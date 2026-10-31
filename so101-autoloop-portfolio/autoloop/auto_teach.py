#!/usr/bin/env python3
"""1-Pose Kinesthetic Teach -> Autonomous Real-Arm Demonstration Collector & Self-Resetting Loop."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import cv2  # Must be imported before PyAV / lerobot.datasets on macOS
import numpy as np
import torch

from lerobot.datasets import LeRobotDataset
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

from .critic import detect_pingpong_ball, evaluate_real_rollout
from .filmstrip import render_rollout_filmstrip, summarize_telemetry
from .hardware_config import CAPTURES_DIR, MOTOR_NAMES, RUNS_DIR, RigConfig
from .watchdog import SafeSOFollower, SafetyTripError

TAUGHT_POSE_PATH = CAPTURES_DIR / "taught_grasp_pose.json"


def capture_taught_grasp_pose(rig_cfg: RigConfig | None = None) -> dict[str, Any]:
    """Read the follower arm's current joint angles (with torque off) as the ground-truth ball grasp pose."""
    cfg = rig_cfg or RigConfig()
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    f_cfg = SOFollowerRobotConfig(port=cfg.follower_port, id=cfg.follower_id)
    follower = SOFollower(f_cfg)
    follower.bus.connect()
    try:
        if not follower.is_calibrated:
            follower.calibration = follower.bus.read_calibration()
            follower.bus.calibration = follower.calibration
            follower._save_calibration()
        pos = follower.bus.sync_read("Present_Position", num_retry=2)
        grasp_pose = {m: round(float(pos[m]), 2) for m in MOTOR_NAMES}
    finally:
        follower.bus.disconnect(disable_torque=True)

    for cam_name, spec in cfg.cameras.items():
        cap = cv2.VideoCapture(spec.index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, spec.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, spec.height)
            frame = None
            for _ in range(20):
                ok, f = cap.read()
                if ok and f is not None:
                    frame = f
                time.sleep(0.03)
            cap.release()
            if frame is not None:
                cv2.imwrite(str(CAPTURES_DIR / f"taught_{cam_name}.png"), frame)

    # Build the 4-waypoint pick-lift-and-place-back cycle from the taught grasp pose
    pre_grasp = dict(grasp_pose)
    pre_grasp["shoulder_lift"] = round(max(-110.0, grasp_pose["shoulder_lift"] - 18.0), 2)
    pre_grasp["elbow_flex"] = round(min(105.0, grasp_pose["elbow_flex"] + 10.0), 2)
    pre_grasp["gripper"] = 48.0  # Open wide enough for 40mm ping-pong ball

    grasp_open = dict(grasp_pose)
    grasp_open["gripper"] = 48.0

    # Close gripper snugly on the 40mm ball (8% tighter than resting contact)
    grasp_closed = dict(grasp_pose)
    grasp_closed["gripper"] = round(max(3.0, grasp_pose["gripper"] - 8.0), 2)

    lift_pose = dict(grasp_closed)
    lift_pose["shoulder_lift"] = round(max(-110.0, grasp_pose["shoulder_lift"] - 26.0), 2)
    lift_pose["elbow_flex"] = round(min(105.0, grasp_pose["elbow_flex"] + 14.0), 2)

    data = {
        "timestamp": time.time(),
        "grasp_pose_taught": grasp_pose,
        "waypoints": {
            "pre_grasp": pre_grasp,
            "grasp_open": grasp_open,
            "grasp_closed": grasp_closed,
            "lift_pose": lift_pose,
        },
    }
    TAUGHT_POSE_PATH.write_text(json.dumps(data, indent=2))
    return data


def generate_waypoint_trajectory(
    start_pose: dict[str, float],
    waypoints: dict[str, dict[str, float]],
    fps: int = 15,
) -> list[dict[str, float]]:
    """Interpolate a smooth trajectory: start -> pre_grasp -> grasp_open -> grasp_closed -> lift_pose."""
    segments = [
        (start_pose, waypoints["pre_grasp"], 1.4),
        (waypoints["pre_grasp"], waypoints["grasp_open"], 1.0),
        (waypoints["grasp_open"], waypoints["grasp_closed"], 0.7),
        (waypoints["grasp_closed"], waypoints["lift_pose"], 1.2),
    ]
    traj: list[dict[str, float]] = []
    for p0, p1, dur_s in segments:
        n = max(2, int(round(dur_s * fps)))
        for i in range(1, n + 1):
            # Cosine easing for zero jerk at waypoint transitions
            alpha = 0.5 * (1.0 - np.cos(np.pi * (i / n)))
            pose = {m: float(p0[m] + alpha * (p1[m] - p0[m])) for m in MOTOR_NAMES}
            traj.append(pose)
    return traj


def self_reset_ball(safe_bot: SafeSOFollower, waypoints: dict[str, dict[str, float]], fps: int = 15) -> None:
    """Place the lifted ping-pong ball gently back into the exact taught pickup spot and open gripper."""
    # 1. Lower back to grasp_closed pose
    safe_bot.smooth_move_to(waypoints["grasp_closed"], duration_s=1.2, fps=fps)
    time.sleep(0.15)
    # 2. Open gripper gently so ball stays in place without rolling
    safe_bot.smooth_move_to(waypoints["grasp_open"], duration_s=0.6, fps=fps)
    time.sleep(0.15)
    # 3. Retract upward to pre_grasp before moving laterally
    safe_bot.smooth_move_to(waypoints["pre_grasp"], duration_s=1.0, fps=fps)


def collect_autonomous_dataset(
    repo_id: str,
    root_dir: Path,
    num_episodes: int = 10,
    fps: int = 15,
    task_description: str = "Pick up the orange ping pong ball and lift it",
    active_cameras: list[str] | None = None,
) -> dict[str, Any]:
    """Autonomously execute & record `num_episodes` real-world demos with start-pose perturbations and self-reset."""
    if not TAUGHT_POSE_PATH.exists():
        raise FileNotFoundError(f"No taught grasp pose found at {TAUGHT_POSE_PATH}. Run 'teach' first.")

    taught = json.loads(TAUGHT_POSE_PATH.read_text())
    waypoints = taught["waypoints"]
    cfg = RigConfig()
    cams = active_cameras if active_cameras is not None else ["front", "wrist"]

    import shutil

    if root_dir.exists():
        shutil.rmtree(root_dir)

    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(MOTOR_NAMES),),
            "names": [f"{m}.pos" for m in MOTOR_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(MOTOR_NAMES),),
            "names": [f"{m}.pos" for m in MOTOR_NAMES],
        },
    }
    for c in cams:
        spec = cfg.cameras[c]
        features[f"observation.images.{c}"] = {
            "dtype": "video",
            "shape": (spec.height, spec.width, 3),
            "names": ["height", "width", "channels"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root_dir,
        robot_type=cfg.follower_type,
        features=features,
        use_videos=True,
    )

    safe_bot = SafeSOFollower(cfg, enable_cameras=True, active_cameras=cams)
    safe_bot.connect()

    rng = np.random.default_rng(42)
    base_start = dict(waypoints["pre_grasp"])
    base_start["shoulder_lift"] = max(-108.0, waypoints["pre_grasp"]["shoulder_lift"] - 10.0)

    last_frames: dict[str, list[np.ndarray]] = {c: [] for c in cams}
    last_telemetry: list[dict[str, Any]] = []

    try:
        # First release the ball cleanly in place and lift straight up to pre_grasp
        safe_bot.smooth_move_to(waypoints["grasp_open"], duration_s=0.6, fps=fps)
        time.sleep(0.15)
        safe_bot.smooth_move_to(waypoints["pre_grasp"], duration_s=1.0, fps=fps)
        time.sleep(0.2)

        for ep_idx in range(num_episodes):
            # Vary start pose slightly each episode so ACT learns spatial generalization
            start_pose = dict(base_start)
            if ep_idx > 0:
                start_pose["shoulder_pan"] += float(rng.uniform(-6.0, 6.0))
                start_pose["shoulder_lift"] += float(rng.uniform(-4.0, 4.0))
                start_pose["elbow_flex"] += float(rng.uniform(-4.0, 4.0))

            safe_bot.smooth_move_to(start_pose, duration_s=1.2, fps=fps)
            time.sleep(0.2)

            traj = generate_waypoint_trajectory(start_pose, waypoints, fps=fps)
            ep_frames: dict[str, list[np.ndarray]] = {c: [] for c in cams}
            ep_tel: list[dict[str, Any]] = []

            for target_pose in traj:
                t0 = time.perf_counter()
                action_dict = {f"{m}.pos": target_pose[m] for m in MOTOR_NAMES}
                obs, tel = safe_bot.step_safe(action_dict)

                frame_dict: dict[str, Any] = {
                    "observation.state": np.array([float(obs[f"{m}.pos"]) for m in MOTOR_NAMES], dtype=np.float32),
                    "action": np.array([float(target_pose[m]) for m in MOTOR_NAMES], dtype=np.float32),
                    "task": task_description,
                }
                for c in cams:
                    frame_dict[f"observation.images.{c}"] = obs[c]
                    ep_frames[c].append(obs[c].copy())

                dataset.add_frame(frame_dict)
                ep_tel.append(dataclasses.asdict(tel))

                dt = time.perf_counter() - t0
                if (1.0 / fps) - dt > 0:
                    time.sleep((1.0 / fps) - dt)

            dataset.save_episode()
            last_frames = ep_frames
            last_telemetry = ep_tel

            # Gently place the ball back into the exact pickup spot for the next episode
            self_reset_ball(safe_bot, waypoints, fps=fps)
            print(f"[Auto-Collect] Saved episode {ep_idx + 1}/{num_episodes}")

        safe_bot.reset_to_home()
    finally:
        dataset.finalize()
        safe_bot.disconnect(disable_torque=True)

    filmstrip_path = CAPTURES_DIR / "auto_collect_last_ep_filmstrip.png"
    render_rollout_filmstrip(
        frames_by_cam=last_frames,
        telemetry_steps=last_telemetry,
        output_png_path=filmstrip_path,
        title=f"Auto-Collected Demo ({num_episodes} episodes)",
    )
    summary = summarize_telemetry(last_telemetry)
    critic = evaluate_real_rollout(last_telemetry, last_frames)
    return {
        "repo_id": repo_id,
        "root_dir": str(root_dir),
        "num_episodes": num_episodes,
        "filmstrip_png": str(filmstrip_path),
        "last_episode_summary": summary,
        "last_episode_critic": critic,
    }
