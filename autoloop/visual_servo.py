#!/usr/bin/env python3
"""Two-Stage Visual Tracker:
1. Static External Camera (`front`) = Global Polar Guide (observes full mat from retracted pose, waits for ball to settle, computes polar angle & radial reach).
2. Arm Wrist Camera (`wrist`) = Local Ball Chase (steers gripper jaws onto the ball during approach/descent).
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2  # Must be imported before PyAV / lerobot.datasets on macOS
import numpy as np

from lerobot.datasets import LeRobotDataset

from .auto_teach import TAUGHT_POSE_PATH
from .critic import detect_pingpong_ball, evaluate_real_rollout
from .filmstrip import render_rollout_filmstrip, summarize_telemetry
from .hardware_config import CAPTURES_DIR, MOTOR_NAMES, RigConfig
from .watchdog import SafeSOFollower

# Static Front Camera (640x480 downscaled from full 1280x800 sensor):
#   Arm turret base pivot in front camera: (FRONT_BASE_X, FRONT_BASE_Y) = (180.0, 115.0)
#   Taught grasp ball position in front camera (lift=-7.1, elbow=45.0, wrist_flex=54.2): (186.0, 358.0)
FRONT_BASE_X = 180.0
FRONT_BASE_Y = 115.0
FRONT_REF_X = 186.0
FRONT_REF_Y = 358.0

FRONT_REF_THETA_DEG = math.degrees(math.atan2(FRONT_REF_X - FRONT_BASE_X, FRONT_REF_Y - FRONT_BASE_Y))
FRONT_REF_RADIUS_PX = math.hypot(FRONT_REF_X - FRONT_BASE_X, FRONT_REF_Y - FRONT_BASE_Y)

# Wrist Camera (640x480) local chase target inside the asymmetric jaw pocket:
#   Left finger in wrist camera (x~=240..300) is the FIXED claw; right finger is the MOVEABLE claw.
#   To avoid the fixed claw hitting the ball, the ball must sit to the right of the fixed claw at cx~=450, cy~=365.
WRIST_CHASE_CX = 450.0
WRIST_CHASE_CY = 365.0
PX_PER_DEG_PAN_WRIST = 17.25
PX_PER_DEG_REACH_WRIST = 7.75


def front_pixel_to_offsets(cx_front: float, cy_front: float) -> tuple[float, float]:
    """Map (cx_front, cy_front) from the static external camera to (pan_offset_deg, reach_offset_deg)."""
    dx = cx_front - FRONT_BASE_X
    dy = max(60.0, cy_front - FRONT_BASE_Y)
    theta_deg = math.degrees(math.atan2(dx, dy))
    radius_px = math.hypot(dx, dy)

    # -4.5 deg offset places the rigid fixed claw to the right of the ball in front camera (left of ball in wrist camera)
    pan_offset_deg = -0.85 * (theta_deg - FRONT_REF_THETA_DEG) - 4.5
    reach_offset_deg = ((radius_px - FRONT_REF_RADIUS_PX) / 2.0) + 5.5

    return (
        float(np.clip(pan_offset_deg, -45.0, 45.0)),
        float(np.clip(reach_offset_deg, -18.0, 28.0)),
    )


def compute_table_pose(
    base_grasp: dict[str, float],
    pan_offset_deg: float,
    reach_offset_deg: float,
    height_above_table_deg: float,
    gripper_pct: float,
) -> dict[str, float]:
    """Compute a kinematically coupled joint pose parallel to the flat workspace mat.

    Both reach joints (`shoulder_lift` and `elbow_flex`) + `wrist_flex` move together:
      - Reaching forward (`reach_offset_deg > 0`):
          shoulder_lift increases (+0.55) to lean upper arm forward,
          elbow_flex decreases (-0.65) to extend forearm forward,
          wrist_flex DECREASES (-0.35) to extend wrist forward instead of curling back!
      - Lifting above table (`height_above_table_deg > 0`):
          shoulder_lift decreases (-1.0) and elbow_flex increases (+0.45) to raise gripper straight up.
    """
    pose = dict(base_grasp)
    pose["shoulder_pan"] = float(np.clip(base_grasp["shoulder_pan"] + pan_offset_deg, -65.0, 65.0))

    lift = base_grasp["shoulder_lift"] + 0.55 * reach_offset_deg - 1.0 * height_above_table_deg
    elbow = base_grasp["elbow_flex"] - 0.65 * reach_offset_deg + 0.45 * height_above_table_deg
    wrist_f = base_grasp["wrist_flex"] - 0.35 * reach_offset_deg + 0.15 * height_above_table_deg

    pose["shoulder_lift"] = float(np.clip(lift, -108.0, 18.0))
    pose["elbow_flex"] = float(np.clip(elbow, 22.0, 98.0))
    pose["wrist_flex"] = float(np.clip(wrist_f, 20.0, 90.0))
    pose["gripper"] = float(np.clip(gripper_pct, 2.0, 72.0))
    return pose


def execute_visual_servo_pick(
    safe_bot: SafeSOFollower,
    base_grasp: dict[str, float],
    fps: int = 15,
    max_attempts: int = 3,
    self_reset_at_end: bool = True,
) -> tuple[dict[str, list[np.ndarray]], list[dict[str, Any]], bool]:
    """
    1. Move to retracted `observe_pose` so the static `front` camera sees the whole mat clearly.
    2. Wait until the ball is steady in the `front` camera, then compute global `(pan_offset, reach_offset)`.
    3. Approach to chase height above the ball and use the `wrist` camera to chase/center the ball between the fingers.
    4. Descend, close gripper, verify load, and lift!
    """
    frames_by_cam: dict[str, list[np.ndarray]] = {c: [] for c in safe_bot.robot.cameras}
    telemetry_steps: list[dict[str, Any]] = []

    # Open jaws wide (68%) so the pointed fingertips clear the 40mm ball when descending vertically from above
    open_gripper = 68.0
    # Command 10.0% on close so the servo squeezes the 40mm ball firmly (ball stalls fingers around 16..24%)
    closed_gripper = 10.0

    # Retracted observation pose so the arm does not block the static front camera
    observe_pose = {
        "shoulder_pan": base_grasp["shoulder_pan"],
        "shoulder_lift": -75.0,
        "elbow_flex": 85.0,
        "wrist_flex": 65.0,
        "wrist_roll": base_grasp["wrist_roll"],
        "gripper": open_gripper,
    }

    def record_move_to(target_pose: dict[str, float], duration_s: float, wait_settle: bool = True) -> dict[str, Any]:
        curr_obs = safe_bot.robot.get_observation()
        curr_pose = {m: float(curr_obs[f"{m}.pos"]) for m in MOTOR_NAMES}
        # Scale step count so no joint exceeds 5.5 deg/tick (prevents watchdog clamping lag!)
        max_joint_diff = max(abs(target_pose[m] - curr_pose[m]) for m in MOTOR_NAMES if m != "gripper")
        min_steps_for_speed = int(math.ceil(max_joint_diff / 5.0))
        n = max(2, int(round(duration_s * fps)), min_steps_for_speed)
        last_obs = curr_obs
        for i in range(1, n + 1):
            t0 = time.perf_counter()
            alpha = 0.5 * (1.0 - np.cos(np.pi * (i / n)))
            cmd_pose = {m: float(curr_pose[m] + alpha * (target_pose[m] - curr_pose[m])) for m in MOTOR_NAMES}
            last_obs, tel = safe_bot.step_safe({f"{m}.pos": cmd_pose[m] for m in MOTOR_NAMES})
            for c in frames_by_cam:
                if c in last_obs:
                    frames_by_cam[c].append(last_obs[c].copy())
            telemetry_steps.append(dataclasses.asdict(tel))
            dt = time.perf_counter() - t0
            if (1.0 / fps) - dt > 0:
                time.sleep((1.0 / fps) - dt)

        # Wait up to 8 extra ticks for physical servos to settle at target_pose before starting next stage
        if wait_settle:
            for _ in range(8):
                arm_err = max(
                    abs(float(last_obs[f"{m}.pos"]) - target_pose[m])
                    for m in ["shoulder_pan", "shoulder_lift", "elbow_flex"]
                )
                if arm_err <= 4.0:
                    break
                t0 = time.perf_counter()
                last_obs, tel = safe_bot.step_safe({f"{m}.pos": target_pose[m] for m in MOTOR_NAMES})
                for c in frames_by_cam:
                    if c in last_obs:
                        frames_by_cam[c].append(last_obs[c].copy())
                telemetry_steps.append(dataclasses.asdict(tel))
                dt = time.perf_counter() - t0
                if (1.0 / fps) - dt > 0:
                    time.sleep((1.0 / fps) - dt)
        return last_obs

    grasped = False
    pan_offset = 0.0
    reach_offset = 0.0

    for attempt in range(max_attempts):
        # STAGE 1: Retract to clear observation pose & use Static Front Camera to locate steady ball
        record_move_to(observe_pose, duration_s=1.2)
        prev_front_xy: tuple[float, float] | None = None
        steady_ticks = 0
        front_det: dict[str, Any] | None = None

        for _ in range(35):
            t0 = time.perf_counter()
            obs, tel = safe_bot.step_safe({f"{m}.pos": observe_pose[m] for m in MOTOR_NAMES})
            for c in frames_by_cam:
                if c in obs:
                    frames_by_cam[c].append(obs[c].copy())
            telemetry_steps.append(dataclasses.asdict(tel))

            det = detect_pingpong_ball(obs.get("front"), is_wrist=False) if "front" in obs else None
            if det is not None:
                front_det = det
                curr_xy = (float(det["cx"]), float(det["cy"]))
                if prev_front_xy is not None:
                    speed = math.hypot(curr_xy[0] - prev_front_xy[0], curr_xy[1] - prev_front_xy[1])
                    if speed <= 2.5:
                        steady_ticks += 1
                        if steady_ticks >= 5:
                            break
                    else:
                        steady_ticks = 0
                prev_front_xy = curr_xy

            dt = time.perf_counter() - t0
            if (1.0 / fps) - dt > 0:
                time.sleep((1.0 / fps) - dt)

        if front_det is not None:
            pan_offset, reach_offset = front_pixel_to_offsets(float(front_det["cx"]), float(front_det["cy"]))
            print(
                f"[FrontGuide] Ball steady at front=({front_det['cx']}, {front_det['cy']}) "
                f"-> pan_offset={pan_offset:+.2f} deg, reach_offset={reach_offset:+.2f} deg"
            )
        front_pan_anchor = pan_offset
        front_reach_anchor = reach_offset

        # STAGE 2: Move in mid-air at HIGH clearance (22 deg above table, ~6cm up) so the arm NEVER touches the ball en route
        approach_pose = compute_table_pose(base_grasp, pan_offset, reach_offset, 22.0, open_gripper)
        record_move_to(approach_pose, duration_s=1.2)

        # Lower to 10.0 deg above table (still clear above the top of the 40mm ball!) for overhead wrist alignment
        pre_chase_pose = compute_table_pose(base_grasp, pan_offset, reach_offset, 10.0, open_gripper)
        record_move_to(pre_chase_pose, duration_s=0.65)

        # STAGE 3: Overhead Alignment with Wrist Camera at safe clearance (height=9 deg, above the ball so it cannot bump it)
        for _chase_step in range(12):
            t0 = time.perf_counter()
            obs = safe_bot.robot.get_observation()
            w_det = detect_pingpong_ball(obs.get("wrist"), is_wrist=True) if "wrist" in obs else None
            if w_det is None:
                break

            err_x = float(w_det["cx"]) - WRIST_CHASE_CX
            err_y = float(w_det["cy"]) - WRIST_CHASE_CY

            if abs(err_x) <= 24.0 and abs(err_y) <= 28.0:
                print(f"[WristChase] Ball aligned under open jaws at wrist=({w_det['cx']}, {w_det['cy']})!")
                break

            d_pan = +0.28 * (err_x / PX_PER_DEG_PAN_WRIST)
            d_reach = -0.40 * (err_y / PX_PER_DEG_REACH_WRIST)

            pan_offset = float(
                np.clip(pan_offset + np.clip(d_pan, -1.8, 1.8), front_pan_anchor - 6.0, front_pan_anchor + 6.0)
            )
            reach_offset = float(
                np.clip(
                    reach_offset + np.clip(d_reach, -2.0, 2.5), front_reach_anchor - 6.0, front_reach_anchor + 14.0
                )
            )

            chase_pose = compute_table_pose(base_grasp, pan_offset, reach_offset, 9.0, open_gripper)
            obs_step, tel = safe_bot.step_safe({f"{m}.pos": chase_pose[m] for m in MOTOR_NAMES})
            for c in frames_by_cam:
                if c in obs_step:
                    frames_by_cam[c].append(obs_step[c].copy())
            telemetry_steps.append(dataclasses.asdict(tel))

            dt = time.perf_counter() - t0
            if (1.0 / fps) - dt > 0:
                time.sleep((1.0 / fps) - dt)

        # STAGE 4: Pure vertical descent onto the mat (height=-1.0 deg) with jaws wide open (68%), wait until settled on mat, THEN close jaws!
        grasp_open_pose = compute_table_pose(base_grasp, pan_offset, reach_offset, -1.0, open_gripper)
        record_move_to(grasp_open_pose, duration_s=0.9, wait_settle=True)
        time.sleep(0.25)

        grasp_closed_pose = compute_table_pose(base_grasp, pan_offset, reach_offset, -1.0, closed_gripper)
        record_move_to(grasp_closed_pose, duration_s=0.7, wait_settle=False)
        time.sleep(0.2)

        check_lift_pose = compute_table_pose(base_grasp, pan_offset, reach_offset, 6.0, closed_gripper)
        obs_check = record_move_to(check_lift_pose, duration_s=0.45)

        last_tel = telemetry_steps[-1]
        grip_pos = float(last_tel["present_pos"]["gripper"])
        grip_load = abs(int(last_tel["present_load"].get("gripper", 0)))
        w_check = detect_pingpong_ball(obs_check.get("wrist"), is_wrist=True) if "wrist" in obs_check else None
        ball_in_jaws = w_check is not None and float(w_check["cy"]) >= 220.0 and float(w_check["radius"]) >= 55.0

        # A 40mm ping-pong ball physically stalls the gripper open >= 14.5% (empty gripper closes to ~10.2..11.0%)
        if grip_pos >= 14.5 or (grip_pos >= 13.0 and ball_in_jaws):
            grasped = True
            print(
                f"[WristChase] Grasp confirmed! (gripper_pos={grip_pos:.1f}%, load={grip_load}, ball_in_jaws={ball_in_jaws}) Lifting ball..."
            )
            full_lift_pose = compute_table_pose(base_grasp, pan_offset, reach_offset, 28.0, closed_gripper)
            record_move_to(full_lift_pose, duration_s=1.2)
            break
        else:
            print(
                f"[WristChase] Attempt {attempt + 1}: gripper closed empty (pos={grip_pos:.1f}%, load={grip_load}), "
                "re-acquiring ball from static front camera..."
            )
            reopen_pose = compute_table_pose(base_grasp, pan_offset, reach_offset, 14.0, open_gripper)
            record_move_to(reopen_pose, duration_s=0.5)

    if self_reset_at_end and grasped:
        place_pose = compute_table_pose(base_grasp, pan_offset, reach_offset, 0.0, closed_gripper)
        safe_bot.smooth_move_to(place_pose, duration_s=1.0, fps=fps)
        time.sleep(0.15)
        release_pose = compute_table_pose(base_grasp, pan_offset, reach_offset, 0.0, open_gripper)
        safe_bot.smooth_move_to(release_pose, duration_s=0.5, fps=fps)
        time.sleep(0.15)
        retract_pose = compute_table_pose(base_grasp, pan_offset, reach_offset, 18.0, open_gripper)
        safe_bot.smooth_move_to(retract_pose, duration_s=0.8, fps=fps)

    return frames_by_cam, telemetry_steps, grasped


def run_visual_servo_session(
    num_episodes: int = 1,
    fps: int = 15,
    record_dataset_repo: str | None = None,
    record_dataset_root: Path | None = None,
    hold_at_top: bool = False,
) -> dict[str, Any]:
    """Run two-stage (Static Front Guide + Wrist Chase) ball tracking & grasping across `num_episodes`."""
    if not TAUGHT_POSE_PATH.exists():
        raise FileNotFoundError(f"No taught grasp pose at {TAUGHT_POSE_PATH}. Run 'teach' first.")

    taught = json.loads(TAUGHT_POSE_PATH.read_text())
    base_grasp = taught["grasp_pose_taught"]
    cfg = RigConfig()
    cams = ["front", "wrist"]

    dataset = None
    if record_dataset_repo and record_dataset_root:
        import shutil

        if record_dataset_root.exists():
            shutil.rmtree(record_dataset_root)
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
            repo_id=record_dataset_repo,
            fps=fps,
            root=record_dataset_root,
            robot_type=cfg.follower_type,
            features=features,
            use_videos=True,
        )

    safe_bot = SafeSOFollower(cfg, enable_cameras=True, active_cameras=cams)
    safe_bot.connect()

    successes = 0
    last_frames: dict[str, list[np.ndarray]] = {c: [] for c in cams}
    last_tel: list[dict[str, Any]] = []

    try:
        for ep in range(num_episodes):
            ep_frames, ep_tel, grasped = execute_visual_servo_pick(
                safe_bot,
                base_grasp=base_grasp,
                fps=fps,
                max_attempts=3,
                self_reset_at_end=(not hold_at_top or ep < num_episodes - 1),
            )
            if grasped:
                successes += 1
            last_frames = ep_frames
            last_tel = ep_tel

            if dataset is not None and len(ep_tel) > 0:
                for step_i, t_step in enumerate(ep_tel):
                    frame_dict: dict[str, Any] = {
                        "observation.state": np.array(
                            [float(t_step["present_pos"][m]) for m in MOTOR_NAMES], dtype=np.float32
                        ),
                        "action": np.array(
                            [float(t_step["clipped_goal_pos"][m]) for m in MOTOR_NAMES], dtype=np.float32
                        ),
                        "task": "Track, pick up, and lift the orange ping pong ball",
                    }
                    for c in cams:
                        frame_dict[f"observation.images.{c}"] = ep_frames[c][step_i]
                    dataset.add_frame(frame_dict)
                dataset.save_episode()
            print(f"[VisualServo] Episode {ep + 1}/{num_episodes}: grasped={grasped} ({len(ep_tel)} steps)")

        if not hold_at_top:
            safe_bot.reset_to_home()
    finally:
        if dataset is not None:
            dataset.finalize()
        safe_bot.disconnect(disable_torque=not hold_at_top)

    filmstrip_path = CAPTURES_DIR / "visual_servo_track_filmstrip.png"
    render_rollout_filmstrip(
        frames_by_cam=last_frames,
        telemetry_steps=last_tel,
        output_png_path=filmstrip_path,
        title=f"Front-Guided + Wrist-Chase Tracker ({successes}/{num_episodes} grasped)",
    )
    summary = summarize_telemetry(last_tel)
    critic = evaluate_real_rollout(last_tel, last_frames)
    return {
        "num_episodes": num_episodes,
        "successes": successes,
        "filmstrip_png": str(filmstrip_path),
        "telemetry_summary": summary,
        "critic_evaluation": critic,
    }
