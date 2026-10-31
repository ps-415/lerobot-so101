#!/usr/bin/env python3
"""Main CLI & Orchestrator for the Real-Arm Autonomous Training Loop."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2  # Must be imported before PyAV / lerobot.datasets on macOS so OpenCV's AVFFrameReceiver wins
import numpy as np
import torch

from .auto_teach import TAUGHT_POSE_PATH, capture_taught_grasp_pose, collect_autonomous_dataset, self_reset_ball
from .critic import evaluate_real_rollout
from .visual_servo import run_visual_servo_session
from .filmstrip import render_rollout_filmstrip, summarize_telemetry
from .hardware_config import CAPTURES_DIR, LEROBOT_ROOT, MOTOR_NAMES, RUNS_DIR, RigConfig
from .watchdog import SafeSOFollower, SafetyTripError, check_hardware_health


def cmd_health(args: argparse.Namespace) -> int:
    cfg = RigConfig()
    report = check_hardware_health(cfg, save_captures=True)
    print(json.dumps(report, indent=2))
    return 0 if report["healthy"] else 1


def cmd_home(args: argparse.Namespace) -> int:
    cfg = RigConfig()
    safe_bot = SafeSOFollower(cfg, enable_cameras=False)
    safe_bot.connect()
    try:
        print("Smoothly moving follower arm to calibrated home pose...")
        safe_bot.reset_to_home()
        diag = safe_bot.read_servo_diagnostics()
        print("Reached home pose:", json.dumps(diag, indent=2))
    finally:
        safe_bot.disconnect(disable_torque=not args.hold_torque)
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    cfg = RigConfig()
    active_cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    lerobot_record_bin = str(LEROBOT_ROOT / ".venv" / "bin" / "lerobot-record")
    cmd = [
        lerobot_record_bin,
        f"--robot.type={cfg.follower_type}",
        f"--robot.port={cfg.follower_port}",
        f"--robot.id={cfg.follower_id}",
        f"--robot.max_relative_target={cfg.max_step_delta_deg}",
        f"--teleop.type={cfg.leader_type}",
        f"--teleop.port={cfg.leader_port}",
        f"--teleop.id={cfg.leader_id}",
        f"--robot.cameras={cfg.lerobot_cameras_arg(active_cams)}",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.single_task={args.task}",
        f"--dataset.num_episodes={args.num_episodes}",
        f"--dataset.episode_time_s={args.episode_time_s}",
        f"--dataset.reset_time_s={args.reset_time_s}",
        f"--dataset.push_to_hub={'true' if args.push_to_hub else 'false'}",
        f"--display_data={'true' if args.display_data else 'false'}",
        "--play_sounds=false",
    ]
    if args.root:
        cmd.append(f"--dataset.root={args.root}")
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(LEROBOT_ROOT))


def run_training_job(
    dataset_repo_id: str,
    output_dir: Path,
    policy_type: str = "act",
    steps: int = 2000,
    batch_size: int = 8,
    save_freq: int = 500,
    device: str = "mps",
    dataset_root: str | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run lerobot-train as a managed subprocess and return training metrics & checkpoint path."""
    lerobot_train_bin = str(LEROBOT_ROOT / ".venv" / "bin" / "lerobot-train")
    cmd = [
        lerobot_train_bin,
        f"--dataset.repo_id={dataset_repo_id}",
        f"--policy.type={policy_type}",
        f"--policy.device={device}",
        f"--output_dir={output_dir}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--save_freq={save_freq}",
        "--wandb.enable=false",
        "--policy.push_to_hub=false",
    ]
    if dataset_root:
        cmd.append(f"--dataset.root={dataset_root}")
    if extra_args:
        cmd.extend(extra_args)

    print("Launching training:", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(LEROBOT_ROOT), capture_output=True, text=True)
    duration_s = round(time.time() - t0, 2)

    ckpt_dir = output_dir / "checkpoints" / "last" / "pretrained_model"
    return {
        "returncode": proc.returncode,
        "duration_s": duration_s,
        "checkpoint_path": str(ckpt_dir) if ckpt_dir.exists() else None,
        "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
    }


def run_safe_rollout(
    checkpoint_path: str | None,
    max_steps: int = 90,
    fps: int = 15,
    active_cameras: list[str] | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Execute a policy checkpoint (or safe diagnostic trajectory if None) under SafeSOFollower watchdog."""
    cfg = RigConfig()
    cams = active_cameras if active_cameras is not None else ["front"]
    run_id = run_name or f"rollout_{int(time.time())}"
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    policy = None
    preprocessor = None
    postprocessor = None
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    safe_bot = SafeSOFollower(cfg, enable_cameras=True, active_cameras=cams)
    frames_by_cam: dict[str, list[np.ndarray]] = {c: [] for c in cams}
    telemetry_steps: list[dict[str, Any]] = []
    safety_tripped = False
    trip_reason = ""

    safe_bot.connect()
    try:
        if checkpoint_path:
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import get_policy_class, make_pre_post_processors

            policy_cfg = PreTrainedConfig.from_pretrained(checkpoint_path)
            policy_cfg.device = str(device)
            policy_cls = get_policy_class(policy_cfg.type)
            policy = policy_cls.from_pretrained(checkpoint_path, config=policy_cfg)
            policy.eval()
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy_cfg,
                pretrained_path=checkpoint_path,
            )

        # Record initial pose for gentle diagnostic wave if no policy checkpoint is supplied
        init_obs = safe_bot.robot.get_observation()
        init_pose = {m: float(init_obs[f"{m}.pos"]) for m in MOTOR_NAMES}

        for step_idx in range(max_steps):
            t_tick = time.perf_counter()
            obs = safe_bot.robot.get_observation()

            if policy is not None and preprocessor is not None and postprocessor is not None:
                # Build observation batch for LeRobot policy
                state_vec = torch.tensor(
                    [[float(obs[f"{m}.pos"]) for m in MOTOR_NAMES]], dtype=torch.float32, device=device
                )
                batch: dict[str, Any] = {"observation.state": state_vec}
                for c in cams:
                    if c in obs:
                        img_t = torch.from_numpy(obs[c]).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                        batch[f"observation.images.{c}"] = img_t
                proc_batch = preprocessor(batch)
                with torch.inference_mode():
                    raw_action = policy.select_action(proc_batch)
                unproc_action = postprocessor(raw_action)
                act_np = unproc_action.squeeze(0).detach().cpu().numpy()
                action_dict = {f"{m}.pos": float(act_np[i]) for i, m in enumerate(MOTOR_NAMES)}
            else:
                # Safe micro-trajectory (hold pose with tiny +-2 deg wrist_roll sine wave to test loop safely)
                phase = 2.0 * math.pi * (step_idx / max(1, max_steps))
                action_dict = {f"{m}.pos": init_pose[m] for m in MOTOR_NAMES}
                action_dict["wrist_roll.pos"] = init_pose["wrist_roll"] + 2.0 * math.sin(phase)

            obs_step, step_tel = safe_bot.step_safe(action_dict)
            for c in cams:
                if c in obs_step and isinstance(obs_step[c], np.ndarray):
                    frames_by_cam[c].append(obs_step[c].copy())
            telemetry_steps.append(dataclasses.asdict(step_tel))

            dt = time.perf_counter() - t_tick
            sleep_s = max(0.0, (1.0 / fps) - dt)
            if sleep_s > 0:
                time.sleep(sleep_s)

        # Return smoothly to home pose (and self-reset ball if taught pose exists) at end of episode when running a policy
        if checkpoint_path is not None:
            if TAUGHT_POSE_PATH.exists():
                taught = json.loads(TAUGHT_POSE_PATH.read_text())
                self_reset_ball(safe_bot, taught["waypoints"], fps=fps)
            safe_bot.reset_to_home()

    except SafetyTripError as e:
        safety_tripped = True
        trip_reason = str(e)
        print(f"[WATCHDOG TRIPPED] {trip_reason}", file=sys.stderr)
    finally:
        safe_bot.disconnect(disable_torque=True)

    filmstrip_path = CAPTURES_DIR / f"{run_id}_filmstrip.png"
    render_rollout_filmstrip(
        frames_by_cam=frames_by_cam,
        telemetry_steps=telemetry_steps,
        output_png_path=filmstrip_path,
        title=f"Rollout: {run_id} ({'policy' if checkpoint_path else 'watchdog_test'})",
    )

    summary = summarize_telemetry(telemetry_steps, safety_tripped=safety_tripped, trip_reason=trip_reason)
    critic_eval = evaluate_real_rollout(
        telemetry_steps, frames_by_cam, safety_tripped=safety_tripped, trip_reason=trip_reason
    )

    result = {
        "run_id": run_id,
        "checkpoint_path": checkpoint_path,
        "filmstrip_png": str(filmstrip_path),
        "telemetry_summary": summary,
        "critic_evaluation": critic_eval,
    }
    (out_dir / "rollout_report.json").write_text(json.dumps(result, indent=2))
    (CAPTURES_DIR / "latest_rollout_report.json").write_text(json.dumps(result, indent=2))
    return result


def cmd_rollout(args: argparse.Namespace) -> int:
    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    res = run_safe_rollout(
        checkpoint_path=args.checkpoint,
        max_steps=args.max_steps,
        fps=args.fps,
        active_cameras=cams,
        run_name=args.run_name,
    )
    print(json.dumps(res, indent=2))
    return 0 if not res["telemetry_summary"]["safety_tripped"] else 2


def cmd_train(args: argparse.Namespace) -> int:
    out_dir = RUNS_DIR / (args.job_name or f"train_{int(time.time())}")
    res = run_training_job(
        dataset_repo_id=args.repo_id,
        output_dir=out_dir,
        policy_type=args.policy_type,
        steps=args.steps,
        batch_size=args.batch_size,
        save_freq=args.save_freq,
        device=args.device,
        dataset_root=args.root,
    )
    print(json.dumps(res, indent=2))
    return res["returncode"]


def cmd_iterate(args: argparse.Namespace) -> int:
    """Closed-loop Outer Loop iteration: Health Check -> Train -> Safe Real Rollout -> Filmstrip + Critic."""
    iter_id = args.job_name or f"iter_{int(time.time())}"
    iter_dir = RUNS_DIR / iter_id
    iter_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== [1/3] Checking Hardware Health for {iter_id} ===")
    health = check_hardware_health(RigConfig(), save_captures=True)
    if not health["healthy"]:
        print("Hardware unhealthy, aborting iteration:", json.dumps(health, indent=2))
        return 1

    ckpt = args.checkpoint
    train_res = None
    if not ckpt:
        print(f"=== [2/3] Training {args.policy_type} policy on {args.repo_id} ({args.steps} steps) ===")
        train_res = run_training_job(
            dataset_repo_id=args.repo_id,
            output_dir=iter_dir / "train",
            policy_type=args.policy_type,
            steps=args.steps,
            batch_size=args.batch_size,
            save_freq=args.save_freq,
            device=args.device,
            dataset_root=args.root,
        )
        if train_res["returncode"] != 0 or not train_res["checkpoint_path"]:
            print("Training failed:", json.dumps(train_res, indent=2))
            return 1
        ckpt = train_res["checkpoint_path"]

    print(f"=== [3/3] Running Safe Real-Arm Rollout with checkpoint: {ckpt} ===")
    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    rollout_res = run_safe_rollout(
        checkpoint_path=ckpt,
        max_steps=args.max_steps,
        fps=args.fps,
        active_cameras=cams,
        run_name=f"{iter_id}_eval",
    )

    full_report = {
        "iteration_id": iter_id,
        "hardware_health": health,
        "training": train_res,
        "rollout": rollout_res,
    }
    report_path = CAPTURES_DIR / f"{iter_id}_report.json"
    report_path.write_text(json.dumps(full_report, indent=2))
    print(json.dumps(full_report, indent=2))
    return 0


def cmd_teach(args: argparse.Namespace) -> int:
    res = capture_taught_grasp_pose()
    print(json.dumps(res, indent=2))
    return 0


def cmd_auto_collect(args: argparse.Namespace) -> int:
    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    root = Path(args.root) if args.root else (RUNS_DIR / "datasets" / args.repo_id.replace("/", "_"))
    res = collect_autonomous_dataset(
        repo_id=args.repo_id,
        root_dir=root,
        num_episodes=args.num_episodes,
        fps=args.fps,
        task_description=args.task,
        active_cameras=cams,
    )
    print(json.dumps(res, indent=2))
    return 0


def cmd_track_pick(args: argparse.Namespace) -> int:
    root = Path(args.root) if args.root else None
    res = run_visual_servo_session(
        num_episodes=args.num_episodes,
        fps=args.fps,
        record_dataset_repo=args.repo_id,
        record_dataset_root=root,
        hold_at_top=args.hold_at_top,
    )
    print(json.dumps(res, indent=2))
    return 0 if res["successes"] > 0 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Real-Arm Autonomous Training Loop for SO-100/SO-101")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health", help="Check servo voltages, temperatures, loads, calibration, and camera snapshots")
    sub.add_parser("teach", help="Capture the follower arm's current hand-placed pose as the ball grasp waypoint")

    p_tp = sub.add_parser("track-pick", help="Wait for ball to stop rolling, visually servo onto it, and pick it up")
    p_tp.add_argument("--num-episodes", type=int, default=1)
    p_tp.add_argument("--fps", type=int, default=15)
    p_tp.add_argument("--repo-id", default=None, help="Optional dataset repo_id to record tracked episodes")
    p_tp.add_argument("--root", default=None, help="Optional dataset root directory to record tracked episodes")
    p_tp.add_argument("--hold-at-top", action="store_true", help="Hold ball in air after final pick")

    p_ac = sub.add_parser("auto-collect", help="Autonomously execute & record pick-lift-reset demos from taught pose")
    p_ac.add_argument("--repo-id", default="local/so101_pingpong")
    p_ac.add_argument("--root", default="/Users/ashi/proj/datasets/so101_pingpong")
    p_ac.add_argument("--task", default="Pick up the orange ping pong ball and lift it")
    p_ac.add_argument("--num-episodes", type=int, default=10)
    p_ac.add_argument("--fps", type=int, default=15)
    p_ac.add_argument("--cameras", default="front,wrist")

    p_home = sub.add_parser("home", help="Smoothly return follower arm to calibrated home pose")
    p_home.add_argument("--hold-torque", action="store_true", help="Keep torque enabled after homing")

    p_rec = sub.add_parser("record", help="Record leader-follower teleop dataset via lerobot-record")
    p_rec.add_argument("--repo-id", required=True, help="Dataset repo_id (e.g. local/so101_task)")
    p_rec.add_argument("--task", required=True, help="Single-sentence task description")
    p_rec.add_argument("--num-episodes", type=int, default=20)
    p_rec.add_argument("--episode-time-s", type=int, default=25)
    p_rec.add_argument("--reset-time-s", type=int, default=8)
    p_rec.add_argument("--cameras", default="front,wrist", help="Comma-separated camera keys (front,wrist)")
    p_rec.add_argument("--root", default=None, help="Optional local dataset root directory")
    p_rec.add_argument("--push-to-hub", action="store_true")
    p_rec.add_argument("--display-data", action="store_true")

    p_train = sub.add_parser("train", help="Train policy on dataset (defaults to ACT on Apple Silicon MPS)")
    p_train.add_argument("--repo-id", required=True)
    p_train.add_argument("--policy-type", default="act")
    p_train.add_argument("--steps", type=int, default=2000)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--save-freq", type=int, default=500)
    p_train.add_argument("--device", default="mps")
    p_train.add_argument("--root", default=None)
    p_train.add_argument("--job-name", default=None)

    p_roll = sub.add_parser("rollout", help="Run safe hardware rollout under watchdog + generate filmstrip PNG")
    p_roll.add_argument("--checkpoint", default=None, help="Path to pretrained_model dir (None = safe watchdog test)")
    p_roll.add_argument("--max-steps", type=int, default=45)
    p_roll.add_argument("--fps", type=int, default=15)
    p_roll.add_argument("--cameras", default="front,wrist")
    p_roll.add_argument("--run-name", default=None)

    p_iter = sub.add_parser("iterate", help="Run full outer-loop iteration: health -> train -> safe rollout -> critic")
    p_iter.add_argument("--repo-id", required=True)
    p_iter.add_argument("--checkpoint", default=None, help="Optional existing checkpoint to skip training")
    p_iter.add_argument("--policy-type", default="act")
    p_iter.add_argument("--steps", type=int, default=2000)
    p_iter.add_argument("--batch-size", type=int, default=8)
    p_iter.add_argument("--save-freq", type=int, default=500)
    p_iter.add_argument("--device", default="mps")
    p_iter.add_argument("--root", default=None)
    p_iter.add_argument("--max-steps", type=int, default=60)
    p_iter.add_argument("--fps", type=int, default=15)
    p_iter.add_argument("--cameras", default="front,wrist")
    p_iter.add_argument("--job-name", default=None)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    dispatch = {
        "health": cmd_health,
        "teach": cmd_teach,
        "track-pick": cmd_track_pick,
        "auto-collect": cmd_auto_collect,
        "home": cmd_home,
        "record": cmd_record,
        "train": cmd_train,
        "rollout": cmd_rollout,
        "iterate": cmd_iterate,
    }
    return dispatch[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
