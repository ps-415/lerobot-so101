#!/usr/bin/env python3
"""Deterministic Hardware Safety Watchdog & Health Diagnostics for SO-100/SO-101."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

from .hardware_config import CAPTURES_DIR, MOTOR_NAMES, RigConfig


class SafetyTripError(RuntimeError):
    """Raised when the deterministic hardware watchdog trips and disables torque."""


def decode_sts3215_load(raw_load: int) -> int:
    """Decode signed load from Feetech STS3215 Present_Load register (handles both rustypot signed int and raw 11-bit)."""
    val = int(raw_load)
    if val < 0:
        return val
    if val >= 1024:
        return -(val & 0x3FF)
    return val


@dataclass
class StepTelemetry:
    step: int
    timestamp: float
    present_pos: dict[str, float]
    goal_pos: dict[str, float]
    clipped_goal_pos: dict[str, float]
    tracking_error: dict[str, float]
    present_load: dict[str, int]
    clamped_joints: list[str]


class SafeSOFollower:
    """Wraps SOFollower with deterministic joint limits, step-delta clamping, and overload cutoff."""

    def __init__(self, rig_cfg: RigConfig, enable_cameras: bool = True, active_cameras: list[str] | None = None):
        self.rig_cfg = rig_cfg
        cam_cfgs = {}
        if enable_cameras:
            keys = active_cameras if active_cameras is not None else list(rig_cfg.cameras.keys())
            for k in keys:
                spec = rig_cfg.cameras[k]
                cam_cfgs[k] = OpenCVCameraConfig(
                    index_or_path=spec.index,
                    width=spec.width,
                    height=spec.height,
                    fps=spec.fps,
                    warmup_s=6,
                )

        self.robot_cfg = SOFollowerRobotConfig(
            port=rig_cfg.follower_port,
            id=rig_cfg.follower_id,
            max_relative_target=None,
            position_p_coefficient=28,
            cameras=cam_cfgs,
        )
        self.robot = SOFollower(self.robot_cfg)
        self._overload_start_time: dict[str, float | None] = {m: None for m in MOTOR_NAMES}
        self._last_goal_pos: dict[str, float] | None = None
        self._step_counter = 0

    def connect(self) -> None:
        self.robot.connect(calibrate=False)
        if not self.robot.is_calibrated:
            self.robot.calibration = self.robot.bus.read_calibration()
            self.robot.bus.calibration = self.robot.calibration
            self.robot._save_calibration()

    def disconnect(self, disable_torque: bool = True) -> None:
        self.robot.config.disable_torque_on_disconnect = disable_torque
        self.robot.disconnect()

    def read_servo_diagnostics(self) -> dict[str, dict[str, float | int]]:
        bus = self.robot.bus
        pos = bus.sync_read("Present_Position", num_retry=2)
        raw_load = bus.sync_read("Present_Load", normalize=False, num_retry=2)
        volt = {m: bus.read("Present_Voltage", m, normalize=False) / 10.0 for m in MOTOR_NAMES}
        temp = {m: int(bus.read("Present_Temperature", m, normalize=False)) for m in MOTOR_NAMES}
        return {
            m: {
                "pos": float(pos[m]),
                "load_signed": decode_sts3215_load(int(raw_load[m])),
                "load_abs": abs(decode_sts3215_load(int(raw_load[m]))),
                "voltage_v": float(volt[m]),
                "temperature_c": int(temp[m]),
            }
            for m in MOTOR_NAMES
        }

    def check_watchdog(self, present_pos: dict[str, float]) -> dict[str, int]:
        """Poll load & temperature and trip torque off if limits are violated."""
        now = time.perf_counter()
        raw_loads = self.robot.bus.sync_read("Present_Load", normalize=False, num_retry=2)
        signed_loads = {m: decode_sts3215_load(int(raw_loads[m])) for m in MOTOR_NAMES}

        for m, load_val in signed_loads.items():
            if abs(load_val) > self.rig_cfg.max_load_raw:
                if self._overload_start_time[m] is None:
                    self._overload_start_time[m] = now
                elif (now - self._overload_start_time[m]) >= self.rig_cfg.overload_duration_s:
                    self.robot.bus.disable_torque()
                    raise SafetyTripError(
                        f"Overload on joint '{m}': |load|={abs(load_val)} > {self.rig_cfg.max_load_raw} "
                        f"for >= {self.rig_cfg.overload_duration_s:.2f}s at pos={present_pos.get(m):.1f}"
                    )
            else:
                self._overload_start_time[m] = None

        # Periodic temperature check (every 90 steps, with confirmation re-read to reject 1-byte UART glitches)
        if self._step_counter > 0 and self._step_counter % 90 == 0:
            for m in MOTOR_NAMES:
                temp = int(self.robot.bus.read("Present_Temperature", m, normalize=False))
                if temp >= self.rig_cfg.max_temperature_c:
                    temp_confirm = int(self.robot.bus.read("Present_Temperature", m, normalize=False))
                    if temp_confirm >= self.rig_cfg.max_temperature_c and temp_confirm <= 100:
                        self.robot.bus.disable_torque()
                        raise SafetyTripError(
                            f"Over-temperature on joint '{m}': {temp_confirm}C >= {self.rig_cfg.max_temperature_c}C"
                        )

        return signed_loads

    def clamp_action(
        self, raw_goal: dict[str, float], present_pos: dict[str, float]
    ) -> tuple[dict[str, float], list[str]]:
        """Apply hard joint limits and per-step slew-rate clamping (without starving PID torque)."""
        clipped: dict[str, float] = {}
        clamped_joints: list[str] = []
        if self._last_goal_pos is None:
            self._last_goal_pos = dict(present_pos)

        for m in MOTOR_NAMES:
            target = float(raw_goal.get(m, present_pos[m]))
            curr = float(present_pos[m])
            prev_goal = float(self._last_goal_pos.get(m, curr))
            lo, hi = self.rig_cfg.joint_limits[m]

            # 1. Hard workspace/joint limits
            bounded = float(np.clip(target, lo, hi))
            # 2. Smooth slew-rate limit from previous commanded goal, plus generous 35 deg envelope from present_pos
            max_slew = self.rig_cfg.max_gripper_delta if m == "gripper" else self.rig_cfg.max_step_delta_deg
            slew_clipped = float(np.clip(bounded, prev_goal - max_slew, prev_goal + max_slew))
            delta_clipped = float(np.clip(slew_clipped, curr - 35.0, curr + 35.0))

            if abs(delta_clipped - target) > 1e-3:
                clamped_joints.append(m)
            clipped[m] = delta_clipped

        self._last_goal_pos = dict(clipped)
        return clipped, clamped_joints

    def step_safe(self, action: dict[str, float]) -> tuple[dict[str, Any], StepTelemetry]:
        """Read observation, verify watchdog, clamp action, and write goal position."""
        obs = self.robot.get_observation()
        present_pos = {m: float(obs[f"{m}.pos"]) for m in MOTOR_NAMES}
        signed_loads = self.check_watchdog(present_pos)

        raw_goal = {k.removesuffix(".pos"): float(v) for k, v in action.items() if k.endswith(".pos") or k in MOTOR_NAMES}
        clipped_goal, clamped_joints = self.clamp_action(raw_goal, present_pos)

        self.robot.bus.sync_write("Goal_Position", clipped_goal)
        self._step_counter += 1

        tracking_error = {m: round(raw_goal.get(m, present_pos[m]) - present_pos[m], 2) for m in MOTOR_NAMES}
        telemetry = StepTelemetry(
            step=self._step_counter,
            timestamp=time.time(),
            present_pos=present_pos,
            goal_pos=raw_goal,
            clipped_goal_pos=clipped_goal,
            tracking_error=tracking_error,
            present_load=signed_loads,
            clamped_joints=clamped_joints,
        )
        return obs, telemetry

    def smooth_move_to(self, target_pose: dict[str, float], duration_s: float = 2.5, fps: int = 30) -> None:
        """Smoothly interpolate from present pose to target_pose under watchdog monitoring."""
        present_pos = self.robot.bus.sync_read("Present_Position", num_retry=2)
        steps = max(1, int(duration_s * fps))
        for i in range(1, steps + 1):
            alpha = i / steps
            interp = {
                m: float(present_pos[m]) + alpha * (float(target_pose.get(m, present_pos[m])) - float(present_pos[m]))
                for m in MOTOR_NAMES
            }
            curr = self.robot.bus.sync_read("Present_Position", num_retry=2)
            self.check_watchdog({m: float(curr[m]) for m in MOTOR_NAMES})
            clipped, _ = self.clamp_action(interp, {m: float(curr[m]) for m in MOTOR_NAMES})
            self.robot.bus.sync_write("Goal_Position", clipped)
            time.sleep(1.0 / fps)

    def reset_to_home(self) -> None:
        """Safe physical reset sequence: open gripper -> lift slightly -> return to home pose -> disable torque."""
        curr = self.robot.bus.sync_read("Present_Position", num_retry=2)
        # Step 1: open gripper to release any held object before moving
        release_pose = {m: float(curr[m]) for m in MOTOR_NAMES}
        release_pose["gripper"] = 45.0
        self.smooth_move_to(release_pose, duration_s=0.6)
        # Step 2: move smoothly to calibrated home pose
        self.smooth_move_to(self.rig_cfg.home_pose, duration_s=2.0)


def check_hardware_health(rig_cfg: RigConfig | None = None, save_captures: bool = True) -> dict[str, Any]:
    """Query follower & leader servo health and capture live camera frames into /Users/ashi/proj/captures."""
    cfg = rig_cfg or RigConfig()
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "timestamp": time.time(),
        "follower": {"port": cfg.follower_port, "connected": False, "is_calibrated": False, "motors": {}},
        "leader": {"port": cfg.leader_port, "connected": False, "is_calibrated": False, "motors": {}},
        "cameras": {},
        "healthy": True,
        "warnings": [],
    }

    # 1. Check Follower Arm
    try:
        f_cfg = SOFollowerRobotConfig(port=cfg.follower_port, id=cfg.follower_id)
        follower = SOFollower(f_cfg)
        follower.bus.connect()
        if not follower.is_calibrated:
            follower.calibration = follower.bus.read_calibration()
            follower.bus.calibration = follower.calibration
            follower._save_calibration()
        pos = follower.bus.sync_read("Present_Position", num_retry=2)
        loads = follower.bus.sync_read("Present_Load", normalize=False, num_retry=2)
        for m in MOTOR_NAMES:
            v = follower.bus.read("Present_Voltage", m, normalize=False) / 10.0
            t = int(follower.bus.read("Present_Temperature", m, normalize=False))
            ld = decode_sts3215_load(int(loads[m]))
            report["follower"]["motors"][m] = {
                "pos_deg": round(float(pos[m]), 2),
                "load": ld,
                "voltage_v": round(v, 2),
                "temperature_c": t,
            }
            if v < cfg.min_voltage_follower_v or v > cfg.max_voltage_follower_v:
                report["warnings"].append(f"Follower {m} voltage {v:.1f}V outside [10V, 14V]")
            if t >= cfg.max_temperature_c:
                report["healthy"] = False
                report["warnings"].append(f"Follower {m} temperature {t}C >= {cfg.max_temperature_c}C")
        report["follower"]["connected"] = True
        report["follower"]["is_calibrated"] = bool(follower.is_calibrated)
        follower.bus.disconnect(disable_torque=False)
    except Exception as e:
        report["healthy"] = False
        report["warnings"].append(f"Follower bus error: {e}")

    # 2. Check Leader Arm
    try:
        l_cfg = SOLeaderTeleopConfig(port=cfg.leader_port, id=cfg.leader_id)
        leader = SOLeader(l_cfg)
        leader.bus.connect()
        if not leader.is_calibrated:
            leader.calibration = leader.bus.read_calibration()
            leader.bus.calibration = leader.calibration
            leader._save_calibration()
        pos = leader.bus.sync_read("Present_Position", num_retry=2)
        for m in MOTOR_NAMES:
            v = leader.bus.read("Present_Voltage", m, normalize=False) / 10.0
            t = int(leader.bus.read("Present_Temperature", m, normalize=False))
            report["leader"]["motors"][m] = {
                "pos_deg": round(float(pos[m]), 2),
                "voltage_v": round(v, 2),
                "temperature_c": t,
            }
        report["leader"]["connected"] = True
        report["leader"]["is_calibrated"] = bool(leader.is_calibrated)
        leader.bus.disconnect(disable_torque=True)
    except Exception as e:
        report["warnings"].append(f"Leader bus error: {e}")

    # 3. Check Cameras (with auto-exposure warmup frames)
    for cam_name, spec in cfg.cameras.items():
        cap = cv2.VideoCapture(spec.index)
        if not cap.isOpened():
            print(
                f"[Camera '{cam_name}'] Waiting up to 10s for macOS Camera permission dialog "
                "(click 'Allow' for Terminal if prompted, or run inside Devin's integrated terminal)..."
            )
            for _ in range(10):
                time.sleep(1.0)
                cap = cv2.VideoCapture(spec.index)
                if cap.isOpened():
                    break
        if not cap.isOpened():
            report["cameras"][cam_name] = {"index": spec.index, "connected": False}
            report["warnings"].append(f"Camera '{cam_name}' (index {spec.index}) failed to open")
            continue

        default_w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        default_h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not (default_w == 1280 and default_h == 800):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, spec.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, spec.height)
        t0 = time.perf_counter()
        frame = None
        frames_read = 0
        # Warm up 25 frames (~1s) so USB UVC auto-exposure settles
        for _ in range(25):
            ok, f = cap.read()
            if ok and f is not None:
                if f.shape[1] != spec.width or f.shape[0] != spec.height:
                    f = cv2.resize(f, (spec.width, spec.height), interpolation=cv2.INTER_AREA)
                frame = f
                frames_read += 1
            time.sleep(0.03)
        elapsed = time.perf_counter() - t0
        cap.release()

        if frame is not None:
            mean_brightness = float(np.mean(frame))
            out_path = CAPTURES_DIR / f"health_{cam_name}_idx{spec.index}.png"
            if save_captures:
                cv2.imwrite(str(out_path), frame)
            report["cameras"][cam_name] = {
                "index": spec.index,
                "connected": True,
                "shape": list(frame.shape),
                "measured_fps": round(frames_read / max(elapsed, 1e-3), 1),
                "mean_brightness": round(mean_brightness, 1),
                "snapshot_path": str(out_path),
            }
            if mean_brightness < 5.0:
                report["warnings"].append(
                    f"Camera '{cam_name}' (index {spec.index}) image is nearly black (brightness={mean_brightness:.1f})"
                )
        else:
            report["cameras"][cam_name] = {"index": spec.index, "connected": False}
            report["warnings"].append(f"Camera '{cam_name}' (index {spec.index}) returned no frames")

    health_json_path = CAPTURES_DIR / "hardware_health.json"
    health_json_path.write_text(json.dumps(report, indent=2))
    return report
