#!/usr/bin/env python3
"""Hardware configuration for the connected SO-100/SO-101 Leader-Follower rig."""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path("/Users/ashi/proj")
LEROBOT_ROOT = PROJECT_ROOT / "lerobot"
CAPTURES_DIR = PROJECT_ROOT / "captures"
RUNS_DIR = PROJECT_ROOT / "autoloop_runs"

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Calibrated rest / home pose (in normalized degrees / 0..100 for gripper)
DEFAULT_HOME_POSE: dict[str, float] = {
    "shoulder_pan": -4.75,
    "shoulder_lift": -103.12,
    "elbow_flex": 97.54,
    "wrist_flex": 77.36,
    "wrist_roll": -89.76,
    "gripper": 10.0,
}

# Safe joint angle bounds (normalized degrees for body joints, 0..100 for gripper)
DEFAULT_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-120.0, 120.0),
    "shoulder_lift": (-115.0, 90.0),
    "elbow_flex": (-90.0, 110.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-175.0, 175.0),
    "gripper": (0.0, 100.0),
}


@dataclass
class CameraSpec:
    name: str
    index: int
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class RigConfig:
    follower_type: str = "so101_follower"
    follower_port: str = "/dev/tty.usbmodem5A7A0581921"
    follower_id: str = "my_follower"

    leader_type: str = "so101_leader"
    leader_port: str = "/dev/tty.usbmodem5A680100721"
    leader_id: str = "my_leader"

    cameras: dict[str, CameraSpec] = field(
        default_factory=lambda: {
            "front": CameraSpec(name="front", index=0, width=640, height=480, fps=30),
            "wrist": CameraSpec(name="wrist", index=1, width=640, height=480, fps=30),
        }
    )

    # Safety watchdog thresholds
    max_step_delta_deg: float = 8.0
    max_gripper_delta: float = 15.0
    max_load_raw: int = 700          # Feetech STS3215 load magnitude (0..1000, bit 10 is direction)
    overload_duration_s: float = 0.20
    max_temperature_c: int = 65
    min_voltage_follower_v: float = 10.0
    max_voltage_follower_v: float = 14.0

    home_pose: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_HOME_POSE))
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_JOINT_LIMITS))

    def lerobot_cameras_arg(self, active_cameras: list[str] | None = None) -> str:
        keys = active_cameras if active_cameras is not None else list(self.cameras.keys())
        parts = []
        for k in keys:
            c = self.cameras[k]
            parts.append(
                f"{c.name}: {{type: opencv, index_or_path: {c.index}, width: {c.width}, height: {c.height}, fps: {c.fps}, warmup_s: 6}}"
            )
        return "{ " + ", ".join(parts) + " }"
