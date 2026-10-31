#!/usr/bin/env python3
"""Visual Feedback Bridge: generates annotated contact-sheet filmstrips + joint telemetry plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .critic import detect_pingpong_ball
from .hardware_config import MOTOR_NAMES


def render_rollout_filmstrip(
    frames_by_cam: dict[str, list[np.ndarray]],
    telemetry_steps: list[dict[str, Any]],
    output_png_path: Path,
    num_keyframes: int = 8,
    title: str = "Real-Arm Rollout Filmstrip",
    frames_are_rgb: bool = True,
) -> Path:
    """Create a contact-sheet PNG combining camera keyframes and joint position/load plots."""
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    n_steps = len(telemetry_steps)
    cam_names = [k for k, v in frames_by_cam.items() if len(v) > 0]

    tile_w, tile_h = 320, 240
    key_indices = (
        np.linspace(0, max(0, n_steps - 1), num=min(num_keyframes, max(1, n_steps)), dtype=int).tolist()
        if n_steps > 0
        else [0]
    )
    cols = len(key_indices)
    cam_rows = max(1, len(cam_names))
    header_h = 44
    plot_h = 220
    canvas_w = max(cols * tile_w, 960)
    canvas_h = header_h + cam_rows * tile_h + plot_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 20, 26))
    draw = ImageDraw.Draw(canvas)

    # Header bar
    draw.rectangle([0, 0, canvas_w, header_h], fill=(30, 34, 45))
    draw.text((12, 12), f"{title}  |  Total steps: {n_steps}", fill=(235, 240, 250))

    # Draw camera keyframe rows
    for r_idx, cam_name in enumerate(cam_names):
        frames = frames_by_cam[cam_name]
        for c_idx, step_idx in enumerate(key_indices):
            x0 = c_idx * tile_w
            y0 = header_h + r_idx * tile_h
            frame_idx = min(step_idx, len(frames) - 1)
            raw_frame = frames[frame_idx]
            rgb = (raw_frame if frames_are_rgb else cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)).copy()
            det = detect_pingpong_ball(rgb, is_wrist=(cam_name == "wrist"))
            if det is not None:
                cv2.circle(rgb, (det["cx"], det["cy"]), int(max(6, det["radius"])), (0, 255, 100), 2)
            pil_tile = Image.fromarray(rgb).resize((tile_w, tile_h))
            canvas.paste(pil_tile, (x0, y0))

            # Overlay telemetry banner on each tile
            draw.rectangle([x0, y0, x0 + tile_w, y0 + 38], fill=(10, 12, 16))
            if step_idx < len(telemetry_steps):
                t_step = telemetry_steps[step_idx]
                pos = t_step.get("present_pos", {})
                loads = t_step.get("present_load", {})
                errs = t_step.get("tracking_error", {})
                grip = pos.get("gripper", 0.0)
                max_ld = max((abs(int(v)) for v in loads.values()), default=0)
                max_err = max((abs(float(v)) for v in errs.values()), default=0.0)
                line1 = f"[{cam_name}] step {step_idx}/{max(0, n_steps-1)} | grip:{grip:.0f}%"
                line2 = f"max|err|:{max_err:.1f}deg | max|load|:{max_ld}"
            else:
                line1 = f"[{cam_name}] frame {frame_idx}"
                line2 = ""

            draw.text((x0 + 6, y0 + 4), line1, fill=(255, 255, 255))
            if line2:
                color2 = (255, 110, 110) if max_ld > 500 or max_err > 15.0 else (170, 220, 255)
                draw.text((x0 + 6, y0 + 20), line2, fill=color2)
            draw.rectangle([x0, y0, x0 + tile_w - 1, y0 + tile_h - 1], outline=(60, 68, 85))

    # Draw bottom telemetry plot strip (joint positions & loads over time)
    plot_y0 = header_h + cam_rows * tile_h
    draw.rectangle([0, plot_y0, canvas_w, canvas_h], fill=(22, 25, 32))

    if n_steps >= 2:
        # Left half: Present_Position (solid) vs Goal_Position (dashed/dim)
        pw = (canvas_w - 36) // 2
        ph = plot_h - 40
        px1, py1 = 12, plot_y0 + 28
        px2, py2 = px1 + pw + 12, plot_y0 + 28

        draw.text((px1, plot_y0 + 6), "Joint Angles (deg / gripper %): Present vs Goal", fill=(210, 220, 235))
        draw.rectangle([px1, py1, px1 + pw, py1 + ph], outline=(70, 75, 90), fill=(14, 16, 22))

        draw.text((px2, plot_y0 + 6), "Servo Load (signed -1000..1000, trip=+-700)", fill=(210, 220, 235))
        draw.rectangle([px2, py1, px2 + pw, py1 + ph], outline=(70, 75, 90), fill=(14, 16, 22))

        palette = [
            (255, 99, 132),   # shoulder_pan
            (54, 162, 235),   # shoulder_lift
            (255, 206, 86),   # elbow_flex
            (75, 192, 192),   # wrist_flex
            (153, 102, 255),  # wrist_roll
            (255, 159, 64),   # gripper
        ]

        # Plot zero line on both panels
        zero_y_pos = py1 + ph // 2
        draw.line([(px1, zero_y_pos), (px1 + pw, zero_y_pos)], fill=(45, 50, 60))
        draw.line([(px2, zero_y_pos), (px2 + pw, zero_y_pos)], fill=(45, 50, 60))

        for m_idx, m_name in enumerate(MOTOR_NAMES):
            col = palette[m_idx % len(palette)]
            pos_pts = []
            load_pts = []
            for s_i, step_d in enumerate(telemetry_steps):
                x_rel = int((s_i / max(1, n_steps - 1)) * (pw - 4)) + 2
                p_val = float(step_d.get("present_pos", {}).get(m_name, 0.0))
                l_val = float(step_d.get("present_load", {}).get(m_name, 0.0))

                # Map [-150, 150] deg -> [py1+ph-2, py1+2]
                y_p = int(py1 + ph / 2 - (np.clip(p_val, -150.0, 150.0) / 150.0) * (ph / 2 - 4))
                # Map [-1000, 1000] load -> [py1+ph-2, py1+2]
                y_l = int(py1 + ph / 2 - (np.clip(l_val, -1000.0, 1000.0) / 1000.0) * (ph / 2 - 4))

                pos_pts.append((px1 + x_rel, y_p))
                load_pts.append((px2 + x_rel, y_l))

            if len(pos_pts) >= 2:
                draw.line(pos_pts, fill=col, width=2)
                draw.line(load_pts, fill=col, width=2)

            # Legend at bottom
            leg_x = px1 + m_idx * (canvas_w // 6)
            draw.text((leg_x, canvas_h - 14), f"■ {m_name}", fill=col)

    canvas.save(output_png_path)
    return output_png_path


def summarize_telemetry(telemetry_steps: list[dict[str, Any]], safety_tripped: bool = False, trip_reason: str = "") -> dict[str, Any]:
    """Generate a compact JSON summary of a real-hardware rollout."""
    if not telemetry_steps:
        return {"num_steps": 0, "safety_tripped": safety_tripped, "trip_reason": trip_reason}

    max_abs_load: dict[str, int] = {m: 0 for m in MOTOR_NAMES}
    max_abs_err: dict[str, float] = {m: 0.0 for m in MOTOR_NAMES}
    min_pos: dict[str, float] = {m: 999.0 for m in MOTOR_NAMES}
    max_pos: dict[str, float] = {m: -999.0 for m in MOTOR_NAMES}
    clamped_counts: dict[str, int] = {m: 0 for m in MOTOR_NAMES}

    for step in telemetry_steps:
        pos = step.get("present_pos", {})
        loads = step.get("present_load", {})
        errs = step.get("tracking_error", {})
        for m in MOTOR_NAMES:
            p = float(pos.get(m, 0.0))
            ld = abs(int(loads.get(m, 0)))
            er = abs(float(errs.get(m, 0.0)))
            min_pos[m] = min(min_pos[m], p)
            max_pos[m] = max(max_pos[m], p)
            max_abs_load[m] = max(max_abs_load[m], ld)
            max_abs_err[m] = round(max(max_abs_err[m], er), 2)
        for m in step.get("clamped_joints", []):
            if m in clamped_counts:
                clamped_counts[m] += 1

    joint_range_swept = {m: round(max_pos[m] - min_pos[m], 2) for m in MOTOR_NAMES}
    return {
        "num_steps": len(telemetry_steps),
        "safety_tripped": safety_tripped,
        "trip_reason": trip_reason,
        "max_abs_load": max_abs_load,
        "max_abs_tracking_error_deg": max_abs_err,
        "joint_range_swept_deg": joint_range_swept,
        "watchdog_clamped_step_counts": clamped_counts,
        "start_pose": telemetry_steps[0].get("present_pos", {}),
        "end_pose": telemetry_steps[-1].get("present_pos", {}),
    }
