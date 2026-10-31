#!/usr/bin/env python3
"""Real-World Rollout Critic: Ping-Pong Ball visual detection + proprioceptive grasp/lift verification."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def detect_pingpong_ball(rgb_frame: np.ndarray, is_wrist: bool = False) -> dict[str, Any] | None:
    """Detect a white or orange ping-pong ball against the dark workspace mat.

    Returns dict with {"cx": int, "cy": int, "radius": float, "color": str, "norm_x": float, "norm_y": float}
    or None if no candidate ball is found.
    """
    if rgb_frame is None or rgb_frame.size == 0:
        return None

    h, w = rgb_frame.shape[:2]
    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)

    # Mask out regions outside the workspace mat on the front camera
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    if is_wrist:
        roi_mask[:, : int(w * 0.92)] = 255
    else:
        # Black WOWROBO mat ROI on the front camera (excludes floor on left and hand/mouse on right)
        roi_mask[int(h * 0.35) : int(h * 0.98), int(w * 0.12) : int(w * 0.81)] = 255

    r = rgb_frame[:, :, 0].astype(np.int16)
    g = rgb_frame[:, :, 1].astype(np.int16)
    b = rgb_frame[:, :, 2].astype(np.int16)
    # Strictly orange pixels: R > G + 18 and Hue in [2..22] so lime-green "ROBO" letters (G >= R) never match.
    orange_rim = (
        (r >= 200)
        & (g >= 60)
        & (b <= 165)
        & ((r - b) >= 65)
        & ((r - g) >= 18)
        & (hsv[:, :, 0] >= 2)
        & (hsv[:, :, 0] <= 22)
    ).astype(np.uint8) * 255
    close_k = 35 if is_wrist else 21
    orange_mask = cv2.morphologyEx(
        orange_rim, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    )

    best: dict[str, Any] | None = None
    best_score = 0.0

    for color_name, mask in [("orange", orange_mask)]:
        masked = cv2.bitwise_and(mask, roi_mask)
        masked = cv2.morphologyEx(masked, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            hull = cv2.convexHull(cnt)
            area = cv2.contourArea(hull)
            min_area = 50 if not is_wrist else 400
            max_area = 12000 if not is_wrist else 240000
            if area < min_area or area > max_area:
                continue
            perim = cv2.arcLength(hull, True)
            if perim <= 1e-3:
                continue
            circularity = 4.0 * np.pi * area / (perim * perim)
            min_circ = 0.25 if is_wrist else 0.35
            if circularity < min_circ:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(hull)
            # In front camera, the 40mm ping-pong ball has radius ~14..34 px (rejects larger human hands)
            if not is_wrist and (radius < 10.0 or radius > 34.5):
                continue
            score = float(area) * float(circularity)
            if score > best_score:
                best_score = score
                best = {
                    "cx": int(round(cx)),
                    "cy": int(round(cy)),
                    "radius": round(float(radius), 1),
                    "color": color_name,
                    "circularity": round(float(circularity), 2),
                    "norm_x": round(float(cx / w), 3),
                    "norm_y": round(float(cy / h), 3),
                }

    return best


def evaluate_real_rollout(
    telemetry_steps: list[dict[str, Any]],
    frames_by_cam: dict[str, list[np.ndarray]],
    safety_tripped: bool = False,
    trip_reason: str = "",
) -> dict[str, Any]:
    """Compute a dense diagnostic reward and failure-mode classification for ping-pong ball pick & place."""
    if safety_tripped:
        return {
            "success": False,
            "reward": -1.0,
            "stage_reached": "safety_trip",
            "failure_mode": f"watchdog_trip: {trip_reason}",
            "metrics": {},
        }

    if len(telemetry_steps) < 2:
        return {
            "success": False,
            "reward": 0.0,
            "stage_reached": "no_motion",
            "failure_mode": "empty_rollout",
            "metrics": {},
        }

    # 1. Proprioceptive signals (joint motion, gripper opening/closing around a ~40mm ping-pong ball)
    shoulder_lifts = [float(s["present_pos"]["shoulder_lift"]) for s in telemetry_steps]
    elbow_flexes = [float(s["present_pos"]["elbow_flex"]) for s in telemetry_steps]
    grippers = [float(s["present_pos"]["gripper"]) for s in telemetry_steps]
    gripper_loads = [abs(int(s["present_load"].get("gripper", 0))) for s in telemetry_steps]
    lift_loads = [abs(int(s["present_load"].get("shoulder_lift", 0))) for s in telemetry_steps]

    reach_delta = max(shoulder_lifts) - min(shoulder_lifts) + max(elbow_flexes) - min(elbow_flexes)
    gripper_opened = max(grippers) > 22.0
    gripper_closed_after_open = False
    grasp_under_load = False

    if gripper_opened:
        open_idx = int(np.argmax(grippers))
        post_open_grippers = grippers[open_idx:]
        post_open_loads = gripper_loads[open_idx:]
        if min(post_open_grippers) < max(grippers) - 8.0:
            gripper_closed_after_open = True
        # A 40mm ping-pong ball stalls the gripper open > 5% with load or steady intermediate width
        for g_pos, g_ld in zip(post_open_grippers, post_open_loads, strict=False):
            if 5.0 < g_pos < 65.0 and g_ld >= 40:
                grasp_under_load = True
                break

    # 2. Visual tracking of the ping-pong ball in front & wrist cameras
    front_frames = frames_by_cam.get("front", [])
    wrist_frames = frames_by_cam.get("wrist", [])

    ball_start_front = detect_pingpong_ball(front_frames[0], is_wrist=False) if front_frames else None
    ball_end_front = detect_pingpong_ball(front_frames[-1], is_wrist=False) if front_frames else None

    ball_displacement_px = 0.0
    if ball_start_front and ball_end_front:
        ball_displacement_px = float(
            np.hypot(ball_end_front["cx"] - ball_start_front["cx"], ball_end_front["cy"] - ball_start_front["cy"])
        )

    wrist_centered_frames = 0
    for wf in wrist_frames[:: max(1, len(wrist_frames) // 10)]:
        det = detect_pingpong_ball(wf, is_wrist=True)
        if det and 0.25 <= det["norm_x"] <= 0.75 and det["norm_y"] >= 0.35:
            wrist_centered_frames += 1

    # 3. Overall visual motion
    visual_motion_score = 0.0
    for cam_frames in frames_by_cam.values():
        if len(cam_frames) >= 2:
            diffs = [
                float(np.mean(np.abs(cam_frames[i].astype(np.float32) - cam_frames[i - 1].astype(np.float32))))
                for i in range(1, len(cam_frames))
            ]
            visual_motion_score = max(visual_motion_score, float(np.mean(diffs)))

    # 4. Stage progression & dense reward in [0, 1]
    reward = 0.0
    stage = "idle"
    failure_mode = "none"

    if reach_delta > 10.0:
        reward += 0.20
        stage = "approached"
    else:
        failure_mode = "insufficient_arm_motion"

    if wrist_centered_frames > 0:
        reward += 0.15
        stage = "aligned_with_ball"

    if gripper_opened:
        reward += 0.15
        stage = "gripper_opened"

    if gripper_closed_after_open:
        reward += 0.20
        stage = "grasp_attempted"
        if not grasp_under_load and ball_displacement_px < 20.0:
            failure_mode = "missed_grasp_closed_empty"

    if grasp_under_load or ball_displacement_px >= 25.0:
        reward += 0.30
        stage = "ball_picked_or_moved"
        failure_mode = "none"

    success = bool((grasp_under_load or ball_displacement_px >= 30.0) and reach_delta > 15.0)
    return {
        "success": success,
        "reward": round(float(reward), 3),
        "stage_reached": stage,
        "failure_mode": failure_mode,
        "metrics": {
            "reach_delta_deg": round(float(reach_delta), 2),
            "gripper_min_pct": round(float(min(grippers)), 2),
            "gripper_max_pct": round(float(max(grippers)), 2),
            "max_gripper_load": int(max(gripper_loads)),
            "max_shoulder_lift_load": int(max(lift_loads)),
            "visual_motion_score": round(visual_motion_score, 2),
            "ball_start_front": ball_start_front,
            "ball_end_front": ball_end_front,
            "ball_displacement_px": round(ball_displacement_px, 1),
            "wrist_ball_aligned_samples": wrist_centered_frames,
        },
    }
