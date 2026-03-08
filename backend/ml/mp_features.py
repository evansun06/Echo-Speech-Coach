from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = "evan_test.mp4"   # can also be .mov if OpenCV can decode it
OUTPUT_DIR = Path("visual_outputs")
WINDOW_SEC = 1.0
TARGET_FPS = 10.0  # process fewer frames for speed; 5-10 is usually enough

ML_DIR = Path(__file__).resolve().parent
MODELS_DIR = ML_DIR / "models"

FACE_MODEL_PATH = str(MODELS_DIR / "face_landmarker.task")
POSE_MODEL_PATH = str(MODELS_DIR / "pose_landmarker_full.task")
HAND_MODEL_PATH = str(MODELS_DIR / "hand_landmarker.task")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TYPED POINTS
# ============================================================

@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


def point_from_landmark(landmark) -> Point2D:
    return Point2D(x=float(landmark.x), y=float(landmark.y))


def dist2d(a: Optional[Point2D], b: Optional[Point2D]) -> float:
    if a is None or b is None:
        return float("nan")
    return math.hypot(a.x - b.x, a.y - b.y)


def mean_point(points: Iterable[Optional[Point2D]]) -> Optional[Point2D]:
    valid = [p for p in points if p is not None]
    if not valid:
        return None
    return Point2D(
        x=float(np.mean([p.x for p in valid])),
        y=float(np.mean([p.y for p in valid])),
    )


def clamp01(x: float) -> float:
    if np.isnan(x):
        return x
    return max(0.0, min(1.0, x))


def nanmean_safe(values) -> float:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def nanstd_safe(values) -> float:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanstd(arr))


# ============================================================
# TASK SETUP
# ============================================================

def create_face_landmarker(model_path: str):
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def create_pose_landmarker(model_path: str):
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def create_hand_landmarker(model_path: str):
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return vision.HandLandmarker.create_from_options(options)


# ============================================================
# RESULT HELPERS
# ============================================================

def first_face(face_result):
    faces = getattr(face_result, "face_landmarks", None)
    if not faces:
        return None
    return faces[0]


def first_pose(pose_result):
    poses = getattr(pose_result, "pose_landmarks", None)
    if not poses:
        return None
    return poses[0]


def split_hands_by_handedness(hand_result):
    """
    Returns (left_hand_landmarks, right_hand_landmarks).
    """
    hand_landmarks = getattr(hand_result, "hand_landmarks", None)
    handedness = getattr(hand_result, "handedness", None)

    left_hand = None
    right_hand = None

    if not hand_landmarks or not handedness:
        return left_hand, right_hand

    for i, hand_lms in enumerate(hand_landmarks):
        if i >= len(handedness) or not handedness[i]:
            continue

        label = handedness[i][0].category_name.lower()
        if label == "left":
            left_hand = hand_lms
        elif label == "right":
            right_hand = hand_lms

    return left_hand, right_hand


def get_point(landmarks, idx: int) -> Optional[Point2D]:
    if landmarks is None:
        return None
    if idx < 0 or idx >= len(landmarks):
        return None
    lm = landmarks[idx]
    if lm is None:
        return None
    return point_from_landmark(lm)


# ============================================================
# FEATURE HELPERS
# ============================================================

def get_face_scale(face_landmarks) -> float:
    left_eye_outer = get_point(face_landmarks, 33)
    right_eye_outer = get_point(face_landmarks, 263)
    return dist2d(left_eye_outer, right_eye_outer)


def get_shoulder_width(pose_landmarks) -> float:
    left_shoulder = get_point(pose_landmarks, 11)
    right_shoulder = get_point(pose_landmarks, 12)
    return dist2d(left_shoulder, right_shoulder)


def compute_forward_attention(face_landmarks) -> float:
    """
    Proxy for forward attention:
    smaller nose horizontal offset from eye midpoint => more forward-facing.
    """
    nose = get_point(face_landmarks, 1)
    left_eye_outer = get_point(face_landmarks, 33)
    right_eye_outer = get_point(face_landmarks, 263)

    if nose is None or left_eye_outer is None or right_eye_outer is None:
        return float("nan")

    eye_mid = mean_point([left_eye_outer, right_eye_outer])
    eye_dist = dist2d(left_eye_outer, right_eye_outer)

    if eye_mid is None or np.isnan(eye_dist) or eye_dist < 1e-6:
        return float("nan")

    horiz_dev = abs(nose.x - eye_mid.x) / eye_dist
    score = 1.0 - min(horiz_dev / 0.35, 1.0)
    return clamp01(score)


def compute_facial_activity(face_landmarks, prev_state):
    """
    This is your per-frame facial expressiveness building block.
    Later we aggregate it into window-level expressiveness_score.
    """
    if face_landmarks is None:
        return float("nan"), None

    face_scale = get_face_scale(face_landmarks)
    if np.isnan(face_scale) or face_scale < 1e-6:
        return float("nan"), None

    upper_lip = get_point(face_landmarks, 13)
    lower_lip = get_point(face_landmarks, 14)
    left_mouth = get_point(face_landmarks, 61)
    right_mouth = get_point(face_landmarks, 291)
    left_brow = get_point(face_landmarks, 65)
    right_brow = get_point(face_landmarks, 295)
    left_eye_upper = get_point(face_landmarks, 159)
    right_eye_upper = get_point(face_landmarks, 386)

    needed = [
        upper_lip, lower_lip, left_mouth, right_mouth,
        left_brow, right_brow, left_eye_upper, right_eye_upper
    ]
    if any(p is None for p in needed):
        return float("nan"), None

    mouth_open = dist2d(upper_lip, lower_lip) / face_scale
    mouth_width = dist2d(left_mouth, right_mouth) / face_scale
    brow_raise = (
        abs(left_brow.y - left_eye_upper.y) / face_scale
        + abs(right_brow.y - right_eye_upper.y) / face_scale
    ) / 2.0

    curr = {
        "mouth_open": mouth_open,
        "mouth_width": mouth_width,
        "brow_raise": brow_raise,
    }

    if prev_state is None:
        return 0.0, curr

    deltas = [
        abs(curr["mouth_open"] - prev_state["mouth_open"]),
        abs(curr["mouth_width"] - prev_state["mouth_width"]),
        abs(curr["brow_raise"] - prev_state["brow_raise"]),
    ]
    activity = float(np.mean(deltas))
    return activity, curr


def compute_posture_deviation(pose_landmarks) -> float:
    """
    Deviation from neutral posture.
    """
    nose = get_point(pose_landmarks, 0)
    left_shoulder = get_point(pose_landmarks, 11)
    right_shoulder = get_point(pose_landmarks, 12)
    left_hip = get_point(pose_landmarks, 23)
    right_hip = get_point(pose_landmarks, 24)

    if any(p is None for p in [nose, left_shoulder, right_shoulder, left_hip, right_hip]):
        return float("nan")

    shoulder_width = dist2d(left_shoulder, right_shoulder)
    if np.isnan(shoulder_width) or shoulder_width < 1e-6:
        return float("nan")

    shoulder_mid = mean_point([left_shoulder, right_shoulder])
    hip_mid = mean_point([left_hip, right_hip])

    if shoulder_mid is None or hip_mid is None:
        return float("nan")

    shoulder_tilt = abs(left_shoulder.y - right_shoulder.y) / shoulder_width
    torso_lean = abs(shoulder_mid.x - hip_mid.x) / shoulder_width
    head_offset = abs(nose.x - shoulder_mid.x) / shoulder_width

    return float(np.mean([shoulder_tilt, torso_lean, head_offset]))


def compute_hand_motion(
    left_hand_landmarks,
    right_hand_landmarks,
    prev_left_wrist: Optional[Point2D],
    prev_right_wrist: Optional[Point2D],
    shoulder_width: float,
):
    """
    Average wrist displacement normalized by shoulder width.
    """
    if np.isnan(shoulder_width) or shoulder_width < 1e-6:
        return float("nan"), None, None

    left_wrist = get_point(left_hand_landmarks, 0)
    right_wrist = get_point(right_hand_landmarks, 0)

    motions = []
    if left_wrist is not None and prev_left_wrist is not None:
        motions.append(dist2d(left_wrist, prev_left_wrist) / shoulder_width)

    if right_wrist is not None and prev_right_wrist is not None:
        motions.append(dist2d(right_wrist, prev_right_wrist) / shoulder_width)

    return nanmean_safe(motions), left_wrist, right_wrist


def compute_hand_near_face(left_hand_landmarks, right_hand_landmarks, face_landmarks) -> float:
    nose = get_point(face_landmarks, 1)
    left_cheek = get_point(face_landmarks, 234)
    right_cheek = get_point(face_landmarks, 454)

    if any(p is None for p in [nose, left_cheek, right_cheek]):
        return float("nan")

    face_scale = dist2d(left_cheek, right_cheek)
    if np.isnan(face_scale) or face_scale < 1e-6:
        return float("nan")

    wrists = [get_point(left_hand_landmarks, 0), get_point(right_hand_landmarks, 0)]

    checks = []
    for wrist in wrists:
        if wrist is not None:
            checks.append(dist2d(wrist, nose) / face_scale < 1.0)

    if not checks:
        return float("nan")

    return 1.0 if any(checks) else 0.0


def compute_body_sway(pose_landmarks, prev_shoulder_mid: Optional[Point2D], shoulder_width: float):
    left_shoulder = get_point(pose_landmarks, 11)
    right_shoulder = get_point(pose_landmarks, 12)

    if left_shoulder is None or right_shoulder is None:
        return float("nan"), None

    shoulder_mid = mean_point([left_shoulder, right_shoulder])
    if shoulder_mid is None:
        return float("nan"), None

    if prev_shoulder_mid is None or np.isnan(shoulder_width) or shoulder_width < 1e-6:
        return 0.0, shoulder_mid

    sway = dist2d(shoulder_mid, prev_shoulder_mid) / shoulder_width
    return sway, shoulder_mid


def compute_fidget_index(hand_motion_energy: float, body_sway: float, hand_near_face: float) -> float:
    vals = []
    if not np.isnan(hand_motion_energy):
        vals.append(hand_motion_energy)
    if not np.isnan(body_sway):
        vals.append(body_sway)
    if not np.isnan(hand_near_face):
        vals.append(0.5 * hand_near_face)

    if not vals:
        return float("nan")
    return float(np.mean(vals))


# ============================================================
# EXTRACTION
# ============================================================

def extract_frame_features(video_path: str) -> tuple[pd.DataFrame, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps is None or native_fps <= 0:
        native_fps = 30.0

    frame_step = max(1, int(round(native_fps / TARGET_FPS)))
    effective_fps = native_fps / frame_step

    frame_records = []
    prev_left_wrist = None
    prev_right_wrist = None
    prev_shoulder_mid = None
    prev_face_state = None

    frame_idx = 0
    processed_idx = 0

    with create_face_landmarker(FACE_MODEL_PATH) as face_landmarker, \
         create_pose_landmarker(POSE_MODEL_PATH) as pose_landmarker, \
         create_hand_landmarker(HAND_MODEL_PATH) as hand_landmarker:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            timestamp_ms = int((frame_idx / native_fps) * 1000)
            timestamp_sec = frame_idx / native_fps

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            face_landmarks = first_face(face_result)
            pose_landmarks = first_pose(pose_result)
            left_hand_landmarks, right_hand_landmarks = split_hands_by_handedness(hand_result)

            shoulder_width = get_shoulder_width(pose_landmarks)

            forward_attention = compute_forward_attention(face_landmarks)
            facial_activity, prev_face_state = compute_facial_activity(face_landmarks, prev_face_state)
            posture_deviation = compute_posture_deviation(pose_landmarks)

            hand_motion_energy, prev_left_wrist, prev_right_wrist = compute_hand_motion(
                left_hand_landmarks,
                right_hand_landmarks,
                prev_left_wrist,
                prev_right_wrist,
                shoulder_width,
            )

            hand_near_face = compute_hand_near_face(
                left_hand_landmarks,
                right_hand_landmarks,
                face_landmarks,
            )

            body_sway, prev_shoulder_mid = compute_body_sway(
                pose_landmarks,
                prev_shoulder_mid,
                shoulder_width,
            )

            fidget_index = compute_fidget_index(
                hand_motion_energy=hand_motion_energy,
                body_sway=body_sway,
                hand_near_face=hand_near_face,
            )

            frame_records.append({
                "processed_frame_idx": processed_idx,
                "source_frame_idx": frame_idx,
                "timestamp_sec": timestamp_sec,
                "hand_motion_energy": hand_motion_energy,
                "forward_attention_ratio": forward_attention,
                "facial_activity": facial_activity,
                "posture_deviation": posture_deviation,
                "fidget_index": fidget_index,
                "body_sway": body_sway,
                "hand_near_face": hand_near_face,
            })

            frame_idx += 1
            processed_idx += 1

    cap.release()
    return pd.DataFrame(frame_records), effective_fps


# ============================================================
# WINDOW AGGREGATION
# ============================================================

def aggregate_windows(frame_df: pd.DataFrame, window_sec: float) -> pd.DataFrame:
    if frame_df.empty:
        return pd.DataFrame()

    df = frame_df.copy()
    df["window_id"] = (df["timestamp_sec"] // window_sec).astype(int)

    rows = []
    for window_id, g in df.groupby("window_id"):
        window_id = int(window_id)
        start_sec = window_id * window_sec
        end_sec = start_sec + window_sec

        hand_motion_energy = nanmean_safe(g["hand_motion_energy"])
        forward_attention_ratio = nanmean_safe(g["forward_attention_ratio"])
        posture_deviation = nanmean_safe(g["posture_deviation"])
        fidget_index = nanmean_safe(g["fidget_index"])

        # window-level expressiveness_score:
        # mostly face activity, with a small gesture contribution
        face_expr = nanmean_safe(g["facial_activity"])
        hand_expr = nanmean_safe(g["hand_motion_energy"])

        face_expr_norm = min(face_expr / 0.02, 1.0) if not np.isnan(face_expr) else float("nan")
        hand_expr_norm = min(hand_expr / 0.04, 1.0) if not np.isnan(hand_expr) else float("nan")
        expressiveness_score = nanmean_safe([face_expr_norm, 0.35 * hand_expr_norm])

        # simple gesture event proxy for overall gesture_rate_per_min
        gesture_event = 1 if (not np.isnan(hand_motion_energy) and hand_motion_energy > 0.015) else 0

        rows.append({
            "window_id": int(window_id),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "hand_motion_energy": hand_motion_energy,
            "forward_attention_ratio": forward_attention_ratio,
            "expressiveness_score": expressiveness_score,
            "posture_deviation": posture_deviation,
            "fidget_index": fidget_index,
            "gesture_event": gesture_event,
        })

    return pd.DataFrame(rows)


# ============================================================
# OVERALL FEATURES
# ============================================================

def minmax_invert_score(value: float, max_ref: float) -> float:
    if np.isnan(value):
        return float("nan")
    v = min(max(value / max_ref, 0.0), 1.0)
    return 1.0 - v


def compute_overall_features(frame_df: pd.DataFrame, window_df: pd.DataFrame) -> dict:
    if frame_df.empty:
        return {}

    total_duration_sec = float(frame_df["timestamp_sec"].max() - frame_df["timestamp_sec"].min())
    total_duration_min = max(total_duration_sec / 60.0, 1e-6)

    forward_attention_ratio = nanmean_safe(frame_df["forward_attention_ratio"])

    gesture_events = int(window_df["gesture_event"].sum()) if not window_df.empty else 0
    gesture_rate_per_min = gesture_events / total_duration_min

    # steadiness is better represented by posture stability
    posture_variability = nanstd_safe(frame_df["posture_deviation"])
    posture_stability_score = minmax_invert_score(posture_variability, max_ref=0.06)

    face_expr = nanmean_safe(frame_df["facial_activity"])
    hand_expr = nanmean_safe(frame_df["hand_motion_energy"])
    face_expr_norm = min(face_expr / 0.02, 1.0) if not np.isnan(face_expr) else float("nan")
    hand_expr_norm = min(hand_expr / 0.04, 1.0) if not np.isnan(hand_expr) else float("nan")
    expressiveness_score = nanmean_safe([face_expr_norm, 0.35 * hand_expr_norm])

    mean_posture_deviation = nanmean_safe(frame_df["posture_deviation"])
    mean_fidget_index = nanmean_safe(frame_df["fidget_index"])

    return {
        "forward_attention_ratio": forward_attention_ratio,
        "gesture_rate_per_min": gesture_rate_per_min,
        "posture_stability_score": posture_stability_score,
        "expressiveness_score": expressiveness_score,
        "mean_posture_deviation": mean_posture_deviation,
        "mean_fidget_index": mean_fidget_index,
        "total_duration_sec": total_duration_sec,
    }


# ============================================================
# SAVE
# ============================================================

def save_outputs(frame_df: pd.DataFrame, window_df: pd.DataFrame, overall_dict: dict, output_dir: Path):
    frame_csv = output_dir / "frame_features.csv"
    window_csv = output_dir / "window_features.csv"
    overall_json = output_dir / "overall_features.json"

    frame_df.to_csv(frame_csv, index=False)
    window_df.to_csv(window_csv, index=False)

    with open(overall_json, "w", encoding="utf-8") as f:
        json.dump(overall_dict, f, indent=2)

    return frame_csv, window_csv, overall_json


# ============================================================
# MAIN
# ============================================================

def main():
    frame_df, effective_fps = extract_frame_features(VIDEO_PATH)
    window_df = aggregate_windows(frame_df, WINDOW_SEC)
    overall = compute_overall_features(frame_df, window_df)
    frame_csv, window_csv, overall_json = save_outputs(frame_df, window_df, overall, OUTPUT_DIR)

    print("Done.")
    print(f"Processed FPS: {effective_fps:.2f}")
    print(f"Saved frame features to:   {frame_csv}")
    print(f"Saved window features to:  {window_csv}")
    print(f"Saved overall features to: {overall_json}")
    print("\nOverall summary:")
    for k, v in overall.items():
        print(f"  {k}: {v}")
    
    frame_df = pd.read_csv("visual_outputs/frame_features.csv")
    window_df = pd.read_csv("visual_outputs/window_features.csv")
    print(frame_df.head())
    print(window_df[[#"window_id", 
                    #"hand_motion_energy",
                    "forward_attention_ratio",
                    "expressiveness_score",
                    "posture_deviation",
                    # "fidget_index",
                    # "gesture_event"
                    ]])


if __name__ == "__main__":
    main()
