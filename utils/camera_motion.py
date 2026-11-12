import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


@dataclass
class CameraMotionCheckResult:
    is_static: bool
    reason: str
    sampled_frames: int
    checked_pairs: int
    valid_pairs: int
    moving_pairs: int
    used_full_frame_fallback_pairs: int
    max_translation_px: float
    median_translation_px: float
    max_rotation_deg: float
    median_rotation_deg: float
    max_scale_delta: float

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key, value in result.items():
            if isinstance(value, np.generic):
                result[key] = value.item()
        return result


def detect_static_camera(
    video_path: str,
    *,
    downscale_width: int = 320,
    max_pairs: int = 6,
    border_fraction: float = 0.2,
    max_corners: int = 250,
    min_tracked_points: int = 14,
) -> CameraMotionCheckResult:
    """
    Quickly classify whether a recording was captured with a mostly static camera.

    The detector samples a few frames, tracks features with Lucas-Kanade optical flow,
    and estimates a robust affine transform between frame pairs. It prioritizes border
    features so body motion in the center of the image is less likely to trigger a
    false "moving camera" result.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video for camera motion check: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames < 3:
            return CameraMotionCheckResult(
                is_static=True,
                reason="Video too short for camera motion check; allowing processing to continue.",
                sampled_frames=max(n_frames, 0),
                checked_pairs=max(n_frames - 1, 0),
                valid_pairs=0,
                moving_pairs=0,
                used_full_frame_fallback_pairs=0,
                max_translation_px=0.0,
                median_translation_px=0.0,
                max_rotation_deg=0.0,
                median_rotation_deg=0.0,
                max_scale_delta=0.0,
            )

        frame_indices = _sample_frame_indices(n_frames=n_frames, fps=fps, max_pairs=max_pairs)
        frames = _load_sampled_frames(
            cap=cap, frame_indices=frame_indices, downscale_width=downscale_width
        )
    finally:
        cap.release()

    if len(frames) < 3:
        return CameraMotionCheckResult(
            is_static=True,
            reason="Could not sample enough frames for camera motion check; allowing processing to continue.",
            sampled_frames=len(frames),
            checked_pairs=max(len(frames) - 1, 0),
            valid_pairs=0,
            moving_pairs=0,
            used_full_frame_fallback_pairs=0,
            max_translation_px=0.0,
            median_translation_px=0.0,
            max_rotation_deg=0.0,
            median_rotation_deg=0.0,
            max_scale_delta=0.0,
        )

    pair_metrics: List[Dict[str, Any]] = []
    for prev_gray, curr_gray in zip(frames, frames[1:]):
        metric = _estimate_pair_motion(
            prev_gray=prev_gray,
            curr_gray=curr_gray,
            border_fraction=border_fraction,
            max_corners=max_corners,
            min_tracked_points=min_tracked_points,
        )
        pair_metrics.append(metric)

    valid_pairs = [metric for metric in pair_metrics if metric["valid"]]
    if not valid_pairs:
        return CameraMotionCheckResult(
            is_static=True,
            reason="Insufficient stable background features for camera motion check; allowing processing to continue.",
            sampled_frames=len(frames),
            checked_pairs=len(pair_metrics),
            valid_pairs=0,
            moving_pairs=0,
            used_full_frame_fallback_pairs=0,
            max_translation_px=0.0,
            median_translation_px=0.0,
            max_rotation_deg=0.0,
            median_rotation_deg=0.0,
            max_scale_delta=0.0,
        )

    translations = np.array([metric["translation_px"] for metric in valid_pairs], dtype=float)
    rotations = np.array([metric["rotation_deg"] for metric in valid_pairs], dtype=float)
    scale_deltas = np.array([metric["scale_delta"] for metric in valid_pairs], dtype=float)
    moving_pairs = sum(1 for metric in valid_pairs if metric["is_moving"])
    full_frame_pairs = sum(1 for metric in valid_pairs if not metric["used_border_mask"])
    strong_motion = any(metric["is_strong_motion"] for metric in valid_pairs)
    sustained_motion = moving_pairs >= max(2, math.ceil(0.5 * len(valid_pairs)))
    is_static = not (strong_motion or sustained_motion)

    if is_static:
        reason = "No meaningful camera movement detected."
    else:
        reason = (
            "Camera movement detected. "
            f"Median translation {np.median(translations):.2f}px, "
            f"max translation {np.max(translations):.2f}px, "
            f"max rotation {np.max(rotations):.2f}deg."
        )

    return CameraMotionCheckResult(
        is_static=is_static,
        reason=reason,
        sampled_frames=len(frames),
        checked_pairs=len(pair_metrics),
        valid_pairs=len(valid_pairs),
        moving_pairs=moving_pairs,
        used_full_frame_fallback_pairs=full_frame_pairs,
        max_translation_px=float(np.max(translations)),
        median_translation_px=float(np.median(translations)),
        max_rotation_deg=float(np.max(rotations)),
        median_rotation_deg=float(np.median(rotations)),
        max_scale_delta=float(np.max(scale_deltas)),
    )


def _sample_frame_indices(n_frames: int, fps: float, max_pairs: int) -> np.ndarray:
    duration_s = n_frames / max(fps, 1.0)
    desired_pairs = int(np.clip(math.ceil(duration_s / 0.75), 2, max_pairs))
    num_frames = desired_pairs + 1
    frame_indices = np.linspace(0, n_frames - 1, num=num_frames, dtype=int)
    return np.unique(frame_indices)


def _load_sampled_frames(
    cap: cv2.VideoCapture, frame_indices: np.ndarray, downscale_width: int
) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frames.append(_preprocess_frame(frame, downscale_width=downscale_width))
    return frames


def _preprocess_frame(frame: np.ndarray, downscale_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width > downscale_width:
        scale = downscale_width / float(width)
        frame = cv2.resize(
            frame,
            (downscale_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _estimate_pair_motion(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    *,
    border_fraction: float,
    max_corners: int,
    min_tracked_points: int,
) -> Dict[str, Any]:
    border_mask = _make_border_mask(prev_gray.shape, border_fraction=border_fraction)
    metric = _track_and_estimate_transform(
        prev_gray=prev_gray,
        curr_gray=curr_gray,
        mask=border_mask,
        max_corners=max_corners,
        min_tracked_points=min_tracked_points,
        thresholds={"translation_px": 3.5, "rotation_deg": 0.8, "scale_delta": 0.010},
        used_border_mask=True,
    )
    if metric["valid"]:
        return metric

    return _track_and_estimate_transform(
        prev_gray=prev_gray,
        curr_gray=curr_gray,
        mask=None,
        max_corners=max_corners,
        min_tracked_points=min_tracked_points,
        thresholds={"translation_px": 6.0, "rotation_deg": 1.5, "scale_delta": 0.015},
        used_border_mask=False,
    )


def _make_border_mask(shape: tuple[int, int], border_fraction: float) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    border_x = max(1, int(round(width * border_fraction)))
    border_y = max(1, int(round(height * border_fraction)))
    mask[:border_y, :] = 255
    mask[-border_y:, :] = 255
    mask[:, :border_x] = 255
    mask[:, -border_x:] = 255
    return mask


def _track_and_estimate_transform(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    *,
    mask: Optional[np.ndarray],
    max_corners: int,
    min_tracked_points: int,
    thresholds: Dict[str, float],
    used_border_mask: bool,
) -> Dict[str, Any]:
    prev_points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
        mask=mask,
    )
    if prev_points is None or len(prev_points) < min_tracked_points:
        return _invalid_metric("Not enough trackable points", used_border_mask)

    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if curr_points is None or status is None:
        return _invalid_metric("Optical flow failed", used_border_mask)

    tracked = status.reshape(-1).astype(bool)
    if tracked.sum() < min_tracked_points:
        return _invalid_metric("Too few tracked points", used_border_mask)

    prev_tracked = prev_points.reshape(-1, 2)[tracked]
    curr_tracked = curr_points.reshape(-1, 2)[tracked]

    affine, inlier_mask = cv2.estimateAffinePartial2D(
        prev_tracked,
        curr_tracked,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.5,
        maxIters=2000,
        confidence=0.99,
    )
    if affine is None or inlier_mask is None:
        return _invalid_metric("Robust transform estimation failed", used_border_mask)

    inlier_count = int(inlier_mask.sum())
    if inlier_count < min_tracked_points:
        return _invalid_metric("Too few inliers", used_border_mask)

    a, b, tx = affine[0]
    c, d, ty = affine[1]
    scale_x = math.hypot(a, c)
    scale_y = math.hypot(b, d)
    scale = 0.5 * (scale_x + scale_y)
    translation_px = math.hypot(tx, ty)
    rotation_deg = abs(math.degrees(math.atan2(c, a)))
    scale_delta = abs(scale - 1.0)

    is_moving = (
        translation_px > thresholds["translation_px"]
        or rotation_deg > thresholds["rotation_deg"]
        or scale_delta > thresholds["scale_delta"]
    )
    is_strong_motion = (
        translation_px > thresholds["translation_px"] * 2.5
        or rotation_deg > thresholds["rotation_deg"] * 2.5
        or scale_delta > thresholds["scale_delta"] * 2.0
    )

    return {
        "valid": True,
        "reason": "",
        "used_border_mask": used_border_mask,
        "tracked_points": int(tracked.sum()),
        "inliers": inlier_count,
        "translation_px": float(translation_px),
        "rotation_deg": float(rotation_deg),
        "scale_delta": float(scale_delta),
        "is_moving": bool(is_moving),
        "is_strong_motion": bool(is_strong_motion),
    }


def _invalid_metric(reason: str, used_border_mask: bool) -> Dict[str, Any]:
    return {
        "valid": False,
        "reason": reason,
        "used_border_mask": used_border_mask,
        "tracked_points": 0,
        "inliers": 0,
        "translation_px": 0.0,
        "rotation_deg": 0.0,
        "scale_delta": 0.0,
        "is_moving": False,
        "is_strong_motion": False,
    }
