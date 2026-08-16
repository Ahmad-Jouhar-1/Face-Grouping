"""
Quality sub-score computation.

Implements the four sub-scores from Point 2 of the locked design:
size, blur, pose, and eye-openness -- each normalized to [0, 1].

Normalization curve specifics (e.g. what raw blur variance maps to 1.0)
were not part of the original 20 locked points -- those were strategic
decisions (formula, weights, gate thresholds). These curves are
implementation-level engineering choices, made here with documented
reasoning and centralized in configs/thresholds.yaml so they're easy to
re-tune empirically later, consistent with how we treated T_match,
HDBSCAN params, etc.
"""
import numpy as np
import cv2

from face_grouping.detection.landmarker import FaceLandmarks
from face_grouping.alignment.aligner import (
    LEFT_EYE_CORNERS,
    RIGHT_EYE_CORNERS,
    NOSE_TIP,
    MOUTH_LEFT,
    MOUTH_RIGHT,
)

# Standard 6-point Eye Aspect Ratio (EAR) landmark indices, verified
# against multiple independent MediaPipe FaceMesh blink-detection
# implementations (consistent across sources).
# Order per eye: (p1=outer corner, p2, p3, p4=inner corner, p5, p6)
# EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
RIGHT_EYE_EAR_POINTS = (33, 159, 158, 133, 153, 145)
LEFT_EYE_EAR_POINTS = (362, 380, 374, 263, 386, 385)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def compute_size_score(
    face_height_px: float,
    hard_floor_px: float,
    reference_px: float,
) -> float:
    """
    0.0 at/below hard_floor_px, ramping linearly to 1.0 at/above
    reference_px. A face at exactly the Point 3 "reliable floor" (60px)
    lands in the low end of this range by design, so it gets caught by
    the hard_exclusion_threshold gate in combination with other
    sub-scores rather than needing a second separate size check there.
    """
    if face_height_px <= hard_floor_px:
        return 0.0
    if face_height_px >= reference_px:
        return 1.0
    return _clip01(
        (face_height_px - hard_floor_px) / (reference_px - hard_floor_px)
    )


def compute_blur_score(aligned_crop_rgb: np.ndarray, blur_reference_variance: float) -> float:
    """
    Laplacian variance on the grayscale aligned crop. Since alignment
    always produces a fixed 112x112 crop (Point 3), no per-resolution
    normalization is needed -- variance scale is consistent across faces.
    """
    gray = cv2.cvtColor(aligned_crop_rgb, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return _clip01(variance / blur_reference_variance)


def compute_pose_score(landmarks: FaceLandmarks, max_yaw_ratio: float):
    """
    Returns (pose_score, yaw_ratio, pitch_ratio).

    yaw_ratio: horizontal nose offset relative to eye-center, normalized
    by inter-eye distance -- the same geometric signal already used
    successfully as `yaw_estimate` in alignment.py, reused here as the
    primary driver of pose_score.

    pitch_ratio: vertical nose offset relative to eye-center, normalized
    by inter-eye distance -- computed and returned for visibility and
    future calibration, but NOT currently folded into pose_score, since
    we don't yet have a validated "frontal baseline" value for it (this
    is a documented simplification, not an oversight -- to be revisited
    once Phase 1 pose-variation (CPLFW-derived) validation data exists).
    """
    pts = landmarks.points
    left_eye = pts[list(LEFT_EYE_CORNERS)].mean(axis=0)
    right_eye = pts[list(RIGHT_EYE_CORNERS)].mean(axis=0)
    eye_center = (left_eye + right_eye) / 2
    eye_dist = float(np.linalg.norm(right_eye - left_eye)) + 1e-6
    nose = pts[NOSE_TIP]

    yaw_ratio = float((nose[0] - eye_center[0]) / eye_dist)
    pitch_ratio = float((nose[1] - eye_center[1]) / eye_dist)

    pose_score = _clip01(1.0 - abs(yaw_ratio) / max_yaw_ratio)
    return pose_score, yaw_ratio, pitch_ratio


def _ear(pts: np.ndarray, indices: tuple) -> float:
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in indices)
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4) + 1e-6
    return float(vertical / (2.0 * horizontal))


def compute_eye_openness_score(
    landmarks: FaceLandmarks, ear_closed: float, ear_open: float
) -> float:
    """
    Averages the Eye Aspect Ratio (EAR) across both eyes and normalizes
    to [0, 1] between the closed/open reference values.
    """
    pts = landmarks.points
    right_ear = _ear(pts, RIGHT_EYE_EAR_POINTS)
    left_ear = _ear(pts, LEFT_EYE_EAR_POINTS)
    avg_ear = (right_ear + left_ear) / 2.0
    return _clip01((avg_ear - ear_closed) / (ear_open - ear_closed))
