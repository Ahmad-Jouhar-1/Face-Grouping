"""
Quality gating.

Combines the four sub-scores into the final weighted quality_score and
applies the two gates locked in Point 2:
  - hard exclusion (face never embedded/clustered)
  - exemplar eligibility (face usable as a cluster representative)
"""
from dataclasses import dataclass

import numpy as np

from face_grouping.detection.detector import FaceDetection
from face_grouping.detection.landmarker import FaceLandmarks
from face_grouping.alignment.aligner import AlignedFace
from face_grouping.config import load_thresholds
from face_grouping.quality.scoring import (
    compute_size_score,
    compute_blur_score,
    compute_pose_score,
    compute_eye_openness_score,
)


@dataclass
class QualityResult:
    size_score: float
    blur_score: float
    pose_score: float
    eye_openness_score: float
    quality_score: float
    yaw_ratio: float
    pitch_ratio: float
    hard_excluded: bool
    hard_exclusion_reason: str  # empty string if not excluded
    exemplar_eligible: bool


def compute_face_quality(
    detection: FaceDetection,
    landmarks: FaceLandmarks,
    aligned_face: AlignedFace,
) -> QualityResult:
    cfg = load_thresholds()["quality"]
    weights = cfg["weights"]

    face_height_px = float(detection.height)

    size_score = compute_size_score(
        face_height_px,
        hard_floor_px=cfg["size_hard_floor_px"],
        reference_px=cfg["size_reference_px"],
    )
    blur_score = compute_blur_score(
        aligned_face.image, blur_reference_variance=cfg["blur_reference_variance"]
    )
    pose_score, yaw_ratio, pitch_ratio = compute_pose_score(
        landmarks, max_yaw_ratio=cfg["max_yaw_ratio"]
    )
    eye_openness_score = compute_eye_openness_score(
        landmarks, ear_closed=cfg["ear_closed"], ear_open=cfg["ear_open"]
    )

    quality_score = (
        weights["size"] * size_score
        + weights["blur"] * blur_score
        + weights["pose"] * pose_score
        + weights["eye_openness"] * eye_openness_score
    )

    # --- Hard exclusion gate (Point 2) ---
    # Combined score below threshold, OR either absolute floor fails
    # regardless of how the combined score looks -- a good blur/size
    # score should never compensate for an unusable extreme-profile pose,
    # and vice versa.
    hard_excluded = False
    reason = ""
    if quality_score < cfg["hard_exclusion_threshold"]:
        hard_excluded = True
        reason = f"quality_score {quality_score:.3f} below threshold"
    elif face_height_px <= cfg["size_hard_floor_px"]:
        hard_excluded = True
        reason = f"face height {face_height_px:.0f}px at/below hard floor"
    elif abs(yaw_ratio) >= cfg["max_yaw_ratio"]:
        hard_excluded = True
        reason = f"yaw_ratio {yaw_ratio:.2f} at/beyond absolute pose floor"

    # --- Exemplar eligibility gate (Point 2) ---
    exemplar_eligible = (not hard_excluded) and (
        quality_score >= cfg["exemplar_eligibility_threshold"]
    )

    return QualityResult(
        size_score=size_score,
        blur_score=blur_score,
        pose_score=pose_score,
        eye_openness_score=eye_openness_score,
        quality_score=float(quality_score),
        yaw_ratio=yaw_ratio,
        pitch_ratio=pitch_ratio,
        hard_excluded=hard_excluded,
        hard_exclusion_reason=reason,
        exemplar_eligible=exemplar_eligible,
    )
