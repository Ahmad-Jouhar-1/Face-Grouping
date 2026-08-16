"""
Face alignment.

Uses a 5-point landmark subset (eye centers, nose tip, mouth corners)
derived from MediaPipe's 478-point FaceLandmarker output to compute a
similarity transform onto the standard ArcFace 112x112 reference
template, producing the 112x112 aligned RGB crop expected by the
production IR-SE50 embedding model.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from face_grouping.detection.landmarker import FaceLandmarks

# MediaPipe FaceMesh (478-point) landmark indices used to approximate the
# standard 5-point ArcFace alignment set. These are well-documented,
# commonly used indices across MediaPipe-based face-alignment
# implementations (eye corner pairs, nose tip, mouth corners).
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
NOSE_TIP = 1
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# Standard ArcFace 112x112 reference (destination) points, published as
# part of the original ArcFace/InsightFace alignment code and used
# broadly across the ecosystem.
ARCFACE_REFERENCE_112 = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float32,
)

TARGET_SIZE = 112


@dataclass
class AlignedFace:
    image: np.ndarray  # (112, 112, 3) aligned RGB crop, uint8
    yaw_estimate: float  # rough pose signal, refined later in quality scoring (Point 2)


def _five_point_subset(landmarks: FaceLandmarks) -> np.ndarray:
    pts = landmarks.points
    left_eye = pts[list(LEFT_EYE_CORNERS)].mean(axis=0)
    right_eye = pts[list(RIGHT_EYE_CORNERS)].mean(axis=0)
    nose_tip = pts[NOSE_TIP]
    mouth_left = pts[MOUTH_LEFT]
    mouth_right = pts[MOUTH_RIGHT]
    return np.array(
        [left_eye, right_eye, nose_tip, mouth_left, mouth_right], dtype=np.float32
    )


def align_face(image_rgb: np.ndarray, landmarks: FaceLandmarks) -> AlignedFace:
    """
    Computes a similarity transform (rotation/scale/translation only, no
    shear or perspective distortion) mapping the detected 5-point subset
    onto the standard ArcFace 112x112 reference template, and warps the
    original image accordingly.
    """
    src_pts = _five_point_subset(landmarks)

    transform, _ = cv2.estimateAffinePartial2D(
        src_pts, ARCFACE_REFERENCE_112, method=cv2.LMEDS
    )
    if transform is None:
        raise ValueError("Could not estimate alignment transform for this face.")

    aligned = cv2.warpAffine(
        image_rgb, transform, (TARGET_SIZE, TARGET_SIZE), borderValue=(0, 0, 0)
    )

    # Rough yaw estimate from horizontal eye-distance vs. nose offset.
    # This is a lightweight placeholder pose signal -- it gets formalized
    # as part of the quality-score pose component when quality.py is
    # built in the next implementation step.
    left_eye, right_eye = src_pts[0], src_pts[1]
    eye_center = (left_eye + right_eye) / 2
    eye_dist = float(np.linalg.norm(right_eye - left_eye)) + 1e-6
    nose = src_pts[2]
    yaw_estimate = float((nose[0] - eye_center[0]) / eye_dist)

    return AlignedFace(image=aligned, yaw_estimate=yaw_estimate)
