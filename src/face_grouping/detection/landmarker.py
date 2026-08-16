"""MediaPipe Face Landmarker wrapper with image-wide one-to-one association.

Each detector box is still landmarked on a padded local crop for robustness on
small faces.  Unlike the older per-detection implementation, candidates from
all detector crops are pooled in original-image coordinates, de-duplicated,
and assigned one-to-one across the whole photo.  The same physical face mesh
can therefore never be used for two detections.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from face_grouping.detection.association import (
    associate_candidates_one_to_one,
    select_landmark_candidate,
)
from face_grouping.detection.detector import FaceDetection


@dataclass
class FaceLandmarks:
    """478 (x, y) landmark points, in original image pixel coordinates."""

    points: np.ndarray  # shape (478, 2), float32


class FaceLandmarkerWrapper:
    """Thin wrapper around MediaPipe Tasks FaceLandmarker."""

    def __init__(
        self,
        model_path: str,
        padding_ratio: float = 0.3,
        max_faces_per_crop: int = 4,
    ):
        if max_faces_per_crop < 1:
            raise ValueError("max_faces_per_crop must be >= 1")

        self.padding_ratio = padding_ratio
        self.max_faces_per_crop = max_faces_per_crop
        base_options = BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=max_faces_per_crop,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def _crop_with_padding(self, image_rgb: np.ndarray, det: FaceDetection):
        h, w = image_rgb.shape[:2]
        pad_x = int(det.width * self.padding_ratio)
        pad_y = int(det.height * self.padding_ratio)

        x1 = max(0, det.x - pad_x)
        y1 = max(0, det.y - pad_y)
        x2 = min(w, det.x2 + pad_x)
        y2 = min(h, det.y2 + pad_y)

        crop = np.ascontiguousarray(image_rgb[y1:y2, x1:x2])
        return crop, x1, y1

    @staticmethod
    def _map_candidate_to_original(
        landmarks, crop_w: int, crop_h: int, offset_x: int, offset_y: int
    ) -> np.ndarray:
        return np.array(
            [
                (lm.x * crop_w + offset_x, lm.y * crop_h + offset_y)
                for lm in landmarks
            ],
            dtype=np.float32,
        )

    def _detect_candidates_for_crop(
        self, image_rgb: np.ndarray, detection: FaceDetection
    ) -> list[np.ndarray]:
        crop, offset_x, offset_y = self._crop_with_padding(image_rgb, detection)
        if crop.size == 0:
            return []

        crop_h, crop_w = crop.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return []

        return [
            self._map_candidate_to_original(
                landmarks, crop_w, crop_h, offset_x, offset_y
            )
            for landmarks in result.face_landmarks
        ]

    def detect_for_detections(
        self,
        image_rgb: np.ndarray,
        detections: Sequence[FaceDetection],
    ) -> list[Optional[FaceLandmarks]]:
        """Landmark all detections jointly and enforce one-to-one association.

        Local MediaPipe candidates are collected from every padded detector
        crop, mapped back to original-image coordinates, and then associated as
        one image-wide set.  One mesh can be consumed by at most one detection.
        """
        if not detections:
            return []

        pooled_candidates: list[np.ndarray] = []
        for detection in detections:
            pooled_candidates.extend(
                self._detect_candidates_for_crop(image_rgb, detection)
            )

        selected = associate_candidates_one_to_one(pooled_candidates, detections)
        return [
            FaceLandmarks(points=points) if points is not None else None
            for points in selected
        ]

    def detect_for_face(
        self, image_rgb: np.ndarray, detection: FaceDetection
    ) -> Optional[FaceLandmarks]:
        """Backward-compatible single-detection API.

        Production photo ingestion uses ``detect_for_detections`` so nearby
        detections compete globally.  This method remains useful for isolated
        single-face tools.
        """
        result = self.detect_for_detections(image_rgb, [detection])
        return result[0] if result else None

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


__all__ = [
    "FaceLandmarks",
    "FaceLandmarkerWrapper",
    "select_landmark_candidate",
]
