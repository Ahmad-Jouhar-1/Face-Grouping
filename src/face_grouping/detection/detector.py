"""
MediaPipe Face Detector wrapper.

Wraps the MediaPipe Tasks API FaceDetector (BlazeFace-based) to produce
face bounding box detections above the confidence threshold locked in
Point 1 of the face-grouping design (default: 0.5). Faces below this
threshold are never returned by this wrapper at all -- filtering happens
inside MediaPipe itself via `min_detection_confidence`, matching our
locked decision that detection is kept permissive and the real quality
filtering happens downstream (Point 2).
"""
from dataclasses import dataclass
from typing import List

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


@dataclass
class FaceDetection:
    """A single detected face, in original image pixel coordinates."""

    x: int
    y: int
    width: int
    height: int
    confidence: float

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height


class FaceDetectorWrapper:
    """Thin wrapper around MediaPipe's Tasks API FaceDetector."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        base_options = BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=confidence_threshold,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def detect(self, image_rgb) -> List[FaceDetection]:
        """
        Run detection on an RGB numpy image (H, W, 3), uint8.

        Returns faces already filtered at/above confidence_threshold --
        no separate discard step is needed on the caller's side since
        MediaPipe applies min_detection_confidence internally.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._detector.detect(mp_image)

        detections = []
        for det in result.detections:
            bbox = det.bounding_box
            confidence = det.categories[0].score if det.categories else 0.0
            detections.append(
                FaceDetection(
                    x=bbox.origin_x,
                    y=bbox.origin_y,
                    width=bbox.width,
                    height=bbox.height,
                    confidence=confidence,
                )
            )
        return detections

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
