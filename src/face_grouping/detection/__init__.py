"""Detection package exports, loaded lazily to keep geometry helpers lightweight."""
from __future__ import annotations

__all__ = [
    "FaceDetectorWrapper",
    "FaceDetection",
    "FaceLandmarkerWrapper",
    "FaceLandmarks",
]


def __getattr__(name):
    if name in {"FaceDetectorWrapper", "FaceDetection"}:
        from face_grouping.detection.detector import FaceDetectorWrapper, FaceDetection
        return {"FaceDetectorWrapper": FaceDetectorWrapper, "FaceDetection": FaceDetection}[name]
    if name in {"FaceLandmarkerWrapper", "FaceLandmarks"}:
        from face_grouping.detection.landmarker import FaceLandmarkerWrapper, FaceLandmarks
        return {"FaceLandmarkerWrapper": FaceLandmarkerWrapper, "FaceLandmarks": FaceLandmarks}[name]
    raise AttributeError(name)
