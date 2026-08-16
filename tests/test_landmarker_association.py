"""Regression tests for image-wide detector-to-landmark association."""
from dataclasses import dataclass

import numpy as np

from face_grouping.detection.association import (
    associate_candidates_one_to_one,
    deduplicate_landmark_candidates,
    select_landmark_candidate,
)


@dataclass
class FaceDetection:
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


def _candidate(center_x: float, center_y: float, radius: float = 20.0) -> np.ndarray:
    pts = np.tile(np.array([[center_x, center_y]], dtype=np.float32), (478, 1))
    # A simple square-ish mesh bbox plus central nose tip.
    pts[0] = [center_x - radius, center_y - radius]
    pts[2] = [center_x + radius, center_y + radius]
    pts[1] = [center_x, center_y]
    return pts


def test_neighbor_in_padding_cannot_steal_detection():
    detection = FaceDetection(x=100, y=100, width=100, height=100, confidence=0.9)
    target = _candidate(150, 150)
    neighbor_only_in_padding = _candidate(75, 150)
    selected = select_landmark_candidate([neighbor_only_in_padding, target], detection)
    assert selected is target


def test_wrong_neighbor_only_returns_none():
    detection = FaceDetection(x=100, y=100, width=100, height=100, confidence=0.9)
    neighbor_only_in_padding = _candidate(75, 150)
    selected = select_landmark_candidate([neighbor_only_in_padding], detection)
    assert selected is None


def test_duplicate_meshes_from_overlapping_crops_are_collapsed():
    first = _candidate(150, 150, 25)
    second = first.copy()
    second += np.array([1.0, -1.0], dtype=np.float32)
    unique = deduplicate_landmark_candidates([first, second])
    assert len(unique) == 1


def test_nested_parent_child_detections_get_different_faces():
    # Mirrors the gallery failure mode: the child's detector lies completely
    # inside a much larger adult detector.
    child_det = FaceDetection(x=201, y=335, width=153, height=153, confidence=0.88)
    adult_det = FaceDetection(x=0, y=139, width=379, height=379, confidence=0.82)

    child_mesh = _candidate(277, 411, radius=58)
    adult_mesh = _candidate(82, 319, radius=112)

    assigned = associate_candidates_one_to_one(
        [child_mesh, adult_mesh], [child_det, adult_det]
    )
    assert assigned[0] is child_mesh
    assert assigned[1] is adult_mesh


def test_same_child_mesh_cannot_be_reused_for_large_parent_detection():
    child_det = FaceDetection(x=64, y=440, width=142, height=142, confidence=0.66)
    adult_det = FaceDetection(x=0, y=425, width=228, height=228, confidence=0.55)
    child_mesh = _candidate(135, 511, radius=54)

    assigned = associate_candidates_one_to_one([child_mesh], [child_det, adult_det])
    assert assigned[0] is child_mesh
    assert assigned[1] is None
