"""Pure geometry helpers for detector-to-landmark association.

The production landmarker may see the same physical face from more than one
padded detector crop.  This module pools those candidates, de-duplicates the
same mesh in original-image coordinates, and assigns at most one landmark mesh
to each detector and at most one detector to each mesh.

No identity/embedding threshold is used here.  All decisions are geometric.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np


class DetectionLike(Protocol):
    x: int
    y: int
    width: int
    height: int
    confidence: float

    @property
    def x2(self) -> int: ...

    @property
    def y2(self) -> int: ...

NOSE_TIP_INDEX = 1
_DUPLICATE_MESH_DISTANCE_RATIO = 0.15


@dataclass(frozen=True)
class CandidateGeometry:
    points: np.ndarray
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float
    width: float
    height: float
    diagonal: float


def candidate_geometry(points: np.ndarray) -> CandidateGeometry:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("landmark candidate must have shape (N, 2)")
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    x1, y1 = float(mins[0]), float(mins[1])
    x2, y2 = float(maxs[0]), float(maxs[1])
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    return CandidateGeometry(
        points=points,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        center_x=(x1 + x2) * 0.5,
        center_y=(y1 + y2) * 0.5,
        width=width,
        height=height,
        diagonal=float(np.hypot(width, height)),
    )


def _point_inside_detection(x: float, y: float, detection: DetectionLike) -> bool:
    return (
        float(detection.x) <= x <= float(detection.x2)
        and float(detection.y) <= y <= float(detection.y2)
    )


def _bbox_iou(candidate: CandidateGeometry, detection: DetectionLike) -> float:
    ix1 = max(candidate.x1, float(detection.x))
    iy1 = max(candidate.y1, float(detection.y))
    ix2 = min(candidate.x2, float(detection.x2))
    iy2 = min(candidate.y2, float(detection.y2))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0.0:
        return 0.0
    candidate_area = candidate.width * candidate.height
    detection_area = max(float(detection.width * detection.height), 1.0)
    union = candidate_area + detection_area - intersection
    return float(intersection / max(union, 1.0))


def _normalized_center_distance(candidate: CandidateGeometry, detection: DetectionLike) -> float:
    det_cx = (float(detection.x) + float(detection.x2)) * 0.5
    det_cy = (float(detection.y) + float(detection.y2)) * 0.5
    dx = (candidate.center_x - det_cx) / max(float(detection.width), 1.0)
    dy = (candidate.center_y - det_cy) / max(float(detection.height), 1.0)
    return float(np.hypot(dx, dy))


def _size_compatibility(candidate: CandidateGeometry, detection: DetectionLike) -> float:
    det_w = max(float(detection.width), 1.0)
    det_h = max(float(detection.height), 1.0)
    width_ratio = min(candidate.width, det_w) / max(candidate.width, det_w)
    height_ratio = min(candidate.height, det_h) / max(candidate.height, det_h)
    return float(np.sqrt(width_ratio * height_ratio))


def candidate_is_compatible(points: np.ndarray, detection: DetectionLike) -> bool:
    """Basic unpadded-box gate used before one-to-one ranking."""
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] <= NOSE_TIP_INDEX:
        return False
    geom = candidate_geometry(points)
    nose_x, nose_y = map(float, points[NOSE_TIP_INDEX])
    return _point_inside_detection(geom.center_x, geom.center_y, detection) and _point_inside_detection(
        nose_x, nose_y, detection
    )


def association_score(points: np.ndarray, detection: DetectionLike) -> Optional[float]:
    """Geometric compatibility score; higher means a better detector/mesh pair.

    IoU and scale compatibility are intentionally prominent.  This matters for
    nested detector boxes: a child's mesh can lie inside a large adult detector
    box, but its scale/overlap usually fits the child's own detector far better.
    """
    if not candidate_is_compatible(points, detection):
        return None
    geom = candidate_geometry(points)
    iou = _bbox_iou(geom, detection)
    size = _size_compatibility(geom, detection)
    center = 1.0 / (1.0 + _normalized_center_distance(geom, detection))
    return float(0.55 * iou + 0.25 * size + 0.20 * center)


def _duplicate_distance_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Median corresponding-landmark displacement normalized by face size."""
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2:
        return float("inf")
    ga = candidate_geometry(a)
    gb = candidate_geometry(b)
    scale = max(min(ga.diagonal, gb.diagonal), 1.0)
    displacement = np.linalg.norm(a.astype(np.float64) - b.astype(np.float64), axis=1)
    return float(np.median(displacement) / scale)


def deduplicate_landmark_candidates(candidates: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Collapse repeated observations of the same physical face across crops."""
    unique: list[np.ndarray] = []
    for candidate in candidates:
        if candidate.ndim != 2 or candidate.shape[1] != 2:
            continue
        if any(
            _duplicate_distance_ratio(candidate, existing) <= _DUPLICATE_MESH_DISTANCE_RATIO
            for existing in unique
        ):
            continue
        unique.append(candidate)
    return unique


def associate_candidates_one_to_one(
    candidates: Sequence[np.ndarray],
    detections: Sequence[DetectionLike],
) -> list[Optional[np.ndarray]]:
    """Assign pooled candidates to detections with an image-wide one-to-one rule.

    All compatible detector/candidate edges are ranked together, then consumed
    from strongest to weakest while reserving both endpoints.  A candidate can
    therefore never be reused for a second detection.  Detections for which no
    unused compatible candidate remains are returned as ``None`` rather than
    being paired with a neighboring face.
    """
    unique = deduplicate_landmark_candidates(candidates)
    assignments: list[Optional[np.ndarray]] = [None] * len(detections)
    if not unique or not detections:
        return assignments

    edges: list[tuple[float, int, int]] = []
    for det_index, detection in enumerate(detections):
        for cand_index, candidate in enumerate(unique):
            score = association_score(candidate, detection)
            if score is not None:
                edges.append((score, det_index, cand_index))

    # Stable tie-breakers favor the more specific (smaller) detector box before
    # index order.  This helps nested detections reserve their best local mesh.
    edges.sort(
        key=lambda item: (
            item[0],
            -float(detections[item[1]].width * detections[item[1]].height),
            -item[1],
            -item[2],
        ),
        reverse=True,
    )

    used_detections: set[int] = set()
    used_candidates: set[int] = set()
    for _score, det_index, cand_index in edges:
        if det_index in used_detections or cand_index in used_candidates:
            continue
        assignments[det_index] = unique[cand_index]
        used_detections.add(det_index)
        used_candidates.add(cand_index)

    return assignments


def select_landmark_candidate(
    candidates: Sequence[np.ndarray], detection: DetectionLike
) -> Optional[np.ndarray]:
    """Backward-compatible single-detection helper used by older diagnostics."""
    return associate_candidates_one_to_one(candidates, [detection])[0]
