"""Lightweight in-memory photo, face, and cluster data structures."""
from dataclasses import dataclass, field
from datetime import datetime
from face_grouping.time_utils import utcnow_naive
from enum import Enum
from typing import Optional

import numpy as np

from face_grouping.matching.assignment import AssignmentState
from face_grouping.matching.exemplars import ExemplarSet


class PhotoProcessingStatus(Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(eq=False)
class Photo:
    photo_id: str
    image_path: str
    image_width: int
    image_height: int
    processing_status: PhotoProcessingStatus = PhotoProcessingStatus.PROCESSING
    processed_at: Optional[datetime] = None
    embedding_model_version: str = "legacy_unknown"
    config_version: str = "legacy_unknown"
    error_message: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.processing_status, str):
            self.processing_status = PhotoProcessingStatus(self.processing_status)


@dataclass(eq=False)
class Face:
    face_id: str
    embedding: np.ndarray
    quality_score: float
    yaw_ratio: float
    cluster_id: Optional[str] = None
    is_manually_corrected: bool = False
    created_at: datetime = field(default_factory=utcnow_naive)

    # Persistent assignment evidence.
    assignment_state: Optional[AssignmentState] = None
    candidate_cluster_id: Optional[str] = None
    best_match_score: Optional[float] = None
    second_best_cluster_id: Optional[str] = None
    second_best_score: Optional[float] = None
    score_margin: Optional[float] = None
    decision_threshold: Optional[float] = None
    decision_reason: Optional[str] = None

    # Stage 3: source-photo traceability and reproducibility metadata.
    photo_id: Optional[str] = None
    face_index: Optional[int] = None
    bbox_x1: Optional[float] = None
    bbox_y1: Optional[float] = None
    bbox_x2: Optional[float] = None
    bbox_y2: Optional[float] = None
    detection_score: Optional[float] = None
    embedding_model_version: str = "legacy_unknown"
    config_version: str = "legacy_unknown"

    # Faces rejected only by the absolute pose floor are retained for a
    # restricted recognition-only path. They can join a mature existing
    # cluster, but can never seed a cluster or become an exemplar.
    recognition_restricted: bool = False
    recognition_restriction_reason: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.assignment_state, str):
            self.assignment_state = AssignmentState(self.assignment_state)
        if self.assignment_state is None:
            self.assignment_state = (
                AssignmentState.CONFIRMED
                if self.cluster_id is not None
                else AssignmentState.UNASSIGNED
            )


@dataclass(eq=False)
class Cluster:
    cluster_id: str
    exemplar_set: ExemplarSet
    face_count: int = 0
    is_user_confirmed: bool = False
    has_manual_correction: bool = False
    created_at: datetime = field(default_factory=utcnow_naive)
    last_updated_at: datetime = field(default_factory=utcnow_naive)
    merged_into: Optional[str] = None
