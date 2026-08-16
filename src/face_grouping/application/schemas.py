"""Primitive, transport-safe DTOs exposed to the backend layer."""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class FaceView:
    face_id: str
    photo_id: Optional[str]
    cluster_id: Optional[str]
    assignment_state: str
    quality_score: float
    bbox: Optional[Tuple[float, float, float, float]]
    recognition_restricted: bool


@dataclass(frozen=True)
class PhotoView:
    photo_id: str
    processing_status: str
    processed_at: Optional[str]


@dataclass(frozen=True)
class PersonView:
    cluster_id: str
    face_count: int
    visibility: str
    is_user_confirmed: bool
    has_manual_correction: bool
    representative_face_id: Optional[str]
    representative_photo_id: Optional[str]


@dataclass(frozen=True)
class SuggestionView:
    suggestion_id: str
    suggestion_type: str
    cluster_ids: Tuple[str, ...]
    status: str
    created_at: str
    payload: Dict[str, Any]
    evidence: Dict[str, Any]


@dataclass(frozen=True)
class LifecycleView:
    photos_since_consolidation: int
    consolidation_due: bool
    due_reason: Optional[str]
    due_since: Optional[str]
    last_photo_completed_at: Optional[str]
    last_consolidated_at: Optional[str]
    consolidation_in_progress: bool
    last_consolidation_error: Optional[str]


@dataclass(frozen=True)
class ProcessPhotoResult:
    photo_id: str
    faces: Tuple[FaceView, ...]
    cached: bool = False
    lifecycle: Optional[LifecycleView] = None


@dataclass(frozen=True)
class ConsolidationRunResult:
    ran: bool
    summary: Dict[str, Any]
    lifecycle: LifecycleView
    skipped_reason: Optional[str] = None
