"""HTTP transport schemas for the optional FastAPI adapter."""
from typing import Any, Literal

from pydantic import BaseModel, Field


class FaceResponse(BaseModel):
    face_id: str
    photo_id: str | None
    cluster_id: str | None
    assignment_state: str
    quality_score: float
    bbox: tuple[float, float, float, float] | None
    recognition_restricted: bool


class LifecycleResponse(BaseModel):
    photos_since_consolidation: int
    consolidation_due: bool
    due_reason: Literal["photo_count", "idle_timeout"] | None
    due_since: str | None
    last_photo_completed_at: str | None
    last_consolidated_at: str | None
    consolidation_in_progress: bool
    last_consolidation_error: str | None


class ProcessPhotoResponse(BaseModel):
    photo_id: str
    faces: list[FaceResponse]
    cached: bool
    lifecycle: LifecycleResponse


class PersonResponse(BaseModel):
    cluster_id: str
    face_count: int
    visibility: str
    is_user_confirmed: bool
    has_manual_correction: bool
    representative_face_id: str | None
    representative_photo_id: str | None


class PhotoResponse(BaseModel):
    photo_id: str
    processing_status: str
    processed_at: str | None


class SuggestionResponse(BaseModel):
    suggestion_id: str
    suggestion_type: str
    cluster_ids: list[str]
    status: str
    created_at: str
    payload: dict[str, Any]
    evidence: dict[str, Any]


class ResolveSuggestionRequest(BaseModel):
    status: Literal["accepted", "rejected", "uncertain"]


class MovePhotosRequest(BaseModel):
    photo_ids: list[str] = Field(min_length=1)
    from_cluster_id: str = Field(min_length=1)
    to_cluster_id: str = Field(min_length=1)


class MovePhotosResponse(BaseModel):
    moved_faces: int


class ConsolidationResponse(BaseModel):
    ran: bool
    summary: dict[str, Any]
    lifecycle: LifecycleResponse
    skipped_reason: Literal["not_due"] | None = None


class DeleteUserDataResponse(BaseModel):
    deleted: dict[str, int]


class IntegrityResponse(BaseModel):
    ok: bool
    issues: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "face-grouping-ai"
