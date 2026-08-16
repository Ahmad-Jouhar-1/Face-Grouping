"""Framework-agnostic application boundary for backend integration."""

from face_grouping.application.service import FaceGroupingService
from face_grouping.application.schemas import (
    ConsolidationRunResult,
    FaceView,
    LifecycleView,
    PersonView,
    PhotoView,
    ProcessPhotoResult,
    SuggestionView,
)

__all__ = [
    "FaceGroupingService",
    "ConsolidationRunResult",
    "FaceView",
    "LifecycleView",
    "PersonView",
    "PhotoView",
    "ProcessPhotoResult",
    "SuggestionView",
]
