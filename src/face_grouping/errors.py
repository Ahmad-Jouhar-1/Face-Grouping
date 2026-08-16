"""Stable application/domain errors that transport adapters can map safely."""


class FaceGroupingError(Exception):
    """Base class for expected face-grouping service errors."""


class PhotoProcessingInProgressError(FaceGroupingError):
    def __init__(self, photo_id: str, retry_after_seconds: int | None = None):
        self.photo_id = photo_id
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Photo {photo_id!r} is already being processed")


class PhotoProcessingLeaseLostError(FaceGroupingError):
    def __init__(self, photo_id: str):
        self.photo_id = photo_id
        super().__init__(f"Processing lease was lost for photo {photo_id!r}; retry the request")


class ConsolidationInProgressError(FaceGroupingError):
    def __init__(self, retry_after_seconds: int | None = None):
        self.retry_after_seconds = retry_after_seconds
        super().__init__("Consolidation is already running for this user")
