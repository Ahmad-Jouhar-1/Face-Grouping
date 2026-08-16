"""Optional FastAPI transport adapter for the face-grouping application service.

The HTTP layer is intentionally thin: authentication remains a backend concern,
while this service trusts a backend-derived tenant header and optionally checks
an internal service-to-service API key.
"""
from __future__ import annotations

import hmac
import os
import sqlite3
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Protocol

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from face_grouping.api.schemas import (
    ConsolidationResponse,
    DeleteUserDataResponse,
    FaceResponse,
    HealthResponse,
    IntegrityResponse,
    LifecycleResponse,
    MovePhotosRequest,
    MovePhotosResponse,
    PersonResponse,
    PhotoResponse,
    ProcessPhotoResponse,
    ResolveSuggestionRequest,
    SuggestionResponse,
)
from face_grouping.application import FaceGroupingService
from face_grouping.storage.schema import LEGACY_USER_ID
from face_grouping.errors import (
    ConsolidationInProgressError,
    PhotoProcessingInProgressError,
    PhotoProcessingLeaseLostError,
)


USER_HEADER = "X-User-ID"
API_KEY_HEADER = "X-Internal-API-Key"
_DEFAULT_MAX_UPLOAD_MB = 25


class _ServiceLike(Protocol):
    def process_photo(self, *, user_id: str, photo_id: str, image_path: str, source_ref: str | None = None): ...
    def list_people(self, *, user_id: str, include_hidden: bool = False): ...
    def list_person_photos(self, *, user_id: str, cluster_id: str): ...
    def consolidate(self, *, user_id: str, force: bool = True): ...
    def get_lifecycle(self, *, user_id: str): ...
    def list_pending_suggestions(self, *, user_id: str): ...
    def resolve_suggestion(self, *, user_id: str, suggestion_id: str, status: str): ...
    def move_photos(self, *, user_id: str, photo_ids, from_cluster_id: str, to_cluster_id: str): ...
    def delete_photo(self, *, user_id: str, photo_id: str): ...
    def delete_user_data(self, *, user_id: str): ...
    def validate_user_storage(self, *, user_id: str): ...
    def close(self): ...


def _max_upload_bytes() -> int:
    raw = os.environ.get("FACE_GROUPING_MAX_UPLOAD_MB", str(_DEFAULT_MAX_UPLOAD_MB))
    try:
        mb = int(raw)
    except ValueError as exc:
        raise RuntimeError("FACE_GROUPING_MAX_UPLOAD_MB must be an integer") from exc
    if mb <= 0:
        raise RuntimeError("FACE_GROUPING_MAX_UPLOAD_MB must be positive")
    return mb * 1024 * 1024


def _internal_api_key() -> str | None:
    value = os.environ.get("FACE_GROUPING_INTERNAL_API_KEY")
    return value if value else None


def _db_path() -> str:
    return os.environ.get("FACE_GROUPING_DB_PATH", "./face_grouping.db")


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be numeric") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


def _service_from_env() -> FaceGroupingService:
    return FaceGroupingService(
        _db_path(),
        consolidate_every_photos=_positive_int_env(
            "FACE_GROUPING_CONSOLIDATE_EVERY_PHOTOS", 50
        ),
        consolidate_idle_hours=_positive_float_env(
            "FACE_GROUPING_CONSOLIDATE_IDLE_HOURS", 24.0
        ),
        photo_processing_lease_seconds=_positive_int_env(
            "FACE_GROUPING_PHOTO_LEASE_SECONDS", 300
        ),
        consolidation_lease_seconds=_positive_int_env(
            "FACE_GROUPING_CONSOLIDATION_LEASE_SECONDS", 1800
        ),
    )


def _validate_user_id(value: str) -> str:
    user_id = value.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail=f"{USER_HEADER} must be non-empty")
    if len(user_id) > 256 or any(ord(ch) < 32 for ch in user_id):
        raise HTTPException(status_code=400, detail=f"Invalid {USER_HEADER}")
    if user_id in {LEGACY_USER_ID, "__service_bootstrap__"}:
        raise HTTPException(status_code=400, detail=f"Reserved {USER_HEADER}")
    return user_id


async def _tenant_dependency(
    x_user_id: str = Header(..., alias=USER_HEADER),
    x_internal_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> str:
    expected = _internal_api_key()
    if expected is not None:
        if x_internal_api_key is None or not hmac.compare_digest(x_internal_api_key, expected):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid internal API key")
    return _validate_user_id(x_user_id)


def _service(request: Request) -> _ServiceLike:
    return request.app.state.face_grouping_service


async def _stage_upload(upload: UploadFile, max_bytes: int) -> Path:
    """Copy an upload to an AI-owned temporary file with a hard size limit."""
    fd, raw_path = tempfile.mkstemp(prefix="face-grouping-upload-", suffix=".bin")
    os.close(fd)
    path = Path(raw_path)
    total = 0
    try:
        with path.open("wb") as output:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Image exceeds {max_bytes // (1024 * 1024)} MB upload limit",
                    )
                output.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="Uploaded image is empty")
        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def _face_response(face) -> FaceResponse:
    return FaceResponse(
        face_id=face.face_id,
        photo_id=face.photo_id,
        cluster_id=face.cluster_id,
        assignment_state=face.assignment_state,
        quality_score=face.quality_score,
        bbox=face.bbox,
        recognition_restricted=face.recognition_restricted,
    )


def _lifecycle_response(lifecycle) -> LifecycleResponse:
    return LifecycleResponse(**lifecycle.__dict__)


def _raise_transport_error(exc: Exception) -> None:
    if isinstance(exc, PhotoProcessingInProgressError):
        headers = {}
        if exc.retry_after_seconds is not None:
            headers["Retry-After"] = str(exc.retry_after_seconds)
        raise HTTPException(status_code=409, detail=str(exc), headers=headers) from exc
    if isinstance(exc, PhotoProcessingLeaseLostError):
        raise HTTPException(
            status_code=409, detail=str(exc), headers={"Retry-After": "1"}
        ) from exc
    if isinstance(exc, ConsolidationInProgressError):
        headers = {}
        if exc.retry_after_seconds is not None:
            headers["Retry-After"] = str(exc.retry_after_seconds)
        raise HTTPException(status_code=409, detail=str(exc), headers=headers) from exc
    if isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower():
        raise HTTPException(
            status_code=503, detail="AI storage is temporarily busy",
            headers={"Retry-After": "1"},
        ) from exc
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        message = str(exc)
        status_code = 404 if message.startswith("Unknown ") else 409 if "already bound" in message else 400
        raise HTTPException(status_code=status_code, detail=message) from exc
    raise exc


def create_app(*, service: _ServiceLike | None = None) -> FastAPI:
    """Create the HTTP adapter.

    Passing ``service`` is primarily useful for tests/embedding. In normal
    deployment, the app owns one ``FaceGroupingService`` per worker process.
    """
    owns_service = service is None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        instance = service or _service_from_env()
        app.state.face_grouping_service = instance
        try:
            yield
        finally:
            if owns_service:
                instance.close()

    app = FastAPI(
        title="Face Grouping AI Service",
        version="1.1.0",
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    @app.get("/health/live", response_model=HealthResponse, tags=["health"])
    async def live() -> HealthResponse:
        return HealthResponse()

    @app.get("/health/ready", response_model=HealthResponse, tags=["health"])
    async def ready(request: Request) -> HealthResponse:
        # If lifespan completed, model assets/runtime and DB startup succeeded.
        _service(request)
        return HealthResponse()

    @app.post(
        "/v1/photos/{photo_id}/process",
        response_model=ProcessPhotoResponse,
        tags=["photos"],
    )
    async def process_photo(
        request: Request,
        photo_id: str,
        image: UploadFile = File(...),
        user_id: str = Depends(_tenant_dependency),
    ) -> ProcessPhotoResponse:
        if not photo_id.strip():
            raise HTTPException(status_code=400, detail="photo_id must be non-empty")
        if image.content_type and not image.content_type.startswith("image/"):
            raise HTTPException(status_code=415, detail="Upload must have an image/* content type")

        path = await _stage_upload(image, _max_upload_bytes())
        # This stable opaque reference is persisted instead of the temporary
        # filesystem path, so retries remain idempotent after staging cleanup.
        source_ref = f"backend-photo://{photo_id}"
        try:
            try:
                result = await run_in_threadpool(
                    _service(request).process_photo,
                    user_id=user_id,
                    photo_id=photo_id,
                    image_path=str(path),
                    source_ref=source_ref,
                )
            except Exception as exc:
                _raise_transport_error(exc)
            return ProcessPhotoResponse(
                photo_id=result.photo_id,
                faces=[_face_response(face) for face in result.faces],
                cached=result.cached,
                lifecycle=_lifecycle_response(result.lifecycle),
            )
        finally:
            path.unlink(missing_ok=True)

    @app.get("/v1/people", response_model=list[PersonResponse], tags=["people"])
    async def list_people(
        request: Request,
        include_hidden: bool = Query(default=False),
        user_id: str = Depends(_tenant_dependency),
    ) -> list[PersonResponse]:
        people = await run_in_threadpool(
            _service(request).list_people,
            user_id=user_id,
            include_hidden=include_hidden,
        )
        return [PersonResponse(**person.__dict__) for person in people]

    @app.get(
        "/v1/people/{cluster_id}/photos",
        response_model=list[PhotoResponse],
        tags=["people"],
    )
    async def list_person_photos(
        request: Request,
        cluster_id: str,
        user_id: str = Depends(_tenant_dependency),
    ) -> list[PhotoResponse]:
        try:
            photos = await run_in_threadpool(
                _service(request).list_person_photos,
                user_id=user_id,
                cluster_id=cluster_id,
            )
        except Exception as exc:
            _raise_transport_error(exc)
        return [PhotoResponse(**photo.__dict__) for photo in photos]

    @app.get(
        "/v1/lifecycle",
        response_model=LifecycleResponse,
        tags=["grouping"],
    )
    async def lifecycle(
        request: Request,
        user_id: str = Depends(_tenant_dependency),
    ) -> LifecycleResponse:
        state = await run_in_threadpool(
            _service(request).get_lifecycle, user_id=user_id
        )
        return _lifecycle_response(state)

    @app.post(
        "/v1/consolidation",
        response_model=ConsolidationResponse,
        tags=["grouping"],
    )
    async def consolidate(
        request: Request,
        force: bool = Query(default=True),
        user_id: str = Depends(_tenant_dependency),
    ) -> ConsolidationResponse:
        try:
            result = await run_in_threadpool(
                _service(request).consolidate, user_id=user_id, force=force
            )
        except Exception as exc:
            _raise_transport_error(exc)
        return ConsolidationResponse(
            ran=result.ran,
            summary=result.summary,
            lifecycle=_lifecycle_response(result.lifecycle),
            skipped_reason=result.skipped_reason,
        )

    @app.get(
        "/v1/suggestions",
        response_model=list[SuggestionResponse],
        tags=["suggestions"],
    )
    async def list_suggestions(
        request: Request,
        user_id: str = Depends(_tenant_dependency),
    ) -> list[SuggestionResponse]:
        items = await run_in_threadpool(
            _service(request).list_pending_suggestions,
            user_id=user_id,
        )
        return [
            SuggestionResponse(
                suggestion_id=item.suggestion_id,
                suggestion_type=item.suggestion_type,
                cluster_ids=list(item.cluster_ids),
                status=item.status,
                created_at=item.created_at,
                payload=item.payload,
                evidence=item.evidence,
            )
            for item in items
        ]

    @app.post(
        "/v1/suggestions/{suggestion_id}/resolve",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["suggestions"],
    )
    async def resolve_suggestion(
        request: Request,
        suggestion_id: str,
        body: ResolveSuggestionRequest,
        user_id: str = Depends(_tenant_dependency),
    ) -> None:
        try:
            await run_in_threadpool(
                _service(request).resolve_suggestion,
                user_id=user_id,
                suggestion_id=suggestion_id,
                status=body.status,
            )
        except Exception as exc:
            _raise_transport_error(exc)

    @app.post(
        "/v1/people/move-photos",
        response_model=MovePhotosResponse,
        tags=["people"],
    )
    async def move_photos(
        request: Request,
        body: MovePhotosRequest,
        user_id: str = Depends(_tenant_dependency),
    ) -> MovePhotosResponse:
        try:
            moved = await run_in_threadpool(
                _service(request).move_photos,
                user_id=user_id,
                photo_ids=body.photo_ids,
                from_cluster_id=body.from_cluster_id,
                to_cluster_id=body.to_cluster_id,
            )
        except Exception as exc:
            _raise_transport_error(exc)
        return MovePhotosResponse(moved_faces=moved)

    @app.delete(
        "/v1/photos/{photo_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["photos"],
    )
    async def delete_photo(
        request: Request,
        photo_id: str,
        user_id: str = Depends(_tenant_dependency),
    ) -> None:
        try:
            await run_in_threadpool(
                _service(request).delete_photo,
                user_id=user_id,
                photo_id=photo_id,
            )
        except Exception as exc:
            _raise_transport_error(exc)

    @app.delete(
        "/v1/users/me/data",
        response_model=DeleteUserDataResponse,
        tags=["account"],
    )
    async def delete_user_data(
        request: Request,
        user_id: str = Depends(_tenant_dependency),
    ) -> DeleteUserDataResponse:
        deleted = await run_in_threadpool(
            _service(request).delete_user_data,
            user_id=user_id,
        )
        return DeleteUserDataResponse(deleted=deleted)

    @app.get(
        "/v1/storage/integrity",
        response_model=IntegrityResponse,
        tags=["diagnostics"],
    )
    async def storage_integrity(
        request: Request,
        user_id: str = Depends(_tenant_dependency),
    ) -> IntegrityResponse:
        issues = await run_in_threadpool(
            _service(request).validate_user_storage,
            user_id=user_id,
        )
        return IntegrityResponse(ok=not issues, issues=list(issues))

    return app


app = create_app()
