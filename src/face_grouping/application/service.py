"""Production application interface consumed by a backend adapter.

This module intentionally has no FastAPI dependency. The backend can wrap these
methods in HTTP routes, RPC handlers, a job queue, or another transport without
letting transport concerns leak into the face-grouping core.
"""
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional

from face_grouping.application.schemas import (
    ConsolidationRunResult,
    FaceView,
    LifecycleView,
    PersonView,
    PhotoView,
    ProcessPhotoResult,
    SuggestionView,
)
from face_grouping.clustering.merge_rules import SuggestionStatus
from face_grouping.errors import ConsolidationInProgressError
from face_grouping.lifecycle.visibility import VisibilityState, get_visibility
from face_grouping.pipeline import FaceGroupingPipeline
from face_grouping.runtime import FaceGroupingRuntime
from face_grouping.storage.store import FaceGroupingStore, LifecycleState


class FaceGroupingService:
    """Backend-facing facade with shared models and per-request tenant state."""

    def __init__(
        self,
        db_path: str,
        *,
        runtime: Optional[FaceGroupingRuntime] = None,
        consolidate_every_photos: int = 50,
        consolidate_idle_hours: float = 24.0,
        photo_processing_lease_seconds: int = 300,
        consolidation_lease_seconds: int = 1800,
    ):
        self.db_path = str(Path(db_path))
        self.runtime = runtime or FaceGroupingRuntime()
        self._owns_runtime = runtime is None
        self.consolidate_every_photos = int(consolidate_every_photos)
        self.consolidate_idle_hours = float(consolidate_idle_hours)
        self.photo_processing_lease_seconds = int(photo_processing_lease_seconds)
        self.consolidation_lease_seconds = int(consolidation_lease_seconds)
        if self.consolidate_every_photos <= 0:
            raise ValueError("consolidate_every_photos must be positive")
        if self.consolidate_idle_hours <= 0:
            raise ValueError("consolidate_idle_hours must be positive")
        if self.photo_processing_lease_seconds < 0 or self.consolidation_lease_seconds < 0:
            raise ValueError("lease durations must be >= 0")

        # Initialize/migrate SQLite during service startup rather than making
        # the first user request pay migration cost or race with another one.
        with FaceGroupingStore(self.db_path, user_id="__service_bootstrap__"):
            pass

    @contextmanager
    def _session(self, user_id: str):
        pipeline = FaceGroupingPipeline(
            self.db_path,
            user_id=user_id,
            runtime=self.runtime,
            photo_processing_lease_seconds=self.photo_processing_lease_seconds,
        )
        try:
            yield pipeline
        finally:
            pipeline.close()

    @staticmethod
    def _face_view(face) -> FaceView:
        bbox = None
        if None not in (face.bbox_x1, face.bbox_y1, face.bbox_x2, face.bbox_y2):
            bbox = (
                float(face.bbox_x1),
                float(face.bbox_y1),
                float(face.bbox_x2),
                float(face.bbox_y2),
            )
        return FaceView(
            face_id=face.face_id,
            photo_id=face.photo_id,
            cluster_id=face.cluster_id,
            assignment_state=face.assignment_state.value,
            quality_score=float(face.quality_score),
            bbox=bbox,
            recognition_restricted=bool(face.recognition_restricted),
        )

    @staticmethod
    def _lifecycle_view(state: LifecycleState) -> LifecycleView:
        return LifecycleView(
            photos_since_consolidation=state.photos_since_consolidation,
            consolidation_due=state.consolidation_due,
            due_reason=state.due_reason,
            due_since=state.due_since.isoformat() if state.due_since else None,
            last_photo_completed_at=(
                state.last_photo_completed_at.isoformat() if state.last_photo_completed_at else None
            ),
            last_consolidated_at=(
                state.last_consolidated_at.isoformat() if state.last_consolidated_at else None
            ),
            consolidation_in_progress=state.consolidation_in_progress,
            last_consolidation_error=state.last_consolidation_error,
        )

    def get_lifecycle(self, *, user_id: str) -> LifecycleView:
        with FaceGroupingStore(self.db_path, user_id=user_id) as store:
            store.mark_count_due_if_needed(photo_threshold=self.consolidate_every_photos)
            state = store.get_lifecycle_state(
                photo_threshold=self.consolidate_every_photos,
                idle_hours=self.consolidate_idle_hours,
            )
            return self._lifecycle_view(state)

    def process_photo(
        self,
        *,
        user_id: str,
        photo_id: str,
        image_path: str,
        source_ref: Optional[str] = None,
    ) -> ProcessPhotoResult:
        """Process one backend-owned photo idempotently for one user."""
        if not photo_id:
            raise ValueError("photo_id must be non-empty")
        with self._session(user_id) as pipeline:
            faces, cached = pipeline.process_photo_with_status(
                image_path,
                photo_id=photo_id,
                source_ref=source_ref,
            )
        lifecycle = self.get_lifecycle(user_id=user_id)
        return ProcessPhotoResult(
            photo_id=photo_id,
            faces=tuple(self._face_view(face) for face in faces),
            cached=cached,
            lifecycle=lifecycle,
        )

    def consolidate(self, *, user_id: str, force: bool = True) -> ConsolidationRunResult:
        with FaceGroupingStore(self.db_path, user_id=user_id) as store:
            claim = store.claim_consolidation(
                force=force,
                photo_threshold=self.consolidate_every_photos,
                idle_hours=self.consolidate_idle_hours,
                lease_seconds=self.consolidation_lease_seconds,
            )
        if claim.status == "not_due":
            return ConsolidationRunResult(
                ran=False,
                summary={},
                lifecycle=self.get_lifecycle(user_id=user_id),
                skipped_reason="not_due",
            )
        if claim.status == "in_progress":
            raise ConsolidationInProgressError(claim.retry_after_seconds)
        token = claim.token
        assert token is not None
        try:
            with self._session(user_id) as pipeline:
                summary = pipeline.run_consolidation(consolidation_token=token)
        except Exception as exc:
            with FaceGroupingStore(self.db_path, user_id=user_id) as store:
                store.fail_consolidation_claim(token, str(exc))
            raise
        return ConsolidationRunResult(
            ran=True,
            summary=summary,
            lifecycle=self.get_lifecycle(user_id=user_id),
        )

    def list_people(
        self,
        *,
        user_id: str,
        include_hidden: bool = False,
    ) -> tuple[PersonView, ...]:
        with FaceGroupingStore(self.db_path, user_id=user_id) as store:
            results = []
            for cluster in store.load_all_clusters(include_merged=False):
                visibility = get_visibility(cluster)
                if not include_hidden and visibility == VisibilityState.HIDDEN:
                    continue
                exemplars = cluster.exemplar_set.all_exemplars()
                representative_face_id = exemplars[0].face_id if exemplars else None
                representative_photo_id = None
                if representative_face_id:
                    face = store.load_face(representative_face_id)
                    representative_photo_id = face.photo_id if face else None
                results.append(
                    PersonView(
                        cluster_id=cluster.cluster_id,
                        face_count=cluster.face_count,
                        visibility=visibility.value,
                        is_user_confirmed=cluster.is_user_confirmed,
                        has_manual_correction=cluster.has_manual_correction,
                        representative_face_id=representative_face_id,
                        representative_photo_id=representative_photo_id,
                    )
                )
            results.sort(key=lambda item: (-item.face_count, item.cluster_id))
            return tuple(results)

    def list_person_photos(self, *, user_id: str, cluster_id: str) -> tuple[PhotoView, ...]:
        with FaceGroupingStore(self.db_path, user_id=user_id) as store:
            if store.load_cluster(cluster_id) is None:
                raise ValueError(f"Unknown cluster_id: {cluster_id}")
            return tuple(
                PhotoView(
                    photo_id=photo.photo_id,
                    processing_status=photo.processing_status.value,
                    processed_at=photo.processed_at.isoformat() if photo.processed_at else None,
                )
                for photo in store.load_photos_by_cluster(cluster_id)
            )

    def list_pending_suggestions(self, *, user_id: str) -> tuple[SuggestionView, ...]:
        with FaceGroupingStore(self.db_path, user_id=user_id) as store:
            return tuple(
                SuggestionView(
                    suggestion_id=item.suggestion_id,
                    suggestion_type=item.suggestion_type.value,
                    cluster_ids=tuple(item.cluster_ids),
                    status=item.status.value,
                    created_at=item.created_at.isoformat(),
                    payload=dict(item.payload),
                    evidence=dict(item.evidence),
                )
                for item in store.load_pending_suggestions()
            )

    def resolve_suggestion(
        self,
        *,
        user_id: str,
        suggestion_id: str,
        status: str | SuggestionStatus,
    ) -> None:
        resolved_status = status if isinstance(status, SuggestionStatus) else SuggestionStatus(status)
        with self._session(user_id) as pipeline:
            pipeline.resolve_suggestion(suggestion_id, resolved_status)

    def move_photos(
        self,
        *,
        user_id: str,
        photo_ids: Iterable[str],
        from_cluster_id: str,
        to_cluster_id: str,
    ) -> int:
        with self._session(user_id) as pipeline:
            return pipeline.move_photos_manually(
                list(photo_ids), from_cluster_id, to_cluster_id
            )

    def delete_photo(self, *, user_id: str, photo_id: str) -> None:
        # Deletion is immediate and atomic. Cadence is event-based, so deleting
        # a previously processed photo does not rewrite historical ingestion count.
        with self._session(user_id) as pipeline:
            pipeline.delete_photo(photo_id)

    def delete_user_data(self, *, user_id: str) -> dict[str, int]:
        with FaceGroupingStore(self.db_path, user_id=user_id) as store:
            return store.delete_user_data()

    def validate_user_storage(self, *, user_id: str) -> tuple[str, ...]:
        with FaceGroupingStore(self.db_path, user_id=user_id) as store:
            return tuple(store.validate_integrity(expected_embedding_dim=512))

    def close(self) -> None:
        if self._owns_runtime:
            self.runtime.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
