"""
Full pipeline orchestration (Step 8).

Wires Steps 1-7 into one continuous flow:
  photo arrives -> detect -> quality-score -> embed -> match against
  known people (Step 4) -> assign to a cluster or create a new one
  -> persist (Step 7)

Plus explicit consolidation (Step 5), merge/split suggestion resolution, and
correction application (Step 6). Production cadence/leases live in the
application/storage layers so the core algorithm remains transport-neutral.

Two documented simplifications vs. the original locked design:

1. Incremental matching (_assign_face) compares a new face against
   EVERY active cluster's exemplars, not a windowed/nearest-neighbor
   subset. This matches our earlier Point 5 (tools/libraries) decision
   that a dedicated ANN search library isn't needed at this scale --
   brute-force is fine for testing-scale cluster counts.

2. Consolidation remains precision-first, but now includes a narrow
   high-confidence structural-correction tier. Tiny history-created fragments
   can be auto-merged into a mature cluster, and only exceptionally clear
   two-way contaminations can be auto-split. Borderline cases remain user
   suggestions.
"""
import uuid
from datetime import datetime
from face_grouping.time_utils import utcnow_naive
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from face_grouping.matching.assignment import AssignmentState
from face_grouping.matching.incremental import IncrementalAssigner
from face_grouping.clustering.data_types import (
    Face, Cluster, Photo, PhotoProcessingStatus,
)
from face_grouping.clustering.consolidation import ConsolidationEngine
from face_grouping.clustering.merge_rules import (
    build_merge_suggestions,
    build_split_suggestions,
    SuggestionStatus,
    SuggestionType,
)
from face_grouping.lifecycle.visibility import get_visibility, VisibilityState
from face_grouping.lifecycle.pruning import find_clusters_to_prune
from face_grouping.storage.store import FaceGroupingStore
from face_grouping.storage.schema import LEGACY_USER_ID
from face_grouping.runtime import FaceGroupingRuntime
from face_grouping.config import normalize_image_path
from face_grouping.errors import PhotoProcessingInProgressError, PhotoProcessingLeaseLostError


def _read_image_bgr(image_path: str):
    """Read an image in a Unicode-safe way on Windows.

    OpenCV's ``cv2.imread`` can fail for valid Windows paths containing
    non-ASCII characters. Python handles those paths correctly, so read the
    raw bytes with ``Path.read_bytes`` and let OpenCV decode the in-memory
    buffer instead.
    """
    try:
        encoded = np.frombuffer(Path(image_path).read_bytes(), dtype=np.uint8)
    except OSError:
        return None

    if encoded.size == 0:
        return None

    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


class FaceGroupingPipeline:
    """One tenant-scoped grouping session backed by a shared model runtime."""

    def __init__(
        self,
        db_path: str,
        user_id: str = LEGACY_USER_ID,
        *,
        runtime: Optional[FaceGroupingRuntime] = None,
        photo_processing_lease_seconds: int = 300,
    ):
        self.runtime = runtime or FaceGroupingRuntime()
        self._owns_runtime = runtime is None
        self.user_id = user_id
        self.photo_processing_lease_seconds = int(photo_processing_lease_seconds)
        if self.photo_processing_lease_seconds < 0:
            raise ValueError("photo_processing_lease_seconds must be >= 0")

        self._align_face = self.runtime.align_face
        self._compute_face_quality = self.runtime.compute_face_quality
        self.detector = self.runtime.detector
        self.landmarker = self.runtime.landmarker
        self.embedder = self.runtime.embedder
        self.embedding_model_version = self.runtime.embedding_model_version
        self.config_version = self.runtime.config_version
        cfg = self.runtime.cfg

        self.store = FaceGroupingStore(db_path, user_id=user_id)
        self.legacy_empty_cluster_repair = self.store.repair_empty_active_clusters()
        self.legacy_suggestions_removed = self.store.remove_legacy_pending_suggestions()

        matching_cfg = cfg["matching"]
        self.t_match = matching_cfg.get("t_match")
        self.band_width = matching_cfg.get("ambiguous_band_width")
        if self.t_match is None or self.band_width is None:
            raise ValueError(
                "T_match/ambiguous_band_width are not set in configs/thresholds.yaml. "
                "Run the Phase 1/2 sweeps (Step 4) first."
            )
        self.top_k = matching_cfg["top_k"]
        self.sparse_cluster_margin = matching_cfg["sparse_cluster_margin"]
        self.exemplar_admission_margin = matching_cfg["exemplar_admission_margin"]
        self.min_cluster_margin = matching_cfg["min_cluster_margin"]
        self.exemplar_quality_bucket_size = matching_cfg["exemplar_set"]["quality_bucket_size"]
        self.exemplar_pose_bucket_size = matching_cfg["exemplar_set"]["pose_bucket_size"]
        self.exemplar_quality_threshold = cfg["quality"]["exemplar_eligibility_threshold"]
        consolidation_cfg = cfg.get("consolidation", {})
        auto_cfg = consolidation_cfg.get("auto_correction", {})
        restricted_pose_cfg = consolidation_cfg.get("restricted_pose_recovery", {})

        self.incremental_assigner = IncrementalAssigner(
            store=self.store,
            t_match=self.t_match,
            band_width=self.band_width,
            top_k=self.top_k,
            sparse_cluster_margin=self.sparse_cluster_margin,
            exemplar_admission_margin=self.exemplar_admission_margin,
            min_cluster_margin=self.min_cluster_margin,
            exemplar_quality_bucket_size=self.exemplar_quality_bucket_size,
            exemplar_pose_bucket_size=self.exemplar_pose_bucket_size,
        )
        self.consolidation_engine = ConsolidationEngine(
            store=self.store,
            assigner=self.incremental_assigner,
            t_match=self.t_match,
            band_width=self.band_width,
            top_k=self.top_k,
            sparse_cluster_margin=self.sparse_cluster_margin,
            min_cluster_margin=self.min_cluster_margin,
            exemplar_admission_margin=self.exemplar_admission_margin,
            exemplar_quality_threshold=self.exemplar_quality_threshold,
            exemplar_quality_bucket_size=self.exemplar_quality_bucket_size,
            exemplar_pose_bucket_size=self.exemplar_pose_bucket_size,
            restricted_pose_recovery_enabled=restricted_pose_cfg.get("enabled", True),
            restricted_pose_mature_cluster_min_faces=restricted_pose_cfg.get(
                "mature_cluster_min_faces", 8
            ),
            auto_correction_enabled=auto_cfg.get("enabled", True),
            auto_correction_max_actions=auto_cfg.get("max_actions_per_run", 12),
            small_fragment_max_faces=auto_cfg.get("small_fragment_max_faces"),
            mature_cluster_min_faces=auto_cfg.get("mature_cluster_min_faces", 8),
            fragment_max_target_ratio=auto_cfg.get("fragment_max_target_ratio", 0.35),
            member_bridge_min_target_support=auto_cfg.get("member_bridge_min_target_support", 2),
            member_bridge_min_high_conf_source_faces=auto_cfg.get("member_bridge_min_high_conf_source_faces", 2),
            member_bridge_min_strong_source_faces=auto_cfg.get("member_bridge_min_strong_source_faces", 2),
            mutual_auto_min_strong_anchors_per_direction=auto_cfg.get("mutual_auto_min_strong_anchors_per_direction", 2),
            auto_split_min_group_faces=auto_cfg.get("auto_split_min_group_faces", 3),
        )

    def close(self):
        self.store.close()
        if self._owns_runtime:
            self.runtime.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ------------------------------------------------------------------
    # Photo ingestion (Steps 1-4, persisted via Step 7)
    # ------------------------------------------------------------------

    def process_photo(
        self,
        image_path: str,
        *,
        photo_id: Optional[str] = None,
        source_ref: Optional[str] = None,
    ) -> List[Face]:
        faces, _ = self.process_photo_with_status(
            image_path, photo_id=photo_id, source_ref=source_ref
        )
        return faces

    def process_photo_with_status(
        self,
        image_path: str,
        *,
        photo_id: Optional[str] = None,
        source_ref: Optional[str] = None,
    ) -> tuple[List[Face], bool]:
        """Process one photo with an SQLite ownership lease.

        Returns ``(faces, cached)``. The lease closes the concurrent-retry race:
        only the current token owner may commit inference results.
        """
        if source_ref is not None:
            normalized_path = source_ref.strip()
            if not normalized_path:
                raise ValueError("source_ref must be non-empty when provided")
        else:
            normalized_path = normalize_image_path(image_path)
        existing_by_path = self.store.get_photo_by_path(normalized_path) if photo_id is None else None
        resolved_photo_id = photo_id or (
            existing_by_path.photo_id if existing_by_path else str(uuid.uuid4())
        )

        claim = self.store.claim_photo_processing(
            photo_id=resolved_photo_id,
            image_path=normalized_path,
            embedding_model_version=self.embedding_model_version,
            config_version=self.config_version,
            lease_seconds=self.photo_processing_lease_seconds,
        )
        if claim.status == "completed":
            return self.store.load_faces_by_photo(resolved_photo_id), True
        if claim.status == "in_progress":
            raise PhotoProcessingInProgressError(
                resolved_photo_id, claim.retry_after_seconds
            )
        token = claim.token
        assert token is not None

        image_bgr = _read_image_bgr(image_path)
        if image_bgr is None:
            self.store.fail_photo_processing_claim(
                resolved_photo_id, token, f"Could not read image: {image_path}"
            )
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image_height, image_width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        photo = Photo(
            photo_id=resolved_photo_id,
            image_path=normalized_path,
            image_width=image_width,
            image_height=image_height,
            processing_status=PhotoProcessingStatus.PROCESSING,
            embedding_model_version=self.embedding_model_version,
            config_version=self.config_version,
        )

        pending_faces = []
        try:
            with self.runtime.inference_lock:
                detections = self.detector.detect(image_rgb)
                landmarks_by_detection = self.landmarker.detect_for_detections(
                    image_rgb, detections
                )
                for detection_index, (det, landmarks) in enumerate(
                    zip(detections, landmarks_by_detection)
                ):
                    if landmarks is None:
                        continue

                    aligned = self._align_face(image_rgb, landmarks)
                    quality = self._compute_face_quality(det, landmarks, aligned)

                    if quality.hard_excluded and not quality.recognition_restricted:
                        continue

                    face = Face(
                        face_id=str(uuid.uuid4()),
                        embedding=self.embedder.embed(aligned.image),
                        quality_score=quality.quality_score,
                        yaw_ratio=quality.yaw_ratio,
                        created_at=utcnow_naive(),
                        assignment_state=AssignmentState.UNASSIGNED,
                        photo_id=photo.photo_id,
                        face_index=detection_index,
                        bbox_x1=float(det.x),
                        bbox_y1=float(det.y),
                        bbox_x2=float(det.x2),
                        bbox_y2=float(det.y2),
                        detection_score=float(det.confidence),
                        embedding_model_version=self.embedding_model_version,
                        config_version=self.config_version,
                        recognition_restricted=bool(quality.recognition_restricted),
                        recognition_restriction_reason=(
                            quality.recognition_restriction_reason or None
                        ),
                    )
                    pending_faces.append((face, quality.exemplar_eligible))

            with self.store.transaction():
                try:
                    self.store.assert_photo_processing_claim(photo.photo_id, token)
                except RuntimeError as exc:
                    raise PhotoProcessingLeaseLostError(photo.photo_id) from exc
                self.store.save_photo(photo)
                assigned_clusters_in_photo = set()
                for face, exemplar_eligible in pending_faces:
                    if face.recognition_restricted:
                        face.cluster_id = None
                        face.assignment_state = AssignmentState.UNASSIGNED
                        face.candidate_cluster_id = None
                        face.best_match_score = None
                        face.second_best_cluster_id = None
                        face.second_best_score = None
                        face.score_margin = None
                        face.decision_threshold = None
                        face.decision_reason = "pose_restricted_pending_consolidation"
                        self.store.save_face(face)
                        continue

                    self._assign_face(
                        face,
                        exemplar_eligible=exemplar_eligible,
                        excluded_cluster_ids=assigned_clusters_in_photo,
                    )
                    if face.cluster_id is not None:
                        assigned_clusters_in_photo.add(face.cluster_id)
                photo.processing_status = PhotoProcessingStatus.COMPLETED
                photo.processed_at = utcnow_naive()
                photo.error_message = None
                self.store.save_photo(photo)
                self.store.complete_photo_processing_claim(photo.photo_id, token)
            return [face for face, _ in pending_faces], False
        except PhotoProcessingLeaseLostError:
            raise
        except Exception as exc:
            # Only the current owner may mark failure. If the lease was already
            # reclaimed, leave the newer worker's state untouched.
            self.store.fail_photo_processing_claim(photo.photo_id, token, str(exc))
            raise

    def _assign_face(
        self,
        face: Face,
        exemplar_eligible: bool,
        excluded_cluster_ids=None,
    ):
        """Assign while respecting same-photo cannot-link constraints."""
        return self.incremental_assigner.assign_face(
            face,
            exemplar_eligible=exemplar_eligible,
            excluded_cluster_ids=set(excluded_cluster_ids or ()),
        )

    # ------------------------------------------------------------------
    # Consolidation (Step 5) -> Suggestions (Step 6)
    # ------------------------------------------------------------------

    def run_consolidation(self, *, consolidation_token: Optional[str] = None) -> dict:
        """Run safe local consolidation as one atomic database operation.

        When a production lifecycle claim token is supplied, resetting the
        cadence happens inside this same SQLite transaction. That prevents a
        photo completed immediately after consolidation from being accidentally
        erased from the next cadence window.
        """
        with self.store.transaction():
            # Pass 1: recover deferred evidence and discover genuinely new
            # people exactly as before.
            recovery = self.consolidation_engine.recover_deferred_faces()
            new_people = self.consolidation_engine.create_clusters_from_unassigned()

            # Pass 2: structural correction of already-confirmed history
            # artifacts. Each action re-audits from fresh state before the next
            # action, so stale pairwise evidence cannot cascade.
            auto = self.consolidation_engine.apply_high_confidence_auto_corrections()

            # A merge/split rebuilds affected exemplars. Give still-deferred
            # faces one new immutable-snapshot chance against that corrected
            # representation.
            post_recovery = (
                self.consolidation_engine.recover_deferred_faces()
                if auto["auto_correction_actions"]
                else {
                    "deferred_checked": 0,
                    "recovered_confirmed": 0,
                    "remaining_ambiguous": 0,
                    "remaining_unassigned": 0,
                }
            )

            # Pose-only restricted faces are deliberately evaluated last,
            # against the most mature/corrected representation available in
            # this consolidation run. They never seed clusters or exemplars.
            restricted_pose = self.consolidation_engine.recover_restricted_pose_faces()

            # Suggestions are generated only from the final corrected state.
            # Anything confident enough for automation has already been acted
            # on; borderline evidence remains human-controlled.
            audit = self.consolidation_engine.audit_confirmed_clusters()
            merge_suggestions = build_merge_suggestions(audit.merge_candidates)
            split_suggestions = build_split_suggestions(audit.split_candidates)
            inserted_merge = sum(self.store.save_suggestion(s) for s in merge_suggestions)
            inserted_split = sum(self.store.save_suggestion(s) for s in split_suggestions)

            remaining_ambiguous = len(
                self.store.load_faces_by_assignment_state(AssignmentState.AMBIGUOUS)
            )
            remaining_unassigned = len(
                self.store.load_faces_by_assignment_state(AssignmentState.UNASSIGNED)
            )

            if consolidation_token is not None:
                self.store.complete_consolidation_claim(consolidation_token)

        # Keep the original summary keys for Step-9 compatibility, while adding
        # the Stage-2 counters needed to explain what actually changed.
        return {
            "clean_matches": recovery["recovered_confirmed"] + post_recovery["recovered_confirmed"],
            "merge_suggestions": inserted_merge,
            "split_suggestions": inserted_split,
            "new_person_labels": new_people["new_person_labels"],
            "noise_labels": new_people["noise_labels"],
            "deferred_checked": recovery["deferred_checked"] + post_recovery["deferred_checked"],
            "initial_deferred_checked": recovery["deferred_checked"],
            "post_correction_deferred_checked": post_recovery["deferred_checked"],
            "recovered_confirmed": recovery["recovered_confirmed"] + post_recovery["recovered_confirmed"],
            "initial_recovered_confirmed": recovery["recovered_confirmed"],
            "post_correction_recovered_confirmed": post_recovery["recovered_confirmed"],
            "restricted_pose_checked": restricted_pose["restricted_pose_checked"],
            "restricted_pose_recovered_confirmed": restricted_pose["restricted_pose_recovered_confirmed"],
            "remaining_restricted_pose": restricted_pose["remaining_restricted_pose"],
            "auto_merges": auto["auto_merges"],
            "auto_splits": auto["auto_splits"],
            "auto_correction_actions": auto["auto_correction_actions"],
            "auto_correction_events": auto["events"],
            "remaining_ambiguous": remaining_ambiguous,
            "remaining_unassigned": remaining_unassigned,
            "unassigned_hdbscan_points": new_people["unassigned_hdbscan_points"],
            "new_clusters": new_people["new_clusters"],
            "new_cluster_faces": new_people["new_cluster_faces"],
            "suspicious_faces": len(audit.suspicious_face_ids),
        }

    # ------------------------------------------------------------------
    # Suggestion resolution & corrections (Step 6)
    # ------------------------------------------------------------------

    def resolve_suggestion(self, suggestion_id: str, status: SuggestionStatus) -> None:
        """Resolve a rare merge/split suggestion after an explicit user choice."""
        with self.store.transaction():
            suggestion = self.store.load_suggestion(suggestion_id)
            if suggestion is None or suggestion.status != SuggestionStatus.PENDING:
                raise ValueError(f"No pending suggestion with id {suggestion_id}")

            if status == SuggestionStatus.REJECTED:
                if suggestion.suggestion_type == SuggestionType.MERGE:
                    self.store.add_cannot_link(
                        suggestion.cluster_ids[0],
                        suggestion.cluster_ids[1],
                        reason="user_rejected_merge",
                    )
                self.store.update_suggestion_status(suggestion_id, status)
                return

            if status == SuggestionStatus.UNCERTAIN:
                # Stable suggestion IDs mean this exact prompt will not keep resurfacing.
                self.store.update_suggestion_status(suggestion_id, status)
                return

            if status != SuggestionStatus.ACCEPTED:
                raise ValueError(f"Unsupported resolution status: {status.value}")

            try:
                if suggestion.suggestion_type == SuggestionType.MERGE:
                    cluster_ids = suggestion.payload.get("cluster_ids", suggestion.cluster_ids)
                    if len(cluster_ids) != 2:
                        raise ValueError("STALE: merge payload is invalid")
                    self.store.execute_merge_atomic(
                        cluster_ids[0],
                        cluster_ids[1],
                        exemplar_quality_threshold=self.exemplar_quality_threshold,
                        suggestion_id=suggestion_id,
                    )
                else:
                    source_cluster_id = suggestion.payload.get("source_cluster_id")
                    groups = suggestion.payload.get("groups")
                    if not source_cluster_id or not groups:
                        raise ValueError("STALE: split payload is missing exact face groups")
                    self.store.execute_split_atomic(
                        source_cluster_id,
                        groups,
                        exemplar_quality_threshold=self.exemplar_quality_threshold,
                        suggestion_id=suggestion_id,
                    )
            except ValueError as exc:
                if str(exc).startswith("STALE:"):
                    self.store.update_suggestion_status(suggestion_id, SuggestionStatus.STALE)
                    return
                raise

            self.store.update_suggestion_status(suggestion_id, SuggestionStatus.ACCEPTED)

    def move_faces_manually(self, face_ids: List[str], to_cluster_id: str, from_cluster_id: str = None) -> int:
        """Explicit user correction only; the algorithm never calls this automatically."""
        return self.store.manual_move_faces_atomic(
            face_ids,
            to_cluster_id,
            from_cluster_id=from_cluster_id,
            exemplar_quality_threshold=self.exemplar_quality_threshold,
        )

    def move_photos_manually(
        self,
        photo_ids: List[str],
        from_cluster_id: str,
        to_cluster_id: str,
    ) -> int:
        """Move the face responsible for each selected photo's source-cluster membership."""
        return self.store.manual_move_photos_atomic(
            photo_ids,
            from_cluster_id,
            to_cluster_id,
            exemplar_quality_threshold=self.exemplar_quality_threshold,
        )

    def apply_correction(self, face_id: str, to_cluster_id: str) -> None:
        """Backward-compatible single-face manual move."""
        face = self.store.load_face(face_id)
        if face is None:
            raise ValueError(f"No face with id {face_id}")
        if face.cluster_id is None:
            raise ValueError(f"Face {face_id} isn't assigned to any cluster; nothing to move it from.")
        self.move_faces_manually([face_id], to_cluster_id, from_cluster_id=face.cluster_id)

    def delete_photo(self, photo_id: str) -> None:
        self.store.delete_photo_atomic(
            photo_id, exemplar_quality_threshold=self.exemplar_quality_threshold
        )

    def validate_storage(self) -> List[str]:
        return self.store.validate_integrity(expected_embedding_dim=512)

    # ------------------------------------------------------------------
    # Lifecycle (Step 6)
    # ------------------------------------------------------------------

    def get_visible_clusters(self) -> dict:
        active_clusters = self.store.load_all_clusters(include_merged=False)
        grouped = {state: [] for state in VisibilityState}
        for cluster in active_clusters:
            grouped[get_visibility(cluster)].append(cluster)
        return grouped

    def run_pruning(self, now: datetime = None) -> int:
        if now is None:
            now = utcnow_naive()
        active_clusters = self.store.load_all_clusters(include_merged=False)
        to_prune = find_clusters_to_prune(active_clusters, now)
        for cluster in to_prune:
            self.store.delete_cluster(cluster.cluster_id)
        return len(to_prune)