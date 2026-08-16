"""Incremental cluster scoring, decision, and persistence application."""
import uuid
from datetime import datetime
from typing import Iterable, List, Optional, Set

from face_grouping.clustering.data_types import Cluster, Face
from face_grouping.matching.assignment import (
    AssignmentDecision,
    AssignmentState,
    ClusterCandidate,
    decide_assignment,
)
from face_grouping.matching.exemplars import Exemplar, ExemplarSet
from face_grouping.matching.similarity import top_k_average_similarity
from face_grouping.storage.store import embedding_versions_compatible


class IncrementalAssigner:
    """Assign one new face without allowing uncertain evidence to cascade."""

    def __init__(
        self,
        *,
        store,
        t_match: float,
        band_width: float,
        top_k: int,
        sparse_cluster_margin: float,
        exemplar_admission_margin: float,
        min_cluster_margin: float,
        exemplar_quality_bucket_size: int,
        exemplar_pose_bucket_size: int,
    ):
        self.store = store
        self.t_match = t_match
        self.band_width = band_width
        self.top_k = top_k
        self.sparse_cluster_margin = sparse_cluster_margin
        self.exemplar_admission_margin = exemplar_admission_margin
        self.min_cluster_margin = min_cluster_margin
        self.exemplar_quality_bucket_size = exemplar_quality_bucket_size
        self.exemplar_pose_bucket_size = exemplar_pose_bucket_size

    def score_clusters(
        self,
        face: Face,
        active_clusters: Iterable[Cluster],
        *,
        excluded_cluster_ids: Optional[Set[str]] = None,
    ) -> List[ClusterCandidate]:
        excluded_cluster_ids = excluded_cluster_ids or set()
        candidates: List[ClusterCandidate] = []
        for cluster in active_clusters:
            if cluster.cluster_id in excluded_cluster_ids:
                continue
            exemplars = cluster.exemplar_set.all_exemplars()
            if not exemplars:
                # Legacy empty clusters are repaired on pipeline startup. Keep
                # this defensive guard so a malformed row cannot influence a
                # decision before repair completes.
                continue

            incompatible = [
                exemplar.embedding_model_version
                for exemplar in exemplars
                if not embedding_versions_compatible(
                    face.embedding_model_version, exemplar.embedding_model_version
                )
            ]
            if incompatible:
                raise ValueError(
                    f"Cannot compare face model {face.embedding_model_version!r} "
                    f"with cluster {cluster.cluster_id} exemplar model {incompatible[0]!r}"
                )

            exemplar_embeddings = [exemplar.embedding for exemplar in exemplars]
            score, similarities = top_k_average_similarity(
                face.embedding,
                exemplar_embeddings,
                k=self.top_k,
            )
            is_sparse = len(exemplar_embeddings) < self.top_k
            effective_threshold = self.t_match + (
                self.sparse_cluster_margin if is_sparse else 0.0
            )
            candidates.append(
                ClusterCandidate(
                    cluster_id=cluster.cluster_id,
                    score=score,
                    exemplar_count=len(exemplar_embeddings),
                    effective_threshold=effective_threshold,
                    similarities=similarities,
                )
            )

        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)

    def evaluate_face(
        self,
        face: Face,
        active_clusters: Iterable[Cluster],
        *,
        exemplar_eligible: bool,
        excluded_cluster_ids: Optional[Set[str]] = None,
    ) -> AssignmentDecision:
        """Purely evaluate a face against a supplied cluster snapshot."""
        candidates = self.score_clusters(
            face, active_clusters, excluded_cluster_ids=excluded_cluster_ids
        )
        return decide_assignment(
            candidates,
            exemplar_eligible=exemplar_eligible,
            ambiguous_band_width=self.band_width,
            min_cluster_margin=self.min_cluster_margin,
        )

    def assign_face(
        self,
        face: Face,
        *,
        exemplar_eligible: bool,
        excluded_cluster_ids: Optional[Set[str]] = None,
    ) -> AssignmentDecision:
        with self.store.transaction():
            now = datetime.utcnow()
            active_clusters = self.store.load_all_clusters(include_merged=False)
            decision = self.evaluate_face(
                face,
                active_clusters,
                exemplar_eligible=exemplar_eligible,
                excluded_cluster_ids=excluded_cluster_ids,
            )
            self._apply_decision(
                face,
                decision,
                active_clusters=active_clusters,
                exemplar_eligible=exemplar_eligible,
                now=now,
            )
        return decision

    def _apply_decision(
        self,
        face: Face,
        decision: AssignmentDecision,
        *,
        active_clusters: List[Cluster],
        exemplar_eligible: bool,
        now: datetime,
    ) -> None:
        self.copy_decision_metadata(face, decision)

        if decision.create_new_cluster:
            if not exemplar_eligible:
                raise AssertionError("A new active cluster must be seeded by an exemplar-eligible face")

            target_cluster = Cluster(
                cluster_id=f"cluster_{uuid.uuid4().hex[:8]}",
                exemplar_set=ExemplarSet(
                    quality_bucket_size=self.exemplar_quality_bucket_size,
                    pose_bucket_size=self.exemplar_pose_bucket_size,
                ),
                face_count=1,
                created_at=now,
                last_updated_at=now,
            )
            face.cluster_id = target_cluster.cluster_id
            face.assignment_state = AssignmentState.CONFIRMED
            added = target_cluster.exemplar_set.try_add(self._to_exemplar(face))
            if not added or len(target_cluster.exemplar_set) == 0:
                raise AssertionError("New active cluster was not seeded with an exemplar")

            self.store.save_cluster(target_cluster)
            self.store.save_face(face)
            target_cluster.face_count = self.store.recompute_cluster_face_count(
                target_cluster.cluster_id
            )
            return

        if decision.state == AssignmentState.CONFIRMED:
            target_cluster = next(
                (cluster for cluster in active_clusters if cluster.cluster_id == decision.assigned_cluster_id),
                None,
            )
            if target_cluster is None:
                raise ValueError(
                    f"Confirmed assignment references missing cluster {decision.assigned_cluster_id}"
                )

            face.cluster_id = target_cluster.cluster_id
            target_cluster.face_count += 1
            target_cluster.last_updated_at = now

            # Membership and exemplar admission have different risk. A face
            # may join confidently but still be barred from shaping future
            # decisions unless it clears the stricter cluster-specific bar.
            exemplar_threshold = (
                (decision.decision_threshold or self.t_match)
                + self.exemplar_admission_margin
            )
            if (
                exemplar_eligible
                and decision.best_score is not None
                and decision.best_score >= exemplar_threshold
            ):
                target_cluster.exemplar_set.try_add(self._to_exemplar(face))

            self.store.save_cluster(target_cluster)
            self.store.save_face(face)
            target_cluster.face_count = self.store.recompute_cluster_face_count(
                target_cluster.cluster_id
            )
            return

        # AMBIGUOUS and UNASSIGNED are persisted for later consolidation,
        # but are not members, do not change face_count, and cannot become
        # exemplars or influence any future incremental decision.
        face.cluster_id = None
        self.store.save_face(face)

    @staticmethod
    def _to_exemplar(face: Face) -> Exemplar:
        return Exemplar(
            embedding=face.embedding,
            quality_score=face.quality_score,
            yaw_ratio=face.yaw_ratio,
            face_id=face.face_id,
            embedding_model_version=face.embedding_model_version,
        )

    @staticmethod
    def copy_decision_metadata(face: Face, decision: AssignmentDecision) -> None:
        face.assignment_state = decision.state
        face.candidate_cluster_id = decision.candidate_cluster_id
        face.best_match_score = decision.best_score
        face.second_best_cluster_id = decision.second_best_cluster_id
        face.second_best_score = decision.second_best_score
        face.score_margin = decision.score_margin
        face.decision_threshold = decision.decision_threshold
        face.decision_reason = decision.reason

# Backward-compatible private alias.
IncrementalAssigner._copy_decision_metadata = staticmethod(IncrementalAssigner.copy_decision_metadata)
