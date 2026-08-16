"""Precision-first consolidation and late-stage structural correction.

Deferred recovery still uses immutable snapshots, but consolidation can now
repair a very narrow class of high-confidence history-dependent errors:
history-created confirmed fragments may be reconciled into mature clusters
using repeated member-bridge evidence, fully mutual cluster pairs may be
auto-merged at a stricter tier than normal suggestions, and an exceptionally
clear two-way contamination may be auto-split. Singleton clusters are never
auto-merged: a one-photo person is valid gallery evidence, not a fragment by
default. Borderline cases remain suggestions. No production matching threshold
is raised.
"""
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import uuid

import numpy as np

from face_grouping.clustering.data_types import Cluster, Face
from face_grouping.clustering.hdbscan_runner import run_hdbscan
from face_grouping.clustering.candidates import (
    CLEAN_MATCH_THRESHOLD,
    AutoMergeCandidate,
    MergeCandidate,
    SplitCandidate,
)
from face_grouping.matching.assignment import AssignmentDecision, AssignmentState
from face_grouping.matching.exemplars import Exemplar, ExemplarSet
from face_grouping.matching.similarity import cosine_similarity, top_k_average_similarity
from face_grouping.storage.store import embedding_versions_compatible, LEGACY_VERSION


@dataclass
class ConsolidationAudit:
    merge_candidates: List[MergeCandidate]
    split_candidates: List[SplitCandidate]
    suspicious_face_ids: List[str]
    auto_merge_candidates: List[AutoMergeCandidate]
    auto_split_candidates: List[SplitCandidate]
    auto_merge_evaluations: List[dict]


class ConsolidationEngine:
    def __init__(
        self,
        *,
        store,
        assigner,
        t_match: float,
        band_width: float,
        top_k: int,
        sparse_cluster_margin: float,
        min_cluster_margin: float,
        exemplar_admission_margin: float,
        exemplar_quality_threshold: float,
        exemplar_quality_bucket_size: int,
        exemplar_pose_bucket_size: int,
        auto_correction_enabled: bool = True,
        auto_correction_max_actions: int = 12,
        # ``small_fragment_max_faces`` is accepted only for backward
        # compatibility with v1 callers. v2 defines a fragment relatively
        # (source/target size ratio), not by an arbitrary absolute cap.
        small_fragment_max_faces: Optional[int] = None,
        mature_cluster_min_faces: int = 8,
        fragment_max_target_ratio: float = 0.35,
        member_bridge_min_target_support: int = 2,
        member_bridge_min_high_conf_source_faces: int = 2,
        member_bridge_min_strong_source_faces: int = 2,
        mutual_auto_min_strong_anchors_per_direction: int = 2,
        auto_split_min_group_faces: int = 3,
    ):
        self.store = store
        self.assigner = assigner
        self.t_match = t_match
        self.band_width = band_width
        self.top_k = top_k
        self.sparse_cluster_margin = sparse_cluster_margin
        self.min_cluster_margin = min_cluster_margin
        self.exemplar_admission_margin = exemplar_admission_margin
        self.exemplar_quality_threshold = exemplar_quality_threshold
        self.exemplar_quality_bucket_size = exemplar_quality_bucket_size
        self.exemplar_pose_bucket_size = exemplar_pose_bucket_size
        self.auto_correction_enabled = bool(auto_correction_enabled)
        self.auto_correction_max_actions = max(0, int(auto_correction_max_actions))
        # Kept only so older construction code does not break. It no longer
        # decides whether a cluster is a fragment.
        self.small_fragment_max_faces = (
            None if small_fragment_max_faces is None else max(2, int(small_fragment_max_faces))
        )
        self.mature_cluster_min_faces = max(2, int(mature_cluster_min_faces))
        self.fragment_max_target_ratio = float(fragment_max_target_ratio)
        if not (0.0 < self.fragment_max_target_ratio < 1.0):
            raise ValueError("fragment_max_target_ratio must be in (0, 1)")
        self.member_bridge_min_target_support = max(2, int(member_bridge_min_target_support))
        self.member_bridge_min_high_conf_source_faces = max(1, int(member_bridge_min_high_conf_source_faces))
        self.member_bridge_min_strong_source_faces = max(1, int(member_bridge_min_strong_source_faces))
        self.mutual_auto_min_strong_anchors_per_direction = max(1, int(mutual_auto_min_strong_anchors_per_direction))
        self.auto_split_min_group_faces = max(2, int(auto_split_min_group_faces))

    # ------------------------------------------------------------------
    # Deferred-face recovery
    # ------------------------------------------------------------------

    def recover_deferred_faces(self) -> Dict[str, int]:
        """Re-evaluate deferred faces against one immutable cluster snapshot.

        ``exemplar_eligible=False`` is intentional while deciding: a deferred
        face with no existing match must remain deferred for group-level new-
        person discovery, not create a singleton cluster during re-checking.
        """
        active_clusters = self.store.load_all_clusters(include_merged=False)
        deferred = (
            self.store.load_faces_by_assignment_state(AssignmentState.AMBIGUOUS)
            + self.store.load_faces_by_assignment_state(AssignmentState.UNASSIGNED)
        )
        deferred = [f for f in deferred if not f.is_manually_corrected]

        confirmed_by_photo = defaultdict(set)
        for cluster in active_clusters:
            for member in self.store.load_faces_by_cluster(cluster.cluster_id):
                if (
                    member.photo_id is not None
                    and member.assignment_state in (AssignmentState.CONFIRMED, AssignmentState.MANUAL)
                ):
                    confirmed_by_photo[member.photo_id].add(cluster.cluster_id)

        evaluated: List[Tuple[Face, AssignmentDecision]] = []
        for face in deferred:
            decision = self.assigner.evaluate_face(
                face,
                active_clusters,
                exemplar_eligible=False,
                excluded_cluster_ids=(
                    confirmed_by_photo.get(face.photo_id, set())
                    if face.photo_id is not None
                    else set()
                ),
            )
            evaluated.append((face, decision))

        cluster_map = {cluster.cluster_id: cluster for cluster in active_clusters}
        recovered_by_cluster: Dict[str, List[Face]] = defaultdict(list)
        remaining_ambiguous = 0
        remaining_unassigned = 0

        # Apply only after every decision was computed, so recovered faces do
        # not change exemplars or scores for later faces in this same run.
        for face, decision in evaluated:
            self.assigner.copy_decision_metadata(face, decision)
            if (
                decision.state == AssignmentState.CONFIRMED
                and decision.assigned_cluster_id in cluster_map
            ):
                face.cluster_id = decision.assigned_cluster_id
                face.decision_reason = f"consolidation_recovery:{decision.reason}"
                recovered_by_cluster[decision.assigned_cluster_id].append(face)
            else:
                face.cluster_id = None
                face.decision_reason = f"consolidation_recheck:{decision.reason}"
                if face.assignment_state == AssignmentState.AMBIGUOUS:
                    remaining_ambiguous += 1
                else:
                    remaining_unassigned += 1
                self.store.save_face(face)

        now = datetime.utcnow()
        recovered = 0
        for cluster_id, faces in recovered_by_cluster.items():
            cluster = cluster_map[cluster_id]
            cluster.face_count += len(faces)
            cluster.last_updated_at = now
            self.store.save_cluster(cluster)
            for face in faces:
                # Recovery intentionally does not update exemplars in this run.
                self.store.save_face(face)
            cluster.face_count = self.store.recompute_cluster_face_count(cluster_id)
            recovered += len(faces)

        return {
            "deferred_checked": len(evaluated),
            "recovered_confirmed": recovered,
            "remaining_ambiguous": remaining_ambiguous,
            "remaining_unassigned": remaining_unassigned,
        }

    # ------------------------------------------------------------------
    # New-person discovery from still-unassigned faces
    # ------------------------------------------------------------------

    def create_clusters_from_unassigned(self) -> Dict[str, int]:
        unassigned = [
            face
            for face in self.store.load_faces_by_assignment_state(AssignmentState.UNASSIGNED)
            if not face.is_manually_corrected
        ]
        unique = {face.face_id: face for face in unassigned}
        faces = list(unique.values())
        known_versions = {
            face.embedding_model_version
            for face in faces
            if face.embedding_model_version != LEGACY_VERSION
        }
        if len(known_versions) > 1:
            raise ValueError(
                f"Cannot consolidate unassigned faces from incompatible embedding models: "
                f"{sorted(known_versions)}"
            )
        if len(faces) < 2:
            return {
                "unassigned_hdbscan_points": len(faces),
                "new_person_labels": 0,
                "new_clusters": 0,
                "new_cluster_faces": 0,
                "noise_labels": len(faces),
            }

        labels = run_hdbscan(
            [face.embedding for face in faces],
            min_cluster_size=2,
            min_samples=1,
            item_ids=[face.face_id for face in faces],
            allow_single_cluster=True,
        )

        new_clusters = 0
        new_cluster_faces = 0
        accepted_labels = 0
        non_noise_labels = sorted({int(label) for label in labels if int(label) != -1})

        single_detected_group = len(non_noise_labels) == 1
        for label in non_noise_labels:
            # sklearn HDBSCAN may mark a border/medoid point as noise even
            # when only one coherent group exists. In that one-group case we
            # validate all points against the trusted medoid below; with
            # multiple groups, noise stays unassigned to avoid order effects.
            group = (
                list(faces)
                if single_detected_group
                else [faces[i] for i, value in enumerate(labels) if int(value) == label]
            )
            photo_ids = [face.photo_id for face in group if face.photo_id is not None]
            if len(photo_ids) != len(set(photo_ids)):
                # Two distinct detections from one photo are a hard cannot-link
                # for person grouping; never seed one identity from both.
                continue

            eligible = [
                face for face in group
                if face.quality_score >= self.exemplar_quality_threshold
            ]
            if not eligible:
                continue

            seed = self._choose_medoid(eligible, group)
            sparse_threshold = self.t_match + self.sparse_cluster_margin
            group_scores = {
                face.face_id: cosine_similarity(face.embedding, seed.embedding)
                for face in group
            }

            # Precision-first validation: every member must independently
            # match the trusted seed at the same stricter threshold used for a
            # one-exemplar incremental cluster.
            if len(group) < 2 or any(
                score < sparse_threshold for score in group_scores.values()
            ):
                continue

            cluster = Cluster(
                cluster_id=f"cluster_{uuid.uuid4().hex[:8]}",
                exemplar_set=ExemplarSet(
                    quality_bucket_size=self.exemplar_quality_bucket_size,
                    pose_bucket_size=self.exemplar_pose_bucket_size,
                ),
                face_count=len(group),
                created_at=datetime.utcnow(),
                last_updated_at=datetime.utcnow(),
            )

            for exemplar_face in sorted(eligible, key=lambda f: f.quality_score, reverse=True):
                cluster.exemplar_set.try_add(self._to_exemplar(exemplar_face))
            if len(cluster.exemplar_set) == 0:
                continue

            self.store.save_cluster(cluster)
            for face in group:
                face.cluster_id = cluster.cluster_id
                face.assignment_state = AssignmentState.CONFIRMED
                face.candidate_cluster_id = cluster.cluster_id
                face.best_match_score = group_scores[face.face_id]
                face.second_best_cluster_id = None
                face.second_best_score = None
                face.score_margin = None
                face.decision_threshold = sparse_threshold
                face.decision_reason = "consolidation_new_person_group"
                self.store.save_face(face)
            cluster.face_count = self.store.recompute_cluster_face_count(cluster.cluster_id)

            accepted_labels += 1
            new_clusters += 1
            new_cluster_faces += len(group)

        rejected_labels = len(non_noise_labels) - accepted_labels
        remaining_unassigned_ids = {
            face.face_id
            for face in self.store.load_faces_by_assignment_state(AssignmentState.UNASSIGNED)
        }
        has_noise_label = int(any(
            int(label) == -1 and faces[i].face_id in remaining_unassigned_ids
            for i, label in enumerate(labels)
        ))
        return {
            "unassigned_hdbscan_points": len(faces),
            "new_person_labels": accepted_labels,
            "new_clusters": new_clusters,
            "new_cluster_faces": new_cluster_faces,
            "noise_labels": rejected_labels + has_noise_label,
        }

    @staticmethod
    def _choose_medoid(eligible: Sequence[Face], group: Sequence[Face]) -> Face:
        """Pick the eligible face with the highest average group similarity."""
        return max(
            eligible,
            key=lambda candidate: sum(
                cosine_similarity(candidate.embedding, other.embedding)
                for other in group
            ) / len(group),
        )

    @staticmethod
    def _to_exemplar(face: Face) -> Exemplar:
        return Exemplar(
            embedding=face.embedding,
            quality_score=face.quality_score,
            yaw_ratio=face.yaw_ratio,
            face_id=face.face_id,
            embedding_model_version=face.embedding_model_version,
        )

    # ------------------------------------------------------------------
    # Conservative audit of already-confirmed clusters
    # ------------------------------------------------------------------

    def audit_confirmed_clusters(self) -> ConsolidationAudit:
        clusters = self.store.load_all_clusters(include_merged=False)
        faces_by_cluster = {
            cluster.cluster_id: [
                face
                for face in self.store.load_faces_by_cluster(cluster.cluster_id)
                if not face.is_manually_corrected
                and face.assignment_state in (AssignmentState.CONFIRMED, AssignmentState.MANUAL)
            ]
            for cluster in clusters
        }

        for cluster_id, faces in faces_by_cluster.items():
            known_versions = {
                face.embedding_model_version
                for face in faces
                if face.embedding_model_version != LEGACY_VERSION
            }
            if len(known_versions) > 1:
                raise ValueError(
                    f"Cluster {cluster_id} mixes incompatible embedding models: "
                    f"{sorted(known_versions)}"
                )

        suspicious_face_ids = self._find_suspicious_faces(clusters, faces_by_cluster)
        merge_candidates = self._find_mutual_merge_candidates(clusters, faces_by_cluster)
        split_candidates = self._find_conservative_split_candidates(
            clusters, faces_by_cluster, set(suspicious_face_ids)
        )
        auto_merge_candidates, auto_merge_evaluations = self._find_auto_merge_candidates_v2(
            clusters, faces_by_cluster
        )
        auto_split_candidates = [
            candidate
            for candidate in split_candidates
            if self._split_candidate_is_auto_safe(candidate, clusters, faces_by_cluster)
        ]
        return ConsolidationAudit(
            merge_candidates=merge_candidates,
            split_candidates=split_candidates,
            suspicious_face_ids=suspicious_face_ids,
            auto_merge_candidates=auto_merge_candidates,
            auto_split_candidates=auto_split_candidates,
            auto_merge_evaluations=auto_merge_evaluations,
        )

    def _score_against_cluster(
        self,
        face: Face,
        cluster: Cluster,
        *,
        exclude_same_face: bool = False,
    ) -> Optional[Tuple[float, float]]:
        exemplars = [
            exemplar
            for exemplar in cluster.exemplar_set.all_exemplars()
            if not (exclude_same_face and exemplar.face_id == face.face_id)
        ]
        if not exemplars:
            return None
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
        score, _ = top_k_average_similarity(
            face.embedding,
            [exemplar.embedding for exemplar in exemplars],
            k=self.top_k,
        )
        threshold = self.t_match + (
            self.sparse_cluster_margin if len(exemplars) < self.top_k else 0.0
        )
        return score, threshold

    def _find_suspicious_faces(
        self,
        clusters: Sequence[Cluster],
        faces_by_cluster: Dict[str, List[Face]],
    ) -> List[str]:
        suspicious: List[str] = []
        for cluster in clusters:
            for face in faces_by_cluster[cluster.cluster_id]:
                own = self._score_against_cluster(face, cluster, exclude_same_face=True)
                own_score = own[0] if own is not None else None

                best_other_score = None
                best_other_threshold = None
                for other in clusters:
                    if other.cluster_id == cluster.cluster_id:
                        continue
                    result = self._score_against_cluster(face, other)
                    if result is None:
                        continue
                    score, threshold = result
                    if best_other_score is None or score > best_other_score:
                        best_other_score = score
                        best_other_threshold = threshold

                if best_other_score is None or best_other_threshold is None:
                    continue
                if (
                    best_other_score >= best_other_threshold
                    and own_score is not None
                    and best_other_score - own_score >= self.min_cluster_margin
                ):
                    suspicious.append(face.face_id)
        return suspicious

    def _find_mutual_merge_candidates(
        self,
        clusters: Sequence[Cluster],
        faces_by_cluster: Dict[str, List[Face]],
    ) -> List[MergeCandidate]:
        candidates: List[MergeCandidate] = []
        for cluster_a, cluster_b in combinations(clusters, 2):
            faces_a = faces_by_cluster[cluster_a.cluster_id]
            faces_b = faces_by_cluster[cluster_b.cluster_id]
            # Product policy: do not bother the user with weak singleton evidence.
            if len(faces_a) < 2 or len(faces_b) < 2:
                continue
            if self.store.has_cannot_link(cluster_a.cluster_id, cluster_b.cluster_id):
                continue
            if self.store.clusters_share_photo_conflict(cluster_a.cluster_id, cluster_b.cluster_id):
                continue

            fraction_a_to_b = self._coverage(faces_a, cluster_b, high_confidence=True)
            fraction_b_to_a = self._coverage(faces_b, cluster_a, high_confidence=True)
            if (
                fraction_a_to_b >= CLEAN_MATCH_THRESHOLD
                and fraction_b_to_a >= CLEAN_MATCH_THRESHOLD
            ):
                candidates.append(
                    MergeCandidate(
                        new_label=-1,
                        contributing_cluster_ids=[
                            cluster_a.cluster_id,
                            cluster_b.cluster_id,
                        ],
                        fractions={
                            cluster_a.cluster_id: fraction_a_to_b,
                            cluster_b.cluster_id: fraction_b_to_a,
                        },
                        total_members=len(faces_a) + len(faces_b),
                    )
                )
        return candidates

    def _coverage(
        self,
        faces: Sequence[Face],
        target: Cluster,
        *,
        high_confidence: bool = False,
    ) -> float:
        matches = 0
        for face in faces:
            result = self._score_against_cluster(face, target)
            if result is None:
                continue
            score, threshold = result
            if high_confidence:
                threshold = max(threshold, self.t_match + self.min_cluster_margin)
            if score >= threshold:
                matches += 1
        return matches / len(faces) if faces else 0.0

    def _cluster_is_automation_protected(self, cluster: Cluster) -> bool:
        """Never let automation override explicit user intent."""
        if cluster.is_user_confirmed or cluster.has_manual_correction:
            return True
        return any(
            face.is_manually_corrected or face.assignment_state == AssignmentState.MANUAL
            for face in self.store.load_faces_by_cluster(cluster.cluster_id)
        )

    def _score_competition_margin(
        self,
        face: Face,
        target: Cluster,
        clusters: Sequence[Cluster],
        source_cluster_id: str,
    ) -> Optional[float]:
        """Return target score minus the best *third-person* cluster score.

        The face's current fragment is excluded: the question during
        reconciliation is whether ``target`` is the unique best alternative,
        not whether it beats the history-created fragment that already owns
        the face.
        """
        target_result = self._score_against_cluster(face, target)
        if target_result is None:
            return None
        target_score, _ = target_result
        competitor_scores = []
        for other in clusters:
            if other.cluster_id in (source_cluster_id, target.cluster_id):
                continue
            result = self._score_against_cluster(face, other)
            if result is not None:
                competitor_scores.append(result[0])
        if not competitor_scores:
            return float("inf")
        return target_score - max(competitor_scores)

    @staticmethod
    def _best_member_similarity(face: Face, target_faces: Sequence[Face]) -> Optional[float]:
        if not target_faces:
            return None
        return max(cosine_similarity(face.embedding, other.embedding) for other in target_faces)

    def _best_third_cluster_member_score(
        self,
        face: Face,
        *,
        clusters: Sequence[Cluster],
        faces_by_cluster: Dict[str, List[Face]],
        excluded_cluster_ids: set,
    ) -> Optional[float]:
        best = None
        for cluster in clusters:
            if cluster.cluster_id in excluded_cluster_ids:
                continue
            value = self._best_member_similarity(face, faces_by_cluster[cluster.cluster_id])
            if value is not None and (best is None or value > best):
                best = value
        return best

    @staticmethod
    def _graph_connected_at_threshold(faces: Sequence[Face], threshold: float) -> bool:
        """Whether member evidence forms one connected component.

        This is deliberately weaker than requiring every pair to match every
        other pair. Appearance changes can form a chain (normal -> bridge ->
        wedding/age/etc.) while still representing one identity.
        """
        if len(faces) <= 1:
            return True
        adjacency = {face.face_id: set() for face in faces}
        for left, right in combinations(faces, 2):
            if cosine_similarity(left.embedding, right.embedding) >= threshold:
                adjacency[left.face_id].add(right.face_id)
                adjacency[right.face_id].add(left.face_id)
        seen = set()
        stack = [faces[0].face_id]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adjacency[node] - seen)
        return len(seen) == len(faces)

    def _pair_is_structurally_safe(self, source: Cluster, target: Cluster) -> Optional[str]:
        if source.cluster_id == target.cluster_id:
            return "same_cluster"
        if self._cluster_is_automation_protected(source):
            return "source_user_protected"
        if self._cluster_is_automation_protected(target):
            return "target_user_protected"
        if self.store.has_cannot_link(source.cluster_id, target.cluster_id):
            return "stored_cannot_link"
        if self.store.clusters_share_photo_conflict(source.cluster_id, target.cluster_id):
            return "same_photo_conflict"
        return None

    def _strong_exemplar_anchor_count(self, faces: Sequence[Face], target: Cluster) -> int:
        count = 0
        for face in faces:
            result = self._score_against_cluster(face, target)
            if result is None:
                continue
            score, threshold = result
            strong_floor = max(threshold, self.t_match + self.exemplar_admission_margin)
            count += int(score >= strong_floor)
        return count

    def _evaluate_mutual_auto_pair(
        self,
        cluster_a: Cluster,
        cluster_b: Cluster,
        *,
        clusters: Sequence[Cluster],
        faces_by_cluster: Dict[str, List[Face]],
    ) -> Tuple[Optional[AutoMergeCandidate], dict]:
        faces_a = faces_by_cluster[cluster_a.cluster_id]
        faces_b = faces_by_cluster[cluster_b.cluster_id]
        diag = {
            "mode": "mutual_full_coverage",
            "cluster_a_id": cluster_a.cluster_id,
            "cluster_b_id": cluster_b.cluster_id,
            "cluster_a_size": len(faces_a),
            "cluster_b_size": len(faces_b),
            "accepted": False,
            "reasons": [],
        }
        # Explicit product rule from Gallery evidence: one-photo people are
        # valid identities. Singleton clusters are not auto-merge material.
        if len(faces_a) < 2 or len(faces_b) < 2:
            diag["reasons"].append("singleton_cluster_protected")
            return None, diag
        blocker = self._pair_is_structurally_safe(cluster_a, cluster_b)
        if blocker:
            diag["reasons"].append(blocker)
            return None, diag

        coverage_a_to_b = self._coverage(faces_a, cluster_b, high_confidence=True)
        coverage_b_to_a = self._coverage(faces_b, cluster_a, high_confidence=True)
        strong_a_to_b = self._strong_exemplar_anchor_count(faces_a, cluster_b)
        strong_b_to_a = self._strong_exemplar_anchor_count(faces_b, cluster_a)
        diag.update({
            "coverage_a_to_b": coverage_a_to_b,
            "coverage_b_to_a": coverage_b_to_a,
            "strong_a_to_b": strong_a_to_b,
            "strong_b_to_a": strong_b_to_a,
        })

        # The normal suggestion tier is 90% mutual coverage. Auto-merge is
        # intentionally stricter: every member in both directions must clear
        # the existing high-confidence floor (100%/100%).
        if coverage_a_to_b < 1.0 or coverage_b_to_a < 1.0:
            diag["reasons"].append("not_full_mutual_high_confidence_coverage")
            return None, diag

        required_a = min(self.mutual_auto_min_strong_anchors_per_direction, len(faces_a))
        required_b = min(self.mutual_auto_min_strong_anchors_per_direction, len(faces_b))
        if strong_a_to_b < required_a or strong_b_to_a < required_b:
            diag["reasons"].append("insufficient_strong_anchors_both_directions")
            return None, diag

        margins: List[float] = []
        for face in faces_a:
            margin = self._score_competition_margin(face, cluster_b, clusters, cluster_a.cluster_id)
            if margin is None or margin < self.min_cluster_margin:
                diag["reasons"].append("third_person_competition_a_to_b")
                return None, diag
            margins.append(margin)
        for face in faces_b:
            margin = self._score_competition_margin(face, cluster_a, clusters, cluster_b.cluster_id)
            if margin is None or margin < self.min_cluster_margin:
                diag["reasons"].append("third_person_competition_b_to_a")
                return None, diag
            margins.append(margin)

        # Orient diagnostics consistently: retire the smaller cluster when
        # sizes differ. The storage layer still chooses the survivor atomically.
        if (len(faces_a), cluster_a.cluster_id) <= (len(faces_b), cluster_b.cluster_id):
            source, target = cluster_a, cluster_b
            source_faces, target_faces = faces_a, faces_b
            source_cov, reverse_cov = coverage_a_to_b, coverage_b_to_a
            source_strong, reverse_strong = strong_a_to_b, strong_b_to_a
        else:
            source, target = cluster_b, cluster_a
            source_faces, target_faces = faces_b, faces_a
            source_cov, reverse_cov = coverage_b_to_a, coverage_a_to_b
            source_strong, reverse_strong = strong_b_to_a, strong_a_to_b

        target_scores = [self._score_against_cluster(face, target)[0] for face in source_faces]
        candidate = AutoMergeCandidate(
            mode="mutual_full_coverage",
            source_cluster_id=source.cluster_id,
            target_cluster_id=target.cluster_id,
            source_size=len(source_faces),
            target_size=len(target_faces),
            source_coverage=source_cov,
            reverse_coverage=reverse_cov,
            strong_anchor_count=source_strong,
            reverse_strong_anchor_count=reverse_strong,
            min_target_score=min(target_scores),
            mean_target_score=float(sum(target_scores) / len(target_scores)),
            min_competition_margin=min(margins) if margins else float("inf"),
        )
        diag["accepted"] = True
        return candidate, diag

    def _evaluate_fragment_bridge_pair(
        self,
        source: Cluster,
        target: Cluster,
        *,
        clusters: Sequence[Cluster],
        faces_by_cluster: Dict[str, List[Face]],
    ) -> Tuple[Optional[AutoMergeCandidate], dict]:
        source_faces = faces_by_cluster[source.cluster_id]
        target_faces = faces_by_cluster[target.cluster_id]
        diag = {
            "mode": "fragment_member_bridge",
            "source_cluster_id": source.cluster_id,
            "target_cluster_id": target.cluster_id,
            "source_size": len(source_faces),
            "target_size": len(target_faces),
            "accepted": False,
            "reasons": [],
        }

        # User observation is encoded as a hard structural policy: a cluster
        # with exactly one face may simply be a person who appeared once in the
        # gallery. Never auto-merge it, regardless of raw similarity.
        if len(source_faces) < 2:
            diag["reasons"].append("singleton_source_protected")
            return None, diag
        if len(target_faces) < self.mature_cluster_min_faces:
            diag["reasons"].append("target_not_mature")
            return None, diag
        if len(source_faces) >= len(target_faces):
            diag["reasons"].append("source_not_smaller_than_target")
            return None, diag
        ratio = len(source_faces) / len(target_faces)
        diag["source_target_size_ratio"] = ratio
        if ratio > self.fragment_max_target_ratio:
            diag["reasons"].append("source_not_fragment_like_by_relative_size")
            return None, diag

        blocker = self._pair_is_structurally_safe(source, target)
        if blocker:
            diag["reasons"].append(blocker)
            return None, diag

        # A fragment may contain an appearance regime (wedding make-up, age,
        # occlusion, etc.), so use a connectivity test rather than all-pairs
        # similarity. A disconnected source is too risky to merge automatically.
        lower_boundary = self.t_match - self.band_width
        if not self._graph_connected_at_threshold(source_faces, lower_boundary):
            diag["reasons"].append("source_fragment_not_internally_connected")
            return None, diag

        normal_floor = self.t_match
        high_floor = self.t_match + self.min_cluster_margin
        strong_floor = self.t_match + self.exemplar_admission_margin
        high_source_count = 0
        strong_source_count = 0
        source_with_normal_support = 0
        support_counts: List[int] = []
        best_scores: List[float] = []
        all_margins: List[float] = []
        bridge_face_ids = set()
        bridge_photo_ids = set()

        for face in source_faces:
            scored = [
                (cosine_similarity(face.embedding, other.embedding), other)
                for other in target_faces
            ]
            scored.sort(key=lambda pair: pair[0], reverse=True)
            if not scored:
                diag["reasons"].append("target_has_no_members")
                return None, diag
            best_score = scored[0][0]
            normal_support = [(score, other) for score, other in scored if score >= normal_floor]
            support_counts.append(len(normal_support))
            best_scores.append(best_score)

            if len(normal_support) >= self.member_bridge_min_target_support:
                source_with_normal_support += 1
                for _, other in normal_support:
                    bridge_face_ids.add(other.face_id)
                    if other.photo_id is not None:
                        bridge_photo_ids.add(other.photo_id)

            is_high = best_score >= high_floor
            is_strong = best_score >= strong_floor
            high_source_count += int(is_high)
            strong_source_count += int(is_strong)

            competitor = self._best_third_cluster_member_score(
                face,
                clusters=clusters,
                faces_by_cluster=faces_by_cluster,
                excluded_cluster_ids={source.cluster_id, target.cluster_id},
            )
            margin = float("inf") if competitor is None else best_score - competitor
            all_margins.append(margin)

            # Every source member must at least prefer the proposed target over
            # every third-person cluster. Stronger bridge faces must beat the
            # competition by the production min-cluster margin.
            if competitor is not None and best_score <= competitor:
                diag["reasons"].append("target_not_best_member_level_alternative")
                return None, diag
            if is_high and margin < self.min_cluster_margin:
                diag["reasons"].append("high_conf_bridge_has_third_person_competition")
                return None, diag

        member_coverage = source_with_normal_support / len(source_faces)
        diag.update({
            "member_bridge_source_coverage": member_coverage,
            "high_conf_bridge_source_count": high_source_count,
            "strong_bridge_source_count": strong_source_count,
            "min_member_support_count": min(support_counts),
            "distinct_bridge_target_faces": len(bridge_face_ids),
            "distinct_bridge_target_photos": len(bridge_photo_ids),
            "min_best_member_score": min(best_scores),
            "mean_best_member_score": float(sum(best_scores) / len(best_scores)),
            "min_member_competition_margin": min(all_margins),
        })

        if member_coverage < 1.0:
            diag["reasons"].append("not_every_source_face_has_repeated_target_member_support")
            return None, diag
        if min(support_counts) < self.member_bridge_min_target_support:
            diag["reasons"].append("insufficient_repeated_target_member_support")
            return None, diag

        required_high = min(self.member_bridge_min_high_conf_source_faces, len(source_faces))
        required_strong = min(self.member_bridge_min_strong_source_faces, len(source_faces))
        if high_source_count < required_high:
            diag["reasons"].append("insufficient_high_conf_bridge_source_faces")
            return None, diag
        if strong_source_count < required_strong:
            diag["reasons"].append("insufficient_strong_bridge_source_faces")
            return None, diag
        if len(bridge_face_ids) < self.member_bridge_min_target_support:
            diag["reasons"].append("bridge_depends_on_too_few_target_faces")
            return None, diag
        if len(bridge_photo_ids) < self.member_bridge_min_target_support:
            diag["reasons"].append("bridge_depends_on_too_few_target_photos")
            return None, diag

        candidate = AutoMergeCandidate(
            mode="fragment_member_bridge",
            source_cluster_id=source.cluster_id,
            target_cluster_id=target.cluster_id,
            source_size=len(source_faces),
            target_size=len(target_faces),
            source_coverage=member_coverage,
            strong_anchor_count=strong_source_count,
            min_target_score=min(best_scores),
            mean_target_score=float(sum(best_scores) / len(best_scores)),
            min_competition_margin=min(all_margins),
            member_bridge_source_coverage=member_coverage,
            high_conf_bridge_source_count=high_source_count,
            strong_bridge_source_count=strong_source_count,
            min_member_support_count=min(support_counts),
            distinct_bridge_target_faces=len(bridge_face_ids),
            distinct_bridge_target_photos=len(bridge_photo_ids),
        )
        diag["accepted"] = True
        return candidate, diag

    def _find_auto_merge_candidates_v2(
        self,
        clusters: Sequence[Cluster],
        faces_by_cluster: Dict[str, List[Face]],
    ) -> Tuple[List[AutoMergeCandidate], List[dict]]:
        """Two-tier structural auto-merge policy.

        Tier A (mutual): 100%/100% high-confidence exemplar coverage plus
        strong anchors and third-person separation. This is deliberately above
        the normal 90%/90% human suggestion tier.

        Tier B (fragment): repeated member-to-member bridge evidence from a
        relatively small cluster into a mature cluster. This recovers
        appearance modes absent from the five current exemplars without
        lowering ``t_match``.
        """
        if not self.auto_correction_enabled:
            return [], []

        candidates: List[AutoMergeCandidate] = []
        evaluations: List[dict] = []

        # Tier A: unordered pair, any non-singleton sizes.
        for cluster_a, cluster_b in combinations(clusters, 2):
            candidate, diag = self._evaluate_mutual_auto_pair(
                cluster_a, cluster_b, clusters=clusters, faces_by_cluster=faces_by_cluster
            )
            evaluations.append(diag)
            if candidate is not None:
                candidates.append(candidate)

        # Tier B: oriented smaller -> mature target. Only evaluate plausible
        # relative fragments to keep the expensive member-level audit bounded.
        for source in clusters:
            source_faces = faces_by_cluster[source.cluster_id]
            if len(source_faces) < 2:
                continue
            for target in clusters:
                if target.cluster_id == source.cluster_id:
                    continue
                target_faces = faces_by_cluster[target.cluster_id]
                if len(target_faces) < self.mature_cluster_min_faces:
                    continue
                if len(source_faces) >= len(target_faces):
                    continue
                if len(source_faces) / len(target_faces) > self.fragment_max_target_ratio:
                    continue
                candidate, diag = self._evaluate_fragment_bridge_pair(
                    source, target, clusters=clusters, faces_by_cluster=faces_by_cluster
                )
                evaluations.append(diag)
                if candidate is not None:
                    candidates.append(candidate)

        # Deduplicate the same oriented pair if both tiers accept it. Full
        # mutual evidence is the stronger explanation and wins.
        by_pair: Dict[Tuple[str, str], AutoMergeCandidate] = {}
        priority = {"mutual_full_coverage": 0, "fragment_member_bridge": 1}
        for candidate in sorted(
            candidates,
            key=lambda c: (
                priority.get(c.mode, 9),
                -c.source_coverage,
                -c.strong_anchor_count,
                -c.min_target_score,
                -c.mean_target_score,
                -c.min_competition_margin,
                c.source_cluster_id,
                c.target_cluster_id,
            ),
        ):
            pair = tuple(sorted((candidate.source_cluster_id, candidate.target_cluster_id)))
            by_pair.setdefault(pair, candidate)

        return list(by_pair.values()), evaluations

    def _split_candidate_is_auto_safe(
        self,
        candidate: SplitCandidate,
        clusters: Sequence[Cluster],
        faces_by_cluster: Dict[str, List[Face]],
    ) -> bool:
        """Stricter tier above the already-conservative split suggestion."""
        if not self.auto_correction_enabled or not candidate.face_groups:
            return False
        cluster = next(
            (c for c in clusters if c.cluster_id == candidate.existing_cluster_id), None
        )
        if cluster is None or self._cluster_is_automation_protected(cluster):
            return False

        by_id = {face.face_id: face for face in faces_by_cluster[cluster.cluster_id]}
        groups: List[List[Face]] = []
        for face_ids in candidate.face_groups.values():
            group = [by_id[fid] for fid in face_ids if fid in by_id]
            if len(group) != len(face_ids) or len(group) < self.auto_split_min_group_faces:
                return False
            groups.append(group)
        if len(groups) != 2:
            return False

        high_conf_floor = self.t_match + self.min_cluster_margin
        for group in groups:
            eligible = [f for f in group if f.quality_score >= self.exemplar_quality_threshold]
            if not eligible:
                return False
            medoid = self._choose_medoid(eligible, group)
            if any(
                cosine_similarity(medoid.embedding, face.embedding) < high_conf_floor
                for face in group
            ):
                return False

        # Cross-group separation was already checked by the conservative
        # finder at the lower ambiguous boundary. Recheck defensively here.
        lower_boundary = self.t_match - self.band_width
        cross_max = max(
            cosine_similarity(a.embedding, b.embedding)
            for a in groups[0] for b in groups[1]
        )
        return cross_max < lower_boundary

    def apply_high_confidence_auto_corrections(self) -> Dict[str, object]:
        """Apply at most one correction per audit, then recompute evidence.

        Re-auditing after every mutation prevents stale pairwise evidence from
        cascading through multiple merges/splits in one snapshot.
        """
        events: List[dict] = []
        if not self.auto_correction_enabled or self.auto_correction_max_actions <= 0:
            return {"auto_merges": 0, "auto_splits": 0, "auto_correction_actions": 0, "events": events}

        for _ in range(self.auto_correction_max_actions):
            audit = self.audit_confirmed_clusters()
            if audit.auto_merge_candidates:
                candidate = audit.auto_merge_candidates[0]
                survivor = self.store.execute_merge_atomic(
                    candidate.source_cluster_id,
                    candidate.target_cluster_id,
                    exemplar_quality_threshold=self.exemplar_quality_threshold,
                )
                events.append({
                    "type": f"auto_merge_{candidate.mode}",
                    "mode": candidate.mode,
                    "source_cluster_id": candidate.source_cluster_id,
                    "target_cluster_id": candidate.target_cluster_id,
                    "survivor_cluster_id": survivor,
                    "source_size": candidate.source_size,
                    "target_size": candidate.target_size,
                    "source_coverage": candidate.source_coverage,
                    "reverse_coverage": candidate.reverse_coverage,
                    "strong_anchor_count": candidate.strong_anchor_count,
                    "reverse_strong_anchor_count": candidate.reverse_strong_anchor_count,
                    "min_target_score": candidate.min_target_score,
                    "mean_target_score": candidate.mean_target_score,
                    "min_competition_margin": candidate.min_competition_margin,
                    "member_bridge_source_coverage": candidate.member_bridge_source_coverage,
                    "high_conf_bridge_source_count": candidate.high_conf_bridge_source_count,
                    "strong_bridge_source_count": candidate.strong_bridge_source_count,
                    "min_member_support_count": candidate.min_member_support_count,
                    "distinct_bridge_target_faces": candidate.distinct_bridge_target_faces,
                    "distinct_bridge_target_photos": candidate.distinct_bridge_target_photos,
                })
                continue

            if audit.auto_split_candidates:
                candidate = sorted(
                    audit.auto_split_candidates,
                    key=lambda c: (c.existing_cluster_id, sorted(c.receiving_new_labels)),
                )[0]
                groups = [
                    sorted(face_ids)
                    for _, face_ids in sorted(candidate.face_groups.items())
                ]
                result_ids = self.store.execute_split_atomic(
                    candidate.existing_cluster_id,
                    groups,
                    exemplar_quality_threshold=self.exemplar_quality_threshold,
                    mark_manual_correction=False,
                    cannot_link_reason="auto_split_high_confidence",
                )
                events.append({
                    "type": "auto_split_high_confidence",
                    "source_cluster_id": candidate.existing_cluster_id,
                    "result_cluster_ids": list(result_ids),
                    "group_sizes": [len(group) for group in groups],
                })
                continue

            break

        return {
            "auto_merges": sum(event["type"].startswith("auto_merge") for event in events),
            "auto_splits": sum(event["type"].startswith("auto_split") for event in events),
            "auto_correction_actions": len(events),
            "events": events,
        }

    def _find_conservative_split_candidates(
        self,
        clusters: Sequence[Cluster],
        faces_by_cluster: Dict[str, List[Face]],
        suspicious_face_ids: set,
    ) -> List[SplitCandidate]:
        """Return only exceptionally clear two-way split suggestions.

        A split is intentionally rarer than a merge suggestion. The complete
        cluster must partition into exactly two non-noise groups, each group
        must be internally cohesive and contain an exemplar-eligible face, and
        every cross-group pair must stay below the lower ambiguous boundary.
        Clusters touched by manual user corrections are never auto-suggested.
        """
        lower_boundary = self.t_match - self.band_width
        results: List[SplitCandidate] = []

        for cluster in clusters:
            all_members = [
                face for face in self.store.load_faces_by_cluster(cluster.cluster_id)
                if face.assignment_state in (AssignmentState.CONFIRMED, AssignmentState.MANUAL)
            ]
            if len(all_members) < 4:
                continue
            if cluster.has_manual_correction or any(face.is_manually_corrected for face in all_members):
                continue
            faces = faces_by_cluster[cluster.cluster_id]
            if len(faces) != len(all_members):
                continue

            labels = run_hdbscan(
                [face.embedding for face in faces],
                min_cluster_size=2,
                min_samples=1,
                item_ids=[face.face_id for face in faces],
            )
            # Any noise means the split is not clear enough to bother the user.
            if any(int(value) == -1 for value in labels):
                continue
            label_values = sorted({int(v) for v in labels})
            if len(label_values) != 2:
                continue
            label_groups = {
                label: [faces[i] for i, value in enumerate(labels) if int(value) == label]
                for label in label_values
            }
            if any(len(group) < 2 for group in label_groups.values()):
                continue
            if any(
                not any(face.quality_score >= self.exemplar_quality_threshold for face in group)
                for group in label_groups.values()
            ):
                continue

            # Each group must independently look like one person around an
            # exemplar-eligible medoid, using the existing match threshold.
            internally_cohesive = True
            for group in label_groups.values():
                eligible = [f for f in group if f.quality_score >= self.exemplar_quality_threshold]
                medoid = self._choose_medoid(eligible, group)
                if any(
                    cosine_similarity(medoid.embedding, face.embedding) < self.t_match
                    for face in group
                ):
                    internally_cohesive = False
                    break
            if not internally_cohesive:
                continue

            left, right = [label_groups[label] for label in label_values]
            cross_max = max(
                cosine_similarity(a.embedding, b.embedding)
                for a in left for b in right
            )
            if cross_max >= lower_boundary:
                continue

            total = len(faces)
            results.append(
                SplitCandidate(
                    existing_cluster_id=cluster.cluster_id,
                    receiving_new_labels=label_values,
                    fractions={label: len(label_groups[label]) / total for label in label_values},
                    face_groups={
                        label: [face.face_id for face in label_groups[label]]
                        for label in label_values
                    },
                )
            )

        return results

