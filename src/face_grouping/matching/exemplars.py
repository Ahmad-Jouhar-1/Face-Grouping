"""
Exemplar set management.

Implements Point 6's bucketed exemplar design: a fixed-size set (default
5) split into a "best-quality" bucket (near-frontal, sharp, eyes open --
also used as the UI-facing representative photo) and a "pose-diversity"
bucket (fills angular gaps so angled photos still match well).

The temporal-anchor bucket described in Point 6 is deliberately deferred
to the backend/consolidation stage -- it requires tracking cluster age
over real elapsed time, which doesn't exist yet at this isolated
matching-module level. For now, quality_bucket_size + pose_bucket_size
sum to the full exemplar set (3 + 2 = 5 by default).

Replacement rule (Point 6): a candidate only competes against the
weakest member of its OWN relevant bucket, never the whole set --
quality bucket uses quality_score, pose bucket uses angular distance
(does the candidate fill a pose gap the current set doesn't cover).
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from face_grouping.matching.similarity import cosine_similarity


@dataclass(eq=False)
class Exemplar:
    embedding: np.ndarray
    quality_score: float
    yaw_ratio: float  # pose signal from Point 2 -- used as the angular-distance proxy
    face_id: Optional[str] = None  # optional, for debugging/traceability
    embedding_model_version: str = "legacy_unknown"


class ExemplarSet:
    def __init__(self, quality_bucket_size: int = 3, pose_bucket_size: int = 2):
        self.quality_bucket_size = quality_bucket_size
        self.pose_bucket_size = pose_bucket_size
        self.quality_bucket: List[Exemplar] = []
        self.pose_bucket: List[Exemplar] = []

    def all_exemplars(self) -> List[Exemplar]:
        return self.quality_bucket + self.pose_bucket

    def all_embeddings(self) -> List[np.ndarray]:
        return [e.embedding for e in self.all_exemplars()]

    def try_add(self, candidate: Exemplar) -> bool:
        """
        Attempts to add a candidate exemplar. Caller is responsible for
        having already confirmed the candidate passes the exemplar
        eligibility gate (Point 2: quality_score >= 0.7) -- this class
        doesn't re-check that, since it may be reused in contexts where
        that's already guaranteed.

        Returns True if the candidate was added (either into an empty
        slot or by replacing a weaker existing exemplar), False if it
        didn't beat anything and was discarded.
        """
        if len(self.quality_bucket) < self.quality_bucket_size:
            self.quality_bucket.append(candidate)
            self._sort_quality_bucket()
            return True

        weakest = min(self.quality_bucket, key=lambda e: e.quality_score)
        if candidate.quality_score > weakest.quality_score:
            self.quality_bucket.remove(weakest)
            self.quality_bucket.append(candidate)
            self._sort_quality_bucket()
            return True

        return self._try_add_pose(candidate)

    def _try_add_pose(self, candidate: Exemplar) -> bool:
        if len(self.pose_bucket) < self.pose_bucket_size:
            self.pose_bucket.append(candidate)
            return True

        current_all = self.all_exemplars()

        def min_angular_distance(target: Exemplar, others: List[Exemplar]) -> float:
            others = [e for e in others if e is not target]
            if not others:
                return float("inf")
            return min(abs(target.yaw_ratio - o.yaw_ratio) for o in others)

        candidate_gap = min_angular_distance(candidate, current_all)

        weakest_pose_exemplar = None
        weakest_gap = None
        for ex in self.pose_bucket:
            gap = min_angular_distance(ex, current_all)
            if weakest_gap is None or gap < weakest_gap:
                weakest_gap = gap
                weakest_pose_exemplar = ex

        if candidate_gap > weakest_gap:
            self.pose_bucket.remove(weakest_pose_exemplar)
            self.pose_bucket.append(candidate)
            return True

        return False

    def _sort_quality_bucket(self):
        self.quality_bucket.sort(key=lambda e: e.quality_score, reverse=True)

    def remove_by_face_id(self, face_id: str) -> bool:
        """
        Removes an exemplar by face_id, if present. Needed for Point 17
        corrections: if a face the user has moved away from this
        cluster happens to be one of its exemplars, it must not keep
        representing this (now known-wrong) person in the UI.
        Returns True if something was removed.
        """
        for bucket in (self.quality_bucket, self.pose_bucket):
            for exemplar in bucket:
                if exemplar.face_id == face_id:
                    bucket.remove(exemplar)
                    return True
        return False

    def __len__(self):
        return len(self.quality_bucket) + len(self.pose_bucket)