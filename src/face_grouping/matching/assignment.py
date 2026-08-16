"""Pure incremental-assignment decision types and rules.

This module deliberately contains no detector/model/storage dependencies, so
its safety-critical decision logic can be unit-tested with synthetic scores.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from face_grouping.matching.thresholds import MatchDecision, decide_match


class AssignmentState(Enum):
    """Persistent state of an automated or manual face assignment."""

    CONFIRMED = "confirmed"
    AMBIGUOUS = "ambiguous"
    UNASSIGNED = "unassigned"
    MANUAL = "manual"


@dataclass(frozen=True)
class ClusterCandidate:
    """One cluster's score and the threshold actually used for it."""

    cluster_id: str
    score: float
    exemplar_count: int
    effective_threshold: float
    similarities: List[float] = field(default_factory=list)


@dataclass(frozen=True)
class AssignmentDecision:
    """Complete, auditable result of one incremental assignment decision."""

    state: AssignmentState
    assigned_cluster_id: Optional[str]
    candidate_cluster_id: Optional[str]
    best_score: Optional[float]
    second_best_cluster_id: Optional[str]
    second_best_score: Optional[float]
    score_margin: Optional[float]
    decision_threshold: Optional[float]
    reason: str
    create_new_cluster: bool = False


def decide_assignment(
    candidates: List[ClusterCandidate],
    *,
    exemplar_eligible: bool,
    ambiguous_band_width: float,
    min_cluster_margin: float,
) -> AssignmentDecision:
    """Choose CONFIRMED, AMBIGUOUS, UNASSIGNED, or a new-cluster seed.

    A score above the best cluster's effective threshold is not sufficient on
    its own: when a second cluster exists, the winner must also be separated by
    ``min_cluster_margin``. Ambiguous faces are never assigned to a cluster.
    A new active cluster is created only when the face is good enough to seed
    its first exemplar.
    """
    if min_cluster_margin < 0:
        raise ValueError("min_cluster_margin must be non-negative")

    ranked = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)

    if not ranked:
        if exemplar_eligible:
            return AssignmentDecision(
                state=AssignmentState.CONFIRMED,
                assigned_cluster_id=None,
                candidate_cluster_id=None,
                best_score=None,
                second_best_cluster_id=None,
                second_best_score=None,
                score_margin=None,
                decision_threshold=None,
                reason="new_cluster_seed_no_existing_candidates",
                create_new_cluster=True,
            )
        return AssignmentDecision(
            state=AssignmentState.UNASSIGNED,
            assigned_cluster_id=None,
            candidate_cluster_id=None,
            best_score=None,
            second_best_cluster_id=None,
            second_best_score=None,
            score_margin=None,
            decision_threshold=None,
            reason="unassigned_no_existing_candidates_and_not_exemplar_eligible",
        )

    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    second_score = second.score if second is not None else None
    margin = best.score - second.score if second is not None else None
    threshold_decision = decide_match(
        best.score,
        best.effective_threshold,
        ambiguous_band_width,
    )

    common = dict(
        best_score=best.score,
        second_best_cluster_id=second.cluster_id if second is not None else None,
        second_best_score=second_score,
        score_margin=margin,
        decision_threshold=best.effective_threshold,
    )

    if threshold_decision == MatchDecision.CONFIDENT_MATCH:
        if second is not None and margin < min_cluster_margin:
            return AssignmentDecision(
                state=AssignmentState.AMBIGUOUS,
                assigned_cluster_id=None,
                candidate_cluster_id=best.cluster_id,
                reason="ambiguous_insufficient_margin_over_second_best",
                **common,
            )
        return AssignmentDecision(
            state=AssignmentState.CONFIRMED,
            assigned_cluster_id=best.cluster_id,
            candidate_cluster_id=best.cluster_id,
            reason="confirmed_score_and_margin_passed",
            **common,
        )

    if threshold_decision == MatchDecision.AMBIGUOUS:
        return AssignmentDecision(
            state=AssignmentState.AMBIGUOUS,
            assigned_cluster_id=None,
            candidate_cluster_id=best.cluster_id,
            reason="ambiguous_score_inside_threshold_band",
            **common,
        )

    # The best candidate is below the ambiguous band. Create an active cluster
    # only when the face can immediately seed a trustworthy exemplar.
    if exemplar_eligible:
        return AssignmentDecision(
            state=AssignmentState.CONFIRMED,
            assigned_cluster_id=None,
            candidate_cluster_id=None,
            reason="new_cluster_seed_no_candidate_above_ambiguous_band",
            create_new_cluster=True,
            **common,
        )

    return AssignmentDecision(
        state=AssignmentState.UNASSIGNED,
        assigned_cluster_id=None,
        candidate_cluster_id=None,
        reason="unassigned_no_match_and_not_exemplar_eligible",
        **common,
    )
