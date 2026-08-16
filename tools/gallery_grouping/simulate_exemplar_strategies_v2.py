#!/usr/bin/env python3
"""Read-only safe exemplar replacement simulator (v2).

This simulator starts from the CURRENT production exemplar sets and tests at
most one new diversity exemplar per mature cluster.  It does NOT rebuild all
five exemplars.  Every change is evaluated one-at-a-time against global
identification safety before it is accepted.

Why v2 exists
-------------
The v1 coverage/diversity rebuild improved lower-tail own-cluster scores but
introduced wrong-confirmed trusted members and much higher foreign attraction.
v2 therefore treats the current production exemplar set as a safety baseline
and only permits a surgical substitution when all hard safety invariants hold.

Hard safety invariants for an accepted substitution:
- no increase in wrong-confirmed trusted members,
- no decrease in correct-confirmed trusted members,
- no increase in competitive foreign attraction,
- the modified target cluster itself gains no competitive foreign faces,
- no deferred face changes its best identity because of the substitution.

Benefit requirement (at least one):
- a previously deferred face whose best identity was ALREADY the target becomes
  confirmed without changing identity (stable deferred gain), or
- trusted-member correct-confirmed count improves, or
- the target cluster's leave-one-out own-score p10 improves by a small minimum
  amount (default 0.005).

Candidate constraints reuse existing production values:
- mature cluster only (default >= 8 trusted faces),
- normal CONFIRMED/MANUAL face,
- not recognition-restricted,
- not final-suspicious,
- existing exemplar quality floor,
- at least N OTHER same-cluster supporters at the existing T_match.

The maximum exemplar count remains unchanged.  A mature cluster gets at most
one accepted substitution by default, leaving four or more of its current
production exemplars untouched.

No production database or threshold is modified.

Typical usage::

    python tools/gallery_grouping/simulate_exemplar_strategies_v2.py \
      --run-output data/gallery_grouping_output_run002 \
      --target-photo IMG-20250423-WA0000.jpg

Outputs are written to::

    <run-output>/exemplar_strategy_simulation_v2_safe/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Reuse the v1 simulator's read-only DB loading and exact production decision
# helpers.  Keeping these shared prevents the two simulators from drifting on
# basic score/threshold semantics.
import simulate_exemplar_strategies as base  # noqa: E402


OUTPUT_DIRNAME = "exemplar_strategy_simulation_v2_safe"
MEMBER_STATES = {"confirmed", "manual"}


@dataclass
class EvalFace:
    face: base.FaceRec
    true_cluster_id: str
    baseline_state: str
    baseline_predicted_cluster_id: Optional[str]
    baseline_best_cluster_id: Optional[str]
    baseline_best_score: Optional[float]
    baseline_second_cluster_id: Optional[str]
    baseline_second_score: Optional[float]
    baseline_margin: Optional[float]
    baseline_outcome: str
    own_score: Optional[float]
    # Sorted candidates under the CURRENT strategy snapshot.  For the true
    # cluster, leave-one-out scoring is already applied.
    ranked_candidates: List[Tuple[str, float, int, float]]


@dataclass
class DeferredEval:
    face: base.FaceRec
    baseline_state: str
    baseline_best_cluster_id: Optional[str]
    baseline_best_score: Optional[float]
    baseline_second_cluster_id: Optional[str]
    baseline_second_score: Optional[float]
    baseline_margin: Optional[float]
    ranked_candidates: List[Tuple[str, float, int, float]]


@dataclass
class Snapshot:
    trusted: List[EvalFace]
    deferred: List[DeferredEval]
    correct_confirmed: int
    wrong_confirmed: int
    ambiguous: int
    unassigned: int
    own_score_p10: Optional[float]
    own_score_mean: Optional[float]
    cluster_own_p10: Dict[str, Optional[float]]
    cluster_own_mean: Dict[str, Optional[float]]
    cluster_correct: Dict[str, int]
    cluster_wrong: Dict[str, int]
    risk_by_target: Dict[str, dict]
    foreign_above_total: int
    competitive_foreign_total: int


@dataclass
class ReplacementEval:
    cluster_id: str
    removed_face_id: str
    added_face_id: str
    strategy: base.StrategySet
    safe: bool
    accepted_reasons: List[str]
    rejected_reasons: List[str]
    correct_confirmed: int
    wrong_confirmed: int
    correct_delta: int
    wrong_delta: int
    target_p10: Optional[float]
    target_p10_delta: Optional[float]
    target_mean: Optional[float]
    target_mean_delta: Optional[float]
    competitive_foreign_total: int
    competitive_foreign_delta: int
    target_competitive_foreign: int
    target_competitive_foreign_delta: int
    foreign_above_total: int
    foreign_above_delta: int
    target_foreign_above: int
    target_foreign_above_delta: int
    deferred_best_cluster_changes: int
    stable_deferred_gains: int
    newly_confirmed_deferred_total: int
    candidate_support_count: int
    candidate_diversity: float
    candidate_quality: float

    def benefit_key(self) -> Tuple[float, ...]:
        # Lexicographic: identity-stable recovery is the strongest benefit;
        # then trusted correct-confirmed; then lower-tail representation.
        p10 = self.target_p10_delta if self.target_p10_delta is not None else -999.0
        mean = self.target_mean_delta if self.target_mean_delta is not None else -999.0
        return (
            float(self.stable_deferred_gains),
            float(self.correct_delta),
            p10,
            mean,
            float(-self.competitive_foreign_delta),
            float(-self.foreign_above_delta),
            float(self.candidate_support_count),
            self.candidate_diversity,
            self.candidate_quality,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Safe one-at-a-time exemplar diversity replacement simulator (v2).")
    p.add_argument("--run-output", default="data/gallery_grouping_output", help="Completed Gallery run directory.")
    p.add_argument("--output", default="", help="Optional explicit output directory.")
    p.add_argument("--target-photo", action="append", default=[], help="Photo basename to highlight; repeatable.")
    p.add_argument("--mature-min-faces", type=int, default=8, help="Minimum trusted members for diversity replacement.")
    p.add_argument("--member-support", type=int, default=2, help="Required OTHER same-cluster supporters at existing T_match.")
    p.add_argument("--max-replacements-per-cluster", type=int, default=1, help="Maximum accepted substitutions per cluster (default 1).")
    p.add_argument("--max-total-actions", type=int, default=13, help="Global cap on accepted substitutions (default 13).")
    p.add_argument("--min-p10-gain", type=float, default=0.005, help="Minimum target-cluster p10 gain if there is no direct recovery gain.")
    p.add_argument(
        "--max-candidates-per-cluster",
        type=int,
        default=0,
        help="Optional diagnostic speed cap after candidate ranking; 0 evaluates all safe candidates.",
    )
    return p.parse_args()


def _safe(v: Optional[float]) -> str | float:
    return "" if v is None else float(v)


def _person(cluster_id: Optional[str], mapping: Dict[str, str]) -> str:
    if not cluster_id:
        return ""
    return mapping.get(cluster_id, cluster_id)


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _strategy_from_faces(cluster_id: str, faces: Sequence[base.FaceRec], labels: Sequence[str]) -> base.StrategySet:
    return base.StrategySet(
        cluster_id=cluster_id,
        exemplar_ids=[f.face_id for f in faces],
        embeddings=[f.embedding for f in faces],
        qualities=[f.quality_score for f in faces],
        yaws=[f.yaw_ratio for f in faces],
        labels=list(labels),
    )


def _effective_threshold(count: int, *, top_k: int, t_match: float, sparse_margin: float) -> float:
    return t_match + sparse_margin if count < top_k else t_match


def _decision_from_ranked(
    ranked: Sequence[Tuple[str, float, int, float]],
    *,
    ambiguous_band_width: float,
    min_cluster_margin: float,
) -> Tuple[str, Optional[str], Optional[str], Optional[float], Optional[str], Optional[float], Optional[float]]:
    """Exact exemplar_eligible=False equivalent of production decide_assignment."""
    if not ranked:
        return ("unassigned", None, None, None, None, None, None)
    ordered = sorted(ranked, key=lambda x: x[1], reverse=True)
    best = ordered[0]
    second = ordered[1] if len(ordered) > 1 else None
    best_cluster, best_score, _, best_threshold = best
    second_cluster = second[0] if second else None
    second_score = second[1] if second else None
    margin = best_score - second_score if second is not None else None

    if best_score >= best_threshold:
        if second is not None and margin is not None and margin < min_cluster_margin:
            return ("ambiguous", None, best_cluster, best_score, second_cluster, second_score, margin)
        return ("confirmed", best_cluster, best_cluster, best_score, second_cluster, second_score, margin)
    if best_score >= best_threshold - ambiguous_band_width:
        return ("ambiguous", None, best_cluster, best_score, second_cluster, second_score, margin)
    return ("unassigned", None, None, best_score, second_cluster, second_score, margin)


def _all_cluster_candidates(
    face: base.FaceRec,
    strategies: Dict[str, base.StrategySet],
    *,
    top_k: int,
    t_match: float,
    sparse_margin: float,
    leave_one_out_cluster_id: Optional[str],
) -> List[Tuple[str, float, int, float]]:
    out: List[Tuple[str, float, int, float]] = []
    for cluster_id, strategy in strategies.items():
        score, count, _ = base._score_with_ids(
            face,
            strategy,
            top_k=top_k,
            leave_one_out=(leave_one_out_cluster_id == cluster_id),
        )
        if score is None or count <= 0:
            continue
        out.append((cluster_id, score, count, _effective_threshold(count, top_k=top_k, t_match=t_match, sparse_margin=sparse_margin)))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def _build_snapshot(
    *,
    strategies: Dict[str, base.StrategySet],
    faces_by_cluster: Dict[str, List[base.FaceRec]],
    faces_by_id: Dict[str, base.FaceRec],
    suspicious_ids: set[str],
    top_k: int,
    t_match: float,
    sparse_margin: float,
    ambiguous_band_width: float,
    min_cluster_margin: float,
) -> Snapshot:
    trusted_rows: List[EvalFace] = []
    deferred_rows: List[DeferredEval] = []
    totals = Counter()
    own_scores_all: List[float] = []
    own_scores_by_cluster: Dict[str, List[float]] = defaultdict(list)
    cluster_correct = Counter()
    cluster_wrong = Counter()

    for cluster_id, members in faces_by_cluster.items():
        for face in members:
            if (
                face.assignment_state not in MEMBER_STATES
                or face.recognition_restricted
                or face.face_id in suspicious_ids
            ):
                continue
            ranked = _all_cluster_candidates(
                face,
                strategies,
                top_k=top_k,
                t_match=t_match,
                sparse_margin=sparse_margin,
                leave_one_out_cluster_id=cluster_id,
            )
            state, predicted, best_id, best_score, second_id, second_score, margin = _decision_from_ranked(
                ranked,
                ambiguous_band_width=ambiguous_band_width,
                min_cluster_margin=min_cluster_margin,
            )
            own_entry = next((x for x in ranked if x[0] == cluster_id), None)
            own_score = own_entry[1] if own_entry else None
            if own_score is not None:
                own_scores_all.append(own_score)
                own_scores_by_cluster[cluster_id].append(own_score)
            if state == "confirmed" and predicted == cluster_id:
                outcome = "correct_confirmed"
                cluster_correct[cluster_id] += 1
            elif state == "confirmed" and predicted != cluster_id:
                outcome = "wrong_confirmed"
                cluster_wrong[cluster_id] += 1
            else:
                outcome = state
            totals[outcome] += 1
            trusted_rows.append(
                EvalFace(
                    face=face,
                    true_cluster_id=cluster_id,
                    baseline_state=state,
                    baseline_predicted_cluster_id=predicted,
                    baseline_best_cluster_id=best_id,
                    baseline_best_score=best_score,
                    baseline_second_cluster_id=second_id,
                    baseline_second_score=second_score,
                    baseline_margin=margin,
                    baseline_outcome=outcome,
                    own_score=own_score,
                    ranked_candidates=ranked,
                )
            )

    for face in faces_by_id.values():
        if face.assignment_state not in {"ambiguous", "unassigned"} or face.recognition_restricted:
            continue
        ranked = _all_cluster_candidates(
            face,
            strategies,
            top_k=top_k,
            t_match=t_match,
            sparse_margin=sparse_margin,
            leave_one_out_cluster_id=None,
        )
        state, _, best_id, best_score, second_id, second_score, margin = _decision_from_ranked(
            ranked,
            ambiguous_band_width=ambiguous_band_width,
            min_cluster_margin=min_cluster_margin,
        )
        deferred_rows.append(
            DeferredEval(
                face=face,
                baseline_state=state,
                baseline_best_cluster_id=best_id,
                baseline_best_score=best_score,
                baseline_second_cluster_id=second_id,
                baseline_second_score=second_score,
                baseline_margin=margin,
                ranked_candidates=ranked,
            )
        )

    # Risk snapshot from exact shared implementation.  This is only computed
    # once per accepted action, not for every candidate trial.
    risk = base._evaluate_cross_cluster_risk(
        strategies=strategies,
        faces_by_cluster=faces_by_cluster,
        suspicious_ids=suspicious_ids,
        top_k=top_k,
        t_match=t_match,
        sparse_margin=sparse_margin,
        min_cluster_margin=min_cluster_margin,
    )
    arr = np.asarray(own_scores_all, dtype=np.float64) if own_scores_all else np.asarray([], dtype=np.float64)
    p10_by_cluster: Dict[str, Optional[float]] = {}
    mean_by_cluster: Dict[str, Optional[float]] = {}
    for cluster_id in strategies:
        vals = np.asarray(own_scores_by_cluster.get(cluster_id, []), dtype=np.float64)
        p10_by_cluster[cluster_id] = float(np.quantile(vals, 0.10)) if vals.size else None
        mean_by_cluster[cluster_id] = float(np.mean(vals)) if vals.size else None

    return Snapshot(
        trusted=trusted_rows,
        deferred=deferred_rows,
        correct_confirmed=totals["correct_confirmed"],
        wrong_confirmed=totals["wrong_confirmed"],
        ambiguous=totals["ambiguous"],
        unassigned=totals["unassigned"],
        own_score_p10=float(np.quantile(arr, 0.10)) if arr.size else None,
        own_score_mean=float(np.mean(arr)) if arr.size else None,
        cluster_own_p10=p10_by_cluster,
        cluster_own_mean=mean_by_cluster,
        cluster_correct=dict(cluster_correct),
        cluster_wrong=dict(cluster_wrong),
        risk_by_target=risk,
        foreign_above_total=sum(v["foreign_faces_above_threshold"] for v in risk.values()),
        competitive_foreign_total=sum(v["competitive_foreign_faces"] for v in risk.values()),
    )


def _vector_scores(
    faces: Sequence[base.FaceRec],
    true_cluster_ids: Sequence[Optional[str]],
    strategy: base.StrategySet,
    *,
    target_cluster_id: str,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Top-K scores against one candidate exemplar set.

    Leave-one-out is applied only when a face belongs to the target cluster and
    that exact face is one of the candidate exemplars.
    """
    n = len(faces)
    m = len(strategy.embeddings)
    if n == 0 or m == 0:
        return np.full(n, np.nan), np.zeros(n, dtype=np.int32)

    q = np.stack([f.embedding for f in faces]).astype(np.float64, copy=False)
    x = np.stack(strategy.embeddings).astype(np.float64, copy=False)
    qn = np.linalg.norm(q, axis=1, keepdims=True)
    xn = np.linalg.norm(x, axis=1, keepdims=True).T
    sims = (q @ x.T) / (qn * xn + 1e-9)

    id_to_col = {fid: j for j, fid in enumerate(strategy.exemplar_ids) if fid is not None}
    counts = np.full(n, m, dtype=np.int32)
    for i, (face, true_id) in enumerate(zip(faces, true_cluster_ids)):
        if true_id == target_cluster_id and face.face_id in id_to_col:
            sims[i, id_to_col[face.face_id]] = -np.inf
            counts[i] -= 1

    sorted_sims = np.sort(sims, axis=1)[:, ::-1]
    scores = np.full(n, np.nan, dtype=np.float64)
    for i, count in enumerate(counts):
        if count <= 0:
            continue
        k = min(top_k, int(count))
        scores[i] = float(np.mean(sorted_sims[i, :k]))
    return scores, counts


def _other_top2(ranked: Sequence[Tuple[str, float, int, float]], target: str) -> List[Tuple[str, float, int, float]]:
    out: List[Tuple[str, float, int, float]] = []
    for item in ranked:
        if item[0] == target:
            continue
        out.append(item)
        if len(out) >= 2:
            break
    return out


def _candidate_diversity(candidate: base.FaceRec, current_set: base.StrategySet) -> float:
    if not current_set.embeddings:
        return 1.0
    return float(min(1.0 - base.cosine_similarity(candidate.embedding, e) for e in current_set.embeddings))


def _candidate_pool(
    *,
    cluster_id: str,
    members: Sequence[base.FaceRec],
    current_set: base.StrategySet,
    suspicious_ids: set[str],
    exemplar_quality_threshold: float,
    t_match: float,
    mature_min_faces: int,
    member_support: int,
    max_candidates: int,
) -> Tuple[List[base.FaceRec], Dict[str, dict], dict]:
    trusted = [
        f for f in members
        if f.assignment_state in MEMBER_STATES
        and not f.recognition_restricted
        and f.face_id not in suspicious_ids
    ]
    current_ids = {fid for fid in current_set.exemplar_ids if fid}
    diag = {
        "cluster_id": cluster_id,
        "trusted_members": len(trusted),
        "current_exemplars": len(current_ids),
        "eligible_noncurrent_candidates": 0,
        "supported_candidates": 0,
        "candidate_cap_applied": False,
        "candidate_face_ids": [],
    }
    if len(trusted) < mature_min_faces:
        diag["reason"] = "cluster_below_mature_min_faces"
        return [], {}, diag

    support_meta: Dict[str, dict] = {}
    pool: List[base.FaceRec] = []
    for candidate in trusted:
        if candidate.face_id in current_ids:
            continue
        if candidate.quality_score < exemplar_quality_threshold:
            continue
        diag["eligible_noncurrent_candidates"] += 1
        support = base._support_count(candidate, trusted, t_match)
        if support < member_support:
            continue
        support_meta[candidate.face_id] = {
            "support_count": support,
            "mean_member_similarity": base._mean_similarity_to_members(candidate, trusted),
            "diversity": _candidate_diversity(candidate, current_set),
        }
        pool.append(candidate)
    diag["supported_candidates"] = len(pool)

    # Optional speed cap.  Rank candidates by repeated internal support first,
    # then actual diversity and quality.  Default 0 evaluates all candidates.
    pool.sort(
        key=lambda f: (
            support_meta[f.face_id]["support_count"],
            support_meta[f.face_id]["diversity"],
            f.quality_score,
            support_meta[f.face_id]["mean_member_similarity"],
        ),
        reverse=True,
    )
    if max_candidates > 0 and len(pool) > max_candidates:
        pool = pool[:max_candidates]
        diag["candidate_cap_applied"] = True
    diag["candidate_face_ids"] = [f.face_id for f in pool]
    diag["reason"] = "candidate_pool_ready" if pool else "no_supported_noncurrent_candidates"
    return pool, support_meta, diag


def _replacement_strategy(
    *,
    target_id: str,
    current_set: base.StrategySet,
    removed_index: int,
    candidate: base.FaceRec,
) -> base.StrategySet:
    ids = list(current_set.exemplar_ids)
    embs = list(current_set.embeddings)
    quals = list(current_set.qualities)
    yaws = list(current_set.yaws)
    labels = list(current_set.labels)
    ids[removed_index] = candidate.face_id
    embs[removed_index] = candidate.embedding
    quals[removed_index] = candidate.quality_score
    yaws[removed_index] = candidate.yaw_ratio
    labels[removed_index] = "safe_diversity_replacement"
    return base.StrategySet(
        cluster_id=target_id,
        exemplar_ids=ids,
        embeddings=embs,
        qualities=quals,
        yaws=yaws,
        labels=labels,
    )


def _replacement_is_safe(
    *,
    wrong_delta: int,
    correct_delta: int,
    competitive_foreign_delta: int,
    target_competitive_foreign_delta: int,
    deferred_best_cluster_changes: int,
    stable_deferred_gains: int,
    target_p10_delta: Optional[float],
    min_p10_gain: float,
) -> Tuple[bool, List[str], List[str]]:
    rejected: List[str] = []
    accepted: List[str] = []
    if wrong_delta > 0:
        rejected.append("wrong_confirmed_increased")
    if correct_delta < 0:
        rejected.append("correct_confirmed_decreased")
    if competitive_foreign_delta > 0:
        rejected.append("competitive_foreign_attraction_increased")
    if target_competitive_foreign_delta > 0:
        rejected.append("target_cluster_competitive_foreign_increased")
    if deferred_best_cluster_changes > 0:
        rejected.append("deferred_best_identity_changed")

    benefit = False
    if stable_deferred_gains > 0:
        benefit = True
        accepted.append("stable_deferred_recovery_gain")
    if correct_delta > 0:
        benefit = True
        accepted.append("trusted_correct_confirmed_gain")
    if target_p10_delta is not None and target_p10_delta >= min_p10_gain:
        benefit = True
        accepted.append("target_lower_tail_gain")
    if not benefit:
        rejected.append("no_material_representation_or_recovery_gain")

    return (not rejected), accepted, rejected


def _evaluate_replacement(
    *,
    target_id: str,
    candidate_strategy: base.StrategySet,
    candidate: base.FaceRec,
    removed_face_id: str,
    snapshot: Snapshot,
    top_k: int,
    t_match: float,
    sparse_margin: float,
    ambiguous_band_width: float,
    min_cluster_margin: float,
    candidate_support_count: int,
    candidate_diversity: float,
    min_p10_gain: float,
) -> ReplacementEval:
    trusted_faces = [r.face for r in snapshot.trusted]
    trusted_true = [r.true_cluster_id for r in snapshot.trusted]
    target_scores, target_counts = _vector_scores(
        trusted_faces,
        trusted_true,
        candidate_strategy,
        target_cluster_id=target_id,
        top_k=top_k,
    )

    correct = 0
    wrong = 0
    target_own_scores: List[float] = []
    target_foreign_above = 0
    target_competitive = 0

    for i, row in enumerate(snapshot.trusted):
        score = target_scores[i]
        count = int(target_counts[i])
        ranked = _other_top2(row.ranked_candidates, target_id)
        if not math.isnan(float(score)) and count > 0:
            ranked.append((
                target_id,
                float(score),
                count,
                _effective_threshold(count, top_k=top_k, t_match=t_match, sparse_margin=sparse_margin),
            ))
        state, predicted, _, _, _, _, _ = _decision_from_ranked(
            ranked,
            ambiguous_band_width=ambiguous_band_width,
            min_cluster_margin=min_cluster_margin,
        )
        if state == "confirmed" and predicted == row.true_cluster_id:
            correct += 1
        elif state == "confirmed" and predicted != row.true_cluster_id:
            wrong += 1

        if row.true_cluster_id == target_id:
            if not math.isnan(float(score)):
                target_own_scores.append(float(score))
        else:
            if not math.isnan(float(score)) and count > 0:
                threshold = _effective_threshold(count, top_k=top_k, t_match=t_match, sparse_margin=sparse_margin)
                if score >= threshold:
                    target_foreign_above += 1
                    # Own score is unchanged because this face belongs to a
                    # different cluster; compare against current own LOO score.
                    if row.own_score is None or score >= row.own_score - min_cluster_margin:
                        target_competitive += 1

    target_arr = np.asarray(target_own_scores, dtype=np.float64) if target_own_scores else np.asarray([], dtype=np.float64)
    target_p10 = float(np.quantile(target_arr, 0.10)) if target_arr.size else None
    target_mean = float(np.mean(target_arr)) if target_arr.size else None
    baseline_p10 = snapshot.cluster_own_p10.get(target_id)
    baseline_mean = snapshot.cluster_own_mean.get(target_id)
    p10_delta = (target_p10 - baseline_p10) if target_p10 is not None and baseline_p10 is not None else None
    mean_delta = (target_mean - baseline_mean) if target_mean is not None and baseline_mean is not None else None

    baseline_target_risk = snapshot.risk_by_target.get(target_id, {})
    baseline_target_foreign = int(baseline_target_risk.get("foreign_faces_above_threshold", 0))
    baseline_target_comp = int(baseline_target_risk.get("competitive_foreign_faces", 0))
    foreign_delta_target = target_foreign_above - baseline_target_foreign
    comp_delta_target = target_competitive - baseline_target_comp
    foreign_total = snapshot.foreign_above_total + foreign_delta_target
    comp_total = snapshot.competitive_foreign_total + comp_delta_target

    # Deferred evaluation: only target-cluster score changes.  A safe gain must
    # preserve the already-leading identity; if the best identity changes at
    # all, the substitution is rejected because deferred faces lack GT here.
    deferred_faces = [d.face for d in snapshot.deferred]
    deferred_true_none: List[Optional[str]] = [None] * len(deferred_faces)
    d_scores, d_counts = _vector_scores(
        deferred_faces,
        deferred_true_none,
        candidate_strategy,
        target_cluster_id=target_id,
        top_k=top_k,
    )
    best_changes = 0
    stable_gains = 0
    newly_confirmed = 0
    for i, drow in enumerate(snapshot.deferred):
        ranked = _other_top2(drow.ranked_candidates, target_id)
        score = d_scores[i]
        count = int(d_counts[i])
        if not math.isnan(float(score)) and count > 0:
            ranked.append((
                target_id,
                float(score),
                count,
                _effective_threshold(count, top_k=top_k, t_match=t_match, sparse_margin=sparse_margin),
            ))
        state, predicted, best_id, _, _, _, _ = _decision_from_ranked(
            ranked,
            ambiguous_band_width=ambiguous_band_width,
            min_cluster_margin=min_cluster_margin,
        )
        if best_id != drow.baseline_best_cluster_id:
            best_changes += 1
        if drow.baseline_state != "confirmed" and state == "confirmed":
            newly_confirmed += 1
            if (
                predicted == target_id
                and drow.baseline_best_cluster_id == target_id
                and best_id == target_id
                and (drow.baseline_margin is None or drow.baseline_margin >= min_cluster_margin)
            ):
                stable_gains += 1

    correct_delta = correct - snapshot.correct_confirmed
    wrong_delta = wrong - snapshot.wrong_confirmed
    comp_delta = comp_total - snapshot.competitive_foreign_total
    foreign_delta = foreign_total - snapshot.foreign_above_total

    safe, accepted_reasons, rejected_reasons = _replacement_is_safe(
        wrong_delta=wrong_delta,
        correct_delta=correct_delta,
        competitive_foreign_delta=comp_delta,
        target_competitive_foreign_delta=comp_delta_target,
        deferred_best_cluster_changes=best_changes,
        stable_deferred_gains=stable_gains,
        target_p10_delta=p10_delta,
        min_p10_gain=min_p10_gain,
    )

    return ReplacementEval(
        cluster_id=target_id,
        removed_face_id=removed_face_id,
        added_face_id=candidate.face_id,
        strategy=candidate_strategy,
        safe=safe,
        accepted_reasons=accepted_reasons,
        rejected_reasons=rejected_reasons,
        correct_confirmed=correct,
        wrong_confirmed=wrong,
        correct_delta=correct_delta,
        wrong_delta=wrong_delta,
        target_p10=target_p10,
        target_p10_delta=p10_delta,
        target_mean=target_mean,
        target_mean_delta=mean_delta,
        competitive_foreign_total=comp_total,
        competitive_foreign_delta=comp_delta,
        target_competitive_foreign=target_competitive,
        target_competitive_foreign_delta=comp_delta_target,
        foreign_above_total=foreign_total,
        foreign_above_delta=foreign_delta,
        target_foreign_above=target_foreign_above,
        target_foreign_above_delta=foreign_delta_target,
        deferred_best_cluster_changes=best_changes,
        stable_deferred_gains=stable_gains,
        newly_confirmed_deferred_total=newly_confirmed,
        candidate_support_count=candidate_support_count,
        candidate_diversity=candidate_diversity,
        candidate_quality=candidate.quality_score,
    )


def _eval_to_row(
    ev: ReplacementEval,
    *,
    faces_by_id: Dict[str, base.FaceRec],
    cluster_to_person: Dict[str, str],
    iteration: int,
) -> dict:
    old = faces_by_id.get(ev.removed_face_id)
    new = faces_by_id.get(ev.added_face_id)
    return {
        "iteration": iteration,
        "person_folder": _person(ev.cluster_id, cluster_to_person),
        "cluster_id": ev.cluster_id,
        "safe": ev.safe,
        "accepted_reasons": ";".join(ev.accepted_reasons),
        "rejected_reasons": ";".join(ev.rejected_reasons),
        "removed_face_id": ev.removed_face_id,
        "removed_photo": old.photo_name if old else "",
        "added_face_id": ev.added_face_id,
        "added_photo": new.photo_name if new else "",
        "candidate_quality": ev.candidate_quality,
        "candidate_support_count": ev.candidate_support_count,
        "candidate_diversity": ev.candidate_diversity,
        "correct_confirmed": ev.correct_confirmed,
        "correct_delta": ev.correct_delta,
        "wrong_confirmed": ev.wrong_confirmed,
        "wrong_delta": ev.wrong_delta,
        "target_p10": _safe(ev.target_p10),
        "target_p10_delta": _safe(ev.target_p10_delta),
        "target_mean": _safe(ev.target_mean),
        "target_mean_delta": _safe(ev.target_mean_delta),
        "competitive_foreign_total": ev.competitive_foreign_total,
        "competitive_foreign_delta": ev.competitive_foreign_delta,
        "target_competitive_foreign": ev.target_competitive_foreign,
        "target_competitive_foreign_delta": ev.target_competitive_foreign_delta,
        "foreign_above_total": ev.foreign_above_total,
        "foreign_above_delta": ev.foreign_above_delta,
        "target_foreign_above": ev.target_foreign_above,
        "target_foreign_above_delta": ev.target_foreign_above_delta,
        "deferred_best_cluster_changes": ev.deferred_best_cluster_changes,
        "stable_deferred_gains": ev.stable_deferred_gains,
        "newly_confirmed_deferred_total": ev.newly_confirmed_deferred_total,
    }


def main() -> int:
    args = parse_args()
    run_output = Path(args.run_output).expanduser().resolve()
    db_path = run_output / base.DB_FILENAME
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")
    output_dir = Path(args.output).expanduser().resolve() if args.output else run_output / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = base._read_thresholds()
    matching = cfg["matching"]
    quality = cfg["quality"]
    t_match = float(matching["t_match"])
    top_k = int(matching["top_k"])
    sparse_margin = float(matching["sparse_cluster_margin"])
    min_cluster_margin = float(matching["min_cluster_margin"])
    ambiguous_band_width = float(matching["ambiguous_band_width"])
    exemplar_quality_threshold = float(quality["exemplar_eligibility_threshold"])

    conn = base._open_readonly(db_path)
    try:
        faces_by_id, faces_by_cluster, exemplars_by_cluster, active_clusters = base._load_data(conn)
    finally:
        conn.close()

    cluster_to_person, _ = base._load_person_map(run_output)
    suspicious_ids = base._load_suspicious_face_ids(run_output)
    current = base._current_strategy(exemplars_by_cluster, active_clusters)
    strategies: Dict[str, base.StrategySet] = dict(current)

    # Candidate pools are rebuilt after every accepted action because a cluster
    # that was modified is no longer eligible for another change by default.
    modified_count = Counter()
    actions: List[dict] = []
    all_evaluation_rows: List[dict] = []
    cluster_diags: Dict[str, dict] = {}

    initial_snapshot = _build_snapshot(
        strategies=strategies,
        faces_by_cluster=faces_by_cluster,
        faces_by_id=faces_by_id,
        suspicious_ids=suspicious_ids,
        top_k=top_k,
        t_match=t_match,
        sparse_margin=sparse_margin,
        ambiguous_band_width=ambiguous_band_width,
        min_cluster_margin=min_cluster_margin,
    )

    iteration = 0
    while len(actions) < args.max_total_actions:
        iteration += 1
        snapshot = _build_snapshot(
            strategies=strategies,
            faces_by_cluster=faces_by_cluster,
            faces_by_id=faces_by_id,
            suspicious_ids=suspicious_ids,
            top_k=top_k,
            t_match=t_match,
            sparse_margin=sparse_margin,
            ambiguous_band_width=ambiguous_band_width,
            min_cluster_margin=min_cluster_margin,
        )

        best: Optional[ReplacementEval] = None
        best_key: Optional[Tuple[float, ...]] = None
        best_meta: Optional[dict] = None
        any_trial = False

        for cluster_id in active_clusters:
            if modified_count[cluster_id] >= args.max_replacements_per_cluster:
                continue
            current_set = strategies[cluster_id]
            if current_set.count() == 0:
                continue
            pool, support_meta, diag = _candidate_pool(
                cluster_id=cluster_id,
                members=faces_by_cluster.get(cluster_id, []),
                current_set=current_set,
                suspicious_ids=suspicious_ids,
                exemplar_quality_threshold=exemplar_quality_threshold,
                t_match=t_match,
                mature_min_faces=args.mature_min_faces,
                member_support=args.member_support,
                max_candidates=args.max_candidates_per_cluster,
            )
            cluster_diags[cluster_id] = diag
            if not pool:
                continue

            for candidate in pool:
                meta = support_meta[candidate.face_id]
                for idx, removed_face_id in enumerate(current_set.exemplar_ids):
                    if removed_face_id is None:
                        continue
                    any_trial = True
                    trial_strategy = _replacement_strategy(
                        target_id=cluster_id,
                        current_set=current_set,
                        removed_index=idx,
                        candidate=candidate,
                    )
                    ev = _evaluate_replacement(
                        target_id=cluster_id,
                        candidate_strategy=trial_strategy,
                        candidate=candidate,
                        removed_face_id=removed_face_id,
                        snapshot=snapshot,
                        top_k=top_k,
                        t_match=t_match,
                        sparse_margin=sparse_margin,
                        ambiguous_band_width=ambiguous_band_width,
                        min_cluster_margin=min_cluster_margin,
                        candidate_support_count=int(meta["support_count"]),
                        candidate_diversity=float(meta["diversity"]),
                        min_p10_gain=args.min_p10_gain,
                    )
                    all_evaluation_rows.append(_eval_to_row(
                        ev,
                        faces_by_id=faces_by_id,
                        cluster_to_person=cluster_to_person,
                        iteration=iteration,
                    ))
                    if not ev.safe:
                        continue
                    key = ev.benefit_key()
                    if best_key is None or key > best_key:
                        best = ev
                        best_key = key
                        best_meta = meta

        if best is None:
            break

        strategies[best.cluster_id] = best.strategy
        modified_count[best.cluster_id] += 1
        old = faces_by_id.get(best.removed_face_id)
        new = faces_by_id.get(best.added_face_id)
        actions.append({
            "action_index": len(actions) + 1,
            "iteration": iteration,
            "person_folder": _person(best.cluster_id, cluster_to_person),
            "cluster_id": best.cluster_id,
            "removed_face_id": best.removed_face_id,
            "removed_photo": old.photo_name if old else "",
            "added_face_id": best.added_face_id,
            "added_photo": new.photo_name if new else "",
            "accepted_reasons": ";".join(best.accepted_reasons),
            "candidate_quality": best.candidate_quality,
            "candidate_support_count": best.candidate_support_count,
            "candidate_diversity": best.candidate_diversity,
            "correct_delta_at_accept": best.correct_delta,
            "wrong_delta_at_accept": best.wrong_delta,
            "target_p10_delta_at_accept": _safe(best.target_p10_delta),
            "competitive_foreign_delta_at_accept": best.competitive_foreign_delta,
            "foreign_above_delta_at_accept": best.foreign_above_delta,
            "deferred_best_cluster_changes_at_accept": best.deferred_best_cluster_changes,
            "stable_deferred_gains_at_accept": best.stable_deferred_gains,
        })

    final_snapshot = _build_snapshot(
        strategies=strategies,
        faces_by_cluster=faces_by_cluster,
        faces_by_id=faces_by_id,
        suspicious_ids=suspicious_ids,
        top_k=top_k,
        t_match=t_match,
        sparse_margin=sparse_margin,
        ambiguous_band_width=ambiguous_band_width,
        min_cluster_margin=min_cluster_margin,
    )

    # Full shared evaluations once at the end for detailed output parity with
    # v1 and easy manual inspection.
    current_member_rows, current_cluster_eval, current_summary = base._evaluate_member_strategy(
        name="current",
        strategies=current,
        faces_by_cluster=faces_by_cluster,
        suspicious_ids=suspicious_ids,
        cluster_to_person=cluster_to_person,
        top_k=top_k,
        t_match=t_match,
        sparse_margin=sparse_margin,
        ambiguous_band_width=ambiguous_band_width,
        min_cluster_margin=min_cluster_margin,
    )
    final_member_rows, final_cluster_eval, final_summary = base._evaluate_member_strategy(
        name="safe_v2",
        strategies=strategies,
        faces_by_cluster=faces_by_cluster,
        suspicious_ids=suspicious_ids,
        cluster_to_person=cluster_to_person,
        top_k=top_k,
        t_match=t_match,
        sparse_margin=sparse_margin,
        ambiguous_band_width=ambiguous_band_width,
        min_cluster_margin=min_cluster_margin,
    )
    current_risk = base._evaluate_cross_cluster_risk(
        strategies=current,
        faces_by_cluster=faces_by_cluster,
        suspicious_ids=suspicious_ids,
        top_k=top_k,
        t_match=t_match,
        sparse_margin=sparse_margin,
        min_cluster_margin=min_cluster_margin,
    )
    final_risk = base._evaluate_cross_cluster_risk(
        strategies=strategies,
        faces_by_cluster=faces_by_cluster,
        suspicious_ids=suspicious_ids,
        top_k=top_k,
        t_match=t_match,
        sparse_margin=sparse_margin,
        min_cluster_margin=min_cluster_margin,
    )
    deferred_rows = base._evaluate_deferred(
        faces_by_id=faces_by_id,
        current=current,
        proposed=strategies,
        cluster_to_person=cluster_to_person,
        top_k=top_k,
        t_match=t_match,
        sparse_margin=sparse_margin,
        ambiguous_band_width=ambiguous_band_width,
        min_cluster_margin=min_cluster_margin,
    )

    target_names = {Path(v).name.lower() for v in args.target_photo}
    target_rows = [r for r in deferred_rows if r["photo"].lower() in target_names] if target_names else []

    cluster_rows: List[dict] = []
    risk_rows: List[dict] = []
    final_exemplar_rows: List[dict] = []
    for cluster_id in active_clusters:
        cur = current_cluster_eval.get(cluster_id, {})
        fin = final_cluster_eval.get(cluster_id, {})
        changed = set(fid for fid in current[cluster_id].exemplar_ids if fid) != set(fid for fid in strategies[cluster_id].exemplar_ids if fid)
        cluster_rows.append({
            "person_folder": _person(cluster_id, cluster_to_person),
            "cluster_id": cluster_id,
            "member_count": len([f for f in faces_by_cluster.get(cluster_id, []) if f.assignment_state in MEMBER_STATES]),
            "changed": changed,
            "accepted_replacements": modified_count[cluster_id],
            "current_correct_rate": cur.get("correct_rate"),
            "final_correct_rate": fin.get("correct_rate"),
            "current_wrong_confirmed": cur.get("wrong_confirmed", 0),
            "final_wrong_confirmed": fin.get("wrong_confirmed", 0),
            "current_own_score_p10": cur.get("own_score_p10"),
            "final_own_score_p10": fin.get("own_score_p10"),
            "p10_delta": (
                fin.get("own_score_p10") - cur.get("own_score_p10")
                if fin.get("own_score_p10") is not None and cur.get("own_score_p10") is not None else None
            ),
        })
        cr = current_risk.get(cluster_id, {})
        fr = final_risk.get(cluster_id, {})
        risk_rows.append({
            "person_folder": _person(cluster_id, cluster_to_person),
            "cluster_id": cluster_id,
            "current_foreign_faces_above_threshold": cr.get("foreign_faces_above_threshold", 0),
            "final_foreign_faces_above_threshold": fr.get("foreign_faces_above_threshold", 0),
            "delta_foreign_faces_above_threshold": fr.get("foreign_faces_above_threshold", 0) - cr.get("foreign_faces_above_threshold", 0),
            "current_competitive_foreign_faces": cr.get("competitive_foreign_faces", 0),
            "final_competitive_foreign_faces": fr.get("competitive_foreign_faces", 0),
            "delta_competitive_foreign_faces": fr.get("competitive_foreign_faces", 0) - cr.get("competitive_foreign_faces", 0),
            "current_max_foreign_score": cr.get("max_foreign_score"),
            "final_max_foreign_score": fr.get("max_foreign_score"),
        })
        current_ids = {fid for fid in current[cluster_id].exemplar_ids if fid}
        for idx, (fid, quality_score, yaw, label) in enumerate(zip(
            strategies[cluster_id].exemplar_ids,
            strategies[cluster_id].qualities,
            strategies[cluster_id].yaws,
            strategies[cluster_id].labels,
        )):
            face = faces_by_id.get(fid or "")
            final_exemplar_rows.append({
                "person_folder": _person(cluster_id, cluster_to_person),
                "cluster_id": cluster_id,
                "slot": idx,
                "face_id": fid or "",
                "photo": face.photo_name if face else "",
                "quality_score": quality_score,
                "yaw_ratio": yaw,
                "role": label,
                "was_current_exemplar": bool(fid and fid in current_ids),
            })

    total_current_foreign = sum(v["foreign_faces_above_threshold"] for v in current_risk.values())
    total_final_foreign = sum(v["foreign_faces_above_threshold"] for v in final_risk.values())
    total_current_comp = sum(v["competitive_foreign_faces"] for v in current_risk.values())
    total_final_comp = sum(v["competitive_foreign_faces"] for v in final_risk.values())

    newly_confirmed = sum(1 for r in deferred_rows if r["newly_confirmed_by_proposed"])
    changed_best = sum(1 for r in deferred_rows if r["proposed_changes_best_cluster"])

    summary = {
        "test": "exemplar_strategy_simulation_v2_safe_one_at_a_time",
        "run_output": str(run_output),
        "production_db_modified": False,
        "active_clusters": len(active_clusters),
        "final_suspicious_faces_excluded": len(suspicious_ids),
        "thresholds_reused_unchanged": {
            "t_match": t_match,
            "top_k": top_k,
            "sparse_cluster_margin": sparse_margin,
            "min_cluster_margin": min_cluster_margin,
            "ambiguous_band_width": ambiguous_band_width,
            "exemplar_quality_threshold": exemplar_quality_threshold,
        },
        "policy": {
            "mature_cluster_min_faces": args.mature_min_faces,
            "new_candidate_min_other_member_support_at_t_match": args.member_support,
            "max_replacements_per_cluster": args.max_replacements_per_cluster,
            "max_total_actions": args.max_total_actions,
            "min_target_p10_gain_without_direct_recovery": args.min_p10_gain,
            "max_candidates_per_cluster": args.max_candidates_per_cluster,
            "hard_safety": [
                "wrong_confirmed_must_not_increase",
                "correct_confirmed_must_not_decrease",
                "competitive_foreign_attraction_must_not_increase",
                "target_competitive_foreign_attraction_must_not_increase",
                "deferred_best_identity_must_not_change",
            ],
            "benefit_required": [
                "stable_deferred_recovery_gain",
                "trusted_correct_confirmed_gain",
                "or_target_p10_gain_at_least_minimum",
            ],
        },
        "selection": {
            "accepted_replacements": len(actions),
            "clusters_changed": sum(1 for v in modified_count.values() if v > 0),
            "replacement_actions": actions,
            "evaluated_replacement_trials": len(all_evaluation_rows),
        },
        "trusted_member_identification": {
            "current": current_summary,
            "safe_v2": final_summary,
            "delta_correct_confirmed": final_summary["correct_confirmed"] - current_summary["correct_confirmed"],
            "delta_wrong_confirmed": final_summary["wrong_confirmed"] - current_summary["wrong_confirmed"],
        },
        "cross_cluster_attraction": {
            "current_foreign_faces_above_threshold_total": total_current_foreign,
            "safe_v2_foreign_faces_above_threshold_total": total_final_foreign,
            "delta_foreign_faces_above_threshold_total": total_final_foreign - total_current_foreign,
            "current_competitive_foreign_faces_total": total_current_comp,
            "safe_v2_competitive_foreign_faces_total": total_final_comp,
            "delta_competitive_foreign_faces_total": total_final_comp - total_current_comp,
        },
        "deferred_faces": {
            "evaluated_ambiguous_unassigned_nonrestricted": len(deferred_rows),
            "newly_confirmed_by_safe_v2": newly_confirmed,
            "safe_v2_changes_best_cluster": changed_best,
        },
        "target_photos_requested": sorted(target_names),
        "target_rows_found": len(target_rows),
        "interpretation_guardrails": [
            "This simulator never writes to production DB.",
            "At most one exemplar is replaced per mature cluster by default; current exemplars remain the baseline.",
            "A deferred gain is treated as high-confidence only when the same cluster was already the best identity before the replacement.",
            "Deferred newly-confirmed rows are still candidates for inspection, not external ground truth.",
            "No face-similarity threshold is raised or lowered.",
        ],
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (output_dir / "selection_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump({"clusters": cluster_diags, "actions": actions}, f, indent=2, ensure_ascii=False)

    action_fields = [
        "action_index", "iteration", "person_folder", "cluster_id",
        "removed_face_id", "removed_photo", "added_face_id", "added_photo",
        "accepted_reasons", "candidate_quality", "candidate_support_count", "candidate_diversity",
        "correct_delta_at_accept", "wrong_delta_at_accept", "target_p10_delta_at_accept",
        "competitive_foreign_delta_at_accept", "foreign_above_delta_at_accept",
        "deferred_best_cluster_changes_at_accept", "stable_deferred_gains_at_accept",
    ]
    _write_csv(output_dir / "replacement_actions.csv", actions, action_fields)
    eval_fields = list(all_evaluation_rows[0].keys()) if all_evaluation_rows else ["iteration"]
    _write_csv(output_dir / "replacement_evaluations.csv", all_evaluation_rows, eval_fields)
    _write_csv(
        output_dir / "cluster_comparison.csv",
        cluster_rows,
        [
            "person_folder", "cluster_id", "member_count", "changed", "accepted_replacements",
            "current_correct_rate", "final_correct_rate", "current_wrong_confirmed", "final_wrong_confirmed",
            "current_own_score_p10", "final_own_score_p10", "p10_delta",
        ],
    )
    _write_csv(
        output_dir / "cross_cluster_risk.csv",
        risk_rows,
        [
            "person_folder", "cluster_id",
            "current_foreign_faces_above_threshold", "final_foreign_faces_above_threshold", "delta_foreign_faces_above_threshold",
            "current_competitive_foreign_faces", "final_competitive_foreign_faces", "delta_competitive_foreign_faces",
            "current_max_foreign_score", "final_max_foreign_score",
        ],
    )
    _write_csv(
        output_dir / "final_exemplars.csv",
        final_exemplar_rows,
        ["person_folder", "cluster_id", "slot", "face_id", "photo", "quality_score", "yaw_ratio", "role", "was_current_exemplar"],
    )
    _write_csv(
        output_dir / "member_identification_current.csv",
        current_member_rows,
        list(current_member_rows[0].keys()) if current_member_rows else ["strategy"],
    )
    _write_csv(
        output_dir / "member_identification_safe_v2.csv",
        final_member_rows,
        list(final_member_rows[0].keys()) if final_member_rows else ["strategy"],
    )
    _write_csv(
        output_dir / "deferred_face_comparison.csv",
        deferred_rows,
        list(deferred_rows[0].keys()) if deferred_rows else ["photo"],
    )
    _write_csv(
        output_dir / "target_photo_report.csv",
        target_rows,
        list(target_rows[0].keys()) if target_rows else ["photo"],
    )

    print("=== EXEMPLAR STRATEGY SIMULATION v2 — SAFE ONE-AT-A-TIME ===")
    print(f"Run output:                    {run_output}")
    print(f"Active clusters:               {len(active_clusters)}")
    print(f"Replacement trials:            {len(all_evaluation_rows)}")
    print(f"Accepted replacements:         {len(actions)}")
    print(f"Clusters changed:              {sum(1 for v in modified_count.values() if v > 0)}")
    print()
    print("=== TRUSTED MEMBER LEAVE-ONE-OUT IDENTIFICATION ===")
    print(f"Current correct confirmed:     {current_summary['correct_confirmed']}/{current_summary['evaluated_members']}")
    print(f"Safe-v2 correct confirmed:     {final_summary['correct_confirmed']}/{final_summary['evaluated_members']}")
    print(f"Current wrong confirmed:       {current_summary['wrong_confirmed']}")
    print(f"Safe-v2 wrong confirmed:       {final_summary['wrong_confirmed']}")
    if current_summary["own_leave_one_out_score_p10"] is not None:
        print(f"Current own-score p10:         {current_summary['own_leave_one_out_score_p10']:.4f}")
    if final_summary["own_leave_one_out_score_p10"] is not None:
        print(f"Safe-v2 own-score p10:         {final_summary['own_leave_one_out_score_p10']:.4f}")
    print()
    print("=== CROSS-CLUSTER ATTRACTION ===")
    print(f"Foreign above threshold:       {total_current_foreign} -> {total_final_foreign}")
    print(f"Competitive foreign faces:     {total_current_comp} -> {total_final_comp}")
    print()
    print("=== DEFERRED FACES ===")
    print(f"Evaluated:                     {len(deferred_rows)}")
    print(f"Newly confirmed by safe-v2:    {newly_confirmed}")
    print(f"Best-cluster changes:          {changed_best}")
    if target_names:
        print(f"Requested target rows found:   {len(target_rows)}")
        for row in target_rows:
            print(
                f"  {row['photo']} face={row['face_index']} | "
                f"current {row['current_sim_state']} {row['current_best_score']} {_person(row['current_best_cluster_id'], cluster_to_person)} | "
                f"safe-v2 {row['proposed_sim_state']} {row['proposed_best_score']} {_person(row['proposed_best_cluster_id'], cluster_to_person)}"
            )
    if actions:
        print()
        print("=== ACCEPTED SAFE REPLACEMENTS ===")
        for action in actions:
            print(
                f"  {_person(action['cluster_id'], cluster_to_person)}: "
                f"{action['removed_photo']} -> {action['added_photo']} | "
                f"reason={action['accepted_reasons']}"
            )
    print(f"Output:                        {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
