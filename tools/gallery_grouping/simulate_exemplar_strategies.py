#!/usr/bin/env python3
"""Read-only simulator for exemplar-selection strategies.

Purpose
-------
Compare the production stored exemplar set (currently 3 quality + 2 pose)
against a proposed mature-cluster rebuild strategy that optimizes internal
coverage and embedding diversity while preserving the existing matching
thresholds and Top-K decision logic.

The simulator NEVER writes to the production SQLite database.

Typical usage from project root::

    python tools/gallery_grouping/simulate_exemplar_strategies.py \
      --run-output data/gallery_grouping_output_run002 \
      --target-photo IMG-20250423-WA0000.jpg

Outputs are written under::

    <run-output>/exemplar_strategy_simulation/

The proposed strategy is intentionally conservative:
- only mature clusters are rebuilt (default >= 8 trusted faces),
- candidates must be normal CONFIRMED/MANUAL faces,
- recognition-restricted and final-suspicious faces never become exemplars,
- candidates must satisfy the existing exemplar quality gate,
- a new non-current candidate must have repeated same-cluster support from
  at least two OTHER trusted faces at the existing T_match,
- maximum exemplar count stays fixed at 5,
- no production similarity threshold is changed.

The proposed selection uses:
1) a central medoid-like anchor,
2) a high-quality anchor,
3) greedy representatives that maximize leave-one-out Top-2 member coverage,
   high-confidence coverage, lower-tail score, mean score, then embedding
   diversity and quality as tie-breakers.

This is a simulator, not a production policy. The current cluster membership is
used as the evaluation reference, after excluding final suspicious faces.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from face_grouping.matching.assignment import (  # noqa: E402
    AssignmentState,
    ClusterCandidate,
    decide_assignment,
)
from face_grouping.matching.similarity import cosine_similarity, top_k_average_similarity  # noqa: E402


DB_FILENAME = "gallery_grouping.db"
OUTPUT_DIRNAME = "exemplar_strategy_simulation"
MEMBER_STATES = {"confirmed", "manual"}


@dataclass
class FaceRec:
    face_id: str
    cluster_id: Optional[str]
    photo_id: Optional[str]
    photo_name: str
    face_index: Optional[int]
    embedding: np.ndarray
    quality_score: float
    yaw_ratio: float
    assignment_state: str
    recognition_restricted: bool
    is_manually_corrected: bool


@dataclass
class ExemplarRec:
    cluster_id: str
    face_id: Optional[str]
    embedding: np.ndarray
    quality_score: float
    yaw_ratio: float
    bucket: str


@dataclass
class StrategySet:
    cluster_id: str
    exemplar_ids: List[Optional[str]]
    embeddings: List[np.ndarray]
    qualities: List[float]
    yaws: List[float]
    labels: List[str]

    def count(self) -> int:
        return len(self.embeddings)


@dataclass
class DecisionResult:
    state: str
    predicted_cluster_id: Optional[str]
    best_cluster_id: Optional[str]
    best_score: Optional[float]
    second_cluster_id: Optional[str]
    second_score: Optional[float]
    margin: Optional[float]
    threshold: Optional[float]
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare current vs coverage/diversity exemplar strategies read-only.")
    parser.add_argument(
        "--run-output",
        default="data/gallery_grouping_output",
        help="Completed gallery run directory containing gallery_grouping.db.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional explicit diagnostic output directory (default: <run-output>/exemplar_strategy_simulation).",
    )
    parser.add_argument(
        "--target-photo",
        action="append",
        default=[],
        help="Optional photo basename to highlight. Can be provided multiple times.",
    )
    parser.add_argument(
        "--max-exemplars",
        type=int,
        default=5,
        help="Maximum proposed exemplars per mature cluster (default: 5).",
    )
    parser.add_argument(
        "--mature-min-faces",
        type=int,
        default=8,
        help="Only rebuild clusters with at least this many trusted evaluation members (default: 8).",
    )
    parser.add_argument(
        "--member-support",
        type=int,
        default=2,
        help="Required OTHER same-cluster supporters for a new diversity candidate at existing T_match (default: 2).",
    )
    return parser.parse_args()


def _read_thresholds() -> dict:
    path = PROJECT_ROOT / "configs" / "thresholds.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def _load_person_map(run_output: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    cluster_to_person: Dict[str, str] = {}
    person_to_cluster: Dict[str, str] = {}
    path = run_output / "clusters.csv"
    if not path.exists():
        return cluster_to_person, person_to_cluster
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cluster_id = (row.get("cluster_id") or "").strip()
            person = (row.get("person_folder") or "").strip()
            if cluster_id and person:
                cluster_to_person[cluster_id] = person
                person_to_cluster[person] = cluster_id
    return cluster_to_person, person_to_cluster


def _load_suspicious_face_ids(run_output: Path) -> set[str]:
    path = run_output / "suspicious_faces.csv"
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            face_id = (row.get("face_id") or "").strip()
            if face_id:
                out.add(face_id)
    return out


def _load_data(conn: sqlite3.Connection) -> Tuple[Dict[str, FaceRec], Dict[str, List[FaceRec]], Dict[str, List[ExemplarRec]], List[str]]:
    active_clusters = [
        row["cluster_id"]
        for row in conn.execute("SELECT cluster_id FROM clusters WHERE merged_into IS NULL ORDER BY cluster_id")
    ]
    active = set(active_clusters)

    photos = {
        row["photo_id"]: Path(row["image_path"]).name
        for row in conn.execute("SELECT photo_id, image_path FROM photos")
    }

    faces_by_id: Dict[str, FaceRec] = {}
    faces_by_cluster: Dict[str, List[FaceRec]] = defaultdict(list)
    for row in conn.execute(
        """
        SELECT face_id, cluster_id, photo_id, face_index, embedding, quality_score,
               yaw_ratio, assignment_state, recognition_restricted, is_manually_corrected
        FROM faces
        """
    ):
        face = FaceRec(
            face_id=row["face_id"],
            cluster_id=row["cluster_id"],
            photo_id=row["photo_id"],
            photo_name=photos.get(row["photo_id"], ""),
            face_index=row["face_index"],
            embedding=_blob_to_embedding(row["embedding"]),
            quality_score=float(row["quality_score"]),
            yaw_ratio=float(row["yaw_ratio"]),
            assignment_state=row["assignment_state"],
            recognition_restricted=bool(row["recognition_restricted"]),
            is_manually_corrected=bool(row["is_manually_corrected"]),
        )
        faces_by_id[face.face_id] = face
        if face.cluster_id in active:
            faces_by_cluster[face.cluster_id].append(face)

    exemplars_by_cluster: Dict[str, List[ExemplarRec]] = defaultdict(list)
    for row in conn.execute(
        """
        SELECT cluster_id, bucket, face_id, embedding, quality_score, yaw_ratio
        FROM exemplars
        ORDER BY cluster_id, CASE bucket WHEN 'quality' THEN 0 ELSE 1 END, id
        """
    ):
        if row["cluster_id"] not in active:
            continue
        exemplars_by_cluster[row["cluster_id"]].append(
            ExemplarRec(
                cluster_id=row["cluster_id"],
                face_id=row["face_id"],
                embedding=_blob_to_embedding(row["embedding"]),
                quality_score=float(row["quality_score"]),
                yaw_ratio=float(row["yaw_ratio"]),
                bucket=row["bucket"],
            )
        )

    return faces_by_id, faces_by_cluster, exemplars_by_cluster, active_clusters


def _current_strategy(exemplars_by_cluster: Dict[str, List[ExemplarRec]], active_clusters: Sequence[str]) -> Dict[str, StrategySet]:
    out: Dict[str, StrategySet] = {}
    for cluster_id in active_clusters:
        exs = exemplars_by_cluster.get(cluster_id, [])
        out[cluster_id] = StrategySet(
            cluster_id=cluster_id,
            exemplar_ids=[e.face_id for e in exs],
            embeddings=[e.embedding for e in exs],
            qualities=[e.quality_score for e in exs],
            yaws=[e.yaw_ratio for e in exs],
            labels=[f"current_{e.bucket}" for e in exs],
        )
    return out


def _score_with_ids(
    face: FaceRec,
    strategy_set: StrategySet,
    *,
    top_k: int,
    leave_one_out: bool,
) -> Tuple[Optional[float], int, List[float]]:
    embeddings: List[np.ndarray] = []
    for face_id, emb in zip(strategy_set.exemplar_ids, strategy_set.embeddings):
        if leave_one_out and face_id is not None and face_id == face.face_id:
            continue
        embeddings.append(emb)
    if not embeddings:
        return None, 0, []
    score, sims = top_k_average_similarity(face.embedding, embeddings, k=top_k)
    return float(score), len(embeddings), [float(v) for v in sims]


def _support_count(candidate: FaceRec, trusted_members: Sequence[FaceRec], t_match: float) -> int:
    count = 0
    for member in trusted_members:
        if member.face_id == candidate.face_id:
            continue
        if cosine_similarity(candidate.embedding, member.embedding) >= t_match:
            count += 1
    return count


def _mean_similarity_to_members(candidate: FaceRec, trusted_members: Sequence[FaceRec]) -> float:
    if not trusted_members:
        return -1.0
    return float(np.mean([cosine_similarity(candidate.embedding, m.embedding) for m in trusted_members]))


def _selection_objective(
    trusted_members: Sequence[FaceRec],
    selected: Sequence[FaceRec],
    *,
    top_k: int,
    t_match: float,
    high_conf_floor: float,
) -> Tuple[int, int, float, float]:
    if not selected:
        return (0, 0, -1.0, -1.0)
    strategy = StrategySet(
        cluster_id="_tmp",
        exemplar_ids=[f.face_id for f in selected],
        embeddings=[f.embedding for f in selected],
        qualities=[f.quality_score for f in selected],
        yaws=[f.yaw_ratio for f in selected],
        labels=["tmp"] * len(selected),
    )
    scores: List[float] = []
    for member in trusted_members:
        score, _, _ = _score_with_ids(member, strategy, top_k=top_k, leave_one_out=True)
        if score is not None:
            scores.append(score)
    if not scores:
        return (0, 0, -1.0, -1.0)
    arr = np.asarray(scores, dtype=np.float64)
    return (
        int(np.sum(arr >= t_match)),
        int(np.sum(arr >= high_conf_floor)),
        float(np.quantile(arr, 0.10)),
        float(np.mean(arr)),
    )


def _candidate_diversity(candidate: FaceRec, selected: Sequence[FaceRec]) -> float:
    if not selected:
        return 1.0
    return float(min(1.0 - cosine_similarity(candidate.embedding, s.embedding) for s in selected))


def _build_proposed_strategy(
    *,
    cluster_id: str,
    members: Sequence[FaceRec],
    current_set: StrategySet,
    suspicious_ids: set[str],
    exemplar_quality_threshold: float,
    t_match: float,
    high_conf_floor: float,
    top_k: int,
    max_exemplars: int,
    mature_min_faces: int,
    member_support: int,
) -> Tuple[StrategySet, dict]:
    trusted_members = [
        f for f in members
        if f.assignment_state in MEMBER_STATES
        and not f.recognition_restricted
        and f.face_id not in suspicious_ids
    ]
    current_ids = {fid for fid in current_set.exemplar_ids if fid}

    diag = {
        "cluster_id": cluster_id,
        "trusted_members": len(trusted_members),
        "eligible_candidates": 0,
        "safe_candidates": 0,
        "strategy": "current_fallback",
        "reason": "",
        "selected": [],
    }

    if len(trusted_members) < mature_min_faces:
        diag["reason"] = "cluster_below_mature_min_faces"
        return current_set, diag

    eligible = [
        f for f in trusted_members
        if f.quality_score >= exemplar_quality_threshold
    ]
    diag["eligible_candidates"] = len(eligible)
    if len(eligible) < 2:
        diag["reason"] = "fewer_than_two_exemplar_eligible_faces"
        return current_set, diag

    supports = {f.face_id: _support_count(f, trusted_members, t_match) for f in eligible}
    safe = [
        f for f in eligible
        if supports[f.face_id] >= member_support or f.face_id in current_ids
    ]
    diag["safe_candidates"] = len(safe)
    if len(safe) < 2:
        diag["reason"] = "fewer_than_two_supported_candidates"
        return current_set, diag

    mean_sims = {f.face_id: _mean_similarity_to_members(f, trusted_members) for f in safe}

    # Central anchor: supported candidate with the strongest mean affinity to
    # the trusted cluster members. Quality breaks near ties.
    central = max(safe, key=lambda f: (mean_sims[f.face_id], f.quality_score, f.face_id))
    selected: List[FaceRec] = [central]
    selection_labels: Dict[str, str] = {central.face_id: "central_medoid"}

    # High-quality anchor. We deliberately choose it independently of pose;
    # the later greedy objective is responsible for coverage/diversity.
    remaining = [f for f in safe if f.face_id != central.face_id]
    if remaining and len(selected) < max_exemplars:
        quality_anchor = max(remaining, key=lambda f: (f.quality_score, mean_sims[f.face_id], f.face_id))
        selected.append(quality_anchor)
        selection_labels[quality_anchor.face_id] = "quality_anchor"

    # Greedy coverage/diversity representatives. Existing thresholds are used
    # only as evaluation floors; no new similarity threshold is introduced.
    while len(selected) < min(max_exemplars, len(safe)):
        best: Optional[FaceRec] = None
        best_key: Optional[Tuple[float, ...]] = None
        for candidate in safe:
            if any(candidate.face_id == s.face_id for s in selected):
                continue
            coverage, high_conf, p10, mean_score = _selection_objective(
                trusted_members,
                [*selected, candidate],
                top_k=top_k,
                t_match=t_match,
                high_conf_floor=high_conf_floor,
            )
            diversity = _candidate_diversity(candidate, selected)
            key: Tuple[float, ...] = (
                float(coverage),
                float(high_conf),
                p10,
                mean_score,
                diversity,
                float(supports[candidate.face_id]),
                candidate.quality_score,
            )
            if best_key is None or key > best_key:
                best = candidate
                best_key = key
        if best is None:
            break
        selected.append(best)
        selection_labels[best.face_id] = "coverage_diversity"

    # If the supported pool is smaller than max_exemplars, retain safe current
    # production anchors first, then any remaining eligible candidate. This is
    # a conservative fallback and never exceeds the current max set size.
    if len(selected) < max_exemplars:
        by_id = {f.face_id: f for f in eligible}
        for face_id in current_set.exemplar_ids:
            if len(selected) >= max_exemplars:
                break
            if not face_id or face_id not in by_id:
                continue
            if any(face_id == s.face_id for s in selected):
                continue
            selected.append(by_id[face_id])
            selection_labels[face_id] = "retained_current_anchor"
    if len(selected) < max_exemplars:
        for candidate in sorted(eligible, key=lambda f: (f.quality_score, mean_sims.get(f.face_id, -1.0)), reverse=True):
            if len(selected) >= max_exemplars:
                break
            if any(candidate.face_id == s.face_id for s in selected):
                continue
            selected.append(candidate)
            selection_labels[candidate.face_id] = "quality_fill"

    proposed = StrategySet(
        cluster_id=cluster_id,
        exemplar_ids=[f.face_id for f in selected],
        embeddings=[f.embedding for f in selected],
        qualities=[f.quality_score for f in selected],
        yaws=[f.yaw_ratio for f in selected],
        labels=[selection_labels[f.face_id] for f in selected],
    )
    diag["strategy"] = "coverage_embedding_diversity"
    diag["reason"] = "mature_cluster_rebuilt"
    diag["selected"] = [
        {
            "face_id": f.face_id,
            "photo": f.photo_name,
            "quality_score": f.quality_score,
            "yaw_ratio": f.yaw_ratio,
            "support_count": supports.get(f.face_id, 0),
            "mean_member_similarity": mean_sims.get(f.face_id, _mean_similarity_to_members(f, trusted_members)),
            "role": selection_labels[f.face_id],
            "was_current_exemplar": f.face_id in current_ids,
        }
        for f in selected
    ]
    return proposed, diag


def _effective_threshold(exemplar_count: int, *, top_k: int, t_match: float, sparse_margin: float) -> float:
    return t_match + sparse_margin if exemplar_count < top_k else t_match


def _decide_face(
    face: FaceRec,
    strategies: Dict[str, StrategySet],
    *,
    top_k: int,
    t_match: float,
    sparse_margin: float,
    ambiguous_band_width: float,
    min_cluster_margin: float,
    leave_one_out_cluster_id: Optional[str] = None,
) -> DecisionResult:
    candidates: List[ClusterCandidate] = []
    for cluster_id, strategy in strategies.items():
        score, count, similarities = _score_with_ids(
            face,
            strategy,
            top_k=top_k,
            leave_one_out=(leave_one_out_cluster_id == cluster_id),
        )
        if score is None or count <= 0:
            continue
        candidates.append(
            ClusterCandidate(
                cluster_id=cluster_id,
                score=score,
                exemplar_count=count,
                effective_threshold=_effective_threshold(
                    count,
                    top_k=top_k,
                    t_match=t_match,
                    sparse_margin=sparse_margin,
                ),
                similarities=similarities,
            )
        )

    decision = decide_assignment(
        candidates,
        exemplar_eligible=False,
        ambiguous_band_width=ambiguous_band_width,
        min_cluster_margin=min_cluster_margin,
    )
    return DecisionResult(
        state=decision.state.value,
        predicted_cluster_id=decision.assigned_cluster_id,
        best_cluster_id=decision.candidate_cluster_id,
        best_score=decision.best_score,
        second_cluster_id=decision.second_best_cluster_id,
        second_score=decision.second_best_score,
        margin=decision.score_margin,
        threshold=decision.decision_threshold,
        reason=decision.reason,
    )


def _score_to_cluster(
    face: FaceRec,
    strategy: StrategySet,
    *,
    top_k: int,
    leave_one_out: bool,
) -> Tuple[Optional[float], int]:
    score, count, _ = _score_with_ids(face, strategy, top_k=top_k, leave_one_out=leave_one_out)
    return score, count


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe(v: Optional[float]) -> str | float:
    return "" if v is None else float(v)


def _person(cluster_id: Optional[str], mapping: Dict[str, str]) -> str:
    if not cluster_id:
        return ""
    return mapping.get(cluster_id, cluster_id)


def _evaluate_member_strategy(
    *,
    name: str,
    strategies: Dict[str, StrategySet],
    faces_by_cluster: Dict[str, List[FaceRec]],
    suspicious_ids: set[str],
    cluster_to_person: Dict[str, str],
    top_k: int,
    t_match: float,
    sparse_margin: float,
    ambiguous_band_width: float,
    min_cluster_margin: float,
) -> Tuple[List[dict], Dict[str, dict], dict]:
    rows: List[dict] = []
    per_cluster: Dict[str, dict] = {}
    totals = Counter()
    own_scores_all: List[float] = []

    for cluster_id, members in faces_by_cluster.items():
        trusted = [
            f for f in members
            if f.assignment_state in MEMBER_STATES
            and not f.recognition_restricted
            and f.face_id not in suspicious_ids
        ]
        cluster_counts = Counter()
        cluster_own_scores: List[float] = []
        for face in trusted:
            result = _decide_face(
                face,
                strategies,
                top_k=top_k,
                t_match=t_match,
                sparse_margin=sparse_margin,
                ambiguous_band_width=ambiguous_band_width,
                min_cluster_margin=min_cluster_margin,
                leave_one_out_cluster_id=cluster_id,
            )
            own_score, own_count = _score_to_cluster(
                face,
                strategies[cluster_id],
                top_k=top_k,
                leave_one_out=True,
            )
            if own_score is not None:
                cluster_own_scores.append(own_score)
                own_scores_all.append(own_score)

            if result.state == "confirmed" and result.predicted_cluster_id == cluster_id:
                outcome = "correct_confirmed"
            elif result.state == "confirmed" and result.predicted_cluster_id != cluster_id:
                outcome = "wrong_confirmed"
            else:
                outcome = result.state
            totals[outcome] += 1
            cluster_counts[outcome] += 1
            totals["evaluated"] += 1
            cluster_counts["evaluated"] += 1

            rows.append({
                "strategy": name,
                "photo": face.photo_name,
                "photo_id": face.photo_id or "",
                "face_index": "" if face.face_index is None else face.face_index,
                "face_id": face.face_id,
                "true_person": _person(cluster_id, cluster_to_person),
                "true_cluster_id": cluster_id,
                "own_leave_one_out_score": _safe(own_score),
                "own_exemplar_count_after_loo": own_count,
                "decision_state": result.state,
                "predicted_person": _person(result.predicted_cluster_id, cluster_to_person),
                "predicted_cluster_id": result.predicted_cluster_id or "",
                "best_person": _person(result.best_cluster_id, cluster_to_person),
                "best_cluster_id": result.best_cluster_id or "",
                "best_score": _safe(result.best_score),
                "second_person": _person(result.second_cluster_id, cluster_to_person),
                "second_cluster_id": result.second_cluster_id or "",
                "second_score": _safe(result.second_score),
                "margin": _safe(result.margin),
                "threshold": _safe(result.threshold),
                "outcome": outcome,
                "reason": result.reason,
            })

        n = cluster_counts["evaluated"]
        arr = np.asarray(cluster_own_scores, dtype=np.float64) if cluster_own_scores else np.asarray([], dtype=np.float64)
        per_cluster[cluster_id] = {
            "evaluated": n,
            "correct_confirmed": cluster_counts["correct_confirmed"],
            "wrong_confirmed": cluster_counts["wrong_confirmed"],
            "ambiguous": cluster_counts["ambiguous"],
            "unassigned": cluster_counts["unassigned"],
            "correct_rate": (cluster_counts["correct_confirmed"] / n) if n else None,
            "wrong_rate": (cluster_counts["wrong_confirmed"] / n) if n else None,
            "own_score_mean": float(np.mean(arr)) if arr.size else None,
            "own_score_p10": float(np.quantile(arr, 0.10)) if arr.size else None,
            "own_score_min": float(np.min(arr)) if arr.size else None,
        }

    n = totals["evaluated"]
    arr_all = np.asarray(own_scores_all, dtype=np.float64) if own_scores_all else np.asarray([], dtype=np.float64)
    summary = {
        "evaluated_members": n,
        "correct_confirmed": totals["correct_confirmed"],
        "wrong_confirmed": totals["wrong_confirmed"],
        "ambiguous": totals["ambiguous"],
        "unassigned": totals["unassigned"],
        "correct_confirmed_rate": (totals["correct_confirmed"] / n) if n else None,
        "wrong_confirmed_rate": (totals["wrong_confirmed"] / n) if n else None,
        "own_leave_one_out_score_mean": float(np.mean(arr_all)) if arr_all.size else None,
        "own_leave_one_out_score_p10": float(np.quantile(arr_all, 0.10)) if arr_all.size else None,
        "own_leave_one_out_score_min": float(np.min(arr_all)) if arr_all.size else None,
    }
    return rows, per_cluster, summary


def _evaluate_cross_cluster_risk(
    *,
    strategies: Dict[str, StrategySet],
    faces_by_cluster: Dict[str, List[FaceRec]],
    suspicious_ids: set[str],
    top_k: int,
    t_match: float,
    sparse_margin: float,
    min_cluster_margin: float,
) -> Dict[str, dict]:
    # Per target cluster, count foreign trusted faces that are attracted above
    # the production threshold, and those competitive with the face's true
    # own-cluster score within the existing min-cluster margin.
    out: Dict[str, dict] = {}
    for target_id in strategies:
        threshold_count = 0
        competitive_count = 0
        max_foreign_score: Optional[float] = None
        mean_scores: List[float] = []
        for true_id, members in faces_by_cluster.items():
            if true_id == target_id:
                continue
            for face in members:
                if (
                    face.assignment_state not in MEMBER_STATES
                    or face.recognition_restricted
                    or face.face_id in suspicious_ids
                ):
                    continue
                foreign_score, foreign_count = _score_to_cluster(
                    face, strategies[target_id], top_k=top_k, leave_one_out=False
                )
                if foreign_score is None:
                    continue
                own_score, _ = _score_to_cluster(
                    face, strategies[true_id], top_k=top_k, leave_one_out=True
                )
                mean_scores.append(foreign_score)
                threshold = _effective_threshold(
                    foreign_count,
                    top_k=top_k,
                    t_match=t_match,
                    sparse_margin=sparse_margin,
                )
                if foreign_score >= threshold:
                    threshold_count += 1
                    if own_score is None or foreign_score >= own_score - min_cluster_margin:
                        competitive_count += 1
                max_foreign_score = foreign_score if max_foreign_score is None else max(max_foreign_score, foreign_score)
        out[target_id] = {
            "foreign_faces_above_threshold": threshold_count,
            "competitive_foreign_faces": competitive_count,
            "max_foreign_score": max_foreign_score,
            "mean_foreign_score": float(np.mean(mean_scores)) if mean_scores else None,
        }
    return out


def _evaluate_deferred(
    *,
    faces_by_id: Dict[str, FaceRec],
    current: Dict[str, StrategySet],
    proposed: Dict[str, StrategySet],
    cluster_to_person: Dict[str, str],
    top_k: int,
    t_match: float,
    sparse_margin: float,
    ambiguous_band_width: float,
    min_cluster_margin: float,
) -> List[dict]:
    rows: List[dict] = []
    for face in faces_by_id.values():
        if face.assignment_state not in {"ambiguous", "unassigned"}:
            continue
        if face.recognition_restricted:
            continue
        cur = _decide_face(
            face,
            current,
            top_k=top_k,
            t_match=t_match,
            sparse_margin=sparse_margin,
            ambiguous_band_width=ambiguous_band_width,
            min_cluster_margin=min_cluster_margin,
        )
        prop = _decide_face(
            face,
            proposed,
            top_k=top_k,
            t_match=t_match,
            sparse_margin=sparse_margin,
            ambiguous_band_width=ambiguous_band_width,
            min_cluster_margin=min_cluster_margin,
        )
        rows.append({
            "photo": face.photo_name,
            "photo_id": face.photo_id or "",
            "face_index": "" if face.face_index is None else face.face_index,
            "face_id": face.face_id,
            "stored_state": face.assignment_state,
            "current_sim_state": cur.state,
            "current_best_person": _person(cur.best_cluster_id, cluster_to_person),
            "current_best_cluster_id": cur.best_cluster_id or "",
            "current_best_score": _safe(cur.best_score),
            "current_second_person": _person(cur.second_cluster_id, cluster_to_person),
            "current_second_score": _safe(cur.second_score),
            "current_margin": _safe(cur.margin),
            "proposed_sim_state": prop.state,
            "proposed_best_person": _person(prop.best_cluster_id, cluster_to_person),
            "proposed_best_cluster_id": prop.best_cluster_id or "",
            "proposed_best_score": _safe(prop.best_score),
            "proposed_second_person": _person(prop.second_cluster_id, cluster_to_person),
            "proposed_second_score": _safe(prop.second_score),
            "proposed_margin": _safe(prop.margin),
            "newly_confirmed_by_proposed": bool(cur.state != "confirmed" and prop.state == "confirmed"),
            "proposed_changes_best_cluster": bool(cur.best_cluster_id != prop.best_cluster_id),
        })
    return rows


def main() -> int:
    args = parse_args()
    run_output = Path(args.run_output).expanduser().resolve()
    db_path = run_output / DB_FILENAME
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")
    output_dir = Path(args.output).expanduser().resolve() if args.output else run_output / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _read_thresholds()
    matching = thresholds["matching"]
    quality_cfg = thresholds["quality"]
    top_k = int(matching["top_k"])
    t_match = float(matching["t_match"])
    ambiguous_band_width = float(matching["ambiguous_band_width"])
    sparse_margin = float(matching["sparse_cluster_margin"])
    min_cluster_margin = float(matching["min_cluster_margin"])
    exemplar_quality_threshold = float(quality_cfg["exemplar_eligibility_threshold"])
    high_conf_floor = t_match + sparse_margin

    cluster_to_person, _ = _load_person_map(run_output)
    suspicious_ids = _load_suspicious_face_ids(run_output)

    with _open_readonly(db_path) as conn:
        faces_by_id, faces_by_cluster, exemplars_by_cluster, active_clusters = _load_data(conn)

    current = _current_strategy(exemplars_by_cluster, active_clusters)
    proposed: Dict[str, StrategySet] = {}
    selection_diags: List[dict] = []
    selection_rows: List[dict] = []

    for cluster_id in active_clusters:
        proposed_set, diag = _build_proposed_strategy(
            cluster_id=cluster_id,
            members=faces_by_cluster.get(cluster_id, []),
            current_set=current[cluster_id],
            suspicious_ids=suspicious_ids,
            exemplar_quality_threshold=exemplar_quality_threshold,
            t_match=t_match,
            high_conf_floor=high_conf_floor,
            top_k=top_k,
            max_exemplars=args.max_exemplars,
            mature_min_faces=args.mature_min_faces,
            member_support=args.member_support,
        )
        proposed[cluster_id] = proposed_set
        selection_diags.append(diag)
        for item in diag.get("selected", []):
            selection_rows.append({
                "person_folder": _person(cluster_id, cluster_to_person),
                "cluster_id": cluster_id,
                **item,
            })

    current_face_rows, current_cluster_eval, current_summary = _evaluate_member_strategy(
        name="current_3quality_2pose_stored",
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
    proposed_face_rows, proposed_cluster_eval, proposed_summary = _evaluate_member_strategy(
        name="proposed_coverage_embedding_diversity",
        strategies=proposed,
        faces_by_cluster=faces_by_cluster,
        suspicious_ids=suspicious_ids,
        cluster_to_person=cluster_to_person,
        top_k=top_k,
        t_match=t_match,
        sparse_margin=sparse_margin,
        ambiguous_band_width=ambiguous_band_width,
        min_cluster_margin=min_cluster_margin,
    )

    current_risk = _evaluate_cross_cluster_risk(
        strategies=current,
        faces_by_cluster=faces_by_cluster,
        suspicious_ids=suspicious_ids,
        top_k=top_k,
        t_match=t_match,
        sparse_margin=sparse_margin,
        min_cluster_margin=min_cluster_margin,
    )
    proposed_risk = _evaluate_cross_cluster_risk(
        strategies=proposed,
        faces_by_cluster=faces_by_cluster,
        suspicious_ids=suspicious_ids,
        top_k=top_k,
        t_match=t_match,
        sparse_margin=sparse_margin,
        min_cluster_margin=min_cluster_margin,
    )

    cluster_rows: List[dict] = []
    risk_rows: List[dict] = []
    rebuilt_clusters = 0
    changed_exemplar_sets = 0
    for cluster_id in active_clusters:
        cur_eval = current_cluster_eval.get(cluster_id, {})
        prop_eval = proposed_cluster_eval.get(cluster_id, {})
        cur_ids = [fid or "" for fid in current[cluster_id].exemplar_ids]
        prop_ids = [fid or "" for fid in proposed[cluster_id].exemplar_ids]
        rebuilt = any(d["cluster_id"] == cluster_id and d["strategy"] == "coverage_embedding_diversity" for d in selection_diags)
        changed = set(cur_ids) != set(prop_ids)
        if rebuilt:
            rebuilt_clusters += 1
        if changed:
            changed_exemplar_sets += 1
        cluster_rows.append({
            "person_folder": _person(cluster_id, cluster_to_person),
            "cluster_id": cluster_id,
            "member_count": len([f for f in faces_by_cluster.get(cluster_id, []) if f.assignment_state in MEMBER_STATES]),
            "current_exemplar_count": current[cluster_id].count(),
            "proposed_exemplar_count": proposed[cluster_id].count(),
            "rebuilt_by_proposed": rebuilt,
            "exemplar_set_changed": changed,
            "current_correct_rate": cur_eval.get("correct_rate"),
            "proposed_correct_rate": prop_eval.get("correct_rate"),
            "correct_rate_delta": (
                (prop_eval.get("correct_rate") - cur_eval.get("correct_rate"))
                if prop_eval.get("correct_rate") is not None and cur_eval.get("correct_rate") is not None
                else None
            ),
            "current_wrong_confirmed": cur_eval.get("wrong_confirmed", 0),
            "proposed_wrong_confirmed": prop_eval.get("wrong_confirmed", 0),
            "current_own_score_p10": cur_eval.get("own_score_p10"),
            "proposed_own_score_p10": prop_eval.get("own_score_p10"),
            "own_score_p10_delta": (
                (prop_eval.get("own_score_p10") - cur_eval.get("own_score_p10"))
                if prop_eval.get("own_score_p10") is not None and cur_eval.get("own_score_p10") is not None
                else None
            ),
            "current_own_score_mean": cur_eval.get("own_score_mean"),
            "proposed_own_score_mean": prop_eval.get("own_score_mean"),
        })
        cr = current_risk.get(cluster_id, {})
        pr = proposed_risk.get(cluster_id, {})
        risk_rows.append({
            "person_folder": _person(cluster_id, cluster_to_person),
            "cluster_id": cluster_id,
            "current_foreign_faces_above_threshold": cr.get("foreign_faces_above_threshold", 0),
            "proposed_foreign_faces_above_threshold": pr.get("foreign_faces_above_threshold", 0),
            "delta_foreign_faces_above_threshold": pr.get("foreign_faces_above_threshold", 0) - cr.get("foreign_faces_above_threshold", 0),
            "current_competitive_foreign_faces": cr.get("competitive_foreign_faces", 0),
            "proposed_competitive_foreign_faces": pr.get("competitive_foreign_faces", 0),
            "delta_competitive_foreign_faces": pr.get("competitive_foreign_faces", 0) - cr.get("competitive_foreign_faces", 0),
            "current_max_foreign_score": cr.get("max_foreign_score"),
            "proposed_max_foreign_score": pr.get("max_foreign_score"),
            "current_mean_foreign_score": cr.get("mean_foreign_score"),
            "proposed_mean_foreign_score": pr.get("mean_foreign_score"),
        })

    deferred_rows = _evaluate_deferred(
        faces_by_id=faces_by_id,
        current=current,
        proposed=proposed,
        cluster_to_person=cluster_to_person,
        top_k=top_k,
        t_match=t_match,
        sparse_margin=sparse_margin,
        ambiguous_band_width=ambiguous_band_width,
        min_cluster_margin=min_cluster_margin,
    )

    target_names = {Path(v).name.lower() for v in args.target_photo}
    target_rows = [row for row in deferred_rows if row["photo"].lower() in target_names] if target_names else []

    current_wrong = current_summary["wrong_confirmed"]
    proposed_wrong = proposed_summary["wrong_confirmed"]
    newly_confirmed = sum(1 for row in deferred_rows if row["newly_confirmed_by_proposed"])
    proposed_changes_best = sum(1 for row in deferred_rows if row["proposed_changes_best_cluster"])
    total_current_foreign = sum(v["foreign_faces_above_threshold"] for v in current_risk.values())
    total_proposed_foreign = sum(v["foreign_faces_above_threshold"] for v in proposed_risk.values())
    total_current_competitive = sum(v["competitive_foreign_faces"] for v in current_risk.values())
    total_proposed_competitive = sum(v["competitive_foreign_faces"] for v in proposed_risk.values())

    summary = {
        "test": "exemplar_strategy_simulation_v1",
        "run_output": str(run_output),
        "production_db_modified": False,
        "active_clusters": len(active_clusters),
        "final_suspicious_faces_excluded_from_candidate_and_reference_evaluation": len(suspicious_ids),
        "thresholds_reused_unchanged": {
            "t_match": t_match,
            "top_k": top_k,
            "sparse_cluster_margin": sparse_margin,
            "min_cluster_margin": min_cluster_margin,
            "ambiguous_band_width": ambiguous_band_width,
            "exemplar_quality_threshold": exemplar_quality_threshold,
        },
        "proposed_policy": {
            "max_exemplars": args.max_exemplars,
            "mature_cluster_min_faces": args.mature_min_faces,
            "new_candidate_min_other_member_support_at_t_match": args.member_support,
            "recognition_restricted_excluded": True,
            "final_suspicious_excluded": True,
            "selection": "central medoid + quality anchor + greedy leave-one-out coverage/high-confidence/lower-tail/mean + embedding diversity tie-break",
        },
        "selection": {
            "mature_clusters_rebuilt": rebuilt_clusters,
            "clusters_whose_exemplar_face_set_changed": changed_exemplar_sets,
        },
        "trusted_member_identification": {
            "current": current_summary,
            "proposed": proposed_summary,
            "delta_correct_confirmed": proposed_summary["correct_confirmed"] - current_summary["correct_confirmed"],
            "delta_wrong_confirmed": proposed_wrong - current_wrong,
        },
        "cross_cluster_attraction": {
            "current_foreign_faces_above_threshold_total": total_current_foreign,
            "proposed_foreign_faces_above_threshold_total": total_proposed_foreign,
            "delta_foreign_faces_above_threshold_total": total_proposed_foreign - total_current_foreign,
            "current_competitive_foreign_faces_total": total_current_competitive,
            "proposed_competitive_foreign_faces_total": total_proposed_competitive,
            "delta_competitive_foreign_faces_total": total_proposed_competitive - total_current_competitive,
        },
        "deferred_faces": {
            "evaluated_ambiguous_unassigned_nonrestricted": len(deferred_rows),
            "newly_confirmed_by_proposed": newly_confirmed,
            "proposed_changes_best_cluster": proposed_changes_best,
        },
        "target_photos_requested": sorted(target_names),
        "target_rows_found": len(target_rows),
        "interpretation_guardrails": [
            "This is a read-only simulation; it does not rebuild production exemplars or reassign faces.",
            "Current final cluster membership is used as reference for leave-one-out evaluation, after excluding final suspicious faces.",
            "A proposed gain is not sufficient for production adoption if wrong-confirmed or competitive cross-cluster attraction increases materially.",
            "Deferred-face newly-confirmed results are candidates for inspection, not ground-truth correctness claims.",
            "No face-similarity threshold is raised or lowered by this simulator.",
        ],
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (output_dir / "selection_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(selection_diags, f, indent=2, ensure_ascii=False)

    _write_csv(
        output_dir / "cluster_comparison.csv",
        cluster_rows,
        [
            "person_folder", "cluster_id", "member_count",
            "current_exemplar_count", "proposed_exemplar_count",
            "rebuilt_by_proposed", "exemplar_set_changed",
            "current_correct_rate", "proposed_correct_rate", "correct_rate_delta",
            "current_wrong_confirmed", "proposed_wrong_confirmed",
            "current_own_score_p10", "proposed_own_score_p10", "own_score_p10_delta",
            "current_own_score_mean", "proposed_own_score_mean",
        ],
    )
    _write_csv(
        output_dir / "proposed_exemplars.csv",
        selection_rows,
        [
            "person_folder", "cluster_id", "face_id", "photo", "role",
            "quality_score", "yaw_ratio", "support_count", "mean_member_similarity",
            "was_current_exemplar",
        ],
    )
    _write_csv(
        output_dir / "member_identification_current.csv",
        current_face_rows,
        list(current_face_rows[0].keys()) if current_face_rows else ["strategy"],
    )
    _write_csv(
        output_dir / "member_identification_proposed.csv",
        proposed_face_rows,
        list(proposed_face_rows[0].keys()) if proposed_face_rows else ["strategy"],
    )
    _write_csv(
        output_dir / "cross_cluster_risk.csv",
        risk_rows,
        [
            "person_folder", "cluster_id",
            "current_foreign_faces_above_threshold", "proposed_foreign_faces_above_threshold", "delta_foreign_faces_above_threshold",
            "current_competitive_foreign_faces", "proposed_competitive_foreign_faces", "delta_competitive_foreign_faces",
            "current_max_foreign_score", "proposed_max_foreign_score",
            "current_mean_foreign_score", "proposed_mean_foreign_score",
        ],
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

    print("=== EXEMPLAR STRATEGY SIMULATION ===")
    print(f"Run output:                    {run_output}")
    print(f"Active clusters:               {len(active_clusters)}")
    print(f"Mature clusters rebuilt:       {rebuilt_clusters}")
    print(f"Changed exemplar sets:         {changed_exemplar_sets}")
    print()
    print("=== TRUSTED MEMBER LEAVE-ONE-OUT IDENTIFICATION ===")
    print(f"Current correct confirmed:     {current_summary['correct_confirmed']}/{current_summary['evaluated_members']}")
    print(f"Proposed correct confirmed:    {proposed_summary['correct_confirmed']}/{proposed_summary['evaluated_members']}")
    print(f"Current wrong confirmed:       {current_wrong}")
    print(f"Proposed wrong confirmed:      {proposed_wrong}")
    print(f"Current own-score p10:         {current_summary['own_leave_one_out_score_p10']:.4f}" if current_summary['own_leave_one_out_score_p10'] is not None else "Current own-score p10:         n/a")
    print(f"Proposed own-score p10:        {proposed_summary['own_leave_one_out_score_p10']:.4f}" if proposed_summary['own_leave_one_out_score_p10'] is not None else "Proposed own-score p10:        n/a")
    print()
    print("=== CROSS-CLUSTER ATTRACTION ===")
    print(f"Foreign above threshold:       {total_current_foreign} -> {total_proposed_foreign}")
    print(f"Competitive foreign faces:     {total_current_competitive} -> {total_proposed_competitive}")
    print()
    print("=== DEFERRED FACES ===")
    print(f"Evaluated:                     {len(deferred_rows)}")
    print(f"Newly confirmed by proposed:   {newly_confirmed}")
    if target_names:
        print(f"Requested target rows found:   {len(target_rows)}")
        for row in target_rows:
            print(
                f"  {row['photo']} face={row['face_index']} | "
                f"current {row['current_sim_state']} {_safe(row['current_best_score'] if isinstance(row['current_best_score'], float) else None)} {_person(row['current_best_cluster_id'], cluster_to_person)} | "
                f"proposed {row['proposed_sim_state']} {_safe(row['proposed_best_score'] if isinstance(row['proposed_best_score'], float) else None)} {_person(row['proposed_best_cluster_id'], cluster_to_person)}"
            )
    print(f"Output:                        {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
