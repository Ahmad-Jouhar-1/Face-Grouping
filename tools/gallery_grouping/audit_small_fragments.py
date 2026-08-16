#!/usr/bin/env python3
"""Audit small confirmed clusters for asymmetric merge evidence.

This is a READ-ONLY post-run diagnostic for ``data/gallery_grouping_output``.
It does not modify clustering state, thresholds, suggestions, or the database.

The goal is to find cases where a tiny fragment is consistently explained by
one mature cluster even though the current *mutual* 90% merge rule fails in
the reverse direction because the tiny fragment has too few exemplars.

Default probe (diagnostic only, NOT production behavior):
  * source cluster has 2-3 confirmed/manual faces;
  * target cluster has >= 4 faces and more faces than source;
  * target has at least ``top_k`` exemplars;
  * no permanent cannot-link and no same-photo conflict;
  * 100% of source faces pass the CURRENT high-confidence merge threshold
    against the target;
  * at least one source face is a "strong anchor" whose target score clears
    ``t_match + exemplar_admission_margin``.

Outputs:
  * small_fragment_pairs.csv       -- every evaluated small->large pair
  * asymmetric_candidates.csv      -- only pairs passing the diagnostic probe
  * candidate_face_scores.csv      -- per-face evidence for those candidates
  * diagnostic_summary.json        -- compact aggregate summary

Typical usage:

    python tools/gallery_grouping/audit_small_fragments.py

The defaults intentionally preserve the product policy that singleton evidence
is too weak for merge suggestions. Use ``--small-min 1`` only for exploration.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from face_grouping.config import load_thresholds
from face_grouping.matching.similarity import cosine_similarity
from face_grouping.storage.store import FaceGroupingStore

DB_FILENAME = "gallery_grouping.db"
CLUSTERS_FILENAME = "clusters.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit all small clusters for conservative asymmetric small->large merge evidence."
    )
    p.add_argument("--output", default="data/gallery_grouping_output")
    p.add_argument("--diagnostic-dir", default="")
    p.add_argument("--small-min", type=int, default=2)
    p.add_argument("--small-max", type=int, default=3)
    p.add_argument("--large-min", type=int, default=4)
    p.add_argument("--required-small-coverage", type=float, default=1.0)
    p.add_argument("--required-anchor-count", type=int, default=1)
    return p.parse_args()


def _load_person_map(clusters_csv: Path) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, dict]]:
    p2c: Dict[str, str] = {}
    c2p: Dict[str, str] = {}
    rows: Dict[str, dict] = {}
    with clusters_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            person = str(row.get("person_folder", "")).strip()
            cluster = str(row.get("cluster_id", "")).strip()
            if not person or not cluster:
                continue
            p2c[person] = cluster
            c2p[cluster] = person
            rows[person] = row
    return p2c, c2p, rows


def _cluster_exemplars(cluster):
    out = []
    for bucket_name, bucket in (
        ("quality", cluster.exemplar_set.quality_bucket),
        ("pose", cluster.exemplar_set.pose_bucket),
    ):
        for ex in bucket:
            out.append((bucket_name, ex))
    return out


def _top_k_average(values: List[float], k: int) -> float:
    if not values:
        raise ValueError("Target cluster has no exemplars")
    ranked = sorted(values, reverse=True)
    chosen = ranked[: min(k, len(ranked))]
    return float(sum(chosen) / len(chosen))


def _effective_threshold(*, exemplar_count: int, top_k: int, t_match: float, sparse_margin: float) -> float:
    return float(t_match + sparse_margin) if exemplar_count < top_k else float(t_match)


def _photo_name(store: FaceGroupingStore, face) -> str:
    if not face.photo_id:
        return ""
    photo = store.load_photo(face.photo_id)
    return Path(photo.image_path).name if photo else ""


def _score_direction(
    *,
    store: FaceGroupingStore,
    source_person: str,
    source_cluster,
    target_person: str,
    target_cluster,
    top_k: int,
    t_match: float,
    sparse_margin: float,
    min_cluster_margin: float,
    exemplar_admission_margin: float,
) -> Tuple[dict, List[dict]]:
    source_faces = store.load_faces_by_cluster(source_cluster.cluster_id)
    target_exemplars = _cluster_exemplars(target_cluster)

    normal_threshold = _effective_threshold(
        exemplar_count=len(target_exemplars),
        top_k=top_k,
        t_match=t_match,
        sparse_margin=sparse_margin,
    )
    # Mirrors ConsolidationEngine._coverage(..., high_confidence=True).
    merge_threshold = max(normal_threshold, t_match + min_cluster_margin)
    # Re-use an existing configured stronger evidence level rather than inventing
    # a new numeric constant. This is ONLY a diagnostic anchor definition.
    anchor_threshold = max(merge_threshold, t_match + exemplar_admission_margin)

    pass_normal = 0
    pass_merge = 0
    strong_anchors = 0
    face_rows: List[dict] = []

    for face in source_faces:
        per_exemplar = []
        for bucket, ex in target_exemplars:
            sim = float(cosine_similarity(face.embedding, ex.embedding))
            per_exemplar.append((sim, bucket, ex))
        per_exemplar.sort(key=lambda item: item[0], reverse=True)
        sims = [item[0] for item in per_exemplar]
        score = _top_k_average(sims, top_k)
        max_sim = float(per_exemplar[0][0]) if per_exemplar else float("nan")

        normal_ok = score >= normal_threshold
        merge_ok = score >= merge_threshold
        anchor_ok = score >= anchor_threshold
        pass_normal += int(normal_ok)
        pass_merge += int(merge_ok)
        strong_anchors += int(anchor_ok)

        best_ex_photo = ""
        best_ex_face_id = ""
        best_ex_bucket = ""
        if per_exemplar:
            _, best_ex_bucket, best_ex = per_exemplar[0]
            best_ex_face_id = best_ex.face_id or ""
            if best_ex.face_id:
                ex_face = store.load_face(best_ex.face_id)
                if ex_face is not None:
                    best_ex_photo = _photo_name(store, ex_face)

        face_rows.append(
            {
                "direction": f"{source_person}->{target_person}",
                "source_person": source_person,
                "source_cluster_id": source_cluster.cluster_id,
                "source_face_id": face.face_id,
                "source_photo": _photo_name(store, face),
                "source_face_index": "" if face.face_index is None else int(face.face_index),
                "source_quality_score": float(face.quality_score),
                "target_person": target_person,
                "target_cluster_id": target_cluster.cluster_id,
                "target_exemplar_count": len(target_exemplars),
                "top_k_average_similarity": score,
                "max_similarity": max_sim,
                "normal_threshold": normal_threshold,
                "merge_high_conf_threshold": merge_threshold,
                "strong_anchor_threshold": anchor_threshold,
                "passes_normal_threshold": normal_ok,
                "passes_merge_high_conf_threshold": merge_ok,
                "is_strong_anchor": anchor_ok,
                "best_target_exemplar_bucket": best_ex_bucket,
                "best_target_exemplar_face_id": best_ex_face_id,
                "best_target_exemplar_photo": best_ex_photo,
            }
        )

    total = len(source_faces)
    summary = {
        "source_face_count": total,
        "target_face_count": len(store.load_faces_by_cluster(target_cluster.cluster_id)),
        "target_exemplar_count": len(target_exemplars),
        "normal_threshold": normal_threshold,
        "merge_high_conf_threshold": merge_threshold,
        "strong_anchor_threshold": anchor_threshold,
        "normal_passing_faces": pass_normal,
        "normal_coverage": (pass_normal / total) if total else 0.0,
        "merge_passing_faces": pass_merge,
        "merge_high_conf_coverage": (pass_merge / total) if total else 0.0,
        "strong_anchor_count": strong_anchors,
    }
    return summary, face_rows


def _versions(store: FaceGroupingStore, cluster_id: str) -> set[str]:
    return {
        str(face.embedding_model_version)
        for face in store.load_faces_by_cluster(cluster_id)
        if face.embedding_model_version
    }


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames:
            with path.open("w", newline="", encoding="utf-8-sig") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        else:
            path.write_text("", encoding="utf-8-sig")
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.small_min < 1 or args.small_max < args.small_min:
        raise ValueError("Require 1 <= --small-min <= --small-max")
    if args.large_min < 1:
        raise ValueError("--large-min must be >= 1")
    if not (0.0 <= args.required_small_coverage <= 1.0):
        raise ValueError("--required-small-coverage must be in [0,1]")
    if args.required_anchor_count < 0:
        raise ValueError("--required-anchor-count must be >= 0")

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()

    db_path = output_dir / DB_FILENAME
    clusters_csv = output_dir / CLUSTERS_FILENAME
    if not db_path.exists():
        raise FileNotFoundError(f"Run DB not found: {db_path}")
    if not clusters_csv.exists():
        raise FileNotFoundError(f"Cluster map not found: {clusters_csv}")

    diag_dir = (
        Path(args.diagnostic_dir).resolve()
        if args.diagnostic_dir
        else output_dir / "small_fragment_audit"
    )
    diag_dir.mkdir(parents=True, exist_ok=True)

    p2c, _, csv_rows = _load_person_map(clusters_csv)

    snapshot = diag_dir / "_diagnostic_snapshot.db"
    shutil.copy2(db_path, snapshot)
    store = FaceGroupingStore(str(snapshot))
    try:
        cfg = load_thresholds()
        mcfg = cfg["matching"]
        top_k = int(mcfg["top_k"])
        t_match = float(mcfg["t_match"])
        sparse_margin = float(mcfg["sparse_cluster_margin"])
        min_cluster_margin = float(mcfg["min_cluster_margin"])
        exemplar_admission_margin = float(mcfg["exemplar_admission_margin"])

        loaded = {}
        sizes = {}
        exemplar_counts = {}
        for person, cluster_id in p2c.items():
            cluster = store.load_cluster(cluster_id)
            if cluster is None:
                continue
            loaded[person] = cluster
            sizes[person] = len(store.load_faces_by_cluster(cluster_id))
            exemplar_counts[person] = len(_cluster_exemplars(cluster))

        small_people = sorted(
            [p for p, n in sizes.items() if args.small_min <= n <= args.small_max],
            key=lambda p: (sizes[p], p),
        )
        target_people = sorted(
            [p for p, n in sizes.items() if n >= args.large_min],
            key=lambda p: (-sizes[p], p),
        )

        pair_rows: List[dict] = []
        candidate_rows: List[dict] = []
        candidate_face_rows: List[dict] = []

        for source_person in small_people:
            source = loaded[source_person]
            for target_person in target_people:
                if target_person == source_person:
                    continue
                if sizes[target_person] <= sizes[source_person]:
                    continue
                target = loaded[target_person]

                cannot_link = store.has_cannot_link(source.cluster_id, target.cluster_id)
                same_photo_conflict = store.clusters_share_photo_conflict(
                    source.cluster_id, target.cluster_id
                )
                model_compatible = len(_versions(store, source.cluster_id) | _versions(store, target.cluster_id)) <= 1

                # Score both directions for diagnosis, even if a safety blocker
                # exists, as long as embeddings are model-compatible.
                if model_compatible and exemplar_counts[target_person] > 0 and exemplar_counts[source_person] > 0:
                    forward, forward_faces = _score_direction(
                        store=store,
                        source_person=source_person,
                        source_cluster=source,
                        target_person=target_person,
                        target_cluster=target,
                        top_k=top_k,
                        t_match=t_match,
                        sparse_margin=sparse_margin,
                        min_cluster_margin=min_cluster_margin,
                        exemplar_admission_margin=exemplar_admission_margin,
                    )
                    reverse, _ = _score_direction(
                        store=store,
                        source_person=target_person,
                        source_cluster=target,
                        target_person=source_person,
                        target_cluster=source,
                        top_k=top_k,
                        t_match=t_match,
                        sparse_margin=sparse_margin,
                        min_cluster_margin=min_cluster_margin,
                        exemplar_admission_margin=exemplar_admission_margin,
                    )
                else:
                    forward = {
                        "source_face_count": sizes[source_person],
                        "target_face_count": sizes[target_person],
                        "target_exemplar_count": exemplar_counts[target_person],
                        "normal_threshold": "",
                        "merge_high_conf_threshold": "",
                        "strong_anchor_threshold": "",
                        "normal_passing_faces": 0,
                        "normal_coverage": 0.0,
                        "merge_passing_faces": 0,
                        "merge_high_conf_coverage": 0.0,
                        "strong_anchor_count": 0,
                    }
                    reverse = {
                        "merge_high_conf_coverage": 0.0,
                        "merge_passing_faces": 0,
                        "source_face_count": sizes[target_person],
                    }
                    forward_faces = []

                current_mutual_merge = bool(
                    not cannot_link
                    and not same_photo_conflict
                    and model_compatible
                    and sizes[source_person] >= 2
                    and sizes[target_person] >= 2
                    and forward["merge_high_conf_coverage"] >= 0.90
                    and reverse["merge_high_conf_coverage"] >= 0.90
                )

                mature_target = bool(exemplar_counts[target_person] >= top_k)
                asymmetric_probe = bool(
                    not cannot_link
                    and not same_photo_conflict
                    and model_compatible
                    and mature_target
                    and forward["merge_high_conf_coverage"] >= args.required_small_coverage
                    and forward["strong_anchor_count"] >= args.required_anchor_count
                )

                row = {
                    "source_person": source_person,
                    "source_cluster_id": source.cluster_id,
                    "source_face_count": sizes[source_person],
                    "source_exemplar_count": exemplar_counts[source_person],
                    "target_person": target_person,
                    "target_cluster_id": target.cluster_id,
                    "target_face_count": sizes[target_person],
                    "target_exemplar_count": exemplar_counts[target_person],
                    "cannot_link": cannot_link,
                    "same_photo_conflict": same_photo_conflict,
                    "model_compatible": model_compatible,
                    "mature_target": mature_target,
                    "small_to_large_normal_coverage": forward["normal_coverage"],
                    "small_to_large_merge_high_conf_coverage": forward["merge_high_conf_coverage"],
                    "small_to_large_passing_faces": forward["merge_passing_faces"],
                    "small_to_large_strong_anchor_count": forward["strong_anchor_count"],
                    "large_to_small_merge_high_conf_coverage": reverse["merge_high_conf_coverage"],
                    "large_to_small_passing_faces": reverse["merge_passing_faces"],
                    "normal_target_threshold": forward["normal_threshold"],
                    "merge_high_conf_threshold": forward["merge_high_conf_threshold"],
                    "strong_anchor_threshold": forward["strong_anchor_threshold"],
                    "current_mutual_90_merge": current_mutual_merge,
                    "asymmetric_probe_candidate": asymmetric_probe,
                }
                pair_rows.append(row)
                if asymmetric_probe:
                    candidate_rows.append(row)
                    candidate_face_rows.extend(forward_faces)

        # Highest small->large evidence first; ties prefer larger targets.
        pair_rows.sort(
            key=lambda r: (
                -float(r["small_to_large_merge_high_conf_coverage"]),
                -int(r["small_to_large_strong_anchor_count"]),
                -int(r["target_face_count"]),
                str(r["source_person"]),
                str(r["target_person"]),
            )
        )
        candidate_rows.sort(
            key=lambda r: (
                -float(r["small_to_large_merge_high_conf_coverage"]),
                -int(r["small_to_large_strong_anchor_count"]),
                -int(r["target_face_count"]),
                str(r["source_person"]),
                str(r["target_person"]),
            )
        )

        _write_csv(diag_dir / "small_fragment_pairs.csv", pair_rows)
        _write_csv(diag_dir / "asymmetric_candidates.csv", candidate_rows)
        _write_csv(diag_dir / "candidate_face_scores.csv", candidate_face_rows)

        report = {
            "test": "small_fragment_asymmetric_merge_audit_v1",
            "run_output": str(output_dir),
            "parameters": {
                "small_min_faces": args.small_min,
                "small_max_faces": args.small_max,
                "large_min_faces": args.large_min,
                "required_small_to_large_high_conf_coverage": args.required_small_coverage,
                "required_strong_anchor_count": args.required_anchor_count,
                "mature_target_requires_exemplars_at_least_top_k": True,
            },
            "matching": {
                "t_match": t_match,
                "top_k": top_k,
                "sparse_cluster_margin": sparse_margin,
                "min_cluster_margin": min_cluster_margin,
                "exemplar_admission_margin": exemplar_admission_margin,
                "current_merge_required_directional_coverage": 0.90,
                "merge_high_conf_floor": t_match + min_cluster_margin,
                "diagnostic_strong_anchor_floor": t_match + exemplar_admission_margin,
            },
            "counts": {
                "active_clusters": len(loaded),
                "small_clusters": len(small_people),
                "eligible_large_clusters": len(target_people),
                "evaluated_ordered_pairs": len(pair_rows),
                "current_mutual_merge_pairs_with_small_source": sum(
                    1 for r in pair_rows if r["current_mutual_90_merge"]
                ),
                "asymmetric_probe_candidates": len(candidate_rows),
            },
            "small_clusters": [
                {
                    "person": p,
                    "face_count": sizes[p],
                    "exemplar_count": exemplar_counts[p],
                }
                for p in small_people
            ],
            "asymmetric_candidates": [
                {
                    "source_person": r["source_person"],
                    "source_face_count": r["source_face_count"],
                    "target_person": r["target_person"],
                    "target_face_count": r["target_face_count"],
                    "small_to_large_high_conf_coverage": r["small_to_large_merge_high_conf_coverage"],
                    "strong_anchor_count": r["small_to_large_strong_anchor_count"],
                    "large_to_small_high_conf_coverage": r["large_to_small_merge_high_conf_coverage"],
                    "current_mutual_90_merge": r["current_mutual_90_merge"],
                }
                for r in candidate_rows
            ],
            "notes": [
                "This audit is read-only and does not create or resolve merge suggestions.",
                "High-confidence coverage mirrors ConsolidationEngine._coverage(..., high_confidence=True).",
                "The asymmetric probe intentionally ignores reverse coverage only for diagnosis; all existing safety blockers remain required.",
                "The strong-anchor floor reuses t_match + exemplar_admission_margin; it is not a new production threshold.",
                "Singletons are excluded by default to preserve the current product policy against weak singleton merge evidence.",
            ],
        }
        with (diag_dir / "diagnostic_summary.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("=== SMALL-FRAGMENT MERGE AUDIT ===")
        print(f"Run:                    {output_dir}")
        print(f"Active clusters:        {len(loaded)}")
        print(f"Small clusters ({args.small_min}-{args.small_max} faces): {len(small_people)}")
        print(f"Evaluated pairs:        {len(pair_rows)}")
        print(f"Asymmetric candidates:  {len(candidate_rows)}")
        print()
        if candidate_rows:
            print("Top asymmetric probe candidates:")
            for row in candidate_rows[:20]:
                print(
                    f"  {row['source_person']} ({row['source_face_count']}) -> "
                    f"{row['target_person']} ({row['target_face_count']}): "
                    f"small->large={float(row['small_to_large_merge_high_conf_coverage']):.1%}, "
                    f"anchors={row['small_to_large_strong_anchor_count']}, "
                    f"large->small={float(row['large_to_small_merge_high_conf_coverage']):.1%}, "
                    f"current_mutual={row['current_mutual_90_merge']}"
                )
        else:
            print("No pair passed the asymmetric diagnostic probe.")
        print()
        print(f"Output:                 {diag_dir}")
        print(f"Summary:                {diag_dir / 'diagnostic_summary.json'}")
        print(f"All pairs:              {diag_dir / 'small_fragment_pairs.csv'}")
        print(f"Candidates:             {diag_dir / 'asymmetric_candidates.csv'}")
        print(f"Candidate face scores:  {diag_dir / 'candidate_face_scores.csv'}")
        return 0
    finally:
        store.close()
        try:
            snapshot.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
