#!/usr/bin/env python3
"""Read-only all-member bridge diagnostic between two current person clusters.

This tool is designed for cases where two folders are known (by visual ground
truth) to contain the same identity, but the production exemplar sets do not
provide enough evidence to merge them.

It compares every member of SOURCE with every member of TARGET using the
embeddings persisted by the completed gallery run.  It also compares each
SOURCE face against TARGET's *current exemplar set* so that an exemplar
representation blind spot can be measured directly.

Typical usage::

    python tools/gallery_grouping/diagnose_member_bridge.py \
        --source person_016 --target person_002

Outputs (default under data/gallery_grouping_output/member_bridge_diagnostics):
  * diagnostic_summary.json
  * source_face_summary.csv
  * source_top10_member_matches.csv
  * all_member_similarities.csv
  * bridge_target_faces.csv

The diagnostic is strictly read-only: it works on a snapshot of the completed
run DB and never reruns detection, landmarking, embedding, assignment,
consolidation, merge/split, or correction logic.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import defaultdict
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
        description=(
            "Compare every source-cluster member against every target-cluster member "
            "and quantify evidence hidden outside the target exemplar set."
        )
    )
    p.add_argument("--output", default="data/gallery_grouping_output")
    p.add_argument("--source", required=True, help="Current person folder, e.g. person_016")
    p.add_argument("--target", required=True, help="Current person folder, e.g. person_002")
    p.add_argument("--top-n", type=int, default=10, help="Top target members to keep per source face")
    p.add_argument("--diagnostic-dir", default="")
    return p.parse_args()


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._") or "item"


def _load_person_map(clusters_csv: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with clusters_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            person = str(row.get("person_folder", "")).strip()
            cluster = str(row.get("cluster_id", "")).strip()
            if person and cluster:
                out[person] = cluster
    return out


def _photo_name(store: FaceGroupingStore, face) -> str:
    if not face.photo_id:
        return ""
    photo = store.load_photo(face.photo_id)
    return Path(photo.image_path).name if photo else ""


def _cluster_exemplars(cluster):
    rows = []
    for bucket_name, bucket in (
        ("quality", cluster.exemplar_set.quality_bucket),
        ("pose", cluster.exemplar_set.pose_bucket),
    ):
        for ex in bucket:
            rows.append((bucket_name, ex))
    return rows


def _top_k_average(values: List[float], k: int) -> float:
    if not values:
        return 0.0
    ranked = sorted(values, reverse=True)
    chosen = ranked[: min(k, len(ranked))]
    return float(sum(chosen) / len(chosen))


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be >= 1")
    if args.source == args.target:
        raise ValueError("--source and --target must be different")

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

    p2c = _load_person_map(clusters_csv)
    for person in (args.source, args.target):
        if person not in p2c:
            raise ValueError(f"Unknown current person folder: {person}")

    diag_dir = (
        Path(args.diagnostic_dir).resolve()
        if args.diagnostic_dir
        else output_dir
        / "member_bridge_diagnostics"
        / f"{_safe_name(args.source)}__vs__{_safe_name(args.target)}"
    )
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the completed run because opening the store may perform schema
    # housekeeping.  The original gallery run remains untouched.
    snapshot = diag_dir / "_diagnostic_snapshot.db"
    shutil.copy2(db_path, snapshot)
    store = FaceGroupingStore(str(snapshot))

    try:
        source_cluster = store.load_cluster(p2c[args.source])
        target_cluster = store.load_cluster(p2c[args.target])
        if source_cluster is None or target_cluster is None:
            raise RuntimeError("One selected cluster could not be loaded")

        source_faces = store.load_faces_by_cluster(source_cluster.cluster_id)
        target_faces = store.load_faces_by_cluster(target_cluster.cluster_id)
        target_exemplars = _cluster_exemplars(target_cluster)
        if not source_faces or not target_faces:
            raise RuntimeError("Selected clusters must both contain member faces")

        cfg = load_thresholds()
        t_match = float(cfg["matching"]["t_match"])
        top_k = int(cfg["matching"]["top_k"])
        sparse_margin = float(cfg["matching"]["sparse_cluster_margin"])
        exemplar_margin = float(cfg["matching"]["exemplar_admission_margin"])
        high_conf_floor = float(t_match + sparse_margin)
        strong_anchor_floor = float(t_match + exemplar_margin)

        exemplar_face_to_bucket = {
            ex.face_id: bucket
            for bucket, ex in target_exemplars
            if ex.face_id
        }

        all_rows: List[dict] = []
        top_rows: List[dict] = []
        source_summary_rows: List[dict] = []
        target_bridge_stats = defaultdict(
            lambda: {
                "top10_hits": 0,
                "top1_hits": 0,
                "source_faces_ge_t_match": 0,
                "source_faces_ge_high_conf": 0,
                "source_faces_ge_strong_anchor": 0,
                "max_similarity": -1.0,
            }
        )

        for source_face in source_faces:
            source_photo = _photo_name(store, source_face)
            member_scores = []

            for target_face in target_faces:
                target_photo = _photo_name(store, target_face)
                sim = float(cosine_similarity(source_face.embedding, target_face.embedding))
                row = {
                    "source_person": args.source,
                    "source_face_id": source_face.face_id,
                    "source_photo": source_photo,
                    "source_face_index": "" if source_face.face_index is None else int(source_face.face_index),
                    "source_quality_score": float(source_face.quality_score),
                    "source_yaw_ratio": float(source_face.yaw_ratio),
                    "target_person": args.target,
                    "target_face_id": target_face.face_id,
                    "target_photo": target_photo,
                    "target_face_index": "" if target_face.face_index is None else int(target_face.face_index),
                    "target_quality_score": float(target_face.quality_score),
                    "target_yaw_ratio": float(target_face.yaw_ratio),
                    "target_is_current_exemplar": target_face.face_id in exemplar_face_to_bucket,
                    "target_exemplar_bucket": exemplar_face_to_bucket.get(target_face.face_id, ""),
                    "similarity": sim,
                    "ge_t_match": sim >= t_match,
                    "ge_high_conf_floor": sim >= high_conf_floor,
                    "ge_strong_anchor_floor": sim >= strong_anchor_floor,
                }
                all_rows.append(row)
                member_scores.append(row)

            member_scores.sort(key=lambda r: r["similarity"], reverse=True)
            top_n_rows = member_scores[: min(args.top_n, len(member_scores))]
            for rank, row in enumerate(top_n_rows, start=1):
                ranked = dict(row)
                ranked["member_rank_for_source_face"] = rank
                top_rows.append(ranked)

                st = target_bridge_stats[row["target_face_id"]]
                st["top10_hits"] += 1
                if rank == 1:
                    st["top1_hits"] += 1
                if row["similarity"] >= t_match:
                    st["source_faces_ge_t_match"] += 1
                if row["similarity"] >= high_conf_floor:
                    st["source_faces_ge_high_conf"] += 1
                if row["similarity"] >= strong_anchor_floor:
                    st["source_faces_ge_strong_anchor"] += 1
                st["max_similarity"] = max(st["max_similarity"], row["similarity"])

            # Production-style evidence against the *current target exemplars*.
            exemplar_scores = [
                float(cosine_similarity(source_face.embedding, ex.embedding))
                for _, ex in target_exemplars
            ]
            exemplar_topk = _top_k_average(exemplar_scores, top_k)
            best_exemplar = max(exemplar_scores) if exemplar_scores else 0.0

            all_member_values = [r["similarity"] for r in member_scores]
            best_member = member_scores[0]
            best_non_exemplar = next(
                (r for r in member_scores if not r["target_is_current_exemplar"]),
                None,
            )

            source_summary_rows.append(
                {
                    "source_person": args.source,
                    "source_face_id": source_face.face_id,
                    "source_photo": source_photo,
                    "source_face_index": "" if source_face.face_index is None else int(source_face.face_index),
                    "source_quality_score": float(source_face.quality_score),
                    "source_yaw_ratio": float(source_face.yaw_ratio),
                    "target_person": args.target,
                    "target_member_count": len(target_faces),
                    "target_current_exemplar_count": len(target_exemplars),
                    "current_exemplar_topk_average": exemplar_topk,
                    "current_exemplar_max_similarity": best_exemplar,
                    "current_exemplar_topk_passes_t_match": exemplar_topk >= t_match,
                    "best_any_member_similarity": float(best_member["similarity"]),
                    "best_any_member_photo": best_member["target_photo"],
                    "best_any_member_face_index": best_member["target_face_index"],
                    "best_any_member_is_exemplar": best_member["target_is_current_exemplar"],
                    "best_non_exemplar_similarity": "" if best_non_exemplar is None else float(best_non_exemplar["similarity"]),
                    "best_non_exemplar_photo": "" if best_non_exemplar is None else best_non_exemplar["target_photo"],
                    "member_gain_over_best_exemplar": float(best_member["similarity"] - best_exemplar),
                    "member_gain_over_exemplar_topk": float(best_member["similarity"] - exemplar_topk),
                    "top5_member_average": _top_k_average(all_member_values, 5),
                    "top10_member_average": _top_k_average(all_member_values, 10),
                    "target_members_ge_t_match": sum(v >= t_match for v in all_member_values),
                    "target_members_ge_high_conf_floor": sum(v >= high_conf_floor for v in all_member_values),
                    "target_members_ge_strong_anchor_floor": sum(v >= strong_anchor_floor for v in all_member_values),
                }
            )

        # Summarize target members that repeatedly serve as bridges for the
        # source appearance mode.  This is useful for identifying evidence that
        # the current exemplar policy discarded.
        target_by_id = {face.face_id: face for face in target_faces}
        bridge_rows: List[dict] = []
        for target_face_id, st in target_bridge_stats.items():
            face = target_by_id[target_face_id]
            bridge_rows.append(
                {
                    "target_person": args.target,
                    "target_face_id": target_face_id,
                    "target_photo": _photo_name(store, face),
                    "target_face_index": "" if face.face_index is None else int(face.face_index),
                    "target_quality_score": float(face.quality_score),
                    "target_yaw_ratio": float(face.yaw_ratio),
                    "is_current_exemplar": target_face_id in exemplar_face_to_bucket,
                    "exemplar_bucket": exemplar_face_to_bucket.get(target_face_id, ""),
                    "top1_hits_from_source_faces": st["top1_hits"],
                    "top10_hits_from_source_faces": st["top10_hits"],
                    "source_face_hits_ge_t_match": st["source_faces_ge_t_match"],
                    "source_face_hits_ge_high_conf": st["source_faces_ge_high_conf"],
                    "source_face_hits_ge_strong_anchor": st["source_faces_ge_strong_anchor"],
                    "max_similarity_from_any_source_face": st["max_similarity"],
                }
            )
        bridge_rows.sort(
            key=lambda r: (
                r["source_face_hits_ge_strong_anchor"],
                r["source_face_hits_ge_high_conf"],
                r["top1_hits_from_source_faces"],
                r["max_similarity_from_any_source_face"],
            ),
            reverse=True,
        )

        _write_csv(diag_dir / "all_member_similarities.csv", all_rows)
        _write_csv(diag_dir / "source_top10_member_matches.csv", top_rows)
        _write_csv(diag_dir / "source_face_summary.csv", source_summary_rows)
        _write_csv(diag_dir / "bridge_target_faces.csv", bridge_rows)

        source_faces_with_any_tmatch = sum(
            int(float(r["best_any_member_similarity"]) >= t_match)
            for r in source_summary_rows
        )
        source_faces_with_any_high = sum(
            int(float(r["best_any_member_similarity"]) >= high_conf_floor)
            for r in source_summary_rows
        )
        source_faces_with_any_strong = sum(
            int(float(r["best_any_member_similarity"]) >= strong_anchor_floor)
            for r in source_summary_rows
        )
        source_faces_exemplar_topk_pass = sum(
            int(bool(r["current_exemplar_topk_passes_t_match"]))
            for r in source_summary_rows
        )
        source_faces_best_member_not_exemplar = sum(
            int(not bool(r["best_any_member_is_exemplar"]))
            for r in source_summary_rows
        )

        report = {
            "test": "all_member_bridge_diagnostic_v1",
            "run_output": str(output_dir),
            "source_person": args.source,
            "source_face_count": len(source_faces),
            "target_person": args.target,
            "target_face_count": len(target_faces),
            "target_current_exemplar_count": len(target_exemplars),
            "pairwise_member_comparisons": len(source_faces) * len(target_faces),
            "top_n_saved_per_source": args.top_n,
            "reference_thresholds": {
                "t_match": t_match,
                "high_conf_floor_t_match_plus_sparse_margin": high_conf_floor,
                "strong_anchor_floor_t_match_plus_exemplar_admission_margin": strong_anchor_floor,
                "note": "These floors are reported diagnostically for member-to-member evidence; production cluster decisions still use exemplar top-k scoring.",
            },
            "aggregate": {
                "source_faces_whose_current_target_exemplar_topk_passes_t_match": source_faces_exemplar_topk_pass,
                "source_faces_with_any_target_member_ge_t_match": source_faces_with_any_tmatch,
                "source_faces_with_any_target_member_ge_high_conf_floor": source_faces_with_any_high,
                "source_faces_with_any_target_member_ge_strong_anchor_floor": source_faces_with_any_strong,
                "source_faces_whose_best_target_member_is_not_a_current_exemplar": source_faces_best_member_not_exemplar,
            },
            "source_faces": [
                {
                    "photo": r["source_photo"],
                    "face_index": r["source_face_index"],
                    "current_exemplar_topk_average": r["current_exemplar_topk_average"],
                    "current_exemplar_max_similarity": r["current_exemplar_max_similarity"],
                    "best_any_member_similarity": r["best_any_member_similarity"],
                    "best_any_member_photo": r["best_any_member_photo"],
                    "best_any_member_is_exemplar": r["best_any_member_is_exemplar"],
                    "best_non_exemplar_similarity": r["best_non_exemplar_similarity"],
                    "best_non_exemplar_photo": r["best_non_exemplar_photo"],
                    "target_members_ge_t_match": r["target_members_ge_t_match"],
                    "target_members_ge_high_conf_floor": r["target_members_ge_high_conf_floor"],
                    "target_members_ge_strong_anchor_floor": r["target_members_ge_strong_anchor_floor"],
                }
                for r in source_summary_rows
            ],
            "notes": [
                "All  source-to-target member similarities use embeddings persisted by the completed production run.",
                "Current exemplar evidence is computed from the target cluster's final stored exemplar set.",
                "A strong member-to-member bridge outside the exemplar set is diagnostic evidence of a representation blind spot; it is not by itself a production merge rule.",
                "No detector, landmarker, IR-SE50 inference, assignment, consolidation, merge/split, or correction is rerun.",
                "This diagnostic does not modify clustering state or thresholds.",
            ],
        }
        with (diag_dir / "diagnostic_summary.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("=== ALL-MEMBER BRIDGE DIAGNOSTIC ===")
        print(f"Run:                   {output_dir}")
        print(f"Pair:                  {args.source} -> {args.target}")
        print(f"Source faces:          {len(source_faces)}")
        print(f"Target faces:          {len(target_faces)}")
        print(f"Pairwise comparisons:  {len(source_faces) * len(target_faces)}")
        print(f"Target exemplars:      {len(target_exemplars)}")
        print()
        print(
            "Exemplar top-k pass:    "
            f"{source_faces_exemplar_topk_pass}/{len(source_faces)} source faces"
        )
        print(
            "Any member >= T_match:  "
            f"{source_faces_with_any_tmatch}/{len(source_faces)} source faces"
        )
        print(
            "Any member >= 0.46:     "
            f"{source_faces_with_any_high}/{len(source_faces)} source faces"
        )
        print(
            "Any member >= 0.51:     "
            f"{source_faces_with_any_strong}/{len(source_faces)} source faces"
        )
        print(
            "Best is non-exemplar:   "
            f"{source_faces_best_member_not_exemplar}/{len(source_faces)} source faces"
        )
        print()
        print(f"Output:                 {diag_dir}")
        print(f"Summary:                {diag_dir / 'diagnostic_summary.json'}")
        print(f"Source face summary:    {diag_dir / 'source_face_summary.csv'}")
        print(f"Top member matches:     {diag_dir / 'source_top10_member_matches.csv'}")
        return 0
    finally:
        store.close()
        try:
            snapshot.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
