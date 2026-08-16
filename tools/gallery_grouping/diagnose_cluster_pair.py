#!/usr/bin/env python3
"""Post-run diagnostic for a pair of exported person clusters.

Reads an existing ``data/gallery_grouping_output`` run and explains the
relationship between two current person folders using the *stored* production
embeddings and final exemplar sets.  It does not rerun detection, landmarking,
IR-SE50, clustering, or modify thresholds/state.

For each direction (A -> B and B -> A) it writes:
  * every member face's similarity to every target exemplar;
  * the production top-k average cluster score;
  * the effective target threshold (including sparse-cluster margin);
  * whether that member passes the target cluster threshold;
  * aggregate directional coverage.

Typical usage:

    python tools/gallery_grouping/diagnose_cluster_pair.py \
        --source person_020 --target person_007

The default run directory is ``data/gallery_grouping_output``.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
        description="Explain why two current person clusters do or do not satisfy merge evidence."
    )
    p.add_argument("--output", default="data/gallery_grouping_output")
    p.add_argument("--source", required=True, help="Current exported person folder, e.g. person_020")
    p.add_argument("--target", required=True, help="Current exported person folder, e.g. person_007")
    p.add_argument("--diagnostic-dir", default="")
    return p.parse_args()


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._") or "item"


def _load_person_map(clusters_csv: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    p2c: Dict[str, str] = {}
    c2p: Dict[str, str] = {}
    with clusters_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            person = str(row.get("person_folder", "")).strip()
            cluster = str(row.get("cluster_id", "")).strip()
            if person and cluster:
                p2c[person] = cluster
                c2p[cluster] = person
    return p2c, c2p


def _top_k_average(values: List[float], k: int) -> float:
    if not values:
        raise ValueError("Target cluster has no exemplars")
    ranked = sorted(values, reverse=True)
    chosen = ranked[: min(k, len(ranked))]
    return float(sum(chosen) / len(chosen))


def _effective_threshold(*, exemplar_count: int, top_k: int, t_match: float, sparse_margin: float) -> float:
    return float(t_match + sparse_margin) if exemplar_count < top_k else float(t_match)


def _face_source(store: FaceGroupingStore, face):
    photo = store.load_photo(face.photo_id) if face.photo_id else None
    return photo


def _exemplar_source(store: FaceGroupingStore, exemplar):
    if not exemplar.face_id:
        return None, None
    face = store.load_face(exemplar.face_id)
    if face is None:
        return None, None
    photo = store.load_photo(face.photo_id) if face.photo_id else None
    return face, photo


def _cluster_exemplars(cluster):
    out = []
    for bucket_name, bucket in (
        ("quality", cluster.exemplar_set.quality_bucket),
        ("pose", cluster.exemplar_set.pose_bucket),
    ):
        for ex in bucket:
            out.append((bucket_name, ex))
    return out


def _direction(
    *,
    store: FaceGroupingStore,
    source_person: str,
    source_cluster,
    target_person: str,
    target_cluster,
    top_k: int,
    t_match: float,
    sparse_margin: float,
) -> Tuple[dict, List[dict], List[dict]]:
    source_faces = store.load_faces_by_cluster(source_cluster.cluster_id)
    target_exemplars = _cluster_exemplars(target_cluster)
    threshold = _effective_threshold(
        exemplar_count=len(target_exemplars),
        top_k=top_k,
        t_match=t_match,
        sparse_margin=sparse_margin,
    )

    face_rows: List[dict] = []
    exemplar_rows: List[dict] = []
    pass_count = 0

    for face in source_faces:
        source_photo = _face_source(store, face)
        per_exemplar = []
        for bucket, ex in target_exemplars:
            sim = float(cosine_similarity(face.embedding, ex.embedding))
            ex_face, ex_photo = _exemplar_source(store, ex)
            entry = {
                "bucket": bucket,
                "exemplar_face_id": ex.face_id or "",
                "exemplar_quality_score": float(ex.quality_score),
                "exemplar_yaw_ratio": float(ex.yaw_ratio),
                "similarity": sim,
                "exemplar_photo": Path(ex_photo.image_path).name if ex_photo else "",
                "exemplar_face_index": "" if ex_face is None or ex_face.face_index is None else int(ex_face.face_index),
            }
            per_exemplar.append(entry)

        per_exemplar.sort(key=lambda r: r["similarity"], reverse=True)
        scores = [r["similarity"] for r in per_exemplar]
        top_avg = _top_k_average(scores, top_k)
        passed = top_avg >= threshold
        if passed:
            pass_count += 1

        top1 = per_exemplar[0]
        top2 = per_exemplar[1] if len(per_exemplar) > 1 else None
        face_rows.append(
            {
                "direction": f"{source_person}->{target_person}",
                "source_person": source_person,
                "source_cluster_id": source_cluster.cluster_id,
                "source_face_id": face.face_id,
                "source_photo": Path(source_photo.image_path).name if source_photo else "",
                "source_face_index": "" if face.face_index is None else int(face.face_index),
                "source_quality_score": float(face.quality_score),
                "source_yaw_ratio": float(face.yaw_ratio),
                "stored_assignment_state": face.assignment_state.value,
                "stored_best_match_score": "" if face.best_match_score is None else float(face.best_match_score),
                "stored_second_best_score": "" if face.second_best_score is None else float(face.second_best_score),
                "stored_decision_threshold": "" if face.decision_threshold is None else float(face.decision_threshold),
                "target_person": target_person,
                "target_cluster_id": target_cluster.cluster_id,
                "target_exemplar_count": len(target_exemplars),
                "target_effective_threshold": threshold,
                "top_k_used": min(top_k, len(scores)),
                "top_k_average_similarity": top_avg,
                "max_similarity": float(top1["similarity"]),
                "passes_target_threshold": passed,
                "top1_exemplar_face_id": top1["exemplar_face_id"],
                "top1_exemplar_photo": top1["exemplar_photo"],
                "top1_exemplar_face_index": top1["exemplar_face_index"],
                "top1_similarity": float(top1["similarity"]),
                "top2_exemplar_face_id": "" if top2 is None else top2["exemplar_face_id"],
                "top2_exemplar_photo": "" if top2 is None else top2["exemplar_photo"],
                "top2_exemplar_face_index": "" if top2 is None else top2["exemplar_face_index"],
                "top2_similarity": "" if top2 is None else float(top2["similarity"]),
            }
        )

        for rank, entry in enumerate(per_exemplar, start=1):
            exemplar_rows.append(
                {
                    "direction": f"{source_person}->{target_person}",
                    "source_person": source_person,
                    "source_face_id": face.face_id,
                    "source_photo": Path(source_photo.image_path).name if source_photo else "",
                    "source_face_index": "" if face.face_index is None else int(face.face_index),
                    "target_person": target_person,
                    "target_exemplar_rank": rank,
                    "is_top_k": rank <= top_k,
                    "target_exemplar_bucket": entry["bucket"],
                    "target_exemplar_face_id": entry["exemplar_face_id"],
                    "target_exemplar_photo": entry["exemplar_photo"],
                    "target_exemplar_face_index": entry["exemplar_face_index"],
                    "target_exemplar_quality_score": entry["exemplar_quality_score"],
                    "target_exemplar_yaw_ratio": entry["exemplar_yaw_ratio"],
                    "similarity": entry["similarity"],
                }
            )

    total = len(source_faces)
    coverage = float(pass_count / total) if total else 0.0
    summary = {
        "direction": f"{source_person}->{target_person}",
        "source_person": source_person,
        "source_cluster_id": source_cluster.cluster_id,
        "source_face_count": total,
        "target_person": target_person,
        "target_cluster_id": target_cluster.cluster_id,
        "target_exemplar_count": len(target_exemplars),
        "top_k": top_k,
        "effective_target_threshold": threshold,
        "passing_source_faces": pass_count,
        "directional_coverage": coverage,
        "required_merge_coverage": 0.90,
        "passes_90_percent_coverage": coverage >= 0.90,
    }
    return summary, face_rows, exemplar_rows


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

    p2c, _ = _load_person_map(clusters_csv)
    for person in (args.source, args.target):
        if person not in p2c:
            raise ValueError(f"Unknown current person folder: {person}")
    if args.source == args.target:
        raise ValueError("--source and --target must be different clusters")

    diag_dir = (
        Path(args.diagnostic_dir).resolve()
        if args.diagnostic_dir
        else output_dir / "cluster_pair_diagnostics" / f"{_safe_name(args.source)}__vs__{_safe_name(args.target)}"
    )
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Work against a snapshot because FaceGroupingStore may perform schema
    # housekeeping when opened. The completed gallery run remains untouched.
    snapshot = diag_dir / "_diagnostic_snapshot.db"
    shutil.copy2(db_path, snapshot)
    store = FaceGroupingStore(str(snapshot))
    try:
        source_cluster = store.load_cluster(p2c[args.source])
        target_cluster = store.load_cluster(p2c[args.target])
        if source_cluster is None or target_cluster is None:
            raise RuntimeError("One selected active cluster could not be loaded")

        cfg = load_thresholds()
        top_k = int(cfg["matching"]["top_k"])
        t_match = float(cfg["matching"]["t_match"])
        sparse_margin = float(cfg["matching"]["sparse_cluster_margin"])

        forward, forward_faces, forward_ex = _direction(
            store=store,
            source_person=args.source,
            source_cluster=source_cluster,
            target_person=args.target,
            target_cluster=target_cluster,
            top_k=top_k,
            t_match=t_match,
            sparse_margin=sparse_margin,
        )
        reverse, reverse_faces, reverse_ex = _direction(
            store=store,
            source_person=args.target,
            source_cluster=target_cluster,
            target_person=args.source,
            target_cluster=source_cluster,
            top_k=top_k,
            t_match=t_match,
            sparse_margin=sparse_margin,
        )

        all_faces = forward_faces + reverse_faces
        all_ex = forward_ex + reverse_ex
        _write_csv(diag_dir / "directional_face_scores.csv", all_faces)
        _write_csv(diag_dir / "all_exemplar_similarities.csv", all_ex)

        # A compact file just for the small/source cluster is convenient for
        # the age-gap case being investigated.
        _write_csv(diag_dir / f"{_safe_name(args.source)}_to_{_safe_name(args.target)}_faces.csv", forward_faces)

        report = {
            "test": "cluster_pair_merge_evidence_diagnostic_v1",
            "run_output": str(output_dir),
            "source_person": args.source,
            "target_person": args.target,
            "matching": {
                "t_match": t_match,
                "top_k": top_k,
                "sparse_cluster_margin": sparse_margin,
                "merge_required_directional_coverage": 0.90,
            },
            "forward": forward,
            "reverse": reverse,
            "mutual_90_percent_merge_condition": bool(
                forward["passes_90_percent_coverage"] and reverse["passes_90_percent_coverage"]
            ),
            "source_faces": [
                {
                    "photo": r["source_photo"],
                    "face_index": r["source_face_index"],
                    "top_k_average_to_target": r["top_k_average_similarity"],
                    "max_similarity_to_target": r["max_similarity"],
                    "passes_target_threshold": r["passes_target_threshold"],
                    "best_target_exemplar_photo": r["top1_exemplar_photo"],
                    "best_target_exemplar_similarity": r["top1_similarity"],
                }
                for r in forward_faces
            ],
            "notes": [
                "All similarities are computed from embeddings persisted by the completed production run.",
                "No detector, landmarker, IR-SE50 inference, assignment, consolidation, or correction is rerun.",
                "The effective threshold includes sparse_cluster_margin when the target has fewer exemplars than top_k.",
                "Directional coverage is the fraction of source members whose production-style top-k score passes the target cluster threshold.",
                "This diagnostic does not alter clustering state or thresholds.",
            ],
        }
        with (diag_dir / "diagnostic_summary.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("=== CLUSTER PAIR DIAGNOSTIC ===")
        print(f"Run:     {output_dir}")
        print(f"Pair:    {args.source} <-> {args.target}")
        print()
        for d in (forward, reverse):
            print(
                f"{d['direction']}: {d['passing_source_faces']}/{d['source_face_count']} "
                f"pass; coverage={d['directional_coverage']:.1%}; "
                f"target exemplars={d['target_exemplar_count']}; "
                f"threshold={d['effective_target_threshold']:.3f}"
            )
        print(f"Mutual 90% merge condition: {report['mutual_90_percent_merge_condition']}")
        print()
        print(f"Output:  {diag_dir}")
        print(f"Summary: {diag_dir / 'diagnostic_summary.json'}")
        print(f"Faces:   {diag_dir / 'directional_face_scores.csv'}")
        return 0
    finally:
        store.close()
        try:
            snapshot.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
