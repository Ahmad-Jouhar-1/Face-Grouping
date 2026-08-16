#!/usr/bin/env python3
"""Run the production Face Grouping pipeline on an unlabelled photo gallery.

Input contract
--------------
Place arbitrary photos directly under ``data/Gallery`` (subdirectories are also
accepted). A photo may contain one or multiple people. No identity-labelled
folders or ground truth are required.

Output contract
---------------
The runner creates a fresh run directory containing one folder per discovered
person cluster. Each person folder receives a copy of every original photo in
which that cluster has a CONFIRMED/MANUAL face. Therefore, a multi-person photo
can intentionally appear in multiple person folders.

Typical PowerShell usage from the project root::

    python tools/gallery_grouping/run_gallery_grouping_test.py

A custom run can be created with::

    python tools/gallery_grouping/run_gallery_grouping_test.py `
      --gallery data/Gallery `
      --output data/gallery_grouping_output `
      --seed 7

This tool does not change production thresholds. It exercises the configured
production consolidation policy, including high-confidence automatic structural
corrections when enabled.

Diagnostic CSVs are also written for final assignment evidence, suspicious
faces, and pending merge suggestions.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from face_grouping.clustering.merge_rules import SuggestionType
from face_grouping.matching.assignment import AssignmentState
from face_grouping.pipeline import FaceGroupingPipeline


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DB_FILENAME = "gallery_grouping.db"
PERSONS_DIRNAME = "persons"
UNGROUPED_DIRNAME = "_ungrouped_photos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the production Face Grouping pipeline on arbitrary Gallery photos "
            "and export one folder per discovered person."
        )
    )
    parser.add_argument(
        "--gallery",
        default="data/Gallery",
        help="Input photo directory. Images may be flat or nested (default: data/Gallery).",
    )
    parser.add_argument(
        "--output",
        default="data/gallery_grouping_output",
        help=(
            "Base output directory. Existing results are preserved by automatically "
            "creating _run002, _run003, ..."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Deterministic shuffled arrival order (default: 7).",
    )
    parser.add_argument(
        "--consolidate-every",
        type=int,
        default=50,
        help="Run consolidation every N photos (default: 50; 0 = final only).",
    )
    parser.add_argument(
        "--max-photos",
        type=int,
        default=0,
        help="Optional smoke-test limit after shuffling; 0 = process all photos.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the exact --output directory instead of auto-suffixing it.",
    )
    return parser.parse_args()


def _prepare_output_dir(requested: Path, overwrite: bool) -> tuple[Path, bool]:
    requested = requested.expanduser().resolve()
    if not requested.exists():
        return requested, False
    if overwrite:
        shutil.rmtree(requested)
        return requested, False

    index = 2
    while True:
        candidate = requested.with_name(f"{requested.name}_run{index:03d}")
        if not candidate.exists():
            return candidate, True
        index += 1


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _scan_images(gallery_dir: Path, excluded_dir: Path | None = None) -> List[Path]:
    images: List[Path] = []
    for path in gallery_dir.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        if excluded_dir is not None and _is_relative_to(path.resolve(), excluded_dir):
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path.resolve())
    return sorted(images, key=lambda p: str(p).lower())


def _copy_unique(source: Path, destination_dir: Path) -> Path:
    """Copy a photo without overwriting an existing file with the same basename."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    candidate = destination_dir / source.name
    if not candidate.exists():
        shutil.copy2(source, candidate)
        return candidate

    stem = source.stem
    suffix = source.suffix
    index = 2
    while True:
        candidate = destination_dir / f"{stem}__{index}{suffix}"
        if not candidate.exists():
            shutil.copy2(source, candidate)
            return candidate
        index += 1


def _state_counts(pipeline: FaceGroupingPipeline) -> dict[str, int]:
    return {
        state.value: len(pipeline.store.load_faces_by_assignment_state(state))
        for state in AssignmentState
    }


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _collect_auto_correction_rows(consolidation_runs: List[dict]) -> List[dict]:
    rows: List[dict] = []
    for run in consolidation_runs:
        for event in run.get("auto_correction_events", []):
            row = {
                "after_photos": run.get("after_photos"),
                "kind": run.get("kind"),
                **event,
            }
            rows.append(row)
    return rows


def _export_person_folders(
    pipeline: FaceGroupingPipeline,
    output_dir: Path,
    all_input_photos: List[Path],
) -> tuple[List[dict], List[Path]]:
    persons_dir = output_dir / PERSONS_DIRNAME
    persons_dir.mkdir(parents=True, exist_ok=True)

    active_clusters = pipeline.store.load_all_clusters(include_merged=False)
    cluster_records = []
    for cluster in active_clusters:
        photos = pipeline.store.load_photos_by_cluster(cluster.cluster_id)
        if photos:
            cluster_records.append((cluster, photos))

    # Human-friendly numbering: largest discovered people first; ties are stable.
    cluster_records.sort(
        key=lambda item: (-len(item[1]), item[0].created_at, item[0].cluster_id)
    )

    exported_source_paths: set[Path] = set()
    rows: List[dict] = []

    for index, (cluster, photos) in enumerate(cluster_records, start=1):
        person_name = f"person_{index:03d}"
        person_dir = persons_dir / person_name
        copied = 0
        source_paths: set[Path] = set()

        for photo in photos:
            source = Path(photo.image_path).resolve()
            if source in source_paths:
                continue
            source_paths.add(source)
            if not source.exists():
                continue
            _copy_unique(source, person_dir)
            exported_source_paths.add(source)
            copied += 1

        rows.append(
            {
                "person_folder": person_name,
                "cluster_id": cluster.cluster_id,
                "cluster_face_count": int(cluster.face_count),
                "exported_photo_count": copied,
                "exemplar_count": len(cluster.exemplar_set),
                "has_manual_correction": bool(cluster.has_manual_correction),
            }
        )

    ungrouped = [photo for photo in all_input_photos if photo not in exported_source_paths]
    if ungrouped:
        review_dir = output_dir / UNGROUPED_DIRNAME
        for photo in ungrouped:
            _copy_unique(photo, review_dir)

    return rows, ungrouped



def _cluster_folder_map(cluster_rows: List[dict]) -> Dict[str, str]:
    """Map internal cluster IDs to the human-friendly exported person folders."""
    return {
        str(row["cluster_id"]): str(row["person_folder"])
        for row in cluster_rows
        if row.get("cluster_id")
    }


def _relative_photo_path(image_path: str, gallery_dir: Path) -> str:
    source = Path(image_path).resolve()
    try:
        return str(source.relative_to(gallery_dir))
    except ValueError:
        return str(source)


def _optional_float(value: Optional[float]) -> str | float:
    return "" if value is None else float(value)


def _collect_face_assignment_rows(
    pipeline: FaceGroupingPipeline,
    gallery_dir: Path,
    cluster_to_folder: Dict[str, str],
) -> List[dict]:
    """Export the final face state plus the evidence stored at assignment time.

    ``best_match_score``/``second_best_score``/``decision_threshold`` are read
    from the production database rather than recomputed, so this file explains
    the actual decision made when the face was processed/recovered.
    """
    rows: List[dict] = []
    for photo in pipeline.store.load_all_photos():
        for face in pipeline.store.load_faces_by_photo(photo.photo_id):
            cluster_id = face.cluster_id or ""
            candidate_id = face.candidate_cluster_id or ""
            second_id = face.second_best_cluster_id or ""
            rows.append(
                {
                    "photo": _relative_photo_path(photo.image_path, gallery_dir),
                    "photo_id": photo.photo_id,
                    "face_index": "" if face.face_index is None else int(face.face_index),
                    "face_id": face.face_id,
                    "assignment_state": face.assignment_state.value,
                    "person_folder": cluster_to_folder.get(cluster_id, ""),
                    "cluster_id": cluster_id,
                    "candidate_person_folder": cluster_to_folder.get(candidate_id, ""),
                    "candidate_cluster_id": candidate_id,
                    "best_match_score": _optional_float(face.best_match_score),
                    "second_best_person_folder": cluster_to_folder.get(second_id, ""),
                    "second_best_cluster_id": second_id,
                    "second_best_score": _optional_float(face.second_best_score),
                    "score_margin": _optional_float(face.score_margin),
                    "decision_threshold": _optional_float(face.decision_threshold),
                    "decision_reason": face.decision_reason or "",
                    "recognition_restricted": bool(face.recognition_restricted),
                    "recognition_restriction_reason": face.recognition_restriction_reason or "",
                    "quality_score": float(face.quality_score),
                    "yaw_ratio": float(face.yaw_ratio),
                    "bbox_x1": _optional_float(face.bbox_x1),
                    "bbox_y1": _optional_float(face.bbox_y1),
                    "bbox_x2": _optional_float(face.bbox_x2),
                    "bbox_y2": _optional_float(face.bbox_y2),
                    "detection_score": _optional_float(face.detection_score),
                    "is_manually_corrected": bool(face.is_manually_corrected),
                }
            )
    rows.sort(key=lambda row: (str(row["photo"]).lower(), str(row["face_index"])))
    return rows


def _collect_suspicious_face_rows(
    pipeline: FaceGroupingPipeline,
    gallery_dir: Path,
    cluster_to_folder: Dict[str, str],
) -> List[dict]:
    """Re-run the final conservative audit and explain each suspicious face.

    This diagnostic intentionally mirrors the production audit scoring. It does
    not modify memberships, exemplars, thresholds, or suggestions.
    """
    audit = pipeline.consolidation_engine.audit_confirmed_clusters()
    suspicious_ids = set(audit.suspicious_face_ids)
    if not suspicious_ids:
        return []

    clusters = pipeline.store.load_all_clusters(include_merged=False)
    cluster_by_id = {cluster.cluster_id: cluster for cluster in clusters}
    rows: List[dict] = []

    for face_id in sorted(suspicious_ids):
        face = pipeline.store.load_face(face_id)
        if face is None or not face.cluster_id:
            continue
        own_cluster = cluster_by_id.get(face.cluster_id)
        if own_cluster is None:
            continue

        own_result = pipeline.consolidation_engine._score_against_cluster(
            face, own_cluster, exclude_same_face=True
        )
        own_score = own_result[0] if own_result is not None else None

        best_other_cluster_id = ""
        best_other_score: Optional[float] = None
        best_other_threshold: Optional[float] = None
        for other in clusters:
            if other.cluster_id == face.cluster_id:
                continue
            result = pipeline.consolidation_engine._score_against_cluster(face, other)
            if result is None:
                continue
            score, threshold = result
            if best_other_score is None or score > best_other_score:
                best_other_cluster_id = other.cluster_id
                best_other_score = score
                best_other_threshold = threshold

        photo = pipeline.store.load_photo(face.photo_id) if face.photo_id else None
        image_path = photo.image_path if photo is not None else ""
        audit_margin = (
            None
            if own_score is None or best_other_score is None
            else best_other_score - own_score
        )
        rows.append(
            {
                "photo": _relative_photo_path(image_path, gallery_dir) if image_path else "",
                "face_index": "" if face.face_index is None else int(face.face_index),
                "face_id": face.face_id,
                "current_person_folder": cluster_to_folder.get(face.cluster_id, ""),
                "current_cluster_id": face.cluster_id,
                "own_cluster_score": _optional_float(own_score),
                "better_person_folder": cluster_to_folder.get(best_other_cluster_id, ""),
                "better_cluster_id": best_other_cluster_id,
                "better_cluster_score": _optional_float(best_other_score),
                "better_cluster_threshold": _optional_float(best_other_threshold),
                "audit_score_margin": _optional_float(audit_margin),
                "required_margin": float(pipeline.min_cluster_margin),
                "quality_score": float(face.quality_score),
                "yaw_ratio": float(face.yaw_ratio),
                "bbox_x1": _optional_float(face.bbox_x1),
                "bbox_y1": _optional_float(face.bbox_y1),
                "bbox_x2": _optional_float(face.bbox_x2),
                "bbox_y2": _optional_float(face.bbox_y2),
                "original_decision_reason": face.decision_reason or "",
                "original_best_match_score": _optional_float(face.best_match_score),
                "original_second_best_score": _optional_float(face.second_best_score),
                "original_score_margin": _optional_float(face.score_margin),
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["audit_score_margin"] or 0.0),
            str(row["current_person_folder"]),
            str(row["photo"]).lower(),
        )
    )
    return rows


def _collect_merge_suggestion_rows(
    pending_suggestions,
    cluster_to_folder: Dict[str, str],
) -> List[dict]:
    rows: List[dict] = []
    for suggestion in pending_suggestions:
        if suggestion.suggestion_type != SuggestionType.MERGE:
            continue
        cluster_ids = list(suggestion.cluster_ids)
        if len(cluster_ids) != 2:
            continue
        a, b = cluster_ids
        coverage = suggestion.evidence.get("mutual_coverage", {})
        rows.append(
            {
                "suggestion_id": suggestion.suggestion_id,
                "status": suggestion.status.value,
                "person_a": cluster_to_folder.get(a, ""),
                "cluster_a": a,
                "person_b": cluster_to_folder.get(b, ""),
                "cluster_b": b,
                "coverage_a_to_b": _optional_float(coverage.get(a)),
                "coverage_b_to_a": _optional_float(coverage.get(b)),
                "total_members": suggestion.evidence.get("total_members", ""),
                "created_at": suggestion.created_at.isoformat(),
            }
        )
    rows.sort(key=lambda row: (str(row["person_a"]), str(row["person_b"])))
    return rows


def main() -> int:
    args = parse_args()
    if args.consolidate_every < 0:
        raise ValueError("--consolidate-every must be >= 0")
    if args.max_photos < 0:
        raise ValueError("--max-photos must be >= 0")

    os.chdir(PROJECT_ROOT)
    gallery_dir = Path(args.gallery).expanduser().resolve()
    if not gallery_dir.exists() or not gallery_dir.is_dir():
        raise FileNotFoundError(f"Gallery directory does not exist: {gallery_dir}")

    requested_output = Path(args.output).expanduser().resolve()
    # Never allow a destructive overwrite of the input gallery itself or one of its parents.
    if requested_output == gallery_dir or _is_relative_to(gallery_dir, requested_output):
        raise ValueError("--output cannot be the Gallery directory or one of its parent directories")

    output_dir, auto_suffixed = _prepare_output_dir(requested_output, args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If someone intentionally puts the output below Gallery, do not recursively ingest it.
    exclude_from_scan = output_dir if _is_relative_to(output_dir, gallery_dir) else None
    photos = _scan_images(gallery_dir, exclude_from_scan)
    if not photos:
        raise ValueError(f"No supported images found under: {gallery_dir}")

    rng = random.Random(args.seed)
    scheduled = list(photos)
    rng.shuffle(scheduled)
    if args.max_photos:
        scheduled = scheduled[: args.max_photos]
    if not scheduled:
        raise ValueError("No photos scheduled")

    print("=== FACE GROUPING VISUAL TEST ===")
    print(f"Gallery:                 {gallery_dir}")
    print(f"Photos scheduled:        {len(scheduled)}")
    print(f"Arrival-order seed:      {args.seed}")
    print(
        f"Consolidation cadence:   every {args.consolidate_every} photos"
        if args.consolidate_every
        else "Consolidation cadence:   final only"
    )
    print(f"Output:                  {output_dir}")
    if auto_suffixed:
        print("Previous output preserved; a new numbered run directory was selected.")
    if args.max_photos:
        print("NOTE: --max-photos is active; this is a smoke run.")
    print()

    failures: List[dict] = []
    consolidation_runs: List[dict] = []
    accepted_faces = 0
    start = time.perf_counter()
    db_path = output_dir / DB_FILENAME

    with FaceGroupingPipeline(str(db_path)) as pipeline:
        for index, photo_path in enumerate(scheduled, start=1):
            try:
                faces = pipeline.process_photo(str(photo_path))
                accepted_faces += len(faces)
            except Exception as exc:
                failures.append(
                    {
                        "photo": str(photo_path.relative_to(gallery_dir)),
                        "error_type": type(exc).__name__,
                        "message": str(exc)[:500],
                    }
                )

            if index % 25 == 0 or index == len(scheduled):
                elapsed = time.perf_counter() - start
                print(
                    f"processed {index}/{len(scheduled)}; "
                    f"accepted faces={accepted_faces}; failures={len(failures)}; "
                    f"elapsed={elapsed:.1f}s"
                )

            if args.consolidate_every and index % args.consolidate_every == 0:
                summary = pipeline.run_consolidation()
                consolidation_runs.append({"after_photos": index, "kind": "periodic", **summary})
                print(
                    f"  consolidation @{index}: recovered={summary['recovered_confirmed']}, "
                    f"new_clusters={summary['new_clusters']}, "
                    f"auto_merges={summary.get('auto_merges', 0)}, "
                    f"auto_splits={summary.get('auto_splits', 0)}, "
                    f"pose_recovered={summary.get('restricted_pose_recovered_confirmed', 0)}, "
                    f"merge_suggestions={summary['merge_suggestions']}, "
                    f"split_suggestions={summary['split_suggestions']}"
                )

        # If the cadence did not land exactly on the final photo, perform the final pass.
        if not args.consolidate_every or len(scheduled) % args.consolidate_every != 0:
            summary = pipeline.run_consolidation()
            consolidation_runs.append({"after_photos": len(scheduled), "kind": "final", **summary})
            print(
                f"final consolidation: recovered={summary['recovered_confirmed']}, "
                f"new_clusters={summary['new_clusters']}, "
                f"auto_merges={summary.get('auto_merges', 0)}, "
                f"auto_splits={summary.get('auto_splits', 0)}, "
                f"pose_recovered={summary.get('restricted_pose_recovered_confirmed', 0)}, "
                f"merge_suggestions={summary['merge_suggestions']}, "
                f"split_suggestions={summary['split_suggestions']}"
            )

        cluster_rows, ungrouped_photos = _export_person_folders(
            pipeline,
            output_dir,
            scheduled,
        )

        storage_errors = pipeline.validate_storage()
        state_counts = _state_counts(pipeline)
        pending = pipeline.store.load_pending_suggestions()
        suggestion_counts = Counter(s.suggestion_type.value for s in pending)
        active_clusters = pipeline.store.load_all_clusters(include_merged=False)

        cluster_to_folder = _cluster_folder_map(cluster_rows)
        face_assignment_rows = _collect_face_assignment_rows(
            pipeline, gallery_dir, cluster_to_folder
        )
        suspicious_face_rows = _collect_suspicious_face_rows(
            pipeline, gallery_dir, cluster_to_folder
        )
        merge_suggestion_rows = _collect_merge_suggestion_rows(
            pending, cluster_to_folder
        )
        auto_correction_rows = _collect_auto_correction_rows(consolidation_runs)
        final_auto_policy_audit = pipeline.consolidation_engine.audit_confirmed_clusters()
        auto_merge_policy_evaluations = final_auto_policy_audit.auto_merge_evaluations

        summary_payload = {
            "test": "unlabelled_gallery_visual_grouping_v2_diagnostics",
            "gallery": str(gallery_dir),
            "output": str(output_dir),
            "seed": args.seed,
            "scheduled_photos": len(scheduled),
            "accepted_faces": accepted_faces,
            "active_clusters": len(active_clusters),
            "person_folders": len(cluster_rows),
            "assignment_states": state_counts,
            "ungrouped_photos": len(ungrouped_photos),
            "processing_failures": len(failures),
            "pending_merge_suggestions": int(suggestion_counts.get(SuggestionType.MERGE.value, 0)),
            "pending_split_suggestions": int(suggestion_counts.get(SuggestionType.SPLIT.value, 0)),
            "total_auto_merges": sum(int(run.get("auto_merges", 0)) for run in consolidation_runs),
            "total_auto_splits": sum(int(run.get("auto_splits", 0)) for run in consolidation_runs),
            "total_restricted_pose_recovered": sum(
                int(run.get("restricted_pose_recovered_confirmed", 0))
                for run in consolidation_runs
            ),
            "final_recognition_restricted_faces": sum(
                1
                for photo in pipeline.store.load_all_photos()
                for face in pipeline.store.load_faces_by_photo(photo.photo_id)
                if face.recognition_restricted
            ),
            "final_recognition_restricted_confirmed": sum(
                1
                for photo in pipeline.store.load_all_photos()
                for face in pipeline.store.load_faces_by_photo(photo.photo_id)
                if face.recognition_restricted
                and face.assignment_state in (AssignmentState.CONFIRMED, AssignmentState.MANUAL)
            ),
            "final_suspicious_faces": len(suspicious_face_rows),
            "diagnostic_face_rows": len(face_assignment_rows),
            "storage_integrity_errors": storage_errors,
            "consolidation_runs": consolidation_runs,
            "notes": [
                "Each person folder contains full original photos, not face crops.",
                "A multi-person photo may correctly appear in multiple person folders.",
                "AMBIGUOUS/UNASSIGNED faces are not forced into person folders.",
                "Pose-only hard exclusions are persisted as recognition-restricted faces and may join only mature existing clusters during consolidation; they never seed clusters or exemplars.",
                "Only the production high-confidence auto-correction tier is applied automatically; borderline Merge/Split suggestions remain pending for user review.",
                "Diagnostic CSVs explain stored assignment evidence and the final audit without changing pipeline behavior.",
            ],
        }

    _write_csv(
        output_dir / "clusters.csv",
        cluster_rows,
        [
            "person_folder",
            "cluster_id",
            "cluster_face_count",
            "exported_photo_count",
            "exemplar_count",
            "has_manual_correction",
        ],
    )
    if failures:
        _write_csv(output_dir / "failures.csv", failures, ["photo", "error_type", "message"])

    _write_csv(
        output_dir / "face_assignments.csv",
        face_assignment_rows,
        [
            "photo", "photo_id", "face_index", "face_id", "assignment_state",
            "person_folder", "cluster_id", "candidate_person_folder",
            "candidate_cluster_id", "best_match_score", "second_best_person_folder",
            "second_best_cluster_id", "second_best_score", "score_margin",
            "decision_threshold", "decision_reason", "recognition_restricted",
            "recognition_restriction_reason", "quality_score", "yaw_ratio",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "detection_score",
            "is_manually_corrected",
        ],
    )
    _write_csv(
        output_dir / "suspicious_faces.csv",
        suspicious_face_rows,
        [
            "photo", "face_index", "face_id", "current_person_folder",
            "current_cluster_id", "own_cluster_score", "better_person_folder",
            "better_cluster_id", "better_cluster_score", "better_cluster_threshold",
            "audit_score_margin", "required_margin", "quality_score", "yaw_ratio",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "original_decision_reason", "original_best_match_score",
            "original_second_best_score", "original_score_margin",
        ],
    )
    _write_csv(
        output_dir / "merge_suggestions.csv",
        merge_suggestion_rows,
        [
            "suggestion_id", "status", "person_a", "cluster_a", "person_b",
            "cluster_b", "coverage_a_to_b", "coverage_b_to_a",
            "total_members", "created_at",
        ],
    )

    auto_fields = [
        "after_photos", "kind", "type", "mode", "source_cluster_id",
        "target_cluster_id", "survivor_cluster_id", "source_size", "target_size",
        "source_coverage", "reverse_coverage", "strong_anchor_count",
        "reverse_strong_anchor_count", "min_target_score", "mean_target_score",
        "min_competition_margin", "member_bridge_source_coverage",
        "high_conf_bridge_source_count", "strong_bridge_source_count",
        "min_member_support_count", "distinct_bridge_target_faces",
        "distinct_bridge_target_photos", "result_cluster_ids", "group_sizes",
        "source_cluster_id",
    ]
    # Remove accidental duplicate field names while preserving order.
    auto_fields = list(dict.fromkeys(auto_fields))
    normalized_auto_rows = [
        {field: row.get(field, "") for field in auto_fields}
        for row in auto_correction_rows
    ]
    _write_csv(output_dir / "auto_corrections.csv", normalized_auto_rows, auto_fields)

    with (output_dir / "auto_merge_policy_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "policy": "auto_correction_v2",
                "singleton_auto_merge_policy": "protected",
                "evaluations": auto_merge_policy_evaluations,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    print("\n=== RESULT ===")
    print(f"Discovered person folders: {summary_payload['person_folders']}")
    print(f"Confirmed/manual faces:    {state_counts.get('confirmed', 0) + state_counts.get('manual', 0)}")
    print(f"Ambiguous faces:           {state_counts.get('ambiguous', 0)}")
    print(f"Unassigned faces:          {state_counts.get('unassigned', 0)}")
    print(f"Restricted-pose faces:     {summary_payload['final_recognition_restricted_faces']}")
    print(f"Restricted-pose confirmed: {summary_payload['final_recognition_restricted_confirmed']}")
    print(f"Ungrouped source photos:   {summary_payload['ungrouped_photos']}")
    print(f"Processing failures:       {summary_payload['processing_failures']}")
    print(f"Storage integrity errors:  {len(summary_payload['storage_integrity_errors'])}")
    print(f"Final suspicious faces:    {summary_payload['final_suspicious_faces']}")
    print(f"Face diagnostics:          {output_dir / 'face_assignments.csv'}")
    print(f"Suspicious diagnostics:    {output_dir / 'suspicious_faces.csv'}")
    print(f"Merge diagnostics:         {output_dir / 'merge_suggestions.csv'}")
    print(f"Auto corrections:          {output_dir / 'auto_corrections.csv'}")
    print(f"Auto-merge policy audit:   {output_dir / 'auto_merge_policy_diagnostics.json'}")
    print(f"Person folders:            {output_dir / PERSONS_DIRNAME}")
    print(f"Summary:                   {output_dir / 'summary.json'}")

    # Photo-level failures are reported but do not erase successful grouping output.
    return 0 if not summary_payload["storage_integrity_errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
