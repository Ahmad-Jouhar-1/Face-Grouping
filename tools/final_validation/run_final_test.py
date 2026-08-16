#!/usr/bin/env python3
"""Private final end-to-end validation runner for Face Grouping.

This tool intentionally lives outside ``src/face_grouping``. It evaluates the
production pipeline without becoming a production dependency or changing any
pipeline threshold/decision rule.

Typical use (PowerShell):

    python tools/final_validation/run_final_test.py `
      --gallery data/Gallery `
      --output data/final_test_runs/seed7 `
      --seed 7

The shareable ZIP contains anonymized metrics only. Private image paths and the
folder-name -> Pxxx mapping remain in ``<output>/private`` and are never added
to that ZIP.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from face_grouping.clustering.merge_rules import SuggestionType
from face_grouping.matching.assignment import AssignmentState
from face_grouping.pipeline import FaceGroupingPipeline

from tools.final_validation.dataset import GalleryIndex, CanonicalPhoto, anonymized_identities, scan_gallery
from tools.final_validation.metrics import clustering_bundle, contamination, fragmentation, photo_set_metrics
from tools.final_validation.reporting import make_shareable_zip, sanitize_message, write_csv, write_json
from tools.final_validation.resolver import FaceView, resolve_folder_ground_truth


SHARE_DIRNAME = "share"
PRIVATE_DIRNAME = "private"
DB_FILENAME = "final_test.db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete production face-grouping pipeline on a private folder-labelled gallery."
    )
    parser.add_argument("--gallery", default="data/Gallery", help="Folder tree: <gallery>/<identity>/images")
    parser.add_argument(
        "--output",
        default="data/final_test_runs/seed7",
        help="Output directory base. If it already exists, a unique _runNNN directory is created automatically.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Deterministic shuffled arrival order")
    parser.add_argument(
        "--consolidate-every",
        type=int,
        default=50,
        help="Run consolidation after every N unique photos (default: 50; 0 = final only)",
    )
    parser.add_argument("--max-photos", type=int, default=0, help="Smoke-test limit after shuffling; 0 = all")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace the exact --output directory instead of creating a new _runNNN directory.",
    )
    return parser.parse_args()


def _prepare_output_dir(requested: Path, overwrite: bool) -> tuple[Path, bool]:
    """Return a safe output directory and whether an automatic suffix was used.

    Normal team usage is non-destructive: the first run uses the requested path, and
    later runs automatically use ``<name>_run002``, ``<name>_run003``, ... .
    ``--overwrite`` remains available for an explicit destructive rerun.
    """
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


def _state_counts(pipeline: FaceGroupingPipeline) -> Dict[str, int]:
    return {
        state.value: len(pipeline.store.load_faces_by_assignment_state(state))
        for state in AssignmentState
    }


def _membership_fingerprint(pipeline: FaceGroupingPipeline) -> Tuple[Tuple[str, str, str], ...]:
    rows = []
    for photo in pipeline.store.load_all_photos():
        for face in pipeline.store.load_faces_by_photo(photo.photo_id):
            rows.append((face.face_id, face.cluster_id or "", face.assignment_state.value))
    return tuple(sorted(rows))


def _snapshot(pipeline: FaceGroupingPipeline) -> dict:
    clusters = pipeline.store.load_all_clusters(include_merged=False)
    pending = pipeline.store.load_pending_suggestions()
    return {
        "assignment_states": _state_counts(pipeline),
        "active_clusters": len(clusters),
        "pending_suggestions": len(pending),
        "pending_merge": sum(1 for s in pending if s.suggestion_type == SuggestionType.MERGE),
        "pending_split": sum(1 for s in pending if s.suggestion_type == SuggestionType.SPLIT),
    }


def _safe_float(value):
    return None if value is None else float(value)


def _build_faces_by_photo(
    pipeline: FaceGroupingPipeline,
    photo_id_to_code: Mapping[str, str],
) -> Tuple[Dict[str, List[FaceView]], Dict[str, object]]:
    result: Dict[str, List[FaceView]] = defaultdict(list)
    face_objects: Dict[str, object] = {}
    for photo in pipeline.store.load_all_photos():
        photo_code = photo_id_to_code.get(photo.photo_id)
        if photo_code is None:
            continue
        for face in pipeline.store.load_faces_by_photo(photo.photo_id):
            result[photo_code].append(
                FaceView(
                    face_id=face.face_id,
                    photo_code=photo_code,
                    face_index=int(face.face_index if face.face_index is not None else -1),
                    cluster_id=face.cluster_id,
                    assignment_state=face.assignment_state.value,
                )
            )
            face_objects[face.face_id] = face
    for photo_code in result:
        result[photo_code].sort(key=lambda f: (f.face_index, f.face_id))
    return dict(result), face_objects


def _same_photo_violations(faces_by_photo: Mapping[str, Sequence[FaceView]]) -> List[dict]:
    violations = []
    for photo_code, faces in faces_by_photo.items():
        clusters = defaultdict(list)
        for face in faces:
            if face.cluster_id and face.assignment_state in (AssignmentState.CONFIRMED.value, AssignmentState.MANUAL.value):
                clusters[face.cluster_id].append(face.face_id)
        for cluster_id, face_ids in clusters.items():
            if len(face_ids) > 1:
                violations.append({
                    "photo_id": photo_code,
                    "cluster_id": cluster_id,
                    "face_count": len(face_ids),
                    "face_ids": face_ids,
                })
    return violations


def _prediction_maps(exact_gt: Mapping[str, str], face_objects: Mapping[str, object]) -> Tuple[Dict[str, str], Dict[str, str]]:
    clustered = {}
    end_to_end = {}
    for face_id in exact_gt:
        face = face_objects.get(face_id)
        if face is None:
            continue
        if face.cluster_id and face.assignment_state in (AssignmentState.CONFIRMED, AssignmentState.MANUAL):
            clustered[face_id] = face.cluster_id
            end_to_end[face_id] = face.cluster_id
        else:
            end_to_end[face_id] = f"singleton::{face_id}"
    return clustered, end_to_end


def _exemplar_purity(pipeline: FaceGroupingPipeline, exact_gt: Mapping[str, str]) -> dict:
    details = {}
    correct = 0
    known = 0
    for cluster in pipeline.store.load_all_clusters(include_merged=False):
        labels = [
            exact_gt[ex.face_id]
            for ex in cluster.exemplar_set.all_exemplars()
            if ex.face_id in exact_gt
        ]
        if not labels:
            continue
        counts = Counter(labels)
        dominant, dominant_count = counts.most_common(1)[0]
        known += len(labels)
        correct += dominant_count
        details[cluster.cluster_id] = {
            "known_exact_exemplars": len(labels),
            "dominant_identity": dominant,
            "identity_counts": dict(sorted(counts.items())),
            "correct_vs_dominant": dominant_count,
        }
    return {
        "known_exact_exemplars": known,
        "correct_vs_cluster_dominant": correct,
        "purity": correct / known if known else 0.0,
        "cluster_details": details,
    }


def _suggestion_rows_and_summary(pipeline: FaceGroupingPipeline, resolution, unique_photos: int):
    pending = pipeline.store.load_pending_suggestions()
    rows = []
    classification_counts = Counter()
    for suggestion in pending:
        classification = "unknown"
        if suggestion.suggestion_type == SuggestionType.MERGE and len(suggestion.cluster_ids) == 2:
            a, b = suggestion.cluster_ids
            ia = resolution.cluster_identity_map.get(a)
            ib = resolution.cluster_identity_map.get(b)
            if ia and ib:
                classification = "likely_helpful_same_identity" if ia == ib else "dangerous_different_identities"
        elif suggestion.suggestion_type == SuggestionType.SPLIT and suggestion.cluster_ids:
            cid = suggestion.cluster_ids[0]
            anchor_counts = resolution.cluster_anchor_counts.get(cid, {})
            if len(anchor_counts) > 1:
                classification = "likely_helpful_mixed_anchor_cluster"
            elif len(anchor_counts) == 1:
                classification = "potentially_unnecessary_single_anchor_identity"
        classification_counts[classification] += 1
        rows.append({
            "suggestion_id": suggestion.suggestion_id,
            "type": suggestion.suggestion_type.value,
            "cluster_ids": "|".join(suggestion.cluster_ids),
            "classification": classification,
            "evidence_json": json.dumps(suggestion.evidence, sort_keys=True),
        })
    total = len(rows)
    return rows, {
        "total": total,
        "merge": sum(1 for s in pending if s.suggestion_type == SuggestionType.MERGE),
        "split": sum(1 for s in pending if s.suggestion_type == SuggestionType.SPLIT),
        "per_1000_photos": total * 1000.0 / unique_photos if unique_photos else 0.0,
        "classifications": dict(classification_counts),
        "dangerous_anchor_known_suggestions": classification_counts.get("dangerous_different_identities", 0),
    }


def _print_metric_line(label: str, metric: dict) -> None:
    print(
        f"{label}: purity={metric['purity']['purity']:.4f}, "
        f"B3 F1={metric['bcubed']['f1']:.4f}, pairwise F1={metric['pairwise']['f1']:.4f}"
    )


def main() -> int:
    args = parse_args()
    if args.consolidate_every < 0:
        raise ValueError("--consolidate-every must be >= 0")
    if args.max_photos < 0:
        raise ValueError("--max-photos must be >= 0")

    os.chdir(PROJECT_ROOT)
    gallery = scan_gallery(args.gallery)
    requested_output_dir = Path(args.output)
    output_dir, auto_suffixed_output = _prepare_output_dir(requested_output_dir, args.overwrite)
    share_dir = output_dir / SHARE_DIRNAME
    private_dir = output_dir / PRIVATE_DIRNAME
    share_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)

    rng = random.Random(args.seed)
    scheduled = list(gallery.photos)
    rng.shuffle(scheduled)
    if args.max_photos:
        scheduled = scheduled[: args.max_photos]
    if not scheduled:
        raise ValueError("No photos scheduled")

    expected_by_photo = {
        photo.photo_code: anonymized_identities(photo, gallery.identity_codes)
        for photo in scheduled
    }
    photo_by_code = {photo.photo_code: photo for photo in scheduled}

    multi_count = sum(1 for p in scheduled if len(p.identities) > 1)
    print("=== FINAL PRIVATE FACE-GROUPING TEST ===")
    print(f"Unique identities:             {len(gallery.identities)}")
    print(f"Unique gallery photos:         {len(scheduled)}")
    print(f"Multi-person labelled photos:  {multi_count}")
    print(f"Arrival-order seed:            {args.seed}")
    print(f"Consolidation cadence:         every {args.consolidate_every} photos" if args.consolidate_every else "Consolidation cadence:         final only")
    print(f"Output directory:              {output_dir}")
    if auto_suffixed_output:
        print("NOTE: Previous output was preserved; a new numbered run directory was selected automatically.")
    if args.max_photos:
        print("NOTE: --max-photos is active; this is a smoke run, not the final validation.")
    if gallery.warnings:
        print("Dataset warnings:")
        for warning in gallery.warnings:
            print(f"  - {warning}")
    print()

    db_path = output_dir / DB_FILENAME
    failures = []
    checkpoints = []
    photo_id_to_code: Dict[str, str] = {}
    private_photo_rows = []
    process_start = time.perf_counter()
    consolidation_seconds = 0.0
    pipeline_faces_created = 0
    zero_accepted_face_photos = 0

    with FaceGroupingPipeline(str(db_path)) as pipeline:
        for index, photo in enumerate(scheduled, start=1):
            try:
                created = pipeline.process_photo(str(photo.canonical_path))
                pipeline_faces_created += len(created)
                if not created:
                    zero_accepted_face_photos += 1
                stored_photo = pipeline.store.get_photo_by_path(str(photo.canonical_path))
                if stored_photo is None:
                    raise RuntimeError("Processed photo was not persisted")
                photo_id_to_code[stored_photo.photo_id] = photo.photo_code
                private_photo_rows.append({
                    "photo_id": photo.photo_code,
                    "canonical_path": str(photo.canonical_path),
                    "identity_folders": "|".join(sorted(photo.identities)),
                    "source_copy_count": len(photo.source_copies),
                })
            except Exception as exc:
                failures.append({
                    "photo_id": photo.photo_code,
                    "error_type": type(exc).__name__,
                    "message": sanitize_message(str(exc), gallery.gallery_dir),
                })

            if index % 25 == 0 or index == len(scheduled):
                elapsed = (time.perf_counter() - process_start) / 60.0
                print(
                    f"processed {index}/{len(scheduled)} unique photos; "
                    f"accepted faces={pipeline_faces_created}; failures={len(failures)}; elapsed={elapsed:.1f} min"
                )

            if args.consolidate_every and index % args.consolidate_every == 0:
                before = _snapshot(pipeline)
                started = time.perf_counter()
                summary = pipeline.run_consolidation()
                consolidation_seconds += time.perf_counter() - started
                after = _snapshot(pipeline)
                checkpoints.append({
                    "photos_processed": index,
                    "kind": "periodic",
                    "before": before,
                    "consolidation": summary,
                    "after": after,
                })
                print(
                    f"  consolidation @{index}: recovered={summary['recovered_confirmed']}, "
                    f"new_clusters={summary['new_clusters']}, merge_suggestions={summary['merge_suggestions']}, "
                    f"split_suggestions={summary['split_suggestions']}"
                )

        # Final consolidation when the cadence did not land exactly on the end,
        # or when periodic consolidation was disabled.
        need_final = (
            not args.consolidate_every
            or len(scheduled) % args.consolidate_every != 0
        )
        if need_final:
            before = _snapshot(pipeline)
            started = time.perf_counter()
            summary = pipeline.run_consolidation()
            consolidation_seconds += time.perf_counter() - started
            after = _snapshot(pipeline)
            checkpoints.append({
                "photos_processed": len(scheduled),
                "kind": "final",
                "before": before,
                "consolidation": summary,
                "after": after,
            })
            print(
                f"final consolidation: recovered={summary['recovered_confirmed']}, "
                f"new_clusters={summary['new_clusters']}, merge_suggestions={summary['merge_suggestions']}, "
                f"split_suggestions={summary['split_suggestions']}"
            )

        fingerprint_before = _membership_fingerprint(pipeline)
        started = time.perf_counter()
        second_summary = pipeline.run_consolidation()
        consolidation_seconds += time.perf_counter() - started
        fingerprint_after = _membership_fingerprint(pipeline)
        second_changed = fingerprint_before != fingerprint_after

        # Re-processing the same path must be idempotent and create no rows.
        probe_photo = scheduled[0]
        before_photo_count = len(pipeline.store.load_all_photos())
        before_face_count = sum(
            len(pipeline.store.load_faces_by_photo(photo.photo_id))
            for photo in pipeline.store.load_all_photos()
        )
        pipeline.process_photo(str(probe_photo.canonical_path))
        after_photo_count = len(pipeline.store.load_all_photos())
        after_face_count = sum(
            len(pipeline.store.load_faces_by_photo(photo.photo_id))
            for photo in pipeline.store.load_all_photos()
        )
        reprocess_idempotent = before_photo_count == after_photo_count and before_face_count == after_face_count

        storage_errors = pipeline.validate_storage()
        fresh_pruned = pipeline.run_pruning()
        final_snapshot = _snapshot(pipeline)

        faces_by_photo, face_objects = _build_faces_by_photo(pipeline, photo_id_to_code)
        same_photo_violations = _same_photo_violations(faces_by_photo)
        resolution = resolve_folder_ground_truth(
            expected_identities_by_photo=expected_by_photo,
            faces_by_photo=faces_by_photo,
        )

        exact_gt = resolution.exact_face_gt
        clustered_pred, end_to_end_pred = _prediction_maps(exact_gt, face_objects)
        exact_clustered = clustering_bundle(exact_gt, clustered_pred)
        exact_end_to_end = clustering_bundle(exact_gt, end_to_end_pred)
        exact_contamination = contamination(exact_gt, clustered_pred)
        exact_fragmentation = fragmentation(exact_gt, clustered_pred)
        exemplar_purity = _exemplar_purity(pipeline, exact_gt)
        photo_sets = photo_set_metrics(expected_by_photo, resolution.predicted_identity_sets)
        suggestion_rows, suggestion_summary = _suggestion_rows_and_summary(
            pipeline, resolution, len(scheduled)
        )

        # Detection/retention count proxy. Under the folder contract, each
        # canonical photo's label count is the expected benchmark-person count.
        count_rows = []
        exact_face_count_match = 0
        under_count = 0
        over_count = 0
        for photo_code, expected_ids in expected_by_photo.items():
            accepted_count = len(faces_by_photo.get(photo_code, ()))
            expected_count = len(expected_ids)
            if accepted_count == expected_count:
                exact_face_count_match += 1
            elif accepted_count < expected_count:
                under_count += 1
            else:
                over_count += 1
            count_rows.append((photo_code, expected_count, accepted_count))

        all_faces = [face for faces in faces_by_photo.values() for face in faces]
        exact_identity_counts = Counter(exact_gt.values())
        exact_identities = len(exact_identity_counts)

        # Cluster diagnostics derived from independent exact labels only.
        exact_by_cluster = defaultdict(Counter)
        for face_id, identity in exact_gt.items():
            face = face_objects.get(face_id)
            if face and face.cluster_id:
                exact_by_cluster[face.cluster_id][identity] += 1

        cluster_rows = []
        for cluster in pipeline.store.load_all_clusters(include_merged=False):
            counts = exact_by_cluster.get(cluster.cluster_id, Counter())
            dominant = counts.most_common(1)[0][0] if counts else ""
            known = sum(counts.values())
            minority = known - (counts.most_common(1)[0][1] if counts else 0)
            cluster_rows.append({
                "cluster_id": cluster.cluster_id,
                "face_count": cluster.face_count,
                "exemplar_count": len(cluster.exemplar_set),
                "exact_gt_faces": known,
                "exact_gt_dominant_identity": dominant,
                "exact_gt_minority_faces": minority,
                "exact_gt_identity_counts": json.dumps(dict(sorted(counts.items()))),
                "anchor_identity": resolution.cluster_identity_map.get(cluster.cluster_id, ""),
                "anchor_conflicted": cluster.cluster_id in resolution.conflicted_anchor_clusters,
            })

        face_rows = []
        problems = []
        for photo_code, faces in sorted(faces_by_photo.items()):
            for view in faces:
                face = face_objects[view.face_id]
                exact_identity = exact_gt.get(view.face_id, "")
                exact_source = resolution.exact_source.get(view.face_id, "")
                cluster_dominant = ""
                if face.cluster_id and exact_by_cluster.get(face.cluster_id):
                    cluster_dominant = exact_by_cluster[face.cluster_id].most_common(1)[0][0]
                exact_wrong = bool(exact_identity and face.cluster_id and cluster_dominant and exact_identity != cluster_dominant)
                face_rows.append({
                    "photo_id": photo_code,
                    "face_index": view.face_index,
                    "face_id": view.face_id,
                    "assignment_state": view.assignment_state,
                    "cluster_id": view.cluster_id or "",
                    "quality_score": _safe_float(face.quality_score),
                    "yaw_ratio": _safe_float(face.yaw_ratio),
                    "detection_score": _safe_float(face.detection_score),
                    "best_match_score": _safe_float(face.best_match_score),
                    "second_best_score": _safe_float(face.second_best_score),
                    "score_margin": _safe_float(face.score_margin),
                    "decision_threshold": _safe_float(face.decision_threshold),
                    "decision_reason": face.decision_reason or "",
                    "exact_gt_identity": exact_identity,
                    "exact_gt_source": exact_source,
                    "exact_gt_cluster_dominant": cluster_dominant,
                    "exact_gt_minority_face": exact_wrong,
                })
                if exact_wrong:
                    problems.append({
                        "severity": "accuracy",
                        "type": "exact_gt_minority_face",
                        "photo_id": photo_code,
                        "face_index": view.face_index,
                        "cluster_id": view.cluster_id or "",
                        "identity": exact_identity,
                        "details": f"cluster exact-GT dominant={cluster_dominant}",
                    })
                if exact_identity and not face.cluster_id:
                    problems.append({
                        "severity": "accuracy",
                        "type": "exact_gt_deferred_face",
                        "photo_id": photo_code,
                        "face_index": view.face_index,
                        "cluster_id": "",
                        "identity": exact_identity,
                        "details": view.assignment_state,
                    })

        photo_rows = []
        for photo_code in sorted(expected_by_photo):
            expected = expected_by_photo[photo_code]
            predicted = resolution.predicted_identity_sets.get(photo_code, [])
            pr = resolution.photo_resolution.get(photo_code, {})
            accepted_count = len(faces_by_photo.get(photo_code, ()))
            photo_rows.append({
                "photo_id": photo_code,
                "expected_identities": "|".join(expected),
                "expected_identity_count": len(expected),
                "multi_person": len(expected) > 1,
                "accepted_face_count": accepted_count,
                "predicted_anchor_mapped_identities": "|".join(predicted),
                "predicted_identity_count": len(predicted),
                "identity_set_exact_match": set(expected) == set(predicted),
                "exact_face_labels": pr.get("exact_face_labels", 0),
                "unresolved_face_count": pr.get("unresolved_face_count", accepted_count),
                "missing_expected_identities": "|".join(pr.get("missing_expected_identities", [])),
                "extra_predicted_identities": "|".join(pr.get("extra_predicted_identities", [])),
            })
            if set(expected) != set(predicted):
                problems.append({
                    "severity": "accuracy",
                    "type": "photo_identity_set_mismatch",
                    "photo_id": photo_code,
                    "face_index": "",
                    "cluster_id": "",
                    "identity": "",
                    "details": f"expected={'|'.join(expected)} predicted={'|'.join(predicted)}",
                })

        for violation in same_photo_violations:
            problems.append({
                "severity": "safety",
                "type": "same_photo_cannot_link_violation",
                "photo_id": violation["photo_id"],
                "face_index": "",
                "cluster_id": violation["cluster_id"],
                "identity": "",
                "details": f"{violation['face_count']} faces share one cluster",
            })
        for cluster_id, counts in resolution.conflicted_anchor_clusters.items():
            problems.append({
                "severity": "safety",
                "type": "anchor_proven_contaminated_cluster",
                "photo_id": "",
                "face_index": "",
                "cluster_id": cluster_id,
                "identity": "",
                "details": json.dumps(counts, sort_keys=True),
            })
        for failure in failures:
            problems.append({
                "severity": "functional",
                "type": "photo_processing_failure",
                "photo_id": failure["photo_id"],
                "face_index": "",
                "cluster_id": "",
                "identity": "",
                "details": f"{failure['error_type']}: {failure['message']}",
            })

        functional_pass = (
            not failures
            and not storage_errors
            and not same_photo_violations
            and not second_changed
            and reprocess_idempotent
            and fresh_pruned == 0
        )

        runtime_seconds = time.perf_counter() - process_start
        report = {
            "benchmark": "private_gallery_final_validation_v1",
            "functional_status": "PASS" if functional_pass else "REVIEW",
            "accuracy_status": "MEASURED_NO_HARDCODED_ACCEPTANCE_THRESHOLD",
            "run": {
                "seed": args.seed,
                "consolidate_every": args.consolidate_every,
                "max_photos": args.max_photos,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "runtime_seconds": runtime_seconds,
                "consolidation_seconds": consolidation_seconds,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "embedding_model_version": pipeline.embedding_model_version,
                "config_version": pipeline.config_version,
                "match_threshold_observed": pipeline.t_match,
            },
            "dataset": {
                "identity_count": len(gallery.identities),
                "scheduled_unique_photos": len(scheduled),
                "multi_person_photos": multi_count,
                "single_person_photos": len(scheduled) - multi_count,
                "label_instances": sum(len(ids) for ids in expected_by_photo.values()),
                "ignored_non_image_files": len(gallery.ignored_files),
                "warnings": gallery.warnings,
                "ground_truth_contract": "folder labels + SHA256 duplicate union; multi-person photo must be byte-identical in every visible benchmark identity folder",
            },
            "processing": {
                "processed_unique_photos": len(scheduled) - len(failures),
                "failed_photos": len(failures),
                "failure_details": failures,
                "pipeline_faces_created": pipeline_faces_created,
                "zero_accepted_face_photos": zero_accepted_face_photos,
                "accepted_face_count_vs_folder_label_count": {
                    "exact_count_match_photos": exact_face_count_match,
                    "under_count_photos": under_count,
                    "over_count_photos": over_count,
                    "exact_count_match_rate": exact_face_count_match / len(scheduled) if scheduled else 0.0,
                },
            },
            "final_pipeline_state": final_snapshot,
            "consolidation_checkpoints": checkpoints,
            "idempotency": {
                "second_consolidation_summary": second_summary,
                "second_consolidation_changed_membership": second_changed,
                "reprocess_same_path_created_new_rows": not reprocess_idempotent,
            },
            "fresh_pruning": {
                "pruned_clusters": fresh_pruned,
                "expected": 0,
            },
            "storage": {
                "integrity_errors": storage_errors,
                "stored_photo_rows": len(pipeline.store.load_all_photos()),
            },
            "same_photo_safety": {
                "violations": len(same_photo_violations),
                "details": same_photo_violations,
            },
            "folder_gt_resolution": {
                "method": "single-person/single-face exact anchors + conservative within-photo elimination; cluster mapping used only for photo-level set metrics",
                "exact_face_labels": len(exact_gt),
                "pipeline_faces": len(all_faces),
                "exact_face_label_coverage": len(exact_gt) / len(all_faces) if all_faces else 0.0,
                "exact_identities_covered": exact_identities,
                "total_identities": len(gallery.identities),
                "exact_face_labels_by_identity": dict(sorted(exact_identity_counts.items())),
                "exact_sources": dict(Counter(resolution.exact_source.values())),
                "anchor_clean_cluster_mappings": len(resolution.cluster_identity_map),
                "anchor_conflicted_clusters": resolution.conflicted_anchor_clusters,
                "unresolved_faces": len(resolution.unresolved_face_ids),
                "important_limitation": "Folder-only labels cannot independently detect a face swap between two known identities when both appear in the same group photo. Exact face-level metrics therefore use only independent anchors/elimination; group behavior is additionally measured at photo identity-set level.",
            },
            "exact_face_level_metrics": {
                "samples": len(exact_gt),
                "assignment_coverage": len(clustered_pred) / len(exact_gt) if exact_gt else 0.0,
                "contamination": exact_contamination,
                "fragmentation": exact_fragmentation,
                "clustered_only": exact_clustered,
                "end_to_end_deferred_as_singletons": exact_end_to_end,
                "exemplar_purity": exemplar_purity,
            },
            "photo_identity_set_metrics": photo_sets,
            "suggestions": suggestion_summary,
        }

        # Shareable, anonymized artifacts.
        write_json(share_dir / "final_report.json", report)
        write_csv(share_dir / "faces.csv", face_rows)
        write_csv(share_dir / "photos.csv", photo_rows)
        write_csv(share_dir / "clusters.csv", cluster_rows)
        write_csv(share_dir / "suggestions.csv", suggestion_rows)
        write_csv(share_dir / "problems.csv", problems)
        checkpoint_rows = []
        for cp in checkpoints:
            row = {
                "photos_processed": cp["photos_processed"],
                "kind": cp["kind"],
                "before_confirmed": cp["before"]["assignment_states"].get("confirmed", 0),
                "before_ambiguous": cp["before"]["assignment_states"].get("ambiguous", 0),
                "before_unassigned": cp["before"]["assignment_states"].get("unassigned", 0),
                "recovered_confirmed": cp["consolidation"].get("recovered_confirmed", 0),
                "new_clusters": cp["consolidation"].get("new_clusters", 0),
                "merge_suggestions_added": cp["consolidation"].get("merge_suggestions", 0),
                "split_suggestions_added": cp["consolidation"].get("split_suggestions", 0),
                "after_confirmed": cp["after"]["assignment_states"].get("confirmed", 0),
                "after_ambiguous": cp["after"]["assignment_states"].get("ambiguous", 0),
                "after_unassigned": cp["after"]["assignment_states"].get("unassigned", 0),
                "active_clusters": cp["after"]["active_clusters"],
                "pending_suggestions": cp["after"]["pending_suggestions"],
            }
            checkpoint_rows.append(row)
        write_csv(share_dir / "checkpoints.csv", checkpoint_rows)

        # Private local aids; never included in shareable_results.zip.
        write_csv(private_dir / "photo_paths.csv", private_photo_rows)
        write_csv(
            private_dir / "identity_map.csv",
            [
                {"identity_code": gallery.identity_codes[name], "folder_name": name}
                for name in gallery.identities
            ],
        )

        summary_lines = [
            "=== FINAL PRIVATE FACE-GROUPING TEST ===",
            f"Functional status:              {report['functional_status']}",
            f"Identities / unique photos:     {len(gallery.identities)} / {len(scheduled)}",
            f"Multi-person photos:            {multi_count}",
            f"Accepted pipeline faces:        {pipeline_faces_created}",
            f"Final assignment states:        {final_snapshot['assignment_states']}",
            f"Storage integrity errors:       {len(storage_errors)}",
            f"Same-photo violations:          {len(same_photo_violations)}",
            f"Second consolidation changed:   {second_changed}",
            f"Reprocess path idempotent:       {reprocess_idempotent}",
            f"Fresh clusters pruned:           {fresh_pruned}",
            "",
            "--- Exact folder-grounded face metrics ---",
            f"Exact GT faces:                  {len(exact_gt)} / {len(all_faces)} ({(len(exact_gt)/len(all_faces) if all_faces else 0):.2%})",
            f"Exact identities covered:        {exact_identities} / {len(gallery.identities)}",
            f"Assigned exact faces:            {len(clustered_pred)} / {len(exact_gt)} ({(len(clustered_pred)/len(exact_gt) if exact_gt else 0):.2%})",
            f"Contaminated clusters:           {exact_contamination['contaminated_clusters']}",
            f"Minority-face rate:              {exact_contamination['false_merge_face_rate']:.4%}",
            f"Fragmented identities:           {exact_fragmentation['fragmented_identities']} / {exact_fragmentation['identities']}",
            f"Micro identity recall:           {exact_fragmentation['micro_identity_recall']:.4%}",
            f"Clustered-only purity:           {exact_clustered['purity']['purity']:.4f}",
            f"Clustered-only B3 F1:            {exact_clustered['bcubed']['f1']:.4f}",
            f"Clustered-only pairwise F1:      {exact_clustered['pairwise']['f1']:.4f}",
            f"End-to-end B3 F1:                {exact_end_to_end['bcubed']['f1']:.4f}",
            f"End-to-end pairwise F1:          {exact_end_to_end['pairwise']['f1']:.4f}",
            f"Known exact exemplar purity:     {exemplar_purity['purity']:.4f} ({exemplar_purity['correct_vs_cluster_dominant']}/{exemplar_purity['known_exact_exemplars']})",
            "",
            "--- Photo-level identity-set metric (especially useful for group photos) ---",
            f"All-photo exact set match:       {photo_sets['all']['exact_set_match_rate']:.4%}",
            f"All-photo macro set recall:      {photo_sets['all']['macro_recall']:.4%}",
            f"Group-photo exact set match:     {photo_sets['multi_person_only']['exact_set_match_rate']:.4%}",
            f"Group-photo macro set recall:    {photo_sets['multi_person_only']['macro_recall']:.4%}",
            "",
            "--- Suggestions ---",
            f"Pending suggestions:             {suggestion_summary['total']} ({suggestion_summary['merge']} merge, {suggestion_summary['split']} split)",
            f"Suggestion burden:               {suggestion_summary['per_1000_photos']:.2f}/1000 photos",
            f"Dangerous anchor-known merges:   {suggestion_summary['dangerous_anchor_known_suggestions']}",
            "",
            "IMPORTANT: shareable_results.zip contains no image files, no embeddings, no absolute paths, and no raw folder/person names.",
            "Upload shareable_results.zip for analysis.",
        ]
        (share_dir / "SUMMARY.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        share_zip = output_dir / "shareable_results.zip"
        make_shareable_zip(share_dir, share_zip)

        print("\n=== FINAL SUMMARY ===")
        for line in summary_lines[:34]:
            print(line)
        print(f"\nShareable result bundle: {share_zip}")
        print(f"Private local diagnostics: {private_dir}")
        print("No images, embeddings, absolute paths, or raw identity-folder names are in the shareable ZIP.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
