#!/usr/bin/env python3
"""Diagnose why final Gallery photos were left in ``_ungrouped_photos``.

This is a READ-ONLY production diagnostic. It never opens the completed run
SQLite database through the production store directly. Instead it creates a
private SQLite backup, opens the backup through ``FaceGroupingPipeline``, and
uses the completed final clusters as an immutable scoring snapshot.

For every ungrouped source photo the tool re-runs only the visual front-end:

    detection -> detection/landmark association -> alignment -> quality

For accepted faces it reuses the embedding persisted by the completed run when
possible. For a hard-excluded face, it may compute a diagnostic-only embedding
so we can answer the useful question: "would this visually clear face match a
mature final cluster if the quality gate had not rejected it?". Such an
embedding is NEVER stored and NEVER changes clustering state.

It then scores each usable face against the FINAL active cluster snapshot using
exactly the production exemplar Top-K scoring/decision rules. A secondary
member-to-member bridge probe is also reported for diagnosis only; it is not a
production assignment rule.

Typical PowerShell usage from the project root::

    python tools/gallery_grouping/diagnose_ungrouped_photos.py `
      --run-output data/gallery_grouping_output

Outputs are written to::

    <run-output>/ungrouped_diagnostics/
        summary.json
        photo_summary.csv
        face_details.csv
        annotated/
        aligned_faces/

Primary diagnostic categories
-----------------------------
NO_DETECTION
NO_LANDMARK
HARD_EXCLUDED
RESTRICTED_POSE_UNRECOVERED
AMBIGUOUS_AFTER_FINAL_STATE
UNASSIGNED_AFTER_FINAL_STATE
FINAL_MATCH_WAS_RECOVERABLE_BUT_MISSED

Additional defensive categories may appear only for unexpected consistency or
runtime issues, e.g. ALIGNMENT_ERROR or PERSISTENCE_MISMATCH.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from face_grouping.clustering.data_types import Face
from face_grouping.matching.assignment import AssignmentState
from face_grouping.matching.similarity import cosine_similarity
from face_grouping.pipeline import FaceGroupingPipeline, _read_image_bgr


CONFIRMED_STATES = {AssignmentState.CONFIRMED, AssignmentState.MANUAL}

CATEGORY_NO_DETECTION = "NO_DETECTION"
CATEGORY_NO_LANDMARK = "NO_LANDMARK"
CATEGORY_HARD_EXCLUDED = "HARD_EXCLUDED"
CATEGORY_RESTRICTED_POSE = "RESTRICTED_POSE_UNRECOVERED"
CATEGORY_AMBIGUOUS_FINAL = "AMBIGUOUS_AFTER_FINAL_STATE"
CATEGORY_UNASSIGNED_FINAL = "UNASSIGNED_AFTER_FINAL_STATE"
CATEGORY_RECOVERABLE_MISSED = "FINAL_MATCH_WAS_RECOVERABLE_BUT_MISSED"
CATEGORY_ALIGNMENT_ERROR = "ALIGNMENT_ERROR"
CATEGORY_PERSISTENCE_MISMATCH = "PERSISTENCE_MISMATCH"
CATEGORY_INCONSISTENT_GROUPING = "INCONSISTENT_GROUPING_STATE"
CATEGORY_FRONTEND_ERROR = "FRONTEND_DIAGNOSTIC_ERROR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain why completed Gallery-run photos remained ungrouped."
    )
    parser.add_argument(
        "--run-output",
        default="data/gallery_grouping_output",
        help="Completed Gallery run directory (default: data/gallery_grouping_output).",
    )
    parser.add_argument(
        "--gallery",
        default=None,
        help=(
            "Optional Gallery override if the path recorded in summary.json moved. "
            "Usually unnecessary."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Diagnostic output directory. Default: <run-output>/ungrouped_diagnostics."
        ),
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Do not write annotated images or aligned face crops.",
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _optional(value):
    return "" if value is None else value


def _optional_float(value: Optional[float]):
    return "" if value is None else float(value)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sqlite_backup(source: Path, destination: Path) -> None:
    """Create a consistent private backup without mutating the source DB."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    src = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(str(destination))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def _load_cluster_folder_map(run_output: Path) -> Dict[str, str]:
    path = run_output / "clusters.csv"
    if not path.exists():
        return {}
    result: Dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cluster_id = (row.get("cluster_id") or "").strip()
            person_folder = (row.get("person_folder") or "").strip()
            if cluster_id:
                result[cluster_id] = person_folder
    return result


def _resolve_gallery(summary: dict, override: Optional[str]) -> Path:
    candidates: List[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    recorded = summary.get("gallery")
    if recorded:
        candidates.append(Path(str(recorded)).expanduser())
    candidates.append(PROJECT_ROOT / "data" / "Gallery")

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_dir():
            return resolved
    rendered = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not resolve the Gallery directory. Checked:\n  - " + rendered
    )


def _relative_photo_path(image_path: str, gallery_dir: Path) -> str:
    source = Path(image_path).resolve()
    try:
        return str(source.relative_to(gallery_dir))
    except ValueError:
        return str(source)


def _safe_stem(photo_id: str, source: Path) -> str:
    digest = hashlib.sha1(str(source).encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"{photo_id[:8]}_{digest}_{source.stem}"


def _write_image_unicode(path: Path, image_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() if path.suffix else ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"
        path = path.with_suffix(suffix)
    ok, encoded = cv2.imencode(suffix, image_bgr)
    if not ok:
        raise RuntimeError(f"Could not encode diagnostic image: {path}")
    encoded.tofile(str(path))


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _match_persisted_face(detection_index: int, detection, persisted_faces: List[Face]) -> Optional[Face]:
    """Prefer the persisted face_index; fall back to a strong bbox IoU match."""
    for face in persisted_faces:
        if face.face_index == detection_index:
            return face

    det_bbox = (float(detection.x), float(detection.y), float(detection.x2), float(detection.y2))
    best = None
    best_iou = 0.0
    for face in persisted_faces:
        if None in (face.bbox_x1, face.bbox_y1, face.bbox_x2, face.bbox_y2):
            continue
        face_bbox = (
            float(face.bbox_x1), float(face.bbox_y1),
            float(face.bbox_x2), float(face.bbox_y2),
        )
        iou = _bbox_iou(det_bbox, face_bbox)
        if iou > best_iou:
            best_iou = iou
            best = face
    return best if best_iou >= 0.75 else None


def _build_member_index(pipeline: FaceGroupingPipeline):
    active_clusters = pipeline.store.load_all_clusters(include_merged=False)
    index = {}
    for cluster in active_clusters:
        members = [
            face
            for face in pipeline.store.load_faces_by_cluster(cluster.cluster_id)
            if face.assignment_state in CONFIRMED_STATES
        ]
        index[cluster.cluster_id] = members
    return active_clusters, index


def _best_member_bridge(
    embedding: np.ndarray,
    member_index: Dict[str, List[Face]],
    pipeline: FaceGroupingPipeline,
    cluster_to_folder: Dict[str, str],
) -> dict:
    best_score = None
    best_face = None
    best_cluster_id = ""
    for cluster_id, members in member_index.items():
        for member in members:
            score = cosine_similarity(embedding, member.embedding)
            if best_score is None or score > best_score:
                best_score = score
                best_face = member
                best_cluster_id = cluster_id

    best_photo = None
    if best_face is not None and best_face.photo_id:
        best_photo = pipeline.store.load_photo(best_face.photo_id)

    return {
        "best_member_cluster_id": best_cluster_id,
        "best_member_person_folder": cluster_to_folder.get(best_cluster_id, ""),
        "best_member_similarity": _optional_float(best_score),
        "best_member_face_id": best_face.face_id if best_face is not None else "",
        "best_member_photo": best_photo.image_path if best_photo is not None else "",
    }


def _score_final_snapshot(
    *,
    pipeline: FaceGroupingPipeline,
    face: Face,
    exemplar_eligible: bool,
    active_clusters,
    cluster_to_folder: Dict[str, str],
    member_index: Dict[str, List[Face]],
) -> dict:
    candidates = pipeline.incremental_assigner.score_clusters(face, active_clusters)
    decision = pipeline.incremental_assigner.evaluate_face(
        face,
        active_clusters,
        exemplar_eligible=exemplar_eligible,
    )
    best = candidates[0] if candidates else None
    second = candidates[1] if len(candidates) > 1 else None

    row = {
        "final_best_cluster_id": best.cluster_id if best else "",
        "final_best_person_folder": cluster_to_folder.get(best.cluster_id, "") if best else "",
        "final_best_score": _optional_float(best.score if best else None),
        "final_best_threshold": _optional_float(best.effective_threshold if best else None),
        "final_second_cluster_id": second.cluster_id if second else "",
        "final_second_person_folder": cluster_to_folder.get(second.cluster_id, "") if second else "",
        "final_second_score": _optional_float(second.score if second else None),
        "final_score_margin": _optional_float(
            (best.score - second.score) if best is not None and second is not None else None
        ),
        "final_decision_state": decision.state.value,
        "final_decision_reason": decision.reason,
        "final_decision_create_new_cluster": bool(decision.create_new_cluster),
        "final_decision_assigned_cluster_id": decision.assigned_cluster_id or "",
        "final_decision_assigned_person_folder": cluster_to_folder.get(
            decision.assigned_cluster_id or "", ""
        ),
    }
    row.update(_best_member_bridge(face.embedding, member_index, pipeline, cluster_to_folder))
    return row


def _classification_for_persisted(face: Face, final_eval: dict) -> str:
    final_state = final_eval.get("final_decision_state")
    final_assigned = final_eval.get("final_decision_assigned_cluster_id")
    if final_state == AssignmentState.CONFIRMED.value and final_assigned:
        return CATEGORY_RECOVERABLE_MISSED
    if face.assignment_state == AssignmentState.AMBIGUOUS:
        return CATEGORY_AMBIGUOUS_FINAL
    if face.assignment_state == AssignmentState.UNASSIGNED:
        return CATEGORY_UNASSIGNED_FINAL
    return CATEGORY_INCONSISTENT_GROUPING


def _photo_primary_reason(face_rows: List[dict]) -> str:
    if not face_rows:
        return CATEGORY_NO_DETECTION
    priority = [
        CATEGORY_RECOVERABLE_MISSED,
        CATEGORY_HARD_EXCLUDED,
        CATEGORY_NO_LANDMARK,
        CATEGORY_AMBIGUOUS_FINAL,
        CATEGORY_UNASSIGNED_FINAL,
        CATEGORY_ALIGNMENT_ERROR,
        CATEGORY_PERSISTENCE_MISMATCH,
        CATEGORY_INCONSISTENT_GROUPING,
        CATEGORY_FRONTEND_ERROR,
        CATEGORY_NO_DETECTION,
    ]
    present = {str(row.get("diagnostic_category") or "") for row in face_rows}
    for category in priority:
        if category in present:
            return category
    return sorted(present)[0] if present else CATEGORY_FRONTEND_ERROR


def _annotation_label(row: dict) -> str:
    category = str(row.get("diagnostic_category") or "")
    score = row.get("final_best_score")
    person = str(row.get("final_best_person_folder") or "")
    if score not in (None, "") and person:
        return f"{category} | {person} {float(score):.3f}"
    return category


def _annotate(image_bgr: np.ndarray, rows: List[dict]) -> np.ndarray:
    out = image_bgr.copy()
    for row in rows:
        if row.get("bbox_x1") in (None, ""):
            continue
        x1 = int(round(float(row["bbox_x1"])))
        y1 = int(round(float(row["bbox_y1"])))
        x2 = int(round(float(row["bbox_x2"])))
        y2 = int(round(float(row["bbox_y2"])))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = _annotation_label(row)
        y_text = max(18, y1 - 7)
        cv2.putText(
            out,
            label[:90],
            (max(0, x1), y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def main() -> int:
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    run_output = Path(args.run_output).expanduser().resolve()
    if not run_output.exists() or not run_output.is_dir():
        raise FileNotFoundError(f"Run output does not exist: {run_output}")

    source_db = run_output / "gallery_grouping.db"
    if not source_db.exists():
        raise FileNotFoundError(f"Completed run database not found: {source_db}")

    summary = _read_json(run_output / "summary.json")
    gallery_dir = _resolve_gallery(summary, args.gallery)
    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else run_output / "ungrouped_diagnostics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = output_dir / "annotated"
    aligned_dir = output_dir / "aligned_faces"

    cluster_to_folder = _load_cluster_folder_map(run_output)

    # Work exclusively on a private DB snapshot because pipeline construction
    # may run benign repair/migration checks. The completed production run is
    # therefore never modified by this diagnostic.
    temp_root = Path(tempfile.mkdtemp(prefix="face_grouping_ungrouped_diag_"))
    temp_db = temp_root / "diagnostic_snapshot.db"
    _sqlite_backup(source_db, temp_db)

    all_face_rows: List[dict] = []
    photo_rows: List[dict] = []

    print("=== UNGROUPED PHOTO DIAGNOSTIC ===")
    print(f"Run output:              {run_output}")
    print(f"Gallery:                 {gallery_dir}")
    print(f"Diagnostic output:       {output_dir}")
    print("Production DB mutation:  NO (private SQLite snapshot)")
    print()

    try:
        with FaceGroupingPipeline(str(temp_db)) as pipeline:
            active_clusters, member_index = _build_member_index(pipeline)
            active_cluster_ids = {cluster.cluster_id for cluster in active_clusters}

            all_photos = pipeline.store.load_all_photos()
            ungrouped = []
            for photo in all_photos:
                faces = pipeline.store.load_faces_by_photo(photo.photo_id)
                grouped = any(
                    face.assignment_state in CONFIRMED_STATES
                    and face.cluster_id in active_cluster_ids
                    for face in faces
                )
                if not grouped:
                    ungrouped.append((photo, faces))

            print(f"Ungrouped photos found:  {len(ungrouped)}")
            print(f"Final active clusters:   {len(active_clusters)}")
            print()

            for photo_number, (photo, persisted_faces) in enumerate(ungrouped, start=1):
                source = Path(photo.image_path)
                if not source.exists():
                    # If absolute DB path moved, use its Gallery-relative tail when possible.
                    fallback = gallery_dir / source.name
                    if fallback.exists():
                        source = fallback

                photo_face_rows: List[dict] = []
                if not source.exists():
                    row = {
                        "photo": _relative_photo_path(photo.image_path, gallery_dir),
                        "photo_id": photo.photo_id,
                        "detection_index": "",
                        "diagnostic_category": CATEGORY_FRONTEND_ERROR,
                        "diagnostic_note": "source_image_not_found",
                    }
                    photo_face_rows.append(row)
                    all_face_rows.append(row)
                    photo_rows.append(
                        {
                            "photo": row["photo"],
                            "photo_id": photo.photo_id,
                            "primary_reason": CATEGORY_FRONTEND_ERROR,
                            "detections": 0,
                            "persisted_faces": len(persisted_faces),
                            "no_landmark_faces": 0,
                            "hard_excluded_faces": 0,
                            "ambiguous_final_faces": 0,
                            "unassigned_final_faces": 0,
                            "recoverable_missed_faces": 0,
                            "best_final_person_folder": "",
                            "best_final_score": "",
                            "best_member_person_folder": "",
                            "best_member_similarity": "",
                        }
                    )
                    continue

                image_bgr = _read_image_bgr(str(source))
                if image_bgr is None:
                    raise RuntimeError(f"Could not decode source image: {source}")
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                try:
                    detections = pipeline.detector.detect(image_rgb)
                except Exception as exc:
                    detections = []
                    row = {
                        "photo": _relative_photo_path(str(source), gallery_dir),
                        "photo_id": photo.photo_id,
                        "detection_index": "",
                        "diagnostic_category": CATEGORY_FRONTEND_ERROR,
                        "diagnostic_note": f"detector_error:{type(exc).__name__}:{exc}",
                    }
                    photo_face_rows.append(row)
                    all_face_rows.append(row)

                if detections:
                    try:
                        landmarks_by_detection = pipeline.landmarker.detect_for_detections(
                            image_rgb, detections
                        )
                    except Exception as exc:
                        landmarks_by_detection = [None] * len(detections)
                        landmark_global_error = f"landmarker_error:{type(exc).__name__}:{exc}"
                    else:
                        landmark_global_error = ""

                    for detection_index, (det, landmarks) in enumerate(
                        zip(detections, landmarks_by_detection)
                    ):
                        persisted = _match_persisted_face(
                            detection_index, det, persisted_faces
                        )
                        base = {
                            "photo": _relative_photo_path(str(source), gallery_dir),
                            "photo_id": photo.photo_id,
                            "detection_index": detection_index,
                            "detection_score": float(det.confidence),
                            "bbox_x1": float(det.x),
                            "bbox_y1": float(det.y),
                            "bbox_x2": float(det.x2),
                            "bbox_y2": float(det.y2),
                            "bbox_width": float(det.width),
                            "bbox_height": float(det.height),
                            "persisted_face_id": persisted.face_id if persisted else "",
                            "persisted_assignment_state": (
                                persisted.assignment_state.value if persisted else ""
                            ),
                            "persisted_candidate_cluster_id": (
                                persisted.candidate_cluster_id or "" if persisted else ""
                            ),
                            "persisted_candidate_person_folder": (
                                cluster_to_folder.get(persisted.candidate_cluster_id or "", "")
                                if persisted else ""
                            ),
                            "persisted_best_match_score": _optional_float(
                                persisted.best_match_score if persisted else None
                            ),
                            "persisted_second_best_score": _optional_float(
                                persisted.second_best_score if persisted else None
                            ),
                            "persisted_score_margin": _optional_float(
                                persisted.score_margin if persisted else None
                            ),
                            "persisted_decision_threshold": _optional_float(
                                persisted.decision_threshold if persisted else None
                            ),
                            "persisted_decision_reason": persisted.decision_reason or "" if persisted else "",
                            "persisted_recognition_restricted": (
                                bool(persisted.recognition_restricted) if persisted else False
                            ),
                            "persisted_recognition_restriction_reason": (
                                persisted.recognition_restriction_reason or "" if persisted else ""
                            ),
                            "landmark_status": "",
                            "alignment_status": "",
                            "size_score": "",
                            "blur_score": "",
                            "pose_score": "",
                            "eye_openness_score": "",
                            "quality_score": "",
                            "yaw_ratio": "",
                            "pitch_ratio": "",
                            "hard_excluded": "",
                            "hard_exclusion_reason": "",
                            "exemplar_eligible": "",
                            "diagnostic_embedding_computed_for_hard_excluded": False,
                            "diagnostic_category": "",
                            "diagnostic_note": "",
                        }

                        if landmarks is None:
                            base["landmark_status"] = "NO_LANDMARK"
                            base["alignment_status"] = "NOT_RUN"
                            base["diagnostic_category"] = CATEGORY_NO_LANDMARK
                            base["diagnostic_note"] = landmark_global_error or "no_unique_landmark_candidate_for_detection"
                            photo_face_rows.append(base)
                            all_face_rows.append(base)
                            continue

                        base["landmark_status"] = "MATCHED"
                        try:
                            aligned = pipeline._align_face(image_rgb, landmarks)
                        except Exception as exc:
                            base["alignment_status"] = "ERROR"
                            base["diagnostic_category"] = CATEGORY_ALIGNMENT_ERROR
                            base["diagnostic_note"] = f"{type(exc).__name__}:{exc}"
                            photo_face_rows.append(base)
                            all_face_rows.append(base)
                            continue

                        base["alignment_status"] = "OK"
                        quality = pipeline._compute_face_quality(det, landmarks, aligned)
                        base.update(
                            {
                                "size_score": float(quality.size_score),
                                "blur_score": float(quality.blur_score),
                                "pose_score": float(quality.pose_score),
                                "eye_openness_score": float(quality.eye_openness_score),
                                "quality_score": float(quality.quality_score),
                                "yaw_ratio": float(quality.yaw_ratio),
                                "pitch_ratio": float(quality.pitch_ratio),
                                "hard_excluded": bool(quality.hard_excluded),
                                "hard_exclusion_reason": quality.hard_exclusion_reason,
                                "recognition_restricted": bool(quality.recognition_restricted),
                                "recognition_restriction_reason": quality.recognition_restriction_reason,
                                "exemplar_eligible": bool(quality.exemplar_eligible),
                            }
                        )

                        if not args.no_images:
                            aligned_bgr = cv2.cvtColor(aligned.image, cv2.COLOR_RGB2BGR)
                            crop_path = aligned_dir / (
                                f"{_safe_stem(photo.photo_id, source)}__face_{detection_index:02d}.jpg"
                            )
                            _write_image_unicode(crop_path, aligned_bgr)
                            base["aligned_crop"] = str(crop_path)
                        else:
                            base["aligned_crop"] = ""

                        embedding = None
                        embedding_source = ""
                        if persisted is not None:
                            embedding = persisted.embedding
                            embedding_source = "persisted_final_run"
                        elif quality.hard_excluded:
                            # Diagnostic-only probe. Production correctly skipped
                            # this embedding because the hard gate fired.
                            try:
                                embedding = pipeline.embedder.embed(aligned.image)
                                embedding_source = "diagnostic_only_hard_excluded"
                                base["diagnostic_embedding_computed_for_hard_excluded"] = True
                            except Exception as exc:
                                base["diagnostic_note"] = (
                                    f"diagnostic_embedding_error:{type(exc).__name__}:{exc}"
                                )

                        base["embedding_source"] = embedding_source

                        if embedding is not None:
                            scoring_face = persisted or Face(
                                face_id=f"diagnostic_{uuid.uuid4().hex}",
                                embedding=embedding,
                                quality_score=float(quality.quality_score),
                                yaw_ratio=float(quality.yaw_ratio),
                                created_at=datetime.utcnow(),
                                assignment_state=AssignmentState.UNASSIGNED,
                                photo_id=photo.photo_id,
                                face_index=detection_index,
                                bbox_x1=float(det.x),
                                bbox_y1=float(det.y),
                                bbox_x2=float(det.x2),
                                bbox_y2=float(det.y2),
                                detection_score=float(det.confidence),
                                embedding_model_version=pipeline.embedding_model_version,
                                config_version=pipeline.config_version,
                            )
                            base.update(
                                _score_final_snapshot(
                                    pipeline=pipeline,
                                    face=scoring_face,
                                    exemplar_eligible=bool(quality.exemplar_eligible),
                                    active_clusters=active_clusters,
                                    cluster_to_folder=cluster_to_folder,
                                    member_index=member_index,
                                )
                            )

                        if quality.hard_excluded:
                            if quality.recognition_restricted:
                                base["diagnostic_category"] = CATEGORY_RESTRICTED_POSE
                                base["diagnostic_note"] = (
                                    persisted.decision_reason
                                    if persisted is not None and persisted.decision_reason
                                    else "pose_only_restriction_not_recovered"
                                )
                            else:
                                base["diagnostic_category"] = CATEGORY_HARD_EXCLUDED
                            if (
                                base.get("diagnostic_category") == CATEGORY_HARD_EXCLUDED
                                and base.get("final_decision_state") == AssignmentState.CONFIRMED.value
                                and base.get("final_decision_assigned_cluster_id")
                            ):
                                base["diagnostic_note"] = (
                                    "hard_gate_blocked_face_that_would_match_final_existing_cluster"
                                )
                            elif not base.get("diagnostic_note"):
                                base["diagnostic_note"] = "production_hard_quality_gate"
                        elif persisted is None:
                            base["diagnostic_category"] = CATEGORY_PERSISTENCE_MISMATCH
                            base["diagnostic_note"] = (
                                "front_end_accepts_detection_now_but_completed_run_has_no_persisted_face"
                            )
                        else:
                            base["diagnostic_category"] = _classification_for_persisted(
                                persisted, base
                            )
                            if base["diagnostic_category"] == CATEGORY_RECOVERABLE_MISSED:
                                base["diagnostic_note"] = (
                                    "final_cluster_snapshot_now_confirms_existing_cluster_but_face_remained_deferred"
                                )

                        photo_face_rows.append(base)
                        all_face_rows.append(base)

                if not detections and not photo_face_rows:
                    row = {
                        "photo": _relative_photo_path(str(source), gallery_dir),
                        "photo_id": photo.photo_id,
                        "detection_index": "",
                        "diagnostic_category": CATEGORY_NO_DETECTION,
                        "diagnostic_note": "face_detector_returned_zero_detections",
                    }
                    photo_face_rows.append(row)
                    all_face_rows.append(row)

                if not args.no_images and detections:
                    annotated = _annotate(image_bgr, photo_face_rows)
                    annotated_path = annotated_dir / (
                        f"{_safe_stem(photo.photo_id, source)}__annotated.jpg"
                    )
                    _write_image_unicode(annotated_path, annotated)

                best_final = None
                best_member = None
                for row in photo_face_rows:
                    if row.get("final_best_score") not in (None, ""):
                        if best_final is None or float(row["final_best_score"]) > float(best_final["final_best_score"]):
                            best_final = row
                    if row.get("best_member_similarity") not in (None, ""):
                        if best_member is None or float(row["best_member_similarity"]) > float(best_member["best_member_similarity"]):
                            best_member = row

                counts = Counter(row.get("diagnostic_category") for row in photo_face_rows)
                photo_rows.append(
                    {
                        "photo": _relative_photo_path(str(source), gallery_dir),
                        "photo_id": photo.photo_id,
                        "primary_reason": _photo_primary_reason(photo_face_rows),
                        "detections": len(detections),
                        "persisted_faces": len(persisted_faces),
                        "no_landmark_faces": int(counts.get(CATEGORY_NO_LANDMARK, 0)),
                        "hard_excluded_faces": int(counts.get(CATEGORY_HARD_EXCLUDED, 0)),
                        "ambiguous_final_faces": int(counts.get(CATEGORY_AMBIGUOUS_FINAL, 0)),
                        "unassigned_final_faces": int(counts.get(CATEGORY_UNASSIGNED_FINAL, 0)),
                        "recoverable_missed_faces": int(counts.get(CATEGORY_RECOVERABLE_MISSED, 0)),
                        "best_final_person_folder": best_final.get("final_best_person_folder", "") if best_final else "",
                        "best_final_score": best_final.get("final_best_score", "") if best_final else "",
                        "best_member_person_folder": best_member.get("best_member_person_folder", "") if best_member else "",
                        "best_member_similarity": best_member.get("best_member_similarity", "") if best_member else "",
                    }
                )

                print(
                    f"[{photo_number:02d}/{len(ungrouped):02d}] "
                    f"{_relative_photo_path(str(source), gallery_dir)} -> "
                    f"{photo_rows[-1]['primary_reason']}"
                )

            category_counts = Counter(
                str(row.get("diagnostic_category") or "") for row in all_face_rows
            )
            primary_reason_counts = Counter(row["primary_reason"] for row in photo_rows)

            summary_payload = {
                "test": "ungrouped_photo_root_cause_diagnostic_v1",
                "run_output": str(run_output),
                "gallery": str(gallery_dir),
                "diagnostic_output": str(output_dir),
                "production_db_modified": False,
                "final_active_clusters": len(active_clusters),
                "ungrouped_photos": len(photo_rows),
                "diagnostic_rows": len(all_face_rows),
                "photo_primary_reason_counts": dict(sorted(primary_reason_counts.items())),
                "face_diagnostic_category_counts": dict(sorted(category_counts.items())),
                "recoverable_missed_faces": int(category_counts.get(CATEGORY_RECOVERABLE_MISSED, 0)),
                "hard_excluded_faces_that_would_match_existing_final_cluster": sum(
                    1
                    for row in all_face_rows
                    if row.get("diagnostic_category") == CATEGORY_HARD_EXCLUDED
                    and row.get("final_decision_state") == AssignmentState.CONFIRMED.value
                    and row.get("final_decision_assigned_cluster_id")
                ),
                "notes": [
                    "The completed production DB is never mutated; analysis uses a private SQLite backup.",
                    "Final exemplar scoring uses the same production Top-K and margin decision logic against the completed active clusters.",
                    "Member-to-member best similarity is diagnostic evidence only and is not used to assign or correct faces.",
                    "Hard-excluded faces may receive a diagnostic-only embedding solely to test whether the quality gate blocked an otherwise strong final identity match.",
                    "NO_LANDMARK means the image-wide one-to-one detection-landmark association could not provide a reliable independent landmark mesh for that detection.",
                ],
            }

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    face_fields = [
        "photo", "photo_id", "detection_index", "diagnostic_category", "diagnostic_note",
        "detection_score", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_width", "bbox_height",
        "landmark_status", "alignment_status", "size_score", "blur_score", "pose_score",
        "eye_openness_score", "quality_score", "yaw_ratio", "pitch_ratio", "hard_excluded",
        "hard_exclusion_reason", "recognition_restricted",
        "recognition_restriction_reason", "exemplar_eligible", "aligned_crop",
        "diagnostic_embedding_computed_for_hard_excluded", "embedding_source",
        "persisted_face_id", "persisted_assignment_state", "persisted_candidate_person_folder",
        "persisted_candidate_cluster_id", "persisted_best_match_score", "persisted_second_best_score",
        "persisted_score_margin", "persisted_decision_threshold", "persisted_decision_reason",
        "persisted_recognition_restricted", "persisted_recognition_restriction_reason",
        "final_best_person_folder", "final_best_cluster_id", "final_best_score", "final_best_threshold",
        "final_second_person_folder", "final_second_cluster_id", "final_second_score", "final_score_margin",
        "final_decision_state", "final_decision_reason", "final_decision_create_new_cluster",
        "final_decision_assigned_person_folder", "final_decision_assigned_cluster_id",
        "best_member_person_folder", "best_member_cluster_id", "best_member_similarity",
        "best_member_face_id", "best_member_photo",
    ]
    normalized_face_rows = [
        {field: row.get(field, "") for field in face_fields}
        for row in all_face_rows
    ]
    _write_csv(output_dir / "face_details.csv", normalized_face_rows, face_fields)

    photo_fields = [
        "photo", "photo_id", "primary_reason", "detections", "persisted_faces",
        "no_landmark_faces", "hard_excluded_faces", "ambiguous_final_faces",
        "unassigned_final_faces", "recoverable_missed_faces",
        "best_final_person_folder", "best_final_score",
        "best_member_person_folder", "best_member_similarity",
    ]
    _write_csv(output_dir / "photo_summary.csv", photo_rows, photo_fields)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    print("\n=== DIAGNOSTIC SUMMARY ===")
    print(f"Ungrouped photos:         {summary_payload['ungrouped_photos']}")
    print(f"Recoverable-but-missed:   {summary_payload['recoverable_missed_faces']}")
    print(
        "Hard-excluded but final-matchable: "
        f"{summary_payload['hard_excluded_faces_that_would_match_existing_final_cluster']}"
    )
    for category, count in summary_payload["photo_primary_reason_counts"].items():
        print(f"  {category}: {count}")
    print(f"Photo summary:            {output_dir / 'photo_summary.csv'}")
    print(f"Face details:             {output_dir / 'face_details.csv'}")
    print(f"Summary:                  {output_dir / 'summary.json'}")
    if not args.no_images:
        print(f"Annotated images:         {annotated_dir}")
        print(f"Aligned crops:            {aligned_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
