#!/usr/bin/env python3
"""Targeted post-run diagnostic for one gallery photo.

This tool reads an *existing* ``gallery_grouping_output`` run. It does not
modify the database and does not rerun clustering. For a selected source photo
and selected exported person folders it:

* reports similarity from every stored face in the target photo to every
  current exemplar in each selected cluster;
* computes the same top-k average used by production matching;
* exports aligned 112x112 crops for target faces and all compared exemplars;
* writes an annotated copy of the target photo for face-index verification.

Typical usage from the project root::

    python tools/gallery_grouping/diagnose_target_photo.py `
      --photo IMG-20240820-WA0010.jpg `
      --persons person_001 person_003

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
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from face_grouping.config import load_model_paths, load_thresholds
from face_grouping.matching.similarity import cosine_similarity
from face_grouping.storage.store import FaceGroupingStore

DB_FILENAME = "gallery_grouping.db"
CLUSTERS_FILENAME = "clusters.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain one gallery photo against selected person-cluster exemplars."
    )
    parser.add_argument(
        "--output",
        default="data/gallery_grouping_output",
        help="Existing gallery grouping output directory.",
    )
    parser.add_argument(
        "--photo",
        required=True,
        help="Target source photo basename or full path, e.g. IMG-20240820-WA0010.jpg.",
    )
    parser.add_argument(
        "--persons",
        nargs="+",
        required=True,
        help="Exported person folders to compare, e.g. person_001 person_003.",
    )
    parser.add_argument(
        "--diagnostic-dir",
        default="",
        help="Optional custom diagnostic directory. Defaults under the output run.",
    )
    return parser.parse_args()


def _read_image_bgr(path: Path) -> Optional[np.ndarray]:
    """Unicode-safe OpenCV image read on Windows."""
    try:
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _write_image(path: Path, image_bgr: np.ndarray) -> None:
    """Unicode-safe OpenCV image write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"
    ok, encoded = cv2.imencode(suffix, image_bgr)
    if not ok:
        raise RuntimeError(f"Could not encode image for {path}")
    path.write_bytes(encoded.tobytes())


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._") or "item"


def _load_person_map(clusters_csv: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    person_to_cluster: Dict[str, str] = {}
    cluster_to_person: Dict[str, str] = {}
    with clusters_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            person = str(row.get("person_folder", "")).strip()
            cluster = str(row.get("cluster_id", "")).strip()
            if person and cluster:
                person_to_cluster[person] = cluster
                cluster_to_person[cluster] = person
    return person_to_cluster, cluster_to_person


def _find_target_photo(store: FaceGroupingStore, query: str):
    query_path = Path(query)
    query_name = query_path.name.casefold()
    normalized_query = str(query_path).casefold()

    exact = []
    basename = []
    for photo in store.load_all_photos():
        stored = Path(photo.image_path)
        if str(stored).casefold() == normalized_query:
            exact.append(photo)
        if stored.name.casefold() == query_name:
            basename.append(photo)

    matches = exact or basename
    if not matches:
        raise FileNotFoundError(f"Photo not found in run database: {query}")
    if len(matches) > 1:
        candidates = "\n  ".join(p.image_path for p in matches)
        raise ValueError(
            f"Photo name is ambiguous; pass a fuller path. Candidates:\n  {candidates}"
        )
    return matches[0]


def _top_k_average(scores: List[float], k: int) -> float:
    if not scores:
        raise ValueError("No exemplar similarities available")
    ranked = sorted(scores, reverse=True)
    selected = ranked[: min(k, len(ranked))]
    return float(sum(selected) / len(selected))


def _stored_detection_from_face(face):
    """Recreate the exact detector box stored for a face in the completed run.

    The diagnostic must not rely on MediaPipe detection order during a later
    rerun: that order is not a stable identifier.  The production pipeline
    already persisted the original detector bbox for every accepted face, so
    use that bbox as the source of truth when recreating visual crops.
    """
    if None in (face.bbox_x1, face.bbox_y1, face.bbox_x2, face.bbox_y2):
        raise RuntimeError(
            f"Stored bbox is unavailable for face_id={face.face_id}"
        )

    from face_grouping.detection.detector import FaceDetection

    x1 = int(round(float(face.bbox_x1)))
    y1 = int(round(float(face.bbox_y1)))
    x2 = int(round(float(face.bbox_x2)))
    y2 = int(round(float(face.bbox_y2)))
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    confidence = (
        float(face.detection_score)
        if face.detection_score is not None
        else 1.0
    )
    return FaceDetection(
        x=x1,
        y=y1,
        width=width,
        height=height,
        confidence=confidence,
    )


def _extract_aligned_for_stored_face(
    image_path: Path,
    face,
    landmarker,
) -> Tuple[np.ndarray, object]:
    """Recreate an aligned crop from the bbox persisted in the original run."""
    image_bgr = _read_image_bgr(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    det = _stored_detection_from_face(face)
    landmarks = landmarker.detect_for_face(image_rgb, det)
    if landmarks is None:
        raise RuntimeError(
            f"Landmarker failed while recreating stored face_id={face.face_id} "
            f"(face_index={face.face_index}) in {image_path.name}"
        )
    from face_grouping.alignment.aligner import align_face
    aligned = align_face(image_rgb, landmarks)
    return cv2.cvtColor(aligned.image, cv2.COLOR_RGB2BGR), det


def _annotate_target(
    image_path: Path,
    faces: Iterable,
    cluster_to_person: Dict[str, str],
) -> np.ndarray:
    image = _read_image_bgr(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    for face in faces:
        if None in (face.bbox_x1, face.bbox_y1, face.bbox_x2, face.bbox_y2):
            continue
        x1, y1 = int(face.bbox_x1), int(face.bbox_y1)
        x2, y2 = int(face.bbox_x2), int(face.bbox_y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        person = cluster_to_person.get(face.cluster_id or "", "unassigned")
        label = f"face_{face.face_index}: {person}"
        cv2.putText(
            image,
            label,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return image


def _face_source(store: FaceGroupingStore, face_id: Optional[str]):
    if not face_id:
        return None, None
    face = store.load_face(face_id)
    if face is None or face.photo_id is None:
        return face, None
    return face, store.load_photo(face.photo_id)


def main() -> int:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()
    db_path = output_dir / DB_FILENAME
    clusters_csv = output_dir / CLUSTERS_FILENAME
    if not db_path.exists():
        raise FileNotFoundError(f"Existing run database not found: {db_path}")
    if not clusters_csv.exists():
        raise FileNotFoundError(f"Cluster map not found: {clusters_csv}")

    person_to_cluster, cluster_to_person = _load_person_map(clusters_csv)
    missing = [p for p in args.persons if p not in person_to_cluster]
    if missing:
        raise ValueError(
            f"Unknown person folder(s): {', '.join(missing)}. Check {clusters_csv.name}."
        )

    stem = _safe_name(Path(args.photo).stem)
    diagnostic_dir = (
        Path(args.diagnostic_dir).resolve()
        if args.diagnostic_dir
        else output_dir / "target_diagnostics" / stem
    )
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds()
    top_k = int(thresholds["matching"]["top_k"])
    model_paths = load_model_paths()

    # FaceGroupingStore performs schema housekeeping on open. Work on a snapshot
    # so the original completed run remains byte-for-byte untouched.
    snapshot_db = diagnostic_dir / "_diagnostic_snapshot.db"
    shutil.copy2(db_path, snapshot_db)
    store = FaceGroupingStore(str(snapshot_db))
    landmarker = None
    try:
        target_photo = _find_target_photo(store, args.photo)
        target_path = Path(target_photo.image_path)
        target_faces = sorted(
            store.load_faces_by_photo(target_photo.photo_id),
            key=lambda f: (999999 if f.face_index is None else f.face_index),
        )
        if not target_faces:
            raise RuntimeError(f"No accepted stored faces for target photo: {target_path}")

        selected_clusters = {
            person: store.load_cluster(person_to_cluster[person])
            for person in args.persons
        }
        for person, cluster in selected_clusters.items():
            if cluster is None:
                raise RuntimeError(f"Cluster for {person} no longer exists")

        similarity_rows: List[dict] = []
        face_summaries: List[dict] = []

        for target_face in target_faces:
            per_cluster = []
            for person, cluster in selected_clusters.items():
                exemplar_entries = []
                for bucket_name, bucket in (
                    ("quality", cluster.exemplar_set.quality_bucket),
                    ("pose", cluster.exemplar_set.pose_bucket),
                ):
                    for exemplar in bucket:
                        score = cosine_similarity(target_face.embedding, exemplar.embedding)
                        exemplar_face, exemplar_photo = _face_source(store, exemplar.face_id)
                        exemplar_entries.append(
                            {
                                "bucket": bucket_name,
                                "face_id": exemplar.face_id or "",
                                "quality_score": float(exemplar.quality_score),
                                "yaw_ratio": float(exemplar.yaw_ratio),
                                "similarity": float(score),
                                "source_photo": exemplar_photo.image_path if exemplar_photo else "",
                                "source_face_index": (
                                    ""
                                    if exemplar_face is None or exemplar_face.face_index is None
                                    else int(exemplar_face.face_index)
                                ),
                            }
                        )

                exemplar_entries.sort(key=lambda x: x["similarity"], reverse=True)
                for rank, entry in enumerate(exemplar_entries, start=1):
                    similarity_rows.append(
                        {
                            "target_photo": target_path.name,
                            "target_face_index": target_face.face_index,
                            "target_face_id": target_face.face_id,
                            "stored_assignment_state": target_face.assignment_state.value,
                            "stored_person_folder": cluster_to_person.get(target_face.cluster_id or "", ""),
                            "stored_best_match_score": target_face.best_match_score if target_face.best_match_score is not None else "",
                            "stored_second_best_score": target_face.second_best_score if target_face.second_best_score is not None else "",
                            "stored_score_margin": target_face.score_margin if target_face.score_margin is not None else "",
                            "stored_decision_threshold": target_face.decision_threshold if target_face.decision_threshold is not None else "",
                            "compared_person_folder": person,
                            "compared_cluster_id": cluster.cluster_id,
                            "exemplar_rank": rank,
                            "is_top_k": rank <= top_k,
                            "exemplar_bucket": entry["bucket"],
                            "exemplar_face_id": entry["face_id"],
                            "exemplar_quality_score": entry["quality_score"],
                            "exemplar_yaw_ratio": entry["yaw_ratio"],
                            "similarity": entry["similarity"],
                            "exemplar_source_photo": Path(entry["source_photo"]).name if entry["source_photo"] else "",
                            "exemplar_source_face_index": entry["source_face_index"],
                        }
                    )

                scores = [entry["similarity"] for entry in exemplar_entries]
                per_cluster.append(
                    {
                        "person_folder": person,
                        "cluster_id": cluster.cluster_id,
                        "exemplar_count": len(exemplar_entries),
                        "top_k": min(top_k, len(exemplar_entries)),
                        "top_k_average_similarity": _top_k_average(scores, top_k),
                        "max_similarity": max(scores),
                        "all_similarities_desc": scores,
                    }
                )

            per_cluster.sort(key=lambda x: x["top_k_average_similarity"], reverse=True)
            face_summaries.append(
                {
                    "face_index": target_face.face_index,
                    "face_id": target_face.face_id,
                    "stored_assignment_state": target_face.assignment_state.value,
                    "stored_person_folder": cluster_to_person.get(target_face.cluster_id or "", ""),
                    "quality_score": float(target_face.quality_score),
                    "yaw_ratio": float(target_face.yaw_ratio),
                    "stored_assignment_evidence": {
                        "best_match_score": target_face.best_match_score,
                        "second_best_score": target_face.second_best_score,
                        "score_margin": target_face.score_margin,
                        "decision_threshold": target_face.decision_threshold,
                        "decision_reason": target_face.decision_reason,
                    },
                    "current_selected_cluster_scores": per_cluster,
                }
            )

        # CSV evidence is useful even if crop recreation later fails.
        csv_fields = [
            "target_photo", "target_face_index", "target_face_id",
            "stored_assignment_state", "stored_person_folder",
            "stored_best_match_score", "stored_second_best_score",
            "stored_score_margin", "stored_decision_threshold",
            "compared_person_folder", "compared_cluster_id", "exemplar_rank",
            "is_top_k", "exemplar_bucket", "exemplar_face_id",
            "exemplar_quality_score", "exemplar_yaw_ratio", "similarity",
            "exemplar_source_photo", "exemplar_source_face_index",
        ]
        with (diagnostic_dir / "exemplar_similarities.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(similarity_rows)

        # Visual recreation uses the detector boxes persisted by the original
        # production run. Only the landmarker is rerun; IR-SE50 weights and a
        # second detector pass are intentionally not needed.
        from face_grouping.detection.landmarker import FaceLandmarkerWrapper
        landmarker = FaceLandmarkerWrapper(model_paths["mediapipe"]["face_landmarker"])

        annotated = _annotate_target(target_path, target_faces, cluster_to_person)
        _write_image(diagnostic_dir / "target_annotated.jpg", annotated)

        target_crop_dir = diagnostic_dir / "target_faces"
        for face in target_faces:
            if face.face_index is None:
                continue
            crop, _ = _extract_aligned_for_stored_face(
                target_path, face, landmarker
            )
            person = cluster_to_person.get(face.cluster_id or "", "unassigned")
            _write_image(
                target_crop_dir / f"face_{int(face.face_index):02d}__{_safe_name(person)}.jpg",
                crop,
            )

        exemplar_visuals: List[dict] = []
        for person, cluster in selected_clusters.items():
            person_dir = diagnostic_dir / "exemplars" / person
            seq = 0
            for bucket_name, bucket in (
                ("quality", cluster.exemplar_set.quality_bucket),
                ("pose", cluster.exemplar_set.pose_bucket),
            ):
                for exemplar in bucket:
                    seq += 1
                    ex_face, ex_photo = _face_source(store, exemplar.face_id)
                    record = {
                        "person_folder": person,
                        "bucket": bucket_name,
                        "exemplar_face_id": exemplar.face_id,
                        "source_photo": ex_photo.image_path if ex_photo else None,
                        "source_face_index": ex_face.face_index if ex_face else None,
                        "crop_file": None,
                        "crop_error": None,
                    }
                    try:
                        if ex_face is None or ex_photo is None or ex_face.face_index is None:
                            raise RuntimeError("Exemplar source face/photo metadata is unavailable")
                        crop, _ = _extract_aligned_for_stored_face(
                            Path(ex_photo.image_path), ex_face, landmarker
                        )
                        filename = (
                            f"{seq:02d}__{bucket_name}__face_{int(ex_face.face_index):02d}__"
                            f"{_safe_name(Path(ex_photo.image_path).stem)}.jpg"
                        )
                        _write_image(person_dir / filename, crop)
                        record["crop_file"] = str(Path("exemplars") / person / filename)
                    except Exception as exc:
                        record["crop_error"] = str(exc)
                    exemplar_visuals.append(record)

        report = {
            "test": "target_photo_exemplar_diagnostic_v2_stored_bbox",
            "run_output": str(output_dir),
            "target_photo": str(target_path),
            "selected_persons": args.persons,
            "top_k": top_k,
            "faces": face_summaries,
            "exemplar_visuals": exemplar_visuals,
            "notes": [
                "Similarity values use the stored face embeddings and the current final exemplar sets.",
                "The stored assignment evidence records the original decision at processing/recovery time.",
                "Aligned crops are recreated from the detector bounding boxes persisted by the original production run, then passed through the same MediaPipe landmarker and ArcFace alignment code.",
                "The diagnostic never relies on rerun detector ordering to identify a stored face.",
                "This tool does not modify clustering state or thresholds.",
            ],
        }
        with (diagnostic_dir / "diagnostic_summary.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("=== TARGET PHOTO DIAGNOSTIC ===")
        print(f"Photo:       {target_path}")
        print(f"Faces:       {len(target_faces)}")
        print(f"Compared:    {', '.join(args.persons)}")
        print(f"Output:      {diagnostic_dir}")
        print()
        for face in face_summaries:
            print(
                f"face_{face['face_index']}: stored={face['stored_person_folder']} "
                f"quality={face['quality_score']:.3f}"
            )
            for score in face["current_selected_cluster_scores"]:
                print(
                    f"  {score['person_folder']}: top-{score['top_k']} avg="
                    f"{score['top_k_average_similarity']:.4f}; max={score['max_similarity']:.4f}"
                )
        print()
        print(f"Similarity CSV: {diagnostic_dir / 'exemplar_similarities.csv'}")
        print(f"Summary:        {diagnostic_dir / 'diagnostic_summary.json'}")
        return 0
    finally:
        if landmarker is not None:
            landmarker.close()
        store.close()
        try:
            snapshot_db.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
