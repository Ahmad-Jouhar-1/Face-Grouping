#!/usr/bin/env python3
"""Visual smoke test for detector/landmark one-to-one association.

Runs only detection, landmarking and alignment on selected photos.  It does not
load IR-SE50, touch the gallery database, or run matching/clustering.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from face_grouping.config import load_model_paths, load_thresholds
from face_grouping.detection.detector import FaceDetectorWrapper
from face_grouping.detection.landmarker import FaceLandmarkerWrapper
from face_grouping.alignment.aligner import align_face


def _read(path: Path):
    try:
        data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def _write(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    path.write_bytes(encoded.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("photos", nargs="+", help="One or more image paths")
    parser.add_argument(
        "--output", default="data/landmark_association_check",
        help="Output folder relative to the project root",
    )
    args = parser.parse_args()

    model_paths = load_model_paths()
    cfg = load_thresholds()
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    detector = FaceDetectorWrapper(
        model_paths["mediapipe"]["face_detector"],
        confidence_threshold=cfg["detection"]["confidence_threshold"],
    )
    landmarker = FaceLandmarkerWrapper(model_paths["mediapipe"]["face_landmarker"])
    try:
        for photo_arg in args.photos:
            path = Path(photo_arg)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            image_bgr = _read(path)
            if image_bgr is None:
                raise FileNotFoundError(path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            detections = detector.detect(image_rgb)
            landmarks = landmarker.detect_for_detections(image_rgb, detections)

            stem_dir = output / path.stem
            stem_dir.mkdir(parents=True, exist_ok=True)
            annotated = image_bgr.copy()
            rows = []
            for i, (det, lm) in enumerate(zip(detections, landmarks)):
                cv2.rectangle(annotated, (det.x, det.y), (det.x2, det.y2), (255,255,255), 2)
                status = "matched" if lm is not None else "NO_LANDMARK"
                cv2.putText(annotated, f"face_{i}: {status}", (det.x, max(18, det.y-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
                row = {"face_index": i, "bbox": [det.x, det.y, det.x2, det.y2], "status": status}
                if lm is not None:
                    aligned = align_face(image_rgb, lm)
                    aligned_bgr = cv2.cvtColor(aligned.image, cv2.COLOR_RGB2BGR)
                    _write(stem_dir / f"face_{i:02d}.jpg", aligned_bgr)
                rows.append(row)
            _write(stem_dir / "annotated.jpg", annotated)
            (stem_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"{path.name}: detections={len(detections)}, matched={sum(x is not None for x in landmarks)}")
    finally:
        landmarker.close()
        detector.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
