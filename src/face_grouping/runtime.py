"""Shared model runtime for production face-grouping sessions.

The heavy detector, landmarker, and embedding network are process-level
resources. User isolation belongs to the persistence/session layer, not to
model loading. A runtime can therefore be shared by many short-lived
``FaceGroupingPipeline`` sessions while SQLite state remains tenant-scoped.
"""
from pathlib import Path
from threading import RLock

from face_grouping.config import (
    get_config_version,
    get_embedding_model_version,
    load_model_paths,
    load_thresholds,
)


class FaceGroupingRuntime:
    """Own the heavyweight inference resources shared across users.

    MediaPipe task objects are treated conservatively as non-thread-safe.
    ``inference_lock`` serializes access inside one process. Horizontal
    throughput can be increased later by running multiple AI worker processes,
    each with its own runtime/model instances.
    """

    def __init__(self):
        self.cfg = load_thresholds()
        model_paths = load_model_paths()

        required_assets = {
            "MediaPipe face detector": model_paths["mediapipe"]["face_detector"],
            "MediaPipe face landmarker": model_paths["mediapipe"]["face_landmarker"],
            "IR-SE50 weights": model_paths["embedding"]["weights"],
        }
        missing = [f"{name}: {path}" for name, path in required_assets.items() if not Path(path).is_file()]
        if missing:
            details = "\n  - ".join(missing)
            raise FileNotFoundError(
                "Missing required face-grouping model assets:\n  - "
                + details
                + "\nSet FACE_GROUPING_ROOT (or explicit config path overrides) "
                  "to the deployed AI asset directory."
            )

        # Keep heavyweight imports after asset validation so startup failures are
        # explicit, and storage-only code can still be imported without loading
        # inference dependencies.
        from face_grouping.alignment.aligner import align_face
        from face_grouping.detection.detector import FaceDetectorWrapper
        from face_grouping.detection.landmarker import FaceLandmarkerWrapper
        from face_grouping.embedding.embedder import EmbedderWrapper
        from face_grouping.quality.gates import compute_face_quality

        self.embedding_model_version = get_embedding_model_version()
        self.config_version = get_config_version()

        self.align_face = align_face
        self.compute_face_quality = compute_face_quality
        self.detector = FaceDetectorWrapper(
            model_paths["mediapipe"]["face_detector"],
            confidence_threshold=self.cfg["detection"]["confidence_threshold"],
        )
        self.landmarker = FaceLandmarkerWrapper(
            model_paths["mediapipe"]["face_landmarker"]
        )
        self.embedder = EmbedderWrapper(model_paths["embedding"]["weights"])
        self.inference_lock = RLock()
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self.detector.close()
        self.landmarker.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
