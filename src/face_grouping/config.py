"""Configuration and asset-path helpers for the face-grouping service."""
import functools
import hashlib
import os
from pathlib import Path
import yaml

# Runtime configuration must not depend on the backend process' current working
# directory. In the repository/editable-install layout, the project root is two
# levels above this module's package directory. Production deployments may mount
# the AI assets elsewhere and point FACE_GROUPING_ROOT at that directory.
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("FACE_GROUPING_ROOT", _DEFAULT_PROJECT_ROOT)).resolve()
THRESHOLDS_PATH = Path(
    os.environ.get("FACE_GROUPING_THRESHOLDS_PATH", PROJECT_ROOT / "configs" / "thresholds.yaml")
).resolve()
MODEL_PATHS_PATH = Path(
    os.environ.get("FACE_GROUPING_MODEL_PATHS_PATH", PROJECT_ROOT / "configs" / "model_paths.yaml")
).resolve()



@functools.lru_cache(maxsize=1)
def load_thresholds() -> dict:
    with THRESHOLDS_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=1)
def load_model_paths() -> dict:
    with MODEL_PATHS_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Only the known filesystem fields are resolved. Other strings such as
    # backbone/model_version remain untouched.
    for section, key in (
        ("mediapipe", "face_detector"),
        ("mediapipe", "face_landmarker"),
        ("embedding", "weights"),
    ):
        raw = cfg.get(section, {}).get(key)
        if raw:
            path = Path(raw)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            cfg[section][key] = str(path.resolve())
    return cfg


@functools.lru_cache(maxsize=1)
def get_embedding_model_version() -> str:
    cfg = load_model_paths()
    return str(cfg["embedding"]["model_version"])


@functools.lru_cache(maxsize=1)
def get_config_version() -> str:
    """Short content hash of the thresholds that shaped the decision."""
    with THRESHOLDS_PATH.open("rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def normalize_image_path(path: str) -> str:
    value = str(path).strip()
    # Opaque backend/object-store references are identifiers, not local paths.
    if "://" in value:
        return value
    return os.path.normcase(os.path.abspath(os.path.normpath(value)))
