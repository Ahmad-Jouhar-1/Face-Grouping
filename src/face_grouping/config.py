"""Small cached configuration helpers."""
import functools
import hashlib
import os

import yaml

THRESHOLDS_PATH = "configs/thresholds.yaml"
MODEL_PATHS_PATH = "configs/model_paths.yaml"


@functools.lru_cache(maxsize=1)
def load_thresholds() -> dict:
    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=1)
def load_model_paths() -> dict:
    with open(MODEL_PATHS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=1)
def get_embedding_model_version() -> str:
    cfg = load_model_paths()
    return str(cfg["embedding"]["model_version"])


@functools.lru_cache(maxsize=1)
def get_config_version() -> str:
    """Short content hash of the thresholds that shaped the decision."""
    with open(THRESHOLDS_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def normalize_image_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(path)))
