"""Small, defensive HDBSCAN wrapper used by consolidation."""
from typing import Optional, Sequence

import numpy as np
from sklearn.cluster import HDBSCAN


def run_hdbscan(
    embeddings: Sequence[np.ndarray],
    min_cluster_size: int = 2,
    min_samples: int = 1,
    item_ids: Optional[Sequence[str]] = None,
    allow_single_cluster: bool = False,
) -> np.ndarray:
    """Return one label per embedding; ``-1`` means noise.

    Stage 2 optionally accepts stable item IDs and fails fast if duplicate
    persisted faces were accidentally supplied. Inputs smaller than the
    requested cluster size are returned as noise instead of asking sklearn to
    fit an impossible clustering problem.
    """
    count = len(embeddings)
    if item_ids is not None:
        if len(item_ids) != count:
            raise ValueError("item_ids must be parallel to embeddings")
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("Duplicate item_ids are not allowed in HDBSCAN input")

    if count == 0:
        return np.array([], dtype=int)
    if count < min_cluster_size:
        return np.full(count, -1, dtype=int)

    matrix = np.stack([np.asarray(e) for e in embeddings])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / (norms + 1e-9)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        copy=True,
        allow_single_cluster=allow_single_cluster,
    )
    return clusterer.fit_predict(normalized)
