"""
Similarity computation.

Implements Point 7's exemplar comparison method: compare a new face's
embedding against all exemplars in a candidate cluster (no sub-sampling),
then combine via top-k average (k=2 by default) rather than max (too
exploitable by a single high-similarity fluke) or full average (defeats
the pose/quality bucket diversity design from Point 6).
"""
from typing import List, Tuple

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def top_k_average_similarity(
    query_embedding: np.ndarray,
    exemplar_embeddings: List[np.ndarray],
    k: int = 2,
) -> Tuple[float, List[float]]:
    """
    Returns (top_k_average, all_similarities_sorted_desc).

    Compares query_embedding against every exemplar embedding (Point 7:
    "no sub-sampling"), sorts descending, and averages the top k. If
    fewer than k exemplars are available (e.g. a brand-new cluster with
    only 1-2 faces so far), averages over whatever is available instead
    of failing -- this is expected during a cluster's early life.
    """
    if not exemplar_embeddings:
        raise ValueError("exemplar_embeddings must contain at least one embedding.")

    similarities = [cosine_similarity(query_embedding, e) for e in exemplar_embeddings]
    similarities.sort(reverse=True)

    effective_k = min(k, len(similarities))
    top_k = similarities[:effective_k]
    top_k_avg = sum(top_k) / len(top_k)

    return top_k_avg, similarities
