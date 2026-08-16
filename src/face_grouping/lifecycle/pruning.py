"""
Cluster pruning (Point 16, simplified per explicit decision).

Clusters with face_count < 3 (never reached the visibility bar, Point
15) that haven't gained a new face in 30 days are hard-deleted. No
archive stage, no quality-based exceptions -- pure size + fixed time
window, per the user's explicit simplification decision. Clusters with
face_count >= 3 are never pruned, regardless of staleness.
"""
from datetime import datetime, timedelta
from typing import List

from face_grouping.clustering.data_types import Cluster

STALENESS_WINDOW = timedelta(days=30)
VISIBILITY_FACE_COUNT_THRESHOLD = 3  # matches Point 15's main-grid bar


def find_clusters_to_prune(clusters: List[Cluster], now: datetime) -> List[Cluster]:
    return [
        c for c in clusters
        if c.merged_into is None  # already-merged clusters aren't independently pruned here
        and c.face_count < VISIBILITY_FACE_COUNT_THRESHOLD
        and (now - c.last_updated_at) >= STALENESS_WINDOW
    ]