"""
Visibility classification (Point 15).

A live, recalculated property based purely on current face_count -- not
a one-time check at cluster creation. Callers should re-evaluate this
any time face_count changes (new photo assigned, correction applied,
merge executed), not cache it.
"""
from enum import Enum

from face_grouping.clustering.data_types import Cluster


class VisibilityState(Enum):
    MAIN_GRID = "main_grid"              # face_count >= 3: shown as a "Person"
    POSSIBLE_PERSON = "possible_person"  # face_count == 2: lower-visibility section
    HIDDEN = "hidden"                    # face_count <= 1: never surfaced


def get_visibility(cluster: Cluster) -> VisibilityState:
    if cluster.face_count >= 3:
        return VisibilityState.MAIN_GRID
    elif cluster.face_count == 2:
        return VisibilityState.POSSIBLE_PERSON
    else:
        return VisibilityState.HIDDEN