"""Shared candidate types used by conservative consolidation and suggestions."""
from dataclasses import dataclass
from typing import Dict, List, Optional

# Mutual cross-cluster evidence required by the normal (human-review) merge audit.
CLEAN_MATCH_THRESHOLD = 0.90


@dataclass
class AutoMergeCandidate:
    """A structural auto-merge candidate backed by conservative evidence.

    ``mode`` distinguishes two intentionally separate auto-correction paths:

    * ``mutual_full_coverage``: both clusters explain every member of the other
      at the existing high-confidence threshold.
    * ``fragment_member_bridge``: a smaller history-created fragment is linked
      to a mature target by repeated member-to-member bridge evidence, even if
      the target's current five exemplars do not represent that appearance.

    No field here introduces a new face-similarity threshold. Similarity floors
    are derived from the existing matching configuration.
    """

    mode: str
    source_cluster_id: str
    target_cluster_id: str
    source_size: int
    target_size: int
    source_coverage: float
    strong_anchor_count: int
    min_target_score: float
    mean_target_score: float
    min_competition_margin: float

    # Optional mode-specific evidence, persisted in diagnostics/events.
    reverse_coverage: Optional[float] = None
    reverse_strong_anchor_count: int = 0
    member_bridge_source_coverage: Optional[float] = None
    high_conf_bridge_source_count: int = 0
    strong_bridge_source_count: int = 0
    min_member_support_count: int = 0
    distinct_bridge_target_faces: int = 0
    distinct_bridge_target_photos: int = 0


@dataclass
class MergeCandidate:
    new_label: int
    contributing_cluster_ids: List[str]
    fractions: Dict[str, float]
    total_members: int


@dataclass
class SplitCandidate:
    existing_cluster_id: str
    receiving_new_labels: List[int]
    fractions: Dict[int, float]
    # Exact raw face groups are stored so user-approved execution never
    # re-runs clustering and accidentally changes the reviewed proposal.
    face_groups: Optional[Dict[int, List[str]]] = None
