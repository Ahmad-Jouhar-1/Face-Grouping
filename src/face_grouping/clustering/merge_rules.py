"""High-precision merge/split suggestion records (Stage 4).

The product policy is intentionally quiet: the system may *suggest* rare,
high-confidence merge/split corrections, but never executes either without
explicit user approval. Face moves are not suggestions at all; they are manual
user commands handled by the storage/pipeline layer.
"""
from dataclasses import dataclass, field
import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from face_grouping.clustering.candidates import MergeCandidate, SplitCandidate


class SuggestionType(Enum):
    MERGE = "merge"
    SPLIT = "split"


class SuggestionStatus(Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"
    STALE = "stale"  # internal: data changed before execution


@dataclass
class Suggestion:
    suggestion_id: str
    suggestion_type: SuggestionType
    cluster_ids: List[str]
    status: SuggestionStatus = SuggestionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    payload: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: datetime | None = None


def _digest_payload(prefix: str, payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def build_merge_suggestions(merge_candidates: List[MergeCandidate]) -> List[Suggestion]:
    suggestions = {}
    for candidate in merge_candidates:
        cluster_ids = sorted(set(candidate.contributing_cluster_ids))
        if len(cluster_ids) != 2:
            continue
        payload = {"cluster_ids": cluster_ids}
        suggestion = Suggestion(
            suggestion_id=_digest_payload("merge", payload),
            suggestion_type=SuggestionType.MERGE,
            cluster_ids=cluster_ids,
            payload=payload,
            evidence={
                "mutual_coverage": dict(candidate.fractions),
                "total_members": candidate.total_members,
            },
        )
        suggestions[suggestion.suggestion_id] = suggestion
    return list(suggestions.values())


def build_split_suggestions(split_candidates: List[SplitCandidate]) -> List[Suggestion]:
    suggestions = {}
    for candidate in split_candidates:
        if not candidate.face_groups:
            continue
        groups = [sorted(ids) for _, ids in sorted(candidate.face_groups.items())]
        if len(groups) != 2 or any(len(group) < 2 for group in groups):
            continue
        payload = {
            "source_cluster_id": candidate.existing_cluster_id,
            "groups": groups,
        }
        suggestion = Suggestion(
            suggestion_id=_digest_payload("split", payload),
            suggestion_type=SuggestionType.SPLIT,
            cluster_ids=[candidate.existing_cluster_id],
            payload=payload,
            evidence={"fractions": dict(candidate.fractions)},
        )
        suggestions[suggestion.suggestion_id] = suggestion
    return list(suggestions.values())


def resolve_suggestion(suggestion: Suggestion, status: SuggestionStatus) -> None:
    suggestion.status = status
    suggestion.updated_at = datetime.utcnow()
    if status in (SuggestionStatus.ACCEPTED, SuggestionStatus.REJECTED, SuggestionStatus.UNCERTAIN, SuggestionStatus.STALE):
        suggestion.resolved_at = suggestion.updated_at
