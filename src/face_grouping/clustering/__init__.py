"""Public clustering API with lazy imports to avoid storage/clustering cycles."""
from face_grouping.clustering.data_types import Face, Cluster, Photo, PhotoProcessingStatus

__all__ = [
    "Face", "Cluster", "Photo", "PhotoProcessingStatus",
    "run_hdbscan", "ConsolidationEngine", "ConsolidationAudit",
    "MergeCandidate", "SplitCandidate",
    "Suggestion", "SuggestionType", "SuggestionStatus",
    "build_merge_suggestions", "build_split_suggestions", "resolve_suggestion",
]


def __getattr__(name):
    if name == "run_hdbscan":
        from face_grouping.clustering.hdbscan_runner import run_hdbscan
        return run_hdbscan
    if name in {"ConsolidationEngine", "ConsolidationAudit"}:
        from face_grouping.clustering.consolidation import ConsolidationEngine, ConsolidationAudit
        return {"ConsolidationEngine": ConsolidationEngine, "ConsolidationAudit": ConsolidationAudit}[name]
    if name in {"MergeCandidate", "SplitCandidate"}:
        from face_grouping.clustering.candidates import MergeCandidate, SplitCandidate
        return {"MergeCandidate": MergeCandidate, "SplitCandidate": SplitCandidate}[name]
    if name in {
        "Suggestion", "SuggestionType", "SuggestionStatus",
        "build_merge_suggestions", "build_split_suggestions", "resolve_suggestion",
    }:
        from face_grouping.clustering.merge_rules import (
            Suggestion, SuggestionType, SuggestionStatus,
            build_merge_suggestions, build_split_suggestions, resolve_suggestion,
        )
        return {
            "Suggestion": Suggestion,
            "SuggestionType": SuggestionType,
            "SuggestionStatus": SuggestionStatus,
            "build_merge_suggestions": build_merge_suggestions,
            "build_split_suggestions": build_split_suggestions,
            "resolve_suggestion": resolve_suggestion,
        }[name]
    raise AttributeError(name)
