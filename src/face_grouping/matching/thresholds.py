"""
Match decision logic.

Applies the T_match / ambiguous-band structure locked in Points 4-5 to a
top-k average similarity score (Point 7).

IMPORTANT: t_match and ambiguous_band_width never receive silent
fallback values. They must be explicitly configured from validated
IR-SE50 calibration; a half-configured system fails loudly rather than
silently making unsafe grouping decisions.
"""
from enum import Enum

from face_grouping.config import load_thresholds


class MatchDecision(Enum):
    CONFIDENT_MATCH = "confident_match"  # >= T_match: auto-assign silently
    AMBIGUOUS = "ambiguous"              # in the band: provisional, flagged for consolidation
    NO_MATCH = "no_match"                # below the band: confidently different person


def load_match_thresholds():
    """
    Reads t_match / ambiguous_band_width from configs/thresholds.yaml's
    matching section. If either value is unset, decide_match() raises
    instead of guessing a production matching threshold.
    """
    cfg = load_thresholds()["matching"]
    return cfg.get("t_match"), cfg.get("ambiguous_band_width")


def decide_match(
    top_k_avg_similarity: float,
    t_match: float,
    ambiguous_band_width: float,
) -> MatchDecision:
    if t_match is None or ambiguous_band_width is None:
        raise ValueError(
            "t_match and ambiguous_band_width are not set. Configure only "
            "values validated for the production IR-SE50 model; do not "
            "hardcode guessed values to bypass this check."
        )

    band_lower_bound = t_match - ambiguous_band_width

    if top_k_avg_similarity >= t_match:
        return MatchDecision.CONFIDENT_MATCH
    elif top_k_avg_similarity >= band_lower_bound:
        return MatchDecision.AMBIGUOUS
    else:
        return MatchDecision.NO_MATCH
