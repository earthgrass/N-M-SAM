"""Static local Group-OBS mask selection with optional APBR tie-breaking."""

from .adapter import (
    APBR_SHUFFLE_SEEDS,
    DELETION_PAIRS,
    MAX_QDD_CONDITION,
    GroupSelectionResult,
    OBSSystem,
    apbr_pair_costs,
    prepare_obs_system,
    select_group_mask,
    static_group_obs_costs,
)

__all__ = [
    "APBR_SHUFFLE_SEEDS",
    "DELETION_PAIRS",
    "MAX_QDD_CONDITION",
    "GroupSelectionResult",
    "OBSSystem",
    "apbr_pair_costs",
    "prepare_obs_system",
    "select_group_mask",
    "static_group_obs_costs",
]
