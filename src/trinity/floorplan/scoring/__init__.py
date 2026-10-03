"""Scoring: hard feasibility, the soft count ``V_rel``, the per-case cost and set totals,
plus the continuous metrics (``continuous``, ``severity``, ``vector``), the bookshelf
metrics and the distribution-level metrics (``descriptors``, ``distribution``).

Importing this package registers the ``full``, ``full_fast``, ``stub`` and ``continuous``
scorers in :data:`trinity.floorplan.registry.SCORER`.
"""

from trinity.floorplan.scoring import (
    continuous,  # noqa: F401  (register: continuous)
    cost,  # noqa: F401  (register: full, full_fast, stub)
)
from trinity.floorplan.scoring.continuous import ContinuousScore, score_continuous
from trinity.floorplan.scoring.cost import (
    CaseScore,
    compute_cost,
    hpwl,
    hpwl_fast,
    use_fast_hpwl,
)
from trinity.floorplan.scoring.feasibility import FeasibilityReport, check_feasibility
from trinity.floorplan.scoring.severity import SeverityScore, score_severity
from trinity.floorplan.scoring.soft import SoftReport, check_soft
from trinity.floorplan.scoring.total import total_blockcount, total_exp, total_mean
from trinity.floorplan.scoring.validate import ValidationResult, validate

__all__ = [
    "CaseScore",
    "ContinuousScore",
    "SeverityScore",
    "score_severity",
    "score_continuous",
    "compute_cost",
    "hpwl",
    "hpwl_fast",
    "use_fast_hpwl",
    "FeasibilityReport",
    "check_feasibility",
    "SoftReport",
    "check_soft",
    "total_blockcount",
    "total_exp",
    "total_mean",
    "ValidationResult",
    "validate",
]
