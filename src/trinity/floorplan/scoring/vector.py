"""The per-layout metric vector of the offline score caches.

The continuous metrics, the gaps to the reference layout and the severity bundle, in the
fixed ``COLS`` order.
"""

from trinity.floorplan.geometry import bbox_area
from trinity.floorplan.scoring.continuous import score_continuous
from trinity.floorplan.scoring.cost import GAP_FLOOR, gap_baselines, hpwl_fast
from trinity.floorplan.scoring.severity import score_severity
from trinity.floorplan.types import Placement

COLS = (
    "wl_norm",
    "compactness",
    "overlap_ratio",
    "boundary_dist",
    "group_gap",
    "mib_var",
    "fixed_dist",
    "score_pre",
    "hpwl_gap",
    "area_gap",
    "sev_group",
    "sev_mib",
    "sev_boundary",
    "sev_rel",
    "soft_cost",
)


def metric_vector(xywh, inst) -> list[float]:
    """The ``COLS`` values of one decoded layout
    ``xywh (n, 4)`` of instance ``inst``."""
    placement = Placement(xywh=xywh, instance=inst)
    cont = score_continuous(placement)
    hpwl_base, area_base = gap_baselines(placement, fast_hpwl=True)
    hpwl_gap = (hpwl_fast(placement) - hpwl_base) / max(hpwl_base, GAP_FLOOR)
    area_gap = (bbox_area(xywh) - area_base) / max(area_base, GAP_FLOOR)
    sev = score_severity(placement, hpwl_gap, area_gap, cont.overlap_ratio)
    return [
        cont.wl_norm,
        cont.compactness,
        cont.overlap_ratio,
        cont.boundary_dist,
        cont.group_gap,
        cont.mib_var,
        cont.fixed_dist,
        cont.score_pre,
        hpwl_gap,
        area_gap,
        sev.grouping,
        sev.mib,
        sev.boundary,
        sev.sev_rel,
        sev.soft_cost,
    ]
