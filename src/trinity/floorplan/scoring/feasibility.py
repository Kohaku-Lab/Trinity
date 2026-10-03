"""Hard-constraint feasibility checks.

A layout is *feasible* iff it has no overlap, every soft block meets its area within 1%,
every soft block's ``w/h`` lies within its aspect bounds (when the instance has them),
and -- when immutability is enforced -- every fixed-shape and preplaced block keeps its
target. Each check returns the violating blocks or pairs.
"""

from dataclasses import dataclass, field

import numpy as np

from trinity.floorplan.geometry import overlapping_pairs
from trinity.floorplan.types import Placement

AREA_TOLERANCE = 0.01
SHAPE_TOLERANCE = 1e-4
ASPECT_TOLERANCE = 1e-6


@dataclass
class FeasibilityReport:
    """Hard-constraint violations of one layout."""

    overlap_pairs: list[tuple[int, int]] = field(default_factory=list)
    area_violations: list[int] = field(default_factory=list)
    fixed_violations: list[int] = field(default_factory=list)
    preplaced_violations: list[int] = field(default_factory=list)
    aspect_violations: list[int] = field(default_factory=list)

    @property
    def overlap_count(self) -> int:
        return len(self.overlap_pairs)

    def is_feasible(self, enforce_immutability: bool = True) -> bool:
        """Whether no checked constraint is violated."""
        ok = (
            self.overlap_count == 0
            and not self.area_violations
            and not self.aspect_violations
        )
        if enforce_immutability:
            ok = ok and not self.fixed_violations and not self.preplaced_violations
        return ok


def check_area(placement: Placement, tolerance: float = AREA_TOLERANCE) -> list[int]:
    """Indices of soft blocks whose ``w*h`` deviates
    from target area by ``> tolerance``."""
    inst = placement.instance
    actual = placement.w * placement.h
    target = inst.area_targets
    soft = inst.is_soft & (target > 0)
    rel = np.abs(actual - target) / np.maximum(target, 1e-12)
    bad = soft & (rel > tolerance)
    return np.nonzero(bad)[0].tolist()


def check_fixed_shape(
    placement: Placement, tolerance: float = SHAPE_TOLERANCE
) -> list[int]:
    """Indices of fixed-shape blocks whose ``(w,h)`` drifted from the target shape."""
    inst = placement.instance
    if inst.target_positions is None:
        return []
    bad = []
    for i in np.nonzero(inst.is_fixed)[0]:
        tw, th = inst.target_positions[i, 2], inst.target_positions[i, 3]
        if tw < 0 or th < 0:
            continue
        if abs(placement.w[i] - tw) > tolerance or abs(placement.h[i] - th) > tolerance:
            bad.append(int(i))
    return bad


def check_aspect(
    placement: Placement, tolerance: float = ASPECT_TOLERANCE
) -> list[int]:
    """Indices of soft blocks whose ``w/h`` leaves their
    ``aspect_bounds`` (relative ``tolerance``)."""
    inst = placement.instance
    if inst.aspect_bounds is None:
        return []
    lo, hi = inst.aspect_bounds[:, 0], inst.aspect_bounds[:, 1]
    bounded = inst.is_soft & (lo > 0) & (hi > 0)
    ratio = placement.w / np.maximum(placement.h, 1e-12)
    bad = bounded & ((ratio < lo * (1 - tolerance)) | (ratio > hi * (1 + tolerance)))
    return np.nonzero(bad)[0].tolist()


def check_preplaced(
    placement: Placement, tolerance: float = SHAPE_TOLERANCE
) -> list[int]:
    """Indices of preplaced blocks whose ``(x,y,w,h)`` drifted from the target."""
    inst = placement.instance
    if inst.target_positions is None:
        return []
    bad = []
    for i in np.nonzero(inst.is_preplaced)[0]:
        target = inst.target_positions[i]
        if np.any(target < 0):
            continue
        if np.abs(placement.xywh[i] - target).max() > tolerance:
            bad.append(int(i))
    return bad


def check_feasibility(placement: Placement) -> FeasibilityReport:
    """Run every hard-constraint check and collect the violations."""
    return FeasibilityReport(
        overlap_pairs=overlapping_pairs(placement.xywh),
        area_violations=check_area(placement),
        fixed_violations=check_fixed_shape(placement),
        preplaced_violations=check_preplaced(placement),
        aspect_violations=check_aspect(placement),
    )
