"""The metrics of the MCNC / GSRC floorplanning literature for a bookshelf case.

Net HPWL (block centers and terminals, per hyperedge), bounding-box area and dead space,
whether the layout fits the fixed outline, and the hard checks (overlap, kept shapes).
``orient_targets`` counts a rotated hard block as keeping its shape; ``clamp_aspect``
clips soft blocks into their aspect bounds.
"""

from dataclasses import dataclass, replace

import numpy as np

from trinity.floorplan.geometry import bbox_area, bounding_box
from trinity.floorplan.scoring.feasibility import check_feasibility
from trinity.floorplan.types import Placement


@dataclass
class BookshelfScore:
    """The bookshelf metrics of one layout
    (``fits_outline`` is ``None`` without an outline)."""

    hpwl: float
    area: float
    width: float
    height: float
    dead_space: float
    fits_outline: bool | None
    overlap_free: bool
    shapes_kept: bool
    n_nets: int


def orient_targets(inst, xywh) -> object:
    """``inst`` with each fixed-shape target
    rotated to the orientation ``xywh`` gives it.

    Returns ``inst`` itself when no block is rotated.
    """
    targets = np.asarray(inst.target_positions, dtype=np.float64)
    boxes = np.asarray(xywh, dtype=np.float64)
    fixed = (
        inst.is_fixed & ~inst.is_preplaced & (targets[:, 2] >= 0) & (targets[:, 3] >= 0)
    )
    as_is = np.abs(boxes[:, 2] - targets[:, 2]) + np.abs(boxes[:, 3] - targets[:, 3])
    swapped = np.abs(boxes[:, 2] - targets[:, 3]) + np.abs(boxes[:, 3] - targets[:, 2])
    rotated = fixed & (swapped < as_is)
    if not rotated.any():
        return inst
    targets = targets.copy()
    targets[rotated, 2], targets[rotated, 3] = (
        targets[rotated, 3].copy(),
        targets[rotated, 2].copy(),
    )
    return replace(inst, target_positions=targets.astype(inst.target_positions.dtype))


def clamp_aspect(inst, xywh) -> np.ndarray:
    """``xywh`` with every bounded soft block's
    ``w/h`` clipped into its ``aspect_bounds``.

    Area and center are kept.
    """
    out = np.asarray(xywh, dtype=np.float64).copy()
    if inst.aspect_bounds is None:
        return out
    lo, hi = inst.aspect_bounds[:, 0], inst.aspect_bounds[:, 1]
    for i in np.nonzero(inst.is_soft & (lo > 0) & (hi > 0))[0]:
        area = out[i, 2] * out[i, 3]
        ratio = np.clip(out[i, 2] / max(out[i, 3], 1e-12), lo[i], hi[i])
        w, h = np.sqrt(area * ratio), np.sqrt(area / ratio)
        cx, cy = out[i, 0] + out[i, 2] / 2, out[i, 1] + out[i, 3] / 2
        out[i] = [cx - w / 2, cy - h / 2, w, h]
    return out


def net_hpwl(placement: Placement) -> float:
    """The sum over hyperedges of the half-perimeter
    of their block centers and terminals."""
    inst = placement.instance
    if inst.nets is None:
        raise ValueError("the instance carries no hyperedges (not a bookshelf case)")
    pins = np.asarray(inst.pins_pos, dtype=np.float64).reshape(-1, 2)
    points = np.concatenate([placement.centers, pins], axis=0)
    total = 0.0
    for net in inst.nets:
        p = points[np.asarray(net)]
        total += float(p[:, 0].max() - p[:, 0].min() + p[:, 1].max() - p[:, 1].min())
    return total if inst.block_count else 0.0


def score_bookshelf(placement: Placement, outline=None) -> BookshelfScore:
    """The bookshelf metrics of ``placement``
    (``outline`` defaults to the instance's)."""
    inst = placement.instance
    xywh = np.asarray(placement.xywh, dtype=np.float64)
    x0, y0, x1, y1 = bounding_box(xywh)
    area = bbox_area(xywh)
    block_area = float((xywh[:, 2] * xywh[:, 3]).sum())
    box = inst.outline if outline is None else outline
    fits = None
    if box is not None:
        fits = bool(x1 - x0 <= box[0] * (1 + 1e-9) and y1 - y0 <= box[1] * (1 + 1e-9))
    report = check_feasibility(placement)
    shape_violations = (
        report.fixed_violations
        or report.preplaced_violations
        or report.aspect_violations
        or report.area_violations
    )
    return BookshelfScore(
        hpwl=net_hpwl(placement),
        area=float(area),
        width=float(x1 - x0),
        height=float(y1 - y0),
        dead_space=float(1.0 - block_area / max(area, 1e-12)),
        fits_outline=fits,
        overlap_free=report.overlap_count == 0,
        shapes_kept=not shape_violations,
        n_nets=len(inst.nets or []),
    )
