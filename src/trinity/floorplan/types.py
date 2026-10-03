"""The two records the problem layer passes around.

``FloorplanInstance`` is one case (the inputs of a solve); ``Placement`` is a candidate
layout for it. Both hold plain numpy arrays, so geometry, scoring and legalization are
torch-free.

A block's kind comes from the 5-column constraint array
``[fixed, preplaced, mib_id, cluster_id, boundary_code]`` (the FloorSet column order): a
block is *preplaced* if ``preplaced != 0`` (position and shape locked), else *fixed-shape*
if ``fixed != 0`` (shape locked, free to move), else *soft* (only its area is constrained).
"""

from dataclasses import dataclass

import numpy as np

# Column indices of the constraint array.
COL_FIXED = 0
COL_PREPLACED = 1
COL_MIB = 2
COL_CLUSTER = 3
COL_BOUNDARY = 4

# Boundary bitmask -> required bbox edges (1=left, 2=right, 4=top, 8=bottom; corners sum).
BOUNDARY_EDGES: dict[int, tuple[str, ...]] = {
    1: ("left",),
    2: ("right",),
    4: ("top",),
    8: ("bottom",),
    5: ("top", "left"),
    6: ("top", "right"),
    9: ("bottom", "left"),
    10: ("bottom", "right"),
}


@dataclass
class FloorplanInstance:
    """One floorplanning case, unpadded to ``n = block_count`` blocks.

    ``target_positions`` is ``(n, 4)`` ``(x, y, w, h)`` with ``-1`` where a coordinate is
    free (all ``-1`` = soft block). ``nets``, ``outline`` and ``aspect_bounds`` are set only
    by the bookshelf loaders.
    """

    block_count: int
    # (n,) target area per block.
    area_targets: np.ndarray
    # (n, 5) [fixed, preplaced, mib_id, cluster_id, boundary_code].
    constraints: np.ndarray
    # (E_b, 3) (block_i, block_j, weight).
    b2b: np.ndarray
    # (E_p, 3) (pin_idx, block_idx, weight).
    p2b: np.ndarray
    # (n_pins, 2) pin positions.
    pins_pos: np.ndarray
    # (n, 4) (x, y, w, h), -1 = free.
    target_positions: np.ndarray
    # (n, 4) ground-truth layout (FloorSet only).
    gt_positions: np.ndarray | None = None
    # (8,) dataset metrics, including the reference baselines (FloorSet only).
    metrics: np.ndarray | None = None
    test_id: int | None = None
    # Hyperedges over node ids: blocks first, then pins (bookshelf sets).
    nets: list[list[int]] | None = None
    # Fixed outline (W, H) when the protocol sets one.
    outline: tuple[float, float] | None = None
    # (n, 2) (min, max) of w/h per soft block, -1 = free (bookshelf soft sets).
    aspect_bounds: np.ndarray | None = None

    @property
    def s(self) -> float:
        """Layout scale ``sqrt(sum area)``."""
        return float(np.sqrt(self.area_targets.sum()))

    @property
    def is_fixed(self) -> np.ndarray:
        return self.constraints[:, COL_FIXED] != 0

    @property
    def is_preplaced(self) -> np.ndarray:
        return self.constraints[:, COL_PREPLACED] != 0

    @property
    def is_soft(self) -> np.ndarray:
        return ~(self.is_fixed | self.is_preplaced)

    @property
    def mib_id(self) -> np.ndarray:
        return self.constraints[:, COL_MIB].astype(np.int64)

    @property
    def cluster_id(self) -> np.ndarray:
        return self.constraints[:, COL_CLUSTER].astype(np.int64)

    @property
    def boundary_code(self) -> np.ndarray:
        return self.constraints[:, COL_BOUNDARY].astype(np.int64)


@dataclass
class Placement:
    """A candidate layout: ``(n, 4)`` ``(x, y, w, h)``, lower-left corner + size."""

    xywh: np.ndarray
    instance: FloorplanInstance

    def copy(self) -> "Placement":
        return Placement(xywh=self.xywh.copy(), instance=self.instance)

    @property
    def x(self) -> np.ndarray:
        return self.xywh[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.xywh[:, 1]

    @property
    def w(self) -> np.ndarray:
        return self.xywh[:, 2]

    @property
    def h(self) -> np.ndarray:
        return self.xywh[:, 3]

    @property
    def centers(self) -> np.ndarray:
        """``[n,2]`` block centers ``(x + w/2, y + h/2)``."""
        return self.xywh[:, :2] + self.xywh[:, 2:] / 2.0

    def to_tuples(self) -> list[tuple[float, float, float, float]]:
        """The layout as a list of ``(x, y, w, h)`` float tuples."""
        return [tuple(float(v) for v in row) for row in self.xywh]
