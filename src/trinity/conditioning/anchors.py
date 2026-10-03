"""Anchors: the latent coordinates of a case that are known, and their values.

In the latent ``z = (cx/s, cy/s, rho)``, fixed-shape blocks anchor ``rho`` and preplaced
blocks anchor all three coordinates.

MIB groups (members share one shape):

* a group with a fixed / preplaced member anchors every soft
  member's ``rho`` to the first known member's ``rho``;
* an all-soft group follows ``mib_anchor``:

  - ``"group_mean"`` -- no anchor; the group id is recorded in ``mib_soft_group`` and
    the ``mib_group_mean`` projection sets every member's ``rho`` to the group mean;
  - ``"square"`` -- anchors every member's ``rho`` to 0;
  - ``"none"`` -- nothing.
"""

import numpy as np

from trinity.floorplan.parameterize import RHO_CLAMP
from trinity.floorplan.types import FloorplanInstance


def _rho_of(w: float, h: float) -> float:
    return float(np.clip(np.log(w / h), -RHO_CLAMP, RHO_CLAMP))


def build_anchors(
    inst: FloorplanInstance, mib_anchor: str = "group_mean"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The anchors of ``inst``: ``(anchor_z (n, 3),
    anchor_mask (n, 3), mib_soft_group (n,))``.

    ``anchor_z`` is in ``/ s`` units; ``mib_soft_group`` is 0 for blocks outside a
    ``group_mean`` group.
    """
    n = inst.block_count
    s = inst.s
    anchor_z = np.zeros((n, 3), dtype=np.float32)
    anchor_mask = np.zeros((n, 3), dtype=np.float32)
    tp = inst.target_positions

    for i in np.nonzero(inst.is_fixed)[0]:
        if tp[i, 2] >= 0 and tp[i, 3] >= 0:
            anchor_z[i, 2] = _rho_of(tp[i, 2], tp[i, 3])
            anchor_mask[i, 2] = 1.0

    for i in np.nonzero(inst.is_preplaced)[0]:
        if np.all(tp[i] >= 0):
            cx = tp[i, 0] + tp[i, 2] / 2.0
            cy = tp[i, 1] + tp[i, 3] / 2.0
            anchor_z[i, 0] = cx / s
            anchor_z[i, 1] = cy / s
            anchor_z[i, 2] = _rho_of(tp[i, 2], tp[i, 3])
            anchor_mask[i, :] = 1.0

    mib_soft_group = _anchor_mib(inst, anchor_z, anchor_mask, mib_anchor)
    return anchor_z, anchor_mask, mib_soft_group


def _anchor_mib(
    inst: FloorplanInstance,
    anchor_z: np.ndarray,
    anchor_mask: np.ndarray,
    mib_anchor: str,
) -> np.ndarray:
    """Write the MIB anchors into ``anchor_z`` / ``anchor_mask`` (in place).

    Returns the ``(n,)`` ``group_mean`` group ids (module docstring).
    """
    mib = inst.mib_id
    soft_group = np.zeros(inst.block_count, dtype=np.int64)
    for g in np.unique(mib[mib > 0]):
        members = np.nonzero(mib == g)[0]
        if members.size < 2:
            continue
        ref_rho = None
        for m in members:
            if anchor_mask[m, 2] > 0.5:
                ref_rho = float(anchor_z[m, 2])
                break
        if ref_rho is not None:
            for m in members:
                if inst.is_soft[m]:
                    anchor_z[m, 2] = ref_rho
                    anchor_mask[m, 2] = 1.0
            continue
        if mib_anchor == "square":
            for m in members:
                if inst.is_soft[m]:
                    anchor_z[m, 2] = 0.0
                    anchor_mask[m, 2] = 1.0
        elif mib_anchor == "group_mean":
            soft_group[members] = int(g)
        elif mib_anchor != "none":
            raise ValueError(
                f"unknown mib_anchor {mib_anchor!r}; "
                "expected group_mean | square | none"
            )
    return soft_group
