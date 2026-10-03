"""Perturb a valid layout to inject specific violations (used to test the scorers).

Each jitter mode is a registered function ``(placement, rng, **kwargs) -> JitterEffect``
that changes a copy of the placement and reports what it broke. Modes cover the hard
constraints (overlap, area, fixed shape, preplaced), the soft ones (grouping, MIB,
boundary), and plain noise. ``apply_jitter`` runs a list of modes in order.
"""

from dataclasses import dataclass

import numpy as np

from trinity.floorplan.registry import JITTER, resolve
from trinity.floorplan.types import Placement


@dataclass
class JitterEffect:
    """The perturbed placement, a description of the change and the touched blocks."""

    placement: Placement
    description: str
    touched: list[int]


def _pick_soft(inst, rng, k: int = 1) -> list[int]:
    soft = np.nonzero(inst.is_soft)[0]
    if soft.size == 0:
        return []
    return rng.choice(soft, size=min(k, soft.size), replace=False).tolist()


@JITTER.register("overlap")
def jitter_overlap(
    placement: Placement, rng: np.random.Generator, fraction: float = 0.5
) -> JitterEffect:
    """Slide one block on top of a neighbour to create a hard overlap."""
    p = placement.copy()
    soft = _pick_soft(p.instance, rng, k=1)
    if not soft:
        return JitterEffect(p, "overlap: no soft block to move", [])
    i = soft[0]
    centers = p.centers
    others = [j for j in range(p.instance.block_count) if j != i]
    j = min(others, key=lambda k: np.abs(centers[i] - centers[k]).sum())
    p.xywh[i, 0] = p.xywh[j, 0] + fraction * p.xywh[j, 2] - p.xywh[i, 2] * 0.5
    p.xywh[i, 1] = p.xywh[j, 1] + fraction * p.xywh[j, 3] - p.xywh[i, 3] * 0.5
    return JitterEffect(p, f"overlap: moved block {i} onto block {j}", [i, j])


@JITTER.register("area")
def jitter_area(
    placement: Placement, rng: np.random.Generator, scale: float = 1.2
) -> JitterEffect:
    """Resize one soft block beyond the 1% area tolerance (keeps aspect)."""
    p = placement.copy()
    soft = _pick_soft(p.instance, rng, k=1)
    if not soft:
        return JitterEffect(p, "area: no soft block", [])
    i = soft[0]
    p.xywh[i, 2] *= scale
    p.xywh[i, 3] *= scale
    return JitterEffect(p, f"area: scaled block {i} by {scale}", [i])


@JITTER.register("fixed")
def jitter_fixed(
    placement: Placement, rng: np.random.Generator, delta: float = 5.0
) -> JitterEffect:
    """Change a fixed-shape block's dimensions (violates shape immutability)."""
    p = placement.copy()
    fixed = np.nonzero(p.instance.is_fixed)[0]
    if fixed.size == 0:
        return JitterEffect(p, "fixed: no fixed-shape block", [])
    i = int(rng.choice(fixed))
    p.xywh[i, 2] += delta
    p.xywh[i, 3] = max(p.xywh[i, 3] - delta, 1.0)
    return JitterEffect(p, f"fixed: changed shape of fixed block {i}", [i])


@JITTER.register("preplaced")
def jitter_preplaced(
    placement: Placement, rng: np.random.Generator, delta: float = 8.0
) -> JitterEffect:
    """Translate a preplaced block off its locked position."""
    p = placement.copy()
    pre = np.nonzero(p.instance.is_preplaced)[0]
    if pre.size == 0:
        return JitterEffect(p, "preplaced: no preplaced block", [])
    i = int(rng.choice(pre))
    p.xywh[i, 0] += delta
    p.xywh[i, 1] += delta
    return JitterEffect(p, f"preplaced: displaced preplaced block {i}", [i])


@JITTER.register("grouping")
def jitter_grouping(
    placement: Placement, rng: np.random.Generator, distance: float = 30.0
) -> JitterEffect:
    """Pull one cluster member far away so the group splits into components."""
    p = placement.copy()
    cid = p.instance.cluster_id
    groups = np.unique(cid[cid > 0])
    if groups.size == 0:
        return JitterEffect(p, "grouping: no cluster", [])
    g = int(rng.choice(groups))
    members = np.nonzero(cid == g)[0]
    if members.size < 2:
        return JitterEffect(p, f"grouping: cluster {g} too small", [])
    i = int(members[-1])
    p.xywh[i, 0] += distance
    p.xywh[i, 1] += distance
    return JitterEffect(p, f"grouping: pulled block {i} out of cluster {g}", [i])


@JITTER.register("mib")
def jitter_mib(
    placement: Placement, rng: np.random.Generator, scale: float = 1.5
) -> JitterEffect:
    """Reshape one MIB-group member so it no longer matches the group shape."""
    p = placement.copy()
    mid = p.instance.mib_id
    groups = np.unique(mid[mid > 0])
    if groups.size == 0:
        return JitterEffect(p, "mib: no mib group", [])
    g = int(rng.choice(groups))
    members = np.nonzero(mid == g)[0]
    if members.size < 2:
        return JitterEffect(p, f"mib: group {g} too small", [])
    i = int(members[-1])
    p.xywh[i, 2] *= scale
    p.xywh[i, 3] /= scale
    return JitterEffect(p, f"mib: reshaped block {i} in group {g}", [i])


@JITTER.register("boundary")
def jitter_boundary(
    placement: Placement, rng: np.random.Generator, distance: float = 20.0
) -> JitterEffect:
    """Move a boundary-coded block inward so it no longer touches its required edge."""
    p = placement.copy()
    coded = np.nonzero(p.instance.boundary_code != 0)[0]
    if coded.size == 0:
        return JitterEffect(p, "boundary: no boundary block", [])
    i = int(rng.choice(coded))
    cx = p.centers[:, 0].mean()
    cy = p.centers[:, 1].mean()
    direction = np.sign([cx - p.centers[i, 0], cy - p.centers[i, 1]])
    p.xywh[i, 0] += direction[0] * distance
    p.xywh[i, 1] += direction[1] * distance
    return JitterEffect(p, f"boundary: moved boundary block {i} inward", [i])


@JITTER.register("noise")
def jitter_noise(
    placement: Placement, rng: np.random.Generator, sigma: float = 5.0
) -> JitterEffect:
    """Add Gaussian noise to every block's position (generic mess)."""
    p = placement.copy()
    p.xywh[:, :2] += rng.normal(0.0, sigma, size=(p.instance.block_count, 2))
    return JitterEffect(
        p,
        f"noise: jittered all positions sigma={sigma}",
        list(range(p.instance.block_count)),
    )


def apply_jitter(
    placement: Placement, modes: list[str | dict], seed: int = 0
) -> tuple[Placement, list[str]]:
    """Apply a configured list of jitter modes in order; return the broken layout + log."""
    rng = np.random.default_rng(seed)
    current = placement.copy()
    log = []
    for mode in modes:
        if isinstance(mode, dict):
            opts = dict(mode)
            fn = resolve(opts.pop("name"), JITTER)
            effect = fn(current, rng, **opts)
        else:
            fn = resolve(mode, JITTER)
            effect = fn(current, rng)
        current = effect.placement
        log.append(effect.description)
    return current, log
