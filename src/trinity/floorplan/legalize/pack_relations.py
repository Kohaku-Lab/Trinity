"""The geometric stages of ``scale_pack``: restore, expand, pair relations, abutment, snap.

A *relation* ``(axis, lo, hi)`` of a block pair says that ``lo`` lies before ``hi`` on
``axis`` (0 = x, 1 = y): ``pos[lo] + size[lo] <= pos[hi]``. The relations of every pair
form the constraint graph the packing LP (:mod:`.pack_lp`) solves over.
"""

import numpy as np

from trinity.floorplan.geometry import OVERLAP_EPS, cand_overlaps_any, overlapping_pairs
from trinity.floorplan.scoring.bookshelf import clamp_aspect
from trinity.floorplan.scoring.feasibility import AREA_TOLERANCE

MAX_EXPAND = 10.0


def _center(xywh: np.ndarray, i: int, axis: int) -> float:
    """The center coordinate of block ``i`` on ``axis``."""
    return xywh[i, axis] + xywh[i, axis + 2] / 2


def restore(inst, xywh: np.ndarray) -> np.ndarray:
    """Stage A: the known shapes and positions written back into ``xywh``.

    Fixed-shape blocks get their target shape about their center, preplaced blocks their
    target box, soft blocks their target area about their center (aspect clipped into the
    instance's aspect bounds when it has them), and every all-soft MIB group of equal areas
    one shared shape (the median log-aspect of its members).
    """
    out = np.asarray(xywh, np.float64).copy()
    targets = inst.target_positions
    for i in np.nonzero(inst.is_fixed & ~inst.is_preplaced)[0]:
        if targets[i, 2] >= 0 and targets[i, 3] >= 0:
            cx, cy = out[i, 0] + out[i, 2] / 2, out[i, 1] + out[i, 3] / 2
            out[i, 2:] = targets[i, 2:]
            out[i, 0], out[i, 1] = cx - targets[i, 2] / 2, cy - targets[i, 3] / 2
    for i in np.nonzero(inst.is_preplaced)[0]:
        if np.all(targets[i] >= 0):
            out[i] = targets[i]

    soft = inst.is_soft & (inst.area_targets > 0)
    idx = np.nonzero(soft)[0]
    if idx.size:
        cx, cy = out[idx, 0] + out[idx, 2] / 2, out[idx, 1] + out[idx, 3] / 2
        current_area = np.maximum(out[idx, 2] * out[idx, 3], 1e-12)
        factor = np.sqrt(inst.area_targets[idx] / current_area)
        out[idx, 2] *= factor
        out[idx, 3] *= factor
        out[idx, 0], out[idx, 1] = cx - out[idx, 2] / 2, cy - out[idx, 3] / 2
    if inst.aspect_bounds is not None:
        out = clamp_aspect(inst, out)

    mib = inst.mib_id
    for g in np.unique(mib[mib > 0]):
        members = np.nonzero(mib == g)[0]
        if len(members) < 2 or not np.all(soft[members]):
            continue
        areas = inst.area_targets[members]
        shared = float(areas.mean())
        if np.max(np.abs(shared - areas) / areas) > 0.5 * AREA_TOLERANCE:
            continue
        log_w = np.log(np.maximum(out[members, 2], 1e-12))
        log_h = np.log(np.maximum(out[members, 3], 1e-12))
        rho = float(np.median(log_w - log_h))
        w = float(np.sqrt(shared * np.exp(rho)))
        h = float(np.sqrt(shared * np.exp(-rho)))
        for i in members:
            cx, cy = out[i, 0] + out[i, 2] / 2, out[i, 1] + out[i, 3] / 2
            out[i] = [cx - w / 2, cy - h / 2, w, h]
    return out


def expand(
    xywh: np.ndarray, mobile=None, cap: float = MAX_EXPAND
) -> tuple[np.ndarray, float]:
    """Stage B: ``(expanded layout, factor)``.

    The ``mobile`` blocks' centers move outward from the layout center by the smallest
    factor (at most ``cap``) at which no pair overlaps; the other blocks stay.
    """
    n = len(xywh)
    mob = np.ones(n, dtype=bool) if mobile is None else np.asarray(mobile, dtype=bool)
    centers = xywh[:, :2] + xywh[:, 2:] / 2
    origin = centers.mean(0)
    offset = centers - origin
    s = float(np.sqrt((xywh[:, 2] * xywh[:, 3]).sum()))
    factor = 1.0
    for i, j in overlapping_pairs(xywh):
        if not (mob[i] or mob[j]):
            continue
        if mob[i] and mob[j] and np.abs(offset[i] - offset[j]).max() < 1e-9 * s:
            offset[j, 0] += 1e-3 * s
        need = (xywh[i, 2:] + xywh[j, 2:]) / 2
        best = np.inf
        for axis in (0, 1):
            if mob[i] and mob[j]:
                distance = abs(offset[i, axis] - offset[j, axis])
                value = need[axis] / distance if distance > 0 else np.inf
            else:
                m, p = (i, j) if mob[i] else (j, i)
                dm, dp = offset[m, axis], offset[p, axis]
                if dm > 0:
                    value = (dp + need[axis]) / dm
                elif dm < 0:
                    value = (dp - need[axis]) / dm
                else:
                    value = np.inf
            best = min(best, value)
        factor = max(factor, float(best))
    factor = min(factor, cap)
    out = xywh.copy()
    out[mob, :2] = origin + factor * offset[mob] - xywh[mob, 2:] / 2
    return out, factor


def ordered(ref: np.ndarray, axis: int, i: int, j: int) -> tuple[int, int, int]:
    """The relation of ``i`` and ``j`` on ``axis``, ordered by their centers in ``ref``."""
    if _center(ref, i, axis) <= _center(ref, j, axis):
        return (axis, i, j)
    return (axis, j, i)


def pair_relation(ref: np.ndarray, i: int, j: int) -> tuple[int, int, int]:
    """The relation of one pair: the axis with the larger gap in ``ref``, ordered by center."""
    gx = max(ref[i, 0] - ref[j, 0] - ref[j, 2], ref[j, 0] - ref[i, 0] - ref[i, 2])
    gy = max(ref[i, 1] - ref[j, 1] - ref[j, 3], ref[j, 1] - ref[i, 1] - ref[i, 3])
    return ordered(ref, 0 if gx >= gy else 1, i, j)


def relations(expanded: np.ndarray, pinned: np.ndarray) -> dict:
    """Stage C: ``{(i, j): (axis, lo, hi)}`` for every pair, read from ``expanded``.

    A pair of two overlapping preplaced blocks gets no relation.
    """
    n = len(expanded)
    overlapping_pins = {
        p for p in overlapping_pairs(expanded) if pinned[p[0]] and pinned[p[1]]
    }
    return {
        (i, j): pair_relation(expanded, i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if (i, j) not in overlapping_pins
    }


def _key(i: int, j: int) -> tuple[int, int]:
    """The relation key ``(min, max)`` of a pair."""
    return (min(i, j), max(i, j))


def clear_between(
    rel: dict, abut: list, expanded: np.ndarray, pinned: np.ndarray
) -> dict:
    """``rel`` with no block ordered between an abutting pair on its contact axis.

    Such a block is ordered against both members on the perpendicular axis by center; a
    pair of two preplaced blocks keeps its relation.
    """
    out = dict(rel)
    for axis, lo, hi in abut:
        for k in range(len(expanded)):
            if k in (lo, hi):
                continue
            before = out.get(_key(lo, k))
            after = out.get(_key(k, hi))
            if before != (axis, lo, k) or after != (axis, k, hi):
                continue
            for other in (lo, hi):
                if not (pinned[k] and pinned[other]):
                    out[_key(k, other)] = ordered(expanded, 1 - axis, k, other)
    return out


def clear_outside(rel: dict, inst, expanded: np.ndarray) -> dict:
    """``rel`` with no block ordered outside a boundary-coded block on its coded side.

    Such a block is ordered on the perpendicular axis by center; a pair of two preplaced
    blocks keeps its relation.
    """
    out = dict(rel)
    codes = inst.boundary_code
    pinned = inst.is_preplaced
    n = len(expanded)
    # (bit, axis, whether the coded block must come first on that axis)
    sides = ((1, 0, True), (2, 0, False), (8, 1, True), (4, 1, False))
    for c in np.nonzero(codes)[0]:
        code = int(codes[c])
        for bit, axis, coded_first in sides:
            if not code & bit:
                continue
            for k in range(n):
                if k == c or (pinned[k] and pinned[c]):
                    continue
                key = _key(c, k)
                relation = out.get(key)
                if relation is None or relation[0] != axis:
                    continue
                outside = (axis, k, c) if coded_first else (axis, c, k)
                if relation == outside:
                    out[key] = ordered(expanded, 1 - axis, c, k)
    return out


def count_between(rel: dict, axis: int, lo: int, hi: int, n: int) -> int:
    """The number of blocks ordered after ``lo`` and before ``hi`` on ``axis``."""
    count = 0
    for k in range(n):
        if k in (lo, hi):
            continue
        before = rel.get(_key(lo, k))
        after = rel.get(_key(k, hi))
        count += int(before == (axis, lo, k) and after == (axis, k, hi))
    return count


def _abut_cost(inst, restored, rel, contact, axis, lo, hi, i, j) -> float:
    """The ranking cost of abutting ``i`` and ``j`` along their relation ``(axis, lo, hi)``.

    Gap on the axis + perpendicular overlap missing to ``contact`` + ``s`` per block in
    between + ``s`` when sliding the free member onto the preplaced one hits another
    preplaced block.
    """
    pinned = inst.is_preplaced
    s = float(inst.s)
    perp = 1 - axis
    gap = restored[hi, axis] - restored[lo, axis] - restored[lo, axis + 2]
    overlap = min(
        restored[i, perp] + restored[i, perp + 2],
        restored[j, perp] + restored[j, perp + 2],
    ) - max(restored[i, perp], restored[j, perp])
    blocked = 0.0
    if pinned[i] != pinned[j]:
        m, p = (lo, hi) if not pinned[lo] else (hi, lo)
        slid = restored[m].copy()
        if m == hi:
            slid[axis] = restored[p, axis] + restored[p, axis + 2]
        else:
            slid[axis] = restored[p, axis] - restored[m, axis + 2]
        others = np.nonzero(pinned)[0]
        self_index = int(np.nonzero(others == p)[0][0])
        if cand_overlaps_any(slid, restored[others], exclude={self_index}):
            blocked = s
    between = count_between(rel, axis, lo, hi, len(restored))
    return max(gap, 0.0) + max(0.0, contact - overlap) + s * between + blocked


def _find_root(parent: dict, k: int) -> int:
    """The union-find root of ``k`` (with path halving)."""
    while parent[k] != k:
        parent[k] = parent[parent[k]]
        k = parent[k]
    return k


def abut_tree(
    inst, restored: np.ndarray, rel: dict, contact: float
) -> list[tuple[int, int, int]]:
    """The abutment edges: per cluster, a Kruskal spanning tree over its member pairs.

    Pairs are ranked by :func:`_abut_cost`; each edge is the pair's relation.
    """
    cluster = inst.cluster_id
    pinned = inst.is_preplaced
    edges = []
    for g in np.unique(cluster[cluster > 0]):
        members = np.nonzero(cluster == g)[0]
        if len(members) < 2:
            continue
        candidates = []
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = int(members[a]), int(members[b])
                if (pinned[i] and pinned[j]) or (i, j) not in rel:
                    continue
                axis, lo, hi = rel[(i, j)]
                cost = _abut_cost(inst, restored, rel, contact, axis, lo, hi, i, j)
                candidates.append((cost, i, j, axis, lo, hi))
        candidates.sort()
        parent = {int(k): int(k) for k in members}
        for _cost, i, j, axis, lo, hi in candidates:
            root_i, root_j = _find_root(parent, i), _find_root(parent, j)
            if root_i != root_j:
                parent[root_i] = root_j
                edges.append((axis, lo, hi))
    return edges


def flip_pairs(rel: dict, pairs, ref: np.ndarray) -> dict:
    """``rel`` with the listed pairs moved to the other axis, ordered by center in ``ref``."""
    out = dict(rel)
    for key in pairs:
        axis, lo, hi = out[key]
        out[key] = ordered(ref, 1 - axis, lo, hi)
    return out


def beyond(rel: dict, pinned: np.ndarray, only=None) -> dict:
    """``rel`` with every (free, preplaced) pair related as the free block above.

    ``only`` restricts this to the listed preplaced blocks.
    """
    out = dict(rel)
    for key, (_axis, lo, hi) in rel.items():
        if pinned[lo] != pinned[hi]:
            p, m = (lo, hi) if pinned[lo] else (hi, lo)
            if only is None or p in only:
                out[key] = (1, p, m)
    return out


def snap_contacts(xywh: np.ndarray, abut: list) -> np.ndarray:
    """Stage F: abutting pairs within ``OVERLAP_EPS`` of contact moved to exact contact.

    A move is kept only when it creates no overlap; up to three sweeps.
    """
    out = xywh.copy()
    for _ in range(3):
        moved = False
        for axis, lo, hi in abut:
            target = out[lo, axis] + out[lo, axis + 2]
            if out[hi, axis] == target or abs(out[hi, axis] - target) > OVERLAP_EPS:
                continue
            cand = out[hi].copy()
            cand[axis] = target
            if not cand_overlaps_any(cand, out, exclude={hi}):
                out[hi] = cand
                moved = True
        if not moved:
            break
    return out
