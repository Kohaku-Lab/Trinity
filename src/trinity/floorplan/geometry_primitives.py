"""The continuous constraint quantities of a layout, as pure numpy functions.

Each primitive maps ``xywh`` (plus per-block data) to one float: a continuous relaxation
of one constraint (overlap, compactness, boundary, grouping, MIB, fixed / preplaced
immutability). The continuous scorer (:mod:`trinity.floorplan.scoring.continuous`) is
built from these; the training losses in ``trinity/losses/aux_terms.py`` are their batched
torch counterparts.

Conventions: ``xywh`` is ``(n, 4)`` lower-left ``(x, y, w, h)``; ``s = sqrt(sum area)``
is the layout scale; group-id arrays are ``(n,)`` ints (``> 0`` = member of that group).
Empty or degenerate inputs return ``0.0``.
"""

import numpy as np

from trinity.floorplan.geometry import bounding_box, connected_components

EPS = 1e-9


def overlap_area(xywh: np.ndarray) -> float:
    """Total pairwise rectangle intersection area."""
    x0, y0 = xywh[:, 0], xywh[:, 1]
    x1, y1 = x0 + xywh[:, 2], y0 + xywh[:, 3]
    ox = np.maximum(
        0.0, np.minimum(x1[:, None], x1[None]) - np.maximum(x0[:, None], x0[None])
    )
    oy = np.maximum(
        0.0, np.minimum(y1[:, None], y1[None]) - np.maximum(y0[:, None], y0[None])
    )
    return float(np.triu(ox * oy, k=1).sum())


def overlap_ratio(xywh: np.ndarray) -> float:
    """``overlap_area / total block area`` (0 = overlap-free)."""
    area = float((xywh[:, 2] * xywh[:, 3]).sum())
    return overlap_area(xywh) / (area + EPS)


def bbox_compactness(xywh: np.ndarray) -> float:
    """``bbox area / total block area`` (``>= 1``; 1 = no dead space)."""
    xmin, ymin, xmax, ymax = bounding_box(xywh)
    area = float((xywh[:, 2] * xywh[:, 3]).sum())
    return float((xmax - xmin) * (ymax - ymin) / (area + EPS))


def boundary_distance(xywh: np.ndarray, boundary_code: np.ndarray, s: float) -> float:
    """Mean distance of each boundary-coded block to its required bbox edges, ``/ s``.

    ``boundary_code`` is the bitmask 1=left 2=right 4=top 8=bottom; a
    block's distance is the mean L1 distance over its required edges.
    """
    coded = np.nonzero(boundary_code != 0)[0]
    if coded.size == 0:
        return 0.0
    xmin, ymin, xmax, ymax = bounding_box(xywh)
    distances = []
    for i in coded:
        code = int(boundary_code[i])
        x0, y0, w, h = xywh[i]
        x1, y1 = x0 + w, y0 + h
        edges = []
        if code & 1:
            edges.append(abs(x0 - xmin))
        if code & 2:
            edges.append(abs(x1 - xmax))
        if code & 4:
            edges.append(abs(y1 - ymax))
        if code & 8:
            edges.append(abs(y0 - ymin))
        if edges:
            distances.append(float(np.mean(edges)))
    return float(np.mean(distances) / (s + EPS)) if distances else 0.0


def pair_gap(a: np.ndarray, b: np.ndarray) -> float:
    """The smallest axis translation that makes
    rectangles ``a`` and ``b`` touch (0 if they do)."""
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    gx = max(0.0, bx0 - ax1, ax0 - bx1)
    gy = max(0.0, by0 - ay1, ay0 - by1)
    y_overlap = min(ay1, by1) - max(ay0, by0) > 0
    x_overlap = min(ax1, bx1) - max(ax0, bx0) > 0
    if y_overlap:
        return gx
    if x_overlap:
        return gy
    return min(gx, gy)


def group_gap(xywh: np.ndarray, cluster_id: np.ndarray, s: float) -> float:
    """Mean gap from each clustered block to its
    nearest same-cluster member, ``/ s``."""
    gaps = []
    for g in np.unique(cluster_id[cluster_id > 0]):
        members = np.nonzero(cluster_id == g)[0]
        if members.size < 2:
            continue
        for i in members:
            gaps.append(min(pair_gap(xywh[i], xywh[j]) for j in members if j != i))
    return float(np.mean(gaps) / (s + EPS)) if gaps else 0.0


def group_split(xywh: np.ndarray, cluster_id: np.ndarray) -> float:
    """The normalized grouping violation ``sum_g
    (components_g - 1) / sum_g (|G_g| - 1)``.

    A count (in ``[0, 1]``, 0 when every cluster is connected),
    so a hard metric, not part of the continuous score.
    """
    violations = worst_case = 0
    for g in np.unique(cluster_id[cluster_id > 0]):
        members = np.nonzero(cluster_id == g)[0]
        if members.size < 2:
            continue
        violations += connected_components(xywh, members) - 1
        worst_case += members.size - 1
    return float(violations / worst_case) if worst_case else 0.0


def mib_log_shape_var(xywh: np.ndarray, mib_id: np.ndarray) -> float:
    """Mean within-MIB-group variance of the log
    shape ``(log w, log h)`` (0 = identical)."""
    variances = []
    for g in np.unique(mib_id[mib_id > 0]):
        members = np.nonzero(mib_id == g)[0]
        if members.size < 2:
            continue
        log_w = np.log(np.maximum(xywh[members, 2], EPS))
        log_h = np.log(np.maximum(xywh[members, 3], EPS))
        log_shape = np.column_stack([log_w, log_h])
        variances.append(float(((log_shape - log_shape.mean(0)) ** 2).sum(1).mean()))
    return float(np.mean(variances)) if variances else 0.0


def fixed_deviation(
    xywh: np.ndarray,
    target_positions: np.ndarray,
    area_targets: np.ndarray,
    is_locked: np.ndarray,
    is_pos_locked: np.ndarray,
    s: float,
) -> float:
    """Mean deviation of the fixed / preplaced blocks from their targets.

    Per block: the center distance ``/ s`` (position-locked blocks) plus the shape
    deviation ``(|dw| + |dh|) / sqrt(area)`` (blocks with a target shape).
    """
    centers = xywh[:, :2] + xywh[:, 2:] / 2
    locked = np.nonzero(is_locked | is_pos_locked)[0]
    deviations = []
    for i in locked:
        target = target_positions[i]
        deviation = 0.0
        if is_pos_locked[i] and target[0] >= 0:
            target_cx = target[0] + target[2] / 2
            target_cy = target[1] + target[3] / 2
            distance = np.hypot(centers[i, 0] - target_cx, centers[i, 1] - target_cy)
            deviation += float(distance / (s + EPS))
        if target[2] >= 0:
            shape_error = abs(xywh[i, 2] - target[2]) + abs(xywh[i, 3] - target[3])
            deviation += float(shape_error / (np.sqrt(area_targets[i]) + EPS))
        deviations.append(deviation)
    return float(np.mean(deviations)) if deviations else 0.0
