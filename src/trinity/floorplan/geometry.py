"""Geometry primitives over ``(x, y, w, h)`` rectangles.

``(x, y)`` is the lower-left corner and a block spans ``[x, x+w] x [y, y+h]``. Overlap is
tested per axis with a ``1e-6`` epsilon, so edge contact is not an overlap. A polygon is
reduced to its axis-aligned bounding box.
"""

import functools

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

OVERLAP_EPS = 1e-6


@functools.lru_cache(maxsize=512)
def _triu_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The strict upper-triangle index arrays of an ``n x n`` matrix (cached)."""
    return np.triu_indices(n, k=1)


def bounding_box(xywh: np.ndarray) -> tuple[float, float, float, float]:
    """Global bbox of a layout as ``(x_min, y_min, x_max, y_max)``."""
    x_min = float(xywh[:, 0].min())
    y_min = float(xywh[:, 1].min())
    x_max = float((xywh[:, 0] + xywh[:, 2]).max())
    y_max = float((xywh[:, 1] + xywh[:, 3]).max())
    return x_min, y_min, x_max, y_max


def bbox_area(xywh: np.ndarray) -> float:
    """Area of the layout's global bounding box (not the sum of block areas)."""
    x_min, y_min, x_max, y_max = bounding_box(xywh)
    return (x_max - x_min) * (y_max - y_min)


def overlapping_pairs(
    xywh: np.ndarray, eps: float = OVERLAP_EPS
) -> list[tuple[int, int]]:
    """Index pairs ``(i,j)`` whose overlap exceeds ``eps`` on *both* axes."""
    x1 = xywh[:, 0][:, None]
    y1 = xywh[:, 1][:, None]
    x2 = (xywh[:, 0] + xywh[:, 2])[:, None]
    y2 = (xywh[:, 1] + xywh[:, 3])[:, None]
    ox = np.minimum(x2, x2.T) - np.maximum(x1, x1.T)
    oy = np.minimum(y2, y2.T) - np.maximum(y1, y1.T)
    overlapping = (ox > eps) & (oy > eps)
    iu, ju = _triu_pairs(xywh.shape[0])
    mask = overlapping[iu, ju]
    return list(zip(iu[mask].tolist(), ju[mask].tolist(), strict=True))


def rects_overlap(a: np.ndarray, b: np.ndarray, eps: float = OVERLAP_EPS) -> bool:
    """Whether two ``(x, y, w, h)`` rectangles penetrate by more than ``eps`` on both axes."""
    ox = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    oy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return ox > eps and oy > eps


def cand_overlaps_any(
    cand: np.ndarray,
    xywh: np.ndarray,
    exclude: set[int] | None = None,
    eps: float = OVERLAP_EPS,
) -> bool:
    """Whether ``cand`` overlaps any row of ``xywh`` outside ``exclude`` (vectorized
    :func:`rects_overlap`)."""
    ox = np.minimum(cand[0] + cand[2], xywh[:, 0] + xywh[:, 2]) - np.maximum(
        cand[0], xywh[:, 0]
    )
    oy = np.minimum(cand[1] + cand[3], xywh[:, 1] + xywh[:, 3]) - np.maximum(
        cand[1], xywh[:, 1]
    )
    hit = (ox > eps) & (oy > eps)
    if exclude:
        hit[list(exclude)] = False
    return bool(hit.any())


def share_edge(a: np.ndarray, b: np.ndarray, eps: float = OVERLAP_EPS) -> bool:
    """Whether rectangles ``a`` and ``b`` share a boundary segment of nonzero length."""
    yov = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if yov > eps and (abs(a[0] + a[2] - b[0]) < eps or abs(b[0] + b[2] - a[0]) < eps):
        return True
    xov = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    return xov > eps and (
        abs(a[1] + a[3] - b[1]) < eps or abs(b[1] + b[3] - a[1]) < eps
    )


def rect_polygon(x: float, y: float, w: float, h: float) -> Polygon:
    """Axis-aligned rectangle as a shapely ``Polygon``."""
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def polygon_to_bbox(vertices: np.ndarray) -> tuple[float, float, float, float]:
    """The axis-aligned bounding box ``(x, y, w, h)`` of polygon ``vertices``.

    Rows whose x is ``-1`` are padding; an empty polygon gives the unit box at the origin.
    """
    valid = vertices[vertices[:, 0] != -1]
    if valid.shape[0] == 0:
        return 0.0, 0.0, 1.0, 1.0
    x_min, y_min = valid.min(axis=0)
    x_max, y_max = valid.max(axis=0)
    return float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)


def connected_components(xywh: np.ndarray, indices: np.ndarray) -> int:
    """The number of connected components among the blocks ``indices``.

    Computed by a shapely ``unary_union``: blocks sharing an edge or overlapping merge, a
    corner-only contact does not.
    """
    if len(indices) <= 1:
        return len(indices)
    polys = [rect_polygon(*xywh[i]) for i in indices]
    union = unary_union(polys)
    if union.geom_type == "MultiPolygon":
        return len(union.geoms)
    return 1


def component_labels(xywh: np.ndarray, indices: np.ndarray) -> dict[int, int]:
    """Map each block in ``indices`` to its 0-based connected-component id.

    Same merge as :func:`connected_components`; a block joins the merged polygon that
    covers its center.
    """
    if len(indices) == 0:
        return {}
    if len(indices) == 1:
        return {int(indices[0]): 0}
    union = unary_union([rect_polygon(*xywh[i]) for i in indices])
    geoms = list(union.geoms) if union.geom_type == "MultiPolygon" else [union]
    labels: dict[int, int] = {}
    for i in indices:
        x, y, w, h = xywh[i]
        pt = Point(x + w / 2, y + h / 2)
        labels[int(i)] = next((c for c, g in enumerate(geoms) if g.covers(pt)), 0)
    return labels
