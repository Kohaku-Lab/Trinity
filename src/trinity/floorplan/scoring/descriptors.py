"""Fixed-length, permutation-invariant layout
descriptors for distribution-level metrics.

Two embeddings of one layout ``xywh`` ``(n, 4)`` with its instance:

* :func:`density_descriptor` -- occupancy density fields on a ``G x G`` grid: all
  blocks, soft blocks only, fixed+preplaced blocks only, and the net-weight-weighted
  centre density (``4 * G^2`` numbers). ``frame="bbox"`` rasterizes over the layout's
  own bounding box; ``frame="fixed"`` over a square of side ``2s`` centred on the
  layout's bbox centre.
* :func:`position_descriptor` -- moments of the block-centre cloud (covariance,
  skewness, kurtosis, radius of gyration) and 16-bin histograms of nearest-neighbour and
  pairwise centre distances, centres normalized by ``s`` and centred on the bbox centre.

Both are plain functions of the geometry: no metric, no learned encoder.
"""

import numpy as np

from trinity.floorplan.types import FloorplanInstance

_EPS = 1e-9
HIST_BINS = 16
# Histogram range of the nearest-neighbour center distance, in units of s.
NN_RANGE = (0.0, 0.5)
# Histogram range of the pairwise center distance, in units of s.
PAIR_RANGE = (0.0, 2.0)


def _frame(xywh: np.ndarray, s: float, frame: str) -> tuple[float, float, float, float]:
    """The ``(x0, y0, w, h)`` of the rasterization window."""
    x0, y0 = xywh[:, 0].min(), xywh[:, 1].min()
    x1, y1 = (xywh[:, 0] + xywh[:, 2]).max(), (xywh[:, 1] + xywh[:, 3]).max()
    if frame == "bbox":
        return (
            float(x0),
            float(y0),
            float(max(x1 - x0, _EPS)),
            float(max(y1 - y0, _EPS)),
        )
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    return float(cx - s), float(cy - s), float(2 * s), float(2 * s)


def _coverage(xywh: np.ndarray, x0, y0, w, h, G: int) -> np.ndarray:
    """Per-block fractional coverage of each cell:
    ``(n, G, G)`` with values in ``[0, 1]``."""
    ex = x0 + np.arange(G + 1) * (w / G)
    ey = y0 + np.arange(G + 1) * (h / G)
    bx0, by0 = xywh[:, 0:1], xywh[:, 1:2]
    bx1, by1 = bx0 + xywh[:, 2:3], by0 + xywh[:, 3:4]
    ox = np.clip(
        np.minimum(bx1, ex[None, 1:]) - np.maximum(bx0, ex[None, :-1]), 0, None
    )
    oy = np.clip(
        np.minimum(by1, ey[None, 1:]) - np.maximum(by0, ey[None, :-1]), 0, None
    )
    cell = (w / G) * (h / G)
    return (oy[:, :, None] * ox[:, None, :]) / max(cell, _EPS)


def density_descriptor(
    xywh: np.ndarray, inst: FloorplanInstance, G: int = 8, frame: str = "bbox"
) -> np.ndarray:
    """The ``4 * G^2`` density descriptor of one layout in the chosen frame."""
    x0, y0, w, h = _frame(xywh, inst.s, frame)
    coverage = _coverage(xywh, x0, y0, w, h, G)
    soft = inst.is_soft
    locked = ~soft
    all_field = coverage.sum(0)
    soft_field = coverage[soft].sum(0) if soft.any() else np.zeros((G, G))
    locked_field = coverage[locked].sum(0) if locked.any() else np.zeros((G, G))
    weight = np.zeros(inst.block_count)
    b2b = np.asarray(inst.b2b).reshape(-1, 3)
    b2b = b2b[b2b[:, 0] != -1]
    for i, j, wt in b2b:
        i, j = int(i), int(j)
        if i < inst.block_count and j < inst.block_count:
            weight[i] += wt
            weight[j] += wt
    p2b = np.asarray(inst.p2b).reshape(-1, 3)
    p2b = p2b[p2b[:, 0] != -1]
    for _, b, wt in p2b:
        b = int(b)
        if b < inst.block_count:
            weight[b] += wt
    centers = xywh[:, :2] + xywh[:, 2:] / 2
    ix = np.clip(((centers[:, 0] - x0) / w * G).astype(int), 0, G - 1)
    iy = np.clip(((centers[:, 1] - y0) / h * G).astype(int), 0, G - 1)
    net_field = np.zeros((G, G))
    np.add.at(net_field, (iy, ix), weight / max(weight.sum(), _EPS))
    fields = (all_field, soft_field, locked_field, net_field)
    return np.concatenate([field.ravel() for field in fields])


def position_descriptor(xywh: np.ndarray, inst: FloorplanInstance) -> np.ndarray:
    """The ``8 + 2 * HIST_BINS`` center-cloud descriptor of one layout."""
    s = inst.s
    c = (xywh[:, :2] + xywh[:, 2:] / 2) / s
    x0, y0 = xywh[:, 0].min() / s, xywh[:, 1].min() / s
    x1, y1 = (xywh[:, 0] + xywh[:, 2]).max() / s, (xywh[:, 1] + xywh[:, 3]).max() / s
    c = c - np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
    n = c.shape[0]
    cov = np.cov(c.T, bias=True) if n > 1 else np.zeros((2, 2))
    std = np.sqrt(np.maximum(np.diag(cov), _EPS))
    z = (c - c.mean(0)) / std
    skew = (z**3).mean(0)
    kurt = (z**4).mean(0) - 3.0
    rg = np.sqrt((c**2).sum(1).mean())
    moments = np.array(
        [cov[0, 0], cov[1, 1], cov[0, 1], skew[0], skew[1], kurt[0], kurt[1], rg]
    )
    if n > 1:
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        iu = np.triu_indices(n, 1)
        pair = d[iu]
        nn = np.where(np.eye(n, dtype=bool), np.inf, d).min(1)
    else:
        pair = np.zeros(0)
        nn = np.zeros(0)
    h_nn = np.histogram(nn, bins=HIST_BINS, range=NN_RANGE)[0] / max(nn.size, 1)
    h_pair = np.histogram(pair, bins=HIST_BINS, range=PAIR_RANGE)[0] / max(pair.size, 1)
    return np.concatenate([moments, h_nn, h_pair])


DENSITY_DIM = {G: 4 * G * G for G in (8, 12)}
POSITION_DIM = 8 + 2 * HIST_BINS
