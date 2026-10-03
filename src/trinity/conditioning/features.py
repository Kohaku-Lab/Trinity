"""Per-block conditioning features and the netlist adjacency of one case.

Per block, a ``FEATURE_DIM = 19`` vector:

  1  block scale ``sqrt(area) / s``
  2  fixed-shape and preplaced flags
  1  connectivity degree (max-normalized)
  4  required boundary edges (left, right, top, bottom)
  2  MIB and cluster membership flags
  3  anchor latent ``(cx/s, cy/s, rho)`` and 3 anchor mask
  3  pin pull: weighted-mean pin position ``/ s`` and ``log1p`` total pin weight

and a dense ``log1p`` b2b adjacency ``(n, n)`` used as the attention graph bias.
"""

from dataclasses import dataclass

import numpy as np
import torch

from trinity.conditioning.anchors import build_anchors
from trinity.floorplan.types import BOUNDARY_EDGES, FloorplanInstance

FEATURE_DIM = 19

# Boundary code -> (left, right, top, bottom) bits; codes not in BOUNDARY_EDGES stay zero.
_MAX_BOUNDARY_CODE = max(BOUNDARY_EDGES)
_EDGE_INDEX = {"left": 0, "right": 1, "top": 2, "bottom": 3}
_BOUNDARY_BITS = np.zeros((_MAX_BOUNDARY_CODE + 1, 4), dtype=np.float32)
for _code, _names in BOUNDARY_EDGES.items():
    for _name in _names:
        _BOUNDARY_BITS[_code, _EDGE_INDEX[_name]] = 1.0


@dataclass(frozen=True)
class CondOptions:
    """How the per-block conditioning is normalized.

    * ``pin_pull_normalize`` -- max-normalize the pin-pull strength to ``[0, 1]`` per case.
    * ``graph_bias_normalize`` -- divide the ``log1p`` b2b adjacency by its per-case max.
    * ``mib_anchor`` -- the anchor of an all-soft MIB group: ``group_mean`` | ``square`` |
      ``none`` (see :mod:`trinity.conditioning.anchors`).
    """

    pin_pull_normalize: bool = False
    graph_bias_normalize: bool = False
    mib_anchor: str = "group_mean"


DEFAULT_COND_OPTIONS = CondOptions()


def build_b2b_dense(inst: FloorplanInstance) -> np.ndarray:
    """The dense symmetric b2b weight matrix ``(n, n)``, repeated pairs summed in edge order."""
    n = inst.block_count
    adj = np.zeros((n, n), dtype=np.float32)
    edges = np.asarray(inst.b2b, dtype=np.float64).reshape(-1, 3)
    if edges.size == 0:
        return adj
    i = edges[:, 0].astype(np.int64)
    j = edges[:, 1].astype(np.int64)
    keep = (i >= 0) & (i < n) & (j < n)
    i, j = i[keep], j[keep]
    w = edges[keep, 2].astype(np.float32)
    np.add.at(adj, (i, j), w)
    np.add.at(adj, (j, i), w)
    return adj


def build_adjacency(
    inst: FloorplanInstance,
    normalize: bool = False,
    b2b_dense: np.ndarray | None = None,
) -> np.ndarray:
    """The ``log1p`` b2b weight matrix ``(n, n)`` of the graph bias.

    ``normalize`` divides it by its max; ``b2b_dense`` reuses an already built
    :func:`build_b2b_dense`.
    """
    adj = np.log1p(build_b2b_dense(inst) if b2b_dense is None else b2b_dense)
    if normalize:
        adj = adj / max(float(adj.max()), 1e-6)
    return adj


def pin_targets(inst: FloorplanInstance) -> tuple[np.ndarray, np.ndarray]:
    """The per-block pin pull ``(pin_xy (n, 2), pin_w (n,))``.

    ``pin_xy`` is the weighted-mean position of a block's pins (absolute coordinates, 0 for
    a block without pins) and ``pin_w`` its total p2b weight.
    """
    n = inst.block_count
    # Columns: sum(w * x), sum(w * y), sum(w).
    acc = np.zeros((n, 3), dtype=np.float64)
    edges = np.asarray(inst.p2b, dtype=np.float64).reshape(-1, 3)
    if edges.size:
        p = edges[:, 0].astype(np.int64)
        b = edges[:, 1].astype(np.int64)
        keep = (b >= 0) & (b < n) & (p >= 0) & (p < inst.pins_pos.shape[0])
        p, b, w = p[keep], b[keep], edges[keep, 2]
        pxy = inst.pins_pos[p]
        # The weighted coordinates are formed in float32.
        w32 = w.astype(np.float32)
        np.add.at(acc, (b, 0), w32 * pxy[:, 0])
        np.add.at(acc, (b, 1), w32 * pxy[:, 1])
        np.add.at(acc, (b, 2), w)
    pin_w = acc[:, 2].astype(np.float32)
    pin_xy = np.zeros((n, 2), dtype=np.float32)
    has = pin_w > 0
    pin_xy[has] = (acc[has, :2] / acc[has, 2:3]).astype(np.float32)
    return pin_xy, pin_w


def pin_edges(inst: FloorplanInstance) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The valid p2b edges: ``(pin_xy (P, 2) absolute, weight (P,), block index (P,))``."""
    n = inst.block_count
    edges = np.asarray(inst.p2b, dtype=np.float64).reshape(-1, 3)
    if edges.size == 0:
        return (
            np.zeros((0, 2), np.float32),
            np.zeros(0, np.float32),
            np.zeros(0, np.int64),
        )
    p = edges[:, 0].astype(np.int64)
    b = edges[:, 1].astype(np.int64)
    keep = (b >= 0) & (b < n) & (p >= 0) & (p < inst.pins_pos.shape[0])
    p, b, w = p[keep], b[keep], edges[keep, 2]
    return inst.pins_pos[p].astype(np.float32), w.astype(np.float32), b


def _pin_pull(
    inst: FloorplanInstance, normalize: bool = False, pins: tuple | None = None
) -> np.ndarray:
    """The ``(n, 3)`` pin-pull features ``(pin x / s, pin y / s, log1p(pin weight))``.

    ``normalize`` max-normalizes the strength column to ``[0, 1]``.
    """
    pin_xy, pin_w = pin_targets(inst) if pins is None else pins
    out = np.zeros((inst.block_count, 3), dtype=np.float32)
    out[:, :2] = pin_xy / inst.s
    strength = np.log1p(pin_w)
    if normalize:
        strength = strength / max(strength.max(), 1e-6)
    out[:, 2] = strength
    return out


def build_features(
    inst: FloorplanInstance,
    anchor_z: np.ndarray,
    anchor_mask: np.ndarray,
    options: CondOptions = DEFAULT_COND_OPTIONS,
    b2b_dense: np.ndarray | None = None,
    pins: tuple | None = None,
) -> np.ndarray:
    """The ``(n, FEATURE_DIM)`` per-block features (layout in the module docstring).

    ``b2b_dense`` / ``pins`` reuse an already built :func:`build_b2b_dense` /
    :func:`pin_targets`.
    """
    s = inst.s
    adj = build_adjacency(inst, b2b_dense=b2b_dense)
    degree = adj.sum(axis=1)
    degree = degree / max(degree.max(), 1e-6)

    boundary_bits = _BOUNDARY_BITS[np.clip(inst.boundary_code, 0, _MAX_BOUNDARY_CODE)]

    feats = np.concatenate(
        [
            (np.sqrt(inst.area_targets) / s)[:, None],
            inst.is_fixed.astype(np.float32)[:, None],
            inst.is_preplaced.astype(np.float32)[:, None],
            degree[:, None],
            boundary_bits,
            (inst.mib_id > 0).astype(np.float32)[:, None],
            (inst.cluster_id > 0).astype(np.float32)[:, None],
            anchor_z,
            anchor_mask,
            _pin_pull(inst, options.pin_pull_normalize, pins),
        ],
        axis=1,
    ).astype(np.float32)
    return feats


def build_conditioning(
    inst: FloorplanInstance, options: CondOptions = DEFAULT_COND_OPTIONS
) -> dict[str, torch.Tensor]:
    """Every conditioning tensor of one case, as CPU torch tensors.

    Keys: ``features``, ``adjacency`` (graph bias), ``adj_raw`` (dense b2b weights, input of
    the graph PE), ``anchor_z`` / ``anchor_mask`` (``/ s`` latent space), ``pin_xy`` /
    ``pin_w``, ``pin_edge_xy`` / ``pin_edge_w`` / ``pin_edge_block``, ``mib_soft_group``.
    """
    anchor_z, anchor_mask, mib_soft_group = build_anchors(inst, options.mib_anchor)
    b2b_dense = build_b2b_dense(inst)
    pins = pin_targets(inst)
    features = build_features(inst, anchor_z, anchor_mask, options, b2b_dense, pins)
    adjacency = build_adjacency(inst, options.graph_bias_normalize, b2b_dense)
    edge_xy, edge_w, edge_block = pin_edges(inst)
    return {
        "features": torch.from_numpy(features),
        "adjacency": torch.from_numpy(adjacency),
        "adj_raw": torch.from_numpy(b2b_dense),
        "anchor_z": torch.from_numpy(anchor_z),
        "anchor_mask": torch.from_numpy(anchor_mask),
        "pin_xy": torch.from_numpy(pins[0]),
        "pin_w": torch.from_numpy(pins[1]),
        "pin_edge_xy": torch.from_numpy(edge_xy),
        "pin_edge_w": torch.from_numpy(edge_w),
        "pin_edge_block": torch.from_numpy(edge_block),
        "mib_soft_group": torch.from_numpy(mib_soft_group),
    }
