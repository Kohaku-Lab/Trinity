"""Parse the FloorSet-Lite tensor format into a :class:`FloorplanInstance`.

On-disk layout of the validation set:

* ``litedata_<id>.pth`` is a ``list`` of layouts; each layout is
  ``[raw[n,6], b2b[E_b,3], p2b[E_p,3], pins[n_pins,2]]`` where ``raw[:,0]`` is the area
  target and ``raw[:,1:6]`` are the 5 constraint columns.
* ``litelabel_<id>.pth`` is a ``list`` of ``(metrics[8], fp_sol)`` where ``fp_sol`` is a
  ``[n, V, 2]`` polygon tensor (``-1``-padded vertices).

The validation set has one layout per file. ``-1`` is the universal padding sentinel;
``block_count`` is recovered from the non-padding area targets.
"""

from pathlib import Path

import numpy as np
import torch

from trinity.floorplan.data.paths import find_floorset_root, validation_case_path
from trinity.floorplan.geometry import polygon_to_bbox
from trinity.floorplan.types import FloorplanInstance, Placement


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def target_positions_from_gt(
    constraints: np.ndarray, gt_positions: np.ndarray | None
) -> np.ndarray:
    """The ``(n, 4)`` targets of a case, read off its ground-truth layout.

    Fixed-shape blocks get ``(-1, -1, w, h)``, preplaced blocks ``(x, y, w, h)``, soft
    blocks all ``-1`` (all ``-1`` without a ground truth).
    """
    n = constraints.shape[0]
    targets = np.full((n, 4), -1.0, dtype=np.float32)
    if gt_positions is None:
        return targets
    fixed = constraints[:, 0] != 0
    preplaced = constraints[:, 1] != 0
    targets[fixed, 2:] = gt_positions[fixed, 2:]
    targets[preplaced, :] = gt_positions[preplaced, :]
    return targets


def parse_layout(
    raw: torch.Tensor,
    b2b: torch.Tensor,
    p2b: torch.Tensor,
    pins: torch.Tensor,
    metrics: torch.Tensor | None = None,
    fp_sol: torch.Tensor | None = None,
    test_id: int | None = None,
) -> FloorplanInstance:
    """Build a :class:`FloorplanInstance` from one raw layout (+ optional label)."""
    raw_np = _np(raw)
    area_targets_full = raw_np[:, 0]
    keep = area_targets_full != -1
    block_count = int(keep.sum())

    area_targets = area_targets_full[keep]
    constraints = raw_np[keep, 1:6]

    gt_positions = None
    if fp_sol is not None:
        polys = _np(fp_sol)
        gt_positions = np.stack(
            [
                np.array(polygon_to_bbox(polys[i]), dtype=np.float32)
                for i in range(block_count)
            ]
        )

    return FloorplanInstance(
        block_count=block_count,
        area_targets=area_targets,
        constraints=constraints,
        b2b=_np(b2b),
        p2b=_np(p2b),
        pins_pos=_np(pins),
        target_positions=target_positions_from_gt(constraints, gt_positions),
        gt_positions=gt_positions,
        metrics=None if metrics is None else _np(metrics),
        test_id=test_id,
    )


def load_validation_case(
    n_blocks: int,
    identifier: int = 1,
    root: str | Path | None = None,
    allow_download: bool = True,
) -> FloorplanInstance:
    """Load one validation case (keyed by block count ``n_blocks in [21,120]``)."""
    floorset_root = find_floorset_root(
        override=None if root is None else str(root), allow_download=allow_download
    )
    data_path, label_path = validation_case_path(floorset_root, n_blocks, identifier)
    layouts = torch.load(data_path, weights_only=False)
    labels = torch.load(label_path, weights_only=False)
    raw, b2b, p2b, pins = layouts[0]
    metrics, fp_sol = labels[0]
    return parse_layout(raw, b2b, p2b, pins, metrics, fp_sol, test_id=n_blocks)


def load_validation_set(
    root: str | Path | None = None, allow_download: bool = True
) -> list[FloorplanInstance]:
    """Load all 100 validation cases (block counts 21..120)."""
    floorset_root = find_floorset_root(
        override=None if root is None else str(root), allow_download=allow_download
    )
    cases = []
    for n in range(21, 121):
        data_path, label_path = validation_case_path(floorset_root, n)
        layouts = torch.load(data_path, weights_only=False)
        labels = torch.load(label_path, weights_only=False)
        raw, b2b, p2b, pins = layouts[0]
        metrics, fp_sol = labels[0]
        cases.append(parse_layout(raw, b2b, p2b, pins, metrics, fp_sol, test_id=n))
    return cases


def ground_truth_placement(instance: FloorplanInstance) -> Placement:
    """The ground-truth layout of ``instance`` as a :class:`Placement`."""
    if instance.gt_positions is None:
        raise ValueError("instance has no ground-truth positions (training case?)")
    return Placement(xywh=instance.gt_positions.copy(), instance=instance)
