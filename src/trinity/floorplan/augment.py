"""Label-preserving rigid transforms of a floorplan case.

The transforms are the dihedral group of the square (the four 90-degree rotations, with an
optional x mirror) plus a global translation. Block boxes, target and ground-truth
positions, pins and boundary codes move together, so HPWL, bbox area and the constraint
violations of the ground truth are unchanged.

* translation -- shifts every coordinate; ``shift`` is in raw layout units.
* rotation by ``k * 90`` degrees -- rotates centers about the origin and swaps ``w`` / ``h``.
* mirror -- negates x.

The boundary bitmask is ``1=left 2=right 4=top 8=bottom``; each op permutes those bits.
"""

from dataclasses import replace

import numpy as np

from trinity.floorplan.types import COL_BOUNDARY, FloorplanInstance

# Boundary-bit permutation of one CCW 90-degree turn: L->B, B->R, R->T, T->L.
_ROT90_CCW_BITS = {1: 8, 8: 2, 2: 4, 4: 1}
# Boundary-bit permutation of the x mirror: L<->R.
_FLIPX_BITS = {1: 2, 2: 1, 4: 4, 8: 8}


def _remap_code(code: int, bit_map: dict[int, int]) -> int:
    """``code`` with every single-edge bit replaced by its image under ``bit_map``."""
    out = 0
    for bit, new_bit in bit_map.items():
        if code & bit:
            out |= new_bit
    return int(out)


def _free_rows(xywh: np.ndarray) -> np.ndarray:
    """The rows that are the all-``-1`` placeholder (a negative width)."""
    return xywh[:, 2] < 0


def _xywh_to_cxywh(xywh: np.ndarray) -> np.ndarray:
    """``(x, y, w, h)`` -> ``(cx, cy, w, h)``, placeholder rows unchanged."""
    out = xywh.copy()
    free = _free_rows(xywh)
    out[~free, 0] = xywh[~free, 0] + xywh[~free, 2] / 2.0
    out[~free, 1] = xywh[~free, 1] + xywh[~free, 3] / 2.0
    return out


def _rot90_centers(cx: np.ndarray, cy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One CCW 90deg rotation about the origin: ``(x,y) -> (-y, x)``."""
    return -cy, cx


def transform_instance(
    inst: FloorplanInstance,
    *,
    rot: int = 0,
    flip: bool = False,
    shift: tuple[float, float] = (0.0, 0.0),
) -> FloorplanInstance:
    """Apply a rigid transform to a whole case; return a new :class:`FloorplanInstance`.

    ``rot`` CCW 90-degree turns are applied first, then the x mirror when ``flip``. The
    ground-truth bbox minimum is then moved back to the origin and the case is translated
    by ``shift = (dx, dy)`` (raw layout units). Free coordinates stay free.
    """
    rot %= 4
    dx, dy = shift

    def _rotflip_center(cx, cy):
        for _ in range(rot):
            cx, cy = _rot90_centers(cx, cy)
        return (-cx, cy) if flip else (cx, cy)

    gt = inst.gt_positions.astype(np.float64)
    gcx, gcy = gt[:, 0] + gt[:, 2] / 2.0, gt[:, 1] + gt[:, 3] / 2.0
    rgx, rgy = _rotflip_center(gcx.copy(), gcy.copy())
    gw, gh = (gt[:, 3], gt[:, 2]) if rot % 2 else (gt[:, 2], gt[:, 3])
    ax = dx - (rgx - gw / 2.0).min()
    ay = dy - (rgy - gh / 2.0).min()

    def move_boxes(xywh: np.ndarray | None) -> np.ndarray | None:
        if xywh is None:
            return None
        free = _free_rows(xywh)
        c = _xywh_to_cxywh(xywh.astype(np.float64))
        cx, cy, w, h = c[:, 0].copy(), c[:, 1].copy(), c[:, 2].copy(), c[:, 3].copy()
        for _ in range(rot):
            cx, cy = _rot90_centers(cx, cy)
            w, h = h.copy(), w.copy()
        if flip:
            cx = -cx
        out = np.stack([cx + ax - w / 2.0, cy + ay - h / 2.0, w, h], axis=1)
        out[free] = -1.0
        return out.astype(xywh.dtype)

    def move_pins(pins: np.ndarray) -> np.ndarray:
        if pins.size == 0:
            return pins.copy()
        p = pins.astype(np.float64)
        px, py = p[:, 0].copy(), p[:, 1].copy()
        # A placeholder pin is (-1, -1) on both coordinates.
        valid = ~((pins[:, 0] == -1) & (pins[:, 1] == -1))
        for _ in range(rot):
            px, py = _rot90_centers(px, py)
        if flip:
            px = -px
        out = np.stack([px + ax, py + ay], axis=1)
        out[~valid] = -1.0
        return out.astype(pins.dtype)

    constraints = inst.constraints.copy()
    codes = constraints[:, COL_BOUNDARY].astype(np.int64)
    new_codes = codes.copy()
    for _ in range(rot):
        new_codes = np.array([_remap_code(int(c), _ROT90_CCW_BITS) for c in new_codes])
    if flip:
        new_codes = np.array([_remap_code(int(c), _FLIPX_BITS) for c in new_codes])
    constraints[:, COL_BOUNDARY] = new_codes.astype(constraints.dtype)

    return replace(
        inst,
        constraints=constraints,
        pins_pos=move_pins(inst.pins_pos),
        target_positions=move_boxes(inst.target_positions),
        gt_positions=move_boxes(inst.gt_positions),
    )


def random_transform(
    inst: FloorplanInstance,
    rng: np.random.Generator,
    *,
    shift_std: float = 0.1,
    rotate: bool = True,
    flip: bool = True,
) -> FloorplanInstance:
    """Apply a random rotation / mirror / Gaussian shift to ``inst``.

    The shift is ``N(0, shift_std * s)`` per axis; ``rotate`` enables a random 90-degree
    turn and ``flip`` a random x mirror.
    """
    rot = int(rng.integers(0, 4)) if rotate else 0
    do_flip = bool(rng.integers(0, 2)) if flip else False
    s = inst.s
    dx = float(rng.normal(0.0, shift_std)) * s
    dy = float(rng.normal(0.0, shift_std)) * s
    return transform_instance(inst, rot=rot, flip=do_flip, shift=(dx, dy))
