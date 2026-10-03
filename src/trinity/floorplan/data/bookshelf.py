"""Parse the Bookshelf floorplanning format into a :class:`FloorplanInstance`.

The block-level Bookshelf triple of GSRC and MCNC (``.blocks`` / ``.pl`` / ``.nets``):

* ``.blocks`` -- ``NumSoftRectangularBlocks`` / ``NumHardRectilinearBlocks`` /
  ``NumTerminals`` header, then one line per object: a hard block (``name
  hardrectilinear 4 (x,y) (x,y) (x,y) (x,y)`` -> bbox ``(w,h)``), a soft block (``name
  softrectangular AREA min_ar max_ar`` -> area and a ``w/h`` range; the GSRC SOFT files
  give ``min_ar == max_ar``, then the protocol range ``SOFT_ASPECT_RANGE`` applies), or
  a terminal (``name terminal``). Files may mix soft and hard blocks.
* ``.pl`` -- ``name x y [: ORIENT]`` lower-left positions for blocks and terminals. A
  file with every block at the origin (an *unsolved* instance, no reference layout)
  still gives real terminal positions; the loader then leaves ``gt_positions`` empty and
  keeps the terminals.
* ``.nets`` -- ``NetDegree : k`` then ``k`` ``node dir`` lines; nodes are blocks or
  terminals. Hyperedges are clique-expanded into pairwise ``b2b`` / ``p2b`` edges, each
  pair weighted ``1/(degree-1)`` (2-pin nets weight 1), and also kept as index lists in
  ``nets``.

These suites carry no preplaced / MIB / cluster / boundary annotations;
hard blocks can be loaded as fixed-shape blocks (``hard_shapes``).
"""

import re
from pathlib import Path

import numpy as np

from trinity.floorplan.types import FloorplanInstance, Placement

_HARD_RE = re.compile(r"\((-?\d+)\s*,\s*(-?\d+)\)")
_HEADER_PREFIXES = ("UCSC", "UCLA", "Num", "#")
# The w/h range of a soft block whose file gives none.
SOFT_ASPECT_RANGE = (1.0 / 3.0, 3.0)


def _data_lines(path: Path) -> list[str]:
    """Non-blank, non-header lines of a Bookshelf text file."""
    lines = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith(_HEADER_PREFIXES):
            continue
        lines.append(line)
    return lines


def parse_blocks(
    path: Path,
) -> tuple[list[str], dict[str, tuple[float, float, float, float, float]], list[str]]:
    """Parse a ``.blocks`` file.

    Returns ``(block_names, {name: (w, h, area, min_ar, max_ar)}, terminal_names)``:
    soft blocks report ``(-1, -1, area, min_ar, max_ar)``, hard blocks ``(w, h, w*h, -1,
    -1)``.
    """
    block_names: list[str] = []
    blocks: dict[str, tuple[float, float, float, float, float]] = {}
    terminals: list[str] = []
    for line in _data_lines(path):
        parts = line.split()
        name = parts[0]
        if "hardrectilinear" in line:
            xs = [int(a) for a, _ in _HARD_RE.findall(line)]
            ys = [int(b) for _, b in _HARD_RE.findall(line)]
            w, h = float(max(xs) - min(xs)), float(max(ys) - min(ys))
            block_names.append(name)
            blocks[name] = (w, h, w * h, -1.0, -1.0)
        elif "softrectangular" in line:
            area = float(parts[2])
            lo = float(parts[3]) if len(parts) > 3 else -1.0
            hi = float(parts[4]) if len(parts) > 4 else -1.0
            block_names.append(name)
            blocks[name] = (-1.0, -1.0, area, lo, hi)
        elif parts[-1] == "terminal":
            terminals.append(name)
    return block_names, blocks, terminals


def parse_pl(path: Path) -> dict[str, tuple[float, float]]:
    """``{name: (x, y)}`` lower-left positions of a ``.pl`` file (blocks and terminals).

    Only the first three tokens of a line are read (an orientation suffix is ignored).
    """
    pos: dict[str, tuple[float, float]] = {}
    for line in _data_lines(path):
        parts = line.split()
        pos[parts[0]] = (float(parts[1]), float(parts[2]))
    return pos


def parse_nets(path: Path) -> list[list[str]]:
    """The nets of a ``.nets`` file, each a list of node names."""
    nets: list[list[str]] = []
    current: list[str] = []
    for line in _data_lines(path):
        if line.startswith("NetDegree"):
            if current:
                nets.append(current)
            current = []
        else:
            current.append(line.split()[0])
    if current:
        nets.append(current)
    return nets


def clique_edges(
    nets: list[list[str]], block_idx: dict[str, int], pin_idx: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Clique-expand nets into ``b2b`` (block-block) and ``p2b`` (terminal-block) edges.

    Each net of degree ``d`` contributes pairwise edges weighted ``1/(d-1)``; repeated
    block pairs are summed into one edge.
    """
    b2b: dict[tuple[int, int], float] = {}
    p2b: list[tuple[int, int, float]] = []
    for net in nets:
        blocks = [block_idx[n] for n in net if n in block_idx]
        pins = [pin_idx[n] for n in net if n in pin_idx]
        weight = 1.0 / max(1, len(net) - 1)
        for a in range(len(blocks)):
            for b in range(a + 1, len(blocks)):
                key = (min(blocks[a], blocks[b]), max(blocks[a], blocks[b]))
                b2b[key] = b2b.get(key, 0.0) + weight
        for p in pins:
            for b in blocks:
                p2b.append((p, b, weight))
    b2b_arr = (
        np.array([[i, j, w] for (i, j), w in b2b.items()], dtype=np.float32)
        if b2b
        else np.zeros((0, 3), dtype=np.float32)
    )
    p2b_arr = (
        np.array(p2b, dtype=np.float32) if p2b else np.zeros((0, 3), dtype=np.float32)
    )
    return b2b_arr, p2b_arr


def parse_bookshelf_case(
    blocks_path: Path,
    pl_path: Path,
    nets_path: Path,
    hard_shapes: bool = False,
    soft_aspect_range: tuple[float, float] = SOFT_ASPECT_RANGE,
) -> FloorplanInstance:
    """Build a :class:`FloorplanInstance` from one
    Bookshelf ``(.blocks, .pl, .nets)`` triple.

    ``gt_positions`` is set only when the ``.pl`` gives a real block layout (every block
    placed, every shape known, not all at the origin); terminal positions are always
    kept. ``nets`` lists the hyperedges as node ids (blocks ``0..n-1``, then terminals).
    With ``hard_shapes`` every block of known ``(w, h)`` is fixed-shape; every other
    block is soft with ``aspect_bounds`` = the file's ``w/h`` range, else
    ``soft_aspect_range``.
    """
    block_names, blocks, terminals = parse_blocks(blocks_path)
    pos = parse_pl(pl_path)
    nets = parse_nets(nets_path)

    n = len(block_names)
    block_idx = {name: i for i, name in enumerate(block_names)}
    pin_idx = {name: i for i, name in enumerate(terminals)}

    area_targets = np.array([blocks[name][2] for name in block_names], dtype=np.float32)
    pins_pos = np.array([pos[name] for name in terminals], dtype=np.float32).reshape(
        -1, 2
    )
    b2b, p2b = clique_edges(nets, block_idx, pin_idx)
    net_ids = [
        [block_idx[m] for m in net if m in block_idx]
        + [n + pin_idx[m] for m in net if m in pin_idx]
        for net in nets
    ]

    gt = np.array(
        [
            (*pos.get(name, (0.0, 0.0)), blocks[name][0], blocks[name][1])
            for name in block_names
        ],
        dtype=np.float32,
    )
    placed = all(name in pos for name in block_names)
    shapes_known = bool((gt[:, 2] > 0).all())
    has_layout = placed and shapes_known and bool((np.abs(gt[:, :2]) > 0).any())
    gt_positions = gt if has_layout else None
    constraints = np.zeros((n, 5), dtype=np.float32)
    target_positions = np.full((n, 4), -1.0, dtype=np.float32)
    known = gt[:, 2] > 0
    if hard_shapes:
        constraints[known, 0] = 1.0
        target_positions[known, 2:] = gt[known, 2:]
    aspect_bounds = np.full((n, 2), -1.0, dtype=np.float32)
    for i, name in enumerate(block_names):
        if hard_shapes and known[i]:
            continue
        lo, hi = blocks[name][3], blocks[name][4]
        aspect_bounds[i] = (lo, hi) if 0 < lo < hi else soft_aspect_range

    return FloorplanInstance(
        block_count=n,
        area_targets=area_targets,
        constraints=constraints,
        b2b=b2b,
        p2b=p2b,
        pins_pos=pins_pos,
        target_positions=target_positions,
        gt_positions=gt_positions,
        test_id=n,
        nets=[net for net in net_ids if len(net) >= 2],
        aspect_bounds=aspect_bounds,
    )


def fixed_outline(
    inst: FloorplanInstance, gamma: float = 0.10, aspect: float = 1.0
) -> tuple[float, float]:
    """The fixed outline ``(W, H)`` of area ``(1 +
    gamma) * sum(area)`` and ``H / W = aspect``."""
    area = float(inst.area_targets.sum()) * (1.0 + gamma)
    return float(np.sqrt(area / aspect)), float(np.sqrt(area * aspect))


def with_fixed_outline(
    inst: FloorplanInstance,
    gamma: float = 0.10,
    aspect: float = 1.0,
    map_pins: bool = False,
) -> FloorplanInstance:
    """A copy of ``inst`` with the fixed outline set and the reference layout dropped.

    Terminals stay at their file positions; with ``map_pins`` they are mapped affinely
    from their bounding box (widened by the reference layout's, when there is one) onto
    ``[0, W] x [0, H]``.
    """
    W, H = fixed_outline(inst, gamma, aspect)
    pins = np.array(inst.pins_pos, dtype=np.float32).reshape(-1, 2)
    if map_pins and len(pins):
        lo, hi = pins.min(0), pins.max(0)
        if inst.gt_positions is not None:
            g = inst.gt_positions
            lo = np.minimum(lo, g[:, :2].min(0))
            hi = np.maximum(hi, (g[:, :2] + g[:, 2:]).max(0))
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        pins = (pins - lo) / span * np.array([W, H], dtype=np.float32)
    return FloorplanInstance(
        block_count=inst.block_count,
        area_targets=inst.area_targets.copy(),
        constraints=inst.constraints.copy(),
        b2b=inst.b2b.copy(),
        p2b=inst.p2b.copy(),
        pins_pos=pins.astype(np.float32),
        target_positions=inst.target_positions.copy(),
        gt_positions=None,
        metrics=None,
        test_id=inst.test_id,
        nets=None if inst.nets is None else [list(net) for net in inst.nets],
        outline=(W, H),
        aspect_bounds=None if inst.aspect_bounds is None else inst.aspect_bounds.copy(),
    )


def ground_truth_placement(instance: FloorplanInstance) -> Placement:
    """The reference layout of ``instance`` as a :class:`Placement`."""
    if instance.gt_positions is None:
        raise ValueError("instance has no ground-truth positions (unsolved case)")
    return Placement(xywh=instance.gt_positions.copy(), instance=instance)
