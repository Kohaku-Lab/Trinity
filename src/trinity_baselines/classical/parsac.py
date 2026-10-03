"""PARSAC's constraints-aware B*-tree simulated annealing as a ``SOLVER`` (``parsac``).

Wraps the C++ annealer of PARSAC (https://github.com/IntelLabs/parsac, Apache-2.0, arXiv
2405.05495) through torch's C++ extension loader. ``scripts/baselines/build_parsac.sh``
clones it into ``third_party/parsac`` and applies ``scripts/baselines/parsac.patch``,
which adds two setters (``set_outline_slope``, ``set_rotate_prob``; both 0 = the
published cost and move set). ``TRINITY_PARSAC`` overrides the source root and
``TRINITY_PARSAC_BUILD`` the ``-O3`` build directory (default
``third_party/parsac/build``).

A :class:`FloorplanInstance` becomes the engine's integer problem: sizes and pins at
``units_per_s`` units per ``s``, boundary codes and cluster ids one to one, fixed-shape,
preplaced and MIB blocks AR-locked (MIB members share one shape), preplaced blocks
anchored, b2b / p2b edges as 2-pin nets repeated by their rounded weight, and the pin
bounding box (or a given outline) as the fixed outline. A run follows the repository's
``main.py``: 100 stages on its temperature ladder, each followed by an
aspect-ratio-search stage, the budget split evenly over the stages; layout snapshots are
taken after the listed stages. ``anneal(..., init=placement)`` starts from the B*-tree
of a legal layout (left-then-down compaction, then the DAC-2000 tree construction)
instead of a random tree.

The engine keeps a process-global wirelength normalization set by the
first layout it scores, so one process should anneal one instance.
"""

import os
import time
from pathlib import Path

import numpy as np
from torch.utils.cpp_extension import load

from trinity.floorplan.registry import SOLVER
from trinity.floorplan.types import FloorplanInstance, Placement
from trinity_baselines.classical.anneal import AnnealResult, Checkpoint

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "third_party" / "parsac"
DEFAULT_BUILD = DEFAULT_ROOT / "build"
SCHEDULE = {0: 1e-3, 10: 5e-4, 30: 2e-4, 50: 1e-4, 70: 5e-5, 90: 1e-5, 95: 1e-6}
BASE_CONFIG = {
    "beta": 1.0,
    "beta_cluster": 1.0,
    "fixing_prob": 1e-4,
    "alpha": 0.5,
    "beta_preplaced": 10.0,
}
INIT_CONFIG = {
    "alpha": 0.0,
    "T0": 1e-4,
    "ws_ratio": 0.0,
    "chip_ar": 1.0,
    "inverse": 0,
    "ar_search": False,
    "ws_threshold": 10,
}
N_STAGES = 100
_ENGINE = None


def source_root() -> Path:
    return Path(os.environ.get("TRINITY_PARSAC", DEFAULT_ROOT))


def engine():
    """The compiled ``ca_sa`` module (built once
    per build directory, cached per process)."""
    global _ENGINE
    if _ENGINE is None:
        build_dir = Path(os.environ.get("TRINITY_PARSAC_BUILD", DEFAULT_BUILD))
        build_dir.mkdir(parents=True, exist_ok=True)
        source = source_root() / "src" / "c_src" / "ca_sa.cpp"
        _ENGINE = load(
            name="ca_sa_o3",
            sources=[str(source)],
            extra_cflags=["-O3"],
            build_directory=str(build_dir),
        )
    return _ENGINE


def engine_available() -> bool:
    """True when the PARSAC source tree is present."""
    return (source_root() / "src" / "c_src" / "ca_sa.cpp").is_file()


def block_shapes(inst: FloorplanInstance) -> np.ndarray:
    """``(n, 2)`` starting sizes: given shapes
    where known, squares of the area otherwise.

    The members of an MIB group share one shape (a shaped member's, if any).
    """
    targets = inst.target_positions
    wh = np.zeros((inst.block_count, 2))
    known = (targets[:, 2] >= 0) & (targets[:, 3] >= 0)
    wh[known] = targets[known, 2:4]
    side = np.sqrt(inst.area_targets[~known])
    wh[~known] = side[:, None]
    mib = inst.mib_id
    for group in np.unique(mib[mib > 0]):
        members = np.where(mib == group)[0]
        shaped = members[known[members]]
        wh[members] = wh[shaped[0] if len(shaped) else members[0]]
    return wh


def net_list(inst: FloorplanInstance, net_rep_cap: int) -> tuple[list, int]:
    """The 2-pin nets (each edge repeated by its
    rounded weight, capped) and the capped count."""
    n = inst.block_count
    nets, capped = [], 0
    edges = [(int(i), int(j), w) for i, j, w in inst.b2b]
    edges += [(int(b), n + int(p), w) for p, b, w in inst.p2b]
    for a, b, weight in edges:
        reps = max(1, int(round(float(weight))))
        capped += reps > net_rep_cap
        nets += [[a, b]] * min(reps, net_rep_cap)
    return nets, capped


def to_engine_problem(
    inst: FloorplanInstance,
    units_per_s: float,
    outline=None,
    net_rep_cap: int = 10,
    shapes: np.ndarray | None = None,
) -> dict:
    """``{"rows", "pins", "nets", "canvas", "scale",
    "all_fixed", "capped"}`` for the engine.

    ``outline`` = ``(W, H)`` in the instance's units; without it the pin bounding box,
    or a square of 1.05 x the block area when there are no pins. ``shapes`` overrides
    the sizes.
    """
    n = inst.block_count
    scale = units_per_s / inst.s
    targets = inst.target_positions
    wh = block_shapes(inst) if shapes is None else np.asarray(shapes, np.float64)
    size = np.maximum(1, np.rint(wh * scale)).astype(int)
    known = (targets[:, 2] >= 0) & (targets[:, 3] >= 0)
    preplaced = inst.is_preplaced
    cluster = np.zeros(n, dtype=int)
    raw_cluster = inst.cluster_id
    for k, group in enumerate(np.unique(raw_cluster[raw_cluster > 0]), start=1):
        cluster[raw_cluster == group] = k
    shape_locked = (known | (inst.mib_id > 0) | preplaced).astype(int)
    px = np.where(preplaced, np.rint(targets[:, 0] * scale), -1).astype(int)
    py = np.where(preplaced, np.rint(targets[:, 1] * scale), -1).astype(int)
    rows = [
        [
            int(size[i, 0]),
            int(size[i, 1]),
            int(inst.boundary_code[i]),
            int(cluster[i]),
            0,
            int(shape_locked[i]),
            int(preplaced[i]),
            int(px[i]),
            int(py[i]),
        ]
        for i in range(n)
    ]
    pins = np.rint(inst.pins_pos * scale).astype(int).tolist()
    nets, capped = net_list(inst, net_rep_cap)
    if outline is not None:
        canvas = (int(np.ceil(outline[0] * scale)), int(np.ceil(outline[1] * scale)))
    elif len(pins) > 0:
        pin_pos = np.asarray(inst.pins_pos)
        canvas = (
            int(np.ceil(pin_pos[:, 0].max() * scale)),
            int(np.ceil(pin_pos[:, 1].max() * scale)),
        )
    else:
        side = int(np.ceil(np.sqrt(float(inst.area_targets.sum()) * 1.05) * scale))
        canvas = (side, side)
    return {
        "rows": rows,
        "pins": pins,
        "nets": nets,
        "canvas": canvas,
        "scale": scale,
        "all_fixed": bool(shape_locked.all()),
        "capped": int(capped),
    }


def compact(
    xywh: np.ndarray, pinned: np.ndarray | None = None, rounds: int = 10
) -> np.ndarray:
    """Move every unpinned block left, then down,
    as far as no overlap allows, until stable."""
    out = np.array(xywh, dtype=np.int64)
    n = len(out)
    pinned = np.zeros(n, bool) if pinned is None else np.asarray(pinned, bool)
    for _ in range(rounds):
        moved = False
        for axis in (0, 1):
            other = 1 - axis
            for i in np.argsort(out[:, axis], kind="stable"):
                if pinned[i]:
                    continue
                lowest = 0
                for j in range(n):
                    if j == i:
                        continue
                    other_lo, other_hi = (
                        out[j, other],
                        out[j, other] + out[j, other + 2],
                    )
                    overlaps = (
                        other_lo < out[i, other] + out[i, other + 2]
                        and out[i, other] < other_hi
                    )
                    end_j = out[j, axis] + out[j, axis + 2]
                    if overlaps and end_j <= out[i, axis]:
                        lowest = max(lowest, int(end_j))
                if lowest < out[i, axis]:
                    out[i, axis] = lowest
                    moved = True
        if not moved:
            break
    return out


def tree_from_layout(xywh: np.ndarray) -> list[list[int]]:
    """B*-tree edges ``[parent, child, 0 = left / 1
    = right]`` in preorder of a compacted layout.

    Left child = the lowest unvisited block whose left edge is the parent's right edge;
    right child = the lowest unvisited block above the parent with the same left edge;
    blocks the depth-first walk does not reach are hung on the nearest free slot.
    """
    n = len(xywh)
    x, y, w, h = (xywh[:, k].astype(np.int64) for k in range(4))
    left = -np.ones(n, dtype=int)
    right = -np.ones(n, dtype=int)
    visited = np.zeros(n, bool)
    bottom = np.where(y == y.min())[0]
    root = int(bottom[np.argmin(x[bottom])])

    def lowest_unvisited(candidates):
        candidates = [c for c in candidates if not visited[c]]
        if not candidates:
            return -1
        return int(min(candidates, key=lambda c: (y[c], x[c])))

    visited[root] = True
    stack = [root]
    while stack:
        p = stack.pop()
        left_child = lowest_unvisited(np.where(x == x[p] + w[p])[0])
        if left_child >= 0:
            left[p] = left_child
            visited[left_child] = True
        right_child = lowest_unvisited(np.where((x == x[p]) & (y > y[p]))[0])
        if right_child >= 0:
            right[p] = right_child
            visited[right_child] = True
        stack += [c for c in (right_child, left_child) if c >= 0]

    for b in sorted(np.where(~visited)[0], key=lambda b: (x[b], y[b])):
        best, best_distance = None, None
        for p in np.where(visited)[0]:
            if left[p] < 0:
                distance = abs(x[p] + w[p] - x[b]) + abs(y[p] - y[b])
                if best_distance is None or distance < best_distance:
                    best, best_distance = (p, 0), distance
            if right[p] < 0:
                distance = abs(x[p] - x[b]) + abs(y[p] + h[p] - y[b])
                if best_distance is None or distance < best_distance:
                    best, best_distance = (p, 1), distance
        p, side = best
        if side == 0:
            left[p] = b
        else:
            right[p] = b
        visited[b] = True

    edges = []
    stack = [root]
    while stack:
        p = stack.pop()
        if left[p] >= 0:
            edges.append([int(p), int(left[p]), 0])
        if right[p] >= 0:
            edges.append([int(p), int(right[p]), 1])
        stack += [c for c in (right[p], left[p]) if c >= 0]
    return edges


@SOLVER.register("parsac")
class ParsacBTree:
    """PARSAC's staged B*-tree annealer.

    ``solve(inst) -> Placement``; ``anneal(inst, budget, seed, init, outline) ->
    AnnealResult`` with a snapshot after every stage in ``checkpoint_stages`` and a
    ``"final"`` one. ``outline_slope`` and ``rotate_prob`` drive the two setters of
    ``parsac.patch``.
    """

    def __init__(
        self,
        budget: int = 100_000,
        seed: int = 1,
        units_per_s: float = 4000.0,
        checkpoint_stages=tuple(range(0, N_STAGES, 5)),
        ar_search: bool = True,
        config: dict | None = None,
        schedule: dict | None = None,
        net_rep_cap: int = 10,
        outline_weight: float = 1.0,
        outline_slope: float = 0.0,
        rotate_prob: float = 0.0,
    ) -> None:
        self.budget = budget
        self.seed = seed
        self.units_per_s = units_per_s
        self.checkpoint_stages = frozenset(checkpoint_stages)
        self.ar_search = ar_search
        self.config = dict(BASE_CONFIG if config is None else config)
        self.schedule = dict(SCHEDULE if schedule is None else schedule)
        self.net_rep_cap = net_rep_cap
        self.outline_weight = outline_weight
        self.outline_slope = outline_slope
        self.rotate_prob = rotate_prob

    def solve(self, inst: FloorplanInstance) -> Placement:
        return self.anneal(inst).placement

    def anneal(
        self,
        inst: FloorplanInstance,
        budget: int | None = None,
        seed: int | None = None,
        init: Placement | None = None,
        outline=None,
    ) -> AnnealResult:
        """Anneal ``inst`` for ``budget`` steps from
        a random tree or from ``init``'s tree."""
        ca = engine()
        budget = self.budget if budget is None else budget
        seed = self.seed if seed is None else seed
        shapes = None if init is None else init.xywh[:, 2:4]
        problem = to_engine_problem(
            inst, self.units_per_s, outline, self.net_rep_cap, shapes
        )
        setters = {
            "T0": ca.set_T0,
            "beta_preplaced": ca.set_beta_preplaced,
            "beta_cluster": ca.set_beta_cluster,
            "beta": ca.set_beta,
            "alpha": ca.set_alpha,
            "fixing_prob": ca.set_fixing_prob,
            "ar_increment": ca.set_ar_increment,
            "ar_search": ca.ar_search,
            "ws_ratio": ca.set_ws_ratio,
            "inverse": ca.set_inverse,
            "chip_ar": ca.set_chip_ar,
            "ws_threshold": ca.set_ws_thresh,
        }
        for key, value in INIT_CONFIG.items():
            setters[key](value)
        ca.set_seed(int(seed))
        width, height = problem["canvas"]
        ca.set_floorplan_boundaries(
            int(width), int(height), float(self.outline_weight), True
        )
        if self.outline_slope:
            ca.set_outline_slope(float(self.outline_slope))
        if self.rotate_prob:
            ca.set_rotate_prob(float(self.rotate_prob))
        ca.read_nets(problem["nets"])
        ca.read_terminals(problem["pins"])
        ca.read_blocks(problem["rows"], False)
        ca.hard_preplace_constraints(True)
        if init is None:
            ca.initialize(1)
        else:
            boxes = np.rint(init.xywh * problem["scale"]).astype(np.int64)
            boxes[:, 2:] = np.maximum(1, boxes[:, 2:])
            ca.set_btree(tree_from_layout(compact(boxes, inst.is_preplaced)))

        passes = 2 if (self.ar_search and not problem["all_fixed"]) else 1
        per_stage = max(10, budget // (N_STAGES * passes))
        stage_config = dict(self.config)
        sa_seconds = 0.0
        checkpoints = []

        def run_stage(ar_search: bool) -> None:
            nonlocal sa_seconds
            for key, value in stage_config.items():
                setters[key](value)
            setters["ar_search"](ar_search)
            if ar_search:
                setters["ar_increment"](0.1)
            ca.clear_trajectory()
            ca.set_step_limit(per_stage)
            started = time.perf_counter()
            ca.sa_refine()
            sa_seconds += time.perf_counter() - started

        def current_layout() -> tuple[np.ndarray, list[float]]:
            cost = [float(v) for v in ca.compute_cost()]
            xywh = np.array(ca.get_bpos(), dtype=np.float64)[:, :4] / problem["scale"]
            return xywh, cost

        for stage in range(N_STAGES):
            if stage in self.schedule:
                stage_config["T0"] = self.schedule[stage]
            run_stage(False)
            if passes == 2:
                run_stage(True)
            if stage in self.checkpoint_stages:
                xywh, cost = current_layout()
                steps = per_stage * passes * (stage + 1)
                checkpoints.append(
                    Checkpoint(steps, sa_seconds, xywh, cost[6], f"stage{stage}")
                )
        ca.go_to_min_cost_sol()
        xywh, cost = current_layout()
        total_steps = per_stage * passes * N_STAGES
        checkpoints.append(Checkpoint(total_steps, sa_seconds, xywh, cost[6], "final"))
        info = {
            "engine_cost": cost,
            "canvas": problem["canvas"],
            "scale": problem["scale"],
            "per_stage": per_stage,
            "passes": passes,
            "capped_nets": problem["capped"],
            "warm_start": init is not None,
        }
        placement = Placement(xywh=xywh, instance=inst)
        return AnnealResult(placement, sa_seconds, total_steps, checkpoints, info)
