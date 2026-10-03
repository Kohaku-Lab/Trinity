"""Sequence-pair simulated annealing, ported from
BBOPlace-Bench (``placer=sp``, ``algorithm=sa``).

Source: https://github.com/lamda-bbo/BBOPlace-Bench (MIT; licenses: NOTICE). Copied from
``src/placer/sp_placer.py`` (the decode), ``src/algorithm/sa/sa.py`` (the annealing
loop), ``src/operators/sampling.py`` (random permutation pair) and pymoo's
``InversionMutation`` (reverse a random segment of each sequence), at
``config/algorithm/sa.yaml`` (``T=100``, ``decay=0.99`` every ``update_freq=100``
evaluations) and ``default.yaml`` (``max_evals=10000``). The DAG longest path is the
same recurrence over the same topological order, in numpy instead of igraph. Fitness is
the scorer's HPWL, which equals theirs at zero pin offset.

A soft block enters at the square of its area; a fixed block at its shape. The layout
the SA returns has no overlap by construction of the sequence pair. ``anneal`` records
the best layout so far at the listed evaluation counts as checkpoints.
"""

import time

import numpy as np

from trinity.floorplan.registry import SOLVER
from trinity.floorplan.scoring.cost import hpwl_fast
from trinity.floorplan.types import FloorplanInstance, Placement
from trinity_baselines.classical.anneal import AnnealResult, Checkpoint


def _shapes(inst: FloorplanInstance) -> np.ndarray:
    """Return ``(n, 2)`` block sizes: known ``(w,
    h)`` where given, else a square of the area."""
    tp = inst.target_positions
    wh = np.zeros((inst.block_count, 2), dtype=np.float64)
    known = (tp[:, 2] >= 0) & (tp[:, 3] >= 0)
    wh[known] = tp[known, 2:4]
    side = np.sqrt(inst.area_targets[~known])
    wh[~known] = np.stack([side, side], axis=1)
    return wh


def decode_sequence_pair(
    seq1: np.ndarray, seq2: np.ndarray, wh: np.ndarray, gap: tuple[float, float]
) -> np.ndarray:
    """Return ``(n, 2)`` lower-left corners for
    sequence pair ``(seq1, seq2)`` by DAG longest path.

    ``seq1`` / ``seq2`` are permutations; ``i`` precedes ``j`` horizontally when it
    precedes it in both, vertically when it precedes in ``seq2`` only. ``gap`` is the
    spacing per axis.
    """
    n = wh.shape[0]
    pos1 = np.empty(n, dtype=np.int64)
    pos1[seq1] = np.arange(n)
    # Row k (in seq2 order): the earlier blocks left of block k, then those below it.
    p1 = pos1[seq2]
    earlier = np.tri(n, n, -1, dtype=bool)
    left = earlier & (p1[None, :] < p1[:, None])
    below = earlier & (p1[None, :] > p1[:, None])
    w = wh[seq2, 0] + gap[0]
    h = wh[seq2, 1] + gap[1]
    x = np.zeros(n)
    y = np.zeros(n)
    for k in range(1, n):
        lk, bk = left[k], below[k]
        if lk.any():
            x[k] = np.max((x + w)[lk])
        if bk.any():
            y[k] = np.max((y + h)[bk])
    out = np.empty((n, 2))
    out[seq2, 0] = x
    out[seq2, 1] = y
    return out


def _invert_segment(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return ``seq`` with one random contiguous segment reversed."""
    n = seq.shape[0]
    a, b = np.sort(rng.integers(0, n, size=2))
    out = seq.copy()
    out[a : b + 1] = out[a : b + 1][::-1]
    return out


@SOLVER.register("sp_sa")
class SequencePairSA:
    """BBOPlace-Bench sequence-pair SA: ``solve(inst) -> Placement`` minimizing HPWL."""

    def __init__(
        self,
        max_evals: int = 10000,
        T: float = 100.0,
        decay: float = 0.99,
        update_freq: int = 100,
        seed: int = 1,
        checkpoint_evals=(),
    ) -> None:
        self.max_evals = max_evals
        self.T0 = T
        self.decay = decay
        self.update_freq = update_freq
        self.seed = seed
        self.checkpoint_evals = frozenset(int(e) for e in checkpoint_evals)

    def _layout(self, inst, wh, gap, seq1, seq2) -> Placement:
        xy = decode_sequence_pair(seq1, seq2, wh, gap)
        return Placement(xywh=np.concatenate([xy, wh], axis=1), instance=inst)

    def solve(self, inst: FloorplanInstance) -> Placement:
        """Anneal from a random sequence pair; return the best HPWL layout found."""
        return self.anneal(inst).placement

    def anneal(
        self,
        inst: FloorplanInstance,
        budget: int | None = None,
        seed: int | None = None,
    ) -> AnnealResult:
        """Anneal for ``budget`` evaluations; the best layout
        so far is snapshotted at ``checkpoint_evals``."""
        budget = self.max_evals if budget is None else budget
        rng = np.random.default_rng(self.seed if seed is None else seed)
        n = inst.block_count
        wh = _shapes(inst)
        gap = (inst.s / 1000.0, inst.s / 1000.0)
        seq1, seq2 = rng.permutation(n), rng.permutation(n)
        t0 = time.perf_counter()
        cur = self._layout(inst, wh, gap, seq1, seq2)
        cur_f = hpwl_fast(cur)
        best, best_f = cur, cur_f
        checkpoints = []
        T = self.T0
        for k in range(1, budget):
            s1 = _invert_segment(seq1, rng)
            s2 = _invert_segment(seq2, rng)
            cand = self._layout(inst, wh, gap, s1, s2)
            f = hpwl_fast(cand)
            if f <= cur_f or rng.random() < np.exp((cur_f - f) / T):
                seq1, seq2, cur, cur_f = s1, s2, cand, f
                if f < best_f:
                    best, best_f = cand, f
            if k % self.update_freq == 0:
                T *= self.decay
            if k in self.checkpoint_evals:
                checkpoints.append(
                    Checkpoint(
                        k,
                        time.perf_counter() - t0,
                        best.xywh.copy(),
                        float(best_f),
                        f"eval{k}",
                    )
                )
        seconds = time.perf_counter() - t0
        checkpoints.append(
            Checkpoint(budget, seconds, best.xywh.copy(), float(best_f), "final")
        )
        return AnnealResult(best, seconds, budget, checkpoints, {"hpwl": float(best_f)})
