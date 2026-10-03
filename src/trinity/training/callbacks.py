"""In-training validation on the held-out dev split (and optionally the official 100 cases).

Every ``every_n_steps`` the callback scores the live (EMA) model in one of two modes:

* ``mode="soft"`` -- the raw sample of every dev case (no refiner, no legalizer), scored with
  the continuous soft metrics; logged as ``val/<metric>``.
* ``mode="full"`` -- sample -> refine -> legalize -> score over the dev cases and, when an
  ``official_shard_fn`` is given, the official 100-case set. The dev cases log the full
  breakdown under ``val/`` (hard cost, feasibility, V_rel, gaps, and the soft metrics of the
  raw sample); the official cases log only ``official_val/cost``.

The sets arrive as ``shard_fn(rank, world)`` callables returning this rank's disjoint slice;
rank zero gathers the per-case rows and writes them straight to the logger experiment.
"""

import os
from collections.abc import Callable

import lightning.pytorch as pl
import torch

from trinity.floorplan.registry import RENDERER, resolve
from trinity.floorplan.scoring.continuous import score_continuous
from trinity.floorplan.scoring.total import total_exp
from trinity.floorplan.types import FloorplanInstance

_DEV = 0  # case tag: dev split (full breakdown -> val/*)
_OFFICIAL = 1  # case tag: official 100-case set (cost only -> official_val/cost)

_SOFT_KEYS = (
    "overlap_ratio",
    "compactness",
    "wl_norm",
    "boundary_dist",
    "group_gap",
    "mib_var",
    "fixed_dist",
    "score_pre",
)


def _soft_row(placement) -> tuple:
    """The continuous soft metrics of ``placement`` in ``_SOFT_KEYS`` order."""
    scores = score_continuous(placement)
    return tuple(getattr(scores, key) for key in _SOFT_KEYS)


class ValidationCallback(pl.Callback):
    """Score the dev (and official) sets with the live model every ``every_n_steps``.

    ``max_dev_cases > 0`` truncates this rank's dev shard in the soft mode.
    """

    def __init__(
        self,
        shard_fn: Callable[[int, int], list[FloorplanInstance]],
        official_shard_fn: Callable[[int, int], list[FloorplanInstance]] | None = None,
        every_n_steps: int = 1000,
        run_at_start: bool = False,
        num_render: int = 6,
        out_dir: str = "outputs/train",
        mode: str = "full",
        max_dev_cases: int = 0,
    ) -> None:
        self.shard_fn = shard_fn
        self.official_shard_fn = official_shard_fn
        self.mode = mode
        self.max_dev_cases = max_dev_cases
        self.every_n_steps = every_n_steps
        self.run_at_start = run_at_start
        self.num_render = num_render
        self.out_dir = out_dir
        self._last_step = -1

    def on_train_start(self, trainer, pl_module) -> None:
        if self.run_at_start:
            self._evaluate(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = trainer.global_step
        if (
            self.every_n_steps > 0
            and step % self.every_n_steps == 0
            and step != self._last_step
        ):
            self._evaluate(trainer, pl_module)
            self._last_step = step

    def _evaluate(self, trainer, pl_module) -> None:
        if self.mode == "soft":
            self._evaluate_soft(trainer, pl_module)
            return
        rank, world = trainer.global_rank, trainer.world_size
        dev = [(_DEV, inst) for inst in self.shard_fn(rank, world)]
        official = (
            [(_OFFICIAL, inst) for inst in self.official_shard_fn(rank, world)]
            if self.official_shard_fn is not None
            else []
        )
        tagged = dev + official
        shard = [inst for _, inst in tagged]
        solved = pl_module.solve_cases(shard, desc="val") if shard else []
        # The soft metrics score the raw sample, not the legalized layout.
        raw = pl_module.raw_sample_cases(shard) if shard else []
        local_scores = []
        for (tag, _), (inst, score, _), raw_placement in zip(
            tagged, solved, raw, strict=True
        ):
            hard = (
                score.cost,
                float(score.feasible),
                score.v_rel,
                score.hpwl_gap,
                score.area_gap,
            )
            local_scores.append(
                (tag, inst.block_count, *hard, *_soft_row(raw_placement))
            )
        scores = self._all_gather(trainer, local_scores)
        if trainer.is_global_zero:
            dev_scores = [s for s in scores if s[0] == _DEV]
            official_scores = [s for s in scores if s[0] == _OFFICIAL]
            self._log(pl_module, dev_scores, "val", cost_only=False)
            self._log(pl_module, official_scores, "official_val", cost_only=True)
            self._render(trainer, pl_module, solved[: len(dev)])

    def _evaluate_soft(self, trainer, pl_module) -> None:
        """Score the raw sample of every dev case with the continuous soft metrics."""
        rank, world = trainer.global_rank, trainer.world_size
        dev = list(self.shard_fn(rank, world))
        if self.max_dev_cases > 0:
            dev = dev[: self.max_dev_cases]
        raw = pl_module.raw_sample_cases(dev) if dev else []
        local = [_soft_row(placement) for placement in raw]
        scores = self._all_gather(trainer, local)
        if not trainer.is_global_zero or not scores:
            return
        n = len(scores)
        metrics = {
            f"val/{key}": sum(row[i] for row in scores) / n
            for i, key in enumerate(_SOFT_KEYS)
        }
        metrics["val/n_cases"] = float(n)
        self._write(pl_module, metrics)

    @staticmethod
    def _write(pl_module, metrics: dict) -> None:
        """Write ``metrics`` to the logger experiment without committing a step."""
        if pl_module.logger is not None and hasattr(pl_module.logger, "experiment"):
            pl_module.logger.experiment.log(metrics, commit=False)

    @staticmethod
    def _all_gather(trainer, local: list[tuple]) -> list[tuple]:
        """Gather every rank's per-case scores into one full-set list (or local if 1 rank)."""
        if trainer.world_size <= 1:
            return local
        buckets: list[list[tuple]] = [None] * trainer.world_size  # type: ignore[list-item]
        torch.distributed.all_gather_object(buckets, local)
        return [row for part in buckets for row in part]

    def _log(
        self, pl_module, scores: list[tuple], prefix: str, cost_only: bool
    ) -> None:
        """Log the aggregate of the full-mode rows ``(tag, n, cost, feasible, v_rel,
        hpwl_gap, area_gap, *soft)``: the exp-weighted total, and the means unless ``cost_only``.
        """
        if not scores:
            return
        block_counts = [int(s[1]) for s in scores]
        costs = [s[2] for s in scores]
        total = total_exp(costs, block_counts)
        if cost_only:
            self._write(pl_module, {f"{prefix}/cost": total})
            return
        n = len(scores)
        keys = ("feasible", "v_rel", "hpwl_gap", "area_gap", *_SOFT_KEYS)
        metrics = {f"{prefix}/total": total}
        for i, key in enumerate(keys):
            metrics[f"{prefix}/{key}"] = sum(s[3 + i] for s in scores) / n
        self._write(pl_module, metrics)

    def _render(self, trainer, pl_module, dev_solved) -> None:
        if self.num_render <= 0 or not dev_solved:
            return
        render = resolve("placement", RENDERER)
        out = f"{self.out_dir}/{pl_module.name}/val"
        os.makedirs(out, exist_ok=True)
        step = trainer.global_step
        stride = max(1, len(dev_solved) // self.num_render)
        for inst, score, placement in dev_solved[::stride][: self.num_render]:
            render(
                placement,
                f"{out}/step{step}_n{inst.block_count}.png",
                score=score,
                title=f"val step {step} n={inst.block_count}",
            )
