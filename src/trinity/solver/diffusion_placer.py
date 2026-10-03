"""The diffusion placer: sample -> refine -> legalize -> select.

``solve_batch`` pads every case (x ``samples`` candidates) to a common length and runs one
sampler + refiner forward over the whole stack, decodes the candidates, legalizes each one
through the legalization portfolio, and keeps the cheapest legalized candidate per case.

Legalization and scoring run on the CPU. With ``legalize_workers > 0`` the candidates of a
batch are legalized in a process pool while the GPU works on the next batch.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from trinity.conditioning import (
    DEFAULT_COND_OPTIONS,
    FEATURE_DIM,
    build_adjacency,
    build_anchors,
    build_b2b_dense,
    build_features,
)
from trinity.decode import z_to_xywh
from trinity.floorplan.legalize import legalize
from trinity.floorplan.scoring import use_fast_hpwl
from trinity.floorplan.types import FloorplanInstance, Placement
from trinity.models import DenoiserCond
from trinity.sampling.refine_case import build_refine_case


def legalize_candidate(inst, candidate_xywh, portfolio, scorer):
    """Legalize one candidate ``(n, 4)``; return ``(feasible, cost, legalized_xywh)``."""
    placement = Placement(xywh=candidate_xywh.astype(np.float64), instance=inst)
    with use_fast_hpwl():
        result = legalize(placement, portfolio=portfolio, scorer=scorer)
    return result.score.feasible, result.score.cost, result.placement.xywh


class DiffusionPlacer:
    """A trained denoiser + sampler + optional refiner + legalization portfolio.

    ``refiner`` is a built ``REFINER`` (``refine(z0, case) -> z``) or ``None``;
    ``legalize_portfolio`` is a list of routes (``None`` = ``scale_pack`` alone);
    ``samples`` candidates are drawn per case and the cheapest legalized one is kept.
    """

    def __init__(
        self,
        backbone,
        sampler,
        refiner=None,
        *,
        param,
        samples: int = 16,
        scorer: str = "full",
        legalize_portfolio: list | None = None,
        device: str = "cpu",
        max_batch: int = 256,
        legalize_workers: int = 0,
        graph_pe,
        cond_options=None,
    ) -> None:
        self.backbone = backbone.to(device).eval()
        self.sampler = sampler
        self.refiner = refiner
        self.param = param
        self.samples = samples
        self.scorer = scorer
        self.graph_pe = graph_pe
        self.cond_options = (
            cond_options if cond_options is not None else DEFAULT_COND_OPTIONS
        )
        self.legalize_portfolio = legalize_portfolio
        self.device = device
        self.max_batch = max_batch
        self.legalize_workers = legalize_workers
        self._pool: ProcessPoolExecutor | None = None
        self.last_candidate_costs: list = []

    def solve(self, inst: FloorplanInstance) -> Placement:
        return self.solve_batch([inst])[0]

    @torch.no_grad()
    def solve_batch(self, instances: list[FloorplanInstance]) -> list[Placement]:
        """Place every instance; generation runs in batches of up to ``max_batch`` rows.

        Sets ``last_candidate_costs``: per case, every candidate's legalized cost, ascending.
        """
        per_batch = max(1, self.max_batch // self.samples)
        pending = []
        self._start_pool()
        try:
            for start in range(0, len(instances), per_batch):
                group = instances[start : start + per_batch]
                for inst, candidates in zip(group, self.candidates(group), strict=True):
                    pending.append((inst, self._submit(inst, candidates)))
            placements, self.last_candidate_costs = [], []
            for inst, jobs in pending:
                placement, costs = self._select(inst, jobs)
                placements.append(placement)
                self.last_candidate_costs.append(costs)
        finally:
            self._stop_pool()
        return placements

    def build_group_ctx(self, group: list[FloorplanInstance]) -> dict:
        """The padded ``len(group) x samples`` conditioning of one GPU forward.

        Returns ``{"cond", "rows", "max_n", "k", "area", "scale"}``: the ``DenoiserCond`` the
        sampler reads, the row count, padded length and candidates per case, and the per-row
        target areas and layout scales the decoder reads.
        """
        k = self.samples
        max_n = max(inst.block_count for inst in group)
        rows = len(group) * k

        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(*shape, dtype=dtype, device=self.device)

        def as_tensor(array):
            return torch.from_numpy(np.asarray(array)).to(self.device)

        features = zeros(rows, max_n, FEATURE_DIM)
        adjacency_log = zeros(rows, max_n, max_n)
        adjacency_raw = zeros(rows, max_n, max_n)
        anchor_z = zeros(rows, max_n, 3)
        anchor_mask = zeros(rows, max_n, 3)
        token_mask = zeros(rows, max_n)
        area = zeros(rows, max_n)
        scale = zeros(rows)
        mib_soft_group = zeros(rows, max_n, dtype=torch.long)

        for gi, inst in enumerate(group):
            n = inst.block_count
            rows_of_case = slice(gi * k, (gi + 1) * k)
            case_anchor_z, case_anchor_mask, case_mib_group = build_anchors(
                inst, self.cond_options.mib_anchor
            )
            case_anchor_z = self.param.to_latent(case_anchor_z)
            feats = build_features(
                inst, case_anchor_z, case_anchor_mask, self.cond_options
            )
            features[rows_of_case, :n] = as_tensor(feats)
            adjacency_log[rows_of_case, :n, :n] = as_tensor(
                build_adjacency(inst, self.cond_options.graph_bias_normalize)
            )
            adjacency_raw[rows_of_case, :n, :n] = as_tensor(build_b2b_dense(inst))
            anchor_z[rows_of_case, :n] = as_tensor(case_anchor_z)
            anchor_mask[rows_of_case, :n] = as_tensor(case_anchor_mask)
            mib_soft_group[rows_of_case, :n] = as_tensor(case_mib_group)
            token_mask[rows_of_case, :n] = 1.0
            area[rows_of_case, :n] = as_tensor(inst.area_targets.astype(np.float32))
            scale[rows_of_case] = float(inst.s)

        key_pad_mask = token_mask > 0.5
        graph_pe = None
        if self.graph_pe.dim > 0:
            graph_pe = self.graph_pe.batched(adjacency_raw, key_pad_mask)
        cond = DenoiserCond(
            features=features,
            adjacency=adjacency_log,
            anchor_z=anchor_z,
            anchor_mask=anchor_mask,
            key_pad_mask=key_pad_mask,
            graph_pe=graph_pe,
            mib_soft_group=mib_soft_group,
        )
        return {
            "cond": cond,
            "rows": rows,
            "max_n": max_n,
            "k": k,
            "area": area,
            "scale": scale,
        }

    def sample_latents(self, ctx: dict, noise=None) -> torch.Tensor:
        """Sampled ``x0`` latents ``(rows, max_n, 3)`` for ``ctx`` (``noise`` = the start state)."""
        shape = (ctx["rows"], ctx["max_n"], 3)
        z = self.sampler.sample(
            self.backbone, ctx["cond"], shape, self.device, noise=noise
        )
        return self.param.from_latent(z)

    def refine_decode(
        self, group: list[FloorplanInstance], z: torch.Tensor, ctx: dict
    ) -> list:
        """Refine ``z`` (when there is a refiner) and decode it: ``[(samples, n, 4)]`` per case."""
        if self.refiner is not None:
            z = self.refiner.refine(z, build_refine_case(group, ctx["k"], self.device))
        return self.decode(group, z, ctx)

    def decode(
        self, group: list[FloorplanInstance], z: torch.Tensor, ctx: dict
    ) -> list:
        """Per-case candidate boxes ``[(samples, n, 4)]`` decoded from ``z``."""
        xywh = z_to_xywh(z, ctx["area"], ctx["scale"]).cpu().numpy()
        k = ctx["k"]
        return [
            xywh[gi * k : (gi + 1) * k, : inst.block_count]
            for gi, inst in enumerate(group)
        ]

    def candidates(self, group: list[FloorplanInstance]) -> list:
        """Sample, refine and decode ``group``: per-case candidate boxes ``[(samples, n, 4)]``."""
        ctx = self.build_group_ctx(group)
        return self.refine_decode(group, self.sample_latents(ctx), ctx)

    def candidates_from_latents(self, group: list[FloorplanInstance], z_cached) -> list:
        """Refine and decode pre-sampled ``x0`` latents ``(len(group) * samples, max_n, 3)``."""
        ctx = self.build_group_ctx(group)
        z = torch.as_tensor(z_cached, dtype=torch.float32, device=self.device)
        return self.refine_decode(group, z, ctx)

    def raw_sample_boxes(self, group: list[FloorplanInstance]) -> list[np.ndarray]:
        """The first raw candidate (no refiner, no legalizer) of each case, ``[(n, 4)]``."""
        ctx = self.build_group_ctx(group)
        boxes = self.decode(group, self.sample_latents(ctx), ctx)
        return [case_boxes[0] for case_boxes in boxes]

    def _start_pool(self) -> None:
        """Start the legalization pool (``forkserver``: the workers must not inherit CUDA)."""
        if self.legalize_workers > 0 and self._pool is None:
            context = mp.get_context("forkserver")
            self._pool = ProcessPoolExecutor(
                max_workers=self.legalize_workers, mp_context=context
            )

    def _stop_pool(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def _submit(self, inst: FloorplanInstance, candidates: np.ndarray) -> list:
        """Legalize the candidates of one case (in the pool when there is one)."""
        args = (self.legalize_portfolio, self.scorer)
        if self._pool is None:
            return [legalize_candidate(inst, cand, *args) for cand in candidates]
        return [
            self._pool.submit(legalize_candidate, inst, cand, *args)
            for cand in candidates
        ]

    def _select(
        self, inst: FloorplanInstance, jobs: list
    ) -> tuple[Placement, list[float]]:
        """The cheapest legalized candidate and every candidate's cost, ascending."""
        results = [job.result() if self._pool is not None else job for job in jobs]
        ranked = sorted(results, key=lambda result: result[1])
        best_xywh = ranked[0][2]
        costs = [float(cost) for _feasible, cost, _xywh in ranked]
        return Placement(xywh=best_xywh, instance=inst), costs
