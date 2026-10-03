"""The direct-regressor trainer (PyTorch Lightning, manual optimization).

Mirrors :class:`trinity.training.DiffusionTrainer` -- same backbone, conditioning, loss
terms, optimizer, scheduler, EMA, refiner, legalizer and ``solve_cases`` interface --
with three differences that define the baseline:

* no framing, time sampler or ODE sampler: one
  deterministic forward maps the conditioning to a layout;
* the regression target is the ground-truth layout in the head's
  space (``z`` or ``xywh``) with a uniform per-sample loss weight;
* evaluation runs the :class:`~trinity_baselines.solver.RegressionPlacer`
  (one candidate per case).
"""

import contextlib
import os

import lightning.pytorch as pl
import numpy as np
import torch
import torch.optim as optim
import wandb
from anyschedule import AnySchedule
from tqdm import tqdm

from trinity.compile_utils import apply_compile, compile_loss_terms
from trinity.conditioning import DEFAULT_COND_OPTIONS
from trinity.decode import z_to_xywh
from trinity.floorplan.data import load_validation_case
from trinity.floorplan.registry import RENDERER, resolve
from trinity.floorplan.scoring import validate
from trinity.floorplan.types import Placement
from trinity.losses.constraint import _ConstraintTerm
from trinity.models import DenoiserCond, get_preset
from trinity.models.ema import EMAModule
from trinity.models.presets import DenoiserArchConfig
from trinity.optim import build_param_groups
from trinity.registry import GRAPH_PE, LATENT_PARAM, REFINER
from trinity.registry import build as trinity_build
from trinity_baselines.losses import LOSS, LossContext
from trinity_baselines.models import DirectRegressor
from trinity_baselines.registry import HEAD, build
from trinity_baselines.solver import RegressionPlacer


class RegressorTrainer(pl.LightningModule):
    """A direct regressor over block-token conditioning.

    ``head`` selects the regression space (``"z"`` or ``"xywh"``); ``refiner`` is a
    ``REFINER`` spec run after the forward at evaluation (``None`` = no refiner);
    ``sample_projections`` are ``PROJECTION`` specs applied to the prediction (none by
    default); ``legalize_portfolio`` is a list of legalization routes (``None`` =
    ``scale_pack``).
    """

    def __init__(
        self,
        *,
        preset: str | None = "ref-d384",
        arch_overrides: dict | None = None,
        head: str = "z",
        losses: list | None = None,
        latent_param: str = "s_only",
        learning_rate: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        optimizer_mode: str = "adamw",
        base_dim: int = 256,
        scheduler_config: dict | None = None,
        gradient_accumulation_steps: int = 1,
        gradient_clip: float = 1.0,
        grad_checkpointing: bool = False,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        compile_config: dict | None = None,
        eval_max_batch: int = 256,
        sample_projections: list | None = None,
        refiner: dict | str | None = None,
        legalize_portfolio: list | None = None,
        eval_scorer: str = "full",
        eval_legalize_workers: int = 0,
        graph_pe=None,
        cond_options=None,
        log_interval: int = 1000,
        sample_block_counts: tuple[int, ...] = (21, 60, 120),
        out_dir: str = "outputs/train",
        name: str = "reg",
        **_unused,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        if preset is None:
            arch = DenoiserArchConfig(**(arch_overrides or {}))
        else:
            arch = get_preset(preset, **(arch_overrides or {}))
        arch.grad_ckpt = grad_checkpointing
        self.head = build(head, HEAD)
        self.arch = arch
        self.backbone = DirectRegressor(arch, self.head)

        self.param = trinity_build(latent_param, LATENT_PARAM)
        loss_specs = losses or [{"name": "denoise"}]
        self.loss_terms = [trinity_build(spec, LOSS) for spec in loss_specs]
        self._needs_geometry = any(
            getattr(term, "needs_geometry", False) for term in self.loss_terms
        )
        if any(isinstance(term, _ConstraintTerm) for term in self.loss_terms):
            raise ValueError("the regressor trainer supports the ref_* aux terms only")
        compile_loss_terms(self.loss_terms, compile_config)

        self.learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay
        self.optimizer_mode = optimizer_mode
        self.base_dim = base_dim
        self.scheduler_config = scheduler_config or {
            "lr": {"mode": "constant", "value": 1.0, "end": -1}
        }
        self.grad_accum = gradient_accumulation_steps
        self.grad_clip = gradient_clip

        self.ema = EMAModule(self.backbone, ema_decay) if use_ema else None
        self.backbone = apply_compile(self.backbone, compile_config)

        self.eval_max_batch = eval_max_batch
        self.sample_projections = list(sample_projections or [])
        self.refiner = refiner
        self.legalize_portfolio = legalize_portfolio
        self.eval_scorer = eval_scorer
        self.eval_legalize_workers = eval_legalize_workers
        self.cond_options = (
            cond_options if cond_options is not None else DEFAULT_COND_OPTIONS
        )
        self.graph_pe = trinity_build(graph_pe or "none", GRAPH_PE)
        if self.graph_pe.dim != arch.graph_pe_dim:
            raise ValueError(
                f"graph_pe builder width {self.graph_pe.dim} != arch.graph_pe_dim "
                f"{arch.graph_pe_dim}; set ARCH_OVERRIDES['graph_pe_dim'] to match "
                "the GRAPH_PE spec"
            )
        self.log_interval = log_interval
        self.sample_block_counts = tuple(sample_block_counts)
        self.out_dir = out_dir
        self.name = name
        self._sample_instances: list | None = None
        self._previous_sample_step = -1
        self.ema_loss = -1.0

    def on_train_start(self) -> None:
        """Check that the loader batch size divides evenly by ``grad_accum``."""
        loader = self.trainer.train_dataloader
        batch_size = getattr(loader, "batch_size", None) if loader is not None else None
        if (
            batch_size is not None
            and self.grad_accum > 1
            and batch_size % self.grad_accum != 0
        ):
            raise ValueError(
                f"batch size {batch_size} must be divisible by "
                f"gradient_accumulation_steps {self.grad_accum}"
            )

    def _micro_loss(self, batch: dict, lo: int, hi: int):
        """Loss and per-term logs of the micro-batch ``batch[lo:hi]``."""
        z_s = self.param.from_latent(batch["z0"][lo:hi])
        token_mask = batch["token_mask"][lo:hi]
        area = batch["area_targets"][lo:hi]
        scale = batch["scale"][lo:hi]
        area_norm = area / scale.unsqueeze(-1) ** 2

        graph_pe = batch["graph_pe"]
        cond = DenoiserCond(
            features=batch["features"][lo:hi],
            adjacency=batch["adjacency"][lo:hi],
            key_pad_mask=token_mask > 0.5,
            graph_pe=graph_pe[lo:hi] if graph_pe is not None else None,
        )

        pred = self.backbone(cond)
        target = self.head.target_from_zs(z_s, area_norm)

        ctx = LossContext(
            pred=pred,
            target=target,
            weight=torch.ones(pred.shape[0], device=pred.device, dtype=pred.dtype),
            t=torch.zeros(pred.shape[0], device=pred.device, dtype=pred.dtype),
            anchor_mask=self.head.expand_mask(batch["anchor_mask"][lo:hi]),
            token_mask=token_mask,
        )
        if self._needs_geometry:
            ones = torch.ones_like(scale)
            ctx.xywh = self.head.to_xywh(pred, area_norm)
            ctx.xywh_gt = z_to_xywh(z_s, area_norm, ones)
            ctx.area_targets = area_norm
            ctx.scale = ones
            ctx.cluster_id = batch["cluster_id"][lo:hi]
            ctx.boundary_code = batch["boundary_code"][lo:hi]
            ctx.mib_id = batch["mib_id"][lo:hi]
            ctx.pin_xy = batch["pin_xy"][lo:hi]
            ctx.pin_w = batch["pin_w"][lo:hi]

        loss = pred.new_zeros(())
        logs: dict[str, torch.Tensor] = {}
        for term in self.loss_terms:
            value, term_logs = term(ctx)
            loss = loss + value
            logs.update(term_logs)
        return loss, logs

    def _batch_graph_pe(self, batch: dict) -> torch.Tensor | None:
        """The graph PE of the whole batch, or ``None`` when the PE is disabled."""
        if self.graph_pe.dim == 0:
            return None
        return self.graph_pe.batched(batch["adj_raw"], batch["token_mask"] > 0.5)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sched = self.lr_schedulers()
        opt.zero_grad(set_to_none=True)

        batch["graph_pe"] = self._batch_graph_pe(batch)
        k = self.grad_accum
        chunk = batch["z0"].shape[0] // k
        total = batch["z0"].new_zeros(())
        total_logs: dict[str, torch.Tensor] = {}
        for i in range(k):
            loss, logs = self._micro_loss(batch, i * chunk, (i + 1) * chunk)
            self.manual_backward(loss / k)
            total = total + loss.detach() / k
            for key, value in logs.items():
                total_logs[key] = total_logs.get(key, 0.0) + value / k

        self.clip_gradients(
            opt, gradient_clip_val=self.grad_clip, gradient_clip_algorithm="norm"
        )
        opt.step()
        sched.step()
        if self.ema is not None:
            self.ema.update(self.backbone)

        self._log_step(total, total_logs)

        if (
            self.log_interval > 0
            and self.global_step % self.log_interval == 0
            and self.global_step != self._previous_sample_step
            and self.trainer.is_global_zero
        ):
            self.log_samples(
                self._sample_cases(), f"{self.out_dir}/{self.name}/samples"
            )
            self._previous_sample_step = self.global_step

    def _log_step(self, total: torch.Tensor, logs: dict) -> None:
        """Log the per-term losses and a step-warmed EMA of the total loss."""
        main = total.item()
        decay = min(0.99, self.global_step / (self.global_step + 10))
        if self.ema_loss < 0:
            self.ema_loss = main
        else:
            self.ema_loss = self.ema_loss * decay + main * (1 - decay)
        self.log("train/loss_ema", self.ema_loss, prog_bar=True, sync_dist=False)
        self.log("train/loss", total, sync_dist=True)
        for key, value in logs.items():
            show = key in ("l2", "denoise")
            self.log(f"train/{key}", value, prog_bar=show, sync_dist=True)

    def _sample_cases(self) -> list:
        """The validation cases rendered in the training loop (loaded once)."""
        if self._sample_instances is None:
            self._sample_instances = [
                load_validation_case(n, allow_download=False)
                for n in self.sample_block_counts
            ]
        return self._sample_instances

    def _resolved_scheduler_config(self) -> dict:
        """The scheduler config with every ``end:
        -1`` replaced by the total step count."""
        try:
            total = int(self.trainer.estimated_stepping_batches)
        except (RuntimeError, ValueError, OverflowError):
            total = 0
        resolved = {}
        for key, sub in self.scheduler_config.items():
            if isinstance(sub, dict) and sub.get("end", -1) in (None, -1) and total > 0:
                sub = {**sub, "end": total}
            resolved[key] = sub
        return resolved

    def configure_optimizers(self):
        params = [p for p in self.backbone.parameters() if p.requires_grad]
        groups = build_param_groups(
            params,
            mode=self.optimizer_mode,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            base_dim=self.base_dim,
        )
        opt = optim.AdamW(
            groups,
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        sched = AnySchedule(opt, config=self._resolved_scheduler_config())
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }

    # ---- inference (sample logging + validation scoring) ---------------------

    @contextlib.contextmanager
    def _eval_mode(self):
        """The backbone in eval mode under EMA
        weights for the duration of the context."""
        was_training = self.backbone.training
        if self.ema is not None:
            ema_ctx = self.ema.use_ema(self.backbone)
        else:
            ema_ctx = contextlib.nullcontext()
        self.backbone.eval()
        try:
            with ema_ctx:
                yield
        finally:
            self.backbone.train(was_training)

    def make_placer(self) -> RegressionPlacer:
        """A one-candidate placer over the current (EMA) regressor."""
        return RegressionPlacer(
            self.backbone,
            trinity_build(self.refiner, REFINER),
            head=self.head,
            projections=self.sample_projections,
            param=self.param,
            samples=1,
            scorer=self.eval_scorer,
            legalize_portfolio=self.legalize_portfolio,
            device=self.device.type,
            max_batch=self.eval_max_batch,
            legalize_workers=self.eval_legalize_workers,
            graph_pe=self.graph_pe,
            cond_options=self.cond_options,
        )

    @torch.no_grad()
    def solve_cases(self, instances, desc: str = "eval: solving"):
        """Place every instance; return ``[(instance,
        score, placement)]`` in input order.

        Sets ``last_candidate_costs`` to each case's
        legalized cost (one candidate per case).
        """
        order = sorted(range(len(instances)), key=lambda i: instances[i].block_count)
        ordered = [instances[i] for i in order]
        per_chunk = max(1, self.eval_max_batch)
        show_bar = self._trainer is None or self.trainer.is_global_zero
        placements_ordered: list = []
        costs_ordered: list = []
        with self._eval_mode():
            placer = self.make_placer()
            with tqdm(
                total=len(ordered), desc=desc, unit="case", disable=not show_bar
            ) as bar:
                for start in range(0, len(ordered), per_chunk):
                    group = ordered[start : start + per_chunk]
                    placements_ordered.extend(placer.solve_batch(group))
                    costs_ordered.extend(placer.last_candidate_costs)
                    bar.update(len(group))
        out: list = [None] * len(instances)
        self.last_candidate_costs = [None] * len(instances)
        for pos, placement, costs in zip(
            order, placements_ordered, costs_ordered, strict=True
        ):
            score = validate(placement, self.eval_scorer).score
            out[pos] = (instances[pos], score, placement)
            self.last_candidate_costs[pos] = costs
        return out

    @torch.no_grad()
    def raw_sample_cases(self, instances):
        """Per case, the raw regressor layout (no
        refiner, no legalizer) as a ``Placement``."""
        order = sorted(range(len(instances)), key=lambda i: instances[i].block_count)
        ordered = [instances[i] for i in order]
        per_chunk = max(1, self.eval_max_batch)
        out = [None] * len(instances)
        with self._eval_mode():
            placer = self.make_placer()
            for start in range(0, len(ordered), per_chunk):
                group = ordered[start : start + per_chunk]
                for gi, boxes in enumerate(placer.raw_sample_boxes(group)):
                    pos = order[start + gi]
                    out[pos] = Placement(
                        xywh=boxes.astype(np.float64), instance=instances[pos]
                    )
        return out

    @torch.no_grad()
    def log_samples(self, instances, out_dir: str) -> list[str]:
        """Render the solved ``instances`` (EMA weights) to PNG; return the paths."""
        os.makedirs(out_dir, exist_ok=True)
        render = resolve("placement", RENDERER)
        paths = []
        for inst, score, placement in self.solve_cases(instances):
            path = f"{out_dir}/step{self.global_step}_n{inst.block_count}.png"
            title = f"step {self.global_step} n={inst.block_count}"
            render(placement, path, score=score, title=title)
            paths.append(path)
        if paths and hasattr(self.logger, "log_image"):
            self.logger.log_image("train/samples", [wandb.Image(p) for p in paths])
        return paths
