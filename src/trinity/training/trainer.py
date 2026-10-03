"""The diffusion trainer (PyTorch Lightning, manual optimization).

``DiffusionTrainer`` builds the backbone, framing, timestep sampler and loss terms once
from their specs and runs a flat manual-optimization step. Downstream code speaks
``x0``: with ``output_kind="x0"`` the network emits it and the framing maps it to the
regression target; with ``output_kind="target"`` the head emits the framing's own target
(eps / v) and ``X0View`` converts it back, so the loss, the sampler and the aux terms
are the same in both cases. The aux terms receive the decoded geometry. AnySchedule
drives the learning rate and an EMA of the backbone is kept for sampling.

The trainer also owns the inference used during training: ``solve_cases`` (sample ->
refine -> legalize -> score) and ``raw_sample_cases`` (sample only), both under the EMA
weights.
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
from trinity.losses.base import LossContext
from trinity.models import DenoiserCond, SetTransformerDenoiser, get_preset
from trinity.models.backbone import X0View
from trinity.models.ema import EMAModule
from trinity.models.presets import DenoiserArchConfig
from trinity.optim import build_param_groups
from trinity.registry import (
    FRAMING,
    GRAPH_PE,
    LATENT_PARAM,
    LOSS,
    REFINER,
    SAMPLER,
    TIME_SAMPLER,
    build,
)
from trinity.solver import DiffusionPlacer

# Batch keys the aux terms read, copied into the loss context sliced to the micro-batch.
_GEOMETRY_KEYS = (
    "cluster_id",
    "boundary_code",
    "mib_id",
    "mob_pos",
    "mob_shape",
    "pin_xy",
    "pin_xy_n",
    "pin_w",
    "pin_edge_xy_n",
    "pin_edge_w",
    "pin_edge_block",
    "area_base",
    "hpwl_base",
    "n_soft",
)


def build_backbone(arch: DenoiserArchConfig, backbone_spec=None):
    """The denoiser of ``arch``: the Trinity set
    transformer, or a ported baseline backbone.

    ``backbone_spec`` is a ``trinity_baselines``
    ``BASELINE_BACKBONE`` spec, or ``None``.
    """
    if backbone_spec is None:
        return SetTransformerDenoiser(arch)
    # Optional dependency: the ported baselines live in their own package.
    import trinity_baselines  # noqa: F401
    from trinity_baselines.registry import BASELINE_BACKBONE
    from trinity_baselines.registry import build as baseline_build

    return baseline_build(
        backbone_spec,
        BASELINE_BACKBONE,
        latent_dim=arch.latent_dim,
        feature_dim=arch.feature_dim,
    )


def build_arch(preset: str | None, arch_overrides: dict | None) -> DenoiserArchConfig:
    """The architecture config: a named preset
    plus overrides, or the overrides alone."""
    if preset is None:
        return DenoiserArchConfig(**(arch_overrides or {}))
    return get_preset(preset, **(arch_overrides or {}))


class DiffusionTrainer(pl.LightningModule):
    """A configurable diffusion model over block-token latents.

    Every component argument is a registry spec (a key, a ``{"name": ..., **kwargs}``
    dict or an instance). ``refiner`` / ``legalize_portfolio`` / ``sample_projections``
    configure only the in-training solve path: ``None`` means no refiner, ``scale_pack``
    alone, and the sampler's default projections respectively. Unknown keyword arguments
    are ignored, so checkpoints written by older versions still load.
    """

    def __init__(
        self,
        *,
        preset: str | None = "ref-d384",
        arch_overrides: dict | None = None,
        backbone: dict | str | None = None,
        framing="ddpm_x0",
        time_sampler="uniform",
        losses: list | None = None,
        latent_param="s_only",
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
        fast_step: bool = False,
        train_cuda_graphs: bool = False,
        sample_steps: int = 30,
        sample_solver: str = "euler",
        sample_projections: list | None = None,
        eval_samples: int = 8,
        eval_max_batch: int = 256,
        refiner: dict | str | None = None,
        legalize_portfolio: list | None = None,
        eval_scorer: str = "full",
        eval_legalize_workers: int = 0,
        graph_pe=None,
        cond_options=None,
        log_interval: int = 1000,
        sample_block_counts: tuple[int, ...] = (21, 60, 120),
        out_dir: str = "outputs/train",
        name: str = "trinity",
        **_unused,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        arch = build_arch(preset, arch_overrides)
        arch.grad_ckpt = grad_checkpointing
        self.arch = arch
        self.backbone = build_backbone(arch, backbone)
        self.param = build(latent_param, LATENT_PARAM)

        self.framing = build(framing, FRAMING)
        self.time_sampler = build(time_sampler, TIME_SAMPLER)
        self.loss_terms = [
            build(spec, LOSS) for spec in (losses or [{"name": "denoise"}])
        ]
        self._needs_geometry = any(
            getattr(term, "needs_geometry", False) for term in self.loss_terms
        )
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
        self.fast_step = fast_step
        self._side_stream = None
        self.train_cuda_graphs = train_cuda_graphs
        self._compile_config = compile_config
        self._cuda_graphs = False
        self._graph_swaps: list = []

        self.ema = EMAModule(self.backbone, ema_decay) if use_ema else None
        self.backbone = apply_compile(self.backbone, compile_config)
        if arch.output_kind == "x0":
            self.x0_model = self.backbone
            self._split_prediction = self._split_x0_head
        else:
            self.x0_model = X0View(self.backbone, self.framing)
            self._split_prediction = self._split_target_head

        self.sample_steps = sample_steps
        self.sample_solver = sample_solver
        self.sample_projections = (
            None if sample_projections is None else list(sample_projections)
        )
        self.eval_samples = eval_samples
        self.eval_max_batch = eval_max_batch
        self.refiner = refiner
        self.legalize_portfolio = legalize_portfolio
        self.eval_scorer = eval_scorer
        self.eval_legalize_workers = eval_legalize_workers
        self.cond_options = (
            cond_options if cond_options is not None else DEFAULT_COND_OPTIONS
        )
        self.graph_pe = build(graph_pe or "none", GRAPH_PE)
        if self.graph_pe.dim != arch.graph_pe_dim:
            raise ValueError(
                f"graph_pe builder width {self.graph_pe.dim} != arch.graph_pe_dim "
                f"{arch.graph_pe_dim}; "
                "set ARCH_OVERRIDES['graph_pe_dim'] to match GRAPH_PE"
            )
        self.log_interval = log_interval
        self.sample_block_counts = tuple(sample_block_counts)
        self.out_dir = out_dir
        self.name = name
        self.last_candidate_costs: list = []
        self._sample_instances: list | None = None
        self._previous_sample_step = -1
        self.ema_loss = -1.0
        self._ema_loss_t: torch.Tensor | None = None

    # ---- runtime hooks -------------------------------------------------------

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Copy the batch; with ``fast_step`` on CUDA, copy
        it and compute its graph PE on a side stream."""
        if not (self.fast_step and device.type == "cuda" and isinstance(batch, dict)):
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
        if self._side_stream is None:
            self._side_stream = torch.cuda.Stream(device)
        main = torch.cuda.current_stream(device)
        with torch.cuda.stream(self._side_stream):
            out = {
                key: (
                    value.to(device, non_blocking=True)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in batch.items()
            }
            out["graph_pe"] = self._batch_graph_pe(out)
        for value in out.values():
            if torch.is_tensor(value):
                value.record_stream(main)
        main.wait_stream(self._side_stream)
        return out

    def on_fit_start(self) -> None:
        """With ``train_cuda_graphs``: recompile the compiled blocks and aux-term
        functions with the inductor ``triton.cudagraphs`` option (needs the ``module``
        compile mode).
        """
        if not self.train_cuda_graphs:
            return
        spec = dict(self._compile_config or {})
        if spec.pop("mode", "module") != "module":
            raise ValueError("train_cuda_graphs needs COMPILE mode 'module'")
        targets = spec.pop("targets", ["blocks"])
        exclude = set(spec.pop("exclude", []))
        spec["options"] = {**(spec.get("options") or {}), "triton.cudagraphs": True}
        for target in targets:
            container = self.backbone.get_submodule(target) if target else self.backbone
            for name, child in container.named_children():
                full_name = f"{target}.{name}" if target else name
                if full_name in exclude:
                    continue
                plain = getattr(child, "_compiled_call_impl", None)
                child.compile(**spec)
                self._graph_swaps.append((child, plain, child._compiled_call_impl))
        for term in self.loss_terms:
            fn = getattr(term, "fn", None)
            if fn is not None:
                original = getattr(fn, "_torchdynamo_orig_callable", fn)
                term.fn = torch.compile(original, **spec)
        self._cuda_graphs = True

    def on_train_start(self) -> None:
        """Check that the loader batch divides evenly
        into ``gradient_accumulation_steps``."""
        loader = self.trainer.train_dataloader
        batch_size = getattr(loader, "batch_size", None) if loader is not None else None
        if (
            batch_size is not None
            and self.grad_accum > 1
            and batch_size % self.grad_accum
        ):
            raise ValueError(
                f"batch size {batch_size} must be divisible by "
                f"gradient_accumulation_steps {self.grad_accum}"
            )

    # ---- training step -------------------------------------------------------

    def _micro_loss(self, batch: dict, lo: int, hi: int):
        """The loss and its per-term logs for the micro-batch ``batch[lo:hi]``."""
        x0 = batch["z0"][lo:hi]
        token_mask = batch["token_mask"][lo:hi]
        graph_pe = batch["graph_pe"]
        # Anchor values reach the network only inside ``features``.
        cond = DenoiserCond(
            features=batch["features"][lo:hi],
            adjacency=batch["adjacency"][lo:hi],
            key_pad_mask=token_mask > 0.5,
            graph_pe=graph_pe[lo:hi] if graph_pe is not None else None,
        )
        t = self.time_sampler(x0.shape[0], x0.device)
        x1 = torch.randn_like(x0)
        x_t, target = self.framing.prepare(x0, x1, t)
        x0_pred, pred = self._split_prediction(x_t, t, cond)

        ctx = LossContext(
            pred=pred,
            target=target,
            weight=self.framing.loss_weight(t),
            t=t,
            anchor_mask=batch["anchor_mask"][lo:hi],
            token_mask=token_mask,
        )
        if self._needs_geometry:
            self._fill_geometry(ctx, batch, x0_pred, lo, hi)

        loss = x0.new_zeros(())
        logs: dict[str, torch.Tensor] = {}
        for term in self.loss_terms:
            value, term_logs = term(ctx)
            loss = loss + value
            logs.update(term_logs)
        return loss, logs

    def _fill_geometry(
        self, ctx: LossContext, batch: dict, x0_pred, lo: int, hi: int
    ) -> None:
        """Add the decoded geometry (normalized units:
        centers ``/s``, areas ``/s^2``) to ``ctx``."""
        area = batch["area_targets"][lo:hi]
        scale = batch["scale"][lo:hi]
        area_norm = area / scale.unsqueeze(-1) ** 2
        ones = torch.ones_like(scale)
        ctx.xywh = z_to_xywh(self.param.from_latent(x0_pred), area_norm, ones)
        ctx.xywh_gt = z_to_xywh(
            self.param.from_latent(batch["z0"][lo:hi]), area_norm, ones
        )
        ctx.area_targets = area_norm
        ctx.scale = ones
        ctx.adjacency = batch["adj_raw"][lo:hi]
        for key in _GEOMETRY_KEYS:
            setattr(ctx, key, batch[key][lo:hi])

    def _split_x0_head(self, x_t, t, cond):
        """``(x0_pred, pred)`` of an x0-emitting
        head (``pred`` mapped to target space)."""
        x0_pred = self.backbone(x_t, t, cond)
        return x0_pred, self.framing.pred_to_target(x_t, x0_pred, t)

    def _split_target_head(self, x_t, t, cond):
        """``(x0_pred, pred)`` of a target-emitting
        head (``x0`` solved from the target)."""
        raw = self.backbone(x_t, t, cond)
        return self.framing.x0_from_target(x_t, raw, t), raw

    def _batch_graph_pe(self, batch: dict) -> torch.Tensor | None:
        """The graph PE of the whole batch from the
        raw adjacency (``None`` without a PE)."""
        if self.graph_pe.dim == 0:
            return None
        return self.graph_pe.batched(batch["adj_raw"], batch["token_mask"] > 0.5)

    @staticmethod
    def _move_grads(params, acc):
        """Add each parameter's ``.grad`` into ``acc``,
        clear ``.grad``, and return ``acc``."""
        if acc is None:
            acc = [None] * len(params)
        both = [
            i for i, p in enumerate(params) if p.grad is not None and acc[i] is not None
        ]
        if both:
            torch._foreach_add_([acc[i] for i in both], [params[i].grad for i in both])
        for i, p in enumerate(params):
            if p.grad is not None and acc[i] is None:
                acc[i] = p.grad.clone()
            p.grad = None
        return acc

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sched = self.lr_schedulers()
        opt.zero_grad(set_to_none=True)

        if "graph_pe" not in batch:
            batch["graph_pe"] = self._batch_graph_pe(batch)
        steps = self.grad_accum
        chunk = batch["z0"].shape[0] // steps
        total = batch["z0"].new_zeros(())
        total_logs: dict[str, torch.Tensor] = {}
        all_params = [p for group in opt.param_groups for p in group["params"]]
        acc = None
        for i in range(steps):
            if self._cuda_graphs:
                torch.compiler.cudagraph_mark_step_begin()
            loss, logs = self._micro_loss(batch, i * chunk, (i + 1) * chunk)
            self.manual_backward(loss / steps)
            total = total + loss.detach() / steps
            for key, value in logs.items():
                total_logs[key] = total_logs.get(key, 0.0) + value / steps
            if self._cuda_graphs:
                acc = self._move_grads(all_params, acc)
        if acc is not None:
            for p, grad in zip(all_params, acc, strict=True):
                p.grad = grad

        if self.fast_step:
            if getattr(self.trainer.precision_plugin, "scaler", None) is not None:
                raise ValueError("fast_step does not support fp16 with a grad scaler")
            torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip)
        else:
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
        """Log every loss term and a warmed-up EMA of the total loss."""
        decay = min(0.99, self.global_step / (self.global_step + 10))
        if self.fast_step:
            value = total.detach().float()
            if self._ema_loss_t is None:
                self._ema_loss_t = value
            else:
                self._ema_loss_t = self._ema_loss_t * decay + value * (1 - decay)
            self.log("train/loss_ema", self._ema_loss_t, prog_bar=True, sync_dist=False)
        else:
            value = total.item()
            if self.ema_loss < 0:
                self.ema_loss = value
            else:
                self.ema_loss = self.ema_loss * decay + value * (1 - decay)
            self.log("train/loss_ema", self.ema_loss, prog_bar=True, sync_dist=False)
        self.log("train/loss", total, sync_dist=True)
        for key, value in logs.items():
            self.log(f"train/{key}", value, prog_bar=(key == "denoise"), sync_dist=True)

    def _sample_cases(self) -> list:
        """The validation cases rendered during training (loaded once)."""
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
            fused=True if self.fast_step else None,
        )
        sched = AnySchedule(opt, config=self._resolved_scheduler_config())
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }

    # ---- inference (sample logging + validation) -----------------------------

    @contextlib.contextmanager
    def _eval_mode(self):
        """Backbone in eval mode under the EMA weights,
        without CUDA graphs, inside the context."""
        was_training = self.backbone.training
        ema_ctx = (
            self.ema.use_ema(self.backbone)
            if self.ema is not None
            else contextlib.nullcontext()
        )
        self.backbone.eval()
        for child, plain, _ in self._graph_swaps:
            child._compiled_call_impl = plain
        try:
            with ema_ctx:
                yield
        finally:
            for child, _, graphed in self._graph_swaps:
                child._compiled_call_impl = graphed
            self.backbone.train(was_training)

    def make_placer(self) -> DiffusionPlacer:
        """A ``DiffusionPlacer`` over the live backbone
        with the eval sampler / refiner config."""
        sampler = build(
            {
                "name": self.sample_solver,
                "num_steps": self.sample_steps,
                "projections": self.sample_projections,
            },
            SAMPLER,
            framing=self.framing,
        )
        return DiffusionPlacer(
            self.x0_model,
            sampler,
            build(self.refiner, REFINER),
            param=self.param,
            samples=self.eval_samples,
            scorer=self.eval_scorer,
            legalize_portfolio=self.legalize_portfolio,
            device=self.device.type,
            max_batch=self.eval_max_batch,
            legalize_workers=self.eval_legalize_workers,
            graph_pe=self.graph_pe,
            cond_options=self.cond_options,
        )

    def _ordered_chunks(self, instances):
        """``(order, chunks)``: instances sorted by
        block count, cut into placer-sized chunks."""
        order = sorted(range(len(instances)), key=lambda i: instances[i].block_count)
        per_chunk = max(1, self.eval_max_batch // self.eval_samples)
        ordered = [instances[i] for i in order]
        chunks = [ordered[i : i + per_chunk] for i in range(0, len(ordered), per_chunk)]
        return order, chunks

    @torch.no_grad()
    def solve_cases(self, instances, desc: str = "eval: solving"):
        """Place every instance; return ``[(instance,
        score, placement)]`` in input order.

        Sets ``last_candidate_costs``: per case, every legalized candidate's cost.
        """
        order, chunks = self._ordered_chunks(instances)
        show_bar = self._trainer is None or self.trainer.is_global_zero
        placements, candidate_costs = [], []
        with self._eval_mode():
            placer = self.make_placer()
            with tqdm(
                total=len(instances), desc=desc, unit="case", disable=not show_bar
            ) as bar:
                for group in chunks:
                    placements.extend(placer.solve_batch(group))
                    candidate_costs.extend(placer.last_candidate_costs)
                    bar.update(len(group))
        out: list = [None] * len(instances)
        self.last_candidate_costs = [None] * len(instances)
        for pos, placement, costs in zip(
            order, placements, candidate_costs, strict=True
        ):
            out[pos] = (
                instances[pos],
                validate(placement, self.eval_scorer).score,
                placement,
            )
            self.last_candidate_costs[pos] = costs
        return out

    @torch.no_grad()
    def raw_sample_cases(self, instances) -> list[Placement]:
        """The first raw sample (no refiner, no
        legalizer) of every instance, in input order."""
        order, chunks = self._ordered_chunks(instances)
        out: list = [None] * len(instances)
        position = 0
        with self._eval_mode():
            placer = self.make_placer()
            for group in chunks:
                for boxes in placer.raw_sample_boxes(group):
                    pos = order[position]
                    out[pos] = Placement(
                        xywh=boxes.astype(np.float64), instance=instances[pos]
                    )
                    position += 1
        return out

    @torch.no_grad()
    def log_samples(self, instances, out_dir: str) -> list[str]:
        """Render solved placements of ``instances``
        to PNG (and wandb); return the paths."""
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
