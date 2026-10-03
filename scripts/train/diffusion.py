"""Train a Trinity diffusion model on FloorSet.

A kohaku-engine script: every typed UPPER_CASE global below is
a knob that a config file (or a typed ``--set``) overrides::

    kogine run scripts/train/diffusion.py --config configs/train/flagship.py
    kogine run scripts/train/diffusion.py \\
        --config configs/train/smoke.py \\
        --set MAX_STEPS=20

The defaults form a small runnable recipe (the 100-case validation set as data). Every
component spec (framing, losses, graph PE, refiner, ...) is resolved by the trainer.
"""

import os
import sys
import warnings

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as ddp_hooks

from trinity.conditioning import CondOptions
from trinity.data import build_loader, train_dataset, validation_dataset
from trinity.floorplan.data import (
    LanceFloorplanStore,
    find_train_lance,
    load_validation_set,
)
from trinity.training import DiffusionTrainer, ValidationCallback
from trinity.training.trainer import build_arch
from trinity.utils import autofill_schedule_steps

if os.environ.get("LOCAL_RANK", "0") != "0":
    warnings.filterwarnings("ignore", ".*")

torch.set_float32_matmul_precision("high")

# ---- Resume ----------------------------------------------------------------
# Checkpoint to initialize the weights from.
CHECKPOINT_PATH: str | None = None
# Also resume the optimizer, scheduler and step counter from CHECKPOINT_PATH.
TRAINER_RESUME: bool = False

# ---- Data ------------------------------------------------------------------
# "train" (the Lance train set) | "validation" (the 100-case set, for smoke runs).
DATA: str = "train"
# Directory holding LiteTensorDataTest/ (None: the project data/ directory).
FLOORSET_ROOT: str | None = None
# The Lance train set (None: data/floorset_lite_mibfix.lance).
TRAIN_LANCE: str | None = None
# Dev split: DEV_PER_N_K layouts per block count + DEV_RANDOM_SIZE random layouts.
DEV_PER_N_K: int = 10
DEV_RANDOM_SIZE: int = 7000
SPLIT_SEED: int = 20090220
ALLOW_DOWNLOAD: bool = True

# ---- Model -----------------------------------------------------------------
# A named preset plus per-field overrides (None: ARCH_OVERRIDES is the whole arch).
PRESET: str | None = "ref-d384"
ARCH_OVERRIDES: dict = {}
# A ported baseline denoiser (trinity_baselines
# BASELINE_BACKBONE spec); None: Trinity's.
BACKBONE: dict | str | None = None

# ---- Objective -------------------------------------------------------------
# ddpm_x0 | ddpm_eps | ddpm_v | rectified_flow | gvp
FRAMING_SPEC: dict | str = "ddpm_x0"
# {"name": "uniform"} | {"name": "logit_normal", "mean": 0.0, "std": 1.0}
TIME_SAMPLER: dict | str = {"name": "uniform"}
# The denoise term plus any aux terms: overlap / group / mib / boundary / wl / area.
LOSSES: list = [{"name": "denoise"}]
LATENT_PARAM: dict | str = "s_only"
# Graph PE spec; its width must equal ARCH_OVERRIDES["graph_pe_dim"].
GRAPH_PE: dict | str | None = None
# Per-instance max-normalization of the pin-pull
# feature and of the graph-bias adjacency.
PIN_PULL_NORMALIZE: bool = False
GRAPH_BIAS_NORMALIZE: bool = False
# Anchor of all-soft MIB groups: "group_mean" | "square" | "none".
MIB_ANCHOR: str = "group_mean"
# Train-time augmentation op list (None: off),
# e.g. [{"rot90": {...}}, {"flip": {"p": 0.5}}].
AUGMENT: list | None = None

# ---- Compute ---------------------------------------------------------------
# Device ids of the GPUs to train on.
GPUS: list[int] = [0]
# Per-GPU loader batch; GRAD_ACC splits it into micro-batches inside one optimizer step.
BATCH_SIZE: int = 8
GRAD_ACC: int = 1
# "same_n" (equal block count per batch) | "pad_to_max" (pad every case to MAX_N).
BATCHING: str = "same_n"
MAX_N: int = 128
# Training length: set one of the two, the other to -1.
MAX_STEPS: int = 2000
EPOCH: int = -1
PRECISION: str = "bf16-mixed"
# Gradient compression of multi-GPU runs: "fp16" | "bf16" | None.
DDP_COMPRESS_HOOK: str | None = "bf16"
NUM_WORKERS: int = 0
SEED: int = 20090220
# Runtime speed options (docs/training.md): pinned loader memory, the sync-free
# step, and CUDA graphs on the compiled blocks (needs COMPILE={"mode": "module"}).
PIN_MEMORY: bool = False
FAST_STEP: bool = False
TRAIN_CUDA_GRAPHS: bool = False
LOG_EVERY_N_STEPS: int = 1
PROGRESS_REFRESH_RATE: int = 1

# ---- Optimizer and schedule ------------------------------------------------
LR: float = 3e-4
BETAS: tuple[float, float] = (0.9, 0.95)
WEIGHT_DECAY: float = 0.0
# "adamw" | "mup" (BASE_DIM is the muP base width).
OPTIMIZER_MODE: str = "adamw"
BASE_DIM: int = 256
# AnySchedule config; "end": -1 is filled with the total step count.
SCHEDULER_CONFIG: dict = {"lr": {"mode": "cosine", "min_value": 0.05, "end": -1}}
SCHED_WARMUP_RATIO: float = 0.0
GRAD_CLIP: float = 1.0

# ---- EMA and compile -------------------------------------------------------
USE_EMA: bool = True
EMA_DECAY: float = 0.999
# None | {"mode": "module"} | {"mode": "model", "dynamic": True}
COMPILE: dict | None = None
GRAD_CKPT: bool = False

# ---- Inference during training (sample renders + validation) ---------------
SAMPLE_STEPS: int = 16
SAMPLE_SOLVER: str = "euler"
# Sampler projections; None: the default (known
# answers + MIB shape agreement), []: none.
SAMPLE_PROJECTIONS: list | None = None
# Candidates sampled per case in the full solve
# path (the cheapest legalized one is kept).
EVAL_BEST_OF_N: int = 16
# Rows (cases x candidates) per GPU forward.
EVAL_MAX_BATCH: int = 2000
# REFINER spec of the full solve path (e.g. "closed"); None: no refiner.
REFINER: dict | str | None = None
# Legalization portfolio (a list of routes); None: scale_pack.
LEGALIZE_PORTFOLIO: list | None = None
EVAL_SCORER: str = "full"
# Processes legalizing candidates in parallel (0: inline).
EVAL_LEGALIZE_WORKERS: int = 0
# Block counts of the validation cases rendered every LOG_INTERVAL steps.
SAMPLE_BLOCK_COUNTS: tuple[int, ...] = (21, 60, 120)

# ---- In-training validation (ValidationCallback) ----------------------------
EVAL_ENABLE: bool = True
# "soft": soft metrics of the raw sample on the dev split; "full": solve and score.
EVAL_MODE: str = "soft"
# Truncate each rank's dev shard in the soft mode (0: all).
EVAL_MAX_DEV_CASES: int = 0
EVAL_EVERY_N_STEPS: int = 1000
EVAL_RUN_AT_START: bool = False
EVAL_NUM_RENDER: int = 6
# Also solve the official 100-case set in the full mode.
EVAL_OFFICIAL: bool = True

# ---- Logging and checkpoints ----------------------------------------------
WANDB_PROJECT: str = "Trinity"
WANDB_OFFLINE: bool = True
# Run name (None: generated from the arch, size and objective).
NAME: str | None = None
# Sample-render cadence in steps (0: off).
LOG_INTERVAL: int = 1000
CKPT_INTERVAL: int = 1000
# Periodic checkpoints kept (-1: all, 0: none); last.ckpt is always kept.
CKPT_SAVE_TOP_K: int = -1
# Per-epoch checkpoint cadence (0: off).
CKPT_EVERY_N_EPOCHS: int = 1
OUT_DIR: str = "outputs/train"


_NAME_TAGS = {
    "layernorm": "ln",
    "rmsnorm": "rms",
    "swiglu": "swi",
    "adaln_shared": "shadaln",
    "additive": "add",
    "token": "tok",
    "ddpm_x0": "x0",
    "ddpm_cosine_x0": "cosx0",
    "ddpm_eps": "eps",
    "ddpm_v": "v",
    "rectified_flow": "rf",
    "spectral_draw": "specdraw",
}


def _spec_name(spec) -> str:
    """The registry key of a spec (a key string or a ``{"name": ...}`` dict)."""
    if isinstance(spec, dict):
        return str(spec.get("name", "?"))
    return str(spec)


def _tag(spec) -> str:
    name = _spec_name(spec)
    return _NAME_TAGS.get(name, name)


def auto_name() -> str:
    """A run name from the arch, size and objective, e.g.
    ``trinity-rms-swi-tok-qk-d768L12-rf-logitn-bs256``."""
    arch = build_arch(PRESET, ARCH_OVERRIDES)
    parts = [
        "trinity",
        _tag(arch.norm),
        _tag(arch.mlp),
        _tag(arch.time_cond),
        "qk" if arch.qk_norm else None,
        "pn" if arch.post_norm else None,
        "nobias" if not arch.graph_bias else None,
        _tag(GRAPH_PE) if GRAPH_PE is not None else None,
        "aug" if AUGMENT else None,
        f"d{arch.hidden}L{arch.depth}",
        _tag(FRAMING_SPEC),
        "logitn" if _spec_name(TIME_SAMPLER) == "logit_normal" else "uni",
        f"bs{BATCH_SIZE * len(GPUS)}",
        "mup" if OPTIMIZER_MODE == "mup" else None,
    ]
    return "-".join(part for part in parts if part)


def _config_path() -> str | None:
    """The ``--config`` / ``-c`` path of the
    launching ``kogine run`` command, if any."""
    for flag in ("--config", "-c"):
        if flag in sys.argv:
            index = sys.argv.index(flag)
            if index + 1 < len(sys.argv):
                return sys.argv[index + 1]
    return None


def _log_config_to_wandb(logger) -> None:
    """Save the config file and the resolved knobs
    to the wandb run (failures only warn)."""
    if not hasattr(logger, "experiment"):
        return
    try:
        path = _config_path()
        if path and os.path.isfile(path):
            logger.experiment.save(path, policy="now")
        resolved = {
            k: v for k, v in globals().items() if k.isupper() and not k.startswith("_")
        }
        resolved["CONFIG_FILE"] = path
        logger.experiment.config.update(resolved, allow_val_change=True)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"could not log config to wandb: {exc}", stacklevel=2)


def _cond_options() -> CondOptions:
    return CondOptions(
        pin_pull_normalize=PIN_PULL_NORMALIZE,
        graph_bias_normalize=GRAPH_BIAS_NORMALIZE,
        mib_anchor=MIB_ANCHOR,
    )


def build_data():
    """``(train_dataset, dev_shard_fn)``; ``dev_shard_fn(rank,
    world)`` is that rank's dev slice."""
    if DATA == "train":
        dataset, splits = train_dataset(
            lance_path=TRAIN_LANCE,
            per_n_k=DEV_PER_N_K,
            random_size=DEV_RANDOM_SIZE,
            seed=SPLIT_SEED,
            latent_param=LATENT_PARAM,
            augment=AUGMENT,
            cond_options=_cond_options(),
        )
        store = LanceFloorplanStore(str(find_train_lance(TRAIN_LANCE)))
        dev_ids = splits.dev_per_n + splits.dev_random
        return dataset, lambda rank, world: store.instances(dev_ids[rank::world])
    if DATA == "validation":
        dataset = validation_dataset(
            root=FLOORSET_ROOT,
            allow_download=ALLOW_DOWNLOAD,
            latent_param=LATENT_PARAM,
            cond_options=_cond_options(),
        )
        cases = load_validation_set(root=FLOORSET_ROOT, allow_download=ALLOW_DOWNLOAD)
        return dataset, lambda rank, world: cases[rank::world]
    raise ValueError(f"unknown DATA source {DATA!r}")


def build_model() -> DiffusionTrainer:
    """The ``DiffusionTrainer`` of the config
    (weights from ``CHECKPOINT_PATH`` if set)."""
    trainer_kwargs = dict(
        preset=PRESET,
        arch_overrides=ARCH_OVERRIDES,
        backbone=BACKBONE,
        framing=FRAMING_SPEC,
        time_sampler=TIME_SAMPLER,
        losses=LOSSES,
        latent_param=LATENT_PARAM,
        learning_rate=LR,
        betas=tuple(BETAS),
        weight_decay=WEIGHT_DECAY,
        optimizer_mode=OPTIMIZER_MODE,
        base_dim=BASE_DIM,
        scheduler_config=SCHEDULER_CONFIG,
        gradient_accumulation_steps=GRAD_ACC,
        gradient_clip=GRAD_CLIP,
        grad_checkpointing=GRAD_CKPT,
        use_ema=USE_EMA,
        ema_decay=EMA_DECAY,
        compile_config=COMPILE,
        fast_step=FAST_STEP,
        train_cuda_graphs=TRAIN_CUDA_GRAPHS,
        sample_steps=SAMPLE_STEPS,
        sample_solver=SAMPLE_SOLVER,
        sample_projections=SAMPLE_PROJECTIONS,
        eval_samples=EVAL_BEST_OF_N,
        eval_max_batch=EVAL_MAX_BATCH,
        refiner=REFINER,
        legalize_portfolio=LEGALIZE_PORTFOLIO,
        eval_scorer=EVAL_SCORER,
        eval_legalize_workers=EVAL_LEGALIZE_WORKERS,
        graph_pe=GRAPH_PE,
        cond_options=_cond_options(),
        log_interval=LOG_INTERVAL,
        sample_block_counts=SAMPLE_BLOCK_COUNTS,
        out_dir=OUT_DIR,
        name=NAME,
    )
    if CHECKPOINT_PATH is not None and not TRAINER_RESUME:
        return DiffusionTrainer.load_from_checkpoint(
            CHECKPOINT_PATH,
            map_location="cpu",
            strict=False,
            weights_only=False,
            **trainer_kwargs,
        )
    return DiffusionTrainer(**trainer_kwargs)


def build_callbacks(dev_shard_fn) -> list:
    """Learning-rate monitor, progress bar, checkpoints, and (if enabled) validation."""
    checkpoint_dir = f"{OUT_DIR}/{NAME}/checkpoints"
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=PROGRESS_REFRESH_RATE),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            every_n_train_steps=CKPT_INTERVAL,
            save_last=True,
            save_top_k=CKPT_SAVE_TOP_K,
        ),
    ]
    if CKPT_EVERY_N_EPOCHS > 0:
        callbacks.append(
            ModelCheckpoint(dirpath=checkpoint_dir, every_n_epochs=CKPT_EVERY_N_EPOCHS)
        )
    if EVAL_ENABLE:
        official_shard_fn = None
        if EVAL_OFFICIAL and DATA == "train":
            official = load_validation_set(
                root=FLOORSET_ROOT, allow_download=ALLOW_DOWNLOAD
            )

            def official_shard_fn(rank, world):
                return official[rank::world]

        callbacks.append(
            ValidationCallback(
                shard_fn=dev_shard_fn,
                official_shard_fn=official_shard_fn,
                every_n_steps=EVAL_EVERY_N_STEPS,
                run_at_start=EVAL_RUN_AT_START,
                num_render=EVAL_NUM_RENDER,
                out_dir=OUT_DIR,
                mode=EVAL_MODE,
                max_dev_cases=EVAL_MAX_DEV_CASES,
            )
        )
    return callbacks


def build_strategy(gpu_count: int):
    """DDP with the configured gradient compression on several GPUs, else ``"auto"``."""
    if gpu_count <= 1:
        return "auto"
    hooks = {"fp16": ddp_hooks.fp16_compress_hook, "bf16": ddp_hooks.bf16_compress_hook}
    return DDPStrategy(
        gradient_as_bucket_view=True, ddp_comm_hook=hooks.get(DDP_COMPRESS_HOOK)
    )


def main() -> None:
    global NAME
    NAME = NAME or auto_name()
    pl.seed_everything(SEED)
    gpu_count = len(GPUS)

    dataset, dev_shard_fn = build_data()
    loader, use_distributed_sampler = build_loader(
        dataset,
        batching=BATCHING,
        batch_size=BATCH_SIZE,
        max_n=MAX_N,
        shuffle=True,
        seed=SEED,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    if MAX_STEPS > 0:
        train_steps = MAX_STEPS
    else:
        train_steps = (1 + len(loader) // gpu_count) * EPOCH + 1
    autofill_schedule_steps(
        SCHEDULER_CONFIG["lr"], train_steps, warmup_ratio=SCHED_WARMUP_RATIO
    )

    model = build_model()
    os.makedirs(OUT_DIR, exist_ok=True)
    logger = WandbLogger(project=WANDB_PROJECT, name=NAME, offline=WANDB_OFFLINE)
    _log_config_to_wandb(logger)
    trainer = pl.Trainer(
        max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
        max_epochs=EPOCH if EPOCH > 0 else None,
        accelerator="auto",
        devices=GPUS,
        strategy=build_strategy(gpu_count),
        precision=PRECISION,
        logger=logger,
        callbacks=build_callbacks(dev_shard_fn),
        use_distributed_sampler=use_distributed_sampler,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        num_sanity_val_steps=0,
    )
    resume_ckpt = CHECKPOINT_PATH if TRAINER_RESUME else None
    trainer.fit(model, loader, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
