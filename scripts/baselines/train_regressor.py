"""Train a direct-regression baseline on FloorSet.

The deterministic counterpart of ``scripts/train/diffusion.py``: the same data, backbone,
conditioning, refiner, legalizer and validation, with the diffusion objective replaced by one
supervised regression of the ground-truth layout. Run with KohakuEngine::

    kogine run scripts/baselines/train_regressor.py --config configs/baselines/train_reg_z.py
    kogine run scripts/baselines/train_regressor.py --set MAX_STEPS=500

Every typed UPPER_CASE global below is a knob a config (or a typed ``--set``) may override.
The defaults are a small debug recipe (the 100-case validation set, ``ref-d384``, ``head="z"``).
"""

import os
import sys
import warnings

if os.environ.get("LOCAL_RANK", "0") != "0":
    warnings.filterwarnings("ignore", ".*")

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
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
from trinity.models import get_preset
from trinity.utils import autofill_schedule_steps
from trinity_baselines.training import RegressorTrainer, ValidationCallback

torch.set_float32_matmul_precision("high")

# ---- Resume ----------------------------------------------------------------
CHECKPOINT_PATH: str | None = None  # .ckpt to load the weights from
TRAINER_RESUME: bool = False  # also resume optimizer, scheduler and global step

# ---- Data ------------------------------------------------------------------
DATA: str = "validation"  # "train" (1M Lance set) | "validation" (100 cases, debug)
FLOORSET_ROOT: str | None = None  # None = the project data directory
TRAIN_LANCE: str | None = None  # None = the project default path
DEV_PER_N_K: int = 10  # dev cases held out per block count
DEV_RANDOM_SIZE: int = 7000  # dev cases held out at random
SPLIT_SEED: int = 20090220
ALLOW_DOWNLOAD: bool = True

# ---- Model -----------------------------------------------------------------
PRESET: str | None = "ref-d384"  # None = a full DenoiserArchConfig from ARCH_OVERRIDES
ARCH_OVERRIDES: dict = {}
HEAD: str = "z"  # "z" (latent cx/s, cy/s, rho) | "xywh" (raw box)

# ---- Objective -------------------------------------------------------------
LOSSES: list = [{"name": "denoise", "reduction": "global"}]
LATENT_PARAM: str = "s_only"
GRAPH_PE: str | dict | None = (
    None  # its width must equal ARCH_OVERRIDES["graph_pe_dim"]
)
PIN_PULL_NORMALIZE: bool = False
GRAPH_BIAS_NORMALIZE: bool = False
MIB_ANCHOR: str = "group_mean"  # "group_mean" | "square" | "none"
AUGMENT: list | bool | None = (
    None  # an ordered augment-op list (see trinity.augment_ops)
)

# ---- Compute ---------------------------------------------------------------
GPUS: list | int = [0]  # device ids, or a device count
BATCH_SIZE: int = 8  # per-GPU batch
BATCHING: str = "same_n"  # "same_n" | "pad_to_max"
MAX_N: int = 128  # pad length of "pad_to_max"
GRAD_ACC: int = 1  # micro-batches per optimizer step (splits BATCH_SIZE)
MAX_STEPS: int = 2000  # -1 = train for EPOCH epochs
EPOCH: int = -1
PRECISION: str = "bf16-mixed"
DDP_COMPRESS_HOOK: str | None = "bf16"  # "fp16" | "bf16" | None
NUM_WORKERS: int = 0
SEED: int = 20090220

# ---- Optimizer and schedule ------------------------------------------------
LR: float = 3e-4
BETAS: tuple = (0.9, 0.95)
WEIGHT_DECAY: float = 0.0
OPTIMIZER_MODE: str = "adamw"  # "adamw" | "mup"
BASE_DIM: int = 256  # muP base width
SCHEDULER_CONFIG: dict = {"lr": {"mode": "cosine", "min_value": 0.05, "end": -1}}
SCHED_WARMUP_RATIO: float = 0.0
GRAD_CLIP: float = 1.0

# ---- EMA and runtime -------------------------------------------------------
USE_EMA: bool = True
EMA_DECAY: float = 0.999
COMPILE: dict | None = None  # None | {"mode": "module"} | {"mode": "model"}
GRAD_CKPT: bool = False

# ---- Inference (one candidate per case) ------------------------------------
EVAL_MAX_BATCH: int = 2000  # cases per forward
SAMPLE_PROJECTIONS: list = []  # PROJECTION specs applied to the prediction
REFINER: str | dict | None = None  # a REFINER spec, e.g. "closed"; None = no refiner
LEGALIZE_PORTFOLIO: list | None = None  # legalization routes; None = scale_pack
EVAL_SCORER: str = "full"  # "full" | "full_fast"
EVAL_LEGALIZE_WORKERS: int = 0  # >0 legalizes in a process pool
SAMPLE_BLOCK_COUNTS: tuple = (21, 60, 120)  # validation cases rendered in the loop

# ---- In-training validation ------------------------------------------------
EVAL_ENABLE: bool = True
EVAL_MODE: str = (
    "soft"  # "soft" (raw-layout metrics) | "full" (refine + legalize + score)
)
EVAL_MAX_DEV_CASES: int = 0  # >0 truncates the dev split
EVAL_EVERY_N_STEPS: int = 1000
EVAL_RUN_AT_START: bool = False
EVAL_NUM_RENDER: int = 6
EVAL_OFFICIAL: bool = True  # also solve the 100 validation cases (DATA="train" only)

# ---- Logging and checkpoints -----------------------------------------------
WANDB_PROJECT: str = "Trinity"
WANDB_OFFLINE: bool = True
NAME: str | None = None  # None = a name built from the head, arch and batch
LOG_INTERVAL: int = 1000  # in-loop sample render cadence (0 = off)
CKPT_INTERVAL: int = 1000
CKPT_SAVE_TOP_K: int = (
    -1
)  # periodic checkpoints kept (-1 all); last.ckpt is always kept
CKPT_EVERY_N_EPOCHS: int = 1  # 0 disables the per-epoch checkpoint
OUT_DIR: str = "outputs/train"


NORM_TAGS = {"layernorm": "ln", "rmsnorm": "rms"}
MLP_TAGS = {"gelu": "gelu", "swiglu": "swi"}
TIME_COND_TAGS = {
    "adaln": "adaln",
    "adaln_shared": "shadaln",
    "additive": "add",
    "token": "tok",
}


def spec_name(spec) -> str:
    """The registry key of a spec (a string or a ``{"name": ...}`` dict)."""
    if isinstance(spec, dict):
        return str(spec.get("name", "?"))
    return str(spec)


def gpu_count() -> int:
    return len(GPUS) if isinstance(GPUS, list) else int(GPUS)


def auto_name() -> str:
    """A run name from the head, arch and batch, e.g. ``reg-z-ln-gelu-adaln-d384L8-bs8``."""
    arch = ARCH_OVERRIDES
    hidden, depth = arch.get("hidden"), arch.get("depth")
    if PRESET is not None and (hidden is None or depth is None):
        preset = get_preset(PRESET, **arch)
        hidden, depth = preset.hidden, preset.depth
    norm = arch.get("norm", "layernorm")
    mlp = arch.get("mlp", "gelu")
    time_cond = spec_name(arch.get("time_cond", "adaln"))
    parts = [
        "reg",
        HEAD,
        NORM_TAGS.get(norm, norm),
        MLP_TAGS.get(mlp, mlp),
        TIME_COND_TAGS.get(time_cond, time_cond),
        "qk" if arch.get("qk_norm") else None,
        "nobias" if arch.get("graph_bias") is False else None,
        "aug" if AUGMENT else None,
        f"d{hidden}L{depth}",
        f"bs{BATCH_SIZE * gpu_count() * GRAD_ACC}",
        "mup" if OPTIMIZER_MODE == "mup" else None,
    ]
    return "-".join(part for part in parts if part)


def config_path() -> str | None:
    """The ``--config`` path of the ``kogine run`` command line, if any."""
    for flag in ("--config", "-c"):
        if flag in sys.argv:
            i = sys.argv.index(flag)
            if i + 1 < len(sys.argv):
                return sys.argv[i + 1]
    return None


def log_config_to_wandb(logger) -> None:
    """Save the config file to the run and record every knob in ``wandb.config``."""
    if not hasattr(logger, "experiment"):
        return
    try:
        path = config_path()
        if path and os.path.isfile(path):
            logger.experiment.save(path, policy="now")
        knobs = {k: v for k, v in globals().items() if k.isupper()}
        knobs["CONFIG_FILE"] = path
        logger.experiment.config.update(knobs, allow_val_change=True)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"could not log the config to wandb: {exc}", stacklevel=2)


def cond_options() -> CondOptions:
    return CondOptions(
        pin_pull_normalize=PIN_PULL_NORMALIZE,
        graph_bias_normalize=GRAPH_BIAS_NORMALIZE,
        mib_anchor=MIB_ANCHOR,
    )


def build_data():
    """``(train dataset, dev_shard_fn)``; ``dev_shard_fn(rank, world)`` is one rank's dev cases."""
    if DATA == "train":
        dataset, splits = train_dataset(
            lance_path=TRAIN_LANCE,
            per_n_k=DEV_PER_N_K,
            random_size=DEV_RANDOM_SIZE,
            seed=SPLIT_SEED,
            latent_param=LATENT_PARAM,
            augment=AUGMENT,
            cond_options=cond_options(),
        )
        store = LanceFloorplanStore(str(find_train_lance(TRAIN_LANCE)))
        dev_ids = splits.dev_per_n + splits.dev_random
        return dataset, lambda rank, world: store.instances(dev_ids[rank::world])
    if DATA == "validation":
        dataset = validation_dataset(
            root=FLOORSET_ROOT,
            allow_download=ALLOW_DOWNLOAD,
            latent_param=LATENT_PARAM,
            cond_options=cond_options(),
        )
        cases = load_validation_set(root=FLOORSET_ROOT, allow_download=ALLOW_DOWNLOAD)
        return dataset, lambda rank, world: cases[rank::world]
    raise ValueError(f"unknown DATA source {DATA!r}")


def build_callbacks(name: str, dev_shard_fn) -> list:
    """Learning-rate monitor, checkpoints and (if enabled) the validation callback."""
    checkpoint_dir = f"{OUT_DIR}/{name}/checkpoints"
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
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


def build_strategy():
    """DDP with gradient compression on several GPUs, Lightning's default on one."""
    if gpu_count() <= 1:
        return "auto"
    hooks = {
        "fp16": ddp_hooks.fp16_compress_hook,
        "bf16": ddp_hooks.bf16_compress_hook,
        None: None,
    }
    return DDPStrategy(
        gradient_as_bucket_view=True, ddp_comm_hook=hooks[DDP_COMPRESS_HOOK]
    )


def main():
    name = NAME or auto_name()
    pl.seed_everything(SEED)

    dataset, dev_shard_fn = build_data()
    loader, use_distributed_sampler = build_loader(
        dataset,
        batching=BATCHING,
        batch_size=BATCH_SIZE,
        max_n=MAX_N,
        shuffle=True,
        seed=SEED,
        num_workers=NUM_WORKERS,
    )

    if MAX_STEPS > 0:
        train_steps = MAX_STEPS
    else:
        train_steps = (1 + len(loader) // gpu_count()) * EPOCH + 1
    autofill_schedule_steps(
        SCHEDULER_CONFIG["lr"], train_steps, warmup_ratio=SCHED_WARMUP_RATIO
    )

    trainer_kwargs = dict(
        preset=PRESET,
        arch_overrides=ARCH_OVERRIDES,
        head=HEAD,
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
        eval_max_batch=EVAL_MAX_BATCH,
        sample_projections=SAMPLE_PROJECTIONS,
        refiner=REFINER,
        legalize_portfolio=LEGALIZE_PORTFOLIO,
        eval_scorer=EVAL_SCORER,
        eval_legalize_workers=EVAL_LEGALIZE_WORKERS,
        graph_pe=GRAPH_PE,
        cond_options=cond_options(),
        log_interval=LOG_INTERVAL,
        sample_block_counts=SAMPLE_BLOCK_COUNTS,
        out_dir=OUT_DIR,
        name=name,
    )
    if CHECKPOINT_PATH is not None and not TRAINER_RESUME:
        model = RegressorTrainer.load_from_checkpoint(
            CHECKPOINT_PATH,
            map_location="cpu",
            strict=False,
            weights_only=False,
            **trainer_kwargs,
        )
    else:
        model = RegressorTrainer(**trainer_kwargs)

    os.makedirs(OUT_DIR, exist_ok=True)
    logger = WandbLogger(project=WANDB_PROJECT, name=name, offline=WANDB_OFFLINE)
    log_config_to_wandb(logger)
    trainer = pl.Trainer(
        max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
        max_epochs=EPOCH if EPOCH > 0 else None,
        accelerator="auto",
        devices=GPUS,
        strategy=build_strategy(),
        precision=PRECISION,
        logger=logger,
        callbacks=build_callbacks(name, dev_shard_fn),
        use_distributed_sampler=use_distributed_sampler,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    resume_ckpt = CHECKPOINT_PATH if TRAINER_RESUME else None
    trainer.fit(model, loader, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
