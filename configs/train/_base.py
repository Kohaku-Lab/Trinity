"""The base training recipe of the paper; every training config builds on it.

A plain d768 / L12 set transformer with the netlist as an additive attention bias and no
graph PE, trained with rectified flow (x0 head against the v target, logit-normal t) and
no aux terms. Frozen for every paper run: effective batch 256 (one card, micro-batch
128), 200k steps. Topic configs override one axis on top, reading a dict back through
``use_config``::

    _base = use_config("_base.py")
    ARCH_OVERRIDES = {**_base.globals_dict["ARCH_OVERRIDES"], "graph_pe_dim": 2}

Run one config per card (select the card with ``CUDA_VISIBLE_DEVICES``)::

    kogine run scripts/train/diffusion.py --config configs/train/flagship.py
"""

# ---- Data ------------------------------------------------------------------
DATA = "train"
# The MIB-consistent train set (every MIB group shares one (w, h)); see docs/data.md.
TRAIN_LANCE = "data/floorset_lite_mibfix.lance"
FLOORSET_ROOT = None
ALLOW_DOWNLOAD = True
# Dev split: 100 per block count (10,000) + 2,000 random = 12,000; 996,000 train rows.
DEV_PER_N_K = 100
DEV_RANDOM_SIZE = 2000
SPLIT_SEED = 20090220

# ---- Model -----------------------------------------------------------------
PRESET = None
ARCH_OVERRIDES = {
    "latent_dim": 3,
    "feature_dim": 19,
    "hidden": 768,
    "depth": 12,
    "heads": 12,
    "norm": "rmsnorm",
    "mlp": "swiglu",
    "mlp_ratio": 4.0,
    "attn": "sdpa_graph",
    "qk_norm": True,
    "graph_bias": True,
    "time_cond": "token",
    "post_norm": False,
    "graph_pe_dim": 0,
    "time_scale": 1000.0,
    "output_kind": "x0",
}

# ---- Objective -------------------------------------------------------------
FRAMING_SPEC = "rectified_flow"
TIME_SAMPLER = {"name": "logit_normal", "mean": 0.0, "std": 1.0}
LOSSES = [
    {"name": "denoise", "weight": 1.0, "cosine_weight": 0.0, "reduction": "global"}
]
LATENT_PARAM = "s_only"
GRAPH_PE = None
PIN_PULL_NORMALIZE = True
GRAPH_BIAS_NORMALIZE = True
MIB_ANCHOR = "group_mean"
# The dihedral group (rot90 x flip), a global shift, and 1% constraint dropout per kind.
AUGMENT = [
    {"rot90": {"choices": [0, 1, 2, 3]}},
    {"flip": {"p": 0.5}},
    {"shift": {"std": 0.05}},
    {
        "cond_dropout": {
            "boundary": 0.01,
            "cluster": 0.01,
            "mib": 0.01,
            "fixed": 0.01,
            "preplaced": 0.01,
            "cluster_per_group": True,
            "mib_per_group": True,
        }
    },
]

# ---- Compute: effective batch 256, 200k steps -------------------------------
GPUS = [0]
BATCH_SIZE = 256
GRAD_ACC = 2
BATCHING = "pad_to_max"
MAX_N = 127
MAX_STEPS = 200000
EPOCH = -1
PRECISION = "bf16-mixed"
DDP_COMPRESS_HOOK = "bf16"
NUM_WORKERS = 16
SEED = 20090220

# ---- Optimizer -------------------------------------------------------------
LR = 2e-4
BETAS = (0.9, 0.98)
WEIGHT_DECAY = 0.001
OPTIMIZER_MODE = "adamw"
BASE_DIM = 256
SCHEDULER_CONFIG = {"lr": {"mode": "cosine", "min_value": 0.01, "end": -1}}
SCHED_WARMUP_RATIO = 0.01
GRAD_CLIP = 1.0

USE_EMA = True
EMA_DECAY = 0.9995
COMPILE = {"mode": "module"}
GRAD_CKPT = False

# ---- In-training validation: soft metrics of the raw sample -----------------
EVAL_ENABLE = True
EVAL_MODE = "soft"
EVAL_MAX_DEV_CASES = 2000
EVAL_EVERY_N_STEPS = 10000
EVAL_RUN_AT_START = False
EVAL_NUM_RENDER = 0
EVAL_OFFICIAL = False
SAMPLE_STEPS = 16
SAMPLE_SOLVER = "euler"
# Paper experiments sample free (no projections); the shipped default projects.
SAMPLE_PROJECTIONS = []
EVAL_BEST_OF_N = 1
EVAL_MAX_BATCH = 2000
EVAL_SCORER = "full_fast"
EVAL_LEGALIZE_WORKERS = 0
REFINER = None
LEGALIZE_PORTFOLIO = None
SAMPLE_BLOCK_COUNTS = (21, 60, 120)

# ---- Logging ---------------------------------------------------------------
WANDB_PROJECT = "Trinity"
WANDB_OFFLINE = False
LOG_INTERVAL = 0
CKPT_INTERVAL = 25000
CKPT_SAVE_TOP_K = 1
CKPT_EVERY_N_EPOCHS = 0
OUT_DIR = "outputs/train"
NAME = "base"
