"""Smoke run: a tiny flagship-shaped model on the 100-case validation set, a few steps, offline
logging. Checks the whole training path end to end in about a minute.

Run::

    kogine run scripts/train/diffusion.py --config configs/train/smoke.py
"""

from kohakuengine import use_config

_flagship = use_config("flagship.py")

DATA = "validation"
ARCH_OVERRIDES = {
    **_flagship.globals_dict["ARCH_OVERRIDES"],
    "hidden": 64,
    "depth": 2,
    "heads": 2,
}
BATCH_SIZE = 8
GRAD_ACC = 1
NUM_WORKERS = 0
MAX_STEPS = 20
COMPILE = None
PRECISION = "32-true"

EVAL_EVERY_N_STEPS = 10
EVAL_MAX_DEV_CASES = 8
SAMPLE_STEPS = 4

WANDB_OFFLINE = True
CKPT_INTERVAL = 20
OUT_DIR = "outputs/smoke"
NAME = "smoke"
