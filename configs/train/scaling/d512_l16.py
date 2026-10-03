"""Scaling rung d512_l16: the flagship recipe at hidden 512, 16 blocks, 8 heads, with the
runtime speed options on (pinned memory, the sync-free step, CUDA graphs).

Run::

    kogine run scripts/train/diffusion.py --config configs/train/scaling/d512_l16.py
"""

from kohakuengine import use_config

_flagship = use_config("../flagship.py")

ARCH_OVERRIDES = {
    **_flagship.globals_dict["ARCH_OVERRIDES"],
    "hidden": 512,
    "heads": 8,
    "depth": 16,
}

PIN_MEMORY = True
FAST_STEP = True
TRAIN_CUDA_GRAPHS = True
LOG_EVERY_N_STEPS = 50
PROGRESS_REFRESH_RATE = 50

NAME = "flagship_d512_l16"
