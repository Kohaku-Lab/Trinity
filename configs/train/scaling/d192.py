"""Scaling rung d192: the flagship recipe at hidden 192, 12 blocks, 3 heads, with
the runtime speed options on (pinned memory, the sync-free step, CUDA graphs).

Run::

    kogine run scripts/train/diffusion.py --config configs/train/scaling/d192.py
"""

from kohakuengine import use_config

_flagship = use_config("../flagship.py")

ARCH_OVERRIDES = {**_flagship.globals_dict["ARCH_OVERRIDES"], "hidden": 192, "heads": 3}

PIN_MEMORY = True
FAST_STEP = True
TRAIN_CUDA_GRAPHS = True
LOG_EVERY_N_STEPS = 50
PROGRESS_REFRESH_RATE = 50

NAME = "flagship_d192"
