"""Scaling rung d256: the flagship recipe at hidden 256, 12 blocks, 4 heads.

Run::

    kogine run scripts/train/diffusion.py --config configs/train/scaling/d256.py
"""

from kohakuengine import use_config

_flagship = use_config("../flagship.py")

ARCH_OVERRIDES = {**_flagship.globals_dict["ARCH_OVERRIDES"], "hidden": 256, "heads": 4}

NAME = "flagship_d256"
