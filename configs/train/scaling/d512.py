"""Scaling rung d512: the flagship recipe at hidden 512, 12 blocks, 8 heads.

Run::

    kogine run scripts/train/diffusion.py --config configs/train/scaling/d512.py
"""

from kohakuengine import use_config

_flagship = use_config("../flagship.py")

ARCH_OVERRIDES = {**_flagship.globals_dict["ARCH_OVERRIDES"], "hidden": 512, "heads": 8}

NAME = "flagship_d512"
