"""Scaling rung d384: the flagship recipe at hidden 384, 12 blocks, 6 heads.

Run::

    kogine run scripts/train/diffusion.py --config configs/train/scaling/d384.py
"""

from kohakuengine import use_config

_flagship = use_config("../flagship.py")

ARCH_OVERRIDES = {**_flagship.globals_dict["ARCH_OVERRIDES"], "hidden": 384, "heads": 6}

NAME = "flagship_d384"
