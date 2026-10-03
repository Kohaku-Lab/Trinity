"""Scaling rung l8: the flagship recipe at depth 8 (hidden 768, 12 heads).

Run::

    kogine run scripts/train/diffusion.py --config configs/train/scaling/l8.py
"""

from kohakuengine import use_config

_flagship = use_config("../flagship.py")

ARCH_OVERRIDES = {**_flagship.globals_dict["ARCH_OVERRIDES"], "depth": 8}

NAME = "flagship_l8"
