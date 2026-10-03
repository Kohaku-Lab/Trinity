"""Scaling rung l4: the flagship recipe at depth 4 (hidden 768, 12 heads).

Run::

    kogine run scripts/train/diffusion.py --config configs/train/scaling/l4.py
"""

from kohakuengine import use_config

_flagship = use_config("../flagship.py")

ARCH_OVERRIDES = {**_flagship.globals_dict["ARCH_OVERRIDES"], "depth": 4}

NAME = "flagship_l4"
