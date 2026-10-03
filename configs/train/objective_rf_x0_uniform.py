"""Objective / time-sampling example: the base recipe with uniform t floored at 1e-2
instead of logit-normal t (x0 head, v target). The other objectives swap
``FRAMING_SPEC`` / ``TIME_SAMPLER`` / ``output_kind`` the same way, e.g. ``ddpm_eps``
with ``ARCH_OVERRIDES["output_kind"] = "target"``.

Run::

    kogine run scripts/train/diffusion.py \\
        --config configs/train/objective_rf_x0_uniform.py
"""

from kohakuengine import use_config

use_config("_base.py")

TIME_SAMPLER = {"name": "uniform", "t_min": 1e-2}

NAME = "objective-rf_x0_uniform_tmin1e-2"
