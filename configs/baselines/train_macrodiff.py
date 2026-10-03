"""MacroDiff+: the ported ``MacroPlacer`` denoiser with its DDPM objective on the Trinity recipe.

Backbone: ``macrodiff_hetero`` at the training configuration (a heterogeneous GATv2 cell / net
GNN, hidden 64, 5 layers, 4 heads; a token U-Net with 32 base channels; the HPWL-composed
epsilon). Objective: epsilon prediction on the variance-preserving schedule (``ddpm_eps``) with
a target-emitting head, uniform t. Everything else is the base recipe plus the wire-dropout
mixture. ``BACKBONE`` needs the ``baselines`` extra.

Run::

    kogine run scripts/train/diffusion.py --config configs/baselines/train_macrodiff.py
"""

from kohakuengine import use_config

_base = use_config("../train/_base.py")

BACKBONE = {"name": "macrodiff_hetero"}
ARCH_OVERRIDES = {**_base.globals_dict["ARCH_OVERRIDES"], "output_kind": "target"}
FRAMING_SPEC = "ddpm_eps"
TIME_SAMPLER = {"name": "uniform"}
COMPILE = None
AUGMENT = [
    *_base.globals_dict["AUGMENT"],
    {
        "wire_dropout": {
            "keep": 0.5,
            "beta": [1.0, 3.0],
            "packing": 0.1,
            "packing_range": [0.9, 1.0],
        }
    },
]

NAME = "baseline-macrodiff"
