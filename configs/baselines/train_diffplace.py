"""DiffPlace: the ported ``VectorGNNV2Global`` denoiser
with its DDPM objective on the Trinity recipe.

Backbone: ``diffplace_vgnn`` at the deployed configuration (hidden 256, 8 blocks of 2
vector message-passing layers, 8 heads, a global supernode after every block).
Objective: epsilon prediction on the variance-preserving schedule (``ddpm_eps``) with a
target-emitting head, uniform t. Micro-batch 64 (``GRAD_ACC = 4``) keeps the effective
batch at 256. Everything else is the base recipe plus the wire-dropout mixture.
``BACKBONE`` needs the ``baselines`` extra.

Run::

    kogine run scripts/train/diffusion.py --config configs/baselines/train_diffplace.py
"""

from kohakuengine import use_config

_base = use_config("../train/_base.py")

BACKBONE = {"name": "diffplace_vgnn"}
ARCH_OVERRIDES = {**_base.globals_dict["ARCH_OVERRIDES"], "output_kind": "target"}
FRAMING_SPEC = "ddpm_eps"
TIME_SAMPLER = {"name": "uniform"}
COMPILE = None
GRAD_ACC = 4
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

NAME = "baseline-diffplace"
