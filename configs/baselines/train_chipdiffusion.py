"""ChipDiffusion: the ported ``AttGNN`` denoiser
with its DDPM objective on the Trinity recipe.

Backbone: ``flowplace_attgnn`` (the same network as FlowPlace). Objective: epsilon
prediction on the cosine variance-preserving schedule (``ddpm_cosine_eps``) with a
target-emitting head, uniform t. Data, split, augmentation (the base recipe plus the
wire-dropout mixture), batch, steps, optimizer and evaluation are the base recipe's.
``BACKBONE`` needs the ``baselines`` extra (``torch_geometric``).

Run::

    kogine run scripts/train/diffusion.py \\
        --config configs/baselines/train_chipdiffusion.py
"""

from kohakuengine import use_config

_base = use_config("../train/_base.py")

BACKBONE = "flowplace_attgnn"
ARCH_OVERRIDES = {**_base.globals_dict["ARCH_OVERRIDES"], "output_kind": "target"}
FRAMING_SPEC = "ddpm_cosine_eps"
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

NAME = "baseline-chipdiffusion"
