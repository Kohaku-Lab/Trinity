"""The direct regressor with the ``z`` head: one deterministic forward per case.

Same backbone, conditioning, data, split, augmentation (the base recipe plus the
wire-dropout mixture), batch, steps, optimizer and in-loop evaluation as the base
training recipe; only the generation mechanism differs.

Run::

    kogine run scripts/baselines/train_regressor.py \\
        --config configs/baselines/train_reg_z.py
"""

from kohakuengine import use_config

_base = use_config("../train/_base.py")

HEAD = "z"
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

NAME = "baseline-reg_z"
