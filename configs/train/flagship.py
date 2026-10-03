"""The flagship: spectral-drawing graph PE + the info-mover residual, all six aux terms at
weight 0.01, and the wire-dropout mixture, on the base recipe (d768 / L12).

Run::

    kogine run scripts/train/diffusion.py --config configs/train/flagship.py
"""

from kohakuengine import use_config

_base = use_config("_base.py")

GRAPH_PE = {"name": "spectral_draw", "k": 2, "normalize": "log1p"}
ARCH_OVERRIDES = {
    **_base.globals_dict["ARCH_OVERRIDES"],
    "graph_pe_dim": 2,
    "graph_mix": {"name": "mp", "normalize": "row"},
}
LOSSES = [
    {"name": "denoise", "weight": 1.0, "cosine_weight": 0.0, "reduction": "global"},
    {"name": "overlap", "weight": 0.01},
    {"name": "group", "weight": 0.01},
    {"name": "mib", "weight": 0.01},
    {"name": "boundary", "weight": 0.01},
    {"name": "wl", "weight": 0.01},
    {"name": "area", "weight": 0.01},
]
# Wire dropout: 50% untouched, 40% at a Beta(1, 3) rate, 10% near-total (0.9-1.0).
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

NAME = "flagship"
