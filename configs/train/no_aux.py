"""The flagship without aux terms: spectral-drawing PE + info mover + wire dropout, trained on
the denoise loss alone.

Run::

    kogine run scripts/train/diffusion.py --config configs/train/no_aux.py
"""

from kohakuengine import use_config

_flagship = use_config("flagship.py")

LOSSES = [
    {"name": "denoise", "weight": 1.0, "cosine_weight": 0.0, "reduction": "global"}
]

NAME = "no_aux"
