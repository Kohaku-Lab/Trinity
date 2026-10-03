"""The flagship pipeline (NFE 8, the 400-step refiner, ``scale_pack``, 16 draws) on the
five MCNC cases with hard blocks, a square outline with 10% white space and terminals at
their file positions.

Run::

    kogine run scripts/eval/bookshelf.py \\
        --config configs/eval/bookshelf/flagship_mcnc_hard.py
"""

CHECKPOINT = "outputs/train/flagship/checkpoints/last.ckpt"
SUITE = "mcnc"
VARIANT = "HARD"
NAMES = ()
GAMMA = 0.10
ASPECT = 1.0
MAP_PINS = False
SETTINGS = ((8, 400, 16),)
SEEDS = (0,)
SAMPLE_SOLVER = "euler"
SAMPLE_PROJECTIONS = []
WEIGHTS = {
    "overlap": 10.0,
    "group": 1.0,
    "mib": 1.0,
    "boundary": 1.0,
    "wl": 1.0,
    "area": 1.0,
    "outline": 1.0,
}
LR = 0.001
LR_SCHEDULE = "linear"
BETAS = (0.0, 0.99)
LEGALIZE_ROUTE = [
    {"name": "scale_pack", "relation": "restored", "reshape": True, "trust": 1.3}
]
WORKERS = 16
OUT = "outputs/eval/bookshelf/flagship_mcnc_hard.json"
