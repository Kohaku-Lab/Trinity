"""The flagship pipeline (NFE 8, the 400-step refiner, ``scale_pack``) on the six GSRC cases
under the soft-block protocol: every block soft with w/h in [1/3, 3], a square outline with
10% white space, terminals at their file positions; 16 and 64 draws, four sampling seeds.

Run::

    kogine run scripts/eval/bookshelf.py --config configs/eval/bookshelf/flagship_gsrc_soft.py
"""

CHECKPOINT = "outputs/train/flagship/checkpoints/last.ckpt"
SUITE = "gsrc"
VARIANT = "SOFT"
NAMES = ()
GAMMA = 0.10
ASPECT = 1.0
MAP_PINS = False
SETTINGS = ((8, 400, 16), (8, 400, 64))
SEEDS = (0, 1, 2, 3)
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
OUT = "outputs/eval/bookshelf/flagship_gsrc_soft_g10.json"
