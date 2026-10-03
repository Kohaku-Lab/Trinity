"""The flagship's NFE x refiner-steps grid through ``scale_pack``: hard cost per cell for a
single sample and best of 4, with the refined latents, legalized boxes and per-draw soft
vectors stored.

Run::

    kogine run scripts/eval/legalize.py --config configs/eval/flagship/legalize.py
"""

RUN = "flagship"
GEN_DIR = "outputs/gen"
SHARD_PREFIX = "free"
NFES = (1, 2, 4, 8, 16, 32)
STEPS = (0, 10, 25, 50, 100, 200, 400)
DRAWS = 0
N_SELECT = (1, 4)
REFINER = "closed"
WEIGHTS = {
    "overlap": 10.0,
    "group": 1.0,
    "mib": 1.0,
    "boundary": 1.0,
    "wl": 1.0,
    "area": 1.0,
}
LR = 0.001
LR_SCHEDULE = "linear"
BETAS = (0.0, 0.99)
LEGALIZE_ROUTE = [
    {"name": "scale_pack", "relation": "restored", "reshape": True, "trust": 1.3}
]
SCORER = "full_fast"
SAVE_LAYOUTS = True
WORKERS = 24
OUT = "outputs/eval/legalize/flagship.json"
