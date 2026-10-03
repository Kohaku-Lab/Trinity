"""The refiner-only baseline: the closed-form refiner at peak lr 1e-2 from the sampler's
starting noise (``prior``, NFE 0) for 0 / 400 / 1000 / 4000 steps, then ``scale_pack``.

Run::

    kogine run scripts/eval/legalize.py --config configs/eval/prior/legalize.py
"""

RUN = "prior"
GEN_DIR = "outputs/gen"
SHARD = "free_nfe0"
NFES = (0,)
STEPS = (0, 400, 1000, 4000)
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
LR = 0.01
LR_SCHEDULE = "linear"
BETAS = (0.0, 0.99)
LEGALIZE_ROUTE = [
    {"name": "scale_pack", "relation": "restored", "reshape": True, "trust": 1.3}
]
SCORER = "full_fast"
SAVE_LAYOUTS = True
WORKERS = 24
OUT = "outputs/eval/legalize/prior_lr1e-2_s4k.json"
