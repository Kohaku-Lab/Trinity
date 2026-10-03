"""The NFE x refiner-steps grid of the flagship with the paper's closed-form refiner:
linear decay from peak lr 1e-3, Adam betas (0, 0.99), overlap weight 10 and the other
five terms at 1, every cached draw, batched GPU scoring.

Run::

    kogine run scripts/eval/refine.py --config configs/eval/flagship/refine.py
"""

RUN = "flagship"
GEN_DIR = "outputs/gen"
SHARD_PREFIX = "free"
NFES = (1, 2, 4, 8, 16, 32)
STEPS = (0, 10, 25, 50, 100, 200, 400)
DRAWS = 0
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
OPTIMIZER = "adam"
BETAS = (0.0, 0.99)
CPU_CHECK_STEPS = ()
OUT = "outputs/eval/refine/flagship.json"
