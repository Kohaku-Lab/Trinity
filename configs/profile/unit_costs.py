"""Per-unit costs on the official 100 cases: the legalizer per call on the flagship's refined
draws (NFE 8, 400 refiner steps, 16 draws, free sampling), PARSAC per annealing step and SP-SA
per evaluation, against the block count.

Run::

    kogine run scripts/profile/unit_costs.py --config configs/profile/unit_costs.py
"""

CHECKPOINT = "outputs/train/flagship/checkpoints/last.ckpt"
SAMPLE_SOLVER = "euler"
SAMPLE_PROJECTIONS = []
NFE = 8
STEPS = 400
DRAWS = 16
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
PARSAC_STEPS = 20_000
SPSA_EVALS = 2_000
SEED = 0
OUT = "outputs/profile/unit_costs.json"
