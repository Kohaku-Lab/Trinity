"""The official 100 cases with the flagship pipeline, per case and timed: free sampling, the
closed-form refiner, ``scale_pack``, one pool worker per draw. The settings are the
time-versus-score ladder: best of {4, 8, 16} at NFE 8 x 400 steps, then lower NFE and fewer
refiner steps at best of 16.

Run::

    kogine run scripts/eval/official.py --config configs/eval/flagship/official.py
"""

CHECKPOINT = "outputs/train/flagship/checkpoints/last.ckpt"
SETTINGS = (
    (8, 400, 16),
    (8, 400, 8),
    (8, 400, 4),
    (4, 400, 16),
    (1, 400, 16),
    (8, 100, 16),
    (4, 100, 16),
    (1, 100, 16),
    (8, 50, 16),
    (1, 50, 16),
)
SAMPLE_SOLVER = "euler"
SAMPLE_PROJECTIONS = []
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
WORKERS = 16
SEED = 0
OUT = "outputs/eval/official/flagship_ladder.json"
