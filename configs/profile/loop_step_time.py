"""One per-step time per correction loop (the closed-form refiner and the three
ported refiners) at 32 / 256 / 1000 layouts per forward on the first 2000 dev cases.

Run::

    kogine run scripts/profile/loop_step_time.py --config configs/profile/loop_step_time.py
"""

N_CASES = 2000
ROWS = (32, 256, 1000)
STEPS = 100
REPEATS = 5
JITTER = 0.05
LOOPS = ("closed", "chipd_scheduled", "diffplace", "macrodiff")
WEIGHTS = {
    "overlap": 10.0,
    "group": 1.0,
    "mib": 1.0,
    "boundary": 1.0,
    "wl": 1.0,
    "area": 1.0,
}
LR = 0.001
BETAS = (0.0, 0.99)
SEED = 20260924
OUT = "outputs/profile/loop_step_time.json"
