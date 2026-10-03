"""Runtime against the block count: the flagship, its no-aux control and the four ported
learned placers (per NFE), the closed-form refiner and the three ported refiners (per
step), at 32 layouts per forward on one idle GPU; N up to 1200 through tiled synthetic
cases.

Run::

    kogine run scripts/profile/time_scaling.py --config configs/profile/time_scaling.py
"""

SAMPLE_PROJECTIONS = []
CHECKPOINTS = {
    "flagship": "outputs/train/flagship/checkpoints/last.ckpt",
    "no_aux": "outputs/train/no_aux/checkpoints/last.ckpt",
    "baseline-flowplace": "outputs/train/baseline-flowplace/checkpoints/last.ckpt",
    "baseline-chipdiffusion": (
        "outputs/train/baseline-chipdiffusion/checkpoints/last.ckpt"
    ),
    "baseline-diffplace": "outputs/train/baseline-diffplace/checkpoints/last.ckpt",
    "baseline-macrodiff": "outputs/train/baseline-macrodiff/checkpoints/last.ckpt",
}
N_LADDER = (24, 40, 60, 80, 100, 120, 240, 360, 480, 720, 960, 1200)
SUB_N = 120
ROWS = 32
DRAWS = 1
NFE_PROBES = (8, 32)
REPEATS = 5
REFINER_STEPS = 100
REFINER_LR = 0.001
PORT_NAMES = ("chipd_scheduled", "diffplace", "macrodiff")
SEED = 20260922
OUT = "outputs/profile/time_scaling.json"
