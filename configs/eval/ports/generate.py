"""Generation cache of the four ported published placers over the 12k dev split.

Free sampling, six NFEs x 4 draws per case, the same noise as the flagship's cache. The
checkpoints come from
``configs/baselines/train_{flowplace,chipdiffusion,diffplace,macrodiff}.py``; the
``*_own_refiner.py`` configs and the motivation-figure script read these shards
(``free_nfe<k>.npz``).

Run::

    kogine run scripts/eval/generate.py --config configs/eval/ports/generate.py
"""

CHECKPOINTS = {
    "baseline-flowplace": "outputs/train/baseline-flowplace/checkpoints/last.ckpt",
    "baseline-chipdiffusion": (
        "outputs/train/baseline-chipdiffusion/checkpoints/last.ckpt"
    ),
    "baseline-diffplace": "outputs/train/baseline-diffplace/checkpoints/last.ckpt",
    "baseline-macrodiff": "outputs/train/baseline-macrodiff/checkpoints/last.ckpt",
}
DEV_PER_N_K = 100
DEV_RANDOM_SIZE = 2000
SPLIT_SEED = 20090220
NFE = (1, 2, 4, 8, 16, 32)
PROJECTION_SETS = {"free": []}
K = 4
SAMPLE_SOLVER = "euler"
NOISE_SEED = 20260826
MAX_ROWS = 2000
OUT_DIR = "outputs/gen"
