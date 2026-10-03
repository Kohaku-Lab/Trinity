"""Generation cache of the flagship and its no-aux control over the 12k dev split.

Free sampling (no projection), six NFEs x 4 draws per case: 288,000 layouts per checkpoint.
Every later step of the flagship evaluation reads these shards (``free_nfe<k>.npz``).

Run::

    kogine run scripts/eval/generate.py --config configs/eval/flagship/generate.py
"""

CHECKPOINTS = {
    "flagship": "outputs/train/flagship/checkpoints/last.ckpt",
    "no_aux": "outputs/train/no_aux/checkpoints/last.ckpt",
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
