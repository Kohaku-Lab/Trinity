"""Distribution metrics (FLD / FPD / MMD²) of the flagship and no-aux caches at NFE 1 and 32,
with the no-aux model as the permutation test's control.

Run::

    kogine run scripts/eval/dist_metrics.py --config configs/eval/flagship/dist_metrics.py
"""

GEN_DIR = "outputs/gen"
EMB_DIR = "outputs/gen/_emb"
RUNS = ["flagship", "no_aux"]
SHARDS = ["free_nfe1", "free_nfe32"]
CONTROL = "no_aux"
GRIDS = (8, 12)
FRAMES = ("bbox", "fixed")
TEST_DESC = "density_g8_bbox"
N_PERM = 200
WORKERS = 24
OUT = "outputs/gen/dist_metrics_flagship.json"
