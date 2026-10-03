"""BBOPlace-Bench's sequence-pair SA over the full dev split (12,000 cases) at its 10k-evaluation
budget, four seeded solves per case, cached as the shard ``free_nfe1``.

Run::

    kogine run scripts/baselines/solve_classical.py --config configs/baselines/solve_sp_sa.py
"""

TRAIN_LANCE = "data/floorset_lite_mibfix.lance"
DEV_PER_N_K = 100
DEV_RANDOM_SIZE = 2000
SPLIT_SEED = 20090220
DEV_LIMIT = 0
SOLVER_SPEC = {"name": "sp_sa", "max_evals": 10000}
NAME = "baseline-sp_sa"
K = 4
WORKERS = 12
OUT_DIR = "outputs/gen"
