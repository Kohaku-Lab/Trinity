"""The direct regressor's layout of every dev
case (12,000), cached as the shard ``reg``.

Run::

    kogine run scripts/baselines/generate_regressor.py \\
        --config configs/baselines/generate_reg_z.py
"""

CKPTS = {"baseline-reg_z": "outputs/train/baseline-reg_z/checkpoints/last.ckpt"}
TRAIN_LANCE = "data/floorset_lite_mibfix.lance"
DEV_PER_N_K = 100
DEV_RANDOM_SIZE = 2000
SPLIT_SEED = 20090220
DEV_LIMIT = 0
SHARD = "reg"
DEVICE = "cuda"
MAX_ROWS = 2000
OUT_DIR = "outputs/gen"
