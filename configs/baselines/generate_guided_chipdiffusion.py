"""ChipDiffusion with its own in-sampler guidance (the ``opt`` mode of its evaluation config):
the dev split, NFE 1-32, K = 4 draws, the noise of the unguided shard.

The checkpoint is ChipDiffusion trained on the Trinity recipe: the ``AttGNN`` backbone of
``configs/baselines/train_flowplace.py`` with ``FRAMING_SPEC = "ddpm_cosine_eps"`` and
``NAME = "baseline-chipdiffusion"``. Sampling is free (no projections) and the shard keeps the
name ``free``, which the scoring, refining and legalizing scripts read.

Run::

    kogine run scripts/baselines/generate_guided.py --config configs/baselines/generate_guided_chipdiffusion.py
"""

NAME = "baseline-chipdiffusion-guided"
CKPT = "outputs/train/baseline-chipdiffusion/checkpoints/last.ckpt"
GUIDANCE = "chipd_opt"
GUIDANCE_KWARGS = {}
PROJECTIONS = []
SHARD = "free"
TRAIN_LANCE = "data/floorset_lite_mibfix.lance"
DEV_PER_N_K = 100
DEV_RANDOM_SIZE = 2000
SPLIT_SEED = 20090220
DEV_LIMIT = 0
NFE = (1, 2, 4, 8, 16, 32)
K = 4
NOISE_SEED = 20260826
DEVICE = "cuda"
MAX_ROWS = 500
OUT_DIR = "outputs/gen"
