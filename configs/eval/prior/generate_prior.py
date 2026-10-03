"""The sampler's starting noise as a generation cache (run ``prior``, shard ``free_nfe0``):
the input of the refiner-only baseline.

Run::

    kogine run scripts/eval/generate_prior.py --config configs/eval/prior/generate_prior.py
"""

NAME = "prior"
SHARD = "free_nfe0"
K = 4
NOISE_SEED = 20260826
OUT_DIR = "outputs/gen"
