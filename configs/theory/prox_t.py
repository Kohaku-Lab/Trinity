"""Theory T1: the proximal correction against t on the first 2000 dev cases at 19 values of t,
for the flagship (lambda = 0.01), the no-aux control (lambda = 0, the reference), a second
and third training seed of each, lambda = 0.05, and the t-weighted lambda = 0.1 t.

Run::

    kogine run scripts/theory/prox_t.py --config configs/theory/prox_t.py
"""

N_CASES = 2000
T_GRID = (
    0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1,
    0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
)  # fmt: skip
MODELS = {
    "flagship": ("outputs/train/flagship/checkpoints/last.ckpt", 0.01, False),
    "no_aux": ("outputs/train/no_aux/checkpoints/last.ckpt", 0.0, False),
    "flagship_s1": ("outputs/train/flagship_s1/checkpoints/last.ckpt", 0.01, False),
    "flagship_s2": ("outputs/train/flagship_s2/checkpoints/last.ckpt", 0.01, False),
    "no_aux_s1": ("outputs/train/no_aux_s1/checkpoints/last.ckpt", 0.0, False),
    "no_aux_s2": ("outputs/train/no_aux_s2/checkpoints/last.ckpt", 0.0, False),
    "aux_w0p05": ("outputs/train/aux_w0p05/checkpoints/last.ckpt", 0.05, False),
    "aux_t0p1": ("outputs/train/aux_t0p1/checkpoints/last.ckpt", 0.1, True),
}
REFERENCE = "no_aux"
NOISE_SEED = 20260924
MAX_ROWS = 256
OUT = "outputs/theory/prox_t.json"
