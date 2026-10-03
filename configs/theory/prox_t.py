"""The proximal correction against t on the first 2000 dev cases at 19 values of t, for
the flagship (lambda = 0.01, the released model) against the no-aux control (lambda = 0,
the reference, trained with ``configs/train/no_aux.py``).

The paper also measures more seeds of both models and other aux weights; train them from
a copy of ``configs/train/flagship.py`` or ``no_aux.py`` with another ``SEED`` or other
term weights and ``NAME``, and add one ``MODELS`` entry per run: ``"<NAME>":
("outputs/train/<NAME>/checkpoints/last.ckpt", lambda, t_weighted)``.

Run::

    kogine run scripts/theory/prox_t.py --config configs/theory/prox_t.py
"""

N_CASES = 2000
T_GRID = (
    0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1,
    0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
)  # fmt: skip
MODELS = {
    "flagship": ("KBlueLeaf/Trinity", 0.01, False),
    "no_aux": ("outputs/train/no_aux/checkpoints/last.ckpt", 0.0, False),
}
SCALES = {"flagship": "flagship"}
REFERENCE = "no_aux"
NOISE_SEED = 20260924
MAX_ROWS = 256
OUT = "outputs/theory/prox_t.json"
