"""Every dev case's reference layout through the three ported refiners at their
published step counts and through the closed-form refiner at the paper weights and at
the training weights (every loop's defaults in the script).

Run::

    kogine run scripts/theory/reference_loops.py \\
        --config configs/theory/reference_loops.py
"""

DEV_LIMIT = 0
LR = 0.001
LR_SCHEDULE = "linear"
BETAS = (0.0, 0.99)
MAX_ROWS = 2000
OUT = "outputs/theory/reference_loops.json"
