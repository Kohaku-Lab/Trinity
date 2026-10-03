"""The ported sequence-pair annealer, from a random pair, on the 100 FloorSet validation cases:
four seeds, one 100k-evaluation run per case and seed, the best-so-far layout snapshotted at
100 / 200 / 500 / 1k / 2k / 5k / 10k / 20k / 50k evaluations and at the end, scored after the
``scale_pack`` legalizer.

Run::

    kogine run scripts/baselines/anneal_curve.py --config configs/baselines/anneal_sp_sa_official.py
"""

TRAIN_LANCE = "data/floorset_lite_mibfix.lance"
CASES = {"source": "official"}
SOLVER_SPEC = {
    "name": "sp_sa",
    "checkpoint_evals": (100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000),
}
BUDGETS = (100_000,)
SEEDS = (1, 2, 3, 4)
INIT = None
LEGALIZE_ROUTE = [
    {"name": "scale_pack", "relation": "restored", "reshape": True, "trust": 1.3}
]
SCORER = "full_fast"
WORKERS = 24
OUT = "outputs/classical/sp_sa_official_1e5_curve.json"
