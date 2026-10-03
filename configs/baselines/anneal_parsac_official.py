"""PARSAC as published, from a random tree, on the 100 FloorSet validation cases: four
seeds, one 10M-step run per case and seed, the layout snapshotted after every one of the
100 stages and scored after the ``scale_pack`` legalizer. Needs
``scripts/baselines/build_parsac.sh``.

Run::

    kogine run scripts/baselines/anneal_curve.py \\
        --config configs/baselines/anneal_parsac_official.py
"""

TRAIN_LANCE = "data/floorset_lite_mibfix.lance"
CASES = {"source": "official"}
SOLVER_SPEC = {
    "name": "parsac",
    "units_per_s": 4000.0,
    "checkpoint_stages": tuple(range(100)),
}
BUDGETS = (10_000_000,)
SEEDS = (1, 2, 3, 4)
INIT = None
LEGALIZE_ROUTE = [
    {"name": "scale_pack", "relation": "restored", "reshape": True, "trust": 1.3}
]
SCORER = "full_fast"
WORKERS = 24
OUT = "outputs/classical/parsac_official_1e7_curve.json"
