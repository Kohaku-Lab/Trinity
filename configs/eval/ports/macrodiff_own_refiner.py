"""MacroDiff+ samples through its own ported guidance loop (500 iterations) at NFE 32, then
``scale_pack``: the published generator with its own refiner, scored by the hard cost
(single sample and best of 4). The layouts it stores feed the motivation figure.

Run::

    kogine run scripts/eval/legalize.py --config configs/eval/ports/macrodiff_own_refiner.py
"""

RUN = "baseline-macrodiff"
GEN_DIR = "outputs/gen"
SHARD_PREFIX = "free"
NFES = (32,)
STEPS = (0, 500)
DRAWS = 0
N_SELECT = (1, 4)
REFINER = "macrodiff"
LEGALIZE_ROUTE = [
    {"name": "scale_pack", "relation": "restored", "reshape": True, "trust": 1.3}
]
SCORER = "full_fast"
SAVE_LAYOUTS = True
WORKERS = 24
OUT = "outputs/eval/legalize/baseline-macrodiff_port_macrodiff.json"
