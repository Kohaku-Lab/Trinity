"""ChipDiffusion samples through its own ported scheduled gradient legalizer (5000 iterations)
at NFE 32, then ``scale_pack``: the published generator with its own refiner, in contest
units (hard cost, single sample and best of 4). The layouts it stores feed theory T3.

Run::

    kogine run scripts/eval/legalize.py --config configs/eval/ports/chipdiffusion_own_refiner.py
"""

RUN = "baseline-chipdiffusion"
GEN_DIR = "outputs/gen"
SHARD_PREFIX = "free"
NFES = (32,)
STEPS = (0, 5000)
DRAWS = 0
N_SELECT = (1, 4)
REFINER = "chipd_scheduled"
LEGALIZE_ROUTE = [
    {"name": "scale_pack", "relation": "restored", "reshape": True, "trust": 1.3}
]
SCORER = "full_fast"
SAVE_LAYOUTS = True
WORKERS = 24
OUT = "outputs/eval/legalize/baseline-chipdiffusion_port_chipd.json"
