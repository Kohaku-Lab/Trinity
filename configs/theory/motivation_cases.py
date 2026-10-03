"""The motivation figure's snapshots. Needs the free-sampling generation caches of
the flagship, ChipDiffusion and MacroDiff+ and the legalization caches of each published
model through its own ported refiner at NFE 32 (``legalize.py`` with ``SAVE_LAYOUTS``).

Run::

    kogine run scripts/theory/motivation_cases.py --config configs/theory/motivation_cases.py
"""

LEGALIZE_DIR = "outputs/eval/legalize"
GEN_DIR = "outputs/gen"
NFE = 32
RAW_SHARD = "free_nfe32"
PORT_CACHES = {
    "chipd_scheduled": ("baseline-chipdiffusion_port_chipd", 5000),
    "macrodiff": ("baseline-macrodiff_port_macrodiff", 500),
    "diffplace": ("baseline-diffplace_port_diffplace", 500),
}
RAW_RUNS = {
    "chipd": "baseline-chipdiffusion",
    "macro": "baseline-macrodiff",
    "trinity": "flagship",
}
BANDS = ((50, 70), (25, 35), (100, 120))
MIN_CLUSTERS = 2
MIN_CLUSTER_SIZE = 3
OUT = "outputs/theory/motivation_cases.json"
