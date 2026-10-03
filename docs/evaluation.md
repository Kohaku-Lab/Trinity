# Evaluation

Evaluation runs as a chain of kohaku-engine scripts over the 12,000-case dev split
([data.md](data.md)). The first step caches raw samples; every later step reads the cache, so
one generation serves the soft metrics, the refiner grid and the legalized (hard) cost.

```bash
kogine run scripts/eval/generate.py     --config configs/eval/flagship/generate.py
kogine run scripts/eval/score.py        --config configs/eval/flagship/score.py
kogine run scripts/eval/dist_metrics.py --config configs/eval/flagship/dist_metrics.py
kogine run scripts/eval/refine.py       --config configs/eval/flagship/refine.py
kogine run scripts/eval/legalize.py     --config configs/eval/flagship/legalize.py
kogine run scripts/eval/official.py     --config configs/eval/flagship/official.py
```

Models are loaded through `trinity.hub.load_model`, so `CHECKPOINT` / `CHECKPOINTS` accept a
training `.ckpt`, a release directory, or the Hugging Face id `KBlueLeaf/Trinity` with a
`SCALE` (`flagship`, `d512`, ...).

## The steps

| script | reads | writes | what it measures |
|---|---|---|---|
| `generate.py` | checkpoints | `outputs/gen/<run>/<set>_nfe<k>.npz` | raw latents of `K` draws per case at every NFE and projection set, from a fixed noise seed per chunk |
| `generate_prior.py` | — | the same format | the sampler's starting noise itself (the refiner-only baseline's input) |
| `score.py` | generation shards | one JSON | the soft metric vector of every draw: the six constraint quantities of [physics.md](physics.md) and the soft cost |
| `dist_metrics.py` | generation shards | one JSON | distribution distances to the ground-truth layouts (FLD, FPD, MMD²) with a permutation test against a control run |
| `refine.py` | generation shards | one JSON | the NFE × refiner-steps grid of the soft metrics after the closed-form refiner (or a ported refiner, `REFINER`) |
| `legalize.py` | generation shards | one JSON (+ layouts) | the same grid through the legalizer: hard cost of a single draw and best of `N_SELECT` |
| `official.py` | a checkpoint | one JSON | the full pipeline on the official 100 validation cases, per case and timed, over a ladder of (NFE, refiner steps, best of N) settings |
| `bookshelf.py` | a checkpoint | one JSON | the full pipeline on GSRC / MCNC under the fixed-outline protocol |

Shard names are `<projection set>_nfe<k>`; the paper configs sample free
(`PROJECTION_SETS = {"free": []}`), so the later steps read `free_nfe<k>`.

## Metrics

* **Soft metrics** — computed on a layout before legalization
  (`trinity.floorplan.scoring.vector.COLS`): the continuous quantities of
  [physics.md](physics.md) (`overlap_ratio`, `group_gap`, `mib_var`, `boundary_dist`,
  `wl_norm`, `compactness`), the wirelength and area gaps to the ground truth, the severity of
  each soft-constraint violation, and the soft cost
  `(1 + α·(hpwl_gap + area_gap)) · exp(β·sev_rel) · exp(β_overlap·overlap_ratio)` — the hard
  cost with violation severities in place of counts and an overlap factor in place of the
  feasibility check.
* **Hard cost** — the FloorSet checker on the legalized layout (`trinity.floorplan.scoring`):
  feasibility, the wirelength and area gaps to the ground truth, and the counted soft-constraint
  violations, combined as `(1 + 0.5·(hpwl_gap⁺ + area_gap⁺)) · exp(2·V_rel)` (10 when
  infeasible).
* **Distribution metrics** — layout embeddings from rasterized block-density descriptors
  (`GRIDS`, `FRAMES`), compared with the dev ground truth.

## Other configs

| config | what it runs |
|---|---|
| `configs/eval/prior/*` | the refiner-only baseline: the closed-form refiner from the starting noise, then the legalizer |
| `configs/eval/bookshelf/*` | the flagship on GSRC (soft blocks) and MCNC (hard blocks) |
| `configs/eval/ports/*` | a published placer's own post-hoc refiner on its samples |
| `configs/profile/*` | runtime: time against block count, per-unit costs on the official set, per-step time of every correction loop |
| `configs/theory/*` | the measurements behind the paper's propositions: the proximal correction against `t`, reference layouts through every correction loop, the motivation figure's snapshots |
