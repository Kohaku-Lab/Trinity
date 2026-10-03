# Baselines

Every baseline lives in `trinity_baselines`, apart from the method. The learned baselines are
trained on the same data, split, augmentation, batch and step budget as Trinity, and evaluated
by the same scripts ([evaluation.md](evaluation.md)).

```bash
pip install -e ".[baselines]"   # torch_geometric, for the ported backbones
```

## Direct regression

One deterministic forward from the conditioning to a layout, on the Trinity backbone
(`trinity_baselines.models.DirectRegressor`, trained by `RegressorTrainer`):

| head | predicts |
|---|---|
| `z` | the latent `(cx/s, cy/s, ρ)`; areas exact by the decode |
| `xywh` | the box `(cx/s, cy/s, w/s, h/s)`; areas free |

```bash
kogine run scripts/baselines/train_regressor.py    --config configs/baselines/train_reg_z.py
kogine run scripts/baselines/generate_regressor.py --config configs/baselines/generate_reg_z.py
```

`generate_regressor.py` writes the generation-cache format, so `score.py`, `refine.py` and
`legalize.py` read it like a diffusion cache.

## Ported published placers

The backbones of published learned placers, registered in `BASELINE_BACKBONE` and trained by
the Trinity trainer when a config sets `BACKBONE`:

| key | source |
|---|---|
| `flowplace_attgnn` | FlowPlace's `AttGNN` (also the ChipDiffusion backbone, trained with `ddpm_cosine_eps`) |
| `diffplace_vgnn` | DiffPlace's `VGNN` |
| `macrodiff_hetero` | MacroDiff+'s heterogeneous GNN |

```bash
kogine run scripts/train/diffusion.py --config configs/baselines/train_flowplace.py
```

Their own post-processing is ported too:

* **post-hoc refiners** — `trinity_baselines.ports.refine.PORTS`: `chipd_scheduled`,
  `chipd_standard`, `chipd_opt` (ChipDiffusion's legalizer modes), `diffplace`, `macrodiff`;
  selected by `REFINER` in `scripts/eval/refine.py` and `legalize.py`.
* **in-sampler guidance** — the `euler_guided` sampler with `GUIDANCE` `chipd_opt`,
  `diffplace` or `macrodiff`:

```bash
kogine run scripts/baselines/generate_guided.py --config configs/baselines/generate_guided_chipdiffusion.py
```

Each ported module names its source repository and file; licenses are listed in
[NOTICE](../NOTICE).

## Classic solvers

| solver | module | what it is |
|---|---|---|
| `parsac` | `trinity_baselines.classical.parsac` | Intel's PARSAC simulated annealer, driven through its Python extension |
| `sp_sa` | `trinity_baselines.classical.sp_sa` | sequence-pair simulated annealing, ported from BBOPlace-Bench |

PARSAC is not vendored. Build it once:

```bash
bash scripts/baselines/build_parsac.sh   # clones IntelLabs/parsac at a pinned commit into third_party/parsac
```

The script applies `scripts/baselines/parsac.patch` (two extra knobs, both off by default) and
builds the `-O3` extension into `third_party/parsac/build`; `TRINITY_PARSAC` and
`TRINITY_PARSAC_BUILD` point elsewhere.

```bash
kogine run scripts/baselines/solve_classical.py --config configs/baselines/solve_sp_sa.py
kogine run scripts/baselines/anneal_curve.py    --config configs/baselines/anneal_parsac_official.py
```

`solve_classical.py` writes a generation-style cache over the dev split; `anneal_curve.py`
records cost against time on the official 100 cases at several budgets and seeds.
