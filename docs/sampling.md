# Sampling

`trinity/sampling/ode.py` (samplers), `trinity/sampling/projection.py` (state projections),
`trinity/solver/diffusion_placer.py` (the full pipeline).

## The sampler

The network predicts the clean latent `x̂0` at every step; the sampler turns it into the
probability-flow drift of the framing (`framing.x0_to_velocity`) and integrates from `t = 1`
(noise) to `t = 1e-3`:

* the time grid is `linspace(1, 1e-3, nfe + 1)`;
* `euler` takes `nfe` explicit steps (`heun` is the second-order variant);
* the returned layout is the network's `x̂0` read at the last time, so a run costs `nfe + 1`
  network evaluations.

```python
from trinity.registry import SAMPLER, build

sampler = build({"name": "euler", "num_steps": 32}, SAMPLER, framing=framing)
z = sampler.sample(model, cond, shape=(rows, max_n, 3), device="cuda")
```

## State projections

A projection writes known answers into the sampler state. Projections are applied to the
initial noise, to the state after every step, and to the returned `x̂0`.

| key | what it writes |
|---|---|
| `anchor_clamp` | preplaced positions and shapes, fixed shapes, and the shape of an MIB group that has a fixed or preplaced member |
| `mib_group_mean` | one shared aspect for every all-soft MIB group: the mean of its members' `ρ` |

**Default.** `projections=None` means `DEFAULT_PROJECTIONS = ("anchor_clamp",
"mib_group_mean")`; this is the shipped behaviour. `projections=[]` is free sampling.

**Paper configs.** Every model in the paper is evaluated with free sampling, so every config
under `configs/` sets the projection list to `[]` explicitly. The projections are an
inference-time option on a finished checkpoint, not part of the trained model.

## The pipeline

`DiffusionPlacer` runs, per batch of cases:

1. `build_group_ctx` — pad `len(group) × samples` rows to one length and build the
   conditioning (features, adjacency, anchors, graph PE);
2. `sample_latents` — one sampler run over all rows;
3. `refine_decode` — the refiner (optional) and the decode to boxes;
4. legalization of every candidate (`trinity.floorplan.legalize.legalize`, in a process pool
   with `legalize_workers > 0`);
5. the cheapest legalized candidate per case.

`raw_sample_boxes` returns the first raw candidate (no refiner, no legalizer), the input of
the soft metrics.
