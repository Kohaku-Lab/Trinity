# Reading the code

## Two packages, two halves

* `trinity.floorplan` is **the problem**: instances, data loaders, geometry, the FloorSet
  checker and the soft metrics, the legalizer, rendering. It knows nothing about models.
* The rest of `trinity` is **the method**: it turns an instance into conditioning, samples a
  layout, refines it and hands it to `trinity.floorplan` for legalization and scoring.
* `trinity_baselines` holds every baseline and is never imported by `trinity`, except lazily
  when a training config names a baseline backbone.

## Registries and specs

Every swappable component is a class registered under a string key and built from a *spec*:
a key, a dotted import path, or `{"name": key, **kwargs}`.

```python
from trinity.registry import SAMPLER, build

sampler = build({"name": "euler", "num_steps": 32}, SAMPLER, framing=framing)
```

`build` runs once, when a script or trainer is set up; the built object is called directly
afterwards. Configs therefore select variants by name, and a new variant is a new registered
class.

| registry | keys | file |
|---|---|---|
| `GRAPH_PE` | `none`, `rwpe`, `spectral_draw` | `trinity/conditioning/graph_pe.py` |
| `ATTENTION` | `sdpa`, `sdpa_graph`, `flex_graph`, ... | `trinity/models/components/attention.py` |
| `TIME_COND` | `token`, `adaln`, `adaln_shared`, `additive` | `trinity/models/components/time_cond.py` |
| `GRAPH_MIX` | `none`, `mp` (the info-mover residual) | `trinity/models/components/graph_mix.py` |
| `LATENT_PARAM` | `s_only` | `trinity/latent_param.py` |
| `FRAMING` | `rectified_flow`, `ddpm_x0`, `ddpm_eps`, `ddpm_v`, `gvp`, ... | `trinity/framing/standard.py` |
| `TIME_SAMPLER` | `uniform`, `logit_normal` | `trinity/framing/time_sampler.py` |
| `LOSS` | `denoise`, the six constraint terms, `ref_*` | `trinity/losses/` |
| `SAMPLER` | `euler`, `heun` (`euler_guided` in the baselines) | `trinity/sampling/ode.py` |
| `PROJECTION` | `anchor_clamp`, `mib_group_mean` | `trinity/sampling/projection.py` |
| `REFINER` | `closed`, `constraint_latent` | `trinity/sampling/refine_*.py` |
| `LEGALIZER` | `scale_pack` | `trinity/floorplan/legalize/scale_pack.py` |
| `SCORER` | `full`, `full_fast`, `continuous`, `stub` | `trinity/floorplan/scoring/` |

## One instance's path

1. **Instance** — `trinity.floorplan.types.FloorplanInstance` (areas, constraint columns,
   netlist, pins), from `trinity.floorplan.data`.
2. **Conditioning** — `trinity.conditioning`: per-block features, the anchors of fixed and
   preplaced blocks, the netlist adjacency (`build_adjacency`) and the graph PE.
3. **Latent** — per block `(cx/s, cy/s, ρ)`; `trinity.decode.z_to_xywh` turns it into boxes
   with exact areas.
4. **Denoiser** — `trinity.models.backbone`: a set transformer over block tokens with the
   netlist as an attention bias (`trinity/models/block.py`, `graph_bias.py`).
5. **Sampling** — `trinity.sampling.ode`: the probability-flow ODE of the framing, with the
   state projections ([sampling.md](sampling.md)).
6. **Refining** — `trinity.sampling.refine_closed`: the hand-written gradient of the six
   terms ([refiner.md](refiner.md)).
7. **Legalizing and scoring** — `trinity.floorplan.legalize.legalize` and
   `trinity.floorplan.scoring.validate` ([legalizer.md](legalizer.md)).

`trinity.solver.DiffusionPlacer` runs steps 2–7 over a batch of instances;
`trinity.hub.load_model(...).placer()` builds one from a released model.

## Training

`trinity.training.DiffusionTrainer` (a Lightning module) draws `t` from the time sampler,
noises the latent with the framing, and sums the registered loss terms on the prediction
(`trinity/losses/`). `trinity.data` provides the dataset over the Lance training set, the
batching (`same_n` / `pad_to_max`) and the dev split; `trinity.augment_ops` the augmentation.
The training script `scripts/train/diffusion.py` builds both from its config
([training.md](training.md)).

## Tests

`tests/` holds equivalence checks (the closed-form refiner against autograd, the fast scorer
against the reference checker, the batched graph PE against the per-instance one), invariance
checks (padding), and end-to-end runs of the trainers, the placers, the legalizer and the
classic solvers. Tests that need FloorSet or a PARSAC build are skipped when it is absent.
