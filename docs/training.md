# Training

```bash
kogine run scripts/train/diffusion.py --config configs/train/flagship.py
```

`scripts/train/diffusion.py` is a kohaku-engine script: every knob is a typed UPPER_CASE
global with a runnable default. A config file overrides any of them, and a single run can
override more on the command line with a typed `--set KEY=VALUE`:

```bash
kogine run scripts/train/diffusion.py --config configs/train/smoke.py --set MAX_STEPS=100
```

## Configs

| config | what it trains |
|---|---|
| `configs/train/_base.py` | the base recipe every config builds on (below) |
| `configs/train/flagship.py` | the flagship: spectral-drawing graph PE, the info-mover residual, all six constraint terms at weight 0.01, the wire-dropout mixture |
| `configs/train/scaling/*.py` | the flagship at width 512 / 384 / 256 / 192 and depth 8 / 6 / 4, and 512 × 16 |
| `configs/train/no_aux.py` | the flagship without the constraint terms |
| `configs/train/objective_rf_x0_uniform.py` | a time-sampling variant: uniform `t` floored at 1e-2 instead of logit-normal |
| `configs/train/smoke.py` | 20 steps on the 100-case validation set |

A config reads the base back through `use_config` and overrides one axis:

```python
from kohakuengine import use_config

_base = use_config("_base.py")
ARCH_OVERRIDES = {**_base.globals_dict["ARCH_OVERRIDES"], "graph_pe_dim": 2}
```

## The base recipe

* **Model** — a set transformer over block tokens (d768, 12 blocks, 12 heads, RMSNorm,
  SwiGLU, QK-norm), the netlist as an additive attention bias, time as a token; the head
  emits the clean latent `x̂0`.
* **Latent** — per block `(cx/s, cy/s, ρ = log(w/h))`; the decode keeps every block's area
  exact.
* **Objective** — rectified flow with logit-normal time sampling; the denoising loss on the
  framing's target, plus the constraint terms of [physics.md](physics.md) when a config
  lists them.
* **Conditioning** — 19 per-block features (area, constraint codes, anchors, pin pull), the
  netlist adjacency, and the graph PE of the config.
* **Augmentation** — the dihedral group (rot90 × flip), a global shift, 1 % dropout of each
  constraint kind; the flagship adds wire dropout.
* **Compute** — effective batch 256 (micro-batch 128, two accumulation steps), 200k steps,
  AdamW (lr 2e-4, betas 0.9 / 0.98, weight decay 1e-3), cosine schedule with 1 % warmup,
  EMA 0.9995, bf16 mixed precision, one GPU.

In-training validation scores the raw sample of 2,000 dev cases with the soft metrics every
10k steps. Checkpoints go to `OUT_DIR/NAME/checkpoints/`.

## Runtime speed options

All off by default; none changes the trained model.

| knob | what it does |
|---|---|
| `PIN_MEMORY` | page-locked loader batches, so the host-to-device copy is asynchronous |
| `FAST_STEP` | fused AdamW; the batch copy and its graph PE on a side CUDA stream; the loss EMA kept on the device |
| `LOG_EVERY_N_STEPS` | Lightning's logger cadence |
| `PROGRESS_REFRESH_RATE` | progress-bar refresh cadence |
| `TRAIN_CUDA_GRAPHS` | CUDA graphs on the compiled blocks and loss terms during training only (needs `COMPILE = {"mode": "module"}` and bf16 / fp32) |

## Releasing a model

```bash
kogine run scripts/tools/export_release.py \
    --set 'RELEASES={"flagship": "outputs/train/flagship/checkpoints/last.ckpt"}'
```

writes `outputs/release/flagship/{config.json, model.safetensors}` (EMA weights), loadable with
`trinity.hub.load_model`.
