# Trinity: One Differentiable Physics for Training, Refining and Scoring Generative Floorplanners

[arXiv (coming soon)](#)

Trinity is a graph-conditioned diffusion model for SoC floorplanning on
[FloorSet](https://github.com/IntelLabs/FloorSet). One set of six differentiable constraint
terms (overlap, grouping, MIB, boundary, wirelength, area) is used three times:

* **training** — as the auxiliary loss of the denoiser;
* **refining** — as the energy a closed-form refiner descends on a finished sample;
* **scoring** — as the soft metrics that evaluate a layout before legalization.

This repository holds the code of the paper: the model, its training, the sampler, the
refiner, the legalizer, every evaluation, and every baseline. It contains no results.

## Install

```bash
git clone https://github.com/Kohaku-Lab/Trinity.git
cd Trinity
pip install -e .                 # trinity + trinity_baselines
pip install -e ".[baselines]"    # + torch_geometric for the ported published placers
pip install -e ".[dev]"          # + pytest, black, ruff
```

Python ≥ 3.13, PyTorch ≥ 2.12.

## Quick start

Released models (the flagship and eight smaller sizes) are on the Hugging Face Hub as
`KBlueLeaf/Trinity`, one subfolder per size:

```python
from trinity.floorplan.data import load_validation_case
from trinity.floorplan.scoring import validate
from trinity.hub import load_model

model = load_model("KBlueLeaf/Trinity", scale="flagship", device="cuda")
placer = model.placer(samples=16)  # sampler -> closed-form refiner -> legalizer, best of 16
instance = load_validation_case(60)  # FloorSet validation case with 60 blocks
placement = placer.solve(instance)
print(validate(placement, "full").score)
```

| size | width × depth | parameters |
|---|---|---|
| `flagship` | 768 × 12 | 92.9 M |
| `d512_l16` | 512 × 16 | 56.0 M |
| `d512` | 512 × 12 | 42.1 M |
| `d384` | 384 × 12 | 23.3 M |
| `d256` | 256 × 12 | 10.6 M |
| `d192` | 192 × 12 | 5.9 M |
| `l8` | 768 × 8 | 62.2 M |
| `l6` | 768 × 6 | 46.8 M |
| `l4` | 768 × 4 | 31.5 M |

The sampler writes the known answers into its state by default (preplaced and fixed blocks,
one shared shape per MIB group; [docs/sampling.md](docs/sampling.md)). The paper's
experiments sample free: `model.placer(projections=[])`.

## Running the experiments

Every script is a [kohaku-engine](https://pypi.org/project/kohaku-engine/) script, run with a
config file; any knob can also be overridden with a typed `--set KEY=VALUE`.

```bash
# data: download FloorSet-Lite and build the training set
python -c "from trinity.floorplan.data import download_train_raw; download_train_raw()"
kogine run scripts/data/transcode_floorset.py

# training
kogine run scripts/train/diffusion.py --config configs/train/flagship.py

# evaluation over the 12k dev split
kogine run scripts/eval/generate.py --config configs/eval/flagship/generate.py
kogine run scripts/eval/score.py    --config configs/eval/flagship/score.py
kogine run scripts/eval/legalize.py --config configs/eval/flagship/legalize.py
```

The configs under `configs/` show one of each experiment of the paper; the other runs change
one or two knobs of these.

## Layout

```
src/trinity/floorplan/   the problem: FloorSet / bookshelf data, geometry, scoring, legalization
src/trinity/             the method: conditioning, models, framings, losses, training,
                         sampling, the refiner, the placer, the model hub loader
src/trinity_baselines/   the baselines: regressors, ported published placers, classic solvers
scripts/                 kohaku-engine scripts (data, train, eval, baselines, profile, theory)
configs/                 config files for the scripts
docs/                    how to run things and how to read the code
tests/                   pytest suite
```

## Documentation

* [docs/code.md](docs/code.md) — how to read the code: the packages, the registries, one
  sample's path through them
* [docs/physics.md](docs/physics.md) — the six constraint terms
* [docs/data.md](docs/data.md) — data sources, the training set, the dev split
* [docs/training.md](docs/training.md) — the training script, configs and recipe
* [docs/sampling.md](docs/sampling.md) — the sampler, state projections, the placer pipeline
* [docs/refiner.md](docs/refiner.md) — the closed-form refiner
* [docs/legalizer.md](docs/legalizer.md) — the legalizer (`scale_pack`)
* [docs/evaluation.md](docs/evaluation.md) — the evaluation chain and its metrics
* [docs/baselines.md](docs/baselines.md) — regressors, ported published placers, classic solvers

## License

Apache-2.0 ([LICENSE](LICENSE)). Third-party code, data and models used by this repository
keep their own licenses; see [NOTICE](NOTICE).
