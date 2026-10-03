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

* [docs/physics.md](docs/physics.md) — the six constraint terms
* [docs/sampling.md](docs/sampling.md) — the sampler, state projections, the placer pipeline
* [docs/refiner.md](docs/refiner.md) — the closed-form refiner
* [docs/legalizer.md](docs/legalizer.md) — the legalizer (`scale_pack`)

## License

Apache-2.0 ([LICENSE](LICENSE)). Third-party code, data and models used by this repository
keep their own licenses; see [NOTICE](NOTICE).
