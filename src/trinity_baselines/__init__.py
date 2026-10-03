"""The baselines of the Trinity paper, kept apart from the method package.

* ``trinity_baselines.models`` / ``solver`` / ``training`` -- the direct-regression
  baselines (one deterministic forward instead of an ODE sample): ``head="z"`` predicts
  the latent ``(cx/s, cy/s, rho)``, ``head="xywh"`` the raw box ``(cx/s, cy/s, w/s,
  h/s)``.
* ``trinity_baselines.ported`` -- the backbones of the published learned placers
  (FlowPlace, ChipDiffusion, DiffPlace, MacroDiff+) behind the Trinity training
  interface.
* ``trinity_baselines.ports`` -- their published
  refiners and in-sampler guidance, ported.
* ``trinity_baselines.classical`` -- the classic solvers: the PARSAC driver,
  sequence-pair simulated annealing, and the shared annealing schedule.

Importing this package populates the baseline
registries (``HEAD``, ``BASELINE_BACKBONE``).
"""

from trinity_baselines import (
    losses,  # noqa: F401  (re-export: shared LOSS registry)
    models,  # noqa: F401  (register: z / xywh heads)
    ported,  # noqa: F401  (register: baseline backbones)
)
from trinity_baselines.losses import LOSS
from trinity_baselines.models import (
    BoxHead,
    DirectRegressor,
    LatentHead,
    RegressionHead,
)
from trinity_baselines.registry import BASELINE_BACKBONE, HEAD, Registry, build, resolve
from trinity_baselines.solver import RegressionPlacer

__all__ = [
    "build",
    "resolve",
    "Registry",
    "HEAD",
    "BASELINE_BACKBONE",
    "LOSS",
    "DirectRegressor",
    "RegressionHead",
    "LatentHead",
    "BoxHead",
    "RegressionPlacer",
    "models",
    "losses",
]
