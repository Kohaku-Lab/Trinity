"""Sampling, the sampler-state projections, and the constraint refiners.

Importing this package registers the ODE samplers ``euler`` / ``heun``, the state projections
``anchor_clamp`` / ``mib_group_mean`` (on by default in every sampler), and the refiners
``closed`` (the closed-form refiner of the paper) and ``constraint_latent`` (its autograd
reference).
"""

from trinity.sampling.ode import (
    DEFAULT_PROJECTIONS,
    EulerSampler,
    HeunSampler,
    ODESampler,
)
from trinity.sampling.projection import AnchorClamp, MIBGroupMean, project_mib_group_rho
from trinity.sampling.refine_case import build_refine_case
from trinity.sampling.refine_closed import ClosedFormRefiner, refine_closed
from trinity.sampling.refine_constraint import (
    ConstraintRefiner,
    RefineCase,
    constraint_energy,
    refine_latent,
)

__all__ = [
    "DEFAULT_PROJECTIONS",
    "ODESampler",
    "EulerSampler",
    "HeunSampler",
    "AnchorClamp",
    "MIBGroupMean",
    "project_mib_group_rho",
    "RefineCase",
    "build_refine_case",
    "constraint_energy",
    "refine_latent",
    "refine_closed",
    "ClosedFormRefiner",
    "ConstraintRefiner",
]
