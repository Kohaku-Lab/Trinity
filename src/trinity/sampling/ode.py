"""Deterministic probability-flow ODE samplers over block-token latents.

The network emits ``x0`` at every step; the drift is formed from it via
``framing.x0_to_velocity`` and integrated from ``t_start`` (noise) to ``t_end`` (data).
Each numerical method is its own class; the shared loop, velocity and projection live in
``ODESampler`` and subclasses implement only the per-step rule ``_step``.

State projections (``trinity/sampling/projection.py``) are an ordered list applied to
the initial noise, to the state after every step, and to the returned ``x0``. The
default, ``DEFAULT_PROJECTIONS``, writes in the known answers (preplaced positions and
shapes, fixed shapes, the shape of an MIB group that has a known member) and gives every
all-soft MIB group one shared shape. Pass ``projections=[]`` for free sampling.
"""

import torch

from trinity.models import DenoiserCond
from trinity.registry import PROJECTION, SAMPLER, build

DEFAULT_PROJECTIONS = ("anchor_clamp", "mib_group_mean")


class ODESampler:
    """Base probability-flow ODE integrator; subclass and define ``_step``.

    ``framing`` supplies the drift, ``num_steps`` the integration steps from ``t_start``
    to ``t_end``, and ``projections`` a list of ``PROJECTION`` specs applied to the
    state each step (``None`` means ``DEFAULT_PROJECTIONS``, ``[]`` means none).
    """

    def __init__(
        self,
        framing,
        num_steps: int = 50,
        t_start: float = 1.0,
        t_end: float = 1e-3,
        projections: list | tuple | None = None,
    ) -> None:
        self.framing = framing
        self.num_steps = num_steps
        self.t_start = t_start
        self.t_end = t_end
        if projections is None:
            projections = DEFAULT_PROJECTIONS
        self.projections = [build(spec, PROJECTION) for spec in projections]

    def _project(self, z: torch.Tensor, cond: DenoiserCond) -> torch.Tensor:
        """Return ``z`` with every configured projection applied in order."""
        for proj in self.projections:
            z = proj(z, cond)
        return z

    def _velocity(self, model, z, t_scalar, cond):
        t = t_scalar.expand(z.shape[0])
        x0_pred = model(z, t, cond)
        return self.framing.x0_to_velocity(z, x0_pred, t)

    def _step(self, model, z, t0, t1, cond):
        """Advance ``z`` from ``t0`` to ``t1`` by one
        step of the method (subclass implements)."""
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self, model, cond: DenoiserCond, shape, device, noise=None
    ) -> torch.Tensor:
        z = torch.randn(shape, device=device) if noise is None else noise
        z = self._project(z, cond)
        ts = torch.linspace(self.t_start, self.t_end, self.num_steps + 1, device=device)
        for i in range(self.num_steps):
            z = self._project(self._step(model, z, ts[i], ts[i + 1], cond), cond)
        t_last = ts[-1].expand(z.shape[0])
        return self._project(model(z, t_last, cond), cond)


@SAMPLER.register("euler")
class EulerSampler(ODESampler):
    """First-order explicit Euler: ``z + dt * v(z, t0)``."""

    def _step(self, model, z, t0, t1, cond):
        dt = t1 - t0
        return z + dt * self._velocity(model, z, t0, cond)


@SAMPLER.register("heun")
class HeunSampler(ODESampler):
    """Second-order Heun (trapezoidal): average the
    drift at ``t0`` and the Euler-predicted ``t1``."""

    def _step(self, model, z, t0, t1, cond):
        dt = t1 - t0
        v = self._velocity(model, z, t0, cond)
        z_euler = self._project(z + dt * v, cond)
        v2 = self._velocity(model, z_euler, t1, cond)
        return z + dt * 0.5 * (v + v2)
