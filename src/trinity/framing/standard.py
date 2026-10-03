"""The standard diffusion / flow framings, each
with closed-form ``(a, b, c, d, a', b')``.

* ``rectified_flow`` / ``gvp`` -- the linear and
  trigonometric interpolants, velocity target;
* ``ddpm_eps`` / ``ddpm_v`` / ``ddpm_x0`` -- the continuous VP schedule
  (``beta_min``, ``beta_max``) with an epsilon, v or x0 target;
* ``ddpm_cosine_eps`` / ``ddpm_cosine_x0`` -- the cosine schedule with an epsilon or x0
  target.
"""

import math

import torch

from trinity.framing.base import Coeffs, Framing
from trinity.registry import FRAMING


@FRAMING.register("rectified_flow")
class RectifiedFlow(Framing):
    """Linear interpolant ``(1 - t) x0 + t x1``; target ``x1 - x0``."""

    velocity_target = True

    def coeffs(self, t: torch.Tensor) -> Coeffs:
        one = torch.ones_like(t)
        return Coeffs(one - t, t, -one, one, -one, one)


@FRAMING.register("gvp")
class GVP(Framing):
    """Trigonometric (generalized-VP) interpolant + velocity."""

    velocity_target = True

    def coeffs(self, t: torch.Tensor) -> Coeffs:
        half_pi = math.pi / 2
        alpha = torch.cos(half_pi * t)
        sigma = torch.sin(half_pi * t)
        alpha_dot, sigma_dot = -half_pi * sigma, half_pi * alpha
        return Coeffs(alpha, sigma, alpha_dot, sigma_dot, alpha_dot, sigma_dot)


def _vp_alpha_sigma(t: torch.Tensor, beta_min: float, beta_max: float, eps: float):
    """The continuous VP schedule at ``t``: ``(alpha, sigma, alpha_dot, sigma_dot)``."""
    log_abar = -0.5 * (beta_min * t + 0.5 * (beta_max - beta_min) * t * t)
    dlog = -0.5 * (beta_min + (beta_max - beta_min) * t)
    abar = torch.exp(log_abar)
    alpha = torch.sqrt(abar)
    sigma = torch.sqrt((1 - abar).clamp_min(eps))
    return alpha, sigma, 0.5 * alpha * dlog, -0.5 * abar * dlog / sigma


class _VPFraming(Framing):
    """Continuous VP schedule; subclasses pick the target."""

    def __init__(
        self, beta_min: float = 0.1, beta_max: float = 20.0, eps: float = 1e-5
    ) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.eps = eps

    def _schedule(self, t: torch.Tensor):
        return _vp_alpha_sigma(t, self.beta_min, self.beta_max, self.eps)


@FRAMING.register("ddpm_eps")
class EpsFraming(_VPFraming):
    """VP schedule, epsilon target ``x1``."""

    def coeffs(self, t: torch.Tensor) -> Coeffs:
        a, b, a_dot, b_dot = self._schedule(t)
        return Coeffs(a, b, torch.zeros_like(t), torch.ones_like(t), a_dot, b_dot)


@FRAMING.register("ddpm_v")
class VPredFraming(_VPFraming):
    """VP schedule, v target ``a x1 - b x0``."""

    def coeffs(self, t: torch.Tensor) -> Coeffs:
        a, b, a_dot, b_dot = self._schedule(t)
        return Coeffs(a, b, -b, a, a_dot, b_dot)


@FRAMING.register("ddpm_x0")
class X0Framing(_VPFraming):
    """VP schedule, x0 target."""

    def coeffs(self, t: torch.Tensor) -> Coeffs:
        a, b, a_dot, b_dot = self._schedule(t)
        return Coeffs(a, b, torch.ones_like(t), torch.zeros_like(t), a_dot, b_dot)


def _cosine_alpha_sigma(
    t: torch.Tensor, offset: float, abar_min: float, abar_max: float
):
    """The cosine schedule at ``t``: ``(alpha, sigma, alpha_dot, sigma_dot)``.

    ``abar(t) = f(t) / f(0)`` with ``f(s) = cos((s + offset) / (1
    + offset) * pi / 2)^2``, clamped to ``[abar_min, abar_max]``.
    """
    half_pi = math.pi / 2
    scale = half_pi / (1.0 + offset)
    phase = (t + offset) * scale
    phase0 = torch.full_like(t, offset * scale)
    abar = ((torch.cos(phase) ** 2) / (torch.cos(phase0) ** 2)).clamp(
        abar_min, abar_max
    )
    alpha = torch.sqrt(abar)
    sigma = torch.sqrt(1 - abar)
    # d(abar)/dt = -sin(2 phase) * scale / cos(phase0)^2
    abar_dot = -torch.sin(2 * phase) * scale / (torch.cos(phase0) ** 2)
    return alpha, sigma, 0.5 * abar_dot / alpha, -0.5 * abar_dot / sigma


@FRAMING.register("ddpm_cosine_eps")
class CosineEpsFraming(Framing):
    """Cosine schedule, epsilon target ``x1``."""

    def __init__(
        self, offset: float = 0.008, abar_min: float = 1e-5, abar_max: float = 0.9999
    ) -> None:
        self.offset = offset
        self.abar_min = abar_min
        self.abar_max = abar_max

    def coeffs(self, t: torch.Tensor) -> Coeffs:
        a, b, a_dot, b_dot = _cosine_alpha_sigma(
            t, self.offset, self.abar_min, self.abar_max
        )
        return Coeffs(a, b, torch.zeros_like(t), torch.ones_like(t), a_dot, b_dot)


@FRAMING.register("ddpm_cosine_x0")
class CosineX0Framing(Framing):
    """Cosine schedule, x0 target."""

    def __init__(
        self, offset: float = 0.008, abar_min: float = 1e-5, abar_max: float = 0.9999
    ) -> None:
        self.offset = offset
        self.abar_min = abar_min
        self.abar_max = abar_max

    def coeffs(self, t: torch.Tensor) -> Coeffs:
        a, b, a_dot, b_dot = _cosine_alpha_sigma(
            t, self.offset, self.abar_min, self.abar_max
        )
        return Coeffs(a, b, torch.ones_like(t), torch.zeros_like(t), a_dot, b_dot)
