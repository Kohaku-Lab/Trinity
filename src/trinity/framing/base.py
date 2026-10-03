"""Diffusion/flow *framing*: the linear ``(a, b, c, d)`` view of every objective.

Every objective is one per-``t`` linear transform over ``(x0 = data, x1 = noise)``::

    x_t    = a(t) * x0 + b(t) * x1
    target = c(t) * x0 + d(t) * x1

A concrete framing implements only :meth:`Framing.coeffs`; ``x_t``, the regression target,
the recovery of ``(x0, x1)`` from a prediction and the sampling velocity are derived here.

The denoiser emits ``x0`` (or, with ``output_kind="target"``, the framing target converted
back by :meth:`Framing.x0_from_target`). :meth:`Framing.pred_to_target` maps a predicted
``x0`` into the framing's target space for the loss; :meth:`Framing.x0_to_velocity` gives
the ODE drift from a predicted ``x0``.
"""

from dataclasses import dataclass

import torch


@dataclass
class Coeffs:
    """Per-sample coefficients (each shape ``(B,)``); ``a_dot``/``b_dot`` optional."""

    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    d: torch.Tensor
    a_dot: torch.Tensor | None = None
    b_dot: torch.Tensor | None = None


def _floor_abs(x: torch.Tensor, eps: float) -> torch.Tensor:
    """``x`` with its magnitude floored at ``eps``, keeping its sign (0 -> +eps)."""
    return torch.where(x >= 0, x.clamp_min(eps), x.clamp_max(-eps))


def _bcast(coef: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Reshape a ``(B,)`` coefficient to broadcast against ``ref`` (``(B, ...)``)."""
    return coef.reshape(coef.shape[0], *([1] * (ref.ndim - 1)))


class Framing:
    """Base framing. Subclasses implement :meth:`coeffs`; the rest is derived."""

    # True when the regression target IS the ODE velocity (flow framings / v-target).
    velocity_target: bool = False

    def coeffs(self, t: torch.Tensor) -> Coeffs:
        raise NotImplementedError

    def loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Per-sample loss weight ``w(t)`` (default: uniform)."""
        return torch.ones_like(t)

    def x_t(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The noised state ``a x0 + b x1``."""
        co = self.coeffs(t)
        return _bcast(co.a, x0) * x0 + _bcast(co.b, x0) * x1

    def target(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """The regression target ``c*x0 + d*x1`` the loss compares against."""
        co = self.coeffs(t)
        return _bcast(co.c, x0) * x0 + _bcast(co.d, x0) * x1

    def prepare(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        """Return ``(x_t, target)`` for a training step (single ``coeffs`` call)."""
        co = self.coeffs(t)
        a, b = _bcast(co.a, x0), _bcast(co.b, x0)
        c, d = _bcast(co.c, x0), _bcast(co.d, x0)
        return a * x0 + b * x1, c * x0 + d * x1

    def pred_to_x0(
        self, x_t: torch.Tensor, x0_pred: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """``x0_pred`` itself (the network emits ``x0``)."""
        return x0_pred

    def pred_to_target(
        self, x_t: torch.Tensor, x0_pred: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Map the network's ``x0`` output into the framing's regression-target space.

        Given ``x_t = a*x0 + b*x1`` and the predicted ``x0``, recover ``x1`` and form
        ``target = c*x0 + d*x1``. For an x0-target framing this returns ``x0_pred``.
        """
        co = self.coeffs(t)
        a, b = _bcast(co.a, x_t), _bcast(co.b, x_t)
        c, d = _bcast(co.c, x_t), _bcast(co.d, x_t)
        x1 = (x_t - a * x0_pred) / b
        return c * x0_pred + d * x1

    def x0_from_target(
        self,
        x_t: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """The ``x0`` implied by a target-space prediction (inverse of :meth:`pred_to_target`).

        The 2x2 solve of :meth:`recover` with ``|det|`` floored at ``eps``.
        """
        co = self.coeffs(t)
        a, b = _bcast(co.a, x_t), _bcast(co.b, x_t)
        c, d = _bcast(co.c, x_t), _bcast(co.d, x_t)
        return (d * x_t - b * target) / _floor_abs(a * d - b * c, eps)

    def target_from_x0(
        self, x_t: torch.Tensor, x0: torch.Tensor, t: torch.Tensor, eps: float = 1e-4
    ) -> torch.Tensor:
        """The target-space value implied by ``x0``: :meth:`pred_to_target` with ``|b|`` floored."""
        co = self.coeffs(t)
        a, b = _bcast(co.a, x_t), _bcast(co.b, x_t)
        c, d = _bcast(co.c, x_t), _bcast(co.d, x_t)
        x1 = (x_t - a * x0) / _floor_abs(b, eps)
        return c * x0 + d * x1

    def recover(self, x_t: torch.Tensor, pred: torch.Tensor, t: torch.Tensor):
        """Solve the 2x2 system for ``(x0, x1)`` given ``(x_t, pred == target)``.

        Well-posed iff ``det = a*d - b*c != 0``.
        """
        co = self.coeffs(t)
        a, b = _bcast(co.a, x_t), _bcast(co.b, x_t)
        c, d = _bcast(co.c, x_t), _bcast(co.d, x_t)
        det = a * d - b * c
        x0 = (d * x_t - b * pred) / det
        x1 = (a * pred - c * x_t) / det
        return x0, x1

    def to_velocity(
        self, x_t: torch.Tensor, pred: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Probability-flow ODE drift ``dx_t/dt = a' x0 + b' x1`` from a *target* pred."""
        co = self.coeffs(t)
        if co.a_dot is None or co.b_dot is None:
            raise ValueError("framing has no a_dot/b_dot; cannot form an ODE velocity")
        x0, x1 = self.recover(x_t, pred, t)
        return _bcast(co.a_dot, x_t) * x0 + _bcast(co.b_dot, x_t) * x1

    def x0_to_velocity(
        self, x_t: torch.Tensor, x0_pred: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """The probability-flow ODE drift from a predicted ``x0``."""
        co = self.coeffs(t)
        if co.a_dot is None or co.b_dot is None:
            raise ValueError("framing has no a_dot/b_dot; cannot form an ODE velocity")
        a, b = _bcast(co.a, x_t), _bcast(co.b, x_t)
        x1 = (x_t - a * x0_pred) / b
        return _bcast(co.a_dot, x_t) * x0_pred + _bcast(co.b_dot, x_t) * x1
