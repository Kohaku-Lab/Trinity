"""A framing built from user-supplied coefficient functions ``a, b, c, d`` (and ``a', b'``)."""

from collections.abc import Callable

import torch

from trinity.framing.base import Coeffs, Framing

CoeffFn = Callable[[torch.Tensor], torch.Tensor]


class GeneralFraming(Framing):
    """Framing built from user-supplied coefficient functions of ``t``.

    ``x_t = a(t)*x0 + b(t)*x1``, ``target = c(t)*x0 + d(t)*x1``; ``a_dot`` / ``b_dot`` are
    optional but required for ODE sampling.
    """

    def __init__(
        self,
        a: CoeffFn,
        b: CoeffFn,
        c: CoeffFn,
        d: CoeffFn,
        a_dot: CoeffFn | None = None,
        b_dot: CoeffFn | None = None,
        velocity_target: bool = False,
    ) -> None:
        self.a, self.b, self.c, self.d = a, b, c, d
        self.a_dot, self.b_dot = a_dot, b_dot
        self.velocity_target = velocity_target

    def coeffs(self, t: torch.Tensor) -> Coeffs:
        return Coeffs(
            self.a(t),
            self.b(t),
            self.c(t),
            self.d(t),
            None if self.a_dot is None else self.a_dot(t),
            None if self.b_dot is None else self.b_dot(t),
        )
