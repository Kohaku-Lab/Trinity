"""In-sampler guidance of three published diffusion placers, ported onto the batched latent, and
the guided Euler sampler (``SAMPLER`` ``euler_guided``) that applies one of them at every step.

Every guidance sees the state ``z_t``, the network's predicted ``x0`` (both in the ``/s`` latent
``(cx, cy, rho)``), the step's times and the batch's :class:`RefineCase`; it moves block positions
only (``rho`` stays), never moves preplaced blocks, and works on each row's canvas as the post-hoc
ports do (the predicted ``x0``'s bounding box with a 5 % margin mapped to ``[-1, 1]^2``, see
``trinity_baselines.ports.refine._setup``). The final ``x0`` is guided as well.

* ``chipd_opt`` -- ChipDiffusion's default evaluation guidance (https://github.com/vint-1/chipdiffusion,
  ``guidance/opt.yaml``, ``ContinuousDiffusionModel.reverse_guidance_opt_force``): per step, 10 Adam steps (lr 8e-3,
  betas (0.8, 0.99)) on ``alpha * legality + 1e-4 * wirelength`` from the predicted ``x0``; the
  per-row dual weight ``alpha`` starts at 0, persists across steps, and takes an Adam step (lr 5e-4,
  betas (0.9, 0.99)) toward the potential target 1e-4 while ``t < 0.5`` (clipped at 15); legality
  softmax factor 20 -> 10 over ``t in [0.1, 1]``. The shift ``g`` enters as ``a(t_next) * g`` on the
  next state (the scheduler keeps its noise prediction).
* ``diffplace`` -- DiffPlace's deploy overlap guidance (https://github.com/HySonLab/DiffPlace,
  ``sample_ddim``, ``deploy_nangate45.py``):
  from ``1 - t >= 0.85``, ``eps <- eps - lambda_t * g`` with ``lambda_t = 200 (1 - t)^2`` and ``g`` the
  gradient of the pairwise overlap of the movable blocks at the noisy state, rescaled per block to
  norm 0.03 and clipped at 0.08; the step continues from the ``x0`` implied by the guided ``eps``.
* ``macrodiff`` -- MacroDiff+'s ``apply_guidance`` (https://github.com/jhy00n/MacroDiff-plus, MIT)
  on the predicted ``x0`` at every step (``refine_macrodiff``: 500 Adam iterations, lr 0.01, the
  HPWL / overlap phases, gradient-norm clip, die clamp); the step continues from the guided ``x0``.

Licenses: NOTICE.
"""

import torch

from trinity.framing.base import _bcast
from trinity.registry import SAMPLER
from trinity.sampling.ode import EulerSampler
from trinity_baselines.ports.refine import (
    CANVAS_MARGIN,
    _chipd_legality,
    _diffplace_overlap,
    _setup,
    _wirelength,
    refine_macrodiff,
)


def _euler(framing, z, x0, t0, t1):
    """One Euler step of the probability-flow ODE from ``z`` at ``t0`` to ``t1`` with the given ``x0``."""
    return z + (t1 - t0) * framing.x0_to_velocity(z, x0, t0.expand(z.shape[0]))


def _position_delta(dX, e):
    """Latent shift ``(B, N, 3)`` from a canvas-space position shift ``dX`` (``rho`` unchanged)."""
    return torch.cat([dX * e[:, None], torch.zeros_like(dX[..., :1])], dim=-1)


class ChipdOptGuidance:
    """ChipDiffusion's ``opt`` guidance with its per-row dual weight kept across the steps."""

    def __init__(
        self,
        steps: int = 10,
        lr: float = 8e-3,
        hpwl_weight: float = 1e-4,
        alpha_init: float = 0.0,
        alpha_lr: float = 5e-4,
        alpha_critical: float = 0.5,
        potential_target: float = 1e-4,
        softmax_min: float = 10.0,
        softmax_max: float = 20.0,
        softmax_critical: float = 0.1,
        margin: float = CANVAS_MARGIN,
    ) -> None:
        self.steps, self.lr, self.hpwl_weight = steps, lr, hpwl_weight
        self.alpha_init, self.alpha_lr, self.alpha_critical, self.target = (
            alpha_init,
            alpha_lr,
            alpha_critical,
            potential_target,
        )
        self.softmax_min, self.softmax_max, self.softmax_critical = (
            softmax_min,
            softmax_max,
            softmax_critical,
        )
        self.margin = margin

    def reset(self, z):
        self.alpha = torch.full(
            (z.shape[0],),
            self.alpha_init,
            device=z.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        self.opt_alpha = torch.optim.Adam(
            [self.alpha], lr=self.alpha_lr, betas=(0.9, 0.99)
        )

    def _softmax_factor(self, t: float) -> float:
        if t <= self.softmax_critical:
            return self.softmax_max
        tp = (t - self.softmax_critical) / (1 - self.softmax_critical)
        return self.softmax_max - tp * (self.softmax_max - self.softmax_min)

    @torch.enable_grad()
    def shift(self, x0, t: float, case):
        """The guidance shift ``g`` of ``x0`` in latent units."""
        X0, S, Q, real, movable, _, e = _setup(x0.float(), case, self.margin)
        X = X0.clone().requires_grad_(True)
        mv = movable[..., None].to(X.dtype)
        opt = torch.optim.Adam([X], lr=self.lr, betas=(0.8, 0.99))
        sf = self._softmax_factor(t)
        for _ in range(self.steps):
            opt.zero_grad(set_to_none=True)
            pot = _chipd_legality(X, S, real, sf)
            (
                self.alpha.detach() * pot
                + self.hpwl_weight * _wirelength(X, Q, case, real)
            ).sum().backward()
            X.grad.mul_(mv)
            opt.step()
            if t < self.alpha_critical:
                self.opt_alpha.zero_grad(set_to_none=True)
                (-self.alpha * (pot.detach() - self.target)).sum().backward()
                self.opt_alpha.step()
                self.alpha.data.clamp_(max=15.0)
        return _position_delta((X - X0).detach(), e).to(x0.dtype)

    def step(self, framing, z, x0, t0, t1, case):
        g = self.shift(x0, float(t0), case)
        a1 = _bcast(framing.coeffs(t1.expand(z.shape[0])).a, z)
        return _euler(framing, z, x0, t0, t1) + a1 * g

    def final(self, framing, z, x0, t, case):
        return x0 + self.shift(x0, float(t), case)


class DiffPlaceGuidance:
    """DiffPlace's deploy overlap guidance on the noise prediction over the last part of sampling."""

    def __init__(
        self,
        scale_max: float = 200.0,
        power: float = 2.0,
        start_frac: float = 0.85,
        grad_norm: float = 0.03,
        grad_clip: float = 0.08,
        sign: float = 1.0,
        margin: float = CANVAS_MARGIN,
    ) -> None:
        self.scale_max, self.power, self.start_frac = scale_max, power, start_frac
        self.grad_norm, self.grad_clip, self.margin = grad_norm, grad_clip, margin
        # 1 = the released update eps - lambda * g; -1 = eps + lambda * g.
        self.sign = sign

    def reset(self, z):
        pass

    @torch.enable_grad()
    def _guided_x0(self, framing, z, x0, t: torch.Tensor, case):
        frac = 1.0 - float(t)
        if frac < self.start_frac:
            return x0
        lam = self.scale_max * frac**self.power
        _, S, _, _, movable, c0, e = _setup(x0.float(), case, self.margin)
        Xt = ((z[..., :2].float() - c0[:, None]) / e[:, None]).requires_grad_(True)
        g = torch.autograd.grad(_diffplace_overlap(Xt, S, movable).sum(), Xt)[0]
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        gn = g.norm(dim=-1, keepdim=True)
        g = (
            g
            * (self.grad_norm / (gn + 1e-12))
            * torch.clamp(self.grad_clip / (gn + 1e-12), max=1.0)
        )
        g = g * movable[..., None].to(g.dtype)
        co = framing.coeffs(t.expand(z.shape[0]))
        a, b = _bcast(co.a, z), _bcast(co.b, z)
        eps = (z - a * x0) / b
        eps = eps - self.sign * lam * _position_delta(g, e).to(z.dtype)
        return (z - b * eps) / a

    def step(self, framing, z, x0, t0, t1, case):
        return _euler(framing, z, self._guided_x0(framing, z, x0, t0, case), t0, t1)

    def final(self, framing, z, x0, t, case):
        return self._guided_x0(framing, z, x0, t, case)


class MacroDiffGuidance:
    """MacroDiff+'s guidance loop on the predicted ``x0`` at every step."""

    def __init__(
        self, iterations: int = 500, lr: float = 0.01, margin: float = CANVAS_MARGIN
    ) -> None:
        self.iterations, self.lr, self.margin = iterations, lr, margin

    def reset(self, z):
        pass

    def _guided(self, x0, case):
        return refine_macrodiff(
            x0.float(), case, self.iterations, lr=self.lr, margin=self.margin
        ).to(x0.dtype)

    def step(self, framing, z, x0, t0, t1, case):
        return _euler(framing, z, self._guided(x0, case), t0, t1)

    def final(self, framing, z, x0, t, case):
        return self._guided(x0, case)


GUIDANCES = {
    "chipd_opt": ChipdOptGuidance,
    "diffplace": DiffPlaceGuidance,
    "macrodiff": MacroDiffGuidance,
}


@SAMPLER.register("euler_guided")
class GuidedEulerSampler(EulerSampler):
    """Euler sampler with one published placer's in-sampler guidance.

    ``guidance`` names a ``GUIDANCES`` entry and ``guidance_kwargs`` its settings; set ``case``
    (the ``RefineCase`` of the batch's rows) before ``sample``.
    """

    def __init__(
        self,
        framing,
        num_steps: int = 50,
        guidance: str = "chipd_opt",
        guidance_kwargs: dict | None = None,
        **kw,
    ) -> None:
        super().__init__(framing, num_steps=num_steps, **kw)
        self.guidance = GUIDANCES[guidance](**(guidance_kwargs or {}))
        self.case = None

    @torch.no_grad()
    def sample(self, model, cond, shape, device, noise=None) -> torch.Tensor:
        if self.case is None:
            raise ValueError(
                "GuidedEulerSampler needs .case set to the batch's RefineCase"
            )
        z = torch.randn(shape, device=device) if noise is None else noise
        z = self._project(z, cond)
        self.guidance.reset(z)
        ts = torch.linspace(self.t_start, self.t_end, self.num_steps + 1, device=device)
        for i in range(self.num_steps):
            x0 = model(z, ts[i].expand(z.shape[0]), cond)
            z = self._project(
                self.guidance.step(self.framing, z, x0, ts[i], ts[i + 1], self.case),
                cond,
            )
        x0 = model(z, ts[-1].expand(z.shape[0]), cond)
        return self._project(
            self.guidance.final(self.framing, z, x0, ts[-1], self.case), cond
        )
