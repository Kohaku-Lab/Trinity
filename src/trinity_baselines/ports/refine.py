"""The post-hoc refiners of three published placers, ported onto the batched latent.

All three move block positions only (sizes and aspects stay), run with autograd and the
optimizers of their code, and work on each row's own canvas: the raw sample's bounding box
(5 % margin) mapped to ``[-1, 1]²``, sizes in the same units. Preplaced blocks never move but
still collide.

* ``refine_chipdiffusion`` -- the gradient legalizer of ChipDiffusion
  (https://github.com/vint-1/chipdiffusion, ``legalization.py``): softmax-smoothed squared
  penetration weighted by relative mass, a squared die-boundary term and weighted Manhattan
  wirelength; ``scheduled`` = SGD with the schedules of the ``standard`` / ``scheduled``
  configs, ``opt`` = Adam (0.8, 0.99) with the dual-ascent legality weight of ``legalize_opt``.
* ``refine_diffplace`` -- the anchored overlap refinement of DiffPlace
  (https://github.com/HySonLab/DiffPlace, ``deploy_nangate45.py``): gradient steps on
  ``Σ_{i<j} relu(dx)·relu(dy) + w·MSE(x, x0)`` with a linear anchor ramp.
* ``refine_macrodiff`` -- the guidance loop of MacroDiff+ (https://github.com/jhy00n/MacroDiff-plus,
  MIT, ``diffuser.apply_guidance``) run once on the final sample: Adam on
  ``w_hpwl · WA-HPWL + w_overlap · log-sum-exp overlap`` with its phase switch, gradient-norm clip
  and die clamp.

``PORTS`` maps a refiner name to ``fn(z0, case, steps, **kwargs)``. Licenses: NOTICE.
"""

import torch
import torch.nn.functional as F

from trinity.decode import z_to_xywh
from trinity.losses import constraint as C
from trinity.sampling.refine_constraint import RefineCase

CANVAS_MARGIN = 0.05


def _canvas(g0, mask, margin):
    """Per-row canvas centre ``(B,2)`` and half-extent ``(B,2)`` from the raw boxes' bounding box."""
    xmin, ymin, xmax, ymax = C.bbox(g0, mask)
    c0 = torch.stack([(xmin + xmax) / 2, (ymin + ymax) / 2], -1)
    e = torch.stack([(xmax - xmin) / 2, (ymax - ymin) / 2], -1) * (1 + margin)
    return c0, e.clamp_min(1e-6)


def _setup(z0, case, margin=CANVAS_MARGIN):
    """Canvas-space centres ``X``, sizes ``S``, pin positions ``Q``, real mask, movable mask and the canvas."""
    g0 = z_to_xywh(z0, case.area_norm, z0.new_ones(z0.shape[0]))
    c0, e = _canvas(g0, case.token_mask, margin)
    X = (g0[..., :2] + g0[..., 2:] / 2 - c0[:, None]) / e[:, None]
    S = g0[..., 2:] / e[:, None]
    Q = (case.pin_edge_xy_n - c0[:, None]) / e[:, None]
    real = case.token_mask > 0.5
    movable = real & (case.mob_pos > 0.5)
    return X, S, Q, real, movable, c0, e


def _finish(z0, X, c0, e):
    """Latent with the canvas-space centres ``X`` written back; ``rho`` unchanged."""
    c = c0[:, None] + X * e[:, None]
    return torch.cat([c, z0[..., 2:]], dim=-1)


class _Snapshots:
    """Collects ``{step: latent}`` at the requested steps of a loop; ``result`` returns the dict (with
    the final latent under ``steps``) when snapshots were requested, else the final latent.
    """

    def __init__(self, snapshots, steps, z0, c0, e):
        self.want = set(int(s) for s in snapshots)
        self.steps = steps
        self.z0, self.c0, self.e = z0, c0, e
        self.out = {}

    def take(self, step, X):
        if step in self.want and step != self.steps:
            self.out[step] = _finish(self.z0, X.detach().clone(), self.c0, self.e)

    def result(self, final):
        if not self.want:
            return final
        self.out[self.steps] = final
        return self.out


def _wirelength(X, Q, case, real):
    """Weighted Manhattan net length in canvas units ``(B,)`` (b2b edges + per-pin p2b edges)."""
    pair = (real[:, :, None] & real[:, None, :]).to(X.dtype)
    pair = pair * (1 - torch.eye(X.shape[1], device=X.device, dtype=X.dtype))[None]
    d = (X[:, :, None] - X[:, None, :]).abs().sum(-1)
    b2b = 0.5 * (case.adjacency * d * pair).sum((1, 2))
    xb = X.gather(1, case.pin_edge_block[:, :, None].expand(-1, -1, 2))
    p2b = (case.pin_edge_w * (xb - Q).abs().sum(-1)).sum(1)
    return b2b + p2b


def _chipd_legality(X, S, real, softmax_factor):
    """ChipDiffusion legality potential ``(B,)``: mass-weighted softmax-smoothed squared penetration
    over real pairs (the partner detached) plus the squared die-boundary term."""
    delta = (X[:, :, None] - X[:, None, :].detach()).abs() - (
        S[:, :, None] + S[:, None, :]
    ) / 2
    smooth_gap = (F.softmax(delta * softmax_factor, dim=-1) * delta).sum(-1)
    h = F.relu(-smooth_gap) ** 2 / 4
    mass = torch.exp(torch.log(S.clamp_min(1e-12)).mean(-1))
    weight = mass[:, None, :] / (mass[:, :, None] + mass[:, None, :] + 1e-12)
    pair = (real[:, :, None] & real[:, None, :]).to(X.dtype)
    pair = pair * (1 - torch.eye(X.shape[1], device=X.device, dtype=X.dtype))[None]
    h_pair = (h * weight * pair).sum((1, 2))
    h_bound = ((F.relu(X.abs() + S / 2 - 1) ** 2 / 2) * real[..., None]).sum((1, 2))
    return h_pair + h_bound


def _linear_schedule(total, start_idx, end_idx, start_val, end_val):
    """``(total,)`` values: ``start_val``, a linear ramp over ``[start_idx, end_idx)``, then ``end_val``."""
    sched = torch.full((total,), float(start_val))
    if end_idx > start_idx:
        sched[start_idx:end_idx] = torch.linspace(
            start_val, end_val, end_idx - start_idx
        )
    sched[end_idx:] = end_val
    return sched


@torch.enable_grad()
def refine_chipdiffusion(
    z0,
    case: RefineCase,
    steps: int = 5000,
    mode: str = "scheduled",
    step_size: float = 0.2,
    softmax_min: float = 5.0,
    softmax_max: float = 50.0,
    legality_weight: float = 1.0,
    hpwl_weight: float = 4e-5,
    softmax_critical: float = 0.75,
    guidance_critical: float = 0.75,
    zero_hpwl: float = 0.9,
    legality_increase: float = 2.0,
    alpha_init: float = 1.0,
    alpha_lr: float = 1e-3,
    potential_target: float = 1e-4,
    opt_hpwl_weight: float = 1e-5,
    opt_step_size: float = 2e-3,
    opt_softmax_critical: float = 0.1,
    margin: float = CANVAS_MARGIN,
    snapshots: tuple[int, ...] = (),
):
    """ChipDiffusion's gradient legalizer on the raw sample ``z0``; returns the refined latent, or
    ``{step: latent}`` at the steps in ``snapshots`` (0 = the input) plus the final step when given.
    """
    X0, S, Q, real, movable, c0, e = _setup(z0, case, margin)
    X = X0.clone().requires_grad_(True)
    mv = movable[..., None].to(X.dtype)
    snaps = _Snapshots(snapshots, steps, z0, c0, e)
    snaps.take(0, X)
    if mode == "scheduled":
        opt = torch.optim.SGD([X], lr=step_size, momentum=0.0)
        soft = _linear_schedule(
            steps, 0, round(steps * softmax_critical), softmax_min, softmax_max
        )
        leg = _linear_schedule(
            steps,
            round(steps * guidance_critical),
            steps,
            legality_weight,
            legality_weight * legality_increase,
        )
        hp = _linear_schedule(
            steps,
            round(steps * guidance_critical),
            round(steps * zero_hpwl),
            hpwl_weight,
            0.0,
        )
        for i in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = leg[i].item() * _chipd_legality(X, S, real, soft[i].item())
            if hpwl_weight:
                loss = loss + hp[i].item() * _wirelength(X, Q, case, real)
            loss.sum().backward()
            X.grad.mul_(mv)
            opt.step()
            snaps.take(i + 1, X)
    elif mode == "opt":
        alpha = torch.full(
            (z0.shape[0],),
            alpha_init,
            dtype=X.dtype,
            device=X.device,
            requires_grad=True,
        )
        opt = torch.optim.Adam([X], lr=opt_step_size, betas=(0.8, 0.99))
        opt_a = torch.optim.Adam([alpha], lr=alpha_lr, betas=(0.9, 0.99))
        soft = _linear_schedule(
            steps, 0, round(steps * opt_softmax_critical), softmax_min, softmax_max
        )
        for i in range(steps):
            opt.zero_grad(set_to_none=True)
            pot = _chipd_legality(X, S, real, soft[i].item())
            loss = alpha.detach() * pot + opt_hpwl_weight * _wirelength(
                X, Q, case, real
            )
            loss.sum().backward()
            X.grad.mul_(mv)
            opt.step()
            opt_a.zero_grad(set_to_none=True)
            (-alpha * (pot.detach() - potential_target)).sum().backward()
            opt_a.step()
            alpha.data.clamp_(max=15.0)
            snaps.take(i + 1, X)
    else:
        raise ValueError(f"unknown mode {mode!r}: scheduled | opt")
    return snaps.result(_finish(z0, X.detach(), c0, e))


def _diffplace_overlap(X, S, real):
    """``Σ_{i<j} relu(dx)·relu(dy)`` over real pairs ``(B,)`` on centres ``X`` and sizes ``S``."""
    x1, x2 = X[..., 0] - S[..., 0] / 2, X[..., 0] + S[..., 0] / 2
    y1, y2 = X[..., 1] - S[..., 1] / 2, X[..., 1] + S[..., 1] / 2
    dx = F.relu(
        torch.minimum(x2[:, :, None], x2[:, None, :])
        - torch.maximum(x1[:, :, None], x1[:, None, :])
    )
    dy = F.relu(
        torch.minimum(y2[:, :, None], y2[:, None, :])
        - torch.maximum(y1[:, :, None], y1[:, None, :])
    )
    pair = (real[:, :, None] & real[:, None, :]).to(X.dtype)
    return torch.triu(dx * dy * pair, diagonal=1).sum((1, 2))


@torch.enable_grad()
def refine_diffplace(
    z0,
    case: RefineCase,
    steps: int = 500,
    lr: float = 0.04,
    anchor_start: float = 0.01,
    anchor_end: float = 0.30,
    margin: float = CANVAS_MARGIN,
    snapshots: tuple[int, ...] = (),
):
    """DiffPlace's anchored overlap refinement on the raw sample ``z0``; returns the refined latent, or
    ``{step: latent}`` at the steps in ``snapshots`` (0 = the input) plus the final step when given.
    """
    X0, S, _, real, movable, c0, e = _setup(z0, case, margin)
    X = X0.clone()
    mv = movable[..., None].to(X.dtype)
    n_real = real.to(X.dtype).sum(1) * 2
    snaps = _Snapshots(snapshots, steps, z0, c0, e)
    snaps.take(0, X)
    for k in range(steps):
        Xg = X.detach().requires_grad_(True)
        loss_ov = _diffplace_overlap(Xg, S, real)
        loss_anchor = (((Xg - X0) ** 2) * real[..., None]).sum(
            (1, 2)
        ) / n_real.clamp_min(1)
        p = 0.0 if steps <= 1 else k / (steps - 1)
        aw = (1 - p) * anchor_start + p * anchor_end
        g = torch.autograd.grad((loss_ov + aw * loss_anchor).sum(), Xg)[0]
        X = (Xg - lr * g * mv).detach().clamp(-1.0, 1.0)
        snaps.take(k + 1, X)
    return snaps.result(_finish(z0, X, c0, e))


def _wa_length(X, Q, case, real, gamma):
    """Weighted-average (log-sum-exp) net length in canvas units ``(B,)`` over b2b and p2b nets."""

    def wa(a, b):
        hi = torch.maximum(a, b)
        lo = torch.minimum(a, b)
        ea, eb = torch.exp((a - hi) / gamma), torch.exp((b - hi) / gamma)
        plus = (a * ea + b * eb) / (ea + eb)
        fa, fb = torch.exp(-(a - lo) / gamma), torch.exp(-(b - lo) / gamma)
        minus = (a * fa + b * fb) / (fa + fb)
        return plus - minus

    pair = (real[:, :, None] & real[:, None, :]).to(X.dtype)
    pair = pair * (1 - torch.eye(X.shape[1], device=X.device, dtype=X.dtype))[None]
    d = wa(X[:, :, None, 0], X[:, None, :, 0]) + wa(X[:, :, None, 1], X[:, None, :, 1])
    b2b = 0.5 * (case.adjacency * d * pair).sum((1, 2))
    xb = X.gather(1, case.pin_edge_block[:, :, None].expand(-1, -1, 2))
    p2b = (
        case.pin_edge_w * (wa(xb[..., 0], Q[..., 0]) + wa(xb[..., 1], Q[..., 1]))
    ).sum(1)
    return b2b + p2b


def _macrodiff_overlap(P, S, real, gamma):
    """MacroDiff+'s smooth overlap ``(B,)``: log-sum-exp max/min of the edges, clamped, summed over ordered pairs."""

    def smax(a, b):
        m = torch.maximum(a, b)
        return (
            gamma * torch.log(torch.exp((a - m) / gamma) + torch.exp((b - m) / gamma))
            + m
        )

    def smin(a, b):
        m = torch.minimum(a, b)
        return (
            -gamma
            * torch.log(torch.exp((-a + m) / gamma) + torch.exp((-b + m) / gamma))
            + m
        )

    left, right = P[..., 0], P[..., 0] + S[..., 0]
    bottom, top = P[..., 1], P[..., 1] + S[..., 1]
    w = (
        smin(right[:, :, None], right[:, None, :])
        - smax(left[:, :, None], left[:, None, :])
    ).clamp_min(0)
    h = (
        smin(top[:, :, None], top[:, None, :])
        - smax(bottom[:, :, None], bottom[:, None, :])
    ).clamp_min(0)
    pair = (real[:, :, None] & real[:, None, :]).to(P.dtype)
    pair = pair * (1 - torch.eye(P.shape[1], device=P.device, dtype=P.dtype))[None]
    return (w * h * pair).sum((1, 2))


@torch.enable_grad()
def refine_macrodiff(
    z0,
    case: RefineCase,
    steps: int = 500,
    lr: float = 0.01,
    gamma: float = 0.01,
    burn_in: int = 50,
    switch_threshold: float = 1e-5,
    margin: float = CANVAS_MARGIN,
    snapshots: tuple[int, ...] = (),
):
    """MacroDiff+'s guidance loop, run once on the raw sample ``z0``; returns the refined latent, or
    ``{step: latent}`` at the steps in ``snapshots`` (0 = the input) plus the final step when given.
    """
    X0, S, Q, real, movable, c0, e = _setup(z0, case, margin)
    P = (X0 - S / 2).clone().requires_grad_(True)
    mv = movable[..., None].to(P.dtype)
    opt = torch.optim.Adam([P], lr=lr)
    hpwl_phase = torch.ones(z0.shape[0], dtype=torch.bool, device=P.device)
    prev = torch.full((z0.shape[0],), float("inf"), device=P.device, dtype=P.dtype)
    snaps = _Snapshots(snapshots, steps, z0, c0, e)
    snaps.take(0, P + S / 2)
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        X = P + S / 2
        loss_h = _wa_length(X, Q, case, real, gamma)
        loss_o = _macrodiff_overlap(P, S, real, gamma)
        if i > burn_in:
            hpwl_phase = hpwl_phase & ((prev - loss_h.detach()) >= switch_threshold)
        prev = loss_h.detach()
        w_h = torch.where(hpwl_phase, 0.01, 0.001)
        (w_h * loss_h + 10.0 * loss_o).sum().backward()
        P.grad.mul_(mv)
        norm = P.grad.flatten(1).norm(dim=1).clamp_min(1e-6)
        P.grad.mul_((1.0 / norm.clamp_min(1.0))[:, None, None])
        opt.step()
        with torch.no_grad():
            P.data = torch.maximum(
                torch.minimum(P.data, 1.0 - S), torch.full_like(P.data, -1.0)
            )
        snaps.take(i + 1, P + S / 2)
    return snaps.result(_finish(z0, (P + S / 2).detach(), c0, e))


PORTS = {
    "chipd_scheduled": lambda z, c, steps, **kw: refine_chipdiffusion(
        z, c, steps, "scheduled", **kw
    ),
    "chipd_standard": lambda z, c, steps, **kw: refine_chipdiffusion(
        z,
        c,
        steps,
        "scheduled",
        hpwl_weight=0.0,
        softmax_critical=1.0,
        guidance_critical=1.0,
        zero_hpwl=1.0,
        legality_increase=1.0,
        **kw,
    ),
    "chipd_opt": lambda z, c, steps, **kw: refine_chipdiffusion(
        z, c, steps, "opt", **kw
    ),
    "diffplace": lambda z, c, steps, **kw: refine_diffplace(z, c, steps, **kw),
    "macrodiff": lambda z, c, steps, **kw: refine_macrodiff(z, c, steps, **kw),
}
