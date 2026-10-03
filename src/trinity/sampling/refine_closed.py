"""The closed-form constraint refiner: the aux terms descended with hand-written gradients.

The energy is ``refine_constraint.constraint_energy`` (the six terms of
``losses/constraint.py``, plus ``outline``) on the latent ``z = (cx/s, cy/s, rho)`` of a
finished sample. Gradient path per step: decode ``z`` to boxes, accumulate every term's
derivative with respect to the four box edges ``(L, B, R, T)``, convert to center + size,
zero the frozen channels (``mob_pos`` / ``mob_shape``), map through the decode Jacobian to
``dE/dz`` and zero the padding.

Per-term edge derivatives (normalized units, ``s^2 = sum area``, ``s`` its root):

* overlap  -- pair ``(i, j)`` with ``ox * oy > 0``: ``dE/dR_i = oy [R_i < R_j] / s^2``,
  ``dE/dL_i = -oy [L_i > L_j] / s^2``; ``y`` likewise with ``ox``.
* group    -- every edge of the Prim tree with gap ``max(gx, gy) > 0``: on the active axis,
  ``dE/dL_i = [L_i > L_j] / s``, ``dE/dR_i = -[R_i < R_j] / s`` (both endpoints).
* mib      -- member ``i`` of a group of ``> 1``: ``dE/drho_i = sign(rho_i - median)``.
* boundary -- coded block ``i``: ``dE/dL_i = [L] / s`` and ``dE/dL_{argmin L} -= [L] / s``;
  right / top / bottom symmetric.
* wl       -- ``dE/dcx_i = [sum_j a_ij sign(cx_i - cx_j) + sum_{p->i} w_p sign(cx_i - q_p)] / (s W)``.
* area     -- ``dE/dL_{argmin L} = -H_bb / s^2``, ``dE/dR_{argmax R} = H_bb / s^2``; ``y`` with ``W_bb``.
* outline  -- ``dE/dL_{argmin L} = -[W_bb > W] / s``, ``dE/dR_{argmax R} = [W_bb > W] / s``;
  ``y`` with ``H`` (zero for a case without an outline).

The descent is Adam (``betas = (0, beta2)`` by default) or plain gradient steps, with a
per-step lr vector from an AnySchedule config, anchored coordinates re-clamped after every
step, and every ``chunk`` steps under one ``torch.compile``.
"""

import torch
from anyschedule.utils import get_scheduler

from trinity.decode import RHO_CLAMP, z_to_xywh
from trinity.losses import constraint as C
from trinity.registry import REFINER
from trinity.sampling.refine_constraint import RefineCase

TERM_NAMES = ("overlap", "group", "mib", "boundary", "wl", "area")
# The refiner weights of the paper: overlap 10, the other five terms 1.
PAPER_WEIGHTS = {
    "overlap": 10.0,
    "group": 1.0,
    "mib": 1.0,
    "boundary": 1.0,
    "wl": 1.0,
    "area": 1.0,
}
_INF = float("inf")


def _bbox_arg(g, mask):
    """The bbox ``(xmin, ymin, xmax, ymax)`` over real blocks and the blocks attaining it."""
    x, y, xr, yt = C._edges(g)
    m = mask > 0.5
    big = 1e9
    xmin = torch.where(m, x, torch.full_like(x, big)).min(1)
    ymin = torch.where(m, y, torch.full_like(y, big)).min(1)
    xmax = torch.where(m, xr, torch.full_like(xr, -big)).max(1)
    ymax = torch.where(m, yt, torch.full_like(yt, -big)).max(1)
    values = (xmin.values, ymin.values, xmax.values, ymax.values)
    indices = (xmin.indices, ymin.indices, xmax.indices, ymax.indices)
    return values, indices


def _overlap_grad(g, pair, s2, margin=0.0):
    """Edge gradients of the overlap term (each box inflated by ``margin``)."""
    x, y, xr, yt = C._edges(g)
    x, y, xr, yt = x - margin / 2, y - margin / 2, xr + margin / 2, yt + margin / 2
    ox = C._pair_overlap(x, xr)
    oy = C._pair_overlap(y, yt)
    act = pair * (ox > 0) * (oy > 0)
    inv = 1.0 / s2[:, None]
    gR = (act * oy * (xr[:, :, None] < xr[:, None, :])).sum(2) * inv
    gL = -(act * oy * (x[:, :, None] > x[:, None, :])).sum(2) * inv
    gT = (act * ox * (yt[:, :, None] < yt[:, None, :])).sum(2) * inv
    gB = -(act * ox * (y[:, :, None] > y[:, None, :])).sum(2) * inv
    return gL, gB, gR, gT


def _mst_edges(gap, cluster_id, mask):
    """``(B, N, N)`` bool: the edges Prim picks in ``cluster_mst_gap``.

    Entry ``[i, j]`` is set when block ``i`` joins the tree through tree block ``j``.
    """
    b, n = cluster_id.shape
    member = (mask > 0.5) & (cluster_id > 0)
    same = (
        (cluster_id[:, :, None] == cluster_id[:, None, :])
        & member[:, :, None]
        & member[:, None, :]
    )
    same = same & ~torch.eye(n, dtype=torch.bool, device=gap.device)[None]
    idx = torch.arange(n, device=gap.device)[None].expand(b, n)
    inf = torch.full_like(gap, _INF)
    big = torch.full((b, n + 1), n, device=gap.device, dtype=torch.long)
    first = big.scatter_reduce(1, cluster_id, torch.where(member, idx, n), "amin")
    in_tree = member & (idx == first.gather(1, cluster_id))
    edges = torch.zeros_like(same)
    for _ in range(C.MAX_GROUP_SIZE - 1):
        allowed = same & in_tree[:, None, :] & ~in_tree[:, :, None]
        cand, arg = torch.where(allowed, gap, inf).min(2)
        has = torch.isfinite(cand)
        cmin = torch.full((b, n + 1), _INF, device=gap.device, dtype=gap.dtype)
        cmin = cmin.scatter_reduce(1, cluster_id, cand, "amin")
        is_min = has & (cand <= cmin.gather(1, cluster_id))
        first_min = big.scatter_reduce(
            1, cluster_id, torch.where(is_min, idx, n), "amin"
        )
        chosen = is_min & (idx == first_min.gather(1, cluster_id))
        edges = edges | (chosen[:, :, None] & (arg[:, :, None] == idx[:, None, :]))
        in_tree = in_tree | chosen
    return edges


def _group_grad(g, cluster_id, mask, s, squared=False):
    """Edge gradients of the group term (``squared``: of the squared gaps)."""
    x, y, xr, yt = C._edges(g)
    gx = C._pair_gap(x, xr)
    gy = C._pair_gap(y, yt)
    gap = torch.maximum(gx, gy)
    edges = _mst_edges(gap, cluster_id, mask)
    sym = (edges | edges.transpose(1, 2)).to(g.dtype)
    if squared:
        sym = sym * 2 * gap
    x_active = sym * (gx > gy) * (gx > 0)
    y_active = sym * (gy >= gx) * (gy > 0)
    inv = 1.0 / s[:, None]
    gL = (x_active * (x[:, :, None] > x[:, None, :])).sum(2) * inv
    gR = -(x_active * (xr[:, :, None] < xr[:, None, :])).sum(2) * inv
    gB = (y_active * (y[:, :, None] > y[:, None, :])).sum(2) * inv
    gT = -(y_active * (yt[:, :, None] < yt[:, None, :])).sum(2) * inv
    return gL, gB, gR, gT


def _mib_sign(g, mib_id, mask):
    """``(B, N)`` ``sign(rho_i - group median rho)`` for members of MIB groups of ``> 1``."""
    rho = torch.log(g[..., 2].clamp_min(C._EPS)) - torch.log(
        g[..., 3].clamp_min(C._EPS)
    )
    b, n = mib_id.shape
    member = (mask > 0.5) & (mib_id > 0)
    gid = torch.arange(n + 1, device=g.device)[None, :, None]
    onehot = member[:, None, :] & (mib_id[:, None, :] == gid)
    expanded = rho[:, None, :].expand(b, n + 1, n)
    masked = torch.where(onehot, expanded, torch.full_like(expanded, _INF))
    ordered = masked.sort(dim=2).values
    cnt = onehot.sum(2)
    lo = ((cnt - 1) // 2).clamp_min(0)
    hi = (cnt // 2).clamp_min(0)
    med = 0.5 * (ordered.gather(2, lo[:, :, None]) + ordered.gather(2, hi[:, :, None]))
    med = med.squeeze(2)
    in_group = member & (cnt.gather(1, mib_id) > 1)
    sign = torch.sign(rho - med.gather(1, mib_id))
    return torch.where(in_group, sign, torch.zeros_like(rho))


def _mib_grad(g, mib_id, mask):
    """Edge gradients of the mib term."""
    sgn = _mib_sign(g, mib_id, mask)
    w, h = g[..., 2].clamp_min(C._EPS), g[..., 3].clamp_min(C._EPS)
    return -sgn / w, sgn / h, sgn / w, -sgn / h


def _boundary_grad(g, boundary_code, mask, s):
    """Edge gradients of the boundary term."""
    code = boundary_code.long()
    m = (mask > 0.5).to(g.dtype)
    inv = 1.0 / s[:, None]
    left = (code & 1 > 0).to(g.dtype) * m
    right = (code & 2 > 0).to(g.dtype) * m
    top = (code & 4 > 0).to(g.dtype) * m
    bottom = (code & 8 > 0).to(g.dtype) * m
    _, (i_xmin, i_ymin, i_xmax, i_ymax) = _bbox_arg(g, mask)
    gL = left * inv
    gR = -right * inv
    gT = -top * inv
    gB = bottom * inv
    gL = gL.scatter_add(1, i_xmin[:, None], -(left.sum(1) * inv[:, 0])[:, None])
    gR = gR.scatter_add(1, i_xmax[:, None], (right.sum(1) * inv[:, 0])[:, None])
    gT = gT.scatter_add(1, i_ymax[:, None], (top.sum(1) * inv[:, 0])[:, None])
    gB = gB.scatter_add(1, i_ymin[:, None], -(bottom.sum(1) * inv[:, 0])[:, None])
    return gL, gB, gR, gT


def _wl_grad(g, pair, adjacency, pin_xy, pin_w, pin_block, s):
    """Edge gradients of the wl term (half of each center gradient on either edge)."""
    c = g[..., :2] + g[..., 2:] / 2
    a = 0.5 * (adjacency + adjacency.transpose(1, 2)) * pair
    dcx = (a * torch.sign(c[:, :, None, 0] - c[:, None, :, 0])).sum(2)
    dcy = (a * torch.sign(c[:, :, None, 1] - c[:, None, :, 1])).sum(2)
    cb = c.gather(1, pin_block[:, :, None].expand(-1, -1, 2))
    pull = pin_w[:, :, None] * torch.sign(cb - pin_xy)
    dcx = dcx.scatter_add(1, pin_block, pull[..., 0])
    dcy = dcy.scatter_add(1, pin_block, pull[..., 1])
    total_w = (0.5 * a.sum((1, 2)) + pin_w.sum(1)).clamp_min(C._EPS)
    inv = 1.0 / (s * total_w)[:, None]
    gcx, gcy = dcx * inv, dcy * inv
    return gcx / 2, gcy / 2, gcx / 2, gcy / 2


def _area_grad(g, mask, s2):
    """Edge gradients of the area term (on the four bbox-attaining blocks)."""
    (xmin, ymin, xmax, ymax), (i_xmin, i_ymin, i_xmax, i_ymax) = _bbox_arg(g, mask)
    wb, hb = (xmax - xmin) / s2, (ymax - ymin) / s2
    zero = torch.zeros_like(g[..., 0])
    gL = zero.scatter(1, i_xmin[:, None], -hb[:, None])
    gR = zero.scatter(1, i_xmax[:, None], hb[:, None])
    gB = zero.scatter(1, i_ymin[:, None], -wb[:, None])
    gT = zero.scatter(1, i_ymax[:, None], wb[:, None])
    return gL, gB, gR, gT


def _outline_grad(g, mask, outline, s):
    """Edge gradients of the outline term (on the four bbox-attaining blocks)."""
    (xmin, ymin, xmax, ymax), (i_xmin, i_ymin, i_xmax, i_ymax) = _bbox_arg(g, mask)
    ax = ((xmax - xmin) > outline[:, 0]).to(g.dtype) / s
    ay = ((ymax - ymin) > outline[:, 1]).to(g.dtype) / s
    zero = torch.zeros_like(g[..., 0])
    gL = zero.scatter(1, i_xmin[:, None], -ax[:, None])
    gR = zero.scatter(1, i_xmax[:, None], ax[:, None])
    gB = zero.scatter(1, i_ymin[:, None], -ay[:, None])
    gT = zero.scatter(1, i_ymax[:, None], ay[:, None])
    return gL, gB, gR, gT


def edge_grads(g, case: RefineCase, weights: dict[str, float]):
    """``(gL, gB, gR, gT)``, each ``(B, N)``: the weighted energy's box-edge derivatives.

    ``weights`` may also carry ``overlap_margin``, ``group_squared`` and ``outline`` (the
    latter active only when the case has an outline).
    """
    pair = C._real_pairs(case.token_mask)
    s2, s = C._scales(case.area_norm, case.token_mask)
    acc = [torch.zeros_like(g[..., 0]) for _ in range(4)]

    def add(w, parts):
        for k in range(4):
            acc[k] = acc[k] + w * parts[k]

    if weights.get("overlap"):
        margin = weights.get("overlap_margin", 0.0)
        add(weights["overlap"], _overlap_grad(g, pair, s2, margin))
    if weights.get("group"):
        squared = bool(weights.get("group_squared", False))
        add(
            weights["group"],
            _group_grad(g, case.cluster_id, case.token_mask, s, squared),
        )
    if weights.get("mib"):
        add(weights["mib"], _mib_grad(g, case.mib_id, case.token_mask))
    if weights.get("boundary"):
        add(
            weights["boundary"],
            _boundary_grad(g, case.boundary_code, case.token_mask, s),
        )
    if weights.get("wl"):
        parts = _wl_grad(
            g,
            pair,
            case.adjacency,
            case.pin_edge_xy_n,
            case.pin_edge_w,
            case.pin_edge_block,
            s,
        )
        add(weights["wl"], parts)
    if weights.get("area"):
        add(weights["area"], _area_grad(g, case.token_mask, s2))
    if weights.get("outline") and case.outline is not None:
        add(weights["outline"], _outline_grad(g, case.token_mask, case.outline, s))
    return acc


def latent_grad(
    z: torch.Tensor, case: RefineCase, weights: dict[str, float], shape: bool = True
) -> torch.Tensor:
    """``dE/dz`` ``(B, N, 3)`` of the weighted energy, frozen channels and padding zeroed.

    ``shape=False`` also zeroes the ``rho`` channel (positions-only descent).
    """
    ones = z.new_ones(z.shape[0])
    g = z_to_xywh(z, case.area_norm, ones)
    gL, gB, gR, gT = edge_grads(g, case, weights)
    gcx = (gL + gR) * case.mob_pos
    gcy = (gB + gT) * case.mob_pos
    gw = (gR - gL) / 2 * case.mob_shape
    gh = (gT - gB) / 2 * case.mob_shape
    live = (z[..., 2].abs() < RHO_CLAMP).to(z.dtype) * float(shape)
    grho = (gw * g[..., 2] / 2 - gh * g[..., 3] / 2) * live
    return torch.stack([gcx, gcy, grho], dim=-1) * case.token_mask[..., None]


_NAMED_SCHEDULES = {
    "constant": {"mode": "constant"},
    "cosine": {"mode": "cosine"},
    "linear": {"mode": "polynomial", "power": 1},
}


def lr_vector(lr: float, schedule, steps: int) -> list[float]:
    """The per-step learning rates (peak ``lr``) of ``schedule`` over ``steps`` updates.

    ``schedule`` is ``constant`` / ``cosine`` / ``linear`` or an AnySchedule dict.
    """
    if isinstance(schedule, str):
        cfg = dict(_NAMED_SCHEDULES[schedule])
    else:
        cfg = dict(schedule)
    cfg.setdefault("value", 1.0)
    cfg.setdefault("end", steps)
    if cfg.get("warmup", 0) >= steps:
        cfg["warmup"] = steps // 2
    sched = get_scheduler(cfg)
    return [lr * float(sched(i)) for i in range(steps)]


def _clamp(z, case):
    """``z`` with the anchored coordinates written back."""
    if case.anchor_mask is None:
        return z
    return torch.where(case.anchor_mask > 0.5, case.anchor_z, z)


def _adam_chunk(
    z, m, v, t0, lrs, case_tensors, weights, betas, eps, k, shape=True, sgd=False
):
    """Advance ``k`` updates from global step ``t0`` (a tensor); return ``(z, m, v)``.

    Adam, or plain gradient steps with ``sgd``.
    """
    case = RefineCase(*case_tensors)
    b1, b2 = betas
    for i in range(k):
        g = latent_grad(z, case, weights, shape)
        if case.anchor_mask is not None:
            g = g * (case.anchor_mask < 0.5)
        if sgd:
            z = _clamp(z - lrs[i] * g, case)
            continue
        t = t0 + (i + 1)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        z = _clamp(z - lrs[i] * mhat / (vhat.sqrt() + eps), case)
    return z, m, v


_COMPILED = {}


def _chunk_fn(compile_loop: bool):
    """``_adam_chunk``, compiled once per process when ``compile_loop``."""
    if not compile_loop:
        return _adam_chunk
    if "fn" not in _COMPILED:
        _COMPILED["fn"] = torch.compile(_adam_chunk, dynamic=True)
    return _COMPILED["fn"]


@torch.no_grad()
def refine_closed(
    z0: torch.Tensor,
    case: RefineCase,
    weights: dict[str, float],
    steps: int,
    lr: float,
    lr_schedule="constant",
    betas: tuple[float, float] = (0.0, 0.99),
    eps: float = 1e-8,
    snapshots: tuple[int, ...] = (),
    chunk: int = 5,
    compile_loop: bool = True,
    shape: bool = True,
    optimizer: str = "adam",
) -> dict[int, torch.Tensor]:
    """Descend the constraint energy from ``z0`` ``(B, N, 3)`` for ``steps`` updates.

    ``optimizer`` is ``adam`` or ``sgd``; ``shape=False`` moves positions only. Returns
    ``{step: z}`` at every step in ``snapshots`` (0 = the clamped input) and at ``steps``.
    """
    want = sorted(set(snapshots) | {steps})
    lrs = torch.tensor(
        lr_vector(lr, lr_schedule, steps), dtype=z0.dtype, device=z0.device
    )
    fields = (
        case.area_norm,
        case.token_mask,
        case.mob_pos,
        case.mob_shape,
        case.cluster_id,
        case.mib_id,
        case.boundary_code,
        case.adjacency,
        case.pin_edge_xy_n,
        case.pin_edge_w,
        case.pin_edge_block,
        case.anchor_z,
        case.anchor_mask,
        case.outline,
    )
    fn = _chunk_fn(compile_loop)
    sgd = optimizer == "sgd"
    z = _clamp(z0.detach().clone(), case)
    m, v = torch.zeros_like(z), torch.zeros_like(z)
    out = {0: z.clone()} if 0 in want else {}
    done = 0
    for target in want:
        while done < target:
            k = min(chunk, target - done)
            t0 = torch.tensor(float(done), dtype=z0.dtype, device=z0.device)
            step_lrs = lrs[done : done + k]
            z, m, v = fn(
                z, m, v, t0, step_lrs, fields, weights, betas, eps, k, shape, sgd
            )
            done += k
        out[target] = z.clone()
    return out


@REFINER.register("closed")
class ClosedFormRefiner:
    """The closed-form refiner (``refine(z0, case) -> z``).

    Defaults are the paper setting: :data:`PAPER_WEIGHTS`, 400 Adam steps with betas
    ``(0, 0.99)``, lr ``1e-3`` decaying linearly.
    """

    def __init__(
        self,
        steps: int = 400,
        lr: float = 1e-3,
        lr_schedule="linear",
        weights: dict | None = None,
        betas: tuple[float, float] = (0.0, 0.99),
        chunk: int = 5,
        compile_loop: bool = True,
    ) -> None:
        self.steps = steps
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.weights = dict(PAPER_WEIGHTS if weights is None else weights)
        self.betas = tuple(betas)
        self.chunk = chunk
        self.compile_loop = compile_loop

    def refine(self, z0: torch.Tensor, case: RefineCase) -> torch.Tensor:
        """``z0`` ``(B, N, 3)`` after ``steps`` updates against ``case``."""
        if self.steps <= 0:
            return z0
        snapshots = refine_closed(
            z0,
            case,
            self.weights,
            self.steps,
            self.lr,
            self.lr_schedule,
            betas=self.betas,
            chunk=self.chunk,
            compile_loop=self.compile_loop,
        )
        return snapshots[self.steps]


def total_steps(schedule, step_counts: tuple[int, ...]) -> int:
    """The refiner updates a sweep over ``step_counts`` spends under ``schedule``.

    One checkpointed descent for ``constant``, one descent per count otherwise.
    """
    if schedule == "constant":
        return max(step_counts)
    return sum(s for s in step_counts if s > 0)
