"""Closed-form refiner gates: gradient equals autograd, padding invariance, frozen channels,
the loop equals ``torch.optim.Adam``, compiled equals eager, schedules, batched scorer equals
the numpy metric vector."""

import numpy as np
import torch

import trinity  # noqa: F401
from trinity.decode import z_to_xywh
from trinity.floorplan.parameterize import z_to_xywh as z_to_xywh_np
from trinity.floorplan.scoring.vector import COLS, metric_vector
from trinity.floorplan.types import FloorplanInstance
from trinity.losses import constraint as C
from trinity.sampling.refine_case import build_refine_case
from trinity.sampling.refine_closed import (
    TERM_NAMES,
    _mst_edges,
    latent_grad,
    lr_vector,
    refine_closed,
)
from trinity.sampling.refine_constraint import (
    RefineCase,
    constraint_energy,
    refine_latent,
)
from trinity.sampling.score_batched import metric_vector_batched

ALL = {k: 1.0 for k in TERM_NAMES}


def _case(b=3, n=10, pad=0, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(b, n + pad, 3, generator=g, dtype=dtype) * 0.3
    area = torch.rand(b, n + pad, generator=g, dtype=dtype) * 0.02 + 0.005
    area[:, n:] = 0.0
    area = area / area.sum(1, keepdim=True)
    adj = torch.rand(b, n + pad, n + pad, generator=g, dtype=dtype) * (
        torch.rand(b, n + pad, n + pad, generator=g) > 0.6
    )
    adj = adj + adj.transpose(1, 2)
    adj[:, n:, :] = 0
    adj[:, :, n:] = 0
    mask = torch.zeros(b, n + pad, dtype=dtype)
    mask[:, :n] = 1.0
    mob_pos, mob_shape = mask.clone(), mask.clone()
    mob_pos[:, 0] = 0
    mob_shape[:, 0] = 0
    mob_shape[:, 1] = 0

    def ids(hi):
        return torch.randint(0, hi, (b, n + pad), generator=g) * mask.long()

    anchor_mask = torch.zeros(b, n + pad, 3, dtype=dtype)
    anchor_mask[:, 0] = 1.0
    anchor_mask[:, 1, 2] = 1.0
    anchor_z = torch.randn(b, n + pad, 3, generator=g, dtype=dtype) * 0.2
    case = RefineCase(
        area_norm=area,
        token_mask=mask,
        mob_pos=mob_pos,
        mob_shape=mob_shape,
        cluster_id=ids(3),
        mib_id=ids(3),
        boundary_code=torch.randint(0, 11, (b, n + pad), generator=g) * mask.long(),
        adjacency=adj,
        pin_edge_xy_n=torch.rand(b, 5, 2, generator=g, dtype=dtype),
        pin_edge_w=torch.rand(b, 5, generator=g, dtype=dtype),
        pin_edge_block=torch.randint(0, n, (b, 5), generator=g),
        anchor_z=anchor_z,
        anchor_mask=anchor_mask,
    )
    return z, case


def _autograd(z, case, weights):
    zz = z.detach().clone().requires_grad_(True)
    constraint_energy(zz, case, weights).sum().backward()
    return zz.grad


def test_closed_form_gradient_matches_autograd_per_term_and_summed():
    z, case = _case()
    for name in TERM_NAMES:
        w = {name: 1.0}
        assert torch.allclose(
            latent_grad(z, case, w), _autograd(z, case, w), atol=1e-6
        ), name
    weights = {
        "overlap": 0.7,
        "group": 1.3,
        "mib": 0.4,
        "boundary": 2.0,
        "wl": 0.9,
        "area": 1.1,
    }
    assert torch.allclose(
        latent_grad(z, case, weights), _autograd(z, case, weights), atol=1e-6
    )


def test_outline_term_matches_autograd_and_is_silent_without_an_outline():
    z, case = _case(seed=11)
    w = {"outline": 1.0, "area": 0.5}
    assert torch.allclose(latent_grad(z, case, w), _autograd(z, case, w), atol=1e-6)
    assert torch.all(latent_grad(z, case, {"outline": 1.0}) == 0)
    g = z_to_xywh(z, case.area_norm, z.new_ones(z.shape[0]))
    x0, y0, x1, y1 = C.bbox(g, case.token_mask)
    case.outline = torch.stack([(x1 - x0) * 0.8, (y1 - y0) * 1.5], dim=1)
    got = latent_grad(z, case, {"outline": 1.0})
    assert torch.allclose(got, _autograd(z, case, {"outline": 1.0}), atol=1e-6)
    assert torch.any(got[..., 0] != 0) and torch.all(got[..., 1] == 0)
    assert torch.allclose(latent_grad(z, case, w), _autograd(z, case, w), atol=1e-6)


def _variant_energy(z, case, margin, squared):
    """Overlap on boxes inflated by ``margin`` plus the squared MST gap, both at weight 1."""
    g = z_to_xywh(z, case.area_norm, z.new_ones(z.shape[0]))
    g = C._freeze(g, case.mob_pos, case.mob_shape)
    infl = torch.cat([g[..., :2] - margin / 2, g[..., 2:] + margin], dim=-1)
    s2, s = C._scales(case.area_norm, case.token_mask)
    e = C.overlap_area(infl, case.token_mask) / s2
    gap = C.gap_matrix(g)
    edges = _mst_edges(gap, case.cluster_id, case.token_mask)
    total = torch.where(edges, gap**2 if squared else gap, torch.zeros_like(gap)).sum(
        (1, 2)
    )
    return e + total / s


def test_variant_gradients_match_autograd_of_the_variant_energy():
    z, case = _case(seed=21)
    for margin, squared in ((0.02, False), (0.0, True), (0.02, True)):
        zz = z.detach().clone().requires_grad_(True)
        _variant_energy(zz, case, margin, squared).sum().backward()
        w = {
            "overlap": 1.0,
            "group": 1.0,
            "overlap_margin": margin,
            "group_squared": squared,
        }
        assert torch.allclose(latent_grad(z, case, w), zz.grad, atol=1e-6), (
            margin,
            squared,
        )


def test_gradient_is_padding_invariant_and_zero_on_padding():
    z, case = _case(pad=0, seed=3)
    zp, casep = _case(pad=4, seed=3)
    n = z.shape[1]
    zp[:, :n] = z
    for f in (
        "area_norm",
        "token_mask",
        "mob_pos",
        "mob_shape",
        "cluster_id",
        "mib_id",
        "boundary_code",
        "anchor_z",
        "anchor_mask",
    ):
        getattr(casep, f)[:, :n] = getattr(case, f)
    casep.adjacency[:, :n, :n] = case.adjacency
    casep.pin_edge_xy_n[:] = case.pin_edge_xy_n
    casep.pin_edge_w[:] = case.pin_edge_w
    casep.pin_edge_block[:] = case.pin_edge_block
    g, gp = latent_grad(z, case, ALL), latent_grad(zp, casep, ALL)
    assert torch.allclose(g, gp[:, :n], atol=1e-9)
    assert torch.all(gp[:, n:] == 0)


def test_frozen_channels_get_zero_gradient():
    z, case = _case(seed=5)
    g = latent_grad(z, case, ALL)
    assert torch.all(g[:, 0] == 0)
    assert torch.all(g[:, 1, 2] == 0)
    assert torch.any(g[:, 1, :2] != 0)


def test_loop_matches_torch_adam_with_momentum_and_without():
    z, case = _case(seed=7)
    for betas in ((0.9, 0.999), (0.0, 0.99)):
        ref = refine_latent(
            z, case, ALL, steps=24, lr=1e-3, snapshots=(0, 8, 16), betas=betas
        )
        got = refine_closed(
            z,
            case,
            ALL,
            steps=24,
            lr=1e-3,
            betas=betas,
            snapshots=(0, 8, 16),
            chunk=4,
            compile_loop=False,
        )
        assert set(got) == {0, 8, 16, 24}
        for s in (8, 16, 24):
            assert torch.allclose(got[s], ref[s], atol=1e-7), (betas, s)
        assert torch.allclose(got[24][:, 0], case.anchor_z[:, 0])


def test_sgd_loop_matches_torch_sgd():
    z, case = _case(seed=8)
    ref = refine_latent(
        z, case, ALL, steps=12, lr=1e-3, optimizer="sgd", snapshots=(6,)
    )
    got = refine_closed(
        z,
        case,
        ALL,
        steps=12,
        lr=1e-3,
        snapshots=(6,),
        chunk=4,
        compile_loop=False,
        optimizer="sgd",
    )
    for s in (6, 12):
        assert torch.allclose(got[s], ref[s], atol=1e-9)


def test_compiled_chunk_matches_eager():
    z, case = _case(seed=9, dtype=torch.float32)
    eager = refine_closed(
        z, case, ALL, steps=12, lr=1e-3, snapshots=(4,), chunk=4, compile_loop=False
    )
    comp = refine_closed(
        z, case, ALL, steps=12, lr=1e-3, snapshots=(4,), chunk=4, compile_loop=True
    )
    for s in (4, 12):
        assert torch.allclose(comp[s], eager[s], atol=1e-5)


def test_lr_vectors():
    const = lr_vector(1e-3, "constant", 10)
    cos = lr_vector(1e-3, "cosine", 10)
    lin = lr_vector(1e-3, "linear", 10)
    assert const == [1e-3] * 10
    assert cos[0] == 1e-3 and cos[-1] < 1e-4
    assert all(a >= b for a, b in zip(cos, cos[1:], strict=False))
    assert lin[0] == 1e-3 and abs(lin[-1] - 1e-4) < 1e-12
    custom = lr_vector(1e-3, {"mode": "cosine", "warmup": 3, "min_value": 0.1}, 10)
    assert custom[0] == 0.0 and custom[3] == 1e-3 and custom[-1] >= 1e-4
    short = lr_vector(1e-3, {"mode": "cosine", "warmup": 10}, 10)
    assert len(short) == 10 and short[5] == 1e-3


def _instance(seed):
    r = np.random.default_rng(seed)
    n = 12
    area = r.uniform(4.0, 30.0, n)
    cons = np.zeros((n, 5), dtype=np.int64)
    cons[1, 0] = 1
    cons[2, 1] = 1
    cons[3:6, 2] = 1
    cons[6:8, 2] = 2
    cons[0:3, 3] = 1
    cons[8:11, 3] = 2
    cons[9, 4] = 1
    cons[10, 4] = 6
    cons[11, 4] = 8
    b2b = np.array([[0, 1, 2.0], [1, 2, 1.0], [2, 5, 3.0], [6, 7, 1.5], [0, 11, 0.5]])
    pins = r.uniform(-5.0, 30.0, (3, 2))
    p2b = np.array([[0, 4, 1.0], [1, 4, 2.0], [2, 9, 0.7]])
    gt = np.zeros((n, 4))
    gt[:, 2] = np.sqrt(area) * r.uniform(0.7, 1.4, n)
    gt[:, 3] = area / gt[:, 2]
    gt[:, 0] = r.uniform(0, 20, n)
    gt[:, 1] = r.uniform(0, 20, n)
    tp = np.full((n, 4), -1.0)
    tp[1, 2:] = gt[1, 2:]
    tp[2] = gt[2]
    return FloorplanInstance(
        block_count=n,
        area_targets=area,
        constraints=cons,
        b2b=b2b,
        p2b=p2b,
        pins_pos=pins.astype(np.float32),
        target_positions=tp,
        gt_positions=gt,
    )


def test_batched_scorer_matches_numpy_metric_vector():
    insts = [_instance(s) for s in range(4)]
    case = build_refine_case(insts, 2, "cpu", dtype=torch.float64)
    r = np.random.default_rng(1)
    z = torch.zeros(8, 12, 3, dtype=torch.float64)
    expected = np.zeros((8, len(COLS)))
    for gi, inst in enumerate(insts):
        for j in range(2):
            row = gi * 2 + j
            zz = r.normal(0, 0.4, (12, 3))
            zz[2, :2] = case.anchor_z[row, 2, :2].numpy()
            zz[1, 2] = case.anchor_z[row, 1, 2].numpy()
            zz[2, 2] = case.anchor_z[row, 2, 2].numpy()
            z[row] = torch.as_tensor(zz)
            expected[row] = metric_vector(
                z_to_xywh_np(zz, inst.area_targets, inst.s), inst
            )
    g = z_to_xywh(z, case.area_norm, torch.ones(8, dtype=torch.float64))
    got = metric_vector_batched(g, case).numpy()
    for c, name in enumerate(COLS):
        assert np.allclose(got[:, c], expected[:, c], atol=1e-6, rtol=1e-5), (
            name,
            got[:, c],
            expected[:, c],
        )
