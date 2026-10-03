"""Constraint refiner: energy equals the aux terms, descent lowers it, frozen channels stay put."""

import torch

import trinity  # noqa: F401
from trinity.decode import z_to_xywh
from trinity.losses.base import LossContext
from trinity.registry import LOSS, build
from trinity.sampling.refine_constraint import (
    RefineCase,
    constraint_energy,
    lr_factor,
    refine_latent,
)


def _case(b=2, n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(b, n, 3, generator=g) * 0.3
    area = torch.rand(b, n, generator=g) * 0.02 + 0.005
    adj = torch.rand(b, n, n, generator=g)
    adj = adj + adj.transpose(1, 2)
    case = RefineCase(
        area_norm=area,
        token_mask=torch.ones(b, n),
        mob_pos=torch.ones(b, n),
        mob_shape=torch.ones(b, n),
        cluster_id=torch.randint(0, 3, (b, n), generator=g),
        mib_id=torch.randint(0, 3, (b, n), generator=g),
        boundary_code=torch.randint(1, 11, (b, n), generator=g),
        adjacency=adj,
        pin_edge_xy_n=torch.rand(b, 5, 2, generator=g),
        pin_edge_w=torch.rand(b, 5, generator=g),
        pin_edge_block=torch.randint(0, n, (b, 5), generator=g),
    )
    return z, case


def test_energy_matches_the_registered_aux_terms():
    z, case = _case()
    xywh = z_to_xywh(z, case.area_norm, torch.ones(2))
    ctx = LossContext(
        pred=z,
        target=z,
        weight=torch.ones(2),
        t=torch.zeros(2),
        anchor_mask=torch.zeros(2, 8, 3),
        xywh=xywh,
        area_targets=case.area_norm,
        scale=torch.ones(2),
        cluster_id=case.cluster_id,
        mib_id=case.mib_id,
        boundary_code=case.boundary_code,
        token_mask=case.token_mask,
    )
    ctx.mob_pos, ctx.mob_shape, ctx.adjacency = (
        case.mob_pos,
        case.mob_shape,
        case.adjacency,
    )
    ctx.pin_edge_xy_n, ctx.pin_edge_w, ctx.pin_edge_block = (
        case.pin_edge_xy_n,
        case.pin_edge_w,
        case.pin_edge_block,
    )
    expected = sum(
        build({"name": nm, "weight": 1.0}, LOSS)(ctx)[0]
        for nm in ("overlap", "group", "mib", "boundary", "wl", "area")
    )
    got = constraint_energy(
        z, case, {k: 1.0 for k in ("overlap", "group", "mib", "boundary", "wl", "area")}
    ).mean()
    assert torch.allclose(got, expected, atol=1e-6)


def test_descent_lowers_the_energy_and_keeps_frozen_channels():
    z, case = _case()
    case.mob_pos[:, 0] = 0
    case.mob_shape[:, 0] = 0
    case.mob_shape[:, 1] = 0
    weights = {k: 1.0 for k in ("overlap", "group", "mib", "boundary", "wl", "area")}
    e0 = constraint_energy(z, case, weights)
    out = refine_latent(z, case, weights, steps=30, lr=0.01, snapshots=(0, 10))
    assert set(out) == {0, 10, 30}
    e1 = constraint_energy(out[30], case, weights)
    assert torch.all(e1 < e0)
    assert torch.allclose(out[30][:, 0], z[:, 0])  # preplaced block untouched
    assert torch.allclose(
        out[30][:, 1, 2], z[:, 1, 2]
    )  # fixed-shape block's rho untouched
    assert not torch.allclose(out[30][:, 1, :2], z[:, 1, :2])  # its position moved


def test_lr_schedules_decay_to_zero_and_still_descend():
    z, case = _case()
    weights = {k: 1.0 for k in ("overlap", "group", "mib", "boundary", "wl", "area")}
    assert (
        lr_factor("cosine", 0, 20) == 1.0 and abs(lr_factor("cosine", 19, 20)) < 1e-12
    )
    assert lr_factor("linear", 0, 20) == 1.0 and lr_factor("linear", 19, 20) == 0.0
    assert lr_factor("constant", 19, 20) == 1.0
    e0 = constraint_energy(z, case, weights)
    for sched in ("cosine", "linear"):
        out = refine_latent(
            z, case, weights, steps=20, lr=0.01, snapshots=(5,), lr_schedule=sched
        )
        assert set(out) == {5, 20}
        assert torch.all(constraint_energy(out[20], case, weights) < e0)
