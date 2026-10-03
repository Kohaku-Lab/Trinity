"""Per-constraint aux terms: exact values on hand-built layouts, zero-iff, gradients, t-weighting."""

import pytest
import torch

import trinity  # noqa: F401  (register all)
from trinity.decode import z_to_xywh
from trinity.losses.base import LossContext
from trinity.losses.constraint import (
    boundary_distance,
    cluster_mst_gap,
    mib_median_deviation,
    net_length,
    overlap_area,
)
from trinity.registry import LOSS, build

TERMS = ("overlap", "group", "mib", "boundary", "wl", "area")


def _ctx(xywh, cluster=None, mib=None, code=None, adjacency=None, pins=None, t=None):
    b, n, _ = xywh.shape
    zeros_l = torch.zeros(b, n, dtype=torch.long)
    p_xy, p_w, p_blk = (
        pins
        if pins is not None
        else (
            torch.zeros(b, 1, 2, dtype=xywh.dtype),
            torch.zeros(b, 1, dtype=xywh.dtype),
            torch.zeros(b, 1, dtype=torch.long),
        )
    )
    ctx = LossContext(
        pred=xywh,
        target=xywh,
        weight=torch.ones(b),
        t=t if t is not None else torch.full((b,), 0.5),
        anchor_mask=torch.zeros(b, n, 3),
        xywh=xywh,
        area_targets=xywh[..., 2] * xywh[..., 3],
        scale=torch.ones(b),
        cluster_id=cluster if cluster is not None else zeros_l,
        mib_id=mib if mib is not None else zeros_l,
        boundary_code=code if code is not None else zeros_l,
        token_mask=torch.ones(b, n),
    )
    ctx.mob_pos = torch.ones(b, n)
    ctx.mob_shape = torch.ones(b, n)
    ctx.adjacency = (
        adjacency if adjacency is not None else torch.zeros(b, n, n, dtype=xywh.dtype)
    )
    ctx.pin_edge_xy_n, ctx.pin_edge_w, ctx.pin_edge_block = p_xy, p_w, p_blk
    return ctx


def _boxes(rows):
    return torch.tensor([rows], dtype=torch.float64)


def test_registered():
    assert set(TERMS) <= set(LOSS.keys())


def test_overlap_exact_area_and_zero_when_apart():
    g = _boxes([[0, 0, 2, 2], [1, 1, 2, 2], [5, 5, 1, 1]])
    assert overlap_area(g, torch.ones(1, 3)).item() == pytest.approx(1.0)
    apart = _boxes([[0, 0, 1, 1], [2, 0, 1, 1], [0, 2, 1, 1]])
    assert overlap_area(apart, torch.ones(1, 3)).item() == 0.0
    loss, _ = build({"name": "overlap", "weight": 1.0}, LOSS)(_ctx(g))
    assert loss.item() == pytest.approx(1.0 / (4 + 4 + 1))


def test_overlap_gradient_pushes_apart():
    g = _boxes([[0, 0, 2, 2], [1, 0, 2, 2]]).requires_grad_(True)
    overlap_area(g, torch.ones(1, 2)).sum().backward()
    # gradient = y-penetration; descent (-grad) moves the left block left and the right block right
    assert g.grad[0, 0, 0] == pytest.approx(2.0) and g.grad[0, 1, 0] == pytest.approx(
        -2.0
    )


def test_group_mst_gap_is_min_total_gap_to_connect():
    g = _boxes([[0, 0, 1, 1], [2, 0, 1, 1], [5, 0, 1, 1]])
    cid = torch.tensor([[1, 1, 1]])
    assert cluster_mst_gap(g, cid, torch.ones(1, 3)).item() == pytest.approx(3.0)
    touching = _boxes([[0, 0, 1, 1], [1, 0, 1, 1], [2, 0, 1, 1]])
    assert cluster_mst_gap(touching, cid, torch.ones(1, 3)).item() == 0.0
    two_pairs = _boxes([[0, 0, 1, 1], [1, 0, 1, 1], [10, 0, 1, 1], [11, 0, 1, 1]])
    assert cluster_mst_gap(
        two_pairs, torch.tensor([[1, 1, 1, 1]]), torch.ones(1, 4)
    ).item() == pytest.approx(8.0)
    g2 = _boxes(
        [
            [0, 0, 1, 1],
            [3, 0, 1, 1],
            [0, 5, 1, 1],
            [0, 8, 1, 1],
            [20, 20, 1, 1],
            [0, 0, 0, 0],
        ]
    )
    cid2 = torch.tensor([[1, 1, 2, 2, 0, 0]])
    mask = torch.tensor([[1.0, 1, 1, 1, 1, 0]])
    assert cluster_mst_gap(g2, cid2, mask).item() == pytest.approx(4.0)


def test_group_gradient_pulls_mst_edges_only():
    g = _boxes([[0, 0, 1, 1], [2, 0, 1, 1], [5, 0, 1, 1]]).requires_grad_(True)
    cluster_mst_gap(g, torch.tensor([[1, 1, 1]]), torch.ones(1, 3)).sum().backward()
    # descent moves a right (toward b) and c left (toward b); b is pulled both ways equally
    assert g.grad[0, 0, 0] == pytest.approx(-1.0)
    assert g.grad[0, 2, 0] == pytest.approx(1.0)
    assert g.grad[0, 1, 0] == pytest.approx(0.0)


def test_mib_median_deviation():
    g = _boxes([[0, 0, 4, 1], [0, 0, 1, 1], [0, 0, 1, 4], [0, 0, 1, 1]])
    v = mib_median_deviation(g, torch.tensor([[1, 1, 1, 0]]), torch.ones(1, 4)).item()
    assert v == pytest.approx(2 * torch.log(torch.tensor(4.0)).item())
    same = _boxes([[0, 0, 2, 1], [3, 0, 2, 1], [6, 0, 2, 1]])
    assert (
        mib_median_deviation(same, torch.tensor([[1, 1, 1]]), torch.ones(1, 3)).item()
        == 0.0
    )


def test_boundary_distance_and_zero_on_edge():
    g = _boxes([[0, 0, 1, 1], [3, 0, 1, 1], [1, 2, 1, 1]])
    assert (
        boundary_distance(g, torch.tensor([[1, 2, 4]]), torch.ones(1, 3)).item() == 0.0
    )
    assert boundary_distance(
        g, torch.tensor([[2, 0, 8]]), torch.ones(1, 3)
    ).item() == pytest.approx(5.0)


def test_net_length_b2b_and_per_pin_exact():
    g = _boxes([[0, 0, 2, 2], [4, 0, 2, 2]])
    adj = torch.zeros(1, 2, 2, dtype=torch.float64)
    adj[0, 0, 1] = adj[0, 1, 0] = 2.0
    pins = (
        torch.tensor([[[0.0, 1.0], [2.0, 1.0]]], dtype=torch.float64),
        torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0, 0]]),
    )
    h, w = net_length(g, adj, *pins, torch.ones(1, 2))
    assert h.item() == pytest.approx(2.0 * 4.0 + 2.0)
    assert w.item() == pytest.approx(4.0)


def test_area_term_equals_compactness_and_t_weighting():
    g = _boxes([[0, 0, 1, 1], [2, 0, 1, 1]])
    ctx = _ctx(g, t=torch.tensor([0.25]))
    fixed = build({"name": "area", "weight": 0.1}, LOSS)(ctx)[0]
    scaled = build({"name": "area", "weight": 0.1, "t_weight": True}, LOSS)(ctx)[0]
    assert fixed.item() == pytest.approx(0.1 * 1.5)
    assert scaled.item() == pytest.approx(0.1 * 0.25 * 1.5)


def test_all_terms_finite_gradients_on_random_batch():
    torch.manual_seed(0)
    b, n = 3, 12
    xywh = (torch.rand(b, n, 4) * 0.5 + 0.05).requires_grad_(True)
    cid, mib = torch.randint(0, 3, (b, n)), torch.randint(0, 3, (b, n))
    code = torch.randint(0, 11, (b, n))
    adj = torch.rand(b, n, n)
    adj = adj + adj.transpose(1, 2)
    pins = (torch.rand(b, 7, 2), torch.rand(b, 7), torch.randint(0, n, (b, 7)))
    ctx = _ctx(xywh, cid, mib, code, adj, pins)
    total = sum(build({"name": nm, "weight": 1.0}, LOSS)(ctx)[0] for nm in TERMS)
    total.backward()
    assert torch.isfinite(xywh.grad).all()


def test_frozen_channels_receive_no_gradient_through_the_decode():
    torch.manual_seed(2)
    b, n = 2, 10
    z = (torch.randn(b, n, 3) * 0.3).requires_grad_(True)
    area = torch.rand(b, n) * 0.02 + 0.005
    xywh = z_to_xywh(z, area, torch.ones(b))
    cid, mib = torch.randint(0, 3, (b, n)), torch.randint(0, 3, (b, n))
    code = torch.randint(1, 11, (b, n))
    adj = torch.rand(b, n, n)
    adj = adj + adj.transpose(1, 2)
    pins = (torch.rand(b, 6, 2), torch.rand(b, 6), torch.randint(0, n, (b, 6)))
    ctx = _ctx(xywh, cid, mib, code, adj, pins)
    ctx.area_targets = area
    ctx.mob_shape[:, 0] = 0  # block 0 fixed-shape: position free, shape frozen
    ctx.mob_pos[:, 1] = 0  # block 1 preplaced: everything frozen
    ctx.mob_shape[:, 1] = 0
    total = sum(build({"name": nm, "weight": 1.0}, LOSS)(ctx)[0] for nm in TERMS)
    total.backward()
    # the fixed-shape block's rho sees only the round-off of the cancelled -w/2 + w/2 decode path
    assert z.grad[:, 0, 2].abs().max() < 1e-6
    assert z.grad[:, 0, :2].abs().max() > 1e-2  # its centre still moves
    assert torch.all(z.grad[:, 1] == 0)  # the preplaced block receives nothing
    assert z.grad[:, 2:, 2].abs().max() > 1e-2  # free blocks do get rho gradient


def test_terms_compile():
    torch.manual_seed(1)
    b, n = 2, 8
    xywh = torch.rand(b, n, 4) * 0.5 + 0.05
    ctx = _ctx(
        xywh,
        torch.randint(0, 3, (b, n)),
        torch.randint(0, 2, (b, n)),
        torch.randint(0, 11, (b, n)),
        torch.rand(b, n, n),
        (torch.rand(b, 5, 2), torch.rand(b, 5), torch.randint(0, n, (b, 5))),
    )
    for nm in TERMS:
        term = build({"name": nm, "weight": 1.0}, LOSS)
        eager = term(ctx)[0].item()
        term.fn = torch.compile(term.fn)
        assert term(ctx)[0].item() == pytest.approx(eager, rel=1e-5, abs=1e-7)
