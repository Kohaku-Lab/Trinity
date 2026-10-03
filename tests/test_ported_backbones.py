"""CPU gates for the ported baseline backbones: build from the registry, forward on real data."""

import pytest
import torch

import trinity_baselines  # noqa: F401  (register)
from trinity.conditioning import build_conditioning
from trinity.floorplan.data import find_floorset_root, load_validation_case
from trinity.models import DenoiserCond
from trinity_baselines.registry import BASELINE_BACKBONE, build

pytest.importorskip("torch_geometric")


def have_data() -> bool:
    try:
        find_floorset_root(allow_download=False)
        return True
    except FileNotFoundError:
        return False


needs_data = pytest.mark.skipif(
    not have_data(), reason="FloorSet validation set not present"
)
SMALL = {
    "name": "flowplace_attgnn",
    "hidden_size": 64,
    "hidden_node_features": (64,),
    "attention_node_features": (32,),
}


def padded_batch(cases):
    """Pad a list of instances into one ``DenoiserCond`` plus a random ``z_t``."""
    n_max = max(c.block_count for c in cases)
    feats, adjs, masks = [], [], []
    for case in cases:
        conditioning = build_conditioning(case)
        n = case.block_count
        features = torch.zeros(n_max, conditioning["features"].shape[1])
        features[:n] = conditioning["features"]
        adjacency = torch.zeros(n_max, n_max)
        adjacency[:n, :n] = conditioning["adjacency"]
        mask = torch.zeros(n_max, dtype=torch.bool)
        mask[:n] = True
        feats.append(features)
        adjs.append(adjacency)
        masks.append(mask)
    cond = DenoiserCond(
        features=torch.stack(feats),
        adjacency=torch.stack(adjs),
        key_pad_mask=torch.stack(masks),
    )
    return torch.randn(len(cases), n_max, 3), cond


def test_registered():
    assert "flowplace_attgnn" in BASELINE_BACKBONE.keys()


@needs_data
def test_flowplace_attgnn_forward_shapes_and_padding():
    cases = [load_validation_case(21), load_validation_case(60)]
    z, cond = padded_batch(cases)
    net = build(SMALL, BASELINE_BACKBONE).eval()
    with torch.no_grad():
        out = net(z, torch.rand(2), cond)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


@needs_data
def test_flowplace_attgnn_rows_are_independent():
    """Row 0's output must not change when row 1's inputs change (no cross-row leakage)."""
    cases = [load_validation_case(21), load_validation_case(60)]
    z, cond = padded_batch(cases)
    net = build(SMALL, BASELINE_BACKBONE).eval()
    t = torch.tensor([0.3, 0.7])
    z2 = z.clone()
    z2[1] = torch.randn_like(z2[1])
    with torch.no_grad():
        a = net(z, t, cond)
        b = net(z2, t, cond)
    assert torch.allclose(a[0], b[0], atol=1e-5)
    assert not torch.allclose(a[1], b[1])


@needs_data
def test_flowplace_attgnn_paper_size_param_count():
    """The paper 'large' configuration builds and has a parameter count in the expected range."""
    net = build("flowplace_attgnn", BASELINE_BACKBONE)
    n_params = sum(p.numel() for p in net.parameters())
    assert 1_000_000 < n_params < 20_000_000, n_params
