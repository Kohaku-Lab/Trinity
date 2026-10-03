"""The DiffPlace port: shape, pad independence,
permutation equivariance, parameter count."""

import torch

import trinity_baselines  # noqa: F401  (register the ported backbones)
from trinity.models import DenoiserCond
from trinity_baselines.registry import BASELINE_BACKBONE, build

SMALL = {
    "name": "diffplace_vgnn",
    "hidden_size": 32,
    "num_blocks": 2,
    "layers_per_block": 1,
    "num_heads": 4,
}


def _case(b, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(b, n, 3, generator=g)
    feats = torch.randn(b, n, 19, generator=g)
    adj = torch.rand(b, n, n, generator=g)
    adj = ((adj + adj.transpose(1, 2)) > 1.2).float() * torch.rand(b, n, n, generator=g)
    adj = adj * (1 - torch.eye(n)).unsqueeze(0)
    adj = torch.maximum(adj, adj.transpose(1, 2))
    return z, feats, adj, torch.rand(b, generator=g)


def _cond(feats, adj, real):
    return DenoiserCond(features=feats, adjacency=adj, key_pad_mask=real)


def test_forward_shape():
    net = build(SMALL, BASELINE_BACKBONE, latent_dim=3, feature_dim=19).eval()
    z, feats, adj, t = _case(2, 9)
    out = net(z, t, _cond(feats, adj, torch.ones(2, 9, dtype=torch.bool)))
    assert out.shape == (2, 9, 3) and torch.isfinite(out).all()


def test_pad_and_batch_mate_independence():
    torch.manual_seed(0)
    net = build(SMALL, BASELINE_BACKBONE, latent_dim=3, feature_dim=19).eval()
    z, feats, adj, t = _case(2, 8)
    ref = net(
        z[:1], t[:1], _cond(feats[:1], adj[:1], torch.ones(1, 8, dtype=torch.bool))
    )
    # padded to 12 tokens and batched with a different case
    zp = torch.cat(
        [
            torch.cat([z[:1], torch.randn(1, 4, 3)], 1),
            torch.cat([z[1:], torch.zeros(1, 4, 3)], 1),
        ]
    )
    fp = torch.cat(
        [
            torch.cat([feats[:1], torch.randn(1, 4, 19)], 1),
            torch.cat([feats[1:], torch.zeros(1, 4, 19)], 1),
        ]
    )
    ap = torch.zeros(2, 12, 12)
    ap[:, :8, :8] = adj
    real = torch.zeros(2, 12, dtype=torch.bool)
    real[:, :8] = True
    out = net(zp, t, _cond(fp, ap, real))
    assert torch.allclose(out[0, :8], ref[0], atol=1e-5)


def test_permutation_equivariance():
    torch.manual_seed(1)
    net = build(SMALL, BASELINE_BACKBONE, latent_dim=3, feature_dim=19).eval()
    z, feats, adj, t = _case(1, 7)
    real = torch.ones(1, 7, dtype=torch.bool)
    out = net(z, t, _cond(feats, adj, real))
    p = torch.randperm(7)
    out_p = net(z[:, p], t, _cond(feats[:, p], adj[:, p][:, :, p], real))
    assert torch.allclose(out[:, p], out_p, atol=1e-5)


def test_parameter_count_at_deployed_config():
    net = build(
        {"name": "diffplace_vgnn"}, BASELINE_BACKBONE, latent_dim=3, feature_dim=19
    )
    n = sum(p.numel() for p in net.parameters())
    print(f"diffplace_vgnn deployed params: {n / 1e6:.2f} M")
    assert 10e6 < n < 40e6
