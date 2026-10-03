"""A real token's output must not depend on padding: neither its amount nor its content."""

import pytest
import torch

import trinity  # noqa: F401  (register all)
from trinity.models import DenoiserCond, SetTransformerDenoiser, get_preset


def _forward(model, z, feats, adj, n_pad, seed):
    """Forward one case padded by ``n_pad`` junk tokens; return the real tokens' output."""
    n = z.shape[1]
    g = torch.Generator().manual_seed(seed)
    zp = torch.cat([z, torch.randn(1, n_pad, 3, generator=g)], dim=1)
    fp = torch.cat([feats, torch.zeros(1, n_pad, feats.shape[-1])], dim=1)
    ap = torch.zeros(1, n + n_pad, n + n_pad)
    ap[:, :n, :n] = adj
    mask = torch.zeros(1, n + n_pad, dtype=torch.bool)
    mask[:, :n] = True
    with torch.no_grad():
        out = model(
            zp,
            torch.full((1,), 0.7),
            DenoiserCond(features=fp, adjacency=ap, key_pad_mask=mask),
        )
    return out[:, :n]


@pytest.mark.parametrize("time_cond", ["token", "adaln", "additive"])
@pytest.mark.parametrize("attn", ["sdpa_graph", "naive_graph"])
def test_real_tokens_independent_of_padding(time_cond, attn):
    torch.manual_seed(0)
    cfg = get_preset(
        "DiT-S", attn=attn, time_cond=time_cond, qk_norm=True, feature_dim=19
    )
    model = SetTransformerDenoiser(cfg).eval()
    # a non-zero output head, so the forward is not the identity-at-init zero map
    torch.nn.init.normal_(model.final.linear.weight, std=0.1)
    n = 12
    z, feats = torch.randn(1, n, 3), torch.randn(1, n, 19)
    adj = torch.rand(1, n, n)
    adj = adj + adj.transpose(1, 2)
    ref = _forward(model, z, feats, adj, n_pad=0, seed=0)
    for n_pad, seed in ((8, 1), (40, 2), (40, 3)):
        out = _forward(model, z, feats, adj, n_pad=n_pad, seed=seed)
        assert (out - ref).abs().max() < 1e-4, (
            f"{time_cond}/{attn}: padding {n_pad} (seed {seed}) changed real tokens by "
            f"{(out - ref).abs().max():.3e}"
        )
