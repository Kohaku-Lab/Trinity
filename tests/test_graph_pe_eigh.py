"""The batched spectral-drawing PE survives a
batch element the batched eigensolver rejects."""

import torch

import trinity  # noqa: F401  (register)
from trinity.conditioning import graph_pe
from trinity.registry import GRAPH_PE, build


def _batch(b, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(b, n, n, generator=g)
    a = a + a.transpose(1, 2)  # dense, distinct weights: a simple spectrum
    a = a * (1 - torch.eye(n)).unsqueeze(0)
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[1, 7:] = False
    a[1, 7:, :] = 0
    a[1, :, 7:] = 0
    return a, mask


def test_batched_matches_reference_and_survives_a_rejected_element(monkeypatch):
    pe = build({"name": "spectral_draw", "k": 2, "normalize": "log1p"}, GRAPH_PE)
    a, mask = _batch(3, 9)
    ref = pe.batched(a, mask)
    for i in range(3):
        n = int(mask[i].sum())
        single = torch.from_numpy(pe(a[i, :n, :n].numpy()))
        assert torch.allclose(ref[i, :n], single, atol=1e-5)
    real_eigh = torch.linalg.eigh
    calls = {"n": 0}

    def flaky(m, *args, **kwargs):
        calls["n"] += 1
        if m.dim() == 3:
            raise torch._C._LinAlgError(
                "linalg.eigh: (Batch element 1): failed to converge"
            )
        return real_eigh(m, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "eigh", flaky)
    out = pe.batched(a, mask)
    assert calls["n"] > 1
    assert torch.allclose(out, ref, atol=1e-6)


def test_padded_case_keeps_both_axes_when_real_spectrum_exceeds_one():
    pe = build({"name": "spectral_draw", "k": 2, "normalize": "log1p"}, GRAPH_PE)
    a, mask = _batch(3, 9)
    out = pe.batched(a, mask)
    assert (
        out[1, :7, 1].abs().sum() > 0.5
    )  # the second axis is a real Fiedler vector, not a pad indicator
    assert torch.all(out[1, 7:] == 0)


def test_jitter_path_returns_finite_vectors():
    eye = torch.eye(5, dtype=torch.float64)
    v = graph_pe._eigh_single(torch.zeros(5, 5, dtype=torch.float64), eye)
    assert v.shape == (5, 5) and torch.isfinite(v).all()
