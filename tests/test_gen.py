"""Correctness gates of the method: framings, backbone, losses,
trainer, placer, graph PE, and the release export / load round trip."""

from types import SimpleNamespace

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import trinity  # noqa: F401  (register all)
from trinity.conditioning import build_b2b_dense
from trinity.data import SameNBatchSampler, collate_same_n, validation_dataset
from trinity.decode import z_to_xywh
from trinity.floorplan.data import (
    find_floorset_root,
    load_validation_case,
    load_validation_set,
)
from trinity.floorplan.scoring import validate
from trinity.floorplan.scoring.total import total_blockcount, total_exp
from trinity.hub import export_release, load_model
from trinity.losses.base import LossContext
from trinity.models import DenoiserCond, SetTransformerDenoiser, get_preset
from trinity.registry import (
    FRAMING,
    GRAPH_PE,
    LATENT_PARAM,
    LOSS,
    REFINER,
    SAMPLER,
    build,
)
from trinity.solver import DiffusionPlacer
from trinity.training import DiffusionTrainer, ValidationCallback

_PE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _have_data() -> bool:
    try:
        find_floorset_root(allow_download=False)
        return True
    except FileNotFoundError:
        return False


needs_data = pytest.mark.skipif(
    not _have_data(), reason="FloorSet validation set not present"
)


def test_registries_populated():
    assert {"ddpm_x0", "ddpm_eps", "ddpm_v", "rectified_flow", "gvp"} <= set(
        FRAMING.keys()
    )
    assert {"denoise", "overlap", "group", "mib", "boundary", "wl", "area"} <= set(
        LOSS.keys()
    )
    assert "euler" in SAMPLER.keys()
    assert "heun" in SAMPLER.keys()
    assert {"closed", "constraint_latent"} <= set(REFINER.keys())


@pytest.mark.parametrize(
    "name", ["ddpm_x0", "ddpm_eps", "ddpm_v", "rectified_flow", "gvp"]
)
def test_framing_recover_and_x0_mapping(name):
    fr = build(name, FRAMING)
    x0, x1, t = torch.randn(4, 8, 3), torch.randn(4, 8, 3), torch.rand(4)
    x_t, target = fr.prepare(x0, x1, t)
    rx0, rx1 = fr.recover(x_t, target, t)
    assert (rx0 - x0).abs().max() < 1e-3 and (rx1 - x1).abs().max() < 1e-3
    # network emits true x0 -> pred_to_target must equal the true target
    assert (fr.pred_to_target(x_t, x0, t) - target).abs().max() < 1e-4


@pytest.mark.parametrize("attn", ["sdpa", "sdpa_graph", "naive_graph"])
def test_backbone_shapes_and_equivariance(attn):
    cfg = get_preset("DiT-S", attn=attn, feature_dim=19)
    model = SetTransformerDenoiser(cfg).eval()
    n = 12
    z, t, feats = torch.randn(1, n, 3), torch.rand(1), torch.randn(1, n, 19)
    adj = torch.rand(1, n, n)
    adj = adj + adj.transpose(1, 2)
    perm = torch.randperm(n)
    with torch.no_grad():
        o1 = model(z, t, DenoiserCond(features=feats, adjacency=adj))
        o2 = model(
            z[:, perm],
            t,
            DenoiserCond(features=feats[:, perm], adjacency=adj[:, perm][:, :, perm]),
        )
    assert o1.shape == (1, n, 3)
    assert (o1[:, perm] - o2).abs().max() < 1e-4


def test_losses_differentiable():
    z = torch.randn(2, 10, 3, requires_grad=True)
    gt = torch.randn(2, 10, 3)
    area = torch.rand(2, 10) * 100 + 10
    scale = area.sum(1).sqrt()
    cid = torch.zeros(2, 10, dtype=torch.long)
    cid[:, :4] = 1
    mib = torch.zeros(2, 10, dtype=torch.long)
    mib[:, :3] = 1
    bcode = torch.zeros(2, 10, dtype=torch.long)
    bcode[:, 0] = 1
    area_norm = area / scale.unsqueeze(-1) ** 2
    ones = torch.ones_like(scale)
    ctx = LossContext(
        pred=z,
        target=gt,
        weight=torch.ones(2),
        t=torch.rand(2),
        anchor_mask=torch.zeros(2, 10, 3),
        xywh=z_to_xywh(z, area_norm, ones),
        xywh_gt=z_to_xywh(gt, area_norm, ones),
        area_targets=area_norm,
        scale=ones,
        cluster_id=cid,
        mib_id=mib,
        boundary_code=bcode,
        token_mask=torch.ones(2, 10),
    )
    names = [
        "denoise",
        "ref_overlap",
        "ref_area",
        "ref_grouping",
        "ref_mib",
        "ref_boundary",
    ]
    total = sum(build({"name": nm}, LOSS)(ctx)[0] for nm in names)
    total.backward()
    assert torch.isfinite(z.grad).all()


@needs_data
def test_trainer_step_runs():
    ds = validation_dataset(allow_download=False)
    sampler = SameNBatchSampler(ds.block_counts, batch_size=2, shuffle=True, seed=0)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_same_n)
    model = DiffusionTrainer(
        preset="DiT-S", arch_overrides={"feature_dim": 19}, use_ema=False
    )
    trainer = pl.Trainer(
        max_steps=5,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, loader)


@needs_data
def test_solver_produces_feasible_layout():
    inst = load_validation_case(21)
    framing = build("ddpm_x0", FRAMING)
    param = build("s_only", LATENT_PARAM)
    model = SetTransformerDenoiser(get_preset("DiT-S", feature_dim=19))
    sampler = build({"name": "euler", "num_steps": 5}, SAMPLER, framing=framing)
    refiner = build({"name": "closed", "steps": 30, "compile_loop": False}, REFINER)
    none_pe = build("none", GRAPH_PE)
    placer = DiffusionPlacer(
        model,
        sampler,
        refiner,
        param=param,
        graph_pe=none_pe,
        samples=2,
        scorer="full",
        device="cpu",
    )
    placement = placer.solve(inst)
    assert validate(placement, "full").score.feasible
    assert np.asarray(placement.to_tuples()).shape == (21, 4)


@needs_data
def test_trainer_solve_cases_and_validation_totals():
    model = DiffusionTrainer(
        preset="DiT-S",
        arch_overrides={"feature_dim": 19},
        use_ema=True,
        eval_samples=1,
        sample_steps=3,
        refiner={"name": "closed", "steps": 10, "compile_loop": False},
    )
    cases = [load_validation_case(n) for n in (21, 40)]
    results = model.solve_cases(cases)
    assert len(results) == 2
    assert all(score.feasible for _, score, _ in results)
    callback = ValidationCallback(
        shard_fn=lambda r, w: cases[r::w], every_n_steps=1, num_render=0
    )
    assert callback.shard_fn(0, 1) == cases
    rows = [
        (i.block_count, s.cost, float(s.feasible), s.v_rel, s.hpwl_gap, s.area_gap)
        for i, s, _ in results
    ]
    gathered = callback._all_gather(SimpleNamespace(world_size=1), rows)
    assert gathered == rows

    costs = [s.cost for _, s, _ in results]
    bcs = [i.block_count for i, _, _ in results]
    assert total_exp(costs, bcs) > 0 and total_blockcount(costs, bcs) > 0


def _pad_batch(cases, device):
    """Stack real instances' raw b2b adjacency into
    ``(B, max_n, max_n)`` + a ``(B, max_n)`` mask."""
    max_n = max(c.block_count for c in cases)
    adj = torch.zeros(len(cases), max_n, max_n)
    mask = torch.zeros(len(cases), max_n, dtype=torch.bool)
    for i, c in enumerate(cases):
        n = c.block_count
        adj[i, :n, :n] = torch.from_numpy(build_b2b_dense(c))
        mask[i, :n] = True
    return adj.to(device), mask.to(device), max_n


@needs_data
@pytest.mark.parametrize(
    "spec",
    [
        {"name": "spectral_draw", "k": 2},
        {"name": "spectral_draw", "k": 2, "normalize": "log1p"},
        {"name": "spectral_draw", "k": 3},
        {"name": "rwpe", "k": 8},
    ],
)
def test_batched_gpu_pe_matches_numpy(spec):
    """The batched PE equals the per-instance numpy reference
    on every real block, and is exactly zero on padded rows."""
    builder = build(spec, GRAPH_PE)
    cases = load_validation_set(allow_download=False)[:24]
    adj, mask, _max_n = _pad_batch(cases, _PE_DEVICE)
    gpu_pe = builder.batched(adj, mask).cpu().numpy()
    max_diff = 0.0
    for i, c in enumerate(cases):
        n = c.block_count
        ref = builder(build_b2b_dense(c))
        max_diff = max(max_diff, float(np.abs(gpu_pe[i, :n] - ref).max()))
        assert not np.any(gpu_pe[i, n:])
    assert max_diff < 1e-3, f"GPU PE vs numpy max abs diff {max_diff:.2e} exceeds 1e-3"


@needs_data
def test_no_graph_pe_is_batched_noop():
    """``graph_pe_dim == 0`` (NoGraphPE) yields an
    empty ``(B, N, 0)`` PE (no eigendecomposition)."""
    builder = build("none", GRAPH_PE)
    assert builder.dim == 0
    cases = load_validation_set(allow_download=False)[:4]
    adj, mask, max_n = _pad_batch(cases, _PE_DEVICE)
    out = builder.batched(adj, mask)
    assert out.shape == (len(cases), max_n, 0)


def test_release_round_trip(tmp_path):
    """A checkpoint exported as a release reloads
    with the EMA weights and the same outputs."""
    model = DiffusionTrainer(
        preset="DiT-S",
        arch_overrides={"feature_dim": 19, "graph_pe_dim": 2},
        graph_pe={"name": "spectral_draw", "k": 2},
        framing="rectified_flow",
        use_ema=True,
    )
    with torch.no_grad():
        for shadow in model.ema.shadow:
            shadow.add_(0.01 * torch.randn_like(shadow))
    ckpt = tmp_path / "model.ckpt"
    torch.save(
        {"hyper_parameters": dict(model.hparams), "state_dict": model.state_dict()},
        ckpt,
    )

    release = export_release(ckpt, tmp_path / "release")
    loaded = load_model(release)
    from_ckpt = load_model(ckpt)

    n = 10
    z, t = torch.randn(1, n, 3), torch.full((1,), 0.5)
    cond = DenoiserCond(
        features=torch.randn(1, n, 19),
        adjacency=torch.rand(1, n, n),
        graph_pe=torch.randn(1, n, 2),
    )
    model.backbone.eval()
    with torch.no_grad(), model.ema.use_ema(model.backbone):
        expected = model.backbone(z, t, cond)
    with torch.no_grad():
        assert torch.allclose(loaded.backbone(z, t, cond), expected, atol=1e-5)
        assert torch.allclose(from_ckpt.backbone(z, t, cond), expected, atol=1e-5)
    assert loaded.config["graph_pe"] == {"name": "spectral_draw", "k": 2}
    assert loaded.placer(samples=2, refiner=None).samples == 2
