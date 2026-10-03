"""Gates of the direct-regression baseline: heads, regressor, trainer and placer."""

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import trinity  # noqa: F401  (register shared losses / refiners)
import trinity_baselines  # noqa: F401  (register heads)
from trinity.data import SameNBatchSampler, collate_same_n, validation_dataset
from trinity.floorplan.data import find_floorset_root, load_validation_case
from trinity.floorplan.scoring import validate
from trinity.models import DenoiserCond, get_preset
from trinity.sampling import AnchorClamp
from trinity_baselines.models import DirectRegressor, LatentHead
from trinity_baselines.registry import HEAD, build
from trinity_baselines.solver import RegressionPlacer
from trinity_baselines.training import RegressorTrainer, ValidationCallback


def have_data() -> bool:
    try:
        find_floorset_root(allow_download=False)
        return True
    except FileNotFoundError:
        return False


needs_data = pytest.mark.skipif(
    not have_data(), reason="FloorSet validation set not present"
)


def test_head_registry_populated():
    assert {"z", "xywh"} <= set(HEAD.keys())


@pytest.mark.parametrize("name,out_dim", [("z", 3), ("xywh", 4)])
def test_head_decode_and_zs_roundtrip(name, out_dim):
    head = build(name, HEAD)
    assert head.out_dim == out_dim
    z_s = torch.randn(2, 6, 3)
    z_s[..., 2] = z_s[..., 2].clamp(-3, 3)
    area_norm = torch.rand(2, 6) * 0.2 + 0.05
    target = head.target_from_zs(z_s, area_norm)
    assert target.shape == (2, 6, out_dim)
    xywh = head.to_xywh(target, area_norm)
    centers = xywh[..., :2] + xywh[..., 2:] / 2
    assert (centers - z_s[..., :2]).abs().max() < 1e-4
    zs_back = head.to_zs(target, area_norm)
    assert (zs_back[..., 2] - z_s[..., 2]).abs().max() < 1e-4


def test_z_head_area_exact():
    """The ``z`` head decodes with ``w * h == area``."""
    head = LatentHead()
    z_s = torch.randn(1, 5, 3)
    area_norm = torch.rand(1, 5) * 0.2 + 0.05
    xywh = head.to_xywh(head.target_from_zs(z_s, area_norm), area_norm)
    wh = xywh[..., 2] * xywh[..., 3]
    assert (wh - area_norm).abs().max() < 1e-5


@pytest.mark.parametrize("name", ["z", "xywh"])
def test_regressor_shapes_and_equivariance(name):
    head = build(name, HEAD)
    reg = DirectRegressor(get_preset("DiT-S", feature_dim=19), head).eval()
    n = 12
    feats = torch.randn(1, n, 19)
    adj = torch.rand(1, n, n)
    adj = adj + adj.transpose(1, 2)
    perm = torch.randperm(n)
    permuted = DenoiserCond(features=feats[:, perm], adjacency=adj[:, perm][:, :, perm])
    with torch.no_grad():
        o1 = reg(DenoiserCond(features=feats, adjacency=adj))
        o2 = reg(permuted)
    assert o1.shape == (1, n, head.out_dim)
    assert (o1[:, perm] - o2).abs().max() < 1e-4


@pytest.mark.parametrize("name", ["z", "xywh"])
def test_regressor_output_is_unclamped(name):
    """The regressor never overwrites its output with anchor values."""
    head = build(name, HEAD)
    reg = DirectRegressor(get_preset("DiT-S", feature_dim=19), head).eval()
    n = 6
    area_norm = torch.rand(1, n) * 0.2 + 0.05
    anchor_zs = torch.randn(1, n, 3)
    mask3 = torch.zeros(1, n, 3)
    mask3[0, 0, :] = 1.0
    value, mask_head = head.anchor(anchor_zs, mask3, area_norm)
    feats = torch.randn(1, n, 19)
    adj = torch.zeros(1, n, n)
    anchored = DenoiserCond(
        features=feats, adjacency=adj, anchor_z=value, anchor_mask=mask_head
    )
    with torch.no_grad():
        plain = reg(DenoiserCond(features=feats, adjacency=adj))
        with_anchor = reg(anchored)
    assert torch.equal(plain, with_anchor)


def test_anchor_clamp_projection_enforces():
    """The ``anchor_clamp`` projection overwrites masked coordinates of a latent."""
    n = 6
    z = torch.randn(1, n, 3)
    anchor = torch.randn(1, n, 3)
    mask = torch.zeros(1, n, 3)
    mask[0, 0, :] = 1.0
    cond = DenoiserCond(
        features=torch.zeros(1, n, 19), anchor_z=anchor, anchor_mask=mask
    )
    out = AnchorClamp()(z, cond)
    assert torch.equal(out[0, 0], anchor[0, 0])
    assert torch.equal(out[0, 1:], z[0, 1:])


@needs_data
@pytest.mark.parametrize("head", ["z", "xywh"])
def test_trainer_step_runs(head):
    ds = validation_dataset(allow_download=False)
    sampler = SameNBatchSampler(ds.block_counts, batch_size=2, shuffle=True, seed=0)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_same_n)
    model = RegressorTrainer(
        preset="DiT-S", arch_overrides={"feature_dim": 19}, head=head, use_ema=False
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
@pytest.mark.parametrize("head", ["z", "xywh"])
def test_solver_produces_layout(head):
    inst = load_validation_case(21)
    model = RegressorTrainer(
        preset="DiT-S",
        arch_overrides={"feature_dim": 19},
        head=head,
        use_ema=True,
        refiner={"name": "closed", "steps": 10, "compile_loop": False},
    )
    results = model.solve_cases([inst])
    assert len(results) == 1
    _inst, _score, placement = results[0]
    assert np.asarray(placement.to_tuples()).shape == (21, 4)
    assert validate(placement, "full").score is not None


@needs_data
def test_regression_placer_single_candidate():
    """The regression placer yields one candidate cost per case."""
    inst = load_validation_case(21)
    model = RegressorTrainer(
        preset="DiT-S", arch_overrides={"feature_dim": 19}, head="z"
    )
    model.solve_cases([inst])
    assert isinstance(model._make_placer(), RegressionPlacer)
    assert len(model.last_candidate_costs[0]) == 1
    callback = ValidationCallback(
        shard_fn=lambda rank, world: [inst][rank::world], every_n_steps=1, num_render=0
    )
    assert callback.shard_fn(0, 1) == [inst]
