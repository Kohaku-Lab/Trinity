"""Training data: the ground-truth latent ``z0`` plus conditioning, and two batching modes.

The denoiser trains on the ground-truth layout encoded to the area-preserving latent
``z = (cx/s, cy/s, rho)``. Two batching strategies, selected by config:

* ``same_n`` -- batches of equal block count (:class:`SameNBatchSampler`), stacked with no
  padding; the sampler shards batches across DDP ranks itself.
* ``pad_to_max`` -- every case padded to ``max_n`` (:class:`PadCollate`) with a
  ``token_mask``; padded tokens are masked out of attention, the loss and the geometry.
  DDP uses Lightning's ``DistributedSampler``.

:func:`build_loader` returns the ``DataLoader`` and whether Lightning should inject its
distributed sampler.
"""

from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info

from trinity.augment_ops import build_pipeline
from trinity.conditioning import DEFAULT_COND_OPTIONS, build_conditioning
from trinity.data.splits import DataSplits, load_or_make_splits
from trinity.floorplan.data import (
    LanceFloorplanStore,
    find_train_lance,
    load_validation_set,
)
from trinity.floorplan.parameterize import xywh_to_z
from trinity.floorplan.scoring.soft import soft_denominator
from trinity.losses.constraint import MAX_GROUP_SIZE
from trinity.registry import LATENT_PARAM, build

# Attention bias on padded keys (finite, so an all-padding row stays well-defined).
PAD_BIAS = -1e4

# Upper bound on p2b pin edges per case; ``PadCollate`` pads every case to it.
MAX_PINS = 8192


def _check_group_sizes(inst) -> None:
    """Raise if a cluster or MIB group exceeds the loss terms' ``MAX_GROUP_SIZE`` bound."""
    for ids in (inst.cluster_id, inst.mib_id):
        ids = ids[ids > 0]
        if ids.size and int(np.bincount(ids).max()) > MAX_GROUP_SIZE:
            raise ValueError(
                f"group of {int(np.bincount(ids).max())} blocks exceeds MAX_GROUP_SIZE={MAX_GROUP_SIZE}"
            )


def encode_instance(
    inst, param, pipeline, rng, cond_options=DEFAULT_COND_OPTIONS
) -> dict[str, torch.Tensor]:
    """Encode one ``FloorplanInstance`` to the tensor dict the collates consume.

    The augment ``pipeline`` transforms the instance first; the ground truth and the anchors
    are mapped through ``param.to_latent``. The dict holds the conditioning of
    ``build_conditioning`` (including the raw b2b adjacency ``adj_raw`` for the batched graph
    PE), ``z0``, the group ids and mobility masks the aux terms read, the pin targets in
    ``/s`` units, and the ground-truth area / HPWL baselines in normalized units.
    """
    inst = pipeline.apply(inst, rng)
    _check_group_sizes(inst)
    cond = build_conditioning(inst, cond_options)
    z_s = xywh_to_z(inst.gt_positions, inst.area_targets, inst.s)
    cond["z0"] = torch.from_numpy(param.to_latent(z_s).astype(np.float32))
    cond["anchor_z"] = torch.from_numpy(param.to_latent(cond["anchor_z"].numpy()))
    cond["area_targets"] = torch.from_numpy(inst.area_targets.astype(np.float32))
    cond["scale"] = torch.tensor(float(inst.s))
    cond["cluster_id"] = torch.from_numpy(inst.cluster_id.astype(np.int64))
    cond["boundary_code"] = torch.from_numpy(inst.boundary_code.astype(np.int64))
    cond["mib_id"] = torch.from_numpy(inst.mib_id.astype(np.int64))
    # Mobility: preplaced blocks freeze position and shape, fixed blocks freeze shape.
    cond["mob_pos"] = torch.from_numpy((~inst.is_preplaced).astype(np.float32))
    cond["mob_shape"] = torch.from_numpy(
        (~(inst.is_fixed | inst.is_preplaced)).astype(np.float32)
    )
    # Ground-truth bbox area (metrics[0]) and HPWL (metrics[6] + metrics[7]), normalized.
    m = inst.metrics
    scale = float(inst.s)
    has_metrics = m is not None and len(m) >= 8
    area_base = float(m[0]) / scale**2 if has_metrics and m[0] > 0 else 0.0
    hpwl_base = float(m[6] + m[7]) / scale if has_metrics and m[6] > 0 else 0.0
    cond["pin_xy_n"] = cond["pin_xy"] / scale
    cond["pin_edge_xy_n"] = cond["pin_edge_xy"] / scale
    cond["area_base"] = torch.tensor(area_base, dtype=torch.float32)
    cond["hpwl_base"] = torch.tensor(hpwl_base, dtype=torch.float32)
    cond["n_soft"] = torch.tensor(float(soft_denominator(inst)), dtype=torch.float32)
    cond["block_count"] = int(inst.block_count)
    return cond


class FloorplanLatentDataset(Dataset):
    """A list of instances encoded once and held in memory (the small validation set)."""

    def __init__(
        self, instances, param, pipeline, cond_options=DEFAULT_COND_OPTIONS
    ) -> None:
        rng = np.random.default_rng(0)
        self.samples = [
            encode_instance(i, param, pipeline, rng, cond_options) for i in instances
        ]
        self.block_counts = [s["block_count"] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


class LanceLayoutDataset(Dataset):
    """A lazy dataset over a subset of Lance row ids, encoding each layout on access.

    Each DataLoader worker opens its own Lance handle on first access. ``block_counts``
    (for same-n bucketing) is read up front from the store's ``n`` column.
    """

    def __init__(
        self, lance_path, row_ids, param, pipeline, cond_options=DEFAULT_COND_OPTIONS
    ) -> None:
        self.lance_path = lance_path
        self.row_ids = list(row_ids)
        self.param = param
        self.pipeline = pipeline
        self.cond_options = cond_options
        self._store: LanceFloorplanStore | None = None
        self._rng: np.random.Generator | None = None
        all_counts = LanceFloorplanStore(lance_path).block_counts
        self.block_counts = [int(all_counts[i]) for i in self.row_ids]

    def _ensure_store(self) -> "LanceFloorplanStore":
        if self._store is None:
            self._store = LanceFloorplanStore(self.lance_path)
        return self._store

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._rng is None:
            info = get_worker_info()
            self._rng = np.random.default_rng(None if info is None else info.seed)
        inst = self._ensure_store().instance(self.row_ids[idx])
        return encode_instance(
            inst, self.param, self.pipeline, self._rng, self.cond_options
        )


class SameNBatchSampler(Sampler):
    """Yield batches of indices that share a block count, sharded across DDP ranks.

    ``rank`` / ``world_size`` default to the live ``torch.distributed`` group (or a single
    process); each rank takes every ``world_size``-th batch of the epoch's shuffled list.
    """

    def __init__(
        self,
        block_counts: list[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        if world_size is None or rank is None:
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            else:
                world_size, rank = 1, 0
        self.rank = rank
        self.world_size = world_size
        self.buckets: dict[int, list[int]] = {}
        for idx, n in enumerate(block_counts):
            self.buckets.setdefault(n, []).append(idx)
        self._epoch = 0

    def _all_batches(self, rng) -> list[list[int]]:
        batches = []
        for indices in self.buckets.values():
            order = list(indices)
            if self.shuffle:
                rng.shuffle(order)
            for i in range(0, len(order), self.batch_size):
                batches.append(order[i : i + self.batch_size])
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        batches = self._all_batches(rng)
        yield from batches[self.rank :: self.world_size]

    def __len__(self) -> int:
        total = sum(
            (len(v) + self.batch_size - 1) // self.batch_size
            for v in self.buckets.values()
        )
        return (total - self.rank + self.world_size - 1) // self.world_size


def _pad_pin_edges(
    batch: list[dict], p_max: int | None = None
) -> dict[str, torch.Tensor]:
    """Pad every sample's pin-edge list to ``p_max`` (default: the batch's longest; weight 0, block 0)."""
    longest = max(int(b["pin_edge_w"].shape[0]) for b in batch)
    if p_max is None:
        p_max = longest
    elif longest > p_max:
        raise ValueError(f"a case has {longest} pin edges, above MAX_PINS={p_max}")
    xy = torch.zeros(len(batch), p_max, 2)
    w = torch.zeros(len(batch), p_max)
    blk = torch.zeros(len(batch), p_max, dtype=torch.long)
    for i, b in enumerate(batch):
        p = int(b["pin_edge_w"].shape[0])
        xy[i, :p] = b["pin_edge_xy_n"]
        w[i, :p] = b["pin_edge_w"]
        blk[i, :p] = b["pin_edge_block"]
    return {"pin_edge_xy_n": xy, "pin_edge_w": w, "pin_edge_block": blk}


def collate_same_n(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Stack a list of same-``n`` samples; the token mask is all-ones (no padding)."""
    out = {
        "z0": torch.stack([b["z0"] for b in batch]),
        "features": torch.stack([b["features"] for b in batch]),
        "adjacency": torch.stack([b["adjacency"] for b in batch]),
        "adj_raw": torch.stack([b["adj_raw"] for b in batch]),
        "anchor_z": torch.stack([b["anchor_z"] for b in batch]),
        "anchor_mask": torch.stack([b["anchor_mask"] for b in batch]),
        "area_targets": torch.stack([b["area_targets"] for b in batch]),
        "scale": torch.stack([b["scale"] for b in batch]),
        "area_base": torch.stack([b["area_base"] for b in batch]),
        "hpwl_base": torch.stack([b["hpwl_base"] for b in batch]),
        "n_soft": torch.stack([b["n_soft"] for b in batch]),
        "cluster_id": torch.stack([b["cluster_id"] for b in batch]),
        "boundary_code": torch.stack([b["boundary_code"] for b in batch]),
        "mib_id": torch.stack([b["mib_id"] for b in batch]),
        "mib_soft_group": torch.stack([b["mib_soft_group"] for b in batch]),
        "mob_pos": torch.stack([b["mob_pos"] for b in batch]),
        "mob_shape": torch.stack([b["mob_shape"] for b in batch]),
        "pin_xy": torch.stack([b["pin_xy"] for b in batch]),
        "pin_xy_n": torch.stack([b["pin_xy_n"] for b in batch]),
        "pin_w": torch.stack([b["pin_w"] for b in batch]),
    }
    out.update(_pad_pin_edges(batch))
    out["token_mask"] = torch.ones(out["z0"].shape[:2])
    return out


class PadCollate:
    """Pad every case to ``max_n`` tokens (zeros) and stack, with ``token_mask`` 1 on real
    blocks and 0 on padding; pin edges are padded to ``MAX_PINS``."""

    def __init__(self, max_n: int) -> None:
        self.max_n = max_n

    def _pad_tokens(self, t: torch.Tensor) -> torch.Tensor:
        pad = self.max_n - t.shape[0]
        if pad <= 0:
            return t[: self.max_n]
        zeros = t.new_zeros((pad, *t.shape[1:]))
        return torch.cat([t, zeros], dim=0)

    def _pad_adjacency(self, a: torch.Tensor) -> torch.Tensor:
        out = a.new_zeros((self.max_n, self.max_n))
        n = min(a.shape[0], self.max_n)
        out[:n, :n] = a[:n, :n]
        return out

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        token_keys = (
            "z0",
            "features",
            "anchor_z",
            "anchor_mask",
            "area_targets",
            "cluster_id",
            "boundary_code",
            "mib_id",
            "mib_soft_group",
            "mob_pos",
            "mob_shape",
            "pin_xy",
            "pin_xy_n",
            "pin_w",
        )
        out = {
            k: torch.stack([self._pad_tokens(b[k]) for b in batch]) for k in token_keys
        }
        out["adjacency"] = torch.stack(
            [self._pad_adjacency(b["adjacency"]) for b in batch]
        )
        out["adj_raw"] = torch.stack([self._pad_adjacency(b["adj_raw"]) for b in batch])
        out["scale"] = torch.stack([b["scale"] for b in batch])
        for k in ("area_base", "hpwl_base", "n_soft"):
            out[k] = torch.stack([b[k] for b in batch])
        out.update(_pad_pin_edges(batch, MAX_PINS))
        mask = torch.zeros(len(batch), self.max_n)
        for i, b in enumerate(batch):
            mask[i, : b["block_count"]] = 1.0
        out["token_mask"] = mask
        return out


def _resolve(latent_param, augment):
    """Build the latent parameterization and the augment pipeline from their specs."""
    param = build(latent_param, LATENT_PARAM)
    pipeline = build_pipeline(augment)
    return param, pipeline


def validation_dataset(
    root: str | None = None,
    allow_download: bool = True,
    latent_param="s_only",
    cond_options=DEFAULT_COND_OPTIONS,
) -> FloorplanLatentDataset:
    """Build a dataset from the 100-case validation set (handy for overfit checks)."""
    param, pipeline = _resolve(latent_param, None)
    cases = load_validation_set(root=root, allow_download=allow_download)
    return FloorplanLatentDataset(cases, param, pipeline, cond_options)


def train_dataset(
    lance_path: str | None = None,
    *,
    per_n_k: int = 10,
    random_size: int = 1000,
    seed: int = 20090220,
    split_cache: str | None = None,
    latent_param="s_only",
    augment=None,
    cond_options=DEFAULT_COND_OPTIONS,
) -> tuple[LanceLayoutDataset, DataSplits]:
    """The Lance-backed train dataset over the train rows, and the :class:`DataSplits`.

    The split is derived deterministically from ``seed`` and cached as JSON next to the
    Lance set (or at ``split_cache``).
    """
    path = str(find_train_lance(lance_path))
    store = LanceFloorplanStore(path)
    cache = split_cache or str(Path(path).parent / "splits.json")
    splits = load_or_make_splits(
        store.block_counts, cache, per_n_k=per_n_k, random_size=random_size, seed=seed
    )
    param, pipeline = _resolve(latent_param, augment)
    return LanceLayoutDataset(path, splits.train, param, pipeline, cond_options), splits


def build_loader(
    dataset: Dataset,
    batching: str,
    batch_size: int,
    *,
    max_n: int = 128,
    shuffle: bool = True,
    seed: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, bool]:
    """The train ``DataLoader`` of ``batching`` (``"same_n"`` | ``"pad_to_max"``).

    Returns ``(loader, use_distributed_sampler)``: ``False`` for ``same_n`` (it shards in
    its own sampler), ``True`` for ``pad_to_max``. Workers use the ``forkserver`` start
    method; ``pin_memory`` returns page-locked batches.
    """
    worker_kw = (
        {"multiprocessing_context": "forkserver", "persistent_workers": True}
        if num_workers > 0
        else {}
    )
    worker_kw["pin_memory"] = pin_memory
    if batching == "same_n":
        sampler = SameNBatchSampler(
            dataset.block_counts, batch_size=batch_size, shuffle=shuffle, seed=seed
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_same_n,
            num_workers=num_workers,
            **worker_kw,
        )
        return loader, False
    if batching == "pad_to_max":
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=PadCollate(max_n),
            num_workers=num_workers,
            drop_last=True,
            **worker_kw,
        )
        return loader, True
    raise ValueError(
        f"unknown batching {batching!r}; expected 'same_n' or 'pad_to_max'"
    )
