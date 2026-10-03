"""Released Trinity models: export a training
checkpoint, load a model back for sampling.

A release is a directory with two files:

* ``config.json`` -- everything needed to rebuild the model: the architecture, the
  latent parameterization, the framing, the graph PE, the conditioning options and the
  sampler defaults it was evaluated with.
* ``model.safetensors`` -- the EMA weights of the backbone.

``load_model`` accepts a training ``.ckpt``, a release directory, or a Hugging Face repo
id (one release per subfolder, picked with ``scale``) and returns a
:class:`TrinityModel`, whose ``placer()`` builds a ready
:class:`~trinity.solver.DiffusionPlacer`::

    model = load_model("KBlueLeaf/Trinity", scale="flagship", device="cuda")
    placer = model.placer(samples=16)
    placement = placer.solve(instance)
"""

import dataclasses
import json
import pickle
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from trinity.conditioning import CondOptions
from trinity.models.backbone import X0View
from trinity.registry import FRAMING, GRAPH_PE, LATENT_PARAM, REFINER, SAMPLER, build
from trinity.solver import DiffusionPlacer
from trinity.training.trainer import build_arch, build_backbone

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
RELEASE_FORMAT = "trinity-release-v1"

# Module paths of checkpoints written before the package was renamed.
_MODULE_RENAMES = (("kohakudifffs", "trinity.floorplan"), ("kdifffs_gen", "trinity"))


class _RenamingUnpickler(pickle.Unpickler):
    """An unpickler that maps the pre-rename module paths onto ``trinity``."""

    def find_class(self, module: str, name: str):
        for old, new in _MODULE_RENAMES:
            if module == old or module.startswith(old + "."):
                module = new + module[len(old) :]
                break
        return super().find_class(module, name)


class _RenamingPickle:
    """The ``pickle_module`` handed to ``torch.load``."""

    Unpickler = _RenamingUnpickler
    load = staticmethod(pickle.load)


@dataclasses.dataclass
class TrinityModel:
    """A loaded model: the ``x0``-emitting backbone and the components that read it.

    ``config`` is the release config the model was built from.
    """

    backbone: torch.nn.Module
    param: object
    framing: object
    graph_pe: object
    cond_options: CondOptions
    config: dict
    device: str = "cpu"

    def sampler(
        self, num_steps: int | None = None, solver: str | None = None, projections=None
    ):
        """The ODE sampler (defaults from the release
        config; ``projections=None`` = default on)."""
        defaults = self.config["sampler"]
        spec = {
            "name": solver or defaults["name"],
            "num_steps": num_steps or defaults["num_steps"],
            "projections": projections,
        }
        return build(spec, SAMPLER, framing=self.framing)

    def placer(
        self,
        samples: int = 16,
        num_steps: int | None = None,
        solver: str | None = None,
        projections=None,
        refiner: dict | str | None = "closed",
        legalize_portfolio: list | None = None,
        **placer_kwargs,
    ) -> DiffusionPlacer:
        """A ``DiffusionPlacer``: sampler ->
        ``refiner`` (``None`` for none) -> legalizer.

        ``legalize_portfolio=None`` is ``scale_pack``; ``placer_kwargs`` go
        to the placer (``scorer``, ``max_batch``, ``legalize_workers``).
        """
        return DiffusionPlacer(
            self.backbone,
            self.sampler(num_steps, solver, projections),
            build(refiner, REFINER),
            param=self.param,
            samples=samples,
            legalize_portfolio=legalize_portfolio,
            device=self.device,
            graph_pe=self.graph_pe,
            cond_options=self.cond_options,
            **placer_kwargs,
        )


def read_checkpoint(ckpt_path: str | Path) -> tuple[dict, dict[str, torch.Tensor]]:
    """``(hparams, weights)`` of a training checkpoint; ``weights`` is the backbone
    state dict with the EMA weights in place of the parameters (when the run kept an
    EMA)."""
    ckpt = torch.load(
        ckpt_path, map_location="cpu", weights_only=False, pickle_module=_RenamingPickle
    )
    hparams = dict(ckpt["hyper_parameters"])
    state = ckpt["state_dict"]
    weights = {
        key.removeprefix("backbone."): value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    if hparams.get("use_ema", True) and "ema.shadow.0" in state:
        arch = build_arch(hparams.get("preset"), hparams.get("arch_overrides"))
        names = [
            name
            for name, _ in build_backbone(
                arch, hparams.get("backbone")
            ).named_parameters()
        ]
        for i, name in enumerate(names):
            weights[name] = state[f"ema.shadow.{i}"]
    return hparams, weights


def release_config(hparams: dict, extra: dict | None = None) -> dict:
    """The release config of a run's ``hparams`` (plus ``extra`` metadata)."""
    arch = build_arch(hparams.get("preset"), hparams.get("arch_overrides"))
    arch.grad_ckpt = False
    cond_options = hparams.get("cond_options") or CondOptions()
    return {
        "format": RELEASE_FORMAT,
        "arch": dataclasses.asdict(arch),
        "backbone": hparams.get("backbone"),
        "latent_param": hparams.get("latent_param", "s_only"),
        "framing": hparams.get("framing", "ddpm_x0"),
        "graph_pe": hparams.get("graph_pe"),
        "cond_options": dataclasses.asdict(cond_options),
        "sampler": {
            "name": hparams.get("sample_solver", "euler"),
            "num_steps": hparams.get("sample_steps", 16),
        },
        **(extra or {}),
    }


def export_release(
    ckpt_path: str | Path, out_dir: str | Path, extra: dict | None = None
) -> Path:
    """Write ``config.json`` + ``model.safetensors``
    of ``ckpt_path`` into ``out_dir``."""
    hparams, weights = read_checkpoint(ckpt_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = release_config(hparams, extra)
    (out_dir / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n")
    tensors = {key: value.detach().contiguous() for key, value in weights.items()}
    save_file(tensors, str(out_dir / WEIGHTS_NAME))
    return out_dir


def build_model(
    config: dict, weights: dict[str, torch.Tensor], device: str = "cpu"
) -> TrinityModel:
    """A :class:`TrinityModel` from a release ``config`` and backbone ``weights``."""
    arch = build_arch(None, config["arch"])
    backbone = build_backbone(arch, config.get("backbone"))
    backbone.load_state_dict(weights, strict=True)
    backbone = backbone.to(device).eval()
    framing = build(config["framing"], FRAMING)
    if arch.output_kind != "x0":
        backbone = X0View(backbone, framing)
    return TrinityModel(
        backbone=backbone,
        param=build(config["latent_param"], LATENT_PARAM),
        framing=framing,
        graph_pe=build(config.get("graph_pe") or "none", GRAPH_PE),
        cond_options=CondOptions(**config["cond_options"]),
        config=config,
        device=device,
    )


def _release_dir(source: str | Path, scale: str | None, revision: str | None) -> Path:
    """The local release directory of ``source``
    (downloading from the Hub when needed)."""
    path = Path(source)
    if not path.exists():
        patterns = [f"{scale}/*"] if scale else None
        path = Path(
            snapshot_download(str(source), revision=revision, allow_patterns=patterns)
        )
    return path / scale if scale else path


def load_model(
    source: str | Path,
    scale: str | None = None,
    device: str = "cpu",
    revision: str | None = None,
) -> TrinityModel:
    """Load a model from a ``.ckpt``, a release directory, or a Hugging Face repo id.

    ``scale`` names the release subfolder (``"flagship"``, ``"d512"``, ...) of a
    directory or repo holding several releases; ``revision`` pins a Hub revision.
    """
    path = Path(source)
    if path.is_file():
        hparams, weights = read_checkpoint(path)
        return build_model(release_config(hparams), weights, device)
    release = _release_dir(source, scale, revision)
    config = json.loads((release / CONFIG_NAME).read_text())
    weights = load_file(str(release / WEIGHTS_NAME))
    return build_model(config, weights, device)
