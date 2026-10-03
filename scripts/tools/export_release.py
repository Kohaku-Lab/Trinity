"""Export training checkpoints as releases (``config.json`` + ``model.safetensors``).

Each entry of ``RELEASES`` maps a release name to a checkpoint; the release is written to
``OUT_DIR/<name>/`` and can be loaded with ``trinity.hub.load_model``::

    kogine run scripts/tools/export_release.py \\
        --set 'RELEASES={"flagship": "outputs/train/flagship/checkpoints/last.ckpt"}'
"""

from trinity.hub import export_release, load_model

# Release name -> training checkpoint path.
RELEASES: dict[str, str] = {}
OUT_DIR: str = "outputs/release"
# Reload every export and print its config as a check.
VERIFY: bool = True


def main() -> None:
    if not RELEASES:
        raise SystemExit("RELEASES is empty; map a release name to a checkpoint path")
    for name, ckpt in RELEASES.items():
        out = export_release(ckpt, f"{OUT_DIR}/{name}", extra={"name": name})
        print(f"{name}: {ckpt} -> {out}")
        if VERIFY:
            model = load_model(out)
            n_params = sum(p.numel() for p in model.backbone.parameters())
            print(
                f"  reloaded: {n_params / 1e6:.1f} M parameters, arch {model.config['arch']}"
            )


if __name__ == "__main__":
    main()
