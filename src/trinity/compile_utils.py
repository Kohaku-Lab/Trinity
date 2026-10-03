"""Configurable ``torch.compile``: the whole
model, or the children of chosen containers.

The spec is a dict (or ``None`` to disable)::

    {"mode": "module" | "model" | "off",
     "targets": ["blocks"],     # (module mode) containers whose children get compiled
     "exclude": ["blocks.0"],   # (module mode) dotted child names to skip
     **compile_kwargs}          # forwarded to torch.compile (dynamic=, fullgraph=, ...)
"""

import torch
import torch.nn as nn


def apply_compile(model: nn.Module, spec: dict | None) -> nn.Module:
    """Apply ``torch.compile`` per ``spec``; return the (possibly wrapped) model."""
    if not spec:
        return model
    opts = dict(spec)
    mode = opts.pop("mode", "module")
    targets = opts.pop("targets", ["blocks"])
    exclude = set(opts.pop("exclude", []))
    match mode:
        case "off":
            return model
        case "model":
            return torch.compile(model, **opts)
        case "module":
            _compile_children(model, targets, exclude, opts)
            return model
        case _:
            raise ValueError(f"unknown compile mode {mode!r}")


def _compile_children(model, targets, exclude, opts) -> int:
    """Compile every child of the ``targets``
    containers not in ``exclude``; return the count."""
    compiled = 0
    for target in targets:
        container = model.get_submodule(target) if target else model
        for name, child in container.named_children():
            full = f"{target}.{name}" if target else name
            if full in exclude:
                continue
            child.compile(**opts)
            compiled += 1
    return compiled


def compile_loss_terms(terms: list, spec: dict | None) -> None:
    """Replace the ``fn`` of every loss term that
    has one by its ``torch.compile`` version.

    Uses the compile kwargs of ``spec``; does nothing when ``spec`` is empty or ``off``.
    """
    if not spec:
        return
    opts = dict(spec)
    mode = opts.pop("mode", "module")
    opts.pop("targets", None)
    opts.pop("exclude", None)
    if mode == "off":
        return
    for term in terms:
        fn = getattr(term, "fn", None)
        if fn is not None:
            term.fn = torch.compile(fn, **opts)
