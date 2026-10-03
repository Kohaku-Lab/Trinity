"""Small helpers shared across the package: dotted-path import, adaLN modulation, and
schedule step filling."""

import importlib
from collections.abc import Callable
from typing import Any


def import_class(path: str) -> Callable[..., Any]:
    """Import and return the object named by a dotted path ``pkg.module.name``."""
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ValueError(f"{path!r} is not a dotted path (expected 'pkg.module.name')")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def modulate(x: Any, shift: Any, scale: Any) -> Any:
    """adaLN modulation ``x * (1 + scale) + shift`` (token-broadcast over dim 1)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def autofill_schedule_steps(
    scheduler_config: dict,
    train_steps: int,
    warmup_ratio: float = 0.0,
    end_from_steps: bool = True,
    warmup_from_steps: bool = True,
) -> dict:
    """Set an AnySchedule sub-config's ``end`` (and ``warmup = warmup_ratio *
    train_steps``) from ``train_steps``; mutates and returns ``scheduler_config``."""
    if end_from_steps:
        scheduler_config["end"] = train_steps
    if warmup_from_steps:
        scheduler_config["warmup"] = int(train_steps * warmup_ratio)
    return scheduler_config
