"""Small helpers shared across ``trinity.floorplan``."""

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
