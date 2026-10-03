"""The registries of the baseline package and its ``build(spec)`` resolver.

* :data:`HEAD` -- the output heads of the direct regressor (``z`` / ``xywh``).
* :data:`BASELINE_BACKBONE` -- the ported denoisers of published placers.

Losses, latent parameterizations and refiners come from ``trinity.registry``. A spec is
an instance (returned as-is), a class (instantiated), a registry key, a dotted import
path, or a ``{"name": <key|path>, **kwargs}`` dict.
"""

from collections.abc import Callable
from typing import Any

from trinity.utils import import_class


class Registry:
    """A named string -> class/callable map with a decorator-based ``register``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, key: str | None = None) -> Callable[[Callable], Callable]:
        def decorator(obj: Callable[..., Any]) -> Callable[..., Any]:
            name = key or obj.__name__
            if name in self._items:
                raise KeyError(f"{self.name!r} already has an entry named {name!r}")
            self._items[name] = obj
            return obj

        return decorator

    def get(self, key: str) -> Callable[..., Any]:
        if key not in self._items:
            raise KeyError(f"unknown {self.name} {key!r}; registered: {self.keys()}")
        return self._items[key]

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def keys(self) -> list[str]:
        return sorted(self._items)


HEAD = Registry("head")
BASELINE_BACKBONE = Registry("baseline_backbone")


def _resolve(name: str, registry: Registry | None) -> Callable[..., Any]:
    if "." in name:
        return import_class(name)
    if registry is None:
        raise ValueError(f"{name!r} is not a dotted path and no registry was provided")
    return registry.get(name)


def resolve(name: str, registry: Registry | None = None) -> Callable[..., Any]:
    """Look up a key / dotted path and return the object *without* calling it."""
    return _resolve(name, registry)


def build(spec: Any, registry: Registry | None = None, **kwargs: Any) -> Any:
    """Resolve ``spec`` to a concrete object, passing ``kwargs`` at construction."""
    match spec:
        case None:
            return None
        case dict():
            opts = dict(spec)
            name = opts.pop("name")
            return _resolve(name, registry)(**opts, **kwargs)
        case str():
            return _resolve(spec, registry)(**kwargs)
        case type():
            return spec(**kwargs)
        case _ if hasattr(spec, "build") and callable(spec.build):
            return spec.build()
        case _:
            return spec
