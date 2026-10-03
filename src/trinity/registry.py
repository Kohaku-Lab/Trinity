"""The component registries of the model side and the ``build(spec)`` resolver.

``build`` resolves a *spec* once, at build time; the built object is held as a plain
attribute and called directly in the hot loop. ``trinity.floorplan.registry`` holds the
registries of the problem side (legalizers, scorers, renderers).

A spec is one of: an instance (returned as-is), a class (instantiated), a registry-key
string, a dotted import path, or a ``{"name": <key|path>, **kw}`` dict.
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


NORM = Registry("norm")
MLP = Registry("mlp")
ATTENTION = Registry("attention")
TIME_COND = Registry("time_cond")
GRAPH_PE = Registry("graph_pe")
GRAPH_MIX = Registry("graph_mix")
LATENT_PARAM = Registry("latent_param")
AUGMENT_OP = Registry("augment_op")
FRAMING = Registry("framing")
TIME_SAMPLER = Registry("time_sampler")
LOSS = Registry("loss")
SAMPLER = Registry("sampler")
PROJECTION = Registry("projection")
REFINER = Registry("refiner")


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
