"""The component registries of the problem side and the ``build(spec)`` resolver.

:func:`build` resolves a *spec* once, at build time; the built object is held as a plain
attribute and called directly afterwards. A spec is one of:

* an already-built instance               -> returned unchanged
* a class                                 -> instantiated with the build-site kwargs
* a registry key string ``"scale_pack"``  -> looked up in ``registry``, then called
* a dotted path ``"pkg.module.Cls"``      -> imported, then called
* a dict ``{"name": <key|path>, **kw}``   -> resolved, called with ``kw`` + kwargs
"""

from collections.abc import Callable
from typing import Any

from trinity.floorplan.utils import import_class


class Registry:
    """A named string -> class/callable map with a decorator-based ``register``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, key: str | None = None) -> Callable[[Callable], Callable]:
        """Return a decorator registering the wrapped object under ``key``.

        ``key`` defaults to the object's ``__name__``.
        """

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


LEGALIZER = Registry("legalizer")
RENDERER = Registry("renderer")
SCORER = Registry("scorer")
JITTER = Registry("jitter")
# Non-learned placers: ``(FloorplanInstance) -> Placement``.
SOLVER = Registry("solver")


def _resolve(name: str, registry: Registry | None) -> Callable[..., Any]:
    if "." in name:
        return import_class(name)
    if registry is None:
        raise ValueError(f"{name!r} is not a dotted path and no registry was provided")
    return registry.get(name)


def resolve(name: str, registry: Registry | None = None) -> Callable[..., Any]:
    """Return the object a registry key or dotted path names, without calling it."""
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
