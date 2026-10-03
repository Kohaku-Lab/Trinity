"""Trinity: graph-conditioned diffusion for constrained floorplanning.

``trinity.floorplan`` is the problem (data, geometry, scoring, legalization); the rest of the
package is the method (conditioning, models, framings, losses, training, sampling, the
refiner and the placer). Importing this package populates every registry, so
``build(spec, REGISTRY)`` works immediately.
"""

from trinity import (
    augment_ops,  # noqa: F401  (register: augment ops)
    conditioning,  # noqa: F401  (register: graph PE builders)
    framing,  # noqa: F401  (register: framings, time samplers)
    latent_param,  # noqa: F401  (register: latent parameterizations)
    losses,  # noqa: F401  (register: denoise + aux losses)
    models,  # noqa: F401  (register: norm / mlp / attention / backbones)
    sampling,  # noqa: F401  (register: samplers, projections, refiners)
)
from trinity.registry import (
    ATTENTION,
    AUGMENT_OP,
    FRAMING,
    GRAPH_PE,
    LATENT_PARAM,
    LOSS,
    MLP,
    NORM,
    PROJECTION,
    REFINER,
    SAMPLER,
    TIME_SAMPLER,
    Registry,
    build,
    resolve,
)

__all__ = [
    "build",
    "resolve",
    "Registry",
    "NORM",
    "MLP",
    "ATTENTION",
    "LATENT_PARAM",
    "GRAPH_PE",
    "AUGMENT_OP",
    "FRAMING",
    "TIME_SAMPLER",
    "LOSS",
    "SAMPLER",
    "PROJECTION",
    "REFINER",
    "framing",
    "models",
    "losses",
    "sampling",
]
