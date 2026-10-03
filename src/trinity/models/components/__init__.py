"""Swappable backbone components and the embedders.

Importing this package registers the ``NORM``, ``MLP``, ``ATTENTION``, ``TIME_COND`` and
``GRAPH_MIX`` variants.
"""

from trinity.models.components import (
    attention,  # noqa: F401  (register: attention variants)
    graph_mix,  # noqa: F401  (register: graph mix variants)
    mlp,  # noqa: F401  (register: mlp variants)
    norm,  # noqa: F401  (register: norm variants)
    time_cond,  # noqa: F401  (register: time-conditioning strategies)
)
from trinity.models.components.embed import InputEmbed, TimestepEmbedder

__all__ = ["InputEmbed", "TimestepEmbedder"]
