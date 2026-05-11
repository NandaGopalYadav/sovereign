"""Territory graph and map registry.

A map is an undirected graph of territories. Each territory carries:
    * a fixed `home_of`   — Nation indicator: 0 invader, 1 defender, 2 neutral, 3 contested
    * a `resource_value`  ∈ [0, 1]
    * a `strategic_value` ∈ [0, 1]

To register a new map, decorate a function returning a :class:`MapSpec` with
:func:`register_map`. The default 9-territory map is the one used in the experiments
section of the spec; an alternate "frontier12" topology is provided for ablations on
graph structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import networkx as nx
import numpy as np


# Nation identifiers. Used as integers everywhere downstream so numpy can hold them.
INVADER = 0
DEFENDER = 1
NEUTRAL = 2
CONTESTED = 3


@dataclass(frozen=True)
class Territory:
    name: str
    home_of: int           # initial controller (INVADER, DEFENDER, NEUTRAL, CONTESTED)
    resource_value: float
    strategic_value: float


@dataclass(frozen=True)
class MapSpec:
    name: str
    territories: tuple[Territory, ...]
    edges: tuple[tuple[int, int], ...]

    @property
    def n(self) -> int:
        return len(self.territories)

    def to_graph(self) -> nx.Graph:
        g = nx.Graph()
        for i, t in enumerate(self.territories):
            g.add_node(
                i,
                name=t.name,
                home_of=t.home_of,
                resource_value=t.resource_value,
                strategic_value=t.strategic_value,
            )
        g.add_edges_from(self.edges)
        return g

    def adjacency(self) -> np.ndarray:
        """Return a dense (n, n) symmetric adjacency matrix."""
        a = np.zeros((self.n, self.n), dtype=np.int8)
        for u, v in self.edges:
            a[u, v] = 1
            a[v, u] = 1
        return a


# --------------------------------------------------------------------------------------
# Map registry
# --------------------------------------------------------------------------------------


_REGISTRY: dict[str, Callable[[], MapSpec]] = {}


def register_map(name: str) -> Callable[[Callable[[], MapSpec]], Callable[[], MapSpec]]:
    def deco(fn: Callable[[], MapSpec]) -> Callable[[], MapSpec]:
        if name in _REGISTRY:
            raise ValueError(f"Map already registered: {name}")
        _REGISTRY[name] = fn
        return fn

    return deco


def get_map(name: str) -> MapSpec:
    if name not in _REGISTRY:
        raise KeyError(f"No such map: {name}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def list_maps() -> list[str]:
    return sorted(_REGISTRY)


# --------------------------------------------------------------------------------------
# rulebook9 — 1 home per nation + 6 contested, matching Sovereign.pdf Section 3.3.
# --------------------------------------------------------------------------------------


@register_map("rulebook9")
def _rulebook9() -> MapSpec:
    territories = (
        Territory("I",  INVADER,   0.8, 0.9),  # 0 — invader home
        Territory("D",  DEFENDER,  0.7, 0.9),  # 1 — defender home
        Territory("N",  NEUTRAL,   0.6, 0.7),  # 2 — neutral home
        Territory("C1", CONTESTED, 0.4, 0.5),  # 3
        Territory("C2", CONTESTED, 0.5, 0.6),  # 4
        Territory("C3", CONTESTED, 0.4, 0.8),  # 5
        Territory("C4", CONTESTED, 0.6, 0.7),  # 6
        Territory("C5", CONTESTED, 0.5, 0.6),  # 7
        Territory("C6", CONTESTED, 0.7, 0.5),  # 8
    )
    edges = (
        (0, 3), (0, 4),
        (1, 7), (1, 8),
        (2, 5), (2, 6),
        (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
        (3, 5), (4, 6), (5, 7),
    )
    return MapSpec("rulebook9", territories, edges)


# --------------------------------------------------------------------------------------
# default9 — legacy simplified map: 3 invader-home + 3 defender-home + 3 contested.
# --------------------------------------------------------------------------------------


@register_map("default9")
def _default9() -> MapSpec:
    territories = (
        Territory("Capital-I", INVADER, 0.4, 0.9),       # 0
        Territory("Industrial-I", INVADER, 0.8, 0.5),    # 1
        Territory("Border-I", INVADER, 0.3, 0.4),        # 2
        Territory("Border-D", DEFENDER, 0.3, 0.4),       # 3
        Territory("Industrial-D", DEFENDER, 0.7, 0.5),   # 4
        Territory("Capital-D", DEFENDER, 0.4, 0.9),      # 5
        Territory("Steppe", CONTESTED, 0.5, 0.6),        # 6
        Territory("Coast", CONTESTED, 0.6, 0.7),         # 7
        Territory("Highlands", CONTESTED, 0.4, 0.8),     # 8
    )
    edges = (
        (0, 1), (0, 2), (1, 2),
        (2, 6), (2, 8),
        (3, 4), (3, 5), (4, 5),
        (3, 6), (3, 7),
        (6, 7), (7, 8), (6, 8),
    )
    return MapSpec("default9", territories, edges)


# --------------------------------------------------------------------------------------
# frontier12 — alternate topology with longer fronts and more contested ground.
# --------------------------------------------------------------------------------------


@register_map("frontier12")
def _frontier12() -> MapSpec:
    territories = (
        Territory("I-Cap", INVADER, 0.4, 0.9),         # 0
        Territory("I-Ind1", INVADER, 0.7, 0.5),        # 1
        Territory("I-Ind2", INVADER, 0.6, 0.5),        # 2
        Territory("I-Border", INVADER, 0.3, 0.4),      # 3
        Territory("D-Border", DEFENDER, 0.3, 0.5),     # 4
        Territory("D-Ind", DEFENDER, 0.7, 0.5),        # 5
        Territory("D-Cap", DEFENDER, 0.4, 0.9),        # 6
        Territory("Marsh", CONTESTED, 0.4, 0.4),       # 7
        Territory("Forest", CONTESTED, 0.5, 0.6),      # 8
        Territory("Plain", CONTESTED, 0.5, 0.5),       # 9
        Territory("Mtn", CONTESTED, 0.3, 0.8),         # 10
        Territory("Coast", CONTESTED, 0.6, 0.7),       # 11
    )
    edges = (
        (0, 1), (0, 2), (1, 3), (2, 3),
        (4, 5), (5, 6), (4, 6),
        (3, 7), (3, 8),
        (4, 8), (4, 9),
        (7, 8), (8, 9), (9, 10), (10, 11),
        (7, 11), (8, 10),
    )
    return MapSpec("frontier12", territories, edges)
