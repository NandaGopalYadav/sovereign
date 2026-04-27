"""Rule-based defender policy.

Heuristic, in plain English:

    1. If any home territory is occupied by the invader, all available defender ground
       units march to retake the closest occupied home territory (counter-offensive).
    2. Else if an adjacent contested or non-home territory has invader strength
       exceeding our local strength, we counter-strike there to deny build-up.
    3. Else we hold and reinforce the most strategically valuable home territory.

The policy returns a target territory plus a discrete tag in
{HOLD, COUNTER_HOME, COUNTER_STRIKE, REINFORCE} so the env logs it clearly.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from sovereign.env.map import DEFENDER, INVADER, MapSpec


HOLD = 0
COUNTER_HOME = 1
COUNTER_STRIKE = 2
REINFORCE = 3


@dataclass
class DefenderDecision:
    tag: int
    target: int


class DefenderPolicy:
    """Reactive rule-based controller for the defender nation."""

    def __init__(self, spec: MapSpec) -> None:
        self.spec = spec
        self._graph = spec.to_graph()
        self._home = tuple(i for i, t in enumerate(spec.territories) if t.home_of == DEFENDER)
        if not self._home:
            raise ValueError("Map has no defender home territories")

    def decide(
        self,
        controller: np.ndarray,     # shape (V,) ints
        invader_units: np.ndarray,  # shape (V,)
        defender_units: np.ndarray, # shape (V,)
    ) -> DefenderDecision:
        # 1. Recapture occupied home — pick the home territory closest to a friendly base.
        occupied_home = [
            i for i in self._home if controller[i] == INVADER and invader_units[i] > 0
        ]
        if occupied_home:
            target = min(
                occupied_home,
                key=lambda i: self._distance_to_friendly(i, controller),
            )
            return DefenderDecision(tag=COUNTER_HOME, target=target)

        # 2. Counter-strike: any adjacent, non-defender territory where invader strength
        #    exceeds our local strength.
        for v in range(self.spec.n):
            if controller[v] == DEFENDER:
                continue
            local_def = sum(defender_units[u] for u in self._graph.neighbors(v))
            if invader_units[v] > local_def and invader_units[v] > 0:
                return DefenderDecision(tag=COUNTER_STRIKE, target=v)

        # 3. Reinforce the most strategically valuable home territory.
        target = max(self._home, key=lambda i: self.spec.territories[i].strategic_value)
        if defender_units[target] >= 2:
            return DefenderDecision(tag=HOLD, target=target)
        return DefenderDecision(tag=REINFORCE, target=target)

    def _distance_to_friendly(self, src: int, controller: np.ndarray) -> int:
        """BFS distance from `src` to the nearest defender-controlled territory."""
        try:
            for node, dist in nx.single_source_shortest_path_length(
                self._graph, src
            ).items():
                if controller[node] == DEFENDER:
                    return dist
        except nx.NetworkXError:
            pass
        return 10_000
