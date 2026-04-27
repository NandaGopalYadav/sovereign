"""Top-level Gymnasium environment.

Three nations on a graph map. The invader is the RL agent; the defender is rule-based;
the neutral is a stochastic political process governed by the drift-diffusion of θ.

The action space is *flat* (`Discrete`) so DQN works out of the box; internally we
decode it as `(political, military, target)` per the spec's joint action requirement.

A "step" of the env corresponds to one full turn for the invader. Inside :meth:`step`
we run the 12 substeps documented in :meth:`_run_turn`. Decoupling the substep loop
from the agent's interface keeps the gym contract (one observation, one action) intact
while preserving the structured turn order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from sovereign.agents.defender import COUNTER_HOME, COUNTER_STRIKE, DefenderPolicy
from sovereign.env.config import (
    MILITARY_ACTIONS,
    POLITICAL_ACTIONS,
    SovereignConfig,
)
from sovereign.env.map import (
    CONTESTED,
    DEFENDER,
    INVADER,
    NEUTRAL,
    MapSpec,
    get_map,
)
from sovereign.env.mechanics import (
    DriftSignals,
    HysteresisState,
    StepTrace,
    compute_reward,
    insurgency_fires,
    resolve_combat,
    step_theta,
    update_threshold_events,
)


# --------------------------------------------------------------------------------------
# Action codec.  Discrete(N_POL · N_MIL · V).
# --------------------------------------------------------------------------------------

N_POL = len(POLITICAL_ACTIONS)
N_MIL = len(MILITARY_ACTIONS)


def _encode_action(pol: int, mil: int, target: int, n_targets: int) -> int:
    return pol * (N_MIL * n_targets) + mil * n_targets + target


def _decode_action(action: int, n_targets: int) -> tuple[int, int, int]:
    pol, rem = divmod(action, N_MIL * n_targets)
    mil, target = divmod(rem, n_targets)
    return pol, mil, target


# --------------------------------------------------------------------------------------
# Observation layout. We flatten everything into a 1-D Box for SB3 compatibility,
# but keep the slicing centralised so policies can recover structured fields.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationLayout:
    n_territories: int

    @property
    def control_size(self) -> int:
        # one-hot over {INVADER, DEFENDER, NEUTRAL, CONTESTED}
        return self.n_territories * 4

    @property
    def units_size(self) -> int:
        return 2 * self.n_territories  # invader, defender per territory

    @property
    def scalar_size(self) -> int:
        # L, E, θ, t_occ_norm, sanctions_active, alliance, supply_open
        return 7

    @property
    def total(self) -> int:
        return self.control_size + self.units_size + self.scalar_size

    def slices(self) -> dict[str, slice]:
        c = self.control_size
        u = self.units_size
        return {
            "control": slice(0, c),
            "units": slice(c, c + u),
            "scalars": slice(c + u, c + u + self.scalar_size),
        }


# --------------------------------------------------------------------------------------
# Internal world state.
# --------------------------------------------------------------------------------------


@dataclass
class WorldState:
    spec: MapSpec
    controller: np.ndarray              # (V,) int — INVADER / DEFENDER / NEUTRAL / CONTESTED
    invader_units: np.ndarray           # (V,) float
    defender_units: np.ndarray          # (V,) float
    neutral_units: np.ndarray           # (V,) float
    invader_strike: float               # global pool of strike units
    defender_strike: float
    legitimacy: float
    economy: float
    theta: float
    t_occ: int                          # consecutive turns >=1 invader-occupied non-home territory
    turn: int = 0
    last_resources: float = 0.0
    last_territory_count: int = 0
    last_pol_action: int = 0
    last_mil_action: int = 0
    last_target: int = 0
    insurgency_event: bool = False
    history: list[StepTrace] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# The Env.
# --------------------------------------------------------------------------------------


class SovereignEnv(gym.Env):
    """Three-nation geopolitical conflict environment.

    Parameters
    ----------
    config:
        :class:`SovereignConfig`. If omitted, defaults are used.
    map_name:
        Override the map specified in the config. Useful for parameter sweeps.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        config: SovereignConfig | None = None,
        map_name: str | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or SovereignConfig()
        if map_name is not None:
            self.cfg = self.cfg.with_overrides(map_name=map_name)
        self.map_spec: MapSpec = get_map(self.cfg.map_name)
        self.graph: nx.Graph = self.map_spec.to_graph()
        self.layout = ObservationLayout(n_territories=self.map_spec.n)
        self.render_mode = render_mode

        self._defender = DefenderPolicy(self.map_spec)
        self._n_targets = self.map_spec.n
        self._n_actions = N_POL * N_MIL * self._n_targets

        self.action_space = spaces.Discrete(self._n_actions)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.layout.total,),
            dtype=np.float32,
        )

        self.state: WorldState | None = None
        self.hysteresis = HysteresisState()
        self.np_random: np.random.Generator | None = None

    # -- gym API ----------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()

        controller = np.array(
            [t.home_of for t in self.map_spec.territories], dtype=np.int8
        )
        # Distribute initial unit stacks: invader piles into one home (capital), defender does too.
        invader_units = np.zeros(self.map_spec.n, dtype=np.float32)
        defender_units = np.zeros(self.map_spec.n, dtype=np.float32)
        neutral_units = np.zeros(self.map_spec.n, dtype=np.float32)
        invader_homes = [i for i, t in enumerate(self.map_spec.territories) if t.home_of == INVADER]
        defender_homes = [i for i, t in enumerate(self.map_spec.territories) if t.home_of == DEFENDER]
        contested = [i for i, t in enumerate(self.map_spec.territories) if t.home_of in (NEUTRAL, CONTESTED)]
        invader_units[invader_homes[0]] = self.cfg.force.invader_ground
        defender_units[defender_homes[-1]] = self.cfg.force.defender_ground
        for c in contested:
            neutral_units[c] = self.cfg.force.neutral_ground / max(len(contested), 1)

        self.hysteresis.reset()
        self.state = WorldState(
            spec=self.map_spec,
            controller=controller,
            invader_units=invader_units,
            defender_units=defender_units,
            neutral_units=neutral_units,
            invader_strike=float(self.cfg.force.invader_strike),
            defender_strike=float(self.cfg.force.defender_strike),
            legitimacy=1.0,
            economy=1.0,
            theta=0.0,
            t_occ=0,
            turn=0,
            last_resources=self._sum_resources(controller, INVADER),
            last_territory_count=int((controller == INVADER).sum()),
        )
        obs = self._encode_observation()
        info = self._info_payload()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        pol, mil, target = _decode_action(int(action), self._n_targets)

        rc, terminal_reason = self._run_turn(pol, mil, target)
        reward = rc.total()

        self.state.turn += 1
        terminated = terminal_reason is not None
        truncated = (not terminated) and self.state.turn >= self.cfg.limits.t_max

        # Trace
        trace = StepTrace(
            t=self.state.turn,
            theta=self.state.theta,
            legitimacy=self.state.legitimacy,
            economy=self.state.economy,
            t_occ=self.state.t_occ,
            sanctions_active=self.hysteresis.sanctions_active,
            insurgency_event=self.state.insurgency_event,
            pol_action=pol,
            mil_action=mil,
            target=target,
            reward=reward,
            reward_components=rc.as_dict(),
        )
        self.state.history.append(trace)

        info = self._info_payload()
        info["terminal_reason"] = terminal_reason
        info["reward_components"] = rc.as_dict()

        if self.render_mode == "human":
            self.render()

        return self._encode_observation(), float(reward), terminated, truncated, info

    def render(self) -> str | None:
        if self.state is None:
            return None
        return self._ascii_render()

    def close(self) -> None:  # pragma: no cover — nothing to release
        return None

    # -- 12-step turn -----------------------------------------------------------------

    def _run_turn(
        self, pol: int, mil: int, target: int
    ) -> tuple[Any, str | None]:
        """Execute the structured 12-substep turn and return (reward components, terminal_reason).

        The substep numbering mirrors the spec; each step is a small pure mutation of
        :attr:`state` and the hysteresis machine. The political action is committed
        before any military move, as required.
        """
        s = self.state
        assert s is not None
        s.last_pol_action = pol
        s.last_mil_action = mil
        s.last_target = target
        s.insurgency_event = False
        terminal_reason: str | None = None

        prev_territory_count = int((s.controller == INVADER).sum())
        prev_resources = self._sum_resources(s.controller, INVADER)

        # 1. Initiative & posture: commit the political action first.
        self._apply_political(pol, target)

        # 2. Compute pre-combat drift signals (but defer applying noise to step 9).
        # 3. Defender decides reaction.
        defender_choice = self._defender.decide(s.controller, s.invader_units, s.defender_units)

        # 4. Invader military action.
        if mil == 1:  # attack
            self._invader_attack(target)
        elif mil == 2:  # redeploy
            self._invader_redeploy(target)
        elif mil == 3:  # strike
            self._invader_strike(target)

        # 5. Defender military action.
        if defender_choice.tag in (COUNTER_HOME, COUNTER_STRIKE):
            self._defender_attack(defender_choice.target)

        # 6. Update territory control bookkeeping (controller field is mutated by combat).
        # 7. Occupation accounting.
        invader_off_home_territories = [
            v for v in range(s.spec.n)
            if s.controller[v] == INVADER and s.spec.territories[v].home_of != INVADER
        ]
        if invader_off_home_territories:
            s.t_occ += 1
        else:
            # Full withdrawal resets the timer — important for the "withdraw" political action.
            s.t_occ = 0

        # 8. Insurgency check.
        if self.cfg.flags.insurgency_enabled and invader_off_home_territories and self.np_random:
            if insurgency_fires(s.t_occ, self.cfg.hazard, self.np_random):
                s.insurgency_event = True
                victim = invader_off_home_territories[
                    int(self.np_random.integers(len(invader_off_home_territories)))
                ]
                s.invader_units[victim] = max(
                    0.0, s.invader_units[victim] - self.cfg.combat.insurgency_unit_loss
                )

        # 9. θ drift + noise + threshold update.
        if self.cfg.flags.neutral_posture_enabled:
            signals = self._collect_drift_signals(pol, mil)
            assert self.np_random is not None
            s.theta = step_theta(s.theta, signals, self.cfg.drift, self.np_random)
            update_threshold_events(
                self.hysteresis,
                s.theta,
                self.cfg.thresholds,
                self.cfg.flags.sanctions_enabled,
            )

        # 10. Economy update — sanctions slowly bleed E.
        if self.hysteresis.sanctions_active and self.cfg.flags.sanctions_enabled:
            s.economy = max(0.0, s.economy - self.cfg.combat.sanction_economy_decay)

        # 11. Legitimacy decay.
        if self.cfg.flags.legitimacy_enabled:
            decay = self.cfg.combat.occupation_legitimacy_decay * len(invader_off_home_territories)
            if mil == 3:  # strike used this turn
                decay += self.cfg.combat.strike_legitimacy_cost
            s.legitimacy = max(0.0, s.legitimacy - decay)

        # 12. Reward decomposition + terminal check.
        new_territory_count = int((s.controller == INVADER).sum())
        new_resources = self._sum_resources(s.controller, INVADER)
        rc = compute_reward(
            delta_territory=float(new_territory_count - prev_territory_count) / max(s.spec.n, 1),
            delta_resources=float(new_resources - prev_resources),
            t_occ=s.t_occ,
            t_max=self.cfg.limits.t_max,
            legitimacy=s.legitimacy,
            sanction_active=self.hysteresis.sanctions_active,
            insurgency_event=s.insurgency_event,
            weights=self.cfg.reward,
            flags=self.cfg.flags,
        )

        terminal_reason = self._check_terminal(pol)
        if terminal_reason is not None:
            rc.terminal_bonus = self._terminal_bonus(terminal_reason)

        s.last_resources = new_resources
        s.last_territory_count = new_territory_count
        return rc, terminal_reason

    # -- mechanics helpers ------------------------------------------------------------

    def _apply_political(self, pol: int, target: int) -> None:
        s = self.state
        assert s is not None
        if pol == 1:  # propaganda — small invader-favourable θ tilt and tiny L bump
            s.theta = float(np.clip(s.theta - 0.01, -1.0, 1.0))
            s.legitimacy = min(1.0, s.legitimacy + 0.005)
        elif pol == 4:  # withdraw
            # Hand back every non-home territory currently controlled by invader.
            for v in range(s.spec.n):
                if s.controller[v] == INVADER and s.spec.territories[v].home_of != INVADER:
                    s.controller[v] = s.spec.territories[v].home_of
                    s.invader_units[v] = 0.0
            s.t_occ = 0
        # Negotiate (pol == 2) is handled in `_check_terminal`.

    def _invader_attack(self, target: int) -> None:
        s = self.state
        assert s is not None
        assert self.np_random is not None
        if target < 0 or target >= s.spec.n:
            return
        # Find the strongest adjacent invader-controlled territory to launch from.
        sources = [
            u for u in self.graph.neighbors(target)
            if s.controller[u] == INVADER and s.invader_units[u] >= 1
        ]
        if not sources:
            return
        src = max(sources, key=lambda u: float(s.invader_units[u]))
        attackers = float(s.invader_units[src])
        defenders = float(s.defender_units[target] + s.neutral_units[target])
        terrain_bonus = (
            self.cfg.force.defender_home_effectiveness - 1.0
            if s.spec.territories[target].home_of == DEFENDER
            else 0.0
        )
        out = resolve_combat(attackers, defenders, terrain_bonus, self.cfg.combat, self.np_random)
        s.invader_units[src] = 0.0
        if out.attacker_won:
            s.controller[target] = INVADER
            s.invader_units[target] += out.attacker_remaining
            s.defender_units[target] = 0.0
            s.neutral_units[target] = 0.0
        else:
            # Attack failed: surviving attackers retreat to the source.
            s.invader_units[src] = out.attacker_remaining
            s.defender_units[target] = max(0.0, out.defender_remaining * 0.5 + s.neutral_units[target])
            s.neutral_units[target] = max(0.0, s.neutral_units[target] - 0.5)

    def _invader_redeploy(self, target: int) -> None:
        s = self.state
        assert s is not None
        if target < 0 or target >= s.spec.n or s.controller[target] != INVADER:
            return
        # Pull one unit from each adjacent friendly territory holding ≥2 units.
        for u in self.graph.neighbors(target):
            if s.controller[u] == INVADER and s.invader_units[u] >= 2:
                s.invader_units[u] -= 1
                s.invader_units[target] += 1

    def _invader_strike(self, target: int) -> None:
        s = self.state
        assert s is not None
        if s.invader_strike <= 0 or target < 0 or target >= s.spec.n:
            return
        s.invader_strike -= 1
        damage = self.cfg.combat.strike_attacker_damage
        s.defender_units[target] = max(0.0, s.defender_units[target] - damage)
        s.neutral_units[target] = max(0.0, s.neutral_units[target] - damage * 0.5)

    def _defender_attack(self, target: int) -> None:
        s = self.state
        assert s is not None
        assert self.np_random is not None
        sources = [
            u for u in self.graph.neighbors(target)
            if s.controller[u] == DEFENDER and s.defender_units[u] >= 1
        ]
        if not sources:
            return
        src = max(sources, key=lambda u: float(s.defender_units[u]))
        attackers = float(s.defender_units[src])
        defenders = float(s.invader_units[target])
        # Defender attacking *into* a non-home is itself: no terrain bonus to either side.
        out = resolve_combat(attackers, defenders, 0.0, self.cfg.combat, self.np_random)
        s.defender_units[src] = 0.0
        if out.attacker_won:
            s.controller[target] = DEFENDER if s.spec.territories[target].home_of == DEFENDER else CONTESTED
            s.defender_units[target] += out.attacker_remaining
            s.invader_units[target] = 0.0
        else:
            s.defender_units[src] = out.attacker_remaining
            s.invader_units[target] = max(0.0, out.defender_remaining)

    # -- drift signal collection ------------------------------------------------------

    def _collect_drift_signals(self, pol: int, mil: int) -> DriftSignals:
        s = self.state
        assert s is not None
        invader_off_home = sum(
            1 for v in range(s.spec.n)
            if s.controller[v] == INVADER and s.spec.territories[v].home_of != INVADER
        )
        non_invader_count = sum(
            1 for v in range(s.spec.n) if s.spec.territories[v].home_of != INVADER
        )
        defender_alive = bool(s.defender_units.sum() > 0)
        defender_capital_held = any(
            s.controller[v] == DEFENDER and s.spec.territories[v].home_of == DEFENDER
            and s.spec.territories[v].strategic_value >= 0.85
            for v in range(s.spec.n)
        )
        return DriftSignals(
            invader_aggression=1.0 if mil in (1, 3) else 0.0,
            occupation_fraction=invader_off_home / max(non_invader_count, 1),
            legitimacy_loss=1.0 - s.legitimacy,
            defender_morale=1.0 if (defender_alive and defender_capital_held) else 0.0,
            invader_concession=1.0 if pol in (2, 4) else 0.0,
            economic_pressure=1.0 - s.economy,
        )

    # -- terminal logic ---------------------------------------------------------------

    def _check_terminal(self, pol: int) -> str | None:
        s = self.state
        assert s is not None
        if self.cfg.flags.legitimacy_enabled and s.legitimacy <= 0.0:
            return "legitimacy_collapse"
        if s.invader_units.sum() <= 0 and s.invader_strike <= 0:
            return "invader_destroyed"
        # Negotiated settlement: invader proposed AND legitimacy is healthy AND defender
        # holds at least one home — the defender is rational and accepts.
        if pol == 2 and s.legitimacy > 0.4:
            defender_alive = bool(s.defender_units.sum() > 0)
            if defender_alive:
                return "negotiated_settlement"
        # Total conquest: invader holds every non-invader-home territory and defender has no units.
        all_taken = all(
            s.controller[v] == INVADER or s.spec.territories[v].home_of == INVADER
            for v in range(s.spec.n)
        )
        if all_taken and s.defender_units.sum() <= 0:
            return "total_conquest"
        return None

    def _terminal_bonus(self, reason: str) -> float:
        t = self.cfg.terminal
        return {
            "legitimacy_collapse": t.legitimacy_collapse,
            "invader_destroyed": t.invader_destroyed,
            "negotiated_settlement": t.negotiated_settlement,
            "timeout": t.timeout,
            "total_conquest": t.total_conquest,
        }[reason]

    # -- observation packing ----------------------------------------------------------

    def _encode_observation(self) -> np.ndarray:
        s = self.state
        assert s is not None
        v = s.spec.n
        obs = np.zeros(self.layout.total, dtype=np.float32)
        # control one-hot
        for i in range(v):
            obs[i * 4 + int(s.controller[i])] = 1.0
        # units (normalised by initial invader force as a stable scale)
        scale = max(self.cfg.force.invader_ground, 1)
        offset = self.layout.control_size
        obs[offset : offset + v] = np.minimum(s.invader_units / scale, 1.0)
        obs[offset + v : offset + 2 * v] = np.minimum(s.defender_units / scale, 1.0)
        # scalars
        sc = offset + 2 * v
        obs[sc + 0] = s.legitimacy
        obs[sc + 1] = s.economy
        obs[sc + 2] = s.theta
        obs[sc + 3] = min(s.t_occ / max(self.cfg.limits.t_max, 1), 1.0)
        obs[sc + 4] = 1.0 if self.hysteresis.sanctions_active else 0.0
        obs[sc + 5] = 1.0 if self.hysteresis.formal_alliance else 0.0
        obs[sc + 6] = 1.0 if self.hysteresis.supply_routes_open else 0.0
        return obs

    def _info_payload(self) -> dict[str, Any]:
        s = self.state
        assert s is not None
        return {
            "turn": s.turn,
            "theta": float(s.theta),
            "legitimacy": float(s.legitimacy),
            "economy": float(s.economy),
            "t_occ": int(s.t_occ),
            "sanctions_active": self.hysteresis.sanctions_active,
            "supply_routes_open": self.hysteresis.supply_routes_open,
            "formal_alliance": self.hysteresis.formal_alliance,
            "invader_units": s.invader_units.tolist(),
            "defender_units": s.defender_units.tolist(),
            "controller": s.controller.tolist(),
            "insurgency_event": s.insurgency_event,
        }

    def _sum_resources(self, controller: np.ndarray, nation: int) -> float:
        return float(
            sum(
                self.map_spec.territories[v].resource_value
                for v in range(self.map_spec.n)
                if controller[v] == nation
            )
        )

    # -- rendering --------------------------------------------------------------------

    def _ascii_render(self) -> str:
        s = self.state
        assert s is not None
        sym = {INVADER: "I", DEFENDER: "D", NEUTRAL: "N", CONTESTED: "·"}
        lines = [f"=== Turn {s.turn} ==="]
        lines.append(
            f"L={s.legitimacy:.2f}  E={s.economy:.2f}  θ={s.theta:+.2f}  "
            f"t_occ={s.t_occ}  sanc={int(self.hysteresis.sanctions_active)}  "
            f"alliance={int(self.hysteresis.formal_alliance)}"
        )
        for i, t in enumerate(s.spec.territories):
            controller_sym = sym[int(s.controller[i])]
            lines.append(
                f"  [{i:2d}] {t.name:<14} ctl={controller_sym}  "
                f"I={s.invader_units[i]:5.1f}  D={s.defender_units[i]:5.1f}  "
                f"N={s.neutral_units[i]:4.1f}"
            )
        out = "\n".join(lines)
        print(out)
        return out

    # -- public introspection ---------------------------------------------------------

    def action_meaning(self, action: int) -> tuple[str, str, int]:
        """Decode a discrete action into (political_label, military_label, target_id)."""
        pol, mil, target = _decode_action(int(action), self._n_targets)
        return POLITICAL_ACTIONS[pol], MILITARY_ACTIONS[mil], target

    def encode_action(self, pol: int, mil: int, target: int) -> int:
        return _encode_action(pol, mil, target, self._n_targets)

    def history(self) -> list[StepTrace]:
        if self.state is None:
            return []
        return list(self.state.history)

    def n_territories(self) -> int:
        return self.map_spec.n
