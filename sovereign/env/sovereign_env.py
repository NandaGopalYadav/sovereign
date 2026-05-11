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
    defender_economy: float
    theta: float
    t_occ: int                          # consecutive turns >=1 invader-occupied non-home territory
    occupied_territory_turns: int = 0
    pressure_streak: int = 0
    turn: int = 0
    last_resources: float = 0.0
    last_territory_count: int = 0
    last_pol_action: int = 0
    last_mil_action: int = 0
    last_target: int = 0
    insurgency_event: bool = False
    last_settlement_verdict: Any = None    # AcceptanceVerdict | None — last gate decision when negotiate was tried
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
            defender_economy=1.0,
            theta=0.0,
            t_occ=0,
            turn=0,
            last_resources=self._connected_non_home_resources(controller),
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
        prev_resources = self._connected_non_home_resources(s.controller)
        # Capture pre-action snapshot for substance-of-concession tests in the
        # drift-signal collector. Iter-3: a `withdraw` only counts as a real
        # concession if there was off-home territory to give back; a
        # `negotiate` only counts if the defender was actually under pressure.
        prev_t_occ = s.t_occ
        prev_territory_share = self._territory_share()
        prev_defender_loss_fraction = self._defender_loss_fraction()
        prev_thresholds = (
            self.hysteresis.neutral_joined_defender,
            self.hysteresis.supply_routes_open,
            self.hysteresis.formal_alliance,
        )
        # Iter-9: settlement acceptance is derived from two rulebook
        # facts — (1) §6.1 specifies that NEGOTIATE increases L by 0.03,
        # so a NEGOTIATE that doesn't actually increase L (because L is
        # already capped at 1.0, or because other effects this turn
        # cancelled it) didn't produce a diplomatic effect, and
        # (2) the defender is rule-based, so its choice to counter-attack
        # this turn (COUNTER_HOME / COUNTER_STRIKE) indicates it isn't
        # open to terms. Settlement requires the defender did NOT
        # counter-attack AND L actually increased over the turn.
        prev_legitimacy = s.legitimacy

        # 1. Initiative & posture: commit the political action first.
        self._apply_political(pol, target, prev_territory_share, prev_defender_loss_fraction)

        # 2. Compute pre-combat drift signals (but defer applying noise to step 9).
        # 3. Defender decides reaction.
        defender_choice = self._defender.decide(s.controller, s.invader_units, s.defender_units)

        # 4. Invader military action.
        #    Iter-7: rulebook military action set is (0=ADVANCE, 1=HOLD,
        #    2=WITHDRAW, 3=STRIKE). REDEPLOY was a code-only addition and
        #    has been removed.
        if mil == 0:  # ADVANCE — claim adjacent territory
            self._invader_attack(target)
        elif mil == 2:  # WITHDRAW — cede one off-home territory
            self._invader_withdraw(target)
        elif mil == 3:  # STRIKE — damage defender units
            self._invader_strike(target)
        # mil == 1 (HOLD) — no military move

        # 5. Defender military action.
        defender_attacked = False
        if defender_choice.tag in (COUNTER_HOME, COUNTER_STRIKE):
            defender_attacked = self._defender_attack(defender_choice.target)

        # 6. Update territory control bookkeeping (controller field is mutated by combat).
        # 7. Occupation accounting.
        #    Iter-8 (rulebook §4.4): a single global t_occ counter — no
        #    per-territory tracking. t_occ increments by 1 each turn the
        #    invader holds *any* off-home territory and resets only on
        #    full withdrawal.
        invader_off_home_territories = [
            v for v in range(s.spec.n)
            if s.controller[v] == INVADER and s.spec.territories[v].home_of != INVADER
        ]
        if invader_off_home_territories:
            s.t_occ += 1
        else:
            s.t_occ = 0
        s.occupied_territory_turns += len(invader_off_home_territories)
        if self._instant_pressure():
            s.pressure_streak += 1
        else:
            s.pressure_streak = 0

        # 8. Insurgency check — rulebook §4.4 / §8.3: a single Bernoulli per
        #    turn using global `t_occ`. Iter-5's per-territory model has been
        #    reverted to comply with the rulebook.
        s.insurgency_event = False
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
            signals = self._collect_drift_signals(
                pol, mil,
                prev_t_occ=prev_t_occ,
                prev_territory_share=prev_territory_share,
                prev_defender_loss_fraction=prev_defender_loss_fraction,
            )
            assert self.np_random is not None
            s.theta = step_theta(s.theta, signals, self.cfg.drift, self.np_random)
            update_threshold_events(
                self.hysteresis,
                s.theta,
                self.cfg.thresholds,
                self.cfg.flags.sanctions_enabled,
            )
            self._apply_threshold_effects(prev_thresholds)

        # 10. Economy update — sanctions slowly bleed E.
        if self.hysteresis.sanctions_active and self.cfg.flags.sanctions_enabled:
            s.economy = max(0.0, s.economy - self.cfg.combat.sanction_economy_decay)

        # 11. Legitimacy decay.
        #     Iter-8: rulebook §6.2 specifies action-specific L costs:
        #     ADVANCE −0.05, STRIKE −0.08. We also keep the generic
        #     per-territory occupation decay (rulebook §4.2 says L decays
        #     under aggressive action but doesn't quantify; we keep
        #     0.005/territory as a reasonable interpretation).
        if self.cfg.flags.legitimacy_enabled:
            decay = self.cfg.combat.occupation_legitimacy_decay * len(invader_off_home_territories)
            if mil == 0:  # ADVANCE
                decay += self.cfg.combat.advance_legitimacy_cost
            if mil == 3:  # STRIKE
                decay += self.cfg.combat.strike_legitimacy_cost
            s.legitimacy = max(0.0, s.legitimacy - decay)

        # 12. Reward decomposition + terminal check.
        new_territory_count = int((s.controller == INVADER).sum())
        new_resources = self._connected_non_home_resources(s.controller)
        rc = compute_reward(
            controlled_resources=new_resources,
            delta_territory=float(new_territory_count - prev_territory_count) / max(s.spec.n, 1),
            delta_resources=float(new_resources - prev_resources),
            t_occ=s.t_occ,
            t_max=self.cfg.limits.t_max,
            legitimacy=s.legitimacy,
            sanction_active=self.hysteresis.sanctions_active,
            economy=s.economy,
            insurgency_event=s.insurgency_event,
            weights=self.cfg.reward,
            flags=self.cfg.flags,
            occupation_multiplier=self._occupation_multiplier(invader_off_home_territories),
        )

        # Iter-9 acceptance conditions (derived from rulebook §6.1):
        # NEGOTIATE settles iff (a) the defender did not actually
        # counter-attack this turn — its rule-based policy either chose
        # a non-counter-attack tag OR chose to counter-attack but had
        # no adjacent units to launch from — AND (b) L actually
        # increased over this turn (NEGOTIATE's specified +0.03 L
        # effect landed; i.e. L was below 1.0 going in and was not
        # overwhelmed by other negative L effects this turn).
        defender_did_not_reply = not defender_attacked
        legitimacy_increased = s.legitimacy > prev_legitimacy
        terminal_reason = self._check_terminal(
            pol, defender_did_not_reply=defender_did_not_reply,
            legitimacy_increased=legitimacy_increased,
        )
        if terminal_reason is not None:
            rc.terminal_bonus = self._terminal_bonus(terminal_reason)

        s.last_resources = new_resources
        s.last_territory_count = new_territory_count
        return rc, terminal_reason

    # -- mechanics helpers ------------------------------------------------------------

    def _apply_political(
        self,
        pol: int,
        target: int,
        prev_territory_share: float,
        prev_defender_loss_fraction: float,
    ) -> None:
        s = self.state
        assert s is not None
        # Iter-8 strict rulebook conformance:
        #   - All θ shifts apply unconditionally (rulebook §6.1 doesn't
        #     specify ablation interactions; iter-6 gating reverted).
        #   - NEGOTIATE's L/θ shifts apply unconditionally (rulebook §6.1;
        #     iter-3 substance-of-leverage gating reverted).
        #   - DO_NOTHING now has the rulebook's "slow" effects:
        #     L decay if L < 0.5, θ drift toward defender if t_occ > 0.
        combat = self.cfg.combat
        if pol == 0:  # SEEK_ALLIANCE
            s.theta = float(np.clip(s.theta + combat.seek_alliance_theta_shift, -1.0, 1.0))
            s.legitimacy = min(1.0, s.legitimacy + combat.seek_alliance_legitimacy_gain)
        elif pol == 1:  # IMPOSE_SANCTION
            if self.cfg.flags.legitimacy_enabled:
                s.legitimacy = max(0.0, s.legitimacy - combat.impose_sanction_legitimacy_cost)
            s.theta = float(np.clip(s.theta + combat.impose_sanction_theta_shift, -1.0, 1.0))
            s.defender_economy = max(
                0.0, s.defender_economy - combat.impose_sanction_defender_economy_decay
            )
        elif pol == 2:  # ISSUE_THREAT
            s.theta = float(np.clip(s.theta + combat.issue_threat_theta_shift, -1.0, 1.0))
            if self.cfg.flags.legitimacy_enabled:
                s.legitimacy = max(0.0, s.legitimacy - combat.issue_threat_legitimacy_cost)
        elif pol == 3:  # NEGOTIATE
            # Settlement-terminal logic is handled in `_check_terminal`.
            s.theta = float(np.clip(s.theta + combat.negotiate_theta_shift, -1.0, 1.0))
            s.legitimacy = min(1.0, s.legitimacy + combat.negotiate_legitimacy_gain)
        elif pol == 4:  # DO_NOTHING — rulebook §6.1 slow effects
            if self.cfg.flags.legitimacy_enabled and s.legitimacy < 0.5:
                s.legitimacy = max(0.0, s.legitimacy - combat.do_nothing_l_decay)
            if s.t_occ > 0:
                s.theta = float(np.clip(s.theta + combat.do_nothing_theta_drift, -1.0, 1.0))

    def _apply_threshold_effects(self, prev_thresholds: tuple[bool, bool, bool]) -> None:
        s = self.state
        assert s is not None
        prev_joined, _, prev_alliance = prev_thresholds
        if self.hysteresis.neutral_joined_defender and not prev_joined:
            defender_home = [
                i for i, t in enumerate(s.spec.territories) if t.home_of == DEFENDER
            ]
            if defender_home:
                target = max(defender_home, key=lambda i: s.spec.territories[i].strategic_value)
                s.defender_units[target] += self.cfg.combat.neutral_join_defender_units
            if self.cfg.flags.legitimacy_enabled:
                s.legitimacy = max(0.0, s.legitimacy - self.cfg.combat.neutral_join_legitimacy_cost)
        if self.hysteresis.formal_alliance and not prev_alliance:
            s.defender_economy = max(
                0.0,
                s.defender_economy - self.cfg.combat.formal_alliance_defender_economy_decay,
            )
            if self.cfg.flags.legitimacy_enabled:
                s.legitimacy = max(0.0, s.legitimacy - self.cfg.combat.formal_alliance_legitimacy_cost)

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

    def _invader_withdraw(self, target: int) -> None:
        """WITHDRAW (military) — cede a single off-home territory back to
        its original controller. Iter-7: per the rulebook, this is a
        per-territory action (not the old "withdraw everything" political
        action), grants +0.02 L, and resets that territory's occupation
        counter. If it's the last off-home held, aggregate t_occ also
        resets to 0.
        """
        s = self.state
        assert s is not None
        if target < 0 or target >= s.spec.n:
            return
        spec_t = s.spec.territories[target]
        if s.controller[target] != INVADER or spec_t.home_of == INVADER:
            # Can only withdraw from an off-home territory we currently hold.
            return
        s.controller[target] = spec_t.home_of
        s.invader_units[target] = 0.0
        if self.cfg.flags.legitimacy_enabled:
            s.legitimacy = min(
                1.0, s.legitimacy + self.cfg.combat.withdraw_legitimacy_gain
            )
        # If no off-home territories remain, reset the aggregate t_occ.
        remaining_off_home = [
            v for v in range(s.spec.n)
            if s.controller[v] == INVADER
            and s.spec.territories[v].home_of != INVADER
        ]
        if not remaining_off_home:
            s.t_occ = 0

    def _invader_strike(self, target: int) -> None:
        s = self.state
        assert s is not None
        if s.invader_strike <= 0 or target < 0 or target >= s.spec.n:
            return
        s.invader_strike -= 1
        damage = self.cfg.combat.strike_attacker_damage
        s.defender_units[target] = max(0.0, s.defender_units[target] - damage)
        s.neutral_units[target] = max(0.0, s.neutral_units[target] - damage * 0.5)

    def _defender_attack(self, target: int) -> bool:
        """Resolve the defender's counter-attack. Returns True if the
        attack actually fired (the defender had adjacent units capable
        of attacking the target). Returns False if the defender's
        rule-based policy *wanted* to attack but had no source —
        which iter-9 treats as 'did not reply' for the negotiation
        acceptance gate."""
        s = self.state
        assert s is not None
        assert self.np_random is not None
        sources = [
            u for u in self.graph.neighbors(target)
            if s.controller[u] == DEFENDER and s.defender_units[u] >= 1
        ]
        if not sources:
            return False
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
        return True

    # -- drift signal collection ------------------------------------------------------

    def _collect_drift_signals(
        self,
        pol: int,
        mil: int,
        prev_t_occ: int = 0,
        prev_territory_share: float = 0.0,
        prev_defender_loss_fraction: float = 0.0,
    ) -> DriftSignals:
        s = self.state
        assert s is not None
        # Iter-8 strict rulebook conformance: drift signals are
        # unconditional (rulebook §7.2 defines them as action-indicator
        # variables, not substance-gated). Iter-3's substance gating on
        # NEGOTIATE is reverted.
        return DriftSignals(
            legitimacy_loss=1.0 - s.legitimacy,
            advance_action=1.0 if mil == 0 else 0.0,
            strike_action=1.0 if mil == 3 else 0.0,
            negotiate_action=1.0 if pol == 3 else 0.0,
            seek_alliance_action=1.0 if pol == 0 else 0.0,
            occupation_duration=min(s.t_occ / max(self.cfg.limits.t_max, 1), 1.0),
        )

    # -- terminal logic ---------------------------------------------------------------

    def _check_terminal(
        self,
        pol: int,
        defender_did_not_reply: bool = True,
        legitimacy_increased: bool = False,
    ) -> str | None:
        s = self.state
        assert s is not None
        if self.cfg.flags.legitimacy_enabled and s.legitimacy <= 0.0:
            return "legitimacy_collapse"
        if s.invader_units.sum() <= 0 and s.invader_strike <= 0:
            return "invader_destroyed"
        # Iter-9 rulebook-derived acceptance gate (per user reading of §6.1):
        # The defender accepts NEGOTIATE iff (a) it does not reply
        # (its rule-based policy chose a non-counter-attack response this
        # turn) AND (b) the international legitimacy increased over the
        # turn (NEGOTIATE's specified +0.03 L effect actually landed, i.e.
        # L was below 1.0 going in and was not overwhelmed by other
        # negative L effects this turn).
        s.last_settlement_verdict = None
        if (
            pol == 3
            and s.defender_units.sum() > 0
            and defender_did_not_reply
            and legitimacy_increased
        ):
            return "negotiated_settlement"
        # Total conquest: invader holds every non-invader-home territory and defender has no units.
        all_taken = all(
            s.controller[v] == INVADER or s.spec.territories[v].home_of == INVADER
            for v in range(s.spec.n)
        )
        if all_taken and s.defender_units.sum() <= 0:
            return "total_conquest"
        return None

    def _territory_share(self) -> float:
        s = self.state
        assert s is not None
        non_home = [v for v in range(s.spec.n) if s.spec.territories[v].home_of != INVADER]
        if not non_home:
            return 0.0
        held = sum(1 for v in non_home if s.controller[v] == INVADER)
        return held / len(non_home)

    def _defender_loss_fraction(self) -> float:
        """Fraction of the defender's *initial* ground force that has been destroyed."""
        s = self.state
        assert s is not None
        initial = max(float(self.cfg.force.defender_ground), 1.0)
        remaining = float(s.defender_units.sum())
        return max(0.0, min(1.0, (initial - remaining) / initial))

    def _terminal_bonus(self, reason: str) -> float:
        # Iter-8 strict rulebook conformance: settlement is now a flat
        # +40 per rulebook §9 — no state-scaled bonus.
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
        payload: dict[str, Any] = {
            "turn": s.turn,
            "theta": float(s.theta),
            "legitimacy": float(s.legitimacy),
            "economy": float(s.economy),
            "defender_economy": float(s.defender_economy),
            "t_occ": int(s.t_occ),
            "occupied_territory_turns": int(s.occupied_territory_turns),
            "pressure_streak": int(s.pressure_streak),
            "sanctions_active": self.hysteresis.sanctions_active,
            "supply_routes_open": self.hysteresis.supply_routes_open,
            "formal_alliance": self.hysteresis.formal_alliance,
            "invader_units": s.invader_units.tolist(),
            "defender_units": s.defender_units.tolist(),
            "controller": s.controller.tolist(),
            "insurgency_event": s.insurgency_event,
            "territory_share": self._territory_share(),
            "defender_loss_fraction": self._defender_loss_fraction(),
            "connected_resource_yield": self._connected_non_home_resources(s.controller),
            "controlled_resource_yield": self._sum_resources(s.controller, INVADER),
        }
        if s.last_settlement_verdict is not None:
            v = s.last_settlement_verdict
            payload["settlement_verdict"] = {
                "accepted": v.accepted,
                "pressure_ok": v.pressure_ok,
                "viability_ok": v.viability_ok,
                "reason": v.reason,
            }
        return payload

    def _sum_resources(self, controller: np.ndarray, nation: int) -> float:
        return float(
            sum(
                self.map_spec.territories[v].resource_value
                for v in range(self.map_spec.n)
                if controller[v] == nation
            )
        )

    def _connected_invader_nodes(self, controller: np.ndarray) -> set[int]:
        invader_nodes = [v for v in range(self.map_spec.n) if controller[v] == INVADER]
        if not invader_nodes:
            return set()
        subgraph = self.graph.subgraph(invader_nodes)
        connected: set[int] = set()
        for home in (
            v for v, t in enumerate(self.map_spec.territories)
            if t.home_of == INVADER and controller[v] == INVADER
        ):
            connected.update(int(n) for n in nx.node_connected_component(subgraph, home))
        return connected

    def _connected_resources(self, controller: np.ndarray, nation: int) -> float:
        if nation != INVADER:
            return self._sum_resources(controller, nation)
        connected = self._connected_invader_nodes(controller)
        return float(
            sum(
                self.map_spec.territories[v].resource_value
                for v in connected
                if controller[v] == INVADER
            )
        )

    def _connected_non_home_resources(self, controller: np.ndarray) -> float:
        # Iter-8 strict rulebook conformance: rulebook §8.1 specifies
        # `w_T · Σ resource_value(v) for v controlled by I` — all
        # invader-held territory including invader-home. The iter-1..7
        # exclusion of home territory has been reverted. (Connectivity
        # restriction is retained because rulebook §3.2 says disconnected
        # occupied territories yield nothing.)
        connected = self._connected_invader_nodes(controller)
        return float(
            sum(
                self.map_spec.territories[v].resource_value
                for v in connected
                if controller[v] == INVADER
            )
        )

    def _instant_pressure(self) -> bool:
        return (
            self._territory_share() >= self.cfg.acceptance.territory_floor
            or (
                self._defender_loss_fraction() >= self.cfg.acceptance.defender_loss_floor
                and self._territory_share() > 0.0
            )
        )

    def _occupation_multiplier(self, invader_off_home_territories: list[int]) -> float:
        multiplier = 1.0
        if self.hysteresis.supply_routes_open:
            multiplier *= 1.0 - self.cfg.combat.supply_route_occupation_discount
        s = self.state
        connected = self._connected_invader_nodes(s.controller) if s is not None else set()
        if any(v not in connected for v in invader_off_home_territories):
            multiplier *= 1.0 + self.cfg.combat.disconnected_occupation_drag
        return multiplier

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
