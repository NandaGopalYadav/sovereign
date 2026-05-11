"""Pure-function game mechanics: drift, hazard, thresholds with hysteresis, reward.

Every function here is side-effect-free and operates on plain data so the env can
checkpoint state cheaply and the test suite can exercise the formulas in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sovereign.env.config import (
    AblationFlags,
    CombatParams,
    DriftCoefficients,
    HazardParams,
    RewardWeights,
    SettlementAcceptance,
    SettlementBonusWeights,
    TerminalPayoffs,
    Thresholds,
)


# --------------------------------------------------------------------------------------
# Hysteretic threshold state.
# --------------------------------------------------------------------------------------


@dataclass
class HysteresisState:
    """Sticky bits for threshold events that latch on/off with a streak rule.

    `sanctions_active` flips on the first step θ exceeds `sanctions_on`. To clear,
    θ must remain below `sanctions_off` for `sanctions_off_streak` consecutive steps —
    a soft commitment device that prevents thrashing across the boundary.
    """

    sanctions_active: bool = False
    sanctions_below_streak: int = 0
    neutral_joined_defender: bool = False
    supply_routes_open: bool = False
    formal_alliance: bool = False

    def reset(self) -> None:
        self.sanctions_active = False
        self.sanctions_below_streak = 0
        self.neutral_joined_defender = False
        self.supply_routes_open = False
        self.formal_alliance = False


def update_threshold_events(
    state: HysteresisState,
    theta: float,
    th: Thresholds,
    sanctions_enabled: bool,
) -> HysteresisState:
    """Mutate `state` in-place applying the threshold rules. Returns it for chaining."""

    if sanctions_enabled:
        if not state.sanctions_active and theta > th.sanctions_on:
            state.sanctions_active = True
            state.sanctions_below_streak = 0
        elif state.sanctions_active:
            if theta < th.sanctions_off:
                state.sanctions_below_streak += 1
                if state.sanctions_below_streak >= th.sanctions_off_streak:
                    state.sanctions_active = False
                    state.sanctions_below_streak = 0
            else:
                state.sanctions_below_streak = 0

    # The remaining latches are sticky: once tripped, they stay on for the episode.
    # This models commitment — a country that joins an alliance does not silently leave.
    if theta > th.neutral_joins_defender:
        state.neutral_joined_defender = True
    if theta < th.supply_routes_open:
        state.supply_routes_open = True
    if theta < th.formal_alliance:
        state.formal_alliance = True

    return state


# --------------------------------------------------------------------------------------
# Neutral drift-diffusion.
# --------------------------------------------------------------------------------------


@dataclass
class DriftSignals:
    """The behavioural inputs that feed the rulebook Section 7.2 drift function."""

    legitimacy_loss: float             # ∈ [0,1] — equals (1 - L)
    advance_action: float              # ∈ {0,1} — military advance/attack
    strike_action: float               # ∈ {0,1} — strike action
    negotiate_action: float            # ∈ {0,1} — substantive negotiation
    seek_alliance_action: float        # ∈ {0,1} — seek-alliance political action
    occupation_duration: float         # ∈ [0,1] — t_occ / T_max


def drift(signals: DriftSignals, c: DriftCoefficients) -> float:
    """Deterministic part of the θ update."""
    return (
        c.alpha * signals.legitimacy_loss
        + c.beta * signals.advance_action
        + c.gamma * signals.strike_action
        - c.delta * signals.negotiate_action
        - c.epsilon * signals.seek_alliance_action
        + c.zeta * signals.occupation_duration
    )


def step_theta(
    theta: float,
    signals: DriftSignals,
    c: DriftCoefficients,
    rng: np.random.Generator,
) -> float:
    """One step of θ with drift + Gaussian noise, clipped to [-1, +1]."""
    mu = drift(signals, c)
    noise = rng.normal(0.0, c.sigma)
    return float(np.clip(theta + mu + noise, -1.0, 1.0))


# --------------------------------------------------------------------------------------
# Insurgency hazard.
# --------------------------------------------------------------------------------------


def insurgency_probability(t_occ: int, h: HazardParams) -> float:
    """p(insurgency | t_occ) = 1 - exp(-λ · t_occ)."""
    if t_occ <= 0:
        return 0.0
    return 1.0 - math.exp(-h.lam * t_occ)


def insurgency_fires(t_occ: int, h: HazardParams, rng: np.random.Generator) -> bool:
    return rng.random() < insurgency_probability(t_occ, h)


# --------------------------------------------------------------------------------------
# Reward decomposition.
# --------------------------------------------------------------------------------------


@dataclass
class RewardComponents:
    """All the parts of a single-step reward, useful for debugging and the UI."""

    controlled_resource_yield: float = 0.0
    territory_gain: float = 0.0
    resource_gain: float = 0.0
    occupation_cost: float = 0.0
    legitimacy_cost: float = 0.0
    sanction_cost: float = 0.0
    insurgency_cost: float = 0.0
    terminal_bonus: float = 0.0

    def total(self) -> float:
        return (
            self.controlled_resource_yield
            + self.territory_gain
            + self.resource_gain
            - self.occupation_cost
            - self.legitimacy_cost
            - self.sanction_cost
            - self.insurgency_cost
            + self.terminal_bonus
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "territory_gain": self.territory_gain,
            "controlled_resource_yield": self.controlled_resource_yield,
            "resource_gain": self.resource_gain,
            "occupation_cost": -self.occupation_cost,
            "legitimacy_cost": -self.legitimacy_cost,
            "sanction_cost": -self.sanction_cost,
            "insurgency_cost": -self.insurgency_cost,
            "terminal_bonus": self.terminal_bonus,
            "total": self.total(),
        }


def compute_reward(
    controlled_resources: float,
    delta_territory: float,
    delta_resources: float,
    t_occ: int,
    t_max: int,
    legitimacy: float,
    sanction_active: bool,
    economy: float,
    insurgency_event: bool | int,
    weights: RewardWeights,
    flags: AblationFlags,
    occupation_multiplier: float = 1.0,
) -> RewardComponents:
    """Apply the per-step reward formula, gated by ablation flags.

    `insurgency_event` accepts either a bool (legacy: 1 if any event fired this
    turn, used by the global-hazard model) or an int (per-territory model: the
    *count* of events that fired this turn). The cost scales with the count so
    that wide-occupation episodes pay more under the per-territory model.
    """
    rc = RewardComponents()
    rc.controlled_resource_yield = weights.territory * controlled_resources
    rc.territory_gain = weights.territory * delta_territory
    rc.resource_gain = weights.resources * delta_resources
    if flags.occupation_cost_enabled:
        rc.occupation_cost = weights.occupation * (t_occ / max(t_max, 1)) * occupation_multiplier
    if flags.legitimacy_enabled:
        rc.legitimacy_cost = weights.legitimacy * (1.0 - legitimacy)
    if flags.sanctions_enabled and sanction_active:
        rc.sanction_cost = weights.sanction * (1.0 - economy)
    if flags.insurgency_enabled:
        n_events = int(insurgency_event) if insurgency_event else 0
        if n_events > 0:
            rc.insurgency_cost = weights.insurgency * n_events
    return rc


# --------------------------------------------------------------------------------------
# Settlement bonus.
#
# Replaces the previous flat +40 settlement payoff. The bonus interpolates between
# `payoffs.settlement_min` and `payoffs.settlement_max` by a quality score over four
# state variables. The point is that the agent has to *plan against* the cost
# mechanisms across the whole episode to reach a high bonus — settling on the
# earliest legal turn from a content-free state pays only the floor.
# --------------------------------------------------------------------------------------


def settlement_bonus(
    territory_share: float,
    legitimacy: float,
    theta: float,
    economy: float,
    weights: SettlementBonusWeights,
    payoffs: TerminalPayoffs,
) -> float:
    """Compute the (state-scaled) negotiated-settlement payoff.

    Parameters
    ----------
    territory_share:
        Fraction of non-invader-home territories controlled by the invader at
        the moment of settlement, ∈ [0, 1]. 0 at reset, 1 at total conquest.
        Saturates at `weights.territory_saturation` — extra captures beyond
        that point give no further settlement credit. The point is to remove
        the incentive for maximalist conquest *as a means of getting a higher
        settlement*; territory beyond the saturation point should be pursued
        only because the agent wants conquest for its own sake (which pays
        the modest +10 conquest terminal, not the larger settlement bonus).
    legitimacy, economy:
        ∈ [0, 1]. Both directly enter the quality score.
    theta:
        ∈ [-1, +1]. Mapped to a `standing` score `(1 - θ) / 2` ∈ [0, 1] so that
        a neutral aligned with the invader (θ = -1) gives full credit and a
        neutral that has joined the defender (θ = +1) gives zero.
    weights:
        Per-dimension weights on the quality score.
    payoffs:
        Provides `settlement_min` and `settlement_max`.
    """
    legitimacy_span = max(1.0 - weights.legitimacy_floor, 1e-6)
    legitimacy_score = (legitimacy - weights.legitimacy_floor) / legitimacy_span
    legitimacy_score = max(0.0, min(1.0, legitimacy_score))
    standing_score = (weights.theta_ceiling - theta) / (weights.theta_ceiling + 1.0)
    standing_score = max(0.0, min(1.0, standing_score))
    sat = max(weights.territory_saturation, 1e-6)
    territory_term = min(territory_share, sat) / sat
    quality = (
        weights.territory * territory_term
        + weights.legitimacy * legitimacy_score
        + weights.standing * standing_score
        + weights.economy * economy
    )
    quality = float(max(0.0, min(1.0, quality)))
    span = payoffs.settlement_max - payoffs.settlement_min
    return payoffs.settlement_min + span * quality


# --------------------------------------------------------------------------------------
# Settlement acceptance.
#
# Defender accepts terms iff the campaign has produced enough pressure on them
# AND the invader's standing is healthy enough that an agreement is politically
# viable. Both sides must hold. See :class:`SettlementAcceptance` for the
# floors.
# --------------------------------------------------------------------------------------


@dataclass
class AcceptanceVerdict:
    accepted: bool
    pressure_ok: bool
    viability_ok: bool
    reason: str            # short tag for logging when `accepted=False`


def settlement_accepted(
    territory_share: float,
    defender_loss_fraction: float,
    pressure_streak: int,
    occupied_territory_turns: int,
    legitimacy: float,
    theta: float,
    sanctions_active: bool,
    cfg: SettlementAcceptance,
) -> AcceptanceVerdict:
    """Decide whether the defender will accept negotiation terms right now."""
    attrition_pressure = (
        defender_loss_fraction >= cfg.defender_loss_floor
        and territory_share > 0.0
    )
    instant_pressure = territory_share >= cfg.territory_floor or attrition_pressure
    pressure_ok = instant_pressure and (
        pressure_streak >= cfg.pressure_streak_floor
        or occupied_territory_turns >= cfg.occupied_turns_floor
    )
    viability_ok = (
        legitimacy >= cfg.legitimacy_floor
        and theta <= cfg.theta_ceiling
        and not (cfg.sanctions_block and sanctions_active)
    )
    if pressure_ok and viability_ok:
        return AcceptanceVerdict(True, True, True, "accepted")
    # Reason tag — first failure wins, useful for debugging policies.
    if not pressure_ok:
        return AcceptanceVerdict(False, False, viability_ok, "no_pressure")
    if legitimacy < cfg.legitimacy_floor:
        return AcceptanceVerdict(False, True, False, "legitimacy_too_low")
    if theta > cfg.theta_ceiling:
        return AcceptanceVerdict(False, True, False, "standing_too_low")
    if cfg.sanctions_block and sanctions_active:
        return AcceptanceVerdict(False, True, False, "sanctions_active")
    return AcceptanceVerdict(False, pressure_ok, viability_ok, "rejected")


# --------------------------------------------------------------------------------------
# Combat resolution.
#
# A pure function: given attacker and defender unit counts, terrain advantage, and a
# random draw, returns the post-combat counts and whether territory changed hands.
# --------------------------------------------------------------------------------------


@dataclass
class CombatOutcome:
    attacker_remaining: float
    defender_remaining: float
    attacker_won: bool


def resolve_combat(
    attackers: float,
    defenders: float,
    terrain_bonus_to_defender: float,
    params: CombatParams,
    rng: np.random.Generator,
) -> CombatOutcome:
    """Lanchester-flavoured one-shot resolution.

    The bonus multiplies the *effective* defender strength. The fight is decided by
    the ratio of effective strengths plus a small noise term so scouting is not perfect.
    """
    if attackers <= 0:
        return CombatOutcome(attackers, defenders, attacker_won=False)

    effective_defenders = defenders * (1.0 + terrain_bonus_to_defender)
    # Casualty model: each side loses a fraction of the *opposing* effective force.
    # Iter-8: rulebook §1 / §5-step-5 says combat is deterministic — noise
    # term that was here under iter-1..7 has been removed.
    attacker_losses = effective_defenders * params.attacker_loss_rate
    defender_losses = attackers * params.defender_loss_rate

    a_after = max(0.0, attackers - attacker_losses)
    d_after = max(0.0, defenders - defender_losses)
    won = a_after > d_after * (1.0 + terrain_bonus_to_defender) and d_after <= 0.5
    return CombatOutcome(a_after, d_after, won)


# --------------------------------------------------------------------------------------
# Convenience builders for trajectory logging.
# --------------------------------------------------------------------------------------


@dataclass
class StepTrace:
    """Lightweight per-step record we keep around for the UI and metrics logger."""

    t: int = 0
    theta: float = 0.0
    legitimacy: float = 1.0
    economy: float = 1.0
    t_occ: int = 0
    sanctions_active: bool = False
    insurgency_event: bool = False
    pol_action: int = 0
    mil_action: int = 0
    target: int = 0
    reward: float = 0.0
    reward_components: dict[str, float] = field(default_factory=dict)
