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
    """The behavioural inputs that feed the drift function μ(s, a).

    Each is a normalized scalar in roughly [0, 1] (legitimacy is already in [0,1]).
    `economic_pressure` is `1 - E` so a depleted economy pushes θ up.
    """

    invader_aggression: float          # ∈ [0,1] — fraction of last turn's mil actions that were attacks/strikes
    occupation_fraction: float         # ∈ [0,1] — share of non-invader territories occupied
    legitimacy_loss: float             # ∈ [0,1] — equals (1 - L)
    defender_morale: float             # ∈ [0,1] — defender alive AND holding home capital
    invader_concession: float          # ∈ [0,1] — 1 if last political move was negotiate or withdraw
    economic_pressure: float           # ∈ [0,1] — equals (1 - E)


def drift(signals: DriftSignals, c: DriftCoefficients) -> float:
    """Deterministic part of the θ update."""
    return (
        c.alpha * signals.invader_aggression
        + c.beta * signals.occupation_fraction
        + c.gamma * signals.legitimacy_loss
        - c.delta * signals.defender_morale
        - c.epsilon * signals.invader_concession
        + c.zeta * signals.economic_pressure
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

    territory_gain: float = 0.0
    resource_gain: float = 0.0
    occupation_cost: float = 0.0
    legitimacy_cost: float = 0.0
    sanction_cost: float = 0.0
    insurgency_cost: float = 0.0
    terminal_bonus: float = 0.0

    def total(self) -> float:
        return (
            self.territory_gain
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
            "resource_gain": self.resource_gain,
            "occupation_cost": -self.occupation_cost,
            "legitimacy_cost": -self.legitimacy_cost,
            "sanction_cost": -self.sanction_cost,
            "insurgency_cost": -self.insurgency_cost,
            "terminal_bonus": self.terminal_bonus,
            "total": self.total(),
        }


def compute_reward(
    delta_territory: float,
    delta_resources: float,
    t_occ: int,
    t_max: int,
    legitimacy: float,
    sanction_active: bool,
    insurgency_event: bool,
    weights: RewardWeights,
    flags: AblationFlags,
) -> RewardComponents:
    """Apply the per-step reward formula, gated by ablation flags."""
    rc = RewardComponents()
    rc.territory_gain = weights.territory * delta_territory
    rc.resource_gain = weights.resources * delta_resources
    if flags.occupation_cost_enabled:
        rc.occupation_cost = weights.occupation * (t_occ / max(t_max, 1))
    if flags.legitimacy_enabled:
        rc.legitimacy_cost = weights.legitimacy * (1.0 - legitimacy)
    if flags.sanctions_enabled and sanction_active:
        rc.sanction_cost = weights.sanction
    if flags.insurgency_enabled and insurgency_event:
        rc.insurgency_cost = weights.insurgency
    return rc


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
    attacker_losses = effective_defenders * params.attacker_loss_rate
    defender_losses = attackers * params.defender_loss_rate

    # Small symmetric noise so identical force ratios don't always resolve identically.
    attacker_losses *= 1.0 + rng.normal(0.0, 0.05)
    defender_losses *= 1.0 + rng.normal(0.0, 0.05)

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
