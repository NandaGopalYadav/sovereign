"""Single source of truth for every coefficient, threshold, and weight in the model.

All ablation regimes are expressed as flags on :class:`AblationFlags`. Edit values here
(or load a YAML and override) and every downstream calculation picks up the change.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------------------
# Drift-diffusion coefficients on the neutral posture θ.
# θ_{t+1} = clip(θ_t + μ(s,a) + ε,  -1, +1),    ε ~ N(0, σ²)
#
# The drift μ(s,a) is a linear combination of behavioural signals; coefficients below
# are documented in RESEARCH.md alongside the Diplomacy-paper context.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftCoefficients:
    """Coefficients on the rulebook Section 7.2 posture drift μ(s, a)."""

    alpha: float = 0.04   # legitimacy-loss coupling: (1 - L)
    beta: float = 0.05    # advance shock
    gamma: float = 0.10   # strike shock
    delta: float = 0.04   # substantive negotiation pull
    epsilon: float = 0.03 # seek-alliance pull
    zeta: float = 0.03    # occupation-duration pressure: t_occ / T_max
    sigma: float = 0.02   # noise std-dev


# --------------------------------------------------------------------------------------
# Threshold events. All are hysteretic: the *enter* threshold is strict, *exit* relaxed.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Thresholds:
    sanctions_on: float = 0.60       # θ > 0.60 triggers sanctions
    sanctions_off: float = 0.50      # θ must fall below this for N consecutive steps
    sanctions_off_streak: int = 5    # number of consecutive steps required to lift
    neutral_joins_defender: float = 0.85   # θ > 0.85
    supply_routes_open: float = -0.60      # θ < -0.60
    formal_alliance: float = -0.85         # θ < -0.85


# --------------------------------------------------------------------------------------
# Insurgency hazard.   p(insurgency | t_occ) = 1 - exp(-λ · t_occ)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HazardParams:
    # Iter-8 (strict rulebook conformance): reverted to the rulebook's
    # single-roll-per-turn hazard model using global `t_occ`, with the
    # rulebook's default λ = 0.05 (was 0.02 under iter-5's per-territory
    # model).
    lam: float = 0.05


# --------------------------------------------------------------------------------------
# Reward weights.  Per-step reward:
#   R = w_T·Σ connected resources + w_R·Δresources
#       - w_O·(t_occ / T_max) - w_L·(1 - L) - w_S·sanction_active·(1-E)
#       - w_I·insurgency_event
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardWeights:
    territory: float = 0.30
    resources: float = 0.20
    occupation: float = 0.25
    legitimacy: float = 0.15
    sanction: float = 0.20
    insurgency: float = 0.10


# Terminal payoffs — rulebook §9.
# Iter-8 reverts the state-scaled settlement bonus added in iter-2 to the
# rulebook's flat +40. The state-scaling fields (settlement_min,
# settlement_max) are removed; SettlementBonusWeights is kept defined for
# backwards compat but no longer read.
@dataclass(frozen=True)
class TerminalPayoffs:
    legitimacy_collapse: float = -50.0       # L ≤ 0
    invader_destroyed: float = -30.0         # all invader units gone
    negotiated_settlement: float = +40.0     # rulebook §9: flat +40
    timeout: float = 0.0
    total_conquest: float = +10.0            # intentionally modest


# --------------------------------------------------------------------------------------
# Settlement bonus weights. The bonus is a clipped weighted sum:
#
#   quality = w_T·territory_share + w_L·legitimacy + w_θ·standing + w_E·economy
#   bonus   = settlement_min + (settlement_max - settlement_min) · clip(quality, 0, 1)
#
# Where:
#   territory_share = (invader-controlled non-invader-home tiles) / (non-invader-home tiles)
#   standing        = (1 - θ) / 2                  ∈ [0, 1]   (θ=-1 → 1, θ=+1 → 0)
#
# Under ablation, mechanisms that are switched off keep their state variable
# at its initial value (e.g. L ≡ 1.0 under `no_legitimacy`), which makes the
# corresponding term effectively a constant. That is the point: when a
# mechanism is ablated, the agent no longer has to *plan against* that
# dimension, so the bonus is more easily maxed and the policy is predicted to
# regress toward conquest-leaning behaviour.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SettlementBonusWeights:
    territory: float = 0.20    # down-weighted; territory is "what was negotiated for", not "what defines a good settlement"
    legitimacy: float = 0.35   # up-weighted; whether the invader still has standing at home
    standing: float = 0.35     # up-weighted; whether the international community accepts the deal
    economy: float = 0.10
    territory_saturation: float = 1.0
    legitimacy_floor: float = 0.65
    theta_ceiling: float = 0.35


# --------------------------------------------------------------------------------------
# Settlement acceptance gate.
#
# Settlement is a *political result* of the campaign, not a button unlocked by
# elapsed time. The defender will only accept terms when they are under enough
# pressure to negotiate (`pressure_*` floors) AND the invader's standing is intact
# enough that an agreement is politically viable (`viability_*` floors).
#
# Both halves must hold. Either side missing means no settlement terminal —
# the agent must keep playing (or hit a different terminal).
#
# Under ablation, dimensions whose underlying state is frozen at its initial
# value (e.g. L≡1.0 under `no_legitimacy`) trivially satisfy their floor.
# That is the predicted regression: ablating a mechanism makes settlement
# easier to reach, so the agent's policy can drift toward conquest-leaning
# play without the compensating cost.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SettlementAcceptance:
    # Pressure floors — defender must be cornered enough to consider terms.
    territory_floor: float = 2.0 / 6.0     # ≥ 1/3 of non-home territory captured
    defender_loss_floor: float = 0.50      # ≥ 50% of defender ground units destroyed
    pressure_streak_floor: int = 5         # pressure must persist, not appear only at settlement
    occupied_turns_floor: int = 15         # cumulative occupied-territory-turns also satisfies pressure
    # Viability floors — international politics must be amenable.
    legitimacy_floor: float = 0.65         # invader still has domestic mandate
    theta_ceiling: float = 0.35            # not pushed into pre-sanctions territory
    sanctions_block: bool = True           # active sanctions block acceptance outright


# --------------------------------------------------------------------------------------
# Action set definitions — Sovereign.pdf Section 6 (rulebook-conformant).
#
# Iter-7: brought into strict alignment with the rulebook. Removed REDEPLOY
# (code-only addition with no rulebook analogue), moved WITHDRAW from political
# to military (rulebook places it under military with per-territory semantics),
# added IMPOSE_SANCTION (rulebook political action that was previously missing).
# --------------------------------------------------------------------------------------


POLITICAL_ACTIONS: tuple[str, ...] = (
    "SEEK_ALLIANCE",     # 0 — +0.01 L, −0.05 θ (toward Invader)
    "IMPOSE_SANCTION",   # 1 — −0.02 L, +0.04 θ (toward Defender), −0.03 to defender E
    "ISSUE_THREAT",      # 2 — −0.03 L, +0.03 θ
    "NEGOTIATE",         # 3 — +0.03 L, −0.04 θ (gated on pressure); proposes settlement
    "DO_NOTHING",        # 4 — no political move
)

MILITARY_ACTIONS: tuple[str, ...] = (
    "ADVANCE",           # 0 — claim adjacent territory, −0.05 L
    "HOLD",              # 1 — maintain current positions
    "WITHDRAW",          # 2 — cede one contested territory, +0.02 L
    "STRIKE",            # 3 — destroy one defender unit, −0.08 L
)


# --------------------------------------------------------------------------------------
# Force composition (initial unit counts and home effectiveness modifiers).
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ForceParams:
    invader_ground: int = 12
    invader_strike: int = 3
    defender_ground: int = 6
    defender_strike: int = 1
    neutral_ground: int = 4
    defender_home_effectiveness: float = 1.20  # +20% effectiveness on home territories


# --------------------------------------------------------------------------------------
# Combat & economy parameters. These are intentionally modest knobs that make the
# combat engine deterministic-ish; pure stochasticity lives in the drift noise.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CombatParams:
    attacker_loss_rate: float = 0.30
    defender_loss_rate: float = 0.40
    strike_attacker_damage: float = 2.5      # units removed from defender per strike
    strike_legitimacy_cost: float = 0.08     # L decrement per strike issued (rulebook §6.2 STRIKE: −0.08)
    advance_legitimacy_cost: float = 0.05    # L decrement per ADVANCE issued (rulebook §6.2 ADVANCE: −0.05)
    occupation_legitimacy_decay: float = 0.005  # L decrement per occupied territory per step
    sanction_economy_decay: float = 0.01     # E loss per step under sanctions (rulebook §7.3)
    insurgency_unit_loss: float = 1.0        # invader units lost when an insurgency fires
    # Iter-8 DO_NOTHING effects (rulebook §6.1 says "slow decay if L<0.5"
    # and "slow drift if t_occ>0" without quantifying — these are
    # 1/10-the-magnitude-of-STRIKE small values).
    do_nothing_l_decay: float = 0.005        # L drop per DO_NOTHING when L < 0.5
    do_nothing_theta_drift: float = 0.005    # θ drift toward defender per DO_NOTHING when t_occ > 0
    issue_threat_legitimacy_cost: float = 0.03
    issue_threat_theta_shift: float = 0.03
    seek_alliance_legitimacy_gain: float = 0.01
    seek_alliance_theta_shift: float = -0.05
    negotiate_legitimacy_gain: float = 0.03
    negotiate_theta_shift: float = -0.04
    # IMPOSE_SANCTION — rulebook Section 6.1 political action. Iter-7: added.
    # Effect is bookkeeping-only: the rulebook drains the target's economy,
    # but no further coupling is added (defender_economy isn't read elsewhere
    # for combat or reward). Documented as a rulebook-faithful action with
    # limited strategic value under the current reward structure.
    impose_sanction_legitimacy_cost: float = 0.02
    impose_sanction_theta_shift: float = 0.04
    impose_sanction_defender_economy_decay: float = 0.03
    # WITHDRAW (military, per-territory) — rulebook Section 6.2.
    withdraw_legitimacy_gain: float = 0.02
    neutral_join_defender_units: float = 2.0
    neutral_join_legitimacy_cost: float = 0.10
    supply_route_occupation_discount: float = 0.30
    formal_alliance_defender_economy_decay: float = 0.10
    formal_alliance_legitimacy_cost: float = 0.05
    disconnected_occupation_drag: float = 0.50


# --------------------------------------------------------------------------------------
# Episode-level limits.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeLimits:
    t_max: int = 60                      # max number of full turns
    twelve_step_substeps: int = 12       # per-turn substep count
    # earliest turn on which `negotiate` is even considered. Acts as a guardrail
    # *behind* the state-conditional acceptance gate. Value derived from
    # mechanism activation latency — see `experiments/derive_min_turn.py`.
    min_negotiation_turn: int = 15


# --------------------------------------------------------------------------------------
# Ablation flags. Toggling these reproduces the 5 regimes from Section 10.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationFlags:
    legitimacy_enabled: bool = True
    occupation_cost_enabled: bool = True
    neutral_posture_enabled: bool = True
    insurgency_enabled: bool = True
    sanctions_enabled: bool = True

    @staticmethod
    def regime(name: str) -> "AblationFlags":
        """Return the ablation regime named in the protocol."""
        match name:
            case "full":
                return AblationFlags()
            case "no_legitimacy":
                return AblationFlags(legitimacy_enabled=False)
            case "no_occupation_cost":
                return AblationFlags(occupation_cost_enabled=False)
            case "no_neutral":
                return AblationFlags(
                    neutral_posture_enabled=False, sanctions_enabled=False
                )
            case "baseline":
                return AblationFlags(
                    legitimacy_enabled=False,
                    occupation_cost_enabled=False,
                    neutral_posture_enabled=False,
                    insurgency_enabled=False,
                    sanctions_enabled=False,
                )
            case _:
                raise ValueError(f"Unknown ablation regime: {name}")


REGIME_NAMES: tuple[str, ...] = (
    "full",
    "no_legitimacy",
    "no_occupation_cost",
    "no_neutral",
    "baseline",
)


# --------------------------------------------------------------------------------------
# Top-level config object.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SovereignConfig:
    drift: DriftCoefficients = field(default_factory=DriftCoefficients)
    thresholds: Thresholds = field(default_factory=Thresholds)
    hazard: HazardParams = field(default_factory=HazardParams)
    reward: RewardWeights = field(default_factory=RewardWeights)
    terminal: TerminalPayoffs = field(default_factory=TerminalPayoffs)
    settlement: SettlementBonusWeights = field(default_factory=SettlementBonusWeights)
    acceptance: SettlementAcceptance = field(default_factory=SettlementAcceptance)
    force: ForceParams = field(default_factory=ForceParams)
    combat: CombatParams = field(default_factory=CombatParams)
    limits: EpisodeLimits = field(default_factory=EpisodeLimits)
    flags: AblationFlags = field(default_factory=AblationFlags)
    map_name: str = "rulebook9"

    def with_flags(self, flags: AblationFlags) -> "SovereignConfig":
        return replace(self, flags=flags)

    def with_overrides(self, **kwargs: Any) -> "SovereignConfig":
        return replace(self, **kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SovereignConfig":
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return _config_from_dict(raw or {})


def _config_from_dict(raw: dict[str, Any]) -> SovereignConfig:
    """Build a SovereignConfig from a partially-specified dict, filling defaults."""
    cfg = SovereignConfig()
    if "drift" in raw:
        cfg = replace(cfg, drift=DriftCoefficients(**(raw["drift"] or {})))
    if "thresholds" in raw:
        cfg = replace(cfg, thresholds=Thresholds(**(raw["thresholds"] or {})))
    if "hazard" in raw:
        cfg = replace(cfg, hazard=HazardParams(**(raw["hazard"] or {})))
    if "reward" in raw:
        cfg = replace(cfg, reward=RewardWeights(**(raw["reward"] or {})))
    if "terminal" in raw:
        cfg = replace(cfg, terminal=TerminalPayoffs(**(raw["terminal"] or {})))
    if "settlement" in raw:
        cfg = replace(cfg, settlement=SettlementBonusWeights(**(raw["settlement"] or {})))
    if "acceptance" in raw:
        cfg = replace(cfg, acceptance=SettlementAcceptance(**(raw["acceptance"] or {})))
    if "force" in raw:
        cfg = replace(cfg, force=ForceParams(**(raw["force"] or {})))
    if "combat" in raw:
        cfg = replace(cfg, combat=CombatParams(**(raw["combat"] or {})))
    if "limits" in raw:
        cfg = replace(cfg, limits=EpisodeLimits(**(raw["limits"] or {})))
    if "flags" in raw:
        cfg = replace(cfg, flags=AblationFlags(**(raw["flags"] or {})))
    if "map_name" in raw:
        cfg = replace(cfg, map_name=raw["map_name"])
    return cfg
