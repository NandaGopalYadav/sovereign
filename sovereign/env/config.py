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
    """Coefficients on the neutral posture drift μ(s, a)."""

    alpha: float = 0.04   # invader military aggression contribution (positive: pushes θ up)
    beta: float = 0.05    # occupation duration contribution
    gamma: float = 0.10   # legitimacy decay coupling
    delta: float = 0.04   # defender morale (defender alive + holding home)
    epsilon: float = 0.03 # invader-side political concession (negotiate / withdraw)
    zeta: float = 0.03    # economic supply pressure on neutral
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
    lam: float = 0.05  # λ in the exponential hazard


# --------------------------------------------------------------------------------------
# Reward weights.  Per-step reward:
#   R = w_T·Δterritory + w_R·Δresources
#       - w_O·(t_occ / T_max) - w_L·(1 - L) - w_S·sanction_active - w_I·insurgency_event
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardWeights:
    territory: float = 0.30
    resources: float = 0.20
    occupation: float = 0.25
    legitimacy: float = 0.15
    sanction: float = 0.20
    insurgency: float = 0.10


# Terminal payoffs added at the *end* of an episode on top of the per-step reward.
@dataclass(frozen=True)
class TerminalPayoffs:
    legitimacy_collapse: float = -50.0  # L ≤ 0
    invader_destroyed: float = -30.0    # all invader units gone
    negotiated_settlement: float = +40.0
    timeout: float = 0.0
    total_conquest: float = +10.0       # intentionally modest


# --------------------------------------------------------------------------------------
# Action set definitions. The "rulebook" referenced in the spec was not attached;
# these are the canonical 5 political and 4 military action labels used internally.
# --------------------------------------------------------------------------------------


POLITICAL_ACTIONS: tuple[str, ...] = (
    "hold",          # 0 — no political move this turn
    "propaganda",    # 1 — claim legitimacy; small θ shift in invader's favour
    "negotiate",     # 2 — offer settlement; if defender accepts, episode ends
    "coerce",        # 3 — ultimatum; pushes θ adversely if rejected
    "withdraw",      # 4 — declare withdrawal from occupied territories; resets t_occ
)

MILITARY_ACTIONS: tuple[str, ...] = (
    "hold",          # 0 — no military move
    "attack",        # 1 — move ground units to target territory, resolve combat
    "redeploy",      # 2 — repositions ground units between friendly territories
    "strike",        # 3 — strike unit on target; high damage, legitimacy cost
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
    strike_legitimacy_cost: float = 0.05     # L decrement per strike issued
    occupation_legitimacy_decay: float = 0.005  # L decrement per occupied territory per step
    sanction_economy_decay: float = 0.02     # E loss per step under sanctions
    insurgency_unit_loss: float = 1.0        # invader units lost when an insurgency fires


# --------------------------------------------------------------------------------------
# Episode-level limits.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeLimits:
    t_max: int = 60                      # max number of full turns
    twelve_step_substeps: int = 12       # per-turn substep count


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
    force: ForceParams = field(default_factory=ForceParams)
    combat: CombatParams = field(default_factory=CombatParams)
    limits: EpisodeLimits = field(default_factory=EpisodeLimits)
    flags: AblationFlags = field(default_factory=AblationFlags)
    map_name: str = "default9"

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
