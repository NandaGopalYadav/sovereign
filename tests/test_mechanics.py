"""Unit tests for the pure-function mechanics layer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sovereign.env.config import (
    AblationFlags,
    DriftCoefficients,
    HazardParams,
    RewardWeights,
    SettlementAcceptance,
    SettlementBonusWeights,
    TerminalPayoffs,
    Thresholds,
)
from sovereign.env.mechanics import (
    DriftSignals,
    HysteresisState,
    compute_reward,
    drift,
    insurgency_probability,
    settlement_accepted,
    settlement_bonus,
    step_theta,
    update_threshold_events,
)


def test_drift_zero_signals_is_zero() -> None:
    s = DriftSignals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert drift(s, DriftCoefficients()) == 0.0


def test_drift_uses_published_coefficients() -> None:
    """μ should follow Section 7.2: α·Lloss + β·advance + γ·strike."""
    s = DriftSignals(1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    c = DriftCoefficients()
    assert drift(s, c) == pytest.approx(c.alpha + c.beta + c.gamma)


def test_step_theta_clamps_to_unit_interval() -> None:
    rng = np.random.default_rng(0)
    s = DriftSignals(1.0, 1.0, 1.0, 0.0, 0.0, 1.0)  # heavy upward push
    theta = 0.95
    for _ in range(10):
        theta = step_theta(theta, s, DriftCoefficients(), rng)
        assert -1.0 <= theta <= 1.0


def test_step_theta_clamps_lower_bound() -> None:
    rng = np.random.default_rng(0)
    s = DriftSignals(0.0, 0.0, 0.0, 1.0, 1.0, 0.0)  # heavy downward push
    theta = -0.95
    for _ in range(10):
        theta = step_theta(theta, s, DriftCoefficients(), rng)
        assert -1.0 <= theta <= 1.0


def test_drift_passive_play_does_not_drift_negative() -> None:
    """Iter-3 fix A: removing `defender_morale` means an idle invader (no
    aggression, no occupation, no concession) produces zero deterministic
    drift. Previously this was -δ per step regardless of action."""
    s = DriftSignals(
        legitimacy_loss=0.0,
        advance_action=0.0,
        strike_action=0.0,
        negotiate_action=0.0,
        seek_alliance_action=0.0,
        occupation_duration=0.0,
    )
    assert drift(s, DriftCoefficients()) == 0.0


def test_insurgency_probability_zero_at_zero_t_occ() -> None:
    assert insurgency_probability(0, HazardParams()) == 0.0


def test_insurgency_probability_increases_with_t_occ() -> None:
    h = HazardParams()
    seq = [insurgency_probability(t, h) for t in range(0, 30, 5)]
    assert all(b > a for a, b in zip(seq, seq[1:]))


def test_insurgency_probability_matches_formula() -> None:
    h = HazardParams(lam=0.05)
    expected = 1.0 - math.exp(-0.05 * 10)
    assert insurgency_probability(10, h) == pytest.approx(expected)


def test_sanctions_latch_with_hysteresis() -> None:
    h = HysteresisState()
    th = Thresholds()
    update_threshold_events(h, theta=0.65, th=th, sanctions_enabled=True)
    assert h.sanctions_active

    # θ dipping just below `sanctions_off` once does NOT clear the latch.
    update_threshold_events(h, theta=0.49, th=th, sanctions_enabled=True)
    assert h.sanctions_active
    assert h.sanctions_below_streak == 1


def test_sanctions_clear_only_after_streak() -> None:
    h = HysteresisState()
    th = Thresholds()
    update_threshold_events(h, theta=0.7, th=th, sanctions_enabled=True)
    assert h.sanctions_active
    for _ in range(th.sanctions_off_streak):
        update_threshold_events(h, theta=0.40, th=th, sanctions_enabled=True)
    assert not h.sanctions_active


def test_sanctions_streak_resets_on_jump_above() -> None:
    h = HysteresisState()
    th = Thresholds()
    update_threshold_events(h, theta=0.65, th=th, sanctions_enabled=True)
    update_threshold_events(h, theta=0.40, th=th, sanctions_enabled=True)
    assert h.sanctions_below_streak == 1
    update_threshold_events(h, theta=0.55, th=th, sanctions_enabled=True)
    assert h.sanctions_below_streak == 0
    assert h.sanctions_active


def test_alliance_and_supply_routes_latch_sticky() -> None:
    h = HysteresisState()
    th = Thresholds()
    update_threshold_events(h, theta=-0.9, th=th, sanctions_enabled=True)
    assert h.formal_alliance
    assert h.supply_routes_open
    # Even if θ swings positive later, alliance does not unlatch.
    update_threshold_events(h, theta=0.0, th=th, sanctions_enabled=True)
    assert h.formal_alliance
    assert h.supply_routes_open


def test_reward_components_apply_ablation_flags() -> None:
    flags = AblationFlags(
        legitimacy_enabled=False,
        occupation_cost_enabled=False,
        sanctions_enabled=False,
        insurgency_enabled=False,
    )
    rc = compute_reward(
        controlled_resources=0.4,
        delta_territory=0.5,
        delta_resources=0.3,
        t_occ=20,
        t_max=60,
        legitimacy=0.0,
        sanction_active=True,
        economy=0.5,
        insurgency_event=True,
        weights=RewardWeights(),
        flags=flags,
    )
    assert rc.occupation_cost == 0.0
    assert rc.legitimacy_cost == 0.0
    assert rc.sanction_cost == 0.0
    assert rc.insurgency_cost == 0.0
    # Only territory + resource gains remain.
    expected = 0.30 * 0.4 + 0.30 * 0.5 + 0.20 * 0.3
    assert rc.total() == pytest.approx(expected)


def test_settlement_is_flat_per_rulebook() -> None:
    """Iter-8 strict rulebook conformance: §9 specifies a flat +40 for
    negotiated settlement. The iter-2 state-scaled bonus formula has
    been reverted; this test is the regression guard."""
    payoffs = TerminalPayoffs()
    assert payoffs.negotiated_settlement == pytest.approx(40.0)


def test_settlement_accepted_requires_pressure() -> None:
    cfg = SettlementAcceptance()
    # Healthy invader, no pressure on defender — refused.
    v = settlement_accepted(
        territory_share=0.0,
        defender_loss_fraction=0.0,
        pressure_streak=0,
        occupied_territory_turns=0,
        legitimacy=1.0,
        theta=0.0,
        sanctions_active=False,
        cfg=cfg,
    )
    assert not v.accepted
    assert v.reason == "no_pressure"


def test_settlement_accepted_blocked_by_low_legitimacy() -> None:
    cfg = SettlementAcceptance()
    v = settlement_accepted(
        territory_share=0.5, defender_loss_fraction=0.0,
        pressure_streak=5, occupied_territory_turns=15,
        legitimacy=0.50, theta=0.0, sanctions_active=False, cfg=cfg,
    )
    assert not v.accepted
    assert v.reason == "legitimacy_too_low"


def test_settlement_accepted_blocked_by_high_theta() -> None:
    cfg = SettlementAcceptance()
    v = settlement_accepted(
        territory_share=0.5, defender_loss_fraction=0.0,
        pressure_streak=5, occupied_territory_turns=15,
        legitimacy=1.0, theta=0.5, sanctions_active=False, cfg=cfg,
    )
    assert not v.accepted
    assert v.reason == "standing_too_low"


def test_settlement_accepted_blocked_by_sanctions() -> None:
    cfg = SettlementAcceptance()
    v = settlement_accepted(
        territory_share=0.5, defender_loss_fraction=0.0,
        pressure_streak=5, occupied_territory_turns=15,
        legitimacy=1.0, theta=0.0, sanctions_active=True, cfg=cfg,
    )
    assert not v.accepted
    assert v.reason == "sanctions_active"


def test_settlement_accepted_when_both_floors_satisfied() -> None:
    cfg = SettlementAcceptance()
    v = settlement_accepted(
        territory_share=0.5, defender_loss_fraction=0.0,
        pressure_streak=5, occupied_territory_turns=15,
        legitimacy=0.85, theta=-0.1, sanctions_active=False, cfg=cfg,
    )
    assert v.accepted
    assert v.pressure_ok and v.viability_ok


def test_settlement_attrition_pressure_requires_foothold() -> None:
    """Defender losses count as pressure only when paired with some territorial foothold."""
    cfg = SettlementAcceptance()
    v = settlement_accepted(
        territory_share=0.0, defender_loss_fraction=0.6,
        pressure_streak=5, occupied_territory_turns=15,
        legitimacy=0.85, theta=-0.1, sanctions_active=False, cfg=cfg,
    )
    assert not v.accepted
    assert v.reason == "no_pressure"
    v = settlement_accepted(
        territory_share=0.1, defender_loss_fraction=0.6,
        pressure_streak=5, occupied_territory_turns=15,
        legitimacy=0.85, theta=-0.1, sanctions_active=False, cfg=cfg,
    )
    assert v.accepted


def test_settlement_rejects_instant_pressure_without_sustain() -> None:
    cfg = SettlementAcceptance()
    v = settlement_accepted(
        territory_share=0.5,
        defender_loss_fraction=0.0,
        pressure_streak=1,
        occupied_territory_turns=3,
        legitimacy=0.85,
        theta=-0.1,
        sanctions_active=False,
        cfg=cfg,
    )
    assert not v.accepted
    assert v.reason == "no_pressure"


def test_reward_full_regime_costs_everything() -> None:
    flags = AblationFlags()
    rc = compute_reward(
        controlled_resources=0.0,
        delta_territory=1.0,
        delta_resources=0.0,
        t_occ=30,
        t_max=60,
        legitimacy=0.5,
        sanction_active=True,
        economy=0.5,
        insurgency_event=True,
        weights=RewardWeights(),
        flags=flags,
    )
    assert rc.occupation_cost > 0
    assert rc.legitimacy_cost > 0
    assert rc.sanction_cost > 0
    assert rc.insurgency_cost > 0
