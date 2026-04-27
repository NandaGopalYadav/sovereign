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
    Thresholds,
)
from sovereign.env.mechanics import (
    DriftSignals,
    HysteresisState,
    compute_reward,
    drift,
    insurgency_probability,
    step_theta,
    update_threshold_events,
)


def test_drift_zero_signals_is_zero() -> None:
    s = DriftSignals(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert drift(s, DriftCoefficients()) == 0.0


def test_drift_uses_published_coefficients() -> None:
    """μ should be α·1 + β·1 + γ·1 - δ·0 - ε·0 + ζ·0 = α + β + γ."""
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
        delta_territory=0.5,
        delta_resources=0.3,
        t_occ=20,
        t_max=60,
        legitimacy=0.0,
        sanction_active=True,
        insurgency_event=True,
        weights=RewardWeights(),
        flags=flags,
    )
    assert rc.occupation_cost == 0.0
    assert rc.legitimacy_cost == 0.0
    assert rc.sanction_cost == 0.0
    assert rc.insurgency_cost == 0.0
    # Only territory + resource gains remain.
    expected = 0.30 * 0.5 + 0.20 * 0.3
    assert rc.total() == pytest.approx(expected)


def test_reward_full_regime_costs_everything() -> None:
    flags = AblationFlags()
    rc = compute_reward(
        delta_territory=1.0,
        delta_resources=0.0,
        t_occ=30,
        t_max=60,
        legitimacy=0.5,
        sanction_active=True,
        insurgency_event=True,
        weights=RewardWeights(),
        flags=flags,
    )
    assert rc.occupation_cost > 0
    assert rc.legitimacy_cost > 0
    assert rc.sanction_cost > 0
    assert rc.insurgency_cost > 0
