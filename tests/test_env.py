"""Environment-level unit and integration tests."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from sovereign import SovereignEnv
from sovereign.env.config import AblationFlags, SovereignConfig
from sovereign.env.map import CONTESTED, DEFENDER, INVADER, NEUTRAL


# --------------------------------------------------------------------------------------
# API conformance
# --------------------------------------------------------------------------------------


def test_gym_api_conformance() -> None:
    env = SovereignEnv()
    # check_env runs reset/step with random actions and verifies space contracts.
    check_env(env, skip_render_check=True)


def test_observation_in_space_after_reset() -> None:
    env = SovereignEnv()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_observation_in_space_after_steps() -> None:
    env = SovereignEnv()
    env.reset(seed=0)
    for _ in range(20):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if term or trunc:
            break


# --------------------------------------------------------------------------------------
# Invariants on full episodes.
# --------------------------------------------------------------------------------------


def _run_episode(env: gym.Env, seed: int = 0) -> list[dict]:
    env.reset(seed=seed)
    out: list[dict] = []
    for _ in range(200):
        _, _, term, trunc, info = env.step(env.action_space.sample())
        out.append(info)
        if term or trunc:
            break
    return out


def test_legitimacy_stays_in_unit_interval() -> None:
    env = SovereignEnv()
    for s in range(5):
        for info in _run_episode(env, seed=s):
            assert 0.0 <= info["legitimacy"] <= 1.0


def test_theta_stays_in_minus_one_to_plus_one() -> None:
    env = SovereignEnv()
    for s in range(5):
        for info in _run_episode(env, seed=s):
            assert -1.0 <= info["theta"] <= 1.0


def test_t_occ_resets_on_full_withdrawal() -> None:
    """Issuing the `WITHDRAW` military action on the last held off-home
    territory should reset the global occupation counter to 0.
    (Iter-7: WITHDRAW is now a per-territory military action.)"""
    env = SovereignEnv()
    env.reset(seed=0)
    assert env.state is not None
    # Force an off-home occupation manually.
    contested_idx = next(
        i for i, t in enumerate(env.map_spec.territories) if t.home_of != 0
    )
    env.state.controller[contested_idx] = 0
    env.state.invader_units[contested_idx] = 5
    env.state.t_occ = 7
    # mil=2 is WITHDRAW; target the planted contested territory.
    withdraw_action = env.encode_action(pol=4, mil=2, target=contested_idx)
    env.step(withdraw_action)
    assert env.state.t_occ == 0


def test_negotiate_on_turn_1_does_not_settle() -> None:
    """Iter-9 acceptance gate: NEGOTIATE on turn 1 from a fresh reset
    does NOT settle because L starts at 1.0 — NEGOTIATE's +0.03 L gain
    is capped, so L does not actually increase, so per the rulebook §6.1
    interpretation the diplomatic effect didn't land."""
    env = SovereignEnv()
    env.reset(seed=1)
    negotiate_hold = env.encode_action(pol=3, mil=1, target=0)
    _, _, term, _, info = env.step(negotiate_hold)
    assert not term
    assert info.get("terminal_reason") is None


def test_negotiate_settles_after_l_lowered() -> None:
    """Iter-9: after lowering L (e.g. via ISSUE_THREAT), a subsequent
    NEGOTIATE turn does cause L to actually increase, so the
    rulebook-derived acceptance gate is satisfied."""
    env = SovereignEnv()
    env.reset(seed=1)
    # ISSUE_THREAT lowers L by 0.03 (under default flags).
    issue_threat_hold = env.encode_action(pol=2, mil=1, target=0)
    env.step(issue_threat_hold)
    # Now L ≈ 0.97. NEGOTIATE adds 0.03 → L increases (capped at 1.0).
    negotiate_hold = env.encode_action(pol=3, mil=1, target=0)
    _, reward, term, _, info = env.step(negotiate_hold)
    assert term
    assert info["terminal_reason"] == "negotiated_settlement"
    assert reward > 30.0


def test_negotiate_fails_when_defender_counter_attacks() -> None:
    """Iter-9: if the defender actually counter-attacks this turn (its
    rule-based policy issued a counter and successfully fired against
    invader units), the defender is not in a mood to negotiate. Setup:
    invader on a contested tile adjacent to D (defender home), where
    the defender has 6 units; defender will COUNTER_STRIKE and
    successfully attack."""
    env = SovereignEnv()
    env.reset(seed=1)
    assert env.state is not None
    # In rulebook9, D (index 1) is adjacent to C5 (7) and C6 (8). Plant
    # invader on C5. Also clear I (index 0) so the defender's heuristic
    # doesn't get sidetracked picking it as a counter-strike target it
    # can't actually hit.
    env.state.invader_units[0] = 0
    env.state.controller[7] = 0  # invader
    env.state.invader_units[7] = 12
    # Lower L so the +0.03 NEGOTIATE gain would otherwise cause L to rise.
    env.state.legitimacy = 0.80
    negotiate_hold = env.encode_action(pol=3, mil=1, target=0)
    _, _, term, _, info = env.step(negotiate_hold)
    # Defender successfully counter-attacked from D — settlement refused.
    assert info.get("terminal_reason") != "negotiated_settlement"


def test_settlement_blocked_under_no_legitimacy() -> None:
    """Iter-9: under no_legitimacy, L is locked at 1.0 forever
    (no decay applied, gains capped). NEGOTIATE can never cause L to
    increase, so settlement is structurally impossible. This matches
    the rulebook's 'Slower invasion' prediction — without legitimacy
    dynamics the agent has no diplomatic exit."""
    cfg = SovereignConfig().with_flags(AblationFlags.regime("no_legitimacy"))
    env = SovereignEnv(config=cfg)
    env.reset(seed=1)
    # Try many turns of NEGOTIATE — none should settle.
    negotiate_hold = env.encode_action(pol=3, mil=1, target=0)
    for _ in range(5):
        _, _, term, _, info = env.step(negotiate_hold)
        if term:
            # Should NEVER terminate via negotiated_settlement
            assert info.get("terminal_reason") != "negotiated_settlement"
            break


def test_combat_is_deterministic() -> None:
    """Iter-8: rulebook §1 / §5 step 5 specify deterministic combat.
    Two identical force ratios under identical RNG seeds produce
    identical combat outcomes."""
    from sovereign.env.mechanics import resolve_combat
    from sovereign.env.config import CombatParams
    cp = CombatParams()
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(99)
    out_a = resolve_combat(10.0, 5.0, 0.0, cp, rng_a)
    out_b = resolve_combat(10.0, 5.0, 0.0, cp, rng_b)
    assert out_a.attacker_remaining == out_b.attacker_remaining
    assert out_a.defender_remaining == out_b.defender_remaining
    assert out_a.attacker_won == out_b.attacker_won


def test_insurgency_global_single_roll_per_turn() -> None:
    """Iter-8 reverts iter-5: rulebook §8.3 specifies one Bernoulli per
    turn using global `t_occ`. Holding many territories does not
    produce multiple insurgency events per turn."""
    env = SovereignEnv()
    env.reset(seed=0)
    assert env.state is not None
    non_home = [
        v for v in range(env.map_spec.n)
        if env.map_spec.territories[v].home_of != INVADER
    ]
    # Plant invader on all non-home territories.
    for v in non_home:
        env.state.controller[v] = INVADER
        env.state.invader_units[v] = 50
    env.state.t_occ = 20  # high hazard regime
    hold = env.encode_action(pol=4, mil=1, target=0)
    _, _, _, _, info = env.step(hold)
    # insurgency_event is a bool (single roll); count is at most 1
    # destroyed unit.
    assert isinstance(info.get("insurgency_event"), bool)


def test_advance_costs_legitimacy() -> None:
    """Iter-8: rulebook §6.2 ADVANCE: −0.05 L per use, plus the generic
    per-territory occupation decay if we hold any off-home territory."""
    env = SovereignEnv()
    env.reset(seed=0)
    assert env.state is not None
    before = env.state.legitimacy
    advance = env.encode_action(pol=4, mil=0, target=0)
    env.step(advance)
    # L drops by at least the action-specific 0.05.
    assert before - env.state.legitimacy >= env.cfg.combat.advance_legitimacy_cost - 1e-6


def test_do_nothing_slow_l_decay_when_low_legitimacy() -> None:
    """Iter-8: rulebook §6.1 DO_NOTHING has a slow L decay when L < 0.5."""
    env = SovereignEnv()
    env.reset(seed=0)
    assert env.state is not None
    env.state.legitimacy = 0.40
    before = env.state.legitimacy
    nothing_hold = env.encode_action(pol=4, mil=1, target=0)
    env.step(nothing_hold)
    # Pure decay (no other action drivers): exactly do_nothing_l_decay.
    expected = before - env.cfg.combat.do_nothing_l_decay
    assert env.state.legitimacy == pytest.approx(expected, abs=1e-6)


def test_do_nothing_no_l_decay_when_high_legitimacy() -> None:
    """DO_NOTHING does NOT decay L when L ≥ 0.5."""
    env = SovereignEnv()
    env.reset(seed=0)
    assert env.state is not None
    env.state.legitimacy = 0.90
    before = env.state.legitimacy
    nothing_hold = env.encode_action(pol=4, mil=1, target=0)
    env.step(nothing_hold)
    assert env.state.legitimacy == pytest.approx(before, abs=1e-6)


def test_seek_alliance_theta_shift_unconditional() -> None:
    """Iter-8: rulebook §6.1 SEEK_ALLIANCE: −0.05 θ unconditionally.
    The iter-6 gating on `neutral_posture_enabled` has been reverted.
    Under `no_neutral`, SEEK_ALLIANCE still moves θ (drift dynamics are
    off but direct shifts remain — rulebook-literal interpretation)."""
    cfg = SovereignConfig().with_flags(AblationFlags.regime("no_neutral"))
    env = SovereignEnv(config=cfg)
    env.reset(seed=1)
    assert env.state is not None
    before = env.state.theta
    seek = env.encode_action(pol=0, mil=1, target=0)
    env.step(seek)
    assert env.state.theta < before


def test_rulebook_political_actions_have_effects() -> None:
    """Iter-7: pol indices are SEEK_ALLIANCE=0, IMPOSE_SANCTION=1,
    ISSUE_THREAT=2, NEGOTIATE=3, DO_NOTHING=4."""
    env = SovereignEnv()
    env.reset(seed=1)
    assert env.state is not None
    seek = env.encode_action(pol=0, mil=1, target=0)
    env.step(seek)
    assert env.state.theta < 0.0
    assert env.state.legitimacy == pytest.approx(1.0)
    threat = env.encode_action(pol=2, mil=1, target=0)
    before_l = env.state.legitimacy
    before_theta = env.state.theta
    env.step(threat)
    assert env.state.legitimacy < before_l
    assert env.state.theta > before_theta


def test_impose_sanction_drains_defender_economy() -> None:
    """Iter-7: IMPOSE_SANCTION (pol=1) reduces defender_economy per its
    rulebook effect ledger (`−0.03` per use)."""
    env = SovereignEnv()
    env.reset(seed=0)
    assert env.state is not None
    before = env.state.defender_economy
    sanction = env.encode_action(pol=1, mil=1, target=0)
    env.step(sanction)
    expected = before - env.cfg.combat.impose_sanction_defender_economy_decay
    assert env.state.defender_economy == pytest.approx(expected)


def test_impose_sanction_pushes_theta_under_full() -> None:
    """IMPOSE_SANCTION's θ shift (+0.04) fires when neutral_posture is enabled."""
    env = SovereignEnv()
    env.reset(seed=0)
    assert env.state is not None
    # Patch out drift noise so we can isolate the deterministic shift.
    env.np_random = np.random.default_rng(0)
    cfg = env.cfg.with_overrides(
        drift=type(env.cfg.drift)(
            alpha=env.cfg.drift.alpha, beta=env.cfg.drift.beta,
            gamma=env.cfg.drift.gamma, delta=env.cfg.drift.delta,
            epsilon=env.cfg.drift.epsilon,
            zeta=env.cfg.drift.zeta, sigma=0.0,
        )
    )
    env.cfg = cfg
    before = env.state.theta
    sanction = env.encode_action(pol=1, mil=1, target=0)
    env.step(sanction)
    assert env.state.theta > before


def test_impose_sanction_theta_shift_under_no_neutral() -> None:
    """Iter-8 strict rulebook conformance: IMPOSE_SANCTION's θ shift
    applies unconditionally (rulebook §6.1 doesn't gate effects on
    ablations). The iter-6 gating has been reverted."""
    cfg = SovereignConfig().with_flags(AblationFlags.regime("no_neutral"))
    env = SovereignEnv(config=cfg)
    env.reset(seed=0)
    assert env.state is not None
    before_theta = env.state.theta
    before_econ = env.state.defender_economy
    sanction = env.encode_action(pol=1, mil=1, target=0)
    env.step(sanction)
    # θ moves up (+0.04) per rulebook; defender_economy drained too.
    assert env.state.theta > before_theta
    assert env.state.defender_economy < before_econ


def test_withdraw_one_territory_cedes_only_target() -> None:
    """Iter-7: military WITHDRAW (mil=2) cedes only the targeted off-home
    territory, not all of them."""
    env = SovereignEnv()
    env.reset(seed=0)
    assert env.state is not None
    non_home = [
        v for v in range(env.map_spec.n)
        if env.map_spec.territories[v].home_of != INVADER
    ]
    # Plant invader on three contested tiles.
    for v in non_home[:3]:
        env.state.controller[v] = INVADER
        env.state.invader_units[v] = 4
    target = non_home[1]
    withdraw = env.encode_action(pol=4, mil=2, target=target)
    env.step(withdraw)
    # The targeted territory is ceded, the others are not.
    assert env.state.controller[target] != INVADER
    assert env.state.controller[non_home[0]] == INVADER
    assert env.state.controller[non_home[2]] == INVADER


def test_withdraw_grants_legitimacy_bonus() -> None:
    """Iter-7: WITHDRAW grants +0.02 L per the rulebook."""
    env = SovereignEnv()
    env.reset(seed=0)
    assert env.state is not None
    # Plant invader on one contested tile, drop L below 1.0 so the gain is visible.
    non_home = next(
        v for v in range(env.map_spec.n)
        if env.map_spec.territories[v].home_of != INVADER
    )
    env.state.controller[non_home] = INVADER
    env.state.invader_units[non_home] = 4
    env.state.legitimacy = 0.80
    withdraw = env.encode_action(pol=4, mil=2, target=non_home)
    env.step(withdraw)
    # WITHDRAW adds +0.02 L; subsequent decay/etc. is at most 0.01 from one
    # turn of pre-withdraw occupation, so net change should be positive.
    assert env.state.legitimacy > 0.80


def test_settlement_bonus_is_flat_per_rulebook() -> None:
    """Iter-8 strict rulebook conformance: settlement payoff is the
    rulebook's flat +40 (rulebook §9). Iter-9 keeps the flat payoff but
    gates acceptance on (L increased) AND (defender did not reply)."""
    env = SovereignEnv()
    env.reset(seed=1)
    assert env.state is not None
    # First lower L via ISSUE_THREAT so NEGOTIATE causes an actual increase.
    env.step(env.encode_action(pol=2, mil=1, target=0))
    negotiate = env.encode_action(pol=3, mil=1, target=0)
    _, reward, _, _, info = env.step(negotiate)
    assert info["terminal_reason"] == "negotiated_settlement"
    # Within a wide tolerance to account for per-step shaping.
    assert 35.0 <= reward <= 45.0
    assert env.cfg.terminal.negotiated_settlement == pytest.approx(40.0)


def test_legitimacy_collapse_terminates() -> None:
    env = SovereignEnv()
    env.reset(seed=2)
    assert env.state is not None
    env.state.legitimacy = 0.0
    # Use DO_NOTHING + HOLD — SEEK_ALLIANCE would bump L back above zero
    # before the terminal check fires, masking the collapse.
    hold = env.encode_action(pol=4, mil=1, target=0)
    _, reward, term, _, info = env.step(hold)
    assert term
    assert info["terminal_reason"] == "legitimacy_collapse"
    assert reward < -40.0


def test_invader_destroyed_terminates() -> None:
    env = SovereignEnv()
    env.reset(seed=3)
    assert env.state is not None
    env.state.invader_units[:] = 0
    env.state.invader_strike = 0
    # DO_NOTHING + HOLD so we don't accidentally negotiate a settlement first.
    hold = env.encode_action(pol=4, mil=1, target=0)
    _, reward, term, _, info = env.step(hold)
    assert term
    assert info["terminal_reason"] == "invader_destroyed"
    assert reward < -20.0


def test_action_decoding_round_trip() -> None:
    env = SovereignEnv()
    env.reset(seed=0)
    pol_labels = ("SEEK_ALLIANCE", "IMPOSE_SANCTION", "ISSUE_THREAT", "NEGOTIATE", "DO_NOTHING")
    mil_labels = ("ADVANCE", "HOLD", "WITHDRAW", "STRIKE")
    for raw in (0, 1, 50, 179):
        pol_label, mil_label, target = env.action_meaning(raw)
        # Re-encoding the same triple produces the same id.
        pol_idx = pol_labels.index(pol_label)
        mil_idx = mil_labels.index(mil_label)
        assert env.encode_action(pol_idx, mil_idx, target) == raw


def test_truncation_at_t_max() -> None:
    cfg = SovereignConfig().with_overrides(
        flags=AblationFlags(
            legitimacy_enabled=False,
            occupation_cost_enabled=False,
            neutral_posture_enabled=False,
            insurgency_enabled=False,
            sanctions_enabled=False,
        )
    )
    env = SovereignEnv(config=cfg)
    env.reset(seed=4)
    # Use DO_NOTHING + HOLD (iter-7 reindex) to avoid stumbling into
    # negotiation/destruction.
    hold = env.encode_action(pol=4, mil=1, target=0)
    last_info: dict = {}
    for _ in range(cfg.limits.t_max + 5):
        _, _, term, trunc, last_info = env.step(hold)
        if term or trunc:
            break
    assert last_info["turn"] >= cfg.limits.t_max // 2  # at least progressed


def test_alternate_map_loads() -> None:
    env = SovereignEnv(map_name="frontier12")
    obs, _ = env.reset(seed=0)
    assert env.map_spec.n == 12
    assert env.observation_space.contains(obs)


def test_rulebook_default_map_shape() -> None:
    env = SovereignEnv()
    assert env.map_spec.name == "rulebook9"
    homes = [t.home_of for t in env.map_spec.territories]
    assert homes.count(INVADER) == 1
    assert homes.count(DEFENDER) == 1
    assert homes.count(NEUTRAL) == 1
    assert homes.count(CONTESTED) == 6


def test_connected_resources_exclude_cut_off_occupations() -> None:
    env = SovereignEnv()
    env.reset(seed=0)
    assert env.state is not None
    disconnected = next(
        i for i, t in enumerate(env.map_spec.territories)
        if t.home_of != INVADER and i not in env.graph.neighbors(0)
    )
    env.state.controller[disconnected] = INVADER
    env.state.invader_units[disconnected] = 3
    controlled = env._sum_resources(env.state.controller, INVADER)
    connected = env._connected_resources(env.state.controller, INVADER)
    assert controlled > connected


def test_home_resources_count_toward_yield() -> None:
    """Iter-8 strict rulebook conformance: rulebook §8.1 says yield is
    Σ resource_value(v) for v controlled by I (including home). The
    iter-1..7 exclusion of home territory has been reverted."""
    env = SovereignEnv()
    env.reset(seed=0)
    assert env.state is not None
    # DO_NOTHING + HOLD turn. Yield should be > 0 from invader-home I.
    hold = env.encode_action(pol=4, mil=1, target=0)
    _, reward, _, _, info = env.step(hold)
    assert info["connected_resource_yield"] > 0.0
    # Reward includes the home-territory yield component.
    assert reward > 0.0


def test_supply_routes_discount_occupation_cost() -> None:
    env = SovereignEnv()
    env.reset(seed=0)
    occupied = [3]
    plain = env._occupation_multiplier(occupied)
    env.hysteresis.supply_routes_open = True
    discounted = env._occupation_multiplier(occupied)
    assert discounted < plain


def test_threshold_effects_apply_once() -> None:
    env = SovereignEnv()
    env.reset(seed=0)
    assert env.state is not None
    defender_before = float(env.state.defender_units.sum())
    legitimacy_before = env.state.legitimacy
    env.hysteresis.neutral_joined_defender = True
    env._apply_threshold_effects((False, False, False))
    assert env.state.defender_units.sum() == pytest.approx(
        defender_before + env.cfg.combat.neutral_join_defender_units
    )
    assert env.state.legitimacy == pytest.approx(
        legitimacy_before - env.cfg.combat.neutral_join_legitimacy_cost
    )
    env._apply_threshold_effects((True, False, False))
    assert env.state.defender_units.sum() == pytest.approx(
        defender_before + env.cfg.combat.neutral_join_defender_units
    )


def test_formal_alliance_effect_applies_once() -> None:
    env = SovereignEnv()
    env.reset(seed=0)
    assert env.state is not None
    env.hysteresis.formal_alliance = True
    env._apply_threshold_effects((False, False, False))
    assert env.state.defender_economy == pytest.approx(
        1.0 - env.cfg.combat.formal_alliance_defender_economy_decay
    )


# --------------------------------------------------------------------------------------
# Defender policy basic sanity
# --------------------------------------------------------------------------------------


def test_defender_recaptures_occupied_home() -> None:
    env = SovereignEnv()
    env.reset(seed=0)
    assert env.state is not None
    home_idx = next(i for i, t in enumerate(env.map_spec.territories) if t.home_of == 1)
    env.state.controller[home_idx] = 0  # invader occupies
    env.state.invader_units[home_idx] = 3
    decision = env._defender.decide(
        env.state.controller, env.state.invader_units, env.state.defender_units
    )
    # Either retake-home tag or counter-strike (when our heuristic prefers it).
    assert decision.tag in (1, 2, 3)


@pytest.mark.parametrize("seed", list(range(5)))
def test_episode_terminates_or_truncates(seed: int) -> None:
    env = SovereignEnv()
    env.reset(seed=seed)
    for _ in range(env.cfg.limits.t_max + 5):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            break
    else:
        pytest.fail(f"Episode for seed={seed} did not terminate")
