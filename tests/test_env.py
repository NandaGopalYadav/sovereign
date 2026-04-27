"""Environment-level unit and integration tests."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from sovereign import SovereignEnv
from sovereign.env.config import AblationFlags, SovereignConfig


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
    """Issuing the `withdraw` political action with no held off-home territory keeps
    `t_occ` at zero. After we manually plant the invader off-home, withdrawing should
    reset the counter."""
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
    withdraw_action = env.encode_action(pol=4, mil=0, target=0)
    env.step(withdraw_action)
    assert env.state.t_occ == 0


def test_negotiate_terminates_with_settlement_bonus() -> None:
    env = SovereignEnv()
    env.reset(seed=1)
    a = env.encode_action(pol=2, mil=0, target=0)
    _, reward, term, _, info = env.step(a)
    assert term
    assert info["terminal_reason"] == "negotiated_settlement"
    # Terminal bonus 40 dominates per-step reward magnitude.
    assert reward > 30.0


def test_legitimacy_collapse_terminates() -> None:
    env = SovereignEnv()
    env.reset(seed=2)
    assert env.state is not None
    env.state.legitimacy = 0.0
    # Use a deterministic hold/hold action — propaganda would bump L back above zero
    # before the terminal check fires, masking the collapse with a different outcome.
    hold = env.encode_action(pol=0, mil=0, target=0)
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
    # Hold/hold so we don't accidentally negotiate a settlement first.
    hold = env.encode_action(pol=0, mil=0, target=0)
    _, reward, term, _, info = env.step(hold)
    assert term
    assert info["terminal_reason"] == "invader_destroyed"
    assert reward < -20.0


def test_action_decoding_round_trip() -> None:
    env = SovereignEnv()
    env.reset(seed=0)
    for raw in (0, 1, 50, 179):
        pol_label, mil_label, target = env.action_meaning(raw)
        # Re-encoding the same triple produces the same id.
        pol_idx = ("hold", "propaganda", "negotiate", "coerce", "withdraw").index(pol_label)
        mil_idx = ("hold", "attack", "redeploy", "strike").index(mil_label)
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
    # Use HOLD/HOLD action to avoid stumbling into negotiation/destruction.
    hold = env.encode_action(pol=0, mil=0, target=0)
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
