"""Demo runner: load a trained PPO checkpoint and play N episodes with a
readable turn-by-turn trace.

Usage:
    uv run python -m sovereign.experiments.demo_episodes \
        --regime full --episodes 3 --seed 0

Prints one block per episode:
    - Turn header with action label, target territory, reward
    - Running L / E / θ / unit counts
    - Settlement / timeout / collapse outcome line
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from sovereign.env.config import (
    MILITARY_ACTIONS,
    POLITICAL_ACTIONS,
    REGIME_NAMES,
    AblationFlags,
    SovereignConfig,
)
from sovereign.env.sovereign_env import SovereignEnv


def _decode(action: int, n_targets: int) -> tuple[str, str, int]:
    pol = action // (len(MILITARY_ACTIONS) * n_targets)
    rem = action % (len(MILITARY_ACTIONS) * n_targets)
    mil = rem // n_targets
    tgt = rem % n_targets
    return POLITICAL_ACTIONS[pol], MILITARY_ACTIONS[mil], tgt


def run_episode(model: PPO, env: SovereignEnv, ep_idx: int) -> dict:
    obs, _ = env.reset(seed=ep_idx)
    state = env.state
    n_targets = env.map_spec.n
    names = [t.name for t in env.map_spec.territories]

    print(f"\n{'=' * 78}")
    print(f"EPISODE {ep_idx}  (seed={ep_idx})")
    print(f"{'=' * 78}")
    print(f"Start:  L={state.legitimacy:.3f}  E={state.economy:.3f}  θ={state.theta:+.3f}"
          f"  I_units={state.invader_units.sum():.1f}  D_units={state.defender_units.sum():.1f}")
    print(f"{'-' * 78}")
    print(f"{'t':>3} | {'POLITICAL':<16} {'MILITARY':<10} {'tgt':<4}"
          f"| {'L':>5} {'E':>5} {'θ':>6} | {'I':>5} {'D':>5} | {'r':>7}")
    print(f"{'-' * 78}")

    total_r = 0.0
    terminated = truncated = False
    info: dict = {}
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        pol_name, mil_name, tgt = _decode(int(action), n_targets)
        obs, r, terminated, truncated, info = env.step(int(action))
        total_r += r
        s = env.state
        print(f"{s.turn:>3} | {pol_name:<16} {mil_name:<10} {names[tgt]:<4}"
              f"| {s.legitimacy:5.3f} {s.economy:5.3f} {s.theta:+5.3f}"
              f" | {s.invader_units.sum():5.1f} {s.defender_units.sum():5.1f}"
              f" | {r:+7.3f}")

    outcome = info.get("terminal_reason") or ("truncated" if truncated else "terminated")
    print(f"{'-' * 78}")
    print(f"Outcome: {outcome.upper()}  |  turns={env.state.turn}  |  return={total_r:+.2f}"
          f"  |  final L={env.state.legitimacy:.3f}  θ={env.state.theta:+.3f}")
    return {"outcome": outcome, "turns": env.state.turn, "return": total_r}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--regime", choices=REGIME_NAMES, default="full")
    p.add_argument("--map", dest="map_name", default="rulebook9")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-dir", default="results/ablations",
                   help="dir containing <regime>/final.zip")
    args = p.parse_args()

    ckpt = Path(args.ckpt_dir) / args.regime / "final.zip"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    cfg = SovereignConfig(map_name=args.map_name).with_flags(
        AblationFlags.regime(args.regime)
    )
    env = SovereignEnv(config=cfg)
    model = PPO.load(str(ckpt), env=env)

    print(f"Loaded: {ckpt}")
    print(f"Regime: {args.regime}  |  Map: {args.map_name}  |  Episodes: {args.episodes}")

    summary = []
    for i in range(args.episodes):
        summary.append(run_episode(model, env, ep_idx=args.seed + i))

    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    n = len(summary)
    settle = sum(1 for s in summary if s["outcome"] == "negotiated_settlement")
    timeout = sum(1 for s in summary if s["outcome"] == "truncated")
    mean_r = np.mean([s["return"] for s in summary])
    mean_t = np.mean([s["turns"] for s in summary])
    print(f"  episodes:  {n}")
    print(f"  settle:    {settle}/{n}")
    print(f"  timeout:   {timeout}/{n}")
    print(f"  mean turns:   {mean_t:.1f}")
    print(f"  mean return:  {mean_r:+.2f}")


if __name__ == "__main__":
    main()
