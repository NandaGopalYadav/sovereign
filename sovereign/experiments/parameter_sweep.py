"""Grid-style parameter sweep for the most policy-sensitive coefficients.

Sweeps the sanction trigger threshold, insurgency hazard λ, the drift coefficients
γ (legitimacy coupling) and α (aggression coupling), and the reward weights for
occupation cost and legitimacy cost. For each cell we train briefly and record the
settlement rate as the headline metric.

This is intentionally a *grid* rather than Bayesian — interpretability beats
sample efficiency for a small mechanistic study.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from sovereign.env.config import (
    DriftCoefficients,
    RewardWeights,
    SovereignConfig,
    Thresholds,
)
from sovereign.env.sovereign_env import SovereignEnv


def _settlement_rate(model: PPO, cfg: SovereignConfig, n: int, seed: int) -> float:
    env = SovereignEnv(config=cfg)
    settlements = 0
    for ep in range(n):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        last_info: dict = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, last_info = env.step(int(action))
            done = term or trunc
        if last_info.get("terminal_reason") == "negotiated_settlement":
            settlements += 1
    return settlements / n


def _train_and_score(cfg: SovereignConfig, steps: int, seed: int, eval_n: int) -> float:
    venv = DummyVecEnv([lambda: SovereignEnv(config=cfg)])
    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        verbose=0,
        seed=seed,
    )
    model.learn(total_timesteps=steps)
    return _settlement_rate(model, cfg, eval_n, seed + 10_000)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/sweep")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.steps = 2048
        args.eval_episodes = 4

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base = SovereignConfig()

    # Define a small but informative grid. Each entry is a (label, builder) pair.
    grid: list[tuple[str, SovereignConfig]] = []

    for sanc in (0.50, 0.60, 0.75):
        cfg = replace(base, thresholds=replace(base.thresholds, sanctions_on=sanc))
        grid.append((f"sanc_on={sanc:.2f}", cfg))

    for lam in (0.02, 0.05, 0.10):
        cfg = replace(base, hazard=replace(base.hazard, lam=lam))
        grid.append((f"lam={lam:.2f}", cfg))

    for gamma in (0.05, 0.10, 0.20):
        cfg = replace(base, drift=replace(base.drift, gamma=gamma))
        grid.append((f"gamma={gamma:.2f}", cfg))

    for w_occ in (0.10, 0.25, 0.40):
        cfg = replace(base, reward=replace(base.reward, occupation=w_occ))
        grid.append((f"w_occ={w_occ:.2f}", cfg))

    results = []
    for label, cfg in grid:
        rate = _train_and_score(cfg, args.steps, args.seed, args.eval_episodes)
        results.append({"label": label, "settlement_rate": rate})
        print(f"  {label:18s}  settlement={rate:.3f}")

    out_path = out / "sweep.json"
    with out_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
