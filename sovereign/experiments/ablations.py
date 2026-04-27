"""Run the five ablation regimes from Section 10 and emit a summary table.

For each regime we train a PPO policy for a fixed number of steps, then evaluate it
across N greedy episodes to measure (a) the rate of negotiated settlements vs.
total-conquest endings and (b) the distribution of returns. The headline result of
the project is which regimes induce settlement-dominant play.

Use ``--quick`` for a short smoke run on the local machine; the default values are
sized for a multi-hour run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from sovereign.env.config import REGIME_NAMES, AblationFlags, SovereignConfig
from sovereign.env.sovereign_env import SovereignEnv
from sovereign.training.metrics import EpisodeMetricsCallback


def _make_env_factory(regime: str, seed: int):
    cfg = SovereignConfig().with_flags(AblationFlags.regime(regime))

    def _ctor():
        env = SovereignEnv(config=cfg)
        env.reset(seed=seed)
        return env

    return _ctor


def _evaluate(model: PPO, regime: str, n_episodes: int, seed_base: int) -> dict[str, float]:
    """Run greedy rollouts and aggregate behavioural statistics."""
    cfg = SovereignConfig().with_flags(AblationFlags.regime(regime))
    env = SovereignEnv(config=cfg)
    returns: list[float] = []
    settlement_count = 0
    conquest_count = 0
    legitimacy_collapse_count = 0
    destroyed_count = 0
    timeout_count = 0
    mean_thetas: list[float] = []
    mean_legitimacies: list[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        done = False
        ep_return = 0.0
        thetas: list[float] = []
        legs: list[float] = []
        last_info: dict = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(int(action))
            ep_return += reward
            thetas.append(info["theta"])
            legs.append(info["legitimacy"])
            last_info = info
            done = term or trunc
        returns.append(ep_return)
        mean_thetas.append(float(np.mean(thetas)))
        mean_legitimacies.append(float(np.mean(legs)))
        reason = last_info.get("terminal_reason")
        if reason == "negotiated_settlement":
            settlement_count += 1
        elif reason == "total_conquest":
            conquest_count += 1
        elif reason == "legitimacy_collapse":
            legitimacy_collapse_count += 1
        elif reason == "invader_destroyed":
            destroyed_count += 1
        else:
            timeout_count += 1

    return {
        "regime": regime,
        "n_episodes": n_episodes,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "settlement_rate": settlement_count / n_episodes,
        "conquest_rate": conquest_count / n_episodes,
        "legitimacy_collapse_rate": legitimacy_collapse_count / n_episodes,
        "destroyed_rate": destroyed_count / n_episodes,
        "timeout_rate": timeout_count / n_episodes,
        "mean_theta": float(np.mean(mean_thetas)),
        "mean_legitimacy": float(np.mean(mean_legitimacies)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--eval-episodes", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/ablations")
    p.add_argument("--regimes", nargs="+", default=list(REGIME_NAMES))
    p.add_argument("--quick", action="store_true",
                   help="2k training steps, 8 eval episodes per regime — sanity run.")
    args = p.parse_args()

    if args.quick:
        args.steps = 2048
        args.eval_episodes = 8

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    for regime in args.regimes:
        run_dir = out / regime
        run_dir.mkdir(parents=True, exist_ok=True)
        venv = DummyVecEnv([_make_env_factory(regime, args.seed)])
        model = PPO(
            "MlpPolicy",
            venv,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            verbose=0,
            seed=args.seed,
            tensorboard_log=str(run_dir / "tb"),
        )
        callbacks = [EpisodeMetricsCallback(run_dir=run_dir, trace_every=50)]
        model.learn(total_timesteps=args.steps, callback=callbacks)
        model.save(run_dir / "final.zip")

        stats = _evaluate(model, regime, args.eval_episodes, args.seed + 10_000)
        with (run_dir / "eval.json").open("w") as fh:
            json.dump(stats, fh, indent=2)
        summary.append(stats)
        print(
            f"[{regime:20s}] return={stats['return_mean']:+7.2f} ± {stats['return_std']:5.2f}  "
            f"settle={stats['settlement_rate']:.2f}  conquest={stats['conquest_rate']:.2f}  "
            f"L_mean={stats['mean_legitimacy']:.2f}  θ_mean={stats['mean_theta']:+.2f}"
        )

    summary_path = out / "summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
