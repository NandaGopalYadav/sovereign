"""Train a DQN agent on SovereignEnv.

DQN is included as a contrast for the experimental protocol: same env, different
optimisation regime. The flat ``Discrete(180)`` action space lets DQN run unmodified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from sovereign.env.config import REGIME_NAMES, AblationFlags, SovereignConfig
from sovereign.env.sovereign_env import SovereignEnv
from sovereign.training.metrics import EpisodeMetricsCallback


def make_env(regime: str, map_name: str, seed: int):
    cfg = SovereignConfig(map_name=map_name).with_flags(AblationFlags.regime(regime))

    def _ctor():
        env = SovereignEnv(config=cfg)
        env.reset(seed=seed)
        return env

    return _ctor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--regime", choices=REGIME_NAMES, default="full")
    p.add_argument("--map", dest="map_name", default="default9")
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", default="results/dqn")
    args = p.parse_args()

    out_dir = Path(args.out) / f"{args.regime}_{args.map_name}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    venv = DummyVecEnv([make_env(args.regime, args.map_name, args.seed)])

    model = DQN(
        "MlpPolicy",
        venv,
        learning_rate=args.lr,
        buffer_size=50_000,
        learning_starts=2_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(out_dir / "tb"),
    )

    callbacks = [
        EpisodeMetricsCallback(run_dir=out_dir),
        CheckpointCallback(
            save_freq=max(args.steps // 10, 1),
            save_path=str(out_dir / "ckpts"),
            name_prefix="dqn",
        ),
    ]

    model.learn(total_timesteps=args.steps, callback=callbacks)
    model.save(out_dir / "final.zip")
    print(f"Saved final model to {out_dir / 'final.zip'}")


if __name__ == "__main__":
    main()
