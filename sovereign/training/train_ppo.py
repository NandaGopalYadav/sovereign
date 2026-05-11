"""Train a PPO agent on SovereignEnv.

Why PPO and not, say, REINFORCE: the reward signal includes both dense per-step
shaping and large terminal payoffs (±50). PPO's clipped surrogate objective handles
the variance better than vanilla policy gradients while staying simple to reason about.

Why Stable-Baselines3 over CleanRL: SB3 ships well-tested PPO/DQN implementations
with TensorBoard logging, vectorised env support, and checkpointing all wired up.
This project is studying environment dynamics, not RL algorithm internals.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

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
    p.add_argument("--map", dest="map_name", default="rulebook9")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--out", default="results/ppo")
    p.add_argument("--subproc", action="store_true",
                   help="Use SubprocVecEnv (faster on >=4 envs, slower at low n).")
    args = p.parse_args()

    out_dir = Path(args.out) / f"{args.regime}_{args.map_name}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ctors = [make_env(args.regime, args.map_name, args.seed + i) for i in range(args.n_envs)]
    vec_cls = SubprocVecEnv if (args.subproc and args.n_envs > 1) else DummyVecEnv
    venv = vec_cls(ctors)

    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(out_dir / "tb"),
    )

    callbacks = [
        EpisodeMetricsCallback(run_dir=out_dir),
        CheckpointCallback(
            save_freq=max(args.steps // 10, 1),
            save_path=str(out_dir / "ckpts"),
            name_prefix="ppo",
        ),
    ]

    model.learn(total_timesteps=args.steps, callback=callbacks)
    model.save(out_dir / "final.zip")
    print(f"Saved final model to {out_dir / 'final.zip'}")


if __name__ == "__main__":
    main()
