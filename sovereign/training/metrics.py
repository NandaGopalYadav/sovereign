"""JSON metrics logger.

Writes one JSON object per *completed* episode to ``<run_dir>/episodes.jsonl`` and
appends a per-step trace under ``<run_dir>/trace_<ep_idx>.jsonl`` for every Nth episode
so we can replay episodes in the UI without having to keep them all in memory.

Stable-Baselines3 has its own logger we use in parallel (TensorBoard); this is the
domain-specific record for analysis.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeMetricsCallback(BaseCallback):
    """SB3 callback that records per-episode summary stats and per-step traces.

    Parameters
    ----------
    run_dir:
        Directory under which to write ``episodes.jsonl`` and trace files.
    trace_every:
        Save the full per-step trace once every N episodes. Set to 1 to keep all,
        or to a large number to keep storage small.
    """

    def __init__(self, run_dir: str | Path, trace_every: int = 25) -> None:
        super().__init__()
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.run_dir / "episodes.jsonl"
        self.trace_every = trace_every

        self._ep_count = 0
        self._reward_buffers: dict[int, list[float]] = {}
        self._action_buffers: dict[int, list[int]] = {}
        self._theta_buffers: dict[int, list[float]] = {}
        self._legitimacy_buffers: dict[int, list[float]] = {}
        self._reward_decompositions: dict[int, list[dict[str, float]]] = {}
        self._start_times: dict[int, float] = {}

    def _on_training_start(self) -> None:  # type: ignore[override]
        # Writing the file fresh on each fit, not appending — avoids stale runs.
        if self.episodes_path.exists():
            self.episodes_path.unlink()

    def _on_step(self) -> bool:  # type: ignore[override]
        # `or []` collapses a length-1 numpy array because its scalar bool is False
        # when the value is 0; we must compare to None explicitly.
        infos = self.locals.get("infos")
        if infos is None:
            infos = []
        rewards = self.locals.get("rewards")
        if rewards is None:
            rewards = np.zeros(0, dtype=np.float32)
        dones = self.locals.get("dones")
        if dones is None:
            dones = np.zeros(0, dtype=bool)
        actions = self.locals.get("actions")

        for i, info in enumerate(infos):
            if i not in self._reward_buffers:
                self._reward_buffers[i] = []
                self._action_buffers[i] = []
                self._theta_buffers[i] = []
                self._legitimacy_buffers[i] = []
                self._reward_decompositions[i] = []
                self._start_times[i] = time.time()

            self._reward_buffers[i].append(float(rewards[i]))
            if actions is not None:
                self._action_buffers[i].append(int(np.asarray(actions)[i]))
            self._theta_buffers[i].append(float(info.get("theta", 0.0)))
            self._legitimacy_buffers[i].append(float(info.get("legitimacy", 0.0)))
            if "reward_components" in info:
                self._reward_decompositions[i].append(info["reward_components"])

            if dones[i]:
                self._flush_episode(i, info)

        return True

    def _flush_episode(self, env_idx: int, last_info: dict[str, Any]) -> None:
        rewards = self._reward_buffers[env_idx]
        thetas = self._theta_buffers[env_idx]
        legitimacies = self._legitimacy_buffers[env_idx]
        actions = self._action_buffers[env_idx]
        decompositions = self._reward_decompositions[env_idx]

        ep_idx = self._ep_count
        self._ep_count += 1

        summary = {
            "episode": ep_idx,
            "env_idx": env_idx,
            "length": len(rewards),
            "return": float(sum(rewards)),
            "mean_theta": float(np.mean(thetas)) if thetas else 0.0,
            "min_legitimacy": float(np.min(legitimacies)) if legitimacies else 0.0,
            "terminal_reason": last_info.get("terminal_reason"),
            "wall_seconds": time.time() - self._start_times[env_idx],
        }
        with self.episodes_path.open("a") as fh:
            fh.write(json.dumps(summary) + "\n")

        if ep_idx % self.trace_every == 0:
            trace_path = self.run_dir / f"trace_{ep_idx:06d}.jsonl"
            with trace_path.open("w") as fh:
                for t, (r, theta, leg, a, dec) in enumerate(
                    zip(rewards, thetas, legitimacies, actions, decompositions)
                ):
                    fh.write(
                        json.dumps(
                            {
                                "t": t,
                                "reward": r,
                                "theta": theta,
                                "legitimacy": leg,
                                "action": a,
                                "reward_components": dec,
                            }
                        )
                        + "\n"
                    )

        # Reset per-env buffers for the next episode in the vec env slot.
        self._reward_buffers[env_idx] = []
        self._action_buffers[env_idx] = []
        self._theta_buffers[env_idx] = []
        self._legitimacy_buffers[env_idx] = []
        self._reward_decompositions[env_idx] = []
        self._start_times[env_idx] = time.time()
