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
        # Mechanism-engagement signals — these are the metrics the ablation
        # comparison needs in order to verify that `full` and `baseline`
        # produce *different trajectories* before terminal settlement, not
        # just different terminal labels.
        self._sanctions_buffers: dict[int, list[bool]] = {}
        self._insurgency_buffers: dict[int, list[bool]] = {}
        self._t_occ_buffers: dict[int, list[int]] = {}
        self._territory_buffers: dict[int, list[float]] = {}
        self._rejection_buffers: dict[int, list[str]] = {}
        self._connected_resource_buffers: dict[int, list[float]] = {}
        self._controlled_resource_buffers: dict[int, list[float]] = {}
        self._pressure_streak_buffers: dict[int, list[int]] = {}
        self._occupied_turn_buffers: dict[int, list[int]] = {}

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
                self._sanctions_buffers[i] = []
                self._insurgency_buffers[i] = []
                self._t_occ_buffers[i] = []
                self._territory_buffers[i] = []
                self._rejection_buffers[i] = []
                self._connected_resource_buffers[i] = []
                self._controlled_resource_buffers[i] = []
                self._pressure_streak_buffers[i] = []
                self._occupied_turn_buffers[i] = []
                self._start_times[i] = time.time()

            self._reward_buffers[i].append(float(rewards[i]))
            if actions is not None:
                self._action_buffers[i].append(int(np.asarray(actions)[i]))
            self._theta_buffers[i].append(float(info.get("theta", 0.0)))
            self._legitimacy_buffers[i].append(float(info.get("legitimacy", 0.0)))
            if "reward_components" in info:
                self._reward_decompositions[i].append(info["reward_components"])
            self._sanctions_buffers[i].append(bool(info.get("sanctions_active", False)))
            # Per-territory model: each turn can produce 0..N insurgency events.
            # Track the count so episode-level totals are accurate.
            self._insurgency_buffers[i].append(int(info.get("insurgency_events_this_step", int(bool(info.get("insurgency_event", False))))))
            self._t_occ_buffers[i].append(int(info.get("t_occ", 0)))
            self._territory_buffers[i].append(float(info.get("territory_share", 0.0)))
            self._connected_resource_buffers[i].append(float(info.get("connected_resource_yield", 0.0)))
            self._controlled_resource_buffers[i].append(float(info.get("controlled_resource_yield", 0.0)))
            self._pressure_streak_buffers[i].append(int(info.get("pressure_streak", 0)))
            self._occupied_turn_buffers[i].append(int(info.get("occupied_territory_turns", 0)))
            verdict = info.get("settlement_verdict")
            if verdict is not None and not verdict.get("accepted", False):
                self._rejection_buffers[i].append(str(verdict.get("reason", "rejected")))

            if dones[i]:
                self._flush_episode(i, info)

        return True

    def _flush_episode(self, env_idx: int, last_info: dict[str, Any]) -> None:
        rewards = self._reward_buffers[env_idx]
        thetas = self._theta_buffers[env_idx]
        legitimacies = self._legitimacy_buffers[env_idx]
        actions = self._action_buffers[env_idx]
        decompositions = self._reward_decompositions[env_idx]
        sanctions = self._sanctions_buffers[env_idx]
        insurgencies = self._insurgency_buffers[env_idx]
        t_occs = self._t_occ_buffers[env_idx]
        territories = self._territory_buffers[env_idx]
        rejections = self._rejection_buffers[env_idx]
        connected_resources = self._connected_resource_buffers[env_idx]
        controlled_resources = self._controlled_resource_buffers[env_idx]
        pressure_streaks = self._pressure_streak_buffers[env_idx]
        occupied_turns = self._occupied_turn_buffers[env_idx]

        ep_idx = self._ep_count
        self._ep_count += 1

        # Cumulative cost components, summed over the episode. The keys in
        # `reward_components` are signed (costs are already negative), so we
        # take absolute value to report cumulative *exposure* rather than net.
        def _cum(name: str) -> float:
            return float(sum(abs(d.get(name, 0.0)) for d in decompositions))

        from collections import Counter
        rejection_counts = dict(Counter(rejections)) if rejections else {}

        n = len(rewards)
        summary = {
            "episode": ep_idx,
            "env_idx": env_idx,
            "length": n,
            "return": float(sum(rewards)),
            "terminal_reason": last_info.get("terminal_reason"),
            "wall_seconds": time.time() - self._start_times[env_idx],
            # θ statistics
            "mean_theta": float(np.mean(thetas)) if thetas else 0.0,
            "max_theta": float(np.max(thetas)) if thetas else 0.0,
            "min_theta": float(np.min(thetas)) if thetas else 0.0,
            # Legitimacy statistics
            "min_legitimacy": float(np.min(legitimacies)) if legitimacies else 0.0,
            "mean_legitimacy": float(np.mean(legitimacies)) if legitimacies else 0.0,
            # Mechanism-engagement signals
            "sanctions_step_fraction": float(np.mean(sanctions)) if sanctions else 0.0,
            "insurgency_event_count": int(sum(insurgencies)),     # total events (per-territory rolls)
            "insurgency_event_turns": int(sum(1 for x in insurgencies if x > 0)),  # turns w/ ≥1 event
            "max_t_occ": int(max(t_occs)) if t_occs else 0,
            "mean_t_occ": float(np.mean(t_occs)) if t_occs else 0.0,
            "max_territory_share": float(np.max(territories)) if territories else 0.0,
            "final_territory_share": float(territories[-1]) if territories else 0.0,
            "mean_connected_resource_yield": float(np.mean(connected_resources)) if connected_resources else 0.0,
            "mean_controlled_resource_yield": float(np.mean(controlled_resources)) if controlled_resources else 0.0,
            "max_pressure_streak": int(max(pressure_streaks)) if pressure_streaks else 0,
            "final_occupied_territory_turns": int(occupied_turns[-1]) if occupied_turns else 0,
            # Cumulative cost exposure (independent of the live ablation flags —
            # whatever the env actually paid out is recorded).
            "cum_occupation_cost": _cum("occupation_cost"),
            "cum_legitimacy_cost": _cum("legitimacy_cost"),
            "cum_sanction_cost": _cum("sanction_cost"),
            "cum_insurgency_cost": _cum("insurgency_cost"),
            # Settlement attempt failures, by reason. Empty dict if the agent
            # never tried to negotiate or always succeeded.
            "settlement_rejections": rejection_counts,
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
        self._sanctions_buffers[env_idx] = []
        self._insurgency_buffers[env_idx] = []
        self._t_occ_buffers[env_idx] = []
        self._territory_buffers[env_idx] = []
        self._rejection_buffers[env_idx] = []
        self._connected_resource_buffers[env_idx] = []
        self._controlled_resource_buffers[env_idx] = []
        self._pressure_streak_buffers[env_idx] = []
        self._occupied_turn_buffers[env_idx] = []
        self._start_times[env_idx] = time.time()
