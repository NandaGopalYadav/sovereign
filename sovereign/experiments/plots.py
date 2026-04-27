"""Plotting utilities. Pure-functions over the JSON outputs of ``ablations.py``.

Generates a single multi-panel figure for the ablation summary and a separate
parameter-sweep bar chart. We deliberately keep the matplotlib styling minimal —
the UI dashboard is where the polished view lives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_ablation_summary(summary_path: str | Path, out_path: str | Path) -> None:
    with open(summary_path) as fh:
        rows = json.load(fh)
    regimes = [r["regime"] for r in rows]
    returns = [r["return_mean"] for r in rows]
    settle = [r["settlement_rate"] for r in rows]
    conquest = [r["conquest_rate"] for r in rows]
    legitimacy = [r["mean_legitimacy"] for r in rows]
    theta = [r["mean_theta"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    x = np.arange(len(regimes))
    axes[0, 0].bar(x, returns)
    axes[0, 0].set_xticks(x, regimes, rotation=20, ha="right")
    axes[0, 0].set_title("Mean episode return by regime")
    axes[0, 0].axhline(0, color="black", linewidth=0.8)

    axes[0, 1].bar(x - 0.2, settle, width=0.4, label="settlement")
    axes[0, 1].bar(x + 0.2, conquest, width=0.4, label="total conquest")
    axes[0, 1].set_xticks(x, regimes, rotation=20, ha="right")
    axes[0, 1].set_title("Termination outcomes")
    axes[0, 1].legend()

    axes[1, 0].bar(x, legitimacy)
    axes[1, 0].set_xticks(x, regimes, rotation=20, ha="right")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Mean legitimacy across episode")

    axes[1, 1].bar(x, theta)
    axes[1, 1].set_xticks(x, regimes, rotation=20, ha="right")
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_ylim(-1, 1)
    axes[1, 1].set_title("Mean θ across episode")

    fig.suptitle("Ablation regimes — behavioural summary")
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def plot_sweep(sweep_path: str | Path, out_path: str | Path) -> None:
    with open(sweep_path) as fh:
        rows = json.load(fh)
    labels = [r["label"] for r in rows]
    rates = [r["settlement_rate"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.barh(labels, rates)
    ax.set_xlim(0, 1)
    ax.set_xlabel("settlement rate (greedy eval)")
    ax.set_title("Parameter sweep — settlement rate")
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ablations", default="results/ablations/summary.json")
    p.add_argument("--sweep", default="results/sweep/sweep.json")
    p.add_argument("--out-dir", default="results/figures")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if Path(args.ablations).exists():
        plot_ablation_summary(args.ablations, out_dir / "ablations.png")
    if Path(args.sweep).exists():
        plot_sweep(args.sweep, out_dir / "sweep.png")


if __name__ == "__main__":
    main()
