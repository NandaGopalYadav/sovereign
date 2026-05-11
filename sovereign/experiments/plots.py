"""Report-quality plots over the JSON outputs of ``ablations.py`` and
``parameter_sweep.py``.

Three figures are produced:

* ``ablations_terminations.png`` — the headline. Stacked bar of every termination
  type by regime. This is the figure that shows the qualitative shift between
  full (settlement-dominant) and baseline (timeout-dominant), with the
  intermediate ablations slotting between them.
* ``ablations_engagement.png`` — diagnostic panel showing *how* each regime's
  trajectories engage the cost mechanisms (insurgency, occupation duration, θ
  drift, L decay, max territory share). This is what supports the "different
  shape, not just different label" claim.
* ``sweep.png`` — settlement rate across the parameter sweep, with cells
  grouped by the parameter being swept and the default cell highlighted.

The plotting is intentionally publication-leaning: white background, large
labels, threshold annotations where they matter. Pure-matplotlib so anyone can
re-render without an extra dependency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------------------
# Style — kept centralised so all figures share a coherent look.
# --------------------------------------------------------------------------------------

REGIME_ORDER = ("baseline", "no_neutral", "no_legitimacy", "no_occupation_cost", "full")
REGIME_LABELS = {
    "full": "full",
    "no_legitimacy": "no_legitimacy",
    "no_occupation_cost": "no_occ_cost",
    "no_neutral": "no_neutral",
    "baseline": "baseline",
}

TERMINATION_COLORS = {
    "negotiated_settlement": "#2b7a3a",   # green — the desired outcome
    "total_conquest": "#7a4dab",           # purple
    "timeout": "#9aa1ad",                  # neutral grey
    "invader_destroyed": "#a83e3e",        # red
    "legitimacy_collapse": "#c97a3e",      # orange
}
TERMINATION_LABELS = {
    "negotiated_settlement": "settlement",
    "total_conquest": "total conquest",
    "timeout": "timeout",
    "invader_destroyed": "destroyed",
    "legitimacy_collapse": "L-collapse",
}

PLT_RC = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
}


def _apply_style() -> None:
    plt.rcParams.update(PLT_RC)


# --------------------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------------------


def _load_summary(path: str | Path) -> dict[str, dict[str, Any]]:
    with open(path) as fh:
        rows = json.load(fh)
    return {r["regime"]: r for r in rows}


def _load_episodes(run_dir: Path) -> list[dict[str, Any]]:
    """Read a regime's episodes.jsonl. Empty list if missing."""
    p = run_dir / "episodes.jsonl"
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    with p.open() as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _ordered(summary: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Return regime rows in the canonical display order, dropping unknowns at the end."""
    seen: set[str] = set()
    out: list[tuple[str, dict[str, Any]]] = []
    for r in REGIME_ORDER:
        if r in summary:
            out.append((r, summary[r]))
            seen.add(r)
    for r, row in summary.items():
        if r not in seen:
            out.append((r, row))
    return out


# --------------------------------------------------------------------------------------
# Figure 1 — termination outcomes (the headline)
# --------------------------------------------------------------------------------------


def plot_termination_summary(summary_path: str | Path, out_path: str | Path) -> None:
    """Stacked-bar figure: every termination type, by regime.

    Top panel: stacked bars of termination rates (the *qualitative* result).
    Bottom panel: mean episode return ± std with the dominant terminal annotated
    so the reader can read the bar height correctly (timeout-with-income looks
    similar to settlement on return alone).
    """
    _apply_style()
    summary = _load_summary(summary_path)
    rows = _ordered(summary)

    fig, axes = plt.subplots(
        2, 1, figsize=(8.4, 6.8), constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 0.7]},
    )

    # --- Top: stacked terminations -------------------------------------------------
    ax = axes[0]
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    keys = (
        "settlement_rate", "conquest_rate", "timeout_rate",
        "destroyed_rate", "legitimacy_collapse_rate",
    )
    pretty = {
        "settlement_rate": "negotiated_settlement",
        "conquest_rate": "total_conquest",
        "timeout_rate": "timeout",
        "destroyed_rate": "invader_destroyed",
        "legitimacy_collapse_rate": "legitimacy_collapse",
    }
    for key in keys:
        terminal = pretty[key]
        vals = np.array([row[key] for _, row in rows])
        ax.bar(
            x, vals, bottom=bottom,
            color=TERMINATION_COLORS[terminal],
            edgecolor="white", linewidth=0.6,
            label=TERMINATION_LABELS[terminal],
        )
        # Annotate any segment ≥ 5% of the column with its percentage.
        for xi, v, b in zip(x, vals, bottom):
            if v >= 0.05:
                ax.text(
                    xi, b + v / 2.0, f"{int(round(v * 100))}%",
                    ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold",
                )
        bottom += vals

    ax.set_xticks(x, [REGIME_LABELS.get(r, r) for r, _ in rows])
    ax.set_ylim(0, 1.001)
    ax.set_ylabel("share of episodes")
    ax.set_title("Termination outcomes by regime  (n=64 greedy eval each)")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=5, columnspacing=1.2, handlelength=1.4,
    )

    # --- Bottom: mean return -------------------------------------------------------
    ax = axes[1]
    means = np.array([row["return_mean"] for _, row in rows])
    stds = np.array([row["return_std"] for _, row in rows])

    # Color each bar by its dominant terminal so the "+63 baseline" bar visibly
    # differs from the "+55 full" bar even though they are similar in magnitude.
    bar_colors: list[str] = []
    for _, row in rows:
        rates = {
            "negotiated_settlement": row["settlement_rate"],
            "total_conquest": row["conquest_rate"],
            "timeout": row["timeout_rate"],
            "invader_destroyed": row["destroyed_rate"],
            "legitimacy_collapse": row["legitimacy_collapse_rate"],
        }
        dom = max(rates, key=rates.get)
        bar_colors.append(TERMINATION_COLORS[dom])

    ax.bar(x, means, yerr=stds, color=bar_colors, edgecolor="white",
           linewidth=0.6, capsize=4, error_kw={"elinewidth": 1, "alpha": 0.7})
    for xi, m, s in zip(x, means, stds):
        ax.text(
            xi, m + max(stds) * 0.1 + 0.5, f"{m:+.1f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_xticks(x, [REGIME_LABELS.get(r, r) for r, _ in rows])
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("mean episode return")
    ax.set_title("Mean return ± std, bar colour = dominant terminal")

    fig.suptitle(
        "SOVEREIGN ablation — qualitative outcome by regime",
        fontsize=12.5, fontweight="bold",
    )
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


# --------------------------------------------------------------------------------------
# Figure 2 — mechanism engagement (diagnostic)
# --------------------------------------------------------------------------------------


def plot_mechanism_engagement(
    ablations_root: str | Path,
    out_path: str | Path,
    last_fraction: float = 0.10,
) -> None:
    """Diagnostic figure: do regimes produce *different trajectories* before
    settlement, or do they only differ in terminal label?

    Reads the last `last_fraction` of each regime's episodes.jsonl and reports
    five mechanism-engagement signals:

    * mean episode length
    * mean |insurgency events per episode|
    * mean max θ reached across episode
    * mean min L reached across episode
    * mean max territory share reached

    These are the metrics the iter-3/iter-4 commentary relied on; bringing
    them onto a single figure makes the "trajectories differ" claim
    inspectable.
    """
    _apply_style()
    root = Path(ablations_root)
    summary = _load_summary(root / "summary.json")
    rows = _ordered(summary)

    metrics: dict[str, list[float | None]] = {
        "length": [],
        "insurgency_events": [],
        "max_theta": [],
        "min_legitimacy": [],
        "max_territory_share": [],
    }
    regime_labels: list[str] = []
    for regime, _ in rows:
        regime_labels.append(REGIME_LABELS.get(regime, regime))
        eps = _load_episodes(root / regime)
        if not eps:
            for k in metrics:
                metrics[k].append(None)
            continue
        k = max(int(len(eps) * last_fraction), 1)
        tail = eps[-k:]
        def _avg(name: str) -> float:
            vals = [e.get(name) for e in tail if e.get(name) is not None]
            return float(np.mean(vals)) if vals else 0.0
        metrics["length"].append(_avg("length"))
        metrics["insurgency_events"].append(_avg("insurgency_event_count"))
        metrics["max_theta"].append(_avg("max_theta"))
        metrics["min_legitimacy"].append(_avg("min_legitimacy"))
        metrics["max_territory_share"].append(_avg("max_territory_share"))

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.5), constrained_layout=True)
    x = np.arange(len(regime_labels))

    def _bar(ax, vals, *, title: str, ylabel: str, color: str,
             threshold: tuple[float, str] | None = None,
             ylim: tuple[float, float] | None = None,
             fmt: str = "{:.2f}") -> None:
        v = np.array([np.nan if x is None else x for x in vals])
        ax.bar(x, v, color=color, edgecolor="white", linewidth=0.6)
        for xi, val in zip(x, v):
            if np.isnan(val):
                continue
            ax.text(xi, val + (np.nanmax(v) - np.nanmin(v)) * 0.04 if not np.isnan(np.nanmax(v)) else val,
                    fmt.format(val), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, regime_labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if threshold is not None:
            t_val, t_label = threshold
            ax.axhline(t_val, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.text(
                len(x) - 0.5, t_val, f"  {t_label}",
                va="center", ha="left", fontsize=8, color="black", alpha=0.7,
            )
        if ylim is not None:
            ax.set_ylim(*ylim)

    _bar(
        axes[0, 0], metrics["length"],
        title="Mean episode length",
        ylabel="turns", color="#3a5a78",
        threshold=(15, "min_negotiation_turn"),
        fmt="{:.1f}",
    )
    _bar(
        axes[0, 1], metrics["insurgency_events"],
        title="Insurgency events per episode",
        ylabel="count", color="#c97a3e",
        fmt="{:.2f}",
    )
    _bar(
        axes[0, 2], metrics["max_territory_share"],
        title="Max territory share reached",
        ylabel="share", color="#7a4dab",
        threshold=(2 / 6, "pressure floor"),
        ylim=(0, 1.05),
    )
    _bar(
        axes[1, 0], metrics["max_theta"],
        title="Max θ reached (sanctions risk)",
        ylabel="θ", color="#a83e3e",
        threshold=(0.6, "sanctions_on"),
        ylim=(-1, 1),
    )
    _bar(
        axes[1, 1], metrics["min_legitimacy"],
        title="Min legitimacy reached",
        ylabel="L", color="#2b7a3a",
        threshold=(0.65, "viability floor"),
        ylim=(0, 1.05),
    )

    axes[1, 2].axis("off")

    fig.suptitle(
        "Mechanism engagement during converged training (last 10% of episodes)",
        fontsize=12.5, fontweight="bold",
    )
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


# --------------------------------------------------------------------------------------
# Figure 3 — parameter sweep
# --------------------------------------------------------------------------------------


def plot_sweep(sweep_path: str | Path, out_path: str | Path) -> None:
    """Parameter sweep, grouped by parameter, with the default value highlighted.

    The sweep was run at 50k steps (5 % of the ablation budget), so this is a
    *learnability* probe rather than an equilibrium-policy claim. A subtitle
    flags this so the figure is not over-read.
    """
    _apply_style()
    with open(sweep_path) as fh:
        rows = json.load(fh)

    # Group cells by the parameter name (everything before "=" in the label).
    groups: dict[str, list[tuple[str, float]]] = {}
    for r in rows:
        label = r["label"]
        param, _, val = label.partition("=")
        groups.setdefault(param, []).append((val, r["settlement_rate"]))

    # Default values to highlight per parameter.
    DEFAULTS = {
        "sanc_on": "0.60",
        "lam": "0.05",
        "gamma": "0.10",
        "w_occ": "0.25",
    }
    PARAM_TITLES = {
        "sanc_on": "sanctions threshold (θ_on)",
        "lam": "insurgency hazard rate (λ)",
        "gamma": "drift coefficient γ — strike shock\n(post-refactor; pre-refactor was L-coupling)",
        "w_occ": "occupation reward weight (w_O)",
    }

    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 4.0), constrained_layout=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (param, cells) in zip(axes, groups.items()):
        cells = sorted(cells, key=lambda c: float(c[0]))
        labels = [c[0] for c in cells]
        rates = [c[1] for c in cells]
        default = DEFAULTS.get(param)
        bar_colors = [
            "#2b7a3a" if r >= 0.99 else "#c97a3e" if r > 0 else "#a83e3e"
            for r in rates
        ]
        bars = ax.bar(labels, rates, color=bar_colors, edgecolor="white", linewidth=0.6)
        for label, bar in zip(labels, bars):
            if default is not None and label == default:
                bar.set_edgecolor("black")
                bar.set_linewidth(2.0)
        for label, r in zip(labels, rates):
            ax.text(
                label, r + 0.02, f"{int(round(r * 100))}%",
                ha="center", va="bottom", fontsize=8,
            )
        ax.set_ylim(0, 1.12)
        ax.set_xlabel(param)
        ax.set_title(PARAM_TITLES.get(param, param), fontsize=9)
    axes[0].set_ylabel("settlement rate (greedy eval)")

    fig.suptitle(
        "Parameter sweep — settlement rate by coefficient choice",
        fontsize=12.5, fontweight="bold",
    )
    fig.text(
        0.5, -0.02,
        "Each cell is a fresh PPO run at 50k steps (5% of ablation budget) — interpret as a "
        "learnability probe, not an equilibrium claim. "
        "Black-bordered bar = default. Green=converged to settlement, orange=partial, red=failed.",
        ha="center", fontsize=8, alpha=0.75,
    )
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


# --------------------------------------------------------------------------------------
# Backward-compat wrapper used by the UI / older callers.
# --------------------------------------------------------------------------------------


def plot_ablation_summary(summary_path: str | Path, out_path: str | Path) -> None:
    """Compatibility shim — delegates to the new termination-outcome figure."""
    plot_termination_summary(summary_path, out_path)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ablations", default="results/ablations/summary.json")
    p.add_argument("--ablations-root", default="results/ablations")
    p.add_argument("--sweep", default="results/sweep/sweep.json")
    p.add_argument("--out-dir", default="results/figures")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if Path(args.ablations).exists():
        plot_termination_summary(args.ablations, out_dir / "ablations_terminations.png")
        # Keep the old filename in place for any existing references.
        plot_termination_summary(args.ablations, out_dir / "ablations.png")
    if Path(args.ablations_root).exists():
        plot_mechanism_engagement(args.ablations_root, out_dir / "ablations_engagement.png")
    if Path(args.sweep).exists():
        plot_sweep(args.sweep, out_dir / "sweep.png")


if __name__ == "__main__":
    main()
