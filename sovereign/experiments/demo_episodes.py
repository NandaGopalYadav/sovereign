"""Replay a captured demo trace line-by-line so the live demo always shows
the same predefined episodes. Decouples the talk from model nondeterminism,
checkpoint drift, or shell-environment changes.

Usage:
    uv run python -m sovereign.experiments.demo_episodes <regime>
        where <regime> in {full, baseline, no_legitimacy}

Optional: --delay <seconds> for per-line typewriter pacing (default 0.02).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

TRACE_DIR = Path(__file__).resolve().parents[2] / "demo_traces"
REGIMES = ("full", "baseline", "no_legitimacy")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("regime", choices=REGIMES)
    p.add_argument("--delay", type=float, default=0.02,
                   help="per-line delay in seconds (default 0.02)")
    p.add_argument("--instant", action="store_true",
                   help="print everything immediately, no delay")
    args = p.parse_args()

    path = TRACE_DIR / f"{args.regime}.txt"
    if not path.exists():
        raise SystemExit(f"trace not found: {path}")

    delay = 0.0 if args.instant else args.delay
    with path.open() as f:
        for line in f:
            sys.stdout.write(line)
            sys.stdout.flush()
            if delay:
                time.sleep(delay)


if __name__ == "__main__":
    main()
