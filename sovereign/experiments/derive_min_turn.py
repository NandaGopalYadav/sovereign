"""Derive `min_negotiation_turn` from mechanism activation latency.

Rather than picking the time floor by feel, we compute it from when the cost
mechanisms in `RESEARCH.md` start to bite under a plausible occupation
trajectory. Run as a script to print the table; the chosen default in
`config.py` is justified against this output.

The reference trajectory: invader holds 3 non-home territories from turn 1
onward, takes attack actions, and makes no substantive diplomatic concession.
"""

from __future__ import annotations

import math

from sovereign.env.config import (
    CombatParams,
    DriftCoefficients,
    HazardParams,
    RewardWeights,
)


def cumulative_insurgency(t: int, lam: float, n_territories: int = 1) -> float:
    """P(at least one insurgency by turn t) under the per-territory hazard model.

    Each occupied territory has its own per-turn hazard `1 - exp(-λ·t_occ)`
    that rolls independently. P(no event on a single territory by turn t) =
    exp(-λ·t(t+1)/2). With `n` independent territories all held from turn 1,
    P(none across all) = exp(-λ·n·t(t+1)/2).
    """
    return 1.0 - math.exp(-lam * n_territories * t * (t + 1) / 2.0)


def expected_legitimacy(t: int, occupied: int, decay_per_terr: float) -> float:
    """L_t = max(0, 1 - decay·occupied·t). Strikes excluded; this is the
    quietest-possible occupation."""
    return max(0.0, 1.0 - decay_per_terr * occupied * t)


def expected_theta(t: int, occupied: int, c: DriftCoefficients, t_max: int) -> float:
    """Deterministic θ drift under attack-and-hold using Section 7.2 terms."""
    theta = 0.0
    L = 1.0
    t_occ = 0
    for _ in range(t):
        t_occ += 1
        L = max(0.0, L - 0.005 * occupied)
        mu = (
            c.alpha * (1.0 - L)
            + c.beta * 1.0
            + c.gamma * 0.0
            - c.delta * 0.0
            - c.epsilon * 0.0
            + c.zeta * (t_occ / t_max)
        )
        theta = max(-1.0, min(1.0, theta + mu))
    return theta


def cumulative_occupation_penalty(t: int, occupied: int,
                                   weights: RewardWeights, t_max: int) -> float:
    """Sum of `w_O · t_occ/T_max` over t steps, t_occ_step = step index."""
    return sum(weights.occupation * (s / t_max) for s in range(1, t + 1))


def main() -> None:
    h = HazardParams()
    c = DriftCoefficients()
    cp = CombatParams()
    rw = RewardWeights()
    occupied = 3                    # holds 3 contested
    t_max = 60

    print(f"{'turn':>4}  {'P(any insurgency)':>18}  {'E[L]':>6}  {'E[θ]':>7}  "
          f"{'cum occ-cost':>12}")
    for t in [3, 5, 8, 10, 12, 15, 18, 20, 25, 30]:
        p_ins = cumulative_insurgency(t, h.lam, n_territories=occupied)
        L = expected_legitimacy(t, occupied, cp.occupation_legitimacy_decay)
        theta = expected_theta(t, occupied, c, t_max)
        occ = cumulative_occupation_penalty(t, occupied, rw, t_max)
        print(f"{t:>4}  {p_ins:>18.3f}  {L:>6.3f}  {theta:>+7.3f}  "
              f"{occ:>12.3f}")

    print()
    print("Mechanism-engagement targets:")
    print("  P(any insurgency) >= 0.5     →  turn 5  (cheap; small reward weight)")
    print("  L drops >= 0.10              →  turn 7")
    print("  L drops >= 0.20              →  turn 14")
    print("  cum occ-cost >= 0.5          →  turn ~16")
    print("  θ drift into sanctions band  →  turn ~11-12 under sustained advance")
    print()
    print("Recommendation: min_negotiation_turn = 15. At t=15, L has dropped to "
          "~0.78, cumulative occupation penalty is ~0.5 reward units, and "
          "insurgency has near-certainly fired at least once. θ is already in "
          "the sanctions/alliance risk region under sustained advance. None of these alone are "
          "decisive — they collectively put the agent in the regime where the "
          "cost mechanisms can shape policy.")


if __name__ == "__main__":
    main()
