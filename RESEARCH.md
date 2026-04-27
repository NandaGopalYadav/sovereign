# Research notes

This file documents *why* the environment is shaped the way it is. It is the
companion to `README.md`, which covers *how* to run it.

## The question

Given a militarily superior agent and a model that is generous about how easy it is
to win battles, do the *political* costs of holding territory — legitimacy decay,
shifts in third-party posture, insurgency — make pure conquest a dominated strategy
under standard policy-gradient training? Equivalently: does the agent learn to
prefer a negotiated settlement even when it could probably win by force?

This is a structural question, not a historical one. The mechanisms below are not
calibrated to any specific country or war. They are knobs whose presence or absence
can be turned on and off so we can ask which mechanism is doing the work.

## Why the design borrows from Diplomacy

The Cicero / Diplomacy paper from FAIR (Bakhtin et al., 2022, *Human-level play in
the game of Diplomacy by combining language models with strategic reasoning*) showed
that cooperative outcomes in adversarial multi-agent games require an agent that
*tracks how its current move affects future trust*. Sovereign isolates one slice
of that idea: there is no language model, only a single scalar (``θ``, the neutral
posture) tracking an external party's drift toward defection / alliance with the
invader's enemy. The reward and termination structure then asks whether the agent
can *plan against θ trajectories* under ablation.

## The five mechanisms, and why each one matters

### 1. Legitimacy `L ∈ [0,1]`

Initialised at 1.0; declines with occupation duration and with use of strike units.
Drops to 0 → terminal payoff −50.

Without `L`, the agent has no sunk-cost-style penalty for prolonged occupation.
Ablation **no_legitimacy** removes this; the prediction is that the agent will
prefer indefinite occupation since territorial gains have no decaying counterweight.

### 2. Occupation cost (per-step shaping `−w_O · t_occ / T_max`)

A small, monotonically rising drag on reward whenever the invader holds at least
one off-home territory. This separates *legitimacy* (which collapses catastrophically
at 0) from *occupation cost* (which creeps in linearly).

Ablation **no_occupation_cost** removes this term. The expected effect is subtler
than removing legitimacy: the agent still has to manage long-run collapse but has
no immediate cost signal pushing it toward shorter occupations.

### 3. Neutral posture `θ ∈ [-1, +1]`

A drift-diffusion process with coefficients on six behavioural signals:

```
μ(s, a) = α·aggression + β·occupation_fraction + γ·legitimacy_loss
       − δ·defender_morale − ε·invader_concession + ζ·economic_pressure
θ_{t+1} = clip(θ_t + μ + N(0, σ²), -1, +1)
```

Three latched threshold events sit on top of θ:

| event                       | trigger        | exit                                             |
|-----------------------------|----------------|--------------------------------------------------|
| sanctions                   | `θ > 0.60`     | `θ < 0.50` for 5 consecutive steps (hysteretic)  |
| neutral joins defender      | `θ > 0.85`     | sticky (does not reverse)                        |
| supply routes open (helpful)| `θ < -0.60`    | sticky                                           |
| formal alliance (helpful)   | `θ < -0.85`    | sticky                                           |

The hysteresis on the sanctions latch matters: without it, the model thrashes
across the boundary and the agent learns to oscillate θ rather than respect the
threshold. With it, sanctions are a commitment device, much like real-world
multilateral sanctions packages.

Ablation **no_neutral** removes both θ and the sanction latch. Predicted effect:
the agent loses the strongest channel by which "everyone else also has agency"
shows up in its return, and it should regress toward conquest-dominant play.

### 4. Insurgency hazard

```
p(insurgency | t_occ) = 1 - exp(-λ · t_occ),    λ = 0.05
```

A standard exponential hazard. Each step where the invader holds at least one
off-home territory rolls against the hazard; on a hit, one unit is removed from
a random occupied territory and a small reward penalty is applied.

This is the only mechanism that directly destroys invader force, and it scales
with how long territory has been held — a sharper proxy for "the longer you stay,
the harder it gets" than legitimacy decay.

### 5. Sanctions / economy

`E ∈ [0,1]`, depleting at a small constant rate while sanctions are active. `E`
feeds back into the drift (`ζ · (1 − E)`), creating a slow positive feedback loop:
sanctions raise θ, which keeps sanctions on, which depletes E, which raises θ
further. Without the hysteresis floor on `sanctions_off`, this loop becomes a
runaway. With the hysteresis, it is a slow squeeze.

## Why a flat `Discrete(5 · 4 · V)` action space

The spec asks for a *joint* `(a_pol, a_mil)` action with the political move
committed before the military move within a turn. Internally we honor this in
``_run_turn`` (substep 1 applies political, substep 4 applies military). Externally
the agent sees a single discrete action which is decoded on entry. This:

* makes DQN usable without a custom Q-decomposition,
* lets PPO see the full joint distribution rather than a factorised approximation,
* keeps the environment usable from any vanilla SB3 policy.

The flat space is `5 × 4 × |V|` and so is `180` for the default 9-territory map,
`240` for ``frontier12``.

## Why the reward weights have the values they have

```
w_T = 0.30   territory gain (the "win the war" signal)
w_R = 0.20   resource value of newly held territory
w_O = 0.25   per-step occupation drag
w_L = 0.15   legitimacy cost
w_S = 0.20   sanction-active flat per-step cost
w_I = 0.10   insurgency event cost
```

The terminal payoffs are an order of magnitude larger than the per-step components
on purpose:

```
−50  legitimacy collapse        # "your own population has rejected this"
−30  invader force destroyed    # "the war was lost militarily"
+40  negotiated settlement      # "you got most of what you wanted, peacefully"
+10  total conquest             # "you won — but it was expensive"
  0  timeout                    # "stalemate"
```

`+40` strictly dominates `+10` per-episode if the per-step shaping is roughly
neutral, which by design it is in the **full** regime once legitimacy is below
~0.7. The point is to make settlement a *better* outcome than total conquest, *but*
only when the cost mechanisms are switched on. Under **baseline** all costs are
gated off and the +10 conquest outcome is unobstructed; we expect that regime to
collapse to invade-and-hold play.

## Predictions before running the ablations

The thing to look at is the settlement rate vs. the conquest rate per regime:

| regime              | predicted dominant terminal reason |
|---------------------|------------------------------------|
| full                | `negotiated_settlement`            |
| no_legitimacy       | `total_conquest` or `timeout`      |
| no_occupation_cost  | `negotiated_settlement` (still)    |
| no_neutral          | `total_conquest`                   |
| baseline            | `total_conquest`                   |

The most interesting prediction is the *no_occupation_cost* row: legitimacy alone
should still be enough, because the catastrophic terminal at L=0 dominates the
absence of the per-step drag. If empirically that fails, the per-step shaping is
load-bearing, not the terminal — a useful finding.

## What this is *not*

* It is not a calibrated forecast of any actual conflict.
* It is not a multi-agent learning study (defender is rule-based, neutral is a
  scalar process). Multi-agent self-play would be a natural follow-up.
* It is not opinionated about whether war or peace is "good" — it is a structural
  question about which mechanisms must exist in the agent's model for it to learn
  the cheaper outcome.

## References

* Bakhtin, A. et al. (2022). *Human-level play in the game of Diplomacy by
  combining language models with strategic reasoning.* Science 378(6624), 1067–1074.
* Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.*
  Nature 518(7540), 529–533. [DQN baseline]
* Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
