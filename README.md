# sovereign

A reinforcement-learning environment for studying whether a militarily superior agent
learns that invasion is a strategically dominated strategy when the model includes
the costs that real coercive action incurs: legitimacy decay under occupation,
insurgency hazard, sanctions and shifting third-party posture.

The environment is small enough to fit in a notebook and is parameterised by a single
configuration object so the structural mechanisms — legitimacy, occupation cost,
neutral pivot — can be ablated independently.

This is a pilot study. It does not claim to model any real conflict. It models a
*structure*: three nations on a graph, two payoffs (territory captured vs. erosion
of standing), and the question of whether a Pareto-improving "settle" outcome is
reachable when the agent is free to choose force.

## Layout

```
sovereign/
  env/                 the gym.Env, map registry, mechanics, config
  agents/              rule-based defender policy
  training/            PPO and DQN entry points + JSON metrics callback
  experiments/         ablation runner, parameter sweep, plotting
  ui/                  FastAPI backend + React/Vite/Tailwind frontend
  tests/               pytest suite (unit, integration, gym conformance)
```

## Install

```bash
pip install -e ".[dev,train,ui]"
```

The three extras are independent. ``dev`` installs the test toolchain only;
``train`` adds Stable-Baselines3 + Torch + TensorBoard; ``ui`` adds FastAPI.

For the frontend:

```bash
cd sovereign/ui/frontend && npm install
```

## The four commands

```bash
# 1. install
pip install -e ".[dev,train,ui]"

# 2. tests
pytest

# 3. train (full regime, default9 map, 1M steps)
python -m sovereign.training.train_ppo --regime full --steps 1000000

# 4. ui (run backend on :8000, then `npm run dev` in ui/frontend on :5173)
python -m sovereign.ui.backend.serve
```

## Architecture

```text
                        +-----------------------+
                        |   sovereign.env       |
                        |  (gym.Env, formulas)  |
                        +-----------+-----------+
                                    |
                +-------------------+-------------------+
                |                                       |
        +-------v--------+                     +--------v--------+
        |  agents.       |                     |  training.      |
        |  defender      |                     |  PPO / DQN      |
        |  (rule-based)  |                     |  metrics logger |
        +----------------+                     +--------+--------+
                                                        |
                                                        |  results/*/episodes.jsonl
                                                        |  results/*/trace_*.jsonl
                                                        v
                                               +----------------+
                                               |  ui.backend    |
                                               |  (FastAPI)     |
                                               +-------+--------+
                                                       |  /api/*
                                                       v
                                               +----------------+
                                               |  ui.frontend   |
                                               |  (Vite/React)  |
                                               +----------------+
```

## Running ablations

```bash
python -m sovereign.experiments.ablations              # full protocol
python -m sovereign.experiments.ablations --quick      # smoke test
python -m sovereign.experiments.parameter_sweep --quick
python -m sovereign.experiments.plots                  # write figures
```

The runner trains a fresh PPO model under each of the five regimes
(``full``, ``no_legitimacy``, ``no_occupation_cost``, ``no_neutral``, ``baseline``),
evaluates it greedily, and writes a row to ``results/ablations/summary.json`` with
the settlement / total-conquest rate, mean return, and behavioural averages.
The UI's *Compare* tab reads this file directly.

## Configuration

Every coefficient lives in ``sovereign.env.config``. To override from a YAML:

```yaml
# experiment_a.yml
hazard:
  lam: 0.10
reward:
  occupation: 0.40
flags:
  legitimacy_enabled: true
```

```python
from sovereign.env.config import SovereignConfig
cfg = SovereignConfig.from_yaml("experiment_a.yml")
```

## Maps

Two topologies ship registered: ``default9`` (3 invader-home + 3 defender-home + 3
contested) and ``frontier12`` (a longer-front variant for sweeps that want to test
graph structure independently of mechanism weights). Register your own:

```python
from sovereign.env.map import MapSpec, Territory, register_map, INVADER, DEFENDER, CONTESTED

@register_map("mymap")
def _mymap() -> MapSpec:
    return MapSpec("mymap", territories=(...), edges=(...))
```

## Inheritance from the Diplomacy work

The mechanics layer is inspired by the Cicero / Diplomacy line of work at FAIR,
specifically the observation that durable cooperative outcomes require *agents that
internalise the cost of unilateral action on third-party trust*. Sovereign is a
much smaller, more transparent setup — one agent, two scripted opponents, six
mechanism levers — designed to ask: which of those mechanisms, individually, is
necessary for the dominated-strategy result to emerge under PPO? See ``RESEARCH.md``
for the full argument.

## License

Research code, all rights reserved by the author for now.
