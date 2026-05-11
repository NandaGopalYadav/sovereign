"""FastAPI backend serving the SOVEREIGN dashboard.

Endpoints:
    GET  /api/maps                 — list registered maps with topology
    GET  /api/runs                 — list completed runs under results/
    GET  /api/runs/{run_id}/episodes  — episode summary timeline for a run
    GET  /api/runs/{run_id}/trace/{ep}  — per-step trace for a single replay
    POST /api/live/reset           — start a fresh live env
    POST /api/live/step            — apply one action, return new state
    GET  /api/live/state           — current state of the live env
    GET  /api/compare              — compare summary metrics across all runs

Static files: the production frontend is served from ``ui/frontend/dist`` if it exists.
In dev, run Vite separately on :5173 and FastAPI on :8000 — CORS is open in dev.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sovereign.env.config import (
    MILITARY_ACTIONS,
    POLITICAL_ACTIONS,
    SovereignConfig,
)
from sovereign.env.map import list_maps
from sovereign.env.sovereign_env import SovereignEnv


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results"
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


app = FastAPI(title="Sovereign API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-process live env, used by the "Live" tab. Single-tenant; this is a research tool.
_live_env: SovereignEnv | None = None
_live_history: list[dict[str, Any]] = []


# --------------------------------------------------------------------------------------
# Pydantic schemas
# --------------------------------------------------------------------------------------


class StepRequest(BaseModel):
    pol: int
    mil: int
    target: int


class ResetRequest(BaseModel):
    map_name: str = "rulebook9"
    regime: str = "full"
    seed: int | None = None


# --------------------------------------------------------------------------------------
# Map metadata
# --------------------------------------------------------------------------------------


@app.get("/api/maps")
def get_maps() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    from sovereign.env.map import get_map

    for name in list_maps():
        spec = get_map(name)
        out.append(
            {
                "name": name,
                "n": spec.n,
                "territories": [
                    {
                        "id": i,
                        "name": t.name,
                        "home_of": t.home_of,
                        "resource_value": t.resource_value,
                        "strategic_value": t.strategic_value,
                    }
                    for i, t in enumerate(spec.territories)
                ],
                "edges": [list(e) for e in spec.edges],
            }
        )
    return out


@app.get("/api/actions")
def get_actions() -> dict[str, list[str]]:
    return {
        "political": list(POLITICAL_ACTIONS),
        "military": list(MILITARY_ACTIONS),
    }


# --------------------------------------------------------------------------------------
# Runs / episodes
# --------------------------------------------------------------------------------------


def _scan_runs(root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not root.exists():
        return runs
    for episodes_path in root.glob("**/episodes.jsonl"):
        run_dir = episodes_path.parent
        rel = run_dir.relative_to(root).as_posix()
        runs.append(
            {
                "id": rel,
                "path": str(run_dir),
                "size": episodes_path.stat().st_size,
            }
        )
    runs.sort(key=lambda r: r["id"])
    return runs


@app.get("/api/runs")
def get_runs() -> list[dict[str, Any]]:
    return _scan_runs(RESULTS_ROOT)


@app.get("/api/runs/{run_id:path}/episodes")
def get_run_episodes(run_id: str) -> list[dict[str, Any]]:
    path = RESULTS_ROOT / run_id / "episodes.jsonl"
    if not path.exists():
        raise HTTPException(404, f"No such run: {run_id}")
    out: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            out.append(json.loads(line))
    return out


@app.get("/api/runs/{run_id:path}/trace/{ep}")
def get_run_trace(run_id: str, ep: int) -> list[dict[str, Any]]:
    path = RESULTS_ROOT / run_id / f"trace_{ep:06d}.jsonl"
    if not path.exists():
        raise HTTPException(404, f"No trace: {run_id} ep={ep}")
    out: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------------------
# Live env
# --------------------------------------------------------------------------------------


def _state_payload(env: SovereignEnv) -> dict[str, Any]:
    if env.state is None:
        return {}
    return {
        "turn": env.state.turn,
        "controller": env.state.controller.tolist(),
        "invader_units": env.state.invader_units.tolist(),
        "defender_units": env.state.defender_units.tolist(),
        "neutral_units": env.state.neutral_units.tolist(),
        "legitimacy": float(env.state.legitimacy),
        "economy": float(env.state.economy),
        "theta": float(env.state.theta),
        "t_occ": int(env.state.t_occ),
        "sanctions_active": env.hysteresis.sanctions_active,
        "supply_routes_open": env.hysteresis.supply_routes_open,
        "formal_alliance": env.hysteresis.formal_alliance,
        "thresholds": {
            "sanctions_on": env.cfg.thresholds.sanctions_on,
            "sanctions_off": env.cfg.thresholds.sanctions_off,
            "neutral_joins_defender": env.cfg.thresholds.neutral_joins_defender,
            "supply_routes_open": env.cfg.thresholds.supply_routes_open,
            "formal_alliance": env.cfg.thresholds.formal_alliance,
        },
        "map_name": env.cfg.map_name,
    }


@app.post("/api/live/reset")
def live_reset(req: ResetRequest) -> dict[str, Any]:
    global _live_env, _live_history
    from sovereign.env.config import AblationFlags

    cfg = SovereignConfig(map_name=req.map_name).with_flags(
        AblationFlags.regime(req.regime)
    )
    _live_env = SovereignEnv(config=cfg)
    _live_env.reset(seed=req.seed if req.seed is not None else 0)
    _live_history = []
    return _state_payload(_live_env)


@app.post("/api/live/step")
def live_step(req: StepRequest) -> dict[str, Any]:
    global _live_env, _live_history
    if _live_env is None:
        raise HTTPException(409, "Call /api/live/reset first")
    action = _live_env.encode_action(req.pol, req.mil, req.target)
    _, reward, term, trunc, info = _live_env.step(action)
    payload = _state_payload(_live_env)
    payload["reward"] = float(reward)
    payload["reward_components"] = info.get("reward_components", {})
    payload["terminated"] = bool(term)
    payload["truncated"] = bool(trunc)
    payload["terminal_reason"] = info.get("terminal_reason")
    payload["last_action"] = {"pol": req.pol, "mil": req.mil, "target": req.target}
    _live_history.append(payload)
    return payload


@app.get("/api/live/state")
def live_state() -> dict[str, Any]:
    if _live_env is None:
        return {}
    return _state_payload(_live_env)


@app.get("/api/live/history")
def live_history() -> list[dict[str, Any]]:
    return _live_history


# --------------------------------------------------------------------------------------
# Compare ablations
# --------------------------------------------------------------------------------------


@app.get("/api/compare")
def compare() -> list[dict[str, Any]]:
    """Aggregate eval.json files from results/ablations/* if present."""
    abl = RESULTS_ROOT / "ablations"
    out: list[dict[str, Any]] = []
    if not abl.exists():
        return out
    for ev in sorted(abl.glob("*/eval.json")):
        with ev.open() as fh:
            out.append(json.load(fh))
    return out


# --------------------------------------------------------------------------------------
# Static frontend (production build)
# --------------------------------------------------------------------------------------


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")


def main() -> None:
    import uvicorn

    uvicorn.run("sovereign.ui.backend.serve:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
