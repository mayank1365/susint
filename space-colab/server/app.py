"""FastAPI server exposing the Suspect X environment.

Endpoints (one env instance per session_id):
    POST /reset?session_id=...       body: {crime_id?, seed?, split?, max_turns?}
    POST /step?session_id=...        body: {action: {...}}
    GET  /state?session_id=...
    GET  /info                       static metadata for the environment

Deployable as-is to Hugging Face Spaces (Docker SDK).
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .secret_factory import SecretFactory
from .suspect_x_environment import SuspectXEnvironment


app = FastAPI(title="Suspect X Environment", version="0.1.0")

# Single shared SecretFactory (loads ~200 JSONs once at startup).
_factory = SecretFactory()
# Per-session env instances. In a real multi-tenant deploy this would be a
# bounded LRU; for the hackathon a plain dict is fine.
_sessions: dict[str, SuspectXEnvironment] = {}


def _get_session(session_id: str) -> SuspectXEnvironment:
    if session_id not in _sessions:
        _sessions[session_id] = SuspectXEnvironment(factory=_factory)
    return _sessions[session_id]


# -------- request/response schemas (kept local so we don't import models.py
# at the server-package level — the server is self-contained for HF Spaces). #

class ResetBody(BaseModel):
    crime_id: Optional[str] = None
    seed: Optional[int] = None
    split: str = "train"
    max_turns: int = 20


class ActionBody(BaseModel):
    action_type: str
    content: str = ""
    accusation_json: dict = {}


class StepBody(BaseModel):
    action: ActionBody


# ---------------------------------------------------------------- endpoints
@app.get("/info")
def info():
    return {
        "name": "suspect_x_env",
        "n_crimes_total": len(_factory),
        "n_train": len(_factory.split("train")),
        "n_heldout": len(_factory.split("heldout")),
        "active_sessions": len(_sessions),
        "max_turns_default": 20,
    }


@app.post("/reset")
def reset(body: ResetBody, session_id: str = Query(default="default")):
    env = _get_session(session_id)
    return env.reset(
        crime_id=body.crime_id,
        seed=body.seed,
        split=body.split,
        max_turns=body.max_turns,
    )


@app.post("/step")
def step(body: StepBody, session_id: str = Query(default="default")):
    env = _get_session(session_id)
    try:
        return env.step(body.action.model_dump())
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state(session_id: str = Query(default="default")):
    env = _get_session(session_id)
    return env.state


@app.delete("/session")
def delete_session(session_id: str = Query(default="default")):
    _sessions.pop(session_id, None)
    return {"deleted": session_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
