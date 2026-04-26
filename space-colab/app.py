"""FastAPI app for the Train-on-Boot Suspect X Space.

On startup:
  - Loads the SecretFactory (always succeeds — pure Python).
  - Spawns a background thread that runs train.train() if `out/done` is
    absent. The thread writes progress to `out/status.json` and the eval
    log to `out/eval_log.jsonl`. After training, `out/done` is touched
    and the reward plot is at `out/reward_curve.png`.

The env endpoints work whether or not training has started/finished.
"""
from __future__ import annotations

import json
import os
import threading
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from server.secret_factory import SecretFactory
from server.suspect_x_environment import SuspectXEnvironment


OUT_DIR = Path(os.environ.get("SUSPECT_X_OUT", "/app/out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATUS_PATH = OUT_DIR / "status.json"
EVAL_LOG_PATH = OUT_DIR / "eval_log.jsonl"
DONE_MARKER = OUT_DIR / "done"
PLOT_PATH = OUT_DIR / "reward_curve.png"


app = FastAPI(title="Suspect X — Train + Serve", version="0.2.0")

_factory = SecretFactory()
_sessions: dict[str, SuspectXEnvironment] = {}
_training_thread: Optional[threading.Thread] = None


def _write_status(**kwargs):
    base = {}
    if STATUS_PATH.exists():
        try:
            base = json.loads(STATUS_PATH.read_text())
        except Exception:
            base = {}
    base.update(kwargs)
    STATUS_PATH.write_text(json.dumps(base, indent=2))


def _training_runner():
    """Run the training loop. Catches and records every exception so the
    Space stays up even if torch fails to load."""
    try:
        _write_status(state="loading_model")
        from train import run_training   # imported lazily — heavy deps

        _write_status(state="training")
        run_training(
            out_dir=str(OUT_DIR),
            status_writer=_write_status,
            eval_log_path=str(EVAL_LOG_PATH),
        )
        DONE_MARKER.touch()
        _write_status(state="done")
    except Exception as exc:
        tb = traceback.format_exc()
        _write_status(state="error", error=str(exc), traceback=tb[-2000:])


def _maybe_start_training():
    global _training_thread
    if DONE_MARKER.exists():
        _write_status(state="done")
        return
    if _training_thread is not None and _training_thread.is_alive():
        return
    _write_status(state="starting", step=0)
    _training_thread = threading.Thread(target=_training_runner, daemon=True)
    _training_thread.start()


@app.on_event("startup")
def _on_startup():
    _maybe_start_training()


# ----------------------------- request models -----------------------------
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


def _get_session(session_id: str) -> SuspectXEnvironment:
    if session_id not in _sessions:
        _sessions[session_id] = SuspectXEnvironment(factory=_factory)
    return _sessions[session_id]


# ------------------------------ env endpoints -----------------------------
@app.get("/info")
def info():
    status = {}
    if STATUS_PATH.exists():
        try:
            status = json.loads(STATUS_PATH.read_text())
        except Exception:
            pass
    return {
        "name": "suspect_x_env",
        "n_crimes_total": len(_factory),
        "n_train": len(_factory.split("train")),
        "n_heldout": len(_factory.split("heldout")),
        "active_sessions": len(_sessions),
        "max_turns_default": 20,
        "training": status,
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


# --------------------------- training endpoints ---------------------------
@app.get("/training/status")
def training_status():
    if not STATUS_PATH.exists():
        return {"state": "not_started"}
    return json.loads(STATUS_PATH.read_text())


@app.get("/training/log", response_class=PlainTextResponse)
def training_log():
    if not EVAL_LOG_PATH.exists():
        return ""
    return EVAL_LOG_PATH.read_text()


@app.get("/training/plot.png")
def training_plot():
    if not PLOT_PATH.exists():
        raise HTTPException(404, "plot not yet generated")
    return FileResponse(PLOT_PATH, media_type="image/png")


@app.post("/training/restart")
def training_restart():
    """Manual re-trigger. Removes done marker, spawns a new thread."""
    if DONE_MARKER.exists():
        DONE_MARKER.unlink()
    _maybe_start_training()
    return {"ok": True}
