"""HTTP client for the Suspect X environment.

Two backends:

  - "http"  : talks to a running FastAPI server (local uvicorn or HF Space).
  - "local" : skips the network entirely and calls the env class in-process.
              Useful for fast unit tests and Colab GRPO loops where running
              an HTTP server adds latency.

The trainer should generally use "local" for speed and switch to "http" for
demos / CI.
"""
from __future__ import annotations

from typing import Any, Optional

from .models import InterrogationAction, Observation, State


class SuspectXEnv:
    def __init__(
        self,
        base_url: Optional[str] = None,
        session_id: str = "default",
        backend: str = "auto",
    ):
        """
        Args:
            base_url: e.g. "http://localhost:8000" or your HF Space URL.
            session_id: server-side session key (one episode per session).
            backend: "http", "local", or "auto" (http if base_url else local).
        """
        self.session_id = session_id
        if backend == "auto":
            backend = "http" if base_url else "local"
        self.backend = backend

        if backend == "http":
            if not base_url:
                raise ValueError("http backend requires base_url")
            import httpx  # imported lazily so local backend has no extra deps

            self._http = httpx.Client(base_url=base_url, timeout=60.0)
            self._env = None
        elif backend == "local":
            from .server.suspect_x_environment import SuspectXEnvironment

            self._env = SuspectXEnvironment()
            self._http = None
        else:
            raise ValueError(f"unknown backend: {backend}")

    # ------------------------------------------------------------ public API
    def reset(
        self,
        crime_id: Optional[str] = None,
        seed: Optional[int] = None,
        split: str = "train",
        max_turns: int = 20,
    ) -> Observation:
        body = {
            "crime_id": crime_id,
            "seed": seed,
            "split": split,
            "max_turns": max_turns,
        }
        if self.backend == "http":
            r = self._http.post(f"/reset?session_id={self.session_id}", json=body)
            r.raise_for_status()
            return Observation(**r.json())
        return Observation(**self._env.reset(**body))

    def step(self, action: InterrogationAction | dict) -> Observation:
        if isinstance(action, InterrogationAction):
            action_dict = action.model_dump()
        else:
            action_dict = action
        if self.backend == "http":
            r = self._http.post(
                f"/step?session_id={self.session_id}",
                json={"action": action_dict},
            )
            r.raise_for_status()
            return Observation(**r.json())
        return Observation(**self._env.step(action_dict))

    def state(self) -> State:
        if self.backend == "http":
            r = self._http.get(f"/state?session_id={self.session_id}")
            r.raise_for_status()
            return State(**r.json())
        return State(**self._env.state)

    def close(self) -> None:
        if self.backend == "http" and self._http is not None:
            self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc: Any):
        self.close()
