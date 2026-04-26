"""Action / Observation / State data models for the Suspect X environment."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


ActionType = Literal["question", "suspect_answer", "submit_accusation"]


class InterrogationAction(BaseModel):
    """A single action submitted to env.step().

    The same action type covers both agents' turns; the env tracks whose turn
    it is via `awaiting` in the previous Observation.
    """

    action_type: ActionType
    content: str = ""  # natural-language turn text (question or suspect answer)
    accusation_json: dict[str, str] = Field(default_factory=dict)  # only for submit_accusation


class Observation(BaseModel):
    done: bool = False
    reward: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class State(BaseModel):
    episode_id: str
    step_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    crime_id: Optional[str] = None  # if set, force a specific crime
    seed: Optional[int] = None
    split: Literal["train", "heldout", "all"] = "train"
    max_turns: int = 20


class StepRequest(BaseModel):
    action: InterrogationAction
