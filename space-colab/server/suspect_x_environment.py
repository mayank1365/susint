"""Core environment class.

Mirrors OpenEnv's `Environment` interface but does not import openenv-core
directly so the file is unit-testable without the framework installed. The
FastAPI layer (app.py) is the integration point with OpenEnv.

Action protocol (the trainer alternates these per turn):
    1. action_type="question"          -> records interrogator's question
    2. action_type="suspect_answer"    -> records suspect's reply
    3. action_type="submit_accusation" -> ends episode, returns rewards

The env itself is policy-free: it does not generate text. The trainer/
rollout loop is responsible for calling the LLM (interrogator or suspect)
and feeding the result back via step().
"""
from __future__ import annotations

from typing import Optional
from uuid import uuid4

from .grader import grade_episode
from .secret_factory import Secret, SecretFactory


class SuspectXEnvironment:
    """One env instance per concurrent session."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, factory: Optional[SecretFactory] = None):
        self.factory = factory or SecretFactory()
        self.secret: Optional[Secret] = None
        self.conversation: list[dict] = []
        self.episode_id: str = ""
        self.max_turns: int = 20
        self.done: bool = False
        # Whose turn we expect next ("interrogator" -> question; "suspect" -> answer).
        self.awaiting: str = "interrogator_question"

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        crime_id: Optional[str] = None,
        seed: Optional[int] = None,
        split: str = "train",
        max_turns: int = 20,
    ) -> dict:
        if crime_id is not None:
            self.secret = self.factory.get(crime_id)
        else:
            self.secret = self.factory.sample(split=split, seed=seed)
        self.conversation = []
        self.episode_id = str(uuid4())
        self.max_turns = max_turns
        self.done = False
        self.awaiting = "interrogator_question"
        return self._observation(reward=0.0, extra={"awaiting": self.awaiting})

    # ------------------------------------------------------------------- step
    def step(self, action: dict) -> dict:
        if self.done:
            raise RuntimeError("episode is already done; call reset()")
        if self.secret is None:
            raise RuntimeError("env not initialised; call reset() first")

        atype = action.get("action_type")
        content = action.get("content", "") or ""

        if atype == "question":
            if self.awaiting != "interrogator_question":
                # Allow it but record an "awaiting" mismatch in metadata.
                pass
            self.conversation.append({"role": "interrogator", "content": content})
            self.awaiting = "suspect_answer"
            return self._observation(reward=0.0)

        if atype == "suspect_answer":
            self.conversation.append({"role": "suspect", "content": content})
            self.awaiting = "interrogator_question"
            return self._observation(reward=0.0)

        if atype == "submit_accusation":
            accusation = action.get("accusation_json", {}) or {}
            return self._finalize(accusation)

        raise ValueError(f"unknown action_type: {atype!r}")

    # --------------------------------------------------------------- finalize
    def _finalize(self, accusation: dict[str, str]) -> dict:
        result = grade_episode(
            self.conversation,
            accusation,
            self.secret,
            max_turns=self.max_turns,
        )
        self.done = True
        self.awaiting = "done"
        return self._observation(
            reward=result.interrogator_reward,
            done=True,
            extra={
                "interrogator_reward": result.interrogator_reward,
                "suspect_reward": result.suspect_reward,
                "extraction_score": result.extraction_score,
                "no_false_facts_score": result.no_false_facts_score,
                "turn_efficiency_score": result.turn_efficiency_score,
                "concealment_score": result.concealment_score,
                "consistency_score": result.consistency_score,
                "plausibility_score": result.plausibility_score,
                "matched_keys": result.matched_keys,
                "per_key": result.per_key,
                "accusation": accusation,
                "secret": self.secret.to_dict(),  # revealed only at episode end
            },
        )

    # -------------------------------------------------------------- inspection
    @property
    def state(self) -> dict:
        n_int = sum(1 for t in self.conversation if t["role"] == "interrogator")
        return {
            "episode_id": self.episode_id,
            "step_count": len(self.conversation),
            "metadata": {
                "turns_used": n_int,
                "turns_remaining": max(0, self.max_turns - n_int),
                "awaiting": self.awaiting,
                "done": self.done,
            },
        }

    # --------------------------------------------------------------- internals
    def _observation(self, reward: float, done: bool = False, extra: dict | None = None) -> dict:
        n_int = sum(1 for t in self.conversation if t["role"] == "interrogator")
        meta = {
            "episode_id": self.episode_id,
            "turns_used": n_int,
            "turns_remaining": max(0, self.max_turns - n_int),
            "conversation": list(self.conversation),
            "awaiting": self.awaiting,
        }
        if self.secret is not None and not done:
            # Only crime_description (public) is exposed mid-episode.
            meta["public"] = self.secret.public_view()
        if extra:
            meta.update(extra)
        return {"done": done or self.done, "reward": reward, "metadata": meta}
