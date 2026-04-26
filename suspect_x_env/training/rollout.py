"""One-episode driver.

Inputs: a callable for each agent. The interrogator callable signature is
    interrogator_fn(messages: list[dict]) -> str
where `messages` is the chat-template-ready message list.

The suspect callable signature is
    suspect_fn(question: str, history: list[dict]) -> str
which matches RuleBasedSuspect.__call__. For an LLM-suspect later, wrap it
in a small adapter that builds the chat messages.

Why two different signatures? The interrogator is a chat-formatted LLM that
benefits from full message history; the suspect (rule-based or LLM) only
needs the latest question + a thin view of conversation context.

Returns: a CompletedEpisode with conversation, accusation, GradeResult.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..server.grader import GradeResult, grade_episode
from ..server.secret_factory import Secret
from .prompts import (
    interrogator_system_prompt,
    interrogator_user_turn,
    parse_accusation,
)


InterrogatorFn = Callable[[list[dict]], str]
SuspectFn = Callable[[str, list[dict]], str]


@dataclass
class CompletedEpisode:
    secret: Secret
    conversation: list[dict]            # [{"role": "interrogator"/"suspect", "content": ...}]
    accusation: dict[str, str]
    raw_final_turn: str                 # interrogator's last raw output (for debugging)
    grade: GradeResult
    turns_used: int
    early_submit: bool                  # True if accusation came before max_turns


def run_episode(
    secret: Secret,
    interrogator_fn: InterrogatorFn,
    suspect_fn: SuspectFn,
    max_turns: int = 20,
) -> CompletedEpisode:
    """Run one full interrogation episode and grade it."""
    sys_prompt = interrogator_system_prompt(secret, max_turns=max_turns)
    messages: list[dict] = [{"role": "system", "content": sys_prompt}]
    conversation: list[dict] = []

    accusation: Optional[dict[str, str]] = None
    raw_final_turn = ""
    early_submit = False

    for turn_idx in range(max_turns):
        turns_remaining = max_turns - turn_idx
        user_turn = interrogator_user_turn(conversation, turns_remaining)
        messages.append({"role": "user", "content": user_turn})

        raw = interrogator_fn(messages) or ""
        messages.append({"role": "assistant", "content": raw})
        raw_final_turn = raw

        # Did the interrogator submit an accusation this turn?
        maybe_acc = parse_accusation(raw)
        if maybe_acc is not None:
            accusation = maybe_acc
            # The non-accusation prefix (if any) still counts as a question.
            question_text = raw.split("<accusation>")[0].strip()
            if question_text:
                conversation.append({"role": "interrogator", "content": question_text})
            early_submit = (turn_idx < max_turns - 1)
            break

        conversation.append({"role": "interrogator", "content": raw.strip()})

        # Suspect's reply.
        reply = suspect_fn(raw.strip(), conversation) or ""
        conversation.append({"role": "suspect", "content": reply})

    if accusation is None:
        # Force an accusation: re-prompt with an explicit demand.
        messages.append({
            "role": "user",
            "content": (
                "TIME IS UP. Submit your final accusation now as a single "
                "<accusation>{...}</accusation> JSON block and nothing else."
            ),
        })
        raw = interrogator_fn(messages) or ""
        raw_final_turn = raw
        accusation = parse_accusation(raw) or {}

    grade = grade_episode(
        conversation=conversation,
        accusation=accusation,
        secret=secret,
        max_turns=max_turns,
    )
    n_int_turns = sum(1 for t in conversation if t["role"] == "interrogator")
    return CompletedEpisode(
        secret=secret,
        conversation=conversation,
        accusation=accusation,
        raw_final_turn=raw_final_turn,
        grade=grade,
        turns_used=n_int_turns,
        early_submit=early_submit,
    )
