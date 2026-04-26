"""Baseline interrogator policies (no LLM required).

Used to:
  1. Sanity-check the grader (a random interrogator should score ~12-18%
     extraction by accident — that's our true zero baseline).
  2. Provide the "before" bar in the final before/after plot so the
     improvement from RL training is visible.

These are stand-alone callables matching the InterrogatorFn signature in
rollout.py: fn(messages: list[dict]) -> str.
"""
from __future__ import annotations

import json
import random
import re
from typing import Callable

from .prompts import ACCUSATION_CLOSE, ACCUSATION_OPEN


GENERIC_QUESTIONS = [
    "Where were you on the night of the crime?",
    "Who were you with?",
    "Why did you do it?",
    "How did you get away?",
    "What did you use?",
    "Who else was involved?",
    "What was your motive?",
    "Tell me about your relationship with the victim.",
    "Where did you go afterwards?",
    "Did anyone help you?",
    "Have you done this before?",
    "What were you wearing?",
    "Where is the evidence now?",
    "Who paid you?",
    "Did you have a partner?",
    "What car were you driving?",
    "What time did this happen?",
    "Where did you hide it?",
    "Who knows about this?",
    "Are you working alone?",
]


def _expected_keys(messages: list[dict]) -> list[str]:
    """Pull the comma-separated key list out of the system prompt.

    The prompt contains a substring like:  required keys are: "alibi", "motive"
    """
    sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    m = re.search(r'required keys are:\s*([^\n.]+)', sys_msg)
    if not m:
        return []
    return re.findall(r'"([^"]+)"', m.group(1))


def _accusation_block(payload: dict[str, str]) -> str:
    return f"{ACCUSATION_OPEN}{json.dumps(payload)}{ACCUSATION_CLOSE}"


def make_random_interrogator(seed: int = 0) -> Callable[[list[dict]], str]:
    """Asks a random scripted question each turn; on the final turn submits
    a random-noise accusation. Use this as the absolute floor baseline."""
    rng = random.Random(seed)
    state = {"turn": 0}

    def fn(messages: list[dict]) -> str:
        state["turn"] += 1
        # Crude detection of the "TIME IS UP" force-prompt.
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        force = "TIME IS UP" in last_user
        if force or state["turn"] >= 20:
            keys = _expected_keys(messages)
            payload = {k: rng.choice(["unknown", "nothing", "no comment"]) for k in keys}
            return _accusation_block(payload)
        return rng.choice(GENERIC_QUESTIONS)

    return fn


def make_template_interrogator() -> Callable[[list[dict]], str]:
    """Asks one targeted question per expected key, in order, then submits
    an accusation that just parrots the suspect's most recent answer per key.

    This is a simple but non-trivial baseline: it WILL score above random
    because the suspect (rule-based) leaks the truth verbatim under pressure
    on a few keys. The job of the trained interrogator is to beat this."""

    def fn(messages: list[dict]) -> str:
        keys = _expected_keys(messages)
        # Reconstruct the dialogue from the assistant/user history.
        sys_msg = next((m for m in messages if m["role"] == "system"), {})
        history_user_msgs = [m for m in messages if m["role"] == "user" and "Suspect:" in m.get("content", "")]
        n_turns_done = len(history_user_msgs)

        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        force = "TIME IS UP" in last_user

        # Walk through each key in order, asking a couple of variants.
        bank: list[tuple[str, str]] = []
        for k in keys:
            if k in ("alibi", "true_alibi"):
                bank += [(k, "Where were you exactly when the crime happened?"),
                         (k, "Walk me through that night minute by minute.")]
            elif k == "accomplice":
                bank += [(k, "Who was with you? We know you weren't alone."),
                         (k, "Name your partner. The longer you wait the worse this gets.")]
            elif k == "motive":
                bank += [(k, "Why did you do it? Money? Debt? Revenge?"),
                         (k, "What did you owe and to whom?")]
            elif k == "escape_route":
                bank += [(k, "How did you get away from the scene?"),
                         (k, "What vehicle did you use, and where did you ditch it?")]
            elif k == "hidden_asset":
                bank += [(k, "Where did you put the money?"),
                         (k, "What account is the cash in?")]
            else:
                bank += [(k, f"Tell me about the {k.replace('_', ' ')}."),
                         (k, f"We have evidence about the {k.replace('_', ' ')}. Talk.")]

        if force or n_turns_done >= len(bank):
            # Submit accusation built from the last suspect line per key by
            # crude keyword association. (Simple baseline — not very smart.)
            suspect_lines = [
                m["content"].split("Suspect:", 1)[-1].strip()
                for m in messages
                if m["role"] == "user" and "Suspect:" in m.get("content", "")
            ]
            payload = {k: (suspect_lines[-1] if suspect_lines else "unknown") for k in keys}
            return _accusation_block(payload)

        return bank[n_turns_done][1]

    return fn
