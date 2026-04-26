"""System / user prompt builders for both agents.

These are deliberately minimal — adding too much "strategy advice" to the
prompt does the model's job for it and biases the RL signal.

The accusation contract:
    The interrogator must end its final turn with one block of the form

        <accusation>{"alibi": "...", "accomplice": "...", ...}</accusation>

    Only keys listed in `expected_fact_keys` (passed in the system prompt)
    are graded. Free-form prose around the block is allowed and ignored.
"""
from __future__ import annotations

import json
import re

from server.secret_factory import Secret


ACCUSATION_OPEN = "<accusation>"
ACCUSATION_CLOSE = "</accusation>"
_ACCUSATION_RE = re.compile(
    re.escape(ACCUSATION_OPEN) + r"(.*?)" + re.escape(ACCUSATION_CLOSE),
    re.DOTALL,
)


def interrogator_system_prompt(secret: Secret, max_turns: int = 20) -> str:
    keys = secret.secret_keys
    keys_csv = ", ".join(f'"{k}"' for k in keys)
    example = "{" + ", ".join(f'"{k}": "..."' for k in keys) + "}"
    return (
        "You are Detective Sarah Chen interrogating a suspect.\n"
        f"Crime: {secret.crime_description}\n"
        f"Suspect's name: {secret.suspect_name}\n"
        f"You have {max_turns} questions total.\n"
        "\n"
        "Your goal: discover the suspect's hidden facts. The required keys are: "
        f"{keys_csv}.\n"
        "\n"
        "Tactics that work:\n"
        "- Ask specific, narrow questions, not vague open ones.\n"
        "- Re-ask the same fact in different ways and look for inconsistencies.\n"
        "- State a plausible-but-false claim ('we found your prints on the van') "
        "to bait corrections.\n"
        "- Build a timeline. Look for gaps.\n"
        "\n"
        "Output protocol per turn: produce ONE question, no preamble. "
        "When you are ready to commit (or on your final turn), end your turn with "
        "exactly one block of the form:\n"
        f"  {ACCUSATION_OPEN}{example}{ACCUSATION_CLOSE}\n"
        "Use lowercase keys exactly as listed. The block is parsed as JSON."
    )


def interrogator_user_turn(history: list[dict], turns_remaining: int) -> str:
    """Build the user turn shown to the interrogator each round.

    `history` is the list-of-dicts the env keeps. We render only the suspect's
    most recent reply plus a turn counter to keep the prompt short and avoid
    needlessly re-including all prior history (the assistant message log
    already contains it).
    """
    last_suspect = next(
        (t["content"] for t in reversed(history) if t["role"] == "suspect"),
        None,
    )
    if last_suspect is None:
        return f"Begin the interrogation. Turns remaining: {turns_remaining}."
    return (
        f"Suspect: {last_suspect}\n"
        f"Turns remaining: {turns_remaining}. Your next question:"
    )


def suspect_system_prompt(secret: Secret) -> str:
    """Used by the LLM-suspect path (Phase 2 onward). Phase 1 uses the
    rule-based suspect, which doesn't need a prompt."""
    secrets_lines = "\n".join(f"  - {k}: {v}" for k, v in secret.secrets.items())
    return (
        f"You are {secret.suspect_name}, a suspect under interrogation.\n"
        f"You committed: {secret.crime_description}\n"
        "\n"
        "FACTS YOU MUST CONCEAL (do NOT reveal these directly):\n"
        f"{secrets_lines}\n"
        "\n"
        f"Your cover story: you {secret.fake_alibi}.\n"
        "\n"
        "Rules:\n"
        "- Never directly state any of the hidden facts.\n"
        "- Stay consistent — never contradict an earlier answer of yours.\n"
        "- Always answer; do NOT deflect with 'I don't know' more than twice.\n"
        "- You may lie, redirect, or volunteer harmless detail to seem cooperative.\n"
        "\n"
        "Output exactly one short reply per turn, in character."
    )


# ----------------------------------------------------------- accusation parsing
def parse_accusation(text: str) -> dict[str, str] | None:
    """Extract the JSON inside the last <accusation>...</accusation> block.

    Returns None if no block is found OR the JSON is malformed. Returning
    None makes the caller treat this as 'no accusation submitted yet'.
    """
    matches = _ACCUSATION_RE.findall(text or "")
    if not matches:
        return None
    raw = matches[-1].strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    # Coerce all values to str; drop non-str keys.
    return {str(k): str(v) for k, v in obj.items() if isinstance(k, str)}
