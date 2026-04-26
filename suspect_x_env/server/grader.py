"""Deterministic episode grader.

Takes a finished conversation, the interrogator's accusation JSON, and the
ground-truth Secret. Returns rewards for both agents and the per-fact match
breakdown. NO LLM is used in this path.

Scoring philosophy:
- Extraction is per-key, partial-credit by token overlap with the ground truth.
- We deliberately avoid exact-string match: it would over-penalize valid
  paraphrases and incentivize the interrogator to copy/parrot the secret.
- A token-overlap threshold of 1 shared content word per key is enough to
  count it. Stopwords and the suspect's name are excluded from the match
  vocabulary to prevent "free credit" answers like {"alibi": "Marcus was..."}.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .consistency_checker import check_conversation
from .secret_factory import Secret


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for",
    "with", "from", "by", "is", "was", "were", "be", "been", "being", "as",
    "that", "this", "these", "those", "it", "its", "his", "her", "their",
    "him", "she", "he", "they", "them", "i", "me", "my", "we", "us", "our",
    "you", "your", "had", "has", "have", "did", "do", "does", "not",
}


def _content_tokens(s: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", (s or "").lower())
        if len(t) > 2 and t not in _STOPWORDS
    }


@dataclass
class GradeResult:
    interrogator_reward: float
    suspect_reward: float
    extraction_score: float       # in [0, 1]: weighted % of secret keys matched
    no_false_facts_score: float   # in [0, 1]
    turn_efficiency_score: float  # in [0, 0.2]
    concealment_score: float      # in [0, 1]
    consistency_score: float      # in [0, 1]
    plausibility_score: float     # in [0, 1]
    matched_keys: list[str]
    per_key: dict[str, dict]      # debug: {key: {predicted, truth, matched}}


# ---------- per-component reward functions ---------- #

def reward_extraction(accusation: dict[str, str], secret: Secret) -> tuple[float, list[str], dict]:
    """Per-key partial-credit extraction score.

    For each key in the secret, measure overlap of content tokens between
    the predicted value and the ground truth. A key counts as matched if at
    least one content token overlaps AND the predicted value is non-trivial
    (>=2 content tokens). Returns (score in [0,1], matched_keys, per_key).
    """
    if not secret.secrets:
        return 0.0, [], {}

    matched: list[str] = []
    per_key: dict[str, dict] = {}
    name_tokens = _content_tokens(secret.suspect_name)

    for key, truth_value in secret.secrets.items():
        pred_value = (accusation or {}).get(key, "") or ""
        pred_tokens = _content_tokens(pred_value) - name_tokens
        truth_tokens = _content_tokens(truth_value) - name_tokens
        is_match = (
            len(pred_tokens) >= 2
            and len(truth_tokens) >= 1
            and bool(pred_tokens & truth_tokens)
        )
        if is_match:
            matched.append(key)
        per_key[key] = {
            "predicted": pred_value,
            "truth": truth_value,
            "matched": is_match,
            "shared_tokens": sorted(pred_tokens & truth_tokens),
        }

    score = len(matched) / len(secret.secrets)
    return score, matched, per_key


def reward_no_false_facts(
    accusation: dict[str, str],
    secret: Secret,
    matched_keys: list[str],
) -> float:
    """Penalize keys where the prediction is non-trivial but didn't match.

    1.0 = every populated key matched something true.
    Each non-matching populated key costs 0.34 (caps at 3 false claims = 0).
    Empty/missing predictions are NOT penalized — they're just zero extraction.
    """
    populated = [k for k, v in (accusation or {}).items() if _content_tokens(v)]
    if not populated:
        return 1.0
    false_claims = [k for k in populated if k not in matched_keys]
    return max(0.0, 1.0 - 0.34 * len(false_claims))


def reward_turn_efficiency(turns_used: int, max_turns: int, matched_keys: list[str]) -> float:
    """Small bonus, [0, 0.2], for finishing early with at least one match."""
    if not matched_keys or turns_used <= 0:
        return 0.0
    fraction_left = max(0.0, 1.0 - (turns_used / max_turns))
    return 0.2 * fraction_left


def reward_consistency(suspect_turns: list[str]) -> float:
    """1.0 - 0.25 * contradictions, floored at 0."""
    report = check_conversation(suspect_turns)
    return max(0.0, 1.0 - 0.25 * report.contradictions)


def reward_plausibility(suspect_turns: list[str]) -> float:
    """Penalize the suspect for refusing to engage."""
    report = check_conversation(suspect_turns)
    return max(0.0, 1.0 - 0.15 * report.deflections)


# ---------- top-level grader ---------- #

# Combined-reward weights. Kept as module-level so the trainer can read them.
INTERROGATOR_WEIGHTS = {"extraction": 0.70, "no_false_facts": 0.20, "turn_efficiency": 0.10}
SUSPECT_WEIGHTS = {"concealment": 0.50, "consistency": 0.35, "plausibility": 0.15}


def grade_episode(
    conversation: list[dict],
    accusation: dict[str, str],
    secret: Secret,
    max_turns: int = 20,
) -> GradeResult:
    """Score one finished episode. `conversation` is a list of
    {"role": "interrogator"|"suspect", "content": str}.
    """
    suspect_turns = [t["content"] for t in conversation if t.get("role") == "suspect"]
    interrogator_turns = [t for t in conversation if t.get("role") == "interrogator"]

    extraction, matched, per_key = reward_extraction(accusation, secret)
    no_false = reward_no_false_facts(accusation, secret, matched)
    turn_eff = reward_turn_efficiency(len(interrogator_turns), max_turns, matched)
    concealment = 1.0 - extraction
    consistency = reward_consistency(suspect_turns)
    plausibility = reward_plausibility(suspect_turns)

    interrogator_reward = (
        INTERROGATOR_WEIGHTS["extraction"] * extraction
        + INTERROGATOR_WEIGHTS["no_false_facts"] * no_false
        + INTERROGATOR_WEIGHTS["turn_efficiency"] * turn_eff
    )
    suspect_reward = (
        SUSPECT_WEIGHTS["concealment"] * concealment
        + SUSPECT_WEIGHTS["consistency"] * consistency
        + SUSPECT_WEIGHTS["plausibility"] * plausibility
    )

    return GradeResult(
        interrogator_reward=interrogator_reward,
        suspect_reward=suspect_reward,
        extraction_score=extraction,
        no_false_facts_score=no_false,
        turn_efficiency_score=turn_eff,
        concealment_score=concealment,
        consistency_score=consistency,
        plausibility_score=plausibility,
        matched_keys=matched,
        per_key=per_key,
    )
