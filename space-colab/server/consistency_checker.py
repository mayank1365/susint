"""Rule-based contradiction detection for the Suspect's turns.

We do NOT use an NLI model — keep the reward path 100% deterministic so the
grader is not exploitable via prompt injection or LLM judge hacking.

Approach:
- For each suspect turn, extract simple "fact assertions" via regex on a
  curated list of patterns (locations, times, named entities, yes/no claims
  about being somewhere).
- Track the first asserted value per fact_key across the conversation.
- If a later turn asserts a value for the same key that does not overlap
  (token-set), count one contradiction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# Regex patterns mapping a fact_key -> capturing pattern. These are
# intentionally generic; they fire on suspect prose, not on questions.
FACT_PATTERNS: dict[str, list[str]] = {
    "location": [
        r"\bI was at (?:the |a |an )?([A-Za-z][\w' ]{2,40})",
        r"\bI was in (?:the |a |an )?([A-Za-z][\w' ]{2,40})",
        r"\bat (?:the |a |an )?([A-Za-z][\w' ]{2,40}?) (?:apartment|house|bar|cinema|gallery|office|garage)",
    ],
    "companion": [
        r"\bwith ([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b",
        r"\b([A-Z][a-z]+) (?:was|were) with me\b",
    ],
    "time": [
        r"\bat (\d{1,2}(?::\d{2})? ?(?:am|pm|AM|PM))\b",
        r"\baround (\d{1,2}(?::\d{2})? ?(?:am|pm|AM|PM))\b",
    ],
    "alone": [
        r"\bI was (alone|by myself)\b",
    ],
    "vehicle": [
        r"\b(?:drove|driving|in) (?:a|my|the) ([\w ]+?(?:car|van|truck|motorcycle|bike))\b",
    ],
}


DEFLECTION_PHRASES = [
    "i don't know",
    "i don't remember",
    "i can't recall",
    "no comment",
    "i refuse to answer",
    "i'd rather not say",
    "i have nothing to say",
]


@dataclass
class ConsistencyReport:
    contradictions: int = 0
    deflections: int = 0
    asserted_facts: dict[str, str] = field(default_factory=dict)
    flagged_turns: list[int] = field(default_factory=list)


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 2}


def _compatible(a: str, b: str) -> bool:
    """Two assertions are compatible if their non-stopword tokens overlap."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return True
    return bool(ta & tb)


def check_conversation(suspect_turns: list[str]) -> ConsistencyReport:
    """Run consistency analysis over the suspect's utterances.

    Args:
        suspect_turns: list of suspect content strings, in order.

    Returns:
        ConsistencyReport with contradiction count, deflection count,
        the first-seen asserted value per fact_key, and turn indices flagged.
    """
    report = ConsistencyReport()

    for idx, text in enumerate(suspect_turns):
        lower = text.lower()
        if any(p in lower for p in DEFLECTION_PHRASES):
            report.deflections += 1

        for key, patterns in FACT_PATTERNS.items():
            for pat in patterns:
                m = re.search(pat, text)
                if not m:
                    continue
                value = m.group(1).strip()
                if key in report.asserted_facts:
                    if not _compatible(report.asserted_facts[key], value):
                        report.contradictions += 1
                        report.flagged_turns.append(idx)
                else:
                    report.asserted_facts[key] = value
    return report
