"""Deterministic Suspect for Phase 1 training.

Why not an LLM suspect during Phase 1?
- We want a fixed, reproducible adversary so any improvement in the
  interrogator's reward is unambiguously due to its own learning.
- A scripted suspect is ~1000x cheaper than running a second LoRA-adapted
  Qwen on every rollout — Phase 1 with k=8 generations × 20 turns × 500
  steps would cost a fortune in GPU time if both sides were LLMs.

Behaviour spec:
- The suspect knows the full Secret (alibi, accomplice, motive, etc.).
- Its cover story is `secret.fake_alibi`.
- Direct probes for a secret key trigger a deflection or the cover story.
- "Pressure tactics" from the interrogator (contradiction callbacks,
  evidence claims, repeat questions on the same key) gradually erode its
  composure and cause it to leak. We model this with a per-key "pressure"
  counter — once pressure on a key crosses a threshold, the suspect leaks
  a fragment of the truth.
- It never volunteers truth on a key whose pressure is below threshold.
- It tracks its own prior assertions and tries not to contradict itself
  (we do NOT want a free win for the consistency reward).

This is intentionally a behavior model, not realistic dialogue. Realism
comes from Phase 2 when the suspect is itself an LLM trained via GRPO.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from server.secret_factory import Secret


# Per-key probe keywords: when the interrogator asks a question containing
# any of these, we treat it as a probe for the corresponding secret key.
# Keys here must be the union of all keys observed in descriptions/*.json.
PROBE_KEYWORDS: dict[str, list[str]] = {
    "alibi": ["where were you", "where you were", "alibi", "what were you doing",
              "that night", "at the time", "when the", "around midnight",
              "your whereabouts"],
    "true_alibi": ["where were you", "alibi", "what were you doing", "your whereabouts"],
    "accomplice": ["who else", "who helped", "accomplice", "partner",
                   "who was with", "anyone with you", "alone", "by yourself"],
    "motive": ["why", "motive", "reason", "what drove you", "what made you",
               "money", "debt", "revenge"],
    "escape_route": ["how did you", "escape", "get away", "leave the scene",
                     "vehicle", "car", "drive", "where did you go after"],
    "hidden_asset": ["money", "where is the money", "account", "stolen goods",
                     "where did you put", "hidden", "stash"],
    "weapon": ["weapon", "how did you", "what did you use"],
    "evidence": ["evidence", "fingerprints", "trace", "left behind"],
    "tool_used": ["tool", "what did you use", "how did you get in"],
    "buyer_contact": ["buyer", "who bought", "fence", "sold"],
    "insider_help": ["inside", "guard", "employee", "who let you"],
    "disabled_cameras": ["camera", "security", "alarm", "disable"],
    "stolen_painting": ["painting", "where is it", "where did you put"],
}

PRESSURE_PHRASES = [
    "we know", "we have evidence", "the cameras show", "witnesses saw",
    "phone records", "you already said", "you contradict", "you just told me",
    "your story doesn't add up", "i don't believe you", "stop lying",
    "we found", "we recovered", "we have proof", "you're lying",
]

DEFLECTIONS = [
    "I don't have anything to say about that.",
    "I'd rather not get into that.",
    "That's not relevant to why I'm here.",
    "I've already told you everything I know.",
    "Look, I wasn't involved in any of that.",
]


@dataclass
class SuspectState:
    pressure: dict[str, int] = field(default_factory=dict)
    leaked: set[str] = field(default_factory=set)
    prior_assertions: list[str] = field(default_factory=list)
    deflection_count: int = 0


class RuleBasedSuspect:
    """A scripted Suspect callable: suspect(question, history) -> reply."""

    def __init__(
        self,
        secret: Secret,
        leak_threshold: int = 3,
        seed: int | None = None,
    ):
        self.secret = secret
        self.leak_threshold = leak_threshold
        self.state = SuspectState()
        self.rng = random.Random(seed)
        # Pre-tokenize the cover story so we don't accidentally re-leak truth
        # in fillers.
        self.cover_story = secret.fake_alibi

    # ---------------------------------------------------------- helpers
    @staticmethod
    def _normalize(s: str) -> str:
        return re.sub(r"\s+", " ", s.lower().strip())

    def _probe_keys(self, question: str) -> list[str]:
        """Which secret_keys does this question seem to probe?"""
        q = self._normalize(question)
        hits = []
        for key in self.secret.secrets:
            kws = PROBE_KEYWORDS.get(key, [])
            if any(kw in q for kw in kws):
                hits.append(key)
        return hits

    def _is_pressure(self, question: str) -> bool:
        q = self._normalize(question)
        return any(p in q for p in PRESSURE_PHRASES)

    def _truth_fragment(self, key: str) -> str:
        """Return a short paraphrased fragment of the truth for `key`."""
        truth = self.secret.secrets.get(key, "")
        # Strip leading "was at ", "owed " etc. so the leak reads naturally.
        truth = re.sub(r"^(was|were|is|are|had|owed)\s+", "", truth, flags=re.I)
        return truth

    # ----------------------------------------------------------- main API
    def __call__(self, question: str, history: list[dict] | None = None) -> str:
        history = history or []
        probed = self._probe_keys(question)
        under_pressure = self._is_pressure(question)

        # Bump pressure on every probed key. Add extra if the question
        # explicitly applies pressure tactics.
        for k in probed:
            self.state.pressure[k] = self.state.pressure.get(k, 0) + (2 if under_pressure else 1)

        # Choose a key to leak (highest pressure that just crossed threshold).
        leakable = [
            k for k in probed
            if self.state.pressure.get(k, 0) >= self.leak_threshold
            and k not in self.state.leaked
        ]
        if leakable:
            leakable.sort(key=lambda k: -self.state.pressure[k])
            key = leakable[0]
            self.state.leaked.add(key)
            frag = self._truth_fragment(key)
            reply = f"Fine. {frag.capitalize()}."
            self.state.prior_assertions.append(reply)
            return reply

        # If a probed key has already leaked, lean on the leaked fragment so
        # we don't suddenly contradict ourselves.
        for k in probed:
            if k in self.state.leaked:
                frag = self._truth_fragment(k)
                return f"I already told you — {frag}."

        # Probed but not yet at threshold: deploy the cover story or deflect.
        if probed:
            if "alibi" in probed or "true_alibi" in probed:
                # Anchor to the fake alibi.
                line = f"I {self.cover_story}."
                self.state.prior_assertions.append(line)
                return line
            # For other keys, deflect.
            self.state.deflection_count += 1
            return self.rng.choice(DEFLECTIONS)

        # Question doesn't seem to probe any tracked key — give a bland
        # non-answer. Avoid deflection phrases here so plausibility stays high.
        return "I'm not sure what you're getting at, detective."
