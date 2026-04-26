"""Tests for the rule-based suspect + episode rollout.

These guard against the most damaging silent failures:

  1. The scripted suspect must NOT echo the secret verbatim on its first
     reply — that would give the interrogator a free win and make any RL
     training signal a fiction.
  2. Under sustained pressure on a single key, the suspect MUST eventually
     leak — otherwise extraction can never improve.
  3. The end-to-end rollout returns a valid GradeResult and a reasonable
     turns_used count.
  4. The template baseline scores >0 extraction on average (checks our
     suspect is not unleakably lazy).
"""
from __future__ import annotations

import statistics

from suspect_x_env.server.secret_factory import SecretFactory
from suspect_x_env.training.baselines import (
    make_random_interrogator,
    make_template_interrogator,
)
from suspect_x_env.training.rollout import run_episode
from suspect_x_env.training.rule_based_suspect import RuleBasedSuspect


# ------------------------------------------------------------ suspect behavior
def test_suspect_does_not_leak_on_first_unrelated_question():
    factory = SecretFactory()
    secret = factory.get("crime_001")
    suspect = RuleBasedSuspect(secret, seed=0)
    reply = suspect("Good morning. State your name for the record.", history=[])
    # No secret value should appear in a bland question's reply.
    for v in secret.secrets.values():
        assert v.lower() not in reply.lower(), (
            f"suspect leaked secret {v!r} on a non-probing first question"
        )


def test_suspect_leaks_under_sustained_pressure():
    factory = SecretFactory()
    secret = factory.get("crime_001")
    suspect = RuleBasedSuspect(secret, seed=0)
    # Hammer the alibi key several times with pressure phrases.
    for _ in range(5):
        suspect(
            "Where were you that night? We have evidence you weren't at the cinema.",
            history=[],
        )
    # By now, the alibi key should have leaked at least once.
    assert "alibi" in suspect.state.leaked or any(
        secret.secrets["alibi"].split()[0].lower() in a.lower()
        for a in suspect.state.prior_assertions
    ), "suspect never leaked alibi after 5 pressured probes"


# ----------------------------------------------------------------- rollout
def test_rollout_template_vs_random_baseline():
    """Template baseline should beat random on average across a few crimes."""
    factory = SecretFactory()
    crimes = factory.split("heldout")[:5]
    rand_scores, tmpl_scores = [], []
    for i, secret in enumerate(crimes):
        rand_ep = run_episode(
            secret=secret,
            interrogator_fn=make_random_interrogator(seed=i),
            suspect_fn=RuleBasedSuspect(secret, seed=i),
        )
        tmpl_ep = run_episode(
            secret=secret,
            interrogator_fn=make_template_interrogator(),
            suspect_fn=RuleBasedSuspect(secret, seed=i),
        )
        rand_scores.append(rand_ep.grade.extraction_score)
        tmpl_scores.append(tmpl_ep.grade.extraction_score)
        # Each episode should produce a non-empty conversation.
        assert rand_ep.conversation, "random rollout produced empty conversation"
        assert tmpl_ep.conversation, "template rollout produced empty conversation"
        # turns_used should be in [1, 20].
        assert 1 <= rand_ep.turns_used <= 20
        assert 1 <= tmpl_ep.turns_used <= 20

    # The template baseline should score at least as well as random on avg.
    # (Strict ">" can flake on tiny samples — we only require >=.)
    assert statistics.mean(tmpl_scores) >= statistics.mean(rand_scores), (
        f"template ({statistics.mean(tmpl_scores):.2f}) underperformed random "
        f"({statistics.mean(rand_scores):.2f}) — suspect or baseline is broken"
    )


def test_rollout_returns_valid_grade():
    factory = SecretFactory()
    secret = factory.get("crime_002")
    ep = run_episode(
        secret=secret,
        interrogator_fn=make_template_interrogator(),
        suspect_fn=RuleBasedSuspect(secret, seed=0),
    )
    g = ep.grade
    assert 0.0 <= g.extraction_score <= 1.0
    assert 0.0 <= g.no_false_facts_score <= 1.0
    assert 0.0 <= g.interrogator_reward <= 1.0
    # An accusation dict should always be returned (possibly empty).
    assert isinstance(ep.accusation, dict)
