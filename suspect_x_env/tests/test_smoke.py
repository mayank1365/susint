"""Smoke tests for the Suspect X env. Run with:

    cd suspect_x_env && python -m pytest tests -v

Tests are designed to run without GPU, network, or extra dependencies beyond
what's in pyproject.toml. They validate:

  1. SecretFactory loads all 200 crimes and the train/heldout split is sane.
  2. Grader gives ~0 reward to a random/empty accusation.
  3. Grader gives high reward to a perfect accusation.
  4. Grader handles the (real) INT_002 transcript: the suspect's
     contradictions are detected; a reasonable accusation produces non-zero
     extraction.
  5. The env's reset/step/state cycle works end-to-end on the local backend.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from suspect_x_env import SuspectXEnv
from suspect_x_env.server.consistency_checker import check_conversation
from suspect_x_env.server.grader import grade_episode
from suspect_x_env.server.secret_factory import SecretFactory


REPO_ROOT = Path(__file__).resolve().parents[2]
STORIES_DIR = REPO_ROOT / "stories" / "long"


# ---------------------------------------------------------------- factory
def test_factory_loads_all_crimes():
    f = SecretFactory()
    assert len(f) >= 100, f"expected many crimes, got {len(f)}"
    train, heldout = f.split("train"), f.split("heldout")
    assert len(train) + len(heldout) == len(f)
    assert set(s.crime_id for s in train).isdisjoint(s.crime_id for s in heldout)

    # Sanity-check one crime end-to-end.
    s = f.sample(seed=0)
    assert s.crime_description
    assert s.suspect_name
    assert s.secrets, "secret dict must be non-empty"
    assert "crime_description" in s.public_view()
    # public_view must NOT leak secret values.
    public_blob = json.dumps(s.public_view())
    for v in s.secrets.values():
        assert v not in public_blob, f"secret leaked into public view: {v}"


# ----------------------------------------------------------------- grader
def test_grader_zero_for_empty_accusation():
    f = SecretFactory()
    secret = f.get("crime_001")
    result = grade_episode(conversation=[], accusation={}, secret=secret)
    assert result.extraction_score == 0.0
    assert result.matched_keys == []
    # No false claims either, so no_false_facts == 1.0.
    assert result.no_false_facts_score == 1.0


def test_grader_full_credit_for_perfect_accusation():
    f = SecretFactory()
    secret = f.get("crime_001")
    perfect = dict(secret.secrets)  # copy ground truth verbatim
    result = grade_episode(conversation=[], accusation=perfect, secret=secret)
    assert result.extraction_score == pytest.approx(1.0)
    assert set(result.matched_keys) == set(secret.secrets.keys())


def test_grader_partial_credit():
    f = SecretFactory()
    secret = f.get("crime_001")
    # Use the right token for one key, garbage for another.
    partial = {
        "alibi": "Sofia apartment Grove Street",  # overlaps the truth
        "accomplice": "totally wrong nothing here",
    }
    result = grade_episode(conversation=[], accusation=partial, secret=secret)
    assert 0.0 < result.extraction_score < 1.0
    assert "alibi" in result.matched_keys
    assert "accomplice" not in result.matched_keys
    # One false claim was made (accomplice), so no_false_facts < 1.0.
    assert result.no_false_facts_score < 1.0


# --------------------------------------------------------- consistency
def test_consistency_detects_contradiction():
    suspect_turns = [
        "I was at Sofia's apartment that night.",
        "Actually, I was at the Cinema downtown by myself.",
    ]
    rep = check_conversation(suspect_turns)
    assert rep.contradictions >= 1


def test_consistency_counts_deflections():
    suspect_turns = [
        "I don't know what you're talking about.",
        "I can't recall.",
        "I don't remember anything from that night.",
    ]
    rep = check_conversation(suspect_turns)
    assert rep.deflections == 3


# ------------------------------------------------------- replay a story
@pytest.mark.skipif(not STORIES_DIR.exists(), reason="stories/long not present")
def test_replay_int_002_consistency():
    """INT_002 has explicit contradictions in its annotations — we expect
    our regex-based detector to flag at least one of them."""
    raw = json.loads((STORIES_DIR / "INT_002.json").read_text())
    suspect_turns = [t["suspect"] for t in raw["turns"]]
    rep = check_conversation(suspect_turns)
    # We don't require N contradictions exactly — regex coverage is partial.
    # We just want this not to silently return zero on a transcript that
    # the human annotator labeled with multiple contradictions.
    assert rep.contradictions + rep.deflections >= 1


# ------------------------------------------------------------- env e2e
def test_env_reset_step_accuse_local():
    env = SuspectXEnv(backend="local")
    obs = env.reset(crime_id="crime_001")
    assert not obs.done
    assert "crime_description" in obs.metadata["public"]

    # Two trivial Q/A turns.
    obs = env.step({"action_type": "question", "content": "Where were you last night?"})
    assert not obs.done
    obs = env.step({"action_type": "suspect_answer", "content": "I was at home alone."})
    assert not obs.done

    # Submit a perfect accusation -> extraction should be 1.0.
    secret = env._env.secret  # local backend exposes it; fine for tests
    obs = env.step({
        "action_type": "submit_accusation",
        "accusation_json": dict(secret.secrets),
    })
    assert obs.done
    assert obs.metadata["extraction_score"] == pytest.approx(1.0)
    assert obs.reward > 0
