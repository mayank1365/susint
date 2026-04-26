"""Loads crime scenarios from descriptions/*.json and exposes a sampler.

The training distribution lives in `metaFinale/descriptions/`. Each file is
one crime with the schema:

    {
      "id": "crime_001",
      "crime_description": "...",
      "difficulty": "easy|medium|hard",
      "num_facts": 3..6,
      "suspect": {"name": "...", "fake_alibi": "..."},
      "secrets": [{"key": "alibi", "value": "..."}, ...]
    }

Held-out split is deterministic: the last `n_heldout` crimes by id. Trainers
should only sample from `split="train"`; eval scripts use `split="heldout"`.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# Path resolution: prefer env var, fall back to ../../descriptions relative
# to this file.
_DEFAULT_DESCRIPTIONS = Path(__file__).resolve().parents[2] / "descriptions"
DESCRIPTIONS_DIR = Path(os.environ.get("SUSPECT_X_DESCRIPTIONS", _DEFAULT_DESCRIPTIONS))

# How many crimes (by sorted id) are held out for evaluation.
N_HELDOUT = 30


@dataclass
class Secret:
    crime_id: str
    crime_description: str
    difficulty: str
    num_facts: int
    suspect_name: str
    fake_alibi: str
    # secrets[key] = value, e.g. {"alibi": "...", "accomplice": "...", ...}
    secrets: dict[str, str] = field(default_factory=dict)

    @property
    def secret_keys(self) -> list[str]:
        return list(self.secrets.keys())

    def public_view(self) -> dict[str, str]:
        """What the interrogator may see at reset."""
        return {
            "crime_id": self.crime_id,
            "crime_description": self.crime_description,
            "difficulty": self.difficulty,
            "suspect_name": self.suspect_name,
            "expected_fact_keys": list(self.secrets.keys()),
        }

    def to_dict(self) -> dict:
        return {
            "crime_id": self.crime_id,
            "crime_description": self.crime_description,
            "difficulty": self.difficulty,
            "num_facts": self.num_facts,
            "suspect_name": self.suspect_name,
            "fake_alibi": self.fake_alibi,
            "secrets": self.secrets,
        }


def _load_one(path: Path) -> Secret:
    raw = json.loads(path.read_text())
    secrets = {item["key"]: item["value"] for item in raw.get("secrets", [])}
    return Secret(
        crime_id=raw["id"],
        crime_description=raw["crime_description"],
        difficulty=raw.get("difficulty", "medium"),
        num_facts=raw.get("num_facts", len(secrets)),
        suspect_name=raw["suspect"]["name"],
        fake_alibi=raw["suspect"]["fake_alibi"],
        secrets=secrets,
    )


class SecretFactory:
    """Loads all crimes once and serves them with deterministic splits."""

    def __init__(self, descriptions_dir: Path | None = None, n_heldout: int = N_HELDOUT):
        self.descriptions_dir = Path(descriptions_dir or DESCRIPTIONS_DIR)
        if not self.descriptions_dir.exists():
            raise FileNotFoundError(
                f"descriptions dir not found: {self.descriptions_dir}. "
                "Set SUSPECT_X_DESCRIPTIONS env var or copy the folder into the env."
            )

        files = sorted(
            p for p in self.descriptions_dir.glob("crime_*.json") if p.is_file()
        )
        self._all: list[Secret] = [_load_one(p) for p in files]
        if not self._all:
            raise RuntimeError(f"no crime_*.json found under {self.descriptions_dir}")

        self._heldout: list[Secret] = self._all[-n_heldout:]
        self._train: list[Secret] = self._all[:-n_heldout]
        self._by_id: dict[str, Secret] = {s.crime_id: s for s in self._all}

    def split(self, name: Literal["train", "heldout", "all"]) -> list[Secret]:
        if name == "train":
            return self._train
        if name == "heldout":
            return self._heldout
        return self._all

    def get(self, crime_id: str) -> Secret:
        return self._by_id[crime_id]

    def sample(
        self,
        split: Literal["train", "heldout", "all"] = "train",
        seed: int | None = None,
    ) -> Secret:
        pool = self.split(split)
        rng = random.Random(seed)
        return rng.choice(pool)

    def __len__(self) -> int:
        return len(self._all)
