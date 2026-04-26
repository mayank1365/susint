"""Run baseline interrogators against the rule-based suspect.

Usage:
    python -m suspect_x_env.scripts.evaluate_baseline --split heldout --n 30

Prints mean extraction rate, mean reward, and a per-baseline table. This is
the "before" number you reference in the final plot.
"""
from __future__ import annotations

import argparse
import statistics
from typing import Callable

from ..server.secret_factory import SecretFactory
from ..training.baselines import make_random_interrogator, make_template_interrogator
from ..training.rollout import run_episode
from ..training.rule_based_suspect import RuleBasedSuspect


def evaluate(
    name: str,
    interrogator_factory: Callable[[], Callable],
    crimes,
    seed_base: int = 0,
) -> dict:
    extractions, rewards = [], []
    for i, secret in enumerate(crimes):
        suspect = RuleBasedSuspect(secret, seed=seed_base + i)
        ep = run_episode(
            secret=secret,
            interrogator_fn=interrogator_factory(),
            suspect_fn=suspect,
            max_turns=20,
        )
        extractions.append(ep.grade.extraction_score)
        rewards.append(ep.grade.interrogator_reward)

    return {
        "name": name,
        "n": len(crimes),
        "mean_extraction": statistics.mean(extractions),
        "mean_reward": statistics.mean(rewards),
        "extraction_p25": sorted(extractions)[len(extractions) // 4],
        "extraction_p75": sorted(extractions)[3 * len(extractions) // 4],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="heldout", choices=["train", "heldout", "all"])
    p.add_argument("--n", type=int, default=0,
                   help="cap number of crimes (0 = all in split)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    factory = SecretFactory()
    crimes = factory.split(args.split)
    if args.n > 0:
        crimes = crimes[: args.n]

    print(f"Evaluating on {len(crimes)} crimes from split={args.split!r}")
    print("-" * 76)
    results = [
        evaluate("random", lambda: make_random_interrogator(seed=args.seed), crimes, args.seed),
        evaluate("template", make_template_interrogator, crimes, args.seed),
    ]
    print(f"{'baseline':<12} {'n':>4} {'mean_extr':>10} {'mean_reward':>12} "
          f"{'p25':>6} {'p75':>6}")
    for r in results:
        print(f"{r['name']:<12} {r['n']:>4} {r['mean_extraction']:>10.3f} "
              f"{r['mean_reward']:>12.3f} {r['extraction_p25']:>6.2f} "
              f"{r['extraction_p75']:>6.2f}")


if __name__ == "__main__":
    main()
