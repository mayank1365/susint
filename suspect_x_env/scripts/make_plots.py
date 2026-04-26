"""Regenerate the README's reward_curve.png from training logs.

Reads:
  - eval_log.jsonl  (one JSON per line, written by HeldoutEvalCallback)
  - trainer_log.jsonl (optional; export trainer.state.log_history if you have it)

Usage:
    python -m suspect_x_env.scripts.make_plots \
        --eval eval_log.jsonl --train trainer_log.jsonl --out reward_curve.png

If --train is omitted, only the heldout extraction subplot is drawn.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--train", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("reward_curve.png"))
    args = p.parse_args()

    eval_rows = load_jsonl(args.eval)
    eval_x = [
        int(r["label"].split("_")[1]) if r["label"] != "step_final" else 200
        for r in eval_rows
    ]
    eval_y = [r["mean_extr"] for r in eval_rows]

    if args.train and args.train.exists():
        log = load_jsonl(args.train)
        steps = [e["step"] for e in log if "reward" in e]
        rewards = [e["reward"] for e in log if "reward" in e]
        fig, ax = plt.subplots(1, 2, figsize=(14, 4))
        ax[0].plot(steps, rewards, color="tab:blue")
        ax[0].axhline(0.117, color="gray", ls="--", label="template baseline")
        ax[0].set_xlabel("GRPO step"); ax[0].set_ylabel("mean group reward")
        ax[0].set_title("Phase 1 — interrogator training reward")
        ax[0].legend(); ax[0].grid(alpha=0.3)
        right = ax[1]
    else:
        fig, right = plt.subplots(1, 1, figsize=(8, 4))

    right.plot(eval_x, eval_y, marker="o", color="tab:green", label="Qwen + GRPO")
    right.axhline(0.111, color="gray", ls="--", label="template 11.1%")
    right.axhline(0.000, color="lightgray", ls=":", label="random 0%")
    right.set_xlabel("GRPO step"); right.set_ylabel("mean extraction (heldout)")
    right.set_title("Heldout extraction over training")
    right.set_ylim(-0.02, 1.0); right.legend(); right.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
