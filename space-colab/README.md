---
title: Suspect X — Train + Serve
emoji: 🕵️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Self-training Qwen 2.5 7B interrogator (GRPO).
---

# Suspect X — Train-on-Boot Space

This Space is **both** the OpenEnv environment server AND the trainer.

When the container starts, a background thread begins GRPO training of
Qwen2.5-7B-Instruct as the interrogator. The env endpoints work
immediately; the training endpoints surface progress.

## Endpoints

### Environment (works at any time)
- `GET  /info`                          — env metadata, training status
- `POST /reset?session_id=...`          — start a new episode
- `POST /step?session_id=...`           — submit an action
- `GET  /state?session_id=...`          — inspect current episode

### Training (read-only)
- `GET /training/status`                — JSON: step / max_steps / latest eval
- `GET /training/log`                   — full eval_log.jsonl as text
- `GET /training/plot.png`              — reward curve (after training)

## Hardware required

A GPU is mandatory. **T4 small** (~$0.60/hr) is the minimum. Set
hardware in Space Settings before pushing.
