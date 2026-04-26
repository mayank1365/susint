---
title: Suspect X Environment
emoji: 🕵️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Adversarial interrogation RL environment for OpenEnv.
---

# Suspect X — AI Interrogation Room

Two-agent adversarial RL environment for the OpenEnv hackathon (Theme 1: Multi-Agent + Theme 4: Self-Improvement).

An **Interrogator** LLM extracts hidden facts from a **Suspect** LLM over 20 turns. The Suspect knows a structured secret (alibi, accomplice, motive, escape route, ...). The Interrogator only knows the public crime description. At the end of the episode, the Interrogator submits a JSON accusation. A 100% deterministic Python grader compares it to ground truth and returns rewards for both agents — **no LLM-as-judge**.

## Endpoints

- `GET  /info`                                 — env metadata, crime counts
- `POST /reset?session_id=...`                 — start a new episode
- `POST /step?session_id=...`                  — submit an action (question / suspect_answer / submit_accusation)
- `GET  /state?session_id=...`                 — inspect current episode
- `DELETE /session?session_id=...`             — clear a session

See the [training repo](#) for the GRPO trainer using Qwen2.5-7B-Instruct.

## Quick test

```bash
curl -s $SPACE_URL/info
curl -s -X POST "$SPACE_URL/reset?session_id=t1" \
  -H 'content-type: application/json' \
  -d '{"crime_id": "crime_001"}'
```
