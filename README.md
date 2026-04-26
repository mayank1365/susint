# Suspect X — AI Interrogation Room

> **OpenEnv Hackathon submission** · Theme 1 (Multi-Agent) + Theme 4 (Self-Improvement)

A two-agent adversarial RL environment in which an **Interrogator** LLM extracts hidden facts from a **Suspect** LLM over 20 turns, then submits a structured JSON accusation that is graded by a **100% deterministic Python reward function** — no LLM-as-judge.

## 🔗 Links

- **HF Space (live env):** https://huggingface.co/spaces/<YOUR_USER>/suspect-x-env
- **Training notebook:** [`suspect_x_env/training/train_interrogator.ipynb`](suspect_x_env/training/train_interrogator.ipynb) (Colab-ready, Unsloth + TRL GRPO, Qwen2.5-7B-Instruct)
- **HF blog post:** <BLOG_URL>

## Problem

Current LLM RL benchmarks heavily test single-agent reasoning (math, code, browsing). They under-test **theory-of-mind**: modelling what *another* agent knows and doesn't know, then exploiting that asymmetry. Interrogation is the cleanest setting for this — the entire game is about the gap between two information sets.

## Environment

```
SecretFactory.generate()
       │
       ▼
┌─────────┐    crime_desc only    ┌──────────────────┐
│ SUSPECT │◄──────────────────────│  INTERROGATOR    │
│ (knows  │                       │  (knows only     │
│ secret) │──── answer (turn) ───►│  crime type)     │
└─────────┘                       └──────────────────┘
     │ ◄─────── 20 turns ──────────│
                                   submit_accusation(JSON)
                                            │
                                            ▼
                                   ┌──────────────┐
                                   │   GRADER     │
                                   │ JSON diff +  │
                                   │ consistency  │
                                   └──────┬───────┘
                                          ▼
                          reward_interrogator, reward_suspect
```

- **200 hand-authored crime scenarios** (`descriptions/crime_001..200.json`) with structured secrets across 6+ keys (alibi, accomplice, motive, escape_route, hidden_asset, ...).
- **Train/heldout split:** 170 / 30, deterministic by id.
- **6 reward signals**, all rule-based:
  - Interrogator: `0.7 × extraction + 0.2 × no_false_facts + 0.1 × turn_efficiency`
  - Suspect: `0.5 × concealment + 0.35 × consistency + 0.15 × plausibility`
- **OpenEnv-compliant:** `reset/step/state` over FastAPI, multi-session, Docker-deployable.

## Training

We trained **Qwen2.5-7B-Instruct** (4-bit + LoRA r=16) as the Interrogator using **GRPO** for 200 steps on Colab A100 (~3 hours). The Suspect is a deterministic rule-based script that holds a fake alibi, deflects on direct probes, and leaks under sustained pressure — which fixes the adversary so any reward improvement is unambiguously due to the Interrogator's policy.

```python
GRPOConfig(num_generations=4, lr=5e-6, beta=0.04,
           max_steps=200, max_completion_length=160, bf16=True)
```

## Results

Mean extraction rate on **30 heldout crimes** (never seen during training):

| Policy                          | Extraction | Reward |
| ------------------------------- | ---------: | -----: |
| Random interrogator             |      0.0%  | 0.013  |
| Template interrogator           |     11.1%  | 0.117  |
| **Qwen2.5-7B + GRPO (Phase 1)** |  **<X>%**  | <Y>    |

![reward curve](reward_curve.png)

Left: training reward per GRPO step.
Right: heldout extraction rate sampled every 50 steps; baselines marked.

## Repository layout

```
metaFinale/
├── descriptions/                   200 crime JSONs (training distribution)
├── stories/long/                   20 gold transcripts (eval/SFT seed corpus)
├── space/                          HF Space deploy bundle
│   ├── Dockerfile
│   ├── README.md
│   └── deploy.sh                   bash space/deploy.sh <user>/<space>
└── suspect_x_env/
    ├── openenv.yaml
    ├── pyproject.toml
    ├── client.py                   SuspectXEnv (http + local backends)
    ├── models.py                   pydantic Action/Observation/State
    ├── server/
    │   ├── secret_factory.py       loads descriptions/, train/heldout split
    │   ├── consistency_checker.py  regex contradiction + deflection counter
    │   ├── grader.py               6 deterministic rewards, no LLM
    │   ├── suspect_x_environment.py reset/step/state
    │   ├── app.py                  FastAPI server
    │   └── Dockerfile
    ├── training/
    │   ├── rule_based_suspect.py   pressure-driven scripted suspect
    │   ├── prompts.py              system/user prompts + accusation parser
    │   ├── rollout.py              run_episode → graded
    │   ├── baselines.py            random + template interrogators
    │   └── train_interrogator.ipynb  Colab GRPO training notebook
    ├── scripts/
    │   ├── evaluate_baseline.py    CLI for baseline numbers
    │   └── make_plots.py           regenerate plots from logs
    └── tests/                      12 tests, all passing
```

## Reproduce

**Run the env locally:**

```bash
cd suspect_x_env && python -m venv .venv && source .venv/bin/activate
pip install fastapi 'uvicorn[standard]' pydantic
PYTHONPATH=. uvicorn server.app:app --port 8000
```

**Run the test suite:**

```bash
PYTHONPATH=. pytest suspect_x_env/tests -v
```

**Reproduce the baselines:**

```bash
PYTHONPATH=. python -m suspect_x_env.scripts.evaluate_baseline --split heldout
```

**Train the interrogator:** open the notebook in Colab on an A100, set `REPO_ROOT`, run all.

## Anti-cheat guarantees

- The Interrogator never sees the secret, only `crime_description`.
- The accusation block is parsed via `json.loads`, not `eval`. Malformed → 0 reward.
- Extraction uses **token overlap, not exact string match**, so the Interrogator can't game the grader by parroting the secret verbatim — and conversely, paraphrased correct answers still get credit.
- The Suspect's prior assertions are tracked; rule-based contradictions count toward its consistency penalty.

## License

MIT.
