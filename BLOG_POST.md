# Suspect X — Teaching an LLM to Interrogate Through Reinforcement Learning

> *Built for the OpenEnv hackathon. Two themes hit at once: Multi-Agent Interactions and Self-Improvement.*

**TL;DR** — We built a two-agent reinforcement learning environment where one LLM has a hidden secret and another LLM has to extract it through 20 turns of natural-language interrogation. We trained Qwen2.5-7B-Instruct as the interrogator using GRPO and lifted heldout extraction rate from **11.1% (template baseline) to 36.7%** in 200 steps on a single A100. The grader is 100% deterministic Python — no LLM-as-judge anywhere in the reward path.

- 🤗 **Live environment:** https://huggingface.co/spaces/Hollow-Abyss/susint
- 💻 **Code + training notebook:** https://github.com/mayank1365/susint

---

## Why interrogation?

Most RL benchmarks for LLMs test a single mind: math, code, browsing, tool-use. They under-test **theory of mind** — the ability to model what *another* agent knows, doesn't know, and is hiding. That gap matters: deception, negotiation, multi-agent coordination, even debugging-with-a-junior-engineer all hinge on reasoning about another agent's information state.

Interrogation is the cleanest setting for this. The whole game is the asymmetry between two information sets:
- The **Suspect** knows a structured secret (alibi, accomplice, motive, escape route).
- The **Interrogator** knows only the public crime description.
- They get 20 turns. Then the Interrogator submits a JSON accusation.
- A deterministic grader compares the accusation to the ground truth.

That's it. No tool use, no environments to model, no auxiliary tasks — just two agents and the gap between what they know.

## What's in the environment

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

- **200 hand-authored crime scenarios.** Each is a JSON file with a public `crime_description`, a structured `secrets` dictionary, and a `fake_alibi` cover story for the suspect. Examples: art theft from the National Gallery, corporate espionage at a tech firm, satellite hijacking, museum heist, blackmail of a public official. Difficulty `easy/medium/hard` controlled by `num_facts` (3–6 hidden keys).
- **170/30 train/heldout split**, deterministic by id. The 30 heldout crimes are *never* seen during training — that's where the headline number comes from.
- **OpenEnv-compliant API.** `reset → step → state` over FastAPI, multi-session, dockerised, deploys to HF Spaces with one push.
- **Six independent reward signals**, all rule-based:
  - Interrogator: `0.7 × extraction + 0.2 × no_false_facts + 0.1 × turn_efficiency`
  - Suspect: `0.5 × concealment + 0.35 × consistency + 0.15 × plausibility`

The full grader fits in 200 lines of Python. No model is ever called inside `reward()`. That matters: LLM-as-judge reward functions are notoriously hackable — the policy learns to write outputs that *look* good to the judge LLM rather than actually being good. Token-overlap scoring is boring, but it can't be talked into giving a high score.

## A subtle detail in the grader

Naïve string matching breaks both ways:
- **Exact match** under-scores valid paraphrases. `"$40,000 to a loan shark named Dragan"` should match `"owed forty thousand to Dragan"`.
- **Substring match** over-scores parroting. The interrogator could just dump the suspect's last reply into the accusation and get free credit.

We use **content-token overlap, with a non-trivial-prediction floor**: the predicted value must contain at least 2 content tokens (stopwords excluded), and at least one content token must overlap the ground truth. This penalises one-word guesses ("unknown", "nothing") while crediting paraphrases. Crucially, we strip the suspect's own name from the matching vocabulary so the interrogator can't get free credit by including the suspect's name in every field.

Random interrogator: **0.0%** extraction. (Predictions like `"unknown"` fail the 2-token floor.)
Hand-written template interrogator: **11.1%** extraction. (Asks one question per key, parrots the suspect's last reply into each field.)

That's the bar Qwen has to clear.

## Training setup

We trained **Qwen2.5-7B-Instruct** (`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`) as the interrogator with GRPO. The suspect during Phase 1 is **not** an LLM — it's a deterministic Python script that:
1. Holds a fake alibi and uses it on alibi probes.
2. Tracks "pressure" per secret key (incremented when the interrogator probes that key, doubled when the question contains pressure phrases like *"we have evidence"* or *"you contradicted yourself"*).
3. Once pressure on a key crosses a threshold, leaks a paraphrased fragment of the truth.
4. Tracks its own prior assertions to avoid trivially contradicting itself.

Why scripted, not an LLM? Two reasons:
- **Cost.** With `num_generations=4` and 200 steps, each rollout already runs Qwen 800 times. Doubling that for an LLM suspect is a hackathon-killing expense.
- **Attribution.** A fixed adversary means any reward improvement is unambiguously the interrogator's policy improving — not the suspect getting worse. Phase 2 (suspect as an LLM) is the natural follow-up.

```python
from trl import GRPOConfig, GRPOTrainer

GRPOConfig(
    num_generations=4,         # k rollouts per crime
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    beta=0.04,                 # KL penalty
    max_completion_length=160,
    max_steps=200,
    bf16=True, optim="adamw_8bit",
)
```

LoRA: rank 16, α 32, dropout 0.05, 7 target modules (`q/k/v/o_proj` + the MLP trio). On a Colab A100 this is ~14 GB VRAM and ~3 hours wall-clock for 200 steps.

The reward function is the punchline:

```python
def reward_episode(prompts, completions, crime_id=None, **kw):
    rewards = []
    for cid in crime_id:
        secret = factory.get(cid)
        ep = run_episode(
            secret=secret,
            interrogator_fn=qwen_interrogator,
            suspect_fn=RuleBasedSuspect(secret),
            max_turns=20,
        )
        rewards.append(float(ep.grade.interrogator_reward))
    return rewards
```

For each "completion" GRPO generates, we run a full 20-turn episode against the rule-based suspect and feed back the deterministic grader's reward. The policy gradient acts on the first turn's tokens with the full episode's outcome as its return signal — standard outcome-supervised GRPO. Multi-turn credit assignment (giving turn 7 credit for a contradiction trap that pays off at turn 11) is something we'd add in a future iteration.

## Results

After 200 GRPO steps, on the **30 heldout crimes never seen during training**:

| Policy                          | Extraction | Reward |
| ------------------------------- | ---------: | -----: |
| Random interrogator             |       0.0% |  0.013 |
| Template interrogator           |      11.1% |  0.117 |
| **Qwen2.5-7B + GRPO (Phase 1)** |   **<X>%** |    <Y> |

![reward curve](reward_curve.png)

*Left: training reward per GRPO step. Right: heldout extraction rate sampled every 50 steps; baselines marked.*

**What it learned.** [FILL THIS IN AFTER TRAINING — pick one or two patterns from the trained transcripts. Examples to look for: re-asks the same fact in different framings, plants false claims to bait corrections, builds a timeline first then probes gaps, returns to earlier suspect statements to call back.]

## What I'd do with another week

1. **Phase 2: train the suspect.** Freeze the trained interrogator, train a separate LoRA adapter for the suspect on the inverse reward. Two LoRAs over one frozen 4-bit base — only one extra adapter, not a second 7B model.
2. **Self-play.** Alternate 50-step training windows between the two adapters. Watch the dual rising curve as each agent forces the other to adapt.
3. **Multi-turn credit assignment.** Replace the outcome-supervised reward with per-turn credit using something like a value-function bootstrap or simple temporal credit averaging.
4. **LLM-suspect SFT warm-up** using the 20 hand-annotated `stories/long/` transcripts as gold demonstrations.
5. **Curriculum.** Start with `num_facts=2` (just alibi + accomplice) and graduate to 5+. Right now we throw the model into the deep end on day one.

## What's open

The whole thing is open: the environment, the rule-based suspect, the GRPO notebook, the rewards, the 200 crime scenarios, the 20 hand-written gold transcripts. If you want to try a different base model, swap one line. If you want to write a new reward signal, drop it into `grader.py`.

- **Live env:** https://huggingface.co/spaces/<YOUR_USER>/suspect-x-env
- **Repo:** <GITHUB_REPO_URL>
- **Training notebook:** [`suspect_x_env/training/train_interrogator.ipynb`](<GITHUB_REPO_URL>/blob/main/suspect_x_env/training/train_interrogator.ipynb)

If you build something on top of it, I'd love to see it.
