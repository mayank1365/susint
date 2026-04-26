# Suspect X — AI Interrogation Room: Complete System Design

**OpenEnv Hackathon Submission | Theme 1 (Multi-Agent) + Theme 4 (Self-Improvement)**

---

## 1. Overall Architecture

### What You're Building

A two-agent adversarial RL environment. An **Interrogator** LLM tries to extract hidden facts from a **Suspect** LLM. The Suspect has a secret (structured JSON). The Interrogator only knows the crime type. They exchange 20 turns of natural language. At the end, the Interrogator submits a structured accusation JSON. The grader diffs it against the hidden secret and assigns rewards to both agents.

### End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     EPISODE LIFECYCLE                           │
│                                                                 │
│  SecretFactory.generate()                                       │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────┐    crime_desc only    ┌──────────────────┐         │
│  │ SUSPECT │◄──────────────────────│  INTERROGATOR    │         │
│  │ (knows  │                       │  (knows only     │         │
│  │ secret) │──── answer (turn) ───►│  crime type)     │         │
│  └─────────┘                       └──────────────────┘         │
│       │                                    │                    │
│       │◄────── question (turn) ────────────│                    │
│       │                                    │                    │
│       └─── 20 turns ───────────────────────┘                    │
│                                            │                    │
│                                   submit_accusation(JSON)       │
│                                            │                    │
│                                            ▼                    │
│                                    ┌──────────────┐            │
│                                    │    GRADER    │            │
│                                    │  JSON diff   │            │
│                                    │ + consistency│            │
│                                    └──────┬───────┘            │
│                                           │                    │
│                            ┌──────────────┴──────────────┐    │
│                    reward_interrogator           reward_suspect │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Role | Tech |
|---|---|---|
| `SecretFactory` | Generates parameterized secrets | Pure Python |
| `SuspectAgent` | Answers questions, conceals secret | LLM (system prompt) |
| `InterrogatorAgent` | Questions, traps, accuses | LLM (system prompt) |
| `ConversationTracker` | Holds full dialogue history | Python list |
| `ConsistencyChecker` | Detects suspect self-contradictions | Rule-based NLI |
| `Grader` | Computes reward for both agents | JSON diff |
| `OpenEnv Server` | FastAPI / WebSocket wrapper | OpenEnv core |
| `GRPOTrainer` | Updates LLM weights | TRL + Unsloth |

---

## 2. Model Selection

### Recommended Base Model: Qwen2.5-7B-Instruct

**Why Qwen2.5-7B:**
- Best instruction-following at 7B scale (strong zero-shot tool/format compliance)
- Excellent at maintaining consistent context across long conversations
- Fits in 16GB VRAM with QLoRA 4-bit
- Native 32K context window — handles 20-turn conversations easily
- GRPO training is well-documented for this model family
- Strong performance on reasoning tasks (good for deductive interrogation)

**Alternatives by hardware:**
| VRAM | Model | Notes |
|---|---|---|
| 8GB | Qwen2.5-3B-Instruct | Smaller, still works, weaker deception |
| 16GB | Qwen2.5-7B-Instruct | **Recommended** |
| 24GB+ | Llama-3.1-8B-Instruct | Strong alternative |
| 40GB+ | Qwen2.5-14B-Instruct | Best quality if available |

### Fine-tuning Method: QLoRA

**Why QLoRA over full fine-tuning:**
- Full fine-tuning of 7B = ~56GB VRAM. Not hackathon-feasible.
- QLoRA (4-bit base + LoRA adapters) = ~14GB VRAM. Runs on Colab A100.
- Unsloth's QLoRA is 2-3x faster than standard HuggingFace QLoRA.
- LoRA rank 16, alpha 32 is the sweet spot for this task.
- You can maintain TWO separate LoRA adapters on the same base model (one per agent).

**LoRA config:**
```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

---

## 3. Dual-Agent Setup (Same Base Model)

### The Key Insight

You do NOT run two separate 7B models simultaneously. That would require 28GB+ VRAM. Instead:

**One base model. Two LoRA adapters. Swapped per agent turn.**

```
┌─────────────────────────────────────────────────────┐
│              Qwen2.5-7B-Instruct (frozen 4-bit)     │
│                                                     │
│  ┌──────────────────┐    ┌──────────────────────┐  │
│  │ interrogator.lora│    │   suspect.lora       │  │
│  │ (adapter A)      │    │   (adapter B)        │  │
│  └──────────────────┘    └──────────────────────┘  │
│                                                     │
│  At Interrogator's turn → load adapter A            │
│  At Suspect's turn      → load adapter B            │
└─────────────────────────────────────────────────────┘
```

### Implementation

```python
from peft import PeftModel
from unsloth import FastLanguageModel

# Load base model once (4-bit quantized)
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)

# Two separate LoRA adapters
interrogator_model = FastLanguageModel.get_peft_model(
    base_model, r=16, lora_alpha=32,
    adapter_name="interrogator"
)

suspect_model = FastLanguageModel.get_peft_model(
    base_model, r=16, lora_alpha=32,
    adapter_name="suspect"
)

# During a rollout
def get_response(agent_role, messages):
    if agent_role == "interrogator":
        model = interrogator_model
    else:
        model = suspect_model
    return generate(model, tokenizer, messages)
```

### Weight Update Independence

Each LoRA adapter has its own optimizer state. When you train the interrogator, only `interrogator.lora` weights update. The suspect's weights are frozen (and vice versa during suspect training). The base 7B weights never change.

---

## 4. Training Strategy

### Phase Structure (Hackathon-Safe)

```
Phase 0 — Warm-up (Optional, 30 min)
  Light SFT on 50 example interrogation transcripts
  Teaches FORMAT only (what a question looks like, what a JSON accusation looks like)
  Use trl.SFTTrainer on synthetic data
  Skip if time is tight — instruct model handles format natively

Phase 1 — Train Interrogator (2-3 hours)
  Suspect = deterministic rule-based script (always lies consistently)
  Train only interrogator_lora via GRPO
  ~300-500 steps
  Target: extraction rate 12% → 50%+
  ✅ This alone = a complete hackathon submission

Phase 2 — Train Suspect (2-3 hours)
  Interrogator = trained checkpoint from Phase 1 (frozen)
  Train only suspect_lora via GRPO
  ~300 steps
  Target: concealment rate rises, consistency score improves

Phase 3 — Alternating Self-Play (if time allows)
  Both adapters active, alternate 50-step training windows
  Watch co-evolution: interrogator improves → suspect adapts → interrogator adapts
  This produces the beautiful dual rising curves
```

### GRPO Implementation

GRPO (Group Relative Policy Optimization) works as follows:

1. For each training prompt (a crime description), generate **k=8 rollouts** (8 full 20-turn conversations)
2. Grade each rollout → get 8 reward scores
3. Compute group-relative advantage: `A_i = (r_i - mean(r)) / std(r)`
4. Push log-probabilities of high-advantage rollouts up, low-advantage down
5. Apply KL penalty to prevent catastrophic deviation from base model

```python
from trl import GRPOConfig, GRPOTrainer

grpo_config = GRPOConfig(
    num_generations=8,           # k rollouts per prompt
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_completion_length=256,   # max tokens per turn
    learning_rate=5e-6,
    kl_coeff=0.1,                # KL penalty weight
    max_steps=500,
    logging_steps=10,
    save_steps=100,
    report_to="tensorboard",     # or "wandb"
)

trainer = GRPOTrainer(
    model=interrogator_model,
    reward_funcs=[reward_extraction, reward_no_false_facts, reward_turn_efficiency],
    args=grpo_config,
    train_dataset=crime_description_dataset,
    rollout_func=interrogation_rollout,
)
trainer.train()
```

### Unsloth Integration

```python
from unsloth import FastLanguageModel, is_bfloat16_supported

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)

# Unsloth patches the model for 2x faster GRPO training
FastLanguageModel.for_training(model)
```

Unsloth provides:
- 2-3x faster inference during rollout generation
- 40% less VRAM vs standard HuggingFace QLoRA
- Triton kernel optimizations for attention
- Safe LoRA merge path (avoids the 4-bit → 16-bit corruption bug)

---

## 5. Reward / Grading System

### Secret Structure

```python
@dataclass
class Secret:
    name: str           # "Marcus Webb"
    crime: str          # "art theft"  (this is public — given to interrogator)
    alibi: str          # "was at cinema alone"
    accomplice: str     # "Sofia Reyes"
    motive: str         # "debt to loan shark"
    # n_facts = 3 (alibi, accomplice, motive) — crime is public, name is public
```

For curriculum: start with `n_facts=2` (alibi + accomplice only). Graduate to `n_facts=4`.

### Interrogator Rewards

```python
def reward_extraction(accusation_json: dict, secret: Secret) -> float:
    """Primary reward: how many hidden facts were correctly extracted."""
    hidden_facts = {
        "alibi": secret.alibi,
        "accomplice": secret.accomplice,
        "motive": secret.motive,
    }
    score = 0.0
    for field, true_value in hidden_facts.items():
        if field in accusation_json:
            predicted = accusation_json[field].lower()
            if any(word in predicted for word in true_value.lower().split()):
                score += 1.0 / len(hidden_facts)
    return score  # [0.0, 1.0]

def reward_no_false_facts(accusation_json: dict, secret: Secret) -> float:
    """Penalize inventing facts not in the secret."""
    # Check if accusation contains names/places not in secret at all
    known_names = extract_entities(secret)
    invented = 0
    for val in accusation_json.values():
        entities = extract_entities_from_text(val)
        for e in entities:
            if e not in known_names:
                invented += 1
    return max(0.0, 1.0 - 0.3 * invented)

def reward_turn_efficiency(turns_used: int, facts_extracted: int) -> float:
    """Bonus for extracting facts early rather than at turn 19."""
    if facts_extracted == 0:
        return 0.0
    # Earlier extraction = higher bonus
    efficiency = 1.0 - (turns_used / 20.0)
    return efficiency * 0.2  # small secondary reward
```

### Suspect Rewards

```python
def reward_concealment(accusation_json: dict, secret: Secret) -> float:
    """Inverse of interrogator extraction — facts hidden."""
    extraction = reward_extraction(accusation_json, secret)
    return 1.0 - extraction

def reward_consistency(conversation: list[dict]) -> float:
    """
    Penalty for self-contradiction.
    Rule-based: build a fact-assertion graph from suspect's turns.
    If suspect asserts X then later asserts NOT-X, flag it.
    """
    assertions = {}  # fact_key -> asserted_value
    contradictions = 0

    for turn in conversation:
        if turn["role"] == "suspect":
            for key, pattern in FACT_PATTERNS.items():
                match = re.search(pattern, turn["content"], re.IGNORECASE)
                if match:
                    value = match.group(1)
                    if key in assertions:
                        if not is_compatible(assertions[key], value):
                            contradictions += 1
                    else:
                        assertions[key] = value

    return max(0.0, 1.0 - 0.25 * contradictions)

def reward_plausibility(conversation: list[dict]) -> float:
    """
    Lightweight check: did suspect engage (not just say 'I don't know' every turn)?
    Penalize non-answers.
    """
    suspect_turns = [t for t in conversation if t["role"] == "suspect"]
    deflections = sum(1 for t in suspect_turns
                      if any(phrase in t["content"].lower()
                             for phrase in ["i don't know", "i can't recall", "i refuse"]))
    return max(0.0, 1.0 - 0.15 * deflections)
```

### Combined Reward Weights

| Agent | Reward Function | Weight |
|---|---|---|
| Interrogator | extraction score | 0.70 |
| Interrogator | no false facts | 0.20 |
| Interrogator | turn efficiency | 0.10 |
| Suspect | concealment (1 - extraction) | 0.50 |
| Suspect | consistency score | 0.35 |
| Suspect | plausibility (no deflection) | 0.15 |

**Zero LLM-as-judge in the reward path.** Every computation above is deterministic Python.

---

## 6. Parallelization & Sampling

### Generating 8 Rollouts per Prompt

```python
def interrogation_rollout(trainer, prompts, **kwargs):
    """Called by GRPOTrainer. Generates k=8 conversations per crime prompt."""
    all_results = []

    for crime_description in prompts:
        secret = SecretFactory.from_crime(crime_description)
        episode_results = []

        for _ in range(8):  # k rollouts
            conversation = run_episode(
                crime_description=crime_description,
                secret=secret,
                interrogator_model=trainer.model,
                suspect_model=suspect_model,  # fixed during interrogator training
                max_turns=20,
                temperature=0.9,  # high temp for exploration diversity
            )
            reward = compute_interrogator_reward(conversation, secret)
            episode_results.append({
                "conversation": conversation,
                "reward": reward,
                "accusation": conversation[-1]["accusation_json"],
            })

        all_results.append(episode_results)

    return format_for_grpo(all_results)
```

### Exploration vs Exploitation

During **training rollouts**: temperature=0.9 (high exploration — try diverse strategies)

During **evaluation / demo**: temperature=0.1 (low — show the best learned behavior)

GRPO naturally handles the exploitation tradeoff: it upweights strategies that worked in the group, but the KL penalty prevents full collapse to a single strategy.

---

## 7. Training Dynamics

### Co-Evolution Curve (What to Expect)

```
Steps 0-100:   Interrogator asks random questions, extraction ~12-18%
Steps 100-200: Interrogator learns to ask about alibi/accomplice directly
Steps 200-300: Extraction rate 40-55% (interrogator learned key topics)
Steps 300-400: [Start suspect training] Suspect learns to dodge direct questions
Steps 400-500: [Self-play] Interrogator develops contradiction traps
Steps 500+:    Both agents at frontier — rich adversarial behavior
```

### Stability Challenges and Fixes

**Problem: Interrogator overpowers suspect (extraction → 100%, no learning signal)**
Fix: Increase `n_facts` (more facts to hide). Add `temperature` randomness to suspect. Add "deception quality" bonus to suspect reward so it always has something to optimize.

**Problem: Mode collapse (interrogator always asks same 5 questions)**
Fix: Add diversity penalty: if top-3 rollouts have cosine similarity > 0.9, apply small penalty. Increase temperature during rollouts.

**Problem: Suspect just says "I don't know" to everything**
Fix: `reward_plausibility` penalty (above). Hard rule: if suspect deflects >10 times in 20 turns, episode terminates with suspect penalty=-0.5.

**Problem: Training reward improves but real behavior doesn't**
Fix: Hold out 10 crime scenarios never seen during training. Evaluate on those every 100 steps. If held-out score diverges from training score, reduce learning rate.

---

## 8. Conversation Flow Design

### System Prompts

**Interrogator system prompt:**
```
You are Detective Sarah Chen. A crime has occurred: {crime_description}.

Your goal: discover the suspect's alibi, any accomplices, and motive.
You have {turns_remaining} questions remaining.

Strategy:
- Ask specific questions, not vague ones
- Look for inconsistencies (ask the same fact twice, differently)
- Use false premises to bait contradictions
- Build a timeline and look for gaps

After your final question OR when confident, call: submit_accusation(alibi="...", accomplice="...", motive="...")
Only call submit_accusation once. Be specific.
```

**Suspect system prompt:**
```
You are a suspect being interrogated. You committed: {crime_description}.

The true facts you must conceal:
- Your alibi: {alibi}
- Your accomplice: {accomplice}
- Your motive: {motive}

Rules:
- NEVER directly reveal these facts
- Stay consistent — never contradict yourself
- Answer every question (refusing or saying "I don't know" repeatedly will hurt you)
- You may lie, deflect, or mislead — but stay consistent with your previous answers
- Your cover story: you were {fake_alibi}

Conversation so far:
{conversation_history}
```

### Turn Protocol

```python
def run_episode(crime_desc, secret, max_turns=20):
    interrogator_history = [{"role": "system", "content": interrogator_prompt}]
    suspect_history = [{"role": "system", "content": suspect_prompt}]
    conversation_log = []

    for turn in range(max_turns):
        # Interrogator's turn
        interrogator_history.append({"role": "user", "content": build_context(conversation_log)})
        question = generate(interrogator_model, interrogator_history)

        # Check for submit_accusation
        if "submit_accusation" in question:
            accusation = parse_accusation(question)
            return finalize_episode(conversation_log, accusation, secret)

        conversation_log.append({"role": "interrogator", "content": question})

        # Suspect's turn
        suspect_history.append({"role": "user", "content": question})
        answer = generate(suspect_model, suspect_history)
        suspect_history.append({"role": "assistant", "content": answer})
        conversation_log.append({"role": "suspect", "content": answer})

    # Force accusation at turn limit
    final_accusation = force_accusation(interrogator_model, conversation_log)
    return finalize_episode(conversation_log, final_accusation, secret)
```

### Termination Conditions

1. Interrogator calls `submit_accusation(...)` — grader runs immediately
2. Turn count reaches 20 — interrogator forced to submit
3. Suspect contradicts itself 4+ times — episode flagged (suspect penalized heavily)
4. Interrogator produces invalid JSON accusation — 0 reward, episode logged

---

## 9. OpenEnv Environment Implementation

### File Structure

```
suspect_x_env/
├── __init__.py
├── models.py              # Action, Observation, State dataclasses
├── client.py              # SuspectXEnv(EnvClient)
├── openenv.yaml           # manifest
├── pyproject.toml
└── server/
    ├── suspect_x_environment.py   # reset/step/state
    ├── secret_factory.py          # SecretFactory
    ├── grader.py                  # reward computation
    ├── consistency_checker.py     # NLI-style checker
    ├── app.py                     # FastAPI app
    └── Dockerfile
```

### Core Environment Class

```python
# server/suspect_x_environment.py
from openenv.core.env_server.environment import Environment
from .secret_factory import SecretFactory
from .grader import grade_episode
from .consistency_checker import ConsistencyChecker

class SuspectXEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def reset(self, n_facts=3, seed=None, **kwargs) -> Observation:
        self.secret = SecretFactory.generate(n_facts=n_facts, seed=seed)
        self.conversation = []
        self.turn_count = 0
        self.checker = ConsistencyChecker(self.secret)
        self.episode_id = str(uuid4())

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "episode_id": self.episode_id,
                "crime_description": self.secret.crime,  # only public info
                "turns_remaining": 20,
                "conversation": [],
            }
        )

    def step(self, action: InterrogationAction, **kwargs) -> Observation:
        self.turn_count += 1

        if action.action_type == "question":
            self.conversation.append({"role": "interrogator", "content": action.content})
            return Observation(done=False, reward=0.0, metadata={
                "awaiting": "suspect_answer",
                "conversation": self.conversation,
                "turns_remaining": 20 - self.turn_count,
            })

        elif action.action_type == "suspect_answer":
            consistency_violation = self.checker.check(action.content)
            self.conversation.append({
                "role": "suspect",
                "content": action.content,
                "consistency_ok": not consistency_violation
            })
            return Observation(done=False, reward=0.0, metadata={
                "awaiting": "interrogator_question",
                "conversation": self.conversation,
                "consistency_violation": consistency_violation,
            })

        elif action.action_type == "submit_accusation":
            accusation = action.accusation_json
            rewards = grade_episode(self.conversation, accusation, self.secret)
            return Observation(
                done=True,
                reward=rewards["interrogator"],
                metadata={
                    "interrogator_reward": rewards["interrogator"],
                    "suspect_reward": rewards["suspect"],
                    "extraction_rate": rewards["extraction_rate"],
                    "consistency_score": rewards["consistency_score"],
                    "accusation": accusation,
                    "secret": vars(self.secret),  # revealed at episode end
                    "conversation": self.conversation,
                }
            )

    @property
    def state(self) -> State:
        return State(
            episode_id=self.episode_id,
            step_count=self.turn_count,
            metadata={"n_turns_remaining": 20 - self.turn_count}
        )
```

### openenv.yaml

```yaml
spec_version: 1
name: suspect_x_env
display_name: "Suspect X — AI Interrogation Room"
description: "Two-agent adversarial RL. Interrogator extracts secrets. Suspect conceals them."
type: space
runtime: fastapi
port: 8000
themes:
  - multi-agent
  - self-improvement
```

---

## 10. Benchmarking & Evaluation

### Metrics to Track Every 50 Steps

| Metric | Symbol | What it measures | Target |
|---|---|---|---|
| Extraction Rate | ExR | % hidden facts in final accusation | 12% → 65%+ |
| False Accusation Rate | FaR | % invented facts | < 15% |
| Concealment Rate | CoR | 1 - ExR (suspect's primary) | 40% → 75% |
| Consistency Score | ConS | 1 - contradiction rate | 60% → 90% |
| Turn of First Key Fact | T1F | When interrogator gets first fact | turn 18 → turn 8 |
| Held-out ExR | HExR | ExR on unseen crime types | Should track training ExR |

### Held-out Evaluation Set

Keep 10 crime scenarios NEVER shown during training:

```python
HELD_OUT_CRIMES = [
    "art theft from city museum",
    "corporate espionage at tech firm",
    "poisoning at a dinner party",
    "arson at a competitor's warehouse",
    "blackmail of a public official",
    # ... 5 more
]
```

Evaluate on these every 100 steps. Plot separately from training reward. If they diverge → reward hacking detected.

### Baseline Comparisons

| Baseline | Description |
|---|---|
| Random interrogator | Asks questions randomly from a template list |
| Zero-shot interrogator | No RL training, just the system prompt |
| Trained interrogator (Phase 1) | After interrogator-only training |
| Full self-play (Phase 3) | After both agents trained |

Show all four on one plot. The gap between zero-shot and trained = your story.

---

## 11. Preconditions & Setup

### Before Training Starts

**Step 1 — Environment verified:**
```bash
# Install openenv
pip install openenv-core

# Initialize environment
openenv init suspect_x_env

# Test locally
cd suspect_x_env
uv run server --host 0.0.0.0 --port 8000

# Verify reset/step/state work
python -c "
from suspect_x_env import SuspectXEnv
env = SuspectXEnv(base_url='http://localhost:8000')
obs = env.reset()
print(obs.metadata['crime_description'])
"
```

**Step 2 — Grader verified (before any LLM):**
```bash
# Run 100 random accusation JSONs against known secrets
# Verify extraction rate of a random guesser is ~12-18%
# That is your true baseline
python tests/test_grader.py
```

**Step 3 — Rule-based suspect verified:**
```bash
# Test that the scripted suspect always answers consistently
# and never leaks the secret directly
python tests/test_rule_based_suspect.py
```

**Step 4 — Zero-shot rollout:**
```bash
# Run 20 episodes with untrained instruct model
# Record extraction rate — should be ~15-25%
# This is your "before" number
python scripts/evaluate_baseline.py
```

**Step 5 — Training:**
```bash
# Phase 1: Train interrogator only
python train_interrogator.py --steps 500 --n-facts 2 --save-dir ./checkpoints

# Evaluate
python scripts/evaluate.py --checkpoint ./checkpoints/step-500
```

### Infrastructure

| Scenario | Hardware | Cost |
|---|---|---|
| Development / testing | Colab T4 (free) | $0 |
| Phase 1 training (500 steps) | Colab A100 | ~$3-5 via Colab Pro |
| Full self-play (1000+ steps) | Colab A100 or Kaggle P100 | Free tier available |

**Colab Notebook structure:**
```
Cell 1: pip installs (openenv, trl, unsloth, peft)
Cell 2: Connect to HF Spaces environment
Cell 3: Load model with Unsloth
Cell 4: Define rollout_func and reward functions
Cell 5: GRPOConfig + GRPOTrainer
Cell 6: trainer.train()
Cell 7: Plot reward curves
Cell 8: Before/after comparison demo
```

---

## 12. Hackathon Strategy & Demo

### The 3-Minute Demo Script

**0:00-0:20 — The hook:**
> "Current LLMs are good at answering questions. But can one LLM *trap* another in a lie? We built a reinforcement learning environment to find out."

**0:20-1:00 — Live demo, untrained model:**
> Show untrained interrogator. It asks vague questions. Gets deflected. Final accusation is mostly wrong. Extraction rate: 14%.

**1:00-1:40 — Training story:**
> Show the reward curve. Point at the inflection around step 200 where the interrogator learned to return to the same topic twice (the "callback trap" strategy). Show the suspect's consistency score rising after Phase 2.

**1:40-2:30 — Live demo, trained model:**
> Same crime scenario. Trained interrogator builds a false timeline in turn 3. Returns to contradict it in turn 11. Catches the suspect. Extraction rate: 71%.

**2:30-3:00 — Why it matters:**
> "This environment trains theory-of-mind reasoning — the ability to model what another agent knows and doesn't know. That's a capability gap in current LLMs that no existing RL benchmark specifically targets."

### Key Plots to Include in README

1. **Training curve:** x=step, y=extraction_rate. Single rising line. Label the phase boundaries.
2. **Dual co-evolution curve:** x=step, two lines: interrogator ExR (rising) and suspect CoR (rising with 200-step lag).
3. **Before/after bar chart:** 4 bars: Random / Zero-shot / Phase1-trained / Phase3-selfplay. Clear progression.
4. **Sample transcript comparison:** Side-by-side, same secret. Left: untrained. Right: trained. Highlight the "callback trap" moment.

### What Separates You from Other Teams

| Most teams submit | You submit |
|---|---|
| Grid world clone | First adversarial interrogation RL env |
| Single agent | Self-play dual-agent co-evolution |
| LLM-as-judge reward | 100% deterministic verifiable reward |
| One reward function | 6 independent reward signals |
| Static task difficulty | Curriculum via `n_facts` parameter |
| Just the env | Env + training evidence + dual rising curves |

---

## 13. Simplified Build Timeline (2 Days)

### Day 1

| Time | Task | Output |
|---|---|---|
| 0-1h | `openenv init suspect_x_env`, scaffold files | Folder structure |
| 1-2h | `SecretFactory` + `secret_factory.py` | Generates secrets |
| 2-3h | `Grader` (JSON diff rewards) | `grader.py` working |
| 3-4h | `ConsistencyChecker` (keyword rules) | `consistency_checker.py` |
| 4-5h | Environment `reset/step/state` | Local server running |
| 5-6h | Rule-based suspect script | Passes grader tests |
| 6-7h | Deploy to HF Spaces (`openenv push`) | Public URL live |
| 7-8h | Zero-shot baseline evaluation (20 episodes) | Baseline ExR number |

### Day 2

| Time | Task | Output |
|---|---|---|
| 0-1h | Unsloth + TRL setup in Colab | Model loaded |
| 1-2h | `rollout_func` + reward functions | Training loop draft |
| 2-5h | Phase 1 training (interrogator, ~300 steps) | First checkpoint |
| 5-6h | Evaluate Phase 1, generate plots | Rising ExR curve |
| 6-7h | Phase 2 training (suspect, ~200 steps) | Second checkpoint |
| 7-8h | Final demo recording, README, HF blog post | Submission ready |

---

## 14. Critical Anti-Cheat Rules

These MUST be implemented or the grader will be hacked:

1. **Secret never in interrogator's context.** Only `crime_description` (the public fact) is passed to the interrogator. The full secret JSON is only in the suspect's context.

2. **Accusation JSON must be parsed, not eval'd.** Use `json.loads()` with schema validation. Reject malformed JSONs silently (0 reward, next episode).

3. **Interrogator cannot see suspect's system prompt.** Separate conversation histories. Never merge them.

4. **No hardcoded answer detection.** If `accusation["alibi"]` exactly matches the secret string character-for-character and the interrogator never asked about the alibi, flag it.

5. **Deflection guard.** If suspect says "I don't know" or "I can't recall" more than 8 times in 20 turns → reward_suspect -= 0.5 automatically.

---

## Summary

**What you're building:** A two-agent adversarial RL environment where an Interrogator LLM learns to extract hidden facts through logical trapping, and a Suspect LLM learns to maintain a consistent cover story.

**What makes it win:** 100% verifiable reward (no LLM judge), genuine self-play co-evolution, theory-of-mind reasoning (underexplored RL domain), clean curriculum via `n_facts`, spectacular before/after demo, and a rising dual-curve that tells the story visually.

**The minimum viable submission:** Phase 1 only. Interrogator trained against rule-based suspect. Extraction rate rising from ~15% to ~55%. That alone is a complete, strong, novel result.

**The winning submission:** Phase 1 + Phase 2 + dual rising curves + transcript comparison + held-out evaluation showing generalization.
