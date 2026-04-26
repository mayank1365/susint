

# OpenEnv Hackathon – Themes & Guidelines



---

## 🚩 Theme #1 – Multi-Agent Interactions

Environments involving:

* Cooperation
* Competition
* Negotiation
* Coalition formation

These environments help agents:

* Model beliefs and incentives of others
* Operate in partially observable settings
* Develop theory-of-mind reasoning
* Exhibit emergent strategic behavior

**Expected Outcome:**
An environment for training multi-agent task handling in an LLM

**Example Environments:**

* Market simulations
* Compute-allocation negotiations
* Collaborative puzzle worlds
* Mixed cooperative/competitive strategy games

---

## 🚩 Theme #2 – Long-Horizon Planning & Instruction Following

Focus on:

* Deep multi-step reasoning
* Sparse/delayed rewards
* Long-running tasks beyond context limits

Agents should learn to:

* Decompose goals
* Track long-term state
* Recover from early mistakes

**Expected Outcome:**
Environment improving long-horizon reasoning and planning

**Example Environments:**

* Research-planning simulators
* Large-scale codebase refactoring
* Strategic resource management
* Complex multi-step workflows

---

## 🚩 Theme #3 – World Modeling

### 🔹 3.1 Professional Tasks

Focus:

* Real interaction with tools, APIs, dynamic systems
* Avoid shortcut-based solutions

Agents should:

* Maintain internal state
* Update beliefs
* Execute multi-step workflows

**Expected Outcome:**
Environment simulating a complex, partially observable world

**Examples:**

* Browser/API ecosystems
* Enterprise tools
* Scientific workflows (papers → code → experiments)
* Economic simulations

---

### 🔹 3.2 Personalized Tasks

Focus:

* Real-life personal assistant scenarios

**Expected Outcome:**
Environment simulating personal task handling

**Examples:**

* Email replies
* Meeting scheduling
* Dinner planning
* Shopping assistants

---

## 🚩 Theme #4 – Self-Improvement

Focus:

* Self-play
* Adaptive curricula
* Recursive capability growth

Agents should:

* Generate new challenges
* Increase difficulty
* Improve iteratively

**Expected Outcome:**
Environment for self-improving LLM agents

**Examples:**

* Self-play negotiation systems
* Auto-generated math tasks
* Coding competitions
* Adaptive RL systems

---

## 🚩 Theme #5 – Wild Card

* Fully open-ended
* Focus on creativity and impact

**Goal:**
Build something novel that improves LLM training

---

# 🧠 Problem Statement Guidelines

* You **do not need** to reuse Round 1 problem
* Must align with one of the themes
* Pre-onsite:

  * Build environment
  * Define agent behavior
  * Design reward model
* Onsite (25–26):

  * Perform training using Hugging Face compute

---

# 🏆 Judging Criteria

### Minimum Requirements

* Use **OpenEnv (latest version)**
* Provide training script (Unsloth or HF TRL, preferably Colab)
* Create:

  * Mini-blog (Hugging Face) OR
  * <2 min YouTube video
* Host environment on **Hugging Face Spaces**

---

## 📊 Evaluation Breakdown

| Criterion                   | Weight |
| --------------------------- | ------ |
| Environment Innovation      | 40%    |
| Storytelling & Presentation | 30%    |
| Reward Improvement Evidence | 20%    |
| Reward + Training Pipeline  | 10%    |

---

# 🔍 What Judges Look For

## TL;DR

Build:

1. A meaningful environment
2. Train an LLM on it
3. Show measurable improvement
4. Explain clearly

---

## ⭐ Standout Submission Tips

### 1. Pick an Ambitious Problem

* Avoid common ideas (chess, tic-tac-toe, etc.)
* Ask:

  * Does this teach something new?
  * Is this underexplored?
  * Could it be publishable?

---

### 2. Design a Strong Reward System

* Rich signal (not just binary)
* Hard to exploit
* Use composable rubrics

---

### 3. Show Real Training

* Not just a script — actual results
* Include:

  * Reward curves
  * Before vs after comparison
  * Baselines

---

### 4. Make Plots Clear

* Label axes properly
* Save as `.png` / `.jpg`
* Compare runs on same graph
* Add captions

---

### 5. Tell a Story

Answer clearly:

* **Problem:** What gap are you solving?
* **Environment:** What happens inside?
* **Results:** What improved?
* **Impact:** Why does it matter?

---

### 6. Engineering Expectations

* Use OpenEnv properly
* Follow Gym-style API:

  * `reset()`
  * `step()`
  * `state()`
* Maintain clean structure
* Provide `openenv.yaml`
* Avoid reserved tool names

---

# 📦 Submission Requirements

* OpenEnv-based environment
* Training script (Colab preferred)
* Evidence of training (plots, metrics)
* README with:

  * Problem explanation
  * Environment details
  * Results
  * Links to:

    * HF Space
    * Blog/video
* No large files in repo (use links instead)

---

# ⚠️ Important Notes

* Only **one submission per team**
* Provide environment **URL** (used for evaluation)
* No updates allowed after submission deadline

---

# 🏁 Final Advice

* Be ambitious
* Focus on meaningful learning
* Demonstrate real improvement
* Communicate clearly

---