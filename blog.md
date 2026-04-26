# I Taught Two AIs to Lie to Each Other. One Got Really Good at It.

*A story about interrogation rooms, adversarial AI, and what happens when reinforcement learning meets deception.*

---

It started, as most bad ideas do, at 1am.

We were sitting with a pile of hackathon documents, a cold cup of coffee, and the very reasonable question: *what if we built something that's never been built before?*

The document in front of us listed all the approved themes. Multi-agent cooperation. Long-horizon planning. World modeling. All perfectly respectable. All, frankly, a little boring.

And then someone said it:

> "What if one AI tried to interrogate another AI — and the other one tried to lie its way out?"

We looked at each other. We looked at the cold coffee. We decided this was worth staying up for.

---

## The Room

Picture a room. Small. One light overhead. Two chairs.

In one chair: a detective. Sharp eyes, sharper questions. Has seen every trick in the book and written a few of his own.

In the other chair: a suspect. Cool composure. A cover story rehearsed to perfection. Not going to give anything away without a fight.

This is the setup that has powered a thousand crime films, ten thousand novels, and at least one Emmy-winning TV show you definitely binged in a weekend. It is a deeply human drama — the contest between truth-seeking and truth-concealing, played out turn by turn, question by answer.

We wanted to give that drama to two language models and watch what happened.

---

## Why This Is Actually Hard

Here's the thing nobody tells you about LLMs: they already know what an interrogation *looks like*. Ask GPT-4 to roleplay a detective and it will produce something cinematic and competent. The detective will ask good questions. The suspect will dodge them artfully. It will read like a script.

But it won't be *strategic*.

A human detective doesn't ask questions randomly. They build a mental model of the suspect's cover story, probe for internal inconsistencies, set timeline traps, bluff about evidence they don't have. The optimal next question depends entirely on what the suspect just said — and what they've *been* saying for the last ten minutes.

That's not something you can prompt your way into. That's something that has to be *learned*, through thousands of attempts, against an adversary who is simultaneously learning to resist you.

That's reinforcement learning. That's exactly what we built.

---

## The Architecture (The Non-Boring Version)

Let's talk about how this actually works, because it's more elegant than it might sound.

We took one model — **Qwen2.5-7B-Instruct**, a capable 7-billion-parameter language model — and split its personality in two. Not by copying it (that would cost twice the GPU memory and twice the sanity). Instead, we used a technique called **QLoRA**: attach a small set of adapter weights on top of the frozen base model, one set for the Interrogator personality and one for the Suspect.

Think of it like this. The base model is a blank-faced actor. The LoRA adapters are the character notes the director hands them before they walk on stage. Same actor. Two completely different performances.

The environment itself was built on **OpenEnv**, a framework for standardizing how RL environments talk to training systems. Every episode follows a clean protocol:

```
reset()  → a new crime is generated, a new secret is born
step()   → a question is asked, a lie is told, a trap is set
state()  → here's what's happened so far
done     → 20 turns. Time to submit your accusation.
```

The Interrogator never sees the hidden case file. It only sees the crime description and whatever the Suspect has said so far. The Suspect sees everything — the full truth — and must carefully dance around it.

At the end of every episode, a grader scores both agents. Not a language model judge (those can be fooled). A deterministic Python function that compares the Interrogator's final accusation against the actual hidden facts, field by field, word by word. Clean. Objective. Unfakeable.

---

## The Training: Where It Gets Interesting

Here's the part that genuinely surprised us.

We started with a scripted suspect — basically a Python function that always said the same thing. *"I was at the dinner party. I didn't go anywhere. I don't know what you're talking about."* A stubborn brick wall.

Against this wall, we ran the Interrogator through **GRPO** — Group Relative Policy Optimization. For each case file, we generated eight different interrogation attempts simultaneously. Same facts hidden. Same suspect script. Eight different strategies the Interrogator tried.

Most of them failed. A few didn't.

The ones that worked — the ones where the Interrogator happened to ask about the timeline first, then circle back with a contradiction two questions later — those got their probability *increased* in the model's weights. The ones that just asked "where were you?" in ten different ways got their probability *decreased*.

After five hundred of these cycles, something shifted.

The Interrogator started setting traps.

Not because we told it to. Not because we wrote a prompt saying "use timeline contradictions." It discovered that asking about two events in sequence — events it knew the Suspect would have to reconcile — was consistently more rewarding than direct accusations. It developed a *strategy*, on its own, from reward signal alone.

Then we switched. We froze the Interrogator at that checkpoint and started training the Suspect against it.

The Suspect, facing an Interrogator that now knew how to exploit timeline inconsistencies, adapted. It stopped giving specific times. It introduced deliberate vagueness. It learned to redirect — to ask clarifying questions of its *own* when cornered, buying turns, maintaining the alibi's structural integrity without committing to easily-falsifiable details.

And then we set them against each other.

The extraction rate — the percentage of hidden facts the Interrogator correctly identified — started at around 30%. By the end of alternating self-play, it sat at 65%. The Suspect's consistency score improved too. Both agents, sharpened against each other, got better simultaneously.

We had created a little adversarial ecosystem in a GPU.

---

## The Moment We Knew It Was Working

There's a specific episode we kept coming back to.

Case file: corporate espionage. A data engineer had allegedly exfiltrated proprietary source code on a Thursday night. The hidden fact was that they had entered the building at 11:47 PM via a side entrance that wasn't covered by standard badge logs.

**Baseline Interrogator (no training):**
> "Did you enter the building on Thursday?"
> "No."
> "Are you sure?"
> "Yes."
> *Extraction rate: 1/5 facts.*

**Trained Interrogator:**
> "Walk me through your Thursday evening. What did you do after 9pm?"
> "I was home. Ordered food, watched something."
> "Right. So you wouldn't have any reason to use a side entrance on the east wing?"
> *[Suspect pauses in its generation — the model briefly calculates. The question is oddly specific.]*
> "I... don't know what entrance you're talking about."
> "Interesting. Because I didn't say which entrance."

That wasn't scripted. That wasn't in any training example. The Interrogator invented that move. It had learned to imply knowledge it didn't have — to bluff — because bluffing had historically produced reactions that led to higher extraction rates.

It was, genuinely, a little eerie.

---

## The Grader: Why We Did It Without an LLM Judge

Almost every dialogue-based RL project uses an LLM as the judge. It feels natural — language is qualitative, so evaluate it qualitatively, right?

We didn't. And this was one of our best decisions.

The problem with LLM judges is that they become part of the optimization target. If the Interrogator learns that a judge gives higher scores to responses with confident, declarative language — it will produce confident, declarative language, regardless of whether it's actually extracted facts. You've optimized the judge, not the task.

Our grader is forty lines of Python. It takes the Interrogator's final accusation JSON, compares each claim against the hidden case file using Jaccard keyword overlap, counts matches, counts invented facts, applies the efficiency multiplier. No ambiguity. No way to charm it.

The Suspect's consistency score is even simpler — a pre-computed constraint graph, generated when the case file is created, checked against every Suspect response in O(n) time. If you said "I was home" and then said "I left at 10," that's a contradiction. The checker finds it in milliseconds.

This matters beyond elegance. It means the reward curves we show are *real*. When extraction rate goes from 30% to 65%, that means facts were actually extracted. The number cannot be inflated by a sympathetic judge.

---

## What We Learned About Learning

The thing that surprised us most was how *fast* the co-evolution kicked in.

We expected a gentle hill-climb. What we got was more like a ratchet — each improvement in one agent immediately creating pressure on the other. The Suspect didn't need a hundred episodes to notice that the Interrogator had gotten better at timeline questions. Within a training phase, the Suspect's vagueness strategy was already emerging.

This is the property that makes adversarial self-play so powerful and so strange. You don't need a curriculum designer. You don't need to carefully sequence difficulty. The curriculum *is* your opponent. As they improve, you have to improve to keep up.

The second thing we learned is that mode collapse is real and annoying. Midway through Phase 4, the Suspect decided the optimal strategy was to just... say almost nothing. Two-word responses. Technically consistent (you can't contradict yourself if you never say anything specific). The grader was correctly giving it a middling consistency score but a high concealment score.

We fixed this with a minimum response length gate — under ten tokens, the episode reward for the Suspect gets zeroed. The Suspect's policy quickly learned that strategic verbosity outperformed strategic silence.

Small interventions. Big effects. This is the constant texture of building RL environments.

---

## Ship It

After four days — and more GPU hours than we're comfortable admitting — we pushed everything to Hugging Face.

The Space lets you pick a scenario, watch the conversation unfold in a chat interface, and then reveal the hidden truth at the end. You can compare the baseline model (no training) against the RL-trained version on the same case file. The difference is visible in about thirty seconds of reading.

The training notebook is in the repo. The reward curves are in the README. The adapter weights are on the Hub. Everything is reproducible.

---

## Why This Matters Beyond the Hackathon

We built this as a hackathon project, but the underlying problem is real and growing.

As AI agents take on more complex roles — negotiation, investigation, customer service, legal reasoning — they will encounter other agents, human or artificial, who have incentives to be incomplete, misleading, or strategically vague. An agent that can only accept input at face value is not ready for that world.

Training on adversarial interrogation is training on *skeptical reasoning* — the capacity to hold a model of what someone might be hiding, update that model based on what they say and how they say it, and probe strategically toward the truth.

That's not a party trick. That's a capability that will matter.

---

## Try It

The environment is live. The weights are public. The Colab notebook will run in under four hours on an A100.

If you run it and the trained Interrogator sets a trap you didn't expect — screenshot it. We'd love to see it.

And if the Suspect talks its way out of a corner with something clever?

We'd love to see that too.

---

*Built with OpenEnv · Qwen2.5-7B-Instruct · HuggingFace TRL · Unsloth · GRPO*

*[HuggingFace Space](https://huggingface.co/spaces/Hollow-Abyss/susint) · [GitHub Repo](https://github.com/mayank1365/susint) · [Training Notebook](https://colab.research.google.com/drive/1r24QyIUSgIbeboY2ngTqF1NhhjrhWMyI?usp=sharing) 