"""GRPO training entrypoint for the Train-on-Boot Space.

Layout choice: we don't try to be clever about T4 vs A100 here. We always
configure for the smallest viable card (T4 small: 16GB, no bf16). On a
larger card the same config still works — it's just slower than necessary.

The function `run_training(out_dir, status_writer, eval_log_path)` is
called from app.py in a background thread. It writes:
  - status.json updates after every logging step (via status_writer)
  - eval_log.jsonl rows at step 0 / 20 / 40 / 60 / 80 / final
  - lora_final/ adapter dir
  - reward_curve.png
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Callable

# These imports happen at module import time, BUT module is only imported
# from the background thread (see app.py), so the FastAPI server can boot
# even if torch fails to install at runtime.


def run_training(
    out_dir: str,
    status_writer: Callable[..., None],
    eval_log_path: str,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eval_log = Path(eval_log_path)

    status_writer(state="loading_torch")
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import GRPOConfig, GRPOTrainer

    from server.secret_factory import SecretFactory
    from training.prompts import interrogator_system_prompt
    from training.rollout import run_episode
    from training.rule_based_suspect import RuleBasedSuspect

    # ---- model + tokenizer ----
    status_writer(state="loading_model")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False  # required for gradient checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.train()
    status_writer(
        state="model_ready",
        device=str(model.device),
        bf16_supported=use_bf16,
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # ---- data ----
    factory = SecretFactory()
    train_crimes = factory.split("train")
    heldout_crimes = factory.split("heldout")[:10]   # A100: 10 heldout per eval

    train_ds = Dataset.from_list([
        {"crime_id": s.crime_id, "prompt": interrogator_system_prompt(s)}
        for s in train_crimes
    ])

    # ---- interrogator inference fn ----
    GEN_KWARGS = dict(
        max_new_tokens=160,        # A100: room for accusation block + question
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    def qwen_interrogator(messages):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, **GEN_KWARGS)
        text = tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return text

    # ---- reward function (full episode rollout per generation) ----
    def reward_episode(prompts, completions, crime_id=None, **kwargs):
        rewards = []
        crime_ids = crime_id if isinstance(crime_id, list) else [crime_id] * len(completions)
        for cid, _completion in zip(crime_ids, completions):
            secret = factory.get(cid)
            ep = run_episode(
                secret=secret,
                interrogator_fn=qwen_interrogator,
                suspect_fn=RuleBasedSuspect(secret, seed=0),
                max_turns=20,
            )
            rewards.append(float(ep.grade.interrogator_reward))
        return rewards

    # ---- periodic heldout eval ----
    def _eval_now(label: str):
        extr, rew = [], []
        for i, secret in enumerate(heldout_crimes):
            ep = run_episode(
                secret=secret,
                interrogator_fn=qwen_interrogator,
                suspect_fn=RuleBasedSuspect(secret, seed=i),
            )
            extr.append(ep.grade.extraction_score)
            rew.append(ep.grade.interrogator_reward)
        row = {
            "label": label,
            "mean_extr": statistics.mean(extr),
            "mean_reward": statistics.mean(rew),
        }
        with open(eval_log, "a") as f:
            f.write(json.dumps(row) + "\n")
        status_writer(latest_eval=row)
        return row

    class HeldoutEvalCallback(TrainerCallback):
        def __init__(self, every: int = 20):
            self.every = every

        def on_step_end(self, args, state, control, **kwargs):
            status_writer(step=int(state.global_step), max_steps=int(args.max_steps))
            if state.global_step > 0 and state.global_step % self.every == 0:
                _eval_now(f"step_{state.global_step}")

    # Step-0 (pre-training) eval anchor.
    if not eval_log.exists():
        _eval_now("step_0")

    # ---- GRPO config (A100 profile) ----
    grpo_config = GRPOConfig(
        output_dir=str(out / "checkpoints"),
        num_generations=4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_prompt_length=1024,
        max_completion_length=160,
        learning_rate=5e-6,
        beta=0.04,
        max_steps=150,
        logging_steps=5,
        save_steps=50,
        report_to="none",
        warmup_ratio=0.1,
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_episode],
        args=grpo_config,
        train_dataset=train_ds,
        callbacks=[HeldoutEvalCallback(every=25)],
    )
    status_writer(state="training", step=0, max_steps=grpo_config.max_steps)
    trainer.train()
    _eval_now("step_final")

    # ---- persist artefacts ----
    status_writer(state="saving")
    model.save_pretrained(str(out / "lora_final"))
    tokenizer.save_pretrained(str(out / "lora_final"))

    # ---- plot ----
    status_writer(state="plotting")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = trainer.state.log_history
    steps = [e["step"] for e in log if "reward" in e]
    rewards = [e["reward"] for e in log if "reward" in e]

    rows = [json.loads(l) for l in eval_log.read_text().splitlines() if l.strip()]
    eval_x = [
        int(r["label"].split("_")[1]) if r["label"] != "step_final" else grpo_config.max_steps
        for r in rows
    ]
    eval_y = [r["mean_extr"] for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].plot(steps, rewards, color="tab:blue")
    ax[0].axhline(0.117, color="gray", ls="--", label="template baseline")
    ax[0].set_xlabel("GRPO step"); ax[0].set_ylabel("mean group reward")
    ax[0].set_title("Phase 1 — interrogator training reward")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(eval_x, eval_y, marker="o", color="tab:green", label="trained Qwen (heldout)")
    ax[1].axhline(0.111, color="gray", ls="--", label="template 11.1%")
    ax[1].axhline(0.000, color="lightgray", ls=":", label="random 0%")
    ax[1].set_xlabel("GRPO step"); ax[1].set_ylabel("mean extraction (heldout)")
    ax[1].set_title(f"Heldout extraction (n={len(heldout_crimes)})")
    ax[1].set_ylim(-0.02, 1.0); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "reward_curve.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # CLI fallback for local tests.
    def _print(**kw):
        print("STATUS:", json.dumps(kw))
    run_training("./out", _print, "./out/eval_log.jsonl")
