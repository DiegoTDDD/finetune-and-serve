"""LoRA fine-tuning on CPU.

The training text is exactly what the model sees at inference: the chat template applied to
the zero-shot prompt, followed by the target answer. Loss is computed only on the answer
tokens, and the language-model head is applied only at those positions: with a 152k-token
vocabulary, full logits for a batch would take more memory than the model itself.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from ftserve.finer import Example
from ftserve.task import messages, target

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    lr: float = 2e-4
    epochs: int = 1
    batch_size: int = 4
    grad_accum: int = 4
    warmup_frac: float = 0.05
    max_len: int = 1024
    seed: int = 0
    max_minutes: float = 300.0
    gradient_checkpointing: bool = True


def encode(tokenizer, example: Example, tags: list[str]) -> dict[str, list[int]]:
    """input_ids and labels (-100 on the prompt) for one training example."""
    msgs = messages(example, tags)
    prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    full = tokenizer.apply_chat_template([*msgs, {"role": "assistant", "content": target(example)}], tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("chat template does not extend the prompt with the answer")
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(full[len(prompt) :], add_special_tokens=False)["input_ids"]
    return {"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids}


def collate(batch: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"])
        labels[i, :k] = torch.tensor(b["labels"])
        mask[i, :k] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def answer_loss(causal_lm, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Next-token cross-entropy on answer tokens only, computing logits only there."""
    hidden = causal_lm.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    shifted = batch["labels"][:, 1:]
    keep = shifted != -100
    logits = causal_lm.lm_head(hidden[:, :-1][keep])
    return F.cross_entropy(logits.float(), shifted[keep])


def lr_at(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def eval_loss(causal_lm, encoded: list[dict], pad_id: int, batch_size: int) -> float:
    causal_lm.eval()
    total, n = 0.0, 0
    for i in range(0, len(encoded), batch_size):
        batch = collate(encoded[i : i + batch_size], pad_id)
        k = int((batch["labels"][:, 1:] != -100).sum())
        total += float(answer_loss(causal_lm, batch)) * k
        n += k
    causal_lm.train()
    return total / max(n, 1)


def train(
    model,
    tokenizer,
    train_set: list[Example],
    val_set: list[Example],
    tags: list[str],
    cfg: TrainConfig,
    out: Path,
    log=print,
) -> dict:
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    enc_train = [encode(tokenizer, e, tags) for e in train_set]
    enc_val = [encode(tokenizer, e, tags) for e in val_set]
    too_long = [e.id for e, x in zip(train_set, enc_train, strict=True) if len(x["input_ids"]) > cfg.max_len]
    if too_long:
        raise ValueError(f"{len(too_long)} training examples exceed max_len {cfg.max_len}")
    lengths = sorted(len(x["input_ids"]) for x in enc_train)
    answers = sorted(sum(t != -100 for t in x["labels"]) for x in enc_train)
    median, answer_median = lengths[len(lengths) // 2], answers[len(answers) // 2]
    log(f"tokens per example: median {median}, max {lengths[-1]}; answer median {answer_median}")

    if cfg.gradient_checkpointing:
        kwargs = {"use_reentrant": False}
        if hasattr(model, "activation_checkpointing_enable"):
            model.activation_checkpointing_enable(activation_checkpointing_kwargs=kwargs)
        else:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
    model.config.use_cache = False
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            target_modules=TARGET_MODULES,
            task_type="CAUSAL_LM",
        ),
    )
    causal_lm = peft_model.base_model.model
    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)

    steps_per_epoch = math.ceil(len(enc_train) / (cfg.batch_size * cfg.grad_accum))
    total = steps_per_epoch * cfg.epochs
    warmup = max(1, int(total * cfg.warmup_frac))
    history = {
        "val_loss": [{"step": 0, "loss": round(eval_loss(causal_lm, enc_val, pad_id, cfg.batch_size), 4)}],
        "train": [],
    }
    log(f"steps {total}, val loss before training {history['val_loss'][0]['loss']}")

    start, step, stopped_early = time.time(), 0, False
    causal_lm.train()
    for epoch in range(cfg.epochs):
        order = list(range(len(enc_train)))
        rng.shuffle(order)
        micro = [order[i : i + cfg.batch_size] for i in range(0, len(order), cfg.batch_size)]
        for i in range(0, len(micro), cfg.grad_accum):
            group = micro[i : i + cfg.grad_accum]
            batches = [collate([enc_train[j] for j in m], pad_id) for m in group]
            n_tok = sum(int((b["labels"][:, 1:] != -100).sum()) for b in batches)
            t0, loss_sum = time.time(), 0.0
            for b in batches:
                k = int((b["labels"][:, 1:] != -100).sum())
                loss = answer_loss(causal_lm, b) * (k / n_tok)
                loss.backward()
                loss_sum += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            for g in opt.param_groups:
                g["lr"] = cfg.lr * lr_at(step, total, warmup)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            history["train"].append(
                {
                    "step": step,
                    "epoch": epoch,
                    "loss": round(loss_sum, 4),
                    "seconds": round(time.time() - t0, 2),
                }
            )
            if step % 10 == 0 or step == total:
                log(f"step {step}/{total} loss {loss_sum:.4f} {time.time() - t0:.1f}s/step")
            if (time.time() - start) / 60 > cfg.max_minutes:
                stopped_early = True
                log(f"time budget of {cfg.max_minutes} min reached at step {step}")
                break
        if stopped_early:
            break

    val = eval_loss(causal_lm, enc_val, pad_id, cfg.batch_size)
    history["val_loss"].append({"step": step, "loss": round(val, 4)})
    log(f"val loss after training {val:.4f}")
    out.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(out / "adapter")
    summary = {
        "config": asdict(cfg),
        "train_examples": len(enc_train),
        "val_examples": len(enc_val),
        "steps": step,
        "planned_steps": total,
        "stopped_early": stopped_early,
        "minutes": round((time.time() - start) / 60, 1),
        "trainable_params": sum(p.numel() for p in params),
        "val_loss_before": history["val_loss"][0]["loss"],
        "val_loss_after": round(val, 4),
        "threads": torch.get_num_threads(),
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "train_history.json").write_text(json.dumps(history), encoding="utf-8")
    return summary
