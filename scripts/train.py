"""Fine-tune the base model with LoRA on the training set, then merge the adapter.

    python scripts/train.py --out build/finetuned [--limit 32 --max-minutes 10]

Writes <out>/adapter (LoRA weights), <out>/merged (full model for GGUF conversion),
<out>/train_summary.json and <out>/train_history.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ftserve.finer import Example  # noqa: E402
from ftserve.train import TrainConfig, train  # noqa: E402

BASE = "Qwen/Qwen2.5-0.5B-Instruct"


def load(path: Path, limit: int | None = None) -> list[Example]:
    rows = [Example.from_json(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]
    return rows[:limit] if limit else rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "finetuned")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--revision", default=os.environ.get("BASE_REVISION", "main"))
    ap.add_argument("--limit", type=int, default=None, help="use only the first N training rows")
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-minutes", type=float, default=300.0)
    args = ap.parse_args()

    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(os.cpu_count() or 4)
    revision = HfApi().model_info(args.base, revision=args.revision).sha
    tokenizer = AutoTokenizer.from_pretrained(args.base, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(args.base, revision=revision, dtype=torch.float32)

    tags = json.loads((args.data / "tags.json").read_text(encoding="utf-8"))
    cfg = TrainConfig(epochs=args.epochs, max_minutes=args.max_minutes)
    summary = train(
        model,
        tokenizer,
        load(args.data / "train.jsonl", args.limit),
        load(args.data / "val.jsonl", args.val_limit),
        tags,
        cfg,
        args.out,
        log=lambda m: print(m, flush=True),
    )
    summary["base_model"] = args.base
    summary["base_revision"] = revision
    (args.out / "train_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(args.base, revision=revision, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, args.out / "adapter").merge_and_unload()
    merged.to(torch.bfloat16).save_pretrained(args.out / "merged")
    tokenizer.save_pretrained(args.out / "merged")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
