"""Run one model on the test set through llama-server and score it.

    python scripts/evaluate.py --name finetuned-q4 --model build/gguf/finetuned-Q4_K_M.gguf \
        --server llama.cpp/build/bin/llama-server [--shots 6] [--limit 20]

Writes reports/runs/<name>.json (scores, latency, tokens) and <name>.predictions.jsonl.
Requests are sent one at a time, so the latency figures are single-user latency.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ftserve.finer import Example  # noqa: E402
from ftserve.metrics import score  # noqa: E402
from ftserve.server import serve  # noqa: E402
from ftserve.task import messages, parse, pick_shots, schema  # noqa: E402


def load(path: Path) -> list[Example]:
    return [Example.from_json(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--server", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "runs")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    tags = json.loads((args.data / "tags.json").read_text(encoding="utf-8"))
    test = load(args.data / "test.jsonl")[: args.limit]
    shots = pick_shots(load(args.data / "val.jsonl"), tags, args.shots) if args.shots else []
    sch = schema(tags)
    args.out.mkdir(parents=True, exist_ok=True)

    gold, pred, valid, rows = [], [], [], []
    with serve(args.server, args.model, threads=args.threads, log_file=args.out / f"{args.name}.server.log") as client:
        client.chat(messages(test[0], tags, shots), sch)  # warm-up: load weights, cache the shared prefix
        warmup_raw = {k: v for k, v in (client.last_raw or {}).items() if k in ("usage", "timings")}
        for i, ex in enumerate(test):
            c = client.chat(messages(ex, tags, shots), sch)
            facts, ok = parse(c.text)
            gold.append(list(ex.facts))
            pred.append(facts)
            valid.append(ok)
            rows.append({"id": ex.id, "output": c.text, "seconds": round(c.seconds, 3),
                         "prompt_tokens": c.prompt_tokens, "cached_tokens": c.cached_tokens,
                         "completion_tokens": c.completion_tokens})  # fmt: skip
            if (i + 1) % 25 == 0:
                print(f"{args.name}: {i + 1}/{len(test)}", flush=True)

    secs = [r["seconds"] for r in rows]
    result = {
        "name": args.name,
        "model_file": args.model.name,
        "model_bytes": args.model.stat().st_size,
        "shots": len(shots),
        "shot_ids": [s.id for s in shots],
        "threads": args.threads,
        "scores": score(gold, pred, valid),
        "latency_s": {
            "mean": round(statistics.mean(secs), 3),
            "p50": round(pct(secs, 0.5), 3),
            "p95": round(pct(secs, 0.95), 3),
        },
        "tokens_per_request": {
            "prompt": round(statistics.mean(r["prompt_tokens"] for r in rows), 1),
            "prompt_not_cached": round(statistics.mean(r["prompt_tokens"] - r["cached_tokens"] for r in rows), 1),
            "completion": round(statistics.mean(r["completion_tokens"] for r in rows), 1),
        },
    }
    result["warmup_response_usage"] = warmup_raw
    (args.out / f"{args.name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (args.out / f"{args.name}.predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    s = result["scores"]
    print(f"{args.name}: fact F1 {s['fact']['f1']} value F1 {s['value']['f1']} exact {s['exact_match']} "
          f"p50 {result['latency_s']['p50']}s")  # fmt: skip


if __name__ == "__main__":
    main()
