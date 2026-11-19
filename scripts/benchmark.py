"""Throughput of one model under concurrent load.

    python scripts/benchmark.py --name finetuned-q4 --model build/gguf/finetuned-Q4_K_M.gguf \
        --server llama.cpp/build/bin/llama-server --concurrency 1 4 --requests 100

For each concurrency level the server runs that many slots (sharing the same CPU threads) and
a client keeps that many requests in flight. Writes reports/bench/<name>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ftserve.finer import Example  # noqa: E402
from ftserve.server import serve  # noqa: E402
from ftserve.task import messages, pick_shots, schema  # noqa: E402


def load(path: Path) -> list[Example]:
    return [Example.from_json(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--server", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "bench")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--requests", type=int, default=100)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    tags = json.loads((args.data / "tags.json").read_text(encoding="utf-8"))
    test = load(args.data / "test.jsonl")[: args.requests]
    shots = pick_shots(load(args.data / "val.jsonl"), tags, args.shots) if args.shots else []
    sch = schema(tags)
    args.out.mkdir(parents=True, exist_ok=True)

    levels = []
    for n in args.concurrency:
        with serve(args.server, args.model, threads=args.threads, parallel=n) as client:
            for _ in range(n):  # warm every slot's prompt cache
                client.chat(messages(test[0], tags, shots), sch)
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=n) as pool:
                done = list(pool.map(lambda ex: client.chat(messages(ex, tags, shots), sch), test))
            wall = time.perf_counter() - t0
        lat = sorted(c.seconds for c in done)
        out_tok = sum(c.completion_tokens for c in done)
        levels.append(
            {
                "concurrency": n,
                "requests": len(done),
                "wall_seconds": round(wall, 2),
                "requests_per_second": round(len(done) / wall, 3),
                "output_tokens_per_second": round(out_tok / wall, 1),
                "latency_p50_s": round(statistics.median(lat), 3),
                "latency_p95_s": round(lat[min(len(lat) - 1, round(0.95 * (len(lat) - 1)))], 3),
            }
        )
        print(json.dumps(levels[-1]), flush=True)
    result = {"name": args.name, "model_file": args.model.name, "shots": len(shots),
              "threads": args.threads, "levels": levels}  # fmt: skip
    (args.out / f"{args.name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
