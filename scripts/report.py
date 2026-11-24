"""Build reports/results.md and reports/results.json from the pipeline outputs.

    python scripts/report.py

Reads reports/runs/*.json (quality and single-user latency), reports/bench/*.json (throughput),
reports/train_summary.json and costs.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"
FINETUNED = "finetuned-0.5b-Q4_K_M"
HOURS_PER_MONTH = 730


def load_dir(path: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(path.glob("*.json"))}


def best_level(bench: dict) -> dict:
    return max(bench["levels"], key=lambda lv: lv["requests_per_second"])


def costs(runs: dict, bench: dict, prices: dict) -> dict:
    """Self-hosted: VM price / measured throughput. API: measured token counts x list prices.
    Token counts come from the few-shot prompt (what an API model without fine-tuning needs);
    the API models' tokenizers differ from Qwen's, so these are estimates."""
    vm = prices["vm"]["usd_per_hour"]
    self_hosted = {}
    for name, b in bench.items():
        lv = best_level(b)
        self_hosted[name] = {
            "concurrency": lv["concurrency"],
            "requests_per_second": lv["requests_per_second"],
            "usd_per_million_requests": round(vm / (lv["requests_per_second"] * 3600) * 1e6, 2),
            "max_requests_per_month": int(lv["requests_per_second"] * 3600 * HOURS_PER_MONTH),
        }
    tokens = {k: runs[k]["tokens_per_request"] for k in ("base-3b-few-shot", "base-3b-zero-shot") if k in runs}
    api = []
    for a in prices["apis"]:
        for prompt, t in tokens.items():
            per_request = (t["prompt"] * a["input_usd_per_mtok"] + t["completion"] * a["output_usd_per_mtok"]) / 1e6
            api.append(
                {
                    "api": a["name"],
                    "prompt": "few-shot" if "few" in prompt else "zero-shot",
                    "usd_per_million_requests": round(per_request * 1e6, 2),
                    "break_even_requests_per_month": int(vm * HOURS_PER_MONTH / per_request) if per_request else None,
                }
            )
    return {"vm_usd_per_month": round(vm * HOURS_PER_MONTH, 2), "self_hosted": self_hosted, "api": api}


def fmt_int(n: int | None) -> str:
    return "-" if n is None else f"{n:,}"


def markdown(runs: dict, bench: dict, train: dict | None, cost: dict, prices: dict) -> str:
    order = sorted(runs.values(), key=lambda r: -r["scores"]["fact"]["f1"])
    out = [
        "# Results",
        "",
        "Test set: 300 sentences from 10-K and 10-Q notes (FiNER-139 test split), 362 tagged facts, 60 "
        "sentences without facts. Every model runs in llama-server on the same 4-core CPU runner with the "
        "same JSON schema enforced by constrained decoding.",
        "",
        "## Quality and single-user latency",
        "",
        "| Model | Prompt | File | Fact F1 | Value F1 | Exact match | p50 latency | p95 latency |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in order:
        s = r["scores"]
        prompt = f"{r['shots']}-shot" if r["shots"] else "zero-shot"
        out.append(
            f"| {r['name']} | {prompt} | {r['model_bytes'] / 1e6:,.0f} MB | **{s['fact']['f1']:.3f}** | "
            f"{s['value']['f1']:.3f} | {s['exact_match']:.3f} | {r['latency_s']['p50']:.2f} s | "
            f"{r['latency_s']['p95']:.2f} s |"
        )
    out += [
        "",
        "Fact F1 counts a prediction only when both the number and its XBRL concept match the auditors' tag; "
        "value F1 ignores the concept; exact match is the share of sentences answered completely right.",
    ]
    if FINETUNED in runs and "base-3b-few-shot" in runs:
        ft, ref = runs[FINETUNED]["scores"]["per_tag"], runs["base-3b-few-shot"]["scores"]["per_tag"]
        out += [
            "",
            "## By concept (F1)",
            "",
            f"| Concept | Test facts | {FINETUNED} | base-3b-few-shot |",
            "| --- | --- | --- | --- |",
        ]
        for tag in sorted(ft, key=lambda t: -ft[t]["gold"]):
            if ft[tag]["gold"]:
                out.append(f"| {tag} | {ft[tag]['gold']} | {ft[tag]['f1']:.2f} | {ref.get(tag, {}).get('f1', 0):.2f} |")
    if bench:
        out += [
            "",
            "## Throughput",
            "",
            "| Model | Concurrent requests | Requests/s | Output tokens/s | p50 latency | p95 latency |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name, b in bench.items():
            for lv in b["levels"]:
                out.append(
                    f"| {name} | {lv['concurrency']} | {lv['requests_per_second']:.2f} | "
                    f"{lv['output_tokens_per_second']:.0f} | {lv['latency_p50_s']:.2f} s | "
                    f"{lv['latency_p95_s']:.2f} s |"
                )
        out += [
            "",
            "## Cost",
            "",
            f"Self-hosted on {prices['vm']['name']} at ${prices['vm']['usd_per_hour']}/hour "
            f"(${cost['vm_usd_per_month']:,.0f}/month always on), at the best measured throughput:",
            "",
            "| Self-hosted model | Requests/s | USD per million requests | Capacity per month |",
            "| --- | --- | --- | --- |",
        ]
        for name, c in cost["self_hosted"].items():
            out.append(
                f"| {name} | {c['requests_per_second']:.2f} | ${c['usd_per_million_requests']:,.2f} | "
                f"{fmt_int(c['max_requests_per_month'])} |"
            )
        out += [
            "",
            "Commercial APIs at list prices (checked " + prices["checked"] + "), with the prompt sizes measured above "
            "(token counts from Qwen's tokenizer, so estimates):",
            "",
            "| API | Prompt | USD per million requests | Requests/month where the VM becomes cheaper |",
            "| --- | --- | --- | --- |",
        ]
        for a in cost["api"]:
            out.append(
                f"| {a['api']} | {a['prompt']} | ${a['usd_per_million_requests']:,.2f} | "
                f"{fmt_int(a['break_even_requests_per_month'])} |"
            )
        out += ["", "The APIs' accuracy on this task was not measured (no paid key is used in this project)."]
    if train:
        per_step = train["config"]["batch_size"] * train["config"]["grad_accum"]
        seen_n = min(train["steps"] * per_step, train["train_examples"])
        seen = f"{train['steps']} optimizer steps of {per_step} examples ({seen_n:,} of {train['train_examples']:,})"
        if train["stopped_early"]:
            seen = (
                f"The {train['config']['max_minutes']:.0f}-minute budget stopped training at step {train['steps']} "
                f"of {train['planned_steps']}: {seen_n:,} of the {train['train_examples']:,} examples were seen"
            )
        out += [
            "",
            "## Training",
            "",
            f"LoRA rank {train['config']['rank']} on all attention and MLP projections of {train['base_model']} "
            f"({train['trainable_params'] / 1e6:.1f}M trainable parameters), {train['minutes']} minutes on "
            f"{train['threads']} CPU threads. {seen}. Validation loss on answer tokens: "
            f"{train['val_loss_before']} before, {train['val_loss_after']} after.",
        ]
    return "\n".join(out) + "\n"


def main() -> None:
    runs = load_dir(REPORTS / "runs")
    bench = load_dir(REPORTS / "bench") if (REPORTS / "bench").exists() else {}
    train_path = REPORTS / "train_summary.json"
    train = json.loads(train_path.read_text(encoding="utf-8")) if train_path.exists() else None
    prices = json.loads((ROOT / "costs.json").read_text(encoding="utf-8"))
    cost = costs(runs, bench, prices)
    (REPORTS / "results.md").write_text(markdown(runs, bench, train, cost, prices), encoding="utf-8")
    summary = {
        "runs": {
            k: {"scores": {x: v["scores"][x] for x in ("fact", "value", "exact_match")}, "latency_s": v["latency_s"]}
            for k, v in runs.items()
        },
        "costs": cost,
    }
    (REPORTS / "results.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print((REPORTS / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
