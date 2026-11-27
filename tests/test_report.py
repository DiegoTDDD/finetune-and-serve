import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("report", Path(__file__).resolve().parents[1] / "scripts" / "report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def run(name, f1, shots=0, prompt=1000.0, completion=20.0):
    s = {"fact": {"f1": f1}, "value": {"f1": f1}, "exact_match": f1, "per_tag": {"T": {"f1": f1, "gold": 3}}}
    return {
        "name": name,
        "shots": shots,
        "model_bytes": 4e8,
        "scores": s,
        "latency_s": {"p50": 0.5, "p95": 0.9},
        "tokens_per_request": {"prompt": prompt, "completion": completion},
    }


PRICES = {
    "checked": "2026-11-19",
    "vm": {"name": "vm", "usd_per_hour": 0.36},
    "apis": [{"name": "api", "input_usd_per_mtok": 1.0, "output_usd_per_mtok": 5.0}],
}


def test_costs():
    runs = {"base-3b-few-shot": run("base-3b-few-shot", 0.5, 6, prompt=1000, completion=20)}
    bench = {
        "ft": {
            "levels": [{"concurrency": 1, "requests_per_second": 1.0}, {"concurrency": 4, "requests_per_second": 2.0}]
        }
    }
    c = report.costs(runs, bench, PRICES)
    # 0.36 $/h at 2 req/s = 7200 req/h -> $50 per million
    assert c["self_hosted"]["ft"] == {
        "concurrency": 4,
        "requests_per_second": 2.0,
        "usd_per_million_requests": 50.0,
        "max_requests_per_month": 2 * 3600 * 730,
    }
    # 1000 * $1/M + 20 * $5/M = $0.0011 per request -> $1100 per million; VM month $262.8 -> 238,909 requests
    (api,) = c["api"]
    assert api["usd_per_million_requests"] == pytest.approx(1100.0)
    assert api["break_even_requests_per_month"] == int(0.36 * 730 / 0.0011)


def test_markdown_orders_by_fact_f1():
    runs = {
        "a": run("a", 0.4),
        "finetuned-0.5b-Q4_K_M": run("finetuned-0.5b-Q4_K_M", 0.9),
        "base-3b-few-shot": run("base-3b-few-shot", 0.6, 6),
    }
    md = report.markdown(runs, {}, None, report.costs(runs, {}, PRICES), PRICES, {"intervals": {}, "differences": []})
    rows = [line for line in md.splitlines() if line.startswith("| ") and "-shot" in line and "MB" in line]
    assert [r.split(" | ")[0] for r in rows] == ["| finetuned-0.5b-Q4_K_M", "| base-3b-few-shot", "| a"]
    assert "## By concept" in md
