"""How sure are the differences between models? A paired bootstrap over sentences.

Every model answered the same 300 test sentences, so the comparison resamples sentences (with
replacement) and recomputes both models' micro fact F1 on the same resample. The spread of the
difference over the resamples gives a 95% interval; when it excludes 0, the difference is not
explained by which sentences happened to be in the test set. Pure Python, seeded, so the report
is the same on every run.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence

from ftserve.finer import Fact
from ftserve.metrics import norm

Counts = tuple[int, int, int]  # true positives, predicted facts, gold facts


def sentence_counts(gold: Sequence[Sequence[Fact]], pred: Sequence[Sequence[Fact]]) -> list[Counts]:
    """Per sentence: (true positives, predicted, gold), as metrics.score counts them."""
    out = []
    for g, p in zip(gold, pred, strict=True):
        gc = Counter((norm(f.value), f.tag) for f in g)
        pc = Counter((norm(f.value), f.tag) for f in p)
        out.append((sum((gc & pc).values()), sum(pc.values()), sum(gc.values())))
    return out


def micro_f1(counts: Sequence[Counts]) -> float:
    tp = sum(c[0] for c in counts)
    n_pred = sum(c[1] for c in counts)
    n_gold = sum(c[2] for c in counts)
    p = tp / n_pred if n_pred else 1.0
    r = tp / n_gold if n_gold else 1.0
    return 2 * p * r / (p + r) if p + r else 0.0


def _quantiles(xs: list[float], level: float) -> tuple[float, float]:
    xs = sorted(xs)
    lo = xs[int((1 - level) / 2 * (len(xs) - 1))]
    hi = xs[int((1 + level) / 2 * (len(xs) - 1))]
    return lo, hi


def bootstrap_interval(counts: Sequence[Counts], resamples: int = 2000, level: float = 0.95,
                       seed: int = 0) -> tuple[float, float]:  # fmt: skip
    """Percentile interval of one model's micro fact F1."""
    rng = random.Random(seed)
    n = len(counts)
    stats = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(micro_f1([counts[i] for i in idx]))
    return _quantiles(stats, level)


def paired_difference(a: Sequence[Counts], b: Sequence[Counts], resamples: int = 2000,
                      level: float = 0.95, seed: int = 0) -> dict[str, float]:  # fmt: skip
    """F1(a) - F1(b) on the full set, its percentile interval over paired resamples, and the
    share of resamples where the sign flips (a two-sided bootstrap p-value)."""
    if len(a) != len(b):
        raise ValueError("both models must answer the same sentences")
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(micro_f1([a[i] for i in idx]) - micro_f1([b[i] for i in idx]))
    observed = micro_f1(a) - micro_f1(b)
    lo, hi = _quantiles(diffs, level)
    flips = sum((d <= 0) if observed > 0 else (d >= 0) for d in diffs) / resamples
    return {"difference": observed, "low": lo, "high": hi, "p_value": min(1.0, 2 * flips)}
