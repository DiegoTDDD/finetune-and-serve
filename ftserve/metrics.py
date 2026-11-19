"""Scores for predicted facts against the auditors' tags.

- fact F1: a prediction counts only if the value and the concept both match (micro-averaged
  over all facts, as multisets, so a number tagged twice must be predicted twice);
- value F1: the same, ignoring the concept (did the model find the right numbers?);
- exact match: the sentence's whole answer is right, including "no facts";
- per-concept F1, to see where models differ.
"""

from __future__ import annotations

from collections import Counter

from ftserve.finer import Fact


def norm(value: str) -> str:
    return value.strip().lstrip("$").rstrip("%").replace(",", "").strip()


def _prf(tp: int, n_pred: int, n_gold: int) -> dict[str, float]:
    p = tp / n_pred if n_pred else 1.0
    r = tp / n_gold if n_gold else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def score(gold: list[list[Fact]], pred: list[list[Fact]], valid: list[bool]) -> dict:
    if not (len(gold) == len(pred) == len(valid)):
        raise ValueError("gold, pred and valid must have the same length")
    tp = tpv = n_pred = n_gold = exact = 0
    per_tag: dict[str, list[int]] = {}
    for g, p in zip(gold, pred, strict=True):
        gc = Counter((norm(f.value), f.tag) for f in g)
        pc = Counter((norm(f.value), f.tag) for f in p)
        hit = gc & pc
        tp += sum(hit.values())
        tpv += sum((Counter(v for v, _ in gc.elements()) & Counter(v for v, _ in pc.elements())).values())
        n_pred += sum(pc.values())
        n_gold += sum(gc.values())
        exact += gc == pc
        for (_, tag), n in gc.items():
            per_tag.setdefault(tag, [0, 0, 0])[2] += n
        for (_, tag), n in pc.items():
            per_tag.setdefault(tag, [0, 0, 0])[1] += n
        for (_, tag), n in hit.items():
            per_tag[tag][0] += n
    return {
        "sentences": len(gold),
        "gold_facts": n_gold,
        "predicted_facts": n_pred,
        "fact": _prf(tp, n_pred, n_gold),
        "value": _prf(tpv, n_pred, n_gold),
        "exact_match": round(exact / len(gold), 4) if gold else 0.0,
        "valid_json": round(sum(valid) / len(valid), 4) if valid else 0.0,
        "per_tag": {t: {**_prf(a, b, c), "gold": c} for t, (a, b, c) in sorted(per_tag.items())},
    }
