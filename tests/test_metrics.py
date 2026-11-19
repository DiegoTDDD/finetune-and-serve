import pytest

from ftserve.finer import Fact
from ftserve.metrics import norm, score

A, B = "DebtInstrumentFaceAmount", "IncomeTaxExpenseBenefit"


def test_norm():
    assert norm(" $1,234 ") == "1234" and norm("7.00%") == "7.00"


def test_perfect_and_negatives():
    s = score([[Fact("1", A)], []], [[Fact("1", A)], []], [True, True])
    assert s["fact"]["f1"] == 1.0 and s["exact_match"] == 1.0 and s["valid_json"] == 1.0


def test_wrong_tag_counts_for_value_not_fact():
    s = score([[Fact("1", A)]], [[Fact("1", B)]], [True])
    assert s["fact"]["f1"] == 0.0 and s["value"]["f1"] == 1.0 and s["exact_match"] == 0.0


def test_multiset_and_partial():
    gold = [[Fact("5", A), Fact("5", A), Fact("9", B)]]
    pred = [[Fact("5", A), Fact("2", B)]]
    s = score(gold, pred, [True])
    assert s["fact"]["precision"] == 0.5 and s["fact"]["recall"] == pytest.approx(0.3333, abs=1e-4)
    assert s["per_tag"][A] == {
        "precision": 1.0,
        "recall": 0.5,
        "f1": pytest.approx(0.6667, abs=1e-4),
        "gold": 2,
    }


def test_false_positive_on_negative_sentence():
    s = score([[]], [[Fact("2019", A)]], [True])
    assert s["fact"]["precision"] == 0.0 and s["exact_match"] == 0.0


def test_length_mismatch():
    with pytest.raises(ValueError):
        score([[]], [], [])
