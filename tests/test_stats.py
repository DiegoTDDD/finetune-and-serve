"""The paired bootstrap: intervals, and differences that are or are not significant."""

import pytest

from ftserve.finer import Fact
from ftserve.stats import bootstrap_interval, micro_f1, paired_difference, sentence_counts

GOLD = [[Fact("345", "DebtInstrumentFaceAmount")], [Fact("20", "EquityMethodInvestmentOwnershipPercentage")]] * 50


def test_counts_and_f1_match_the_metric():
    pred = [[Fact("$345", "DebtInstrumentFaceAmount")], []] * 50  # "$345" normalises to "345"
    counts = sentence_counts(GOLD, pred)
    assert counts[:2] == [(1, 1, 1), (0, 0, 1)]
    assert micro_f1(counts) == pytest.approx(2 * 1.0 * 0.5 / 1.5)


def test_interval_contains_the_estimate():
    pred = [g if i % 3 else [] for i, g in enumerate(GOLD)]
    counts = sentence_counts(GOLD, pred)
    lo, hi = bootstrap_interval(counts, resamples=500)
    assert lo <= micro_f1(counts) <= hi and hi - lo > 0


def test_a_clear_gap_is_significant_and_a_tie_is_not():
    perfect = sentence_counts(GOLD, GOLD)
    half = sentence_counts(GOLD, [g if i % 2 else [] for i, g in enumerate(GOLD)])
    clear = paired_difference(perfect, half, resamples=500)
    assert clear["low"] > 0 and clear["p_value"] < 0.01
    tie = paired_difference(perfect, perfect, resamples=500)
    assert tie["difference"] == 0 and tie["low"] <= 0 <= tie["high"]


def test_paired_needs_the_same_sentences():
    with pytest.raises(ValueError):
        paired_difference([(1, 1, 1)], [(1, 1, 1), (0, 0, 1)])
