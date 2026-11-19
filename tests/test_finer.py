from ftserve.finer import Example, Fact, detokenize, facts, in_scope, parse_labels, tag_counts

SCRIPT = """
_DESCRIPTION = "x"
_LABELS = [
    "O",
    "B-DebtInstrumentFaceAmount",
    "I-DebtInstrumentFaceAmount",
    "B-IncomeTaxExpenseBenefit",
]
"""
LABELS = parse_labels(SCRIPT)


def test_parse_labels():
    assert LABELS == [
        "O",
        "B-DebtInstrumentFaceAmount",
        "I-DebtInstrumentFaceAmount",
        "B-IncomeTaxExpenseBenefit",
    ]


def test_detokenize():
    toks = [
        "In",
        "2019",
        ",",
        "the",
        "Company",
        "issued",
        "$",
        "100",
        "million",
        "(",
        "net",
        ")",
        "at",
        "7.00",
        "%",
        ".",
    ]
    assert detokenize(toks) == "In 2019, the Company issued $100 million (net) at 7.00%."


def test_facts_spans():
    toks = ["$", "1", ",", "234", "and", "56", "and", "78"]
    tags = [0, 1, 2, 2, 0, 3, 0, 1]
    assert facts(toks, tags, LABELS) == [
        Fact("1,234", "DebtInstrumentFaceAmount"),
        Fact("56", "IncomeTaxExpenseBenefit"),
        Fact("78", "DebtInstrumentFaceAmount"),
    ]


def test_adjacent_b_tags_are_separate_facts():
    assert facts(["5", "6"], [1, 1], LABELS) == [
        Fact("5", "DebtInstrumentFaceAmount"),
        Fact("6", "DebtInstrumentFaceAmount"),
    ]


def test_scope_and_counts(examples):
    tags = {"DebtInstrumentFaceAmount", "DebtInstrumentInterestRateStatedPercentage"}
    assert [in_scope(e, tags) for e in examples] == [True, False, True, True]
    assert tag_counts(examples)["DebtInstrumentInterestRateStatedPercentage"] == 2


def test_json_round_trip(examples):
    assert all(Example.from_json(e.to_json()) == e for e in examples)
