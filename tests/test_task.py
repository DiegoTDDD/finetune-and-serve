import json

from conftest import TAGS

from ftserve.finer import Fact
from ftserve.task import messages, parse, pick_shots, readable, schema, system_prompt, target


def test_readable():
    assert readable("DebtInstrumentFaceAmount") == "Debt Instrument Face Amount"
    assert readable("EffectiveIncomeTaxRateReconciliationAtFederalStatutoryIncomeTaxRate").startswith(
        "Effective Income Tax"
    )


def test_target_matches_schema_order_and_parses(examples):
    t = target(examples[0])
    assert (
        t
        == '{"facts": [{"value": "100", "tag": "DebtInstrumentFaceAmount"}, {"value": "7.00", "tag": "DebtInstrumentInterestRateStatedPercentage"}]}'
    )
    assert parse(t) == (list(examples[0].facts), True)
    assert target(examples[2]) == '{"facts": []}'
    item = schema(TAGS)["properties"]["facts"]["items"]
    assert list(item["properties"]) == ["value", "tag"] and item["properties"]["tag"]["enum"] == TAGS


def test_parse_invalid():
    assert parse("not json") == ([], False)
    assert parse('{"other": 1}') == ([], False)
    assert parse('{"facts": [{"value": 5, "tag": "X"}]}') == ([Fact("5", "X")], True)


def test_messages_and_prompt(examples):
    m = messages(examples[1], TAGS, shots=[examples[0]])
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user"]
    assert json.loads(m[2]["content"])["facts"][0]["value"] == "100"
    assert "- IncomeTaxExpenseBenefit: Income Tax Expense Benefit" in system_prompt(TAGS)


def test_pick_shots(examples):
    shots = pick_shots(examples, TAGS, 3)
    assert shots[-1].id == "t-3"  # the negative
    assert len(shots) == 3 and len({s.id for s in shots}) == 3
