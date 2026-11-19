"""The extraction task: prompt, output schema, few-shot examples and output parsing.

Every model gets the same system prompt and the same JSON schema, enforced by constrained
decoding in llama-server, so all outputs are valid JSON with a tag from the list; what differs
between models is only whether they pick the right numbers and the right concepts.
"""

from __future__ import annotations

import json
import re

from ftserve.finer import Example, Fact


def readable(tag: str) -> str:
    """DebtInstrumentFaceAmount -> Debt Instrument Face Amount."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", tag)


def system_prompt(tags: list[str]) -> str:
    lines = "\n".join(f"- {t}: {readable(t)}" for t in tags)
    return (
        "You tag numeric facts in sentences from SEC 10-K and 10-Q filings with XBRL concepts.\n"
        "Return every number in the sentence that reports one of the concepts below, written "
        "exactly as in the sentence (digits only, without $, % or units), with its concept. "
        "Numbers that report anything else, such as years, dates, counts or other concepts, "
        'are not facts. If there are none, return {"facts": []}.\n\n'
        f"Concepts:\n{lines}"
    )


def schema(tags: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "tag": {"type": "string", "enum": tags},
                    },
                    "required": ["value", "tag"],
                },
            }
        },
        "required": ["facts"],
    }


def target(example: Example) -> str:
    """The answer a model should produce, in the key order the schema's grammar enforces."""
    return json.dumps(
        {"facts": [{"value": f.value, "tag": f.tag} for f in example.facts]},
        ensure_ascii=False,
        separators=(", ", ": "),
    )


def messages(example: Example, tags: list[str], shots: list[Example] | None = None) -> list[dict[str, str]]:
    out = [{"role": "system", "content": system_prompt(tags)}]
    for s in shots or []:
        out += [
            {"role": "user", "content": s.text},
            {"role": "assistant", "content": target(s)},
        ]
    out.append({"role": "user", "content": example.text})
    return out


def parse(output: str) -> tuple[list[Fact], bool]:
    """Facts from a model's answer, and whether the answer was valid."""
    try:
        data = json.loads(output)
        return [Fact(str(f["value"]), str(f["tag"])) for f in data["facts"]], True
    except (json.JSONDecodeError, KeyError, TypeError):
        return [], False


def pick_shots(pool: list[Example], tags: list[str], n: int) -> list[Example]:
    """Few-shot examples from the validation split: one sentence per most frequent concept
    (shortest first, so the prompt stays small), plus one sentence with no facts."""
    shots: list[Example] = []
    by_len = sorted(pool, key=lambda e: (len(e.text), e.id))
    for tag in tags:
        if len(shots) >= n - 1:
            break
        for e in by_len:
            if e not in shots and e.facts and all(f.tag == tag for f in e.facts):
                shots.append(e)
                break
    negative = next((e for e in by_len if not e.facts and len(e.text) >= 60), None)
    if negative:
        shots.append(negative)
    return shots
