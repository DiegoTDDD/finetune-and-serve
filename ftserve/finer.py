"""FiNER-139: sentences from 10-K and 10-Q notes, numeric tokens tagged by auditors with XBRL
concepts (Loukas et al., ACL 2022; CC BY-SA 4.0).

The raw release is a zip of JSON Lines files ({"id", "tokens", "ner_tags"}) plus a loading
script whose `_LABELS` list maps tag ids to IOB2 labels. This module turns a raw row into the
extraction task used here: the sentence text and the list of tagged facts, each a value as
written in the sentence and its XBRL concept.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

NO_SPACE_BEFORE = {",", ".", ";", ":", "%", ")", "]", "?", "!", "'s", "’s"}
NO_SPACE_AFTER = {"$", "(", "["}


def parse_labels(script: str) -> list[str]:
    """The `_LABELS = [...]` list from the dataset's loading script."""
    m = re.search(r"_LABELS\s*=\s*(\[.*?\])", script, re.S)
    if not m:
        raise ValueError("no _LABELS list in the loading script")
    labels = ast.literal_eval(m.group(1))
    if labels[0] != "O":
        raise ValueError(f"unexpected first label {labels[0]!r}")
    return labels


def detokenize(tokens: list[str]) -> str:
    out = ""
    for i, tok in enumerate(tokens):
        if i and tok not in NO_SPACE_BEFORE and tokens[i - 1] not in NO_SPACE_AFTER:
            out += " "
        out += tok
    return out


@dataclass(frozen=True)
class Fact:
    value: str
    tag: str


def label_of(tag: int | str, labels: list[str]) -> str:
    """The raw files store tags as label strings; the Hugging Face loader turns them into ids."""
    if isinstance(tag, int):
        return labels[tag]
    if tag.isdigit():
        return labels[int(tag)]
    if tag not in labels:
        raise ValueError(f"unknown label {tag!r}")
    return tag


def facts(tokens: list[str], tags: list[int | str], labels: list[str]) -> list[Fact]:
    """IOB2 spans to facts; a span's value is its tokens joined without spaces (spans are
    numbers, so "1,234" or "7.00" stay as written)."""
    out: list[Fact] = []
    value, tag = None, None
    for tok, t in zip(tokens, tags, strict=True):
        label = label_of(t, labels)
        if label.startswith("B-") or (label.startswith("I-") and label[2:] != tag):
            if value is not None:
                out.append(Fact(value, tag))
            value, tag = tok, label[2:]
        elif label.startswith("I-"):
            value += tok
        else:
            if value is not None:
                out.append(Fact(value, tag))
            value, tag = None, None
    if value is not None:
        out.append(Fact(value, tag))
    return out


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    facts: tuple[Fact, ...]

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "facts": [{"value": f.value, "tag": f.tag} for f in self.facts],
        }

    @classmethod
    def from_json(cls, d: dict) -> Example:
        return cls(d["id"], d["text"], tuple(Fact(f["value"], f["tag"]) for f in d["facts"]))


def to_example(split: str, row: dict, labels: list[str]) -> Example:
    return Example(
        f"{split}-{row['id']}",
        detokenize(row["tokens"]),
        tuple(facts(row["tokens"], row["ner_tags"], labels)),
    )


def tag_counts(examples: Iterable[Example]) -> Counter:
    return Counter(f.tag for e in examples for f in e.facts)


def in_scope(example: Example, tags: set[str]) -> bool:
    """Every fact in the sentence uses one of the selected tags (a sentence with facts outside
    the selection would have a correct answer the prompt cannot express)."""
    return all(f.tag in tags for f in example.facts)
