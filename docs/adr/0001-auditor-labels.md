# ADR 0001: Train and test on labels written by auditors

**Status:** accepted

## Context
Fine-tuning projects often label data with a larger model and then measure the small model
against those same labels. That measures how well it copies the teacher, mistakes included.

## Decision
Use FiNER-139: sentences from the notes of 10-K and 10-Q filings, with each numeric fact tagged
by professional auditors with its XBRL concept (the tags companies file with the SEC). The task
keeps the twelve most frequent concepts and is scored against the auditors' tags. Train,
validation and test sets come from the dataset's own splits, so test sentences are from filings
the model never trained on. A fifth of every set is sentences with no facts, because most
numbers in filings (years, counts, untagged amounts) must be left alone.

## Consequences
- The score means the same thing for every model, including the fine-tuned one.
- The dataset is CC BY-SA 4.0; the sampled sets are committed with a sha256 manifest and the
  Hugging Face revision they came from, and CI fails if they change without a new manifest.
- Auditor tags are not perfect (the dataset card notes some discrepancies); the ceiling is below 1.0.
