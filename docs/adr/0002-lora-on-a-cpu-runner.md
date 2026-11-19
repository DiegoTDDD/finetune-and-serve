# ADR 0002: LoRA in full precision on a free CPU runner, not QLoRA on a GPU

**Status:** accepted

## Context
The project must cost nothing and be reproducible by anyone who forks it. Free GPU notebooks
cannot be driven from CI and their hardware changes between sessions. QLoRA quantizes the frozen
weights to 4 bits so that large models fit in GPU memory; a 0.5B model fits in a 16 GB runner
without that.

## Decision
Train LoRA adapters (rank 16, all attention and MLP projections) on Qwen2.5-0.5B-Instruct in
float32 on the standard 4-core GitHub runner, in the same workflow that exports, evaluates and
benchmarks the result. Two choices make it fit:
- the loss is computed only on the answer tokens, and the language-model head is applied only at
  those positions: full logits over the 152k-token vocabulary would need more memory than the model;
- activation checkpointing trades recomputation for memory.

## Consequences
- Every number in the results comes from one CI run that anyone can repeat with one click.
- Training takes hours instead of minutes and is limited to a small model and a few thousand
  examples. The same code runs on a GPU unchanged.
