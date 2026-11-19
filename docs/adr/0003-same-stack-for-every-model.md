# ADR 0003: Serve every model the same way, with the output schema enforced

**Status:** accepted

## Context
Comparisons between a fine-tuned model and prompted baselines are easy to skew: different
runtimes, different sampling, or counting malformed JSON as a model failure when a production
system would constrain the output anyway.

## Decision
Every model is converted to GGUF and served by the same llama-server build (llama.cpp, pinned
tag) on the same runner, with the same threads, greedy decoding, and the same JSON schema
enforced by grammar-constrained decoding (the concept field is an enum of the twelve concepts).
The fine-tuned model is evaluated with the zero-shot prompt it was trained with; baselines get
the same prompt with and without six worked examples from the validation set.

## Consequences
- All outputs are valid JSON, so the scores measure only whether the right numbers get the right
  concepts.
- Speed differences come from model size and prompt length, which is what a deployment would see.
- Quantization is measured directly: the fine-tuned model is evaluated as F16, Q8_0 and Q4_K_M.
