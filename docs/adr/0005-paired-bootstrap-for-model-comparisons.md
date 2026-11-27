# ADR 0005: compare models with a paired bootstrap over the test sentences

**Status:** Accepted

## Context
The results table ranks seven configurations on 300 test sentences. Some gaps are large (0.816
against 0.653), others are a point or two (Q8_0 at 0.799 against Q4_K_M at 0.816), and the
README called the small ones "within the noise" without measuring the noise.

## Decision
`ftserve/stats.py` rescoring from the saved predictions, with no new model run:

- each configuration's fact F1 gets a 95% percentile interval from 2,000 bootstrap resamples of
  the sentences;
- the comparisons the conclusions rest on (fine-tuned against the 3B model with examples,
  against the same 0.5B model with examples, and the three quantizations against each other)
  use a **paired** bootstrap: both models are rescored on the same resampled sentences, and the
  interval is for the difference. A difference is called real only when its interval excludes 0.

Pure Python, seeded, so `scripts/report.py` writes the same tables on every run; the micro F1
recomputed per sentence equals the pipeline's score for all seven runs.

## Consequences
- Fine-tuning's gains are real (+0.162 over the 3B model, interval +0.103 to +0.225); the
  quantization differences are not distinguishable at 300 sentences (intervals include 0).
- The bootstrap covers which sentences made the test set, not training randomness: one
  training run cannot say how much a different seed would move the fine-tuned scores.
