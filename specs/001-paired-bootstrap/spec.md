# Spec 001: intervals and significance for model comparisons

## Problem
Small gaps between configurations were called noise without measuring it.

## Requirements
- R1. Each configuration's fact F1 has a 95% bootstrap interval over the 300 sentences.
- R2. The key comparisons have a paired-bootstrap interval for the difference and a p-value;
  "real" only when the interval excludes 0.
- R3. Computed from the saved predictions; seeded; equal to the pipeline's F1 on the full set.
- R4. results.md, results.json and the README show them; the Limits say what they do not cover.
