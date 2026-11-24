# ADR 0004: Compare cost at list prices with measured token counts

**Status:** accepted

## Context
Self-hosting looks cheap per request and expensive per month; APIs look the other way round.
A single "cheaper" verdict hides the volume at which the answer flips.

## Decision
- Self-hosted cost is the price of a cloud VM the same size as the runner that produced the
  measurements (AWS m7i.xlarge, 4 vCPU, 16 GiB, on-demand) divided by the best throughput measured
  on that runner. That is the cost at full load.
- API cost uses each provider's list price per million input and output tokens, applied to the
  prompt and answer sizes measured in the evaluation (zero-shot and six-shot prompts). Counts come
  from Qwen's tokenizer, so they are estimates for other tokenizers.
- The report gives the monthly volume at which an always-on VM becomes cheaper than each API.
- Prices are kept in `costs.json` with their source and the date they were checked.

## Consequences
- The conclusion depends on volume, and the report says so instead of picking a winner.
- The APIs' accuracy on this task is not measured, because the project uses no paid keys; the
  comparison is cost per request for a model of known quality on one side and unknown quality on
  the other.
- Prompt caching discounts, reserved or spot VM prices and idle time are left out; each would move
  the break-even point and is named in the README's limits.
