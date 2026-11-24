# finetune-and-serve

[![ci](https://github.com/DiegoTDDD/finetune-and-serve/actions/workflows/ci.yml/badge.svg)](https://github.com/DiegoTDDD/finetune-and-serve/actions/workflows/ci.yml)
[![pipeline](https://github.com/DiegoTDDD/finetune-and-serve/actions/workflows/pipeline.yml/badge.svg)](https://github.com/DiegoTDDD/finetune-and-serve/actions/workflows/pipeline.yml)

A 0.5B model fine-tuned with LoRA to tag numeric facts in 10-K sentences with their XBRL
concepts, trained, quantized, served and benchmarked on the free CPU runner of GitHub Actions.
**On 300 test sentences tagged by auditors, the fine-tuned model quantized to 398 MB scores
0.816 F1, against 0.653 for a 3B model prompted with six worked examples, and serves 3.7 times
more requests per second on the same 4-core CPU.**

## Problem

Structured extraction is usually done by prompting a large model. That works, but every request
pays for a long prompt, on an API or on hardware big enough for the model. A small model trained
for the one job might do it better and cheaper, and the only way to know is to measure both on
the same data and the same machine.

## Task and data

[FiNER-139](https://huggingface.co/datasets/nlpaueb/finer-139) (CC BY-SA 4.0): sentences from the
notes of 10-K and 10-Q filings, each number tagged by auditors with the XBRL concept it reports.
Given a sentence, the model returns the facts as JSON:

```text
This agreement covers the Partnership ’ s ownership of 22.7% in Ursa and 25% in KM Phoenix.
{"facts": [{"value": "22.7", "tag": "EquityMethodInvestmentOwnershipPercentage"},
           {"value": "25", "tag": "EquityMethodInvestmentOwnershipPercentage"}]}
```

(sentence `test-1043111` from the test set, with the auditors' tags)

The twelve most frequent concepts are in scope. 2,000 training, 150 validation and 300 test
sentences are sampled from the dataset's own splits; a fifth of each set has no facts, because
most numbers in filings (years, counts, untagged amounts) must be left alone. The sets are
committed with a sha256 manifest ([`data/`](data/)), and CI fails if they change
([ADR 0001](docs/adr/0001-auditor-labels.md)).

## Solution

```
FiNER-139 ──► prepare_data.py ──► data/{train,val,test}.jsonl (+ manifest)
                                        │
Qwen2.5-0.5B-Instruct ──► LoRA on CPU ──┤  loss on answer tokens only
                                        ▼
                     merge ──► GGUF F16 ──► Q8_0, Q4_K_M  (llama.cpp)
                                        │
          llama-server, same build, same schema-constrained decoding for every model
                                        ▼
       evaluate.py (quality, latency) · benchmark.py (throughput) · report.py (cost)
```

- **Training** ([ADR 0002](docs/adr/0002-lora-on-a-cpu-runner.md)): LoRA rank 16 on every
  attention and MLP projection, in float32 on 4 CPU threads. The language-model head runs only at
  the answer positions: full logits over a 152k-token vocabulary took more memory than the model.
- **Serving** ([ADR 0003](docs/adr/0003-same-stack-for-every-model.md)): every model is a GGUF
  file served by the same llama-server with the same JSON schema enforced by the grammar, so all
  outputs are valid and the scores measure only the facts.
- **Baselines**: the same 0.5B model and Qwen2.5-3B-Instruct (official Q4_K_M), each with the
  zero-shot prompt and with six worked examples from the validation set.

The whole pipeline is one workflow ([`pipeline.yml`](.github/workflows/pipeline.yml)); the run
that produced these results committed [`reports/`](reports/) and published the
[model release](https://github.com/DiegoTDDD/finetune-and-serve/releases/tag/model-44606410559)
(adapter, Q4_K_M and Q8_0 GGUF).

## Results

From [`reports/results.md`](reports/results.md):

| Model | Prompt | File | Fact F1 | Value F1 | Exact match | p50 latency |
| --- | --- | --- | --- | --- | --- | --- |
| Fine-tuned 0.5B, F16 | zero-shot | 994 MB | **0.827** | 0.881 | 0.743 | 1.49 s |
| Fine-tuned 0.5B, Q4_K_M | zero-shot | 398 MB | **0.816** | 0.883 | 0.747 | 0.91 s |
| Fine-tuned 0.5B, Q8_0 | zero-shot | 531 MB | 0.799 | 0.866 | 0.723 | 0.72 s |
| Qwen2.5-3B | 6-shot | 2,105 MB | 0.653 | 0.782 | 0.600 | 3.19 s |
| Qwen2.5-0.5B | 6-shot | 398 MB | 0.253 | 0.476 | 0.160 | 0.92 s |
| Qwen2.5-3B | zero-shot | 2,105 MB | 0.240 | 0.296 | 0.287 | 1.07 s |
| Qwen2.5-0.5B | zero-shot | 398 MB | 0.189 | 0.442 | 0.117 | 0.97 s |

Fact F1 counts a prediction only if both the number and the concept match the auditors' tag;
value F1 ignores the concept; exact match is the share of sentences answered completely right.

- Fine-tuning moves the 0.5B model from 0.19 to 0.82 fact F1. It beats the 3B model with
  examples on ten of the twelve concepts; the 3B model is better on credit-line capacity
  (0.80 vs 0.73) and debt face amounts (0.69 vs 0.61).
- Quantizing to 4 bits costs about one point of F1 (0.827 to 0.816) for a file 2.5 times
  smaller. Q8_0 scoring slightly below Q4_K_M is within the noise of 300 sentences.
- Without examples, the 3B model answers "no facts" for 257 of 300 sentences: it is cautious,
  not confused, and six examples are what make it usable.

**Throughput** (4 vCPU, 100 requests): the fine-tuned Q4_K_M model serves 1.60 requests/s with
four concurrent requests (p50 2.1 s), the 3B model with examples 0.43 requests/s (p50 8.1 s).

**Cost** ([ADR 0004](docs/adr/0004-cost-at-list-prices.md)): on an always-on AWS m7i.xlarge
($0.2016/hour, the runner's size), the fine-tuned model costs **$35 per million requests** at
full load, against $131 for the 3B model. At list prices and the measured prompt sizes, Claude
Haiku 5.5 would cost $45 (zero-shot) to $91 (six examples) per million requests and GPT-5.4 mini
$344 to $706. The VM is cheaper than Haiku with examples from about 1.6 million requests a month,
and than GPT-5.4 mini from about 210,000. Below those volumes the API is cheaper. The APIs'
accuracy on this task was not measured.

**Training**: 308 minutes on 4 CPU threads; the 300-minute budget stopped it at step 99 of 125, so
1,584 of the 2,000 examples were seen. Validation loss on answer tokens fell from 0.570 to 0.023.

## Run it

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q                                   # 26 tests: data, task, metrics, training, server, report
gh workflow run pipeline.yml -f smoke=true  # the whole pipeline on a handful of examples
gh workflow run pipeline.yml -f smoke=false # full run: about 6.5 hours on the free runner
```

## Design decisions

- [ADR 0001](docs/adr/0001-auditor-labels.md): train and test on labels written by auditors.
- [ADR 0002](docs/adr/0002-lora-on-a-cpu-runner.md): LoRA in full precision on a free CPU runner, not QLoRA on a GPU.
- [ADR 0003](docs/adr/0003-same-stack-for-every-model.md): serve every model the same way, with the output schema enforced.
- [ADR 0004](docs/adr/0004-cost-at-list-prices.md): compare cost at list prices with measured token counts.

## Limits

- One training run and one evaluation run: differences of a point or two between models (such as
  Q8_0 against Q4_K_M) are not significant at 300 sentences.
- Training stopped at 79% of one epoch because of the time budget; a longer run might do better.
- Twelve concepts out of the 139 in FiNER; the prompt would not scale to all of them as written.
- Cost figures assume the VM is fully used and ignore prompt caching discounts the APIs offer.

## Stack

Python 3.11, PyTorch, Transformers, PEFT, llama.cpp (llama-server, GGUF quantization), Hugging
Face Hub, pytest, ruff, GitHub Actions.

## License

MIT. The dataset files in `data/` are derived from FiNER-139 and keep its CC BY-SA 4.0 license.
