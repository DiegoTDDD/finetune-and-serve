# Results

Test set: 300 sentences from 10-K and 10-Q notes (FiNER-139 test split), 362 tagged facts, 60 sentences without facts. Every model runs in llama-server on the same 4-core CPU runner with the same JSON schema enforced by constrained decoding.

## Quality and single-user latency

| Model | Prompt | File | Fact F1 | Value F1 | Exact match | p50 latency | p95 latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| finetuned-0.5b-F16 | zero-shot | 994 MB | **0.827** | 0.881 | 0.743 | 1.49 s | 3.37 s |
| finetuned-0.5b-Q4_K_M | zero-shot | 398 MB | **0.816** | 0.883 | 0.747 | 0.91 s | 2.03 s |
| finetuned-0.5b-Q8_0 | zero-shot | 531 MB | **0.799** | 0.866 | 0.723 | 0.72 s | 1.67 s |
| base-3b-few-shot | 6-shot | 2,105 MB | **0.653** | 0.782 | 0.600 | 3.19 s | 7.83 s |
| base-0.5b-few-shot | 6-shot | 398 MB | **0.253** | 0.476 | 0.160 | 0.92 s | 1.70 s |
| base-3b-zero-shot | zero-shot | 2,105 MB | **0.240** | 0.296 | 0.287 | 1.07 s | 4.90 s |
| base-0.5b-zero-shot | zero-shot | 398 MB | **0.189** | 0.442 | 0.117 | 0.97 s | 2.35 s |

Fact F1 counts a prediction only when both the number and its XBRL concept match the auditors' tag; value F1 ignores the concept; exact match is the share of sentences answered completely right.

## By concept (F1)

| Concept | Test facts | finetuned-0.5b-Q4_K_M | base-3b-few-shot |
| --- | --- | --- | --- |
| DebtInstrumentInterestRateStatedPercentage | 49 | 0.82 | 0.72 |
| LineOfCreditFacilityMaximumBorrowingCapacity | 49 | 0.73 | 0.80 |
| DebtInstrumentFaceAmount | 39 | 0.61 | 0.69 |
| ConcentrationRiskPercentage1 | 35 | 0.96 | 0.39 |
| AllocatedShareBasedCompensationExpense | 31 | 0.89 | 0.69 |
| EffectiveIncomeTaxRateContinuingOperations | 31 | 0.93 | 0.69 |
| DebtInstrumentBasisSpreadOnVariableRate1 | 30 | 0.74 | 0.54 |
| AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount | 26 | 0.92 | 0.67 |
| IncomeTaxExpenseBenefit | 24 | 0.73 | 0.55 |
| EquityMethodInvestmentOwnershipPercentage | 22 | 0.88 | 0.47 |
| AmortizationOfIntangibleAssets | 15 | 0.97 | 0.97 |
| ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriod | 11 | 0.91 | 0.21 |

## Throughput

| Model | Concurrent requests | Requests/s | Output tokens/s | p50 latency | p95 latency |
| --- | --- | --- | --- | --- | --- |
| base-3b-few-shot | 1 | 0.28 | 8 | 3.27 s | 7.47 s |
| base-3b-few-shot | 4 | 0.43 | 13 | 8.12 s | 20.16 s |
| finetuned-0.5b-Q4_K_M | 1 | 1.01 | 34 | 0.90 s | 1.92 s |
| finetuned-0.5b-Q4_K_M | 4 | 1.60 | 54 | 2.13 s | 5.22 s |

## Cost

Self-hosted on AWS EC2 m7i.xlarge, us-east-1, on-demand Linux (4 vCPU, 16 GiB: the same size as the GitHub runner that produced the measurements) at $0.2016/hour ($147/month always on), at the best measured throughput:

| Self-hosted model | Requests/s | USD per million requests | Capacity per month |
| --- | --- | --- | --- |
| base-3b-few-shot | 0.43 | $130.54 | 1,127,412 |
| finetuned-0.5b-Q4_K_M | 1.60 | $34.89 | 4,217,940 |

Commercial APIs at list prices (checked 2026-11-19), with the prompt sizes measured above (token counts from Qwen's tokenizer, so estimates):

| API | Prompt | USD per million requests | Requests/month where the VM becomes cheaper |
| --- | --- | --- | --- |
| Claude Haiku 5.5 | few-shot | $91.11 | 1,615,278 |
| Claude Haiku 5.5 | zero-shot | $44.76 | 3,287,935 |
| GPT-5.4 mini | few-shot | $705.98 | 208,460 |
| GPT-5.4 mini | zero-shot | $343.88 | 427,969 |

The APIs' accuracy on this task was not measured (no paid key is used in this project).

## Training

LoRA rank 16 on all attention and MLP projections of Qwen/Qwen2.5-0.5B-Instruct (8.8M trainable parameters), 2000 examples, 99 steps, 308.5 minutes on 4 CPU threads. Validation loss on answer tokens: 0.5698 before, 0.0234 after.
