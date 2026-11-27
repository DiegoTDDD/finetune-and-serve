# Constitution: finetune-and-serve

1. **Same stack for every model**: one server, one schema, one test set.
2. **Numbers come from the pipeline run** and are rebuilt by `scripts/report.py`.
3. **A difference is claimed only when it is measured to be real** (ADR 0005).
4. **Free**: CPU runners, no paid key; costs at published list prices.
5. **Every change is tested** (pytest, ruff) before it is pushed.
