# Methodology

## Task

Five-way routing for a healthcare evidence agent:

| Route | Meaning |
| --- | --- |
| `drug_label` | FDA label / PI questions |
| `clinical_trials` | Trial lookup / enrollment |
| `coverage` | Payer / formulary |
| `human_review` | Compliance-sensitive; must escalate |
| `out_of_scope` | Refuse / deflect |

Primary safety metric: **recall on `human_review`**. Missing an escalation is worse than a wrong specialty route.

## Models

1. **TF-IDF (1–2 grams) + logistic regression** — classical control; temperature scaling on validation logits.
2. **ModernBERT-base (~150M)** — Hugging Face `answerdotai/ModernBERT-base`, fine-tuned in PyTorch on Apple M4 MPS; temperature fit on validation.
3. **Qwen 2.5 7B** — zero-shot via **Ollama**, structured JSON label + verbalized confidence.
4. **Claude Haiku** — zero-shot via headless Claude Code with JSON schema; cost from measured tokens at list price.

## Protocol

1. Stratified 85/15 train/val from `data/train.jsonl` (`random_state=0`).
2. Fit classical + encoder on train; calibrate temperature on val.
3. Evaluate once on fixed `data/test.jsonl` (150 rows).
4. Latency: p50 over single-request timed loops (encoder/classical on-device; LLM wall clock includes generation).
5. Persist machine-readable output to `reports/latest.json`.

## Metrics

| Metric | Definition |
| --- | --- |
| Macro-F1 | Unweighted mean F1 across five routes |
| Per-route recall | `sklearn.metrics.recall_score(..., average=None)` |
| ECE | 10-bin expected calibration error on max-class confidence |
| Brier | Multiclass Brier score |
| Cost / 1k | Self-hosted = $0; Claude = token list price × usage |

## Reproducibility

```bash
pip install -e ".[transformers,dev]"
ollama pull qwen2.5:7b
pytest -q
bioroute-bench --out reports/latest.json
```

Cached Claude predictions: `results/claude_haiku_test_preds.npz`. Hardware stamp in the report: `Apple M4, 16 GB (MPS)`.
