# BioRoute — Calibrated Biomedical Model Router

Decide whether a **self-hosted** biomedical/classifier stack can replace a **hosted LLM** for fast routing / safety-review decisions — with **calibration**, latency, and cost.

## Routes

Incoming healthcare business request →

1. `drug_labeling` → FDA agent  
2. `clinical_evidence` → Trials agent  
3. `coverage` → CMS agent  
4. `human_review` → escalate  

## Models compared (v0)

| Candidate | Role |
| --- | --- |
| TF-IDF + Logistic Regression | Cheap interpretable baseline |
| Calibrated TF-IDF + LR | Same + sigmoid calibration |
| Hosted LLM stub | Quality/cost comparison placeholder for Claude classification |
| DistilBERT / BiomedBERT | Optional (`pip install -e ".[transformers]"`) — next iteration |

## Metrics

- Macro-F1 + per-class recall
- Expected Calibration Error (ECE)
- Brier score
- Human-review false-negative rate
- Abstention vs coverage
- Latency + estimated $/1k requests
- License / data-handling notes

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
bioroute-bench
```

Output: `reports/latest.json`

## Decision memo (fill from report)

Which classifier is accurate enough, fast enough, legally usable, and economical enough to serve as an agent front-door?
