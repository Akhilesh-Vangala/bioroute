# Architecture

```text
                    ┌─────────────────────────────────────────┐
   user / agent     │              BioRoute bench             │
   request text ───►│  data.load → routers → metrics → report │
                    └─────────────────────────────────────────┘
                                      │
          ┌───────────────┬───────────┼───────────┬───────────────┐
          ▼               ▼           ▼           ▼               ▼
     TF-IDF+LR      ModernBERT     Qwen 7B     Claude API    reports/
     (sklearn)      (PyTorch/HF)   (Ollama)    (hosted)      latest.json
```

## Components

| Module | Responsibility |
| --- | --- |
| `data.py` | Route enum, JSONL loaders, repo root paths |
| `encoders.py` | Fine-tune loop, logits, MPS-aware latency, temperature fit |
| `llm.py` | Ollama HTTP client + Claude CLI JSON router |
| `metrics.py` | ECE, Brier, abstain coverage helpers |
| `benchmark.py` | Orchestrates train → calibrate → evaluate → write |
| `cli.py` | `bioroute-bench` flags (`--skip-llms`, `--out`) |

## Serving implication

BioRoute is a **benchmark**, not a production gateway. The intended production shape after the study:

1. **Online path (quality):** Qwen 2.5 7B behind Ollama / vLLM on private GPU — PHI stays local.
2. **Online path (latency):** Distilled ModernBERT (or BiomedBERT) once labeled traffic closes the F1 gap.
3. **Policy gate:** If `human_review` posterior &lt; τ or model abstains → force human queue.

## Threats to validity

- Synthetic train phrasing vs hand-written test (encoder underfits distribution shift).
- Verbalized LLM confidence ≠ calibrated logits.
- Small test n; publish intervals before claiming SOTA.
