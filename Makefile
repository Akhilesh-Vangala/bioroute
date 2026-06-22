.PHONY: help install install-ml test bench bench-encoders plot

help:
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Core deps
	pip install -e ".[dev]"

install-ml: ## Core + PyTorch/transformers
	pip install -e ".[transformers,dev]"

test: ## Unit tests
	pytest -q

bench: ## Full benchmark (needs Ollama qwen2.5:7b for LLM rows)
	bioroute-bench --out reports/latest.json

bench-encoders: ## Classical + ModernBERT only
	bioroute-bench --skip-llms --out reports/encoders-only.json

plot: ## Regenerate docs/assets charts from reports/latest.json
	python scripts/plot_results.py
