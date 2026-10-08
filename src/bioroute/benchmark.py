"""Benchmark routers for an agent workflow: self-hosted encoders vs hosted and local LLMs.

Compares accuracy (macro-F1, per-route recall, human_review recall), calibration
(ECE, Brier, before and after temperature scaling), latency, cost, and licensing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split

from bioroute.data import ROOT, ROUTES, load
from bioroute.metrics import brier_multiclass, expected_calibration_error

HR = ROUTES.index("human_review")

LICENSES = {
    "tfidf_logreg": {"license": "BSD-3 (scikit-learn)", "hosting": "self-hosted", "data_handling": "stays on our infrastructure"},
    "answerdotai/ModernBERT-base": {"license": "Apache-2.0", "hosting": "self-hosted", "data_handling": "stays on our infrastructure"},
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext": {"license": "MIT", "hosting": "self-hosted", "data_handling": "stays on our infrastructure"},
    "qwen2.5:7b": {"license": "Apache-2.0 (Qwen2.5-7B)", "hosting": "self-hosted (Ollama)", "data_handling": "stays on our infrastructure"},
    "claude-haiku-5-5": {"license": "Anthropic commercial terms", "hosting": "hosted API", "data_handling": "sent to vendor; governed by contract and retention settings"},
}


def score(name: str, probs: np.ndarray, y: np.ndarray, latency_ms: float, cost_per_1k: float, extra: dict | None = None) -> dict:
    pred = probs.argmax(1)
    return {
        "model": name,
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 3),
        "accuracy": round(float((pred == y).mean()), 3),
        "recall_by_route": {r: round(float(v), 3) for r, v in zip(ROUTES, recall_score(y, pred, average=None, labels=range(len(ROUTES))))},
        "human_review_recall": round(float(((pred == HR) & (y == HR)).sum() / max((y == HR).sum(), 1)), 3),
        "ece": round(expected_calibration_error(probs, y), 3),
        "brier": round(brier_multiclass(probs, y, len(ROUTES)), 3),
        "latency_ms_p50_single_request": round(latency_ms, 1),
        "cost_per_1k_requests_usd": round(cost_per_1k, 4),
        **LICENSES.get(name.split(" ")[0], {}),
        **(extra or {}),
    }


def run(include_llms: bool = True, encoders: tuple[str, ...] = ("answerdotai/ModernBERT-base",)) -> dict:
    x_all, y_all = load("train")
    x_test, y_test = load("test")
    y_test_a = np.array(y_test)
    x_tr, x_val, y_tr, y_val = train_test_split(x_all, y_all, test_size=0.15, stratify=y_all, random_state=0)
    results = []

    # TF-IDF + logistic regression, with and without temperature scaling.
    from bioroute.encoders import fit_temperature, softmax

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    lr = LogisticRegression(max_iter=2000, C=4.0).fit(vec.fit_transform(x_tr), y_tr)
    start = time.perf_counter()
    for t in x_test[:50]:
        lr.decision_function(vec.transform([t]))
    lat = (time.perf_counter() - start) * 1000 / 50
    val_lg = lr.decision_function(vec.transform(x_val))
    test_lg = lr.decision_function(vec.transform(x_test))
    temp = fit_temperature(val_lg, y_val)
    results.append(score("tfidf_logreg (uncalibrated)", softmax(test_lg), y_test_a, lat, 0.0))
    results.append(score("tfidf_logreg (temperature-scaled)", softmax(test_lg, temp), y_test_a, lat, 0.0, {"temperature": round(temp, 3)}))

    # Fine-tuned open-weight encoders.
    from bioroute.encoders import fine_tune, logits, single_query_latency_ms

    for model_id in encoders:
        model, tok, train_s = fine_tune(model_id, x_tr, y_tr, len(ROUTES))
        val_lg = logits(model, tok, x_val)
        test_lg = logits(model, tok, x_test)
        temp = fit_temperature(val_lg, y_val)
        lat = single_query_latency_ms(model, tok, x_test)
        n_params = sum(p.numel() for p in model.parameters())
        extra = {"params_millions": round(n_params / 1e6, 1), "train_seconds_m4": round(train_s, 1)}
        results.append(score(f"{model_id} (fine-tuned, uncalibrated)", softmax(test_lg), y_test_a, lat, 0.0, extra))
        results.append(score(f"{model_id} (fine-tuned, temperature-scaled)", softmax(test_lg, temp), y_test_a, lat, 0.0,
                             {**extra, "temperature": round(temp, 3)}))
        del model

    if include_llms:
        from bioroute.llm import claude_classify, ollama_classify

        q = ollama_classify(x_test)
        results.append(score("qwen2.5:7b (zero-shot, verbalized confidence)", q["probs"], y_test_a, q["latency_ms_p50"], 0.0))
        cache = ROOT / "results" / "claude_haiku_test_preds.npz"
        if cache.exists():
            z = np.load(cache)
            c = {"probs": z["probs"], "latency_ms_p50": float(z["latency_ms_p50"]), "cost_per_1k_usd": float(z["cost_per_1k_usd"])}
        else:
            c = claude_classify(x_test)
        results.append(score("claude-haiku-5-5 (zero-shot, verbalized confidence)", c["probs"], y_test_a, c["latency_ms_p50"],
                             c["cost_per_1k_usd"], {"cost_note": "API list price computed from measured tokens; latency via headless Claude Code includes CLI overhead"}))

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "routes": ROUTES,
        "n_train": len(x_tr), "n_val": len(x_val), "n_test": len(x_test),
        "hardware": "Apple M4, 16 GB (MPS)",
        "models": results,
    }


def write_report(path: Path, include_llms: bool = True) -> dict:
    report = run(include_llms=include_llms)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return report
