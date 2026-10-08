from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from bioroute.data import ROUTES, TEST, TRAIN
from bioroute.metrics import abstain_coverage, brier_multiclass, expected_calibration_error


def _hosted_llm_stub_predict(texts: list[str], classes: list[str]) -> np.ndarray:
    """Keyword heuristic standing in for a hosted Claude classifier (demo/cost baseline)."""
    n = len(classes)
    probs = np.full((len(texts), n), 1.0 / n)
    for i, text in enumerate(texts):
        t = text.lower()
        scores = np.zeros(n)
        if any(k in t for k in ["label", "warning", "fda", "indication", "adverse"]):
            scores[classes.index("drug_labeling")] += 2
        if any(k in t for k in ["trial", "nct", "clinicaltrials", "phase"]):
            scores[classes.index("clinical_evidence")] += 2
        if any(k in t for k in ["cms", "medicare", "coverage", "ncd", "lcd"]):
            scores[classes.index("coverage")] += 2
        if any(k in t for k in ["off-label", "superior", "promotional", "phi", "approve"]):
            scores[classes.index("human_review")] += 2.5
        if scores.sum() == 0:
            scores[:] = 1
        probs[i] = scores / scores.sum()
    return probs


def run_benchmark() -> dict:
    le = LabelEncoder()
    le.fit(ROUTES)
    x_train = [t for t, _ in TRAIN]
    y_train = le.transform([y for _, y in TRAIN])
    x_test = [t for t, _ in TEST]
    y_test = le.transform([y for _, y in TEST])

    # TF-IDF + Logistic Regression baseline
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            (
                "clf",
                LogisticRegression(max_iter=2000),
            ),
        ]
    )
    start = time.perf_counter()
    pipe.fit(x_train, y_train)
    raw_probs = pipe.predict_proba(x_test)
    cal_pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            (
                "clf",
                CalibratedClassifierCV(
                    LogisticRegression(max_iter=2000),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ]
    )
    cal_pipe.fit(x_train, y_train)
    cal_probs = cal_pipe.predict_proba(x_test)
    tfidf_latency = (time.perf_counter() - start) * 1000 / max(len(x_test), 1)

    hosted_probs = _hosted_llm_stub_predict(x_test, list(le.classes_))

    def pack(name: str, probs: np.ndarray, latency_ms: float, cost_per_1k: float) -> dict:
        pred = probs.argmax(axis=1)
        macro_f1 = float(f1_score(y_test, pred, average="macro"))
        per_class = {
            cls: float(recall_score(y_test, pred, labels=[i], average="macro", zero_division=0))
            for i, cls in enumerate(le.classes_)
        }
        # false-negative rate on human_review class
        hr = list(le.classes_).index("human_review")
        hr_mask = y_test == hr
        fnr = float(((pred[hr_mask] != hr).mean()) if np.any(hr_mask) else 0.0)
        return {
            "model": name,
            "macro_f1": macro_f1,
            "per_class_recall": per_class,
            "ece": expected_calibration_error(probs, y_test),
            "brier": brier_multiclass(probs, y_test, len(le.classes_)),
            "human_review_fnr": fnr,
            "abstention": abstain_coverage(probs, y_test, threshold=0.55),
            "latency_ms_per_request_est": latency_ms,
            "cost_per_1000_requests_usd_est": cost_per_1k,
            "license_note": (
                "TF-IDF/LR: self-hosted, no model license issue. "
                "Hosted LLM stub: replace with real Claude API pricing. "
                "BiomedBERT/DistilBERT: add under optional transformers extra."
            ),
        }

    report = {
        "routes": list(le.classes_),
        "n_train": len(TRAIN),
        "n_test": len(TEST),
        "models": [
            pack("tfidf_logreg_raw", raw_probs, tfidf_latency, 0.02),
            pack("tfidf_logreg_calibrated", cal_probs, tfidf_latency * 1.05, 0.02),
            pack("hosted_llm_stub", hosted_probs, 350.0, 3.50),
        ],
        "decision_memo_prompt": (
            "Which classifier is accurate enough, calibrated enough, fast enough, "
            "and economical enough to front a CMG healthcare agent?"
        ),
    }
    return report


def write_report(path: Path) -> dict:
    report = run_benchmark()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
