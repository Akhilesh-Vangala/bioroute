#!/usr/bin/env python3
"""Render latency–accuracy chart from reports/latest.json."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "latest.json"
OUT = ROOT / "docs" / "assets" / "latency-accuracy.png"


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("pip install matplotlib") from e

    report = json.loads(REPORT.read_text())
    # Prefer temperature-scaled / primary rows for the four headline routers
    wanted = [
        ("tfidf_logreg (temperature-scaled)", "TF-IDF+LR"),
        ("answerdotai/ModernBERT-base (fine-tuned, temperature-scaled)", "ModernBERT 150M"),
        ("qwen2.5:7b (zero-shot, verbalized confidence)", "Qwen 2.5 7B"),
        ("claude-haiku-5-5 (zero-shot, verbalized confidence)", "Claude Haiku"),
    ]
    by_name = {m["model"]: m for m in report["models"]}
    xs, ys, labels = [], [], []
    for key, label in wanted:
        m = by_name[key]
        xs.append(max(m["latency_ms_p50_single_request"], 0.05))
        ys.append(m["macro_f1"])
        labels.append(label)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#FFFFFF")
    colors = ["#64748B", "#2563EB", "#059669", "#7C3AED"]
    ax.scatter(xs, ys, s=120, c=colors, zorder=3, edgecolors="white", linewidths=1.2)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9, color="#0F172A")
    ax.set_xscale("log")
    ax.set_xlabel("p50 latency (ms, log scale)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("BioRoute — latency vs accuracy (n=150 held-out)")
    ax.set_ylim(0.7, 1.02)
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.axhline(0.99, color="#059669", lw=0.8, alpha=0.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
