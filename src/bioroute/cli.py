from __future__ import annotations

import argparse
import json
from pathlib import Path

from bioroute.benchmark import write_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BioRoute: benchmark encoder vs self-hosted and hosted LLM routers",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[2] / "reports" / "latest.json"),
        help="Path for the JSON report",
    )
    parser.add_argument(
        "--skip-llms",
        action="store_true",
        help="Skip Ollama/Claude routers (classical + encoder only)",
    )
    parser.add_argument(
        "--encoders",
        nargs="*",
        default=["answerdotai/ModernBERT-base"],
        help="Hugging Face model ids to fine-tune",
    )
    args = parser.parse_args()
    report = write_report(
        Path(args.out),
        include_llms=not args.skip_llms,
        encoders=tuple(args.encoders),
    )
    # Compact console summary
    print(f"wrote {args.out}")
    for m in report.get("models", []):
        print(
            f"  {m['model'][:56]:<56}  F1={m['macro_f1']:.3f}  "
            f"HR-R={m['human_review_recall']:.3f}  "
            f"p50={m['latency_ms_p50_single_request']:.1f}ms"
        )
    if not args.skip_llms:
        print(json.dumps({"n_test": report.get("n_test"), "hardware": report.get("hardware")}, indent=2))


if __name__ == "__main__":
    main()
