from __future__ import annotations

import argparse
import json
from pathlib import Path

from bioroute.benchmark import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BioRoute calibration benchmark")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[2] / "reports" / "latest.json"),
    )
    args = parser.parse_args()
    report = write_report(Path(args.out))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
