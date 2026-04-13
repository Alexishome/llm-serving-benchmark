from __future__ import annotations

import argparse
import json
from pathlib import Path

from quality.quality_evaluator import QualityEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate output validity and lightweight task quality.")
    parser.add_argument(
        "--request-metrics",
        required=True,
        help="Path to request_metrics.csv or request_metrics.json.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save evaluation JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = QualityEvaluator()
    request_df = evaluator.load_request_records(args.request_metrics)
    results = evaluator.evaluate(request_df)
    rendered = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
