from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from quality.quality_evaluator import QualityEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare output validity and lightweight quality between two engine runs."
    )
    parser.add_argument("--baseline", required=True, help="Path to baseline request_metrics csv/json.")
    parser.add_argument("--candidate", required=True, help="Path to candidate/vLLM request_metrics csv/json.")
    parser.add_argument("--output", help="Optional path to save JSON comparison.")
    return parser.parse_args()


def _normalize_text(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def _load(path: str | Path) -> pd.DataFrame:
    return QualityEvaluator().load_request_records(path)


def _validity_and_quality(df: pd.DataFrame) -> dict[str, Any]:
    return QualityEvaluator().evaluate(df)


def _aligned(df_left: pd.DataFrame, df_right: pd.DataFrame) -> pd.DataFrame:
    if "request_id" not in df_left.columns or "request_id" not in df_right.columns:
        return pd.DataFrame()

    left = df_left.copy()
    right = df_right.copy()
    left["request_id"] = left["request_id"].astype(str)
    right["request_id"] = right["request_id"].astype(str)
    merged = left.merge(
        right,
        on="request_id",
        how="inner",
        suffixes=("_baseline", "_candidate"),
    )
    return merged


def _generic_output_agreement(aligned_df: pd.DataFrame) -> dict[str, Any]:
    if aligned_df.empty:
        return {"count": 0, "exact_match_rate": 0.0, "normalized_exact_match_rate": 0.0}

    baseline = aligned_df.get("output_text_baseline", pd.Series(dtype=str)).fillna("").astype(str)
    candidate = aligned_df.get("output_text_candidate", pd.Series(dtype=str)).fillna("").astype(str)

    exact = (baseline == candidate).mean() if len(aligned_df) else 0.0
    normalized = (
        baseline.map(_normalize_text) == candidate.map(_normalize_text)
    ).mean() if len(aligned_df) else 0.0

    return {
        "count": int(len(aligned_df)),
        "exact_match_rate": float(exact),
        "normalized_exact_match_rate": float(normalized),
    }


def _pubhealth_agreement(aligned_df: pd.DataFrame) -> dict[str, Any]:
    if aligned_df.empty or "dataset_name_baseline" not in aligned_df.columns:
        return {"count": 0, "label_agreement": 0.0}

    pubhealth = aligned_df[
        aligned_df["dataset_name_baseline"].fillna("").astype(str).str.lower() == "pubhealth"
    ].copy()
    if pubhealth.empty:
        return {"count": 0, "label_agreement": 0.0}

    normalize = QualityEvaluator._normalize_pubhealth_prediction
    baseline = pubhealth.get("output_text_baseline", pd.Series(dtype=str)).fillna("").astype(str).map(normalize)
    candidate = pubhealth.get("output_text_candidate", pd.Series(dtype=str)).fillna("").astype(str).map(normalize)
    agreement = (baseline == candidate).mean() if len(pubhealth) else 0.0

    return {
        "count": int(len(pubhealth)),
        "label_agreement": float(agreement),
    }


def main() -> None:
    args = parse_args()
    baseline_df = _load(args.baseline)
    candidate_df = _load(args.candidate)
    aligned = _aligned(baseline_df, candidate_df)

    results = {
        "baseline": _validity_and_quality(baseline_df),
        "candidate": _validity_and_quality(candidate_df),
        "engine_to_engine": {
            "generic_output_agreement": _generic_output_agreement(aligned),
            "pubhealth_label_agreement": _pubhealth_agreement(aligned),
        },
    }

    rendered = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
