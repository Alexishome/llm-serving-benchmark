from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class QualityEvaluator:
    def load_request_records(self, path: str | Path) -> pd.DataFrame:
        request_path = Path(path)
        if request_path.suffix.lower() == ".json":
            return pd.DataFrame(json.loads(request_path.read_text(encoding="utf-8")))
        return pd.read_csv(request_path)

    def evaluate(self, request_df: pd.DataFrame) -> dict[str, Any]:
        results: dict[str, Any] = {
            "validity": self._validity_metrics(request_df),
        }

        pubhealth_df = self._subset_dataset(request_df, "pubhealth")
        if not pubhealth_df.empty:
            results["pubhealth"] = self._evaluate_pubhealth(pubhealth_df)

        return results

    @staticmethod
    def _subset_dataset(request_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        if "dataset_name" not in request_df.columns:
            return pd.DataFrame()
        return request_df[
            request_df["dataset_name"].fillna("").astype(str).str.lower() == dataset_name.lower()
        ].copy()

    @staticmethod
    def _validity_metrics(request_df: pd.DataFrame) -> dict[str, Any]:
        if request_df.empty:
            return {
                "request_count": 0,
                "success_rate": 0.0,
                "non_empty_rate": 0.0,
                "valid_response_rate": 0.0,
            }

        output_text = request_df["output_text"].fillna("").astype(str) if "output_text" in request_df.columns else pd.Series([""] * len(request_df))
        non_empty = output_text.str.strip() != ""
        success = request_df["success"].astype(bool) if "success" in request_df.columns else pd.Series([False] * len(request_df))
        valid = success & non_empty

        return {
            "request_count": int(len(request_df)),
            "success_rate": float(success.mean()),
            "non_empty_rate": float(non_empty.mean()),
            "valid_response_rate": float(valid.mean()),
        }

    def _evaluate_pubhealth(self, request_df: pd.DataFrame) -> dict[str, Any]:
        if request_df.empty:
            return {
                "count": 0,
                "accuracy": 0.0,
                "macro_f1": 0.0,
            }

        expected = request_df["expected_output"].fillna("").astype(str).map(self._normalize_pubhealth_label)
        predicted = request_df["output_text"].fillna("").astype(str).map(self._normalize_pubhealth_prediction)
        valid_mask = expected != ""
        expected = expected[valid_mask]
        predicted = predicted[valid_mask]

        if expected.empty:
            return {
                "count": 0,
                "accuracy": 0.0,
                "macro_f1": 0.0,
            }

        accuracy = float((expected == predicted).mean())
        labels = sorted(set(expected) | set(predicted))
        f1_scores = []
        for label in labels:
            tp = int(((predicted == label) & (expected == label)).sum())
            fp = int(((predicted == label) & (expected != label)).sum())
            fn = int(((predicted != label) & (expected == label)).sum())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)

        return {
            "count": int(len(expected)),
            "accuracy": accuracy,
            "macro_f1": float(sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0,
        }

    @staticmethod
    def _normalize_pubhealth_label(value: str) -> str:
        normalized = value.strip().lower().replace("-", " ").replace("_", " ")
        normalized = " ".join(normalized.split())
        return normalized

    @classmethod
    def _normalize_pubhealth_prediction(cls, value: str) -> str:
        normalized = cls._normalize_pubhealth_label(value)
        for candidate in (
            "true",
            "false",
            "mixture",
            "mostly true",
            "mostly false",
            "unproven",
        ):
            if candidate in normalized:
                return candidate
        return normalized
