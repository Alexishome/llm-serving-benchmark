from __future__ import annotations

import json
import math
import re
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

        numeric_df = self._numeric_label_subset(request_df)
        if not numeric_df.empty:
            results["numeric_regression"] = self._evaluate_numeric_regression(numeric_df)

        mimic_df = self._subset_dataset(request_df, "mimic_bhc")
        if not mimic_df.empty:
            results["mimic_bhc"] = self._evaluate_summary_overlap(mimic_df)

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

    @staticmethod
    def _numeric_label_subset(request_df: pd.DataFrame) -> pd.DataFrame:
        if "expected_output" not in request_df.columns or "output_text" not in request_df.columns:
            return pd.DataFrame()
        df = request_df.copy()
        if "task_family" in df.columns:
            task_family = df["task_family"].fillna("").astype(str).str.lower()
            df = df[task_family.isin({"sentence_similarity", "semantic_similarity"})].copy()
        elif "dataset_name" in df.columns:
            dataset_name = df["dataset_name"].fillna("").astype(str).str.lower()
            df = df[dataset_name.isin({"biosses"})].copy()
        else:
            return pd.DataFrame()
        if df.empty:
            return df
        expected = df["expected_output"].fillna("").astype(str).map(QualityEvaluator._parse_first_float)
        predicted = df["output_text"].fillna("").astype(str).map(QualityEvaluator._parse_first_float)
        df["expected_numeric"] = expected
        df["predicted_numeric"] = predicted
        return df[df["expected_numeric"].notna() & df["predicted_numeric"].notna()].copy()

    @staticmethod
    def _parse_first_float(value: str) -> float | None:
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    def _evaluate_numeric_regression(self, request_df: pd.DataFrame) -> dict[str, Any]:
        expected = request_df["expected_numeric"].astype(float)
        predicted = request_df["predicted_numeric"].astype(float)
        errors = predicted - expected
        abs_errors = errors.abs()
        squared_errors = errors.pow(2)

        return {
            "count": int(len(request_df)),
            "mae": float(abs_errors.mean()) if len(request_df) else 0.0,
            "rmse": float(math.sqrt(squared_errors.mean())) if len(request_df) else 0.0,
            "pearson": self._safe_corr(expected, predicted),
            "within_0_5_rate": float((abs_errors <= 0.5).mean()) if len(request_df) else 0.0,
            "within_1_0_rate": float((abs_errors <= 1.0).mean()) if len(request_df) else 0.0,
        }

    @staticmethod
    def _safe_corr(left: pd.Series, right: pd.Series) -> float:
        if len(left) < 2:
            return 0.0
        value = left.corr(right)
        if pd.isna(value):
            return 0.0
        return float(value)

    def _evaluate_summary_overlap(self, request_df: pd.DataFrame) -> dict[str, Any]:
        if "expected_output" not in request_df.columns or "output_text" not in request_df.columns:
            return {
                "count": 0,
                "rouge_l_f1_mean": 0.0,
                "output_to_reference_length_ratio_mean": 0.0,
            }

        rows = request_df[["expected_output", "output_text"]].fillna("").astype(str)
        scores = []
        length_ratios = []
        for _, row in rows.iterrows():
            reference_tokens = self._tokenize_for_overlap(row["expected_output"])
            output_tokens = self._tokenize_for_overlap(row["output_text"])
            if reference_tokens and output_tokens:
                scores.append(self._rouge_l_f1(reference_tokens, output_tokens))
                length_ratios.append(len(output_tokens) / len(reference_tokens))
            else:
                scores.append(0.0)
                length_ratios.append(0.0)

        return {
            "count": int(len(rows)),
            "rouge_l_f1_mean": float(sum(scores) / len(scores)) if scores else 0.0,
            "output_to_reference_length_ratio_mean": float(sum(length_ratios) / len(length_ratios))
            if length_ratios
            else 0.0,
        }

    @staticmethod
    def _tokenize_for_overlap(value: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", value.lower())

    @staticmethod
    def _rouge_l_f1(reference_tokens: list[str], output_tokens: list[str]) -> float:
        lcs = QualityEvaluator._lcs_length(reference_tokens, output_tokens)
        if lcs == 0:
            return 0.0
        precision = lcs / len(output_tokens)
        recall = lcs / len(reference_tokens)
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    @staticmethod
    def _lcs_length(left: list[str], right: list[str]) -> int:
        if len(left) < len(right):
            short, long = left, right
        else:
            short, long = right, left
        previous = [0] * (len(short) + 1)
        for token in long:
            current = [0]
            for index, short_token in enumerate(short, start=1):
                if token == short_token:
                    current.append(previous[index - 1] + 1)
                else:
                    current.append(max(previous[index], current[-1]))
            previous = current
        return previous[-1]

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
