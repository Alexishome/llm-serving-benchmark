from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from engines.base_engine import EngineResponse
from metrics.gpu_monitor import GPUMonitor, GPUSample


@dataclass
class ExperimentSummary:
    engine: str
    model: str
    trial: int
    concurrency: int
    request_rate: float
    batch_size: int
    scheduler_policy: str
    input_tokens: float
    output_tokens: float
    latency_mean: float
    latency_std: float
    latency_p95: float
    latency_p99: float
    ttft_mean: float | None
    ttft_std: float | None
    tokens_per_second_mean: float
    throughput_rps: float
    success_rate: float
    error_rate: float
    queue_delay_mean: float
    queue_delay_std: float
    queue_delay_p95: float
    non_empty_rate: float
    valid_response_rate: float
    predicted_cost_mean: float
    predicted_cost_total: float
    cost_efficiency_tokens_per_cost: float | None
    cost_efficiency_requests_per_cost: float | None
    gpu_utilization_mean: float | None
    gpu_memory_mb_mean: float | None
    cost_per_1m_tokens: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MetricsCollector:
    def __init__(self) -> None:
        self.request_records: list[dict[str, Any]] = []
        self.summary_records: list[dict[str, Any]] = []

    def collect_request_records(
        self,
        responses: list[EngineResponse],
        trial_metadata: dict[str, Any],
    ) -> pd.DataFrame:
        records = []
        for response in responses:
            tokens_per_second = (
                response.completion_tokens / response.latency if response.latency > 0 else 0.0
            )
            record = {
                **trial_metadata,
                "request_id": response.request_id,
                "engine": response.engine,
                "model": response.model,
                "input_tokens": response.prompt_tokens,
                "output_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
                "latency": response.latency,
                "ttft": response.ttft,
                "tokens_per_second": tokens_per_second,
                "success": response.success,
                "error": response.error,
                "output_text": response.output_text,
                "queue_delay": response.metadata.get("queue_delay", 0.0),
                "scheduler_policy": response.metadata.get(
                    "scheduler_policy",
                    trial_metadata.get("scheduler_policy", "fifo"),
                ),
                "task_type": response.metadata.get("task_type"),
                "input_bucket": response.metadata.get("input_bucket"),
                "output_bucket": response.metadata.get("output_bucket"),
                "predicted_cost": float(response.metadata.get("predicted_cost", 0.0)),
                "benchmark": response.metadata.get("benchmark"),
                "dataset_name": response.metadata.get("dataset_name"),
                "task_family": response.metadata.get("task_family"),
                "expected_output": response.metadata.get("expected_output"),
            }
            self.request_records.append(record)
            records.append(record)
        return pd.DataFrame(records)

    def summarize_experiment(
        self,
        request_df: pd.DataFrame,
        gpu_samples: list[GPUSample],
        trial_metadata: dict[str, Any],
        trial_duration_seconds: float,
        cost_per_1m_tokens: float | None = None,
    ) -> ExperimentSummary:
        success_df = request_df[request_df["success"] == True]  # noqa: E712
        success_rate = float(request_df["success"].mean()) if not request_df.empty else 0.0
        error_rate = 1.0 - success_rate
        throughput_rps = (len(success_df) / trial_duration_seconds) if trial_duration_seconds > 0 else 0.0
        ttft_series = success_df["ttft"].dropna()
        queue_delay_series = request_df["queue_delay"] if "queue_delay" in request_df.columns else pd.Series(dtype=float)
        predicted_cost_series = (
            request_df["predicted_cost"] if "predicted_cost" in request_df.columns else pd.Series(dtype=float)
        )
        non_empty_series = (
            request_df["output_text"].fillna("").astype(str).str.strip() != ""
            if "output_text" in request_df.columns
            else pd.Series(dtype=bool)
        )
        valid_response_series = (
            request_df["success"].astype(bool) & non_empty_series
            if not request_df.empty and not non_empty_series.empty
            else pd.Series(dtype=bool)
        )

        if cost_per_1m_tokens is not None:
            total_tokens = float(success_df["total_tokens"].sum())
            normalized_cost = (total_tokens / 1_000_000.0) * cost_per_1m_tokens
        else:
            normalized_cost = None

        predicted_cost_total = float(predicted_cost_series.sum()) if not predicted_cost_series.empty else 0.0
        cost_efficiency_tokens_per_cost = (
            float(success_df["total_tokens"].sum()) / predicted_cost_total
            if predicted_cost_total > 0
            else None
        )
        cost_efficiency_requests_per_cost = (
            float(len(success_df)) / predicted_cost_total if predicted_cost_total > 0 else None
        )

        gpu_utilization_mean = (
            sum(sample.utilization_gpu for sample in gpu_samples) / len(gpu_samples)
            if gpu_samples
            else None
        )
        gpu_memory_mb_mean = (
            sum(sample.memory_used_mb for sample in gpu_samples) / len(gpu_samples)
            if gpu_samples
            else None
        )

        summary = ExperimentSummary(
            engine=str(trial_metadata["engine"]),
            model=str(trial_metadata["model"]),
            trial=int(trial_metadata["trial"]),
            concurrency=int(trial_metadata["concurrency"]),
            request_rate=float(trial_metadata["request_rate"]),
            batch_size=int(trial_metadata["batch_size"]),
            scheduler_policy=str(trial_metadata.get("scheduler_policy", "fifo")),
            input_tokens=float(request_df["input_tokens"].mean()) if not request_df.empty else 0.0,
            output_tokens=float(request_df["output_tokens"].mean()) if not request_df.empty else 0.0,
            latency_mean=float(success_df["latency"].mean()) if not success_df.empty else 0.0,
            latency_std=float(success_df["latency"].std(ddof=0)) if not success_df.empty else 0.0,
            latency_p95=float(success_df["latency"].quantile(0.95)) if not success_df.empty else 0.0,
            latency_p99=float(success_df["latency"].quantile(0.99)) if not success_df.empty else 0.0,
            ttft_mean=float(ttft_series.mean()) if not ttft_series.empty else None,
            ttft_std=float(ttft_series.std(ddof=0)) if not ttft_series.empty else None,
            tokens_per_second_mean=(
                float(success_df["tokens_per_second"].mean()) if not success_df.empty else 0.0
            ),
            throughput_rps=throughput_rps,
            success_rate=success_rate,
            error_rate=error_rate,
            queue_delay_mean=float(queue_delay_series.mean()) if not queue_delay_series.empty else 0.0,
            queue_delay_std=float(queue_delay_series.std(ddof=0)) if not queue_delay_series.empty else 0.0,
            queue_delay_p95=float(queue_delay_series.quantile(0.95)) if not queue_delay_series.empty else 0.0,
            non_empty_rate=float(non_empty_series.mean()) if not non_empty_series.empty else 0.0,
            valid_response_rate=float(valid_response_series.mean()) if not valid_response_series.empty else 0.0,
            predicted_cost_mean=float(predicted_cost_series.mean()) if not predicted_cost_series.empty else 0.0,
            predicted_cost_total=predicted_cost_total,
            cost_efficiency_tokens_per_cost=cost_efficiency_tokens_per_cost,
            cost_efficiency_requests_per_cost=cost_efficiency_requests_per_cost,
            gpu_utilization_mean=gpu_utilization_mean,
            gpu_memory_mb_mean=gpu_memory_mb_mean,
            cost_per_1m_tokens=normalized_cost,
        )
        summary_dict = summary.to_dict()
        self.summary_records.append(summary_dict)
        return summary

    def save_results(
        self,
        request_output_csv: str | Path,
        summary_output_csv: str | Path,
        request_output_json: str | Path,
        summary_output_json: str | Path,
    ) -> None:
        request_df = pd.DataFrame(self.request_records)
        summary_df = pd.DataFrame(self.summary_records)

        for path in [request_output_csv, summary_output_csv, request_output_json, summary_output_json]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        request_df.to_csv(request_output_csv, index=False)
        summary_df.to_csv(summary_output_csv, index=False)
        Path(request_output_json).write_text(
            json.dumps(self.request_records, indent=2),
            encoding="utf-8",
        )
        Path(summary_output_json).write_text(
            json.dumps(self.summary_records, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def aggregate_trials(summary_df: pd.DataFrame) -> pd.DataFrame:
        group_cols = ["engine", "model", "concurrency", "request_rate", "batch_size", "scheduler_policy"]
        numeric_cols = [
            "input_tokens",
            "output_tokens",
            "latency_mean",
            "latency_p95",
            "latency_p99",
            "ttft_mean",
            "tokens_per_second_mean",
            "throughput_rps",
            "success_rate",
            "error_rate",
            "queue_delay_mean",
            "queue_delay_p95",
            "non_empty_rate",
            "valid_response_rate",
            "predicted_cost_mean",
            "predicted_cost_total",
            "cost_efficiency_tokens_per_cost",
            "cost_efficiency_requests_per_cost",
            "gpu_utilization_mean",
            "gpu_memory_mb_mean",
            "cost_per_1m_tokens",
        ]
        available_numeric_cols = [column for column in numeric_cols if column in summary_df.columns]
        if summary_df.empty:
            return summary_df

        aggregated = (
            summary_df.groupby(group_cols)[available_numeric_cols]
            .agg(["mean", "std"])
            .reset_index()
        )
        aggregated.columns = [
            "_".join(col).strip("_") if isinstance(col, tuple) else col for col in aggregated.columns
        ]
        return aggregated
