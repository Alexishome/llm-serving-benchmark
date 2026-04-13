from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from engines.base_engine import BaseEngine, EngineResponse
from metrics.gpu_monitor import GPUMonitor
from metrics.metrics_collector import MetricsCollector
from scheduler.request_scheduler import RequestScheduler
from workload.types import WorkloadRequest


@dataclass
class TrialResult:
    request_df: pd.DataFrame
    summary: dict[str, Any]
    responses: list[EngineResponse]
    duration_seconds: float


class ExperimentRunner:
    def __init__(
        self,
        engine: BaseEngine,
        metrics_collector: MetricsCollector,
        gpu_monitor: GPUMonitor | None = None,
        scheduler_config: dict[str, Any] | None = None,
    ) -> None:
        self.engine = engine
        self.metrics_collector = metrics_collector
        self.gpu_monitor = gpu_monitor
        self.scheduler = RequestScheduler(engine=engine, scheduler_config=scheduler_config)

    async def run_trial(
        self,
        workload: list[WorkloadRequest],
        concurrency: int,
        request_rate: float,
        batch_size: int,
        trial_index: int,
        engine_kwargs: dict[str, Any] | None = None,
        cost_per_1m_tokens: float | None = None,
    ) -> TrialResult:
        engine_kwargs = engine_kwargs or {}
        trial_start = time.perf_counter()
        gpu_sample_start = self.gpu_monitor.sample_count if self.gpu_monitor is not None else 0
        responses = await self.scheduler.execute(
            workload=workload,
            concurrency=concurrency,
            request_rate=request_rate,
            batch_size=batch_size,
            engine_kwargs=engine_kwargs,
        )
        duration_seconds = time.perf_counter() - trial_start
        trial_metadata = {
            "trial": trial_index,
            "engine": self.engine.engine_name,
            "model": self.engine.model_name,
            "concurrency": concurrency,
            "request_rate": request_rate,
            "batch_size": batch_size,
            "scheduler_policy": self.scheduler.policy.name,
        }
        request_df = self.metrics_collector.collect_request_records(
            responses=responses,
            trial_metadata=trial_metadata,
        )
        gpu_samples = (
            self.gpu_monitor.samples[gpu_sample_start:] if self.gpu_monitor is not None else []
        )
        summary = self.metrics_collector.summarize_experiment(
            request_df=request_df,
            gpu_samples=gpu_samples,
            trial_metadata=trial_metadata,
            trial_duration_seconds=duration_seconds,
            cost_per_1m_tokens=cost_per_1m_tokens,
        )
        return TrialResult(
            request_df=request_df,
            summary=summary.to_dict(),
            responses=responses,
            duration_seconds=duration_seconds,
        )
