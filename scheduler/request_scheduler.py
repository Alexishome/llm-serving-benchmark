from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from engines.base_engine import BaseEngine, EngineResponse
from scheduler.policies import BaseSchedulingPolicy, build_policy
from workload.types import WorkloadRequest


@dataclass
class ScheduledRequest:
    request: WorkloadRequest
    arrival_index: int
    planned_arrival: float


class RequestScheduler:
    """Online request scheduler that dispatches according to a configurable policy."""

    def __init__(self, engine: BaseEngine, scheduler_config: dict[str, Any] | None = None) -> None:
        self.engine = engine
        self.config = scheduler_config or {}
        self.policy: BaseSchedulingPolicy = build_policy(self.config)

    async def execute(
        self,
        workload: list[WorkloadRequest],
        concurrency: int,
        request_rate: float,
        batch_size: int,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> list[EngineResponse]:
        engine_kwargs = engine_kwargs or {}
        queue: asyncio.PriorityQueue[tuple[tuple[Any, ...], int, ScheduledRequest | None]] = (
            asyncio.PriorityQueue()
        )
        responses: list[EngineResponse] = []
        response_lock = asyncio.Lock()
        start_time = time.perf_counter()
        worker_count = max(1, concurrency)
        interval_seconds = (1.0 / request_rate) if request_rate > 0 else 0.0

        async def producer() -> None:
            for arrival_index, request in enumerate(workload):
                planned_arrival = arrival_index * interval_seconds
                now = time.perf_counter() - start_time
                delay = max(0.0, planned_arrival - now)
                if delay > 0:
                    await asyncio.sleep(delay)

                scheduled_request = ScheduledRequest(
                    request=request,
                    arrival_index=arrival_index,
                    planned_arrival=planned_arrival,
                )
                priority = self.policy.priority(request, arrival_index)
                await queue.put((priority, arrival_index, scheduled_request))

            for sentinel_index in range(worker_count):
                await queue.put(((float("inf"),), len(workload) + sentinel_index, None))

        async def worker() -> None:
            while True:
                _, _, scheduled = await queue.get()
                try:
                    if scheduled is None:
                        return

                    dispatch_started = time.perf_counter()
                    response = await self.engine.send_request(
                        request_id=scheduled.request.request_id,
                        prompt=scheduled.request.prompt,
                        max_tokens=scheduled.request.max_output_tokens,
                        prompt_tokens=scheduled.request.input_tokens,
                        batch_size=batch_size,
                        task_type=scheduled.request.task_type,
                        **engine_kwargs,
                    )
                    response.metadata.update(
                        {
                            "scheduler_policy": self.policy.name,
                            "queue_delay": dispatch_started
                            - (start_time + scheduled.planned_arrival),
                            "planned_arrival": scheduled.planned_arrival,
                            "dispatch_offset": dispatch_started - start_time,
                            "task_type": scheduled.request.task_type,
                            "input_bucket": scheduled.request.metadata.get("input_bucket"),
                            "output_bucket": scheduled.request.metadata.get("output_bucket"),
                            "predicted_cost": scheduled.request.metadata.get("predicted_cost", 0.0),
                        }
                    )
                    async with response_lock:
                        responses.append(response)
                finally:
                    queue.task_done()

        producer_task = asyncio.create_task(producer())
        workers = [asyncio.create_task(worker()) for _ in range(worker_count)]

        await producer_task
        await queue.join()
        await asyncio.gather(*workers)
        return responses
