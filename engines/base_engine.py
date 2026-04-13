from __future__ import annotations

import abc
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EngineResponse:
    request_id: str
    engine: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    start_time: float
    ttft: float | None
    latency: float
    output_text: str
    success: bool
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseEngine(abc.ABC):
    def __init__(self, engine_name: str, model_name: str) -> None:
        self.engine_name = engine_name
        self.model_name = model_name

    @abc.abstractmethod
    async def send_request(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        prompt_tokens: int,
        **kwargs: Any,
    ) -> EngineResponse:
        raise NotImplementedError

    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    def _error_response(
        self,
        request_id: str,
        prompt_tokens: int,
        start_time: float,
        error: Exception,
        ttft: float | None = None,
    ) -> EngineResponse:
        latency = time.perf_counter() - start_time
        return EngineResponse(
            request_id=request_id,
            engine=self.engine_name,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            total_tokens=prompt_tokens,
            start_time=start_time,
            ttft=ttft,
            latency=latency,
            output_text="",
            success=False,
            error=str(error),
        )
