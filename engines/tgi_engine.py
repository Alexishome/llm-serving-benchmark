from __future__ import annotations

import json
import time
from typing import Any

import aiohttp

from engines.base_engine import BaseEngine, EngineResponse


class TGIEngine(BaseEngine):
    """Adapter for Hugging Face Text Generation Inference."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout_seconds: int = 300,
    ) -> None:
        super().__init__(engine_name="tgi", model_name=model_name)
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._session: aiohttp.ClientSession | None = None

    async def startup(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)

    async def shutdown(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def send_request(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        prompt_tokens: int,
        **kwargs: Any,
    ) -> EngineResponse:
        if self._session is None:
            raise RuntimeError("Engine session not initialized. Call startup() first.")

        start_time = time.perf_counter()
        ttft: float | None = None
        output_parts: list[str] = []
        completion_tokens = 0
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": kwargs.get("temperature", 0.0),
                "return_full_text": False,
            },
            "stream": True,
        }

        try:
            async with self._session.post(
                f"{self.base_url}/generate_stream",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for raw_chunk in response.content:
                    line = raw_chunk.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    parsed = json.loads(line.removeprefix("data:").strip())
                    token = parsed.get("token", {})
                    token_text = token.get("text", "")
                    if token_text and ttft is None:
                        ttft = time.perf_counter() - start_time
                    if token_text and not token.get("special", False):
                        output_parts.append(token_text)
                        completion_tokens += max(1, len(token_text.split()))

            latency = time.perf_counter() - start_time
            return EngineResponse(
                request_id=request_id,
                engine=self.engine_name,
                model=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                start_time=start_time,
                ttft=ttft,
                latency=latency,
                output_text="".join(output_parts),
                success=True,
                metadata={"base_url": self.base_url},
            )
        except Exception as error:  # pragma: no cover - network dependent
            return self._error_response(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                start_time=start_time,
                error=error,
                ttft=ttft,
            )
