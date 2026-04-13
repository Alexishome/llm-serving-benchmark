from __future__ import annotations

import asyncio
import time
from typing import Any

from engines.base_engine import BaseEngine, EngineResponse


class HFLocalEngine(BaseEngine):
    """Local transformers-based generation baseline."""

    def __init__(
        self,
        model_name: str,
        device: str | int = "cuda",
        torch_dtype: str | None = "auto",
    ) -> None:
        super().__init__(engine_name="hf_transformers", model_name=model_name)
        self.device = device
        self.torch_dtype = torch_dtype
        self._pipeline = None

    async def startup(self) -> None:
        import torch
        from transformers import pipeline

        pipeline_device: str | int
        if self.device == "cpu":
            pipeline_device = -1
        elif self.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "HF baseline requested CUDA, but torch.cuda.is_available() is False. "
                    "This usually means the installed torch build is not compatible with the "
                    "current NVIDIA driver/runtime on the pod."
                )
            pipeline_device = 0
        else:
            pipeline_device = self.device

        self._pipeline = await asyncio.to_thread(
            pipeline,
            "text-generation",
            model=self.model_name,
            device=pipeline_device,
            torch_dtype=self.torch_dtype,
        )

    async def send_request(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        prompt_tokens: int,
        **kwargs: Any,
    ) -> EngineResponse:
        if self._pipeline is None:
            raise RuntimeError("Transformers pipeline not initialized. Call startup() first.")

        start_time = time.perf_counter()
        try:
            outputs = await asyncio.to_thread(
                self._pipeline,
                prompt,
                max_new_tokens=max_tokens,
                do_sample=kwargs.get("do_sample", False),
                temperature=kwargs.get("temperature", 0.0),
                return_full_text=False,
            )
            latency = time.perf_counter() - start_time
            output_text = outputs[0]["generated_text"]
            completion_tokens = max(1, len(output_text.split())) if output_text.strip() else 0

            return EngineResponse(
                request_id=request_id,
                engine=self.engine_name,
                model=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                start_time=start_time,
                ttft=None,
                latency=latency,
                output_text=output_text,
                success=True,
                metadata={"device": self.device},
            )
        except Exception as error:  # pragma: no cover - model/runtime dependent
            return self._error_response(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                start_time=start_time,
                error=error,
            )
