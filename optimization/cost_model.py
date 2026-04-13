from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


DEFAULT_TASK_WEIGHTS = {
    "question_answering": 0.9,
    "information_extraction": 1.0,
    "summarization": 1.2,
}


class SupportsCostFeatures(Protocol):
    input_tokens: int
    max_output_tokens: int
    task_type: str


@dataclass(frozen=True)
class CostModelConfig:
    input_weight: float = 1.0
    output_weight: float = 1.0
    kv_cache_weight: float = 0.0
    task_weights: dict[str, float] | None = None


class InferenceCostModel:
    """Lightweight, interpretable cost model for heterogeneous serving workloads."""

    def __init__(self, config: CostModelConfig | None = None) -> None:
        self.config = config or CostModelConfig()

    def predict(self, request: SupportsCostFeatures) -> float:
        task_weights = self.config.task_weights or DEFAULT_TASK_WEIGHTS
        task_weight = task_weights.get(request.task_type, 1.0)
        kv_cache_cost = request.input_tokens + request.max_output_tokens
        return float(
            task_weight
            * (
                self.config.input_weight * request.input_tokens
                + self.config.output_weight * request.max_output_tokens
                + self.config.kv_cache_weight * kv_cache_cost
            )
        )


def build_cost_model(config: dict[str, Any] | None = None) -> InferenceCostModel:
    raw_config = config or {}
    cost_model_config = CostModelConfig(
        input_weight=float(raw_config.get("input_weight", 1.0)),
        output_weight=float(raw_config.get("output_weight", 1.0)),
        kv_cache_weight=float(raw_config.get("kv_cache_weight", 0.0)),
        task_weights=raw_config.get("task_weights"),
    )
    return InferenceCostModel(config=cost_model_config)
