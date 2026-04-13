from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from workload.types import WorkloadRequest


@dataclass(frozen=True)
class PolicyConfig:
    name: str = "fifo"
    input_weight: float = 1.0
    output_weight: float = 1.0
    task_priorities: dict[str, int] | None = None


class BaseSchedulingPolicy:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    def priority(self, request: WorkloadRequest, arrival_index: int) -> tuple[Any, ...]:
        raise NotImplementedError


class FIFOPolicy(BaseSchedulingPolicy):
    def priority(self, request: WorkloadRequest, arrival_index: int) -> tuple[Any, ...]:
        return (arrival_index,)


class ShortestInputFirstPolicy(BaseSchedulingPolicy):
    def priority(self, request: WorkloadRequest, arrival_index: int) -> tuple[Any, ...]:
        return (request.input_tokens, request.max_output_tokens, arrival_index)


class PredictedCostPolicy(BaseSchedulingPolicy):
    def priority(self, request: WorkloadRequest, arrival_index: int) -> tuple[Any, ...]:
        score = float(
            request.metadata.get(
                "predicted_cost",
                self.config.input_weight * request.input_tokens
                + self.config.output_weight * request.max_output_tokens,
            )
        )
        return (score, request.input_tokens, request.max_output_tokens, arrival_index)


class TaskPriorityPolicy(BaseSchedulingPolicy):
    def priority(self, request: WorkloadRequest, arrival_index: int) -> tuple[Any, ...]:
        task_priorities = self.config.task_priorities or {}
        task_priority = task_priorities.get(request.task_type, 100)
        return (task_priority, request.input_tokens, arrival_index)


class HybridPolicy(BaseSchedulingPolicy):
    def priority(self, request: WorkloadRequest, arrival_index: int) -> tuple[Any, ...]:
        task_priorities = self.config.task_priorities or {}
        task_priority = task_priorities.get(request.task_type, 100)
        cost_score = float(
            request.metadata.get(
                "predicted_cost",
                self.config.input_weight * request.input_tokens
                + self.config.output_weight * request.max_output_tokens,
            )
        )
        return (task_priority, cost_score, arrival_index)


def build_policy(config: dict[str, Any] | None = None) -> BaseSchedulingPolicy:
    raw_config = config or {}
    name = str(raw_config.get("policy", "fifo")).lower()
    policy_config = PolicyConfig(
        name=name,
        input_weight=float(raw_config.get("input_weight", 1.0)),
        output_weight=float(raw_config.get("output_weight", 1.0)),
        task_priorities=raw_config.get("task_priorities"),
    )
    policies = {
        "fifo": FIFOPolicy,
        "shortest_input_first": ShortestInputFirstPolicy,
        "predicted_cost_first": PredictedCostPolicy,
        "task_priority": TaskPriorityPolicy,
        "hybrid": HybridPolicy,
    }
    if name not in policies:
        supported = ", ".join(sorted(policies))
        raise ValueError(f"Unsupported scheduler policy '{name}'. Expected one of: {supported}")
    return policies[name](policy_config)
