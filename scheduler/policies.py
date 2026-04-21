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


class ServiceClassPriorityPolicy(BaseSchedulingPolicy):
    """Prioritize requests by workflow service class, then by predicted cost.

    This is the first bridge between the new control-plane metadata and the
    existing lane-level scheduler. Lower priority values run earlier.
    """

    DEFAULT_SERVICE_CLASS_PRIORITIES = {
        "interactive": 0,
        "verification": 1,
        "standard": 2,
        "standard_generation": 3,
        "cache_affine": 3,
        "batch_long_context": 4,
        "background": 5,
    }

    def priority(self, request: WorkloadRequest, arrival_index: int) -> tuple[Any, ...]:
        configured_priorities = self.config.task_priorities or {}
        priorities = {
            **self.DEFAULT_SERVICE_CLASS_PRIORITIES,
            **configured_priorities,
        }
        service_class = str(request.metadata.get("service_class") or "standard")
        service_priority = priorities.get(service_class, 100)
        cost_score = float(
            request.metadata.get(
                "predicted_cost",
                self.config.input_weight * request.input_tokens
                + self.config.output_weight * request.max_output_tokens,
            )
        )
        return (service_priority, cost_score, arrival_index)


class LongPrefillIsolationPolicy(BaseSchedulingPolicy):
    """Protect short interactive traffic from long-prefill / high-risk requests.

    This policy is intentionally route-aware. It uses metadata from the
    control-plane profiler/router and defers long-context clinical traffic
    behind standard, cache-affine, and interactive requests. Within each lane,
    it still sorts by predicted cost to avoid unnecessarily expensive requests
    blocking cheaper requests of the same class.
    """

    DEFAULT_ROUTE_PRIORITIES = {
        "standard": 0,
        "cache_affine": 1,
        "long_prefill": 3,
        "high_risk": 4,
    }
    DEFAULT_SERVICE_CLASS_PRIORITIES = {
        "interactive": 0,
        "verification": 0,
        "standard": 1,
        "standard_generation": 2,
        "cache_affine": 2,
        "batch_long_context": 4,
        "background": 5,
    }

    def priority(self, request: WorkloadRequest, arrival_index: int) -> tuple[Any, ...]:
        route_priorities = {
            **self.DEFAULT_ROUTE_PRIORITIES,
            **(self.config.task_priorities or {}),
        }
        route_name = str(request.metadata.get("route_name") or "standard")
        service_class = str(request.metadata.get("service_class") or "standard")
        route_priority = route_priorities.get(route_name, 100)
        service_priority = self.DEFAULT_SERVICE_CLASS_PRIORITIES.get(service_class, 100)
        risk_score = float(request.metadata.get("profile_quality_risk_score") or 0.0)
        prefill_cost = float(
            request.metadata.get("profile_prefill_cost") or request.input_tokens
        )
        cost_score = float(
            request.metadata.get(
                "predicted_cost",
                self.config.input_weight * request.input_tokens
                + self.config.output_weight * request.max_output_tokens,
            )
        )
        return (
            route_priority,
            service_priority,
            risk_score,
            prefill_cost,
            cost_score,
            arrival_index,
        )


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
        "service_class_priority": ServiceClassPriorityPolicy,
        "long_prefill_isolation": LongPrefillIsolationPolicy,
        "task_priority": TaskPriorityPolicy,
        "hybrid": HybridPolicy,
    }
    if name not in policies:
        supported = ", ".join(sorted(policies))
        raise ValueError(f"Unsupported scheduler policy '{name}'. Expected one of: {supported}")
    return policies[name](policy_config)
