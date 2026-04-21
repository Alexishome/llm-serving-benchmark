from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RequestProfile:
    request_id: str
    task_family: str
    workflow_stage: str
    service_class: str
    prefill_cost: float
    decode_cost: float
    total_cost: float
    cache_affinity_score: float
    quality_risk_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteDecision:
    route_name: str
    reason: str
    score: float
    profile: RequestProfile
