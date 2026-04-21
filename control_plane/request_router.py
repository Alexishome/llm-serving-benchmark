from __future__ import annotations

from dataclasses import dataclass

from control_plane.types import RequestProfile, RouteDecision


@dataclass(frozen=True)
class RouterConfig:
    high_risk_threshold: float = 0.65
    long_prefill_threshold: float = 1800.0
    long_decode_threshold: float = 256.0
    high_cache_affinity_threshold: float = 0.6


class HeuristicRequestRouter:
    """First router prototype for the control plane.

    Routes requests into a small number of conceptual serving lanes:
    - standard
    - long_prefill
    - high_risk

    This does not yet require multiple physical engines. It can be used
    immediately to:
    - label requests with route decisions
    - evaluate whether a multi-lane control plane would make sense
    """

    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()

    def route(self, profile: RequestProfile) -> RouteDecision:
        if profile.quality_risk_score >= self.config.high_risk_threshold:
            return RouteDecision(
                route_name="high_risk",
                reason="quality_risk_score exceeded threshold",
                score=profile.quality_risk_score,
                profile=profile,
            )

        if (
            profile.prefill_cost >= self.config.long_prefill_threshold
            or profile.decode_cost >= self.config.long_decode_threshold
        ):
            return RouteDecision(
                route_name="long_prefill",
                reason="prefill/decode cost exceeded long-request threshold",
                score=max(profile.prefill_cost, profile.decode_cost),
                profile=profile,
            )

        if profile.cache_affinity_score >= self.config.high_cache_affinity_threshold:
            return RouteDecision(
                route_name="cache_affine",
                reason="cache_affinity_score exceeded threshold",
                score=profile.cache_affinity_score,
                profile=profile,
            )

        return RouteDecision(
            route_name="standard",
            reason="default route",
            score=profile.total_cost,
            profile=profile,
        )
