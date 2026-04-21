from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optimization.cost_model import InferenceCostModel, build_cost_model
from workload.types import WorkloadRequest


@dataclass(frozen=True)
class ProfilerConfig:
    long_input_threshold: int = 1800
    long_output_threshold: int = 256
    cacheable_prefix_length: int = 512
    cacheable_task_families: tuple[str, ...] = (
        "clinical_summarization",
        "evidence_summarization",
        "document_classification",
    )


class ClinicalRequestProfiler:
    """Lightweight request profiler for control-plane decisions.

    This first version is intentionally heuristic:
    - it reuses the existing interpretable cost model
    - it adds cheap task-aware features
    - it can be replaced later with a learned predictor
    """

    def __init__(
        self,
        cost_model_config: dict[str, Any] | None = None,
        profiler_config: ProfilerConfig | None = None,
    ) -> None:
        self.cost_model: InferenceCostModel = build_cost_model(cost_model_config)
        self.config = profiler_config or ProfilerConfig()

    def profile(self, request: WorkloadRequest):
        from control_plane.types import RequestProfile

        task_family = str(
            request.metadata.get("task_family")
            or request.metadata.get("dataset_name")
            or request.task_type
        )
        workflow_stage = self._workflow_stage(request, task_family)
        service_class = self._service_class(request, task_family, workflow_stage)
        prefill_cost = float(request.input_tokens)
        decode_cost = float(request.max_output_tokens)
        total_cost = float(self.cost_model.predict(request))
        cache_affinity_score = self._cache_affinity_score(request, task_family)
        quality_risk_score = self._quality_risk_score(request, task_family)

        return RequestProfile(
            request_id=request.request_id,
            task_family=task_family,
            workflow_stage=workflow_stage,
            service_class=service_class,
            prefill_cost=prefill_cost,
            decode_cost=decode_cost,
            total_cost=total_cost,
            cache_affinity_score=cache_affinity_score,
            quality_risk_score=quality_risk_score,
            metadata={
                "input_tokens": request.input_tokens,
                "max_output_tokens": request.max_output_tokens,
                "task_type": request.task_type,
                "dataset_name": request.metadata.get("dataset_name"),
                "benchmark": request.metadata.get("benchmark"),
                "workflow_stage": workflow_stage,
                "service_class": service_class,
            },
        )

    def enrich_request_metadata(self, request: WorkloadRequest) -> WorkloadRequest:
        profile = self.profile(request)
        metadata = dict(request.metadata)
        metadata.update(
            {
                "profile_task_family": profile.task_family,
                "workflow_stage": profile.workflow_stage,
                "service_class": profile.service_class,
                "profile_prefill_cost": profile.prefill_cost,
                "profile_decode_cost": profile.decode_cost,
                "profile_total_cost": profile.total_cost,
                "profile_cache_affinity_score": profile.cache_affinity_score,
                "profile_quality_risk_score": profile.quality_risk_score,
            }
        )
        return WorkloadRequest(
            request_id=request.request_id,
            prompt=request.prompt,
            input_tokens=request.input_tokens,
            max_output_tokens=request.max_output_tokens,
            task_type=request.task_type,
            metadata=metadata,
        )

    def _cache_affinity_score(self, request: WorkloadRequest, task_family: str) -> float:
        score = 0.0
        if request.input_tokens >= self.config.cacheable_prefix_length:
            score += 0.4
        if task_family in self.config.cacheable_task_families:
            score += 0.3
        if request.metadata.get("subjects"):
            score += 0.1
        if request.metadata.get("benchmark") == "acl":
            score += 0.1
        return min(1.0, score)

    def _quality_risk_score(self, request: WorkloadRequest, task_family: str) -> float:
        score = 0.0
        if request.input_tokens >= self.config.long_input_threshold:
            score += 0.4
        if request.max_output_tokens >= self.config.long_output_threshold:
            score += 0.2
        if task_family in {"clinical_summarization", "evidence_summarization"}:
            score += 0.3
        if task_family in {"named_entity_recognition", "relation_extraction"}:
            score += 0.1
        return min(1.0, score)

    def _workflow_stage(self, request: WorkloadRequest, task_family: str) -> str:
        explicit_stage = request.metadata.get("workflow_stage")
        if explicit_stage:
            return str(explicit_stage)

        dataset_name = str(request.metadata.get("dataset_name") or "").lower()
        if task_family in {"clinical_summarization", "evidence_summarization"}:
            return "summarization"
        if dataset_name == "pubhealth" or task_family == "fact_checking":
            return "verification"
        if task_family in {"relation_extraction", "named_entity_recognition"}:
            return "extraction"
        if task_family == "sentence_similarity":
            return "reranking"
        if task_family == "document_classification":
            return "classification"
        return "generation"

    def _service_class(
        self,
        request: WorkloadRequest,
        task_family: str,
        workflow_stage: str,
    ) -> str:
        explicit_class = request.metadata.get("service_class")
        if explicit_class:
            return str(explicit_class)

        if workflow_stage in {"verification", "classification", "extraction", "reranking"}:
            return "interactive"
        if task_family == "clinical_summarization" and request.input_tokens >= self.config.long_input_threshold:
            return "batch_long_context"
        if workflow_stage == "summarization":
            return "standard_generation"
        return "standard"
