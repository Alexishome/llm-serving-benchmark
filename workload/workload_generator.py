from __future__ import annotations

import random
from typing import Any

from control_plane.request_profiler import ClinicalRequestProfiler
from control_plane.request_router import HeuristicRequestRouter
from optimization.cost_model import build_cost_model
from workload.benchmark_adapters import ACLWorkloadAdapter, BLUEWorkloadAdapter, LEvalWorkloadAdapter
from workload.dataset_loader import DatasetLoader
from workload.types import WorkloadRequest


TASK_TEMPLATES = {
    "summarization": "Summarize the following clinical note:\n\n{body}",
    "question_answering": (
        "Read the clinical context and answer the question clearly.\n\n"
        "Context:\n{body}\n\nQuestion: What are the key clinical findings?"
    ),
    "information_extraction": (
        "Extract structured clinical entities from the note below.\n\n{body}\n\n"
        "Return medications, diagnoses, labs, and procedures."
    ),
}

class WorkloadGenerator:
    """Build dataset-backed or synthetic benchmark requests."""

    def __init__(self, seed: int | None = None) -> None:
        self._random = random.Random(seed)
        self._dataset_loader = DatasetLoader()
        self._tokenizer = None
        self._tokenizer_name: str | None = None
        self._cost_model = build_cost_model()
        self._profiler = ClinicalRequestProfiler()
        self._router = HeuristicRequestRouter()

    def generate(self, config: dict[str, Any]) -> list[WorkloadRequest]:
        self._configure_tokenizer(config)
        self._configure_cost_model(config)
        mode = config.get("mode", "synthetic")
        if mode == "leval":
            return self._finalize_requests(self._load_leval(config))
        if mode == "blue":
            return self._finalize_requests(self._load_blue(config))
        if mode == "pubhealth":
            return self._finalize_requests(self._load_pubhealth(config))
        if mode == "cochrane":
            return self._finalize_requests(self._load_cochrane(config))
        if mode == "mimic_bhc":
            return self._finalize_requests(self._load_mimic_bhc(config))
        if mode == "mixed_clinical":
            return self._finalize_requests(self._load_mixed_clinical(config))
        if mode == "dataset":
            return self._finalize_requests(self._from_dataset(config))
        return self._finalize_requests(self._synthetic(config))

    def _from_dataset(self, config: dict[str, Any]) -> list[WorkloadRequest]:
        dataset_path = config["dataset_path"]
        records = self._dataset_loader.load(dataset_path)
        num_requests = int(config.get("num_requests", len(records)))
        requests: list[WorkloadRequest] = []

        for index in range(num_requests):
            record = records[index % len(records)]
            prompt = record["prompt"]
            input_tokens = self._estimate_tokens(prompt)
            max_output_tokens = int(record.get("max_tokens", 128))
            metadata = dict(record.get("metadata", {}))
            request_stub = WorkloadRequest(
                request_id=f"req-{index:05d}",
                prompt=prompt,
                input_tokens=input_tokens,
                max_output_tokens=max_output_tokens,
                task_type=str(record.get("task_type", "summarization")),
                metadata=metadata,
            )
            metadata["predicted_cost"] = self._cost_model.predict(request_stub)
            requests.append(
                WorkloadRequest(
                    request_id=request_stub.request_id,
                    prompt=request_stub.prompt,
                    input_tokens=request_stub.input_tokens,
                    max_output_tokens=request_stub.max_output_tokens,
                    task_type=request_stub.task_type,
                    metadata=metadata,
                )
            )
        return requests

    def _synthetic(self, config: dict[str, Any]) -> list[WorkloadRequest]:
        num_requests = int(config.get("num_requests", 100))
        task_types = config.get(
            "task_types",
            ["summarization", "question_answering", "information_extraction"],
        )
        input_buckets = config.get(
            "input_length_distribution",
            {"short": [128, 512], "medium": [512, 2048], "long": [2048, 4096]},
        )
        output_buckets = config.get(
            "output_length_distribution",
            {"short": [32, 128], "medium": [128, 256], "long": [256, 512]},
        )
        bucket_weights = config.get(
            "bucket_weights",
            {"short": 0.4, "medium": 0.4, "long": 0.2},
        )

        requests: list[WorkloadRequest] = []
        for index in range(num_requests):
            task_type = self._random.choice(task_types)
            input_bucket = self._weighted_bucket_choice(bucket_weights)
            output_bucket = self._weighted_bucket_choice(bucket_weights)
            input_tokens = self._random.randint(*input_buckets[input_bucket])
            output_tokens = self._random.randint(*output_buckets[output_bucket])
            prompt = self._build_synthetic_prompt(task_type=task_type, target_tokens=input_tokens)
            metadata = {"input_bucket": input_bucket, "output_bucket": output_bucket}
            request_stub = WorkloadRequest(
                request_id=f"req-{index:05d}",
                prompt=prompt,
                input_tokens=input_tokens,
                max_output_tokens=output_tokens,
                task_type=task_type,
                metadata=metadata,
            )
            metadata["predicted_cost"] = self._cost_model.predict(request_stub)

            requests.append(
                WorkloadRequest(
                    request_id=request_stub.request_id,
                    prompt=request_stub.prompt,
                    input_tokens=request_stub.input_tokens,
                    max_output_tokens=request_stub.max_output_tokens,
                    task_type=request_stub.task_type,
                    metadata=metadata,
                )
            )
        return requests

    def _load_leval(self, config: dict[str, Any]) -> list[WorkloadRequest]:
        root_dir = config.get("dataset_path", "data/leval")
        adapter = LEvalWorkloadAdapter(token_counter=self._estimate_tokens)
        requests = adapter.load(root_dir)
        num_requests = int(config.get("num_requests", len(requests)))
        return requests[:num_requests]

    def _load_blue(self, config: dict[str, Any]) -> list[WorkloadRequest]:
        root_dir = config.get("dataset_path", "data/blue_data")
        adapter = BLUEWorkloadAdapter(token_counter=self._estimate_tokens)
        requests = adapter.load(root_dir)
        num_requests = int(config.get("num_requests", len(requests)))
        return requests[:num_requests]

    def _load_pubhealth(self, config: dict[str, Any]) -> list[WorkloadRequest]:
        root_dir = config.get("dataset_path", "data/data acl")
        adapter = ACLWorkloadAdapter(token_counter=self._estimate_tokens)
        num_requests = int(config.get("num_requests", 0)) or None
        requests = adapter.load_pubhealth(root_dir, limit=num_requests)
        return requests if num_requests is None else requests[:num_requests]

    def _load_cochrane(self, config: dict[str, Any]) -> list[WorkloadRequest]:
        root_dir = config.get("dataset_path", "data/data acl")
        adapter = ACLWorkloadAdapter(token_counter=self._estimate_tokens)
        num_requests = int(config.get("num_requests", 0)) or None
        requests = adapter.load_cochrane(root_dir, limit=num_requests)
        return requests if num_requests is None else requests[:num_requests]

    def _load_mimic_bhc(self, config: dict[str, Any]) -> list[WorkloadRequest]:
        root_dir = config.get("dataset_path", "data/data acl")
        adapter = ACLWorkloadAdapter(token_counter=self._estimate_tokens)
        num_requests = int(config.get("num_requests", 0)) or None
        requests = adapter.load_mimic_bhc(root_dir, limit=num_requests)
        return requests if num_requests is None else requests[:num_requests]

    def _load_mixed_clinical(self, config: dict[str, Any]) -> list[WorkloadRequest]:
        """Build a heterogeneous clinical / biomedical workload.

        This mode is designed for control-plane and scheduler experiments where
        service class matters. It mixes light verification, long clinical
        summarization, and biomedical extraction/classification requests.
        """

        num_requests = int(config.get("num_requests", 60))
        mix = config.get(
            "mix",
            {
                "pubhealth": 0.4,
                "mimic_bhc": 0.3,
                "blue": 0.3,
            },
        )
        source_requests = {
            "pubhealth": self._load_pubhealth(
                {
                    **config,
                    "mode": "pubhealth",
                    "dataset_path": config.get("acl_dataset_path", "data/data acl"),
                    "num_requests": num_requests,
                }
            ),
            "mimic_bhc": self._load_mimic_bhc(
                {
                    **config,
                    "mode": "mimic_bhc",
                    "dataset_path": config.get("acl_dataset_path", "data/data acl"),
                    "num_requests": num_requests,
                }
            ),
            "blue": self._load_blue(
                {
                    **config,
                    "mode": "blue",
                    "dataset_path": config.get("blue_dataset_path", "data/blue_data"),
                    "num_requests": num_requests,
                }
            ),
        }

        available_sources = {
            name: requests for name, requests in source_requests.items() if requests
        }
        if not available_sources:
            return []

        weights = {
            name: float(mix.get(name, 0.0))
            for name in available_sources
        }
        if sum(weights.values()) <= 0:
            weights = {name: 1.0 for name in available_sources}

        source_positions = {name: 0 for name in available_sources}
        mixed_requests: list[WorkloadRequest] = []
        source_names = list(available_sources)
        source_weights = [weights[name] for name in source_names]

        while len(mixed_requests) < num_requests:
            source_name = self._random.choices(source_names, weights=source_weights, k=1)[0]
            requests = available_sources[source_name]
            position = source_positions[source_name] % len(requests)
            source_positions[source_name] += 1
            original = requests[position]
            metadata = dict(original.metadata)
            metadata["mixed_source"] = source_name
            metadata.setdefault("workflow_id", f"mixed-{len(mixed_requests):05d}")
            mixed_requests.append(
                WorkloadRequest(
                    request_id=f"mixed-{len(mixed_requests):05d}-{original.request_id}",
                    prompt=original.prompt,
                    input_tokens=original.input_tokens,
                    max_output_tokens=original.max_output_tokens,
                    task_type=original.task_type,
                    metadata=metadata,
                )
            )

        return mixed_requests

    def _weighted_bucket_choice(self, weights: dict[str, float]) -> str:
        buckets = list(weights.keys())
        return self._random.choices(buckets, weights=[weights[b] for b in buckets], k=1)[0]

    def _build_synthetic_prompt(self, task_type: str, target_tokens: int) -> str:
        template = TASK_TEMPLATES.get(task_type, TASK_TEMPLATES["summarization"])
        base_sentence = (
            "Patient is a 63 year old with hypertension, diabetes, chronic kidney disease, "
            "and recent hospitalization for shortness of breath. "
        )
        repeated_body = " ".join([base_sentence] * max(1, target_tokens // 20))
        prompt = template.format(body=repeated_body)
        return prompt

    def _configure_tokenizer(self, config: dict[str, Any]) -> None:
        tokenizer_name = config.get("tokenizer_name")
        if not tokenizer_name:
            self._tokenizer = None
            self._tokenizer_name = None
            return
        if tokenizer_name == self._tokenizer_name and self._tokenizer is not None:
            return

        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self._tokenizer_name = tokenizer_name
        except Exception:
            self._tokenizer = None
            self._tokenizer_name = None

    def _estimate_tokens(self, text: str) -> int:
        if self._tokenizer is not None:
            return max(1, len(self._tokenizer.encode(text, add_special_tokens=False)))
        return max(1, len(text.split()))

    def _configure_cost_model(self, config: dict[str, Any]) -> None:
        self._cost_model = build_cost_model(config.get("cost_model"))
        self._profiler = ClinicalRequestProfiler(cost_model_config=config.get("cost_model"))

    def _finalize_requests(self, requests: list[WorkloadRequest]) -> list[WorkloadRequest]:
        finalized: list[WorkloadRequest] = []
        for request in requests:
            metadata = dict(request.metadata)
            metadata["predicted_cost"] = self._cost_model.predict(request)
            profile = self._profiler.profile(request)
            route_decision = self._router.route(profile)
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
                    "route_name": route_decision.route_name,
                    "route_reason": route_decision.reason,
                    "route_score": route_decision.score,
                }
            )
            finalized.append(
                WorkloadRequest(
                    request_id=request.request_id,
                    prompt=request.prompt,
                    input_tokens=request.input_tokens,
                    max_output_tokens=request.max_output_tokens,
                    task_type=request.task_type,
                    metadata=metadata,
                )
            )
        return finalized
