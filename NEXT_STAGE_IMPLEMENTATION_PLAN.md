# Next-Stage Implementation Plan

## Project Direction

The project should move from **single-request LLM serving benchmarking** toward a **clinical workflow serving control plane**.

The target system should optimize future clinical and biomedical AI workloads where token use comes from:

- chart review
- retrieval-heavy clinical assistance
- long-context summarization
- extraction and coding pipelines
- verification / factuality loops
- background batch jobs mixed with interactive requests

The main research direction is:

> Build a workflow-aware, cost-aware, quality-risk-aware serving control plane that routes clinical and biomedical workload stages across appropriate serving paths.

## Focused Direction After EMNLP Review

The direction does not need to be replaced, but it should be narrowed.

The project should not try to be:

- a generic vLLM benchmark
- a full multi-agent hospital system
- a broad survey of every possible serving optimization
- a large model-scaling study with many unrelated experiments

The focused paper direction is:

> Clinical workload-aware control plane improves serving efficiency without hurting output quality.

This means the next version should center on:

1. a mixed clinical workload that better approximates future clinical AI systems
2. a lightweight profiler/router that classifies requests before engine execution
3. a service-class-aware scheduler that uses workflow metadata
4. evidence that the policy improves latency / queue delay / GPU behavior
5. evidence that quality is preserved

This framing is a better fit for an EMNLP short-paper-style contribution because it turns the project from "benchmarking engines" into "a system and evaluation methodology for clinical NLP serving."

## Architecture Layers

1. Request profiler
   - predicts task family, workflow stage, service class, prefill/decode cost, cache affinity, and quality risk
2. Front-door router
   - routes requests before they enter the serving lane
3. Lane-specific scheduler
   - keeps FIFO / predicted-cost-first / future SLO-aware policies inside each lane
4. Multi-path executor
   - eventually dispatches to standard, long-prefill, cache-affine, high-risk, or verifier paths
5. Quality and route-aware evaluation
   - evaluates latency, throughput, GPU use, route distribution, and task quality

## Implementation Phases

### Phase 1: Request Profiling and Route Logging

Status: implemented.

Implemented:

- `control_plane/request_profiler.py`
- `control_plane/request_router.py`
- `control_plane/analyze_routes.py`
- request metadata fields:
  - `profile_task_family`
  - `workflow_stage`
  - `service_class`
  - `profile_prefill_cost`
  - `profile_decode_cost`
  - `profile_total_cost`
  - `profile_cache_affinity_score`
  - `profile_quality_risk_score`
  - `route_name`
  - `route_reason`
  - `route_score`

Goal:

- make route decisions visible in request-level results
- verify that MIMIC-BHC is recognized as high-risk / long-context traffic
- verify that PubHealth and BLUE produce lighter and more heterogeneous profiles

### Phase 2: Route-Aware Analysis

Status: implemented as a standalone analyzer.

Implemented:

- route summary markdown generation
- route distribution by dataset
- route distribution by workflow stage and service class
- route-level request profile summaries

Still useful to add later:

- mean latency / queue delay / GPU behavior by route inside the standard run summary

Minimal output:

- `route_summary.json`
- `route_summary.md`

### Phase 3: Logical Multi-Lane Simulation

Status: started through route metadata and mixed-workload experiments.

Next:

- keep one physical backend
- simulate lanes using metadata:
  - `standard`
  - `long_prefill`
  - `cache_affine`
  - `high_risk`
- evaluate whether routing decisions identify meaningful traffic classes before adding real multi-engine execution

Why:

- low implementation risk
- produces evidence that the control plane is meaningful
- avoids overengineering before the route signal is validated

### Phase 4: Lane-Specific Scheduling

Status: first policy implemented.

Implemented:

- `service_class_priority`

Next:

- add service-class-aware scheduling
- add uncertainty / tail-risk-aware cost scheduling
- compare against FIFO and predicted-cost-first

Candidate policies:

- `service_class_priority`
- `risk_aware_cost_first`
- `long_prefill_isolation`

Status update:

- `service_class_priority` has been added as the first service-class-aware scheduler.
- It uses control-plane metadata from the profiler:
  - `interactive`
  - `verification`
  - `standard`
  - `standard_generation`
  - `batch_long_context`
  - `background`
- This policy is intentionally simple so it can be compared against FIFO and predicted-cost-first before adding more complex routing or multi-lane execution.

### Phase 5: Real Multi-Path Execution

Later:

- standard vLLM lane
- long-prefill lane
- high-risk lane
- optional smaller/faster model lane
- optional verifier lane

This is where the system becomes a true multi-engine / multi-mode optimization layer.

### Phase 6: Quality-Aware Routing

Later:

- PubHealth: accuracy / macro-F1
- MIMIC-BHC: ROUGE-L / BERTScore
- BLUE: subtask-specific metrics

Use quality signals to decide:

- cheap path for easy requests
- safer or stronger path for high-risk clinical requests
- verifier path when output risk is high

## Immediate Next Tasks

1. Finish the missing 7B vLLM single-dataset runs if the server is already available:

- `MIMIC-BHC`
- `BLUE`

These complete the clean 7B HF vs vLLM comparison table.

2. Run the mixed clinical scheduler suite after pulling the control-plane metadata changes:

```bash
bash run_mixed_clinical_scheduler_suite_qwen25_7b.sh
```

3. Generate route analysis using:

```bash
python -m control_plane.analyze_routes \
  --request-metrics results/<run>/request_metrics.json
```

4. Compare `fifo`, `predicted_cost_first`, and `service_class_priority` on the same mixed workload.
5. Add task-quality and agreement checks to avoid making a speed-only claim.

## Recommended Paper Framing

The strongest framing is:

> A workflow-aware serving control plane for clinical and biomedical LLM workloads.

The system contribution is not just faster inference. It is:

- profiling heterogeneous clinical workflow stages
- routing by cost, prefill burden, cache affinity, service class, and quality risk
- preserving quality-aware escalation as a future extension
- treating scheduling as lane-level execution policy rather than the whole contribution

## Paper Readiness Criteria

For a credible short-paper submission, the project should have:

- a clear claim: workflow-aware control improves serving efficiency for clinical LLM workloads
- one mixed clinical benchmark built from real datasets
- at least three policies compared on the same workload: `fifo`, `predicted_cost_first`, `service_class_priority`
- latency, tail latency, queue delay, throughput, and GPU metrics
- quality-preservation evidence:
  - PubHealth accuracy / macro-F1
  - BLUE task-level metrics where feasible
  - MIMIC-BHC summary similarity metrics
  - vLLM-vs-HF output agreement where exact labels are not enough
- an ablation showing what the profiler/router metadata changes compared with cost-only scheduling

## Deferred Future Work

These remain valuable, but should not distract from the core paper story:

- full multi-engine router
- small-model / large-model escalation
- verifier model routing
- full clinical agent workflow implementation
- many model-size sweeps
- extensive vLLM internal parameter tuning

They can be framed as the natural extension after the control-plane signal is validated.
