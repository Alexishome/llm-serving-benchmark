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

Status: started.

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

Next to implement:

- route summary markdown generation
- route distribution by dataset
- route distribution by workflow stage and service class
- mean latency / queue delay / GPU behavior by route

Minimal output:

- `route_summary.json`
- `route_summary.md`

### Phase 3: Logical Multi-Lane Simulation

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

1. Run the mixed clinical scheduler suite after pulling the control-plane metadata changes:

```bash
bash run_mixed_clinical_scheduler_suite_qwen25_7b.sh
```

2. Generate route analysis using:

```bash
python -m control_plane.analyze_routes \
  --request-metrics results/<run>/request_metrics.json
```

3. Add route summary generation to the normal experiment output.
4. Compare `fifo`, `predicted_cost_first`, and `service_class_priority` on the same mixed workload.

## Recommended Paper Framing

The strongest framing is:

> A workflow-aware serving control plane for clinical and biomedical LLM workloads.

The system contribution is not just faster inference. It is:

- profiling heterogeneous clinical workflow stages
- routing by cost, prefill burden, cache affinity, service class, and quality risk
- preserving quality-aware escalation as a future extension
- treating scheduling as lane-level execution policy rather than the whole contribution
