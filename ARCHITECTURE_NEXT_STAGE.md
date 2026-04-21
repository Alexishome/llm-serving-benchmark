# Next-Stage Serving System Architecture

This document describes the recommended next-stage architecture for this project after the initial `HF baseline vs vLLM` benchmarking phase.

The goal is no longer just to compare engines. The goal is to add a **serving optimization layer above the engine** that can:

- reduce latency
- reduce wasted compute
- use GPU resources more efficiently
- protect output quality on risky clinical / biomedical requests

## System Diagram

```text
                           +----------------------+
                           |  Real Requests /     |
                           |  Benchmark Workloads |
                           |  PubHealth / BLUE /  |
                           |  MIMIC-BHC           |
                           +----------+-----------+
                                      |
                                      v
                         +------------+-------------+
                         |  Request Profiler        |
                         |--------------------------|
                         | - task family            |
                         | - prefill cost estimate  |
                         | - decode cost estimate   |
                         | - cache affinity score   |
                         | - quality risk score     |
                         +------------+-------------+
                                      |
                                      v
                         +------------+-------------+
                         |  Front-Door Router       |
                         |--------------------------|
                         | chooses serving path:    |
                         | - standard lane          |
                         | - long-prefill lane      |
                         | - high-risk lane         |
                         | - future: speculative    |
                         |   or quantized lane      |
                         +-----+-----------+--------+
                               |           |
                 +-------------+           +--------------+
                 |                                           
                 v                                            
   +-------------+-------------+        +---------------------+-------------+
   | Standard Lane / Engine    |        | Specialized Lane / Engine         |
   |---------------------------|        |-----------------------------------|
   | vLLM or baseline path     |        | long-prefill or safer path        |
   | local scheduler           |        | local scheduler                   |
   | FIFO / cost-aware / SLO   |        | uncertainty-aware / service class |
   +-------------+-------------+        +---------------------+-------------+
                 |                                            |
                 +-------------------+------------------------+
                                     |
                                     v
                      +--------------+---------------+
                      | Metrics + Quality Evaluation |
                      |------------------------------|
                      | latency / throughput         |
                      | queue delay / GPU            |
                      | validity / task quality      |
                      | route decision diagnostics   |
                      +------------------------------+
```

## Why This Architecture Fits This Project

The current system already has:

- real workload generation from PubHealth, BLUE, and MIMIC-BHC
- engine abstraction (`hf_transformers`, `vllm`, `tgi`)
- scheduling policies inside a serving lane
- cost estimation
- GPU monitoring
- request-level and summary-level metrics

That means the strongest next step is **not** to rebuild the engine. It is to add a **control plane** on top of the current system.

## Layer Responsibilities

### 1. Request Profiler

This layer predicts lightweight request properties before the request enters a lane.

Recommended outputs:

- `task_family`
- `prefill_cost`
- `decode_cost`
- `total_cost`
- `cache_affinity_score`
- `quality_risk_score`
- `route_reason`

This is where we connect domain knowledge to serving behavior.

Examples:

- `PubHealth`:
  - typically lower quality risk
  - medium prefill cost
  - short decode
- `MIMIC-BHC`:
  - high prefill cost
  - high decode cost
  - higher quality risk
- `BLUE`:
  - heterogeneous
  - often light, but subtask-dependent

### 2. Front-Door Router

This layer decides which serving path should handle a request.

Recommended first prototype lanes:

- `standard`
  - default path for ordinary requests
- `long_prefill`
  - for very long prompts, especially MIMIC-BHC-like summarization
- `high_risk`
  - for requests predicted to have higher quality risk

This layer is the main candidate for the next paper contribution.

### 3. Lane-Specific Scheduler

Each lane can still use its own scheduler.

Examples:

- `standard` lane:
  - FIFO
  - predicted-cost-first
- `long_prefill` lane:
  - prefill-aware ordering
  - long-input isolation
- `high_risk` lane:
  - more conservative service class
  - future: stronger model or verifier path

This preserves your current scheduler work rather than discarding it.

## Recommended Implementation Phases

### Phase 1: Profiler + Router Skeleton

Add:

- a `control_plane/` package
- a request profiler
- a heuristic router
- route decision logging

No lane splitting is required yet. The first goal is:

- predict request features
- make a route decision
- store those decisions in metadata

### Phase 2: Multi-Lane Simulation

Extend the benchmark framework so that requests can be assigned to:

- `standard`
- `long_prefill`
- `high_risk`

Initially, these lanes can still point to the same backend while collecting route-level measurements.

This lets you test whether the control-plane decision logic is meaningful before you add real multi-path infrastructure.

### Phase 3: Real Multi-Path Execution

Expose differentiated paths, for example:

- standard vLLM lane
- long-prefill vLLM lane
- future speculative lane
- future quantized or smaller-model lane

At this point, routing becomes a real efficiency optimization rather than only a simulated decision.

### Phase 4: Quality-Aware Routing

Use dataset-specific quality signals to refine routing:

- `PubHealth`: accuracy / macro-F1
- `MIMIC-BHC`: ROUGE-L / BERTScore
- `BLUE`: subtask-specific metrics

The long-term objective is:

- spend cheap paths on easy requests
- reserve expensive paths for risky requests

## Recommended First Prototype

The best first prototype for this repository is:

1. add a lightweight request profiler
2. add a heuristic router over 3 lanes:
   - `standard`
   - `long_prefill`
   - `high_risk`
3. log the route decision per request
4. evaluate whether MIMIC-BHC is consistently identified as a long-prefill / higher-risk workload

This is the smallest step that:

- fits the current codebase
- preserves your existing scheduler work
- moves the system toward a publishable control-plane contribution

## What We Should Build Next

Immediate next module order:

1. `control_plane/request_profiler.py`
2. `control_plane/request_router.py`
3. route-aware metadata logging
4. optional lane-aware scheduler dispatch

That is the implementation path this repository should follow next.
