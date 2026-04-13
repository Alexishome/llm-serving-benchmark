# LLM Serving Benchmark Framework for Clinical NLP Workloads

This project provides a reproducible benchmarking framework for evaluating LLM serving systems under clinical NLP workloads. It includes workload generation, pluggable engine adapters, async experiment orchestration, GPU monitoring, metrics export, and visualization.

## Project Layout

```text
llm-benchmark/
├── workload/
│   ├── dataset_loader.py
│   └── workload_generator.py
├── engines/
│   ├── base_engine.py
│   ├── hf_engine.py
│   ├── tgi_engine.py
│   └── vllm_engine.py
├── metrics/
│   ├── gpu_monitor.py
│   └── metrics_collector.py
├── scheduler/
│   ├── policies.py
│   └── request_scheduler.py
├── experiments/
│   ├── experiment_config.yaml
│   ├── experiment_config_blue.yaml
│   ├── experiment_config_cochrane.yaml
│   ├── experiment_config_leval.yaml
│   ├── experiment_config_mimic_bhc.yaml
│   ├── experiment_config_pubhealth.yaml
│   └── experiment_runner.py
├── results/
├── visualization/
│   └── plot_results.py
├── main.py
├── requirements.txt
└── README.md
```

## Features

- Synthetic or dataset-backed workload generation
- Configurable request rate, concurrency, batch size, and task mix
- Interpretable predicted inference cost modeling for heterogeneous clinical NLP requests
- Cost-aware and workload-aware scheduling policies including FIFO, shortest-input-first, predicted-cost-first, and hybrid task-aware routing
- Unified serving interface for `vLLM`, `TGI`, and local `Transformers`
- Request-level latency, TTFT, throughput, queue delay, predicted cost, success/error, and token-rate metrics
- GPU utilization and memory sampling via `pynvml` or `nvidia-smi`
- CSV and JSON result export
- Standard plots for latency, throughput, tokens/sec, cost, and GPU utilization

## Installation

```bash
cd llm-benchmark
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Experiments

Update [experiments/experiment_config.yaml](/Users/alextang/Downloads/llm serving benchmark/llm-benchmark/experiments/experiment_config.yaml) to enable the serving backends you want to test.

```bash
cd llm-benchmark
python main.py --config experiments/experiment_config.yaml
```

For the benchmark-backed workloads added in this project:

```bash
python main.py --config experiments/experiment_config_leval.yaml
python main.py --config experiments/experiment_config_blue.yaml
python main.py --config experiments/experiment_config_pubhealth.yaml
python main.py --config experiments/experiment_config_cochrane.yaml
python main.py --config experiments/experiment_config_mimic_bhc.yaml
```

For the simplest remote GPU smoke test on the same machine that is hosting `vLLM`:

```bash
python main.py --config experiments/experiment_config_remote_synthetic_vllm.yaml
```

Results are written to the `results/` directory:

- Request-level metrics in CSV and JSON
- Trial summary metrics in CSV and JSON
- GPU samples in CSV
- Benchmark plots in `results/plots/`
- Per-run summaries in `run_summary_*.md` / `run_summary_*.json`
- A global run index in `results/run_history.csv`

## Remote Run Workflow

If you use a remote GPU machine for `vLLM`, a simple long-term workflow is:

```bash
cd /workspace
git clone https://github.com/Alexishome/llm-serving-benchmark.git
cd llm-serving-benchmark
source /workspace/vllm-env/bin/activate
pip install -r requirements.txt
bash run_remote.sh
```

Notes:

- Keep `vllm serve ... --host 0.0.0.0 --port 8000` running in a separate terminal.
- The remote smoke-test config uses `http://127.0.0.1:8000`, so benchmark requests and GPU monitoring happen on the same machine.
- This is the easiest way to populate `gpu_utilization_mean` and `gpu_memory_mb_mean`.

To automate a same-machine comparison between `vLLM` and the `HF Transformers` baseline:

```bash
bash run_remote_compare_vllm_vs_hf.sh
```

This script will:

- stop any existing `vllm serve` process
- start a temporary `vLLM` server
- wait until `http://127.0.0.1:8000/v1/models` is ready
- run the remote `vLLM` experiment
- stop `vLLM`
- run the remote `HF Transformers` baseline

## Configuration Notes

- Use `workload.mode: dataset` with `dataset_path` to benchmark real prompts from JSON or CSV.
- Set `workload.mode: synthetic` to simulate controllable document and output-length distributions.
- Set `workload.mode: leval` to turn the local LEval benchmark into a long-context serving workload.
- Set `workload.mode: blue` to turn the local BLUE benchmark into a biomedical / clinical serving workload.
- Set `workload.mode: pubhealth` to build a public-health claim verification workload.
- Set `workload.mode: cochrane` to build an evidence-review summarization workload.
- Set `workload.mode: mimic_bhc` to build a clinical hospital-course summarization workload from discharge notes.
- Set `workload.tokenizer_name` to use a real Hugging Face tokenizer for token estimation; otherwise the framework falls back to whitespace-based counts.
- Configure `workload.cost_model` to estimate per-request inference cost from input length, output length, KV-cache pressure, and task type.
- Enable or disable backends under `engines.backends`.
- Choose a scheduling policy under `scheduler.policy` to compare FIFO against workload-aware serving strategies.
- Add a new backend by subclassing `BaseEngine` and extending `build_engine()` in [main.py](/Users/alextang/Downloads/llm serving benchmark/llm-benchmark/main.py).
- Summary results now include `latency_p95`, `latency_p99`, and queueing metrics to evaluate scheduling behavior.
- Plot generation now includes scheduler-aware tail-latency, queue-delay, and predicted-cost efficiency figures when those metrics are present.

## Dataset Format

JSON records should look like:

```json
[
  {
    "prompt": "Summarize this discharge summary ...",
    "task_type": "summarization",
    "max_tokens": 128
  }
]
```

CSV files should include at least a `prompt` column and may also include `task_type` and `max_tokens`.

## Benchmark-Backed Workloads

The framework can also build serving workloads from benchmark corpora instead of generic CSV/JSON prompts.

- `LEval` support is implemented in [workload/benchmark_adapters.py](/Users/alextang/Downloads/llm%20serving%20benchmark/llm-benchmark/workload/benchmark_adapters.py).
  The adapter converts each long document plus instruction pair into a request and maps tasks into `question_answering` or `summarization`.
- `BLUE` support is implemented in [workload/benchmark_adapters.py](/Users/alextang/Downloads/llm%20serving%20benchmark/llm-benchmark/workload/benchmark_adapters.py).
  The current adapter covers the locally available subsets `BIOSSES`, `ChemProt`, `ddi2013-type`, `hoc`, and `BC5CDR`.
  These are mapped into serving-style prompts for sentence similarity, relation extraction, document classification, and entity extraction.

Current local dataset paths:

- [data/leval](/Users/alextang/Downloads/llm%20serving%20benchmark/llm-benchmark/data/leval)
- [data/blue_data](/Users/alextang/Downloads/llm%20serving%20benchmark/llm-benchmark/data/blue_data)
- [data/data acl](/Users/alextang/Downloads/llm%20serving%20benchmark/llm-benchmark/data/data%20acl)

ACL-backed workload modes currently supported:

- `pubhealth`
  Maps claim plus evidence articles into fact-checking style requests.
- `cochrane`
  Maps evidence-review abstracts into conclusion summarization requests.
- `mimic_bhc`
  Maps clinical discharge notes into hospital-course summarization requests.

## Research Extensions

- Plug in domain-specific prompt distributions in `workload/workload_generator.py`
- Replace the heuristic cost model in `optimization/cost_model.py` with a learned predictor or hardware-aware estimator
- Add new serving systems in `engines/`
- Extend the workload-aware scheduler with additional policies or routing strategies in `scheduler/`
- Extend experiment sweeps in `experiments/experiment_config.yaml`
- Add richer metrics or dashboards on top of the exported result tables

## Lightweight Quality Evaluation

The framework now preserves `output_text`, `expected_output`, `dataset_name`, and related metadata in request-level results so you can run lightweight quality checks after benchmarking.

Example:

```bash
python quality/evaluate_quality.py \
  --request-metrics results/mimic_bhc/request_metrics.json
```

Current support includes:

- general validity metrics such as `success_rate`, `non_empty_rate`, and `valid_response_rate`
- label-based automatic evaluation for `pubhealth` with `accuracy` and `macro_f1`

This is intended as a lightweight quality sanity check to complement serving metrics, not as a full task-benchmark replacement.
