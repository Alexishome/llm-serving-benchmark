from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import yaml

from experiments.experiment_runner import ExperimentRunner
from metrics.gpu_monitor import GPUMonitor
from metrics.metrics_collector import MetricsCollector
from metrics.run_recorder import RunRecorder
from workload.workload_generator import WorkloadGenerator


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_engine(engine_config: dict[str, Any]):
    name = engine_config["name"]
    model = engine_config["model"]
    if name == "vllm":
        from engines.vllm_engine import VLLMEngine

        return VLLMEngine(base_url=engine_config["base_url"], model_name=model)
    if name == "tgi":
        from engines.tgi_engine import TGIEngine

        return TGIEngine(base_url=engine_config["base_url"], model_name=model)
    if name == "hf_transformers":
        from engines.hf_engine import HFLocalEngine

        return HFLocalEngine(
            model_name=model,
            device=engine_config.get("device", "cuda"),
            torch_dtype=engine_config.get("torch_dtype", "auto"),
        )
    raise ValueError(f"Unsupported engine backend: {name}")


async def run(config: dict[str, Any], config_path: str | Path) -> None:
    workload_generator = WorkloadGenerator(seed=config["project"].get("seed"))
    workload = workload_generator.generate(config["workload"])
    metrics_collector = MetricsCollector()
    run_recorder = RunRecorder(project_root=Path(__file__).resolve().parent)

    gpu_monitor = None
    if config.get("gpu_monitor", {}).get("enabled", True):
        gpu_monitor = GPUMonitor(
            poll_interval=float(config["gpu_monitor"].get("poll_interval", 1.0))
        )
        await gpu_monitor.start()

    try:
        for engine_config in config["engines"]["backends"]:
            if not engine_config.get("enabled", False):
                continue

            engine = build_engine(engine_config)
            await engine.startup()
            runner = ExperimentRunner(
                engine=engine,
                metrics_collector=metrics_collector,
                gpu_monitor=gpu_monitor,
                scheduler_config=config.get("scheduler"),
            )

            try:
                for trial in range(1, int(config["experiments"]["trials"]) + 1):
                    for concurrency in config["experiments"]["concurrency_levels"]:
                        for request_rate in config["experiments"]["request_rates"]:
                            for batch_size in config["experiments"]["batch_sizes"]:
                                await runner.run_trial(
                                    workload=workload,
                                    concurrency=int(concurrency),
                                    request_rate=float(request_rate),
                                    batch_size=int(batch_size),
                                    trial_index=trial,
                                    engine_kwargs=config["engines"].get("default_kwargs", {}),
                                    cost_per_1m_tokens=config["experiments"].get("cost_per_1m_tokens"),
                                )
            finally:
                await engine.shutdown()
    finally:
        if gpu_monitor is not None:
            await gpu_monitor.stop()

    results = config["results"]
    metrics_collector.save_results(
        request_output_csv=results["request_csv"],
        summary_output_csv=results["summary_csv"],
        request_output_json=results["request_json"],
        summary_output_json=results["summary_json"],
    )

    if gpu_monitor is not None:
        gpu_monitor.save_csv(results["gpu_csv"])

    try:
        from visualization.plot_results import ResultPlotter

        result_plotter = ResultPlotter()
        result_plotter.plot_all(
            summary_csv=results["summary_csv"],
            output_dir=results["plots_dir"],
        )
    except ModuleNotFoundError as error:
        print(f"Warning: plotting skipped because an optional dependency is missing: {error}")

    run_recorder.record(
        config=config,
        config_path=config_path,
        request_csv=results["request_csv"],
        summary_csv=results["summary_csv"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM serving benchmark experiments.")
    parser.add_argument(
        "--config",
        default="experiments/experiment_config.yaml",
        help="Path to YAML experiment configuration.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(load_config(args.config), args.config))
