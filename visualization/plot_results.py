from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ResultPlotter:
    def __init__(self, style: str = "whitegrid") -> None:
        sns.set_theme(style=style)

    def plot_all(self, summary_csv: str | Path, output_dir: str | Path) -> None:
        summary_df = pd.read_csv(summary_csv)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if summary_df.empty:
            return

        primary_hue = self._choose_primary_hue(summary_df)
        self._plot(
            summary_df,
            x="concurrency",
            y="latency_mean",
            hue=primary_hue,
            title="Latency vs Concurrency",
            ylabel="Latency (s)",
            output_path=output_path / "latency_vs_concurrency.png",
        )
        self._plot(
            summary_df,
            x="concurrency",
            y="throughput_rps",
            hue=primary_hue,
            title="Throughput vs Concurrency",
            ylabel="Throughput (req/s)",
            output_path=output_path / "throughput_vs_concurrency.png",
        )
        self._plot(
            summary_df,
            x="input_tokens",
            y="tokens_per_second_mean",
            hue=primary_hue,
            title="Tokens/sec vs Input Length",
            ylabel="Tokens / sec",
            output_path=output_path / "tokens_per_second_vs_input_length.png",
        )
        if "latency_p95" in summary_df.columns:
            self._plot(
                summary_df,
                x="concurrency",
                y="latency_p95",
                hue=primary_hue,
                title="P95 Latency vs Concurrency",
                ylabel="P95 Latency (s)",
                output_path=output_path / "latency_p95_vs_concurrency.png",
            )
        if "queue_delay_mean" in summary_df.columns:
            self._plot(
                summary_df,
                x="concurrency",
                y="queue_delay_mean",
                hue=primary_hue,
                title="Queue Delay vs Concurrency",
                ylabel="Queue Delay (s)",
                output_path=output_path / "queue_delay_vs_concurrency.png",
            )
        if "queue_delay_p95" in summary_df.columns:
            self._plot(
                summary_df,
                x="concurrency",
                y="queue_delay_p95",
                hue=primary_hue,
                title="P95 Queue Delay vs Concurrency",
                ylabel="P95 Queue Delay (s)",
                output_path=output_path / "queue_delay_p95_vs_concurrency.png",
            )
        if "predicted_cost_total" in summary_df.columns:
            self._plot(
                summary_df,
                x="concurrency",
                y="predicted_cost_total",
                hue=primary_hue,
                title="Predicted Cost vs Concurrency",
                ylabel="Predicted Cost",
                output_path=output_path / "predicted_cost_vs_concurrency.png",
            )
        if "cost_efficiency_tokens_per_cost" in summary_df.columns:
            self._plot(
                summary_df,
                x="throughput_rps",
                y="cost_efficiency_tokens_per_cost",
                hue=primary_hue,
                title="Cost Efficiency vs Throughput",
                ylabel="Tokens per Predicted Cost",
                output_path=output_path / "cost_efficiency_vs_throughput.png",
            )
        if "predicted_cost_total" in summary_df.columns:
            self._plot(
                summary_df,
                x="predicted_cost_total",
                y="latency_mean",
                hue=primary_hue,
                title="Latency vs Predicted Cost",
                ylabel="Latency (s)",
                output_path=output_path / "latency_vs_predicted_cost.png",
            )

        if "cost_per_1m_tokens" in summary_df.columns and summary_df["cost_per_1m_tokens"].notna().any():
            self._plot(
                summary_df,
                x="throughput_rps",
                y="cost_per_1m_tokens",
                hue=primary_hue,
                title="Cost vs Throughput",
                ylabel="Cost",
                output_path=output_path / "cost_vs_throughput.png",
            )

        if "gpu_utilization_mean" in summary_df.columns and summary_df["gpu_utilization_mean"].notna().any():
            self._plot(
                summary_df,
                x="batch_size",
                y="gpu_utilization_mean",
                hue=primary_hue,
                title="GPU Utilization vs Batch Size",
                ylabel="GPU Utilization (%)",
                output_path=output_path / "gpu_utilization_vs_batch_size.png",
            )

    @staticmethod
    def _choose_primary_hue(df: pd.DataFrame) -> str:
        if "scheduler_policy" in df.columns and df["scheduler_policy"].nunique() > 1:
            return "scheduler_policy"
        return "engine"

    def _plot(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        hue: str,
        title: str,
        ylabel: str,
        output_path: Path,
    ) -> None:
        plt.figure(figsize=(9, 6))
        sns.lineplot(data=df, x=x, y=y, hue=hue, marker="o")
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(x.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
