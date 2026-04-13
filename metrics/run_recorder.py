from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


class RunRecorder:
    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root)

    def record(
        self,
        config: dict[str, Any],
        config_path: str | Path,
        request_csv: str | Path,
        summary_csv: str | Path,
    ) -> dict[str, Path]:
        request_csv_path = self.project_root / Path(request_csv)
        summary_csv_path = self.project_root / Path(summary_csv)
        output_dir = self.project_root / Path(config["results"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        if not summary_csv_path.exists():
            return {}

        summary_df = pd.read_csv(summary_csv_path)
        request_df = pd.read_csv(request_csv_path) if request_csv_path.exists() else pd.DataFrame()
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        run_snapshot = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config_path": str(config_path),
            "project_name": config.get("project", {}).get("name"),
            "workload_mode": config.get("workload", {}).get("mode"),
            "dataset_path": config.get("workload", {}).get("dataset_path"),
            "num_requests_configured": config.get("workload", {}).get("num_requests"),
            "scheduler_policy": config.get("scheduler", {}).get("policy", "fifo"),
            "request_csv": str(request_csv),
            "summary_csv": str(summary_csv),
            "output_dir": str(config["results"]["output_dir"]),
            "summary_records": summary_df.to_dict(orient="records"),
            "request_count_logged": int(len(request_df)),
            "success_count_logged": (
                int(request_df["success"].sum()) if "success" in request_df.columns else None
            ),
        }

        json_path = output_dir / f"run_summary_{run_id}.json"
        md_path = output_dir / f"run_summary_{run_id}.md"
        latest_md_path = output_dir / "run_summary_latest.md"

        json_path.write_text(json.dumps(run_snapshot, indent=2), encoding="utf-8")
        markdown = self._build_markdown(run_snapshot, summary_df)
        md_path.write_text(markdown, encoding="utf-8")
        latest_md_path.write_text(markdown, encoding="utf-8")

        self._append_history(config, config_path, summary_df, request_df, run_id)
        return {"json": json_path, "markdown": md_path, "latest_markdown": latest_md_path}

    def _append_history(
        self,
        config: dict[str, Any],
        config_path: str | Path,
        summary_df: pd.DataFrame,
        request_df: pd.DataFrame,
        run_id: str,
    ) -> None:
        history_dir = self.project_root / "results"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / "run_history.csv"

        base_record = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config_path": str(config_path),
            "project_name": config.get("project", {}).get("name"),
            "workload_mode": config.get("workload", {}).get("mode"),
            "dataset_path": config.get("workload", {}).get("dataset_path"),
            "num_requests_configured": config.get("workload", {}).get("num_requests"),
            "scheduler_policy_config": config.get("scheduler", {}).get("policy", "fifo"),
            "result_dir": str(config["results"]["output_dir"]),
            "request_count_logged": int(len(request_df)),
            "success_count_logged": (
                int(request_df["success"].sum()) if "success" in request_df.columns else None
            ),
        }

        records = []
        for _, row in summary_df.iterrows():
            row_record = {**base_record}
            row_record.update(row.to_dict())
            records.append(row_record)

        history_df = pd.DataFrame(records)
        if history_path.exists():
            existing_df = pd.read_csv(history_path)
            history_df = pd.concat([existing_df, history_df], ignore_index=True)
        history_df.to_csv(history_path, index=False)

    @staticmethod
    def _build_markdown(run_snapshot: dict[str, Any], summary_df: pd.DataFrame) -> str:
        lines = [
            f"# Run Summary: {run_snapshot['run_id']}",
            "",
            "## Context",
            f"- Timestamp: `{run_snapshot['timestamp']}`",
            f"- Config: `{run_snapshot['config_path']}`",
            f"- Workload mode: `{run_snapshot['workload_mode']}`",
            f"- Dataset path: `{run_snapshot['dataset_path']}`",
            f"- Scheduler policy: `{run_snapshot['scheduler_policy']}`",
            f"- Configured requests: `{run_snapshot['num_requests_configured']}`",
            f"- Logged requests: `{run_snapshot['request_count_logged']}`",
            f"- Successful requests: `{run_snapshot['success_count_logged']}`",
            "",
            "## Summary Rows",
        ]

        if summary_df.empty:
            lines.append("- No summary rows recorded.")
            return "\n".join(lines) + "\n"

        key_fields = [
            "engine",
            "model",
            "trial",
            "concurrency",
            "request_rate",
            "batch_size",
            "scheduler_policy",
            "latency_mean",
            "latency_p95",
            "throughput_rps",
            "queue_delay_mean",
            "queue_delay_p95",
            "tokens_per_second_mean",
            "predicted_cost_total",
            "success_rate",
        ]

        for index, (_, row) in enumerate(summary_df.iterrows(), start=1):
            lines.append(f"### Row {index}")
            for field in key_fields:
                if field in row and pd.notna(row[field]):
                    lines.append(f"- {field}: `{row[field]}`")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"
