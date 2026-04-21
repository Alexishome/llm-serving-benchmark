from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze route-aware request metadata produced by the control plane."
    )
    parser.add_argument(
        "--request-metrics",
        required=True,
        help="Path to request_metrics.json or request_metrics.csv.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the rendered JSON summary.",
    )
    parser.add_argument(
        "--markdown-output",
        help="Optional path to save a markdown route summary.",
    )
    return parser.parse_args()


def load_request_records(path: str | Path) -> pd.DataFrame:
    request_path = Path(path)
    if request_path.suffix.lower() == ".json":
        return pd.DataFrame(json.loads(request_path.read_text(encoding="utf-8")))
    return pd.read_csv(request_path)


def _value_counts(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    cleaned = series.fillna("unknown").astype(str)
    return {str(key): int(value) for key, value in cleaned.value_counts().items()}


def _group_summary(df: pd.DataFrame, group_col: str) -> list[dict[str, Any]]:
    if group_col not in df.columns or df.empty:
        return []

    grouped_rows: list[dict[str, Any]] = []
    for key, group in df.groupby(group_col, dropna=False):
        route_name = "unknown" if pd.isna(key) else str(key)
        grouped_rows.append(
            {
                group_col: route_name,
                "count": int(len(group)),
                "share": float(len(group) / len(df)) if len(df) else 0.0,
                "success_rate": float(group["success"].astype(bool).mean())
                if "success" in group.columns
                else 0.0,
                "mean_input_tokens": float(group["input_tokens"].mean())
                if "input_tokens" in group.columns
                else 0.0,
                "mean_output_tokens": float(group["output_tokens"].mean())
                if "output_tokens" in group.columns
                else 0.0,
                "mean_predicted_cost": float(group["predicted_cost"].mean())
                if "predicted_cost" in group.columns
                else 0.0,
                "mean_latency": float(group["latency"].mean())
                if "latency" in group.columns
                else 0.0,
                "p95_latency": float(group["latency"].quantile(0.95))
                if "latency" in group.columns and len(group)
                else 0.0,
                "mean_queue_delay": float(group["queue_delay"].mean())
                if "queue_delay" in group.columns
                else 0.0,
                "p95_queue_delay": float(group["queue_delay"].quantile(0.95))
                if "queue_delay" in group.columns and len(group)
                else 0.0,
                "mean_profile_prefill_cost": float(group["profile_prefill_cost"].mean())
                if "profile_prefill_cost" in group.columns
                else 0.0,
                "mean_profile_decode_cost": float(group["profile_decode_cost"].mean())
                if "profile_decode_cost" in group.columns
                else 0.0,
                "mean_profile_total_cost": float(group["profile_total_cost"].mean())
                if "profile_total_cost" in group.columns
                else 0.0,
                "mean_cache_affinity_score": float(group["profile_cache_affinity_score"].mean())
                if "profile_cache_affinity_score" in group.columns
                else 0.0,
                "mean_quality_risk_score": float(group["profile_quality_risk_score"].mean())
                if "profile_quality_risk_score" in group.columns
                else 0.0,
            }
        )

    grouped_rows.sort(key=lambda row: (-row["count"], row[group_col]))
    return grouped_rows


def analyze_routes(request_df: pd.DataFrame) -> dict[str, Any]:
    if request_df.empty:
        return {
            "request_count": 0,
            "routes": [],
            "datasets": [],
            "route_by_dataset": {},
            "task_families": {},
            "workflow_stages": {},
            "service_classes": {},
        }

    route_by_dataset: dict[str, dict[str, int]] = {}
    if "dataset_name" in request_df.columns and "route_name" in request_df.columns:
        for dataset_name, dataset_df in request_df.groupby("dataset_name", dropna=False):
            dataset_key = "unknown" if pd.isna(dataset_name) else str(dataset_name)
            route_by_dataset[dataset_key] = _value_counts(dataset_df["route_name"])

    summary = {
        "request_count": int(len(request_df)),
        "routes": _group_summary(request_df, "route_name"),
        "datasets": _group_summary(request_df, "dataset_name"),
        "route_by_dataset": route_by_dataset,
        "task_families": _value_counts(
            request_df["profile_task_family"]
            if "profile_task_family" in request_df.columns
            else request_df["task_family"]
            if "task_family" in request_df.columns
            else pd.Series(dtype=str)
        ),
        "workflow_stages": _value_counts(
            request_df["workflow_stage"]
            if "workflow_stage" in request_df.columns
            else pd.Series(dtype=str)
        ),
        "service_classes": _value_counts(
            request_df["service_class"]
            if "service_class" in request_df.columns
            else pd.Series(dtype=str)
        ),
    }
    return summary


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Route-Aware Request Summary",
        "",
        f"- Request count: `{summary.get('request_count', 0)}`",
        "",
        "## Route Distribution",
        "",
        "| Route | Count | Share | Success | Mean Latency | P95 Latency | Mean Queue | P95 Queue | Mean Input | Mean Output | Mean Cost | Mean Risk | Mean Cache Affinity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in summary.get("routes", []):
        lines.append(
            "| {route} | {count} | {share} | {success} | {latency} | {p95_latency} | {queue} | {p95_queue} | {input_tokens} | {output_tokens} | {cost} | {risk} | {cache} |".format(
                route=row.get("route_name", "unknown"),
                count=row.get("count", 0),
                share=_format_float(row.get("share", 0.0)),
                success=_format_float(row.get("success_rate", 0.0)),
                latency=_format_float(row.get("mean_latency", 0.0)),
                p95_latency=_format_float(row.get("p95_latency", 0.0)),
                queue=_format_float(row.get("mean_queue_delay", 0.0)),
                p95_queue=_format_float(row.get("p95_queue_delay", 0.0)),
                input_tokens=_format_float(row.get("mean_input_tokens", 0.0)),
                output_tokens=_format_float(row.get("mean_output_tokens", 0.0)),
                cost=_format_float(row.get("mean_predicted_cost", 0.0)),
                risk=_format_float(row.get("mean_quality_risk_score", 0.0)),
                cache=_format_float(row.get("mean_cache_affinity_score", 0.0)),
            )
        )

    lines.extend(
        [
            "",
            "## Dataset Distribution",
            "",
            "| Dataset | Count | Share | Mean Latency | P95 Latency | Mean Queue | P95 Queue | Mean Input | Mean Output | Mean Risk |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary.get("datasets", []):
        lines.append(
            "| {dataset} | {count} | {share} | {latency} | {p95_latency} | {queue} | {p95_queue} | {input_tokens} | {output_tokens} | {risk} |".format(
                dataset=row.get("dataset_name", "unknown"),
                count=row.get("count", 0),
                share=_format_float(row.get("share", 0.0)),
                latency=_format_float(row.get("mean_latency", 0.0)),
                p95_latency=_format_float(row.get("p95_latency", 0.0)),
                queue=_format_float(row.get("mean_queue_delay", 0.0)),
                p95_queue=_format_float(row.get("p95_queue_delay", 0.0)),
                input_tokens=_format_float(row.get("mean_input_tokens", 0.0)),
                output_tokens=_format_float(row.get("mean_output_tokens", 0.0)),
                risk=_format_float(row.get("mean_quality_risk_score", 0.0)),
            )
        )

    lines.extend(["", "## Route By Dataset", ""])
    route_by_dataset = summary.get("route_by_dataset", {})
    if route_by_dataset:
        for dataset, routes in route_by_dataset.items():
            rendered_routes = ", ".join(f"`{route}`: {count}" for route, count in routes.items())
            lines.append(f"- `{dataset}`: {rendered_routes}")
    else:
        lines.append("- No route-by-dataset data available.")

    lines.extend(["", "## Workflow Stages", ""])
    workflow_stages = summary.get("workflow_stages", {})
    if workflow_stages:
        for stage, count in workflow_stages.items():
            lines.append(f"- `{stage}`: {count}")
    else:
        lines.append("- No workflow-stage data available.")

    lines.extend(["", "## Service Classes", ""])
    service_classes = summary.get("service_classes", {})
    if service_classes:
        for service_class, count in service_classes.items():
            lines.append(f"- `{service_class}`: {count}")
    else:
        lines.append("- No service-class data available.")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    request_df = load_request_records(args.request_metrics)
    results = analyze_routes(request_df)
    rendered = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    if args.markdown_output:
        Path(args.markdown_output).write_text(render_markdown(results), encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
