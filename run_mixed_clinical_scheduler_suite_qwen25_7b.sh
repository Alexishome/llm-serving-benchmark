#!/usr/bin/env bash
set -euo pipefail

cd /workspace/llm-serving-benchmark

if [ -d "/workspace/vllm7-env" ]; then
  source /workspace/vllm7-env/bin/activate
elif [ -d "/workspace/vllm2-env" ]; then
  source /workspace/vllm2-env/bin/activate
else
  echo "Missing /workspace/vllm7-env or /workspace/vllm2-env." >&2
  exit 1
fi

CONFIGS=(
  "experiments/experiment_config_remote_mixed_clinical_vllm_fifo_60_qwen25_7b.yaml"
  "experiments/experiment_config_remote_mixed_clinical_vllm_predicted_cost_60_qwen25_7b.yaml"
  "experiments/experiment_config_remote_mixed_clinical_vllm_service_class_60_qwen25_7b.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo
  echo "=== Running ${config} ==="
  python3 main.py --config "${config}"
done

echo
echo "=== Mixed clinical scheduler summaries ==="
cat results/remote_mixed_clinical_vllm_fifo_60_qwen25_7b/summary_metrics.csv
cat results/remote_mixed_clinical_vllm_predicted_cost_60_qwen25_7b/summary_metrics.csv
cat results/remote_mixed_clinical_vllm_service_class_60_qwen25_7b/summary_metrics.csv

echo
echo "=== Route summaries ==="
python3 -m control_plane.analyze_routes \
  --request-metrics results/remote_mixed_clinical_vllm_fifo_60_qwen25_7b/request_metrics.json \
  --markdown-output results/remote_mixed_clinical_vllm_fifo_60_qwen25_7b/route_summary.md
python3 -m control_plane.analyze_routes \
  --request-metrics results/remote_mixed_clinical_vllm_predicted_cost_60_qwen25_7b/request_metrics.json \
  --markdown-output results/remote_mixed_clinical_vllm_predicted_cost_60_qwen25_7b/route_summary.md
python3 -m control_plane.analyze_routes \
  --request-metrics results/remote_mixed_clinical_vllm_service_class_60_qwen25_7b/request_metrics.json \
  --markdown-output results/remote_mixed_clinical_vllm_service_class_60_qwen25_7b/route_summary.md
