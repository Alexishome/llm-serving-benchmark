#!/usr/bin/env bash
set -euo pipefail

if [ ! -d "/workspace/vllm2-env" ]; then
  echo "Missing /workspace/vllm2-env. Create or restore the vLLM environment first." >&2
  exit 1
fi

cd /workspace/llm-serving-benchmark
source /workspace/vllm2-env/bin/activate

CONFIGS=(
  "experiments/experiment_config_remote_pubhealth_vllm_fifo_50_qwen25_7b.yaml"
  "experiments/experiment_config_remote_mimic_bhc_vllm_fifo_50_qwen25_7b.yaml"
  "experiments/experiment_config_remote_blue_vllm_fifo_50_qwen25_7b.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo
  echo "=== Running ${config} ==="
  python3 main.py --config "${config}"
done

echo
echo "=== vLLM 7B summaries ==="
cat results/remote_pubhealth_vllm_fifo_50_qwen25_7b/summary_metrics.csv
cat results/remote_mimic_bhc_vllm_fifo_50_qwen25_7b/summary_metrics.csv
cat results/remote_blue_vllm_fifo_50_qwen25_7b/summary_metrics.csv
