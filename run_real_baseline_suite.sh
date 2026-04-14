#!/usr/bin/env bash
set -euo pipefail

if [ ! -d "/workspace/bench-env" ]; then
  echo "Missing /workspace/bench-env. Create or restore the baseline environment first." >&2
  exit 1
fi

cd /workspace/llm-serving-benchmark
source /workspace/bench-env/bin/activate

CONFIGS=(
  "experiments/experiment_config_remote_pubhealth_hf_baseline_50.yaml"
  "experiments/experiment_config_remote_mimic_bhc_hf_baseline_50.yaml"
  "experiments/experiment_config_remote_blue_hf_baseline_50.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo
  echo "=== Running ${config} ==="
  python3 main.py --config "${config}"
done

echo
echo "=== Baseline summaries ==="
cat results/remote_pubhealth_hf_baseline_50/summary_metrics.csv
cat results/remote_mimic_bhc_hf_baseline_50/summary_metrics.csv
cat results/remote_blue_hf_baseline_50/summary_metrics.csv
