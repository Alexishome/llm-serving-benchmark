#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-experiments/experiment_config_remote_synthetic_vllm.yaml}"

if [ ! -d "/workspace/vllm-env" ]; then
  echo "Missing /workspace/vllm-env. Create or restore the virtual environment first." >&2
  exit 1
fi

source /workspace/vllm-env/bin/activate

echo "Using config: ${CONFIG_PATH}"
python3 main.py --config "${CONFIG_PATH}"

echo
echo "Run complete. Summary:"
SUMMARY_PATH=$(python3 - "${CONFIG_PATH}" <<'PY'
import sys, yaml
from pathlib import Path
config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
print(config["results"]["summary_csv"])
PY
)

if [ -f "${SUMMARY_PATH}" ]; then
  cat "${SUMMARY_PATH}"
else
  echo "Summary file not found: ${SUMMARY_PATH}" >&2
fi
