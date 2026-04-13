#!/usr/bin/env bash
set -euo pipefail

VLLM_CONFIG="${1:-experiments/experiment_config_remote_synthetic_vllm.yaml}"
HF_CONFIG="${2:-experiments/experiment_config_remote_synthetic_hf_baseline.yaml}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_LOG="${VLLM_LOG:-/workspace/vllm_remote_compare.log}"
VLLM_READY_TIMEOUT_SECONDS="${VLLM_READY_TIMEOUT_SECONDS:-420}"
VLLM_READY_POLL_SECONDS="${VLLM_READY_POLL_SECONDS:-2}"

if [ ! -d "/workspace/vllm-env" ]; then
  echo "Missing /workspace/vllm-env. Create or restore the virtual environment first." >&2
  exit 1
fi

source /workspace/vllm-env/bin/activate

cleanup() {
  if [ -n "${VLLM_PID:-}" ] && kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "Stopping temporary vLLM server (PID ${VLLM_PID})..."
    kill "${VLLM_PID}" || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT

echo "Stopping any existing vLLM serve process on port ${VLLM_PORT}..."
pkill -f "vllm serve" || true
sleep 2

echo "Starting vLLM server for model ${VLLM_MODEL}..."
nohup vllm serve "${VLLM_MODEL}" --host 0.0.0.0 --port "${VLLM_PORT}" >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM readiness..."
READY_ATTEMPTS=$((VLLM_READY_TIMEOUT_SECONDS / VLLM_READY_POLL_SECONDS))
for _ in $(seq 1 "${READY_ATTEMPTS}"); do
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then
    echo "vLLM is ready."
    break
  fi
  sleep "${VLLM_READY_POLL_SECONDS}"
done

if ! curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then
  echo "vLLM failed to become ready within ${VLLM_READY_TIMEOUT_SECONDS} seconds. Recent log tail:" >&2
  tail -n 50 "${VLLM_LOG}" >&2 || true
  exit 1
fi

echo
echo "Running vLLM benchmark with ${VLLM_CONFIG}..."
bash run_remote.sh "${VLLM_CONFIG}"

echo
echo "Stopping vLLM before HF baseline..."
cleanup
unset VLLM_PID
sleep 3

echo
echo "Running HF baseline benchmark with ${HF_CONFIG}..."
bash run_remote.sh "${HF_CONFIG}"

echo
echo "Comparison complete."
echo "vLLM summary: $(python3 - "${VLLM_CONFIG}" <<'PY'
import sys, yaml
from pathlib import Path
config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
print(config["results"]["summary_csv"])
PY
)"
echo "HF summary: $(python3 - "${HF_CONFIG}" <<'PY'
import sys, yaml
from pathlib import Path
config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
print(config["results"]["summary_csv"])
PY
)"
