#!/usr/bin/env bash
# Four real vLLM servers, four policy directories, one GPU.
#
# What this establishes that a mock cannot: four distinct server instances
# derived from four real /proc process pairs, per-lane listen-socket
# ownership, api_pid != engine_core_pid, and -- the property the third
# review found had no production caller at all -- a *live scheduler*
# observing a seal request and producing the terminal seal and retirement
# artifacts itself.
#
# What it cannot establish, and does not claim: four distinct *physical*
# GPU UUIDs. All four processes share one card, so the lane-set check must
# refuse them, and that refusal is retained as evidence the check is live.
# Real MIG identity is likewise out of reach here.
#
# Usage: scripts/four_process_execution_plane_smoke.sh [artifact-dir]
set -euo pipefail

PYTHON=${PYTHON:-/home/zazzi/miniconda3/envs/llm-finetune/bin/python}
MODEL=${GLADIUS_SMOKE_MODEL:?set GLADIUS_SMOKE_MODEL to a local model directory}
ARTIFACTS=${1:-$(pwd)/docs/superpowers/artifacts/third-review/four-process-smoke}
BASE_PORT=${BASE_PORT:-8410}
LANES=${LANES:-4}
MEM_UTIL=${MEM_UTIL:-0.17}

mkdir -p "$ARTIFACTS"
RUN_DIR=$(mktemp -d /tmp/gladius-smoke-XXXXXX)
echo "artifacts: $ARTIFACTS"
echo "run dir:   $RUN_DIR"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT

{
  echo "# Four-process execution-plane smoke"
  echo
  echo "started:  $(date -Is)"
  echo "model:    $MODEL"
  echo "vllm:     $($PYTHON -c 'import vllm; print(vllm.__version__)')"
  echo "commit:   $(git rev-parse HEAD)"
  echo "gpu:      $(rocm-smi --showproductname --csv 2>/dev/null | tail -n +2 | head -1 || echo unknown)"
  echo
} | tee "$ARTIFACTS/environment.txt"

# --- launch -------------------------------------------------------------
for lane in $(seq 0 $((LANES - 1))); do
  port=$((BASE_PORT + lane))
  policy_dir="$RUN_DIR/policy$lane"
  mkdir -p "$policy_dir"
  # A per-lane nonce: two lanes sharing one would let either adopt the
  # other's receipt.
  nonce=$($PYTHON -c "import secrets; print(secrets.token_hex(32))")
  echo "$nonce" > "$policy_dir/.nonce"

  GLADIUS_POLICY_DIR="$policy_dir" \
  GLADIUS_ENGINE_ID="gladius-smoke-gpu$lane" \
  GLADIUS_ATTESTATION_NONCE="$nonce" \
  GLADIUS_TELEMETRY_SAMPLE_N=1 \
  VLLM_LOGGING_LEVEL=WARNING \
    "$PYTHON" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --served-model-name "gladius-smoke" \
      --port "$port" \
      --host 127.0.0.1 \
      --gpu-memory-utilization "$MEM_UTIL" \
      --max-model-len 2048 \
      --max-num-seqs 8 \
      --scheduler-cls gladius_vllm.scheduler.GladiusScheduler \
      > "$RUN_DIR/server$lane.log" 2>&1 &
  PIDS+=($!)
  echo "lane $lane: launched pid ${PIDS[-1]} on port $port"
done

# --- wait for readiness --------------------------------------------------
for lane in $(seq 0 $((LANES - 1))); do
  port=$((BASE_PORT + lane))
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
      echo "lane $lane: healthy"
      break
    fi
    sleep 2
  done
  curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1 || {
    echo "lane $lane never became healthy; last log lines:" >&2
    tail -30 "$RUN_DIR/server$lane.log" >&2
    exit 1
  }
done

# --- attest --------------------------------------------------------------
for lane in $(seq 0 $((LANES - 1))); do
  port=$((BASE_PORT + lane))
  policy_dir="$RUN_DIR/policy$lane"
  nonce=$(cat "$policy_dir/.nonce")
  "$PYTHON" -m gladius_vllm.attest publish \
    --policy-dir "$policy_dir" \
    --nonce "$nonce" \
    --host 127.0.0.1 \
    --port "$port" | tee "$ARTIFACTS/receipt-lane$lane.json"
done

# --- drive traffic, then seal through the live scheduler ------------------
"$PYTHON" scripts/four_process_smoke_driver.py \
  --run-dir "$RUN_DIR" \
  --artifacts "$ARTIFACTS" \
  --lanes "$LANES" \
  --base-port "$BASE_PORT" \
  | tee "$ARTIFACTS/driver-report.json"

cp "$RUN_DIR"/server*.log "$ARTIFACTS/" 2>/dev/null || true
for lane in $(seq 0 $((LANES - 1))); do
  mkdir -p "$ARTIFACTS/policy$lane"
  cp -r "$RUN_DIR/policy$lane"/. "$ARTIFACTS/policy$lane/" 2>/dev/null || true
done
echo "finished: $(date -Is)" | tee -a "$ARTIFACTS/environment.txt"
