#!/usr/bin/env bash
# Launch the FLOSSY verifier (CPU) and prover (GPU) as two independent processes.
# All extra args (--model, --prompt, -n, --slots, --tol, ...) are forwarded to both.
#
#   bash run.sh --model <path-or-id> --prompt "..." -n 64
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${FLOSSY_PYTHON:-/hpc/home/e1374338/anaconda3/envs/python310/bin/python}"
GPU="${FLOSSY_GPU:-0}"

RUN_ID="$$_$RANDOM"
PORT_FILE="/tmp/flossy_port_${RUN_ID}"
SHM_NAME="flossy_ring_${RUN_ID}"
LOG_DIR="${FLOSSY_LOG_DIR:-/tmp}"
rm -f "$PORT_FILE"

COMMON=(--port-file "$PORT_FILE" --shm-name "$SHM_NAME" "$@")

echo "[run] verifier -> $LOG_DIR/flossy_verifier_${RUN_ID}.log"
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 "$PY" "$HERE/verifier.py" "${COMMON[@]}" \
    2>&1 | tee "$LOG_DIR/flossy_verifier_${RUN_ID}.log" &
VERIFIER_PID=$!

echo "[run] prover (GPU $GPU) -> $LOG_DIR/flossy_prover_${RUN_ID}.log"
CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 "$PY" "$HERE/prover.py" "${COMMON[@]}" \
    2>&1 | tee "$LOG_DIR/flossy_prover_${RUN_ID}.log" &
PROVER_PID=$!

trap 'kill $VERIFIER_PID $PROVER_PID 2>/dev/null || true; rm -f "$PORT_FILE"' EXIT

RC=0
wait $PROVER_PID || RC=$?
wait $VERIFIER_PID || RC=$?
rm -f "$PORT_FILE"
echo "[run] done (rc=$RC)"
exit $RC
