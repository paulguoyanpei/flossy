#!/usr/bin/env bash
# Launch the CPU verifier and the GPU prover as two separate OS processes (the
# real-cluster / TEE deploy pattern; in-sandbox use run_e2e_spawn.py instead).
#   bash run_mp.sh <workers> <verifier-threads> -- <shared args...> [--prover <prover-only args...>]
# <workers>/<verifier-threads> configure the verifier only. Args after `--` go to
# BOTH processes (e.g. --model, --prompt, -n). Anything after a literal `--prover`
# token goes to the prover only (e.g. --batch).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${FLOSSY_PYTHON:-python}"
GPU="${FLOSSY_GPU:-0}"
# Optional: isolate the CPU verifier on a NUMA node away from co-located jobs to
# protect its memory bandwidth (verify_step is bandwidth-bound). e.g.
#   FLOSSY_VERIFIER_NUMA="--cpunodebind=1 --membind=1"
VNUMA="${FLOSSY_VERIFIER_NUMA:-}"
VWRAP=(); [ -n "$VNUMA" ] && VWRAP=(numactl $VNUMA)
WORKERS="$1"; VTHREADS="$2"; shift 2; [ "${1:-}" = "--" ] && shift

# split shared args from prover-only args at the `--prover` sentinel
SHARED=(); PROVER_ONLY=(); seen_prover=0
for a in "$@"; do
  if [ "$a" = "--prover" ]; then seen_prover=1; continue; fi
  if [ "$seen_prover" -eq 1 ]; then PROVER_ONLY+=("$a"); else SHARED+=("$a"); fi
done

RUN_ID="$$_$RANDOM"; PF="/tmp/flossy_port_${RUN_ID}"; SHM="flossy_mp_${RUN_ID}"; rm -f "$PF"
COMMON=(--port-file "$PF" --shm-name "$SHM" "${SHARED[@]}")

CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 "${VWRAP[@]}" "$PY" -u "$HERE/verifier.py" "${COMMON[@]}" \
    --workers "$WORKERS" --threads "$VTHREADS" 2>&1 | sed -u 's/^/[V] /' &
VP=$!
CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 "$PY" -u "$HERE/prover.py" "${COMMON[@]}" \
    --threads 4 "${PROVER_ONLY[@]}" 2>&1 | sed -u 's/^/[P] /' &
PP=$!
trap 'kill $VP $PP 2>/dev/null || true; rm -f "$PF"' EXIT
RC=0; wait $PP || RC=$?; wait $VP || RC=$?; rm -f "$PF"; exit $RC
