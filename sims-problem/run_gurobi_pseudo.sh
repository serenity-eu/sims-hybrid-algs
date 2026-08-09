#!/usr/bin/env bash
# =============================================================================
# Generate BOTH exact-front pseudo datasets with the native Gurobi backend at a
# 10-minute (600 s) per-instance budget, for all 20 publication instances:
#
#   GPBA-A      -> sims-core/tests/data/gpbaa_2d_gurobi/
#   Aneja & Nair-> sims-core/tests/data/an_2d_gurobi/
#
# These replace the 120 s CSV-converted sets (convert_gurobi_pseudo.py) with
# freshly solved 600 s fronts. Both are produced by the SAME solver front-end
# (generate_pseudo.py --solver gurobi); this wrapper just fans the two
# methods × 20 instances out with a robust PID-polling semaphore.
#
# PREREQUISITES (one-time):
#   * Build sims-problem with the Gurobi backend:
#       scripts/fetch-gurobi.sh            # provisions GUROBI_HOME + rpath
#       uv pip install -e sims-problem --reinstall-package sims-problem \
#         --config-settings=build-args="--release --features gurobi"
#   * A valid Gurobi license reachable at runtime (named-user/WLS/academic;
#     the pip license is rejected by the C API).
#
# Usage:
#   ./run_gurobi_pseudo.sh                 # both methods, all 20 instances
#   ./run_gurobi_pseudo.sh tokyo           # regex filter on instance name
#   METHODS="gpba" ./run_gurobi_pseudo.sh  # just GPBA-A (or "aneja")
#   TIMEOUT=900 MAX_CONCURRENT=2 ./run_gurobi_pseudo.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

FILTER="${1:-}"
TIMEOUT="${TIMEOUT:-600}"                 # 10 minutes per instance
# Gurobi solves are multi-threaded and may be license-seat limited, so default
# to a lower fan-out than the HiGHS scripts (which run 8-wide).
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
METHODS="${METHODS:-gpba aneja}"
LOG_DIR="${LOG_DIR:-logs/gurobi_gen}"
mkdir -p "$LOG_DIR"

CITIES=(lagos_nigeria mexico_city paris rio_de_janeiro tokyo_bay)
SIZES=(100 150 200 250)

# Build the (method, instance) job list. 250s first so the long tail doesn't
# dominate wall-clock.
JOBS=()
for method in $METHODS; do
    for size in 250 200 150 100; do
        for city in "${CITIES[@]}"; do
            inst="${city}_${size}"
            [[ -n "$FILTER" ]] && ! echo "$inst" | grep -qE "$FILTER" && continue
            JOBS+=("${method}:${inst}")
        done
    done
done

echo "Gurobi pseudo-generation: ${#JOBS[@]} jobs (methods: $METHODS)"
echo "  timeout: ${TIMEOUT}s   concurrency: $MAX_CONCURRENT   logs: $LOG_DIR/"
echo ""

PIDS=()
for job in "${JOBS[@]}"; do
    method="${job%%:*}"
    inst="${job#*:}"

    # Robust semaphore: count live PIDs via kill -0 (works under nohup, unlike
    # jobs -rp / wait -n which silently no-op non-interactively).
    while true; do
        alive=0
        for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done
        [ "$alive" -lt "$MAX_CONCURRENT" ] && break
        sleep 3
    done

    echo "launching $method/$inst (timeout=${TIMEOUT}s) -> $LOG_DIR/${method}_${inst}.log"
    RUST_LOG=off uv run python generate_pseudo.py \
        --solver gurobi \
        --method "$method" \
        --timeout "$TIMEOUT" \
        --filter "$inst" \
        > "$LOG_DIR/${method}_${inst}.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} jobs. Waiting…"
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=$((FAILED+1)); done

echo ""
if [[ $FAILED -eq 0 ]]; then
    echo "GUROBI_GEN_DONE — gpbaa_2d_gurobi/ and an_2d_gurobi/ regenerated at ${TIMEOUT}s."
else
    echo "GUROBI_GEN_DONE — $FAILED job(s) failed. Check $LOG_DIR/*.log"
    exit 1
fi
