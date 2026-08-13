#!/usr/bin/env bash
# =============================================================================
# Generate BOTH exact-front pseudo datasets with the native Gurobi backend at a
# 10-minute (600 s) per-instance budget, for either instance set:
#
#   publication    (default): 5 cities x 100-250        -> sims-core/tests/data/{gpbaa,an}_2d_gurobi/
#   random-clouds            : 5 cities x 100-500 step 50 -> sims-core/tests/data/{gpbaa,an}_2d_gurobi_rc/
#
# These replace the 120 s CSV-converted sets (convert_gurobi_pseudo.py) with
# freshly solved 600 s fronts. Both are produced by the SAME solver front-end
# (generate_pseudo.py --solver gurobi); this wrapper just fans the jobs
# (methods x instances) out with a robust PID-polling semaphore.
#
# PREREQUISITES (one-time):
#   * Build sims-problem with the Gurobi backend:
#       scripts/fetch-gurobi.sh            # provisions GUROBI_HOME + rpath
#   * A valid Gurobi license reachable at runtime (named-user/WLS/academic;
#     the pip license is rejected by the C API).
#
# This script always (re)builds sims-problem in --release before launching
# jobs (see the maturin develop call below): `uv sync`'s default dev/debug
# profile is fine for iterating on code, but the .dzn parsing + MILP model
# construction that runs before every solve is measurably (>10x) slower
# unoptimized, and that cost is paid once per instance across the whole
# batch. Skip this if you've already got a release build installed and
# don't want the ~15s rebuild check: SKIP_RELEASE_BUILD=1 ./run_gurobi_pseudo.sh
#
# Usage:
#   ./run_gurobi_pseudo.sh                             # publication set, both methods
#   ./run_gurobi_pseudo.sh tokyo                        # regex filter on instance name
#   METHODS="gpba" ./run_gurobi_pseudo.sh               # just GPBA-A (or "aneja")
#   INSTANCE_SET=random-clouds ./run_gurobi_pseudo.sh   # the 100-500 benchmark set
#   TIMEOUT=900 ./run_gurobi_pseudo.sh     # only raise MAX_CONCURRENT if the
#                                           # box has cores to spare (see below)
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

if [[ "${SKIP_RELEASE_BUILD:-0}" != "1" ]]; then
    echo "Building sims-problem in --release (skip with SKIP_RELEASE_BUILD=1)..."
    uv run maturin develop --release \
        --features "pyo3/extension-module,milp,scalarized_selection,gurobi"
    echo ""
fi

FILTER="${1:-}"
TIMEOUT="${TIMEOUT:-600}"                 # 10 minutes per instance
# Gurobi solves are unbounded-thread by default (no Threads cap is set), so
# concurrent solves oversubscribe the box's cores badly — observed wall-clock
# blowups of several multiples of --timeout on an 8-core server at
# MAX_CONCURRENT=4. Default to serial (1) until Gurobi's own thread usage is
# capped; raise deliberately only if you know the box has cores to spare.
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
METHODS="${METHODS:-gpba aneja}"
INSTANCE_SET="${INSTANCE_SET:-publication}"
LOG_DIR="${LOG_DIR:-logs/gurobi_gen}"
mkdir -p "$LOG_DIR"

CITIES=(lagos_nigeria mexico_city paris rio_de_janeiro tokyo_bay)
case "$INSTANCE_SET" in
    publication)    SIZES=(100 150 200 250) ;;
    random-clouds)  SIZES=(100 150 200 250 300 350 400 450 500) ;;
    *) echo "error: INSTANCE_SET must be publication or random-clouds (got '$INSTANCE_SET')" >&2; exit 2 ;;
esac

# Build the (method, instance) job list, smallest size first so quick wins
# land early and any per-size regression surfaces before the long tail.
SIZES_ASC=($(printf '%s\n' "${SIZES[@]}" | sort -n))
JOBS=()
for method in $METHODS; do
    for size in "${SIZES_ASC[@]}"; do
        for city in "${CITIES[@]}"; do
            inst="${city}_${size}"
            [[ -n "$FILTER" ]] && ! echo "$inst" | grep -qE "$FILTER" && continue
            JOBS+=("${method}:${inst}")
        done
    done
done

echo "Gurobi pseudo-generation: ${#JOBS[@]} jobs (methods: $METHODS, instance-set: $INSTANCE_SET)"
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
    # Invoke the venv's python directly, NOT `uv run python`: `uv run` performs
    # an implicit sync/rebuild check on every invocation, and sims-problem's
    # cache-keys (needed for uv to notice source changes in its path
    # dependencies) historically lagged behind — a change to augmecon-rs could
    # go undetected, causing `uv run` to silently reinstall a stale cached
    # wheel over the release build from the step above (and, worse, rebuild it
    # in --profile=dev per this project's config-settings). Going straight to
    # the venv sidesteps that resync entirely for the actual solve jobs.
    RUST_LOG=off .venv/bin/python generate_pseudo.py \
        --solver gurobi \
        --method "$method" \
        --instance-set "$INSTANCE_SET" \
        --timeout "$TIMEOUT" \
        --filter "$inst" \
        > "$LOG_DIR/${method}_${inst}.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} jobs. Waiting…"
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=$((FAILED+1)); done

OUT_SUFFIX=""
[[ "$INSTANCE_SET" == "random-clouds" ]] && OUT_SUFFIX="_rc"
echo ""
if [[ $FAILED -eq 0 ]]; then
    echo "GUROBI_GEN_DONE — gpbaa_2d_gurobi${OUT_SUFFIX}/ and an_2d_gurobi${OUT_SUFFIX}/ regenerated at ${TIMEOUT}s."
else
    echo "GUROBI_GEN_DONE — $FAILED job(s) failed. Check $LOG_DIR/*.log"
    exit 1
fi
