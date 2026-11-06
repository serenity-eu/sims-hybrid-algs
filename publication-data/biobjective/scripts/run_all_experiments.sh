#!/bin/bash
# Full biobjective experiment run: hybrid PLS and all three EAs, both seeding sources.
#
#   run_all_experiments.sh <timeout_s> <runs> <workers> <out_root> [instance_regex]
#
# One job per (source, instance), each job doing all 17 configs for <runs>
# runs. Jobs are dealt round-robin to <workers>, each pinned to its own core
# with taskset -- validated at 7 workers (45 observations, worst deviation
# -0.03%), so a worker gets a full core and the wall-clock budget each config
# is given equals the CPU it actually receives. Core 7 is left for the OS.
set -u
T=${1:-200}; RUNS=${2:-10}; W=${3:-7}; ROOT=${4:-$HOME/biobjective-run}; FILTER=${5:-}
# Defaults resolve to this checkout; override any of REPO, PYTHON,
# LD_LIBRARY_PATH or GRB_LICENSE_FILE in the environment instead of editing.
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO/sims-problem" || exit 1
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/gurobi1203/linux64/lib}"
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/opt/gurobi/gurobi.lic}"
PY="${PYTHON:-$REPO/.venv/bin/python}"
LOG="$ROOT/run_all_experiments.log"

CONFIGS=(
  "GPBA-A"
  "Default PLS"        "Hybrid 75:25"          "Hybrid 50:50"          "Hybrid 25:75"
  "Baseline NSGA-II"   "Hybrid NSGA-II 75:25"  "Hybrid NSGA-II 50:50"  "Hybrid NSGA-II 25:75"
  "Baseline NSGA-III"  "Hybrid NSGA-III 75:25" "Hybrid NSGA-III 50:50" "Hybrid NSGA-III 25:75"
  "Baseline MOEA/D"    "Hybrid MOEA/D 75:25"   "Hybrid MOEA/D 50:50"   "Hybrid MOEA/D 25:75"
)
CITIES=(lagos_nigeria mexico_city paris rio_de_janeiro tokyo_bay)
SIZES=(100 125 150 175 200)
SOURCES=(gpbaa_2d_gurobi_hard an_2d_gurobi_hard)

mkdir -p "$ROOT"
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Build the job list.
JOBS=()
for src in "${SOURCES[@]}"; do
  for c in "${CITIES[@]}"; do
    for s in "${SIZES[@]}"; do
      inst="${c}_${s}"
      [ -n "$FILTER" ] && [[ ! "$inst" =~ $FILTER ]] && continue
      JOBS+=("$src|$inst")
    done
  done
done
say "jobs=${#JOBS[@]} workers=$W runs=$RUNS timeout=${T}s configs=${#CONFIGS[@]}"

run_job() { # $1="src|inst"  $2=core
  local src="${1%%|*}" inst="${1#*|}" core="$2"
  local tag="${src%%_*}"; [ "$tag" = "an" ] && tag=an || tag=gpba
  local out="$ROOT/$tag/$inst"
  mkdir -p "$out"
  taskset -c "$core" "$PY" run_hv_experiments.py \
      --hard-instances --pseudo-source "$src" --timeout "$T" --runs "$RUNS" \
      --filter "^${inst}\$" --output-dir "$out" --configs "${CONFIGS[@]}" \
      > "$out/run.log" 2>&1
  echo "$? $tag/$inst" >> "$ROOT/rc.txt"
}

worker() { # $1=index
  local idx=$1 i
  for (( i=idx; i<${#JOBS[@]}; i+=W )); do
    say "  w$idx core$idx -> ${JOBS[$i]}"
    run_job "${JOBS[$i]}" "$idx"
  done
  say "  w$idx finished"
}

: > "$ROOT/rc.txt"
for (( w=0; w<W; w++ )); do worker "$w" & done
wait

say "all workers done; rc summary:"
sort "$ROOT/rc.txt" | awk '{print $1}' | sort | uniq -c | tee -a "$LOG"
fails=$(awk '$1!=0' "$ROOT/rc.txt" | wc -l)
say "failed jobs: $fails"
[ "$fails" -gt 0 ] && awk '$1!=0' "$ROOT/rc.txt" | tee -a "$LOG"

# Consolidate per-source: one directory the figure/table generators can read.
for tag in gpba an; do
  dst="$ROOT/${tag}_all"; mkdir -p "$dst/traces"
  n=0
  for d in "$ROOT/$tag"/*/; do
    [ -d "$d" ] || continue
    cp "$d"/*.json "$dst/" 2>/dev/null && n=$((n+1))
    cp "$d"/traces/* "$dst/traces/" 2>/dev/null
    cp "$d"/*.trace.tar.gz "$dst/" 2>/dev/null
  done
  rm -f "$dst/all_experiments.json"
  say "consolidated $n instances into $dst ($(ls "$dst"/traces 2>/dev/null | wc -l) traces)"
done
say ALL_EXPERIMENTS_DONE
