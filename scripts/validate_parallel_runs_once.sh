#!/bin/bash
# A/B: does running N workers concurrently degrade results, given the budget is
# wall-clock? Arm A runs ONE instance alone. Arm B runs the same instance while
# N-1 sibling workers occupy other cores. Same instance, same seeds, same
# budget -- so any HV difference is CPU starvation, not stochasticity.
#
#   validate_parallel_runs_once.sh <timeout_s> <workers>
set -u
T=${1:-200}
N=${2:-7}
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO/sims-problem" || exit 1
export LD_LIBRARY_PATH=/opt/gurobi1203/linux64/lib GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic
PY="${PYTHON:-$REPO/.venv/bin/python}"
D="$REPO/sims-core/tests/data"
CFG=(--hard-instances --pseudo-source gpbaa_2d_gurobi_hard --timeout "$T" --runs 1
     --configs "GPBA-A" "Hybrid NSGA-II 75:25" "Hybrid NSGA-II 50:50" "Hybrid NSGA-II 25:75" "Baseline NSGA-II")
PROBE=paris_150
SIBS=(lagos_nigeria_150 mexico_city_150 rio_de_janeiro_150 tokyo_bay_150 paris_125 paris_175 lagos_nigeria_125)

run_one() { # $1=instance $2=outdir $3=core
    taskset -c "$3" "$PY" run_hv_experiments.py "${CFG[@]}" \
        --filter "^$1\$" --output-dir "$2" > "$2.log" 2>&1
}

# Explicit names, not a glob: `ab_{solo,par}*` also matched this script,
# which deleted itself mid-run.
rm -rf "$HOME"/abrun_solo "$HOME"/abrun_par "$HOME"/abrun_sib* \
       "$HOME"/abrun_solo.log "$HOME"/abrun_par.log "$HOME"/abrun_sib*.log 2>/dev/null

echo "[$(date +%T)] arm A: $PROBE alone on core 0"
t0=$(date +%s)
run_one "$PROBE" "$HOME/abrun_solo" 0
echo "[$(date +%T)] arm A done in $(( $(date +%s) - t0 ))s"

echo "[$(date +%T)] arm B: $PROBE on core 0 plus $((N-1)) siblings on cores 1..$((N-1))"
t0=$(date +%s)
run_one "$PROBE" "$HOME/abrun_par" 0 &
main=$!
for i in $(seq 1 $((N-1))); do
    run_one "${SIBS[$((i-1))]}" "$HOME/abrun_sib$i" "$i" &
done
wait $main
echo "[$(date +%T)] arm B probe done in $(( $(date +%s) - t0 ))s"
wait

"$PY" - <<PY
import json
a=json.load(open("$HOME/abrun_solo/$PROBE.json"))["configs"]
b=json.load(open("$HOME/abrun_par/$PROBE.json"))["configs"]
print()
print("%-24s %10s %10s %9s" % ("config","solo HV","par HV","delta %"))
worst=0.0
for k in a:
    x=a[k].get("final_hv",0.0); y=b.get(k,{}).get("final_hv",0.0)
    d=(y-x)/x*100 if x else 0.0
    worst=min(worst,d)
    print("%-24s %10.6f %10.6f %+8.2f%%" % (k,x,y,d))
print()
print("worst degradation: %+.2f%%" % worst)
print("VERDICT:", "parallel is safe" if worst > -0.5 else "PARALLEL DEGRADES RESULTS")
PY
