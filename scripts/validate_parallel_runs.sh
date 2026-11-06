#!/bin/bash
# Repeated A/B. The solo arm is deterministic, so it is run ONCE per probe;
# the parallel arm is repeated, since contention is what varies. Several probe
# instances of different sizes, so a result is not an artifact of one instance.
#
#   validate_parallel_runs.sh <timeout_s> <workers> <reps>
set -u
T=${1:-200}; N=${2:-7}; REPS=${3:-3}
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO/sims-problem" || exit 1
export LD_LIBRARY_PATH=/opt/gurobi1203/linux64/lib GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic
PY="${PYTHON:-$REPO/.venv/bin/python}"
CFG=(--hard-instances --pseudo-source gpbaa_2d_gurobi_hard --timeout "$T" --runs 1
     --configs "GPBA-A" "Hybrid NSGA-II 75:25" "Hybrid NSGA-II 50:50" "Hybrid NSGA-II 25:75" "Baseline NSGA-II")
PROBES=(paris_150 lagos_nigeria_200 tokyo_bay_125)
FILL=(mexico_city_150 rio_de_janeiro_150 lagos_nigeria_125 paris_175 tokyo_bay_175 mexico_city_125 rio_de_janeiro_175)

run_one() { taskset -c "$3" "$PY" run_hv_experiments.py "${CFG[@]}" --filter "^$1\$" --output-dir "$2" > "$2.log" 2>&1; }

rm -rf "$HOME"/abr_* 2>/dev/null
for probe in "${PROBES[@]}"; do
  echo "[$(date +%T)] === $probe : solo"
  s=$(date +%s); run_one "$probe" "$HOME/abr_${probe}_solo" 0; echo "    solo wall $(( $(date +%s) - s ))s"
  for rep in $(seq 1 "$REPS"); do
    echo "[$(date +%T)] === $probe : parallel rep $rep ($N workers)"
    s=$(date +%s)
    run_one "$probe" "$HOME/abr_${probe}_par$rep" 0 &
    main=$!
    for i in $(seq 1 $((N-1))); do run_one "${FILL[$((i-1))]}" "$HOME/abr_fill${i}_r$rep" "$i" & done
    wait $main; echo "    probe wall $(( $(date +%s) - s ))s"
    wait
  done
done

"$PY" - <<PY
import json, glob, os, statistics as st
probes = "${PROBES[*]}".split()
reps = $REPS
print()
print("%-20s %-24s %9s %9s" % ("instance","config","solo HV","par deltas %"))
alld=[]
for p in probes:
    sp=f"$HOME/abr_{p}_solo/{p}.json"
    if not os.path.exists(sp): print(f"  {p}: solo missing"); continue
    a=json.load(open(sp))["configs"]
    for k in a:
        x=a[k].get("final_hv",0.0)
        ds=[]
        for r in range(1,reps+1):
            f=f"$HOME/abr_{p}_par{r}/{p}.json"
            if os.path.exists(f):
                y=json.load(open(f))["configs"].get(k,{}).get("final_hv",0.0)
                ds.append((y-x)/x*100 if x else 0.0)
        alld += ds
        print("%-20s %-24s %9.6f  %s" % (p,k,x," ".join(f"{d:+.2f}" for d in ds)))
print()
if alld:
    print("all %d observations: mean %+.3f%%  median %+.3f%%  worst %+.3f%%"
          % (len(alld), st.mean(alld), st.median(alld), min(alld)))
    print("observations worse than -0.5%%: %d of %d" % (sum(1 for d in alld if d < -0.5), len(alld)))
PY
