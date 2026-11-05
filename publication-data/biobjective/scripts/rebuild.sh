#!/usr/bin/env bash
# Rebuild every figure and table in the biobjective SIMS paper from the
# committed results. No solving, no Gurobi, no network.
#
#   ./rebuild.sh [out_dir]
#
# Writes LaTeX to <out_dir>, EPS to <out_dir>_eps and PNG to <out_dir>_png.
# paper/elsa-paper/README.md lists which file goes where in the manuscript.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${1:-$PWD/paper-artifacts}"

R="$ROOT/publication-data/biobjective"
D="$ROOT/sims-core/tests/data"
# The workspace venv, so the script works without activating it first.
if [[ -n "${PYTHON:-}" ]]; then PY="$PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then PY="$ROOT/.venv/bin/python"
else PY="python"; fi

COMMON=(
  --highs-dir "$R/gpba" --an-dir "$R/an"
  --pf-seeds-gpbaa "$D/gpbaa_2d_gurobi_hard"
  --pf-seeds-an    "$D/an_2d_gurobi_hard"
  --output-dir "$OUT"
)
# The two instances the main article discusses; the other 23 go to the
# supplementary, and the main-article tables cover these two only.
EXAMPLES='^(lagos_nigeria_150|rio_de_janeiro_200)$'
EA_FRONTS='^(paris_150|tokyo_bay_200)$'

cd "$ROOT/sims-problem"
run() { echo "== $*"; "$PY" run_hv_experiments.py "$@"; }

# Figures
run --combined-figures  --combined-canvas cas "${COMMON[@]}"
run --ea-front-figures  --combined-canvas cas "${COMMON[@]}" --filter "$EA_FRONTS"

# Critical-difference diagrams: one per indicator, selected by
# --friedman-indicator (the flag --cd-diagram shares with --wilcoxon-ranking).
for ind in hv igd_plus_c; do
  run --cd-diagram --friedman-indicator "$ind" "${COMMON[@]}"
done

# Rank tests: wilcoxon_summary*.tex and wilcoxon_matrix*.tex
for ind in hv igd_plus_c; do
  run --wilcoxon-ranking --friedman-indicator "$ind" "${COMMON[@]}"
done

# Supplementary longtables, all 25 instances
for ind in hv igd_plus_c cardinality; do
  run --ea-tables --ea-tables-indicator "$ind" "${COMMON[@]}"
done

# Main-article tables, the two example instances only
for ind in hv igd_plus_c; do
  run --ea-tables --ea-tables-indicator "$ind" --ea-tables-two-column \
      "${COMMON[@]}" --filter "$EXAMPLES"
done

echo
echo "LaTeX in $OUT, EPS in ${OUT}_eps, PNG in ${OUT}_png"
