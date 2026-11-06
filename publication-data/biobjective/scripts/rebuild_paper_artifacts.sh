#!/usr/bin/env bash
# Rebuild every figure and table in the biobjective SIMS paper from the
# committed results. No solving, no Gurobi, no network.
#
#   ./rebuild_paper_artifacts.sh [out_dir]
#
# Writes LaTeX to <out_dir>, EPS to <out_dir>_eps and PNG to <out_dir>_png.
#
# Reads the stored result set by default. To rebuild from a fresh run instead,
# set GPBA_DIR and AN_DIR to its consolidated output directories, and
# SEEDS_GPBAA / SEEDS_AN if the exact phase was re-solved somewhere else:
#
#   GPBA_DIR=~/biobjective-run/gpba_all AN_DIR=~/biobjective-run/an_all ./rebuild_paper_artifacts.sh out
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Pin the string-hash seed so the output is byte-reproducible. The critical
# difference diagrams come from scikit-posthocs, whose clique grouping iterates
# a set keyed on method names; with Python's hash randomisation left on, two
# runs of identical code over identical data emit the same crossbars traversed
# in a different order. The geometry is unaffected -- the rendered figures
# differ in 41 of 769,494 pixels, at stroke caps where bars overlap -- but the
# EPS is not byte-identical, which makes "regenerated unchanged" unverifiable.
export PYTHONHASHSEED=0
# matplotlib stamps %%CreationDate into every EPS from the wall clock, which
# makes otherwise identical output differ by one line. The PS backend honours
# SOURCE_DATE_EPOCH, so pin it and the figures become byte-reproducible.
export SOURCE_DATE_EPOCH=0
# Resolved against the caller's directory, not against sims-problem/: the
# generator runs from there, so a relative argument would otherwise land
# inside the source tree instead of where it was asked for.
OUT="${1:-paper-artifacts}"
[[ "$OUT" = /* ]] || OUT="$PWD/$OUT"

R="$ROOT/publication-data/biobjective"
D="$ROOT/sims-core/tests/data"
# The workspace venv, so the script works without activating it first.
if [[ -n "${PYTHON:-}" ]]; then PY="$PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then PY="$ROOT/.venv/bin/python"
else PY="python"; fi

COMMON=(
  --highs-dir "${GPBA_DIR:-$R/gpba}" --an-dir "${AN_DIR:-$R/an}"
  --pf-seeds-gpbaa "${SEEDS_GPBAA:-$D/gpbaa_2d_gurobi_hard}"
  --pf-seeds-an    "${SEEDS_AN:-$D/an_2d_gurobi_hard}"
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
