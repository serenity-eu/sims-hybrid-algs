#!/usr/bin/env bash
#
# Prepare a fresh clone for building.
#
# Four manifests -- augmecon-rs, sims-problem, sims-heuristics and psbox --
# declare `[patch.crates-io]` entries pointing into `target/patch/`. Cargo
# resolves `[patch]` while reading a manifest, so until `cargo patch-crate`
# has written those directories, no cargo or maturin command in the workspace
# runs at all, and `uv sync` fails while building sims-problem.
#
# Usage:
#   scripts/bootstrap.sh              # patches, Gurobi runtime, uv sync
#   scripts/bootstrap.sh --no-gurobi  # skip the Gurobi download
#   scripts/bootstrap.sh --no-sync    # stop before uv sync
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

want_gurobi=1
want_sync=1
for arg in "$@"; do
    case "$arg" in
        --no-gurobi) want_gurobi=0 ;;
        --no-sync)   want_sync=0 ;;
        -h|--help)   sed -n '3,14p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *) echo "error: unknown option '$arg'" >&2; exit 2 ;;
    esac
done

step() { printf '\n==> %s\n' "$1"; }

for tool in cargo; do
    command -v "$tool" >/dev/null || { echo "error: $tool not on PATH" >&2; exit 1; }
done

step "cargo patch-crate"
if ! command -v cargo-patch-crate >/dev/null; then
    echo "patch-crate not installed; installing it"
    cargo install patch-crate
fi
# Both are required. sims-problem and psbox read augmecon-rs' patched grb, and
# sims-problem depends on sims-heuristics, which patches optirustic.
for crate in augmecon-rs sims-heuristics; do
    (cd "$crate" && cargo patch-crate)
done

if (( want_gurobi )); then
    step "Gurobi runtime"
    if [[ -n "${GUROBI_HOME:-}" ]]; then
        echo "GUROBI_HOME already set to $GUROBI_HOME; leaving it alone"
    elif grep -q 'GUROBI_HOME' .cargo/config.toml 2>/dev/null; then
        echo ".cargo/config.toml already sets GUROBI_HOME; leaving it alone"
    else
        ./scripts/fetch-gurobi.sh
    fi
fi

if (( want_sync )); then
    step "uv sync"
    command -v uv >/dev/null || { echo "error: uv not on PATH" >&2; exit 1; }
    uv sync
    printf '\nDone. Activate the environment with:\n    source .venv/bin/activate\n'
else
    printf '\nDone (uv sync skipped).\n'
fi
