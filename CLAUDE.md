# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-objective optimization solver for the **Satellite Image Mosaic Selection (SIMS)** problem. Combines exact algorithms (MILP via AUGMECON/GPBA) and heuristics (Pareto Local Search) to find Pareto-optimal satellite image selections balancing cost, cloud coverage, resolution, and incidence angle.

## Architecture

Hybrid Python/Rust monorepo with a `uv` workspace:

```
sims-hybrid-algs/
├── src/sims_cli.py           # CLI entry point (sims solve/prepare/plots)
├── sims-core/                # Python: high-level experiment framework
├── sims-solvers/             # Python: MILP/CP solvers (Gurobi, OR-Tools, MiniZinc)
├── sims-problem/             # Rust cdylib → Python via PyO3/maturin
├── sims-heuristics/          # Rust: Pareto Local Search (PLS, concurrent PLS)
├── augmecon-rs/              # Rust: AUGMECON/GPBA multi-objective MILP solver
├── pareto/                   # Rust: ParetoFront data structure
└── nd-tree/                  # Rust: ND-Tree for non-domination queries
```

**Data flow:**
```
Python CLI (sims_cli.py)
    ↓
sims-core (experiment.py) → sims-solvers (Gurobi/OR-Tools)
    ↓
sims-problem PyO3 bindings
    ↓ (PyO3)
sims-heuristics (PLS / ConcurrentPLS) ← → augmecon-rs (GPBA-A/B/C)
```

**Key integration:** `sims-problem` is a Rust cdylib built via `maturin`. Python imports it as `sims_problem`. Type stubs are at `sims-problem/sims_problem.pyi`. The `milp` feature flag gates the `augmecon-rs` dependency.

The Rust toolchain is **nightly** (see `rust-toolchain.toml`). Both `nd-tree` and `pareto` use `#![feature(adt_const_params)]`.

## Build Commands

```bash
# Initial setup — installs Python deps AND builds Rust extensions
uv sync
source .venv/bin/activate

# Rebuild sims-problem after Rust changes (debug, default)
cd sims-problem
uv pip install -e . --reinstall-package sims-problem

# Release build of sims-problem
uv pip install -e . --reinstall-package sims-problem \
  --config-settings=build-args="--release"

# Build with milp feature enabled (requires Gurobi)
uv pip install -e . --reinstall-package sims-problem \
  --config-settings=build-args="--release --features milp"

# Standalone PLS binary (for PLS_PATH env var)
cd sims-heuristics
cargo build --release --bin pls
export PLS_PATH=$PWD/target/release/pls
```

## Testing

### Python tests

```bash
# Run all Python tests from workspace root
uv run pytest sims-core/tests/
uv run pytest sims-problem/tests/
uv run pytest sims-solvers/tests/

# Run a single test file
uv run pytest sims-core/tests/test_trace.py -v

# Skip slow tests
uv run pytest sims-problem/tests/ -m "not slow"

# Use pseudo-solver (no Gurobi required) for sims-core tests
uv run pytest sims-core/tests/ --use-pseudo-solver
```

### Rust tests

```bash
# augmecon-rs (run from augmecon-rs/)
cargo test
cargo test test_gpba_phases           # Run a single test by name
RUST_LOG=debug cargo test -- --nocapture

# sims-heuristics feature-gated tests
cargo test --features parallel --test test_concurrent_pls
cargo test --features external_solvers --test test_external_solvers

# nd-tree / pareto (from their directories)
cargo test
```

### Lint & format (Rust)

```bash
cargo fmt --check
cargo clippy -- -D warnings
```

Python linting uses `ruff` (dev dependency in root `pyproject.toml`).

## Key Algorithms

### Pareto Local Search (`sims-heuristics`)

- Main loop: `sims-heuristics/src/pareto_local_search.rs`
- Concurrent variant: `sims-heuristics/src/concurrent_pls/orchestrator.rs`
- Solution types: `BitsetEncodedSolution` (active), `VecEncodedSolution` (legacy)
- Archive types: `NdTreeSolutionSet` (default for 3D/4D), `BTreeSolutionSet`, `LinkedListSolutionSet`
- Problem input: MiniZinc `.dzn` files, parsed via `Problem::from_minizinc_datafile()`

**Feature flags** (sims-heuristics):
- `bitmaps` — BitSet-based solution encoding (in default)
- `parallel` — ConcurrentPLS with crossbeam channels
- `external_solvers` — moors/optirustic integration
- `probabilistic_probing` — experimental probing
- `scalarized_selection` — Chebycheff-based archive queries

### AUGMECON/GPBA (`augmecon-rs`)

- Core ε-constraint: `augmecon-rs/src/epsilon_constraint.rs`
- GPBA-A main algorithm: `augmecon-rs/src/gpba.rs`
- Phased testable API: `augmecon-rs/src/gpba_phases/`
- Options builder: `augmecon-rs/src/options.rs`

GPBA-A works internally in **maximization form** (objectives negated). Solutions in the Pareto front are stored in minimization form. See the sign convention note in `augmecon-rs/src/gpba.rs`.

**Solver backends** (feature flags): `highs`, `coin_cbc`, `scip`. The `good_lp` dependency points to a forked repo at `github.com/hlvlad/good_lp`.

### Python MILP (reference implementation)

`sims-solvers/sims_solvers/` contains the Python GPBA-A reference implementation used for cross-validation. When debugging MILP behavior, cross-reference `augmecon-rs/src/gpba.rs` against it. Known divergences are documented in `augmecon-rs_GPBA_Inconsistencies_Analysis.md`.

## PyO3 Entry Points (`sims-problem/src/solver.rs`)

```rust
solve_with_pls(sims_instance, timeout, ...)  -> SolvingResult
solve_with_milp(sims_instance, timeout, ...) -> SolvingResult   // milp feature
solve_with_hybrid(...)                        -> SolvingResult   // milp feature
```

## Instance Data

- Format: MiniZinc `.dzn` files
- Test instances: `sims-problem/tests/data/{city}_{size}.dzn`
  - Examples: `lagos_nigeria_30`, `rio_de_janeiro_150`, `tokyo_bay_50`
- Publication experiments: `publication-data/experiments/`

## Environment

```bash
# Required for MILP algorithms
export PLS_PATH=/path/to/sims-heuristics/target/release/pls

# Gurobi must be licensed; verify with:
python -c "import gurobipy; gurobipy.Model()"
```

Copy `.env.dev` to `.env` and adjust paths for local configuration.

## Trace Files

Binary trace format (`trace.tar.gz`): little-endian u64/u32 encoding of objectives, dominated-count, timestamp, and hypervolume per step. Spec: `TRACE_SPECIFICATION.md`, `TRACE_ARCHIVE_SPECIFICATION.md`. Trace I/O is in `sims-problem/src/trace.rs`.
