# SIMS Hybrid Algorithms - AI Agent Instructions

## Project Overview

Multi-objective optimization solver for the **Satellite Image Mosaic Selection (SIMS)** problem. Combines exact algorithms (MILP with AUGMECON/GPBA) and heuristics (Pareto Local Search) to find Pareto-optimal satellite image selections balancing cost, cloud coverage, resolution, and incidence angle.

## Architecture

### Component Structure (Hybrid Python/Rust)

```
sims-hybrid-algs/              # Root workspace
├── sims-core/                 # Python: High-level experiment framework
├── sims-solvers/              # Python: MILP/CP solvers (Gurobi, OR-Tools)
├── sims-problem/              # Rust→Python: PyO3 bindings exposing PLS & MILP
├── sims-heuristics/           # Rust: Pareto Local Search implementation
├── augmecon-rs/               # Rust: AUGMECON/GPBA multi-objective solver
├── pareto/                    # Rust: Pareto front data structure
└── src/sims_cli.py           # CLI entry point
```

**Key Integration Points:**
- `sims-problem` is a **Rust cdylib** with PyO3 bindings (`sims_problem.pyi`)
- Uses `maturin` to build Rust→Python bridge
- Python code imports `sims_problem` to call Rust solvers
- `augmecon-rs` is used by `sims-problem` for MILP solving (feature-gated: `milp`)

### Data Flow

```
Python CLI (sims_cli.py)
    ↓
sims-core (experiment.py)
    ↓
sims-problem Python bindings
    ↓ (PyO3)
Rust: solve_with_pls() / solve_with_milp() / solve_with_hybrid()
    ↓
sims-heuristics (PLS) ← → augmecon-rs (MILP/GPBA)
```

## Build System & Workflows

### CRITICAL Build Pattern

**ALWAYS use `uv` for Python package management and Rust→Python builds**. This is a `uv` workspace with:
- Root-level `pyproject.toml` defining workspace members
- `uv sync` installs all dependencies AND builds Rust extensions automatically
- No need to invoke `maturin` directly - `uv` handles it

### Standard Development Commands

```bash
# Initial setup and build (from workspace root)
uv sync                        # Install all Python deps + build Rust extensions
source .venv/bin/activate      # Activate venv

# Rebuild Rust components after changes
cd sims-problem
uv pip install -e . --reinstall-package sims-problem  # Debug build
uv pip install -e . --reinstall-package sims-problem --config-settings=build-args="--release"  # Release build

# Build standalone PLS binary
cd sims-heuristics
cargo build --release --bin pls
# OR with Docker:
./build-docker.sh

# Run via CLI
sims solve --experiments-dir <dir> --timeout-s 120 --front-strategy aneja-nair
sims prepare --satellite-data-dir <dir> --experiments-dir <output>
sims plots --experiments-dir <dir> --results-dir <dir> --output-dir <dir>

# Run benchmarks (from sims-problem/)
uv run python benchmark.py hybrid --instances-dir tests/data --validate-solutions
uv run python benchmark.py pls --instances-dir tests/data --max-iterations 100000
uv run python benchmark.py milp --instances-dir tests/data --grid-points 100
```

### VS Code Tasks

Use built-in tasks (Terminal → Run Task) - all use `uv`:
- **Build SIMS Project** - Builds release version with `uv`
- **Build SIMS Debug** - Builds debug version with `uv`
- **Test Lagos Nigeria 30** - Quick validation test with PLS
- **Full Build and Test** - Complete pipeline

## Key Conventions & Patterns

### Rust Code Organization

**Solution Representations** (`sims-heuristics/src/solution*.rs`):
- `EncodedSolution` - Vec-based solution (legacy)
- `BitsetEncodedSolution` - Bitset-based solution (optimized, current)
- `ResidualSolution` - Partial solutions for hybrid algorithms

**Problem Representations** (`sims-heuristics/src/problem*.rs`):
- Uses **MiniZinc `.dzn` format** for instance data
- Parse with `Problem::from_minizinc_datafile(path)`
- BitSet optimizations for coverage tracking

**Pareto Front Management**:
- Custom `ParetoFront` in `pareto/` crate
- BTreeMap-based archive in `sims-heuristics/src/solution_set_impl/`
- Domination checking via `dominates()` method

### Python-Rust Interop

**PyO3 Entry Points** (`sims-problem/src/solver.rs`):
```rust
#[pyfunction]
pub fn solve_with_pls(sims_instance: &SimsDiscreteProblem, ...) -> PyResult<SolvingResult>

#[pyfunction]
pub fn solve_with_milp(sims_instance: &SimsDiscreteProblem, ...) -> PyResult<SolvingResult>

#[pyfunction]
pub fn solve_with_hybrid(...) -> PyResult<SolvingResult>
```

**Python calls via type stubs** (`sims-problem/sims_problem.pyi`):
```python
from sims_problem import solve_with_pls, SimsDiscreteProblem, Solution
result = solve_with_pls(problem, timeout=timedelta(seconds=300), ...)
```

### Multi-Objective Solver Architecture

**AUGMECON Method** (`augmecon-rs/`):
- Epsilon-constraint with augmented objectives
- Grid-based Pareto front exploration
- GPBA-A/B/C variants for adaptive grid refinement
- Supports multiple solvers: Gurobi, COIN-CBC, HiGHS, SCIP

**Configuration Pattern** (`augmecon-rs/src/options.rs`):
```rust
let options = Options::new()
    .with_grid_points(50)
    .with_bypass_coefficient(true)  // AUGMECON2 optimization
    .with_flag_array(true)          // AUGMECON-R optimization
    .with_early_exit(true)
    .with_solver_option("log", "0");
```

### Testing & Validation

**Instance Naming Convention**:
- Test data: `tests/data/{city_name}_{size}.dzn`
- Examples: `lagos_nigeria_30`, `paris_100`, `tokyo_bay_50`

**Solution Validation** (in `benchmark.py`):
- `--validate-solutions` flag enables comprehensive checks
- Validates: solution structure, objective values, coverage constraints, Pareto dominance
- Use `--no-validate-solutions` for faster debugging

**Trace Files** (see `TRACE_SPECIFICATION.md`):
- Binary format: `trace.tar.gz` with objectives/dominated/timestamp/hypervolume
- Little-endian u64/u32 encoding
- Used for temporal analysis of optimization runs

## Common Pitfalls & Gotchas

### Shell Commands with run_in_terminal
**CRITICAL**: Append `; echo ""` to all shell commands when using `run_in_terminal` (see `augmecon-rs/.github/copilot-instructions.md`)

### Gurobi License Requirement
MILP algorithms require **Gurobi license** installed. Check with:
```bash
python -c "import gurobipy; gurobipy.Model()"
```

### Feature Flags
- `sims-problem` default build **excludes MILP** (faster compilation)
- Enable with: `uv pip install -e sims-problem --reinstall-package sims-problem --config-settings=build-args="--release --features milp"`
- `sims-heuristics` has `plotting` feature for visualization

### AUGMECON vs GPBA Inconsistencies
See `augmecon-rs_GPBA_Inconsistencies_Analysis.md`:
- `augmecon-rs` implementation differs from `sims-solvers` Python GPBA-A
- Interval management, solution deduplication, augmentation parameters vary
- Cross-reference both implementations when debugging MILP behavior

## File Navigation Shortcuts

**Core Algorithm Implementations**:
- PLS main loop: [sims-heuristics/src/pareto_local_search.rs](sims-heuristics/src/pareto_local_search.rs)
- MILP solver: [sims-problem/src/solver.rs](sims-problem/src/solver.rs)
- AUGMECON core: [augmecon-rs/src/epsilon_constraint.rs](augmecon-rs/src/epsilon_constraint.rs)
- GPBA phases: [augmecon-rs/src/gpba_phases.rs](augmecon-rs/src/gpba_phases.rs)

**Problem Instances**:
- Test data: [sims-problem/tests/data/](sims-problem/tests/data/)
- Instance parser: [sims-heuristics/src/problem.rs](sims-heuristics/src/problem.rs)

**CLI & Experiments**:
- Main CLI: [src/sims_cli.py](src/sims_cli.py)
- Benchmarking: [sims-problem/benchmark.py](sims-problem/benchmark.py)

## Documentation References

- Algorithm analysis: `SIMS_MILP_Algorithm_Analysis.md`
- Trace format: `TRACE_SPECIFICATION.md`, `TRACE_ARCHIVE_SPECIFICATION.md`
- AUGMECON: `augmecon-rs/docs/` (getting-started, solver-configuration, sims-cp/milp)
