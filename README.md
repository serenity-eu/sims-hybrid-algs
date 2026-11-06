# SIMS Hybrid Algorithms

Solvers and heuristics for Satellite Image Mozaic Selection (SIMS) problem.

## Overview

This repository provides solvers and heuristics for the Satellite Image Mosaic Selection (SIMS) problem. The SIMS problem involves selecting an optimal set of satellite images to create a seamless mosaic, considering constraints such as coverage, quality, and cost. The algorithms implemented here are designed to efficiently tackle various instances of this problem.

## Prerequisites

- **Rust nightly** -- pinned in `rust-toolchain.toml`, so `rustup` picks it up
  automatically
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** -- installs
  Python and every Python dependency, `maturin` included
- **A C linker** -- `sudo apt install build-essential`, which `rustc` needs to
  link; nothing here compiles C or C++ of its own
- **Gurobi**, licensed, with `GUROBI_HOME` set -- see `scripts/fetch-gurobi.sh`.
  Required to re-run the experiments. *Not* required to rebuild the paper's
  tables and figures from the stored results.

That is the whole list: the Gurobi bindings ship pregenerated, so the build
pulls no system library other than libgurobi itself.

Two extras, each needed only outside the path above:

- `libssl-dev`, for `cargo install patch-crate`
- `cmake`, `pkg-config` and `coinor-libcbc-dev`, only if you opt into the other
  MILP backends with `--features gurobi,highs` or `--features gurobi,coin_cbc`.
  HiGHS is compiled from source and CBC is linked from the system, which is why
  they are not in the default build -- the published results use Gurobi
  throughout.

Optional: `patchelf`, which silences a harmless `maturin` rpath warning.

## Installation

Set up the workspace:

```bash
git clone https://github.com/serenity-eu/sims-hybrid-algs.git
cd sims-hybrid-algs
uv sync
source .venv/bin/activate
```

`uv sync` builds the Rust extension, which is how Pareto Local Search is
called -- in process, through the Python bindings. There is no separate
binary to build and no `PLS_PATH` to set.

The Gurobi backend needs a licensed Gurobi install (`scripts/fetch-gurobi.sh`
sets up `GUROBI_HOME`), one patched dependency, and the Python extension:

```bash
cargo install patch-crate
(cd augmecon-rs  && cargo patch-crate)
(cd sims-problem && maturin develop --release --features gurobi)
```


## Reproducing the published results

> [!NOTE]
> The commands under **Usage** below are the original single-run pipeline, not
> what produced the biobjective paper's results. Use this section for those.

### Rebuild the tables and figures

Reads the stored results. No solver, no licence, no network.

```bash
cd publication-data/biobjective
./scripts/rebuild_paper_artifacts.sh out
```

LaTeX in `out/`, EPS in `out_eps/`, PNG in `out_png/`.

### Re-run the experiments

Needs a licensed Gurobi. Run from the repository root, in order.

```bash
# 1. Problem instances
(cd sims-preprocessing && ./generate_hard_an_benchmark.sh \
    ../publication-data/satellite-data/instances_hard_an 100,125,150,175,200)

# 2. Exact phase
(cd sims-problem && python generate_pseudo.py --method gpba  --solver gurobi --instance-set hard)
(cd sims-problem && python generate_pseudo.py --method aneja --solver gurobi --instance-set hard)

# 3. Hybrid experiments
publication-data/biobjective/scripts/run_all_experiments.sh 200 10 7 ~/biobjective-run

# 4. Tables and figures from the new results
(cd publication-data/biobjective && \
    GPBA_DIR=~/biobjective-run/gpba_all AN_DIR=~/biobjective-run/an_all ./scripts/rebuild_paper_artifacts.sh out)
```

Step 3 resolves this checkout and `.venv` by itself; set `LD_LIBRARY_PATH`
and `GRB_LICENSE_FILE` if your Gurobi is not in `/opt`.

Step 1 is optional: its 25 instances are already in the repository, and it
rewrites them byte for byte. Step 2 writes their exact fronts to
`sims-core/tests/data/{gpbaa,an}_2d_gurobi_hard/`, which step 3 reads. Step 3
takes about 20 h on 8 cores.

Steps 1 and 2 are deterministic. Step 3 is not: its budget is 200 s of wall
clock per configuration, so a re-run agrees statistically, not bitwise.
Rebuild from the stored results to reproduce the published figures exactly.

### Protocol

- Objectives: `min_cost`, `cloud_coverage`
- Instances: 5 AOIs x {100, 125, 150, 175, 200} images = 25
- Budget: 200 s wall clock per configuration, uniform across sizes
- Repetitions: 10 per configuration, except the deterministic exact phase
- Exact:heuristic ratios: 100:0, 75:25, 50:50, 25:75, 0:100
- Exact phase: GPBA-A (`gpba/`) and anytime Aneja & Nair (`an/`)
- Second-phase algorithms: PLS, NSGA-II, NSGA-III, MOEA/D
- 17 configurations per instance: the exact phase alone, each second-phase
  algorithm at the three intermediate ratios, and each run alone
- Gurobi 12.0.3, Intel Xeon Gold 5320 @ 2.20 GHz, 8 cores, 32 GB, Ubuntu

Results are in `publication-data/biobjective/`: `gpba/` seeded by GPBA-A, `an/`
by anytime Aneja & Nair, one JSON per instance covering all 17 configurations,
with `run_hvs` giving the per-run final hypervolume and `final_hv_std` its
standard deviation.

## Usage

### Solving

> [!IMPORTANT]
> This algorithm utilizes the **Gurobi** solver for optimization.  
> Ensure that the Gurobi license is already installed and properly configured on your machine before running this code.

After installation, you can run the hybrid solver with Anytime Aneja Nair method as follows:

```bash
sims solve --experiments-dir ./publication-data/experiments --timeout-s 120 --front-strategy aneja-nair --results-dir ./publication-data/new-results/aneja-nair
```

You can run the hybrid solver with GPBA-A method as follows:

```bash
sims solve --experiments-dir ./publication-data/experiments  --timeout-s 120 --front-strategy gpba-a  --results-dir ./publication-data/new-results/gpba-a
```

For more options and help:

```bash
sims solve --help
```

### Generating SIMS problem instances from satelite images

To regenerate SIMS problem instances from original satellite data, run following command

```bash
sims prepare --satellite-data-dir ./publication-data/satellite-data --experiments-dir ./publication-data/new-experiments
```

### Generating plots from experiment results

To generate plots for experiments results, run following command

```bash
sims plots --experiments-dir ./publication-data/experiments/ --results-dir ./publication-data/results/gpba-a ./publication-data/results/aneja-nair/ --output-dir ./publication-data/new-plots/
```