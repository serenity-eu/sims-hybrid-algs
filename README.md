# SIMS Hybrid Algorithms

Solvers and heuristics for Satellite Image Mozaic Selection (SIMS) problem.

## Overview

This repository provides solvers and heuristics for the Satellite Image Mosaic Selection (SIMS) problem. The SIMS problem involves selecting an optimal set of satellite images to create a seamless mosaic, considering constraints such as coverage, quality, and cost. The algorithms implemented here are designed to efficiently tackle various instances of this problem.

## Installation

It is recommended to use [uv](https://github.com/astral-sh/uv) for fast and reliable Python package management.
You can find installation instruction for uv [here](https://docs.astral.sh/uv/getting-started/installation/).

After uv is installed, proceed with following commands to setup workspace:

```bash
git clone https://github.com/serenity-eu/sims-hybrid-algs.git
cd sims-hybrid-algs
uv sync
source .venv/bin/activate
```

You also need to build binary executable for Pareto Local Search from `sims-heuristics` Rust project. If you don't have Rust toolchain installed already, use [this instruction](http://rust-lang.org/tools/install) for installation.

Run following command to build PLS:

```bash
cd sims-heuristics
cargo build --release pls
cp target/release/pls .
```

Alternatively, you can build PLS binary with Docker using following script

```bash
cd sims-heuristics
./build-docker.sh
```

After PLS executable is built, store path to it in `PLS_PATH` environment variable

```bash
export PLS_PATH=$PWD/sims-heuristics/pls
```


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