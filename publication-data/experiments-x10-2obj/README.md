# Biobjective 10-run experiment set (100–200 ladder)

Raw results behind the tables and figures of the biobjective SIMS paper. Every
configuration was run **10 times**; `run_hvs` in each JSON holds the per-run
final hypervolume, and `final_hv_std` its standard deviation.

## Protocol

| | |
|---|---|
| Objectives | `min_cost`, `cloud_coverage` |
| Instances | 5 AOIs × {100, 125, 150, 175, 200} images = 25 |
| Instance source | `publication-data/satellite-data/instances_hard_an` |
| Budget | 200 s wall clock per configuration, uniform across sizes |
| Runs | 10 per configuration |
| Exact:heuristic ratios | 100:0, 75:25, 50:50, 25:75, 0:100 |
| Second-phase algorithms | PLS, NSGA-II, NSGA-III, MOEA/D |
| Exact phase | GPBA-A (`gpba/`) and anytime Aneja & Nair (`an/`) |
| MILP solver | Gurobi 12.0.3 |
| Machine | Intel Xeon Gold 5320 @ 2.20 GHz, 8 cores, 32 GB, Ubuntu |

17 configurations per instance: the exact phase alone, each of the four
second-phase algorithms at the three intermediate ratios, and each run alone.

## Layout

    gpba/                 GPBA-A-seeded results
      <instance>.json     per-instance results, all 17 configurations
      traces/             anytime traces for the canonical run
    an/                   the same, seeded by anytime Aneja & Nair
    scripts/              the runners used to produce this set

Only the canonical run's traces are included. Per-run traces for runs 2–10 are
~200 MB and are not needed to rebuild any table or figure: every generator
reads the canonical trace plus `run_hvs`.

## Reproducing the tables and figures

From `sims-problem/`, with the seed sets in `sims-core/tests/data/`:

    R=../publication-data/experiments-x10-2obj
    D=../sims-core/tests/data
    COMMON="--highs-dir $R/gpba --an-dir $R/an \
            --pf-seeds-gpbaa $D/gpbaa_2d_gurobi_hard \
            --pf-seeds-an $D/an_2d_gurobi_hard \
            --filter '_(100|125|150|175|200)$' --output-dir out"

    # per-indicator longtables (hv, 1-IGD+, front cardinality)
    for ind in hv igd_plus_c cardinality; do
      python run_hv_experiments.py --ea-tables --ea-tables-indicator $ind $COMMON
    done

    # rank tests, per indicator
    for ind in hv igd_plus_c; do
      python run_hv_experiments.py --wilcoxon-ranking --friedman-indicator $ind $COMMON
      python run_hv_experiments.py --friedman         --friedman-indicator $ind $COMMON
    done

    # figures
    python run_hv_experiments.py --pareto-fronts $COMMON
    python run_hv_experiments.py --ea-bars       $COMMON
    python run_hv_experiments.py --ea-bars-indicator igd_plus_c $COMMON

## Re-running the experiments

`scripts/x10.sh <timeout_s> <runs> <workers> <out_root>` reproduces this set:

    ./x10.sh 200 10 7 ~/x10

It deals one job per (source, instance) round-robin to `<workers>` processes,
each pinned to a core with `taskset`. 7 workers on an 8-core machine was
validated against a sequential baseline before use: 45 paired observations,
worst deviation −0.03 % (`scripts/ab_repeat.sh`). Do not raise the worker count
without repeating that check — the per-configuration budget is wall clock, so a
worker that does not hold a full core silently gets less compute for the same
nominal budget.

Wall clock for this set: 19 h 48 m across 7 workers; ~277 CPU-hours.
