# Biobjective experiment set

Raw results behind the tables and figures of the biobjective SIMS paper. Each
JSON holds every configuration for one instance, with `run_hvs` giving the
per-run final hypervolume and `final_hv_std` its standard deviation.

## Protocol

- Objectives: `min_cost`, `cloud_coverage`
- Instances: 5 AOIs x {100, 125, 150, 175, 200} images = 25
- Budget: 200 s wall clock per configuration, uniform across sizes
- Repetitions: 10 per configuration, except the deterministic exact phase
- Exact:heuristic ratios: 100:0, 75:25, 50:50, 25:75, 0:100
- Second-phase algorithms: PLS, NSGA-II, NSGA-III, MOEA/D
- Exact phase: GPBA-A (`gpba/`) and anytime Aneja & Nair (`an/`)
- 17 configurations per instance: the exact phase alone, each second-phase algorithm at the three intermediate ratios, and each run alone
- Gurobi 12.0.3, Intel Xeon Gold 5320 @ 2.20 GHz, 8 cores, 32 GB, Ubuntu
- Wall clock: 19 h 48 m across 7 workers, ~277 CPU-hours

## Layout

```
gpba/                 GPBA-A-seeded results
  <instance>.json     per-instance results, all 17 configurations
  traces/             anytime traces for the canonical run
an/                   the same, seeded by anytime Aneja & Nair
scripts/              rebuild.sh and the runners that produced this set
```

The 25 problem instances live in `../satellite-data/instances_hard_an/`, which
is otherwise gitignored: the five sizes this paper uses are committed, the
larger sizes 225-300 are not.

Only the canonical run's traces are committed. The remaining per-run traces
are ~200 MB and no generator reads them.

## Regenerating the paper's figures and tables

Everything needed is committed, so this needs no solver and no network. Run
following command:

```bash
./scripts/rebuild.sh out
```

LaTeX lands in `out/`, EPS in `out_eps/`, PNG in `out_png/`.
`paper/elsa-paper/README.md` lists which file goes where in the manuscript.

## Re-running the experiments

Requires a Gurobi licence. Nothing needs recreating: the 25 problem instances
and the exact-phase seed sets are both committed.

The instances also regenerate byte-for-byte, so run following command if you
would rather rebuild them than trust the committed copies:

```bash
cd sims-preprocessing
./generate_hard_an_benchmark.sh ../publication-data/satellite-data/instances_hard_an 100,125,150,175,200
```

The per-instance seed is derived from the city name, the size and `--seed`
(default 0). Verify against `paper/elsa-paper/problem_instances.tex`: in each
`meta.json`, `N` is the image count and `universe` the fragment count.

Then run following command to re-run the experiments:

```bash
scripts/x10.sh 200 10 7 ~/x10
```

Arguments are `<timeout_s> <runs> <workers> <out_root>`. One job per
(source, instance) is dealt round-robin to the workers, each pinned to a core
with `taskset`. 7 workers on 8 cores was validated against a sequential
baseline first: 45 paired observations, worst deviation -0.03 %
(`scripts/ab_repeat.sh`). Do not raise the worker count without repeating that
check, because the per-configuration budget is wall clock and a worker without
a full core silently gets less compute.

The script hardcodes the run machine's checkout path and Gurobi locations.
Adjust them for your setup. Results land in `<out_root>/{gpba,an}_all/`, which
is what `gpba/` and `an/` here were copied from.

## Notes

- To regenerate the exact-phase seed sets, run `generate_pseudo.py --method
  gpba --solver gurobi --instance-set hard` from `sims-problem/`, and again
  with `--method aneja`. `--instance-set hard` covers sizes 125-350; the
  size-100 files came from a separate invocation and are committed.
- `problem_instances.tex` has no generator; it was written from the `N` and
  `universe` fields of `instances_hard_an/*.meta.json`.
