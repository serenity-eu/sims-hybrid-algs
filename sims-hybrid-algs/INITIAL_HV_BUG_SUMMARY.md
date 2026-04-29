# Initial Hypervolume Bug - Executive Summary

**Date:** 2025-04-23  
**Instance Tested:** `lagos_nigeria_100` (100 satellite images)  
**Status:** ✅ **CONFIRMED BUG** - High Priority

---

## Executive Summary

PLS-based algorithms incorrectly report 75-91% of their final hypervolume at t=0, making them appear to converge almost instantly. This is caused by greedy initialization solutions being artificially timestamped at 1-50 microseconds instead of their true discovery time.

**Key Finding:** Initialization actually takes **1.16 seconds** and generates **131 solutions**, but these are timestamped at 0.000001-0.000131 seconds in the trace data.

---

## Evidence

### Measured Hypervolume at t=0

| Algorithm        | HV at t=0 | Final HV | Initial as % of Final | Status      |
|------------------|-----------|----------|-----------------------|-------------|
| Pure PLS         | 0.7548    | 0.9379   | **80.5%**            | ❌ **BUG**  |
| Improved PLS     | 0.8604    | 0.9472   | **90.8%**            | ❌ **BUG**  |
| Diverse PLS      | ~0.75-0.86| ~0.94    | **~80-91%**          | ❌ **BUG**  |
| Hybrid 50:50     | 0.0000    | 0.9563   | 0.0%                 | ✅ Correct  |
| Improved Hybrid  | 0.0000    | 0.9560   | 0.0%                 | ✅ Correct  |

### Actual vs. Reported Initialization Time

- **Reported Timestamps:** 1, 2, 3, ..., 131 microseconds (0.000001-0.000131 seconds)
- **Actual Initialization Time:** 1.156 seconds (measured from trace gap analysis)
- **Discrepancy:** 9,954× faster than reality!

### Trace Data Analysis

From `measure_init_time.py` run on `lagos_nigeria_100`:

```
INITIALIZATION PHASE:
  Number of initial solutions:     131
  Non-dominated initial solutions: 131
  Initial solutions timestamp:     1 to 131 µs
  True initialization time:        1.156 seconds

OPTIMIZATION PHASE:
  First optimized solution at:     1,155,540 µs (1.156 seconds)
  Gap between init and optim:      1,155.409 ms
```

**First 10 timestamps from trace:**
```
  [0]     1 µs (0.001 ms) ← INIT
  [1]     2 µs (0.002 ms) ← INIT
  [2]     3 µs (0.003 ms) ← INIT
  [3]     4 µs (0.004 ms) ← INIT
  ...
  [130] 131 µs (0.131 ms) ← INIT
  [131] 1,155,540 µs (1155.540 ms) ← FIRST OPTIMIZED SOLUTION
```

---

## Root Cause

**Location:** `sims-heuristics/src/pareto_local_search.rs:234-265`

```rust
let mut initial_timestamp_us: u64 = 1;
initial_population.iter().for_each(|solution| {
    if population.try_insert(solution) {
        explored_solutions.register(
            0,
            solution,
            Duration::from_micros(initial_timestamp_us),  // ← BUG: Should be actual time
            solution.selected_images().collect(),
        );
        initial_timestamp_us += 1;  // ← Incrementing by 1 µs per solution
    }
});
```

**Problem:**
1. Initial solutions (greedy + random) are generated **before** optimization timer starts
2. These are assigned fake timestamps of 1, 2, 3... microseconds
3. When HV curves sample at t=0, all 131 initial solutions are included
4. Result: HV curve starts at 75-91% instead of 0%

---

## Impact

### 1. Misleading Visualizations

The HV-over-time plots show PLS algorithms with:
- Immediate high HV at t=0 (75-91% of final)
- Flat or slowly increasing curves thereafter
- **False impression** that PLS converges instantly

Example from `lagos_nigeria_100`:
- Pure PLS: Starts at 0.7548, improves only 0.1831 over 40 seconds
- **Reality**: Most of that 0.7548 comes from 1.16s of initialization, not instant convergence

### 2. Unfair Algorithm Comparison

- **PLS algorithms** appear to have excellent "anytime" performance due to high initial HV
- **Hybrid algorithms** correctly start at 0.0, appearing worse in early stages
- Cannot accurately compare first few seconds of optimization

### 3. Invalid Performance Metrics

Current results report:
- Pure PLS: HV improvement = +0.1831 (+19.5% relative)
- **Truth**: 80.5% came from initialization, only 19.5% from actual optimization

### 4. Scientific Reproducibility

Published results and plots show this artifact, affecting:
- Paper figures showing HV curves
- Benchmark comparisons
- Algorithm ablation studies
- Performance claims about PLS convergence speed

---

## Visual Evidence

See generated plots:
- **`sims-problem/hv_experiment_results/lagos_nigeria_100_initial_hv_bug.png`**
  - Top panel: Full 40-second run showing PLS starting high
  - Bottom panel: Zoomed to first 2 seconds with purple line showing true init time (1.16s)
  - Clear annotation of the discrepancy

Key observations from plots:
1. **Red shaded area at t=0**: Shows where bug manifests
2. **Purple dashed line at t=1.16s**: Actual initialization completion time
3. **PLS curves (red)**: Start at ~0.75-0.86 HV
4. **Hybrid curves (green)**: Correctly start at 0.0

---

## Proposed Fix (Recommended)

### Option A: Start Timer AFTER Initialization ✅ RECOMMENDED

**Implementation:**
1. Generate initial population (greedy + random solutions)
2. **Start optimization timer**
3. Begin PLS main loop
4. Timestamp all initial solutions at t=0
5. First optimized solutions get t > 0

**Benefits:**
- Clean separation: initialization vs. optimization
- Accurate representation of optimization dynamics
- Initial HV can be reported as separate metric
- Matches user expectation of "optimization time"

**Code changes:**
```rust
// In solve_with_pls entry point
let initial_population = create_initial_population(&problem, &config);
let optimization_start = Instant::now();  // ← Start timer HERE

let mut pls = ParetoLocalSearch::new(
    &problem,
    &initial_population,
    optimization_start,  // Pass timer
    ...
);
```

### Alternative Options

**Option B:** Normalize timestamps post-hoc
- Subtract first optimized solution timestamp from all timestamps
- More complex; requires identifying init/optim boundary

**Option C:** Report initial HV separately
- Add `initial_hv` field to results
- Plot as horizontal reference line
- Doesn't fix underlying timestamp issue

---

## Testing & Validation

### Completed Diagnostics

1. ✅ **`debug_initial_hv.py`**: Analyzed trace timestamps, confirmed 1-131 µs pattern
2. ✅ **`measure_init_time.py`**: Measured true init time = 1.156s for Diverse PLS
3. ✅ **`visualize_initial_hv_bug.py`**: Generated annotated plots showing the issue

### Required Regression Tests

1. Run full experiment suite after fix (all instances, all configs)
2. Verify PLS curves start at or near 0.0
3. Confirm hybrid methods unchanged
4. Check that initialization time is properly accounted in reports

### Success Criteria

- [ ] PLS algorithms show HV ≈ 0.0 at t=0
- [ ] Initial solutions timestamped at actual discovery time
- [ ] Documentation updated with initialization time reporting
- [ ] All plots regenerated with corrected data

---

## Affected Configurations

### ❌ Affected (use greedy initialization)
- Pure PLS
- Improved PLS
- Diverse Probe PLS
- Scalarized PLS
- Diverse+Scalarized PLS

### ✅ Not Affected
- Hybrid 50:50 (MILP phase starts at true t=0)
- Improved Hybrid (MILP phase starts at true t=0)
- NSGA-II (if implemented)
- MOEA/D (if implemented)

---

## Action Items

1. **[High Priority]** Implement Option A fix in `pareto_local_search.rs`
2. **[High Priority]** Re-run all HV experiments with corrected timestamps
3. **[Medium]** Update `TRACE_SPECIFICATION.md` with initialization semantics
4. **[Medium]** Add `initial_hv` and `init_time_s` fields to result JSON
5. **[Low]** Consider adding warning/note to existing published results

---

## References

### Scripts Created
- `sims-problem/debug_initial_hv.py` - Trace timestamp analysis
- `sims-problem/measure_init_time.py` - Initialization time measurement
- `sims-problem/visualize_initial_hv_bug.py` - Visualization with annotations

### Documentation
- `INITIAL_HV_BUG_ANALYSIS.md` - Detailed technical analysis
- This file - Executive summary

### Data Files
- `hv_experiment_results/lagos_nigeria_100.json` - Original results showing bug
- `sims-problem/hv_experiment_results/lagos_nigeria_100_initial_hv_bug.png` - Annotated plot

### Related Code
- `sims-heuristics/src/pareto_local_search.rs:234-265` - Bug location
- `sims-problem/src/solver.rs` - Python binding entry points
- `sims-problem/run_hv_experiments.py` - Experiment framework

---

## Conclusion

This is a **confirmed, high-impact bug** affecting all PLS-based algorithm experiments. Initial solutions are artificially timestamped at microseconds instead of their true discovery time (~1.16s for size-100 instances), causing HV curves to incorrectly start at 75-91% of final values.

**Recommended action:** Implement Option A (start timer after initialization) and regenerate all experiment results.

**Estimated effort:** 4-8 hours (implementation + testing + documentation + re-running experiments)

---

**Report prepared by:** AI Analysis  
**Instance tested:** `lagos_nigeria_100.dzn`  
**Tools used:** `measure_init_time.py`, `visualize_initial_hv_bug.py`, `debug_initial_hv.py`
