# Initial Hypervolume Bug Analysis

## Executive Summary

**Problem:** Hypervolume (HV) curves for pure PLS algorithms start at 75-86% of their final value at timestamp t=0, making it appear that most optimization progress occurs instantly.

**Root Cause:** Greedy initial solutions are assigned timestamps of 1, 2, 3... microseconds, which effectively round to t=0 when sampling HV curves.

**Impact:** Misleading visualization and analysis of optimization dynamics; makes PLS appear to converge instantly rather than showing gradual improvement.

**Status:** Confirmed in `lagos_nigeria_100` and likely affects all instances with greedy initialization.

---

## Problem Description

### Observed Behavior

When running HV experiments with `run_hv_experiments.py`, PLS-based algorithms show anomalously high initial HV values:

| Algorithm        | HV at t=0 | Final HV | Initial as % of Final |
|-----------------|-----------|----------|-----------------------|
| Pure PLS        | 0.7548    | 0.9379   | **80.5%**            |
| Improved PLS    | 0.8604    | 0.9472   | **90.8%**            |
| Diverse PLS     | (similar) | (similar)| **~75-90%**          |
| Hybrid 50:50    | 0.0000    | 0.9563   | 0.0% (correct)       |
| Improved Hybrid | 0.0000    | 0.9560   | 0.0% (correct)       |

**This means PLS algorithms appear to achieve 75-91% of their final HV value before any actual optimization occurs!**

### Visual Impact

The HV curve plots show:
- PLS curves starting high and increasing slowly
- Hybrid curves starting at 0 and ramping up after phase 1 (correct behavior)
- Misleading comparison suggesting hybrids have better "anytime" performance

---

## Root Cause Analysis

### Code Location

**File:** `sims-heuristics/src/pareto_local_search.rs`  
**Function:** `ParetoLocalSearch::new()`  
**Lines:** 234-265

```rust
let mut initial_timestamp_us: u64 = 1;
initial_population.iter().for_each(|solution| {
    if population.try_insert(solution) {
        explored_solutions.register(
            0,
            solution,
            Duration::from_micros(initial_timestamp_us),  // ← BUG HERE
            solution.selected_images().collect(),
        );
        initial_timestamp_us += 1;  // Incrementing by 1 microsecond
    }
});

// Greedy initialization (when enabled)
if optimizations.use_greedy_initial_population {
    let greedy_solutions = T::greedy_initial_solutions(problem);
    for solution in &greedy_solutions {
        if population.try_insert(solution) {
            explored_solutions.register(
                0,
                solution,
                Duration::from_micros(initial_timestamp_us),  // ← BUG HERE
                solution.selected_images().collect(),
            );
            initial_timestamp_us += 1;  // Still microseconds!
        }
    }
}
```

### The Issue

1. **Initial solutions** (from random or greedy initialization) are generated **before** the optimization timer starts
2. These solutions are assigned timestamps of **1µs, 2µs, 3µs, ...** (microseconds)
3. When HV curves are sampled at t=0 seconds, all these solutions are included
4. Result: The HV curve starts high instead of showing the true optimization trajectory

### Evidence from Trace Data

For `lagos_nigeria_100` with 10-second timeout:

```
First 20 unique timestamps (µs):
  t=     1 µs (0.001 ms):   1 solutions,   1 non-dominated
  t=     2 µs (0.002 ms):   1 solutions,   1 non-dominated
  t=     3 µs (0.003 ms):   1 solutions,   1 non-dominated
  ...
  t=    20 µs (0.020 ms):   1 solutions,   1 non-dominated

SOLUTIONS AT t ≤ 1.0 ms:
  Total: 45 solutions
  Non-dominated: 25 solutions
```

**25 non-dominated solutions exist at t < 1ms**, contributing to the initial HV spike.

---

## Impact Assessment

### Scientific Impact

1. **Misleading Progress Visualization:**
   - Makes PLS appear to converge instantly
   - Obscures actual search dynamics
   - Difficult to assess algorithm improvements

2. **Unfair Algorithm Comparison:**
   - PLS curves appear "better" early on due to initial bias
   - Hybrid algorithms correctly start at 0.0, appearing worse
   - Cannot accurately compare anytime performance

3. **Invalid Benchmarking:**
   - HV improvement metrics are artificially deflated for PLS
   - Example: Pure PLS shows only 19.5% improvement when 80% came from initialization

### Affected Configurations

All PLS-based algorithms with `use_greedy_initial_population=True`:
- ✅ Pure PLS
- ✅ Improved PLS  
- ✅ Diverse Probe PLS
- ✅ Scalarized PLS
- ✅ Diverse+Scalarized PLS
- ❌ Hybrid methods (correctly start MILP phase at t=0)
- ❌ NSGA-II (if implemented)
- ❌ MOEA/D (if implemented)

---

## Proposed Solutions

### Option A: Start Timer AFTER Initialization (Recommended)

**Approach:** Consider initialization as "pre-search" and start the optimization timer after initial population is created.

**Pros:**
- Clean separation of initialization vs. search
- Initial solutions get timestamp=0
- True search progress visible from t>0
- Matches user expectation: "optimization time"

**Cons:**
- Changes existing behavior
- Initialization time not counted in total time
- Need to document clearly

**Implementation:**
```rust
// In solve_with_pls or similar entry point
let initial_population = create_initial_population(&problem, &config);
let timer = Instant::now();  // Start timer AFTER initialization
let mut pls = ParetoLocalSearch::new(
    &problem,
    &initial_population,
    timer,  // Pass timer to PLS
    ...
);
```

### Option B: Normalize Timestamps Post-Hoc

**Approach:** During trace generation, subtract the timestamp of the first non-initial solution from all timestamps.

**Pros:**
- No changes to core algorithm
- Can be applied retroactively to existing traces
- Initialization time still recorded

**Cons:**
- Complex logic to identify "first real solution"
- May need metadata flag to distinguish initial vs. optimized solutions

### Option C: Separate Initial HV Metric

**Approach:** Report initial HV as a separate metric, plot HV curve starting from first improvement.

**Pros:**
- Preserves all information
- Clear distinction between initialization quality and search effectiveness
- Easy to implement in plotting code

**Cons:**
- Doesn't fix underlying timestamp issue
- Requires updating all plots and analysis scripts

### Option D: Assign Large Initial Offset

**Approach:** Assign initial solutions timestamps like `BASE + 1, BASE + 2, ...` where `BASE = 1_000_000` (1 second).

**Pros:**
- Minimal code change
- Initial solutions clearly separated in time

**Cons:**
- Artificial time offset
- Still counts initialization in total time
- Confusing semantics

---

## Recommendation

**Implement Option A + Option C:**

1. **Option A (Primary Fix):** Start optimization timer after initialization completes
   - Initialization solutions get natural timestamps (1µs, 2µs, ...)
   - These map to t=0 in the optimization phase
   - Report "initialization HV" as separate metric

2. **Option C (Enhanced Reporting):** 
   - Add `initial_hv` field to results
   - Plot annotation showing initial HV as horizontal dashed line
   - Report both "initial HV" and "HV improvement" metrics

### Example Output:
```
Pure PLS (lagos_nigeria_100):
  Initial HV:      0.7548  (from 25 greedy solutions)
  Final HV:        0.9379
  HV Improvement:  +0.1831 (+24.3% relative)
  Optimization Time: 40.0s
  Initialization Time: 0.003s
```

---

## Testing Plan

1. **Verify Fix:**
   - Run `lagos_nigeria_100` with Pure PLS
   - Confirm HV at t=0 equals initial_hv
   - Confirm curve shows improvement from t>0

2. **Regression Testing:**
   - Run full experiment suite (all sizes, all configs)
   - Verify hybrid methods unchanged
   - Check all plots render correctly

3. **Documentation:**
   - Update `run_hv_experiments.py` docstring
   - Add explanation to plot captions
   - Update TRACE_SPECIFICATION.md if needed

---

## Related Files

- `sims-heuristics/src/pareto_local_search.rs` - Core PLS implementation
- `sims-problem/src/solver.rs` - Python binding entry points
- `sims-problem/run_hv_experiments.py` - Experiment runner
- `sims-problem/src/trace.rs` - Trace archive generation
- `sims-problem/debug_initial_hv.py` - Diagnostic script (newly created)

---

## References

- Trace data: `hv_experiment_results/lagos_nigeria_100.json`
- Affected plot: `hv_experiment_results/lagos_nigeria_100.png`
- Diagnostic output: See `debug_initial_hv.py` execution log

---

**Status:** Bug confirmed and documented  
**Priority:** High (affects all published results with PLS algorithms)  
**Estimated Effort:** 4-8 hours (implementation + testing + documentation)  
**Date:** 2025-04-24