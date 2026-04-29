# Add Exact Solver Curves - Batch Processing Script

## Overview

This document describes `add_exact_curves_batch.py`, a Python script that enhances PLS algorithm experiment plots by adding Exact Solver (pseudo-solver) hypervolume curves for comparison.

## Purpose

When analyzing PLS algorithm performance, it's valuable to compare against the exact Pareto front to understand:
- How close PLS gets to the optimal solutions
- How quickly PLS converges compared to exhaustive search
- Which parts of the Pareto front are harder for PLS to discover

This script automates the process of adding Exact Solver curves to existing experiment plots.

## Quick Start

```bash
# From sims-problem directory
cd sims-problem

# Process default results directory
uv run python add_exact_curves_batch.py

# Process specific directory
uv run python add_exact_curves_batch.py --results-dir /path/to/results

# Quiet mode (minimal output)
uv run python add_exact_curves_batch.py --quiet
```

## What the Script Does

### Input
- **Experiment Results**: JSON files containing PLS algorithm results (HV curves, bounds, timeout, etc.)
- **Pseudo Solutions**: Pre-computed exact Pareto fronts from `sims-core/tests/data/pseudo_solver_solutions/`

### Processing
1. For each instance JSON file in the results directory:
   - Loads experiment configuration (objectives, bounds, timeout)
   - Loads PLS algorithm HV curves
   - Loads corresponding pseudo-solver solutions
   - Filters exact solutions to those within PLS experiment bounds
   - Computes normalized hypervolume curve over time
   - Creates new plot with Exact Solver curve added

### Output
- New PNG files with `_with_exact` suffix
- Enhanced plots showing PLS vs Exact Solver performance
- Processing statistics and filtering reports

## Technical Details

### Hypervolume Computation

The script uses **normalized hypervolume** computation to prevent integer overflow:

```python
hv = sims_problem.compute_hypervolume(
    discovered, 
    bounds, 
    normalized=True  # Returns HV in [0,1] range
)
```

This is critical because:
- Raw objective values can be very large (e.g., cost > 30 million)
- Integer multiplication in HV calculation can overflow
- Normalized HV is in [0,1] range and directly comparable

### Solution Filtering

**Important**: The script filters exact solver solutions to only include those within PLS experiment bounds.

**Why filtering is necessary:**

1. **Fair Comparison**: PLS explored a specific region of objective space. Comparing HV values requires using the same bounds/region.

2. **Prevents Overflow**: Exact solver may find solutions far outside PLS bounds (e.g., cost=32M when PLS explored [2M-13M]). Including these requires expanding bounds, which causes integer overflow.

3. **Meaningful Metrics**: HV values computed on different bounds are not comparable.

**Example filtering:**
```
Processing: lagos_nigeria_100
  ✓ Loaded 27 pseudo-solver solutions
    NOTE: Filtered 18/27 exact solver solutions outside PLS bounds
  ℹ 18 exact solver solution(s) outside PLS bounds were excluded
  ✓ Computed HV curve: 30 points, final HV = 0.179638
```

This means:
- Exact solver found 27 total solutions
- 18 were outside PLS bounds (better in some objectives)
- 9 solutions within bounds were used for HV comparison

### Visual Styling

**Exact Solver** (default):
- Color: Green (`#2ca02c`)
- Marker: Diamond (`D`)
- Line: Solid, width 2.5
- Z-order: 10 (draws on top)

**PLS Algorithms** (consistent with existing conventions):
- Pure PLS: Gray circles
- Scalarized PLS: Red X markers
- Diverse Probe PLS: Pink stars
- Improved PLS: Orange diamonds (dashed)

## Command-Line Options

### `--results-dir PATH`
Directory containing experiment JSON files.

**Default**: `hv_experiment_results_pure_vs_scalarized_lagos100`

**Example**:
```bash
uv run python add_exact_curves_batch.py --results-dir my_experiments
```

### `--no-exact-label`
Use "Reference" instead of "Exact Solver" in plot labels.

Useful for publications where generic terminology is preferred.

**Example**:
```bash
uv run python add_exact_curves_batch.py --no-exact-label
```

### `--quiet`
Suppress detailed progress output. Only shows summary statistics.

**Example**:
```bash
uv run python add_exact_curves_batch.py --quiet
```

## Expected Directory Structure

### Input Directory
```
results_dir/
├── instance1.json          # Experiment results with PLS curves
├── instance1.png           # Original plot (not modified)
├── instance2.json
├── instance2.png
├── all_experiments.json    # Combined results (skipped by script)
└── ...
```

### Pseudo Solutions Directory
Located at: `sims-core/tests/data/pseudo_solver_solutions/`

```
pseudo_solver_solutions/
├── instance1.json          # Exact Pareto front for instance1
├── instance2.json
└── ...
```

**Note**: The pseudo solutions directory path is hardcoded relative to the script location.

### Output
```
results_dir/
├── instance1_with_exact.png    # NEW: Plot with exact curve
├── instance2_with_exact.png    # NEW: Plot with exact curve
└── ...
```

Original files are **not modified**.

## Example Session

```bash
$ cd sims-problem
$ uv run python add_exact_curves_batch.py

Found 10 instance(s) in hv_experiment_results_pure_vs_scalarized_lagos100/
Pseudo solutions directory: /home/user/sims-core/tests/data/pseudo_solver_solutions

======================================================================
Processing: lagos_nigeria_30
  ✓ Loaded 120 pseudo-solver solutions
  ✓ Computed HV curve: 30 points, final HV = 0.753709
  ✓ Saved: lagos_nigeria_30_with_exact.png

Processing: lagos_nigeria_100
  ✓ Loaded 27 pseudo-solver solutions
    NOTE: Filtered 18/27 exact solver solutions outside PLS bounds
  ℹ 18 exact solver solution(s) outside PLS bounds were excluded
  ✓ Computed HV curve: 30 points, final HV = 0.179638
  ✓ Saved: lagos_nigeria_100_with_exact.png

Processing: mexico_city_30
  ✓ Loaded 54 pseudo-solver solutions
    NOTE: Filtered 6/54 exact solver solutions outside PLS bounds
  ℹ 6 exact solver solution(s) outside PLS bounds were excluded
  ✓ Computed HV curve: 30 points, final HV = 0.111466
  ✓ Saved: mexico_city_30_with_exact.png

... (more instances)

======================================================================

Processing complete!
  Total instances:   10
  Successful:        10
  Skipped (no data): 0
  Failed:            0

Generated 10 plot(s) with '_with_exact.png' suffix
```

## Interpreting Results

### High Exact Solver HV
If exact solver HV is close to 1.0 (e.g., 0.75-0.90):
- The instance has a rich Pareto front
- Exact solver found many high-quality solutions
- Good benchmark for evaluating PLS performance

### Low Exact Solver HV
If exact solver HV is low (e.g., 0.10-0.30):
- Many exact solutions were filtered (outside PLS bounds)
- PLS explored different region of objective space
- Direct HV comparison may be less meaningful

### Filtered Solutions
When many solutions are filtered:
```
NOTE: Filtered 18/27 exact solver solutions outside PLS bounds
```

This indicates:
- Exact solver found solutions PLS didn't explore
- PLS may have focused on different trade-offs
- Consider expanding PLS exploration parameters

## Troubleshooting

### Problem: No pseudo solutions found
```
⚠ No pseudo solutions found - skipping
```

**Solution**: Ensure pseudo solutions exist for the instance in:
```
sims-core/tests/data/pseudo_solver_solutions/<instance_name>.json
```

### Problem: Empty HV curve
```
⚠ Empty HV curve - skipping
```

**Cause**: All exact solutions were filtered (outside PLS bounds).

**Solution**: This is expected when exact solver found solutions in completely different objective space regions. The instance will be skipped.

### Problem: HV computation error
```
ERROR: Failed to compute HV curve: <error message>
```

**Common causes**:
1. Malformed objective data in pseudo solutions
2. Incompatible bounds format
3. Missing objective fields

**Solution**: Check that pseudo solution JSON format matches expected structure (see example below).

### Problem: Plot creation fails
```
ERROR: Failed to create plot: <error message>
```

**Solution**: Verify matplotlib is installed and results JSON has valid structure.

## Pseudo Solution JSON Format

Expected format for pseudo solver solutions:

```json
{
  "instance_name": "lagos_nigeria_30",
  "test_type": "4d",
  "objectives": ["min_cost", "cloud_coverage", "min_max_incidence_angle", "min_resolution"],
  "num_solutions": 120,
  "solutions": [
    {
      "selected_images": [2, 4, 6, 8, 10, 11, 24, 25, 28],
      "cost": 2736640,
      "cloudy_area": 511693,
      "max_incidence_angle": 333,
      "min_resolutions_sum": 19990,
      "timestamp_s": 0.059224,
      "phase": "exact",
      "index": 0
    },
    ...
  ]
}
```

**Required fields per solution**:
- `cost`: Integer
- `cloudy_area`: Integer
- `max_incidence_angle`: Integer
- `min_resolutions_sum`: Integer
- `timestamp_s`: Float (discovery time in seconds)

## Customization

### Changing Exact Solver Styling

Edit the `EXACT_SOLVER_CONFIG` dictionary in the script:

```python
EXACT_SOLVER_CONFIG = {
    "label": "Exact Solver",
    "color": "#2ca02c",      # Green
    "linestyle": "-",        # Solid line
    "linewidth": 2.5,
    "marker": "D",           # Diamond
    "markersize": 7,
    ...
}
```

### Changing PLS Algorithm Styles

Edit the `PLS_ALGORITHM_STYLES` dictionary:

```python
PLS_ALGORITHM_STYLES = {
    "Pure PLS": {
        "color": "#7f7f7f",   # Gray
        "marker": "o",         # Circle
        ...
    },
    ...
}
```

### Adjusting Pseudo Solutions Path

Modify `PSEUDO_SOLUTIONS_DIR` at the top of the script:

```python
PSEUDO_SOLUTIONS_DIR = Path("/custom/path/to/pseudo_solver_solutions")
```

## Requirements

- **Python 3.8+**
- **matplotlib**: For plotting
- **sims_problem**: PyO3 module with `compute_hypervolume` function
- **Pseudo solutions**: Must exist for instances being processed

Install with:
```bash
# From workspace root
uv sync
```

## Related Scripts

- **`add_exact_solver_curves.py`**: Original single-instance version
- **`run_hv_experiments.py`**: Generates the input JSON files
- **`benchmark.py`**: Runs PLS algorithm benchmarks

## References

- **TRACE_SPECIFICATION.md**: Details on trace file format
- **sims_problem.pyi**: Type stubs for `compute_hypervolume`
- **augmecon-rs/**: Exact solver implementation (AUGMECON/GPBA)

## License

Part of the SIMS Hybrid Algorithms project.