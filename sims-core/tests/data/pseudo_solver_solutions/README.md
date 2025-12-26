# Pseudo-Solver Solution Data

This directory contains pre-recorded solutions from real solver runs (100_0 ratio) that can be used by the pseudo-solver for fast testing.

## Overview

The pseudo-solver allows you to run tests without actually executing the expensive MILP and PLS solvers. Instead, it replays pre-recorded solutions filtered by timestamp based on the timeout parameter.

## Usage

To run tests with the pseudo-solver, add the `--use-pseudo-solver` flag:

```bash
# Run a single test with pseudo-solver
pytest tests/test_two_phases_instances.py::TestTwoPhaseInstances::test_solve_two_phase_4d_on_small_instances --use-pseudo-solver -v

# Run all small instance tests with pseudo-solver
pytest tests/test_two_phases_instances.py -k "small" --use-pseudo-solver -v
```

## Benefits

1. **Speed**: Tests complete in ~0.01s instead of seconds/minutes
2. **Reproducibility**: Same solutions every time
3. **No Dependencies**: Doesn't require MILP solvers to be installed
4. **CI/CD**: Great for fast feedback in continuous integration

## Data Structure

Each JSON file contains:
- `instance_name`: Name of the problem instance
- `test_type`: Type of test (2d, 3d, 4d)
- `objectives`: List of objectives being optimized
- `num_solutions`: Total number of solutions
- `solutions`: Array of solution objects with:
  - `selected_images`: List of selected image indices
  - `cost`, `cloudy_area`, `max_incidence_angle`, `min_resolutions_sum`: Objective values
  - `timestamp_s`: When the solution was found (in seconds)
  - `phase`: Which phase found it ("exact" or "heuristic")
  - `index`: Solution index

## Regenerating Data

To regenerate the solution data from test artifacts:

```bash
cd sims-core/tests
python extract_pseudo_solver_data.py
```

This will search for all 100_0 ratio test results in `test_artifacts/` and aggregate them into this directory.

## Coverage

Currently includes solutions for:
- Small instances (30, 50 images): 24 instances
- Medium instances (100 images): 5 instances  
- Large instances (145-150 images): 5 instances
- Huge instances (200 images): 4 instances

Total: ~1,600 solutions across 24 unique problem instances.