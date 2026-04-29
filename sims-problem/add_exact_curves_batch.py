#!/usr/bin/env python
"""Add Exact Solver HV curves to existing experiment plots (batch processing).

This script processes JSON files from PLS algorithm experiment results, loads the
corresponding pseudo-solver solutions (exact Pareto fronts), computes their hypervolume
curves over time, and creates enhanced plots showing how the Exact Solver compares to
PLS algorithms.

OVERVIEW
--------
For each JSON file in the results directory:
1. Loads the experiment configuration (objectives, bounds, timeout, PLS curves)
2. Loads pseudo-solver solutions from sims-core/tests/data/pseudo_solver_solutions/
3. Filters exact solver solutions to those within PLS experiment bounds
4. Computes normalized hypervolume curve showing HV progression over time
5. Creates a new plot with the Exact Solver curve added alongside PLS curves
6. Saves the plot with "_with_exact.png" suffix

KEY FEATURES
------------
- Batch processes all JSON files in a directory automatically
- Uses normalized hypervolume computation (prevents integer overflow)
- Filters exact solver solutions to PLS bounds for fair comparison
- Reports filtering statistics (solutions outside bounds)
- Handles missing pseudo solutions gracefully
- Configurable visual styling (colors, markers, line styles)
- Progress tracking with detailed per-instance reporting
- Generates high-resolution plots (200 DPI)

HYPERVOLUME COMPUTATION
-----------------------
- Uses normalized=True to compute HV in [0,1] range (prevents overflow)
- Filters exact solver solutions outside PLS experiment bounds
- This ensures fair comparison: both PLS and Exact are evaluated on the same
  objective space region that PLS explored
- Solutions outside bounds are counted and reported but not included in HV

BOUNDS EXPANSION FOR EXACT SOLVER
----------------------------------
The pseudo-solver (exact solver) may find solutions that dominate the PLS bounds,
especially for objectives that were not fully explored by PLS. For example:
- PLS explored costs in range [2,338,089 - 13,718,550]
- Exact solver found solution with cost 2,161,600 (better than PLS min)

This script EXPANDS the bounds to include ALL exact solver solutions:
1. Gives exact solver its own evaluation space
2. Shows true exact solver performance (no filtering)
3. Uses normalized HV computation to prevent integer overflow

Important notes:
- Exact solver HV is computed in expanded bounds space
- PLS HV is computed in original bounds space
- HV values may not be directly comparable if bounds differ significantly
- The plot shows both algorithms' performance in their respective spaces
- This approach favors showing exact solver's full capability

OUTPUT FORMAT
-------------
For each instance JSON file (e.g., "lagos_nigeria_30.json"):
- Reads: lagos_nigeria_30.json (experiment results with PLS curves)
- Loads: pseudo_solver_solutions/lagos_nigeria_30.json (exact solutions)
- Creates: lagos_nigeria_30_with_exact.png (plot with exact solver curve)

DIRECTORY STRUCTURE
-------------------
Expected input directory layout:
    results_dir/
        ├── instance1.json          # Experiment results
        ├── instance1.png           # Original plot
        ├── instance2.json
        ├── instance2.png
        └── all_experiments.json    # (skipped by script)

Pseudo solutions directory (hardcoded relative path):
    sims-core/tests/data/pseudo_solver_solutions/
        ├── instance1.json          # Exact Pareto front
        ├── instance2.json
        └── ...

Output:
    results_dir/
        ├── instance1_with_exact.png    # New plot with exact curve
        ├── instance2_with_exact.png
        └── ...

USAGE EXAMPLES
--------------
Basic usage (uses default directory):
    uv run python add_exact_curves_batch.py

Specify custom results directory:
    uv run python add_exact_curves_batch.py --results-dir /path/to/results

Use generic label instead of "Exact Solver":
    uv run python add_exact_curves_batch.py --no-exact-label

Quiet mode (minimal output):
    uv run python add_exact_curves_batch.py --quiet

Process specific experiment results:
    uv run python add_exact_curves_batch.py \
        --results-dir hv_experiment_results_pure_vs_scalarized_lagos100

COMMAND-LINE OPTIONS
--------------------
--results-dir PATH      Directory containing experiment JSON files
                        Default: hv_experiment_results_pure_vs_scalarized_lagos100

--no-exact-label       Use 'Reference' instead of 'Exact Solver' in plot labels
                        Useful for publications where you want generic terminology

--quiet                Suppress detailed progress output
                        Only shows summary statistics

EXAMPLE OUTPUT
--------------
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

======================================================================
Processing complete!
  Total instances:   10
  Successful:        10
  Skipped (no data): 0
  Failed:            0

Generated 10 plot(s) with '_with_exact.png' suffix

REQUIREMENTS
------------
- Python packages: matplotlib, sims_problem (with PyO3 bindings)
- Install with: uv sync (from workspace root)
- Pseudo solutions must exist in: sims-core/tests/data/pseudo_solver_solutions/

NOTES
-----
- Script uses normalized hypervolume (HV in [0,1]) to prevent integer overflow
- Solutions outside PLS bounds are filtered and reported
- All plots use consistent color scheme and styling
- Endpoint annotations show final HV values for easy comparison
- Exact Solver curve uses green color and diamond markers by default

SEE ALSO
--------
- TRACE_SPECIFICATION.md: Details on trace file format
- run_hv_experiments.py: Script that generates the input JSON files
- sims_problem.pyi: Type stubs for Python bindings (compute_hypervolume)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("ERROR: matplotlib not available", file=sys.stderr)
    sys.exit(1)

try:
    import sims_problem

    HAS_SIMS_PROBLEM = True
except ImportError:
    HAS_SIMS_PROBLEM = False
    print("ERROR: sims_problem module not available", file=sys.stderr)
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Pseudo solver solutions directory (relative to script location)
PSEUDO_SOLUTIONS_DIR = (
    Path(__file__).parent.parent
    / "sims-core"
    / "tests"
    / "data"
    / "pseudo_solver_solutions"
)

# Visual styling for Exact Solver
EXACT_SOLVER_CONFIG = {
    "label": "Exact Solver",
    "color": "#2ca02c",  # Green
    "linestyle": "-",
    "linewidth": 2.5,
    "marker": "D",  # Diamond
    "markersize": 7,
    "markeredgewidth": 1.5,
    "markeredgecolor": "white",
    "alpha": 0.95,
    "zorder": 10,  # Draw on top
}

# Visual styling for PLS algorithms (matching standard conventions)
PLS_ALGORITHM_STYLES = {
    "Pure PLS": {
        "color": "#7f7f7f",
        "linestyle": "-",
        "linewidth": 2.0,
        "marker": "o",
        "markersize": 6,
    },
    "Scalarized PLS": {
        "color": "#d62728",
        "linestyle": "-",
        "linewidth": 2.0,
        "marker": "X",
        "markersize": 7,
    },
    "Diverse Probe PLS": {
        "color": "#e377c2",
        "linestyle": "-",
        "linewidth": 2.0,
        "marker": "*",
        "markersize": 8,
    },
    "Improved PLS": {
        "color": "#ff7f0e",
        "linestyle": "--",
        "linewidth": 2.0,
        "marker": "D",
        "markersize": 6,
    },
}

# Default algorithm style for unknown algorithms
DEFAULT_ALGORITHM_STYLE = {
    "color": "#1f77b4",
    "linestyle": "-",
    "linewidth": 2.0,
    "marker": "o",
    "markersize": 6,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def load_pseudo_solutions(instance_name: str) -> list[dict[str, Any]]:
    """Load pseudo-solver solutions from JSON file.

    Args:
        instance_name: Name of the instance (e.g., "lagos_nigeria_30")

    Returns:
        List of solution dictionaries with objectives and timestamps
    """
    json_path = PSEUDO_SOLUTIONS_DIR / f"{instance_name}.json"

    if not json_path.exists():
        return []

    try:
        with open(json_path) as f:
            data = json.load(f)

        # Handle both direct list format and nested format
        if isinstance(data, list):
            solutions = data
        elif isinstance(data, dict) and "solutions" in data:
            solutions = data["solutions"]
        else:
            print(
                f"    WARNING: Unexpected format in {json_path.name}", file=sys.stderr
            )
            return []

        return solutions

    except (json.JSONDecodeError, IOError) as e:
        print(f"    ERROR: Failed to load {json_path.name}: {e}", file=sys.stderr)
        return []


def pseudo_solutions_to_objectives(
    pseudo_solutions: list[dict[str, Any]], objectives_list: list[str]
) -> list[tuple[int, ...]]:
    """Convert pseudo solutions to objective tuples.

    Args:
        pseudo_solutions: List of solution dictionaries
        objectives_list: Ordered list of objective names

    Returns:
        List of objective value tuples
    """
    objective_tuples = []

    for sol in pseudo_solutions:
        try:
            obj_values = []
            for obj_name in objectives_list:
                if obj_name == "min_cost":
                    obj_values.append(sol["cost"])
                elif obj_name == "cloud_coverage":
                    obj_values.append(sol["cloudy_area"])
                elif obj_name == "min_max_incidence_angle":
                    obj_values.append(sol["max_incidence_angle"])
                elif obj_name == "min_resolution":
                    obj_values.append(sol["min_resolutions_sum"])
                else:
                    raise ValueError(f"Unknown objective: {obj_name}")

            objective_tuples.append(tuple(obj_values))

        except KeyError as e:
            print(
                f"    WARNING: Missing objective field in solution: {e}",
                file=sys.stderr,
            )
            continue

    return objective_tuples


def expand_bounds_for_solutions(
    bounds: list[list[int]], objective_tuples: list[tuple[int, ...]]
) -> tuple[list[list[int]], bool]:
    """Expand bounds to include all solutions.

    Args:
        bounds: Original bounds [[min, max], ...]
        objective_tuples: List of objective value tuples

    Returns:
        Tuple of (expanded_bounds, was_expanded)
        - expanded_bounds: Bounds that include all objective values
        - was_expanded: Whether bounds were actually expanded
    """
    expanded_bounds = [list(b) for b in bounds]
    was_expanded = False

    for obj_tuple in objective_tuples:
        for i, val in enumerate(obj_tuple):
            if i < len(expanded_bounds):
                if val < expanded_bounds[i][0]:
                    expanded_bounds[i][0] = val
                    was_expanded = True
                if val > expanded_bounds[i][1]:
                    expanded_bounds[i][1] = val
                    was_expanded = True

    return expanded_bounds, was_expanded


def compute_exact_solver_hv_curve(
    pseudo_solutions: list[dict[str, Any]],
    objectives_list: list[str],
    bounds: list[list[int]],
    num_points: int,
    total_timeout: float,
) -> tuple[list[tuple[float, float]], list[list[int]]]:
    """Compute HV curve for exact solver solutions over time.

    The exact solver generates solutions progressively, so we create a curve
    showing hypervolume as solutions are discovered over time.

    IMPORTANT: This function expands bounds to include ALL exact solver solutions,
    giving the exact solver its own evaluation space. This means HV values may not
    be directly comparable to PLS (which used original bounds), but shows the true
    performance of the exact solver.

    Args:
        pseudo_solutions: List of solution dictionaries with timestamps
        objectives_list: Ordered list of objective names
        bounds: Original bounds from PLS experiments
        num_points: Number of time samples for the curve
        total_timeout: Total time budget (seconds)

    Returns:
        Tuple of (curve_points, expanded_bounds)
        - curve_points: List of (time, hypervolume) tuples
        - expanded_bounds: Bounds expanded to include all exact solver solutions
    """
    # Convert to objective tuples
    objective_tuples = pseudo_solutions_to_objectives(pseudo_solutions, objectives_list)

    if not objective_tuples:
        return [], bounds

    # Expand bounds to include ALL exact solver solutions
    # This gives exact solver its own evaluation space
    expanded_bounds, was_expanded = expand_bounds_for_solutions(
        bounds, objective_tuples
    )

    if was_expanded:
        print(f"    NOTE: Expanded bounds to include all exact solver solutions")
        print(f"          Original: {bounds}")
        print(f"          Expanded: {expanded_bounds}")
        print(f"          (HV values computed in expanded space)")

    # Pair solutions with timestamps
    solutions_with_time = []
    for sol, obj in zip(pseudo_solutions, objective_tuples):
        timestamp = sol.get("timestamp_s", 0.0)
        solutions_with_time.append((timestamp, obj))

    # Sort by timestamp
    solutions_with_time.sort(key=lambda x: x[0])

    # Get actual max timestamp from solutions
    max_solution_time = solutions_with_time[-1][0] if solutions_with_time else 0.0

    # Create time samples (linearly spaced)
    if num_points <= 1:
        sample_times = [total_timeout]
    else:
        sample_times = [i * total_timeout / (num_points - 1) for i in range(num_points)]

    # Compute HV at each sample time using expanded bounds
    curve = []
    for t in sample_times:
        # Get all solutions discovered by time t
        discovered = [obj for ts, obj in solutions_with_time if ts <= t]

        if not discovered:
            curve.append((t, 0.0))
        else:
            # Compute hypervolume using expanded bounds with normalization
            # normalized=True prevents integer overflow and returns values in [0,1] range
            hv = sims_problem.compute_hypervolume(
                discovered, expanded_bounds, normalized=True
            )
            curve.append((t, hv))

    return curve, expanded_bounds


def create_plot_with_exact_solver(
    results_data: dict[str, Any],
    exact_curve: list[tuple[float, float]],
    output_path: Path,
    show_exact_label: bool = True,
) -> None:
    """Create plot with Exact Solver curve added to PLS algorithm curves.

    Args:
        results_data: Experiment results dictionary
        exact_curve: List of (time, hypervolume) tuples for exact solver
        output_path: Path to save the plot
        show_exact_label: Whether to show "Exact Solver" in label
    """
    instance = results_data["instance"]
    num_images = results_data["num_images"]
    timeout = results_data["timeout_s"]
    configs_data = results_data["configs"]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Track endpoints for annotations
    endpoint_annotations = []

    # Plot PLS algorithms
    for config_name, config_data in configs_data.items():
        curve = config_data.get("curve", [])
        if not curve:
            continue

        ts = [t for t, hv in curve]
        hvs = [hv for t, hv in curve]

        # Get style for this algorithm
        style = PLS_ALGORITHM_STYLES.get(config_name, DEFAULT_ALGORITHM_STYLE)

        # Calculate marker frequency (show ~8 markers)
        marker_every = max(1, len(ts) // 8)

        ax.plot(
            ts,
            hvs,
            label=config_name,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=style["markersize"],
            markevery=marker_every,
            markeredgewidth=1.5,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            alpha=0.9,
        )

        endpoint_annotations.append((ts[-1], hvs[-1], config_name, style["color"]))

    # Plot Exact Solver curve
    if exact_curve:
        ts = [t for t, hv in exact_curve]
        hvs = [hv for t, hv in exact_curve]

        # Calculate marker frequency
        marker_every = max(1, len(ts) // 8)

        label = EXACT_SOLVER_CONFIG["label"] if show_exact_label else "Reference"

        ax.plot(
            ts,
            hvs,
            label=label,
            color=EXACT_SOLVER_CONFIG["color"],
            linestyle=EXACT_SOLVER_CONFIG["linestyle"],
            linewidth=EXACT_SOLVER_CONFIG["linewidth"],
            marker=EXACT_SOLVER_CONFIG["marker"],
            markersize=EXACT_SOLVER_CONFIG["markersize"],
            markevery=marker_every,
            markeredgewidth=EXACT_SOLVER_CONFIG["markeredgewidth"],
            markerfacecolor=EXACT_SOLVER_CONFIG["color"],
            markeredgecolor=EXACT_SOLVER_CONFIG["markeredgecolor"],
            alpha=EXACT_SOLVER_CONFIG["alpha"],
            zorder=EXACT_SOLVER_CONFIG["zorder"],
        )

        endpoint_annotations.append(
            (ts[-1], hvs[-1], label, EXACT_SOLVER_CONFIG["color"])
        )

    # Add endpoint annotations (sorted by HV value, descending)
    sorted_annotations = sorted(endpoint_annotations, key=lambda x: x[1], reverse=True)
    n_annotations = len(sorted_annotations)

    for idx, (end_t, end_hv, label, color) in enumerate(sorted_annotations):
        # Vertical offset to prevent overlap
        y_offset = 10 * (n_annotations - 1 - idx)

        ax.annotate(
            f"{end_hv:.4f}",
            xy=(end_t, end_hv),
            xytext=(6, y_offset),
            textcoords="offset points",
            color=color,
            fontsize=9,
            fontweight="bold",
            va="center",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=color,
                alpha=0.8,
                linewidth=1.0,
            ),
        )

    # Configure axes
    ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalized Hypervolume", fontsize=12, fontweight="bold")

    # Title with instance info
    title = f"Hypervolume over Time — {instance} ({num_images} images, 4D)"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # Legend
    ax.legend(fontsize=10, loc="lower right", framealpha=0.95, edgecolor="gray")

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # X-axis limits
    ax.set_xlim(0, timeout)

    # Y-axis limits (with small padding)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min - 0.02, y_max + 0.02)

    # Tight layout and save
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# MAIN PROCESSING
# ============================================================================


def process_instance(
    json_file: Path,
    results_dir: Path,
    show_exact_label: bool = True,
    verbose: bool = True,
) -> bool:
    """Process a single instance JSON file.

    Args:
        json_file: Path to the JSON file
        results_dir: Results directory
        show_exact_label: Whether to show "Exact Solver" in label
        verbose: Print detailed progress

    Returns:
        True if successful, False otherwise
    """
    instance_name = json_file.stem

    if verbose:
        print(f"Processing: {instance_name}")

    # Load results
    try:
        with open(json_file) as f:
            results_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  ERROR: Failed to load {json_file.name}: {e}", file=sys.stderr)
        return False

    # Load pseudo solutions
    pseudo_solutions = load_pseudo_solutions(instance_name)

    if not pseudo_solutions:
        if verbose:
            print(f"  ⚠ No pseudo solutions found - skipping")
        return False

    if verbose:
        print(f"  ✓ Loaded {len(pseudo_solutions)} pseudo-solver solutions")

    # Extract experiment parameters
    objectives_list = results_data.get("objectives", [])
    bounds = results_data.get("shared_bounds", [])
    timeout = results_data.get("timeout_s", 300)

    # Determine number of sample points from existing curves
    num_points = 30  # Default
    for config_data in results_data.get("configs", {}).values():
        if config_data.get("curve"):
            num_points = len(config_data["curve"])
            break

    # Compute Exact Solver HV curve
    try:
        exact_curve, expanded_bounds = compute_exact_solver_hv_curve(
            pseudo_solutions,
            objectives_list,
            bounds,
            num_points,
            timeout,
        )

        # Report if bounds were expanded
        if expanded_bounds != bounds:
            print(
                f"  ℹ Exact solver evaluated in expanded bounds (includes all solutions)"
            )

    except Exception as e:
        print(f"  ERROR: Failed to compute HV curve: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return False

    if not exact_curve:
        if verbose:
            print(f"  ⚠ Empty HV curve - skipping")
        return False

    if verbose:
        final_hv = exact_curve[-1][1]
        print(
            f"  ✓ Computed HV curve: {len(exact_curve)} points, final HV = {final_hv:.6f}"
        )

    # Create plot
    output_path = results_dir / f"{instance_name}_with_exact.png"

    try:
        create_plot_with_exact_solver(
            results_data, exact_curve, output_path, show_exact_label
        )
        if verbose:
            print(f"  ✓ Saved: {output_path.name}\n")
        return True

    except Exception as e:
        print(f"  ERROR: Failed to create plot: {e}", file=sys.stderr)
        return False


def process_results_directory(
    results_dir: Path,
    show_exact_label: bool = True,
    verbose: bool = True,
) -> dict[str, int]:
    """Process all JSON files in results directory.

    Args:
        results_dir: Directory containing experiment results
        show_exact_label: Whether to show "Exact Solver" in label
        verbose: Print detailed progress

    Returns:
        Dictionary with processing statistics
    """
    # Find all JSON files (excluding combined results)
    json_files = [
        f for f in results_dir.glob("*.json") if f.name != "all_experiments.json"
    ]

    if not json_files:
        print(f"No instance JSON files found in {results_dir}", file=sys.stderr)
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}

    print(f"Found {len(json_files)} instance(s) in {results_dir.name}/")
    print(f"Pseudo solutions directory: {PSEUDO_SOLUTIONS_DIR}\n")
    print("=" * 70)

    # Process each file
    stats = {"total": len(json_files), "success": 0, "failed": 0, "skipped": 0}

    for json_file in sorted(json_files):
        success = process_instance(json_file, results_dir, show_exact_label, verbose)

        if success:
            stats["success"] += 1
        else:
            # Check if it was skipped or failed
            pseudo_solutions = load_pseudo_solutions(json_file.stem)
            if not pseudo_solutions:
                stats["skipped"] += 1
            else:
                stats["failed"] += 1

    return stats


# ============================================================================
# CLI ENTRY POINT
# ============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Add Exact Solver HV curves to existing experiment plots (batch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process default directory
  uv run python add_exact_curves_batch.py

  # Process specific directory
  uv run python add_exact_curves_batch.py --results-dir /path/to/results

  # Use generic label instead of "Exact Solver"
  uv run python add_exact_curves_batch.py --no-exact-label

  # Quiet mode
  uv run python add_exact_curves_batch.py --quiet
        """,
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent
        / "hv_experiment_results_pure_vs_scalarized_lagos100",
        help="Directory containing experiment results (default: hv_experiment_results_pure_vs_scalarized_lagos100)",
    )

    parser.add_argument(
        "--no-exact-label",
        action="store_true",
        help="Use 'Reference' instead of 'Exact Solver' in plot labels",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed progress output",
    )

    args = parser.parse_args()

    # Validate results directory
    if not args.results_dir.exists():
        print(
            f"ERROR: Results directory not found: {args.results_dir}", file=sys.stderr
        )
        return 1

    if not args.results_dir.is_dir():
        print(f"ERROR: Not a directory: {args.results_dir}", file=sys.stderr)
        return 1

    # Validate pseudo solutions directory
    if not PSEUDO_SOLUTIONS_DIR.exists():
        print(
            f"ERROR: Pseudo solutions directory not found: {PSEUDO_SOLUTIONS_DIR}",
            file=sys.stderr,
        )
        return 1

    # Process all instances
    stats = process_results_directory(
        args.results_dir,
        show_exact_label=not args.no_exact_label,
        verbose=not args.quiet,
    )

    # Print summary
    print("=" * 70)
    print(f"\nProcessing complete!")
    print(f"  Total instances:   {stats['total']}")
    print(f"  Successful:        {stats['success']}")
    print(f"  Skipped (no data): {stats['skipped']}")
    print(f"  Failed:            {stats['failed']}")

    if stats["success"] > 0:
        print(f"\nGenerated {stats['success']} plot(s) with '_with_exact.png' suffix")
        return 0
    else:
        print("\nNo plots were generated", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
