#!/usr/bin/env python
"""Add Exact Solver HV curves to existing experiment plots.

This script reads JSON files from hv_experiment_results directory,
loads pseudo-solver solutions, computes their HV curve, and creates
new plots with the Exact Solver curve included.

Usage:
    uv run python add_exact_solver_curves.py
    uv run python add_exact_solver_curves.py --results-dir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("ERROR: matplotlib not available", file=sys.stderr)
    sys.exit(1)

import sims_problem

# Pseudo solver solutions directory
PSEUDO_SOLUTIONS_DIR = (
    Path(__file__).parent.parent
    / "sims-core"
    / "tests"
    / "data"
    / "pseudo_solver_solutions"
)

# Algorithm configuration for visual consistency
EXACT_SOLVER_COLOR = "#2ca02c"  # Green
EXACT_SOLVER_LINESTYLE = "-"
EXACT_SOLVER_LINEWIDTH = 2.5
EXACT_SOLVER_MARKER = "D"  # Diamond

# Marker styles for other algorithms (matching run_hv_experiments.py)
MARKER_STYLES = {
    "Pure PLS": "o",
    "Scalarized PLS": "X",
    "Diverse Probe PLS": "*",
    "Improved PLS": "D",
}


def load_pseudo_solutions(instance_name: str) -> list[dict]:
    """Load pseudo-solver solutions from JSON."""
    json_path = PSEUDO_SOLUTIONS_DIR / f"{instance_name}.json"
    if not json_path.exists():
        print(
            f"  WARNING: No pseudo solutions found for {instance_name}", file=sys.stderr
        )
        return []

    with open(json_path) as f:
        data = json.load(f)

    solutions = data if isinstance(data, list) else data.get("solutions", [])
    print(f"  Loaded {len(solutions)} pseudo-solver solutions from {json_path.name}")
    return solutions


def pseudo_solutions_to_objectives(
    pseudo_solutions: list[dict], objectives_list: list[str]
) -> list[tuple[int, ...]]:
    """Convert pseudo solutions to objective tuples matching the experiment's objectives."""
    objective_tuples = []

    for sol in pseudo_solutions:
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

    return objective_tuples


def compute_exact_solver_hv_curve(
    pseudo_solutions: list[dict],
    objectives_list: list[str],
    bounds: list[list[int]],
    num_points: int,
    total_timeout: float,
) -> list[tuple[float, float]]:
    """Compute HV curve for exact solver solutions.

    The exact solver generates solutions over time, so we create a curve
    showing HV as solutions are discovered.
    """
    # Convert to objective tuples
    objective_tuples = pseudo_solutions_to_objectives(pseudo_solutions, objectives_list)

    if not objective_tuples:
        return []

    # Expand bounds to include pseudo solutions
    # (they might not have been included in the original experiment bounds)
    expanded_bounds = [list(b) for b in bounds]
    for obj_tuple in objective_tuples:
        for i, val in enumerate(obj_tuple):
            expanded_bounds[i][0] = min(expanded_bounds[i][0], val)
            expanded_bounds[i][1] = max(expanded_bounds[i][1], val)

    print(f"  Expanded bounds to include pseudo solutions: {expanded_bounds}")</text>

    # Sort by timestamp
    solutions_with_time = [
        (sol.get("timestamp_s", 0.0), obj)
        for sol, obj in zip(pseudo_solutions, objective_tuples)
    ]
    solutions_with_time.sort(key=lambda x: x[0])

    # Get max timestamp
    max_time = solutions_with_time[-1][0] if solutions_with_time else 0.0

    # Create time samples
    if num_points <= 1:
        sample_times = [total_timeout]
    else:
        sample_times = [i * total_timeout / (num_points - 1) for i in range(num_points)]

    # Compute HV at each sample time
    curve = []
    for t in sample_times:
        # Get all solutions discovered by time t
        discovered = [obj for ts, obj in solutions_with_time if ts <= t]

        if not discovered:
            curve.append((t, 0.0))
        else:
            # Compute HV
            hv = sims_problem.compute_hypervolume(discovered, bounds)
            curve.append((t, hv))

    return curve


def create_plot_with_exact_solver(
    results_data: dict,
    exact_curve: list[tuple[float, float]],
    output_path: Path,
) -> None:
    """Create plot with Exact Solver curve added."""
    instance = results_data["instance"]
    num_images = results_data["num_images"]
    timeout = results_data["timeout_s"]
    configs_data = results_data["configs"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot PLS algorithms
    endpoint_annotations = []
    for config_name, config_data in configs_data.items():
        curve = config_data.get("curve", [])
        if not curve:
            continue

        ts = [t for t, hv in curve]
        hvs = [hv for t, hv in curve]

        # Get marker style
        marker = MARKER_STYLES.get(config_name, "o")
        marker_every = max(1, len(ts) // 8)

        # Determine color and style (use some defaults)
        color_map = {
            "Pure PLS": "#7f7f7f",
            "Scalarized PLS": "#d62728",
            "Diverse Probe PLS": "#e377c2",
            "Improved PLS": "#ff7f0e",
        }
        linestyle_map = {
            "Pure PLS": "-",
            "Scalarized PLS": "-",
            "Diverse Probe PLS": "-",
            "Improved PLS": "--",
        }

        color = color_map.get(config_name, "#1f77b4")
        linestyle = linestyle_map.get(config_name, "-")
        linewidth = 2.0

        ax.plot(
            ts,
            hvs,
            label=config_name,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=6,
            markevery=marker_every,
            markeredgewidth=1.5,
            markerfacecolor=color,
            markeredgecolor="white",
            alpha=0.9,
        )
        endpoint_annotations.append((ts[-1], hvs[-1], config_name, color))

    # Plot Exact Solver curve
    if exact_curve:
        ts = [t for t, hv in exact_curve]
        hvs = [hv for t, hv in exact_curve]
        marker_every = max(1, len(ts) // 8)

        ax.plot(
            ts,
            hvs,
            label="Exact Solver",
            color=EXACT_SOLVER_COLOR,
            linestyle=EXACT_SOLVER_LINESTYLE,
            linewidth=EXACT_SOLVER_LINEWIDTH,
            marker=EXACT_SOLVER_MARKER,
            markersize=7,
            markevery=marker_every,
            markeredgewidth=1.5,
            markerfacecolor=EXACT_SOLVER_COLOR,
            markeredgecolor="white",
            alpha=0.9,
        )
        endpoint_annotations.append(
            (ts[-1], hvs[-1], "Exact Solver", EXACT_SOLVER_COLOR)
        )

    # Add endpoint annotations
    sorted_annotations = sorted(endpoint_annotations, key=lambda x: x[1], reverse=True)
    n_annotations = len(sorted_annotations)
    for idx, (end_t, end_hv, label, color) in enumerate(sorted_annotations):
        ax.annotate(
            f"{end_hv:.4f}",
            xy=(end_t, end_hv),
            xytext=(6, 10 * (n_annotations - 1 - idx)),
            textcoords="offset points",
            color=color,
            fontsize=9,
            va="center",
            ha="left",
        )

    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Normalized Hypervolume", fontsize=12)
    ax.set_title(
        f"HV over Time — {instance} ({num_images} images, 4D)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, timeout)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def process_results_directory(results_dir: Path) -> None:
    """Process all JSON files in results directory."""
    json_files = list(results_dir.glob("*.json"))

    # Filter out combined results
    json_files = [f for f in json_files if f.name != "all_experiments.json"]

    if not json_files:
        print(f"No JSON files found in {results_dir}", file=sys.stderr)
        return

    print(f"Processing {len(json_files)} instances from {results_dir}\n")

    for json_file in sorted(json_files):
        instance_name = json_file.stem
        print(f"Processing {instance_name}...")

        # Load results
        with open(json_file) as f:
            results_data = json.load(f)

        # Load pseudo solutions
        pseudo_solutions = load_pseudo_solutions(instance_name)
        if not pseudo_solutions:
            print(f"  Skipping {instance_name} (no pseudo solutions)")
            continue

        # Compute Exact Solver HV curve
        objectives_list = results_data["objectives"]
        bounds = results_data["shared_bounds"]

        # Get number of points from first config's curve
        num_points = 30
        for config_data in results_data["configs"].values():
            if config_data.get("curve"):
                num_points = len(config_data["curve"])
                break

        timeout = results_data["timeout_s"]

        exact_curve = compute_exact_solver_hv_curve(
            pseudo_solutions,
            objectives_list,
            bounds,
            num_points,
            timeout,
        )

        if exact_curve:
            print(
                f"  Computed Exact Solver curve: {len(exact_curve)} points, "
                f"final HV={exact_curve[-1][1]:.6f}"
            )

        # Create new plot
        output_path = results_dir / f"{instance_name}_with_exact.png"
        create_plot_with_exact_solver(results_data, exact_curve, output_path)
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add Exact Solver HV curves to existing experiment plots"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(
            "/home/hlvlad/code/serenity/sims-hybrid-algs/sims-problem/hv_experiment_results_pure_vs_scalarized_lagos100"
        ),
        help="Directory containing experiment results",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(
            f"ERROR: Results directory not found: {args.results_dir}", file=sys.stderr
        )
        return 1

    process_results_directory(args.results_dir)

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
