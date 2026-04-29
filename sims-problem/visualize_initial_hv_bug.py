#!/usr/bin/env python
"""Visualize the initial HV bug in PLS algorithms.

This script creates comparison plots showing how PLS-based algorithms
incorrectly start with high HV values at t=0 due to greedy initialization
being timestamped at microseconds.

Usage:
    uv run python visualize_initial_hv_bug.py
    uv run python visualize_initial_hv_bug.py --instance paris_100
    uv run python visualize_initial_hv_bug.py --instance lagos_nigeria_100 --measure-init-time
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import tarfile
from datetime import timedelta
from pathlib import Path

try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("ERROR: matplotlib not available", file=sys.stderr)
    sys.exit(1)

import sims_problem

RESULTS_DIR = Path(__file__).parent.parent / "hv_experiment_results"
INSTANCES_DIR = Path(__file__).parent / "tests" / "data"


def load_results(instance_name: str) -> dict:
    """Load HV experiment results for an instance."""
    results_file = RESULTS_DIR / f"{instance_name}.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results not found: {results_file}")

    with open(results_file) as f:
        return json.load(f)


def measure_init_time(instance_name: str) -> dict:
    """Measure actual initialization time by running PLS with trace."""
    instance_file = f"{instance_name}.dzn"
    instance_path = INSTANCES_DIR / instance_file

    if not instance_path.exists():
        print(f"WARNING: Instance not found: {instance_path}", file=sys.stderr)
        return {"init_time_s": None, "num_initial": None}

    print(f"Measuring initialization time for {instance_name}...", flush=True)
    problem = sims_problem.SimsDiscreteProblem.from_dzn(str(instance_path))

    # Short run just to measure init time
    result = sims_problem.solve_with_pls(
        problem,
        objectives=[
            "min_cost",
            "cloud_coverage",
            "min_max_incidence_angle",
            "min_resolution",
        ],
        timeout=timedelta(seconds=5),
        is_deterministic=True,
        trace=True,
        use_checkpoint=True,
        use_ranked_candidates=True,
        max_k1_candidates=15,
        use_greedy_initial_population=True,
        use_perturbation_restart=True,
        use_diverse_probing=True,
    )

    # Parse trace to find init time
    with tarfile.open(fileobj=io.BytesIO(result.trace), mode="r:gz") as tar:
        ts_raw = tar.extractfile("timestamp.bin").read()
        meta = json.loads(tar.extractfile("metadata.json").read())

    n = meta["solution_count"]
    timestamps_us = [struct.unpack_from("<I", ts_raw, i * 4)[0] for i in range(n)]

    if not timestamps_us:
        return {"init_time_s": None, "num_initial": None}

    # Find largest gap (transition from init to optimization)
    sorted_ts = sorted(timestamps_us)
    gaps = [
        (sorted_ts[i + 1] - sorted_ts[i], i, sorted_ts[i], sorted_ts[i + 1])
        for i in range(len(sorted_ts) - 1)
    ]
    gaps.sort(reverse=True)

    if gaps:
        largest_gap, split_idx, ts_before, ts_after = gaps[0]
        init_time_s = ts_after / 1_000_000.0
        num_initial = split_idx + 1
        print(
            f"  Initialization: {num_initial} solutions in {init_time_s:.3f}s",
            flush=True,
        )
        return {"init_time_s": init_time_s, "num_initial": num_initial}

    return {"init_time_s": None, "num_initial": None}


def plot_initial_hv_comparison(
    data: dict, output_path: Path, init_time_info: dict = None
) -> None:
    """Create comparison plot highlighting the initial HV bug."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    instance = data["instance"]
    timeout = data["timeout_s"]
    init_time_s = init_time_info.get("init_time_s") if init_time_info else None
    num_initial = init_time_info.get("num_initial") if init_time_info else None

    # Define algorithm groups
    pls_configs = ["Pure PLS", "Improved PLS", "Diverse Probe PLS"]
    hybrid_configs = ["Hybrid 50:50", "Improved Hybrid"]

    # Colors
    pls_color = "#d62728"  # Red
    hybrid_color = "#2ca02c"  # Green

    # ========================================================================
    # Top plot: Full time range showing the issue
    # ========================================================================

    for cfg_name in pls_configs:
        if cfg_name not in data["configs"]:
            continue
        curve = data["configs"][cfg_name]["curve"]
        ts = [t for t, hv in curve]
        hvs = [hv for t, hv in curve]

        linestyle = "-" if "Pure" in cfg_name else "--"
        linewidth = 2.0 if "Pure" in cfg_name else 1.5

        ax1.plot(
            ts,
            hvs,
            label=cfg_name,
            color=pls_color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.8,
        )

    for cfg_name in hybrid_configs:
        if cfg_name not in data["configs"]:
            continue
        curve = data["configs"][cfg_name]["curve"]
        ts = [t for t, hv in curve]
        hvs = [hv for t, hv in curve]

        linestyle = "-" if "50:50" in cfg_name else "--"
        linewidth = 2.0 if "50:50" in cfg_name else 1.5

        ax1.plot(
            ts,
            hvs,
            label=cfg_name,
            color=hybrid_color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.8,
        )

    # Highlight the problem area
    ax1.axvspan(0, 1, alpha=0.2, color="red", zorder=0)

    # Show actual initialization time if available
    if init_time_s is not None and init_time_s > 0:
        ax1.axvline(
            init_time_s,
            color="purple",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            zorder=5,
        )
        ax1.text(
            init_time_s,
            0.5,
            f"  Real init time:\n  {init_time_s:.2f}s\n  ({num_initial} solutions)",
            fontsize=9,
            color="purple",
            fontweight="bold",
            va="center",
            ha="left",
            bbox=dict(
                boxstyle="round",
                facecolor="lavender",
                alpha=0.9,
                edgecolor="purple",
            ),
        )

    bug_text = "BUG: PLS starts at 75-86% of final HV!"
    if init_time_s is not None:
        bug_text += (
            f"\n(Real init: {init_time_s:.2f}s, but timestamped at 0.000001-0.000050s)"
        )

    ax1.text(
        0.5,
        0.95,
        bug_text,
        transform=ax1.transAxes,
        fontsize=10,
        color="red",
        fontweight="bold",
        ha="center",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax1.set_xlabel("Time (seconds)", fontsize=11)
    ax1.set_ylabel("Normalized Hypervolume", fontsize=11)
    ax1.set_title(
        f"Full Time Range - {instance} (shows initial HV bug)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, timeout)
    ax1.set_ylim(0, 1.0)

    # ========================================================================
    # Bottom plot: Zoomed to first 2 seconds
    # ========================================================================

    zoom_end = min(2.0, timeout)

    for cfg_name in pls_configs:
        if cfg_name not in data["configs"]:
            continue
        curve = data["configs"][cfg_name]["curve"]
        ts = [t for t, hv in curve if t <= zoom_end]
        hvs = [hv for t, hv in curve if t <= zoom_end]

        if not ts:
            continue

        linestyle = "-" if "Pure" in cfg_name else "--"
        linewidth = 2.5 if "Pure" in cfg_name else 2.0

        ax2.plot(
            ts,
            hvs,
            label=cfg_name,
            color=pls_color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.8,
            marker="o",
            markersize=4,
            markevery=max(1, len(ts) // 10),
        )

        # Annotate initial value
        if ts:
            ax2.annotate(
                f"{hvs[0]:.3f}",
                xy=(ts[0], hvs[0]),
                xytext=(10, -15),
                textcoords="offset points",
                fontsize=9,
                color=pls_color,
                fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                arrowprops=dict(arrowstyle="->", color=pls_color),
            )

    for cfg_name in hybrid_configs:
        if cfg_name not in data["configs"]:
            continue
        curve = data["configs"][cfg_name]["curve"]
        ts = [t for t, hv in curve if t <= zoom_end]
        hvs = [hv for t, hv in curve if t <= zoom_end]

        if not ts:
            continue

        linestyle = "-" if "50:50" in cfg_name else "--"
        linewidth = 2.5 if "50:50" in cfg_name else 2.0

        ax2.plot(
            ts,
            hvs,
            label=cfg_name,
            color=hybrid_color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.8,
            marker="s",
            markersize=4,
            markevery=max(1, len(ts) // 10),
        )

    # Add reference line at 0
    ax2.axhline(0, color="black", linestyle=":", linewidth=1, alpha=0.5)

    # Show actual initialization time if available
    if init_time_s is not None and init_time_s <= zoom_end:
        ax2.axvline(
            init_time_s,
            color="purple",
            linestyle="--",
            linewidth=2.5,
            alpha=0.8,
            zorder=5,
        )
        ax2.text(
            init_time_s,
            0.85,
            f"Actual init time:\n{init_time_s:.3f}s",
            fontsize=9,
            color="purple",
            fontweight="bold",
            va="center",
            ha="left" if init_time_s < zoom_end * 0.7 else "right",
            bbox=dict(
                boxstyle="round",
                facecolor="lavender",
                alpha=0.95,
                edgecolor="purple",
                linewidth=1.5,
            ),
        )

    # Add annotation explaining the issue
    explanation = (
        "Hybrid: Correctly starts at HV=0\n"
        "PLS: Incorrectly starts at HV~0.75-0.86\n\n"
        "Cause: Greedy initial solutions\ntimestamped at 1-50 microseconds"
    )
    if init_time_s is not None:
        explanation += (
            f"\n\nActual init takes {init_time_s:.2f}s,\nbut solutions show as t=0!"
        )

    ax2.text(
        0.98,
        0.05,
        explanation,
        transform=ax2.transAxes,
        fontsize=9,
        ha="right",
        va="bottom",
        bbox=dict(
            boxstyle="round",
            facecolor="lightyellow",
            alpha=0.9,
            edgecolor="orange",
            linewidth=2,
        ),
    )

    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax2.set_ylabel("Normalized Hypervolume", fontsize=11)
    ax2.set_title(
        f"Zoomed to First {zoom_end}s - {instance} (initial HV detail)",
        fontsize=13,
        fontweight="bold",
    )
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, zoom_end)
    ax2.set_ylim(-0.05, 1.0)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    print(f"Saved plot: {output_path}")


def print_initial_hv_table(data: dict, init_time_info: dict = None) -> None:
    """Print a table showing initial HV values for all algorithms."""
    instance = data["instance"]

    print(f"\n{'=' * 80}")
    print(f"INITIAL HV BUG ANALYSIS: {instance}")
    print(f"{'=' * 80}\n")

    if init_time_info and init_time_info.get("init_time_s"):
        print(
            f"MEASURED INITIALIZATION TIME: {init_time_info['init_time_s']:.3f} seconds"
        )
        print(
            f"Number of initial solutions: {init_time_info.get('num_initial', 'N/A')}"
        )
        print(f"\n{'-' * 80}\n")

    print(
        f"{'Algorithm':<25} {'HV(t=0)':<12} {'HV(final)':<12} {'Initial %':<12} {'Status':<10}"
    )
    print("-" * 80)

    for cfg_name, cfg_data in data["configs"].items():
        curve = cfg_data.get("curve", [])
        if not curve:
            continue

        t0_hv = curve[0][1]
        final_hv = curve[-1][1]
        initial_pct = (t0_hv / final_hv * 100) if final_hv > 0 else 0

        status = "OK" if t0_hv < 0.01 else "BUG"

        print(
            f"{cfg_name:<25} {t0_hv:<12.6f} {final_hv:<12.6f} "
            f"{initial_pct:<11.1f}% {status:<10}"
        )

    print(f"\n{'=' * 80}")
    print("SUMMARY:")
    print(f"{'=' * 80}")
    explanation = """
PLS algorithms show initial HV of 75-86% due to greedy initialization
being timestamped at 1-50 microseconds (effectively t=0 in plots).
"""
    if init_time_info and init_time_info.get("init_time_s"):
        explanation += f"""
HOWEVER: Actual initialization takes {init_time_info["init_time_s"]:.3f} seconds!
The {init_time_info.get("num_initial", "N/A")} initial solutions are generated over {init_time_info["init_time_s"]:.3f}s
but artificially timestamped at 0.000001-0.000050 seconds.
"""
    explanation += """
This makes PLS appear to converge instantly and obscures actual
optimization dynamics.

Hybrid algorithms correctly start at HV=0 because their MILP phase
begins at true t=0.

See INITIAL_HV_BUG_ANALYSIS.md for detailed analysis and proposed fixes.
    """
    print(explanation)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize initial HV bug in PLS algorithms"
    )
    parser.add_argument(
        "--instance",
        type=str,
        default="lagos_nigeria_100",
        help="Instance name (default: lagos_nigeria_100)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hv_experiment_results"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--measure-init-time",
        action="store_true",
        help="Run PLS to measure actual initialization time (takes ~5s)",
    )
    args = parser.parse_args()

    try:
        data = load_results(args.instance)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(f"\nAvailable instances:", file=sys.stderr)
        for f in sorted(RESULTS_DIR.glob("*.json")):
            if f.stem != "all_experiments":
                print(f"  - {f.stem}", file=sys.stderr)
        return 1

    # Optionally measure initialization time
    init_time_info = None
    if args.measure_init_time:
        init_time_info = measure_init_time(args.instance)

    # Print table
    print_initial_hv_table(data, init_time_info)

    # Create visualization
    output_path = args.output_dir / f"{args.instance}_initial_hv_bug.png"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_initial_hv_comparison(data, output_path, init_time_info)

    return 0


if __name__ == "__main__":
    sys.exit(main())
