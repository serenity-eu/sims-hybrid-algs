#!/usr/bin/env python
"""Measure actual initialization time and analyze initial solution timestamps.

This script runs PLS algorithms and analyzes their traces to determine:
1. How many initial solutions are generated
2. When they are timestamped (should be 1, 2, 3... microseconds)
3. When actual optimization begins (the gap after initialization)
4. The true initialization time

Usage:
    uv run python measure_init_time.py --instance lagos_nigeria_100
    uv run python measure_init_time.py --instance paris_50 --timeout 5
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
from typing import Any

import sims_problem

INSTANCES_DIR = Path(__file__).parent / "tests" / "data"


def parse_trace(trace_bytes: bytes) -> dict[str, Any]:
    """Parse trace archive and extract timestamps and metadata."""
    with tarfile.open(fileobj=io.BytesIO(trace_bytes), mode="r:gz") as tar:
        meta = json.loads(tar.extractfile("metadata.json").read())
        ts_raw = tar.extractfile("timestamp.bin").read()
        obj_raw = tar.extractfile("objectives.bin").read()
        dom_raw = tar.extractfile("dominated.bin").read()

    n = meta["solution_count"]
    ndim = len(meta["objectives"])

    timestamps_us = [struct.unpack_from("<I", ts_raw, i * 4)[0] for i in range(n)]
    dominated = [struct.unpack_from("<I", dom_raw, i * 4)[0] for i in range(n)]
    objectives = [
        tuple(
            struct.unpack_from("<Q", obj_raw, (i * ndim + j) * 8)[0]
            for j in range(ndim)
        )
        for i in range(n)
    ]

    return {
        "metadata": meta,
        "timestamps_us": timestamps_us,
        "dominated": dominated,
        "objectives": objectives,
    }


def analyze_initialization_phase(trace_data: dict[str, Any]) -> dict[str, Any]:
    """Analyze trace to identify initialization phase."""
    timestamps_us = trace_data["timestamps_us"]
    dominated = trace_data["dominated"]
    objectives = trace_data["objectives"]

    if not timestamps_us:
        return {
            "num_initial_solutions": 0,
            "num_initial_non_dominated": 0,
            "init_end_timestamp_us": 0,
            "first_optimized_timestamp_us": 0,
            "true_init_time_ms": 0,
        }

    # Sort timestamps to analyze in order
    sorted_ts = sorted(enumerate(timestamps_us), key=lambda x: x[1])

    # Find largest gap in timestamps (indicates transition from init to optimization)
    gaps = []
    for i in range(len(sorted_ts) - 1):
        idx1, ts1 = sorted_ts[i]
        idx2, ts2 = sorted_ts[i + 1]
        gap = ts2 - ts1
        gaps.append((gap, i, ts1, ts2, idx1, idx2))

    gaps.sort(reverse=True)

    # The largest gap likely separates initialization from optimization
    # Initial solutions have timestamps 1, 2, 3... (small increments)
    # Then there's a jump to the actual optimization start time
    largest_gap, split_idx, ts_before, ts_after, _, _ = gaps[0]

    # Count solutions before and after the gap
    init_end_timestamp_us = ts_before
    first_optimized_timestamp_us = ts_after

    num_initial = sum(1 for ts in timestamps_us if ts <= init_end_timestamp_us)
    num_initial_non_dominated = sum(
        1
        for i, ts in enumerate(timestamps_us)
        if ts <= init_end_timestamp_us and dominated[i] == 0xFFFFFFFF
    )

    # The true initialization time is approximately the timestamp of the first
    # optimized solution (since init solutions are artificially timestamped at 1, 2, 3...)
    true_init_time_ms = first_optimized_timestamp_us / 1000.0

    return {
        "num_initial_solutions": num_initial,
        "num_initial_non_dominated": num_initial_non_dominated,
        "init_end_timestamp_us": init_end_timestamp_us,
        "first_optimized_timestamp_us": first_optimized_timestamp_us,
        "true_init_time_ms": true_init_time_ms,
        "largest_gap_us": largest_gap,
        "largest_gap_ms": largest_gap / 1000.0,
    }


def print_detailed_analysis(
    trace_data: dict[str, Any], init_analysis: dict[str, Any]
) -> None:
    """Print detailed analysis of initialization phase."""
    timestamps_us = trace_data["timestamps_us"]
    dominated = trace_data["dominated"]
    total_duration = trace_data["metadata"]["total_duration"]

    print(f"\n{'=' * 80}")
    print("INITIALIZATION TIME ANALYSIS")
    print(f"{'=' * 80}\n")

    print(f"Total solutions in trace: {len(timestamps_us)}")
    print(f"Total optimization time:  {total_duration / 1_000_000:.3f} seconds\n")

    print(f"{'─' * 80}")
    print("INITIALIZATION PHASE:")
    print(f"{'─' * 80}")
    print(
        f"  Number of initial solutions:     {init_analysis['num_initial_solutions']}"
    )
    print(
        f"  Non-dominated initial solutions: {init_analysis['num_initial_non_dominated']}"
    )
    print(
        f"  Initial solutions timestamp:     1 to {init_analysis['init_end_timestamp_us']} µs"
    )
    print(
        f"  True initialization time:        {init_analysis['true_init_time_ms']:.3f} ms"
    )
    print(f"  (= {init_analysis['true_init_time_ms'] / 1000:.6f} seconds)\n")

    print(f"{'─' * 80}")
    print("OPTIMIZATION PHASE:")
    print(f"{'─' * 80}")
    print(
        f"  First optimized solution at:     {init_analysis['first_optimized_timestamp_us']:,} µs"
    )
    print(
        f"  (= {init_analysis['first_optimized_timestamp_us'] / 1_000_000:.3f} seconds)"
    )
    print(
        f"  Gap between init and optim:      {init_analysis['largest_gap_ms']:.3f} ms\n"
    )

    # Show first few timestamps to illustrate the pattern
    sorted_ts = sorted(timestamps_us)[:30]
    print(f"{'─' * 80}")
    print("FIRST 30 TIMESTAMPS (showing artificial 1, 2, 3... pattern):")
    print(f"{'─' * 80}")
    for i, ts in enumerate(sorted_ts):
        marker = (
            " ← INIT" if ts <= init_analysis["init_end_timestamp_us"] else " ← OPTIM"
        )
        print(f"  [{i:2d}] {ts:10,} µs ({ts / 1000:8.3f} ms){marker}")

    print(f"\n{'=' * 80}")
    print("THE PROBLEM:")
    print(f"{'=' * 80}")
    print(
        f"""
Initial solutions are timestamped at 1, 2, 3... microseconds.
When plotting HV curves at t=0 seconds, these all get included!

- Artificial timestamps:  {init_analysis["num_initial_solutions"]} solutions at 0.000001-0.0000{init_analysis["init_end_timestamp_us"]:02d} seconds
- Real initialization time: {init_analysis["true_init_time_ms"]:.3f} milliseconds
- Result: HV curve starts high at t=0 instead of showing true optimization progress

PROPOSED FIX:
- Measure initialization time ({init_analysis["true_init_time_ms"]:.3f} ms)
- Timestamp initial solutions at t = -{init_analysis["true_init_time_ms"]:.3f} ms
- OR start optimization timer AFTER initialization completes
- OR report initial HV as separate metric and normalize timestamps
"""
    )


def run_and_analyze(instance_name: str, timeout_s: int, config_name: str) -> None:
    """Run PLS algorithm and analyze initialization time."""
    instance_file = f"{instance_name}.dzn"
    instance_path = INSTANCES_DIR / instance_file

    if not instance_path.exists():
        print(f"ERROR: Instance not found: {instance_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading problem: {instance_name}")
    problem = sims_problem.SimsDiscreteProblem.from_dzn(str(instance_path))

    objectives = [
        "min_cost",
        "cloud_coverage",
        "min_max_incidence_angle",
        "min_resolution",
    ]

    # Run appropriate algorithm
    print(f"Running {config_name} for {timeout_s} seconds with trace enabled...")

    if config_name == "Pure PLS":
        result = sims_problem.solve_with_pls(
            problem,
            objectives=objectives,
            timeout=timedelta(seconds=timeout_s),
            is_deterministic=True,
            trace=True,
            use_greedy_initial_population=True,
        )
    elif config_name == "Improved PLS":
        result = sims_problem.solve_with_pls(
            problem,
            objectives=objectives,
            timeout=timedelta(seconds=timeout_s),
            is_deterministic=True,
            trace=True,
            use_checkpoint=True,
            use_ranked_candidates=True,
            max_k1_candidates=15,
            use_greedy_initial_population=True,
            use_perturbation_restart=True,
        )
    elif config_name == "Diverse PLS":
        result = sims_problem.solve_with_pls(
            problem,
            objectives=objectives,
            timeout=timedelta(seconds=timeout_s),
            is_deterministic=True,
            trace=True,
            use_checkpoint=True,
            use_ranked_candidates=True,
            max_k1_candidates=15,
            use_greedy_initial_population=True,
            use_perturbation_restart=True,
            use_diverse_probing=True,
        )
    else:
        print(f"ERROR: Unknown config: {config_name}", file=sys.stderr)
        sys.exit(1)

    print(f"Generated trace: {len(result.trace):,} bytes")
    print(f"Final solutions: {len(result.final_solutions)}")

    # Parse and analyze trace
    trace_data = parse_trace(result.trace)
    init_analysis = analyze_initialization_phase(trace_data)

    # Print detailed analysis
    print_detailed_analysis(trace_data, init_analysis)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure initialization time from PLS traces"
    )
    parser.add_argument(
        "--instance",
        type=str,
        default="lagos_nigeria_100",
        help="Instance name (e.g., lagos_nigeria_100)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Diverse PLS",
        choices=["Pure PLS", "Improved PLS", "Diverse PLS"],
        help="Algorithm configuration (default: Diverse PLS)",
    )
    args = parser.parse_args()

    run_and_analyze(args.instance, args.timeout, args.config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
