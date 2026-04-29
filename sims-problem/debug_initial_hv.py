#!/usr/bin/env python
"""Debug script to analyze why HV curves start high at timestamp 0.

This script examines trace archives to understand the distribution of
initial solution timestamps and their contribution to the hypervolume
at t=0.

Usage:
    uv run python debug_initial_hv.py --instance lagos_nigeria_100
    uv run python debug_initial_hv.py --trace-file path/to/trace.tar.gz
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import tarfile
from pathlib import Path
from typing import Any

import sims_problem

INSTANCES_DIR = Path(__file__).parent / "tests" / "data"


def load_trace_data(trace_bytes: bytes) -> dict[str, Any]:
    """Extract trace data from compressed archive."""
    with tarfile.open(fileobj=io.BytesIO(trace_bytes), mode="r:gz") as tar:
        meta_member = tar.extractfile("metadata.json")
        obj_member = tar.extractfile("objectives.bin")
        dom_member = tar.extractfile("dominated.bin")
        ts_member = tar.extractfile("timestamp.bin")

        if not all([meta_member, obj_member, dom_member, ts_member]):
            raise ValueError("Trace archive is missing required members")

        meta = json.loads(meta_member.read())
        n = meta["solution_count"]
        ndim = len(meta["objectives"])

        obj_raw = obj_member.read()
        dom_raw = dom_member.read()
        ts_raw = ts_member.read()

    # Parse timestamps (u32, microseconds)
    timestamps_us = [struct.unpack_from("<I", ts_raw, i * 4)[0] for i in range(n)]

    # Parse objectives (u64 each)
    objectives = [
        tuple(
            struct.unpack_from("<Q", obj_raw, (i * ndim + j) * 8)[0]
            for j in range(ndim)
        )
        for i in range(n)
    ]

    # Parse domination (u32, index or 0xFFFFFFFF)
    dominated = [struct.unpack_from("<I", dom_raw, i * 4)[0] for i in range(n)]

    return {
        "metadata": meta,
        "timestamps_us": timestamps_us,
        "objectives": objectives,
        "dominated": dominated,
    }


def analyze_initial_solutions(trace_data: dict[str, Any]) -> None:
    """Analyze solutions with very early timestamps."""
    timestamps_us = trace_data["timestamps_us"]
    objectives = trace_data["objectives"]
    dominated = trace_data["dominated"]
    total_duration = trace_data["metadata"]["total_duration"]

    print(f"\n{'=' * 72}")
    print("TRACE OVERVIEW")
    print(f"{'=' * 72}")
    print(f"Total solutions: {len(timestamps_us)}")
    print(f"Total duration: {total_duration:,} µs ({total_duration / 1_000_000:.2f}s)")
    print(f"Objectives: {trace_data['metadata']['objectives']}")

    # Group solutions by timestamp
    timestamp_groups: dict[int, list[int]] = {}
    for idx, ts in enumerate(timestamps_us):
        if ts not in timestamp_groups:
            timestamp_groups[ts] = []
        timestamp_groups[ts].append(idx)

    print(f"\n{'=' * 72}")
    print("INITIAL TIMESTAMP ANALYSIS")
    print(f"{'=' * 72}")

    # Show first 20 unique timestamps
    sorted_timestamps = sorted(timestamp_groups.keys())[:20]
    print(f"\nFirst 20 unique timestamps (µs):")
    for ts in sorted_timestamps:
        count = len(timestamp_groups[ts])
        non_dominated = sum(
            1 for idx in timestamp_groups[ts] if dominated[idx] == 0xFFFFFFFF
        )
        print(
            f"  t={ts:6d} µs ({ts / 1000:.3f} ms): {count:3d} solutions, "
            f"{non_dominated:3d} non-dominated"
        )

    # Analyze t=0 to t=1ms
    early_cutoff_us = 1000  # 1 millisecond
    early_solutions = [
        (idx, ts) for idx, ts in enumerate(timestamps_us) if ts <= early_cutoff_us
    ]

    print(f"\n{'=' * 72}")
    print(f"SOLUTIONS AT t ≤ {early_cutoff_us / 1000:.1f} ms")
    print(f"{'=' * 72}")
    print(f"Total: {len(early_solutions)} solutions")

    early_non_dominated = [
        idx for idx, ts in early_solutions if dominated[idx] == 0xFFFFFFFF
    ]
    print(f"Non-dominated: {len(early_non_dominated)} solutions")

    if early_non_dominated:
        print(
            f"\nFirst 10 non-dominated solutions at t ≤ {early_cutoff_us / 1000:.1f} ms:"
        )
        for i, idx in enumerate(early_non_dominated[:10]):
            ts = timestamps_us[idx]
            obj = objectives[idx]
            print(f"  [{i}] t={ts:6d} µs, objectives={obj}")


def compute_hv_at_timestamp(
    trace_data: dict[str, Any],
    cutoff_us: int,
    bounds: list[tuple[int, int]],
) -> float:
    """Compute HV for all non-dominated solutions up to cutoff_us."""
    timestamps_us = trace_data["timestamps_us"]
    objectives = trace_data["objectives"]
    dominated = trace_data["dominated"]

    # Build reverse domination index
    n = len(timestamps_us)
    rev_dom: list[list[int]] = [[] for _ in range(n)]
    for i, d in enumerate(dominated):
        if d != 0xFFFFFFFF and d < n:
            rev_dom[d].append(i)

    # Replay trace up to cutoff
    in_front = [False] * n
    for idx in range(n):
        if timestamps_us[idx] > cutoff_us:
            break
        in_front[idx] = True
        for victim in rev_dom[idx]:
            in_front[victim] = False

    # Extract current front
    front_objectives = [objectives[i] for i in range(n) if in_front[i]]

    if not front_objectives:
        return 0.0

    # Compute normalized HV
    from sims_problem import compute_hypervolume

    hv = compute_hypervolume(front_objectives, bounds)
    return hv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Debug initial HV values in PLS traces"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--instance",
        type=str,
        help="Instance name (e.g., lagos_nigeria_100)",
    )
    group.add_argument(
        "--trace-file",
        type=Path,
        help="Path to trace.tar.gz file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Pure PLS",
        help="Algorithm config to analyze (default: Pure PLS)",
    )
    args = parser.parse_args()

    # Load trace data
    if args.trace_file:
        print(f"Loading trace from: {args.trace_file}")
        with open(args.trace_file, "rb") as f:
            trace_bytes = f.read()
        print(f"Loaded {len(trace_bytes):,} bytes")
    else:
        # Run the algorithm to get trace
        instance_file = f"{args.instance}.dzn"
        instance_path = INSTANCES_DIR / instance_file

        if not instance_path.exists():
            print(f"ERROR: Instance not found: {instance_path}", file=sys.stderr)
            return 1

        print(f"Running {args.config} on {args.instance}...")
        problem = sims_problem.SimsDiscreteProblem.from_dzn(str(instance_path))

        from datetime import timedelta

        # Short run for debugging
        result = sims_problem.solve_with_pls(
            problem,
            objectives=[
                "min_cost",
                "cloud_coverage",
                "min_max_incidence_angle",
                "min_resolution",
            ],
            timeout=timedelta(seconds=10),
            is_deterministic=True,
            trace=True,
            use_greedy_initial_population=True,
        )

        trace_bytes = result.trace
        print(f"Generated trace: {len(trace_bytes):,} bytes")

    # Parse trace
    trace_data = load_trace_data(trace_bytes)

    # Analyze initial solutions
    analyze_initial_solutions(trace_data)

    # Compute HV at different early timestamps
    print(f"\n{'=' * 72}")
    print("HYPERVOLUME EVOLUTION (first 10 seconds)")
    print(f"{'=' * 72}")

    # Extract bounds from trace
    objectives = trace_data["objectives"]
    ndim = len(objectives[0])
    bounds = [
        (min(obj[i] for obj in objectives), max(obj[i] for obj in objectives))
        for i in range(ndim)
    ]

    print(f"\nBounds: {bounds}")

    # Sample HV at various timestamps
    sample_times = [0, 100, 1000, 10000, 100000, 1000000, 5000000, 10000000]
    print(f"\n{'Time (µs)':>12} {'Time (s)':>10} {'HV':>12} {'Front Size':>12}")
    print("-" * 50)

    for cutoff_us in sample_times:
        if cutoff_us > trace_data["metadata"]["total_duration"]:
            break

        hv = compute_hv_at_timestamp(trace_data, cutoff_us, bounds)

        # Count non-dominated solutions at this timestamp
        timestamps_us = trace_data["timestamps_us"]
        dominated = trace_data["dominated"]
        n = len(timestamps_us)

        rev_dom: list[list[int]] = [[] for _ in range(n)]
        for i, d in enumerate(dominated):
            if d != 0xFFFFFFFF and d < n:
                rev_dom[d].append(i)

        in_front = [False] * n
        for idx in range(n):
            if timestamps_us[idx] > cutoff_us:
                break
            in_front[idx] = True
            for victim in rev_dom[idx]:
                in_front[victim] = False

        front_size = sum(in_front)

        print(
            f"{cutoff_us:>12,} {cutoff_us / 1_000_000:>10.3f} {hv:>12.6f} {front_size:>12}"
        )

    print(f"\n{'=' * 72}")
    print("DIAGNOSIS")
    print(f"{'=' * 72}")
    print("""
The high initial HV at t=0 is caused by:

1. Greedy initial solutions are generated before the optimization timer starts
2. These solutions are assigned timestamps of 1, 2, 3... microseconds
3. When sampling the HV curve at t=0, all these solutions are included
4. Result: The HV curve appears to start at ~0.75-0.86 instead of 0.0

POTENTIAL FIXES:
A. Start timer BEFORE generating initial solutions
B. Assign initial solutions timestamp=0, record as separate "initial HV"
C. Adjust HV curve to start from first "real" optimization timestamp
D. Normalize timestamps: first non-initial solution → t=0
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
