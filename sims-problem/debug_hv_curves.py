#!/usr/bin/env python
"""Debug HV curve computation for PLS algorithms.

Usage:
    uv run python debug_hv_curves.py --timeout 60
    uv run python debug_hv_curves.py --timeout 120 --config "Scalarized PLS"
"""

import argparse
import io
import json
import struct
import sys
import tarfile
from datetime import timedelta
from pathlib import Path

import sims_problem

INSTANCES_DIR = Path(__file__).parent / "tests" / "data"
OBJECTIVES = ["min_cost", "cloud_coverage", "min_max_incidence_angle", "min_resolution"]


def parse_trace_metadata(trace_bytes: bytes) -> dict:
    """Extract detailed trace metadata."""
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


def analyze_trace(label: str, trace_bytes: bytes):
    """Analyze trace timestamps and solution distribution."""
    print(f"\n{'=' * 80}")
    print(f"TRACE ANALYSIS: {label}")
    print(f"{'=' * 80}")

    trace_data = parse_trace_metadata(trace_bytes)
    timestamps_us = trace_data["timestamps_us"]
    dominated = trace_data["dominated"]
    objectives = trace_data["objectives"]
    total_duration = trace_data["metadata"]["total_duration"]

    print(f"Total solutions: {len(timestamps_us)}")
    print(f"Total duration: {total_duration:,} µs ({total_duration / 1_000_000:.2f}s)")

    # Timestamp distribution
    sorted_ts = sorted(timestamps_us)
    print(f"\nTimestamp range:")
    print(f"  Min: {sorted_ts[0]:,} µs ({sorted_ts[0] / 1_000_000:.6f}s)")
    print(f"  Max: {sorted_ts[-1]:,} µs ({sorted_ts[-1] / 1_000_000:.6f}s)")
    print(
        f"  Median: {sorted_ts[len(sorted_ts) // 2]:,} µs ({sorted_ts[len(sorted_ts) // 2] / 1_000_000:.6f}s)"
    )

    # First 20 timestamps
    print(f"\nFirst 20 timestamps:")
    for i, ts in enumerate(sorted_ts[:20]):
        dom_status = (
            "NON-DOM"
            if dominated[timestamps_us.index(ts)] == 0xFFFFFFFF
            else "dominated"
        )
        print(f"  [{i:2d}] {ts:10,} µs ({ts / 1_000_000:8.6f}s) - {dom_status}")

    # Count non-dominated at different time points
    time_checkpoints = [0, 100, 1000, 10000, 100000, 1000000, total_duration]
    print(f"\nNon-dominated solutions at checkpoints:")
    for checkpoint_us in time_checkpoints:
        if checkpoint_us > total_duration:
            checkpoint_us = total_duration

        count = sum(
            1
            for i, ts in enumerate(timestamps_us)
            if ts <= checkpoint_us and dominated[i] == 0xFFFFFFFF
        )
        print(
            f"  t={checkpoint_us:10,} µs ({checkpoint_us / 1_000_000:8.3f}s): {count:5d} non-dominated"
        )

    return trace_data


def compute_hv_with_logging(
    label: str, trace_bytes: bytes, bounds: list, num_points: int
):
    """Compute HV curve with detailed logging."""
    print(f"\n{'=' * 80}")
    print(f"HV CURVE COMPUTATION: {label}")
    print(f"{'=' * 80}")

    print(f"Bounds: {bounds}")
    print(f"Num points: {num_points}")

    # Compute HV curve
    curve = sims_problem.compute_hv_curve_from_trace(trace_bytes, bounds, num_points)

    print(f"\nCurve has {len(curve)} points")
    print(f"\nFirst 10 points:")
    for i, (t, hv) in enumerate(curve[:10]):
        print(f"  [{i:2d}] t={t:8.4f}s, HV={hv:.8f}")

    print(f"\nLast 10 points:")
    for i, (t, hv) in enumerate(curve[-10:], start=len(curve) - 10):
        print(f"  [{i:2d}] t={t:8.4f}s, HV={hv:.8f}")

    initial_hv = curve[0][1]
    final_hv = curve[-1][1]
    improvement = final_hv - initial_hv

    print(f"\nSummary:")
    print(f"  Initial HV (t={curve[0][0]:.4f}s): {initial_hv:.8f}")
    print(f"  Final HV (t={curve[-1][0]:.4f}s):   {final_hv:.8f}")
    print(
        f"  Improvement:                       {improvement:.8f} ({improvement / final_hv * 100 if final_hv > 0 else 0:.4f}%)"
    )

    return curve


def run_config(config_name: str, problem, timeout_s: int):
    """Run a specific configuration."""
    print(f"\n{'#' * 80}")
    print(f"# RUNNING: {config_name}")
    print(f"{'#' * 80}")

    if config_name == "Pure PLS":
        result = sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=timeout_s),
            is_deterministic=True,
            trace=True,
            use_greedy_initial_population=True,
        )
    elif config_name == "Diverse Probe PLS":
        result = sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
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
    elif config_name == "Scalarized PLS":
        result = sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=timeout_s),
            is_deterministic=True,
            trace=True,
            use_checkpoint=True,
            use_ranked_candidates=True,
            max_k1_candidates=15,
            use_greedy_initial_population=True,
            use_perturbation_restart=True,
            solution_selection_mode="scalarized-chebycheff",
            scalarized_selection_source="archive",
            scalarized_parent_budget=4,
            scalarized_weight_samples=4,
            scalarized_rho=1e-3,
            use_nd_tree_scalarized_query=True,
        )
    else:
        raise ValueError(f"Unknown config: {config_name}")

    print(f"\nRun completed:")
    print(f"  Final solutions: {len(result.final_solutions)}")
    print(f"  Trace size: {len(result.trace):,} bytes")

    return result


def main():
    parser = argparse.ArgumentParser(description="Debug HV curve computation")
    parser.add_argument("--instance", default="lagos_nigeria_100", help="Instance name")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds")
    parser.add_argument("--num-points", type=int, default=30, help="HV curve points")
    parser.add_argument(
        "--config",
        choices=["Pure PLS", "Diverse Probe PLS", "Scalarized PLS"],
        default="Diverse Probe PLS",
        help="Configuration to test",
    )
    args = parser.parse_args()

    # Load problem
    instance_path = INSTANCES_DIR / f"{args.instance}.dzn"
    if not instance_path.exists():
        print(f"ERROR: Instance not found: {instance_path}", file=sys.stderr)
        return 1

    print(f"Loading instance: {args.instance}")
    problem = sims_problem.SimsDiscreteProblem.from_dzn(str(instance_path))

    # Run configuration
    result = run_config(args.config, problem, args.timeout)

    # Analyze trace
    trace_data = analyze_trace(args.config, result.trace)

    # Extract all objectives for bounds
    all_objectives = trace_data["objectives"]
    ndim = len(all_objectives[0])
    bounds = [
        (min(obj[i] for obj in all_objectives), max(obj[i] for obj in all_objectives))
        for i in range(ndim)
    ]

    # Compute HV curve
    curve = compute_hv_with_logging(args.config, result.trace, bounds, args.num_points)

    # Check if normalization would apply
    print(f"\n{'=' * 80}")
    print(f"NORMALIZATION CHECK")
    print(f"{'=' * 80}")

    first_hv = curve[0][1]
    threshold = 0.05
    print(f"First HV value: {first_hv:.8f}")
    print(f"Threshold: {threshold}")

    if first_hv > threshold:
        print(f"✓ Would be NORMALIZED (prepend (0, 0) point)")
        print(f"  New curve would start at HV=0.0")
    else:
        print(f"✗ Would NOT be normalized")
        print(f"  Curve keeps original initial HV={first_hv:.8f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
