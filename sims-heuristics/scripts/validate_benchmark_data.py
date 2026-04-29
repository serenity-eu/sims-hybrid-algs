#!/usr/bin/env python3
"""Validate benchmark JSON data against SIMS problem instances.

Checks performed per algorithm front:
  1. Pareto non-dominance: no solution dominates another in the same front.
  2. No duplicates: all objective vectors are unique.
  3. Objective bounds feasibility: every objective value falls within the
     theoretically achievable range for the instance.
  4. Hypervolume recomputation: recompute HV via sims_problem.compute_hypervolume
     (Rust) and compare against stored values.
  5. Archive size consistency: stored archive_size matches len(objectives).
  6. HV in [0, 1] range.

Usage (from workspace root):
    .venv/bin/python sims-heuristics/scripts/validate_benchmark_data.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
HEURISTICS_DIR = SCRIPT_DIR.parent
BENCHMARK_DATA_DIR = HEURISTICS_DIR / "docs" / "benchmark_data"

DZN_SEARCH_PATHS = [
    HEURISTICS_DIR / "tests" / "data",
    HEURISTICS_DIR.parent / "sims-problem" / "tests" / "data",
    HEURISTICS_DIR.parent / "sims-core" / "tests" / "data",
]

# ---------------------------------------------------------------------------
# Imports from sims_problem (Rust via PyO3)
# ---------------------------------------------------------------------------

try:
    from sims_problem import SimsDiscreteProblem, compute_hypervolume
except ImportError:
    print(
        "ERROR: cannot import sims_problem. "
        "Run from the workspace venv:\n"
        "  .venv/bin/python sims-heuristics/scripts/validate_benchmark_data.py",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REF_VOLUME = 1.1**4  # ≈ 1.4641
# Scale factor: we map normalised [0,1] floats to integers [0, SCALE]
# so that compute_hypervolume (u64-based) can be used.
SCALE = 100_000
SCALED_REF = int(1.1 * SCALE)  # 110_000
NUM_OBJ = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_dzn(instance_name: str) -> Path | None:
    for d in DZN_SEARCH_PATHS:
        p = d / f"{instance_name}.dzn"
        if p.exists():
            return p
    return None


def check_nondominance_numpy(objectives: list[list[int]]) -> list[str]:
    """Vectorised non-dominance check using numpy.  O(n^2) memory for the
    broadcast but very fast in practice for n < 10_000."""
    errors: list[str] = []
    arr = np.asarray(objectives, dtype=np.int64)
    n = len(arr)
    if n <= 1:
        return errors

    # Process in batches to limit memory for very large fronts
    BATCH = 2000
    dominated_indices: list[int] = []

    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        # batch[i] vs all[j]: check if any j dominates batch[i]
        batch = arr[start:end]  # (B, D)
        # diff[i, j, d] = arr[j, d] - batch[i, d]
        # j dominates i iff all diffs <= 0 and at least one < 0
        diff = arr[np.newaxis, :, :] - batch[:, np.newaxis, :]  # (B, N, D)
        all_leq = np.all(diff <= 0, axis=2)  # (B, N)
        any_lt = np.any(diff < 0, axis=2)  # (B, N)
        dom_matrix = all_leq & any_lt  # (B, N)

        # Exclude self-domination
        for local_i in range(end - start):
            dom_matrix[local_i, start + local_i] = False

        # Any row with a True entry means that solution is dominated
        is_dominated = np.any(dom_matrix, axis=1)
        for local_i in np.where(is_dominated)[0]:
            global_i = start + local_i
            dominator = int(np.where(dom_matrix[local_i])[0][0])
            dominated_indices.append(global_i)
            if len(errors) < 10:
                errors.append(
                    f"  Solution {dominator} dominates solution {global_i}: "
                    f"{objectives[dominator]} vs {objectives[global_i]}"
                )

    if len(dominated_indices) > 10:
        errors.append(
            f"  ... and {len(dominated_indices) - 10} more dominated solutions"
        )

    return errors


def check_duplicates_numpy(objectives: list[list[int]]) -> list[str]:
    """Fast duplicate detection via numpy unique."""
    arr = np.asarray(objectives, dtype=np.int64)
    _, idx, counts = np.unique(arr, axis=0, return_index=True, return_counts=True)
    dups = np.where(counts > 1)[0]
    errors: list[str] = []
    for d in dups[:10]:
        orig = int(idx[d])
        errors.append(
            f"  Duplicate objective vector at index {orig} (appears {int(counts[d])}x): "
            f"{objectives[orig]}"
        )
    if len(dups) > 10:
        errors.append(f"  ... and {len(dups) - 10} more duplicate groups")
    return errors


def check_objective_bounds(
    objectives: list[list[int]],
    problem: SimsDiscreteProblem,
) -> list[str]:
    """Check that each objective is within the feasible range for the instance."""
    errors: list[str] = []

    max_cost = sum(problem.costs)
    min_cost = int(min(problem.costs))  # at least one image selected

    max_resolution = int(max(problem.resolution))
    min_resolution = int(min(problem.resolution))

    max_angle = int(max(problem.incidence_angle))
    min_angle = int(min(problem.incidence_angle))

    max_cloud = int(problem.max_cloud_area)

    obj_names = ["TotalCost", "CloudyArea", "MinResolution", "MaxIncidenceAngle"]
    bounds = [
        (min_cost, max_cost),
        (0, max_cloud),
        (min_resolution, max_resolution),
        (min_angle, max_angle),
    ]

    arr = np.asarray(objectives, dtype=np.int64)
    for dim in range(NUM_OBJ):
        lo, hi = bounds[dim]
        col = arr[:, dim]
        below = np.where(col < lo)[0]
        above = np.where(col > hi)[0]
        for i in below[:3]:
            errors.append(
                f"  Solution {int(i)}: {obj_names[dim]} = {int(col[i])} < lower bound {lo}"
            )
        for i in above[:3]:
            errors.append(
                f"  Solution {int(i)}: {obj_names[dim]} = {int(col[i])} > upper bound {hi}"
            )
        if len(below) > 3:
            errors.append(
                f"  ... {len(below) - 3} more {obj_names[dim]} below-bound violations"
            )
        if len(above) > 3:
            errors.append(
                f"  ... {len(above) - 3} more {obj_names[dim]} above-bound violations"
            )

    return errors


def recompute_hv_rust(
    objectives: list[list[int]],
    shared_lo: list[int],
    shared_hi: list[int],
) -> float:
    """Recompute normalised HV ratio using Rust compute_hypervolume.

    Replicates the benchmark methodology:
      1. Normalise to [0, 1] using shared_lo / shared_hi.
      2. Scale to integers [0, SCALE].
      3. Reference point at [SCALED_REF]^4 (= 1.1 * SCALE).
      4. Compute HV via Rust.
      5. Divide by (SCALE^4 * 1.1^4) to get ratio in [0, 1].
    """
    if not objectives:
        return 0.0

    ranges = [max(float(shared_hi[d] - shared_lo[d]), 1.0) for d in range(NUM_OBJ)]

    # Normalise + scale to integer coords
    scaled_pts: list[list[int]] = []
    for obj in objectives:
        pt = []
        for d in range(NUM_OBJ):
            norm = (obj[d] - shared_lo[d]) / ranges[d]
            norm = max(0.0, min(1.0, norm))
            pt.append(int(round(norm * SCALE)))
        scaled_pts.append(pt)

    ref = [SCALED_REF] * NUM_OBJ
    bounds = [[0, SCALED_REF]] * NUM_OBJ

    raw_hv = compute_hypervolume(
        scaled_pts, bounds, reference_point=ref, normalized=False
    )

    return raw_hv / (float(SCALE) ** NUM_OBJ * REF_VOLUME)


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------


def validate_benchmark_files() -> bool:
    json_files = sorted(BENCHMARK_DATA_DIR.glob("bench_*.json"))
    if not json_files:
        print(f"ERROR: No benchmark JSON files found in {BENCHMARK_DATA_DIR}")
        return False

    print(f"Found {len(json_files)} benchmark file(s) in {BENCHMARK_DATA_DIR}\n")

    total_errors = 0
    total_warnings = 0
    total_algorithms = 0
    total_solutions = 0
    t0 = time.monotonic()

    for json_path in json_files:
        print(f"{'=' * 76}")
        print(f"File: {json_path.name}")
        print(f"{'=' * 76}")

        with open(json_path) as f:
            data = json.load(f)

        for inst in data["instances"]:
            inst_name = inst["name"]
            num_images = inst["num_images"]
            num_elements = inst["num_elements"]
            print(
                f"\n  Instance: {inst_name} ({num_images} images, {num_elements} elements)"
            )

            # --- Load problem instance ---
            dzn_path = find_dzn(inst_name)
            problem: SimsDiscreteProblem | None = None
            if dzn_path is None:
                print(
                    f"    WARNING: .dzn not found for {inst_name}, skipping bounds check"
                )
                total_warnings += 1
            else:
                problem = SimsDiscreteProblem.from_dzn(str(dzn_path))
                try:
                    problem.validate()
                except ValueError as e:
                    print(f"    ERROR: Problem instance validation failed: {e}")
                    total_errors += 1

                if problem.num_images != num_images:
                    print(
                        f"    ERROR: JSON num_images={num_images} != dzn={problem.num_images}"
                    )
                    total_errors += 1
                if problem.universe != num_elements:
                    print(
                        f"    ERROR: JSON num_elements={num_elements} != dzn={problem.universe}"
                    )
                    total_errors += 1

            # --- Shared bounds for HV computation ---
            all_objs: list[list[int]] = []
            for algo in inst["algorithms"]:
                all_objs.extend(algo["objectives"])

            if all_objs:
                arr_all = np.asarray(all_objs, dtype=np.int64)
                shared_lo = arr_all.min(axis=0).tolist()
                shared_hi = arr_all.max(axis=0).tolist()
                for d in range(NUM_OBJ):
                    if shared_hi[d] <= shared_lo[d]:
                        shared_hi[d] = shared_lo[d] + 1
            else:
                shared_lo = [0] * NUM_OBJ
                shared_hi = [1] * NUM_OBJ

            # --- Validate each algorithm ---
            for algo in inst["algorithms"]:
                algo_name = algo["name"]
                stored_hv = algo["hypervolume"]
                stored_size = algo["archive_size"]
                objectives: list[list[int]] = algo["objectives"]
                total_algorithms += 1
                total_solutions += len(objectives)

                errors: list[str] = []
                warnings: list[str] = []

                # Check 1: archive_size consistency
                if stored_size != len(objectives):
                    errors.append(
                        f"  archive_size={stored_size} != len(objectives)={len(objectives)}"
                    )

                if not objectives:
                    warnings.append("  Empty front (no solutions)")
                    _print_result(algo_name, 0, errors, warnings)
                    total_errors += len(errors)
                    total_warnings += len(warnings)
                    continue

                # Check 2: duplicates (numpy)
                errors.extend(check_duplicates_numpy(objectives))

                # Check 3: Pareto non-dominance (numpy)
                errors.extend(check_nondominance_numpy(objectives))

                # Check 4: objective bounds feasibility
                if problem is not None:
                    errors.extend(check_objective_bounds(objectives, problem))

                # Check 5: HV recomputation (Rust)
                recomputed_hv = recompute_hv_rust(objectives, shared_lo, shared_hi)
                hv_diff = abs(recomputed_hv - stored_hv)
                if hv_diff > 0.002:
                    errors.append(
                        f"  HV mismatch: stored={stored_hv:.6f}, "
                        f"recomputed={recomputed_hv:.6f}, diff={hv_diff:.6f}"
                    )
                elif hv_diff > 1e-4:
                    warnings.append(
                        f"  HV minor diff: stored={stored_hv:.6f}, "
                        f"recomputed={recomputed_hv:.6f}, diff={hv_diff:.6f}"
                    )

                # Check 6: HV in [0, 1]
                if stored_hv < -1e-9 or stored_hv > 1.0 + 1e-9:
                    errors.append(f"  HV out of [0, 1] range: {stored_hv:.6f}")

                _print_result(algo_name, len(objectives), errors, warnings)
                total_errors += len(errors)
                total_warnings += len(warnings)

    elapsed = time.monotonic() - t0

    print(f"\n{'=' * 76}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 76}")
    print(f"  Algorithms validated : {total_algorithms}")
    print(f"  Solutions checked    : {total_solutions}")
    print(f"  Errors               : {total_errors}")
    print(f"  Warnings             : {total_warnings}")
    print(f"  Elapsed              : {elapsed:.2f}s")
    print()
    if total_errors == 0:
        print("  ✅ ALL CHECKS PASSED")
    else:
        print(f"  ❌ {total_errors} ERROR(S) FOUND")

    return total_errors == 0


def _print_result(
    algo_name: str,
    num_solutions: int,
    errors: list[str],
    warnings: list[str],
) -> None:
    status = "✅" if not errors else "❌"
    parts = [f"{status} {algo_name:<45} [{num_solutions:>5} sol]"]
    if errors:
        parts.append(f" {len(errors)} error(s)")
    if warnings:
        parts.append(f" {len(warnings)} warning(s)")
    print(f"    {''.join(parts)}")
    for e in errors:
        print(f"      ERROR: {e}")
    for w in warnings:
        print(f"      WARN:  {w}")


def main() -> None:
    success = validate_benchmark_files()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
