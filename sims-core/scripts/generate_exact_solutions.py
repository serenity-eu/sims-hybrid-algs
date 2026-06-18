#!/usr/bin/env python
"""Generate exact Pareto front solutions for SIMS instances using Gurobi (via ortools-py path).

Runs GPBA-A through the same code path as the two-phase integration tests
(solve_with_two_phases with ratio=(100, 0)) and writes JSON compatible with
the pseudo-solver format.  Includes a `pareto_front_complete` flag that is
True when the solver exhausted the search space before the timeout.

Usage (from repo root):
    uv run python sims-core/scripts/generate_exact_solutions.py
    uv run python sims-core/scripts/generate_exact_solutions.py --timeout 3600
    uv run python sims-core/scripts/generate_exact_solutions.py --filter "_30|_50"
    uv run python sims-core/scripts/generate_exact_solutions.py \\
        --instances-dir sims-core/tests/data \\
        --output-dir sims-core/tests/data/pseudo_solver_solutions \\
        --timeout 3600
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTANCES_DIR = REPO_ROOT / "sims-core" / "tests" / "data"
OUTPUT_DIR = REPO_ROOT / "sims-core" / "tests" / "data" / "pseudo_solver_solutions"

DEFAULT_OBJECTIVES = [
    "min_cost",
    "cloud_coverage",
    "min_max_incidence_angle",
    "min_resolution",
]

OBJ_FIELD_MAP = {
    "min_cost": "cost",
    "cloud_coverage": "cloudy_area",
    "min_max_incidence_angle": "max_incidence_angle",
    "min_resolution": "min_resolutions_sum",
}


def _objectives_label(objectives: list[str]) -> str:
    return f"{len(objectives)}d"


def _solution_to_dict(sol, idx: int) -> dict:
    return {
        "selected_images": sorted(sol.selected_images),
        "cost": sol.cost,
        "cloudy_area": sol.cloudy_area,
        "max_incidence_angle": sol.max_incidence_angle,
        "min_resolutions_sum": sol.min_resolutions_sum,
        "timestamp_s": sol.timestamp_s.total_seconds(),
        "phase": "exact",
        "index": idx,
    }


def process_instance(
    dzn_path: Path,
    timeout_s: int,
    objectives: list[str],
    output_dir: Path,
) -> dict:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "sims-core" / "src"))
    sys.path.insert(0, str(REPO_ROOT / "sims-core" / "tests"))

    from sims.core.sims.problem import ProblemInstance
    from sims.core.sims.solver import solve_with_two_phases
    from sims.core.sims.solver_config import (
        FrontStrategy,
        SolverType,
        TwoPhaseSolverConfig,
    )

    instance_name = dzn_path.stem
    print(
        f"[{instance_name}]  timeout={timeout_s}s  objectives={len(objectives)}D",
        flush=True,
    )

    problem_instance = ProblemInstance.from_dzn(dzn_path)

    with tempfile.TemporaryDirectory() as tmp:
        experiment_path = Path(tmp)

        solver_config = TwoPhaseSolverConfig(
            exact_solver_type=SolverType.OR_TOOLS,
            front_strategy=FrontStrategy.GPBA_A,
            timeout_s=timeout_s,
            ratio=(100, 0),
        )

        t0 = time.time()
        try:
            result = solve_with_two_phases(
                problem_instance=problem_instance,
                problem_path=dzn_path,
                experiment_path=experiment_path,
                solver_config=solver_config,
                objectives=objectives,
            )
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  ERROR after {elapsed:.1f}s: {exc}", flush=True)
            return {"instance": instance_name, "error": str(exc)}

    elapsed = time.time() - t0
    exact = result.exact_solver_result
    solutions = exact.pareto_front if exact else []
    is_complete = exact.pareto_front_complete if exact else False

    status = "COMPLETE" if is_complete else f"TIMEOUT ({elapsed:.0f}s)"
    print(f"  {status}  |  solutions={len(solutions)}  time={elapsed:.1f}s", flush=True)

    payload = {
        "instance_name": instance_name,
        "test_type": _objectives_label(objectives),
        "objectives": objectives,
        "num_solutions": len(solutions),
        "pareto_front_complete": is_complete,
        "solutions": [_solution_to_dict(s, i) for i, s in enumerate(solutions)],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{instance_name}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Written → {out_path}", flush=True)

    return {
        "instance": instance_name,
        "solutions": len(solutions),
        "complete": is_complete,
        "elapsed_s": round(elapsed, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--instances-dir",
        type=Path,
        default=INSTANCES_DIR,
        help="Directory containing .dzn instance files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to write solution JSON files",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Solver timeout in seconds per instance (default: 3600)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Regex filter on instance name (e.g. '_30|_50')",
    )
    parser.add_argument(
        "--objectives",
        type=str,
        nargs="+",
        default=DEFAULT_OBJECTIVES,
        help="Objectives to optimize (default: all 4)",
    )
    args = parser.parse_args()

    dzn_files = sorted(args.instances_dir.glob("*.dzn"))
    if not dzn_files:
        print(f"No .dzn files found in {args.instances_dir}", file=sys.stderr)
        return 1

    if args.filter:
        pat = re.compile(args.filter)
        dzn_files = [p for p in dzn_files if pat.search(p.stem)]

    if not dzn_files:
        print("No instances matched the filter.", file=sys.stderr)
        return 1

    print(f"Instances dir : {args.instances_dir}")
    print(f"Output dir    : {args.output_dir}")
    print(f"Timeout       : {args.timeout}s per instance")
    print(f"Objectives    : {args.objectives}")
    print(f"Instances     : {len(dzn_files)}\n")

    results = []
    for dzn_path in dzn_files:
        r = process_instance(dzn_path, args.timeout, args.objectives, args.output_dir)
        results.append(r)
        print()

    print("=" * 70)
    print(f"{'Instance':30s}  {'Status':10s}  {'Solutions':>9}  {'Time(s)':>7}")
    print("-" * 70)
    has_errors = False
    for r in results:
        if "error" in r:
            print(f"{r['instance']:30s}  ERROR: {r['error']}")
            has_errors = True
            continue
        status = "COMPLETE" if r["complete"] else "TIMEOUT"
        print(
            f"{r['instance']:30s}  {status:10s}  {r['solutions']:>9}  {r['elapsed_s']:>7.1f}"
        )
    print("=" * 70)

    complete_count = sum(1 for r in results if r.get("complete"))
    print(
        f"\n{complete_count}/{len(results)} instances fully solved (pareto_front_complete=true)"
    )
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
