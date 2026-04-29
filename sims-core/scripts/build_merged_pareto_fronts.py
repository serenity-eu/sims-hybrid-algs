#!/usr/bin/env python3
"""
Build merged Pareto fronts from manuels_results.csv.

For each instance:
  1. Parse every CSV row (all solvers / strategies / models).
  2. For each solution, take the image selection as ground truth and
     RECOMPUTE all 4 objectives from the DZN instance data —
     this fixes the ~16 Gurobi export bugs where recorded objectives
     didn't match the recorded image selection.
  3. Deduplicate by image selection (identical sets → identical recomputed objectives).
  4. Compute the true 4-objective Pareto front (non-dominated set).
  5. Store everything in a single manuels_pareto_fronts.json.

Usage:
    uv run python scripts/build_merged_pareto_fronts.py

Output:
    sims-problem/tests/data/manuels_pareto_fronts.json
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

# -- path setup so we can import sims.core even when run as a script ----------
SIMS_CORE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIMS_CORE_DIR / "src"))

from sims.core.sims.problem import SimsDiscreteProblem
from sims.core.sims.solver_result import Solution

csv.field_size_limit(sys.maxsize // 10)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DZN_DIR = SIMS_CORE_DIR / "tests" / "data"
CSV_PATH = (
    SIMS_CORE_DIR.parent / "sims-problem" / "tests" / "data" / "manuels_results.csv"
)
OUTPUT_PATH = (
    SIMS_CORE_DIR.parent
    / "sims-problem"
    / "tests"
    / "data"
    / "manuels_pareto_fronts.json"
)

OBJ_NAMES = ["min_cost", "cloud_coverage", "min_resolution", "min_max_incidence_angle"]

# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def parse_selection_sets(raw: str) -> list[frozenset[int]]:
    """Parse '{[0-3-7],[1-4],…}' into list of 0-indexed frozensets."""
    matches = re.findall(r"\[([^\]]+)\]", raw)
    result: list[frozenset[int]] = []
    for m in matches:
        indices = frozenset(int(x) for x in m.split("-"))
        result.append(indices)
    return result


@dataclass
class CsvRow:
    instance: str
    solver: str
    strategy: str
    model: str
    selections: list[frozenset[int]]


def load_csv() -> list[CsvRow]:
    rows: list[CsvRow] = []
    with open(CSV_PATH) as f:
        lines = f.readlines()
    # lines[0] = title, lines[1] = header, lines[2:] = data
    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        fields = line.split(";")
        if len(fields) < 18:
            continue
        selections = parse_selection_sets(fields[17])
        rows.append(
            CsvRow(
                instance=fields[0],
                solver=fields[1],
                strategy=fields[2],
                model=fields[7],
                selections=selections,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Instance cache
# ---------------------------------------------------------------------------

_instance_cache: dict[str, SimsDiscreteProblem | None] = {}


def get_instance(name: str) -> SimsDiscreteProblem | None:
    if name not in _instance_cache:
        dzn = DZN_DIR / f"{name}.dzn"
        if not dzn.exists():
            log.warning("DZN file not found: %s", dzn)
            _instance_cache[name] = None
        else:
            _instance_cache[name] = SimsDiscreteProblem.from_dzn(dzn)
    return _instance_cache[name]


# ---------------------------------------------------------------------------
# Pareto-front computation
# ---------------------------------------------------------------------------


def is_dominated(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    """Return True if *a* is dominated by *b* (all objectives minimised)."""
    all_leq = True
    any_lt = False
    for va, vb in zip(a, b):
        if vb > va:
            all_leq = False
            break
        if vb < va:
            any_lt = True
    return all_leq and any_lt


def pareto_filter(solutions: list[dict]) -> list[dict]:
    """Return the non-dominated subset.  O(n²) but n is small enough."""
    n = len(solutions)
    dominated_flags = [False] * n
    for i in range(n):
        if dominated_flags[i]:
            continue
        oi = solutions[i]["_obj_tuple"]
        for j in range(n):
            if i == j or dominated_flags[j]:
                continue
            oj = solutions[j]["_obj_tuple"]
            if is_dominated(oi, oj):
                dominated_flags[i] = True
                break
    return [s for s, d in zip(solutions, dominated_flags) if not d]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.monotonic()

    log.info("Loading CSV from %s", CSV_PATH)
    rows = load_csv()
    log.info("Loaded %d CSV rows", len(rows))

    # Group selections by instance
    instance_selections: dict[str, set[frozenset[int]]] = {}
    source_counts: dict[str, dict[str, int]] = {}  # instance -> {solver: count}

    for row in rows:
        inst = row.instance
        if inst not in instance_selections:
            instance_selections[inst] = set()
            source_counts[inst] = {}
        for sel in row.selections:
            instance_selections[inst].add(sel)
        key = f"{row.solver}/{row.strategy}/{row.model}"
        source_counts[inst][key] = source_counts[inst].get(key, 0) + len(row.selections)

    log.info("Found %d unique instances", len(instance_selections))

    # Build merged Pareto fronts
    output: dict[str, dict] = {}
    total_input = 0
    total_deduped = 0
    total_pareto = 0
    total_fixed = 0

    for inst_name in sorted(instance_selections):
        problem = get_instance(inst_name)
        if problem is None:
            log.warning("Skipping %s — no DZN file", inst_name)
            continue

        unique_selections = instance_selections[inst_name]
        n_raw = sum(source_counts[inst_name].values())
        n_unique = len(unique_selections)
        total_input += n_raw
        total_deduped += n_unique

        log.info(
            "%s: %d raw solutions -> %d unique selections, recomputing objectives…",
            inst_name,
            n_raw,
            n_unique,
        )

        # Recompute objectives for each unique selection
        solutions: list[dict] = []
        n_fixed_here = 0

        for sel in unique_selections:
            s = Solution(
                selected_images=sel,
                cost=0,
                cloudy_area=0,
                timestamp_s=timedelta(0),
                max_incidence_angle=0,
                min_resolutions_sum=0,
            )

            # Feasibility check — should always pass, but guard against bad data
            if not s.validate(problem):
                log.warning(
                    "%s: infeasible selection %s — including anyway with recomputed objectives",
                    inst_name,
                    sorted(sel),
                )

            computed = s.compute_objectives(problem, OBJ_NAMES)
            cost = computed["min_cost"]
            cloud = computed["cloud_coverage"]
            res = computed["min_resolution"]
            angle = computed["min_max_incidence_angle"]

            solutions.append(
                {
                    "selected_images": sorted(sel),
                    "cost": cost,
                    "cloudy_area": cloud,
                    "min_resolutions_sum": res,
                    "max_incidence_angle": angle,
                    # temp key for pareto filtering
                    "_obj_tuple": (cost, cloud, res, angle),
                }
            )

        # Deduplicate by objective tuple (different image sets could theoretically
        # produce identical objectives — keep all distinct image sets though)
        # We already deduped by image set above, so just filter dominated.

        pf = pareto_filter(solutions)

        # Sort Pareto front by cost (primary), then cloud, res, angle
        pf.sort(key=lambda s: s["_obj_tuple"])

        # Strip internal key
        for s in pf:
            del s["_obj_tuple"]

        n_pareto = len(pf)
        total_pareto += n_pareto

        # Count how many solutions had their objectives fixed by recomputation
        # (We can't know exactly which ones were wrong in the CSV, but we can
        # report the delta between unique inputs and pareto size)

        output[inst_name] = {
            "num_solutions": n_pareto,
            "num_raw_from_csv": n_raw,
            "num_unique_selections": n_unique,
            "sources": source_counts[inst_name],
            "solutions": pf,
        }

        log.info(
            "  -> %d Pareto-optimal solutions (from %d unique)",
            n_pareto,
            n_unique,
        )

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = time.monotonic() - t0

    print()
    print("=" * 68)
    print("  MERGED PARETO FRONTS — SUMMARY")
    print("=" * 68)
    print(f"  Instances processed : {len(output)}")
    print(f"  Raw solutions (CSV) : {total_input}")
    print(f"  After dedup (image) : {total_deduped}")
    print(f"  Pareto-optimal      : {total_pareto}")
    print(f"  Elapsed             : {elapsed:.1f}s")
    print(f"  Output              : {OUTPUT_PATH}")
    print("=" * 68)
    print()

    # Per-instance table
    print(f"  {'Instance':<25s} {'Raw':>6s} {'Unique':>7s} {'Pareto':>7s}")
    print(f"  {'-' * 25} {'-' * 6} {'-' * 7} {'-' * 7}")
    for inst_name, data in sorted(output.items()):
        print(
            f"  {inst_name:<25s} {data['num_raw_from_csv']:>6d} "
            f"{data['num_unique_selections']:>7d} {data['num_solutions']:>7d}"
        )
    print()


if __name__ == "__main__":
    main()
