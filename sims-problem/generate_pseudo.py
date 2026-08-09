#!/usr/bin/env python
"""Generate 2D (biobjective) exact-front pseudo-solver solutions using the Rust
augmecon-rs solver.

The first-phase method (--method: gpba / aneja) and MILP backend (--solver:
highs / gurobi / coin_cbc) are both selectable; output is keyed by the pair, e.g.
gpbaa_2d_highs, an_2d_highs, gpbaa_2d_gurobi, an_2d_gurobi (see out_dir_for).

Writes one JSON per instance under sims-core/tests/data/ in the same format as
gpbaa_2d_pub/ (list of solutions with selected_images, objectives and a
per-solution timestamp_s taken from the trace). Timestamps let the hybrid configs
filter which exact solutions are available at each exact-phase handoff.

Usage:
    uv run python generate_pseudo.py --filter lagos_nigeria_100
    uv run python generate_pseudo.py --method aneja --solver gurobi --timeout 600
    uv run python generate_pseudo.py                 # all publication instances
"""

import argparse
import json
import time
from datetime import timedelta
from pathlib import Path

import sims_problem

PUB_DIR = Path(__file__).parent.parent / "publication-data" / "experiments"
OUT_DIR = Path(__file__).parent.parent / "sims-core" / "tests" / "data" / "gpbaa_2d_highs"

# Generation budget per instance size (matches the experiment timeouts so the
# recorded discovery timeline covers every exact-phase ratio up to 80:20).
TIMEOUT_BY_SIZE = {100: 120, 150: 300, 200: 250, 250: 400}

CITIES = ["lagos_nigeria", "mexico_city", "paris", "rio_de_janeiro", "tokyo_bay"]
SIZES = [100, 150, 200, 250]


def out_dir_for(method: str, solver: str = "highs") -> Path:
    """Exact-front dataset dir, keyed by first-phase method and MILP backend.

    method: gpba -> gpbaa_2d_*, aneja -> an_2d_*
    solver: backend suffix (highs, gurobi, coin_cbc, ...). Keeping the backend in
    the directory name stops a Gurobi run from overwriting the HiGHS dataset (and
    vice versa) — they are distinct pseudo-sources for run_hv_experiments.py.
    """
    prefix = "an_2d" if method in ("aneja", "aneja_nair", "an") else "gpbaa_2d"
    name = f"{prefix}_{solver}"
    return Path(__file__).parent.parent / "sims-core" / "tests" / "data" / name


def run_instance(inst: str, size: int, solver: str = "highs",
                 timeout_override: int | None = None, method: str = "gpba") -> int:
    dzn = PUB_DIR / inst / f"{inst}.dzn"
    problem = sims_problem.SimsDiscreteProblem.from_dzn(str(dzn))
    timeout = timeout_override or TIMEOUT_BY_SIZE[size]
    t0 = time.monotonic()
    res = sims_problem.solve_with_milp(
        problem,
        objectives=["min_cost", "cloud_coverage"],
        grid_points=200,
        timeout=timedelta(seconds=timeout),
        solver_name=solver,
        method=method,
    )

    solutions = []
    for idx, s in enumerate(res.final_solutions):
        cost, cloud, inc, resn = s.compute_objectives(problem)
        ts = s.timestamp.total_seconds()  # real GPBA-A discovery time
        solutions.append({
            "selected_images": s.get_selected_images_list(),
            "cost": int(cost),
            "cloudy_area": int(cloud),
            "max_incidence_angle": int(inc),
            "min_resolutions_sum": int(resn),
            "timestamp_s": round(ts, 6),
            "phase": "gpba-a",
            "index": idx,
        })

    out_dir = out_dir_for(method, solver)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{inst}.json").write_text(json.dumps({"solutions": solutions}))
    print(
        f"{inst}: {len(solutions)} solutions "
        f"(ts range {min((x['timestamp_s'] for x in solutions), default=0):.1f}"
        f"–{max((x['timestamp_s'] for x in solutions), default=0):.1f}s) "
        f"in {time.monotonic() - t0:.1f}s",
        flush=True,
    )
    return len(solutions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default=None)
    ap.add_argument("--solver", default="highs",
                    help="MILP backend: highs (default) or coin_cbc. CBC honors "
                         "the timeout on instances where HiGHS presolve hangs.")
    ap.add_argument("--timeout", type=int, default=None,
                    help="Override per-instance budget (s); for very hard instances "
                         "whose ideal-bounds solve needs more than the size default.")
    ap.add_argument("--method", default="gpba", choices=["gpba", "aneja"],
                    help="First-phase exact method: gpba (GPBA-A -> gpbaa_2d_highs) "
                         "or aneja (Anytime Aneja & Nair -> an_2d_highs).")
    args = ap.parse_args()
    for city in CITIES:
        for size in SIZES:
            inst = f"{city}_{size}"
            if args.filter and args.filter not in inst:
                continue
            run_instance(inst, size, args.solver, args.timeout, args.method)


if __name__ == "__main__":
    main()
