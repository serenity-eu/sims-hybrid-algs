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
    uv run python generate_pseudo.py                 # all publication instances (100-250)
    uv run python generate_pseudo.py --instance-set random-clouds --timeout 600
                                                       # all random-clouds instances (100-500)
"""

import argparse
import gzip
import json
import shutil
import time
from datetime import timedelta
from pathlib import Path

import sims_problem

ROOT = Path(__file__).parent.parent


def resolve_dzn(path: Path) -> Path:
    """Return `path`, decompressing `path.gz` into it first if `path` is
    missing. The random-clouds instances are committed gzipped only (raw .dzn
    is gitignored — see .gitignore and commit 188208b): a fresh checkout has
    `{inst}.dzn.gz` but not `{inst}.dzn` until decompressed once.
    """
    if path.exists():
        return path
    gz_path = path.with_name(path.name + ".gz")
    if not gz_path.exists():
        raise FileNotFoundError(f"neither {path} nor {gz_path} exist")
    with gzip.open(gz_path, "rb") as src, open(path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return path


# Two independent instance sets, each with its own .dzn layout:
#   publication    : ROOT/publication-data/experiments/{inst}/{inst}.dzn, sizes 100-250
#   random-clouds  : ROOT/publication-data/satellite-data/instances_random_clouds/{inst}.dzn (flat), sizes 100-500
PUB_DIR = ROOT / "publication-data" / "experiments"
RC_DIR = ROOT / "publication-data" / "satellite-data" / "instances_random_clouds"
HARD_DIR = ROOT / "publication-data" / "satellite-data" / "instances_hard_an"

CITIES = ["lagos_nigeria", "mexico_city", "paris", "rio_de_janeiro", "tokyo_bay"]

INSTANCE_SETS = {
    "publication": {
        "sizes": [100, 150, 200, 250],
        "dzn_path": lambda inst: PUB_DIR / inst / f"{inst}.dzn",
        # Generation budget per instance size (matches the experiment timeouts so
        # the recorded discovery timeline covers every exact-phase ratio up to 80:20).
        "timeout_by_size": {100: 120, 150: 300, 200: 250, 250: 400},
        "out_suffix": "",
    },
    "random-clouds": {
        "sizes": [100, 150, 200, 250, 300, 350, 400, 450, 500],
        "dzn_path": lambda inst: RC_DIR / f"{inst}.dzn",
        # No tuned per-size schedule exists for this set (up to 500 images) —
        # --timeout is required.
        "timeout_by_size": {},
        # Separate output dirs so this set never collides with the publication
        # pseudo-solutions (e.g. gpbaa_2d_gurobi_rc vs gpbaa_2d_gurobi).
        "out_suffix": "_rc",
    },
    "hard": {
        # The A&N-calibrated set (instance_generation_difficulty.md, Part II).
        # Ladders differ per city, so `sizes` is the union and missing files are
        # skipped by the caller.
        "sizes": [125, 150, 175, 200, 225, 250, 275, 300, 350],
        "dzn_path": lambda inst: HARD_DIR / f"{inst}.dzn",
        "timeout_by_size": {},
        "out_suffix": "_hard",
    },
}


def if_prefix(method: str) -> str:
    """Output-directory prefix for a first-phase method."""
    if method in ("aneja", "aneja_nair", "an"):
        return "an_2d"
    if method == "psbox":
        return "psbox_2d"
    if method in ("quadtree", "qt"):
        return "qt_2d"
    return "gpbaa_2d"


def out_dir_for(method: str, solver: str = "highs", instance_set: str = "publication") -> Path:
    """Exact-front dataset dir, keyed by first-phase method, MILP backend, and
    instance set.

    method: gpba -> gpbaa_2d_*, aneja -> an_2d_*
    solver: backend suffix (highs, gurobi, coin_cbc, ...). Keeping the backend in
    the directory name stops a Gurobi run from overwriting the HiGHS dataset (and
    vice versa) — they are distinct pseudo-sources for run_hv_experiments.py.
    instance_set: appends out_suffix (e.g. "_rc" for random-clouds) so the two
    instance sets' pseudo-solutions never collide.
    """
    prefix = if_prefix(method)
    suffix = INSTANCE_SETS[instance_set]["out_suffix"]
    name = f"{prefix}_{solver}{suffix}"
    return ROOT / "sims-core" / "tests" / "data" / name


def run_instance(inst: str, size: int, solver: str = "highs",
                 timeout_override: int | None = None, method: str = "gpba",
                 instance_set: str = "publication") -> int:
    spec = INSTANCE_SETS[instance_set]
    dzn = resolve_dzn(spec["dzn_path"](inst))
    problem = sims_problem.SimsDiscreteProblem.from_dzn(str(dzn))
    timeout = timeout_override or spec["timeout_by_size"].get(size)
    if timeout is None:
        raise SystemExit(
            f"No default timeout for {instance_set} size {size}; pass --timeout explicitly."
        )
    t0 = time.monotonic()
    res = sims_problem.solve_with_milp(
        problem,
        objectives=["min_cost", "cloud_coverage"],
        grid_points=200,
        timeout=timedelta(seconds=timeout),
        solver_name=solver,
        method=method,
    )

    # Stopped here, not after the loop below. The timeout governs the solver;
    # re-evaluating objectives in Python costs about two seconds per solution on
    # these instances, so folding it in made a run look further over budget the
    # more solutions it found -- penalising exactly the methods that found most.
    solve_secs = time.monotonic() - t0
    post = time.monotonic()

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

    out_dir = out_dir_for(method, solver, instance_set)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{inst}.json").write_text(json.dumps({"solutions": solutions}))
    print(
        f"{inst}: {len(solutions)} solutions "
        f"(ts range {min((x['timestamp_s'] for x in solutions), default=0):.1f}"
        f"–{max((x['timestamp_s'] for x in solutions), default=0):.1f}s) "
        f"in {solve_secs:.1f}s solving (+{time.monotonic() - post:.1f}s reporting)",
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
    ap.add_argument("--method", default="gpba",
                    choices=["gpba", "aneja", "psbox", "quadtree"],
                    help="First-phase exact method: gpba (GPBA-A -> gpbaa_2d_highs), "
                         "aneja (Anytime Aneja & Nair -> an_2d_highs), psbox "
                         "(Kirlik & Sayin rectangle search -> psbox_2d_*), or "
                         "quadtree (Quadtree Search Method -> qt_2d_*). psbox "
                         "reports only points whose solve proved optimality, so it "
                         "never emits an unproven incumbent when the budget runs out; "
                         "quadtree asks only whether a region holds a feasible point, "
                         "which is far cheaper, and reports the points no unexplored "
                         "region can dominate.")
    ap.add_argument("--instance-set", default="publication",
                    choices=list(INSTANCE_SETS),
                    help="publication (5 cities x 100-250, default) or "
                         "random-clouds (5 cities x 100-500 step 50). Output "
                         "directories never collide between the two sets.")
    args = ap.parse_args()
    for city in CITIES:
        for size in INSTANCE_SETS[args.instance_set]["sizes"]:
            inst = f"{city}_{size}"
            if args.filter and args.filter not in inst:
                continue
            # Size ladders are per city in the "hard" set, so most (city, size)
            # pairs in the union simply do not exist. Skip rather than abort.
            spec = INSTANCE_SETS[args.instance_set]
            probe = spec["dzn_path"](inst)
            if not (probe.exists() or probe.with_suffix(".dzn.gz").exists()):
                continue
            run_instance(inst, size, args.solver, args.timeout, args.method, args.instance_set)


if __name__ == "__main__":
    main()
