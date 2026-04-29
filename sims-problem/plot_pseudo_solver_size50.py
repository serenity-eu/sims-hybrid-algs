#!/usr/bin/env python
"""Plot pseudo-solver HV curves for selected instance sizes.

This script uses only pre-recorded pseudo-solver JSON files. It does not run any
solver. For the requested instance size, it builds a synthetic trace spanning a
shared time horizon for each instance, computes normalised HV curves with shared
bounds, and plots all instances of that size on the same chart.

Usage::

    uv run python sims-problem/plot_pseudo_solver_size50.py

    uv run python sims-problem/plot_pseudo_solver_size50.py --size 30

    uv run python sims-problem/plot_pseudo_solver_size50.py --size 145

    uv run python sims-problem/plot_pseudo_solver_size50.py --size 200 --num-points 50

    uv run python sims-problem/plot_pseudo_solver_size50.py --output-dir hv_experiment_results
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import tarfile
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import sims_problem

OBJECTIVES = [
    "min_cost",
    "cloud_coverage",
    "min_max_incidence_angle",
    "min_resolution",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PSEUDO_SOLUTIONS_DIR = (
    PROJECT_ROOT / "sims-core" / "tests" / "data" / "pseudo_solver_solutions"
)

INSTANCES_BY_SIZE: dict[int, list[str]] = {
    30: [
        "lagos_nigeria_30",
        "mexico_city_30",
        "paris_30",
        "rio_de_janeiro_30",
        "tokyo_bay_30",
    ],
    50: [
        "lagos_nigeria_50",
        "mexico_city_50",
        "paris_50",
        "rio_de_janeiro_50",
        "tokyo_bay_50",
    ],
    100: [
        "lagos_nigeria_100",
        "mexico_city_100",
        "paris_100",
        "rio_de_janeiro_100",
        "tokyo_bay_100",
    ],
    145: [
        "lagos_nigeria_145",
    ],
    150: [
        "lagos_nigeria_145",
        "mexico_city_150",
        "paris_150",
        "rio_de_janeiro_150",
        "tokyo_bay_150",
    ],
    200: [
        "mexico_city_200",
        "paris_200",
        "rio_de_janeiro_200",
        "tokyo_bay_200",
    ],
}


def _load_pseudo_solutions(instance_name: str) -> list[dict]:
    json_path = PSEUDO_SOLUTIONS_DIR / f"{instance_name}.json"
    if not json_path.exists():
        return []

    with open(json_path) as f:
        data = json.load(f)

    return data if isinstance(data, list) else data.get("solutions", [])


def _make_exact_phase_trace(
    pseudo_solutions: list[dict],
    exact_time_s: float,
    objectives: list[str],
) -> bytes | None:
    """Create a synthetic trace from pseudo-solver solutions over [0, exact_time_s]."""
    if not pseudo_solutions:
        return None

    converted: list[sims_problem.Solution] = []
    n = len(pseudo_solutions)

    for idx, sol_dict in enumerate(pseudo_solutions):
        images = sol_dict.get("selected_images", [])
        if not images:
            continue

        if n == 1:
            ts_seconds = exact_time_s
        else:
            ts_seconds = exact_time_s * (idx / (n - 1))

        try:
            sol = sims_problem.Solution.create(
                selected_images=images,
                cost=sol_dict.get("cost"),
                cloudy_area=sol_dict.get("cloudy_area"),
                max_incidence_angle=sol_dict.get("max_incidence_angle"),
                timestamp_us=int(ts_seconds * 1_000_000),
                min_resolutions_sum=sol_dict.get("min_resolutions_sum"),
            )
            converted.append(sol)
        except Exception:
            continue

    if not converted:
        return None

    points: list[list[int]] = []
    for s in converted:
        row: list[int] = []
        for obj in objectives:
            if obj == "min_cost":
                row.append(s.cost or 0)
            elif obj == "cloud_coverage":
                row.append(s.cloudy_area or 0)
            elif obj == "min_max_incidence_angle":
                row.append(s.max_incidence_angle or 0)
            elif obj == "min_resolution":
                row.append(s.min_resolutions_sum or 0)
        points.append(row)

    ndim = len(objectives)
    bounds: list[list[int]] = []
    for j in range(ndim):
        vals = [p[j] for p in points]
        lo, hi = min(vals), max(vals)
        rng = max(hi - lo, 1)
        bounds.append([max(0, lo - 1), hi + int(rng * 0.1) + 1])

    bounds_pairs = [[int(b[0]), int(b[1])] for b in bounds]
    ref_point = [b[1] + 1 for b in bounds_pairs]

    try:
        return sims_problem.generate_trace(
            solutions=converted,
            objectives=objectives,
            algorithm="Exact-Pseudo",
            num_objectives=len(objectives),
            objective_bounds=bounds_pairs,
            reference_point=ref_point,
            include_dominated=False,
        )
    except Exception:
        return None


def extract_all_objectives(trace_bytes: bytes, ndim: int) -> list[list[int]]:
    with tarfile.open(fileobj=io.BytesIO(trace_bytes), mode="r:gz") as tar:
        meta_member = tar.extractfile("metadata.json")
        obj_member = tar.extractfile("objectives.bin")
        if meta_member is None or obj_member is None:
            raise ValueError("Trace archive is missing required members")

        meta = json.loads(meta_member.read())
        n = meta["solution_count"]
        obj_raw = obj_member.read()

        pts: list[list[int]] = []
        for i in range(n):
            row = [
                struct.unpack_from("<Q", obj_raw, (i * ndim + j) * 8)[0]
                for j in range(ndim)
            ]
            pts.append(row)
        return pts


def compute_shared_bounds(all_points: list[list[int]], ndim: int) -> list[list[int]]:
    bounds: list[list[int]] = []
    for i in range(ndim):
        vals = [p[i] for p in all_points]
        lo, hi = min(vals), max(vals)
        rng = max(hi - lo, 1)
        bounds.append([max(0, lo - 1), hi + int(rng * 0.1) + 1])
    return bounds


def _safe_label(name: str) -> str:
    return name.replace("_", " ").title()


def _plot_curves(
    curves: list[tuple[str, list[tuple[float, float]]]],
    max_time_s: int,
    output_path: Path,
    size: int,
) -> None:
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is required to generate plots")

    fig, ax = plt.subplots(figsize=(10, 6))

    endpoint_annotations: list[tuple[float, float, str]] = []
    for idx, (instance_name, curve) in enumerate(curves):
        if not curve:
            continue

        ts = [t for t, hv in curve]
        hvs = [hv for t, hv in curve]
        color = f"C{idx % 10}"

        ax.plot(
            ts,
            hvs,
            label=_safe_label(instance_name),
            color=color,
            linewidth=2.0,
        )
        endpoint_annotations.append((ts[-1], hvs[-1], color))

    sorted_annotations = sorted(
        endpoint_annotations, key=lambda item: item[1], reverse=True
    )
    n_annotations = len(sorted_annotations)
    for idx, (end_t, end_hv, color) in enumerate(sorted_annotations):
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
        f"Pseudo-Solver HV over Time — Size-{size} Instances (4D)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_time_s)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot pseudo-solver HV curves for all instances of a given size",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hv_experiment_results"),
        help="Output directory for plot and JSON summary",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=30,
        help="Sample points per HV curve (default: 30)",
    )
    parser.add_argument(
        "--max-time-s",
        type=int,
        default=60,
        help="Shared synthetic time horizon in seconds (default: 60)",
    )
    parser.add_argument(
        "--size",
        type=int,
        choices=[30, 50, 100, 145, 150, 200],
        default=50,
        help="Instance size to plot (default: 50)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    traces: list[tuple[str, bytes]] = []
    all_points: list[list[int]] = []
    instance_names = INSTANCES_BY_SIZE[args.size]

    for instance_name in instance_names:
        pseudo_solutions = _load_pseudo_solutions(instance_name)
        pseudo_trace = _make_exact_phase_trace(
            pseudo_solutions=pseudo_solutions,
            exact_time_s=float(args.max_time_s),
            objectives=OBJECTIVES,
        )
        if not pseudo_trace:
            print(f"Skipping {instance_name}: no usable pseudo-solver data", flush=True)
            continue

        try:
            all_points.extend(extract_all_objectives(pseudo_trace, len(OBJECTIVES)))
            traces.append((instance_name, pseudo_trace))
        except Exception as e:
            print(
                f"Skipping {instance_name}: could not extract objectives: {e}",
                flush=True,
            )

    if not traces or not all_points:
        print(
            f"No pseudo-solver traces available for size-{args.size} instances.",
            flush=True,
        )
        return 1

    bounds = compute_shared_bounds(all_points, len(OBJECTIVES))

    curves: list[tuple[str, list[tuple[float, float]]]] = []
    summary: list[dict] = []

    for instance_name, pseudo_trace in traces:
        try:
            curve = sims_problem.compute_hv_curve_from_trace(
                pseudo_trace, bounds, args.num_points
            )
            curves.append((instance_name, curve))
            final_hv = curve[-1][1] if curve else 0.0
            summary.append(
                {
                    "instance": instance_name,
                    "num_points": args.num_points,
                    "max_time_s": args.max_time_s,
                    "final_hv": final_hv,
                    "curve": curve,
                }
            )
            print(f"{instance_name:>25s}  final HV={final_hv:.6f}", flush=True)
        except Exception as e:
            print(f"Skipping {instance_name}: HV computation failed: {e}", flush=True)

    if not curves:
        print("No pseudo-solver HV curves could be computed.", flush=True)
        return 1

    plot_path = args.output_dir / f"fig_pseudo_solver_size{args.size}_comparison.png"
    _plot_curves(curves, args.max_time_s, plot_path, args.size)
    print(
        f"\nPseudo-solver size-{args.size} comparison plot: {plot_path}",
        flush=True,
    )

    json_path = args.output_dir / f"pseudo_solver_size{args.size}_curves.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Pseudo-solver size-{args.size} data: {json_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
