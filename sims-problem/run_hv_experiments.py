#!/usr/bin/env python
"""HV-over-time experiment runner comparing multiple algorithm configurations.

Produces per-instance line-plots (X = time, Y = normalised hypervolume)
with configurable series including:

  - Pure PLS baseline
  - Hybrid 50:50 baseline
  - NSGA-II
  - MOEA/D
  - Improved PLS / Hybrid
  - Diverse-probe PLS / Hybrid
  - Scalarized PLS / Hybrid
  - Diverse+Scalarized PLS / Hybrid

All series share the same reference point / bounds per instance so that
hypervolume values are directly comparable.

Usage::

    # All instances up to 100 images
    uv run python run_hv_experiments.py --max-size 100

    # Single instance
    uv run python run_hv_experiments.py --filter lagos_nigeria_50

    # Only size-50 instances, 50 HV curve points
    uv run python run_hv_experiments.py --max-size 50 --num-points 50
"""

from __future__ import annotations

import argparse
import io
import json
import re
import struct
import sys
import tarfile
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional, TypedDict

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import sims_problem

# ─── Instance registry ────────────────────────────────────────────────

INSTANCES_DIR = Path(__file__).parent / "tests" / "data"

# (display_name, filename, num_images)
ALL_INSTANCES: list[tuple[str, str, int]] = []

_CITIES = [
    ("lagos_nigeria", [30, 50, 100, 145]),
    ("mexico_city", [30, 50, 100, 150, 200]),
    ("paris", [30, 50, 100, 150, 200]),
    ("rio_de_janeiro", [30, 50, 100, 150, 200]),
    ("tokyo_bay", [30, 50, 100, 150, 200]),
]

for _city, _sizes in _CITIES:
    for _size in _sizes:
        _fname = f"{_city}_{_size}.dzn"
        if (INSTANCES_DIR / _fname).exists():
            ALL_INSTANCES.append((f"{_city}_{_size}", _fname, _size))

OBJECTIVES = [
    "min_cost",
    "cloud_coverage",
    "min_max_incidence_angle",
    "min_resolution",
]


# ─── Timeout heuristic ────────────────────────────────────────────────


def timeout_for_size(num_images: int) -> int:
    """PLS timeout in seconds – longer budgets for full HV ablation runs."""
    if num_images <= 30:
        return 300
    if num_images <= 50:
        return 300
    if num_images <= 100:
        return 1200
    if num_images <= 150:
        return 1200
    return 1200


# ─── Algorithm configurations ─────────────────────────────────────────


@dataclass
class AlgorithmConfig:
    """One series on the plot."""

    label: str
    color: str
    linestyle: str
    linewidth: float = 2.0

    def run(
        self,
        problem: Any,
        timeout_s: int,
        seed: int = 42,
    ) -> sims_problem.SolvingResult:
        """Run the algorithm and return a SolvingResult with trace."""
        raise NotImplementedError


class PurePLS(AlgorithmConfig):
    """Baseline PLS – all improvements disabled, no exact phase."""

    def run(self, problem, timeout_s, seed=42):
        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=timeout_s),
            is_deterministic=True,
            trace=True,
            include_dominated=False,
            use_checkpoint=True,
            use_ranked_candidates=False,
            use_greedy_initial_population=False,
            use_perturbation_restart=False,
        )


class HybridBaseline(AlgorithmConfig):
    """Hybrid 50:50 – baseline PLS seeded with pseudo-solver solutions."""

    def run(self, problem, timeout_s, seed=42):
        exact_time = timeout_s // 2
        pls_time = timeout_s - exact_time

        exact_solutions = _get_pseudo_solutions(problem)
        initial_pop = (
            _solutions_to_sims(exact_solutions, problem) if exact_solutions else None
        )

        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=pls_time),
            is_deterministic=True,
            trace=True,
            include_dominated=False,
            initial_population=initial_pop,
            use_checkpoint=False,
            use_ranked_candidates=False,
            use_greedy_initial_population=False,
            use_perturbation_restart=False,
        )


class ImprovedHybrid(AlgorithmConfig):
    """Improved Hybrid 50:50 – all PLS enhancements, seeded with pseudo-solver."""

    def run(self, problem, timeout_s, seed=42):
        exact_time = timeout_s // 2
        pls_time = timeout_s - exact_time

        exact_solutions = _get_pseudo_solutions(problem)
        initial_pop = (
            _solutions_to_sims(exact_solutions, problem) if exact_solutions else None
        )

        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=pls_time),
            is_deterministic=True,
            trace=True,
            include_dominated=False,
            initial_population=initial_pop,
            use_checkpoint=True,
            use_ranked_candidates=True,
            max_k1_candidates=15,
            use_greedy_initial_population=True,
            use_perturbation_restart=True,
        )


class ImprovedPurePLS(AlgorithmConfig):
    """Improved Pure PLS – all PLS enhancements enabled, no exact phase."""

    def run(self, problem, timeout_s, seed=42):
        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=timeout_s),
            is_deterministic=True,
            trace=True,
            include_dominated=False,
            use_checkpoint=True,
            use_ranked_candidates=True,
            max_k1_candidates=15,
            use_greedy_initial_population=True,
            use_perturbation_restart=True,
        )


class DiverseProbePLS(AlgorithmConfig):
    """Improved PLS with diverse probing – farthest-point subset selection."""

    def run(self, problem, timeout_s, seed=42):
        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=timeout_s),
            is_deterministic=True,
            trace=True,
            include_dominated=False,
            use_checkpoint=True,
            use_ranked_candidates=True,
            max_k1_candidates=15,
            use_greedy_initial_population=True,
            use_perturbation_restart=True,
            use_diverse_probing=True,
        )


class DiverseProbeHybrid(AlgorithmConfig):
    """Improved Hybrid with diverse probing – farthest-point subset selection."""

    def run(self, problem, timeout_s, seed=42):
        exact_time = timeout_s // 2
        pls_time = timeout_s - exact_time

        exact_solutions = _get_pseudo_solutions(problem)
        initial_pop = (
            _solutions_to_sims(exact_solutions, problem) if exact_solutions else None
        )

        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=pls_time),
            is_deterministic=True,
            trace=True,
            include_dominated=False,
            initial_population=initial_pop,
            use_checkpoint=True,
            use_ranked_candidates=True,
            max_k1_candidates=15,
            use_greedy_initial_population=True,
            use_perturbation_restart=True,
            use_diverse_probing=True,
        )


class ScalarizedPLS(AlgorithmConfig):
    """Improved PLS with scalarized parent selection."""

    scalarized_selection_source: str = "archive"
    scalarized_parent_budget: int = 4
    scalarized_weight_samples: int = 4
    scalarized_rho: float = 1e-3
    use_nd_tree_scalarized_query: bool = True

    def run(self, problem, timeout_s, seed=42):
        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=timeout_s),
            is_deterministic=True,
            trace=True,
            include_dominated=False,
            use_checkpoint=True,
            use_ranked_candidates=False,
            use_greedy_initial_population=False,
            use_perturbation_restart=False,
            solution_selection_mode="scalarized-chebycheff",
            scalarized_selection_source=self.scalarized_selection_source,
            scalarized_parent_budget=self.scalarized_parent_budget,
            scalarized_weight_samples=self.scalarized_weight_samples,
            scalarized_rho=self.scalarized_rho,
            use_nd_tree_scalarized_query=self.use_nd_tree_scalarized_query,
        )


class ScalarizedHybrid(AlgorithmConfig):
    """Improved Hybrid with scalarized parent selection."""

    scalarized_selection_source: str = "archive"
    scalarized_parent_budget: int = 4
    scalarized_weight_samples: int = 4
    scalarized_rho: float = 1e-3
    use_nd_tree_scalarized_query: bool = True

    def run(self, problem, timeout_s, seed=42):
        exact_time = timeout_s // 2
        pls_time = timeout_s - exact_time

        exact_solutions = _get_pseudo_solutions(problem)
        initial_pop = (
            _solutions_to_sims(exact_solutions, problem) if exact_solutions else None
        )

        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=pls_time),
            is_deterministic=True,
            trace=True,
            include_dominated=False,
            initial_population=initial_pop,
            use_checkpoint=True,
            use_ranked_candidates=True,
            max_k1_candidates=15,
            use_greedy_initial_population=True,
            use_perturbation_restart=True,
            solution_selection_mode="scalarized-chebycheff",
            scalarized_selection_source=self.scalarized_selection_source,
            scalarized_parent_budget=self.scalarized_parent_budget,
            scalarized_weight_samples=self.scalarized_weight_samples,
            scalarized_rho=self.scalarized_rho,
            use_nd_tree_scalarized_query=self.use_nd_tree_scalarized_query,
        )


class DiverseScalarizedPLS(AlgorithmConfig):
    """Improved PLS with diverse prefiltering and scalarized parent selection."""

    diverse_probe_budget: int = 8
    scalarized_selection_source: str = "archive"
    scalarized_parent_budget: int = 4
    scalarized_weight_samples: int = 4
    scalarized_rho: float = 1e-3
    use_nd_tree_scalarized_query: bool = True

    def run(self, problem, timeout_s, seed=42):
        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=timeout_s),
            is_deterministic=True,
            trace=True,
            include_dominated=False,
            use_checkpoint=True,
            use_ranked_candidates=True,
            max_k1_candidates=15,
            use_greedy_initial_population=True,
            use_perturbation_restart=True,
            solution_selection_mode="diverse-then-scalarized-chebycheff",
            diverse_probe_budget=self.diverse_probe_budget,
            scalarized_selection_source=self.scalarized_selection_source,
            scalarized_parent_budget=self.scalarized_parent_budget,
            scalarized_weight_samples=self.scalarized_weight_samples,
            scalarized_rho=self.scalarized_rho,
            use_nd_tree_scalarized_query=self.use_nd_tree_scalarized_query,
        )


class DiverseScalarizedHybrid(AlgorithmConfig):
    """Improved Hybrid with diverse prefiltering and scalarized parent selection."""

    diverse_probe_budget: int = 8
    scalarized_selection_source: str = "archive"
    scalarized_parent_budget: int = 4
    scalarized_weight_samples: int = 4
    scalarized_rho: float = 1e-3
    use_nd_tree_scalarized_query: bool = True

    def run(self, problem, timeout_s, seed=42):
        exact_time = timeout_s // 2
        pls_time = timeout_s - exact_time

        exact_solutions = _get_pseudo_solutions(problem)
        initial_pop = (
            _solutions_to_sims(exact_solutions, problem) if exact_solutions else None
        )

        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=pls_time),
            is_deterministic=True,
            trace=True,
            include_dominated=False,
            initial_population=initial_pop,
            use_checkpoint=True,
            use_ranked_candidates=True,
            max_k1_candidates=15,
            use_greedy_initial_population=True,
            use_perturbation_restart=True,
            solution_selection_mode="diverse-then-scalarized-chebycheff",
            diverse_probe_budget=self.diverse_probe_budget,
            scalarized_selection_source=self.scalarized_selection_source,
            scalarized_parent_budget=self.scalarized_parent_budget,
            scalarized_weight_samples=self.scalarized_weight_samples,
            scalarized_rho=self.scalarized_rho,
            use_nd_tree_scalarized_query=self.use_nd_tree_scalarized_query,
        )


class NSGA2Config(AlgorithmConfig):
    """Custom NSGA-II evolutionary algorithm."""

    def run(self, problem, timeout_s, seed=42):
        return sims_problem.solve_with_nsga2(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=timeout_s),
            population_size=200,
            max_generations=500_000,
            seed=seed,
            trace=True,
            include_dominated=False,
            crossover_rate=0.95,
            swap_mutation_rate=0.6,
            add_prune_mutation_rate=0.45,
            bitflip_mutation_rate=0.0,
            multi_swap_max_removals=4,
            multi_swap_rate=0.35,
            shift_mutation_rate=0.4,
            coverage_biased_crossover_fraction=0.7,
            ensure_mutation=True,
            stagnation_limit=10,
        )


class MOEADConfig(AlgorithmConfig):
    """MOEA/D decomposition-based evolutionary algorithm."""

    def run(self, problem, timeout_s, seed=42):
        return sims_problem.solve_with_moead(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=timeout_s),
            population_size=300,
            max_generations=500_000,
            seed=seed,
            trace=True,
            include_dominated=False,
            neighbourhood_size=30,
            delta=0.7,
            max_replacements=8,
            crossover_rate=1.0,
            swap_mutation_rate=0.5,
            add_prune_mutation_rate=0.35,
            multi_swap_max_removals=4,
            multi_swap_rate=0.3,
            shift_mutation_rate=0.3,
            coverage_biased_crossover_fraction=0.7,
            ensure_mutation=True,
            auto_divisions=True,
            use_pbi=True,
            pbi_theta=3.0,
            stagnation_limit=15,
        )


# The series in plot order.
CONFIGS: list[AlgorithmConfig] = [
    PurePLS(
        label="Pure PLS",
        color="#7f7f7f",
        linestyle="-",
        linewidth=1.5,
    ),
    HybridBaseline(
        label="Hybrid 50:50",
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
    ),
    NSGA2Config(
        label="NSGA-II",
        color="#ff7f0e",
        linestyle="-.",
    ),
    MOEADConfig(
        label="MOEA/D",
        color="#9467bd",
        linestyle=":",
    ),
    ImprovedPurePLS(
        label="Improved PLS",
        color="#d62728",
        linestyle="-",
        linewidth=1.5,
    ),
    ImprovedHybrid(
        label="Improved Hybrid",
        color="#2ca02c",
        linestyle="-",
        linewidth=2.5,
    ),
    DiverseProbePLS(
        label="Diverse Probe PLS",
        color="#e377c2",
        linestyle="-",
        linewidth=1.5,
    ),
    DiverseProbeHybrid(
        label="Diverse Probe Hybrid",
        color="#17becf",
        linestyle="-",
        linewidth=2.5,
    ),
    ScalarizedPLS(
        label="Scalarized PLS",
        color="#8c564b",
        linestyle="-",
        linewidth=1.5,
    ),
    ScalarizedHybrid(
        label="Scalarized Hybrid",
        color="#bcbd22",
        linestyle="-",
        linewidth=2.5,
    ),
    DiverseScalarizedPLS(
        label="Diverse+Scalarized PLS",
        color="#ff9896",
        linestyle="-",
        linewidth=1.5,
    ),
    DiverseScalarizedHybrid(
        label="Diverse+Scalarized Hybrid",
        color="#98df8a",
        linestyle="-",
        linewidth=2.5,
    ),
]


# ─── Pseudo-solver helpers ────────────────────────────────────────────

_PSEUDO_SOLUTIONS_DIR = (
    Path(__file__).parent.parent
    / "sims-core"
    / "tests"
    / "data"
    / "pseudo_solver_solutions"
)

_pseudo_cache: dict[str, list[dict]] = {}


def _solution_to_objective_dict(
    sol: sims_problem.Solution, problem: sims_problem.SimsDiscreteProblem
) -> dict[str, object]:
    cost, cloudy_area, max_incidence_angle, min_resolutions_sum = (
        sol.compute_objectives(problem)
    )
    return {
        "selected_images": sorted(sol.get_selected_images_list()),
        "cost": cost,
        "cloudy_area": cloudy_area,
        "max_incidence_angle": max_incidence_angle,
        "min_resolutions_sum": min_resolutions_sum,
        "timestamp_us": int(sol.timestamp.total_seconds() * 1_000_000),
    }


def _load_pseudo_solutions(instance_name: str) -> list[dict]:
    """Load pre-recorded exact solver solutions from JSON."""
    if instance_name in _pseudo_cache:
        return _pseudo_cache[instance_name]

    json_path = _PSEUDO_SOLUTIONS_DIR / f"{instance_name}.json"
    if not json_path.exists():
        _pseudo_cache[instance_name] = []
        return []

    with open(json_path) as f:
        data = json.load(f)

    solutions = data if isinstance(data, list) else data.get("solutions", [])
    _pseudo_cache[instance_name] = solutions
    return solutions


def _get_pseudo_solutions(problem: sims_problem.SimsDiscreteProblem) -> list[dict]:
    """Get pseudo-solver solutions for a problem instance."""
    # Try to determine instance name from the problem
    # The problem doesn't expose its name, so we try all cached names
    # This is called after we've loaded the instance by name
    return _current_pseudo_solutions


_current_pseudo_solutions: list[dict] = []


class _RebuiltTraceEntry(TypedDict):
    old_index: int
    objectives: tuple[int, ...]
    timestamp_us: int
    is_pseudo_match: bool


def _solutions_to_sims(
    solutions: list[dict],
    problem: sims_problem.SimsDiscreteProblem,
) -> list[sims_problem.Solution]:
    """Convert pseudo-solver solution dicts to sims_problem.Solution objects."""
    result = []
    for sol_dict in solutions:
        images = sol_dict.get("selected_images", [])
        if not images:
            continue
        try:
            sol = sims_problem.Solution.create(
                selected_images=images,
                cost=sol_dict.get("cost"),
                cloudy_area=sol_dict.get("cloudy_area"),
                max_incidence_angle=sol_dict.get("max_incidence_angle"),
                timestamp_us=0,
                min_resolutions_sum=sol_dict.get("min_resolutions_sum"),
            )
            result.append(sol)
        except Exception:
            pass
    return result


def _patch_hybrid_trace_timestamps(
    pls_trace_bytes: bytes,
    pseudo_solutions: list[dict],
    exact_time_s: float,
    objectives: list[str],
    allowed_pseudo_objectives: set[tuple[int, ...]] | None = None,
) -> bytes:
    """Patch a PLS trace for hybrid display.

    Rebuild the trace by:
      1. filtering pseudo-solver matches to an allowed shared set,
      2. patching only one occurrence per allowed pseudo objective vector,
      3. shifting all other timestamps by exact_time_s,
      4. sorting the whole trace by timestamp,
      5. recomputing domination links for the full reordered trace,
      6. clearing stale hypervolume data.

    This keeps Hybrid and Improved Hybrid aligned on the same pseudo-solver
    phase-1 solutions and avoids stale domination / hypervolume artifacts after
    reordering.
    """
    import gzip

    def _dominates(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> bool:
        return all(a <= b for a, b in zip(lhs, rhs)) and any(
            a < b for a, b in zip(lhs, rhs)
        )

    # 1. Build lookup: objective-vector -> earliest real pseudo-solver timestamp (us)
    obj_to_pseudo_ts: dict[tuple[int, ...], int] = {}
    for sol_dict in pseudo_solutions:
        obj_key = tuple(
            int(sol_dict.get(k, 0))
            for k in (
                "cost",
                "cloudy_area",
                "max_incidence_angle",
                "min_resolutions_sum",
            )
        )
        if (
            allowed_pseudo_objectives is not None
            and obj_key not in allowed_pseudo_objectives
        ):
            continue
        ts_us = int(sol_dict.get("timestamp_s", 0.0) * 1_000_000)
        if obj_key not in obj_to_pseudo_ts or ts_us < obj_to_pseudo_ts[obj_key]:
            obj_to_pseudo_ts[obj_key] = ts_us

    # 2. Extract trace archive
    with tarfile.open(fileobj=io.BytesIO(pls_trace_bytes), mode="r:gz") as tar:
        meta_member = tar.extractfile("metadata.json")
        obj_member = tar.extractfile("objectives.bin")
        ts_member = tar.extractfile("timestamp.bin")
        if meta_member is None or obj_member is None or ts_member is None:
            raise ValueError("Hybrid trace archive is missing required members")
        meta = json.loads(meta_member.read())
        n = meta["solution_count"]
        ndim = len(meta["objectives"])

        obj_raw = obj_member.read()
        ts_raw = ts_member.read()

    offset_us = int(exact_time_s * 1_000_000)

    # 3. Decode, filter/patch, and shift all entries
    entries: list[_RebuiltTraceEntry] = []
    patched = 0
    seen_patched_objectives: set[tuple[int, ...]] = set()

    for i in range(n):
        obj = tuple(
            struct.unpack_from("<Q", obj_raw, (i * ndim + j) * 8)[0]
            for j in range(ndim)
        )
        old_ts = struct.unpack_from("<I", ts_raw, i * 4)[0]
        new_ts = old_ts + offset_us
        is_pseudo_match = False

        pseudo_ts = obj_to_pseudo_ts.get(obj)
        if pseudo_ts is not None and obj not in seen_patched_objectives:
            new_ts = pseudo_ts
            is_pseudo_match = True
            seen_patched_objectives.add(obj)
            patched += 1

        entries.append(
            _RebuiltTraceEntry(
                old_index=i,
                objectives=obj,
                timestamp_us=new_ts,
                is_pseudo_match=is_pseudo_match,
            )
        )

    # 4. Sort the whole trace by patched timestamp, preserving original order on ties
    entries.sort(key=lambda entry: (entry["timestamp_us"], entry["old_index"]))

    # 5. Rebuild objectives/timestamps and recompute domination for the full trace
    obj_row_size = ndim * 8
    new_obj_data = bytearray(n * obj_row_size)
    new_ts_data = bytearray(n * 4)
    new_dom_data = bytearray(n * 4)

    sorted_objectives: list[tuple[int, ...]] = []
    for new_idx, entry in enumerate(entries):
        obj = entry["objectives"]
        ts = entry["timestamp_us"]
        sorted_objectives.append(obj)

        dst = new_idx * obj_row_size
        for j, value in enumerate(obj):
            struct.pack_into("<Q", new_obj_data, dst + j * 8, value)
        struct.pack_into("<I", new_ts_data, new_idx * 4, ts & 0xFFFFFFFF)

    for i in range(n):
        dominator = 0xFFFFFFFF
        for j in range(i + 1, n):
            if _dominates(sorted_objectives[j], sorted_objectives[i]):
                dominator = j
                break
        struct.pack_into("<I", new_dom_data, i * 4, dominator)

    # 6. Update metadata and clear stale hypervolume data
    meta["total_duration"] = meta["total_duration"] + offset_us
    meta_bytes = json.dumps(meta).encode("utf-8")
    hv_data = b""

    print(
        f"    patched {patched} pseudo-solver solutions as phase-1, "
        f"shifted {n - patched} other solutions by {exact_time_s:.0f}s",
        flush=True,
    )

    # 7. Rebuild tar.gz
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as out_tar:
        for name, data in [
            ("objectives.bin", bytes(new_obj_data)),
            ("dominated.bin", bytes(new_dom_data)),
            ("timestamp.bin", bytes(new_ts_data)),
            ("hypervolume.bin", hv_data),
            ("metadata.json", meta_bytes),
        ]:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            out_tar.addfile(info, io.BytesIO(data))

    return gzip.compress(buf.getvalue())


def _make_exact_phase_trace(
    pseudo_solutions: list[dict],
    exact_time_s: float,
    objectives: list[str],
) -> bytes | None:
    """Create a synthetic exact-phase trace from pseudo-solver solutions.

    The synthetic trace spans [0, exact_time_s] and is used to prepend exact
    solutions before the PLS trace for hybrid series.
    """
    if not pseudo_solutions:
        return None

    # Convert pseudo-solver dicts to Solution objects with synthetic timestamps.
    converted: list[sims_problem.Solution] = []
    n = len(pseudo_solutions)
    for idx, sol_dict in enumerate(pseudo_solutions):
        images = sol_dict.get("selected_images", [])
        if not images:
            continue

        # Uniformly spread exact solutions in [0, exact_time_s].
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
            # Skip malformed pseudo-solver rows.
            continue

    if not converted:
        return None

    # Build broad bounds from the provided exact solutions.
    points = []
    for s in converted:
        row = []
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


# ─── Trace extraction helpers ─────────────────────────────────────────


def extract_all_objectives(trace_bytes: bytes, ndim: int) -> list[list[int]]:
    """Extract raw objective points from a trace archive."""
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


def _extract_objective_set(trace_bytes: bytes, ndim: int) -> set[tuple[int, ...]]:
    """Extract unique objective vectors from a trace archive."""
    return {tuple(row) for row in extract_all_objectives(trace_bytes, ndim)}


def _extract_shared_hybrid_pseudo_objectives(
    traces: dict[str, bytes],
    pseudo_solutions: list[dict],
) -> set[tuple[int, ...]]:
    """Return pseudo objective vectors present in both hybrid traces."""
    hybrid = traces.get("Hybrid 50:50")
    improved = traces.get("Improved Hybrid")
    if not hybrid or not improved:
        return set()

    pseudo_objectives = {
        tuple(
            int(sol.get(k, 0))
            for k in (
                "cost",
                "cloudy_area",
                "max_incidence_angle",
                "min_resolutions_sum",
            )
        )
        for sol in pseudo_solutions
    }

    hybrid_objectives = _extract_objective_set(hybrid, len(OBJECTIVES))
    improved_objectives = _extract_objective_set(improved, len(OBJECTIVES))
    return pseudo_objectives & hybrid_objectives & improved_objectives


def compute_shared_bounds(all_points: list[list[int]], ndim: int) -> list[list[int]]:
    """Compute [min, max] bounds per objective with margin."""
    bounds: list[list[int]] = []
    for i in range(ndim):
        vals = [p[i] for p in all_points]
        lo, hi = min(vals), max(vals)
        rng = max(hi - lo, 1)
        bounds.append([max(0, lo - 1), hi + int(rng * 0.1) + 1])
    return bounds


def _extract_front_snapshots(
    trace_bytes: bytes, num_points: int
) -> list[tuple[float, list[tuple[int, ...]]]]:
    """Reconstruct Pareto-front snapshots exactly like compute_hv_curve_from_trace."""
    with tarfile.open(fileobj=io.BytesIO(trace_bytes), mode="r:gz") as tar:
        meta_member = tar.extractfile("metadata.json")
        obj_member = tar.extractfile("objectives.bin")
        dom_member = tar.extractfile("dominated.bin")
        ts_member = tar.extractfile("timestamp.bin")
        if (
            meta_member is None
            or obj_member is None
            or dom_member is None
            or ts_member is None
        ):
            raise ValueError("Trace archive is missing required members")
        meta = json.loads(meta_member.read())
        n = meta["solution_count"]
        ndim = len(meta["objectives"])
        obj_raw = obj_member.read()
        dom_raw = dom_member.read()
        ts_raw = ts_member.read()

    timestamps_us = [struct.unpack_from("<I", ts_raw, i * 4)[0] for i in range(n)]
    objectives = [
        tuple(
            struct.unpack_from("<Q", obj_raw, (i * ndim + j) * 8)[0]
            for j in range(ndim)
        )
        for i in range(n)
    ]
    dominated = [struct.unpack_from("<I", dom_raw, i * 4)[0] for i in range(n)]

    rev_dom: list[list[int]] = [[] for _ in range(n)]
    for i, d in enumerate(dominated):
        if d != 0xFFFFFFFF and d < n:
            rev_dom[d].append(i)

    total_duration_us = meta["total_duration"]
    interval_us = (
        total_duration_us if num_points <= 1 else total_duration_us // (num_points - 1)
    )
    sample_times_us = [
        total_duration_us if i == num_points - 1 else i * interval_us
        for i in range(num_points)
    ]

    in_front = [False] * n
    trace_cursor = 0
    snapshots: list[tuple[float, list[tuple[int, ...]]]] = []
    for sample_us in sample_times_us:
        while trace_cursor < n and timestamps_us[trace_cursor] <= sample_us:
            idx = trace_cursor
            in_front[idx] = True
            for victim in rev_dom[idx]:
                in_front[victim] = False
            trace_cursor += 1
        front = sorted(objectives[i] for i in range(trace_cursor) if in_front[i])
        snapshots.append((sample_us / 1_000_000.0, front))
    return snapshots


def _assert_hybrid_phase1_alignment(
    traces: dict[str, bytes],
    num_points: int,
    exact_time_s: float,
) -> None:
    """Assert Hybrid and Improved Hybrid have identical phase-1 fronts and HVs."""
    hybrid = traces.get("Hybrid 50:50")
    improved = traces.get("Improved Hybrid")
    if not hybrid or not improved:
        return

    hybrid_snaps = _extract_front_snapshots(hybrid, num_points)
    improved_snaps = _extract_front_snapshots(improved, num_points)

    hybrid_points = extract_all_objectives(hybrid, len(OBJECTIVES))
    improved_points = extract_all_objectives(improved, len(OBJECTIVES))
    bounds = compute_shared_bounds(hybrid_points + improved_points, len(OBJECTIVES))
    hybrid_curve = sims_problem.compute_hv_curve_from_trace(hybrid, bounds, num_points)
    improved_curve = sims_problem.compute_hv_curve_from_trace(
        improved, bounds, num_points
    )

    for idx, (((t_h, front_h), (t_i, front_i)), ((_, hv_h), (_, hv_i))) in enumerate(
        zip(zip(hybrid_snaps, improved_snaps), zip(hybrid_curve, improved_curve))
    ):
        if t_h >= exact_time_s:
            break
        assert abs(t_h - t_i) < 1e-9, (
            f"Hybrid timestamp mismatch at sample {idx}: {t_h} vs {t_i}"
        )
        if front_h != front_i or abs(hv_h - hv_i) >= 1e-9:
            raise AssertionError(
                f"Hybrid phase-1 mismatch at sample {idx} (t={t_h:.4f}s). "
                f"Hybrid front ({len(front_h)} pts): {front_h} | "
                f"Improved front ({len(front_i)} pts): {front_i} | "
                f"HVs: {hv_h} vs {hv_i}"
            )


# ─── Hybrid trace adjustment ─────────────────────────────────────────


def _make_hybrid_trace_with_offset(
    trace_bytes: bytes,
    exact_time_s: float,
) -> bytes:
    """Shift all timestamps in a trace by exact_time_s to account for exact phase."""
    with tarfile.open(fileobj=io.BytesIO(trace_bytes), mode="r:gz") as tar:
        meta_member = tar.extractfile("metadata.json")
        obj_member = tar.extractfile("objectives.bin")
        dom_member = tar.extractfile("dominated.bin")
        ts_member = tar.extractfile("timestamp.bin")
        if (
            meta_member is None
            or obj_member is None
            or dom_member is None
            or ts_member is None
        ):
            raise ValueError("Trace archive is missing required members")
        meta = json.loads(meta_member.read())
        n = meta["solution_count"]

        obj_data = obj_member.read()
        dom_data = dom_member.read()
        ts_data = ts_member.read()

        # hypervolume.bin may be empty (skipped for large traces)
        hv_member = tar.extractfile("hypervolume.bin")
        hv_data = hv_member.read() if hv_member else b""

    # Shift timestamps
    offset_us = int(exact_time_s * 1_000_000)
    new_ts = bytearray(len(ts_data))
    for i in range(n):
        off = i * 4
        old_val = struct.unpack_from("<I", ts_data, off)[0]
        new_val = old_val + offset_us
        struct.pack_into("<I", new_ts, off, new_val & 0xFFFFFFFF)

    # Update metadata
    meta["total_duration"] = meta["total_duration"] + offset_us
    meta_bytes = json.dumps(meta).encode("utf-8")

    # Rebuild tar.gz
    import gzip

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as out_tar:
        for name, data in [
            ("objectives.bin", obj_data),
            ("dominated.bin", dom_data),
            ("timestamp.bin", bytes(new_ts)),
            ("hypervolume.bin", hv_data),
            ("metadata.json", meta_bytes),
        ]:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            out_tar.addfile(info, io.BytesIO(data))

    raw_tar = buf.getvalue()
    return gzip.compress(raw_tar)


def _prepend_exact_trace_to_hybrid(
    pls_trace_bytes: bytes,
    pseudo_solutions: list[dict],
    exact_time_s: float,
    objectives: list[str],
) -> bytes:
    """Prepend synthetic exact-phase trace before shifted PLS trace."""
    shifted_pls = _make_hybrid_trace_with_offset(pls_trace_bytes, exact_time_s)
    exact_trace = _make_exact_phase_trace(pseudo_solutions, exact_time_s, objectives)

    if exact_trace is None:
        # Fallback: at least keep shifted PLS aligned to global timeline.
        return shifted_pls

    try:
        ndim = len(objectives)
        all_points: list[list[int]] = []

        try:
            all_points.extend(extract_all_objectives(exact_trace, ndim))
        except Exception:
            pass

        try:
            all_points.extend(extract_all_objectives(shifted_pls, ndim))
        except Exception:
            pass

        if all_points:
            shared_bounds = compute_shared_bounds(all_points, ndim)
        else:
            # Conservative fallback bounds if extraction fails.
            shared_bounds = [[0, 1] for _ in range(ndim)]

        return sims_problem.merge_traces(
            first_trace=exact_trace,
            second_trace=shifted_pls,
            combined_algorithm="Hybrid-Exact+PLS",
            objective_bounds=shared_bounds,
            reference_point=[b[1] + 1 for b in shared_bounds],
        )
    except Exception:
        # Fallback to shifted PLS if merge fails for any reason.
        return shifted_pls


# ─── Main experiment loop ─────────────────────────────────────────────


def run_instance(
    display_name: str,
    filename: str,
    num_images: int,
    output_dir: Path,
    num_points: int,
    configs: list[AlgorithmConfig],
) -> dict:
    """Run all algorithm configs for one instance, compute HV curves, plot."""

    global _current_pseudo_solutions

    instance_path = INSTANCES_DIR / filename
    problem = sims_problem.SimsDiscreteProblem.from_dzn(str(instance_path))
    total_timeout = timeout_for_size(num_images)

    # Load pseudo-solver solutions for hybrid configs
    _current_pseudo_solutions = _load_pseudo_solutions(display_name)
    n_pseudo = len(_current_pseudo_solutions)

    print(f"\n{'=' * 72}", flush=True)
    print(
        f"  {display_name}  ({num_images} images, timeout={total_timeout}s, "
        f"{len(configs)} configs, {n_pseudo} pseudo-solver solutions)",
        flush=True,
    )
    print(f"{'=' * 72}", flush=True)

    # ── Phase 1: run each algorithm and collect traces ──────────────────

    traces: dict[str, bytes] = {}
    run_meta: dict[str, dict] = {}

    for cfg in configs:
        print(f"\n  [{cfg.label}] running for {total_timeout}s …", flush=True)
        t0 = time.time()
        try:
            result = cfg.run(problem, total_timeout)
            wall = time.time() - t0

            trace_data = result.trace
            traces[cfg.label] = trace_data

            # Final solution validation disabled for HV experiments
            n_final = len(result.final_solutions)
            invalid_count = 0

            run_meta[cfg.label] = dict(
                final_solutions=n_final,
                wall_seconds=round(wall, 1),
                trace_bytes=len(trace_data) if trace_data else 0,
                invalid_solutions=invalid_count,
            )
            print(
                f"  [{cfg.label}] {n_final} final sols, "
                f"{wall:.0f}s wall, trace={len(trace_data) // 1024 if trace_data else 0}KB",
                flush=True,
            )
        except Exception as e:
            wall = time.time() - t0
            print(f"  [{cfg.label}] ERROR after {wall:.0f}s: {e}", flush=True)
            import traceback

            traceback.print_exc()
            run_meta[cfg.label] = dict(
                final_solutions=0,
                wall_seconds=round(wall, 1),
                error=str(e),
            )

    # ── Phase 2: patch hybrid traces, then compute shared bounds ───────

    hybrid_labels = {
        "Hybrid 50:50",
        "Improved Hybrid",
        "Diverse Probe Hybrid",
        "Scalarized Hybrid",
        "Diverse+Scalarized Hybrid",
    }
    if {"Hybrid 50:50", "Improved Hybrid"}.issubset(traces):
        shared_pseudo_objectives = _extract_shared_hybrid_pseudo_objectives(
            traces, _current_pseudo_solutions
        )
        if shared_pseudo_objectives:
            exact_time = total_timeout // 2
            for label in (
                "Hybrid 50:50",
                "Improved Hybrid",
                "Diverse Probe Hybrid",
                "Scalarized Hybrid",
                "Diverse+Scalarized Hybrid",
            ):
                trace_data = traces.get(label)
                if not trace_data:
                    continue
                traces[label] = _patch_hybrid_trace_timestamps(
                    pls_trace_bytes=trace_data,
                    pseudo_solutions=_current_pseudo_solutions,
                    exact_time_s=exact_time,
                    objectives=OBJECTIVES,
                    allowed_pseudo_objectives=shared_pseudo_objectives,
                )
        else:
            print(
                "  WARNING: no shared pseudo-solver objectives found across hybrid traces",
                flush=True,
            )

    print(f"\n  Computing shared bounds …", flush=True)
    ndim = len(OBJECTIVES)
    all_points: list[list[int]] = []
    for label, trace_data in traces.items():
        if trace_data:
            try:
                all_points.extend(extract_all_objectives(trace_data, ndim))
            except Exception as e:
                print(
                    f"  WARNING: could not extract objectives from {label}: {e}",
                    flush=True,
                )

    if not all_points:
        print("  ERROR: no trace data available for any config", flush=True)
        return {"instance": display_name, "error": "no trace data"}

    bounds = compute_shared_bounds(all_points, ndim)
    print(f"  {len(all_points)} total trace points", flush=True)

    # Assert baseline and improved hybrid phase-1 alignment before plotting/HV reporting
    _assert_hybrid_phase1_alignment(traces, num_points, total_timeout // 2)

    # ── Phase 3: compute HV curves ──────────────────────────────────────

    curves: dict[str, list[tuple[float, float]]] = {}
    for cfg in configs:
        label = cfg.label
        if label not in traces or not traces[label]:
            curves[label] = []
            continue

        print(f"\n  [{label}] computing HV curve ({num_points} points) …", flush=True)
        t0 = time.time()
        try:
            curve = sims_problem.compute_hv_curve_from_trace(
                traces[label], bounds, num_points
            )
            curves[label] = curve
            elapsed = time.time() - t0

            # Log raw curve data
            if curve:
                initial_hv = curve[0][1]
                final_hv = curve[-1][1]
                print(
                    f"  [{label}] RAW curve: {len(curve)} points, "
                    f"initial_HV={initial_hv:.8f}, final_HV={final_hv:.8f}, "
                    f"improvement={final_hv - initial_hv:.8f}",
                    flush=True,
                )
            print(f"  [{label}] done in {elapsed:.1f}s", flush=True)
        except Exception as e:
            print(f"  [{label}] HV curve failed: {e}", flush=True)
            curves[label] = []

    # Normalize curves to start at HV=0 for PLS algorithms with high initial HV
    print(f"\n  {'=' * 60}", flush=True)
    print(f"  NORMALIZATION STEP", flush=True)
    print(f"  {'=' * 60}", flush=True)
    for label, curve in curves.items():
        if curve and len(curve) > 0:
            first_t, first_hv = curve[0]
            print(
                f"  [{label}] checking: first_hv={first_hv:.8f}, threshold=0.05",
                flush=True,
            )
            # If curve starts with high HV (> 0.05), prepend (0, 0) point
            if first_hv > 0.05:
                curves[label] = [(0.0, 0.0)] + curve
                print(
                    f"  [{label}] ✓ NORMALIZED: prepended (0, 0) - was starting at HV={first_hv:.8f}",
                    flush=True,
                )
            else:
                print(f"  [{label}] ✗ NOT normalized (below threshold)", flush=True)

    # Normalize curves to start at HV=0 for PLS algorithms with high initial HV
    print(f"\n  Normalizing curves to start at HV=0 …", flush=True)
    for label, curve in curves.items():
        if curve and len(curve) > 0:
            first_t, first_hv = curve[0]
            # If curve starts with high HV (> 0.05), prepend (0, 0) point
            if first_hv > 0.05:
                curves[label] = [(0.0, 0.0)] + curve
                print(
                    f"  [{label}] prepended (0, 0) - was starting at HV={first_hv:.4f}",
                    flush=True,
                )

    # ── Phase 4: generate plot ──────────────────────────────────────────

    plot_path = None
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(10, 6))

        endpoint_annotations: list[tuple[float, float, AlgorithmConfig]] = []
        for cfg in configs:
            curve = curves.get(cfg.label, [])
            if not curve:
                continue
            ts = [t for t, hv in curve]
            hvs = [hv for t, hv in curve]
            # Use different markers for each config for better distinguishability
            marker_styles = {
                "Pure PLS": "o",
                "Hybrid 50:50": "s",
                "NSGA-II": "^",
                "MOEA/D": "v",
                "Improved PLS": "D",
                "Improved Hybrid": "p",
                "Diverse Probe PLS": "*",
                "Diverse Probe Hybrid": "h",
                "Scalarized PLS": "X",
                "Scalarized Hybrid": "P",
                "Diverse+Scalarized PLS": "d",
                "Diverse+Scalarized Hybrid": "<",
            }
            marker = marker_styles.get(cfg.label, "o")
            # Show marker every N points to avoid clutter
            marker_every = max(1, len(ts) // 8)

            ax.plot(
                ts,
                hvs,
                label=cfg.label,
                color=cfg.color,
                linestyle=cfg.linestyle,
                linewidth=cfg.linewidth + 0.5,  # Make lines slightly thicker
                marker=marker,
                markersize=6,
                markevery=marker_every,
                markeredgewidth=1.5,
                markerfacecolor=cfg.color,
                markeredgecolor="white",
                alpha=0.9,
            )
            endpoint_annotations.append((ts[-1], hvs[-1], cfg))

        sorted_annotations = sorted(
            endpoint_annotations, key=lambda item: item[1], reverse=True
        )
        n_annotations = len(sorted_annotations)
        for idx, (end_t, end_hv, cfg) in enumerate(sorted_annotations):
            ax.annotate(
                f"{end_hv:.4f}",
                xy=(end_t, end_hv),
                xytext=(6, 10 * (n_annotations - 1 - idx)),
                textcoords="offset points",
                color=cfg.color,
                fontsize=9,
                va="center",
                ha="left",
            )

        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_ylabel("Normalized Hypervolume", fontsize=12)
        ax.set_title(
            f"HV over Time — {display_name} ({num_images} images, 4D)",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, total_timeout)

        plot_path = output_dir / f"{display_name}.png"
        fig.tight_layout()
        fig.savefig(str(plot_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Plot saved: {plot_path}", flush=True)

    # ── Phase 5: build result artifact ──────────────────────────────────

    def _final_hv(label: str) -> float:
        c = curves.get(label, [])
        return c[-1][1] if c else 0.0

    pure_pls_hv = _final_hv("Pure PLS")

    print(f"\n  {'=' * 60}", flush=True)
    print(f"  FINAL HV VALUES (after normalization)", flush=True)
    print(f"  {'=' * 60}", flush=True)
    for cfg in configs:
        curve = curves.get(cfg.label, [])
        if curve:
            initial = curve[0][1]
            final = curve[-1][1]
            improvement = final - initial
            print(
                f"  [{cfg.label:20s}] initial={initial:.8f}, final={final:.8f}, "
                f"improvement={improvement:.8f} ({improvement / final * 100 if final > 0 else 0:.4f}%)",
                flush=True,
            )

    print(f"\n  {'=' * 60}", flush=True)
    print(f"  FINAL HV SUMMARY:", flush=True)
    print(f"  {'=' * 60}", flush=True)

    result_artifact = dict(
        instance=display_name,
        num_images=num_images,
        timeout_s=total_timeout,
        objectives=OBJECTIVES,
        shared_bounds=bounds,
        total_trace_points=len(all_points),
        configs={},
    )

    for cfg in configs:
        fhv = _final_hv(cfg.label)
        delta = (fhv - pure_pls_hv) / pure_pls_hv * 100 if pure_pls_hv > 0 else 0.0
        result_artifact["configs"][cfg.label] = dict(
            **run_meta.get(cfg.label, {}),
            final_hv=round(fhv, 8),
            delta_vs_pure_pls_pct=round(delta, 4),
            curve=[(round(t, 4), round(hv, 8)) for t, hv in curves.get(cfg.label, [])],
        )

    for cfg in configs:
        fhv = _final_hv(cfg.label)
        delta = (fhv - pure_pls_hv) / pure_pls_hv * 100 if pure_pls_hv > 0 else 0.0
        result_artifact["configs"][cfg.label] = dict(
            **run_meta.get(cfg.label, {}),
            final_hv=round(fhv, 8),
            delta_vs_pure_pls_pct=round(delta, 4),
            curve=[(round(t, 4), round(hv, 8)) for t, hv in curves.get(cfg.label, [])],
        )

    artifact_path = output_dir / f"{display_name}.json"
    with open(artifact_path, "w") as f:
        json.dump(result_artifact, f, indent=2)
    print(f"  Artifact saved: {artifact_path}", flush=True)

    # ── Summary ─────────────────────────────────────────────────────────

    print(f"\n  Summary for {display_name}:", flush=True)
    for cfg in configs:
        info = result_artifact["configs"].get(cfg.label, {})
        fhv = info.get("final_hv", 0)
        nsol = info.get("final_solutions", 0)
        delta = info.get("delta_vs_pure_pls_pct", 0)
        tag = "" if cfg.label == "Pure PLS" else f"Δ={delta:+.1f}%"
        print(
            f"    {cfg.label:22s}  HV={fhv:.6f}  sols={nsol:6d}  {tag}",
            flush=True,
        )

    return result_artifact


# ─── Combined summary figures ─────────────────────────────────────────


def generate_combined_figures(
    all_results: list[dict],
    output_dir: Path,
    configs: list[AlgorithmConfig],
) -> None:
    """Generate a combined multi-panel convergence plot and a bar chart."""
    if not HAS_MATPLOTLIB or not all_results:
        return

    # ── Multi-panel convergence plot ────────────────────────────────────
    n_results = len(all_results)
    ncols = 2
    nrows = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 18))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax_idx, r in enumerate(all_results):
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]
        endpoint_annotations: list[tuple[float, float, AlgorithmConfig]] = []
        for cfg in configs:
            curve_data = r["configs"].get(cfg.label, {}).get("curve", [])
            if not curve_data:
                continue
            ts = [t for t, hv in curve_data]
            hvs = [hv for t, hv in curve_data]
            # Use different markers for each config for better distinguishability
            marker_styles = {
                "Pure PLS": "o",
                "Hybrid 50:50": "s",
                "NSGA-II": "^",
                "MOEA/D": "v",
                "Improved PLS": "D",
                "Improved Hybrid": "p",
                "Diverse Probe PLS": "*",
                "Diverse Probe Hybrid": "h",
                "Scalarized PLS": "X",
                "Scalarized Hybrid": "P",
                "Diverse+Scalarized PLS": "d",
                "Diverse+Scalarized Hybrid": "<",
            }
            marker = marker_styles.get(cfg.label, "o")
            marker_every = max(1, len(ts) // 6)

            ax.plot(
                ts,
                hvs,
                label=cfg.label,
                color=cfg.color,
                linestyle=cfg.linestyle,
                linewidth=cfg.linewidth + 0.5,
                marker=marker,
                markersize=5,
                markevery=marker_every,
                markeredgewidth=1.2,
                markerfacecolor=cfg.color,
                markeredgecolor="white",
                alpha=0.9,
            )
            endpoint_annotations.append((ts[-1], hvs[-1], cfg))

        sorted_annotations = sorted(
            endpoint_annotations, key=lambda item: item[1], reverse=True
        )
        n_annotations = len(sorted_annotations)
        for idx, (end_t, end_hv, cfg) in enumerate(sorted_annotations):
            ax.annotate(
                f"{end_hv:.4f}",
                xy=(end_t, end_hv),
                xytext=(4, 7 * (n_annotations - 1 - idx)),
                textcoords="offset points",
                color=cfg.color,
                fontsize=12,
                va="center",
                ha="left",
            )

        ax.set_title(
            f"{r['instance']} ({r['num_images']} imgs)",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("Time (s)", fontsize=14)
        ax.set_ylabel("Normalized HV", fontsize=14)
        ax.legend(fontsize=12, loc="lower right")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, r["timeout_s"])

    # Hide unused axes
    for ax_idx in range(n_results, len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle(
        "Hypervolume Convergence: Multi-Algorithm Comparison (4 Objectives)",
        fontsize=22,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    path = output_dir / "fig_combined.png"
    fig.savefig(str(path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCombined convergence plot: {path}", flush=True)

    # ── Bar chart: final HV for each algorithm × instance ───────────────
    fig2, ax2 = plt.subplots(figsize=(18, max(7, 0.8 * n_results * len(configs))))

    instance_names = [r["instance"].replace("_", " ").title() for r in all_results]
    n_configs = len(configs)
    bar_height = 0.15
    y_positions = range(len(all_results))

    for cfg_idx, cfg in enumerate(configs):
        hvs = []
        for r in all_results:
            cfg_data = r["configs"].get(cfg.label, {})
            hvs.append(cfg_data.get("final_hv", 0))

        offsets = [
            y + (cfg_idx - n_configs / 2 + 0.5) * bar_height for y in y_positions
        ]
        bars = ax2.barh(
            offsets,
            hvs,
            height=bar_height,
            label=cfg.label,
            color=cfg.color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.3,
        )
        for bar, hv in zip(bars, hvs):
            ax2.text(
                bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{hv:.4f}",
                va="center",
                ha="left",
                fontsize=11,
                color="black",
            )

    ax2.set_yticks(list(y_positions))
    ax2.set_yticklabels(instance_names, fontsize=13)
    ax2.set_xlabel("Normalized Hypervolume", fontsize=16)
    ax2.set_title(
        "Final HV by Algorithm and Instance (4D)",
        fontsize=20,
        fontweight="bold",
    )
    ax2.legend(fontsize=12, loc="lower right")
    ax2.grid(True, axis="x", alpha=0.3)
    fig2.tight_layout()
    path2 = output_dir / "fig_barchart.png"
    fig2.savefig(str(path2), dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Bar chart: {path2}", flush=True)


# ─── CLI ──────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="HV-over-time experiment with configurable PLS and hybrid ablations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Only instances with ≤ N images",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Regex filter on instance name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hv_experiment_results"),
        help="Output directory for plots and JSON",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=30,
        help="Sample points per HV curve (default: 30)",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=None,
        help="Subset of series labels to run (default: all 5)",
    )
    args = parser.parse_args()

    # Filter instances
    instances = ALL_INSTANCES
    if args.max_size is not None:
        instances = [(n, f, s) for n, f, s in instances if s <= args.max_size]
    if args.filter is not None:
        pat = re.compile(args.filter)
        instances = [(n, f, s) for n, f, s in instances if pat.search(n)]

    if not instances:
        print("No instances matched the filters.", file=sys.stderr)
        return 1

    # Filter configs
    configs = CONFIGS
    if args.configs is not None:
        requested = set(args.configs)
        configs = [c for c in CONFIGS if c.label in requested]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(instances)} instances × {len(configs)} configs")
    print(f"Instances: {[n for n, _, _ in instances]}")
    print(f"Configs:   {[c.label for c in configs]}")
    print(f"Output:    {args.output_dir}")
    print(f"HV points: {args.num_points}")

    all_results: list[dict] = []
    t_start = time.time()

    for display_name, filename, num_images in instances:
        try:
            result = run_instance(
                display_name,
                filename,
                num_images,
                args.output_dir,
                args.num_points,
                configs,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n  FATAL ERROR on {display_name}: {e}", flush=True)
            import traceback

            traceback.print_exc()

    # ── Global summary ──────────────────────────────────────────────────

    total_time = time.time() - t_start
    print(f"\n{'=' * 72}", flush=True)
    print(
        f"  ALL EXPERIMENTS COMPLETE  ({total_time:.0f}s = {total_time / 60:.1f} min)",
        flush=True,
    )
    print(f"{'=' * 72}", flush=True)

    if all_results:
        labels = [c.label for c in configs]
        header = f"{'Instance':>25s}  {'Size':>5s}"
        for lbl in labels:
            header += f"  {lbl:>14s}"
        print(f"\n{header}", flush=True)
        print("-" * len(header), flush=True)

        for r in all_results:
            row = f"{r['instance']:>25s}  {r['num_images']:>5d}"
            for lbl in labels:
                fhv = r.get("configs", {}).get(lbl, {}).get("final_hv", 0)
                row += f"  {fhv:>14.6f}"
            print(row, flush=True)

    # Combined figures
    generate_combined_figures(all_results, args.output_dir, configs)

    # Save combined results
    combined_path = args.output_dir / "all_experiments.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results: {combined_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
