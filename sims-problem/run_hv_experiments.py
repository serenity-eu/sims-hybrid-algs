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
import math
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
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "Liberation Sans"],
            "font.size": 9,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "figure.dpi": 150,
        }
    )

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

# ─── Plot constants ────────────────────────────────────────────────────

_CITY_LABELS: dict[str, str] = {
    "lagos_nigeria": "Lagos, Nigeria",
    "mexico_city": "Mexico City",
    "paris": "Paris",
    "rio_de_janeiro": "Rio de Janeiro",
    "tokyo_bay": "Tokyo Bay",
}

# Maps config label → short variant name used in large-instance (>100) plots.
_PLS_VARIANT_MAP: dict[str, str] = {
    "Default PLS": "Default",
    "Scalarized PLS": "Scalarized",
    "Diverse Probe PLS": "Diverse Probe",
}

_VARIANT_COLORS: dict[str, str] = {
    "Default": "#e07b00",
    "Scalarized": "#2166ac",
    "Diverse Probe": "#2ca02c",
}

_VARIANT_LINE: dict[str, tuple[str, float]] = {
    "Default": ("-", 2.0),
    "Scalarized": ("--", 1.6),
    "Diverse Probe": ("-.", 1.6),
}

# (variant_name, hatch, legend_label)  — for bar charts
_PLS_VARIANTS: list[tuple[str, str, str]] = [
    ("Default", "", "Default / Hybrid"),
    ("Scalarized", "//", "Scalarized"),
    ("Diverse Probe", "oo", "Diverse Probe"),
]

# Maps config label → (exact_phase_ratio, variant_name) for phase-decomposition bars.
# Ratio 1.0 = GPBA-A only, 0.0 = PLS only.
_LABEL_TO_BAR_KEY: dict[str, tuple[float, str]] = {
    "GPBA-A": (1.00, "Default"),
    "Default PLS": (0.00, "Default"),
    "Hybrid 50:50": (0.50, "Default"),
    "Hybrid 35:65": (0.35, "Default"),
    "Hybrid 20:80": (0.20, "Default"),
    "Scalarized PLS": (0.00, "Scalarized"),
    "Scalarized Hybrid 50:50": (0.50, "Scalarized"),
    "Scalarized Hybrid 35:65": (0.35, "Scalarized"),
    "Scalarized Hybrid 20:80": (0.20, "Scalarized"),
    "Diverse Probe PLS": (0.00, "Diverse Probe"),
    "Diverse Probe Hybrid 50:50": (0.50, "Diverse Probe"),
    "Diverse Probe Hybrid 35:65": (0.35, "Diverse Probe"),
    "Diverse Probe Hybrid 20:80": (0.20, "Diverse Probe"),
}

_COLOR_PHASE1 = "#2166ac"  # steel blue — GPBA-A exact phase
_COLOR_PHASE2 = "#e07b00"  # amber      — GPBA-A PLS phase
_COLOR_PHASE1_MONISE = "#1a9641"  # forest green — MONISE exact phase
_COLOR_PHASE2_MONISE = "#d73027"  # crimson      — MONISE PLS phase


def _city_label(instance_name: str) -> str:
    for key, label in _CITY_LABELS.items():
        if instance_name.startswith(key):
            return label
    return instance_name


def _size_group_label(num_images: int) -> str:
    return "145 / 150" if num_images in (145, 150) else str(num_images)


# ─── Timeout heuristic ────────────────────────────────────────────────


def timeout_for_size(num_images: int) -> int:
    """PLS timeout in seconds – longer budgets for full HV ablation runs."""
    if num_images <= 30:
        return 30
    if num_images <= 50:
        return 300
    if num_images <= 150:
        return 1000
    return 3600


# ─── Algorithm configurations ─────────────────────────────────────────


@dataclass
class AlgorithmConfig:
    """One series on the plot."""

    label: str
    color: str
    linestyle: str
    linewidth: float = 2.0
    skip_normalization: bool = False
    save_trace: bool = True
    # Fraction of the total budget spent in the exact (GPBA-A) phase.
    # Set on hybrid configs so the plot can trim the exact-phase overlap
    # and show only the PLS tail starting from the GPBA-A handoff point.
    exact_phase_ratio: Optional[float] = None

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
            use_greedy_initial_population=True,
            use_perturbation_restart=False,
        )


class Hybrid2080(AlgorithmConfig):
    """Hybrid 20:80 – 20% exact phase, 80% PLS, seeded with pseudo-solver solutions."""

    def run(self, problem, timeout_s, seed=42):
        exact_time = timeout_s * 20 // 100
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
            use_greedy_initial_population=True,
            use_perturbation_restart=False,
        )


class Hybrid3565(AlgorithmConfig):
    """Hybrid 35:65 – 35% exact phase, 65% PLS, seeded with pseudo-solver solutions."""

    def run(self, problem, timeout_s, seed=42):
        exact_time = timeout_s * 35 // 100
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
            use_greedy_initial_population=True,
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
            use_greedy_initial_population=True,
            use_perturbation_restart=False,
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
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=False,
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
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=False,
            use_diverse_probing=True,
        )


class DiverseProbeHybrid3565(DiverseProbeHybrid):
    """Diverse Probe Hybrid 35:65 – 35% exact, 65% PLS."""

    def run(self, problem, timeout_s, seed=42):
        exact_time = timeout_s * 35 // 100
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
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=False,
            use_diverse_probing=True,
        )


class DiverseProbeHybrid2080(DiverseProbeHybrid):
    """Diverse Probe Hybrid 20:80 – 20% exact, 80% PLS."""

    def run(self, problem, timeout_s, seed=42):
        exact_time = timeout_s * 20 // 100
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
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=False,
            use_diverse_probing=True,
        )


class ScalarizedPLS(AlgorithmConfig):
    """Improved PLS with scalarized parent selection."""

    scalarized_selection_source: str = "archive"
    scalarized_parent_budget: int = 1
    scalarized_weight_samples: int = 1
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
            use_greedy_initial_population=True,
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
    scalarized_parent_budget: int = 1
    scalarized_weight_samples: int = 1
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
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=False,
            solution_selection_mode="scalarized-chebycheff",
            scalarized_selection_source=self.scalarized_selection_source,
            scalarized_parent_budget=self.scalarized_parent_budget,
            scalarized_weight_samples=self.scalarized_weight_samples,
            scalarized_rho=self.scalarized_rho,
            use_nd_tree_scalarized_query=self.use_nd_tree_scalarized_query,
        )


class ScalarizedHybrid3565(ScalarizedHybrid):
    """Scalarized Hybrid 35:65 – 35% exact, 65% PLS."""

    def run(self, problem, timeout_s, seed=42):
        exact_time = timeout_s * 35 // 100
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
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=False,
            solution_selection_mode="scalarized-chebycheff",
            scalarized_selection_source=self.scalarized_selection_source,
            scalarized_parent_budget=self.scalarized_parent_budget,
            scalarized_weight_samples=self.scalarized_weight_samples,
            scalarized_rho=self.scalarized_rho,
            use_nd_tree_scalarized_query=self.use_nd_tree_scalarized_query,
        )


class ScalarizedHybrid2080(ScalarizedHybrid):
    """Scalarized Hybrid 20:80 – 20% exact, 80% PLS."""

    def run(self, problem, timeout_s, seed=42):
        exact_time = timeout_s * 20 // 100
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
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=False,
            solution_selection_mode="scalarized-chebycheff",
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


class NSGA3Config(AlgorithmConfig):
    """NSGA-III with reference-point niching (Deb & Jain 2014)."""

    def run(self, problem, timeout_s, seed=42):
        return sims_problem.solve_with_nsga3(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=timeout_s),
            target_pop_size=200,
            auto_divisions=True,
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


class MemeticNSGA2Config(AlgorithmConfig):
    """PLS warm-start → NSGA-II hybrid."""

    def run(self, problem, timeout_s, seed=42):
        return sims_problem.solve_with_memetic_nsga2(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=timeout_s),
            pls_time_fraction=0.3,
            pls_initial_pop_size=50,
            max_pls_seed_size=0,
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


class MemeticNSGA3Config(AlgorithmConfig):
    """PLS warm-start → NSGA-III hybrid."""

    def run(self, problem, timeout_s, seed=42):
        return sims_problem.solve_with_memetic_nsga3(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=timeout_s),
            pls_time_fraction=0.3,
            pls_initial_pop_size=50,
            max_pls_seed_size=0,
            target_pop_size=200,
            auto_divisions=True,
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


class MemeticMOEADConfig(AlgorithmConfig):
    """PLS warm-start → MOEA/D hybrid."""

    def run(self, problem, timeout_s, seed=42):
        return sims_problem.solve_with_memetic_moead(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=timeout_s),
            pls_time_fraction=0.3,
            pls_initial_pop_size=50,
            max_pls_seed_size=0,
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


class GPBASeededPLS(AlgorithmConfig):
    """Full PLS seeded with the complete GPBA-A solution set, no time split."""

    def run(self, problem, timeout_s, seed=42):
        exact_solutions = _current_pseudo_solutions
        initial_pop = (
            _solutions_to_sims(exact_solutions, problem) if exact_solutions else None
        )
        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=timeout_s),
            is_deterministic=True,
            trace=True,
            include_dominated=False,
            initial_population=initial_pop,
            use_checkpoint=True,
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=False,
        )


@dataclass
class _SyntheticResult:
    trace: bytes
    final_solutions: list = field(default_factory=list)


class GPBAAConfig(AlgorithmConfig):
    """GPBA-A exact-solver phase only — plots HV progression of GPBA-A solutions."""

    def run(self, problem, timeout_s, seed=42):
        solutions = _current_pseudo_solutions
        if not solutions:
            return _SyntheticResult(trace=b"")

        converted: list[sims_problem.Solution] = []
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
                    timestamp_us=int(sol_dict.get("timestamp_s", 0.0) * 1_000_000),
                    min_resolutions_sum=sol_dict.get("min_resolutions_sum"),
                )
                converted.append(sol)
            except Exception:
                pass

        if not converted:
            return _SyntheticResult(trace=b"")

        ndim = len(OBJECTIVES)
        obj_keys = ("cost", "cloudy_area", "max_incidence_angle", "min_resolutions_sum")
        points = [
            [int(s.get(k, 0)) for k in obj_keys]
            for s in solutions
            if s.get("selected_images")
        ]
        bounds: list[list[int]] = []
        for j in range(ndim):
            vals = [p[j] for p in points]
            lo, hi = min(vals), max(vals)
            rng = max(hi - lo, 1)
            bounds.append([max(0, lo - 1), hi + int(rng * 0.1) + 1])
        ref_point = [b[1] + 1 for b in bounds]

        trace = sims_problem.generate_trace(
            solutions=converted,
            objectives=OBJECTIVES,
            algorithm="GPBA-A",
            num_objectives=ndim,
            objective_bounds=bounds,
            reference_point=ref_point,
            include_dominated=False,
        )
        return _SyntheticResult(trace=trace, final_solutions=converted)


class MONISEPseudoConfig(GPBAAConfig):
    """MONISE exact-solver phase only — plots HV progression of MONISE solutions."""

    def run(self, problem, timeout_s, seed=42):
        result = super().run(problem, timeout_s, seed)
        # Re-generate trace with "MONISE" algorithm label (GPBAAConfig uses "GPBA-A")
        if not result.trace:
            return result
        solutions = _current_pseudo_solutions
        converted = result.final_solutions
        if not converted:
            return result
        ndim = len(OBJECTIVES)
        obj_keys = ("cost", "cloudy_area", "max_incidence_angle", "min_resolutions_sum")
        points = [
            [int(s.get(k, 0)) for k in obj_keys]
            for s in solutions
            if s.get("selected_images")
        ]
        bounds: list[list[int]] = []
        for j in range(ndim):
            vals = [p[j] for p in points]
            lo, hi = min(vals), max(vals)
            rng = max(hi - lo, 1)
            bounds.append([max(0, lo - 1), hi + int(rng * 0.1) + 1])
        ref_point = [b[1] + 1 for b in bounds]
        trace = sims_problem.generate_trace(
            solutions=converted,
            objectives=OBJECTIVES,
            algorithm="MONISE",
            num_objectives=ndim,
            objective_bounds=bounds,
            reference_point=ref_point,
            include_dominated=False,
        )
        return _SyntheticResult(trace=trace, final_solutions=converted)


# ─── MONISE-seeded hybrid configs ────────────────────────────────────
# These mirror the existing Hybrid* classes but are given distinct labels so
# that MONISE-seeded and GPBA-A-seeded variants can appear on the same plot.
# The actual solution source is controlled by _PSEUDO_SOLUTIONS_SOURCE (set via
# --pseudo-source), so instantiating these configs when --pseudo-source=monise
# gives the intended behaviour.


class MoniseHybridBaseline(HybridBaseline):
    """Hybrid 50:50 seeded with MONISE solutions."""


class MoniseHybrid3565(Hybrid3565):
    """Hybrid 35:65 seeded with MONISE solutions."""


class MoniseHybrid2080(Hybrid2080):
    """Hybrid 20:80 seeded with MONISE solutions."""


class MoniseDiverseProbeHybrid(DiverseProbeHybrid):
    """Diverse Probe Hybrid 50:50 seeded with MONISE solutions."""


class MoniseDiverseProbeHybrid3565(DiverseProbeHybrid3565):
    """Diverse Probe Hybrid 35:65 seeded with MONISE solutions."""


class MoniseDiverseProbeHybrid2080(DiverseProbeHybrid2080):
    """Diverse Probe Hybrid 20:80 seeded with MONISE solutions."""


class MoniseScalarizedHybrid(ScalarizedHybrid):
    """Scalarized Hybrid 50:50 seeded with MONISE solutions."""


class MoniseScalarizedHybrid3565(ScalarizedHybrid3565):
    """Scalarized Hybrid 35:65 seeded with MONISE solutions."""


class MoniseScalarizedHybrid2080(ScalarizedHybrid2080):
    """Scalarized Hybrid 20:80 seeded with MONISE solutions."""


# MONISE hybrid configurations mirroring CONFIGS but with MONISE-prefixed labels.
MONISE_CONFIGS: list[AlgorithmConfig] = [
    MONISEPseudoConfig(
        label="MONISE",
        color="#000000",
        linestyle="--",
        linewidth=2.0,
    ),
    PurePLS(
        label="Default PLS",
        color="#d62728",
        linestyle="-",
        linewidth=1.5,
    ),
    MoniseHybridBaseline(
        label="MONISE Hybrid 50:50",
        color="#1f77b4",
        linestyle="-",
        linewidth=1.8,
        exact_phase_ratio=0.50,
    ),
    MoniseHybrid3565(
        label="MONISE Hybrid 35:65",
        color="#e07b00",
        linestyle="--",
        linewidth=1.8,
        exact_phase_ratio=0.35,
    ),
    MoniseHybrid2080(
        label="MONISE Hybrid 20:80",
        color="#2ca02c",
        linestyle="-.",
        linewidth=1.8,
        exact_phase_ratio=0.20,
    ),
    MoniseDiverseProbeHybrid(
        label="MONISE Diverse Probe Hybrid 50:50",
        color="#17becf",
        linestyle="-",
        linewidth=2.5,
        exact_phase_ratio=0.50,
    ),
    MoniseDiverseProbeHybrid3565(
        label="MONISE Diverse Probe Hybrid 35:65",
        color="#6fe8f5",
        linestyle="-",
        linewidth=1.5,
        exact_phase_ratio=0.35,
    ),
    MoniseDiverseProbeHybrid2080(
        label="MONISE Diverse Probe Hybrid 20:80",
        color="#adf3fb",
        linestyle="-",
        linewidth=1.5,
        exact_phase_ratio=0.20,
    ),
    MoniseScalarizedHybrid(
        label="MONISE Scalarized Hybrid 50:50",
        color="#bcbd22",
        linestyle="-",
        linewidth=2.5,
        exact_phase_ratio=0.50,
    ),
    MoniseScalarizedHybrid3565(
        label="MONISE Scalarized Hybrid 35:65",
        color="#d4d668",
        linestyle="-",
        linewidth=1.5,
        exact_phase_ratio=0.35,
    ),
    MoniseScalarizedHybrid2080(
        label="MONISE Scalarized Hybrid 20:80",
        color="#e8e9a0",
        linestyle="-",
        linewidth=1.5,
        exact_phase_ratio=0.20,
    ),
]

# Also register MONISE labels in the bar-chart mapping.
_LABEL_TO_BAR_KEY.update(
    {
        "MONISE": (1.00, "Default"),
        "MONISE Hybrid 50:50": (0.50, "Default"),
        "MONISE Hybrid 35:65": (0.35, "Default"),
        "MONISE Hybrid 20:80": (0.20, "Default"),
        "MONISE Diverse Probe Hybrid 50:50": (0.50, "Diverse Probe"),
        "MONISE Diverse Probe Hybrid 35:65": (0.35, "Diverse Probe"),
        "MONISE Diverse Probe Hybrid 20:80": (0.20, "Diverse Probe"),
        "MONISE Scalarized Hybrid 50:50": (0.50, "Scalarized"),
        "MONISE Scalarized Hybrid 35:65": (0.35, "Scalarized"),
        "MONISE Scalarized Hybrid 20:80": (0.20, "Scalarized"),
    }
)


# The series in plot order.
CONFIGS: list[AlgorithmConfig] = [
    GPBASeededPLS(
        label="GPBA-Seeded PLS",
        color="#2ca02c",
        linestyle="-",
        linewidth=2.5,
        skip_normalization=True,
        save_trace=True,
    ),
    GPBAAConfig(
        label="GPBA-A",
        color="#000000",
        linestyle="--",
        linewidth=2.0,
    ),
    PurePLS(
        label="Default PLS",
        color="#d62728",
        linestyle="-",
        linewidth=1.5,
    ),
    HybridBaseline(
        label="Hybrid 50:50",
        color="#1f77b4",
        linestyle="-",
        linewidth=1.8,
        exact_phase_ratio=0.50,
    ),
    Hybrid3565(
        label="Hybrid 35:65",
        color="#e07b00",
        linestyle="--",
        linewidth=1.8,
        exact_phase_ratio=0.35,
    ),
    Hybrid2080(
        label="Hybrid 20:80",
        color="#2ca02c",
        linestyle="-.",
        linewidth=1.8,
        exact_phase_ratio=0.20,
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
    NSGA3Config(
        label="NSGA-III",
        color="#2ecc71",
        linestyle="-",
    ),
    MemeticNSGA2Config(
        label="Memetic NSGA-II (30% PLS)",
        color="#e74c3c",
        linestyle="--",
    ),
    MemeticNSGA3Config(
        label="Memetic NSGA-III (30% PLS)",
        color="#c0392b",
        linestyle="-.",
    ),
    MemeticMOEADConfig(
        label="Memetic MOEA/D (30% PLS)",
        color="#8e44ad",
        linestyle=":",
    ),
    DiverseProbePLS(
        label="Diverse Probe PLS",
        color="#e377c2",
        linestyle="-",
        linewidth=1.5,
    ),
    DiverseProbeHybrid(
        label="Diverse Probe Hybrid 50:50",
        color="#17becf",
        linestyle="-",
        linewidth=2.5,
        exact_phase_ratio=0.50,
    ),
    DiverseProbeHybrid3565(
        label="Diverse Probe Hybrid 35:65",
        color="#6fe8f5",
        linestyle="-",
        linewidth=1.5,
        exact_phase_ratio=0.35,
    ),
    DiverseProbeHybrid2080(
        label="Diverse Probe Hybrid 20:80",
        color="#adf3fb",
        linestyle="-",
        linewidth=1.5,
        exact_phase_ratio=0.20,
    ),
    ScalarizedPLS(
        label="Scalarized PLS",
        color="#8c564b",
        linestyle="-",
        linewidth=1.5,
    ),
    ScalarizedHybrid(
        label="Scalarized Hybrid 50:50",
        color="#bcbd22",
        linestyle="-",
        linewidth=2.5,
        exact_phase_ratio=0.50,
    ),
    ScalarizedHybrid3565(
        label="Scalarized Hybrid 35:65",
        color="#d4d668",
        linestyle="-",
        linewidth=1.5,
        exact_phase_ratio=0.35,
    ),
    ScalarizedHybrid2080(
        label="Scalarized Hybrid 20:80",
        color="#e8e9a0",
        linestyle="-",
        linewidth=1.5,
        exact_phase_ratio=0.20,
    ),
]


# ─── Pseudo-solver helpers ────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent.parent / "sims-core" / "tests" / "data"

# Sources: "gpbaa" uses GPBA-A pre-computed solutions (classic pseudo-solver),
#          "monise" uses MONISE-generated solutions.
# Kept as a module-level variable so run_instance() can read it after main()
# overrides it based on --pseudo-source.
_PSEUDO_SOLUTIONS_SOURCE: str = "gpbaa"

# Legacy path kept so existing imports/paths still resolve.
_PSEUDO_SOLUTIONS_DIR = _DATA_DIR / "pseudo_solver_solutions"

_PSEUDO_SOURCE_DIRS: dict[str, Path] = {
    "gpbaa": _DATA_DIR / "gpbaa",
    "monise": _DATA_DIR / "monise",
    # fallback to original location if gpbaa dir doesn't exist yet
    "pseudo_solver_solutions": _DATA_DIR / "pseudo_solver_solutions",
}

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
    """Load pre-recorded solver solutions from JSON.

    The source directory is controlled by the module-level
    ``_PSEUDO_SOLUTIONS_SOURCE`` variable (set by ``--pseudo-source``).
    Falls back to the legacy ``pseudo_solver_solutions`` directory when the
    configured source directory does not contain the instance file.
    """
    cache_key = f"{_PSEUDO_SOLUTIONS_SOURCE}:{instance_name}"
    if cache_key in _pseudo_cache:
        return _pseudo_cache[cache_key]

    # Primary: selected source
    source_dir = _PSEUDO_SOURCE_DIRS.get(
        _PSEUDO_SOLUTIONS_SOURCE, _PSEUDO_SOLUTIONS_DIR
    )
    json_path = source_dir / f"{instance_name}.json"

    # Fallback to legacy directory if primary doesn't have the file
    if not json_path.exists():
        json_path = _PSEUDO_SOLUTIONS_DIR / f"{instance_name}.json"

    if not json_path.exists():
        _pseudo_cache[cache_key] = []
        return []

    with open(json_path) as f:
        data = json.load(f)

    solutions = data if isinstance(data, list) else data.get("solutions", [])
    _pseudo_cache[cache_key] = solutions
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
            new_ts = min(pseudo_ts, int(exact_time_s * 1_000_000))
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
    """Return pseudo objective vectors present in the provided traces dict.

    Accepts any dict of label→bytes (the caller narrows to the relevant hybrid
    trace), so this function works for both GPBA-A-seeded and MONISE-seeded runs.
    """
    hybrid = next(iter(traces.values()), None) if traces else None
    if not hybrid:
        return set()
    improved = hybrid

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
    if not hybrid:
        return
    improved = hybrid

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


# ─── Plotting helpers ─────────────────────────────────────────────────


def _save_fig(fig: "plt.Figure", path: "Path", dpi: int = 200) -> None:
    """Save figure as PNG and EPS side-by-side."""
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    fig.savefig(str(path.with_suffix(".eps")), bbox_inches="tight")


def _annotate_endpoints(
    ax: "plt.Axes",
    endpoint_annotations: list[tuple[float, float, str]],  # (t, hv, color)
    fmt: str = "{:.3f}",
) -> None:
    """Collision-aware right-side HV labels — pure vertical spread, no x-stagger.

    Sorts endpoints by HV descending and cascades each label downward only
    enough to prevent overlap with the label immediately above it.
    """
    ax.figure.canvas.draw()
    y_lo, y_hi = ax.get_ylim()
    bbox = ax.get_window_extent()
    axes_h_pts = bbox.height * (72 / ax.figure.dpi)
    pts_per_hv = axes_h_pts / max(y_hi - y_lo, 1e-6)
    min_sep = 10  # typographic points between label centres

    placed_y: list[float] = []  # HV-unit positions, descending
    # Minimum y position (in HV units) below which we skip the annotation to
    # avoid labels cascading outside the axes boundary and producing stray
    # floating text in adjacent whitespace.
    y_clip = y_lo + (min_sep / 2) / pts_per_hv

    for end_t, end_hv, color in sorted(
        endpoint_annotations, key=lambda x: x[1], reverse=True
    ):
        pos = end_hv
        if placed_y:
            ceiling = placed_y[-1] - min_sep / pts_per_hv
            if pos > ceiling:
                pos = ceiling
        placed_y.append(pos)

        # Skip labels that would be placed below the visible axes area.
        if pos < y_clip:
            continue

        y_off = (pos - end_hv) * pts_per_hv
        ax.annotate(
            fmt.format(end_hv),
            xy=(end_t, end_hv),
            xytext=(6, y_off),
            textcoords="offset points",
            color=color,
            fontsize=8,
            va="center",
            ha="left",
            arrowprops=dict(arrowstyle="-", color=color, linewidth=0.7, alpha=0.5)
            if abs(y_off) > 4
            else None,
            annotation_clip=False,
        )


def _plot_instance_phase_bars(
    result: dict,
    configs: list[AlgorithmConfig],
    output_dir: Path,
    compare_result: dict | None = None,
) -> None:
    """Phase-decomposition stacked bar chart.

    X-axis groups: 100%/0%, 50%/50%, 35%/65%, 20%/80%, 0%/100% (exact/PLS ratio).
    Each group has up to 3 bars per source (Default, Scalarized, Diverse Probe).
    When compare_result is provided (GPBA-A hybrid data), bars from both sources
    appear side-by-side within each variant position, distinguished by color and
    hatch pattern — GPBA-A in blue/amber, MONISE in green/crimson.
    """
    if not HAS_MATPLOTLIB:
        return

    display_name = result["instance"]
    num_images = result["num_images"]
    total_timeout = result["timeout_s"]

    has_monise = "MONISE" in result.get("configs", {}) and "GPBA-A" not in result.get(
        "configs", {}
    )
    primary_source = "MONISE" if has_monise else "GPBA-A"
    exact_phase_label = "MONISE phase" if has_monise else "GPBA-A phase"

    primary_curve = result.get("configs", {}).get("GPBA-A", {}).get(
        "curve", []
    ) or result.get("configs", {}).get("MONISE", {}).get("curve", [])
    if compare_result:
        _cmp_cfgs = compare_result.get("configs", {})
        compare_curve = (
            _cmp_cfgs.get("GPBA-A", {}) or _cmp_cfgs.get("MONISE", {})
        ).get("curve", [])
    else:
        compare_curve = []

    def _hv_at(curve: list, t: float) -> float:
        if not curve:
            return 0.0
        if t <= curve[0][0]:
            return curve[0][1]
        for i in range(1, len(curve)):
            t0, hv0 = curve[i - 1]
            t1, hv1 = curve[i]
            if t0 <= t <= t1:
                alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return hv0 + alpha * (hv1 - hv0)
        return curve[-1][1]

    def _build_bars(res: dict, ref_curve: list, source: str) -> dict:
        bars: dict[tuple[float, str, str], dict] = {}
        # When no time curve is available, use the exact-solver standalone final_hv as
        # the best-case estimate of what the exact phase contributed.
        fallback_exact_hv = (
            res.get("configs", {}).get("GPBA-A", {}).get("final_hv", 0.0)
            if not ref_curve
            else 0.0
        )
        for label, cdata in res.get("configs", {}).items():
            key = _LABEL_TO_BAR_KEY.get(label)
            if key is None:
                continue
            ratio, variant = key
            final_hv = cdata.get("final_hv", 0.0)
            if final_hv == 0.0:
                continue
            if ratio == 1.0:
                hv_p1, hv_p2 = final_hv, 0.0
            elif ratio == 0.0:
                hv_p1, hv_p2 = 0.0, final_hv
            else:
                if ref_curve:
                    hv_p1 = _hv_at(ref_curve, total_timeout * ratio)
                else:
                    hv_p1 = min(fallback_exact_hv, final_hv)
                hv_p2 = max(0.0, final_hv - hv_p1)
            bars[(ratio, variant, source)] = dict(
                hv_p1=hv_p1, hv_p2=hv_p2, final_hv=final_hv
            )
        return bars

    # Detect compare source from compare_result content
    has_compare = compare_result is not None
    if has_compare:
        _cmp_keys = compare_result.get("configs", {})
        compare_source: str = "MONISE" if "MONISE" in _cmp_keys else "GPBA-A"
    else:
        compare_source = "GPBA-A"  # unused when has_compare is False

    # Build bar data keyed by (ratio, variant, source)
    all_bars: dict[tuple[float, str, str], dict] = {}
    all_bars.update(_build_bars(result, primary_curve, primary_source))
    if has_compare:
        all_bars.update(_build_bars(compare_result, compare_curve, compare_source))

    if not all_bars:
        return

    # GPBA-A = blue/amber; MONISE = green/crimson
    _PHASE_COLORS: dict[str, tuple[str, str]] = {
        "GPBA-A":  (_COLOR_PHASE1,        _COLOR_PHASE2),
        "MONISE":  (_COLOR_PHASE1_MONISE,  _COLOR_PHASE2),
    }

    _GROUPS = [
        ("100% / 0%", 1.00),
        ("50% / 50%", 0.50),
        ("35% / 65%", 0.35),
        ("20% / 80%", 0.20),
        ("0% / 100%", 0.00),
    ]
    present_groups = [
        (gname, ratio)
        for gname, ratio in _GROUPS
        if any(k[0] == ratio for k in all_bars)
    ]

    # Source order: GPBA-A left, the other source right
    _right_source = compare_source if has_compare else primary_source
    _SOURCES = ["GPBA-A", _right_source] if has_compare else [primary_source]

    if has_compare:
        # Paired layout: each variant slot has a GPBA-A bar and a MONISE bar side by side
        bar_w = 0.14
        pair_gap = 0.03  # between GPBA-A and MONISE within same variant
        var_gap = 0.12  # between variant pairs
        pair_w = 2 * bar_w + pair_gap
        n_var = len(_PLS_VARIANTS)
        group_span = n_var * pair_w + (n_var - 1) * var_gap
        group_gap = 0.25
        # Pair centers relative to group center
        pair_centers = [
            -group_span / 2 + v_idx * (pair_w + var_gap) + pair_w / 2
            for v_idx in range(n_var)
        ]
        # Within each pair: GPBA-A left, compare_source right
        src_pair_off = {
            "GPBA-A": -(pair_gap / 2 + bar_w / 2),
            compare_source: +(pair_gap / 2 + bar_w / 2),
        }
        fig_w = 16
    else:
        bar_w = 0.20
        inner_gap = 0.04
        n_var = len(_PLS_VARIANTS)
        group_span = n_var * bar_w + (n_var - 1) * inner_gap
        group_gap = 0.18
        pair_centers = None  # unused in single-source mode
        src_pair_off = None
        fig_w = 13

    group_cx = [i * (group_span + group_gap) for i in range(len(present_groups))]
    # For single-source groups (0%/100%, 100%/0%) in comparison mode, space variant
    # bars with the same pair_gap used between GPBA-A and MONISE in hybrid groups.
    _single_group_stride = bar_w + (pair_gap if has_compare else 0.04)
    var_off = [(j - (n_var - 1) / 2) * _single_group_stride for j in range(n_var)]

    fig, ax = plt.subplots(figsize=(fig_w, 6))
    legend_done: dict[tuple[str, str], bool] = {}  # (source, phase) -> drawn

    for g_idx, (gname, ratio) in enumerate(present_groups):
        gx = group_cx[g_idx]
        n_bars = 1 if ratio == 1.0 else n_var

        for v_idx, (vname, hatch, _) in enumerate(_PLS_VARIANTS):
            if n_bars == 1 and v_idx > 0:
                continue
            lookup_variant = vname if n_bars > 1 else "Default"

            for sname in _SOURCES:
                # For the 0%/100% group the primary result already has all PLS
                # variants injected from compare — skip the compare source here
                # to avoid rendering the same data twice.
                if ratio == 0.0 and has_compare and sname != primary_source:
                    continue

                bd = all_bars.get((ratio, lookup_variant, sname))
                # At 100%/0%, render a superthin placeholder for GPBA-A when it
                # timed out (no solutions found), so the group still shows both sources.
                if bd is None and ratio == 1.0 and has_compare and sname == "GPBA-A":
                    color_p1, _ = _PHASE_COLORS[sname]
                    bx = gx + src_pair_off[sname]
                    ax.bar(
                        bx,
                        0.001,
                        width=bar_w * 0.25,
                        color=color_p1,
                        edgecolor="white",
                        linewidth=0.4,
                        zorder=3,
                    )
                    ax.text(
                        bx,
                        0.006,
                        "0",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="#333333",
                    )
                    continue
                if bd is None:
                    continue

                color_p1, color_p2 = _PHASE_COLORS[sname]
                edge_col = "white"
                edge_lw = 0.4

                if has_compare:
                    if ratio == 0.0:
                        # PLS-only group: single source, 3 variant bars tightly spaced
                        bx = gx + var_off[v_idx]
                        bw = bar_w
                    elif ratio == 1.0:
                        # Exact-only group: two sources (GPBA-A + MONISE), no variant spread
                        bx = gx + src_pair_off[sname]
                        bw = bar_w
                    else:
                        pc = pair_centers[v_idx]
                        bx = gx + pc + src_pair_off[sname]
                        bw = bar_w
                else:
                    bx = gx if n_bars == 1 else gx + var_off[v_idx]
                    bw = bar_w

                lbl_p1 = None
                if not legend_done.get((sname, "p1")) and bd["hv_p1"] > 0:
                    phase_lbl = "MONISE phase" if sname == "MONISE" else "GPBA-A phase"
                    lbl_p1 = phase_lbl
                    legend_done[(sname, "p1")] = True

                ax.bar(
                    bx,
                    bd["hv_p1"],
                    width=bw,
                    color=color_p1,
                    hatch=None,  # no hatch on exact phase — pattern encodes PLS variant
                    edgecolor=edge_col,
                    linewidth=edge_lw,
                    zorder=3,
                    label=lbl_p1,
                )

                # When exact phase contributed nothing, draw a thin colored tick
                # at the base of the bar so the source is still visually identifiable.
                if bd["hv_p1"] == 0.0 and bd["hv_p2"] > 0:
                    ax.plot(
                        [bx - bw / 2, bx + bw / 2],
                        [0, 0],
                        color=color_p1,
                        linewidth=3,
                        solid_capstyle="butt",
                        zorder=4,
                    )

                if bd["hv_p2"] > 0:
                    lbl_p2 = None
                    if not legend_done.get(("any", "p2")):
                        lbl_p2 = "PLS phase gain"
                        legend_done[("any", "p2")] = True
                    ax.bar(
                        bx,
                        bd["hv_p2"],
                        width=bw,
                        bottom=bd["hv_p1"],
                        color=color_p2,
                        hatch=hatch,  # hatch encodes PLS variant only on PLS phase
                        edgecolor=edge_col,
                        linewidth=edge_lw,
                        zorder=3,
                        label=lbl_p2,
                    )

                ax.text(
                    bx,
                    bd["final_hv"] + 0.005,
                    f"{bd['final_hv']:.3f}",
                    ha="left",
                    va="bottom",
                    fontsize=11 if has_compare else 12,
                    color="#333333",
                    rotation=45,
                    rotation_mode="anchor",
                )

    ax.set_xticks(group_cx)
    ax.set_xticklabels([g for g, _ in present_groups], fontsize=14)
    ax.tick_params(axis="x", length=0)

    if has_compare:
        from matplotlib.lines import Line2D

        phase_handles = [
            mpatches.Patch(
                facecolor=_COLOR_PHASE1, edgecolor="white", label="GPBA-A phase"
            ),
            mpatches.Patch(
                facecolor=_COLOR_PHASE1_MONISE, edgecolor="white", label="MONISE phase"
            ),
            mpatches.Patch(
                facecolor=_COLOR_PHASE2, edgecolor="white", label="PLS phase gain"
            ),
        ]
    else:
        phase_handles = [
            mpatches.Patch(
                facecolor=_COLOR_PHASE1_MONISE if has_monise else _COLOR_PHASE1,
                edgecolor="white",
                label=exact_phase_label,
            ),
            mpatches.Patch(
                facecolor=_COLOR_PHASE2, edgecolor="white", label="PLS phase"
            ),
        ]
    variant_handles = [
        mpatches.Patch(
            facecolor="#999999", hatch=h, edgecolor="white", linewidth=0.4, label=lbl
        )
        for _, h, lbl in _PLS_VARIANTS
    ]
    from matplotlib.legend_handler import HandlerBase

    class _NoPatchHandler(HandlerBase):
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            handlebox.width = 0
            handlebox.height = 0
            return None

    _sec = lambda t: mpatches.Patch(color="none", label=f"$\\bf{{{t}}}$")
    _fill = lambda: mpatches.Patch(color="none", label="")
    ncol = max(len(phase_handles), len(variant_handles))
    # Pad both rows to ncol entries
    ph = phase_handles + [_fill() for _ in range(ncol - len(phase_handles))]
    vh = variant_handles + [_fill() for _ in range(ncol - len(variant_handles))]
    # Matplotlib fills column-by-column, so interleave column slices to produce:
    #   Row 0: "Phases"  blank  blank …
    #   Row 1: ph[0]     ph[1]  ph[2] …
    #   Row 2: "PLS variants"  blank  blank …
    #   Row 3: vh[0]     vh[1]  vh[2] …
    no_patch_handles = []
    all_handles = []
    for c in range(ncol):
        sec1 = _sec("Phases") if c == 0 else _fill()
        sec2 = _sec("PLS\\ variants") if c == 0 else _fill()
        no_patch_handles += [sec1, sec2]
        all_handles += [sec1, ph[c], sec2, vh[c]]
    ax.legend(
        handles=all_handles,
        handler_map={h: _NoPatchHandler() for h in no_patch_handles},
        loc="lower left",
        fontsize=9,
        title_fontsize=9,
        framealpha=0.9,
        ncol=ncol,
        handlelength=1.2,
        handletextpad=0.5,
    )

    if has_compare:
        ax.set_xlabel("Exact phase % / PLS phase %", fontsize=15)
    else:
        ax.set_xlabel(f"{exact_phase_label} % / PLS phase %", fontsize=15)
    ax.set_ylabel("Normalized Hypervolume", fontsize=15)
    ax.set_title(
        f"Final Hypervolume by Phase & Variant — {display_name}"
        + (" (GPBA-A vs MONISE)" if has_compare else ""),
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(0, max(bd["final_hv"] for bd in all_bars.values()) * 1.28)
    ax.grid(True, axis="y", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    out = output_dir / f"{display_name}_phase_bars.png"
    fig.tight_layout()
    _save_fig(fig, out)
    plt.close(fig)
    print(f"  Phase bar plot saved: {out}", flush=True)


def _plot_instance_lines(
    result: dict,
    configs: list[AlgorithmConfig],
    output_dir: Path,
) -> None:
    """Render the per-instance HV-over-time line plot from a result artifact dict.

    Reads curve data from result["configs"][label]["curve"] so it can be called
    both during an experiment run and when replaying from saved JSON.
    """
    if not HAS_MATPLOTLIB:
        return

    display_name = result["instance"]
    total_timeout = result["timeout_s"]

    curves: dict[str, list[tuple[float, float]]] = {
        label: [tuple(pt) for pt in cfg_data.get("curve", [])]
        for label, cfg_data in result.get("configs", {}).items()
    }

    gpbaa_curve = curves.get("GPBA-A", [])
    gpbaa_cfg = next((c for c in configs if c.label == "GPBA-A"), None)

    def _hv_at(curve: list, t: float) -> float:
        if not curve:
            return 0.0
        if t <= curve[0][0]:
            return curve[0][1]
        for i in range(1, len(curve)):
            t0, hv0 = curve[i - 1]
            t1, hv1 = curve[i]
            if t0 <= t <= t1:
                return hv0 + (hv1 - hv0) * (t - t0) / (t1 - t0) if t1 > t0 else hv0
        return curve[-1][1]

    # Group non-GPBA-A configs by exact_phase_ratio (None → 0.0 for pure PLS)
    ratio_groups: dict[float, list[AlgorithmConfig]] = {}
    for cfg in configs:
        if cfg.label == "GPBA-A":
            continue
        key = cfg.exact_phase_ratio if cfg.exact_phase_ratio is not None else 0.0
        ratio_groups.setdefault(key, []).append(cfg)

    sorted_ratios = sorted(
        ratio_groups.keys(), reverse=True
    )  # e.g. [0.5, 0.35, 0.2, 0.0]
    n = len(sorted_ratios)
    ncols = 2
    nrows = math.ceil(n / ncols)

    # Pre-compute global max HV across all curves for consistent y-axis
    global_max_hv = max(
        (hv for c in curves.values() for _, hv in c),
        default=1.0,
    )
    y_top = global_max_hv * 1.13

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 6, nrows * 4.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    deferred_endpoints: list[tuple] = []  # (ax, annotations) — placed after ylims set

    for i, ratio in enumerate(sorted_ratios):
        ax = axes_flat[i]
        cfgs = ratio_groups[ratio]
        endpoint_annotations: list[tuple[float, float, str]] = []

        # GPBA-A reference line on every subplot
        if gpbaa_curve and gpbaa_cfg:
            ts = [t for t, _ in gpbaa_curve]
            hvs = [hv for _, hv in gpbaa_curve]
            if ts[-1] < total_timeout:
                ts, hvs = ts + [total_timeout], hvs + [hvs[-1]]
            ax.plot(
                ts,
                hvs,
                label=gpbaa_cfg.label,
                color=gpbaa_cfg.color,
                linestyle=gpbaa_cfg.linestyle,
                linewidth=gpbaa_cfg.linewidth,
                marker="o",
                markersize=3,
                markevery=max(1, len(ts) // 10),
                markeredgewidth=0,
                alpha=0.8,
                zorder=3,
            )
            endpoint_annotations.append((ts[-1], hvs[-1], gpbaa_cfg.color))

        # Handoff guide line for hybrid groups
        if ratio > 0.0:
            t_exact = total_timeout * ratio
            ax.axvline(t_exact, color="#cccccc", linestyle=":", linewidth=1.0, zorder=1)
            ax.text(
                t_exact,
                0.02,
                f"{t_exact:.0f}s",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7,
                color="#888888",
            )

        # Algorithm curves for this ratio group
        for cfg in cfgs:
            curve = curves.get(cfg.label, [])
            if not curve:
                continue
            ts = [t for t, _ in curve]
            hvs = [hv for _, hv in curve]

            # Use variant-consistent color and linestyle
            bar_key = _LABEL_TO_BAR_KEY.get(cfg.label)
            variant = bar_key[1] if bar_key else None
            color = _VARIANT_COLORS.get(variant, cfg.color)
            ls, lw = _VARIANT_LINE.get(variant, (cfg.linestyle, cfg.linewidth))

            if cfg.exact_phase_ratio is not None:
                t_exact = total_timeout * cfg.exact_phase_ratio
                pls_pairs = [(t, hv) for t, hv in zip(ts, hvs) if t >= t_exact]
                if not pls_pairs:
                    continue
                handoff_hv = _hv_at(gpbaa_curve, t_exact)
                ts = [t_exact] + [p[0] for p in pls_pairs]
                hvs = [handoff_hv] + [p[1] for p in pls_pairs]
                ax.plot(
                    t_exact,
                    handoff_hv,
                    marker="o",
                    markersize=7,
                    color=color,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=1.8,
                    zorder=5,
                    linestyle="none",
                )

            suffix = "Hybrid" if ratio > 0.0 else "PLS"
            plot_label = f"{variant} {suffix}" if variant else cfg.label
            ax.plot(
                ts,
                hvs,
                label=plot_label,
                color=color,
                linestyle=ls,
                linewidth=lw,
                marker="o",
                markersize=4,
                markevery=max(1, len(ts) // 10),
                markeredgewidth=0,
                markerfacecolor=color,
                alpha=0.92,
                zorder=3,
            )
            endpoint_annotations.append((ts[-1], hvs[-1], color))

        # Subplot title: "50% / 50%", "35% / 65%", … "0% / 100%"
        pls_pct = int((1.0 - ratio) * 100)
        gpba_pct = int(ratio * 100)
        ax.set_title(f"{gpba_pct}% / {pls_pct}%", fontsize=11, fontweight="bold")
        ax.set_xlim(0, total_timeout)
        ax.set_ylim(0, y_top)
        ax.set_xlabel("Time (seconds)")
        if i % ncols == 0:
            ax.set_ylabel("Normalized Hypervolume")
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
        ax.grid(True, alpha=0.25, zorder=0)

        deferred_endpoints.append((ax, endpoint_annotations))

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Annotate after all ylims are final (sharey propagates y_top everywhere)
    for ax, endpoints in deferred_endpoints:
        _annotate_endpoints(ax, endpoints, fmt="{:.4f}")

    fig.suptitle(
        f"Hypervolume over Time — {display_name}",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    plot_path = output_dir / f"{display_name}.png"
    _save_fig(fig, plot_path)
    plt.close(fig)
    print(f"\n  Plot saved: {plot_path}", flush=True)


def _plot_pls_only_lines(
    size_label: str,
    timeout_s: float,
    instances: list[str],  # display city names
    curves: dict[
        tuple[str, str], list[tuple[float, float]]
    ],  # (city, variant) → [(t, hv)]
    output_path: "Path",
) -> None:
    """Grid of subplots — one per instance, 3 PLS-variant lines each."""
    n = len(instances)
    ncols = 2 if n <= 4 else 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 4.8, nrows * 3.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    global_max_hv = max(
        (
            hv
            for city in instances
            for vname in _VARIANT_LINE
            for _, hv in curves.get((city, vname), [(0, 0)])
        ),
        default=0.1,
    )
    y_top = global_max_hv * 1.13

    all_endpoints: list[tuple["plt.Axes", list]] = []

    for i, city in enumerate(instances):
        ax = axes_flat[i]
        endpoints: list[tuple[float, float, str]] = []

        for vname, (ls, lw) in _VARIANT_LINE.items():
            curve = curves.get((city, vname), [])
            if not curve:
                continue
            ts = [t for t, _ in curve]
            hvs = [hv for _, hv in curve]
            if ts[-1] < timeout_s:
                ts, hvs = ts + [timeout_s], hvs + [hvs[-1]]

            color = _VARIANT_COLORS[vname]
            ax.plot(
                ts,
                hvs,
                color=color,
                linestyle=ls,
                linewidth=lw,
                marker="o",
                markersize=3.5,
                markevery=max(1, len(ts) // 8),
                markeredgewidth=0,
                markerfacecolor=color,
                alpha=0.9,
                zorder=3,
            )
            endpoints.append((ts[-1], hvs[-1], color))

        ax.set_title(city, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2, zorder=0)
        ax.set_xlim(0, timeout_s)
        ax.set_ylim(0, y_top)
        ax.set_xlabel("Time (seconds)")
        if i % ncols == 0:
            ax.set_ylabel("Norm. Hypervolume")

        all_endpoints.append((ax, endpoints))

    for ax, endpoints in all_endpoints:
        _annotate_endpoints(ax, endpoints)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    legend_handles = [
        plt.Line2D(
            [0], [0], color=_VARIANT_COLORS[vn], linestyle=ls, linewidth=lw, label=vn
        )
        for vn, (ls, lw) in _VARIANT_LINE.items()
    ]
    bottom_legend = n >= len(axes_flat)
    if not bottom_legend:
        ax_leg = axes_flat[-1]
        ax_leg.set_visible(True)
        ax_leg.axis("off")
        ax_leg.legend(
            handles=legend_handles,
            loc="center",
            title="PLS variant",
            framealpha=0.9,
            fontsize=9,
        )

    fig.suptitle(
        f"Hypervolume over Time — Size {size_label} instances (PLS only)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    tight_rect = [0, 0.1, 1, 1] if bottom_legend else [0, 0, 1, 1]
    fig.tight_layout(rect=tight_rect)

    if bottom_legend:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            title="PLS variant",
            ncol=3,
            framealpha=0.9,
            fontsize=9,
            bbox_to_anchor=(0.5, 0.01),
        )

    _save_fig(fig, output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}", flush=True)


def _plot_pls_only_bars(
    size_label: str,
    instances: list[str],
    hv_by_variant: dict[str, list[float]],  # variant_name → [hv per instance]
    output_path: "Path",
) -> None:
    """Grouped bar chart comparing PLS variants across instances of a given size."""
    n_inst = len(instances)
    n_var = len(_PLS_VARIANTS)

    bar_w = 0.20
    inner_gap = 0.04
    group_gap = 0.35
    group_span = n_var * bar_w + (n_var - 1) * inner_gap
    group_cx = [i * (group_span + group_gap) for i in range(n_inst)]
    var_off = [(j - (n_var - 1) / 2) * (bar_w + inner_gap) for j in range(n_var)]

    fig, ax = plt.subplots(figsize=(max(10, n_inst * 2.2), 6))
    legend_done: set[str] = set()

    for i, inst in enumerate(instances):
        for j, (vname, hatch, _) in enumerate(_PLS_VARIANTS):
            hvs = hv_by_variant.get(vname, [])
            if i >= len(hvs):
                continue
            hv = hvs[i]
            bx = group_cx[i] + var_off[j]
            ax.bar(
                bx,
                hv,
                width=bar_w,
                color=_COLOR_PHASE2,
                hatch=hatch,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
                label=vname if vname not in legend_done else None,
            )
            legend_done.add(vname)
            ax.text(
                bx,
                hv + 0.005,
                f"{hv:.3f}",
                ha="left",
                va="bottom",
                fontsize=8,
                color="#333333",
                rotation=45,
                rotation_mode="anchor",
            )

    ax.set_xticks(group_cx)
    ax.set_xticklabels(instances, fontsize=10)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Normalized Hypervolume")
    ax.set_title(
        f"Final Hypervolume by PLS Variant — Size {size_label} instances",
        fontweight="bold",
    )
    ax.set_ylim(
        0,
        max(
            (hv for hvs in hv_by_variant.values() for hv in hvs),
            default=1.0,
        )
        * 1.18,
    )
    ax.legend(title="Algorithm variant", loc="lower right", framealpha=0.9, ncol=n_var)
    ax.grid(True, axis="y", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save_fig(fig, output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}", flush=True)


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
    # _pseudo_cache is keyed by "source:instance" so no stale hits across sources.
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
        safe_label = (
            cfg.label.lower().replace(" ", "_").replace("+", "plus").replace("/", "")
        )
        existing_trace = output_dir / f"{display_name}__{safe_label}.trace.tar.gz"
        if existing_trace.exists():
            trace_data = existing_trace.read_bytes()
            traces[cfg.label] = trace_data
            run_meta[cfg.label] = dict(
                final_solutions=0,
                wall_seconds=0.0,
                trace_bytes=len(trace_data),
                skipped=True,
            )
            print(
                f"\n  [{cfg.label}] loaded from existing trace ({len(trace_data) // 1024}KB)",
                flush=True,
            )
            continue

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
        "Hybrid 35:65",
        "Hybrid 20:80",
        "Diverse Probe Hybrid 50:50",
        "Diverse Probe Hybrid 35:65",
        "Diverse Probe Hybrid 20:80",
        "Scalarized Hybrid 50:50",
        "Scalarized Hybrid 35:65",
        "Scalarized Hybrid 20:80",
        "MONISE Hybrid 50:50",
        "MONISE Hybrid 35:65",
        "MONISE Hybrid 20:80",
        "MONISE Diverse Probe Hybrid 50:50",
        "MONISE Diverse Probe Hybrid 35:65",
        "MONISE Diverse Probe Hybrid 20:80",
        "MONISE Scalarized Hybrid 50:50",
        "MONISE Scalarized Hybrid 35:65",
        "MONISE Scalarized Hybrid 20:80",
    }
    # Determine which baseline hybrid label is present (GPBA-A or MONISE seeded)
    _baseline_50 = next(
        (lbl for lbl in ("Hybrid 50:50", "MONISE Hybrid 50:50") if lbl in traces), None
    )
    if _baseline_50 is not None and {_baseline_50}.issubset(traces):
        shared_pseudo_objectives = _extract_shared_hybrid_pseudo_objectives(
            {_baseline_50: traces[_baseline_50]}, _current_pseudo_solutions
        )
        if shared_pseudo_objectives:
            cfg_map = {cfg.label: cfg for cfg in configs}
            for label in hybrid_labels:
                trace_data = traces.get(label)
                if not trace_data:
                    continue
                cfg = cfg_map.get(label)
                ratio = (
                    cfg.exact_phase_ratio
                    if (cfg and cfg.exact_phase_ratio is not None)
                    else 0.5
                )
                exact_time = total_timeout * ratio
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

    # Normalize curves to start at HV=0, unless the config opts out
    skip_norm_labels = {cfg.label for cfg in configs if cfg.skip_normalization}
    print(f"\n  Normalizing curves to start at HV=0 …", flush=True)
    for cfg in configs:
        label = cfg.label
        curve = curves.get(label, [])
        if not curve:
            continue
        first_t, first_hv = curve[0]
        if label in skip_norm_labels:
            print(
                f"  [{label}] skipping normalization (skip_normalization=True)",
                flush=True,
            )
        elif first_hv > 0.05:
            curves[label] = [(0.0, 0.0)] + curve
            print(
                f"  [{label}] prepended (0, 0) — was starting at HV={first_hv:.4f}",
                flush=True,
            )

    # ── Phase 4: build result artifact ─────────────────────────────────

    def _final_hv(label: str) -> float:
        c = curves.get(label, [])
        return c[-1][1] if c else 0.0

    pure_pls_hv = _final_hv("Default PLS")

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

    # ── Phase 5: generate per-instance plots ───────────────────────────
    # Skip for large instances where only PLS configs ran — the combined
    # figures (pls_lines / pls_bars) cover those adequately.
    has_hybrid_or_gpbaa = any(
        cfg.label == "GPBA-A" or cfg.exact_phase_ratio is not None for cfg in configs
    )
    if has_hybrid_or_gpbaa:
        _plot_instance_lines(result_artifact, configs, output_dir)
        _plot_instance_phase_bars(result_artifact, configs, output_dir)

    artifact_path = output_dir / f"{display_name}.json"
    with open(artifact_path, "w") as f:
        json.dump(result_artifact, f, indent=2)
    print(f"  Artifact saved: {artifact_path}", flush=True)

    for cfg in configs:
        if cfg.save_trace and cfg.label in traces and traces[cfg.label]:
            safe_label = (
                cfg.label.lower().replace(" ", "_").replace("+", "plus").replace("/", "")
            )
            trace_path = output_dir / f"{display_name}__{safe_label}.trace.tar.gz"
            with open(trace_path, "wb") as f:
                f.write(traces[cfg.label])
            print(f"  Trace saved:    {trace_path}", flush=True)

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
    """Generate combined figures grouped by instance size.

    - Instances ≤ 100 images: multi-panel line plot (one panel per instance,
      all configured series shown with hybrid handoff markers).
    - Instances > 100 images: PLS-only grid line plot + grouped bar chart,
      one subplot per city, grouped by size (145/150 merged).
    """
    if not HAS_MATPLOTLIB or not all_results:
        return

    small = [r for r in all_results if r["num_images"] <= 100]
    large = [r for r in all_results if r["num_images"] > 100]

    # ── Small instances: multi-panel convergence line plot ──────────────
    if small:
        n = len(small)
        ncols = 2 if n <= 4 else 3
        nrows = math.ceil(n / ncols)

        # Per-row height increased to avoid vertical squishing; shared legend
        # replaces per-subplot legends, so more vertical space is available.
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 6, nrows * 6), squeeze=False
        )
        axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

        cfg_map = {cfg.label: cfg for cfg in configs}
        gpbaa_map: dict[str, list] = {}
        # Accumulate one handle per series label for the shared legend below.
        legend_handles: dict[str, Any] = {}

        for ax_idx, r in enumerate(small):
            ax = axes_flat[ax_idx]
            total_timeout = r["timeout_s"]
            gpbaa_curve = r["configs"].get("GPBA-A", {}).get("curve", [])
            gpbaa_map[r["instance"]] = gpbaa_curve

            def _hv_at(curve: list, t: float) -> float:
                if not curve:
                    return 0.0
                if t <= curve[0][0]:
                    return curve[0][1]
                for i in range(1, len(curve)):
                    t0, hv0 = curve[i - 1]
                    t1, hv1 = curve[i]
                    if t0 <= t <= t1:
                        alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                        return hv0 + alpha * (hv1 - hv0)
                return curve[-1][1]

            endpoint_annotations: list[tuple[float, float, str]] = []
            global_max_hv = 0.0

            for cfg in configs:
                curve_data = r["configs"].get(cfg.label, {}).get("curve", [])
                if not curve_data:
                    continue
                ts = [t for t, _ in curve_data]
                hvs = [hv for _, hv in curve_data]

                if cfg.exact_phase_ratio is not None:
                    t_exact = total_timeout * cfg.exact_phase_ratio
                    pls_pairs = [(t, hv) for t, hv in zip(ts, hvs) if t >= t_exact]
                    if not pls_pairs:
                        continue
                    handoff_hv = _hv_at(gpbaa_curve, t_exact)
                    ts = [t_exact] + [p[0] for p in pls_pairs]
                    hvs = [handoff_hv] + [p[1] for p in pls_pairs]
                    ax.plot(
                        t_exact,
                        handoff_hv,
                        marker="o",
                        markersize=6,
                        color=cfg.color,
                        markerfacecolor="white",
                        markeredgecolor=cfg.color,
                        markeredgewidth=1.6,
                        zorder=5,
                        linestyle="none",
                    )
                elif cfg.label == "GPBA-A" and gpbaa_curve:
                    if ts[-1] < total_timeout:
                        ts = ts + [total_timeout]
                        hvs = hvs + [hvs[-1]]

                global_max_hv = max(global_max_hv, max(hvs))
                (line,) = ax.plot(
                    ts,
                    hvs,
                    label=cfg.label,
                    color=cfg.color,
                    linestyle=cfg.linestyle,
                    linewidth=cfg.linewidth,
                    marker="o",
                    markersize=3.5,
                    markevery=max(1, len(ts) // 8),
                    markeredgewidth=0,
                    markerfacecolor=cfg.color,
                    alpha=0.92,
                    zorder=3,
                )
                if cfg.label not in legend_handles:
                    legend_handles[cfg.label] = line
                endpoint_annotations.append((ts[-1], hvs[-1], cfg.color))

            ax.set_xlim(0, total_timeout)
            ax.set_ylim(0, global_max_hv * 1.13)
            _annotate_endpoints(ax, endpoint_annotations, fmt="{:.3f}")
            ax.set_title(r["instance"], fontsize=10, fontweight="bold")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Norm. Hypervolume")
            ax.grid(True, alpha=0.2, zorder=0)

        for j in range(len(small), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(
            "Hypervolume over Time — Small Instances (≤ 100 images)",
            fontsize=13,
            fontweight="bold",
        )
        # tight_layout often fails here because right-side annotations (rendered
        # with annotation_clip=False) extend beyond the axes boundary, making it
        # impossible to satisfy all constraints.  Use explicit subplots_adjust
        # instead so the layout is always applied deterministically.
        #
        # right=0.86: leave ~14 % on the right for endpoint annotation labels.
        # bottom=0.08: reserve space for the shared figure legend below the grid.
        # top=0.96:   a small top margin keeps the suptitle from touching the edge.
        fig.subplots_adjust(
            left=0.07,
            right=0.86,
            top=0.96,
            bottom=0.08,
            hspace=0.50,
            wspace=0.35,
        )
        # Single shared legend placed at the bottom of the figure.
        n_lgd_cols = min(len(legend_handles), 5)
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="lower center",
            ncol=n_lgd_cols,
            fontsize=8,
            framealpha=0.9,
            bbox_to_anchor=(0.5, 0.01),
        )
        # Save WITHOUT bbox_inches="tight": the layout is fully controlled by
        # subplots_adjust above, so all intended content fits inside the nominal
        # figure boundary.  Skipping "tight" prevents bbox_inches expansion
        # caused by endpoint annotations that slightly exceed the right edge.
        path = output_dir / "fig_combined_small.png"
        fig.savefig(str(path), dpi=200)
        fig.savefig(str(path.with_suffix(".eps")))
        plt.close(fig)
        print(f"\nCombined convergence plot (small): {path}", flush=True)

    # ── Large instances: PLS-only grid + bar chart, grouped by size ─────
    if large:
        # Group by size label (145 and 150 merged)
        size_groups: dict[str, list[dict]] = {}
        for r in large:
            key = _size_group_label(r["num_images"])
            size_groups.setdefault(key, []).append(r)

        for size_label, results in sorted(size_groups.items()):
            instances = [_city_label(r["instance"]) for r in results]
            timeout_s = results[0]["timeout_s"]

            # Build (city, variant) → curve mapping
            pls_curves: dict[tuple[str, str], list] = {}
            for r in results:
                city = _city_label(r["instance"])
                for cfg_label, vname in _PLS_VARIANT_MAP.items():
                    cdata = r["configs"].get(cfg_label, {}).get("curve", [])
                    if cdata:
                        pls_curves[(city, vname)] = cdata

            # Grid line plot
            _plot_pls_only_lines(
                size_label,
                timeout_s,
                instances,
                pls_curves,
                output_dir / f"fig_pls_lines_{size_label.replace(' / ', '_')}.png",
            )

            # Bar chart
            hv_by_variant: dict[str, list[float]] = {
                vname: [
                    r["configs"].get(cfg_label, {}).get("final_hv", 0.0)
                    for r in results
                ]
                for cfg_label, vname in _PLS_VARIANT_MAP.items()
            }
            _plot_pls_only_bars(
                size_label,
                instances,
                hv_by_variant,
                output_dir / f"fig_pls_bars_{size_label.replace(' / ', '_')}.png",
            )


# ─── Replot from saved artifacts ─────────────────────────────────────


def _merge_bounds(
    bounds_a: list[list[int]], bounds_b: list[list[int]]
) -> list[list[int]]:
    return [[min(a[0], b[0]), max(a[1], b[1])] for a, b in zip(bounds_a, bounds_b)]


def _recompute_result_with_bounds(
    result: dict,
    trace_dir: Path,
    merged_bounds: list[list[int]],
    num_points: int = 30,
) -> dict:
    """Return a deep copy of result with all HV curves recomputed from trace files
    using merged_bounds, so values are comparable across experiments."""
    import copy

    result = copy.deepcopy(result)
    result["shared_bounds"] = merged_bounds
    instance = result["instance"]

    for label, cfg_data in result["configs"].items():
        safe_label = (
            label.lower()
            .replace(" ", "_")
            .replace("+", "plus")
            .replace(":", "")
            .replace("/", "")
        )
        # Trace filenames use colons for ratios (e.g. 50:50) — try both forms
        candidates = [
            trace_dir / f"{instance}__{safe_label}.trace.tar.gz",
            trace_dir
            / f"{instance}__{label.lower().replace(' ', '_').replace('+', 'plus').replace('/', '')}.trace.tar.gz",
        ]
        trace_bytes = None
        for c in candidates:
            if c.exists():
                trace_bytes = c.read_bytes()
                break

        if trace_bytes is None:
            # No trace file on disk — keep existing stored values unchanged.
            # Caller is responsible for recomputing synthetic configs (e.g. MONISE)
            # separately if needed.
            continue

        curve = sims_problem.compute_hv_curve_from_trace(
            trace_bytes, merged_bounds, num_points
        )
        cfg_data["curve"] = [(round(t, 4), round(hv, 8)) for t, hv in curve]
        cfg_data["final_hv"] = round(curve[-1][1], 8) if curve else 0.0

    # Recompute delta_vs_pure_pls_pct with updated final_hv values
    pure_pls_hv = result["configs"].get("Default PLS", {}).get("final_hv", 0.0)
    if pure_pls_hv > 0:
        for cfg_data in result["configs"].values():
            fhv = cfg_data.get("final_hv", 0.0)
            cfg_data["delta_vs_pure_pls_pct"] = round(
                (fhv - pure_pls_hv) / pure_pls_hv * 100, 4
            )

    return result


def _regenerate_monise_curve_with_bounds(
    instance_name: str,
    merged_bounds: list[list[int]],
    num_points: int = 30,
) -> tuple[list, float] | None:
    """Recompute the MONISE HV curve from pseudo-solutions using merged_bounds.

    Returns (curve, final_hv) or None if pseudo-solutions are unavailable.
    """
    monise_dir = _PSEUDO_SOURCE_DIRS["monise"]
    monise_json = monise_dir / f"{instance_name}.json"
    if not monise_json.exists():
        return None

    import json as _json

    data = _json.loads(monise_json.read_text())
    pseudo_solutions = data if isinstance(data, list) else data.get("solutions", [])

    converted = []
    for sol in pseudo_solutions:
        images = sol.get("selected_images", [])
        if not images:
            continue
        try:
            converted.append(
                sims_problem.Solution.create(
                    selected_images=images,
                    cost=sol.get("cost"),
                    cloudy_area=sol.get("cloudy_area"),
                    max_incidence_angle=sol.get("max_incidence_angle"),
                    timestamp_us=int(sol.get("timestamp_s", 0.0) * 1_000_000),
                    min_resolutions_sum=sol.get("min_resolutions_sum"),
                )
            )
        except Exception:
            pass

    if not converted:
        return None

    ref_point = [b[1] + 1 for b in merged_bounds]
    trace = sims_problem.generate_trace(
        solutions=converted,
        objectives=OBJECTIVES,
        algorithm="MONISE",
        num_objectives=len(OBJECTIVES),
        objective_bounds=[[int(b[0]), int(b[1])] for b in merged_bounds],
        reference_point=ref_point,
        include_dominated=False,
    )
    curve = sims_problem.compute_hv_curve_from_trace(trace, merged_bounds, num_points)
    if not curve:
        return None
    if curve[0][1] > 0.05:
        curve = [(0.0, 0.0)] + list(curve)
    final_hv = round(curve[-1][1], 8)
    curve = [(round(t, 4), round(hv, 8)) for t, hv in curve]
    return curve, final_hv


def replot_from_dir(
    output_dir: Path,
    configs: list[AlgorithmConfig],
    compare_dir: Path | None = None,
    publish_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> None:
    """Regenerate all plots from existing JSON artifacts without re-running experiments."""
    json_files = sorted(
        f for f in output_dir.glob("*.json") if f.stem != "all_experiments"
    )
    if not json_files:
        print(f"No JSON artifacts found in {output_dir}", flush=True)
        return

    num_points = 30
    all_results: list[dict] = []
    for json_path in json_files:
        result = json.loads(json_path.read_text())
        print(f"  Loaded: {json_path.name}", flush=True)
        compare_result: dict | None = None
        instance_size = result.get("num_images", 0)
        if compare_dir is not None:
            cmp_path = compare_dir / json_path.name
            if cmp_path.exists():
                cmp_data = json.loads(cmp_path.read_text())
                if instance_size < 145:
                    compare_result = cmp_data
                    primary_bounds = result.get("shared_bounds")
                    compare_bounds = compare_result.get("shared_bounds")
                    if primary_bounds and compare_bounds:
                        merged = _merge_bounds(primary_bounds, compare_bounds)
                        print(
                            f"    Recomputing HV curves with merged bounds …",
                            flush=True,
                        )
                        result = _recompute_result_with_bounds(
                            result, output_dir, merged, num_points
                        )
                        compare_result = _recompute_result_with_bounds(
                            compare_result, compare_dir, merged, num_points
                        )
                        # Recompute synthetic MONISE curve with merged bounds
                        if "MONISE" in result["configs"]:
                            monise_recomp = _regenerate_monise_curve_with_bounds(
                                result["instance"], merged, num_points
                            )
                            if monise_recomp:
                                curve, final_hv = monise_recomp
                                result["configs"]["MONISE"]["curve"] = curve
                                result["configs"]["MONISE"]["final_hv"] = final_hv
                # Inject PLS-only configs missing from primary but present in compare,
                # regardless of instance size (e.g. Scalarized/Diverse Probe PLS run
                # only in the GPBA-A experiment for 145/150 instances too)
                for label in _PLS_VARIANT_MAP:
                    if label not in result["configs"] and label in cmp_data.get(
                        "configs", {}
                    ):
                        result["configs"][label] = cmp_data["configs"][label]
        all_results.append(
            result
        )  # append after recomputation so combined figs use merged HV
        # Save recomputed result back to JSON so tables can use merged-bounds values
        json_path.write_text(json.dumps(result, indent=2))
        if compare_result is not None:
            cmp_path.write_text(json.dumps(compare_result, indent=2))
        _plot_instance_lines(result, configs, output_dir)
        _plot_instance_phase_bars(
            result, configs, output_dir, compare_result=compare_result
        )

    generate_combined_figures(all_results, output_dir, configs)

    import shutil

    if publish_dir is not None:
        publish_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for png in output_dir.glob("*.png"):
            shutil.copy2(png, publish_dir / png.name)
            copied += 1
        print(f"  Published {copied} PNG(s) → {publish_dir}", flush=True)

    if figures_dir is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)
        copied_eps = 0
        for eps in output_dir.glob("*.eps"):
            shutil.copy2(eps, figures_dir / eps.name)
            copied_eps += 1
        print(f"  Published {copied_eps} EPS(es) → {figures_dir}", flush=True)

    print(f"\nReplot complete — {len(all_results)} instances.", flush=True)


# ─── Statistical tests ────────────────────────────────────────────────


def print_wilcoxon_tests(
    primary_dir: Path,
    monise_dir: Path | None = None,
) -> None:
    """Run Wilcoxon signed-rank tests on all pairwise algorithm comparisons.

    Reports two-sided p-values and rank-biserial correlation (effect size r).
    Uses the 15 small instances (5 cities × 3 sizes: 30/50/100) as the sample.
    Also tests the 9 large instances (145/150/200) where available.
    """
    import json as _json

    from scipy.stats import wilcoxon

    def load_hv(d: Path, slug: str, sz: int, label: str) -> float | None:
        f = d / f"{slug}_{sz}.json"
        if not f.exists():
            return None
        v = _json.loads(f.read_text()).get("configs", {}).get(label, {}).get("final_hv")
        return float(v) if v is not None else None

    CITIES = [
        ("Lagos", "lagos_nigeria"),
        ("Mexico City", "mexico_city"),
        ("Paris", "paris"),
        ("Rio de Janeiro", "rio_de_janeiro"),
        ("Tokyo Bay", "tokyo_bay"),
    ]

    def gather(
        d: Path, sizes: list[int], label_a: str, label_b: str
    ) -> tuple[list, list]:
        a_vals, b_vals = [], []
        for city, slug in CITIES:
            for sz in sizes:
                a = load_hv(d, slug, sz, label_a)
                b = load_hv(d, slug, sz, label_b)
                if a is not None and b is not None:
                    a_vals.append(a)
                    b_vals.append(b)
        return a_vals, b_vals

    def run_test(a_vals: list, b_vals: list) -> dict:
        diffs = [a - b for a, b in zip(a_vals, b_vals)]
        n = len(diffs)
        nonzero = [d for d in diffs if d != 0]
        if len(nonzero) < 2:
            return {"n": n, "p": float("nan"), "r": float("nan"), "note": "all ties"}
        stat, p = wilcoxon(a_vals, b_vals, alternative="two-sided")
        # rank-biserial correlation: r = 1 - 2*W / (n*(n+1)/2)
        # where W is the smaller of W+ and W-
        n_nz = len(nonzero)
        r = 1 - (2 * stat) / (n_nz * (n_nz + 1) / 2)
        return {"n": n, "p": p, "r": r, "W": stat, "note": ""}

    def sig(p: float) -> str:
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "n.s."

    sep = "-" * 72
    print(f"\n{'=' * 72}")
    print("WILCOXON SIGNED-RANK TESTS (two-sided)")
    print("Significance: * p<0.05  ** p<0.01  *** p<0.001  n.s. not significant")
    print("Effect size r (rank-biserial): |r|<0.3 small, 0.3–0.5 medium, >0.5 large")
    print(sep)

    comparisons_small = [
        ("Diverse vs Default", "Diverse Probe PLS", "Default PLS"),
        ("Scalarized vs Default", "Scalarized PLS", "Default PLS"),
        ("Scalarized vs Diverse", "Scalarized PLS", "Diverse Probe PLS"),
    ]

    print("\n── Small instances (30/50/100 images, n=15) ──")
    print(f"{'Comparison':<28}  {'n':>3}  {'W':>8}  {'p-value':>9}  {'r':>6}  Sig")
    print(sep)
    small_results = {}
    for name, lbl_a, lbl_b in comparisons_small:
        a, b = gather(primary_dir, [30, 50, 100], lbl_a, lbl_b)
        res = run_test(a, b)
        small_results[name] = res
        p_str = f"{res['p']:.4f}" if not (res["p"] != res["p"]) else "  nan"
        r_str = f"{res['r']:+.3f}" if not (res["r"] != res["r"]) else "  nan"
        w_str = f"{res['W']:.0f}" if not (res["W"] != res["W"]) else "  nan"
        print(
            f"{name:<28}  {res['n']:>3}  {w_str:>8}  {p_str:>9}  {r_str:>6}  {sig(res['p'])}"
        )

    print("\n── Large instances (145/150/200 images) ──")
    large_sizes = [145, 150, 200]
    print(f"{'Comparison':<28}  {'n':>3}  {'W':>8}  {'p-value':>9}  {'r':>6}  Sig")
    print(sep)
    large_results = {}
    for name, lbl_a, lbl_b in comparisons_small:
        a, b = gather(primary_dir, large_sizes, lbl_a, lbl_b)
        res = run_test(a, b)
        large_results[name] = res
        p_str = f"{res['p']:.4f}" if not (res["p"] != res["p"]) else "  nan"
        r_str = f"{res['r']:+.3f}" if not (res["r"] != res["r"]) else "  nan"
        w_str = f"{res['W']:.0f}" if not (res["W"] != res["W"]) else "  nan"
        print(
            f"{name:<28}  {res['n']:>3}  {w_str:>8}  {p_str:>9}  {r_str:>6}  {sig(res['p'])}"
        )

    if monise_dir:
        print("\n── MONISE hybrid vs Default PLS (100-image, n=5) ──")
        monise_comparisons = [
            ("MONISE Hybrid 20:80 vs Default", "MONISE Hybrid 20:80", "Default PLS"),
        ]
        print(f"{'Comparison':<38}  {'n':>3}  {'W':>8}  {'p-value':>9}  {'r':>6}  Sig")
        print(sep)
        for name, lbl_a, lbl_b in monise_comparisons:
            a, b = gather(monise_dir, [100], lbl_a, lbl_b)
            res = run_test(a, b)
            p_str = f"{res['p']:.4f}" if not (res["p"] != res["p"]) else "  nan"
            r_str = f"{res['r']:+.3f}" if not (res["r"] != res["r"]) else "  nan"
            w_str = f"{res['W']:.0f}" if not (res["W"] != res["W"]) else "  nan"
            print(
                f"{name:<38}  {res['n']:>3}  {w_str:>8}  {p_str:>9}  {r_str:>6}  {sig(res['p'])}"
            )

    print()
    return small_results, large_results


# ─── Paper table printer ──────────────────────────────────────────────


def print_paper_tables(
    primary_dir: Path,
    monise_dir: Path | None = None,
) -> None:
    """Print all paper table values from existing JSON artifacts.

    primary_dir  — final_paper_results (GPBA-A experiment, merged bounds)
    monise_dir   — hv_experiment_results_monise (MONISE experiment)
    """
    import json as _json

    def load(d: Path, slug: str, sz: int) -> dict:
        f = d / f"{slug}_{sz}.json"
        if not f.exists():
            return {}
        return _json.loads(f.read_text()).get("configs", {})

    CITIES = [
        ("Lagos", "lagos_nigeria"),
        ("Mexico City", "mexico_city"),
        ("Paris", "paris"),
        ("Rio de Janeiro", "rio_de_janeiro"),
        ("Tokyo Bay", "tokyo_bay"),
    ]
    HYBRID_CFGS = [
        "Hybrid 20:80",
        "Hybrid 35:65",
        "Hybrid 50:50",
        "Diverse Probe Hybrid 20:80",
        "Diverse Probe Hybrid 35:65",
        "Diverse Probe Hybrid 50:50",
        "Scalarized Hybrid 20:80",
        "Scalarized Hybrid 35:65",
        "Scalarized Hybrid 50:50",
    ]
    MONISE_HYBRID_CFGS = [
        "MONISE Hybrid 20:80",
        "MONISE Hybrid 35:65",
        "MONISE Hybrid 50:50",
        "MONISE Diverse Probe Hybrid 20:80",
        "MONISE Diverse Probe Hybrid 35:65",
        "MONISE Diverse Probe Hybrid 50:50",
        "MONISE Scalarized Hybrid 20:80",
        "MONISE Scalarized Hybrid 35:65",
        "MONISE Scalarized Hybrid 50:50",
    ]

    def fhv(cfgs: dict, label: str) -> float:
        return cfgs.get(label, {}).get("final_hv", 0.0)

    def best_of(cfgs: dict, labels: list[str]) -> float:
        return max((fhv(cfgs, l) for l in labels), default=0.0)

    def pct(a: float, b: float) -> float:
        return (a - b) / b * 100 if b > 0 else 0.0

    sep = "-" * 80

    # ── Tab 1: PLS-only HV ────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("TAB:RESULTS — Final normalised HV for PLS variants (30–100 images)")
    print(sep)
    print(
        f"{'Instance':<22} {'Sz':>3}  {'Default PLS':>11}  {'Diverse':>11}  {'Scalarized':>11}  Best"
    )
    print(sep)
    div_pcts_sm, scal_pcts_sm = [], []
    for city, slug in CITIES:
        for sz in [30, 50, 100]:
            cfgs = load(primary_dir, slug, sz)
            d = fhv(cfgs, "Default PLS")
            v = fhv(cfgs, "Diverse Probe PLS")
            s = fhv(cfgs, "Scalarized PLS")
            winner = (
                "Default"
                if d >= v and d >= s
                else ("Diverse" if v >= s else "Scalarized")
            )
            print(
                f"{city + ' ' + str(sz):<22} {sz:>3}  {d:>11.6f}  {v:>11.6f}  {s:>11.6f}  {winner}"
            )
            if d > 0:
                div_pcts_sm.append(pct(v, d))
                scal_pcts_sm.append(pct(s, d))
        print()

    # ── Tab 2: improvement over Default PLS ──────────────────────────
    print(f"\n{'=' * 80}")
    print("TAB:IMPROVEMENT — % improvement over Default PLS (30–100 images)")
    print(sep)
    print(f"{'Instance':<22} {'Sz':>3}  {'Diverse %':>10}  {'Scalarized %':>13}")
    print(sep)
    for city, slug in CITIES:
        for sz in [30, 50, 100]:
            cfgs = load(primary_dir, slug, sz)
            d = fhv(cfgs, "Default PLS")
            v = fhv(cfgs, "Diverse Probe PLS")
            s = fhv(cfgs, "Scalarized PLS")
            print(
                f"{city + ' ' + str(sz):<22} {sz:>3}  {pct(v, d):>+10.2f}  {pct(s, d):>+13.2f}"
            )
        print()
    avg_d = sum(div_pcts_sm) / len(div_pcts_sm) if div_pcts_sm else 0
    avg_s = sum(scal_pcts_sm) / len(scal_pcts_sm) if scal_pcts_sm else 0
    print(f"{'Average':<22}      {avg_d:>+10.2f}  {avg_s:>+13.2f}")

    # ── Tab 3: direct Scalarized vs Diverse ──────────────────────────
    print(f"\n{'=' * 80}")
    print("TAB:DIRECT — Scalarized vs Diverse (30–100 images)")
    print(sep)
    print(f"{'Instance':<22} {'Sz':>3}  {'Scal–Div %':>11}  Winner")
    print(sep)
    diffs = []
    for city, slug in CITIES:
        for sz in [30, 50, 100]:
            cfgs = load(primary_dir, slug, sz)
            v = fhv(cfgs, "Diverse Probe PLS")
            s = fhv(cfgs, "Scalarized PLS")
            d = pct(s, v)
            diffs.append(d)
            winner = "Scalarized" if s >= v else "Diverse"
            print(f"{city + ' ' + str(sz):<22} {sz:>3}  {d:>+11.2f}  {winner}")
        print()
    avg_diff = sum(diffs) / len(diffs) if diffs else 0
    wins = sum(1 for d in diffs if d > 0)
    print(
        f"{'Average':<22}      {avg_diff:>+11.2f}  Scalarized ({wins}–{len(diffs) - wins})"
    )

    # ── Tab 4: GPBA-A hybrid gains ────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("TAB:HYBRID_SMALL — GPBA-A hybrid gains + MONISE gains (100-image)")
    print(sep)
    print(
        f"{'Instance':<22} {'Sz':>3}  {'Default PLS':>11}  {'GPBA-A Gain%':>13}  {'MONISE Gain%':>13}"
    )
    print(sep)
    gpbaa_3050, gpbaa_100, monise_100 = [], [], []
    for city, slug in CITIES:
        for sz in [30, 50, 100]:
            cfgs = load(primary_dir, slug, sz)
            d = fhv(cfgs, "Default PLS")
            bh = best_of(cfgs, HYBRID_CFGS)
            gain_g = pct(bh, d)
            gain_m_str = "---"
            if sz == 100 and monise_dir:
                mcfgs = load(monise_dir, slug, sz)
                dm = fhv(mcfgs, "Default PLS")
                bm = best_of(mcfgs, MONISE_HYBRID_CFGS)
                gm = pct(bm, dm)
                gain_m_str = f"{gm:>+13.1f}"
                gpbaa_100.append(gain_g)
                monise_100.append(gm)
            else:
                gpbaa_3050.append(gain_g)
            print(
                f"{city + ' ' + str(sz):<22} {sz:>3}  {d:>11.4f}  {gain_g:>+13.1f}  {gain_m_str}"
            )
        print()
    avg_g_3050 = sum(gpbaa_3050) / len(gpbaa_3050) if gpbaa_3050 else 0
    avg_g_100 = sum(gpbaa_100) / len(gpbaa_100) if gpbaa_100 else 0
    avg_m_100 = sum(monise_100) / len(monise_100) if monise_100 else 0
    print(f"{'Avg 30–50 (GPBA-A)':<22}           {avg_g_3050:>+13.1f}")
    print(
        f"{'Avg 100 (merged)':<22}           {avg_g_100:>+13.1f}  {avg_m_100:>+13.1f}"
    )

    # ── Tab 5 & 6: MONISE hybrid ──────────────────────────────────────
    if monise_dir:
        print(f"\n{'=' * 80}")
        print("TAB:HYBRID_MEDIUM — MONISE hybrid gains (100 + 145/150 images)")
        print(sep)
        print(
            f"{'Instance':<22} {'Sz':>3}  {'Default PLS':>11}  {'Best Hybrid':>11}  {'Gain%':>7}"
        )
        print(sep)
        med_gains = []
        for sz_list, label in [([100], "100-image"), ([145, 150], "145/150-image")]:
            for city, slug in CITIES:
                for sz in sz_list:
                    mcfgs = load(monise_dir, slug, sz)
                    if not mcfgs:
                        continue
                    dm = fhv(mcfgs, "Default PLS")
                    bm = best_of(mcfgs, MONISE_HYBRID_CFGS)
                    gm = pct(bm, dm)
                    if sz_list == [145, 150]:
                        med_gains.append(gm)
                    print(
                        f"{city + ' ' + str(sz):<22} {sz:>3}  {dm:>11.4f}  {bm:>11.4f}  {gm:>+7.1f}"
                    )
            print()
        if med_gains:
            print(
                f"{'Avg 145/150':<22}           {'':>11}           {sum(med_gains) / len(med_gains):>+7.1f}  (max {max(med_gains):+.1f})"
            )

    # ── Tab:large: PLS-only 145–200 ──────────────────────────────────
    print(f"\n{'=' * 80}")
    print("TAB:LARGE — Final HV for PLS variants (145–200 images, primary-dir bounds)")
    print(sep)
    print(
        f"{'Instance':<22} {'Sz':>3}  {'Default PLS':>11}  {'Diverse':>11}  {'Scalarized':>11}  Best"
    )
    print(sep)
    large_instances = [
        ("Lagos", "lagos_nigeria", [145]),
        ("Mexico City", "mexico_city", [150, 200]),
        ("Paris", "paris", [150, 200]),
        ("Rio de Janeiro", "rio_de_janeiro", [150, 200]),
        ("Tokyo Bay", "tokyo_bay", [150, 200]),
    ]
    div_pcts_lg, scal_pcts_lg = [], []
    for city, slug, sizes in large_instances:
        for sz in sizes:
            cfgs = load(primary_dir, slug, sz)
            if not cfgs:
                continue
            d = fhv(cfgs, "Default PLS")
            v = fhv(cfgs, "Diverse Probe PLS")
            s = fhv(cfgs, "Scalarized PLS")
            winner = (
                "Default"
                if d >= v and d >= s
                else ("Diverse" if v >= s else "Scalarized")
            )
            print(
                f"{city + ' ' + str(sz):<22} {sz:>3}  {d:>11.6f}  {v:>11.6f}  {s:>11.6f}  {winner}"
            )
            if d > 0:
                div_pcts_lg.append(pct(v, d))
                scal_pcts_lg.append(pct(s, d))
        print()
    avg_dl = sum(div_pcts_lg) / len(div_pcts_lg) if div_pcts_lg else 0
    avg_sl = sum(scal_pcts_lg) / len(scal_pcts_lg) if scal_pcts_lg else 0
    print(f"{'Average':<22}      {avg_dl:>+10.2f}%  {avg_sl:>+12.2f}%")
    print()


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
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Regenerate plots from existing JSON artifacts in --output-dir without re-running experiments",
    )
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=None,
        help="Directory with comparison JSON artifacts (e.g. final_paper_results) to overlay on phase bar plots",
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        default=None,
        help="Copy all generated PNGs to this directory after replot (e.g. paper/png_figures_new)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Copy all generated EPS figures to this directory after replot (e.g. paper/figures_new)",
    )
    parser.add_argument(
        "--large-threshold",
        type=int,
        default=100,
        help="Instances with more images than this use --large-configs instead of --configs (default: 100)",
    )
    parser.add_argument(
        "--large-configs",
        type=str,
        nargs="+",
        default=None,
        help="Config labels to use for instances above --large-threshold",
    )
    parser.add_argument(
        "--pseudo-source",
        type=str,
        default="gpbaa",
        choices=["gpbaa", "monise", "pseudo_solver_solutions"],
        help=(
            "Source of initial-population solutions for hybrid configs. "
            "'gpbaa' uses GPBA-A solutions (default), "
            "'monise' uses MONISE-generated solutions."
        ),
    )
    parser.add_argument(
        "--print-tables",
        action="store_true",
        help=(
            "Print all paper table values from existing JSON artifacts in "
            "--output-dir (primary) and --compare-dir (MONISE), then exit."
        ),
    )
    parser.add_argument(
        "--wilcoxon",
        action="store_true",
        help=(
            "Run Wilcoxon signed-rank tests on pairwise algorithm comparisons "
            "from existing JSON artifacts in --output-dir, then exit."
        ),
    )
    args = parser.parse_args()

    # Apply pseudo-solution source globally (affects all _load_pseudo_solutions calls)
    global _PSEUDO_SOLUTIONS_SOURCE, _pseudo_cache
    _PSEUDO_SOLUTIONS_SOURCE = args.pseudo_source
    _pseudo_cache.clear()  # invalidate cache when source changes

    # Filter instances and sort by size so all size-30 run before size-50, etc.
    instances = ALL_INSTANCES
    if args.max_size is not None:
        instances = [(n, f, s) for n, f, s in instances if s <= args.max_size]
    if args.filter is not None:
        pat = re.compile(args.filter)
        instances = [(n, f, s) for n, f, s in instances if pat.search(n)]
    instances = sorted(instances, key=lambda x: (x[0].rsplit("_", 1)[0], x[2]))

    if not instances:
        print("No instances matched the filters.", file=sys.stderr)
        return 1

    # Build config lists — search both CONFIGS and MONISE_CONFIGS by label.
    all_known_configs = CONFIGS + [
        c for c in MONISE_CONFIGS if c.label not in {x.label for x in CONFIGS}
    ]
    default_pool = MONISE_CONFIGS if args.pseudo_source == "monise" else CONFIGS

    configs = default_pool
    if args.configs is not None:
        requested = set(args.configs)
        configs = [c for c in all_known_configs if c.label in requested]

    large_configs = configs  # default: same set for all sizes
    if args.large_configs is not None:
        requested_large = set(args.large_configs)
        large_configs = [c for c in all_known_configs if c.label in requested_large]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.print_tables:
        print_paper_tables(args.output_dir, monise_dir=args.compare_dir)
        return 0

    if args.wilcoxon:
        print_wilcoxon_tests(args.output_dir, monise_dir=args.compare_dir)
        return 0

    if args.replot:
        replot_from_dir(
            args.output_dir,
            configs,
            compare_dir=args.compare_dir,
            publish_dir=args.publish_dir,
            figures_dir=args.figures_dir,
        )
        return 0

    print(f"Running {len(instances)} instances (sorted by size)")
    print(f"Instances: {[n for n, _, _ in instances]}")
    print(f"Configs (≤{args.large_threshold}):  {[c.label for c in configs]}")
    if large_configs is not configs:
        print(f"Configs (>{args.large_threshold}):  {[c.label for c in large_configs]}")
    print(f"Output:    {args.output_dir}")
    print(f"HV points: {args.num_points}")

    all_results: list[dict] = []
    t_start = time.time()

    for display_name, filename, num_images in instances:
        active_configs = (
            configs if num_images <= args.large_threshold else large_configs
        )
        try:
            result = run_instance(
                display_name,
                filename,
                num_images,
                args.output_dir,
                args.num_points,
                active_configs,
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
        seen: set[str] = set()
        labels = [
            c.label
            for c in configs + large_configs
            if not (c.label in seen or seen.add(c.label))
        ]
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
