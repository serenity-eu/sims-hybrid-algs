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

# ─── Optional sims_solvers import (for live MONISE / GPBA-A phase 1) ──
try:
    from types import SimpleNamespace

    from sims_solvers import constants as _sims_constants
    from sims_solvers.FrontGenerators.CoverageGridPoint import CoverageGridPoint
    from sims_solvers.FrontGenerators.MONISE import MONISE as _MONISE
    from sims_solvers.Instances.InstanceGeneric import InstanceGeneric as _InstanceGeneric
    from sims_solvers.Instances.InstanceSIMS import InstanceSIMS as _InstanceSIMS
    from sims_solvers.Models.OrtoolsCPModels.SatelliteImageMosaicSelectionOrtoolsCPModel import (
        SatelliteImageMosaicSelectionOrtoolsCPModel,
    )
    from sims_solvers.Solvers.OrtoolsCPSolver import OrtoolsCPSolver
    from sims_solvers.Timer import Timer as _SolversTimer

    HAS_SIMS_SOLVERS = True
except ImportError:
    HAS_SIMS_SOLVERS = False

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

# Maps config label → (short_name, hex_color) for hybrid comparison bars.
# Grouped by algorithm family: PLS, NSGA-II, NSGA-III, MOEA/D.
_HYBRID_BAR_STYLES: list[tuple[str, str, str]] = [
    # (config_label, display_name, color)
    # Grouped by algorithm family, ratios 80:20→50:50→20:80 left to right.
    # Default PLS (no exact phase) appears last/rightmost as the baseline reference.
    ("Hybrid 80:20",          "PLS 80:20",      "#08306b"),
    ("Hybrid 50:50",          "PLS 50:50",      "#2171b5"),
    ("Hybrid 20:80",          "PLS 20:80",      "#6baed6"),
    ("Default PLS",           "PLS 0:100",      "#9ecae1"),
    ("Hybrid NSGA-II 80:20",  "NSGA-II 80:20",  "#7f2704"),
    ("Hybrid NSGA-II 50:50",  "NSGA-II 50:50",  "#d94801"),
    ("Hybrid NSGA-II 20:80",  "NSGA-II 20:80",  "#f16913"),
    ("Baseline NSGA-II",      "NSGA-II 0:100",  "#fdae6b"),
    ("Hybrid NSGA-III 80:20", "NSGA-III 80:20", "#00441b"),
    ("Hybrid NSGA-III 50:50", "NSGA-III 50:50", "#238b45"),
    ("Hybrid NSGA-III 20:80", "NSGA-III 20:80", "#41ab5d"),
    ("Baseline NSGA-III",     "NSGA-III 0:100", "#74c476"),
    ("Hybrid MOEA/D 80:20",   "MOEA/D 80:20",   "#3f007d"),
    ("Hybrid MOEA/D 50:50",   "MOEA/D 50:50",   "#756bb1"),
    ("Hybrid MOEA/D 20:80",   "MOEA/D 20:80",   "#9e9ac8"),
    ("Baseline MOEA/D",       "MOEA/D 0:100",   "#bcbddc"),
]

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
    "Hybrid 75:25": (0.75, "Default"),
    "Hybrid 50:50": (0.50, "Default"),
    "Hybrid 35:65": (0.35, "Default"),
    "Hybrid 25:75": (0.25, "Default"),
    "Hybrid 20:80": (0.20, "Default"),
    "Scalarized PLS": (0.00, "Scalarized"),
    "Scalarized Hybrid 75:25": (0.75, "Scalarized"),
    "Scalarized Hybrid 50:50": (0.50, "Scalarized"),
    "Scalarized Hybrid 35:65": (0.35, "Scalarized"),
    "Scalarized Hybrid 25:75": (0.25, "Scalarized"),
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


_timeout_override: int | None = None

# When True, PLS configs run with is_deterministic=False so multiple runs
# produce different results (set automatically when --runs > 1).
_force_nondeterministic: bool = False

# PLS perturbation restart was disabled throughout because it panicked with
# "Solution set contains dominated solution!" — a real bug in
# inject_perturbed_archive_solutions (population inserted via insert_unchecked,
# bypassing the domination prune), fixed in sims-heuristics. With restart off,
# PLS halts at its first local-optimum set: measured using 2s of a 190s budget on
# paris_200 and 16.8s of 150s on rio_de_janeiro_200. Opt in via --pls-restart.
_pls_perturbation_restart: bool = False

# Merge into an existing per-instance artifact rather than overwriting it
# (see --append-configs).
_append_configs: bool = False


def timeout_for_size(num_images: int) -> int:
    """PLS timeout in seconds – longer budgets for full HV ablation runs."""
    if _timeout_override is not None:
        return _timeout_override
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
            is_deterministic=not _force_nondeterministic,
            trace=True,
            include_dominated=False,
            use_checkpoint=True,
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=_pls_perturbation_restart,
        )


class Hybrid2080(AlgorithmConfig):
    """Hybrid 20:80 – 20% exact phase, 80% PLS, seeded with pseudo-solver solutions."""

    def run(self, problem, timeout_s, seed=42):
        exact_time = timeout_s * 20 // 100
        pls_time = timeout_s - exact_time

        exact_solutions = _get_pseudo_solutions(problem, max_timestamp_s=exact_time)
        initial_pop = (
            _solutions_to_sims(exact_solutions, problem) if exact_solutions else None
        )

        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=pls_time),
            is_deterministic=not _force_nondeterministic,
            trace=True,
            include_dominated=False,
            initial_population=initial_pop,
            use_checkpoint=False,
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=_pls_perturbation_restart,
        )


class HybridPLSParam(AlgorithmConfig):
    """PLS hybrid at an arbitrary exact:PLS split read from `exact_phase_ratio`.

    The per-ratio classes above (Hybrid2080, HybridBaseline, ...) each hardcode
    their split; this one is parameterised so the 10% grid can be filled without
    a new class per ratio. Solver flags are identical to those classes.
    """

    def run(self, problem, timeout_s, seed=42):
        ratio = self.exact_phase_ratio if self.exact_phase_ratio is not None else 0.5
        exact_time = int(timeout_s * ratio)
        pls_time = timeout_s - exact_time

        exact_solutions = _get_pseudo_solutions(problem, max_timestamp_s=exact_time)
        initial_pop = (
            _solutions_to_sims(exact_solutions, problem) if exact_solutions else None
        )

        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=pls_time),
            is_deterministic=not _force_nondeterministic,
            trace=True,
            include_dominated=False,
            initial_population=initial_pop,
            use_checkpoint=False,
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=_pls_perturbation_restart,
        )


class Hybrid3565(AlgorithmConfig):
    """Hybrid 35:65 – 35% exact phase, 65% PLS, seeded with pseudo-solver solutions."""

    def run(self, problem, timeout_s, seed=42):
        exact_time = timeout_s * 35 // 100
        pls_time = timeout_s - exact_time

        exact_solutions = _get_pseudo_solutions(problem, max_timestamp_s=exact_time)
        initial_pop = (
            _solutions_to_sims(exact_solutions, problem) if exact_solutions else None
        )

        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=pls_time),
            is_deterministic=not _force_nondeterministic,
            trace=True,
            include_dominated=False,
            initial_population=initial_pop,
            use_checkpoint=False,
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=_pls_perturbation_restart,
        )


class HybridBaseline(AlgorithmConfig):
    """Hybrid 50:50 – baseline PLS seeded with pseudo-solver solutions."""

    def run(self, problem, timeout_s, seed=42):
        exact_time = timeout_s // 2
        pls_time = timeout_s - exact_time

        exact_solutions = _get_pseudo_solutions(problem, max_timestamp_s=exact_time)
        initial_pop = (
            _solutions_to_sims(exact_solutions, problem) if exact_solutions else None
        )

        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=pls_time),
            is_deterministic=not _force_nondeterministic,
            trace=True,
            include_dominated=False,
            initial_population=initial_pop,
            use_checkpoint=False,
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=_pls_perturbation_restart,
        )


class Hybrid7525(AlgorithmConfig):
    """Hybrid 75:25 – 75% exact phase, 25% PLS, seeded with pseudo-solver solutions."""

    def run(self, problem, timeout_s, seed=42):
        exact_time = timeout_s * 75 // 100
        pls_time = timeout_s - exact_time

        exact_solutions = _get_pseudo_solutions(problem, max_timestamp_s=exact_time)
        initial_pop = (
            _solutions_to_sims(exact_solutions, problem) if exact_solutions else None
        )

        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=pls_time),
            is_deterministic=not _force_nondeterministic,
            trace=True,
            include_dominated=False,
            initial_population=initial_pop,
            use_checkpoint=False,
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=_pls_perturbation_restart,
        )


class Hybrid2575(AlgorithmConfig):
    """Hybrid 25:75 – 25% exact phase, 75% PLS, seeded with pseudo-solver solutions."""

    def run(self, problem, timeout_s, seed=42):
        exact_time = timeout_s * 25 // 100
        pls_time = timeout_s - exact_time

        exact_solutions = _get_pseudo_solutions(problem, max_timestamp_s=exact_time)
        initial_pop = (
            _solutions_to_sims(exact_solutions, problem) if exact_solutions else None
        )

        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=pls_time),
            is_deterministic=not _force_nondeterministic,
            trace=True,
            include_dominated=False,
            initial_population=initial_pop,
            use_checkpoint=False,
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=_pls_perturbation_restart,
        )


class Hybrid8020(AlgorithmConfig):
    """Hybrid 80:20 – 80% exact phase, 20% PLS, seeded with pseudo-solver solutions."""

    def run(self, problem, timeout_s, seed=42):
        exact_time = timeout_s * 80 // 100
        pls_time = timeout_s - exact_time

        exact_solutions = _get_pseudo_solutions(problem, max_timestamp_s=exact_time)
        initial_pop = (
            _solutions_to_sims(exact_solutions, problem) if exact_solutions else None
        )

        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=pls_time),
            is_deterministic=not _force_nondeterministic,
            trace=True,
            include_dominated=False,
            initial_population=initial_pop,
            use_checkpoint=False,
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=_pls_perturbation_restart,
        )


class DiverseProbePLS(AlgorithmConfig):
    """Improved PLS with diverse probing – farthest-point subset selection."""

    def run(self, problem, timeout_s, seed=42):
        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=timeout_s),
            is_deterministic=not _force_nondeterministic,
            trace=True,
            include_dominated=False,
            use_checkpoint=True,
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=_pls_perturbation_restart,
            use_diverse_probing=True,
        )


class DiverseProbeHybrid(AlgorithmConfig):
    """Improved Hybrid with diverse probing – farthest-point subset selection."""

    def run(self, problem, timeout_s, seed=42):
        exact_time = timeout_s // 2
        pls_time = timeout_s - exact_time

        exact_solutions = _get_pseudo_solutions(problem, max_timestamp_s=exact_time)
        initial_pop = (
            _solutions_to_sims(exact_solutions, problem) if exact_solutions else None
        )

        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=pls_time),
            is_deterministic=not _force_nondeterministic,
            trace=True,
            include_dominated=False,
            initial_population=initial_pop,
            use_checkpoint=True,
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=_pls_perturbation_restart,
            use_diverse_probing=True,
        )


class DiverseProbeHybrid3565(DiverseProbeHybrid):
    """Diverse Probe Hybrid 35:65 – 35% exact, 65% PLS."""

    def run(self, problem, timeout_s, seed=42):
        exact_time = timeout_s * 35 // 100
        pls_time = timeout_s - exact_time
        exact_solutions = _get_pseudo_solutions(problem, max_timestamp_s=exact_time)
        initial_pop = (
            _solutions_to_sims(exact_solutions, problem) if exact_solutions else None
        )
        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=pls_time),
            is_deterministic=not _force_nondeterministic,
            trace=True,
            include_dominated=False,
            initial_population=initial_pop,
            use_checkpoint=True,
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=_pls_perturbation_restart,
            use_diverse_probing=True,
        )


class DiverseProbeHybrid2080(DiverseProbeHybrid):
    """Diverse Probe Hybrid 20:80 – 20% exact, 80% PLS."""

    def run(self, problem, timeout_s, seed=42):
        exact_time = timeout_s * 20 // 100
        pls_time = timeout_s - exact_time
        exact_solutions = _get_pseudo_solutions(problem, max_timestamp_s=exact_time)
        initial_pop = (
            _solutions_to_sims(exact_solutions, problem) if exact_solutions else None
        )
        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=pls_time),
            is_deterministic=not _force_nondeterministic,
            trace=True,
            include_dominated=False,
            initial_population=initial_pop,
            use_checkpoint=True,
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=_pls_perturbation_restart,
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
            is_deterministic=not _force_nondeterministic,
            trace=True,
            include_dominated=False,
            use_checkpoint=True,
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=_pls_perturbation_restart,
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

        exact_solutions = _get_pseudo_solutions(problem, max_timestamp_s=exact_time)
        initial_pop = (
            _solutions_to_sims(exact_solutions, problem) if exact_solutions else None
        )

        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=pls_time),
            is_deterministic=not _force_nondeterministic,
            trace=True,
            include_dominated=False,
            initial_population=initial_pop,
            use_checkpoint=True,
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=_pls_perturbation_restart,
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
        exact_solutions = _get_pseudo_solutions(problem, max_timestamp_s=exact_time)
        initial_pop = (
            _solutions_to_sims(exact_solutions, problem) if exact_solutions else None
        )
        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=pls_time),
            is_deterministic=not _force_nondeterministic,
            trace=True,
            include_dominated=False,
            initial_population=initial_pop,
            use_checkpoint=True,
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=_pls_perturbation_restart,
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
        exact_solutions = _get_pseudo_solutions(problem, max_timestamp_s=exact_time)
        initial_pop = (
            _solutions_to_sims(exact_solutions, problem) if exact_solutions else None
        )
        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=pls_time),
            is_deterministic=not _force_nondeterministic,
            trace=True,
            include_dominated=False,
            initial_population=initial_pop,
            use_checkpoint=True,
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=_pls_perturbation_restart,
            solution_selection_mode="scalarized-chebycheff",
            scalarized_selection_source=self.scalarized_selection_source,
            scalarized_parent_budget=self.scalarized_parent_budget,
            scalarized_weight_samples=self.scalarized_weight_samples,
            scalarized_rho=self.scalarized_rho,
            use_nd_tree_scalarized_query=self.use_nd_tree_scalarized_query,
        )


class ScalarizedHybrid7525(ScalarizedHybrid):
    """Scalarized Hybrid 75:25 – 75% exact, 25% PLS."""

    def run(self, problem, timeout_s, seed=42):
        exact_time = timeout_s * 75 // 100
        pls_time = timeout_s - exact_time
        exact_solutions = _get_pseudo_solutions(problem, max_timestamp_s=exact_time)
        initial_pop = (
            _solutions_to_sims(exact_solutions, problem) if exact_solutions else None
        )
        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=pls_time),
            is_deterministic=not _force_nondeterministic,
            trace=True,
            include_dominated=False,
            initial_population=initial_pop,
            use_checkpoint=True,
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=_pls_perturbation_restart,
            solution_selection_mode="scalarized-chebycheff",
            scalarized_selection_source=self.scalarized_selection_source,
            scalarized_parent_budget=self.scalarized_parent_budget,
            scalarized_weight_samples=self.scalarized_weight_samples,
            scalarized_rho=self.scalarized_rho,
            use_nd_tree_scalarized_query=self.use_nd_tree_scalarized_query,
        )


class ScalarizedHybrid2575(ScalarizedHybrid):
    """Scalarized Hybrid 25:75 – 25% exact, 75% PLS."""

    def run(self, problem, timeout_s, seed=42):
        exact_time = timeout_s * 25 // 100
        pls_time = timeout_s - exact_time
        exact_solutions = _get_pseudo_solutions(problem, max_timestamp_s=exact_time)
        initial_pop = (
            _solutions_to_sims(exact_solutions, problem) if exact_solutions else None
        )
        return sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=pls_time),
            is_deterministic=not _force_nondeterministic,
            trace=True,
            include_dominated=False,
            initial_population=initial_pop,
            use_checkpoint=True,
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=_pls_perturbation_restart,
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


class NSGA2BaselineConfig(AlgorithmConfig):
    """Literal NSGA-II baseline (Deb, Pratap, Agarwal & Meyarivan, 2002)."""

    def run(self, problem, timeout_s, seed=42):
        return sims_problem.solve_with_nsga2_baseline(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=timeout_s),
            population_size=100,
            max_generations=500_000,
            seed=seed,
            trace=True,
            include_dominated=False,
            crossover_rate=0.9,
            mutation_rate=0.01,
        )


class NSGA3BaselineConfig(AlgorithmConfig):
    """Literal NSGA-III baseline (Deb & Jain, 2014)."""

    def run(self, problem, timeout_s, seed=42):
        return sims_problem.solve_with_nsga3_baseline(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=timeout_s),
            num_divisions=99 if len(OBJECTIVES) == 2 else 12,
            max_generations=500_000,
            seed=seed,
            trace=True,
            include_dominated=False,
            crossover_rate=0.9,
            mutation_rate=0.01,
        )


class MOEADBaselineConfig(AlgorithmConfig):
    """Literal MOEA/D baseline (Zhang & Li, 2007)."""

    def run(self, problem, timeout_s, seed=42):
        return sims_problem.solve_with_moead_baseline(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=timeout_s),
            num_divisions=99 if len(OBJECTIVES) == 2 else 12,
            neighbourhood_size=10,
            max_generations=500_000,
            seed=seed,
            trace=True,
            include_dominated=False,
            crossover_rate=1.0,
            mutation_rate=0.01,
        )


class PseudoSeededNSGA2BaselineConfig(AlgorithmConfig):
    """Pseudosolver-seeded literal NSGA-II baseline.

    `exact_phase_ratio` simulates the fraction of the budget consumed by
    GPBA-A; the EA receives the remaining `(1 - exact_phase_ratio) * timeout_s`
    seconds, matching the accounting used by the Hybrid 50:50/35:65/20:80 PLS
    configs.
    """

    def run(self, problem, timeout_s, seed=42):
        ratio = self.exact_phase_ratio if self.exact_phase_ratio is not None else 0.0
        ea_timeout = int(timeout_s * (1.0 - ratio))
        initial_pop = _get_pseudo_solutions(problem, max_timestamp_s=timeout_s - ea_timeout)
        initial_population = (
            _solutions_to_sims(initial_pop, problem) if initial_pop else None
        )
        return sims_problem.solve_with_pseudo_seeded_nsga2_baseline(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=ea_timeout),
            initial_population=initial_population,
            population_size=100,
            max_generations=500_000,
            seed=seed,
            trace=True,
            include_dominated=False,
            crossover_rate=0.9,
            mutation_rate=0.01,
        )


class PseudoSeededNSGA3BaselineConfig(AlgorithmConfig):
    """Pseudosolver-seeded literal NSGA-III baseline.

    `exact_phase_ratio` simulates the fraction of the budget consumed by
    GPBA-A; the EA receives the remaining `(1 - exact_phase_ratio) * timeout_s`
    seconds.
    """

    def run(self, problem, timeout_s, seed=42):
        ratio = self.exact_phase_ratio if self.exact_phase_ratio is not None else 0.0
        ea_timeout = int(timeout_s * (1.0 - ratio))
        initial_pop = _get_pseudo_solutions(problem, max_timestamp_s=timeout_s - ea_timeout)
        initial_population = (
            _solutions_to_sims(initial_pop, problem) if initial_pop else None
        )
        return sims_problem.solve_with_pseudo_seeded_nsga3_baseline(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=ea_timeout),
            initial_population=initial_population,
            num_divisions=99 if len(OBJECTIVES) == 2 else 12,
            max_generations=500_000,
            seed=seed,
            trace=True,
            include_dominated=False,
            crossover_rate=0.9,
            mutation_rate=0.01,
        )


class PseudoSeededMOEADBaselineConfig(AlgorithmConfig):
    """Pseudosolver-seeded literal MOEA/D baseline.

    `exact_phase_ratio` simulates the fraction of the budget consumed by
    GPBA-A; the EA receives the remaining `(1 - exact_phase_ratio) * timeout_s`
    seconds.
    """

    def run(self, problem, timeout_s, seed=42):
        ratio = self.exact_phase_ratio if self.exact_phase_ratio is not None else 0.0
        ea_timeout = int(timeout_s * (1.0 - ratio))
        initial_pop = _get_pseudo_solutions(problem, max_timestamp_s=timeout_s - ea_timeout)
        initial_population = (
            _solutions_to_sims(initial_pop, problem) if initial_pop else None
        )
        return sims_problem.solve_with_pseudo_seeded_moead_baseline(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=ea_timeout),
            initial_population=initial_population,
            num_divisions=99 if len(OBJECTIVES) == 2 else 12,
            neighbourhood_size=10,
            max_generations=500_000,
            seed=seed,
            trace=True,
            include_dominated=False,
            crossover_rate=1.0,
            mutation_rate=0.01,
        )


class PseudoSeededNSGA2Config(AlgorithmConfig):
    """Pseudosolver-seeded SIMS-tailored NSGA-II (improved-tier hybrid).

    Same GPBA-A-ratio accounting as ``PseudoSeededNSGA2BaselineConfig``, but the
    seed population feeds the tailored ``solve_with_pseudo_seeded_nsga2`` (composite
    mutation, stagnation injection, contribution-distance selection) instead of the
    literal-paper baseline. Tuning mirrors the standalone ``NSGA2Config``.
    """

    def run(self, problem, timeout_s, seed=42):
        ratio = self.exact_phase_ratio if self.exact_phase_ratio is not None else 0.0
        ea_timeout = int(timeout_s * (1.0 - ratio))
        initial_pop = _get_pseudo_solutions(problem, max_timestamp_s=timeout_s - ea_timeout)
        initial_population = (
            _solutions_to_sims(initial_pop, problem) if initial_pop else None
        )
        return sims_problem.solve_with_pseudo_seeded_nsga2(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=ea_timeout),
            initial_population=initial_population,
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


class PseudoSeededNSGA3Config(AlgorithmConfig):
    """Pseudosolver-seeded SIMS-tailored NSGA-III (improved-tier hybrid)."""

    def run(self, problem, timeout_s, seed=42):
        ratio = self.exact_phase_ratio if self.exact_phase_ratio is not None else 0.0
        ea_timeout = int(timeout_s * (1.0 - ratio))
        initial_pop = _get_pseudo_solutions(problem, max_timestamp_s=timeout_s - ea_timeout)
        initial_population = (
            _solutions_to_sims(initial_pop, problem) if initial_pop else None
        )
        return sims_problem.solve_with_pseudo_seeded_nsga3(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=ea_timeout),
            initial_population=initial_population,
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


class PseudoSeededMOEADConfig(AlgorithmConfig):
    """Pseudosolver-seeded SIMS-tailored MOEA/D (improved-tier hybrid).

    Uses the Li & Zhang (2009) enhancements (delta mating, nr replacement cap, PBI)
    exposed by ``solve_with_pseudo_seeded_moead``. Tuning mirrors ``MOEADConfig``.
    """

    def run(self, problem, timeout_s, seed=42):
        ratio = self.exact_phase_ratio if self.exact_phase_ratio is not None else 0.0
        ea_timeout = int(timeout_s * (1.0 - ratio))
        initial_pop = _get_pseudo_solutions(problem, max_timestamp_s=timeout_s - ea_timeout)
        initial_population = (
            _solutions_to_sims(initial_pop, problem) if initial_pop else None
        )
        return sims_problem.solve_with_pseudo_seeded_moead(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=ea_timeout),
            initial_population=initial_population,
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


class NSGA2MoorsConfig(AlgorithmConfig):
    """NSGA-II via the `moors` crate (native binary operators) -- external sanity check."""

    def run(self, problem, timeout_s, seed=42):
        return sims_problem.solve_with_nsga2_moors(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=timeout_s),
            population_size=100,
            num_iterations=500_000,
            seed=seed,
            trace=True,
            include_dominated=False,
        )


class NSGA2OptirusticConfig(AlgorithmConfig):
    """NSGA-II via the `optirustic` crate (SBX + polynomial mutation) -- external sanity check."""

    def run(self, problem, timeout_s, seed=42):
        return sims_problem.solve_with_nsga2_optirustic(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=timeout_s),
            population_size=100,
            max_generations=500_000,
            seed=seed,
            trace=True,
            include_dominated=False,
        )


class NSGA3OptirusticConfig(AlgorithmConfig):
    """NSGA-III via the `optirustic` crate (SBX + polynomial mutation + niching) -- external sanity check."""

    def run(self, problem, timeout_s, seed=42):
        return sims_problem.solve_with_nsga3_optirustic(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=timeout_s),
            population_size=100,
            max_generations=500_000,
            seed=seed,
            trace=True,
            include_dominated=False,
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
            is_deterministic=not _force_nondeterministic,
            trace=True,
            include_dominated=False,
            initial_population=initial_pop,
            use_checkpoint=True,
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=_pls_perturbation_restart,
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


# ─── EA-phase-1 → PLS-phase-2 hybrid configs ─────────────────────────
# These configs run a multi-objective EA (NSGA-II/III/MOEA/D baseline) live as
# the first phase for `exact_phase_ratio * timeout_s` seconds, then seed PLS
# with the EA's Pareto front for the remaining time.  Unlike the Pseudo-seeded
# configs, there is no reliance on pre-recorded solutions: the EA is executed
# during the experiment.  The two sub-phase traces are merged with
# sims_problem.merge_traces() so the combined trace spans the full timeline.


def _bounds_from_solutions(
    solutions: list[sims_problem.Solution],
    ndim: int,
) -> list[list[int]]:
    """Compute [[lo, hi], ...] bounds from a list of Solution objects."""
    obj_keys = ("cost", "cloudy_area", "max_incidence_angle", "min_resolutions_sum")
    result = []
    for j in range(ndim):
        key = obj_keys[j]
        vals = []
        for sol in solutions:
            d = sol.to_json()
            v = d.get(key)
            if v is not None:
                vals.append(int(v))
        if not vals:
            result.append([0, 1])
            continue
        lo, hi = min(vals), max(vals)
        rng = max(hi - lo, 1)
        result.append([max(0, lo - 1), hi + int(rng * 0.1) + 1])
    return result


class EAPhase1PLSHybrid(AlgorithmConfig):
    """EA (phase 1) → PLS (phase 2) live hybrid.

    Subclasses override ``_run_ea_phase`` to select the algorithm.
    ``exact_phase_ratio`` is the fraction of the budget given to the EA.
    """

    def _run_ea_phase(
        self, problem: sims_problem.SimsDiscreteProblem, ea_time_s: int, seed: int
    ) -> sims_problem.SolvingResult:
        raise NotImplementedError

    def run(self, problem, timeout_s, seed=42):
        ratio = self.exact_phase_ratio if self.exact_phase_ratio is not None else 0.0
        ea_time = int(timeout_s * ratio)
        pls_time = timeout_s - ea_time

        if ea_time <= 0:
            # Pure PLS — no EA phase
            return sims_problem.solve_with_pls(
                problem,
                objectives=OBJECTIVES,
                timeout=timedelta(seconds=pls_time),
                is_deterministic=not _force_nondeterministic,
                trace=True,
                include_dominated=False,
                initial_population=None,
                use_checkpoint=False,
                use_ranked_candidates=False,
                use_greedy_initial_population=True,
                use_perturbation_restart=_pls_perturbation_restart,
            )

        ea_result = self._run_ea_phase(problem, ea_time, seed)

        if pls_time <= 0:
            # Pure EA — no PLS phase
            return ea_result

        ea_solutions = ea_result.final_solutions

        pls_result = sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=pls_time),
            is_deterministic=not _force_nondeterministic,
            trace=True,
            include_dominated=False,
            initial_population=ea_solutions if ea_solutions else None,
            use_checkpoint=False,
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=_pls_perturbation_restart,
        )

        # Merge traces so the combined timeline spans 0 → total_timeout.
        ea_trace = ea_result.trace
        pls_trace = pls_result.trace
        if ea_trace and pls_trace:
            all_sols = (ea_solutions or []) + (pls_result.final_solutions or [])
            ndim = len(OBJECTIVES)
            bounds = _bounds_from_solutions(all_sols, ndim) if all_sols else [
                [0, 1] for _ in range(ndim)
            ]
            ref_point = [b[1] + 1 for b in bounds]
            merged_trace = sims_problem.merge_traces(
                ea_trace, pls_trace, self.label, bounds, ref_point
            )
        else:
            merged_trace = pls_trace or ea_trace or b""

        final_sols = pls_result.final_solutions or ea_solutions or []
        if merged_trace:
            return sims_problem.SolvingResult.with_trace(final_sols, merged_trace)
        return sims_problem.SolvingResult(final_sols)


class NSGA2Phase1PLS(EAPhase1PLSHybrid):
    """NSGA-II baseline (phase 1) → PLS (phase 2)."""

    def _run_ea_phase(self, problem, ea_time_s, seed):
        return sims_problem.solve_with_nsga2_baseline(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=ea_time_s),
            population_size=100,
            max_generations=500_000,
            seed=seed,
            trace=True,
            include_dominated=False,
            crossover_rate=0.9,
            mutation_rate=0.01,
        )


class NSGA3Phase1PLS(EAPhase1PLSHybrid):
    """NSGA-III baseline (phase 1) → PLS (phase 2)."""

    def _run_ea_phase(self, problem, ea_time_s, seed):
        num_div = 99 if len(OBJECTIVES) == 2 else 12
        return sims_problem.solve_with_nsga3_baseline(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=ea_time_s),
            num_divisions=num_div,
            max_generations=500_000,
            seed=seed,
            trace=True,
            include_dominated=False,
            crossover_rate=0.9,
            mutation_rate=0.01,
        )


class MOEADPhase1PLS(EAPhase1PLSHybrid):
    """MOEA/D baseline (phase 1) → PLS (phase 2)."""

    def _run_ea_phase(self, problem, ea_time_s, seed):
        num_div = 99 if len(OBJECTIVES) == 2 else 12
        return sims_problem.solve_with_moead_baseline(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=ea_time_s),
            num_divisions=num_div,
            neighbourhood_size=10,
            max_generations=500_000,
            seed=seed,
            trace=True,
            include_dominated=False,
            crossover_rate=1.0,
            mutation_rate=0.01,
        )


# ─── Live exact-phase-1 → PLS-phase-2 hybrid configs ─────────────────
# These configs run MONISE or GPBA-A (via OR-Tools, no Gurobi required) LIVE
# for `exact_phase_ratio * timeout_s` seconds, then seed PLS with the exact
# solver's Pareto front for the remaining time.  Traces are merged so the
# combined trace spans the full timeline.  Requires HAS_SIMS_SOLVERS=True.


class _InstanceSimsDirect(_InstanceSIMS):
    """Adapts SimsDiscreteProblem to the InstanceSIMS attribute interface.

    InstanceSIMS.__init__ adjusts 1-based MiniZinc indices to 0-based.
    SimsDiscreteProblem data is already 0-based, so we bypass that correction
    by calling the grandparent InstanceGeneric.__init__ directly.
    """

    def __init__(self, problem: sims_problem.SimsDiscreteProblem) -> None:
        # Bypass InstanceSIMS.__init__ (which expects 1-based MiniZinc dict)
        _InstanceGeneric.__init__(
            self,
            is_minizinc=False,
            problem_name=_sims_constants.Problem.SATELLITE_IMAGE_SELECTION_PROBLEM.value,
        )
        self.images = [set(img) for img in problem.images]
        self.clouds = [set(cl) for cl in problem.clouds]
        self.costs = list(problem.costs)
        self.areas = list(problem.areas)
        self.max_cloud_area = problem.max_cloud_area
        self.resolution = list(problem.resolution)
        self.incidence_angle = list(problem.incidence_angle)
        self.cloud_covered_by_image, self.clouds_id_area = self.get_clouds_covered_by_image()


def _build_ortools_solver(problem: sims_problem.SimsDiscreteProblem) -> "OrtoolsCPSolver":
    """Build an OR-Tools CP-SAT solver wired to the SIMS model for `problem`."""
    sims_inst = _InstanceSimsDirect(problem)
    config = SimpleNamespace(objectives=OBJECTIVES)
    model = SatelliteImageMosaicSelectionOrtoolsCPModel(sims_inst, config)
    return OrtoolsCPSolver(model, statistics={}, threads=1, free_search=True)


def _collect_exact_solver_solutions(
    front_generator: Any,
    timer: "Timer",
    phase_time_s: float,
) -> list[sims_problem.Solution]:
    """Drive a FrontGeneratorStrategy generator and convert solutions.

    Returns a list of sims_problem.Solution objects with timestamps set to
    the cumulative wall-clock time within the phase.
    """
    obj_name_to_idx: dict[str, int] = {name: i for i, name in enumerate(OBJECTIVES)}
    ndim = len(OBJECTIVES)

    solutions: list[sims_problem.Solution] = []
    try:
        for sol_result in front_generator.solve():
            ts_s = phase_time_s - timer.time_budget_sec
            objs = sol_result.solution.objs          # list[int], one per OBJECTIVE
            selected = sol_result.solution.solution_values   # list[int] of image indices

            # map objective names to keyword args for Solution.create
            obj_vals: dict[str, int] = {}
            for name, idx in obj_name_to_idx.items():
                obj_vals[name] = objs[idx]

            solutions.append(
                sims_problem.Solution.create(
                    selected_images=selected,
                    cost=obj_vals.get("min_cost"),
                    cloudy_area=obj_vals.get("cloud_coverage"),
                    max_incidence_angle=obj_vals.get("min_max_incidence_angle"),
                    min_resolutions_sum=obj_vals.get("min_resolution"),
                    timestamp_us=int(ts_s * 1_000_000),
                )
            )
    except TimeoutError:
        pass
    return solutions


class ExactLivePhase1PLSHybrid(AlgorithmConfig):
    """Base: live exact solver (phase 1) → PLS (phase 2).

    Subclasses override ``_make_front_generator`` to choose the algorithm.
    ``exact_phase_ratio`` is the fraction of the total budget given to the
    exact solver; the remainder goes to PLS.
    """

    algo_label: str = "Exact"  # used in generate_trace algorithm field

    def _make_front_generator(
        self, solver: "OrtoolsCPSolver", timer: "Timer"
    ) -> Any:
        raise NotImplementedError

    def run(self, problem, timeout_s, seed=42):
        if not HAS_SIMS_SOLVERS:
            raise RuntimeError(
                f"{self.label}: sims_solvers not available — "
                "install with `uv sync` from the workspace root."
            )

        ratio = self.exact_phase_ratio if self.exact_phase_ratio is not None else 0.5
        exact_time_s = timeout_s * ratio
        pls_time_s = timeout_s - exact_time_s
        ndim = len(OBJECTIVES)

        if exact_time_s <= 0:
            return sims_problem.solve_with_pls(
                problem,
                objectives=OBJECTIVES,
                timeout=timedelta(seconds=int(pls_time_s)),
                is_deterministic=not _force_nondeterministic,
                trace=True,
                include_dominated=False,
                initial_population=None,
                use_checkpoint=False,
                use_ranked_candidates=False,
                use_greedy_initial_population=True,
                use_perturbation_restart=_pls_perturbation_restart,
            )

        # ── Phase 1: run exact solver ───────────────────────────────────
        solver = _build_ortools_solver(problem)
        timer = _SolversTimer(exact_time_s)
        fg = self._make_front_generator(solver, timer)
        exact_solutions = _collect_exact_solver_solutions(fg, timer, exact_time_s)

        if not exact_solutions:
            print(f"  WARN: {self.label}: exact phase found 0 solutions", flush=True)

        # Build a trace for the exact phase
        if exact_solutions:
            ex_bounds = _bounds_from_solutions(exact_solutions, ndim)
            ex_ref = [b[1] + 1 for b in ex_bounds]
            exact_trace = sims_problem.generate_trace(
                solutions=exact_solutions,
                objectives=OBJECTIVES,
                algorithm=self.algo_label,
                num_objectives=ndim,
                objective_bounds=ex_bounds,
                reference_point=ex_ref,
                include_dominated=False,
            )
        else:
            exact_trace = b""

        if pls_time_s <= 0:
            if exact_trace:
                return sims_problem.SolvingResult.with_trace(exact_solutions, exact_trace)
            return sims_problem.SolvingResult(exact_solutions)

        # ── Phase 2: PLS seeded with exact solutions ────────────────────
        pls_result = sims_problem.solve_with_pls(
            problem,
            objectives=OBJECTIVES,
            timeout=timedelta(seconds=int(pls_time_s)),
            is_deterministic=not _force_nondeterministic,
            trace=True,
            include_dominated=False,
            initial_population=exact_solutions if exact_solutions else None,
            use_checkpoint=False,
            use_ranked_candidates=False,
            use_greedy_initial_population=True,
            use_perturbation_restart=_pls_perturbation_restart,
        )

        pls_trace = pls_result.trace
        final_sols = pls_result.final_solutions or exact_solutions or []

        # ── Merge traces into a unified timeline ────────────────────────
        if exact_trace and pls_trace:
            all_sols = list(exact_solutions) + list(pls_result.final_solutions or [])
            merged_bounds = _bounds_from_solutions(all_sols, ndim) if all_sols else [[0, 1]] * ndim
            merged_ref = [b[1] + 1 for b in merged_bounds]
            merged = sims_problem.merge_traces(
                exact_trace, pls_trace, self.label, merged_bounds, merged_ref
            )
            return sims_problem.SolvingResult.with_trace(final_sols, merged)
        elif pls_trace:
            return pls_result
        elif exact_trace:
            return sims_problem.SolvingResult.with_trace(exact_solutions, exact_trace)
        return sims_problem.SolvingResult(final_sols)


class MONISELivePhase1PLS(ExactLivePhase1PLSHybrid):
    """Live MONISE (OR-Tools) → PLS hybrid."""

    algo_label = "MONISE"

    def _make_front_generator(self, solver, timer):
        return _MONISE(solver, timer)


class GPBAALivePhase1PLS(ExactLivePhase1PLSHybrid):
    """Live GPBA-A / CoverageGridPoint (OR-Tools) → PLS hybrid."""

    algo_label = "GPBA-A"

    def _make_front_generator(self, solver, timer):
        return CoverageGridPoint(solver, timer)


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
    Hybrid8020(
        label="Hybrid 80:20",
        color="#08306b",
        linestyle="-",
        linewidth=1.5,
        exact_phase_ratio=0.80,
    ),
    Hybrid7525(
        label="Hybrid 75:25",
        color="#08306b",
        linestyle="--",
        linewidth=1.5,
        exact_phase_ratio=0.75,
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
    Hybrid2575(
        label="Hybrid 25:75",
        color="#74c476",
        linestyle=":",
        linewidth=1.5,
        exact_phase_ratio=0.25,
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
    NSGA2BaselineConfig(
        label="Baseline NSGA-II",
        color="#ff7f0e",
        linestyle=":",
        linewidth=1.2,
    ),
    MOEADBaselineConfig(
        label="Baseline MOEA/D",
        color="#9467bd",
        linestyle=":",
        linewidth=1.2,
    ),
    NSGA3BaselineConfig(
        label="Baseline NSGA-III",
        color="#2ecc71",
        linestyle=":",
        linewidth=1.2,
    ),
    PseudoSeededNSGA2BaselineConfig(
        label="Hybrid NSGA-II 80:20",
        color="#ff7f0e",
        linestyle="-",
        linewidth=1.6,
        exact_phase_ratio=0.80,
    ),
    PseudoSeededNSGA2BaselineConfig(
        label="Hybrid NSGA-II 75:25",
        color="#ff7f0e",
        linestyle="-",
        linewidth=1.4,
        exact_phase_ratio=0.75,
    ),
    PseudoSeededNSGA2BaselineConfig(
        label="Hybrid NSGA-II 50:50",
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.2,
        exact_phase_ratio=0.50,
    ),
    PseudoSeededNSGA2BaselineConfig(
        label="Hybrid NSGA-II 35:65",
        color="#ff7f0e",
        linestyle="-.",
        linewidth=1.1,
        exact_phase_ratio=0.35,
    ),
    PseudoSeededNSGA2BaselineConfig(
        label="Hybrid NSGA-II 25:75",
        color="#ff7f0e",
        linestyle=":",
        linewidth=1.0,
        exact_phase_ratio=0.25,
    ),
    PseudoSeededNSGA2BaselineConfig(
        label="Hybrid NSGA-II 20:80",
        color="#ff7f0e",
        linestyle=(0, (3, 1, 1, 1)),
        linewidth=1.0,
        exact_phase_ratio=0.20,
    ),
    PseudoSeededNSGA3BaselineConfig(
        label="Hybrid NSGA-III 80:20",
        color="#2ecc71",
        linestyle="-",
        linewidth=1.6,
        exact_phase_ratio=0.80,
    ),
    PseudoSeededNSGA3BaselineConfig(
        label="Hybrid NSGA-III 75:25",
        color="#2ecc71",
        linestyle="-",
        linewidth=1.4,
        exact_phase_ratio=0.75,
    ),
    PseudoSeededNSGA3BaselineConfig(
        label="Hybrid NSGA-III 50:50",
        color="#2ecc71",
        linestyle="--",
        linewidth=1.2,
        exact_phase_ratio=0.50,
    ),
    PseudoSeededNSGA3BaselineConfig(
        label="Hybrid NSGA-III 35:65",
        color="#2ecc71",
        linestyle="-.",
        linewidth=1.1,
        exact_phase_ratio=0.35,
    ),
    PseudoSeededNSGA3BaselineConfig(
        label="Hybrid NSGA-III 25:75",
        color="#2ecc71",
        linestyle=":",
        linewidth=1.0,
        exact_phase_ratio=0.25,
    ),
    PseudoSeededNSGA3BaselineConfig(
        label="Hybrid NSGA-III 20:80",
        color="#2ecc71",
        linestyle=(0, (3, 1, 1, 1)),
        linewidth=1.0,
        exact_phase_ratio=0.20,
    ),
    PseudoSeededMOEADBaselineConfig(
        label="Hybrid MOEA/D 80:20",
        color="#9467bd",
        linestyle="-",
        linewidth=1.6,
        exact_phase_ratio=0.80,
    ),
    PseudoSeededMOEADBaselineConfig(
        label="Hybrid MOEA/D 75:25",
        color="#9467bd",
        linestyle="-",
        linewidth=1.4,
        exact_phase_ratio=0.75,
    ),
    PseudoSeededMOEADBaselineConfig(
        label="Hybrid MOEA/D 50:50",
        color="#9467bd",
        linestyle="--",
        linewidth=1.2,
        exact_phase_ratio=0.50,
    ),
    PseudoSeededMOEADBaselineConfig(
        label="Hybrid MOEA/D 35:65",
        color="#9467bd",
        linestyle="-.",
        linewidth=1.1,
        exact_phase_ratio=0.35,
    ),
    PseudoSeededMOEADBaselineConfig(
        label="Hybrid MOEA/D 25:75",
        color="#9467bd",
        linestyle=":",
        linewidth=1.0,
        exact_phase_ratio=0.25,
    ),
    PseudoSeededMOEADBaselineConfig(
        label="Hybrid MOEA/D 20:80",
        color="#9467bd",
        linestyle=(0, (3, 1, 1, 1)),
        linewidth=1.0,
        exact_phase_ratio=0.20,
    ),
    # ── Improved-tier: SIMS-tailored EAs (standalone + pseudo-seeded hybrids) ──
    NSGA2Config(
        label="Improved NSGA-II",
        color="#ff7f0e",
        linestyle="-.",
        linewidth=1.2,
    ),
    PseudoSeededNSGA2Config(
        label="Improved NSGA-II 80:20",
        color="#ff7f0e",
        linestyle="-",
        linewidth=1.6,
        exact_phase_ratio=0.80,
    ),
    PseudoSeededNSGA2Config(
        label="Improved NSGA-II 50:50",
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.4,
        exact_phase_ratio=0.50,
    ),
    PseudoSeededNSGA2Config(
        label="Improved NSGA-II 20:80",
        color="#ff7f0e",
        linestyle=":",
        linewidth=1.2,
        exact_phase_ratio=0.20,
    ),
    NSGA3Config(
        label="Improved NSGA-III",
        color="#2ecc71",
        linestyle="-.",
        linewidth=1.2,
    ),
    PseudoSeededNSGA3Config(
        label="Improved NSGA-III 80:20",
        color="#2ecc71",
        linestyle="-",
        linewidth=1.6,
        exact_phase_ratio=0.80,
    ),
    PseudoSeededNSGA3Config(
        label="Improved NSGA-III 50:50",
        color="#2ecc71",
        linestyle="--",
        linewidth=1.4,
        exact_phase_ratio=0.50,
    ),
    PseudoSeededNSGA3Config(
        label="Improved NSGA-III 20:80",
        color="#2ecc71",
        linestyle=":",
        linewidth=1.2,
        exact_phase_ratio=0.20,
    ),
    MOEADConfig(
        label="Improved MOEA/D",
        color="#9467bd",
        linestyle="-.",
        linewidth=1.2,
    ),
    PseudoSeededMOEADConfig(
        label="Improved MOEA/D 80:20",
        color="#9467bd",
        linestyle="-",
        linewidth=1.6,
        exact_phase_ratio=0.80,
    ),
    PseudoSeededMOEADConfig(
        label="Improved MOEA/D 50:50",
        color="#9467bd",
        linestyle="--",
        linewidth=1.4,
        exact_phase_ratio=0.50,
    ),
    PseudoSeededMOEADConfig(
        label="Improved MOEA/D 20:80",
        color="#9467bd",
        linestyle=":",
        linewidth=1.2,
        exact_phase_ratio=0.20,
    ),
    NSGA2MoorsConfig(
        label="NSGA-II (moors)",
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.0,
    ),
    NSGA2OptirusticConfig(
        label="NSGA-II (optirustic)",
        color="#ff7f0e",
        linestyle="-",
        linewidth=1.0,
    ),
    NSGA3OptirusticConfig(
        label="NSGA-III (optirustic)",
        color="#2ecc71",
        linestyle="-",
        linewidth=1.0,
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
    ScalarizedHybrid7525(
        label="Scalarized Hybrid 75:25",
        color="#6b7d00",
        linestyle="-",
        linewidth=1.5,
        exact_phase_ratio=0.75,
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
    ScalarizedHybrid2575(
        label="Scalarized Hybrid 25:75",
        color="#adb800",
        linestyle=":",
        linewidth=1.5,
        exact_phase_ratio=0.25,
    ),
    ScalarizedHybrid2080(
        label="Scalarized Hybrid 20:80",
        color="#e8e9a0",
        linestyle="-",
        linewidth=1.5,
        exact_phase_ratio=0.20,
    ),
    # ── EA-phase-1 → PLS-phase-2 hybrids ─────────────────────────────────
    NSGA2Phase1PLS(
        label="EA NSGA-II PLS 75:25",
        color="#d62728",
        linestyle="-",
        linewidth=1.4,
        exact_phase_ratio=0.75,
    ),
    NSGA2Phase1PLS(
        label="EA NSGA-II PLS 50:50",
        color="#d62728",
        linestyle="--",
        linewidth=1.2,
        exact_phase_ratio=0.50,
    ),
    NSGA2Phase1PLS(
        label="EA NSGA-II PLS 25:75",
        color="#d62728",
        linestyle=":",
        linewidth=1.0,
        exact_phase_ratio=0.25,
    ),
    NSGA3Phase1PLS(
        label="EA NSGA-III PLS 75:25",
        color="#2ca02c",
        linestyle="-",
        linewidth=1.4,
        exact_phase_ratio=0.75,
    ),
    NSGA3Phase1PLS(
        label="EA NSGA-III PLS 50:50",
        color="#2ca02c",
        linestyle="--",
        linewidth=1.2,
        exact_phase_ratio=0.50,
    ),
    NSGA3Phase1PLS(
        label="EA NSGA-III PLS 25:75",
        color="#2ca02c",
        linestyle=":",
        linewidth=1.0,
        exact_phase_ratio=0.25,
    ),
    MOEADPhase1PLS(
        label="EA MOEAD PLS 75:25",
        color="#9467bd",
        linestyle="-",
        linewidth=1.4,
        exact_phase_ratio=0.75,
    ),
    MOEADPhase1PLS(
        label="EA MOEAD PLS 50:50",
        color="#9467bd",
        linestyle="--",
        linewidth=1.2,
        exact_phase_ratio=0.50,
    ),
    MOEADPhase1PLS(
        label="EA MOEAD PLS 25:75",
        color="#9467bd",
        linestyle=":",
        linewidth=1.0,
        exact_phase_ratio=0.25,
    ),
    # ── Live MONISE-phase-1 → PLS-phase-2 hybrids ────────────────────────
    MONISELivePhase1PLS(
        label="MONISE Live 75:25",
        color="#1a9641",
        linestyle="-",
        linewidth=1.4,
        exact_phase_ratio=0.75,
    ),
    MONISELivePhase1PLS(
        label="MONISE Live 50:50",
        color="#1a9641",
        linestyle="--",
        linewidth=1.2,
        exact_phase_ratio=0.50,
    ),
    MONISELivePhase1PLS(
        label="MONISE Live 35:65",
        color="#1a9641",
        linestyle="-.",
        linewidth=1.1,
        exact_phase_ratio=0.35,
    ),
    MONISELivePhase1PLS(
        label="MONISE Live 25:75",
        color="#1a9641",
        linestyle=":",
        linewidth=1.0,
        exact_phase_ratio=0.25,
    ),
    MONISELivePhase1PLS(
        label="MONISE Live 20:80",
        color="#1a9641",
        linestyle=(0, (3, 1, 1, 1)),
        linewidth=1.0,
        exact_phase_ratio=0.20,
    ),
    # ── Live GPBA-A-phase-1 → PLS-phase-2 hybrids ────────────────────────
    GPBAALivePhase1PLS(
        label="GPBA-A Live 75:25",
        color="#08306b",
        linestyle="-",
        linewidth=1.4,
        exact_phase_ratio=0.75,
    ),
    GPBAALivePhase1PLS(
        label="GPBA-A Live 50:50",
        color="#08306b",
        linestyle="--",
        linewidth=1.2,
        exact_phase_ratio=0.50,
    ),
    GPBAALivePhase1PLS(
        label="GPBA-A Live 35:65",
        color="#08306b",
        linestyle="-.",
        linewidth=1.1,
        exact_phase_ratio=0.35,
    ),
    GPBAALivePhase1PLS(
        label="GPBA-A Live 25:75",
        color="#08306b",
        linestyle=":",
        linewidth=1.0,
        exact_phase_ratio=0.25,
    ),
    GPBAALivePhase1PLS(
        label="GPBA-A Live 20:80",
        color="#08306b",
        linestyle=(0, (3, 1, 1, 1)),
        linewidth=1.0,
        exact_phase_ratio=0.20,
    ),
]


# ─── 10%-step grid fill ───────────────────────────────────────────────
# The registry above defines hybrids only at 80:20, 75:25, 50:50, 35:65, 25:75,
# 20:80. To reproduce the paper's 10%-step ratio sweep for every second-phase
# algorithm (PLS, NSGA-II, NSGA-III, MOEA/D), append configs at the six missing
# on-grid ratios. The EA seeded classes already read `exact_phase_ratio`, and
# HybridPLSParam does the same for PLS, so this is pure construction -- no new
# behaviour. The three ratios already present (80:20/50:50/20:80) are skipped so
# labels stay unique.
# Exact-phase percentages, as integers to keep labels exact (0.1 is not
# representable, so int((1-0.9)*100) is 9, not 10).
_GRID_MISSING_PCT = [90, 70, 60, 40, 30, 10]
_GRID_SECOND_PHASE = [
    ("Hybrid {r}", HybridPLSParam, "#1f77b4", "-"),
    ("Hybrid NSGA-II {r}", PseudoSeededNSGA2BaselineConfig, "#ff7f0e", "-"),
    ("Hybrid NSGA-III {r}", PseudoSeededNSGA3BaselineConfig, "#9467bd", "-"),
    ("Hybrid MOEA/D {r}", PseudoSeededMOEADBaselineConfig, "#17becf", "-"),
]
for _pct in _GRID_MISSING_PCT:
    _r_label = f"{_pct}:{100 - _pct}"
    for _tmpl, _cls, _col, _ls in _GRID_SECOND_PHASE:
        CONFIGS.append(
            _cls(
                label=_tmpl.format(r=_r_label),
                color=_col,
                linestyle=_ls,
                linewidth=1.5,
                exact_phase_ratio=_pct / 100.0,
            )
        )


# ─── Pseudo-solver helpers ────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent.parent / "sims-core" / "tests" / "data"

# Sources: "gpbaa" uses GPBA-A pre-computed solutions (classic pseudo-solver),
#          "monise" uses MONISE-generated solutions.
# Kept as a module-level variable so run_instance() can read it after main()
# overrides it based on --pseudo-source.
_PSEUDO_SOLUTIONS_SOURCE: str = "gpbaa"

_PSEUDO_SOLUTIONS_DIR = _DATA_DIR / "pseudo_solver_solutions"

_PSEUDO_SOURCE_DIRS: dict[str, Path] = {
    "gpbaa":               _DATA_DIR / "gpbaa",
    "monise":              _DATA_DIR / "monise",
    "gpbaa_2d":            _DATA_DIR / "gpbaa_2d",
    "monise_2d":           _DATA_DIR / "monise_2d",
    "gpbaa_2d_pub":        _DATA_DIR / "gpbaa_2d_pub",
    "gpbaa_2d_highs":      _DATA_DIR / "gpbaa_2d_highs",
    "an_2d_highs":         _DATA_DIR / "an_2d_highs",
    "gpbaa_2d_gurobi":     _DATA_DIR / "gpbaa_2d_gurobi",
    "an_2d_gurobi":        _DATA_DIR / "an_2d_gurobi",
    "monise_2d_pub":       _DATA_DIR / "monise_2d_pub",
    "pseudo_solver_solutions": _DATA_DIR / "pseudo_solver_solutions",
    # random-clouds (family C) pseudo-solutions, generated via generate_pseudo.py
    # --instance-set random-clouds --solver gurobi (see out_suffix="_rc").
    "gpbaa_2d_gurobi_rc":  _DATA_DIR / "gpbaa_2d_gurobi_rc",
    # A&N-calibrated "hard" set (see instance_generation_difficulty.md, Part II).
    "an_2d_gurobi_hard":   _DATA_DIR / "an_2d_gurobi_hard",
    # Same solutions as an_2d_gurobi_hard, trimmed to each instance's experiment
    # timeout (see trim_pseudo.py). The full sets were generated with a generous
    # budget; this holds exactly the prefix the exact phase could have found.
    "an_2d_gurobi_hard_t": _DATA_DIR / "an_2d_gurobi_hard_t",
    "gpbaa_2d_gurobi_hard": _DATA_DIR / "gpbaa_2d_gurobi_hard",
    "gpbaa_2d_gurobi_hard_t": _DATA_DIR / "gpbaa_2d_gurobi_hard_t",
    # Both exact methods trimmed to a SINGLE fixed handoff budget (T=200s)
    # across the whole 150-300 ladder, rather than a per-size schedule. A
    # schedule that grows with N hands bigger instances more compute in
    # roughly the proportion needed to offset their difficulty, which flattens
    # the seed count to ~4 at every size and erases the size effect entirely.
    # At a fixed 200s the mean seed count falls monotonically 24.0 -> 9.2 ->
    # 7.4 -> 5.8 -> 5.2 -> 4.8 over N=150..300, and no instance is left with
    # an empty exact phase.
    "an_2d_gurobi_hard_t200":   _DATA_DIR / "an_2d_gurobi_hard_t200",
    "gpbaa_2d_gurobi_hard_t200": _DATA_DIR / "gpbaa_2d_gurobi_hard_t200",
    "an_2d_gurobi_rc":     _DATA_DIR / "an_2d_gurobi_rc",
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
    Returns an empty list (with a warning) when the file is absent or contains
    no solutions; does not fall back to other directories.
    """
    cache_key = f"{_PSEUDO_SOLUTIONS_SOURCE}:{instance_name}"
    if cache_key in _pseudo_cache:
        return _pseudo_cache[cache_key]

    source_dir = _PSEUDO_SOURCE_DIRS.get(
        _PSEUDO_SOLUTIONS_SOURCE, _PSEUDO_SOLUTIONS_DIR
    )
    json_path = source_dir / f"{instance_name}.json"

    if not json_path.exists():
        print(
            f"  WARN: pseudo-solver file not found: {json_path}",
            flush=True,
        )
        _pseudo_cache[cache_key] = []
        return []

    with open(json_path) as f:
        data = json.load(f)

    solutions = data if isinstance(data, list) else data.get("solutions", [])
    if not solutions:
        print(
            f"  WARN: pseudo-solver file has 0 solutions: {json_path}",
            flush=True,
        )
    _pseudo_cache[cache_key] = solutions
    return solutions


def _get_pseudo_solutions(
    problem: sims_problem.SimsDiscreteProblem,
    max_timestamp_s: float | None = None,
) -> list[dict]:
    """Get pseudo-solver solutions for a problem instance.

    If `max_timestamp_s` is given, only solutions with `timestamp_s` <=
    that value are returned, simulating GPBA-A having run for that duration.
    """
    solutions = _current_pseudo_solutions
    if max_timestamp_s is not None:
        solutions = [
            s for s in solutions
            if s.get("timestamp_s", 0.0) <= max_timestamp_s
        ]
    return solutions


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


def _pseudo_point(sol: dict, objectives: list[str]) -> list[int] | None:
    """Objective vector of a stored pseudo-solution, in `objectives` order."""
    key = {
        "min_cost": "cost",
        "cloud_coverage": "cloudy_area",
        "min_max_incidence_angle": "max_incidence_angle",
        "min_resolution": "min_resolutions_sum",
    }
    row: list[int] = []
    for obj in objectives:
        k = key.get(obj)
        if k is None or sol.get(k) is None:
            return None
        row.append(int(sol[k]))
    return row


# Fraction of the (nadir - ideal) range added above the nadir to place the HV
# reference point. Kept small: its only job is to give the extreme points
# non-zero volume (at the nadir exactly, each extreme contributes zero height).
_NADIR_MARGIN = 0.05


def compute_shared_bounds(
    all_points: list[list[int]], ndim: int, front: list[list[int]] | None = None
) -> list[list[int]]:
    """Per-objective [min, max] defining the HV reference box.

    `front`, when given, is a Pareto-optimal front (A&N's exact solutions). Its
    per-objective max IS the true nadir -- for two objectives the nadir is fixed
    by the lexicographic optima, and unsupported Pareto points lie strictly
    between them, so they cannot exceed it. The reference point is that nadir
    plus `_NADIR_MARGIN` of the range.

    Falling back to the max over `all_points` (every point ever recorded in
    every trace, including dominated early-generation EA candidates) makes the
    box 1.3-7.6x wider than the front -- measured on the N=200 row, up to 22x
    the front's area on mexico_city. In that box the two payoff extremes alone
    sweep ~99% of the hypervolume, every algorithm scores 0.97-0.99, and the
    indicator stops discriminating. Points beyond the nadir score zero, which is
    correct: they are dominated by some Pareto-optimal solution and are of no
    use to a decision maker holding the exact front.
    """
    bounds: list[list[int]] = []
    for i in range(ndim):
        vals = [p[i] for p in all_points]
        lo = min(vals)
        if front:
            hi = max(p[i] for p in front)
            rng = max(hi - lo, 1)
            hi = hi + int(rng * _NADIR_MARGIN)
        else:
            hi = max(vals)
            rng = max(hi - lo, 1)
            hi = hi + int(rng * 0.1)
        bounds.append([max(0, lo - 1), hi + 1])
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
    num_runs: int = 1,
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

    # Configs already recorded for this instance, when appending. Their traces
    # are reloaded rather than re-run, and phases 2-4 below recompute bounds
    # and every HV curve over the union -- so adding a config later yields the
    # same artifact as having run them all together, instead of a file whose
    # older entries were scored against different bounds.
    prior_artifact: dict = {}
    if _append_configs:
        prior_path = output_dir / f"{display_name}.json"
        if prior_path.exists():
            try:
                prior_artifact = json.loads(prior_path.read_text())
            except (OSError, json.JSONDecodeError) as e:
                print(f"  WARNING: cannot read {prior_path} to append: {e}", flush=True)

    def _stored_trace_path(label: str) -> Path | None:
        """Locate a previously written trace for `label`, if any.

        Two layouts exist: the current one writes into a `traces/` subdirectory
        keeping the label's case ("Hybrid_50-50"), while older runs wrote a
        lowercased flat file. Try the recorded path first, then both
        conventions, so appending works against either.
        """
        prior_cfg = prior_artifact.get("configs", {}).get(label)
        if prior_cfg is None:
            # A trace file on its own is not evidence of a completed config: an
            # interrupted run leaves orphan traces behind (the trace is written
            # as each config finishes, the artifact only at the end). Reusing
            # one would silently resurrect a partial run and report it with
            # final_solutions=0, so require a recorded entry and re-run
            # otherwise.
            return None
        rec = prior_cfg.get("trace_file")
        if rec:
            cand = output_dir / rec
            if cand.exists():
                return cand
        safe_new = label.replace("/", "-").replace(" ", "_").replace(":", "-")
        safe_old = (
            label.lower().replace(" ", "_").replace("+", "plus").replace("/", "").replace(":", "")
        )
        for cand in (
            output_dir / "traces" / f"{display_name}__{safe_new}.trace.tar.gz",
            output_dir / f"{display_name}__{safe_old}.trace.tar.gz",
        ):
            if cand.exists():
                return cand
        return None

    for cfg in configs:
        existing_trace = _stored_trace_path(cfg.label) if _append_configs else None
        if existing_trace is not None and num_runs == 1:
            trace_data = existing_trace.read_bytes()
            traces[cfg.label] = trace_data
            # Carry the recorded metadata forward; only the HV numbers are
            # recomputed, so solution counts and wall times stay truthful.
            prior_cfg = dict(prior_artifact.get("configs", {}).get(cfg.label) or {})
            for drop in ("final_hv", "delta_vs_pure_pls_pct", "curve"):
                prior_cfg.pop(drop, None)
            prior_cfg.setdefault("final_solutions", 0)
            prior_cfg.setdefault("wall_seconds", 0.0)
            prior_cfg["trace_bytes"] = len(trace_data)
            prior_cfg["reused_trace"] = True
            run_meta[cfg.label] = prior_cfg
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

            # Persist the raw trace and the final Pareto front, not just their
            # sizes. Without these the run cannot be re-analysed: recomputing HV
            # under a different reference point, or computing IGD+/spacing,
            # needs the actual objective vectors, and re-deriving them means
            # re-running every solver.
            trace_rel = None
            if trace_data:
                tdir = output_dir / "traces"
                tdir.mkdir(parents=True, exist_ok=True)
                safe = cfg.label.replace("/", "-").replace(" ", "_").replace(":", "-")
                tpath = tdir / f"{display_name}__{safe}.trace.tar.gz"
                tpath.write_bytes(trace_data)
                trace_rel = str(tpath.relative_to(output_dir))

            front = []
            for sol in result.final_solutions:
                pt = _pseudo_point(
                    {
                        "cost": getattr(sol, "cost", None),
                        "cloudy_area": getattr(sol, "cloudy_area", None),
                        "max_incidence_angle": getattr(sol, "max_incidence_angle", None),
                        "min_resolutions_sum": getattr(sol, "min_resolutions_sum", None),
                    },
                    OBJECTIVES,
                )
                if pt:
                    front.append(pt)

            run_meta[cfg.label] = dict(
                final_solutions=n_final,
                wall_seconds=round(wall, 1),
                trace_bytes=len(trace_data) if trace_data else 0,
                trace_file=trace_rel,
                final_front=front,
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

    # Prefer the exact front's nadir as the HV reference: it is the true nadir
    # (bi-objective, from A&N's lexicographic optima) and is independent of which
    # heuristics are being compared. Falls back to the trace-max box when no
    # exact solutions are loaded for this instance.
    _exact_front: list[list[int]] = []
    for _s in _current_pseudo_solutions:
        _pt = _pseudo_point(_s, OBJECTIVES)
        if _pt and len(_pt) == ndim:
            _exact_front.append(_pt)
    bounds = compute_shared_bounds(all_points, ndim, _exact_front or None)
    print(
        f"  {len(all_points)} total trace points; HV reference from "
        f"{'exact-front nadir' if _exact_front else 'trace max (fallback)'}",
        flush=True,
    )

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
                cfg.label.lower().replace(" ", "_").replace("+", "plus").replace("/", "").replace(":", "")
            )
            trace_path = output_dir / f"{display_name}__{safe_label}.trace.tar.gz"
            with open(trace_path, "wb") as f:
                f.write(traces[cfg.label])
            print(f"  Trace saved:    {trace_path}", flush=True)

    # ── Multi-run additional iterations ──────────────────────────────────
    if num_runs > 1:
        # run_hvs[label][0] = HV from run 0 (canonical run above)
        run_hvs: dict[str, list[float]] = {
            cfg.label: [result_artifact["configs"][cfg.label].get("final_hv", 0.0)]
            for cfg in configs
        }

        for run_idx in range(1, num_runs):
            print(f"\n  ── Additional run {run_idx + 1}/{num_runs} ──────────────────────────", flush=True)
            for cfg in configs:
                try:
                    run_result = cfg.run(problem, total_timeout, seed=run_idx)
                    trace_data = run_result.trace
                    hv = 0.0
                    if trace_data:
                        try:
                            curve = sims_problem.compute_hv_curve_from_trace(
                                trace_data, bounds, 2
                            )
                            hv = curve[-1][1] if curve else 0.0
                        except BaseException:
                            hv = 0.0
                    run_hvs[cfg.label].append(hv)
                    # Persist this run's full trace so its final front (the
                    # non-dominated subset of the trace) is reconstructable and
                    # HV can be re-normalized under any reference later. Run 0
                    # keeps its unsuffixed name for backward compat; runs >=1
                    # get a __run{n} suffix.
                    if cfg.save_trace and trace_data:
                        safe_label = (
                            cfg.label.lower().replace(" ", "_").replace("+", "plus").replace("/", "").replace(":", "")
                        )
                        run_trace_path = (
                            output_dir
                            / f"{display_name}__{safe_label}__run{run_idx}.trace.tar.gz"
                        )
                        with open(run_trace_path, "wb") as rf:
                            rf.write(trace_data)
                    print(
                        f"    [{cfg.label}] run {run_idx + 1}: HV={hv:.6f}", flush=True
                    )
                except Exception as e:
                    print(
                        f"    [{cfg.label}] run {run_idx + 1} ERROR: {e}", flush=True
                    )
                    run_hvs[cfg.label].append(0.0)

        # Compute mean and std over all runs; update artifact
        for cfg in configs:
            hvs = run_hvs[cfg.label]
            valid = [h for h in hvs if h > 0]
            if not valid:
                continue
            mean_hv = sum(valid) / len(valid)
            variance = sum((h - mean_hv) ** 2 for h in valid) / max(len(valid) - 1, 1)
            std_hv = variance ** 0.5
            result_artifact["configs"][cfg.label]["run_hvs"] = [round(h, 8) for h in hvs]
            result_artifact["configs"][cfg.label]["final_hv"] = round(mean_hv, 8)
            result_artifact["configs"][cfg.label]["final_hv_std"] = round(std_hv, 8)

        # Re-save JSON with multi-run statistics
        with open(artifact_path, "w") as f:
            json.dump(result_artifact, f, indent=2)
        print(f"\n  Multi-run artifact ({num_runs} runs) saved: {artifact_path}", flush=True)

    # ── Summary ─────────────────────────────────────────────────────────

    print(f"\n  Summary for {display_name}:", flush=True)
    for cfg in configs:
        info = result_artifact["configs"].get(cfg.label, {})
        fhv = info.get("final_hv", 0)
        nsol = info.get("final_solutions", 0)
        delta = info.get("delta_vs_pure_pls_pct", 0)
        std = info.get("final_hv_std", 0)
        std_str = f" ±{std:.4f}" if std > 0 else ""
        tag = "" if cfg.label == "Pure PLS" else f"Δ={delta:+.1f}%"
        print(
            f"    {cfg.label:22s}  HV={fhv:.6f}{std_str}  sols={nsol:6d}  {tag}",
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

    # Hybrid comparison bar chart across ALL instances (both small and large)
    if any(
        any(cfg_label in r["configs"] for cfg_label, _, _ in _HYBRID_BAR_STYLES)
        for r in all_results
    ):
        _plot_hybrid_comparison_bars(
            all_results,
            output_dir / "fig_hybrid_comparison_bars.png",
        )


def _plot_hybrid_comparison_bars(
    all_results: list[dict],
    output_path: "Path",
) -> None:
    """Grouped bar chart comparing all hybrid configs (PLS + EA variants) across
    every instance in all_results, ordered by size then city."""
    if not all_results:
        return

    # Sort: 100-image instances first, then 150
    sorted_results = sorted(all_results, key=lambda r: (r["num_images"], r["instance"]))
    instance_labels = [
        f"{_city_label(r['instance'])}\n{r['num_images']}" for r in sorted_results
    ]

    n_inst = len(sorted_results)
    inner_gap = 0.01
    group_gap = 0.45

    fig, ax = plt.subplots(figsize=(max(14, n_inst * 2.8), 6))

    # Compute per-instance GPBA-A HV from the FULL pseudo-solver solution set
    # (all solutions regardless of timestamp — equivalent to running GPBA-A for
    # the full pre-generation budget, not the reduced hybrid exact-phase slice).
    gpbaa_hvs: list[float] = []
    for r in sorted_results:
        inst = r["instance"]
        bounds = r.get("shared_bounds")
        sols = _load_pseudo_solutions(inst)
        hv = 0.0
        if sols and bounds:
            pts = [
                [s["cost"], s["cloudy_area"]] for s in sols
                if s["cost"] <= bounds[0][1] and s["cloudy_area"] <= bounds[1][1]
            ]
            try:
                hv = sims_problem.compute_hypervolume(pts, bounds, normalized=True) if pts else 0.0
            except BaseException:
                hv = 0.0
        gpbaa_hvs.append(hv)

    # Total bars = GPBA-A synthetic bar + configured styles
    _ALL_BARS: list[tuple[str | None, str, str]] = [
        (None, "GPBA-A (100:0)", "#1a1a2e"),
        *((lbl, dn, col) for lbl, dn, col in _HYBRID_BAR_STYLES),
    ]
    n_cfg = len(_ALL_BARS)
    bar_w = 0.65 / n_cfg
    group_span = n_cfg * bar_w + (n_cfg - 1) * inner_gap
    group_cx = [i * (group_span + group_gap) for i in range(n_inst)]
    var_off = [(j - (n_cfg - 1) / 2) * (bar_w + inner_gap) for j in range(n_cfg)]

    # Separator lines between size groups
    sizes_seen: list[int] = []
    for i, r in enumerate(sorted_results):
        if r["num_images"] not in sizes_seen:
            if sizes_seen:
                sep_x = (group_cx[i - 1] + group_cx[i]) / 2
                ax.axvline(sep_x, color="#cccccc", linewidth=1.0, zorder=1)
            sizes_seen.append(r["num_images"])

    legend_done: set[str] = set()
    for j, (cfg_label, display_name, color) in enumerate(_ALL_BARS):
        for i, r in enumerate(sorted_results):
            if cfg_label is None:
                hv = gpbaa_hvs[i]
                std = 0.0
            else:
                cfg_data = r["configs"].get(cfg_label, {})
                hv = cfg_data.get("final_hv", 0.0)
                std = cfg_data.get("final_hv_std", 0.0)
            if not hv:
                continue
            bx = group_cx[i] + var_off[j]
            ax.bar(
                bx,
                hv,
                width=bar_w,
                color=color,
                edgecolor="white",
                linewidth=0.3,
                zorder=3,
                label=display_name if display_name not in legend_done else None,
                yerr=std if std > 0 else None,
                error_kw=dict(ecolor="#444444", capsize=1.5, elinewidth=0.7, capthick=0.7, zorder=4),
            )
            legend_done.add(display_name)

    ax.set_xticks(group_cx)
    ax.set_xticklabels(instance_labels, fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Normalized Hypervolume")
    ax.set_title(
        "Hybrid Algorithm Comparison — All Instances (PLS + EA variants)",
        fontweight="bold",
    )
    all_hvs = gpbaa_hvs + [
        r["configs"].get(cfg_label, {}).get("final_hv", 0.0)
        for r in sorted_results
        for cfg_label, _, _ in _HYBRID_BAR_STYLES
        if cfg_label is not None
    ]
    y_min = max(0.0, min((h for h in all_hvs if h > 0), default=0.0) - 0.05)
    y_max = max((h for h in all_hvs), default=1.0) * 1.06
    ax.set_ylim(y_min, y_max)
    ax.legend(
        fontsize=7,
        ncol=3,
        loc="lower right",
        framealpha=0.9,
        title="Algorithm",
        title_fontsize=7,
    )
    ax.yaxis.grid(True, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save_fig(fig, output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}", flush=True)


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
            / f"{instance}__{label.lower().replace(' ', '_').replace('+', 'plus').replace('/', '').replace(':', '')}.trace.tar.gz",
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


# ── Pareto-front figures (paper `{instance}_pareto_fronts.png` reconstruction) ──
#
# Two-row layout mirroring the PPSN figure:
#   row 0 = GPBA-A (+PLS), row 1 = Aneja & Nair (+PLS)
#   col a) = phase-separated, hatched HV bar chart across 5 ratios
#           (exact-phase HV bottom / PLS-phase gain hatched top)
#   cols b/c/d = Pareto-front scatter at ratios 100:0, 50:50, 0:100
#
# Data source is the trace artifacts (not the broken sims-core geodata pipeline):
#   HV / PLS points  → eval_publication_{highs,an} traces
#   exact-phase front → exact-method pseudo-solutions trimmed by ratio × timeout
#     (gpbaa_2d_highs for GPBA-A, an_2d_highs for A&N)
# The two rows are normalised against a SHARED per-instance reference (the union
# of both experiments' bounds) so their HV bars are directly comparable — this
# is what the two auto-computed per-experiment references failed to guarantee.

# ratio → (bar label, trace slug for the PLS/hybrid part). ratio 1.0 has no
# trace (pure exact, from pseudo seeds); 0.0 = pure PLS (default_pls trace).
_PF_RATIOS: list[tuple[float, str]] = [
    (1.00, "100:0"),
    (0.80, "80:20"),
    (0.50, "50:50"),
    (0.20, "20:80"),
    (0.00, "0:100"),
]
_PF_RATIO_TO_SLUG: dict[float, str] = {
    0.80: "hybrid_8020",
    0.50: "hybrid_5050",
    0.20: "hybrid_2080",
    0.00: "default_pls",
}
# Scatter panels shown (matching the paper: pure exact, balanced, pure PLS).
_PF_SCATTER_RATIOS: list[float] = [1.00, 0.50, 0.00]

# ── On-page sizing ────────────────────────────────────────────────────────────
# The figure is included as `\includegraphics[width=\textwidth,
# height=0.4\textheight,keepaspectratio]`, so LaTeX rescales it and every font
# shrinks by that same factor. Sizes must be chosen *on the page*, not in the
# figure. The old 24x12 in figure was scaled by 4.80/24 = 0.20, printing a
# `fontsize=15` axis label at 3 pt — the reason labels look unreadable.
#
# Fix: build the figure at exactly its printed size, so 1 matplotlib point == 1
# printed point and the numbers below mean what they say. This also requires
# dropping `bbox_inches="tight"`, which silently grows the canvas past the size
# we asked for (and therefore reintroduces an unknown shrink factor).
_LNCS_TEXTWIDTH_IN = 122.0 / 25.4  # 4.803 in
_LNCS_TEXTHEIGHT_IN = 193.0 / 25.4  # 7.598 in
_PF_FIG_W = _LNCS_TEXTWIDTH_IN
_PF_FIG_H = 0.40 * _LNCS_TEXTHEIGHT_IN  # both \includegraphics limits bind at 1:1

# Printed points. LNCS body is 10 pt, captions 9 pt. Eight panels across 122 mm
# leaves ~30 mm per panel, so caption parity is not reachable; 6.5-8 pt is the
# usable band, and it is only reachable at all because the scatter panels share
# axes (see below) and so pay for tick labels once per row instead of per panel.
_PF_PANEL_LETTERS = "abcdefgh"
_PF_TITLE_PT = 8.0
_PF_LABEL_PT = 7.5
_PF_TICK_PT = 6.5
_PF_LEGEND_PT = 7.0

# Ink that is not data: kept thin and grey so it recedes behind the markers.
_PF_AXIS_GREY = "#4d4d4d"
_PF_GRID_GREY = "#d9d9d9"
_PF_SPINE_LW = 0.6
_PF_GRID_LW = 0.4


def _pf_style_axes(ax: "plt.Axes", grid_axis: str = "both") -> None:
    """Apply the shared panel styling: despined, grid behind data, grey rules.

    Top/right spines carry no information and, at 30 mm per panel, a four-sided
    black box is the single heaviest element on the page — removing it is what
    makes the markers read as the subject of the panel.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(_PF_SPINE_LW)
        ax.spines[side].set_color(_PF_AXIS_GREY)
    ax.tick_params(
        labelsize=_PF_TICK_PT, colors=_PF_AXIS_GREY,
        width=_PF_SPINE_LW, length=2.5, pad=1.5,
    )
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color("black")  # ticks recede, but their labels must stay legible
    ax.grid(axis=grid_axis, color=_PF_GRID_GREY, linewidth=_PF_GRID_LW, zorder=0)
    ax.set_axisbelow(True)


def _pf_save(fig: "plt.Figure", output_dir: "Path", stem: str) -> "Path":
    """Write `stem` as PNG and EPS into sibling `_png` / `_eps` directories.

    LaTeX wants one format and reviewers/preprints often want the other, so the
    two are kept apart rather than interleaved in one directory — a glob for
    figures then never has to filter by extension.

    No `bbox_inches="tight"`: the canvas must stay exactly _PF_FIG_W wide so
    `width=\\textwidth` scales it by 1.0 and the point sizes hold on the page.
    """
    png_dir = output_dir.with_name(output_dir.name + "_png")
    eps_dir = output_dir.with_name(output_dir.name + "_eps")
    png_dir.mkdir(parents=True, exist_ok=True)
    eps_dir.mkdir(parents=True, exist_ok=True)
    png = png_dir / f"{stem}.png"
    fig.savefig(str(png), dpi=600)
    fig.savefig(str(eps_dir / f"{stem}.eps"))
    return png


def _pf_panel_title(ax: "plt.Axes", letter: str, text: str) -> None:
    """Left-aligned panel title with a bold letter and a regular-weight name.

    Left alignment is the journal convention and, unlike a centred title, it
    stays put as the panel width changes between the bar and scatter columns.
    """
    ax.set_title(
        f"$\\bf{{{letter})}}$  {text}",
        fontsize=_PF_TITLE_PT, loc="left", pad=5,
    )


def _pf_tint(hex_color: str, amount: float = 0.80) -> tuple[float, float, float]:
    """Blend `hex_color` toward white by `amount` (1.0 == white)."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return tuple(c + (1.0 - c) * amount for c in (r, g, b))


def _pf_front(
    ax: "plt.Axes",
    points: list[tuple[float, float]],
    marker: str,
    size: float,
    color: str,
    z: int,
) -> None:
    """Draw a 2-objective front as unconnected markers.

    No connecting line: each point is a distinct solution, and a line between
    them would imply intermediate solutions exist along it, which they do not.
    """
    ordered = sorted(points)
    ax.scatter(
        [x for x, _y in ordered], [y for _x, y in ordered],
        marker=marker, s=size, c=color, linewidths=0, edgecolors="none",
        zorder=z,
    )


def _pf_pareto_filter(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Non-dominated subset for a 2-objective minimisation problem."""
    if not points:
        return []
    pts = sorted(set(points))  # sort by cost asc, then cloudy asc
    front: list[tuple[float, float]] = []
    best_y = float("inf")
    for x, y in pts:
        if y < best_y:
            front.append((x, y))
            best_y = y
    return front


def _pf_exact_front(
    seeds: list[dict], exact_time: float
) -> list[tuple[float, float]]:
    """Non-dominated exact-phase points: pseudo seeds with timestamp ≤ exact_time."""
    pts = [
        (float(s["cost"]), float(s["cloudy_area"]))
        for s in seeds
        if s.get("timestamp_s", 0.0) <= exact_time
    ]
    return _pf_pareto_filter(pts)


def _pf_exact_hv(
    seeds: list[dict], bounds: list[list[int]], exact_time: float
) -> float:
    """HV of the exact-phase front — pseudo seeds cut off at exact_time — computed
    against the same reference point as the stored final HVs (upper bound + 1)."""
    # compute_hypervolume requires integer points (Vec<Vec<u64>>).
    pts = [[int(x), int(y)] for x, y in _pf_exact_front(seeds, exact_time)]
    if not pts:
        return 0.0
    # Reference point = the upper bound (the nadir), matching the normalisation
    # used for the stored final HVs. Must lie within bounds, so it is b[1], not b[1]+1.
    reference_point = [b[1] for b in bounds]
    return sims_problem.compute_hypervolume(
        pts, bounds, reference_point=reference_point, normalized=True
    )


def _pf_trace_front(
    trace_dir: Path, instance: str, slug: str
) -> list[tuple[float, float]]:
    """Non-dominated points from a trace file, or [] if the trace is absent."""
    path = trace_dir / f"{instance}__{slug}.trace.tar.gz"
    if not path.exists():
        return []
    try:
        raw = extract_all_objectives(path.read_bytes(), 2)
    except Exception:
        return []
    return _pf_pareto_filter([(float(p[0]), float(p[1])) for p in raw])


def _pf_widen_bounds(
    bounds: list[list[int]], seed_sets: list[list[dict]]
) -> list[list[int]]:
    """Widen `bounds` so every pseudo-seed point lies inside it.

    The seed sets are generated with their own (longer) budget, so a seed can
    sit outside the bounds derived from the run traces — `compute_hypervolume`
    rejects any point outside its bounds. Widening keeps every bar in the figure
    on one common reference, which is what makes them comparable.
    """
    out = [list(b) for b in bounds]
    for seeds in seed_sets:
        for sol in seeds:
            for i, key in enumerate(("cost", "cloudy_area")):
                if i >= len(out):
                    break
                v = int(float(sol[key]))
                out[i][0] = min(out[i][0], v)
                out[i][1] = max(out[i][1], v)
    return out


def _pf_row_data(
    result: dict,
    trace_dir: Path,
    seeds: list[dict],
    merged_bounds: list[list[int]],
    pls_result: dict,
    pls_dir: Path,
    num_points: int,
) -> dict:
    """Assemble bar + scatter data for one method row against merged_bounds.

    ``pls_result`` / ``pls_dir`` supply the pure-PLS (0:100) config, which is
    method-independent and taken from the GPBA-A experiment (the A&N experiment
    never ran a standalone PLS config).
    """
    instance = result["instance"]
    timeout = float(result.get("timeout_s", 0.0))
    recomputed = _recompute_result_with_bounds(
        result, trace_dir, merged_bounds, num_points
    )
    cfgs = recomputed.get("configs", {})
    orig_cfgs = result.get("configs", {})

    def _avg_final(label: str) -> float:
        """Merged-bounds 10-run MEAN final HV.

        Per-run raw points exist only for run 0, so the mean cannot be recomputed
        against merged bounds directly. Instead rescale the merged run-0 HV by the
        stored mean/run-0 ratio (both in the experiment's own normalisation). run-0
        tracks the mean to <0.3%, so this is a faithful correction that preserves
        cross-row comparability of the merged reference.
        """
        merged_run0 = float(cfgs.get(label, {}).get("final_hv", 0.0))
        c = orig_cfgs.get(label, {})
        old_mean = float(c.get("final_hv", 0.0))
        rh = c.get("run_hvs", [])
        old_run0 = float(rh[0]) if rh else old_mean
        if old_run0 > 0 and old_mean > 0:
            return merged_run0 * (old_mean / old_run0)
        return merged_run0

    # Pure-PLS (0:100), merged-bounds 10-run mean via the same rescale.
    pls_recomp = _recompute_result_with_bounds(
        pls_result, pls_dir, merged_bounds, num_points
    )
    pls_merged_run0 = float(
        pls_recomp.get("configs", {}).get("Default PLS", {}).get("final_hv", 0.0)
    )
    _plsc = pls_result.get("configs", {}).get("Default PLS", {})
    _pls_old_mean = float(_plsc.get("final_hv", 0.0))
    _pls_rh = _plsc.get("run_hvs", [])
    _pls_old_run0 = float(_pls_rh[0]) if _pls_rh else _pls_old_mean
    pls_final_hv = (
        pls_merged_run0 * (_pls_old_mean / _pls_old_run0)
        if (_pls_old_run0 > 0 and _pls_old_mean > 0)
        else pls_merged_run0
    )

    bars: dict[float, dict] = {}
    for ratio, _name in _PF_RATIOS:
        if ratio == 1.00:
            # Pure exact: deterministic seed front, no run averaging.
            hv_p1 = _pf_exact_hv(seeds, merged_bounds, timeout)
            bars[ratio] = dict(hv_p1=hv_p1, hv_p2=0.0, final_hv=hv_p1)
        elif ratio == 0.00:
            bars[ratio] = dict(hv_p1=0.0, hv_p2=pls_final_hv, final_hv=pls_final_hv)
        else:
            label = {0.80: "Hybrid 80:20", 0.50: "Hybrid 50:50", 0.20: "Hybrid 20:80"}[ratio]
            final_hv = _avg_final(label)
            if final_hv == 0.0:
                continue
            hv_p1 = _pf_exact_hv(seeds, merged_bounds, ratio * timeout)
            hv_p1 = min(hv_p1, final_hv)
            bars[ratio] = dict(
                hv_p1=hv_p1, hv_p2=max(0.0, final_hv - hv_p1), final_hv=final_hv
            )

    # Scatter fronts (raw objective units; normalised at plot time).
    scatter: dict[float, dict] = {}
    for ratio in _PF_SCATTER_RATIOS:
        exact = _pf_exact_front(seeds, (timeout if ratio == 1.0 else ratio * timeout))
        if ratio == 1.00:
            pls_pts: list[tuple[float, float]] = []
        elif ratio == 0.00:
            pls_pts = _pf_trace_front(pls_dir, instance, "default_pls")
            exact = []
        else:
            pls_pts = _pf_trace_front(trace_dir, instance, _PF_RATIO_TO_SLUG[ratio])
        scatter[ratio] = dict(exact=exact, pls=pls_pts)

    return dict(bars=bars, scatter=scatter)


# Second-phase engines, in column order. Each entry supplies the labels the
# experiment JSON uses for that engine at each ratio: the three seeded hybrids
# and the cold-start (0:100) run.
#
# Only the GPBA-A experiment ran the cold-start configs — with no exact phase
# there is nothing method-specific about them, so the A&N row reuses the same
# numbers rather than leaving its 0:100 bar empty. This mirrors how the front
# figure already sources "Default PLS" for both rows.
_EA_ENGINES: list[tuple[str, str, dict[float, str], str]] = [
    # (column title, colour, {ratio: hybrid label}, cold-start label)
    (
        "PLS", "#1f77b4",
        {0.80: "Hybrid 80:20", 0.50: "Hybrid 50:50", 0.20: "Hybrid 20:80"},
        "Default PLS",
    ),
    (
        "NSGA-II", "#2ca02c",
        {0.80: "Hybrid NSGA-II 80:20", 0.50: "Hybrid NSGA-II 50:50",
         0.20: "Hybrid NSGA-II 20:80"},
        "Baseline NSGA-II",
    ),
    (
        "NSGA-III", "#9467bd",
        {0.80: "Hybrid NSGA-III 80:20", 0.50: "Hybrid NSGA-III 50:50",
         0.20: "Hybrid NSGA-III 20:80"},
        "Baseline NSGA-III",
    ),
    (
        "MOEA/D", "#17becf",
        {0.80: "Hybrid MOEA/D 80:20", 0.50: "Hybrid MOEA/D 50:50",
         0.20: "Hybrid MOEA/D 20:80"},
        "Baseline MOEA/D",
    ),
]


def _ea_bar_data(
    result: dict,
    trace_dir: Path,
    seeds: list[dict],
    merged_bounds: list[list[int]],
    cold_result: dict,
    cold_dir: Path,
    num_points: int,
) -> dict[str, dict[float, dict]]:
    """Phase-decomposed HV bars per engine per ratio, against merged bounds.

    Returns {engine title: {ratio: {hv_p1, hv_p2, final_hv, std}}}. The exact
    phase (hv_p1) is the seed front truncated at `ratio * timeout`, so it is the
    same for every engine at a given ratio — which is the point of the figure:
    the engines differ only in what they add on top of an identical seeding.
    """
    timeout = float(result.get("timeout_s", 0.0))
    recomputed = _recompute_result_with_bounds(
        result, trace_dir, merged_bounds, num_points
    )
    cfgs = recomputed.get("configs", {})
    orig_cfgs = result.get("configs", {})

    cold_recomp = _recompute_result_with_bounds(
        cold_result, cold_dir, merged_bounds, num_points
    )
    cold_cfgs = cold_recomp.get("configs", {})
    cold_orig = cold_result.get("configs", {})

    def _rescaled(label: str, merged: dict, original: dict) -> tuple[float, float]:
        """Merged-bounds 10-run mean and std for `label`.

        Per-run raw points exist only for run 0, so the mean cannot be recomputed
        against merged bounds directly; rescale the merged run-0 HV by the stored
        mean/run-0 ratio, exactly as the front figure does. The std is carried
        across as a *relative* spread, which the same rescale preserves.
        """
        merged_run0 = float(merged.get(label, {}).get("final_hv", 0.0))
        c = original.get(label, {})
        old_mean = float(c.get("final_hv", 0.0))
        rh = c.get("run_hvs", [])
        old_run0 = float(rh[0]) if rh else old_mean
        mean = (
            merged_run0 * (old_mean / old_run0)
            if (old_run0 > 0 and old_mean > 0)
            else merged_run0
        )
        old_std = float(c.get("final_hv_std", 0.0))
        std = mean * (old_std / old_mean) if old_mean > 0 else 0.0
        return mean, std

    out: dict[str, dict[float, dict]] = {}
    for title, _colour, hybrid_labels, cold_label in _EA_ENGINES:
        bars: dict[float, dict] = {}
        for ratio, _name in _PF_RATIOS:
            if ratio == 1.00:
                # Pure exact: a deterministic seed front, so no run spread.
                hv = _pf_exact_hv(seeds, merged_bounds, timeout)
                bars[ratio] = dict(hv_p1=hv, hv_p2=0.0, final_hv=hv, std=0.0)
            elif ratio == 0.00:
                mean, std = _rescaled(cold_label, cold_cfgs, cold_orig)
                if mean == 0.0:
                    continue
                bars[ratio] = dict(hv_p1=0.0, hv_p2=mean, final_hv=mean, std=std)
            else:
                mean, std = _rescaled(hybrid_labels[ratio], cfgs, orig_cfgs)
                if mean == 0.0:
                    continue
                hv_p1 = min(_pf_exact_hv(seeds, merged_bounds, ratio * timeout), mean)
                bars[ratio] = dict(
                    hv_p1=hv_p1, hv_p2=max(0.0, mean - hv_p1),
                    final_hv=mean, std=std,
                )
        out[title] = bars
    return out


# Width of the parenthesised standard deviation box. The column headers are
# padded with an empty box of exactly this width (see `generate_ea_tables`), so
# that a right-aligned header lands over the mean rather than over the mean plus
# its SD. Both uses must stay in lockstep, hence the shared constant — and both
# must be set in \tiny, since `em` is relative to the current font size.
_EA_STD_BOX = "2.9em"


def _ea_std_tex(std: float) -> str:
    """Render a standard deviation for the HV tables.

    Several configurations are effectively deterministic across the 10 runs, and
    at 4 decimals their SD renders as a column of `0.0000`, which reads as a
    missing value rather than as "smaller than the displayed precision".
    """
    # Parentheses rather than "$\pm$", and 3 decimals: six columns of
    # "0.9621 $\pm$ 0.0001" overflow \textwidth even at \footnotesize.
    if std <= 0.0:
        body = "(0.000)"
    elif std < 5e-4:
        body = "($<$.001)"
    else:
        body = f"({std:.3f})"
    # Fixed-width box so every cell has the same trailing element. Without it a
    # cell whose SD is absent or narrower renders shorter, and since the columns
    # are right-aligned the *means* end up on different vertical lines.
    return f"\\tiny{{\\makebox[{_EA_STD_BOX}][r]{{{body}}}}}"


def _bergmann_hommel_adjust(pvals: dict[tuple, float], k: int) -> dict[tuple, float]:
    """Bergmann-Hommel adjusted p-values for all-pairs comparisons.

    Neither scipy nor statsmodels provides this. `statsmodels`' `'hommel'` is
    Hommel's procedure over a flat family of p-values; Bergmann-Hommel is the
    all-pairs procedure that additionally exploits the *logical* dependencies
    between pairwise hypotheses (if A=B and B=C then A=C cannot be false), which
    is why Garcia & Herrera (JMLR 2008) recommend it for exactly this setting.

    A set of hypotheses is *exhaustive* when it is the set of all within-group
    pairs of some partition of the k algorithms. The adjusted p-value for H_ij
    is then

        max over exhaustive E containing H_ij of  |E| * min_{h in E} p_h.

    Enumerating set partitions costs Bell(k) -- 4140 for k=8, 21147 for k=9 --
    which is why the literature caps this procedure at 9 algorithms.
    """
    idx = list(range(k))

    def partitions(collection: list):
        if len(collection) == 1:
            yield [collection]
            return
        first, rest = collection[0], collection[1:]
        for smaller in partitions(rest):
            for n, subset in enumerate(smaller):
                yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
            yield [[first]] + smaller

    adjusted = {pair: 0.0 for pair in pvals}
    for part in partitions(idx):
        exhaustive = [
            (min(a, b), max(a, b))
            for block in part
            for i, a in enumerate(block)
            for b in block[i + 1:]
        ]
        if not exhaustive:
            continue
        min_p = min(pvals[h] for h in exhaustive)
        candidate = len(exhaustive) * min_p
        for h in exhaustive:
            if candidate > adjusted[h]:
                adjusted[h] = candidate
    # The raw max-over-exhaustive-sets values can invert (a hypothesis with a
    # smaller raw p ending up with a larger adjusted p). scmamp -- the reference
    # implementation, and almost certainly the source of the paper's published
    # ranking -- fixes this with `correctForMonotocity`: a running maximum over
    # the p-values ordered by raw p ascending. Reproduced here so the two agree;
    # it can only raise values, so it is the conservative direction.
    capped = {h: min(1.0, v) for h, v in adjusted.items()}
    running = 0.0
    for h in sorted(capped, key=lambda h: pvals[h]):
        running = max(running, capped[h])
        capped[h] = running
    return capped


# Treatments for the Friedman test, extending the paper's Eq. 1 with the EA
# baselines. Each entry is (label, row index into `_ea_bar_data` output, engine,
# ratio); row 0 is GPBA-A-seeded, row 1 is Aneja & Nair-seeded.
_FRIEDMAN_TREATMENTS: list[tuple[str, int, str, float]] = [
    ("A\\&N+PLS", 1, "PLS", 0.50),
    ("GPBA-A+PLS", 0, "PLS", 0.50),
    ("A\\&N", 1, "PLS", 1.00),
    ("GPBA-A", 0, "PLS", 1.00),
    ("PLS", 0, "PLS", 0.00),
    ("NSGA-II", 0, "NSGA-II", 0.00),
    ("NSGA-III", 0, "NSGA-III", 0.00),
    ("MOEA/D", 0, "MOEA/D", 0.00),
]


def run_friedman_test(
    highs_dir: Path,
    an_dir: Path,
    output_dir: Path,
    filter_regex: str | None = None,
    num_points: int = 30,
    alpha: float = 0.05,
) -> None:
    """Friedman omnibus + all-pairs post-hoc with Bergmann-Hommel correction.

    Blocks are instances, treatments are algorithms, and the response is the
    mean hypervolume. Ranks are computed within each instance, so the result is
    invariant to any per-instance monotone rescaling -- but *only* if every
    treatment in one block shares one normalisation, which is why the matrix is
    built from `_ea_bar_data` against merged bounds rather than from the raw
    per-experiment `final_hv` values (the GPBA-A and A&N experiments normalise
    against different bounds).
    """
    import numpy as np
    from scipy.stats import friedmanchisquare

    gpbaa_seeds_dir = _PSEUDO_SOURCE_DIRS["gpbaa_2d_highs"]
    an_seeds_dir = _PSEUDO_SOURCE_DIRS["an_2d_highs"]

    def _load(d: Path, inst: str) -> dict | None:
        p = d / f"{inst}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def _seeds(d: Path, inst: str) -> list[dict]:
        p = d / f"{inst}.json"
        return json.loads(p.read_text()).get("solutions", []) if p.exists() else []

    pat = re.compile(filter_regex) if filter_regex else None
    instances = sorted(
        p.stem
        for p in highs_dir.glob("*.json")
        if p.stem != "all_experiments" and (an_dir / p.name).exists()
    )
    if pat:
        instances = [i for i in instances if pat.search(i)]

    labels = [t[0] for t in _FRIEDMAN_TREATMENTS]
    matrix: list[list[float]] = []
    used: list[str] = []
    for inst in instances:
        highs, an = _load(highs_dir, inst), _load(an_dir, inst)
        if highs is None or an is None:
            continue
        hb, ab = highs.get("shared_bounds"), an.get("shared_bounds")
        if not hb or not ab:
            continue
        merged = _merge_bounds(hb, ab)
        rows = [
            _ea_bar_data(highs, highs_dir, _seeds(gpbaa_seeds_dir, inst),
                         merged, highs, highs_dir, num_points),
            _ea_bar_data(an, an_dir, _seeds(an_seeds_dir, inst),
                         merged, highs, highs_dir, num_points),
        ]
        try:
            row = [
                rows[r][engine][ratio]["final_hv"]
                for _lbl, r, engine, ratio in _FRIEDMAN_TREATMENTS
            ]
        except KeyError as exc:
            print(f"  {inst}: missing {exc}, skipping", flush=True)
            continue
        if any(v <= 0 for v in row):
            print(f"  {inst}: non-positive HV, skipping", flush=True)
            continue
        matrix.append(row)
        used.append(inst)

    if len(matrix) < 3:
        print(f"Not enough complete instances ({len(matrix)})", flush=True)
        return

    data = np.asarray(matrix)  # (N instances, k treatments)
    n, k = data.shape

    # Rank within each instance, best (highest HV) = rank 1.
    order = (-data).argsort(axis=1).argsort(axis=1) + 1.0
    # Average tied ranks, or two algorithms with identical HV would be split
    # arbitrarily by argsort -- and identical values are common here.
    ranks = np.empty_like(order)
    for i in range(n):
        for j in range(k):
            tied = np.isclose(data[i], data[i][j], rtol=0, atol=5e-7)
            ranks[i][j] = order[i][tied].mean()
    avg = ranks.mean(axis=0)

    stat, p_omni = friedmanchisquare(*[data[:, j] for j in range(k)])

    # Pairwise p-values from scikit-posthocs rather than hand-rolled: Conover's
    # post-hoc for a Friedman design, unadjusted (the Bergmann-Hommel step is
    # applied separately below).
    import scikit_posthocs as sp

    pmatrix = sp.posthoc_conover_friedman(data, p_adjust=None).values
    pvals: dict[tuple, float] = {
        (i, j): float(pmatrix[i][j]) for i in range(k) for j in range(i + 1, k)
    }

    adj = _bergmann_hommel_adjust(pvals, k)

    print(f"\nFriedman test over {n} instances, {k} algorithms")
    print(f"  chi2 = {stat:.4f}   p = {p_omni:.3e}"
          f"   ({'reject' if p_omni < alpha else 'retain'} H0 at alpha={alpha})")
    print("\nAverage ranks (1 = best):")
    for lbl, r in sorted(zip(labels, avg), key=lambda t: t[1]):
        print(f"  {lbl.replace(chr(92), ''):14s} {r:.2f}")

    print(f"\nPairwise Bergmann-Hommel adjusted p-values (alpha={alpha}):")
    sig: set[tuple] = set()
    for (i, j), p in sorted(adj.items(), key=lambda kv: kv[1]):
        mark = "*" if p < alpha else " "
        if p < alpha:
            sig.add((i, j))
        print(f"  {mark} {labels[i].replace(chr(92),''):14s} vs "
              f"{labels[j].replace(chr(92),''):14s} p = {p:.4g}")

    # Chain notation, as in the paper's Eq. 1: consecutive ranked algorithms are
    # joined by "succ" only when their difference survives the correction.
    ordered = sorted(range(k), key=lambda j: avg[j])
    parts = []
    for pos, j in enumerate(ordered):
        parts.append(f"\\text{{{labels[j]} ({avg[j]:.2f})}}")
        if pos + 1 < len(ordered):
            nxt = ordered[pos + 1]
            pair = (min(j, nxt), max(j, nxt))
            parts.append("\\succ" if pair in sig else "\\sim")
    equation = " ".join(parts)
    print("\nLaTeX (\\succ = significant at alpha, \\sim = not significant):")
    print(" ", equation)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "friedman_ranking.tex"
    out.write_text(equation + "\n")
    print(f"\n  Saved: {out}", flush=True)


def generate_ea_tables(
    highs_dir: Path,
    an_dir: Path,
    output_dir: Path,
    filter_regex: str | None = None,
    num_points: int = 30,
) -> None:
    """Emit one LaTeX longtable per exact method comparing the second-phase
    algorithms: `ea_hv_gpbaa.tex` and `ea_hv_aneja.tex`.

    Numbers come from `_ea_bar_data`, the same function that feeds the bar
    figures, so a table cell and its bar can never disagree.

    The 100:0 row is the exact phase alone -- no second-phase algorithm has run
    yet -- so it spans the four algorithm columns instead of repeating one value
    four times.
    """
    gpbaa_seeds_dir = _PSEUDO_SOURCE_DIRS["gpbaa_2d_highs"]
    an_seeds_dir = _PSEUDO_SOURCE_DIRS["an_2d_highs"]
    engines = [title for title, _c, _h, _cold in _EA_ENGINES]

    def _load(d: Path, inst: str) -> dict | None:
        p = d / f"{inst}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def _seeds(d: Path, inst: str) -> list[dict]:
        p = d / f"{inst}.json"
        return json.loads(p.read_text()).get("solutions", []) if p.exists() else []

    output_dir.mkdir(parents=True, exist_ok=True)
    pat = re.compile(filter_regex) if filter_regex else None
    instances = sorted(
        p.stem
        for p in highs_dir.glob("*.json")
        if p.stem != "all_experiments" and (an_dir / p.name).exists()
    )
    if pat:
        instances = [i for i in instances if pat.search(i)]
    if not instances:
        print(f"No shared instances found in {highs_dir} and {an_dir}", flush=True)
        return

    # row index -> (which row of `rows` to read, output filename, caption, label)
    targets = [
        (0, "ea_hv_gpbaa.tex", "GPBA-A", "tab:ea_hv_gpbaa"),
        (1, "ea_hv_aneja.tex", "Anytime Aneja \\& Nair", "tab:ea_hv_aneja"),
    ]
    collected: list[dict[str, dict[str, dict[float, dict]]]] = []

    for inst in instances:
        highs, an = _load(highs_dir, inst), _load(an_dir, inst)
        if highs is None or an is None:
            continue
        hb, ab = highs.get("shared_bounds"), an.get("shared_bounds")
        if not hb or not ab:
            print(f"  {inst}: missing shared_bounds, skipping", flush=True)
            continue
        merged = _merge_bounds(hb, ab)
        collected.append(
            {
                "instance": inst,
                "rows": [
                    _ea_bar_data(highs, highs_dir, _seeds(gpbaa_seeds_dir, inst),
                                 merged, highs, highs_dir, num_points),
                    _ea_bar_data(an, an_dir, _seeds(an_seeds_dir, inst),
                                 merged, highs, highs_dir, num_points),
                ],
            }
        )

    for row_idx, fname, method, label in targets:
        lines: list[str] = []
        ncol = len(engines)
        colspec = "ll" + "r" * ncol
        # Pad each algorithm header by the SD box width so its right edge lines
        # up with the means below it, not with the SDs.
        # \scriptsize: with the pad added, "NSGA-III" at \footnotesize becomes
        # wider than "0.9981" plus its box, so the header — not the data — would
        # set the column width and push the table 16pt past \textwidth.
        pad = f"\\tiny{{\\makebox[{_EA_STD_BOX}][r]{{}}}}"
        header = (
            " & ".join(
                ["\\scriptsize{Instance}", "\\scriptsize{Ratio}"]
                + [f"\\scriptsize{{{e}}}{pad}" for e in engines]
            )
            + " \\\\"
        )
        cap = (
            f"Hypervolume by second-phase algorithm, seeded by {method}. "
            "Values are the mean over 10 runs, with the standard deviation in "
            "parentheses; the best value for each instance is in bold, "
            "normalised against bounds shared by both exact methods. The "
            "100:0 row is the exact phase alone."
        )
        lines += [
            # Six columns of "0.9621 +/- 0.0001" overflow \textwidth by ~42pt at
            # \normalsize. The group is closed after \end{longtable}; longtable
            # tolerates being wrapped this way as long as the font and column
            # separation are set before it starts.
            "\\begingroup",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{2.5pt}",
            f"\\begin{{longtable}}[!htb]{{{colspec}}}",
            f"\\caption{{{cap}}} \\label{{{label}}} \\\\",
            "\\toprule", header, "\\midrule", "\\endfirsthead",
            f"\\caption[]{{{cap} (cont.)}} \\\\",
            "\\toprule", header, "\\midrule", "\\endhead",
            "\\midrule",
            f"\\multicolumn{{{ncol + 2}}}{{r}}{{Continued on next page}} \\\\",
            "\\midrule", "\\endfoot",
            "\\bottomrule", "\\endlastfoot",
        ]
        for rec in collected:
            bars = rec["rows"][row_idx]
            ratios = [r for r, _n in _PF_RATIOS if r in bars.get(engines[0], {})]
            inst_tex = rec["instance"].replace("_", "\\_")
            # Plain text, not \texttt: the monospace instance names are the
            # widest column and the original tables did not use it either.
            lines.append(f"\\multirow[t]{{{len(ratios)}}}{{*}}{{{inst_tex}}}")

            # Best hypervolume anywhere in this instance's block, including the
            # exact-only 100:0 row. Compared at the 4 decimals actually printed,
            # so two cells that read the same are both marked rather than one
            # winning on digits the reader cannot see.
            best = max(
                round(bars[e][r]["final_hv"], 4)
                for e in engines
                for r in ratios
                if r in bars.get(e, {})
            )

            def _hv(engine: str, ratio: float) -> str:
                v = bars[engine][ratio]["final_hv"]
                txt = f"{v:.4f}"
                return f"\\textbf{{{txt}}}" if round(v, 4) >= best else txt

            for i, ratio in enumerate(ratios):
                # "100:0" as in the figures, not "100\% : 0\%" -- the percent
                # signs cost ~22pt of table width and the caption defines the
                # notation anyway.
                name = dict(_PF_RATIOS)[ratio]
                if ratio == 1.00:
                    cells = (
                        f"\\multicolumn{{{ncol}}}{{c}}{{{_hv(engines[0], ratio)}}}"
                    )
                else:
                    cells = " & ".join(
                        f"{_hv(e, ratio)} {_ea_std_tex(bars[e][ratio]['std'])}"
                        if ratio in bars.get(e, {}) else "---"
                        for e in engines
                    )
                # The instance name sits in the \multirow cell emitted above, so
                # every data row starts with an empty first column.
                lines.append(f" & {name} & {cells} \\\\")
            lines.append(f"\\cline{{1-{ncol + 2}}}")
        lines += ["\\end{longtable}", "\\endgroup"]

        out = output_dir / fname
        out.write_text("\n".join(lines) + "\n")
        print(f"  Saved: {out}  ({len(collected)} instances)", flush=True)


def generate_ea_bar_figures(
    highs_dir: Path,
    an_dir: Path,
    output_dir: Path,
    filter_regex: str | None = None,
    num_points: int = 30,
    gpbaa_seeds_override: Path | None = None,
    an_seeds_override: Path | None = None,
) -> None:
    """Per-instance `{instance}_ea_bars.png`: the same 2x4 layout as the front
    figure, but with every panel a phase-decomposed HV bar chart.

    Rows are the exact method that produced the seeds (GPBA-A, Aneja & Nair);
    columns are the second-phase engine (PLS, NSGA-II, NSGA-III, MOEA/D). Every
    panel shares one y range, so the question the figure answers -- does exact
    seeding help the EAs the way it helps PLS? -- is read off directly by
    comparing the hatched gain across columns.
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator

    gpbaa_seeds_dir = Path(gpbaa_seeds_override or _PSEUDO_SOURCE_DIRS["gpbaa_2d_highs"])
    an_seeds_dir = Path(an_seeds_override or _PSEUDO_SOURCE_DIRS["an_2d_highs"])
    matplotlib.rcParams["hatch.linewidth"] = 0.5
    _HATCH = "////"
    # A row is rendered only when its results directory actually exists, so a
    # run that produced just one exact method renders a one-row figure rather
    # than failing or leaving an empty half-canvas.
    _have_gpbaa = highs_dir is not None and Path(highs_dir).is_dir()
    _have_an = an_dir is not None and Path(an_dir).is_dir()
    # (full name for the row's y label, short name for the per-panel ratio label,
    #  exact-phase colour)
    _ALL_ROW_LABELS = [
        ("GPBA-A", "GPBA-A", "#e07b00"),
        ("Anytime Aneja & Nair", "A&N", "#d62728"),
    ]
    _ROW_LABELS = (
        ([_ALL_ROW_LABELS[0]] if _have_gpbaa else [])
        + ([_ALL_ROW_LABELS[1]] if _have_an else [])
    )
    if not _ROW_LABELS:
        print(f"Neither {highs_dir} nor {an_dir} exists", flush=True)
        return

    def _load_json(d: Path, inst: str) -> dict | None:
        p = d / f"{inst}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def _load_seeds(d: Path, inst: str) -> list[dict]:
        p = d / f"{inst}.json"
        if not p.exists():
            return []
        return json.loads(p.read_text()).get("solutions", [])

    pat = re.compile(filter_regex) if filter_regex else None
    _base = Path(highs_dir) if _have_gpbaa else Path(an_dir)
    instances = sorted(
        p.stem
        for p in _base.glob("*.json")
        if p.stem != "all_experiments"
        and not (_have_gpbaa and _have_an and not (Path(an_dir) / p.name).exists())
    )
    if pat:
        instances = [i for i in instances if pat.search(i)]
    if not instances:
        print(f"No instances found in {_base}", flush=True)
        return

    for inst in instances:
        highs = _load_json(highs_dir, inst) if _have_gpbaa else None
        an = _load_json(an_dir, inst) if _have_an else None
        if (_have_gpbaa and highs is None) or (_have_an and an is None):
            continue
        hb = highs.get("shared_bounds") if highs else None
        ab = an.get("shared_bounds") if an else None
        if not (hb or ab):
            print(f"  {inst}: missing shared_bounds, skipping", flush=True)
            continue
        # With one row there is nothing to merge; keep both rows on a common
        # reference when both are present so their HV bars stay comparable.
        merged = _merge_bounds(hb, ab) if (hb and ab) else (hb or ab)

        gpbaa_seeds = _load_seeds(gpbaa_seeds_dir, inst) if _have_gpbaa else []
        an_seeds = _load_seeds(an_seeds_dir, inst) if _have_an else []
        # A missing seed file reads as an empty exact-phase front, which plots
        # as a zero-height bar rather than an error -- indistinguishable on the
        # page from a genuine zero. Say so instead of drawing a wrong figure.
        for _who, _dir, _got in (("GPBA-A", gpbaa_seeds_dir, gpbaa_seeds if _have_gpbaa else None),
                                 ("A&N", an_seeds_dir, an_seeds if _have_an else None)):
            if _got is not None and not _got:
                print(f"  WARNING: {inst}: no {_who} seeds in {_dir} -- "
                      f"exact-phase values will be empty", flush=True)
        # Seed sets are generated with their own budget and can hold points
        # outside the run-derived bounds; compute_hypervolume rejects those.
        merged = _pf_widen_bounds(merged, [gpbaa_seeds, an_seeds])

        # Reference experiment for axis scaling; fall back to whichever exists.
        _ref, _ref_dir = (highs, highs_dir) if _have_gpbaa else (an, an_dir)
        rows = []
        if _have_gpbaa:
            rows.append(_ea_bar_data(highs, highs_dir, gpbaa_seeds,
                                     merged, _ref, _ref_dir, num_points))
        if _have_an:
            rows.append(_ea_bar_data(an, an_dir, an_seeds,
                                     merged, _ref, _ref_dir, num_points))

        # One y range over every panel in the figure. Comparing a hatched gain in
        # the MOEA/D column against the PLS column is the whole purpose here, and
        # per-panel autoscaling would silently defeat it.
        vals = [b["final_hv"] for row in rows for bars in row.values()
                for b in bars.values() if b["final_hv"] > 0]
        # The floor is 0, not `min(...) * 0.98`. These are STACKED bars: the
        # exact-phase base and the second-phase gain are meant to be read as
        # two parts of one total. A cropped baseline hides the base entirely
        # whenever the exact phase contributes little -- exactly the cases the
        # figure exists to show -- and misstates the ratio between the segments
        # in every other case.
        lo = 0.0
        hi = max(vals) * 1.02 if vals else 1.0

        # Height scales with the number of rows so a single-row figure is not
        # stretched to a two-row canvas.
        fig = plt.figure(
            figsize=(_PF_FIG_W, _PF_FIG_H * len(_ROW_LABELS) / 2.0),
            layout="constrained",
        )
        gs = GridSpec(len(_ROW_LABELS), 4, figure=fig)

        for r, (mlabel, mshort, mcol) in enumerate(_ROW_LABELS):
            for ci, (title, ecol, _hy, _cold) in enumerate(_EA_ENGINES):
                ax = fig.add_subplot(gs[r, ci])
                bars = rows[r][title]
                present = [(ratio, name) for ratio, name in _PF_RATIOS if ratio in bars]
                xs = list(range(len(present)))
                for xi, (ratio, _name) in zip(xs, present):
                    bd = bars[ratio]
                    if bd["hv_p1"] > 0:
                        ax.bar(xi, bd["hv_p1"], width=0.68, color=mcol,
                               linewidth=0, zorder=3)
                    if bd["hv_p2"] > 0:
                        ax.bar(xi, bd["hv_p2"], width=0.68, bottom=bd["hv_p1"],
                               facecolor=_pf_tint(ecol), hatch=_HATCH,
                               edgecolor=ecol, linewidth=_PF_SPINE_LW, zorder=3)
                    if bd["std"] > 0:
                        ax.errorbar(xi, bd["final_hv"], yerr=bd["std"], fmt="none",
                                    ecolor=_PF_AXIS_GREY, elinewidth=0.5,
                                    capsize=1.5, capthick=0.5, zorder=5)
                ax.set_xticks(xs)
                ax.set_xticklabels([name for _r, name in present],
                                   fontsize=_PF_TICK_PT, rotation=45,
                                   ha="right", rotation_mode="anchor")
                # Each panel names the actual pair it plots, so the label has to
                # be per-panel: the ratio splits the budget between *this* row's
                # exact method and *this* column's second-phase algorithm.
                ax.set_xlabel(f"{mshort} : {title}", fontsize=_PF_LABEL_PT)
                _pf_panel_title(ax, _PF_PANEL_LETTERS[r * 4 + ci], title)
                ax.set_ylim(lo, hi)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                _pf_style_axes(ax, grid_axis="y")
                ax.tick_params(axis="x", length=0)
                # The shared y scale is stated once per row; repeating the tick
                # labels four times across identical axes is noise.
                if ci == 0:
                    # With one row the exact method is already named by the
                    # legend and by every panel's x label, and the two-line
                    # form overflows the (halved) canvas height.
                    ax.set_ylabel(
                        "Final HV" if len(_ROW_LABELS) == 1
                        else f"{mlabel}\nFinal HV",
                        fontsize=_PF_LABEL_PT,
                    )
                else:
                    ax.tick_params(labelleft=False)

        from matplotlib.lines import Line2D  # noqa: F401  (kept for parity)

        handles = [
            mpatches.Patch(facecolor=_col, linewidth=0,
                           label=f"{_full} exact phase")
            for _full, _short, _col in _ROW_LABELS
        ] + [
            mpatches.Patch(facecolor=_pf_tint(col), hatch=_HATCH, edgecolor=col,
                           linewidth=_PF_SPINE_LW, label=f"{title} phase gain")
            for title, col, _h, _c in _EA_ENGINES
        ]
        leg = fig.legend(
            handles=handles, loc="outside lower center", ncol=3,
            fontsize=_PF_LEGEND_PT, frameon=False, handletextpad=0.5,
            columnspacing=1.6, labelspacing=0.35, handlelength=1.4,
        )
        for txt in leg.get_texts():
            txt.set_color("black")

        out = _pf_save(fig, output_dir, f"{inst}_ea_bars")
        plt.close(fig)
        print(f"  Saved: {out}", flush=True)


# ── Non-HV indicator variants of the EA-bars figure ─────────────────────────
#
# Same 2x4 layout and phase-decomposed hatched-bar mechanic as
# generate_ea_bar_figures, but the metric is front cardinality / spacing /
# IGD+ instead of hypervolume. These answer the "HV barely moved, but did the
# front actually get denser / better resolved?" question HV can't show on its
# own (see performance-indicators discussion).
#
# Scope difference from the HV version: HV's bars average 10 pre-stored runs
# (`final_hv`/`run_hvs` in the experiment JSON); no equivalent per-run raw
# front data was ever retained for these indicators (only run 0's trace
# survives per config -- see `_rescaled`'s docstring above), so these bars are
# single-run (run 0) values, no error bars. This is exactly the same scope
# the paper's own Pareto-front *scatter* figures already have ("Presented
# Pareto fronts were generated in a single (the first) run").

_INDICATOR_LABEL: dict[str, str] = {
    "cardinality": "Front Cardinality",
    "spacing": "Spacing",
    "igd_plus": "IGD+",
    "igd_plus_c": "1 - IGD+",
}

# Indicators reported as a single final value per bar, with no exact/heuristic
# phase split. IGD+ is not additively decomposable the way HV is -- phase 2
# moves the distance up or down rather than adding area, and it moves it the
# *wrong* way in over half of the hybrid runs (the seeded EAs discard the seed
# points their bounded populations cannot keep). A stacked bar would have to
# render those as negative segments, so these indicators plot the final value
# alone.
_FINAL_ONLY_INDICATORS: frozenset[str] = frozenset({"igd_plus_c"})

# Indicators where a larger value is better. `igd_plus_c` is 1 - IGD+, so it
# inverts IGD+'s direction and bars grow upward with quality.
_HIGHER_IS_BETTER: frozenset[str] = frozenset({"cardinality", "igd_plus_c"})


def _pf_trace_path_for_label(trace_dir: Path, instance: str, label: str) -> Path | None:
    """Trace filename for a config label, matching the derivation in
    `_recompute_result_with_bounds` (kept in lockstep with it deliberately --
    both must agree on how a config label maps to a trace filename)."""
    safe = (
        label.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace(":", "")
        .replace("/", "")
    )
    path = trace_dir / f"{instance}__{safe}.trace.tar.gz"
    return path if path.exists() else None


def _pf_indicator_value(
    front: list,
    bounds: list[list[int]],
    indicator: str,
    reference_set: list[list[float]] | None,
) -> float:
    """Dispatch to the right sims_problem indicator function for one front."""
    pts = [list(p) for p in front]
    if not pts:
        return 0.0
    if indicator == "cardinality":
        return float(sims_problem.front_cardinality(pts))
    if indicator == "spacing":
        return sims_problem.compute_spacing(pts, bounds, normalized=True)
    if indicator in ("igd_plus", "igd_plus_c"):
        ref = reference_set if reference_set else pts
        val = sims_problem.compute_igd(pts, ref, bounds, normalized=True, plus=True)
        # The complement is a display transform only: 1 - x is strictly
        # decreasing, so rankings and statistical tests are identical to IGD+.
        # Report plain IGD+ in tables; use this only for figures where "taller
        # is better" has to hold. Note 1 is not a true upper bound -- normalized
        # IGD+ in d dimensions can reach sqrt(d) -- so a catastrophically bad
        # front could go negative. Warn rather than plot a nonsense bar.
        if indicator == "igd_plus_c":
            if val > 1.0:
                print(f"  WARNING: IGD+ = {val:.4f} > 1, so 1 - IGD+ is "
                      f"negative; the complement is not meaningful here",
                      flush=True)
            return 1.0 - val
        return val
    raise ValueError(f"Unknown indicator: {indicator!r}")


def _pf_build_reference_set(
    instance: str,
    highs_dir: Path,
    an_dir: Path,
    gpbaa_seeds: list[dict],
    an_seeds: list[dict],
) -> list[list[float]]:
    """Union of every front ever discovered for this instance -- both exact
    methods' full seed sets, plus every engine/ratio's final trace front from
    both experiment directories -- filtered to non-dominated. Used as
    compute_igd's reference set: the best-known approximation to the true
    front, built from everything already computed for this instance rather
    than a separate generous-timeout solve.
    """
    points: set[tuple[float, float]] = set()
    for s in gpbaa_seeds + an_seeds:
        points.add((float(s["cost"]), float(s["cloudy_area"])))
    for trace_dir in (highs_dir, an_dir):
        for path in trace_dir.glob(f"{instance}__*.trace.tar.gz"):
            try:
                front = sims_problem.front_from_trace_at_time(path.read_bytes(), 1e12)
            except Exception:
                continue
            for p in front:
                points.add((float(p[0]), float(p[1])))
    return [list(p) for p in _pf_pareto_filter(sorted(points))]


def _pf_reference_bounds(reference_set: list[list[float]]) -> list[list[float]]:
    """Per-objective [min, max] (ideal/nadir) of the reference set itself.

    Used to normalise spacing/IGD+ instead of `shared_bounds` -- shared_bounds
    is built from *every* point ever recorded in every trace (including
    dominated/early-generation candidates from every EA config, see
    `compute_shared_bounds`), which is the right choice for HV's reference
    point but is typically 1.5-3x wider than the region the actual
    Pareto-optimal front occupies. Normalising a density/convergence metric by
    that inflated, algorithm-exploration-dependent range understates the
    values and makes them less comparable across instances. The reference
    set's own ideal/nadir is the standard choice for these indicators (matches
    pymoo/moocore convention) and is already computed for IGD+ elsewhere.
    """
    ndim = len(reference_set[0])
    return [
        [min(p[i] for p in reference_set), max(p[i] for p in reference_set)]
        for i in range(ndim)
    ]


def _ea_bar_data_indicator(
    indicator: str,
    result: dict,
    trace_dir: Path,
    seeds: list[dict],
    merged_bounds: list[list[int]],
    reference_set: list[list[float]] | None,
) -> dict[str, dict[float, dict]]:
    """Phase-decomposed indicator bars per engine per ratio.

    Returns {engine title: {ratio: {p1, p2, final}}}, mirroring
    `_ea_bar_data`'s {hv_p1, hv_p2, final_hv} shape. Unlike HV, `p2` is NOT
    floored at 0: for spacing/IGD+ (lower is better), a second phase that
    improves the front makes `p2` negative by construction (final < p1), and
    that is real signal, not noise -- flooring it would silently hide
    improvements. The renderer draws negative `p2` extending *below* `p1`,
    so the bar's total height (`p1 + p2`) still always equals `final`.
    """
    timeout = float(result.get("timeout_s", 0.0))
    instance = result["instance"]

    out: dict[str, dict[float, dict]] = {}
    for title, _colour, hybrid_labels, cold_label in _EA_ENGINES:
        bars: dict[float, dict] = {}
        for ratio, _name in _PF_RATIOS:
            _final_only = indicator in _FINAL_ONLY_INDICATORS
            if ratio == 1.00:
                front = _pf_exact_front(seeds, timeout)
                val = _pf_indicator_value(front, merged_bounds, indicator, reference_set)
                bars[ratio] = dict(p1=val, p2=0.0, final=val)
            elif ratio == 0.00:
                path = _pf_trace_path_for_label(trace_dir, instance, cold_label)
                if path is None:
                    continue
                front = sims_problem.front_from_trace_at_time(path.read_bytes(), timeout)
                val = _pf_indicator_value(front, merged_bounds, indicator, reference_set)
                bars[ratio] = dict(p1=0.0, p2=val, final=val)
            else:
                path = _pf_trace_path_for_label(trace_dir, instance, hybrid_labels[ratio])
                if path is None:
                    continue
                final_front = sims_problem.front_from_trace_at_time(
                    path.read_bytes(), timeout
                )
                final = _pf_indicator_value(
                    final_front, merged_bounds, indicator, reference_set
                )
                if _final_only:
                    # One bar spanning [0, final]; the renderer draws the
                    # delta segment when p1 is 0.
                    bars[ratio] = dict(p1=0.0, p2=final, final=final)
                else:
                    p1_front = _pf_exact_front(seeds, ratio * timeout)
                    p1 = _pf_indicator_value(
                        p1_front, merged_bounds, indicator, reference_set
                    )
                    bars[ratio] = dict(p1=p1, p2=final - p1, final=final)
        out[title] = bars
    return out


def generate_ea_bar_figures_indicator(
    indicator: str,
    highs_dir: Path,
    an_dir: Path,
    output_dir: Path,
    filter_regex: str | None = None,
    gpbaa_seeds_override: Path | None = None,
    an_seeds_override: Path | None = None,
) -> None:
    """Per-instance `{instance}_ea_bars.png` using `indicator` (one of
    "cardinality", "spacing", "igd_plus") instead of hypervolume. Same 2x4
    layout, same phase-decomposed hatched-bar mechanic as
    generate_ea_bar_figures -- see the module comment above this section for
    what differs (single-run bars, unclipped `p2`).
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator

    if indicator not in _INDICATOR_LABEL:
        raise ValueError(
            f"Unknown indicator {indicator!r}, expected one of {sorted(_INDICATOR_LABEL)}"
        )

    gpbaa_seeds_dir = Path(gpbaa_seeds_override or _PSEUDO_SOURCE_DIRS["gpbaa_2d_highs"])
    an_seeds_dir = Path(an_seeds_override or _PSEUDO_SOURCE_DIRS["an_2d_highs"])
    # Render a row only when its results directory exists, so a grid that ran
    # one exact method produces a one-row figure instead of failing.
    _have_gpbaa = highs_dir is not None and Path(highs_dir).is_dir()
    _have_an = an_dir is not None and Path(an_dir).is_dir()
    matplotlib.rcParams["hatch.linewidth"] = 0.5
    _HATCH = "////"
    _ALL_ROW_LABELS = [
        ("GPBA-A", "GPBA-A", "#e07b00"),
        ("Anytime Aneja & Nair", "A&N", "#d62728"),
    ]
    _ROW_LABELS = (
        ([_ALL_ROW_LABELS[0]] if _have_gpbaa else [])
        + ([_ALL_ROW_LABELS[1]] if _have_an else [])
    )
    if not _ROW_LABELS:
        print(f"Neither {highs_dir} nor {an_dir} exists", flush=True)
        return

    def _load_json(d: Path, inst: str) -> dict | None:
        p = d / f"{inst}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def _load_seeds(d: Path, inst: str) -> list[dict]:
        p = d / f"{inst}.json"
        if not p.exists():
            return []
        return json.loads(p.read_text()).get("solutions", [])

    pat = re.compile(filter_regex) if filter_regex else None
    _base = Path(highs_dir) if _have_gpbaa else Path(an_dir)
    instances = sorted(
        p.stem
        for p in _base.glob("*.json")
        if p.stem != "all_experiments"
        and not (_have_gpbaa and _have_an and not (Path(an_dir) / p.name).exists())
    )
    if pat:
        instances = [i for i in instances if pat.search(i)]
    if not instances:
        print(f"No instances found in {_base}", flush=True)
        return

    ylabel = _INDICATOR_LABEL[indicator]

    for inst in instances:
        highs = _load_json(highs_dir, inst) if _have_gpbaa else None
        an = _load_json(an_dir, inst) if _have_an else None
        if (_have_gpbaa and highs is None) or (_have_an and an is None):
            continue
        hb = highs.get("shared_bounds") if highs else None
        ab = an.get("shared_bounds") if an else None
        if not (hb or ab):
            print(f"  {inst}: missing shared_bounds, skipping", flush=True)
            continue
        merged = _merge_bounds(hb, ab) if (hb and ab) else (hb or ab)

        gpbaa_seeds = _load_seeds(gpbaa_seeds_dir, inst) if _have_gpbaa else []
        an_seeds = _load_seeds(an_seeds_dir, inst) if _have_an else []
        # A missing seed file reads as an empty exact-phase front, which plots
        # as a zero-height bar rather than an error -- indistinguishable on the
        # page from a genuine zero. Say so instead of drawing a wrong figure.
        for _who, _dir, _got in (("GPBA-A", gpbaa_seeds_dir, gpbaa_seeds if _have_gpbaa else None),
                                 ("A&N", an_seeds_dir, an_seeds if _have_an else None)):
            if _got is not None and not _got:
                print(f"  WARNING: {inst}: no {_who} seeds in {_dir} -- "
                      f"exact-phase values will be empty", flush=True)

        reference_set = (
            _pf_build_reference_set(inst, highs_dir, an_dir, gpbaa_seeds, an_seeds)
            if indicator != "cardinality"
            else None
        )
        indicator_bounds = _pf_reference_bounds(reference_set) if reference_set else merged

        rows = []
        if _have_gpbaa:
            rows.append(_ea_bar_data_indicator(
                indicator, highs, highs_dir, gpbaa_seeds, indicator_bounds, reference_set))
        if _have_an:
            rows.append(_ea_bar_data_indicator(
                indicator, an, an_dir, an_seeds, indicator_bounds, reference_set))

        all_vals = [
            v
            for row in rows
            for bars in row.values()
            for bd in bars.values()
            for v in (bd["p1"], bd["final"])
        ]
        # A phase-decomposed bar must sit on zero -- its segments have to sum
        # to the total. A final-only bar carries no segments, and these values
        # occupy a narrow band near 1, so a zero baseline renders every method
        # as the same full-height bar. Crop to the data instead. This IS an
        # axis truncation: state it in the caption.
        # Range over the FINAL values only. `all_vals` also carries the p1
        # entries, which final-only mode pins to 0, so using it would put the
        # floor back at zero and undo the crop.
        _finals = [
            bd["final"]
            for row in rows for bars in row.values() for bd in bars.values()
        ]
        if indicator in _FINAL_ONLY_INDICATORS and _finals:
            _span = max(_finals) - min(_finals)
            _pad = max(_span * 0.15, max(_finals) * 0.005)
            lo, hi = min(_finals) - _pad, max(_finals) + _pad
        else:
            lo = min(0.0, min(all_vals) * 1.02) if all_vals else 0.0
            hi = max(all_vals) * 1.02 if all_vals else 1.0

        fig = plt.figure(figsize=(_PF_FIG_W, _PF_FIG_H), layout="constrained")
        gs = GridSpec(len(_ROW_LABELS), 4, figure=fig)

        # The indicator values behind the bars. A zero-height bar is
        # ambiguous on the page (absent config vs. genuine zero), and for
        # spacing/IGD+ a genuine zero is a strong claim worth checking.
        for _lbl, _rd in zip([r[1] for r in _ROW_LABELS], rows):
            for _eng, _bars in _rd.items():
                _s = ", ".join(
                    f"{_n}:{_bars[_r]['final']:.4f}"
                    for _r, _n in _PF_RATIOS if _r in _bars
                )
                print(f"  {inst} [{_lbl}/{_eng}] {indicator} -> {_s}", flush=True)

        for r, (mlabel, mshort, mcol) in enumerate(_ROW_LABELS):
            for ci, (title, ecol, _hy, _cold) in enumerate(_EA_ENGINES):
                ax = fig.add_subplot(gs[r, ci])
                bars = rows[r][title]
                present = [(ratio, name) for ratio, name in _PF_RATIOS if ratio in bars]
                xs = list(range(len(present)))
                for xi, (ratio, _name) in zip(xs, present):
                    bd = bars[ratio]
                    p1, final = bd["p1"], bd["final"]
                    # The solid bar always spans [0, p1] so the exact-phase
                    # value reads identically across every engine column
                    # (it IS identical — p1 doesn't depend on the phase-2
                    # engine). The phase-2 delta is drawn as a second
                    # rectangle spanning [min(p1, final), max(p1, final)].
                    # When final >= p1 (HV-like, monotone improvement) it
                    # sits above the solid bar and is filled solid, matching
                    # the original stacked-bar look. When final < p1
                    # (lower-is-better indicators improving in phase 2) it
                    # would otherwise overlap and mask the solid bar, so it
                    # is drawn with a transparent fill — only the hatch
                    # pattern and edge — letting the solid p1 color show
                    # through underneath.
                    if p1 != 0:
                        ax.bar(
                            xi, p1, width=0.68, color=mcol,
                            linewidth=0, zorder=3,
                        )
                    delta = final - p1
                    if delta != 0:
                        improving = delta < 0
                        ax.bar(
                            xi, delta, width=0.68, bottom=p1,
                            facecolor="none" if improving else _pf_tint(ecol),
                            hatch=_HATCH, edgecolor=ecol,
                            linewidth=_PF_SPINE_LW, zorder=4,
                        )
                ax.set_xticks(xs)
                ax.set_xticklabels(
                    [name for _r, name in present], fontsize=_PF_TICK_PT,
                    rotation=45, ha="right", rotation_mode="anchor",
                )
                ax.set_xlabel(f"{mshort} : {title}", fontsize=_PF_LABEL_PT)
                _pf_panel_title(ax, _PF_PANEL_LETTERS[r * 4 + ci], title)
                ax.set_ylim(lo, hi)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                _pf_style_axes(ax, grid_axis="y")
                ax.tick_params(axis="x", length=0)
                if ci == 0:
                    # One row: the method is already named by the legend and
                    # by every panel's x label, and the two-line form overflows
                    # the halved canvas height.
                    ax.set_ylabel(
                        f"Final {ylabel}" if len(_ROW_LABELS) == 1
                        else f"{mlabel}\nFinal {ylabel}",
                        fontsize=_PF_LABEL_PT,
                    )
                else:
                    ax.tick_params(labelleft=False)

        _fo = indicator in _FINAL_ONLY_INDICATORS
        handles = [
            mpatches.Patch(
                facecolor=_col, linewidth=0,
                label=f"{_full} ({'100:0' if _fo else 'exact phase'})",
            )
            for _full, _short, _col in _ROW_LABELS
        ] + [
            mpatches.Patch(facecolor=_pf_tint(col), hatch=_HATCH, edgecolor=col,
                           linewidth=_PF_SPINE_LW,
                           label=title if _fo else f"{title} phase Δ")
            for title, col, _h, _c in _EA_ENGINES
        ]
        leg = fig.legend(
            handles=handles, loc="outside lower center", ncol=3,
            fontsize=_PF_LEGEND_PT, frameon=False, handletextpad=0.5,
            columnspacing=1.6, labelspacing=0.35, handlelength=1.4,
        )
        for txt in leg.get_texts():
            txt.set_color("black")

        out = _pf_save(fig, output_dir, f"{inst}_ea_bars")
        plt.close(fig)
        print(f"  Saved: {out}", flush=True)


# ── Lollipop variant: direct engine-vs-engine comparison ────────────────────
#
# The bar figure above only ever compares phase 1 vs phase 2 *within* one
# engine's own column -- it never puts two engines' final values in the same
# axes, so "is PLS better than NSGA-II" can't be read off it directly. Here
# the panel axis is the ratio (not the engine), and each panel plots all four
# engines' `final` indicator value as a lollipop (stem + marker, no fill
# area) so heights are directly comparable and the chart doesn't imply
# "taller = better" for lower-is-better indicators (spacing, IGD+).

_PF_ENGINE_MARKERS: dict[str, str] = {
    "PLS": "o",
    "NSGA-II": "s",
    "NSGA-III": "^",
    "MOEA/D": "D",
}


def _pf_lollipop_ratios(rows: list[dict], _row_idx: int) -> list[float]:
    """Ratios (excluding the pure-exact 1.00 baseline) with data for at least
    one engine, in `_PF_RATIOS` order."""
    row = rows[_row_idx]
    present = {ratio for bars in row.values() for ratio in bars if ratio != 1.00}
    return [ratio for ratio, _name in _PF_RATIOS if ratio in present]


def generate_ea_lollipop_figures_indicator(
    indicator: str,
    highs_dir: Path,
    an_dir: Path,
    output_dir: Path,
    filter_regex: str | None = None,
    gpbaa_seeds_override: Path | None = None,
    an_seeds_override: Path | None = None,
) -> None:
    """Per-instance `{instance}_ea_lollipop.png`: one panel per ratio (not per
    engine), each showing all four engines' final `indicator` value side by
    side as lollipops, plus a dashed reference line at the exact-only (100:0)
    value. Direct visual comparison across engines, unlike
    generate_ea_bar_figures_indicator's phase-decomposed bars.
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    if indicator not in _INDICATOR_LABEL:
        raise ValueError(
            f"Unknown indicator {indicator!r}, expected one of {sorted(_INDICATOR_LABEL)}"
        )

    gpbaa_seeds_dir = Path(gpbaa_seeds_override or _PSEUDO_SOURCE_DIRS["gpbaa_2d_highs"])
    an_seeds_dir = Path(an_seeds_override or _PSEUDO_SOURCE_DIRS["an_2d_highs"])
    # Render a row only when its results directory exists, so a grid that ran
    # one exact method produces a one-row figure instead of failing.
    _have_gpbaa = highs_dir is not None and Path(highs_dir).is_dir()
    _have_an = an_dir is not None and Path(an_dir).is_dir()
    _ALL_ROW_LABELS = [
        ("GPBA-A", "GPBA-A", "#e07b00"),
        ("Anytime Aneja & Nair", "A&N", "#d62728"),
    ]
    _ROW_LABELS = (
        ([_ALL_ROW_LABELS[0]] if _have_gpbaa else [])
        + ([_ALL_ROW_LABELS[1]] if _have_an else [])
    )
    if not _ROW_LABELS:
        print(f"Neither {highs_dir} nor {an_dir} exists", flush=True)
        return
    higher_is_better = indicator in _HIGHER_IS_BETTER
    arrow = "↑" if higher_is_better else "↓"

    def _load_json(d: Path, inst: str) -> dict | None:
        p = d / f"{inst}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def _load_seeds(d: Path, inst: str) -> list[dict]:
        p = d / f"{inst}.json"
        if not p.exists():
            return []
        return json.loads(p.read_text()).get("solutions", [])

    pat = re.compile(filter_regex) if filter_regex else None
    _base = Path(highs_dir) if _have_gpbaa else Path(an_dir)
    instances = sorted(
        p.stem
        for p in _base.glob("*.json")
        if p.stem != "all_experiments"
        and not (_have_gpbaa and _have_an and not (Path(an_dir) / p.name).exists())
    )
    if pat:
        instances = [i for i in instances if pat.search(i)]
    if not instances:
        print(f"No instances found in {_base}", flush=True)
        return

    ylabel = _INDICATOR_LABEL[indicator]

    for inst in instances:
        highs = _load_json(highs_dir, inst) if _have_gpbaa else None
        an = _load_json(an_dir, inst) if _have_an else None
        if (_have_gpbaa and highs is None) or (_have_an and an is None):
            continue
        hb = highs.get("shared_bounds") if highs else None
        ab = an.get("shared_bounds") if an else None
        if not (hb or ab):
            print(f"  {inst}: missing shared_bounds, skipping", flush=True)
            continue
        merged = _merge_bounds(hb, ab) if (hb and ab) else (hb or ab)

        gpbaa_seeds = _load_seeds(gpbaa_seeds_dir, inst) if _have_gpbaa else []
        an_seeds = _load_seeds(an_seeds_dir, inst) if _have_an else []
        # A missing seed file reads as an empty exact-phase front, which plots
        # as a zero-height bar rather than an error -- indistinguishable on the
        # page from a genuine zero. Say so instead of drawing a wrong figure.
        for _who, _dir, _got in (("GPBA-A", gpbaa_seeds_dir, gpbaa_seeds if _have_gpbaa else None),
                                 ("A&N", an_seeds_dir, an_seeds if _have_an else None)):
            if _got is not None and not _got:
                print(f"  WARNING: {inst}: no {_who} seeds in {_dir} -- "
                      f"exact-phase values will be empty", flush=True)

        reference_set = (
            _pf_build_reference_set(inst, highs_dir, an_dir, gpbaa_seeds, an_seeds)
            if indicator != "cardinality"
            else None
        )
        indicator_bounds = _pf_reference_bounds(reference_set) if reference_set else merged

        rows = []
        if _have_gpbaa:
            rows.append(_ea_bar_data_indicator(
                indicator, highs, highs_dir, gpbaa_seeds, indicator_bounds, reference_set))
        if _have_an:
            rows.append(_ea_bar_data_indicator(
                indicator, an, an_dir, an_seeds, indicator_bounds, reference_set))

        all_vals = [
            bd["final"]
            for row in rows
            for bars in row.values()
            for ratio, bd in bars.items()
        ]
        lo = min(0.0, min(all_vals) * 1.02) if all_vals else 0.0
        hi = max(all_vals) * 1.05 if all_vals else 1.0
        # Markers sitting at/near lo=0 get clipped in half by the axes frame
        # (matplotlib clips at the exact ylim); pad the *displayed* floor
        # below the data floor so nothing touches the frame. Stems are still
        # drawn from the true y=0 baseline below -- this margin is purely
        # visual headroom, not a change to what's plotted.
        lo_display = lo - 0.03 * (hi - lo)

        ratio_cols = [_pf_lollipop_ratios(rows, i) for i in range(len(rows))]
        ncols = max((len(rc) for rc in ratio_cols), default=0)
        if ncols == 0:
            continue

        _hscale = len(_ROW_LABELS) / 2.0 if len(_ROW_LABELS) >= 2 else 0.62
        fig = plt.figure(
            figsize=(_PF_FIG_W, _PF_FIG_H * _hscale), layout="constrained"
        )
        gs = GridSpec(len(_ROW_LABELS), ncols, figure=fig)

        for r, (mlabel, mshort, mcol) in enumerate(_ROW_LABELS):
            baseline = None
            for bars in rows[r].values():
                if 1.00 in bars:
                    baseline = bars[1.00]["final"]
                    break
            for ci, ratio in enumerate(ratio_cols[r]):
                ax = fig.add_subplot(gs[r, ci])
                if baseline is not None:
                    ax.axhline(
                        baseline, color=mcol, linestyle="--", linewidth=_PF_SPINE_LW,
                        zorder=2,
                    )
                xs, labels = [], []
                for xi, (title, ecol, _hy, _cold) in enumerate(_EA_ENGINES):
                    bd = rows[r][title].get(ratio)
                    if bd is None:
                        continue
                    xs.append(xi)
                    labels.append(title)
                    val = bd["final"]
                    is_pls = title == "PLS"
                    ax.vlines(
                        xi, 0, val, color=ecol,
                        linewidth=1.1 if is_pls else 0.7, zorder=3,
                        clip_on=False,
                    )
                    ax.plot(
                        xi, val, marker=_PF_ENGINE_MARKERS[title],
                        markersize=5.5 if is_pls else 4.0,
                        markerfacecolor=ecol, markeredgecolor=ecol,
                        markeredgewidth=0.8 if is_pls else 0.4,
                        zorder=4, clip_on=False,
                    )
                ax.set_xticks(range(len(_EA_ENGINES)))
                ax.set_xticklabels(
                    [t for t, *_ in _EA_ENGINES], fontsize=_PF_TICK_PT,
                    rotation=45, ha="right", rotation_mode="anchor",
                )
                ax.set_xlim(-0.6, len(_EA_ENGINES) - 0.4)
                _, ratio_name = next(p for p in _PF_RATIOS if p[0] == ratio)
                ax.set_xlabel(f"{mshort} : {ratio_name}", fontsize=_PF_LABEL_PT)
                _pf_panel_title(ax, _PF_PANEL_LETTERS[r * ncols + ci], ratio_name)
                ax.set_ylim(lo_display, hi)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                _pf_style_axes(ax, grid_axis="y")
                ax.tick_params(axis="x", length=0)
                if ci == 0:
                    # One row: the method is already named by the legend and
                    # by every panel's x label, and the two-line form overflows
                    # the halved canvas height.
                    ax.set_ylabel(
                        f"Final {ylabel}" if len(_ROW_LABELS) == 1
                        else f"{mlabel}\nFinal {ylabel}",
                        fontsize=_PF_LABEL_PT,
                    )
                else:
                    ax.tick_params(labelleft=False)

        fig.suptitle(
            f"{arrow} {'higher' if higher_is_better else 'lower'} is better",
            fontsize=_PF_LEGEND_PT,
        )

        handles = [
            Line2D(
                [0], [0], marker=_PF_ENGINE_MARKERS[title], color=col,
                markerfacecolor=col, markeredgecolor=col,
                markeredgewidth=0.8 if title == "PLS" else 0.4,
                linewidth=1.1 if title == "PLS" else 0.7, label=title,
            )
            for title, col, _h, _c in _EA_ENGINES
        ] + [
            Line2D(
                [0], [0], color=_col, linestyle="--",
                linewidth=_PF_SPINE_LW, label=f"{_short} (100:0) reference",
            )
            for _full, _short, _col in _ROW_LABELS
        ]
        leg = fig.legend(
            handles=handles, loc="outside lower center", ncol=3,
            fontsize=_PF_LEGEND_PT, frameon=False, handletextpad=0.5,
            columnspacing=1.6, labelspacing=0.35, handlelength=1.4,
        )
        for txt in leg.get_texts():
            txt.set_color("black")

        out = _pf_save(fig, output_dir, f"{inst}_ea_lollipop")
        plt.close(fig)
        print(f"  Saved: {out}", flush=True)


def compute_ea_rank_summary(
    indicator: str,
    highs_dir: Path,
    an_dir: Path,
    filter_regex: str | None = None,
) -> "pd.DataFrame":
    """Cross-instance proof table: for every (exact method, ratio), rank the
    four engines' final `indicator` value (1 = best) on each instance and
    aggregate mean rank + win count over all instances. Ranking (rather than
    averaging raw values) sidesteps the fact that spacing/IGD+/cardinality
    scales differ per instance, so raw cross-instance averages would be
    dominated by whichever instances happen to have larger scales.
    """
    import pandas as pd

    if indicator not in _INDICATOR_LABEL:
        raise ValueError(
            f"Unknown indicator {indicator!r}, expected one of {sorted(_INDICATOR_LABEL)}"
        )
    higher_is_better = indicator in _HIGHER_IS_BETTER

    gpbaa_seeds_dir = _PSEUDO_SOURCE_DIRS["gpbaa_2d_highs"]
    an_seeds_dir = _PSEUDO_SOURCE_DIRS["an_2d_highs"]
    _ROW_LABELS = [("GPBA-A", "GPBA-A"), ("Anytime Aneja & Nair", "A&N")]

    def _load_json(d: Path, inst: str) -> dict | None:
        p = d / f"{inst}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def _load_seeds(d: Path, inst: str) -> list[dict]:
        p = d / f"{inst}.json"
        if not p.exists():
            return []
        return json.loads(p.read_text()).get("solutions", [])

    pat = re.compile(filter_regex) if filter_regex else None
    instances = sorted(
        p.stem
        for p in highs_dir.glob("*.json")
        if p.stem != "all_experiments" and (an_dir / p.name).exists()
    )
    if pat:
        instances = [i for i in instances if pat.search(i)]

    records = []
    for inst in instances:
        highs = _load_json(highs_dir, inst)
        an = _load_json(an_dir, inst)
        if highs is None or an is None:
            continue
        hb, ab = highs.get("shared_bounds"), an.get("shared_bounds")
        if not hb or not ab:
            continue
        merged = _merge_bounds(hb, ab)

        gpbaa_seeds = _load_seeds(gpbaa_seeds_dir, inst)
        an_seeds = _load_seeds(an_seeds_dir, inst)
        reference_set = (
            _pf_build_reference_set(inst, highs_dir, an_dir, gpbaa_seeds, an_seeds)
            if indicator != "cardinality"
            else None
        )
        indicator_bounds = _pf_reference_bounds(reference_set) if reference_set else merged

        rows = [
            _ea_bar_data_indicator(
                indicator, highs, highs_dir, gpbaa_seeds, indicator_bounds, reference_set
            ),
            _ea_bar_data_indicator(
                indicator, an, an_dir, an_seeds, indicator_bounds, reference_set
            ),
        ]

        for r, (mlabel, mshort) in enumerate(_ROW_LABELS):
            for ratio in _pf_lollipop_ratios(rows, r):
                vals = {
                    title: rows[r][title][ratio]["final"]
                    for title, *_ in _EA_ENGINES
                    if ratio in rows[r][title]
                }
                if len(vals) < 2:
                    continue
                ordered = sorted(
                    vals, key=lambda t: vals[t], reverse=higher_is_better
                )
                ranks = {t: ordered.index(t) + 1 for t in vals}
                for title, rank in ranks.items():
                    records.append(
                        dict(
                            instance=inst, exact_method=mshort, ratio=ratio,
                            engine=title, value=vals[title], rank=rank,
                            is_best=rank == 1,
                        )
                    )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    summary = (
        df.groupby(["exact_method", "ratio", "engine"])
        .agg(mean_rank=("rank", "mean"), wins=("is_best", "sum"), n=("rank", "size"))
        .reset_index()
        .sort_values(["exact_method", "ratio", "mean_rank"])
    )
    return summary


def generate_pareto_front_figures(
    highs_dir: Path,
    an_dir: Path,
    output_dir: Path,
    filter_regex: str | None = None,
    num_points: int = 30,
    gpbaa_seeds_override: Path | None = None,
    an_seeds_override: Path | None = None,
) -> None:
    """Reconstruct the paper's per-instance `{instance}_pareto_fronts.png` figures
    from trace artifacts, with phase-separated + hatched HV bars.

    Row 0 = GPBA-A (highs_dir + gpbaa_2d_highs pseudo seeds),
    Row 1 = Aneja & Nair (an_dir + an_2d_highs pseudo seeds).
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator

    # Seed sources default to the publication *_highs datasets, but any instance
    # set generated later (e.g. the A&N-calibrated "hard" grid) keeps its
    # pseudo-solutions elsewhere, so allow an explicit override. Without this the
    # exact-phase portion of every bar silently comes out empty.
    gpbaa_seeds_dir = gpbaa_seeds_override or _PSEUDO_SOURCE_DIRS["gpbaa_2d_highs"]
    an_seeds_dir = an_seeds_override or _PSEUDO_SOURCE_DIRS["an_2d_highs"]

    # A row is rendered only if its results directory was supplied. Passing just
    # one of --highs-dir/--an-dir produces the corresponding single-row figure,
    # which is what you want before the second exact method has been run.
    # --highs-dir/--an-dir both carry defaults, so "not supplied" cannot be
    # detected by None. Treat a directory that does not exist as absent, which
    # is what happens when only one exact method has been run.
    _have_gpbaa = highs_dir is not None and Path(highs_dir).is_dir()
    _have_an = an_dir is not None and Path(an_dir).is_dir()
    if not (_have_gpbaa or _have_an):
        print("neither --highs-dir nor --an-dir given", flush=True)
        return

    def _load_json(d: Path, inst: str) -> dict | None:
        if d is None:
            return None
        p = d / f"{inst}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def _load_seeds(d: Path, inst: str) -> list[dict]:
        p = d / f"{inst}.json"
        if not p.exists():
            return []
        return json.loads(p.read_text()).get("solutions", [])

    pat = re.compile(filter_regex) if filter_regex else None

    _base = highs_dir if _have_gpbaa else an_dir
    instances = sorted(
        p.stem
        for p in Path(_base).glob("*.json")
        if p.stem != "all_experiments"
        and (not (_have_gpbaa and _have_an) or (Path(an_dir) / p.name).exists())
    )
    if pat:
        instances = [i for i in instances if pat.search(i)]
    if not instances:
        print(f"No instances found in {_base}", flush=True)
        return

    # Unified colour scheme — colour encodes the *algorithm*, identically in the
    # bars and the scatter panels:
    #   GPBA-A = amber, Aneja & Nair = red, Pareto Local Search = blue.
    # In panel a) the exact-phase base takes the method colour (matching that
    # method's scatter marker) and the PLS-phase gain is drawn in the PLS blue
    # with a hatch (matching the PLS scatter marker); texture marks the phase.
    _PLS_COLOR = "#1f77b4"
    # Finer, thinner hatching: "///" at 0.5 pt lines aliases badly once the
    # figure is rasterised at page size.
    _PLS_HATCH = "////"
    matplotlib.rcParams["hatch.linewidth"] = 0.5
    # Row style: (full method label, short name, exact/method colour, exact marker)
    _ALL_ROWS = [
        ("GPBA-A", "GPBA-A", "#e07b00", "X"),
        ("Anytime Aneja & Nair", "Aneja & Nair", "#d62728", "X"),
    ]
    _ROWS = ([_ALL_ROWS[0]] if _have_gpbaa else []) + ([_ALL_ROWS[1]] if _have_an else [])

    for inst in instances:
        highs = _load_json(highs_dir, inst)
        an = _load_json(an_dir, inst)
        if (_have_gpbaa and highs is None) or (_have_an and an is None):
            continue
        hb = highs.get("shared_bounds") if highs else None
        ab = an.get("shared_bounds") if an else None
        if not (hb or ab):
            print(f"  {inst}: missing shared_bounds, skipping", flush=True)
            continue
        # With one row there is nothing to merge; keep both rows on a common
        # reference when both are present so their HV bars stay comparable.
        merged = _merge_bounds(hb, ab) if (hb and ab) else (hb or ab)

        gpbaa_seeds = _load_seeds(gpbaa_seeds_dir, inst) if _have_gpbaa else []
        an_seeds = _load_seeds(an_seeds_dir, inst) if _have_an else []
        # A missing seed file reads as an empty exact-phase front, which plots
        # as a zero-height bar rather than an error -- indistinguishable on the
        # page from a genuine zero. Say so instead of drawing a wrong figure.
        for _who, _dir, _got in (("GPBA-A", gpbaa_seeds_dir, gpbaa_seeds if _have_gpbaa else None),
                                 ("A&N", an_seeds_dir, an_seeds if _have_an else None)):
            if _got is not None and not _got:
                print(f"  WARNING: {inst}: no {_who} seeds in {_dir} -- "
                      f"exact-phase values will be empty", flush=True)
        merged = _pf_widen_bounds(merged, [gpbaa_seeds, an_seeds])

        # `_pf_row_data`'s last two arguments are the reference experiment used
        # for axis scaling; fall back to whichever row exists.
        _ref, _ref_dir = (highs, highs_dir) if _have_gpbaa else (an, an_dir)
        row_data = []
        if _have_gpbaa:
            row_data.append(
                _pf_row_data(highs, highs_dir, gpbaa_seeds, merged, _ref, _ref_dir, num_points))
        if _have_an:
            row_data.append(
                _pf_row_data(an, an_dir, an_seeds, merged, _ref, _ref_dir, num_points))

        # Exact-phase share per ratio — the number the "why heuristics at all?"
        # question turns on, and not otherwise readable off the zoomed bars.
        for _lbl, rd in zip([r[1] for r in _ROWS], row_data):
            _sh = ", ".join(
                f"{_n}:{100.0 * rd['bars'][_r]['hv_p1'] / rd['bars'][_r]['final_hv']:.0f}%"
                for _r, _n in _PF_RATIOS
                if _r in rd["bars"] and rd["bars"][_r]["final_hv"] > 0
            )
            print(f"  {inst} [{_lbl}] exact-phase share of final HV -> {_sh}",
                  flush=True)

        # Shared scatter-axis normalisation. Divide by the max over the actual
        # plotted points (not the wide HV upper bound), so fronts fill the panel
        # with a ~[0.5, 1.0] spread on every instance size — matching the paper.
        _all_sx = [
            x
            for rd in row_data
            for sc in rd["scatter"].values()
            for x, _y in (sc.get("exact", []) + sc.get("pls", []))
        ]
        _all_sy = [
            y
            for rd in row_data
            for sc in rd["scatter"].values()
            for _x, y in (sc.get("exact", []) + sc.get("pls", []))
        ]
        ux = max(_all_sx) if _all_sx else 1.0
        uy = max(_all_sy) if _all_sy else 1.0

        # Shared front-panel axis limits (normalised units) so every scatter — all
        # ratios, both rows — is on one identical scale and directly comparable.
        _nx_lo = (min(_all_sx) / ux) if _all_sx else 0.0
        _ny_lo = (min(_all_sy) / uy) if _all_sy else 0.0
        _mx = (1.0 - _nx_lo) * 0.06 + 1e-6
        _my = (1.0 - _ny_lo) * 0.06 + 1e-6
        sc_xlim = (_nx_lo - _mx, 1.0 + _mx)
        sc_ylim = (_ny_lo - _my, 1.0 + _my)

        # Shared panel-a) y-range across both rows so the bar charts are directly
        # comparable top-to-bottom.
        _all_hv = [
            bd["final_hv"] for rd in row_data for bd in rd["bars"].values()
            if bd["final_hv"] > 0
        ]
        # Stacked bars must sit on zero -- see the note in
        # generate_ea_bar_figures. A cropped baseline swallowed the exact-phase
        # base whenever it was the smallest value in the figure, which is the
        # common case: the exact phase contributes ~10% of the final HV once
        # it has returned only its first extreme point.
        bar_lo = 0.0
        bar_hi = max(_all_hv) * 1.02 if _all_hv else 1.0

        # A single row does not get exactly half the height: the panel titles
        # and the rotated ratio labels cost the same absolute space at any row
        # count, so a literal halving squeezes the axes until panel a)'s title
        # runs into panel b). 0.62 leaves that furniture room.
        _hscale = len(_ROWS) / 2.0 if len(_ROWS) >= 2 else 0.62
        fig = plt.figure(
            figsize=(_PF_FIG_W, _PF_FIG_H * _hscale), layout="constrained"
        )
        # `constrained_layout` (set on the figure) sizes the gutters from the
        # actual text extents, which is what keeps 7-8 pt labels from colliding
        # at this width; hand-tuned hspace/wspace cannot adapt to the tick label
        # widths, which vary per instance.
        gs = GridSpec(len(_ROWS), 4, figure=fig, width_ratios=[1.3, 1, 1, 1])

        for r, ((mlabel, mshort, mcol, emark), rd) in enumerate(zip(_ROWS, row_data)):
            bottom_row = r == len(_ROWS) - 1
            # ── Panel a) phase-separated hatched bars ──
            # Exact-phase base = method colour (solid); PLS-phase gain = PLS blue
            # (hatched). Colour matches the scatter markers; texture marks phase.
            axa = fig.add_subplot(gs[r, 0])
            bars = rd["bars"]
            present = [(ratio, name) for ratio, name in _PF_RATIOS if ratio in bars]
            xs = list(range(len(present)))
            for xi, (ratio, name) in zip(xs, present):
                bd = bars[ratio]
                if bd["hv_p1"] > 0:
                    axa.bar(xi, bd["hv_p1"], width=0.68, color=mcol,
                            linewidth=0, zorder=3)
                if bd["hv_p2"] > 0:
                    # Hatch drawn in the method colour on a pale wash of itself,
                    # so the stacked segment reads as "same algorithm, second
                    # phase" rather than as a third, unrelated series.
                    axa.bar(xi, bd["hv_p2"], width=0.68, bottom=bd["hv_p1"],
                            facecolor=_pf_tint(_PLS_COLOR), hatch=_PLS_HATCH,
                            edgecolor=_PLS_COLOR, linewidth=_PF_SPINE_LW,
                            zorder=3)
            axa.set_xticks(xs)
            # Five "100:0"-style ratios need ~90 pt of text against a ~75 pt plot
            # area, so they cannot be set upright at any column width that still
            # leaves the scatter panels usable. 45 degrees with `rotation_mode=
            # "anchor"` keeps each label's right end under its own bar.
            axa.set_xticklabels([name for _r, name in present],
                                fontsize=_PF_TICK_PT, rotation=45,
                                ha="right", rotation_mode="anchor")
            axa.set_ylabel("Final HV", fontsize=_PF_LABEL_PT)
            axa.set_xlabel("Exact : PLS ratio", fontsize=_PF_LABEL_PT)
            _pf_panel_title(axa, _PF_PANEL_LETTERS[r * 4], mlabel)
            axa.set_ylim(bar_lo, bar_hi)
            axa.yaxis.set_major_locator(MaxNLocator(nbins=4))
            # Bars sit on a category axis: vertical gridlines would cut through
            # them without helping anyone read a value off the chart.
            _pf_style_axes(axa, grid_axis="y")
            axa.tick_params(axis="x", length=0)

            # ── Panels b/c/d) Pareto-front scatter ──
            for ci, ratio in enumerate(_PF_SCATTER_RATIOS):
                ax = fig.add_subplot(gs[r, ci + 1])
                sc = rd["scatter"].get(ratio, {})
                exact = sc.get("exact", [])
                pls = sc.get("pls", [])
                # PLS underneath: it is the denser cloud, so drawing the sparser
                # exact front on top keeps the seeds visible where they overlap.
                if pls:
                    _pf_front(ax, [(x / ux, y / uy) for x, y in pls],
                              marker="^", size=7, color=_PLS_COLOR, z=2)
                if exact:
                    _pf_front(ax, [(x / ux, y / uy) for x, y in exact],
                              marker=emark, size=9, color=mcol, z=4)
                # Same "exact:PLS" notation as the bar axis tick labels.
                pct = f"{int(ratio * 100)}:{int((1 - ratio) * 100)}"
                _pf_panel_title(ax, _PF_PANEL_LETTERS[r * 4 + ci + 1], pct)
                ax.set_xlim(*sc_xlim)
                ax.set_ylim(*sc_ylim)
                # Every scatter panel shares one axis range (set above), so the
                # tick labels and axis titles are drawn once per row/column
                # instead of eight times. That reclaimed space is what pays for
                # the larger type — the labels below are the reason this is
                # legible at 122 mm, not the point sizes alone.
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                _pf_style_axes(ax)
                # Both rows carry their own x axis. The rows are independent
                # methods rather than a continuation of one scale, so making the
                # top row borrow the bottom row's axis forced a reader to look
                # four panels away to read a cost off panel b.
                ax.set_xlabel("Cost", fontsize=_PF_LABEL_PT)
                if ci == 0:
                    ax.set_ylabel("Cloudy area", fontsize=_PF_LABEL_PT)
                else:
                    ax.tick_params(labelleft=False)

        # One legend, placed *outside* the axes so `constrained_layout` reserves
        # room for it. The previous pair of legends was anchored below the figure
        # with negative bbox coordinates, which only stayed visible because of
        # `bbox_inches="tight"` — the same crop that broke the on-page scale.
        # Colour = algorithm throughout; marker entries name the scatter series,
        # the hatched patch names the PLS-phase gain in the bars.
        from matplotlib.lines import Line2D

        # Build one front/exact-phase legend pair per rendered row, so a
        # single-row figure does not try to unpack two.
        _exact_handles = []
        for _lbl, _short, _col, _mk in _ROWS:
            _exact_handles.append(
                Line2D([], [], marker=_mk, color=_col, linestyle="None",
                       markersize=3.2, label=f"{_lbl} front"))
            _exact_handles.append(
                mpatches.Patch(facecolor=_col, linewidth=0,
                               label=f"{_short} exact phase"))
        # Six entries in three columns. Matplotlib fills a legend column-major,
        # so this ordering puts each algorithm in its own column: the front
        # marker on top, the bar swatch that encodes the same algorithm below.
        # The previous legend listed only the hatched PLS-phase patch, leaving
        # the two solid bar colours — the largest areas of ink in panels a/e —
        # undefined anywhere in the figure.
        handles = _exact_handles + [
            Line2D([], [], marker="^", color=_PLS_COLOR, linestyle="None",
                   markersize=3.0, label="Pareto Local Search front"),
            mpatches.Patch(facecolor=_pf_tint(_PLS_COLOR), hatch=_PLS_HATCH,
                           edgecolor=_PLS_COLOR, linewidth=_PF_SPINE_LW,
                           label="PLS-phase HV gain"),
        ]
        leg = fig.legend(
            handles=handles, loc="outside lower center", ncol=3,
            fontsize=_PF_LEGEND_PT, frameon=False, handletextpad=0.5,
            columnspacing=1.6, labelspacing=0.35, handlelength=1.4,
        )
        for txt in leg.get_texts():
            txt.set_color("black")

        out = _pf_save(fig, output_dir, f"{inst}_pareto_fronts")
        plt.close(fig)
        print(f"  Saved: {out}", flush=True)


# ── Phase-1 budget sweep ────────────────────────────────────────────────────
#
# The other phase figures sample the exact:heuristic split at the four or five
# ratios that were actually RUN. This one sweeps the exact phase alone in 10%
# steps of the same wall-clock budget, which needs no extra runs: the exact
# phase is replayed from the pseudo-solution seed set by cutting it at
# `f * timeout`, exactly as `_pf_exact_hv` does for the run figures.
#
# Every bar is drawn to a total height of 1.0. The solid segment is the share
# of the instance's achievable HV that the exact phase has secured by that
# budget; the hatched remainder is what the heuristic phase is left to close.
# The normaliser is the best final HV any configuration reached on that
# instance, recomputed against the same bounds as the bars themselves so the
# ratio is between two commensurable numbers.

_PHASE1_STEPS: list[float] = [round(0.1 * i, 1) for i in range(1, 11)]


def generate_phase1_sweep_figures(
    an_dir: Path,
    output_dir: Path,
    an_seeds_override: Path | None = None,
    filter_regex: str | None = None,
    num_points: int = 30,
) -> None:
    """Per-instance `{instance}_phase1_sweep.png`: exact-phase HV share at 10%
    steps of the budget, each bar completed to 1.0 by the heuristic phase."""
    from matplotlib.ticker import MaxNLocator

    an_seeds_dir = Path(an_seeds_override or _PSEUDO_SOURCE_DIRS["an_2d_highs"])
    an_dir = Path(an_dir)
    if not an_dir.is_dir():
        print(f"{an_dir} does not exist", flush=True)
        return

    matplotlib.rcParams["hatch.linewidth"] = 0.5
    _EXACT_COL = "#d62728"
    _PLS_COL = "#1f77b4"

    pat = re.compile(filter_regex) if filter_regex else None
    instances = sorted(
        p.stem for p in an_dir.glob("*.json") if p.stem != "all_experiments"
    )
    if pat:
        instances = [i for i in instances if pat.search(i)]
    if not instances:
        print(f"No instances found in {an_dir}", flush=True)
        return

    for inst in instances:
        result = json.loads((an_dir / f"{inst}.json").read_text())
        bounds = result.get("shared_bounds")
        if not bounds:
            print(f"  {inst}: missing shared_bounds, skipping", flush=True)
            continue
        seed_path = an_seeds_dir / f"{inst}.json"
        seeds = (
            json.loads(seed_path.read_text()).get("solutions", [])
            if seed_path.exists()
            else []
        )
        if not seeds:
            print(f"  WARNING: {inst}: no seeds in {an_seeds_dir}, skipping",
                  flush=True)
            continue
        bounds = _pf_widen_bounds(bounds, [seeds])
        timeout = float(result.get("timeout_s", 0.0))

        # Normaliser: best final HV on this instance, recomputed against the
        # same (widened) bounds the bars use.
        recomp = _recompute_result_with_bounds(result, an_dir, bounds, num_points)
        finals = [
            float(c.get("final_hv", 0.0)) for c in recomp.get("configs", {}).values()
        ]
        best = max(finals) if finals else 0.0
        if best <= 0:
            print(f"  {inst}: no positive final HV, skipping", flush=True)
            continue

        shares, raw = [], []
        for f in _PHASE1_STEPS:
            hv1 = _pf_exact_hv(seeds, bounds, f * timeout)
            raw.append(hv1)
            shares.append(min(1.0, hv1 / best))

        fig, ax = plt.subplots(
            figsize=(_PF_FIG_W * 0.62, _PF_FIG_H * 0.62), layout="constrained"
        )
        xs = list(range(len(_PHASE1_STEPS)))
        ax.bar(xs, shares, width=0.72, color=_EXACT_COL, linewidth=0, zorder=3,
               label="Aneja & Nair exact phase")
        ax.bar(
            xs, [1.0 - s for s in shares], width=0.72, bottom=shares,
            facecolor=_pf_tint(_PLS_COL), hatch="////", edgecolor=_PLS_COL,
            linewidth=_PF_SPINE_LW, zorder=3, label="remaining to best HV",
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{int(f * 100)}" for f in _PHASE1_STEPS],
                           fontsize=_PF_TICK_PT)
        ax.set_xlabel("Exact-phase budget (% of timeout)", fontsize=_PF_LABEL_PT)
        ax.set_ylabel("Share of best final HV", fontsize=_PF_LABEL_PT)
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        _pf_style_axes(ax, grid_axis="y")
        ax.tick_params(axis="x", length=0)
        ax.set_title(
            f"{inst}  (T = {timeout:.0f} s, best HV = {best:.4f})",
            fontsize=_PF_LABEL_PT,
        )
        # Bars span the full height at every x, so any in-axes legend sits on
        # top of data; put it under the axes instead.
        leg = fig.legend(
            loc="outside lower center", ncol=2, fontsize=_PF_LEGEND_PT,
            frameon=False, handletextpad=0.5, columnspacing=1.6,
            labelspacing=0.35, handlelength=1.4,
        )
        for t in leg.get_texts():
            t.set_color("black")

        out = _pf_save(fig, output_dir, f"{inst}_phase1_sweep")
        plt.close(fig)
        pct = ", ".join(f"{int(f * 100)}%:{s:.3f}"
                        for f, s in zip(_PHASE1_STEPS, shares))
        print(f"  {inst} exact-phase share -> {pct}", flush=True)
        print(f"  Saved: {out}", flush=True)


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


# ─── Publication ranking plots ───────────────────────────────────────

_DISPLAY_LABELS: dict[str, str] = {
    "Default PLS":       "PLS",
    "Baseline NSGA-II":  "EA (NSGA-II)",
    "Baseline NSGA-III": "EA (NSGA-III)",
    "Baseline MOEA/D":   "EA (MOEA/D)",
}


def _display_label(label: str) -> str:
    return _DISPLAY_LABELS.get(label, label)


def _load_ranking_data(output_dir: Path) -> "dict[str, list[float]]":
    """Load final_hv for every config across all per-instance JSON files."""
    data: dict[str, list[float]] = {}
    for f in sorted(output_dir.glob("*.json")):
        if f.stem == "all_experiments":
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        for label, cfg in d.get("configs", {}).items():
            hv = cfg.get("final_hv")
            if hv is not None:
                data.setdefault(label, []).append(float(hv))
    if not data:
        return {}
    n = max(len(v) for v in data.values())
    return {k: v for k, v in data.items() if len(v) == n}


def plot_cd_diagram(output_dir: Path, output_path: "Path") -> None:
    """Critical Difference diagram: Friedman omnibus + post-hoc Wilcoxon + Holm."""
    if not HAS_MATPLOTLIB:
        return
    try:
        import pandas as pd
        import scikit_posthocs as sp
        from scipy.stats import friedmanchisquare
    except ImportError as e:
        print(f"  Skipping CD diagram (missing dependency: {e})", flush=True)
        return

    data = _load_ranking_data(output_dir)
    if not data:
        print("  No data found for CD diagram.", flush=True)
        return

    n_inst = max(len(v) for v in data.values())
    # rows = instances, cols = methods (display labels)
    df = pd.DataFrame(
        {_display_label(k): v for k, v in data.items() if len(v) == n_inst}
    )

    # ── Friedman omnibus test ────────────────────────────────────────────
    stat, p_friedman = friedmanchisquare(*[df[c].values for c in df.columns])
    print(f"  Friedman: χ²={stat:.3f}, p={p_friedman:.3e}  ({len(df.columns)} methods, "
          f"{n_inst} instances)", flush=True)

    # ── Average ranks (rank 1 = highest HV = best) ───────────────────────
    ranks = df.rank(axis=1, ascending=False).mean().sort_values()

    # Post-hoc pairwise Wilcoxon + Holm: expects long-format DataFrame
    df_long = df.melt(var_name="method", value_name="hv")
    p_matrix = sp.posthoc_wilcoxon(df_long, val_col="hv", group_col="method", p_adjust="holm")

    # ── Draw ─────────────────────────────────────────────────────────────
    n_methods = len(ranks)
    fig, ax = plt.subplots(figsize=(11, max(7, n_methods * 0.44)))
    sp.critical_difference_diagram(
        ranks,
        p_matrix,
        ax=ax,
        alpha=0.05,
        left_only=True,
        label_fmt_left="{label}  ({rank:.2f})",
        label_props={"fontsize": 8.5},
        crossbar_props={"color": "#2c7bb6", "linewidth": 2.8, "zorder": 3},
        elbow_props={"color": "#2c7bb6", "linewidth": 1.3},
        marker_props={"marker": "o", "s": 30, "color": "#2c7bb6", "zorder": 4},
    )
    ax.set_title(
        f"Critical Difference Diagram\n"
        f"Friedman χ²={stat:.2f}, p={p_friedman:.1e}  ·  "
        f"Post-hoc: pairwise Wilcoxon + Holm correction, α=0.05\n"
        f"Bars connect methods with no statistically significant difference",
        fontsize=9,
        fontweight="bold",
        pad=10,
    )
    fig.tight_layout()
    _save_fig(fig, output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}", flush=True)


def _draw_simplex_ax(
    ax: "plt.Axes",
    samples: "Any",  # (N, 3) array: col0=p_left (A wins), col1=p_rope, col2=p_right (B wins)
    p_left: float,
    p_rope: float,
    p_right: float,
    name_a: str,
    name_b: str,
    rope: float,
) -> None:
    """Draw a Bayesian simplex triangle with posterior sample cloud on a given Axes."""
    import numpy as np

    h = math.sqrt(3) / 2  # height of equilateral triangle

    # Vertices: left=(0,0) → A wins, right=(1,0) → B wins, top=(0.5,h) → rope
    # Barycentric → 2D: point = p_left*(0,0) + p_right*(1,0) + p_rope*(0.5,h)
    def bary_to_2d(b):  # b: (..., 3) → (..., 2)
        xs = b[..., 2] * 1.0 + b[..., 1] * 0.5
        ys = b[..., 1] * h
        return xs, ys

    # ── Background region colouring ──────────────────────────────────────
    # Divide triangle into 3 regions by the three medians meeting at centroid.
    # Region A (left): centroid, left vertex, midpoints of left edges
    # Use a fine mesh approach: for each pixel inside triangle, colour by dominant dim.
    res = 200
    xs_grid = np.linspace(0, 1, res)
    ys_grid = np.linspace(0, h, res)
    Xg, Yg = np.meshgrid(xs_grid, ys_grid)
    # Back-project to barycentric
    # y = p_rope * h  →  p_rope = y/h
    # x = p_right + p_rope/2  →  p_right = x - p_rope/2
    # p_left = 1 - p_right - p_rope
    p_r_g = Yg / h
    p_B_g = Xg - p_r_g * 0.5
    p_A_g = 1.0 - p_B_g - p_r_g
    inside = (p_A_g >= 0) & (p_B_g >= 0) & (p_r_g >= 0)
    dominant = np.argmax(
        np.stack([p_A_g, p_r_g, p_B_g], axis=-1), axis=-1
    )  # 0=A wins, 1=rope, 2=B wins
    # RGBA image
    colours = np.ones((res, res, 4))  # white
    colours[inside & (dominant == 0)] = [0.70, 0.85, 1.00, 0.45]  # blue: A
    colours[inside & (dominant == 1)] = [0.80, 0.95, 0.80, 0.45]  # green: rope
    colours[inside & (dominant == 2)] = [1.00, 0.80, 0.75, 0.45]  # red: B
    colours[~inside] = [1, 1, 1, 0]  # transparent outside
    ax.imshow(
        colours,
        extent=[0, 1, 0, h],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        zorder=0,
    )

    # ── Posterior sample cloud ────────────────────────────────────────────
    sx, sy = bary_to_2d(samples)
    ax.scatter(
        sx, sy,
        s=1.2, c="#333333", alpha=0.06, linewidths=0, zorder=2,
        rasterized=True,
    )

    # ── Mean point ───────────────────────────────────────────────────────
    mx, my = bary_to_2d(np.array([[p_left, p_rope, p_right]]))
    ax.scatter(mx, my, s=60, c="#111111", marker="*", zorder=5)

    # ── Triangle outline ─────────────────────────────────────────────────
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, h, 0]
    ax.plot(tri_x, tri_y, "k-", linewidth=1.0, zorder=3)

    # ── Vertex labels ────────────────────────────────────────────────────
    offset = 0.06
    ax.text(-offset, -offset * 0.7, f"{name_a}\nwins",
            ha="center", va="top", fontsize=7.5, color="#1a6db5", fontweight="bold")
    ax.text(1 + offset, -offset * 0.7, f"{name_b}\nwins",
            ha="center", va="top", fontsize=7.5, color="#c0392b", fontweight="bold")
    ax.text(0.5, h + offset * 0.5, f"equivalent\n(ROPE ±{rope:.0%})",
            ha="center", va="bottom", fontsize=7.0, color="#27ae60", fontweight="bold")

    # ── Probability annotations ───────────────────────────────────────────
    ax.text(0.02, 0.02, f"P={p_left:.3f}", transform=ax.transAxes,
            fontsize=8, color="#1a6db5", fontweight="bold")
    ax.text(0.98, 0.02, f"P={p_right:.3f}", transform=ax.transAxes,
            ha="right", fontsize=8, color="#c0392b", fontweight="bold")
    ax.text(0.50, 0.92, f"P={p_rope:.3f}", transform=ax.transAxes,
            ha="center", fontsize=8, color="#27ae60", fontweight="bold")

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.18, h + 0.18)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_bayesian_tests(output_dir: Path, output_path: "Path") -> None:
    """Bayesian signed-rank simplex plots (Benavoli 2014) for key comparisons."""
    if not HAS_MATPLOTLIB:
        return
    try:
        import baycomp
        import numpy as np
    except ImportError as e:
        print(f"  Skipping Bayesian plots (missing: {e})", flush=True)
        return

    data = _load_ranking_data(output_dir)
    if not data:
        return

    ROPE = 0.01  # 1 % HV difference = practically equivalent

    # (title, key_a, display_a, key_b, display_b)
    comparisons = [
        ("Hybrid 80:20",          "Default PLS",
         "Hybrid 80:20",          "PLS"),
        ("Hybrid 80:20",          "Baseline NSGA-II",
         "Hybrid 80:20",          "EA (NSGA-II)"),
        ("Hybrid 80:20",          "Baseline MOEA/D",
         "Hybrid 80:20",          "EA (MOEA/D)"),
        ("Improved NSGA-II 80:20", "Baseline NSGA-II",
         "Imp.NSGA-II 80:20",     "EA (NSGA-II)"),
        ("Improved NSGA-III 80:20", "Baseline NSGA-III",
         "Imp.NSGA-III 80:20",    "EA (NSGA-III)"),
        ("Improved MOEA/D 80:20", "Baseline MOEA/D",
         "Imp.MOEA/D 80:20",      "EA (MOEA/D)"),
    ]

    ncols = 3
    nrows = math.ceil(len(comparisons) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.2, nrows * 4.0),
        squeeze=False,
    )

    for idx, (key_a, key_b, name_a, name_b) in enumerate(comparisons):
        ax = axes[idx // ncols][idx % ncols]
        x = np.array(data.get(key_a, []))
        y = np.array(data.get(key_b, []))
        if len(x) == 0 or len(y) == 0:
            ax.set_visible(False)
            continue
        # Posterior samples: col0=P(x>y), col1=P(rope), col2=P(y>x)
        samples = baycomp.SignedRankTest.sample(x, y, rope=ROPE, nsamples=50_000)
        p_left  = float(samples[:, 0].mean())
        p_rope  = float(samples[:, 1].mean())
        p_right = float(samples[:, 2].mean())
        _draw_simplex_ax(ax, samples, p_left, p_rope, p_right, name_a, name_b, ROPE)
        ax.set_title(
            f"{name_a}  vs  {name_b}",
            fontsize=9, fontweight="bold", pad=4,
        )

    for j in range(len(comparisons), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        f"Bayesian Signed-Rank Tests  (Benavoli et al. 2014,  ROPE = {ROPE:.0%})\n"
        "Posterior cloud over 50 000 Dirichlet samples  ·  ★ = posterior mean  "
        "·  blue = A wins  ·  green = equivalent  ·  red = B wins",
        fontsize=9.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    _save_fig(fig, output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}", flush=True)


def plot_cluster_bars(
    output_dir: Path,
    output_path: "Path",
) -> None:
    """Bar chart: top-ranked cluster vs PLS vs Baseline EAs, with Wilcoxon brackets."""
    if not HAS_MATPLOTLIB:
        return
    try:
        import numpy as np
        from scipy.stats import wilcoxon as _wilcoxon
    except ImportError:
        return

    data = _load_ranking_data(output_dir)
    if not data:
        return

    # ── Define groups ────────────────────────────────────────────────────
    TOP_CLUSTER   = ["Hybrid 80:20", "Hybrid 50:50"]
    REFERENCE_KEYS = [
        ("Default PLS",    "PLS",           "#4daf4a"),
        ("Baseline NSGA-II",  "EA (NSGA-II)",  "#e41a1c"),
        ("Baseline NSGA-III", "EA (NSGA-III)", "#ff7f00"),
        ("Baseline MOEA/D",   "EA (MOEA/D)",   "#984ea3"),
    ]
    TOP_COLORS = {"Hybrid 80:20": "#2166ac", "Hybrid 50:50": "#4393c3"}

    def _get_color(k: str) -> str:
        if k in TOP_COLORS:
            return TOP_COLORS[k]
        for rk, _, c in REFERENCE_KEYS:
            if rk == k:
                return c
        return "#999999"

    all_keys    = TOP_CLUSTER + [k for k, _, _ in REFERENCE_KEYS]
    all_labels  = [_display_label(k) for k in all_keys]
    all_colors  = [_get_color(k) for k in all_keys]
    present = [(k, lbl, col) for k, lbl, col in zip(all_keys, all_labels, all_colors)
               if k in data]
    keys_p   = [t[0] for t in present]
    labels_p = [t[1] for t in present]
    colors_p = [t[2] for t in present]

    means = np.array([np.mean(data[k]) for k in keys_p])
    stds  = np.array([np.std(data[k])  for k in keys_p])

    fig, ax = plt.subplots(figsize=(max(8, len(present) * 1.15), 6.0))
    x = np.arange(len(present))

    ax.bar(
        x, means, yerr=stds,
        color=colors_p,
        edgecolor="white",
        linewidth=0.5,
        width=0.58,
        zorder=3,
        error_kw=dict(ecolor="#333333", capsize=4,
                      elinewidth=1.0, capthick=1.0, zorder=4),
    )

    # ── Significance brackets: Hybrid 80:20 vs each reference ────────────
    top_key = "Hybrid 80:20"
    bracket_base = max(means + stds) + 0.010
    step = 0.022
    if top_key in keys_p:
        top_idx = keys_p.index(top_key)

        for ref_i, (ref_key, _, _) in enumerate(REFERENCE_KEYS):
            if ref_key not in keys_p:
                continue
            ref_idx = keys_p.index(ref_key)
            a_vals = data[top_key]
            b_vals = data[ref_key]
            diffs  = [a - b for a, b in zip(a_vals, b_vals)]
            nz     = [d for d in diffs if d != 0]
            if len(nz) >= 2:
                _, pval = _wilcoxon(a_vals, b_vals, alternative="two-sided")
                sig_str = "***" if pval < 0.001 else ("**" if pval < 0.01
                          else ("*" if pval < 0.05 else "n.s."))
            else:
                sig_str = "ties"

            y_br = bracket_base + ref_i * step
            bx0  = float(x[min(top_idx, ref_idx)])
            bx1  = float(x[max(top_idx, ref_idx)])
            ax.plot(
                [bx0, bx0, bx1, bx1],
                [y_br - 0.003, y_br, y_br, y_br - 0.003],
                lw=1.0, color="#333333", zorder=5,
            )
            ax.text(
                (bx0 + bx1) / 2, y_br + 0.001,
                sig_str,
                ha="center", va="bottom", fontsize=8.5, color="#333333", zorder=5,
            )

    # ── Vertical separator between top cluster and references ─────────────
    sep_x = len(TOP_CLUSTER) - 0.5
    ax.axvline(sep_x, color="#aaaaaa", linestyle="--", linewidth=0.9, zorder=2)

    # Determine y limits before adding annotations
    y_bottom = max(0.0, min(means) - 4 * max(stds) - 0.01)
    y_ceiling = bracket_base + len(REFERENCE_KEYS) * step + 0.030
    ax.set_ylim(y_bottom, y_ceiling)

    # Label the two sections inside the plot area near the bottom
    y_label = y_bottom + (y_ceiling - y_bottom) * 0.03
    ax.text(sep_x - len(TOP_CLUSTER) / 2, y_label,
            "top cluster", ha="center", va="bottom", fontsize=8,
            color="#555555", style="italic")
    ax.text(sep_x + (len(present) - len(TOP_CLUSTER)) / 2, y_label,
            "reference methods", ha="center", va="bottom", fontsize=8,
            color="#555555", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_p, fontsize=9.5, rotation=18, ha="right")
    ax.set_ylabel("Mean Normalised Hypervolume", fontsize=10)
    ax.set_title(
        "Top Cluster vs Reference Methods"
        "   (mean \u00b1 std over all instances  \u00b7  brackets: Wilcoxon two-sided)",
        fontsize=10,
        fontweight="bold",
    )
    ax.yaxis.grid(True, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save_fig(fig, output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}", flush=True)


def generate_ranking_plots(output_dir: Path) -> None:
    """Generate CD diagram, Bayesian simplex plots, and cluster bar chart."""
    print("\nGenerating ranking plots …", flush=True)
    plot_cd_diagram(output_dir,     output_dir / "fig_cd_diagram.png")
    plot_bayesian_tests(output_dir, output_dir / "fig_bayesian_tests.png")
    plot_cluster_bars(output_dir,   output_dir / "fig_cluster_bars.png")
    print("Ranking plots done.", flush=True)


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
        "--biobjective",
        action="store_true",
        help=(
            "Run in 2-objective mode (min_cost + cloud_coverage) instead of "
            "the default 4 objectives. Affects OBJECTIVES globally for this run."
        ),
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
        # Derived from _PSEUDO_SOURCE_DIRS rather than duplicated: a hardcoded
        # copy silently rejects any source added to the dict, which is an
        # argparse error at launch rather than anything the caller can see
        # coming.
        choices=sorted(_PSEUDO_SOURCE_DIRS),
        help=(
            "Source of initial-population solutions for hybrid configs. "
            "'gpbaa'/'monise' use 4D pre-recorded solutions; "
            "'gpbaa_2d'/'monise_2d' use 2D biobjective OR-Tools solutions on test instances; "
            "'gpbaa_2d_pub'/'monise_2d_pub' use 2D solutions on publication instances."
        ),
    )
    parser.add_argument(
        "--pub-instances",
        action="store_true",
        help=(
            "Use publication-data instances (100/150/200/250 images, 5 cities) "
            "instead of the default test instances."
        ),
    )
    parser.add_argument(
        "--rc-instances",
        action="store_true",
        help=(
            "Use publication-data/satellite-data/instances_random_clouds "
            "instances (family C / random-clouds) instead of the default test "
            "instances. Sizes 50/75/100/125/150/175/200, 5 cities."
        ),
    )
    parser.add_argument(
        "--pls-restart",
        action="store_true",
        help=(
            "Enable PLS perturbation restart. Off by default because it used to "
            "panic ('Solution set contains dominated solution!'); that bug is "
            "fixed in sims-heuristics. With restart off PLS stops at its first "
            "local-optimum set, using a few percent of its budget."
        ),
    )
    parser.add_argument(
        "--hard-instances",
        action="store_true",
        help=(
            "Use publication-data/satellite-data/instances_hard_an — the set "
            "calibrated so Aneja & Nair returns 5-15 solutions at the experiment "
            "budget, leaving the heuristic phase real headroom. Per-city size "
            "ladders are read from the directory (they differ: pool limits and "
            "the A&N difficulty window do not line up across cities)."
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
    parser.add_argument(
        "--ranking-plots",
        action="store_true",
        help=(
            "Generate CD diagram, Bayesian signed-rank simplex plots, and top-cluster "
            "bar chart from existing JSON artifacts in --output-dir, then exit."
        ),
    )
    parser.add_argument(
        "--pf-seeds-gpbaa",
        type=Path,
        default=None,
        help=(
            "Directory of GPBA-A pseudo-solutions for --pareto-fronts. Defaults "
            "to the gpbaa_2d_highs publication dataset; override when the "
            "instance set keeps its seeds elsewhere (e.g. the A&N-calibrated "
            "hard grid), otherwise the exact-phase bars come out empty."
        ),
    )
    parser.add_argument(
        "--pf-seeds-an",
        type=Path,
        default=None,
        help="Directory of A&N pseudo-solutions for --pareto-fronts (see --pf-seeds-gpbaa).",
    )
    parser.add_argument(
        "--pareto-fronts",
        action="store_true",
        help=(
            "Reconstruct the paper's per-instance `{instance}_pareto_fronts.png` "
            "figures (2 rows: GPBA-A + Aneja & Nair; phase-separated hatched HV "
            "bars + front scatters) from trace artifacts, then exit. Uses "
            "--highs-dir and --an-dir; writes to --output-dir."
        ),
    )
    parser.add_argument(
        "--friedman",
        action="store_true",
        help=(
            "Friedman test over instances with all-pairs post-hoc and "
            "Bergmann-Hommel correction (verified against scmamp). Prints "
            "average ranks and the adjusted p-matrix, and writes "
            "`friedman_ranking.tex` to --output-dir, then exits."
        ),
    )
    parser.add_argument(
        "--ea-tables",
        action="store_true",
        help=(
            "Emit `ea_hv_gpbaa.tex` / `ea_hv_aneja.tex`: LaTeX longtables of "
            "hypervolume by second-phase algorithm (PLS, NSGA-II, NSGA-III, "
            "MOEA/D), mean +/- SD over 10 runs. Same numbers as --ea-bars. "
            "Uses --highs-dir and --an-dir; writes to --output-dir, then exits."
        ),
    )
    parser.add_argument(
        "--ea-bars",
        action="store_true",
        help=(
            "Per-instance `{instance}_ea_bars.png`: same 2x4 layout as "
            "--pareto-fronts, but every panel is a phase-decomposed HV bar "
            "chart. Rows = exact method (GPBA-A, Aneja & Nair), columns = "
            "second-phase engine (PLS, NSGA-II, NSGA-III, MOEA/D). Uses "
            "--highs-dir and --an-dir; writes to --output-dir, then exits."
        ),
    )
    parser.add_argument(
        "--append-configs",
        action="store_true",
        help=(
            "Merge into an existing per-instance JSON instead of overwriting it. "
            "Configs whose trace is already stored are reloaded rather than "
            "re-run; shared bounds and every HV curve are then recomputed over "
            "the union, so the merged artifact matches what a single run of all "
            "configs would have produced. Use this to add a ratio or engine to a "
            "finished grid without re-running what is already there."
        ),
    )
    parser.add_argument(
        "--phase1-sweep",
        action="store_true",
        help=(
            "Per-instance `{instance}_phase1_sweep.png`: exact-phase HV in 10%% "
            "steps of the budget, replayed from the pseudo-solution seeds "
            "against the same bounds as the grid results. Each bar is completed "
            "to 1.0 by a hatched 'remaining to best HV' segment. Uses --an-dir "
            "and --pf-seeds-an; writes to --output-dir, then exits."
        ),
    )
    parser.add_argument(
        "--ea-bars-indicator",
        choices=sorted(_INDICATOR_LABEL),
        default=None,
        help=(
            "Same figure as --ea-bars (2x4 phase-decomposed bar chart, same "
            "rows/columns), but using this performance indicator instead of "
            "hypervolume -- for showcasing density/spread gains HV alone "
            "can't show. Single-run (run 0) bars, no error bars (see "
            "generate_ea_bar_figures_indicator's docstring for why). Uses "
            "--highs-dir and --an-dir; writes to "
            "--output-dir/plots_ea_<indicator>_png, then exits."
        ),
    )
    parser.add_argument(
        "--ea-lollipop-indicator",
        choices=sorted(_INDICATOR_LABEL),
        default=None,
        help=(
            "Per-instance `{instance}_ea_lollipop.png`: same 2xN grid as "
            "--ea-bars-indicator, but panels are indexed by ratio (not "
            "engine) and each panel plots all four engines' final indicator "
            "value as lollipops (stem + marker) side by side, so engines are "
            "directly comparable in one axes -- unlike the phase-decomposed "
            "bars, which only compare phase 1 vs phase 2 within one engine. "
            "Uses --highs-dir and --an-dir; writes to "
            "--output-dir/plots_ea_lollipop_<indicator>_png, then exits."
        ),
    )
    parser.add_argument(
        "--ea-rank-summary",
        choices=sorted(_INDICATOR_LABEL),
        default=None,
        help=(
            "Cross-instance proof table: ranks the four engines' final "
            "indicator value per instance/ratio/exact-method and aggregates "
            "mean rank + win count over all instances (rank sidesteps "
            "per-instance scale differences that a raw average can't). "
            "Prints the table and writes "
            "--output-dir/ea_rank_summary_<indicator>.csv, then exits."
        ),
    )
    parser.add_argument(
        "--highs-dir",
        type=Path,
        default=Path("results/eval_publication_highs"),
        help="GPBA-A trace-artifact dir for --pareto-fronts (default: results/eval_publication_highs)",
    )
    parser.add_argument(
        "--an-dir",
        type=Path,
        default=Path("results/eval_publication_an"),
        help="Aneja & Nair trace-artifact dir for --pareto-fronts (default: results/eval_publication_an)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Override timeout (seconds) for all instances, ignoring the size-based schedule",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help=(
            "Number of independent runs per config for statistical error bars "
            "(default: 1). When > 1, PLS configs run non-deterministically and "
            "EA configs use seed=run_index. JSON output gains run_hvs, "
            "final_hv (mean) and final_hv_std fields."
        ),
    )

    args = parser.parse_args()

    # Apply timeout override globally.
    global _timeout_override
    _timeout_override = args.timeout

    # Enable non-deterministic PLS when running multiple iterations.
    global _force_nondeterministic
    if args.runs > 1:
        _force_nondeterministic = True
        print(f"Non-deterministic mode enabled ({args.runs} runs per config)", flush=True)

    global _append_configs
    _append_configs = args.append_configs
    if _append_configs:
        print("Append mode: existing configs will be reused, bounds recomputed", flush=True)

    global _pls_perturbation_restart
    if args.pls_restart:
        _pls_perturbation_restart = True
        print("PLS perturbation restart ENABLED", flush=True)

    # Apply biobjective mode globally (affects OBJECTIVES used by every config's
    # .run() and by bounds/HV-curve computation throughout run_instance()).
    global OBJECTIVES
    if args.biobjective:
        OBJECTIVES = ["min_cost", "cloud_coverage"]
        print(f"Biobjective mode: OBJECTIVES = {OBJECTIVES}")

    # Apply pseudo-solution source globally (affects all _load_pseudo_solutions calls)
    global _PSEUDO_SOLUTIONS_SOURCE, _pseudo_cache
    _PSEUDO_SOLUTIONS_SOURCE = args.pseudo_source
    _pseudo_cache.clear()  # invalidate cache when source changes

    # Pareto-front figure reconstruction (self-contained; no instances/solves).
    if args.pareto_fronts:
        generate_pareto_front_figures(
            args.highs_dir,
            args.an_dir,
            args.output_dir,
            gpbaa_seeds_override=args.pf_seeds_gpbaa,
            an_seeds_override=args.pf_seeds_an,
            filter_regex=args.filter,
            num_points=args.num_points,
        )
        return 0

    if args.friedman:
        run_friedman_test(
            args.highs_dir,
            args.an_dir,
            args.output_dir,
            filter_regex=args.filter,
            num_points=args.num_points,
        )
        return 0

    if args.ea_tables:
        generate_ea_tables(
            args.highs_dir,
            args.an_dir,
            args.output_dir,
            filter_regex=args.filter,
            num_points=args.num_points,
        )
        return 0

    if args.ea_bars:
        generate_ea_bar_figures(
            args.highs_dir,
            args.an_dir,
            args.output_dir,
            filter_regex=args.filter,
            num_points=args.num_points,
            gpbaa_seeds_override=args.pf_seeds_gpbaa,
            an_seeds_override=args.pf_seeds_an,
        )
        return 0

    if args.phase1_sweep:
        generate_phase1_sweep_figures(
            args.an_dir,
            args.output_dir,
            an_seeds_override=args.pf_seeds_an,
            filter_regex=args.filter,
            num_points=args.num_points,
        )
        return 0

    if args.ea_bars_indicator:
        # _pf_save appends "_png"/"_eps" to whatever base dir it's given (see
        # its docstring), so pass the un-suffixed base here to land in
        # sibling plots_ea_<indicator>_png / plots_ea_<indicator>_eps dirs --
        # matching the plots_ea_png / plots_ea_eps convention exactly.
        generate_ea_bar_figures_indicator(
            args.ea_bars_indicator,
            args.highs_dir,
            args.an_dir,
            args.output_dir / f"plots_ea_{args.ea_bars_indicator}",
            filter_regex=args.filter,
            gpbaa_seeds_override=args.pf_seeds_gpbaa,
            an_seeds_override=args.pf_seeds_an,
        )
        return 0

    if args.ea_lollipop_indicator:
        generate_ea_lollipop_figures_indicator(
            args.ea_lollipop_indicator,
            args.highs_dir,
            args.an_dir,
            args.output_dir / f"plots_ea_lollipop_{args.ea_lollipop_indicator}",
            filter_regex=args.filter,
            gpbaa_seeds_override=args.pf_seeds_gpbaa,
            an_seeds_override=args.pf_seeds_an,
        )
        return 0

    if args.ea_rank_summary:
        import pandas as pd

        summary = compute_ea_rank_summary(
            args.ea_rank_summary,
            args.highs_dir,
            args.an_dir,
            filter_regex=args.filter,
        )
        if summary.empty:
            print("No data to rank.", flush=True)
            return 0
        with pd.option_context("display.max_rows", None, "display.width", 120):
            print(summary.to_string(index=False), flush=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_csv = args.output_dir / f"ea_rank_summary_{args.ea_rank_summary}.csv"
        summary.to_csv(out_csv, index=False)
        print(f"  Saved: {out_csv}", flush=True)
        return 0

    # Override instance set for publication experiments.
    global INSTANCES_DIR
    if args.pub_instances:
        INSTANCES_DIR = Path(__file__).parent.parent / "publication-data" / "experiments"
        _pub_cities = [
            ("lagos_nigeria", [100, 150, 200, 250]),
            ("mexico_city", [100, 150, 200, 250]),
            ("paris", [100, 150, 200, 250]),
            ("rio_de_janeiro", [100, 150, 200, 250]),
            ("tokyo_bay", [100, 150, 200, 250]),
        ]
        instances = [
            (f"{city}_{size}", f"{city}_{size}/{city}_{size}.dzn", size)
            for city, sizes in _pub_cities
            for size in sizes
            if (INSTANCES_DIR / f"{city}_{size}" / f"{city}_{size}.dzn").exists()
        ]
    elif args.hard_instances:
        INSTANCES_DIR = (
            Path(__file__).parent.parent
            / "publication-data"
            / "satellite-data"
            / "instances_hard_an"
        )
        # Ladders are per city, so discover them from disk instead of hardcoding
        # a common list that would silently drop instances.
        instances = []
        for _p in sorted(INSTANCES_DIR.glob("*.dzn")):
            _city, _, _size = _p.stem.rpartition("_")
            instances.append((_p.stem, _p.name, int(_size)))
        instances.sort(key=lambda t: (t[0].rpartition("_")[0], t[2]))
    elif args.rc_instances:
        INSTANCES_DIR = (
            Path(__file__).parent.parent
            / "publication-data"
            / "satellite-data"
            / "instances_random_clouds"
        )
        _rc_cities = [
            ("lagos_nigeria", [50, 75, 100, 125, 150, 175, 200]),
            ("mexico_city", [50, 75, 100, 125, 150, 175, 200]),
            ("paris", [50, 75, 100, 125, 150, 175, 200]),
            ("rio_de_janeiro", [50, 75, 100, 125, 150, 175, 200]),
            ("tokyo_bay", [50, 75, 100, 125, 150, 175, 200]),
        ]
        instances = [
            (f"{city}_{size}", f"{city}_{size}.dzn", size)
            for city, sizes in _rc_cities
            for size in sizes
            if (INSTANCES_DIR / f"{city}_{size}.dzn").exists()
        ]
    else:
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
    default_pool = MONISE_CONFIGS if args.pseudo_source in ("monise", "monise_2d", "monise_2d_pub") else CONFIGS

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

    if args.ranking_plots:
        generate_ranking_plots(args.output_dir)
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
    print(f"Runs/config: {args.runs}")

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
                num_runs=args.runs,
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
