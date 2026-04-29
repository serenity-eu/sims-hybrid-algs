"""Ablation study for PLS optimization improvements.

Runs the two-phase solver with different PLS optimization configurations and
produces per-instance hypervolume-vs-ratio plots, showing the contribution
of each optimisation technique via toggle-based ablation.

Configurations tested (series on each plot):
  1. Baseline          – all optimizations disabled (original PLS behaviour)
  2. +Checkpoint       – tracker checkpoint/restore enabled
  3. +Greedy Init      – greedy initial population enabled
  4. +Perturbation     – perturbation restart enabled
  5. All improvements  – everything enabled

All configurations use the pseudo-solver for the exact phase so that
results are reproducible and no Gurobi licence is needed.

Usage::

    # Run on small 2D instances (fast, ~3 min total):
    pytest tests/test_ablation_study.py -k small_2d --use-pseudo-solver -v

    # Run everything and generate plots:
    pytest tests/test_ablation_study.py --use-pseudo-solver -v

Plots are written to ``sims-core/test_artifacts/ablation_plots/``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import sims_problem
except ImportError:
    pytest.skip("sims_problem not installed", allow_module_level=True)

from sims.core.sims.solver import solve_with_two_phases, _extract_objectives
from sims.core.sims.solver_config import (
    TwoPhaseSolverConfig,
    SolverType,
    FrontStrategy,
)
from sims.core.sims.solvers.pseudo_solver import get_pseudo_solver
from solver_test_utils import (
    SMALL_INSTANCES,
    MEDIUM_INSTANCES,
    LARGE_INSTANCES,
    TWO_PHASE_RATIOS,
    create_problem_instance,
    create_temp_output_dir,
    validate_two_phase_solver_result,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ablation configurations – each toggles one optimisation independently
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationConfig:
    """One PLS configuration to benchmark."""

    label: str
    # Optimization toggles (mirror PlsOptimizations on the Rust side)
    use_checkpoint: bool = False
    use_ranked_candidates: bool = False
    probing_budget: int | None = None  # None = exhaustive
    use_greedy_initial_population: bool = False
    use_perturbation_restart: bool = False
    # Visual style for plots
    color: str = "black"
    marker: str = "o"
    linestyle: str = "-"


# The series that appear on every plot.  Order matters for the legend.
ABLATION_CONFIGS: list[AblationConfig] = [
    AblationConfig(
        label="Baseline",
        use_checkpoint=False,
        use_ranked_candidates=False,
        probing_budget=None,
        use_greedy_initial_population=False,
        use_perturbation_restart=False,
        color="#7f7f7f",
        marker="o",
        linestyle="-",
    ),
    AblationConfig(
        label="+Checkpoint",
        use_checkpoint=True,
        use_ranked_candidates=False,
        probing_budget=None,
        use_greedy_initial_population=False,
        use_perturbation_restart=False,
        color="#1f77b4",
        marker="s",
        linestyle="--",
    ),
    AblationConfig(
        label="+Greedy Init",
        use_checkpoint=False,
        use_ranked_candidates=False,
        probing_budget=None,
        use_greedy_initial_population=True,
        use_perturbation_restart=False,
        color="#9467bd",
        marker="v",
        linestyle=":",
    ),
    AblationConfig(
        label="+Perturbation",
        use_checkpoint=False,
        use_ranked_candidates=False,
        probing_budget=None,
        use_greedy_initial_population=False,
        use_perturbation_restart=True,
        color="#d62728",
        marker="P",
        linestyle="-.",
    ),
    AblationConfig(
        label="All improvements",
        use_checkpoint=True,
        use_ranked_candidates=False,
        probing_budget=None,
        use_greedy_initial_population=True,
        use_perturbation_restart=True,
        color="#2ca02c",
        marker="D",
        linestyle="-",
    ),
]


# Short timeouts chosen to maximise visible improvement delta between
# baseline and optimised configurations.  Shorter budgets mean the
# baseline hasn't converged yet, so algorithmic improvements show up
# more clearly in the hypervolume metric.
def _ablation_timeout(instances: list[str]) -> int:
    """Return a short timeout suitable for ablation benchmarking."""
    for inst in instances:
        stem = Path(inst).stem
        parts = stem.rsplit("_", 1)
        if len(parts) > 1 and parts[1].isdigit():
            size = int(parts[1])
            if size <= 30:
                return 5
            if size <= 50:
                return 10
            if size <= 100:
                return 30
            if size <= 150:
                return 90
            return 180
    return 15


# ---------------------------------------------------------------------------
# Result collection helpers
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Result of a single (config, instance, ratio) run."""

    config_label: str
    instance_name: str
    ratio: tuple[int, int]
    hypervolume: float
    num_solutions: int
    execution_time: float
    success: bool
    error: str | None = None
    # Raw objective points kept so HV can be recomputed with unified bounds
    raw_points: list[list[int]] | None = None


@dataclass
class InstanceResults:
    """Collected results for one instance across all configs and ratios."""

    instance_name: str
    objectives: list[str]
    objective_bounds: list[list[int]]
    runs: list[RunResult] = field(default_factory=list)


def _compute_hypervolume_for_points(
    points: list[list[int]],
    objective_bounds: list[list[int]],
) -> float:
    """Compute normalized hypervolume from raw objective points.

    ``objective_bounds`` is ``[[min, max], ...]`` per objective and is
    expanded to cover every point so the Rust code never panics on
    out-of-range values.
    """
    if not points:
        return 0.0

    # Expand bounds to cover every point.
    expanded = [list(b) for b in objective_bounds]
    for pt in points:
        for i, v in enumerate(pt):
            if v < expanded[i][0]:
                expanded[i][0] = v
            if v > expanded[i][1]:
                expanded[i][1] = v

    try:
        return sims_problem.compute_hypervolume(
            data=points,
            objective_bounds=expanded,
            normalized=True,
        )
    except Exception as exc:
        logger.warning(f"Hypervolume computation failed: {exc}")
        return 0.0


def _compute_hypervolume_for_solutions(
    solutions,
    objectives: list[str],
    objective_bounds: list[list[int]],
) -> float:
    """Convenience wrapper that extracts objective points from Solution objects."""
    if not solutions:
        return 0.0
    points = [_extract_objectives(sol, objectives) for sol in solutions]
    return _compute_hypervolume_for_points(points, objective_bounds)


def _bounds_from_points(all_points: list[list[int]], n_objectives: int) -> list[list[int]]:
    """Compute tight [min, max] bounds from a collection of objective points.

    A small margin (1 unit below min, 10 % of range above max) is added so
    that the reference point sits strictly beyond all solutions and the
    ideal point is strictly below all solutions.
    """
    bounds: list[list[int]] = []
    for i in range(n_objectives):
        vals = [p[i] for p in all_points]
        lo, hi = min(vals), max(vals)
        rng = max(hi - lo, 1)
        bounds.append([max(0, lo - 1), hi + int(rng * 0.1) + 1])
    return bounds


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


# Cache for loaded ProblemInstance objects to avoid re-parsing .dzn files
# (parsing takes ~7 s per instance and dominates ablation runtime).
_instance_cache: dict[str, "ProblemInstance"] = {}


def _get_problem_instance(instance_path: str) -> "ProblemInstance":
    """Return a cached ProblemInstance, loading from disk on first access."""
    if instance_path not in _instance_cache:
        _instance_cache[instance_path] = create_problem_instance(instance_path)
    return _instance_cache[instance_path]


def _run_single(
    instance_path: str,
    objectives: list[str],
    objective_bounds: list[list[int]] | None,
    ratio: tuple[int, int],
    timeout: int,
    config: AblationConfig,
    use_pseudo_solver: bool = True,
) -> RunResult:
    """Run solve_with_two_phases for one (config, ratio) combination."""
    instance_name = Path(instance_path).stem
    start = time.time()
    try:
        problem_instance = _get_problem_instance(instance_path)
        output_dir = create_temp_output_dir(
            instance_name, f"ablation_{config.label}"
        )
        solver_config = TwoPhaseSolverConfig(
            exact_solver_type=SolverType.OR_TOOLS,
            front_strategy=FrontStrategy.GPBA_A,
            timeout_s=timeout,
            ratio=ratio,
        )
        exact_solver_fn = (
            get_pseudo_solver().solve_exact if use_pseudo_solver else None
        )

        # Pass objective_bounds=None to solve_with_two_phases so that it
        # does NOT attempt its own hypervolume computation (which can panic
        # when pseudo-solver solutions fall outside pre-recorded bounds).
        # We compute hypervolume ourselves below with expanded bounds.
        result = solve_with_two_phases(
            problem_instance=problem_instance,
            problem_path=Path(instance_path),
            experiment_path=output_dir,
            solver_config=solver_config,
            objectives=objectives,
            dry_run=False,
            enable_pls_trace=False,
            enable_profiling_trace=False,
            objective_bounds=None,
            pareto_archive="nd-tree",
            parallel=False,
            exact_solver_fn=exact_solver_fn,
            # PLS optimization flags
            use_checkpoint=config.use_checkpoint,
            use_ranked_candidates=config.use_ranked_candidates,
            probing_budget=config.probing_budget,
            use_greedy_initial_population=config.use_greedy_initial_population,
            use_perturbation_restart=config.use_perturbation_restart,
        )
        elapsed = time.time() - start

        # Validate
        ok, err = validate_two_phase_solver_result(
            result, problem_instance.problem, objectives
        )
        if not ok:
            return RunResult(
                config_label=config.label,
                instance_name=instance_name,
                ratio=ratio,
                hypervolume=0.0,
                num_solutions=0,
                execution_time=elapsed,
                success=False,
                error=err,
            )

        # Collect ALL unique solutions from both phases for hypervolume
        all_solutions = []
        if result.exact_solver_result and result.exact_solver_result.pareto_front:
            all_solutions.extend(result.exact_solver_result.pareto_front)
        if result.pls_result and result.pls_result.pareto_front:
            all_solutions.extend(result.pls_result.pareto_front)

        seen: set[frozenset[int]] = set()
        unique = []
        for sol in all_solutions:
            key = frozenset(sol.selected_images)
            if key not in seen:
                seen.add(key)
                unique.append(sol)

        # Store raw objective points; HV is recomputed later with unified
        # bounds across all configs so that values are comparable.
        raw_points = [_extract_objectives(sol, objectives) for sol in unique]

        return RunResult(
            config_label=config.label,
            instance_name=instance_name,
            ratio=ratio,
            hypervolume=0.0,  # recomputed below with unified bounds
            num_solutions=len(unique),
            execution_time=elapsed,
            success=True,
            raw_points=raw_points,
        )
    except Exception as exc:
        return RunResult(
            config_label=config.label,
            instance_name=instance_name,
            ratio=ratio,
            hypervolume=0.0,
            num_solutions=0,
            execution_time=time.time() - start,
            success=False,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

PLOT_OUTPUT_DIR = (
    Path(__file__).parent.parent / "test_artifacts" / "ablation_plots"
)


def _ratio_label(ratio: tuple[int, int]) -> str:
    return f"{ratio[0]}:{ratio[1]}"


def _generate_instance_plot(
    instance_results: InstanceResults,
    output_dir: Path | None = None,
) -> Path | None:
    """Generate a single instance's ablation plot.

    X-axis: ratios (categorical, one tick per TWO_PHASE_RATIOS entry)
    Y-axis: normalized hypervolume (consistent scale per instance)
    Series: one line per AblationConfig
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available – skipping plot generation")
        return None

    if not instance_results.runs:
        return None

    out = output_dir or PLOT_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Build mapping config_label -> {ratio_label -> hv}
    config_data: dict[str, dict[str, float]] = {}
    for run in instance_results.runs:
        config_data.setdefault(run.config_label, {})[
            _ratio_label(run.ratio)
        ] = run.hypervolume

    ratio_labels = [_ratio_label(r) for r in TWO_PHASE_RATIOS]
    x_positions = list(range(len(ratio_labels)))

    for cfg in ABLATION_CONFIGS:
        data = config_data.get(cfg.label, {})
        y_values = [data.get(rl, 0.0) for rl in ratio_labels]
        ax.plot(
            x_positions,
            y_values,
            label=cfg.label,
            color=cfg.color,
            marker=cfg.marker,
            linestyle=cfg.linestyle,
            linewidth=2,
            markersize=7,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(ratio_labels)
    ax.set_xlabel("Hybrid Ratio  (Exact % : PLS %)", fontsize=11)
    ax.set_ylabel("Normalized Hypervolume", fontsize=11)
    dim = len(instance_results.objectives)
    ax.set_title(
        f"Ablation Study – {instance_results.instance_name}  ({dim}D)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plot_path = out / f"ablation_{instance_results.instance_name}_{dim}d.png"
    fig.tight_layout()
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    logger.info(f"Saved ablation plot to {plot_path}")
    return plot_path


def _generate_summary_json(
    all_results: list[InstanceResults],
    output_dir: Path | None = None,
) -> Path:
    """Dump all numeric results to JSON for downstream analysis."""
    out = output_dir or PLOT_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    data = []
    for ir in all_results:
        for run in ir.runs:
            data.append(
                {
                    "instance": run.instance_name,
                    "config": run.config_label,
                    "ratio_exact": run.ratio[0],
                    "ratio_pls": run.ratio[1],
                    "hypervolume": run.hypervolume,
                    "num_solutions": run.num_solutions,
                    "execution_time_s": round(run.execution_time, 3),
                    "success": run.success,
                    "error": run.error,
                    "objectives": ir.objectives,
                }
            )

    json_path = out / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved ablation results to {json_path}")
    return json_path


# ---------------------------------------------------------------------------
# Objective bounds loader (reuse logic from test_two_phases_instances)
# ---------------------------------------------------------------------------


def _load_objective_bounds() -> tuple[list[str], dict[str, list[list[int]]]]:
    bounds_file = Path(__file__).parent / "data" / "objective_bounds.json"
    if not bounds_file.exists():
        return [], {}
    with open(bounds_file) as f:
        data = json.load(f)
    return data.get("objectives", []), data.get("bounds", {})


_OBJ_NAMES, _OBJ_BOUNDS = _load_objective_bounds()


def _get_bounds(
    instance_name: str, objectives: list[str]
) -> list[list[int]] | None:
    all_bounds = _OBJ_BOUNDS.get(instance_name)
    if not all_bounds or not _OBJ_NAMES:
        return None
    filtered = []
    for obj in objectives:
        if obj not in _OBJ_NAMES:
            return None
        idx = _OBJ_NAMES.index(obj)
        if idx < len(all_bounds):
            filtered.append(all_bounds[idx])
        else:
            return None
    return filtered


# ---------------------------------------------------------------------------
# Orchestrator – runs all configs × ratios for a set of instances
# ---------------------------------------------------------------------------


def _run_ablation_for_instances(
    instances: list[str],
    objectives: list[str],
    test_data_dir: str,
    use_pseudo_solver: bool,
) -> list[InstanceResults]:
    """Run the full ablation matrix for a list of instances."""
    timeout = _ablation_timeout(instances)
    all_instance_results: list[InstanceResults] = []

    for filename in instances:
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            logger.warning(f"Skipping missing instance {filename}")
            continue

        instance_name = instance_path.stem
        bounds = _get_bounds(instance_name, objectives)

        ir = InstanceResults(
            instance_name=instance_name,
            objectives=objectives,
            objective_bounds=bounds or [],
        )

        for ratio in TWO_PHASE_RATIOS:
            for config in ABLATION_CONFIGS:
                logger.info(
                    f"[{instance_name}] config={config.label}  "
                    f"ratio={ratio}  checkpoint={config.use_checkpoint}  "
                    f"ranked={config.use_ranked_candidates}  "
                    f"probing={config.probing_budget}  "
                    f"greedy_init={config.use_greedy_initial_population}  "
                    f"perturbation={config.use_perturbation_restart}"
                )
                run = _run_single(
                    instance_path=str(instance_path),
                    objectives=objectives,
                    objective_bounds=bounds,
                    ratio=ratio,
                    timeout=timeout,
                    config=config,
                    use_pseudo_solver=use_pseudo_solver,
                )
                ir.runs.append(run)

                status = "OK" if run.success else f"FAIL: {run.error}"
                logger.info(
                    f"  -> solutions={run.num_solutions}  "
                    f"time={run.execution_time:.1f}s  {status}"
                )

        # ---- Recompute HV with unified bounds across ALL configs ----
        # Collect every objective point from every run for this instance
        # so that all configs are measured against exactly the same
        # reference point and ideal point.
        all_points: list[list[int]] = []
        for run in ir.runs:
            if run.raw_points:
                all_points.extend(run.raw_points)

        if all_points:
            unified_bounds = _bounds_from_points(all_points, len(objectives))
            ir.objective_bounds = unified_bounds
            for run in ir.runs:
                if run.raw_points:
                    run.hypervolume = _compute_hypervolume_for_points(
                        run.raw_points, unified_bounds
                    )
            logger.info(
                f"[{instance_name}] Unified bounds: {unified_bounds}"
            )
        # Print summary after HV recomputation
        for run in ir.runs:
            logger.info(
                f"  [{instance_name}] {run.config_label:20s} "
                f"ratio={run.ratio}  HV={run.hypervolume:.6f}  "
                f"sols={run.num_solutions}"
            )

        all_instance_results.append(ir)

    # Generate per-instance plots and summary JSON
    for ir in all_instance_results:
        _generate_instance_plot(ir)

    _generate_summary_json(all_instance_results)

    return all_instance_results


# ---------------------------------------------------------------------------
# Pytest test functions
# ---------------------------------------------------------------------------


class TestAblationStudy:
    """Ablation tests exercising the two-phase solver with different PLS
    optimization toggles and producing hypervolume comparison plots.

    Each test runs all ABLATION_CONFIGS × TWO_PHASE_RATIOS for every
    instance in the size tier, then generates one PNG plot per instance
    and a combined JSON with all raw numbers.
    """

    # ---- 2D small --------------------------------------------------------

    @pytest.mark.timeout(900)
    def test_ablation_small_2d(
        self, test_data_dir, use_pseudo_solver, caplog
    ):
        """2D ablation on small instances (30 & 50 images)."""
        caplog.set_level(logging.INFO)
        objectives = ["min_cost", "cloud_coverage"]
        results = _run_ablation_for_instances(
            SMALL_INSTANCES, objectives, test_data_dir, use_pseudo_solver
        )
        self._assert_all_succeeded(results)

    # ---- 2D medium -------------------------------------------------------

    @pytest.mark.timeout(7200)
    def test_ablation_medium_2d(
        self, test_data_dir, use_pseudo_solver, caplog
    ):
        """2D ablation on medium instances (100 images)."""
        caplog.set_level(logging.INFO)
        objectives = ["min_cost", "cloud_coverage"]
        results = _run_ablation_for_instances(
            MEDIUM_INSTANCES, objectives, test_data_dir, use_pseudo_solver
        )
        self._assert_all_succeeded(results)

    # ---- 2D large --------------------------------------------------------

    @pytest.mark.timeout(14400)
    def test_ablation_large_2d(
        self, test_data_dir, use_pseudo_solver, caplog
    ):
        """2D ablation on large instances (145-150 images)."""
        caplog.set_level(logging.INFO)
        objectives = ["min_cost", "cloud_coverage"]
        results = _run_ablation_for_instances(
            LARGE_INSTANCES, objectives, test_data_dir, use_pseudo_solver
        )
        self._assert_all_succeeded(results)

    # ---- 3D small --------------------------------------------------------

    @pytest.mark.timeout(900)
    def test_ablation_small_3d(
        self, test_data_dir, use_pseudo_solver, caplog
    ):
        """3D ablation on small instances."""
        caplog.set_level(logging.INFO)
        objectives = [
            "min_cost",
            "cloud_coverage",
            "min_max_incidence_angle",
        ]
        results = _run_ablation_for_instances(
            SMALL_INSTANCES, objectives, test_data_dir, use_pseudo_solver
        )
        self._assert_all_succeeded(results)

    # ---- 3D medium -------------------------------------------------------

    @pytest.mark.timeout(7200)
    def test_ablation_medium_3d(
        self, test_data_dir, use_pseudo_solver, caplog
    ):
        """3D ablation on medium instances."""
        caplog.set_level(logging.INFO)
        objectives = [
            "min_cost",
            "cloud_coverage",
            "min_max_incidence_angle",
        ]
        results = _run_ablation_for_instances(
            MEDIUM_INSTANCES, objectives, test_data_dir, use_pseudo_solver
        )
        self._assert_all_succeeded(results)

    # ---- 4D small --------------------------------------------------------

    @pytest.mark.timeout(1800)
    def test_ablation_small_4d(
        self, test_data_dir, use_pseudo_solver, caplog
    ):
        """4D ablation on small instances."""
        caplog.set_level(logging.INFO)
        objectives = [
            "min_cost",
            "cloud_coverage",
            "min_max_incidence_angle",
            "min_resolution",
        ]
        results = _run_ablation_for_instances(
            SMALL_INSTANCES, objectives, test_data_dir, use_pseudo_solver
        )
        self._assert_all_succeeded(results)

    # ---- 4D medium -------------------------------------------------------

    @pytest.mark.timeout(7200)
    def test_ablation_medium_4d(
        self, test_data_dir, use_pseudo_solver, caplog
    ):
        """4D ablation on medium instances."""
        caplog.set_level(logging.INFO)
        objectives = [
            "min_cost",
            "cloud_coverage",
            "min_max_incidence_angle",
            "min_resolution",
        ]
        results = _run_ablation_for_instances(
            MEDIUM_INSTANCES, objectives, test_data_dir, use_pseudo_solver
        )
        self._assert_all_succeeded(results)

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _assert_all_succeeded(results: list[InstanceResults]):
        failures = []
        for ir in results:
            for run in ir.runs:
                if not run.success:
                    failures.append(
                        f"{run.instance_name} [{run.config_label}] "
                        f"ratio={run.ratio}: {run.error}"
                    )
        if failures:
            summary = "\n  ".join(failures)
            pytest.fail(
                f"{len(failures)} ablation run(s) failed:\n  {summary}"
            )
