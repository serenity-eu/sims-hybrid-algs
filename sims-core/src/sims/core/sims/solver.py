import logging
from collections.abc import Callable
from pathlib import Path
from typing import Optional

try:
    import sims_problem
except ImportError:
    raise ImportError(
        "sims_problem module not found. Please ensure the sims-problem package is installed."
    )

from .problem import ProblemInstance
from .solver_config import FrontStrategy, SolverType, TwoPhaseSolverConfig
from .solver_result import Solution, SolverResult, TwoPhaseSolverResult
from .solvers import gurobi, ortools, pareto_local_search, python_milp

log = logging.getLogger(Path(__file__).stem)


def _compute_hypervolume_for_result(
    result: SolverResult,
    objectives: list[str],
    objective_bounds: list[list[int]],
) -> float:
    """
    Compute hypervolume for a SolverResult using sims_problem.compute_hypervolume.
    
    Args:
        result: The solver result containing the Pareto front
        objectives: List of objective names (e.g., ["min_cost", "cloud_coverage"])
        objective_bounds: Objective bounds as [[min, max], ...] for each objective.
    
    Returns:
        The computed hypervolume value
    """
    if not result.pareto_front:
        log.info("_compute_hypervolume_for_result: empty pareto_front, returning 0.0")
        return 0.0
    
    # Extract raw objective values as points
    points = [
        _extract_objectives(sol, objectives)
        for sol in result.pareto_front
    ]
    
    log.info(f"_compute_hypervolume_for_result: {len(points)} points, first 3: {points[:3]}")
    
    try:
        hv = sims_problem.compute_hypervolume(
            data=points,
            objective_bounds=objective_bounds,
            normalized=True,
        )
        log.info(f"_compute_hypervolume_for_result: computed hypervolume = {hv}")
        return hv
    except Exception as e:
        log.warning(f"Failed to compute hypervolume: {e}")
        return 0.0


def _extract_objectives(sol: Solution, objectives: list[str]) -> list[int]:
    """Extract objective values from a Solution in the order specified by objectives."""
    values = []
    for obj_name in objectives:
        match obj_name:
            case "min_cost":
                values.append(sol.cost)
            case "cloud_coverage":
                values.append(sol.cloudy_area)
            case "min_max_incidence_angle":
                values.append(sol.max_incidence_angle or 0)
            case "min_resolution":
                values.append(sol.min_resolutions_sum or 0)
    return values


def solve(
    solver_type: SolverType,
    problem_instance: ProblemInstance,
    problem_path: Path,
    timeout_s: int,
    output_path: Path,
    objectives: list[str],
    front_strategy: FrontStrategy = FrontStrategy.NON_APLICABLE,
    initial_population: list[Solution] | None = None,
    enable_trace: bool = False,
    enable_profiling_trace: bool = False,
    objective_bounds: list[list[int]] | None = None,
    include_dominated: bool = False,
    max_solutions_count: int | None = None,
    pareto_archive: str = "nd-tree",
    parallel: bool = False,
    num_parallel_threads: int = 0,
) -> SolverResult:
    match solver_type:
        case SolverType.OR_TOOLS:
            result = ortools.solve(
                problem_instance, problem_path, timeout_s, output_path, front_strategy, objectives, enable_trace, include_dominated, max_solutions_count
            )
        case SolverType.GUROBI:
            result = gurobi.solve(
                problem_instance, problem_path, timeout_s, output_path, front_strategy, objectives, enable_trace, include_dominated, max_solutions_count
            )
        case SolverType.PYTHON_MILP:
            result = python_milp.solve(
                problem_instance, problem_path, timeout_s, output_path, front_strategy, objectives, enable_trace, include_dominated, max_solutions_count
            )
        case SolverType.PLS:
            result = pareto_local_search.solve(
                problem_instance,
                timeout_s,
                objectives,
                initial_population,
                enable_trace=enable_trace,
                enable_profiling_trace=enable_profiling_trace,
                objective_bounds=objective_bounds,
                include_dominated=include_dominated,
                pareto_archive=pareto_archive,
                parallel=parallel,
                num_parallel_threads=num_parallel_threads,
            )
        case _:
            raise ValueError(f"Solver type {solver_type} is not supported")
    
    # Compute hypervolume for the result if bounds are provided
    if objective_bounds is not None:
        log.info(f"Computing hypervolume with bounds: {objective_bounds}, objectives: {objectives}, pareto_front size: {len(result.pareto_front)}")
        result.hypervolume = _compute_hypervolume_for_result(result, objectives, objective_bounds)
        log.info(f"Computed hypervolume: {result.hypervolume}")
    else:
        log.info("No objective_bounds provided, skipping hypervolume computation")
    return result


def _split_time_by_ratio(total_time: int, ratio: tuple[int, int]) -> tuple[int, int]:
    return int(total_time * ratio[0] / 100), int(total_time * ratio[1] / 100)


def solve_with_two_phases(
    problem_instance: ProblemInstance,
    problem_path: Path,
    experiment_path: Path,
    solver_config: TwoPhaseSolverConfig,
    objectives: list[str],
    dry_run: bool = False,
    enable_pls_trace: bool = False,
    enable_profiling_trace: bool = False,
    objective_bounds: list[list[int]] | None = None,
    include_dominated: bool = False,
    max_solutions_count: int | None = None,
    pareto_archive: str = "nd-tree",
    parallel: bool = False,
    num_parallel_threads: int = 0,
    exact_solver_fn: Optional[Callable[..., SolverResult]] = None,
) -> TwoPhaseSolverResult:
    """
    Run the two-phase solver (exact + PLS).

    Args:
        exact_solver_fn: Optional override for the exact phase.  When provided it
            is called instead of the configured exact solver (OR-Tools / Gurobi).
            Signature: ``fn(problem_instance, problem_path, timeout_s, objectives) -> SolverResult``.
            Use this to inject a pseudo-solver for testing without duplicating
            two-phase orchestration logic.
    """
    exact_solver_result = None
    pls_result = None
    timeout_s = solver_config.timeout_s
    ratio = solver_config.ratio
    exact_solver_type = SolverType(solver_config.exact_solver_type)
    front_strategy = solver_config.front_strategy

    summary_path = experiment_path / f"{problem_instance.name.rsplit('_', maxsplit=1)[0]}.csv"

    exact_solver_time, pls_time = _split_time_by_ratio(timeout_s, ratio)

    if exact_solver_time != 0:
        log.info(
            f"[{problem_instance.name}] - running {repr(exact_solver_type)} for {exact_solver_time} seconds..."
        )

        if not dry_run:
            if exact_solver_fn is not None:
                exact_solver_result = exact_solver_fn(
                    problem_instance=problem_instance,
                    problem_path=problem_path,
                    timeout_s=exact_solver_time,
                    objectives=objectives,
                )
            else:
                exact_solver_result = solve(
                    exact_solver_type,
                    problem_instance,
                    problem_path,
                    exact_solver_time,
                    summary_path,
                    objectives,
                    front_strategy,
                    enable_trace=enable_pls_trace,
                    include_dominated=include_dominated,
                    max_solutions_count=max_solutions_count,
                )

        log.info(
            f"[{problem_instance.name}] - running {repr(exact_solver_type)} for {exact_solver_time} seconds...Done"
        )

    if pls_time != 0:
        log.info(
            f"[{problem_instance.name}] - running Pareto Local Search for {pls_time} seconds..."
        )

        if not dry_run:
            # Convert initial population from first phase results if available
            initial_population = None
            if exact_solver_result is not None and exact_solver_result.pareto_front:
                # Use solutions from the first phase as initial population for PLS
                initial_population = exact_solver_result.pareto_front
                log.info(f"[{problem_instance.name}] - seeding PLS with {len(initial_population)} solutions from MILP phase")
            
            pls_result = solve(
                SolverType.PLS,
                problem_instance,
                problem_path,
                pls_time,
                summary_path,
                objectives,
                initial_population=initial_population,
                enable_trace=enable_pls_trace,
                enable_profiling_trace=enable_profiling_trace,
                objective_bounds=objective_bounds,
                include_dominated=include_dominated,
                pareto_archive=pareto_archive,
                parallel=parallel,
                num_parallel_threads=num_parallel_threads,
            )

            log.info(f"[{problem_instance.name}][solve_with_two_phases] - PLS found {len(pls_result.pareto_front)} solutions")

        log.info(
            f"[{problem_instance.name}] - running Pareto Local Search for {pls_time} seconds...Done"
        )

    two_phase_result = TwoPhaseSolverResult.from_results_pair(exact_solver_result, pls_result, solver_config, filter_invalid=False)
    
    # Compute hypervolume for both phase results if bounds are provided
    if objective_bounds is not None:
        if two_phase_result.exact_solver_result is not None:
            two_phase_result.exact_solver_result.hypervolume = _compute_hypervolume_for_result(
                two_phase_result.exact_solver_result, objectives, objective_bounds
            )
        if two_phase_result.pls_result is not None:
            two_phase_result.pls_result.hypervolume = _compute_hypervolume_for_result(
                two_phase_result.pls_result, objectives, objective_bounds
            )
    
    return two_phase_result
