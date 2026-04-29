import logging
from datetime import timedelta
from pathlib import Path
from typing import Sequence

try:
    import sims_problem
except ImportError:
    raise ImportError(
        "sims_problem module not found. Please ensure the sims-problem package is installed."
    )

from ..problem import ProblemInstance
from ..solver_result import SolverResult, Solution

log = logging.getLogger(Path(__file__).stem)


def solve(
    problem_instance: ProblemInstance,
    timeout_s: int,
    objectives: list[str],
    initial_population: Sequence[Solution] | None = None,
    initial_population_size: int = 100,
    max_iterations: int = 50000,
    neighborhood_size_min: int = 1,
    neighborhood_size_max: int = 6,
    enable_trace: bool = False,
    enable_profiling_trace: bool = False,
    objective_bounds: list[list[int]] | None = None,
    include_dominated: bool = False,
    pareto_archive: str = "nd-tree",
    is_deterministic: bool = True,
    parallel: bool = False,
    num_parallel_threads: int = 0,
    neighborhood_budget: int | None = None,
    use_checkpoint: bool = True,
    use_ranked_candidates: bool = True,
    max_k1_candidates: int = 15,
    probing_budget: int | None = None,
    use_greedy_initial_population: bool = True,
    use_perturbation_restart: bool = True,
) -> SolverResult:
    """
    Solve the SIMS problem using Pareto Local Search via sims_problem.solve_with_pls.

    Args:
        problem_instance: The SIMS problem instance to solve
        problem_path: Path to the problem file (not used in new implementation)
        timeout_s: Timeout in seconds
        output_path: Output path (not used in new implementation)
        objectives: List of objectives to optimize
        initial_population: Initial population of solutions (optional)
        initial_population_size: Size of initial population when not providing custom initial_population
        max_iterations: Maximum number of iterations for the PLS algorithm
        neighborhood_size_min: Minimum neighborhood size for local search
        neighborhood_size_max: Maximum neighborhood size for local search
        enable_trace: Whether to enable tracing for debugging/analysis
        enable_profiling_trace: Whether to enable Chrome profiling trace for performance analysis
        objective_bounds: Optional list of [min, max] bounds for each objective (for trace generation)
        include_dominated: If False, filters out dominated solutions from traces (default: False)
        pareto_archive: Pareto archive implementation ("nd-tree", "linked-list", "vector")
        is_deterministic: Whether to use deterministic mode (default: True)
        parallel: Whether to use ConcurrentPLS (multi-threaded) instead of sequential PLS (default: False)
        num_parallel_threads: Number of threads for ConcurrentPLS; 0 = auto-detect CPU count (default: 0)

    Returns:
        SolverResult: The solving result with Pareto front solutions
    """

    # Convert timeout to timedelta
    timeout = timedelta(seconds=timeout_s)

    # Convert sims-core SimsDiscreteProblem to sims-problem SimsDiscreteProblem
    sims_problem_instance = sims_problem.SimsDiscreteProblem(
        num_images=problem_instance.problem.num_images,
        universe=problem_instance.problem.universe,
        images=[list(image_set) for image_set in problem_instance.problem.images],
        costs=problem_instance.problem.costs,
        clouds=[list(cloud_set) for cloud_set in problem_instance.problem.clouds],
        areas=problem_instance.problem.areas,
        resolution=problem_instance.problem.resolution,
        incidence_angle=problem_instance.problem.incidence_angle,
        max_cloud_area=problem_instance.problem.max_cloud_area
    )

    if initial_population is not None:
        # Convert initial population to sims_problem.Solution format
        # Note: -1 sentinel values mean "not set" for optional objectives, convert to None
        initial_population_sims = [
            sims_problem.Solution.create(
                selected_images=list(sol.selected_images),
                cost=sol.cost if sol.cost != -1 else None,  # Pass None for unset objective
                cloudy_area=sol.cloudy_area if sol.cloudy_area != -1 else None,  # Pass None for unset objective
                timestamp_us=int(sol.timestamp_s.total_seconds() * 1_000_000),  # Convert to total microseconds
                max_incidence_angle=sol.max_incidence_angle if sol.max_incidence_angle != -1 else None,
                min_resolutions_sum=sol.min_resolutions_sum if sol.min_resolutions_sum != -1 else None
            ) for sol in initial_population
        ]
    else:
        initial_population_sims = None


    log.debug(f"Solving with sims_problem.solve_with_pls, timeout: {timeout_s}s, objectives: {objectives}")

    try:
        # Call the Rust-based PLS solver
        solving_result = sims_problem.solve_with_pls(
            sims_instance=sims_problem_instance,
            objectives=objectives,
            plots=False,
            plot_output_path=None,
            timeout=timeout,
            max_iterations=max_iterations,
            is_deterministic=is_deterministic,
            initial_population_size=initial_population_size,
            initial_population=initial_population_sims,
            neighborhood_size_min=neighborhood_size_min,
            neighborhood_size_max=neighborhood_size_max,
            trace=enable_trace,
            profiling_trace=enable_profiling_trace,
            objective_bounds=objective_bounds,
            include_dominated=include_dominated,
            pareto_archive=pareto_archive,
            parallel=parallel,
            num_parallel_threads=num_parallel_threads,
            neighborhood_budget=neighborhood_budget,
            use_checkpoint=use_checkpoint,
            use_ranked_candidates=use_ranked_candidates,
            max_k1_candidates=max_k1_candidates,
            probing_budget=probing_budget,
            use_greedy_initial_population=use_greedy_initial_population,
            use_perturbation_restart=use_perturbation_restart,
        )

    except Exception as e:
        log.error(f"Error calling sims_problem.solve_with_pls: {e}")
        raise e

    # Convert sims_problem.Solution to sims-core Solution
    def convert_solution(sims_problem_solution) -> Solution:
        return Solution(
            selected_images=frozenset(sims_problem_solution.selected_images),
            cost=sims_problem_solution.cost,
            cloudy_area=sims_problem_solution.cloudy_area,
            timestamp_s=sims_problem_solution.timestamp,
            max_incidence_angle=sims_problem_solution.max_incidence_angle,
            min_resolutions_sum=sims_problem_solution.min_resolutions_sum
        )

    # Convert solutions
    pareto_front = [convert_solution(sol) for sol in solving_result.final_solutions]

    log.debug(f"Found {len(pareto_front)} Pareto-optimal solutions")

    # Calculate execution time from solutions if available
    execution_time_sec = 0.0
    if pareto_front:
        # Use the maximum timestamp as an approximation of execution time
        execution_time_sec = max(sol.timestamp_s.total_seconds() for sol in pareto_front)

    from ..solver_config import SolverType

    return SolverResult(
        pareto_front=pareto_front,
        timeout_sec=timeout_s,
        execution_time_sec=execution_time_sec,
        hypervolume=0.0,
        solver_type=SolverType.PLS,
        problem_instance=problem_instance,
        front_strategy=None,
        pareto_front_snapshots=[],
        trace_data=solving_result.trace,
        profiling_trace_data=solving_result.profiling_trace_data if hasattr(solving_result, 'profiling_trace_data') else None
    )
