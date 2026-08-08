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
from ..solver_config import SolverType
from ..solver_result import SolverResult, Solution

log = logging.getLogger(Path(__file__).stem)

# Tuned parameters matching the PseudoSeeded*Config classes in run_hv_experiments.py.
# Used when an initial population is provided (hybrid / seeded mode).
_NSGA2_SEEDED_DEFAULTS = dict(
    population_size=200,
    max_generations=500_000,
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
    use_contribution_distance=True,
)

_NSGA3_SEEDED_DEFAULTS = dict(
    target_pop_size=200,
    auto_divisions=True,
    max_generations=500_000,
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

_MOEAD_SEEDED_DEFAULTS = dict(
    population_size=200,
    max_generations=500_000,
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
    neighborhood_size=20,
    delta=0.9,
    nr=2,
    use_pbi=True,
)


def _build_sims_instance(problem_instance: ProblemInstance):
    return sims_problem.SimsDiscreteProblem(
        num_images=problem_instance.problem.num_images,
        universe=problem_instance.problem.universe,
        images=[list(image_set) for image_set in problem_instance.problem.images],
        costs=problem_instance.problem.costs,
        clouds=[list(cloud_set) for cloud_set in problem_instance.problem.clouds],
        areas=problem_instance.problem.areas,
        resolution=problem_instance.problem.resolution,
        incidence_angle=problem_instance.problem.incidence_angle,
        max_cloud_area=problem_instance.problem.max_cloud_area,
    )


def _convert_initial_population(initial_population: Sequence[Solution]) -> list:
    return [
        sims_problem.Solution.create(
            selected_images=list(sol.selected_images),
            cost=sol.cost if sol.cost != -1 else None,
            cloudy_area=sol.cloudy_area if sol.cloudy_area != -1 else None,
            timestamp_us=int(sol.timestamp_s.total_seconds() * 1_000_000),
            max_incidence_angle=sol.max_incidence_angle if sol.max_incidence_angle != -1 else None,
            min_resolutions_sum=sol.min_resolutions_sum if sol.min_resolutions_sum != -1 else None,
        )
        for sol in initial_population
    ]


def _to_solver_result(
    solving_result,
    solver_type: SolverType,
    timeout_s: int,
    problem_instance: ProblemInstance,
) -> SolverResult:
    def convert(sol) -> Solution:
        return Solution(
            selected_images=frozenset(sol.selected_images),
            cost=sol.cost,
            cloudy_area=sol.cloudy_area,
            timestamp_s=sol.timestamp,
            max_incidence_angle=sol.max_incidence_angle,
            min_resolutions_sum=sol.min_resolutions_sum,
        )

    pareto_front = [convert(s) for s in solving_result.final_solutions]
    execution_time_sec = (
        max(s.timestamp_s.total_seconds() for s in pareto_front) if pareto_front else 0.0
    )
    return SolverResult(
        pareto_front=pareto_front,
        timeout_sec=timeout_s,
        execution_time_sec=execution_time_sec,
        hypervolume=0.0,
        solver_type=solver_type,
        problem_instance=problem_instance,
        front_strategy=None,
        pareto_front_snapshots=[],
        trace_data=solving_result.trace,
        profiling_trace_data=getattr(solving_result, "profiling_trace_data", None),
    )


def solve(
    solver_type: SolverType,
    problem_instance: ProblemInstance,
    timeout_s: int,
    objectives: list[str],
    initial_population: Sequence[Solution] | None = None,
    objective_bounds: list[list[int]] | None = None,
    include_dominated: bool = False,
    seed: int = 42,
    trace: bool = False,
) -> SolverResult:
    """
    Solve the SIMS problem with an EA (NSGA-II, NSGA-III, or MOEA/D).

    When *initial_population* is provided the seeded ('pseudo-seeded') Rust variant
    is used, which is the hybrid mode (exact phase → EA phase).  Without an initial
    population the standalone tailored variant is used.
    """
    sims_instance = _build_sims_instance(problem_instance)
    timeout = timedelta(seconds=timeout_s)

    seeded_pop = (
        _convert_initial_population(initial_population)
        if initial_population
        else None
    )

    log.info(
        f"[{problem_instance.name}] EA solve: type={solver_type!r}, "
        f"timeout={timeout_s}s, objectives={objectives}, "
        f"seeded={seeded_pop is not None and len(seeded_pop)}"
    )

    try:
        if solver_type == SolverType.NSGA2:
            if seeded_pop is not None:
                result = sims_problem.solve_with_pseudo_seeded_nsga2(
                    sims_instance,
                    objectives=objectives,
                    timeout=timeout,
                    initial_population=seeded_pop,
                    seed=seed,
                    trace=trace,
                    objective_bounds=objective_bounds,
                    include_dominated=include_dominated,
                    **_NSGA2_SEEDED_DEFAULTS,
                )
            else:
                result = sims_problem.solve_with_nsga2(
                    sims_instance,
                    objectives=objectives,
                    timeout=timeout,
                    population_size=200,
                    max_generations=500_000,
                    seed=seed,
                    trace=trace,
                    objective_bounds=objective_bounds,
                    include_dominated=include_dominated,
                )

        elif solver_type == SolverType.NSGA3:
            if seeded_pop is not None:
                result = sims_problem.solve_with_pseudo_seeded_nsga3(
                    sims_instance,
                    objectives=objectives,
                    timeout=timeout,
                    initial_population=seeded_pop,
                    seed=seed,
                    trace=trace,
                    objective_bounds=objective_bounds,
                    include_dominated=include_dominated,
                    **_NSGA3_SEEDED_DEFAULTS,
                )
            else:
                result = sims_problem.solve_with_nsga3(
                    sims_instance,
                    objectives=objectives,
                    timeout=timeout,
                    target_pop_size=200,
                    auto_divisions=True,
                    max_generations=500_000,
                    seed=seed,
                    trace=trace,
                    objective_bounds=objective_bounds,
                    include_dominated=include_dominated,
                )

        elif solver_type == SolverType.MOEAD:
            if seeded_pop is not None:
                result = sims_problem.solve_with_pseudo_seeded_moead(
                    sims_instance,
                    objectives=objectives,
                    timeout=timeout,
                    initial_population=seeded_pop,
                    seed=seed,
                    trace=trace,
                    objective_bounds=objective_bounds,
                    include_dominated=include_dominated,
                    **_MOEAD_SEEDED_DEFAULTS,
                )
            else:
                result = sims_problem.solve_with_moead(
                    sims_instance,
                    objectives=objectives,
                    timeout=timeout,
                    population_size=200,
                    max_generations=500_000,
                    seed=seed,
                    trace=trace,
                    objective_bounds=objective_bounds,
                    include_dominated=include_dominated,
                )

        else:
            raise ValueError(f"Unsupported EA solver type: {solver_type}")

    except Exception as e:
        log.error(f"Error calling EA solver {solver_type!r}: {e}")
        raise

    return _to_solver_result(result, solver_type, timeout_s, problem_instance)
