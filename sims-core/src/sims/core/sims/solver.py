import logging
from pathlib import Path

from .problem import ProblemInstance
from .solver_config import FrontStrategy, SolverType, TwoPhaseSolverConfig
from .solver_result import Solution, SolverResult, TwoPhaseSolverResult
from .solvers import gurobi, ortools, pareto_local_search

log = logging.getLogger(Path(__file__).stem)


def solve(
    solver_type: SolverType,
    problem_instance: ProblemInstance,
    problem_path: Path,
    timeout_s: int,
    output_path: Path,
    objectives: list[str],
    front_strategy: FrontStrategy = FrontStrategy.NON_APLICABLE,
    initial_population: list[Solution] | None = None,
) -> SolverResult:
    match solver_type:
        case SolverType.OR_TOOLS:
            return ortools.solve(
                problem_instance, problem_path, timeout_s, output_path, front_strategy, objectives
            )
        case SolverType.GUROBI:
            return gurobi.solve(
                problem_instance, problem_path, timeout_s, output_path, front_strategy, objectives
            )
        case SolverType.PLS:
            return pareto_local_search.solve(
                problem_instance,
                timeout_s,
                objectives,
                initial_population,
            )
        case _:
            raise ValueError(f"Solver type {solver_type} is not supported")


def _split_time_by_ratio(total_time: int, ratio: tuple[int, int]) -> tuple[int, int]:
    return int(total_time * ratio[0] / 100), int(total_time * ratio[1] / 100)


def solve_with_two_phases(
    problem_instance: ProblemInstance,
    problem_path: Path,
    experiment_path: Path,
    solver_config: TwoPhaseSolverConfig,
    objectives: list[str],
    dry_run: bool = False,
) -> TwoPhaseSolverResult:
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
            exact_solver_result = solve(
                exact_solver_type,
                problem_instance,
                problem_path,
                exact_solver_time,
                summary_path,
                objectives,
                front_strategy,
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
            )

        log.info(
            f"[{problem_instance.name}] - running Pareto Local Search for {pls_time} seconds...Done"
        )

    return TwoPhaseSolverResult.from_results_pair(exact_solver_result, pls_result, solver_config)
