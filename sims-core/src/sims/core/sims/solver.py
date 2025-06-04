import logging
from pathlib import Path

from .problem import ProblemInstance
from .solver_config import FrontStrategy, SolverType, TwoPhaseSolverConfig
from .solver_result import SolverResult, TwoPhaseSolverResult
from .solvers import gurobi, ortools, pareto_local_search

log = logging.getLogger(Path(__file__).stem)


def solve(
    solver_type: SolverType,
    problem_instance: ProblemInstance,
    problem_path: Path,
    timeout_s: int,
    output_path: Path,
    front_strategy: FrontStrategy = FrontStrategy.NON_APLICABLE,
    initial_population_csv: Path | None = None,
) -> SolverResult:
    match solver_type:
        case SolverType.OR_TOOLS:
            return ortools.solve(
                problem_instance, problem_path, timeout_s, output_path, front_strategy
            )
        case SolverType.GUROBI:
            return gurobi.solve(
                problem_instance, problem_path, timeout_s, output_path, front_strategy
            )
        case SolverType.PLS:
            return pareto_local_search.solve(
                problem_instance,
                problem_path,
                timeout_s,
                output_path,
                initial_population_csv=initial_population_csv,
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
    dry_run: bool = False,
) -> TwoPhaseSolverResult:
    exact_solver_result = None
    pls_result = None
    timeout_s = solver_config.timeout_s
    ratio = solver_config.ratio
    exact_solver_type = SolverType(solver_config.exact_solver_type)
    front_strategy = solver_config.front_strategy

    summary_path = experiment_path / f"{problem_instance.name.rsplit('_', maxsplit=1)[0]}.csv"
    ortools_output_path = None

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
                front_strategy,
            )

        log.info(
            f"[{problem_instance.name}] - running {repr(exact_solver_type)} for {exact_solver_time} seconds...Done"
        )

        # Set ortools output path is the input for PLS
        ortools_output_path = summary_path

    if pls_time != 0:
        log.info(
            f"[{problem_instance.name}] - running Pareto Local Search for {pls_time} seconds..."
        )

        if not dry_run:
            pls_result = solve(
                SolverType.PLS,
                problem_instance,
                problem_path,
                pls_time,
                summary_path,
                initial_population_csv=ortools_output_path,
            )

        log.info(
            f"[{problem_instance.name}] - running Pareto Local Search for {pls_time} seconds...Done"
        )

    return TwoPhaseSolverResult.from_results_pair(exact_solver_result, pls_result, solver_config)
