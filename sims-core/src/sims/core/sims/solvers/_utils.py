import logging
import subprocess
from pathlib import Path
from subprocess import PIPE, STDOUT, CompletedProcess, Popen

import sims_problem
import sims_solvers
from sims_solvers import Config, MZN_MODEL_PATH

from ..problem import ProblemInstance
from ..solver_config import FrontStrategy, SolverType
from ..solver_result import SolverResult

log = logging.getLogger(__name__)


def run_command(cmd, log: logging.Logger, realtime_output: bool = False):
    if realtime_output:
        process = Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True)
        with process.stdout as pipe:
            for line in iter(pipe.readline, ""):
                log.debug(line.strip())
            log.debug("Closing command's stdout pipe.")
        returncode = process.wait()
        stderr = process.stderr.read()
        completed_process = CompletedProcess(args=cmd, returncode=returncode, stderr=stderr)
    else:
        completed_process = subprocess.run(cmd, capture_output=True, text=True)

    completed_process.check_returncode()


def run_sims_solver(
    problem_instance: ProblemInstance,
    problem_path: Path,
    timeout_s: int,
    summary_path: Path,
    solver_type: SolverType,
    front_strategy: FrontStrategy,
    objectives: list[str],
    enable_trace: bool = False,
    include_dominated: bool = False,
    max_solutions_count: int | None = None,
):
    DZN_DIR = problem_path.parent

    if not DZN_DIR.exists():
        raise FileNotFoundError(f"DZN directory {DZN_DIR} does not exist.")

    if not problem_path.exists():
        raise FileNotFoundError(f"Problem file {problem_path} does not exist.")

    # Create and clean the summary directory
    summary_path.parent.mkdir(exist_ok=True, parents=True)

    solver_name = solver_type.value.lower()

    config = Config(
        minizinc_data=False,
        instance_name=problem_path.stem,
        data_sets_folder=DZN_DIR,
        input_mzn=MZN_MODEL_PATH,
        dzn_dir=DZN_DIR,
        problem_name="sims",
        solver_name=str(solver_type),
        front_strategy=str(front_strategy),
        solver_timeout_sec=timeout_s,
        summary_filename=str(summary_path),
        solver_search_strategy="free",
        fzn_optimisation_level=1,
        cores=4,
        threads=8,
        objectives=objectives,
        max_solutions_count=max_solutions_count,
    )

    log.debug("Running command SIMS solver.")
    log.info(f"CSV summary will be written to: {summary_path}")
    try:
        sims_solvers.solve_milp(config)
    except Exception as e:
        log.error(f"sims_solvers.solve_milp failed: {e}")
        raise e

    log.debug(f"Reading summary from {summary_path}")
    if not summary_path.exists():
        log.warning(f"Summary file not found: {summary_path}")
        return SolverResult(
            problem_instance=problem_instance,
            pareto_front=[],
            timeout_sec=timeout_s,
            execution_time_sec=0.0,
            hypervolume=0.0,
            solver_type=solver_type,
            front_strategy=front_strategy,
            trace_data=None,
        )

    try:
        solver_result = SolverResult.from_summary_csv(summary_path, problem_instance, objectives=objectives, no_headers=True)
    except (IndexError, ValueError) as e:
        log.warning(f"Failed to parse CSV file {summary_path}: {e}")
        return SolverResult(
            problem_instance=problem_instance,
            pareto_front=[],
            timeout_sec=timeout_s,
            execution_time_sec=0.0,
            hypervolume=0.0,
            solver_type=solver_type,
            front_strategy=front_strategy,
            trace_data=None,
        )

    # Generate trace data from the parsed solutions
    trace_data = None
    if enable_trace and solver_result.pareto_front:
        try:
            log.debug(f"Generating trace data from {len(solver_result.pareto_front)} MILP solutions")

            sims_problem_solutions = []
            for i, solution in enumerate(solver_result.pareto_front):
                selected_images_list = list(solution.selected_images)
                timestamp_us = int(solution.timestamp_s.total_seconds() * 1_000_000)
                cost_val = abs(int(solution.cost)) if solution.cost is not None else 0
                cloudy_area_val = abs(int(solution.cloudy_area)) if solution.cloudy_area is not None else 0
                max_incidence_val = abs(int(solution.max_incidence_angle or 0))
                min_res_val = abs(int(solution.min_resolutions_sum or 0))

                sims_solution = sims_problem.Solution.create(
                    selected_images=selected_images_list,
                    cost=cost_val,
                    cloudy_area=cloudy_area_val,
                    timestamp_us=timestamp_us,
                    max_incidence_angle=max_incidence_val,
                    min_resolutions_sum=min_res_val
                )
                sims_problem_solutions.append(sims_solution)

            objective_attr_map = {
                'min_cost': 'cost',
                'cloud_coverage': 'cloudy_area',
                'min_max_incidence_angle': 'max_incidence_angle',
                'min_resolutions_sum': 'min_resolutions_sum',
                'min_resolution': 'min_resolutions_sum'
            }

            objective_values = {}
            for solution in solver_result.pareto_front:
                for obj_name in objectives:
                    if obj_name not in objective_attr_map:
                        raise ValueError(f"Unknown objective: {obj_name}")
                    attr_name = objective_attr_map[obj_name]
                    attr_value = getattr(solution, attr_name, None)
                    assert attr_value is not None, f"Solution has None {attr_name}: {solution}"
                    if obj_name not in objective_values:
                        objective_values[obj_name] = []
                    objective_values[obj_name].append(attr_value)

            objective_bounds = []
            for obj_name in objectives:
                values = objective_values[obj_name]
                min_val, max_val = int(min(values)), int(max(values))
                objective_bounds.append([min_val, max_val])

            reference_point = [int(bound[1] + 1) for bound in objective_bounds]

            trace_data = sims_problem.generate_trace(
                solutions=sims_problem_solutions,
                objectives=objectives,
                algorithm="MILP",
                num_objectives=len(objectives),
                objective_bounds=objective_bounds,
                reference_point=reference_point,
                include_dominated=include_dominated
            )
        except Exception as e:
            log.error(f"Failed to generate trace data: {e}")
            log.exception("Full traceback for trace generation failure:")
    else:
        if not enable_trace:
            log.debug("Trace generation disabled")
        else:
            log.debug("No solutions found, skipping trace generation")

    solver_result.trace_data = trace_data
    return solver_result
