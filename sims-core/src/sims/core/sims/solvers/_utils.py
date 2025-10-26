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
):
    import sys
    print("DEBUG: run_sims_solver called", flush=True)
    print(f"DEBUG: enable_trace = {enable_trace}", flush=True)
    print(f"DEBUG: summary_path = {summary_path}", flush=True)
    sys.stdout.flush()
    log.critical("CRITICAL: run_sims_solver called")
    log.critical(f"CRITICAL: enable_trace = {enable_trace}")
    log.critical(f"CRITICAL: summary_path = {summary_path}")
    
    DZN_DIR = problem_path.parent

    if not DZN_DIR.exists():
        raise FileNotFoundError(f"DZN directory {DZN_DIR} does not exist.")

    if not problem_path.exists():
        raise FileNotFoundError(f"Problem file {problem_path} does not exist.")

    # Create and clean the summary directory
    summary_path.parent.mkdir(exist_ok=True, parents=True)
    
    solver_name = solver_type.value.lower()

    config = Config(
        minizinc_data=True,
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
    )

    log.debug("Running command SIMS solver.")
    log.info(f"CSV summary will be written to: {summary_path}")
    print(f"DEBUG: CSV summary will be written to: {summary_path}")
    print(f"DEBUG: Config summary_filename = {config.summary_filename}")
    log.critical("CRITICAL: About to call sims_solvers.solve_milp")
    try:
        sims_solvers.solve_milp(config)
        log.info("sims_solvers.solve_milp completed successfully")
        print("DEBUG: sims_solvers.solve_milp completed successfully")
    except Exception as e:
        log.error(f"sims_solvers.solve_milp failed: {e}")
        print(f"DEBUG: sims_solvers.solve_milp failed: {e}")
        raise e

    log.debug(f"Reading summary from {summary_path}")
    print(f"DEBUG: Checking if CSV file exists: {summary_path.exists()}", flush=True)
    log.info(f"Checking if CSV file exists: {summary_path.exists()}")
    if summary_path.exists():
        file_size = summary_path.stat().st_size
        print(f"DEBUG: CSV file found, size: {file_size} bytes", flush=True)
        log.info(f"CSV file found, size: {file_size} bytes")
        if file_size > 0:
            with open(summary_path, 'r') as f:
                first_lines = f.read(500)  # Read first 500 chars
                print(f"DEBUG: CSV file first 500 chars: {first_lines[:100]}...", flush=True)
                log.info(f"CSV file first 500 chars: {first_lines}")
        else:
            print("DEBUG: CSV file exists but is empty", flush=True)
            log.warning("CSV file exists but is empty")
    else:
        print("DEBUG: CSV file was not created by sims_solvers.solve_milp", flush=True)
        log.warning("CSV file was not created by sims_solvers.solve_milp")
    
    # First, check if summary file exists and has content
    if not summary_path.exists():
        log.warning(f"Summary file not found: {summary_path}")
        # Create a minimal SolverResult with empty pareto front
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
    
    # Parse the CSV to get the solutions
    try:
        print(f"DEBUG: Attempting to parse CSV file: {summary_path}", flush=True)
        log.info(f"Attempting to parse CSV file: {summary_path}")
        solver_result = SolverResult.from_summary_csv(summary_path, problem_instance, objectives=objectives, no_headers=True)
        print(f"DEBUG: Successfully parsed CSV, found {len(solver_result.pareto_front)} solutions", flush=True)
        log.info(f"Successfully parsed CSV, found {len(solver_result.pareto_front)} solutions")
        if solver_result.pareto_front:
            print(f"DEBUG: First solution example: cost={solver_result.pareto_front[0].cost}, cloudy_area={solver_result.pareto_front[0].cloudy_area}", flush=True)
            log.info(f"First solution example: cost={solver_result.pareto_front[0].cost}, cloudy_area={solver_result.pareto_front[0].cloudy_area}")
    except (IndexError, ValueError) as e:
        log.warning(f"Failed to parse CSV file {summary_path}: {e}")
        log.warning("Creating minimal SolverResult with empty pareto front")
        # Create a minimal SolverResult with empty pareto front
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
    print(f"DEBUG: Trace generation settings: enable_trace={enable_trace}, solutions_count={len(solver_result.pareto_front) if solver_result.pareto_front else 0}", flush=True)
    log.info(f"Trace generation settings: enable_trace={enable_trace}, solutions_count={len(solver_result.pareto_front) if solver_result.pareto_front else 0}")
    if enable_trace and solver_result.pareto_front:
        try:
            print(f"DEBUG: Starting trace generation from {len(solver_result.pareto_front)} MILP solutions", flush=True)
            log.info(f"Starting trace generation from {len(solver_result.pareto_front)} MILP solutions")
            log.debug(f"Generating trace data from {len(solver_result.pareto_front)} MILP solutions")
            
            # Convert sims-core Solution objects to sims-problem Solution objects
            sims_problem_solutions = []
            for i, solution in enumerate(solver_result.pareto_front):
                # Convert selected_images frozenset to list
                selected_images_list = list(solution.selected_images)
                
                # Convert timestamp to microseconds
                timestamp_us = int(solution.timestamp_s.total_seconds() * 1_000_000)
                
                # Ensure all values are positive (they may be negative due to minimization)
                cost_val = abs(int(solution.cost)) if solution.cost is not None else 0
                cloudy_area_val = abs(int(solution.cloudy_area)) if solution.cloudy_area is not None else 0
                max_incidence_val = abs(int(solution.max_incidence_angle or 0))
                min_res_val = abs(int(solution.min_resolutions_sum or 0))
                
                log.debug(f"Converting solution {i}: images={len(selected_images_list)}, cost={cost_val}, cloudy_area={cloudy_area_val}")
                
                # Create sims-problem Solution object with proper data types
                sims_solution = sims_problem.Solution.create(
                    selected_images=selected_images_list,
                    cost=cost_val,
                    cloudy_area=cloudy_area_val,
                    timestamp_us=timestamp_us,
                    max_incidence_angle=max_incidence_val,
                    min_resolutions_sum=min_res_val
                )
                sims_problem_solutions.append(sims_solution)
            
            log.info(f"Converted {len(sims_problem_solutions)} solutions, calling sims_problem.generate_trace()")
            log.info(f"Trace generation parameters: objectives={objectives}, algorithm='MILP', num_objectives={len(objectives)}")
            
            # Calculate objective bounds dynamically from the actual solution data
            if len(sims_problem_solutions) == 0:
                raise ValueError("No solutions available to calculate objective bounds")
            
            # Create a mapping from objective names to solution attributes
            objective_attr_map = {
                'min_cost': 'cost',
                'cloud_coverage': 'cloudy_area', 
                'min_max_incidence_angle': 'max_incidence_angle',
                'min_resolutions_sum': 'min_resolutions_sum',
                'min_resolution': 'min_resolutions_sum'  # Alias for singular form
            }
            
            # Extract objective values dynamically based on objectives array
            objective_values = {}
            
            log.info("Extracting objective values from solutions for bounds calculation")
            for solution in solver_result.pareto_front:
                for obj_name in objectives:
                    if obj_name not in objective_attr_map:
                        raise ValueError(f"Unknown objective: {obj_name}")
                    
                    attr_name = objective_attr_map[obj_name]
                    attr_value = getattr(solution, attr_name, None)
                    
                    # Validate that the attribute exists and has a valid value
                    assert attr_value is not None, f"Solution has None {attr_name}: {solution}"
                    
                    # Different validation based on objective type
                    if obj_name == 'min_cost':
                        assert attr_value > 0, f"Cost must be > 0, got {attr_value} in solution: {solution}"
                    elif obj_name in ['cloud_coverage', 'min_max_incidence_angle']:
                        assert attr_value >= 0, f"{obj_name} must be >= 0, got {attr_value} in solution: {solution}"
                    elif obj_name in ['min_resolutions_sum', 'min_resolution']:
                        assert attr_value >= 0, f"Min resolution must be >= 0, got {attr_value} in solution: {solution}"
                    
                    # Collect values for this objective
                    if obj_name not in objective_values:
                        objective_values[obj_name] = []
                    objective_values[obj_name].append(attr_value)
            
            # Calculate bounds for each objective dynamically
            objective_bounds = []
            for obj_name in objectives:
                values = objective_values[obj_name]
                min_val, max_val = int(min(values)), int(max(values))
                
                objective_bounds.append([min_val, max_val])
                log.info(f"Calculated bounds for {obj_name}: [{min_val}, {max_val}]")
            
            # Compute reference_point as max + 1 for each objective (as done in Rust code)
            reference_point = [int(bound[1] + 1) for bound in objective_bounds]
            
            log.info(f"Calculated objective_bounds from data: {objective_bounds}")
            log.info(f"Computed reference_point from bounds: {reference_point}")
            log.debug(f"Based on {len(solver_result.pareto_front)} solutions")
            for i, obj_name in enumerate(objectives):
                values = objective_values[obj_name]
                log.debug(f"{obj_name} range: [{min(values)}, {max(values)}]")
            
            # Generate the trace
            print(f"DEBUG: Calling sims_problem.generate_trace with {len(sims_problem_solutions)} solutions", flush=True)
            print(f"DEBUG: Objective bounds: {objective_bounds}", flush=True)
            print(f"DEBUG: Reference point: {reference_point}", flush=True)
            log.info(f"Calling sims_problem.generate_trace with {len(sims_problem_solutions)} solutions")
            log.info(f"Objective bounds: {objective_bounds}")
            log.info(f"Reference point: {reference_point}")
            
            # Use original objective names for trace generation to match PLS
            trace_data = sims_problem.generate_trace(
                solutions=sims_problem_solutions,
                objectives=objectives,  # Use original objectives, no mapping needed
                algorithm="MILP",
                num_objectives=len(objectives),
                objective_bounds=objective_bounds,
                reference_point=reference_point,
                include_dominated=include_dominated
            )
            
            print(f"DEBUG: sims_problem.generate_trace returned: {type(trace_data)}", flush=True)
            if trace_data:
                print(f"DEBUG: Successfully generated trace data: {len(trace_data)} bytes", flush=True)
                log.info(f"Successfully generated trace data: {len(trace_data)} bytes")
            else:
                print("DEBUG: Trace generation returned empty/None data", flush=True)
                log.warning("Trace generation returned empty/None data")
            log.debug(f"Generated trace data: {len(trace_data)} bytes")
            
        except Exception as e:
            print(f"DEBUG: Failed to generate trace data: {e}", flush=True)
            print(f"DEBUG: Exception type: {type(e)}", flush=True)
            log.error(f"Failed to generate trace data: {e}")
            log.exception("Full traceback for trace generation failure:")
            # Continue without trace data rather than failing
    else:
        if not enable_trace:
            log.info("Trace generation disabled (enable_trace=False)")
            log.debug("Trace generation disabled")
        else:
            log.warning(f"No solutions found for trace generation. Enable_trace={enable_trace}, Solutions count: {len(solver_result.pareto_front) if solver_result.pareto_front else 0}")
            log.debug("No solutions found, skipping trace generation")
    
    # Update the solver result with trace data
    log.info(f"Final result: {len(solver_result.pareto_front)} solutions, trace_data: {len(trace_data) if trace_data else 0} bytes")
    solver_result.trace_data = trace_data
    return solver_result
