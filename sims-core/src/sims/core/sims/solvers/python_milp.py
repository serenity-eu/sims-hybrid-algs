"""Python MILP solver wrapper for sims-core.

This module provides a wrapper around the Python sims_solvers.solve_milp_inlined() implementation.
This wrapper directly calls solve_milp_inlined() and converts results to SolverResult format.
"""

import logging
from datetime import timedelta
from pathlib import Path

from sims_solvers.Config import Config
from sims_solvers.solve import solve_milp_inlined

from ..problem import ProblemInstance
from ..solver_config import FrontStrategy, SolverType
from ..solver_result import Solution, SolverResult

log = logging.getLogger(__name__)


def solve(
    problem_instance: ProblemInstance,
    problem_path: Path,
    timeout_s: int,
    summary_path: Path,
    front_strategy: FrontStrategy,
    objectives: list[str],
    enable_trace: bool = False,
    include_dominated: bool = False,
) -> SolverResult:
    """
    Solve SIMS problem using Python MILP implementation (directly calling solve_milp_inlined).
    
    This wrapper directly calls sims_solvers.solve_milp_inlined() and converts the returned
    solution data to SolverResult format, avoiding CSV serialization/parsing.
    
    Args:
        problem_instance: The problem instance to solve
        problem_path: Path to the DZN file
        timeout_s: Timeout in seconds
        summary_path: Path where solver summary CSV will be written
        front_strategy: Pareto front generation strategy (must be GPBA_A for MILP)
        objectives: List of objective names to optimize
        enable_trace: Whether to enable optimization trace
        include_dominated: Whether to include dominated solutions
    
    Returns:
        SolverResult containing the Pareto front
    
    Raises:
        ValueError: If front_strategy is not GPBA_A
    """
    # Validate front strategy
    if front_strategy != FrontStrategy.GPBA_A:
        raise ValueError(f"Python MILP solver only supports GPBA_A strategy, got: {front_strategy}")
    
    # Validate objectives
    valid_objectives = ["min_cost", "cloud_coverage", "min_resolution", "min_max_incidence_angle"]
    for obj in objectives:
        if obj not in valid_objectives:
            raise ValueError(f"Invalid objective '{obj}'. Valid objectives: {valid_objectives}")
    
    log.info(
        f"Running Python MILP solver (inlined) with objectives: {objectives}, "
        f"timeout: {timeout_s}s, strategy: {front_strategy.value}"
    )
    
    # Create config for solve_milp_inlined (no translation needed - same names)
    config = Config(
        minizinc_data=False,
        instance_name=problem_path.stem,
        data_sets_folder=problem_path.parent,
        input_mzn=Path("dummy.mzn"),  # Not used by inlined solver
        dzn_dir=problem_path.parent,
        solver_name="gurobi",
        problem_name="sims",
        front_strategy=front_strategy.value,
        solver_timeout_sec=timeout_s,
        summary_filename=str(summary_path),
        solver_search_strategy="free",
        fzn_optimisation_level=0,
        cores=1,
        threads=8,
        objectives=objectives,
    )
    
    # Call solve_milp_inlined
    result = solve_milp_inlined(config, objectives=objectives)
    
    # Convert results to SolverResult format (use original objective names for mapping)
    pareto_front = _convert_solutions_to_pareto_front(
        result["pareto_solutions"],
        objectives,  # Use original objectives, not translated
        problem_instance
    )
    
    # Extract hypervolume from statistics if available
    hypervolume = result["statistics"].get("hypervolume", 0.0)
    
    return SolverResult(
        pareto_front=pareto_front,
        timeout_sec=timeout_s,
        execution_time_sec=result["execution_time_sec"],
        hypervolume=hypervolume,
        solver_type=SolverType.GUROBI,
        problem_instance=problem_instance,
        front_strategy=front_strategy,
    )


def _convert_solutions_to_pareto_front(
    pareto_solutions: list[dict],
    objectives: list[str],
    problem_instance: ProblemInstance,
) -> list[Solution]:
    """
    Convert solve_milp_inlined solution format to Solution objects.
    
    Args:
        pareto_solutions: List of solutions from solve_milp_inlined, each with:
            - objs: list of objective values
            - solution_values: list of selected image IDs
            - timestamp: relative time in seconds
        objectives: List of objective names in order
        problem_instance: Problem instance for validation
    
    Returns:
        List of Solution objects
    """
    solutions = []
    
    # Map objective names to Solution field names
    objective_field_map = {
        "min_cost": "cost",
        "cloud_coverage": "cloudy_area",
        "min_resolution": "min_resolutions_sum",
        "min_max_incidence_angle": "max_incidence_angle",
    }
    
    for sol_data in pareto_solutions:
        # Extract objective values
        obj_values = sol_data["objs"]
        
        # Create kwargs for Solution constructor
        solution_kwargs = {
            "selected_images": frozenset(sol_data["solution_values"]),
            "timestamp_s": timedelta(seconds=sol_data["timestamp"]),
        }
        
        # Map objectives to their corresponding fields
        for i, obj_name in enumerate(objectives):
            field_name = objective_field_map.get(obj_name)
            if field_name:
                solution_kwargs[field_name] = int(obj_values[i])
        
        # Ensure cost and cloudy_area are always set (required fields)
        if "cost" not in solution_kwargs:
            solution_kwargs["cost"] = 0
        if "cloudy_area" not in solution_kwargs:
            solution_kwargs["cloudy_area"] = 0
        
        solution = Solution(**solution_kwargs)
        solutions.append(solution)
    
    return solutions
