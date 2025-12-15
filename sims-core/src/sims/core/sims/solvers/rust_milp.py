"""Rust MILP solver wrapper for sims-core.

This module provides a wrapper around the Rust sims_problem.solve_with_milp() implementation.
This wrapper directly uses the Rust solver output without CSV parsing.
"""

import logging
from datetime import timedelta
from pathlib import Path

import sims_problem

from ..problem import ProblemInstance
from ..solver_config import FrontStrategy
from ..solver_result import SolverResult, Solution

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
    Solve SIMS problem using Rust MILP implementation (direct, no CSV parsing).
    
    This wrapper calls sims_problem.solve_with_milp() directly and converts
    the Rust Solution objects to sims-core Solution objects.
    
    Args:
        problem_instance: The problem instance to solve
        problem_path: Path to the DZN file (for metadata)
        timeout_s: Timeout in seconds
        summary_path: Path where solver summary will be written (not used by Rust solver)
        front_strategy: Pareto front generation strategy (must be GPBA_A for MILP)
        objectives: List of objective names to optimize
        enable_trace: Whether to enable optimization trace
        include_dominated: Whether to include dominated solutions
    
    Returns:
        SolverResult containing the Pareto front
    
    Raises:
        ValueError: If front_strategy is not GPBA_A or if MILP feature is not enabled
    """
    # Validate front strategy
    if front_strategy != FrontStrategy.GPBA_A:
        raise ValueError(f"Rust MILP solver only supports GPBA_A strategy, got: {front_strategy}")
    
    # Validate objectives
    valid_objectives = ["min_cost", "cloud_coverage", "min_resolution", "min_max_incidence_angle"]
    for obj in objectives:
        if obj not in valid_objectives:
            raise ValueError(f"Invalid objective '{obj}'. Valid objectives: {valid_objectives}")
    
    log.info(
        f"Running Rust MILP solver (direct) with objectives: {objectives}, "
        f"timeout: {timeout_s}s, strategy: {front_strategy.value}"
    )
    
    # Debug: Check cloud data
    log.info(f"Problem has {problem_instance.problem.num_images} images")
    log.info(f"Problem universe size: {problem_instance.problem.universe}")
    log.info(f"Number of cloud sets: {len(problem_instance.problem.clouds)}")
    
    # Check if clouds have data
    total_cloudy_elements = sum(len(cloud_set) for cloud_set in problem_instance.problem.clouds)
    log.info(f"Total cloudy elements across all images: {total_cloudy_elements}")
    
    if total_cloudy_elements == 0:
        log.warning("⚠️  NO CLOUD DATA FOUND! All cloud sets are empty!")
    else:
        # Show sample cloud data
        for i, cloud_set in enumerate(problem_instance.problem.clouds[:3]):
            log.info(f"Image {i} has {len(cloud_set)} cloudy elements: {sorted(list(cloud_set))[:10]}")

    # Create sims_problem.SimsDiscreteProblem from problem_instance
    # This is necessary because the Python wrapper needs a fresh Rust object
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

    # Call Rust MILP solver directly
    timeout_duration = timedelta(seconds=timeout_s)

    try:
        solving_result = sims_problem.solve_with_milp(
            sims_instance=sims_problem_instance,
            objectives=objectives,
            grid_points=50,
            timeout=timeout_duration,
            bypass_coefficient=False,  # Disable for full exploration
            early_exit=False,  # Disable early exit to find all solutions
            flag_array=False,  # Disable flag array for full exploration
            solver_name="coin_cbc",
        )
    except AttributeError as e:
        raise ValueError(
            "Rust MILP solver not available. "
            "Please ensure sims-problem was built with the 'milp' feature enabled."
        ) from e
    
    # Convert Rust sims_problem.Solution objects to sims-core Solution objects
    rust_solutions = solving_result.final_solutions
    log.info(f"Rust MILP found {len(rust_solutions)} Pareto optimal solutions")
    
    pareto_front = []
    for rust_sol in rust_solutions:
        # Convert Rust Solution to sims-core Solution
        # Rust Solution fields can be None, so use default of 0 if None
        core_solution = Solution(
            selected_images=frozenset(rust_sol.selected_images),
            cost=rust_sol.cost if rust_sol.cost is not None else 0,
            cloudy_area=rust_sol.cloudy_area if rust_sol.cloudy_area is not None else 0,
            timestamp_s=rust_sol.timestamp,  # Rust already returns timedelta
            max_incidence_angle=rust_sol.max_incidence_angle,  # Keep None as is
            min_resolutions_sum=rust_sol.min_resolutions_sum,  # Keep None as is
        )
        pareto_front.append(core_solution)
    
    log.info(f"Converted {len(pareto_front)} Rust solutions to sims-core format")
    
    # Calculate hypervolume (placeholder for now)
    hypervolume = 0.0
    
    # Handle trace generation if requested
    trace_data = None
    if enable_trace and pareto_front:
        try:
            log.info(f"Generating trace data from {len(rust_solutions)} Rust MILP solutions")
            
            # Calculate objective bounds dynamically
            objective_attr_map = {
                'min_cost': 'cost',
                'cloud_coverage': 'cloudy_area',
                'min_max_incidence_angle': 'max_incidence_angle',
                'min_resolution': 'min_resolutions_sum'
            }
            
            objective_values = {}
            for solution in pareto_front:
                for obj_name in objectives:
                    attr_name = objective_attr_map[obj_name]
                    attr_value = getattr(solution, attr_name, None)
                    
                    if attr_value is None:
                        attr_value = 0
                    
                    if obj_name not in objective_values:
                        objective_values[obj_name] = []
                    objective_values[obj_name].append(attr_value)
            
            objective_bounds = []
            for obj_name in objectives:
                values = objective_values[obj_name]
                min_val, max_val = int(min(values)), int(max(values))
                objective_bounds.append([min_val, max_val])
            
            reference_point = [int(bound[1] + 1) for bound in objective_bounds]
            
            log.info(f"Objective bounds: {objective_bounds}, Reference point: {reference_point}")
            
            # Use the original Rust solutions for trace generation
            trace_data = sims_problem.generate_trace(
                solutions=rust_solutions,
                objectives=objectives,
                algorithm="MILP-Rust",
                num_objectives=len(objectives),
                objective_bounds=objective_bounds,
                reference_point=reference_point,
                include_dominated=include_dominated
            )
            
            if trace_data:
                log.info(f"Successfully generated trace data: {len(trace_data)} bytes")
            
        except Exception as e:
            log.error(f"Failed to generate trace data: {e}")
            log.exception("Full traceback:")
    
    # Create SolverResult with RUST_MILP solver type
    from ..solver_config import SolverType
    
    solver_result = SolverResult(
        problem_instance=problem_instance,
        pareto_front=pareto_front,
        timeout_sec=timeout_s,
        execution_time_sec=0.0,
        hypervolume=hypervolume,
        solver_type=SolverType.RUST_MILP,
        front_strategy=front_strategy,
        trace_data=trace_data,
    )
    
    return solver_result
