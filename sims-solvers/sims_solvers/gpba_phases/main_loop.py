"""
GPBA-A Main Solving Loop

This module contains the complete iterative GPBA-A loop that explores the Pareto front
using adaptive interval management. This is the core of the GPBA-A algorithm.

Extracted from solve.py lines 700-900+
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import gurobipy as gp

from .epsilon_adjustment import adjust_parameter_ef_array
from .interval_manager import IntervalManager
from .relaxation_search import (
    save_solution_information,
    search_previous_solutions_relaxation,
)


logger = logging.getLogger(__name__)


def convert_solution_value_to_str(solution_values: List[float]) -> str:
    """Convert solution objective values to string for deduplication."""
    return str([round(float(x), 6) for x in solution_values])


def run_gpba_loop(
    model: gp.Model,
    objectives_exprs: List[Any],
    constraint_indices: List[int],
    main_obj_index: int,
    best_objective_values: List[float],
    nadir_objectives_values: List[float],
    slack_vars: List[Optional[Any]],
    ef_intervals: List[IntervalManager],
    config: Any,
    images_id: List[int],
    costs: List[float],
    cloud_covered: Optional[Dict[int, Any]],
    clouds_id_area: Dict[int, float],
    resolution_element: Optional[Dict[int, Any]],
    elements: List[int],
    current_max_incidence_angle: Optional[Any],
    add_to_pareto_front: Callable[[List[float], List[int]], None],
    statistics: Dict[str, Any],
    select_image: Dict[int, Any],
    gamma: int = 1,
    max_iterations: int = 10000,
) -> Tuple[int, bool]:
    """
    Run the complete GPBA-A main loop with adaptive interval exploration.
    
    This function iterates through the search space using epsilon-constraint method
    combined with interval management. It explores the Pareto front by adaptively
    selecting epsilon values based on solution outcomes.
    
    Args:
        model: Gurobi model with objective set and ready to solve
        objectives_exprs: List of objective expressions (in MAXIMIZATION form)
        constraint_indices: Indices of objectives to use as constraints
        main_obj_index: Index of the main objective to optimize
        best_objective_values: Ideal point (best values for each objective, in MAX form)
        nadir_objectives_values: Nadir point (worst values, in MAX form)
        slack_vars: Slack variables for augmented constraint method (or None)
        ef_intervals: List of IntervalManager objects for each constraint objective
        config: Problem configuration object with objectives attribute
        images_id: List of image indices
        costs: List of image costs
        cloud_covered: Dict of cloud coverage variables (or None)
        clouds_id_area: Dict of cloud areas
        resolution_element: Dict of resolution variables (or None)
        elements: List of element indices
        current_max_incidence_angle: Variable for max incidence angle (or None)
        add_to_pareto_front: Callback to add solutions to Pareto front
        statistics: Dict to track solving statistics
        select_image: Dict of binary decision variables for image selection
        gamma: Coverage parameter (1 = complete coverage)
        max_iterations: Safety limit on iterations
        
    Returns:
        Tuple of (iteration_count, completed) where completed=True if fully explored
    """
    # Initialize epsilon-constraint array to nadir values
    ef_array = [nadir_objectives_values[i] for i in constraint_indices]
    previous_solutions: Set[str] = set()
    previous_solution_information: List[Dict[str, Any]] = []
    
    logger.critical("GPBA-A MAIN LOOP: Starting complete iterative exploration")
    logger.critical(f"GPBA-A MAIN LOOP: Initial ef_array: {ef_array}")
    logger.critical(f"GPBA-A MAIN LOOP: Best values: {best_objective_values}")
    logger.critical(f"GPBA-A MAIN LOOP: Nadir values: {nadir_objectives_values}")
    logger.critical(f"GPBA-A MAIN LOOP: Constraint indices: {constraint_indices}")
    
    # Initialize Relative Worst Values (RWV) for search space pruning
    rwv = [best_objective_values[i] for i in constraint_indices]
    
    # Track objective values at each ef point for precise interval updates
    obj_k_at_ef_k: List[Optional[float]] = [None] * len(constraint_indices)
    
    # Create epsilon-constraint variables
    constraint_vars: List[Any] = []
    for i, constraint_idx in enumerate(constraint_indices):
        if slack_vars[i] is not None:
            constr = model.addConstr(
                objectives_exprs[constraint_idx] - slack_vars[i] == ef_array[i],
                name=f"epsilon_constraint_{constraint_idx}",
            )
        else:
            constr = model.addConstr(
                objectives_exprs[constraint_idx] == ef_array[i],
                name=f"epsilon_constraint_{constraint_idx}",
            )
        constraint_vars.append(constr)
    
    # Main GPBA-A loop with adaptive interval-based exploration
    iteration_count = 0
    completed = True
    
    while (
        ef_array[0] <= best_objective_values[constraint_indices[0]]
        and iteration_count < max_iterations
    ):
        if iteration_count % 50 == 0:
            logger.critical(
                f"GPBA-A MAIN LOOP: Iteration {iteration_count}, ef_array = {ef_array}"
            )
        
        # Initialize one_solution to empty list
        one_solution: List[float] = []
        
        # Check if this configuration was already explored (relaxation)
        previous_relaxation, previous_values = search_previous_solutions_relaxation(
            ef_array, previous_solution_information, constraint_indices
        )
        
        logger.critical(
            f"GPBA-A MAIN LOOP: ef_array={ef_array}, "
            f"previous_relaxation={previous_relaxation}, "
            f"prev_info_count={len(previous_solution_information)}"
        )
        
        if previous_relaxation:
            logger.critical(
                f"GPBA-A MAIN LOOP: Found previous relaxation! "
                f"previous_values={'infeasible' if previous_values == 'infeasible' else 'solution'}"
            )
        
        if previous_relaxation:
            if previous_values == "infeasible":
                one_solution = []
            else:
                one_solution = previous_values
        else:
            # Update constraint RHS values
            # NOTE: objectives_exprs are ALREADY in maximization form (negated)
            # So we use them directly: objectives_exprs[i] - slack = ef_array
            for i, constr in enumerate(constraint_vars):
                model.remove(constr)
                if slack_vars[i] is not None:
                    constr = model.addConstr(
                        objectives_exprs[constraint_indices[i]] - slack_vars[i]
                        == ef_array[i],
                        name=f"epsilon_constraint_{constraint_indices[i]}_iter{iteration_count}",
                    )
                else:
                    constr = model.addConstr(
                        objectives_exprs[constraint_indices[i]] == ef_array[i],
                        name=f"epsilon_constraint_{constraint_indices[i]}_iter{iteration_count}",
                    )
                constraint_vars[i] = constr
            
            # Solve current configuration
            solve_start = time.time()
            model.optimize()
            solve_time = time.time() - solve_start
            
            logger.critical(
                f"GPBA-A MAIN LOOP: After solve - status={model.status}, "
                f"objVal={model.objVal if model.status == gp.GRB.OPTIMAL else 'N/A'}"
            )
            
            if model.status == gp.GRB.INFEASIBLE:
                save_solution_information(
                    ef_array, "infeasible", previous_solution_information
                )
                one_solution = []
            elif model.status == gp.GRB.OPTIMAL:
                # Extract solution
                solution_values = [
                    int(select_image[j].X > 0.5) for j in images_id
                ]
                current_objs: List[float] = []
                
                # Calculate all objective values (in MAXIMIZATION form)
                # Use binary solution_values to avoid floating-point precision issues
                for obj_name in config.objectives:
                    if obj_name == "min_cost":
                        current_objs.append(
                            -sum(solution_values[k] * costs[k] for k in images_id)
                        )
                    elif obj_name == "cloud_coverage":
                        cloud_val = 0.0
                        if cloud_covered is not None:
                            # Round each cloud coverage to avoid floating-point errors
                            cloud_val = sum(
                                round(cloud_covered[c].X) * clouds_id_area[c]
                                for c in clouds_id_area.keys()
                                if c in cloud_covered
                            )
                        total_cloud_area = sum(clouds_id_area.values())
                        current_objs.append(-int(total_cloud_area - cloud_val))
                    elif obj_name == "min_resolution":
                        if resolution_element is not None:
                            current_objs.append(
                                -int(
                                    round(
                                        sum(
                                            resolution_element[e].X
                                            for e in elements
                                        )
                                    )
                                )
                            )
                        else:
                            current_objs.append(0)
                    elif obj_name == "min_max_incidence_angle":
                        if current_max_incidence_angle is not None:
                            current_objs.append(
                                -int(round(current_max_incidence_angle.X))
                            )
                        else:
                            current_objs.append(0)
                
                logger.critical(
                    f"GPBA-A MAIN LOOP: Found solution (in max form): {current_objs}"
                )
                
                # Check if this is a new solution
                sol_str = convert_solution_value_to_str(current_objs)
                logger.critical(f"GPBA-A MAIN LOOP: Solution string for dedup: {sol_str}")
                logger.critical(f"GPBA-A MAIN LOOP: Already seen? {sol_str in previous_solutions}")
                if sol_str not in previous_solutions:
                    previous_solutions.add(sol_str)
                    
                    # Convert back to minimization for Pareto front storage
                    current_objs_min = [-x for x in current_objs]
                    add_to_pareto_front(current_objs_min, solution_values)
                    
                    statistics["number_of_solutions"] += 1
                    statistics["time_solver_sec"] += solve_time
                    statistics["solutions_time_list"].append(solve_time)
                    
                    logger.critical(
                        f"GPBA-A MAIN LOOP: ➕ NEW solution #{statistics['number_of_solutions']}: {current_objs_min}"
                    )
                    
                    one_solution = current_objs
                    save_solution_information(
                        ef_array, one_solution, previous_solution_information
                    )
                else:
                    one_solution = current_objs
            elif model.status == gp.GRB.TIME_LIMIT:
                logger.critical("GPBA-A MAIN LOOP: Timeout during loop")
                completed = False
                break
            else:
                logger.critical(
                    f"GPBA-A MAIN LOOP: Unexpected solver status: {model.status}"
                )
                one_solution = []
        
        # Update RWV and obj_k_at_ef_k based on solution found
        id_interval = -1  # Last constraint objective
        obj_k_at_ef_k[id_interval] = None
        
        if len(one_solution) > 0:
            for i in range(len(rwv)):
                rwv[i] = min(rwv[i], one_solution[constraint_indices[i]])
            obj_k_at_ef_k[id_interval] = one_solution[constraint_indices[id_interval]]
        
        # Adjust ef_array using interval management (core of GPBA-A)
        ef_intervals[id_interval] = adjust_parameter_ef_array(
            id_interval,
            ef_array,
            obj_k_at_ef_k[id_interval],
            ef_intervals[id_interval],
            constraint_indices,
            best_objective_values,
            nadir_objectives_values,
            gamma,
        )
        
        # Multi-dimensional cascading updates for higher constraint objectives
        for i in range(len(constraint_indices) - 1, 0, -1):
            if ef_array[i] > best_objective_values[constraint_indices[i]]:
                # Reset this dimension and update previous dimension
                ef_array[i] = nadir_objectives_values[constraint_indices[i]]
                rwv[i] = best_objective_values[constraint_indices[i]]
                id_interval = i - 1
                ef_intervals[id_interval] = adjust_parameter_ef_array(
                    id_interval,
                    ef_array,
                    obj_k_at_ef_k[id_interval],
                    ef_intervals[id_interval],
                    constraint_indices,
                    best_objective_values,
                    nadir_objectives_values,
                    gamma,
                )
                obj_k_at_ef_k[id_interval] = None
                if i == 1:
                    rwv[id_interval] = best_objective_values[
                        constraint_indices[id_interval]
                    ]
            else:
                break  # Only update higher dimensions if current exceeded
        
        iteration_count += 1
    
    logger.critical(
        f"GPBA-A MAIN LOOP: Completed after {iteration_count} iterations "
        f"(exhaustive={completed})"
    )
    
    return iteration_count, completed
