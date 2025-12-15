"""
Phase: Payoff Table Computation

This module implements the payoff table computation phase of GPBA-A algorithm.
This is the ONLY phase that loads and builds the Gurobi model from DZN file.

The payoff table consists of:
- Ideal points: Best value for each objective when optimized individually
- Nadir points: Worst value for each objective across all extreme solutions

This phase solves 2*n single-objective problems (n minimizations + n maximizations).
"""

from typing import Dict, List, Tuple, Any, Callable
import gurobipy as gp
import time
import logging

logger = logging.getLogger(__name__)


def compute_payoff_table_with_gurobi(
    model: gp.Model,
    objectives_exprs: List[gp.LinExpr],
    config: Any,
    problem_data_dict: Dict[str, Any],
    statistics: Dict[str, Any],
    add_to_pareto_front: Callable[[List[int], List[int]], None]
) -> Tuple[List[int], List[int]]:
    """
    Compute ideal and nadir points by solving 2*n single-objective optimizations.
    
    This is Phase 1 (Payoff Table) of GPBA-A. It's the most expensive phase as it
    solves 2*n MILP problems (for n objectives).
    
    Args:
        model: Gurobi model with all variables and constraints
        objectives_exprs: List of objective expressions (one per objective)
        config: Configuration object with objectives list and solver_timeout_sec
        problem_data_dict: Dictionary containing problem data (images, clouds, etc.)
        statistics: Statistics dict to update (modified in-place)
        add_to_pareto_front: Function to add solutions to Pareto front
    
    Returns:
        Tuple of (best_objective_values_max, nadir_objectives_values_max)
        Both lists are in MAXIMIZATION form (negated from minimization)
    """
    num_objectives = len(config.objectives)
    best_objective_values = [0] * num_objectives
    nadir_objectives_values = [0] * num_objectives
    
    model.setParam('TimeLimit', config.solver_timeout_sec)
    
    # Extract problem data for objective calculation
    images_id = problem_data_dict["images_id"]
    costs = problem_data_dict["costs"]
    clouds_id = problem_data_dict.get("clouds_id", [])
    clouds_id_area = problem_data_dict.get("clouds_id_area", {})
    elements = problem_data_dict.get("elements", [])
    select_image = problem_data_dict["select_image"]
    cloud_covered = problem_data_dict.get("cloud_covered")
    resolution_element = problem_data_dict.get("resolution_element")
    current_max_incidence_angle = problem_data_dict.get("current_max_incidence_angle")
    cloud_covered_by_image = problem_data_dict.get("cloud_covered_by_image", {})
    
    try:
        # STAGE 1: Find ideal points (best value for each objective individually)
        logger.critical(f"INLINED: STAGE 1 - Computing {num_objectives} extreme points")
        for i in range(num_objectives):
            model.setObjective(objectives_exprs[i], gp.GRB.MINIMIZE)
            
            solve_start = time.time()
            model.optimize()
            solve_time = time.time() - solve_start
            
            if model.status == gp.GRB.OPTIMAL:
                obj_value = int(model.objVal)
                best_objective_values[i] = obj_value
                logger.critical(f"INLINED: Best value for objective {i} ({config.objectives[i]}): {obj_value}")
                
                # Extract solution values
                solution_values = [int(select_image[j].X > 0.5) for j in images_id]
                current_objs = []
                
                # Log detailed solution info for cloud_coverage optimization
                if config.objectives[i] == "cloud_coverage":
                    logger.critical("INLINED: CLOUD_COVERAGE OPTIMIZATION - Details:")
                    selected_images = [j for j in images_id if select_image[j].X > 0.5]
                    logger.critical(f"INLINED: Selected images: {selected_images}")
                    logger.critical(f"INLINED: Total images available: {len(images_id)}")
                    if cloud_covered is not None:
                        covered_clouds = {c: cloud_covered[c].X for c in clouds_id if c in cloud_covered}
                        logger.critical(f"INLINED: Cloud coverage variables (first 10): {dict(list(covered_clouds.items())[:10])}")
                        total_cloud_area = sum(clouds_id_area.values())
                        covered_cloud_area = sum(cloud_covered[c].X * clouds_id_area[c] for c in clouds_id if c in cloud_covered)
                        logger.critical(f"INLINED: Total cloud area: {total_cloud_area}")
                        logger.critical(f"INLINED: Covered cloud area: {covered_cloud_area}")
                        logger.critical(f"INLINED: Uncovered cloud area: {total_cloud_area - covered_cloud_area}")
                        
                        # Debug: Check which clouds are not covered
                        uncovered_clouds = []
                        covered_area = 0
                        uncovered_area = 0
                        for c in clouds_id:
                            if c in cloud_covered and cloud_covered[c].X < 0.5:
                                uncovered_clouds.append((c, clouds_id_area[c]))
                                uncovered_area += clouds_id_area[c]
                            elif c in cloud_covered and cloud_covered[c].X >= 0.5:
                                covered_area += clouds_id_area[c]
                        logger.critical(f"INLINED: Uncovered clouds (first 10): {uncovered_clouds[:10]}")
                        logger.critical(f"INLINED: Covered area: {covered_area}, Uncovered area: {uncovered_area}")
                        logger.critical(f"INLINED: Total uncovered clouds: {len(uncovered_clouds)}")
                        
                        # Debug: Check some uncovered clouds with non-zero area
                        nonzero_uncovered = [(c, area) for c, area in uncovered_clouds if area > 0]
                        logger.critical(f"INLINED: Uncovered clouds with non-zero area (first 5): {nonzero_uncovered[:5]}")
                        
                        # Debug: Check image-cloud relationships for uncovered clouds
                        if uncovered_clouds:
                            sample_cloud = uncovered_clouds[0][0]
                            covering_images = []
                            for img in images_id:
                                if sample_cloud in cloud_covered_by_image.get(img, set()):
                                    covering_images.append(img)
                            logger.critical(f"INLINED: Sample uncovered cloud {sample_cloud} can be covered by images: {covering_images}")
                    else:
                        logger.critical("INLINED: No cloud_covered variables - cloud objective may be disabled")
                
                # Calculate all objective values for this solution
                # Use binary solution_values to avoid floating-point precision issues
                for j, obj_name in enumerate(config.objectives):
                    if obj_name == "min_cost":
                        current_objs.append(sum(solution_values[k] * costs[k] for k in images_id))
                    elif obj_name == "cloud_coverage":
                        cloud_val = 0
                        if cloud_covered is not None:
                            # Round each cloud coverage to avoid floating-point errors
                            cloud_val = sum(round(cloud_covered[c].X) * clouds_id_area[c] 
                                          for c in clouds_id if c in cloud_covered)
                        # Add uncoverable clouds to the uncovered area calculation
                        total_cloud_area = sum(clouds_id_area.values())
                        current_objs.append(int(total_cloud_area - cloud_val))
                    elif obj_name == "min_resolution":
                        if resolution_element is not None:
                            current_objs.append(int(round(sum(resolution_element[e].X for e in elements))))
                        else:
                            current_objs.append(0)
                    elif obj_name == "min_max_incidence_angle":
                        if current_max_incidence_angle is not None:
                            current_objs.append(int(round(current_max_incidence_angle.X)))
                        else:
                            current_objs.append(0)
                
                # Ensure all objectives are integers
                current_objs = [int(x) for x in current_objs]
                logger.critical(f"INLINED: Extreme point {i}: {current_objs}")
                
                # Log types of objectives
                obj_types = [f"{config.objectives[j]}:{type(val).__name__}" for j, val in enumerate(current_objs)]
                logger.critical(f"INLINED: Extreme point {i} types: {obj_types}")
                
                add_to_pareto_front(current_objs, solution_values)
                
                statistics["number_of_solutions"] += 1
                statistics["time_solver_sec"] += solve_time
                statistics["solutions_time_list"].append(solve_time)
                
            elif model.status == gp.GRB.TIME_LIMIT:
                raise TimeoutError(f"Timeout while optimizing objective {i}")
            else:
                raise RuntimeError(f"Failed to optimize objective {i}: status {model.status}")
                
        # STAGE 1b: Find nadir points (worst values)
        logger.critical(f"INLINED: Computing {num_objectives} nadir points")
        for i in range(num_objectives):
            model.setObjective(objectives_exprs[i], gp.GRB.MAXIMIZE)
            solve_start = time.time()
            model.optimize()
            
            if model.status == gp.GRB.OPTIMAL:
                nadir_objectives_values[i] = int(model.objVal)
                logger.critical(f"INLINED: Nadir value for objective {i} ({config.objectives[i]}): {nadir_objectives_values[i]}")
            elif model.status == gp.GRB.TIME_LIMIT:
                raise TimeoutError(f"Timeout while finding nadir for objective {i}")
        
        logger.critical(f"INLINED: Best values (minimization): {best_objective_values}")
        logger.critical(f"INLINED: Nadir values (minimization): {nadir_objectives_values}")
        
        # STAGE 1.5: Convert to maximization for uniform handling
        logger.critical("INLINED: Converting objectives to maximization")
        best_objective_values_max = [-x for x in best_objective_values]
        nadir_objectives_values_max = [-x for x in nadir_objectives_values]
        
        logger.critical(f"INLINED: Best values (maximization): {best_objective_values_max}")
        logger.critical(f"INLINED: Nadir values (maximization): {nadir_objectives_values_max}")
        
        return best_objective_values_max, nadir_objectives_values_max
        
    except (TimeoutError, RuntimeError) as e:
        logger.error(f"Payoff table computation failed: {e}")
        raise
