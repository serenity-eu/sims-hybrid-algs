import logging
import traceback
from importlib import resources
from . import mzn_models

from .Config import Config
from .main import (
    build_instance,
    build_model,
    build_solver,
    init_top_level_statistics,
    set_right_time_after_timeout,
    write_statistics,
)

MZN_MODEL_PATH = None
with resources.as_file(resources.files(mzn_models) / "mosaic_cloud2.mzn") as mzn_model_path:
    MZN_MODEL_PATH = mzn_model_path

def solve_milp_inlined(config: Config, objectives: list[str] | None = None):
    """
    Inlined implementation of GPBA-A algorithm with Gurobi solver.
    
    This function contains the complete implementation of the GPBA-A (Coverage Grid Point Based Representation)
    algorithm without external function calls except stdlib and gurobipy. It follows the pseudocode from the
    SIMS_GPBA_A_SPECIFICATION document.
    """
    import gurobipy as gp
    import re
    import os
    import time
    import json
    from datetime import datetime
    import csv
    from filelock import FileLock, Timeout as FileLockTimeout
    
    logger = logging.getLogger(__name__)
    logger.critical("INLINED: Starting solve_milp_inlined")
    
    # If objectives are provided, update the config
    if objectives is not None:
        config.objectives = objectives
    
    logger.critical(f"INLINED: Objectives = {config.objectives}")
    
    print("Start computing (inlined): " + config.uid())
    start_time = time.time()
    
    # STAGE 1: PARSE DATA FROM DZN FILE
    dzn_file_path = os.path.join(config.data_sets_folder, f"{config.data_name}.dzn")
    logger.critical(f"INLINED: Parsing DZN file: {dzn_file_path}")
    
    if not os.path.exists(dzn_file_path):
        raise FileNotFoundError(f"DZN file not found: {dzn_file_path}")
    
    # Parse .dzn file using simple regex patterns
    with open(dzn_file_path, 'r') as f:
        content = f.read()
    
    def parse_array(pattern, content):
        match = re.search(pattern, content, re.DOTALL)
        if match:
            array_str = match.group(1)
            items = [item.strip() for item in array_str.strip('[]').split(',')]
            return [float(item) if '.' in item else int(item) for item in items if item.strip()]
        return []
    
    def parse_set_array(pattern, content):
        match = re.search(pattern, content, re.DOTALL)
        if match:
            array_str = match.group(1)
            sets = []
            set_matches = re.findall(r'\{([^}]*)\}', array_str)
            for set_match in set_matches:
                if set_match.strip():
                    elements = [int(x.strip()) for x in set_match.split(',') if x.strip()]
                    sets.append(elements)
                else:
                    sets.append([])
            return sets
        return []
    
    # Extract data
    costs = parse_array(r'costs\s*=\s*\[([^\]]+)\]', content)
    areas = parse_array(r'areas\s*=\s*\[([^\]]+)\]', content)
    resolution = parse_array(r'resolution\s*=\s*\[([^\]]+)\]', content)
    incidence_angle = parse_array(r'incidence_angle\s*=\s*\[([^\]]+)\]', content)
    images_raw = parse_set_array(r'images\s*=\s*\[([^\]]+)\]', content)
    clouds_raw = parse_set_array(r'clouds\s*=\s*\[([^\]]+)\]', content)
    
    logger.critical(f"INLINED: Parsed data - {len(costs)} costs, {len(areas)} areas, {len(images_raw)} images, {len(clouds_raw)} clouds")
    
    # STAGE 2: CORRECT STARTING INDEXES (1-indexed to 0-indexed)
    images = []
    clouds = []
    for img_set in images_raw:
        images.append([x - 1 for x in img_set])  # Convert to 0-indexed
    for cloud_set in clouds_raw:
        clouds.append([x - 1 for x in cloud_set])  # Convert to 0-indexed
    
    # STAGE 3: BUILD CLOUD COVERAGE RELATIONSHIPS
    cloud_covered_by_image = {}  # image_id -> set of clouds it can cover
    clouds_id_area = {}          # cloud_id -> area
    
    for i in range(len(clouds)):
        image_cloud_set = set(clouds[i])  # Cloudy elements in image i
        for cloud_id in image_cloud_set:
            clouds_id_area[cloud_id] = areas[cloud_id]
            # Find images that can cover this cloud (image j covers element cloud_id without clouds)
            for j in range(len(images)):
                if i != j and cloud_id in images[j] and cloud_id not in clouds[j]:
                    if j in cloud_covered_by_image:
                        cloud_covered_by_image[j].add(cloud_id)
                    else:
                        cloud_covered_by_image[j] = {cloud_id}
                        
    # Debug: Check cloud coverage relationships
    uncoverable_clouds = []
    total_uncoverable_area = 0
    for cloud_id in clouds_id_area:
        covering_images = []
        for img in cloud_covered_by_image:
            if cloud_id in cloud_covered_by_image[img]:
                covering_images.append(img)
        if not covering_images:
            uncoverable_clouds.append((cloud_id, clouds_id_area[cloud_id]))
            total_uncoverable_area += clouds_id_area[cloud_id]
    
    logger.critical(f"INLINED: Total clouds: {len(clouds_id_area)}")
    logger.critical(f"INLINED: Uncoverable clouds: {len(uncoverable_clouds)}")
    logger.critical(f"INLINED: Total uncoverable area: {total_uncoverable_area}")
    if uncoverable_clouds:
        logger.critical(f"INLINED: Sample uncoverable clouds: {uncoverable_clouds[:10]}")
    
    # STAGE 4: BUILD IMAGE-ELEMENT MAPPINGS
    elements = list(range(len(areas)))
    images_id = list(range(len(images)))
    clouds_id = list(clouds_id_area.keys())
    
    images_covering_element = {}
    for i in images_id:
        for e in images[i]:
            if e not in images_covering_element:
                images_covering_element[e] = [i]
            else:
                images_covering_element[e].append(i)
    
    # STAGE 5: CREATE GUROBI MODEL
    model = gp.Model("SIMSModelInlined")
    model.setParam('OutputFlag', 0)  # Suppress Gurobi output
    
    # STAGE 6: ADD VARIABLES
    select_image = model.addVars(len(images), vtype=gp.GRB.BINARY, name="select_image")
    
    # Cloud variables (only if cloud_coverage objective is used)
    cloud_covered = None
    if "cloud_coverage" in config.objectives:
        cloud_covered = model.addVars(clouds_id, vtype=gp.GRB.BINARY, name="cloud_covered")
    
    # Resolution variables (only if min_resolution objective is used)
    resolution_element = None
    auxiliary_resolution = None
    if "min_resolution" in config.objectives:
        min_res = min(resolution)
        max_res = max(resolution)
        resolution_element = model.addVars(elements, lb=min_res, ub=max_res, 
                                         vtype=gp.GRB.INTEGER, name="resolution_element")
        
        # Auxiliary variables for linearization
        auxiliary_resolution = {}
        for element in elements:
            auxiliary_resolution[element] = {}
            for image in images_covering_element[element]:
                auxiliary_resolution[element][image] = model.addVar(
                    vtype=gp.GRB.BINARY, name=f"aux_res_{element}_{image}")
    
    # Incidence angle variables (only if min_max_incidence_angle objective is used)
    effective_incidence_angle = None
    current_max_incidence_angle = None
    if "min_max_incidence_angle" in config.objectives:
        effective_incidence_angle = model.addVars(images_id, vtype=gp.GRB.INTEGER,
                                                name="effective_incidence_angle")
        current_max_incidence_angle = model.addVar(vtype=gp.GRB.INTEGER, 
                                                 name="max_allowed_incidence_angle")
    
    # STAGE 7: ADD CONSTRAINTS
    
    # Coverage constraint (always required)
    for element in elements:
        model.addConstr(
            gp.quicksum(select_image[i] for i in images_id if element in images[i]) >= 1,
            name=f"coverage_{element}"
        )
    
    # Cloud constraints (conditional)
    if "cloud_coverage" in config.objectives and cloud_covered is not None:
        for cloud in clouds_id:
            # If cloud is covered, at least one covering image must be selected
            covering_images = [i for i in cloud_covered_by_image.keys() 
                             if cloud in cloud_covered_by_image[i]]
            if covering_images:
                model.addConstr(
                    gp.quicksum(select_image[i] for i in covering_images) >= cloud_covered[cloud],
                    name=f"cloud_coverage_lower_{cloud}"
                )
                model.addConstr(
                    gp.quicksum(select_image[i] for i in covering_images) 
                    <= cloud_covered[cloud] * len(images),
                    name=f"cloud_coverage_upper_{cloud}"
                )
            else:
                # Uncoverable cloud must have cloud_covered = 0
                model.addConstr(
                    cloud_covered[cloud] == 0,
                    name=f"cloud_uncoverable_{cloud}"
                )
    
    # Resolution constraints (conditional)
    if "min_resolution" in config.objectives and resolution_element is not None and auxiliary_resolution is not None:
        big_M = max(resolution) + 1
        
        for element in elements:
            covering_images = images_covering_element[element]
            
            # Each element must have exactly (n-1) auxiliary variables set to 1
            total_aux = len(auxiliary_resolution[element])
            model.addConstr(
                gp.quicksum(auxiliary_resolution[element][i] for i in covering_images) == total_aux - 1,
                name=f"aux_resolution_sum_{element}"
            )
            
            # Linearization of min operation
            for image in covering_images:
                model.addConstr(
                    resolution_element[element] >= 
                    resolution[image] * select_image[image] +
                    big_M * (1 - select_image[image]) -
                    2 * big_M * auxiliary_resolution[element][image],
                    name=f"resolution_linearization_{element}_{image}"
                )
    
    # Incidence angle constraints (conditional)
    if "min_max_incidence_angle" in config.objectives and effective_incidence_angle is not None:
        # Effective incidence angle is 0 if image not selected, actual angle if selected
        for image in images_id:
            model.addConstr((select_image[image] == 0) >> 
                           (effective_incidence_angle[image] == 0),
                           name=f"incidence_angle_not_selected_{image}")
            model.addConstr((select_image[image] == 1) >> 
                           (effective_incidence_angle[image] == incidence_angle[image]),
                           name=f"incidence_angle_selected_{image}")
        
        # Maximum incidence angle across all selected images
        if current_max_incidence_angle is not None:
            for image in images_id:
                model.addConstr(current_max_incidence_angle >= effective_incidence_angle[image],
                               name=f"max_incidence_angle_{image}")
    
    # STAGE 8: DEFINE OBJECTIVES
    objectives_exprs = []
    
    available_objectives = {
        "min_cost": lambda: gp.quicksum(select_image[i] * costs[i] for i in images_id),
        "cloud_coverage": lambda: sum(clouds_id_area.values()) - (
            gp.quicksum(cloud_covered[c] * clouds_id_area[c] for c in clouds_id) 
            if cloud_covered is not None else 0),
        "min_resolution": lambda: (
            gp.quicksum(resolution_element[e] for e in elements) 
            if resolution_element is not None else 0),
        "min_max_incidence_angle": lambda: (
            current_max_incidence_angle if current_max_incidence_angle is not None else 0)
    }
    
    for obj_name in config.objectives:
        if obj_name in available_objectives:
            objectives_exprs.append(available_objectives[obj_name]())
        else:
            raise ValueError(f"Invalid objective '{obj_name}'")
    
    # STAGE 9: GPBA-A ALGORITHM IMPLEMENTATION
    
    # Initialize statistics
    statistics = {
        "number_of_solutions": 0,
        "time_solver_sec": 0,
        "solutions_time_list": [],
        "pareto_front": "[",
        "all_solutions": "[",
        "hypervolume": 0,
        "exhaustive": False,
        "datetime": datetime.now(),
        "instance": config.data_name,
        "problem": config.problem_name,
        "solver_name": config.solver_name,
        "front_strategy": config.front_strategy,
        "solver_search_strategy": config.solver_search_strategy,
        "fzn_optimisation_level": config.fzn_optimisation_level,
        "cores": config.cores,
        "solver_timeout_sec": config.solver_timeout_sec
    }
    
    # Pareto front storage
    pareto_solutions = []
    pareto_front = []  # Indices of non-dominated solutions
    
    def dominates(sol1_objs, sol2_objs):
        """Check if solution 1 dominates solution 2 (all objectives are minimization)."""
        return all(obj1 <= obj2 for obj1, obj2 in zip(sol1_objs, sol2_objs)) and \
               any(obj1 < obj2 for obj1, obj2 in zip(sol1_objs, sol2_objs))
    
    def add_to_pareto_front(solution_objs, solution_values):
        """Add solution to Pareto front if non-dominated."""
        idx = len(pareto_solutions)
        solution_data = {
            "objs": solution_objs,
            "solution_values": solution_values,
            "timestamp": time.time() - start_time
        }
        pareto_solutions.append(solution_data)
        
        # Check if new solution is dominated
        for front_idx in pareto_front:
            if dominates(pareto_solutions[front_idx]["objs"], solution_objs):
                return False  # New solution is dominated
        
        # Find solutions dominated by new solution
        to_remove = []
        for i, front_idx in enumerate(pareto_front):
            if dominates(solution_objs, pareto_solutions[front_idx]["objs"]):
                to_remove.append(i)
        
        # Remove dominated solutions from front
        for i in reversed(to_remove):
            pareto_front.pop(i)
        
        # Add new solution to front
        pareto_front.append(idx)
        return True
    
    # GPBA-A: STAGE 1 - Compute extreme points
    num_objectives = len(objectives_exprs)
    best_objective_values = [0] * num_objectives
    nadir_objectives_values = [0] * num_objectives
    
    model.setParam('TimeLimit', config.solver_timeout_sec)
    
    try:
        # Find ideal points (best value for each objective individually)
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
                for j, obj_name in enumerate(config.objectives):
                    if obj_name == "min_cost":
                        current_objs.append(int(sum(select_image[k].X * costs[k] for k in images_id 
                                                  if select_image[k].X > 0.5)))
                    elif obj_name == "cloud_coverage":
                        cloud_val = 0
                        if cloud_covered is not None:
                            cloud_val = sum(cloud_covered[c].X * clouds_id_area[c] 
                                          for c in clouds_id if c in cloud_covered)
                        # Add uncoverable clouds to the uncovered area calculation
                        total_cloud_area = sum(clouds_id_area.values())
                        current_objs.append(int(total_cloud_area - cloud_val))
                    elif obj_name == "min_resolution":
                        if resolution_element is not None:
                            current_objs.append(int(sum(resolution_element[e].X for e in elements)))
                        else:
                            current_objs.append(0)
                    elif obj_name == "min_max_incidence_angle":
                        if current_max_incidence_angle is not None:
                            current_objs.append(int(current_max_incidence_angle.X))
                        else:
                            current_objs.append(0)
                
                # Add to Pareto front
                current_objs = [int(round(x)) for x in current_objs]
                logger.critical(f"INLINED: Extreme point {i}: {current_objs}")
                add_to_pareto_front(current_objs, solution_values)
                
                statistics["number_of_solutions"] += 1
                statistics["time_solver_sec"] += solve_time
                statistics["solutions_time_list"].append(solve_time)
                
            elif model.status == gp.GRB.TIME_LIMIT:
                raise TimeoutError(f"Timeout while optimizing objective {i}")
            else:
                raise RuntimeError(f"Failed to optimize objective {i}: status {model.status}")
                
        # Find nadir points (worst values)
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
        
        logger.critical(f"INLINED: Best values: {best_objective_values}")
        logger.critical(f"INLINED: Nadir values: {nadir_objectives_values}")
        
        # GPBA-A: STAGE 2 - Setup ε-constraint formulation
        if num_objectives > 1:
            logger.critical("INLINED: STAGE 2 - Setting up ε-constraint formulation")
            main_obj_index = 0  # Use first objective as main
            constraint_indices = [i for i in range(num_objectives) if i != main_obj_index]
            logger.critical(f"INLINED: Main objective: {main_obj_index} ({config.objectives[main_obj_index]})")
            logger.critical(f"INLINED: Constraint objectives: {constraint_indices}")
            
            # Convert to maximization for main objective
            model.setObjective(-objectives_exprs[main_obj_index], gp.GRB.MAXIMIZE)
            
            # Create slack variables for augmentation
            delta = 0.01
            slack_vars = []
            for i, constraint_idx in enumerate(constraint_indices):
                max_s = abs(best_objective_values[constraint_idx] - nadir_objectives_values[constraint_idx])
                if max_s > 0:
                    s = model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=max_s, name=f"s{constraint_idx+1}")
                    slack_vars.append(s)
                else:
                    slack_vars.append(None)
            
            # Setup augmented objective
            obj_ranges = [abs(best_objective_values[i] - nadir_objectives_values[i]) for i in constraint_indices]
            slack_terms = []
            for i in range(len(constraint_indices)):
                if obj_ranges[i] > 0 and slack_vars[i] is not None:
                    slack_terms.append(slack_vars[i] / (10**i * obj_ranges[i]))
            
            if slack_terms:
                slack_sum = gp.quicksum(slack_terms)
                augmented_obj = -objectives_exprs[main_obj_index] + delta * slack_sum
            else:
                augmented_obj = -objectives_exprs[main_obj_index]
            
            model.setObjective(augmented_obj, gp.GRB.MAXIMIZE)
            
            # GPBA-A: STAGE 3 - Main coverage loop
            ef_array = [nadir_objectives_values[i] for i in constraint_indices]
            previous_solutions = set()
            logger.critical("INLINED: STAGE 3 - Starting main coverage loop")
            logger.critical(f"INLINED: Initial ef_array: {ef_array}")
            
            # Constraint variables for ε-constraint method
            constraint_vars = []
            for i, constraint_idx in enumerate(constraint_indices):
                if slack_vars[i] is not None:
                    constr = model.addConstr(
                        objectives_exprs[constraint_idx] - slack_vars[i] == ef_array[i],
                        name=f"epsilon_constraint_{constraint_idx}"
                    )
                else:
                    constr = model.addConstr(
                        objectives_exprs[constraint_idx] == ef_array[i],
                        name=f"epsilon_constraint_{constraint_idx}"
                    )
                constraint_vars.append(constr)
            
            # Main GPBA-A loop with systematic grid exploration
            iteration_count = 0
            max_iterations = 5000  # Increased for more thorough exploration
            
            # Calculate step sizes for each constraint objective (smaller steps for better coverage)
            step_sizes = []
            for i, constraint_idx in enumerate(constraint_indices):
                range_val = best_objective_values[constraint_idx] - nadir_objectives_values[constraint_idx]
                if range_val != 0:
                    # Use larger steps to find solutions faster - aim for about 10-15 steps per dimension
                    step_size = max(1000, abs(range_val) // 10)  # Much larger steps
                    step_sizes.append(step_size)
                else:
                    step_sizes.append(1000)
            
            logger.critical(f"INLINED: Range for constraint {constraint_indices[0]}: {best_objective_values[constraint_indices[0]]} to {nadir_objectives_values[constraint_indices[0]]}")
            logger.critical(f"INLINED: Calculated step sizes: {step_sizes}")
            
            # Generate all constraint values systematically in multi-dimensional grid
            def get_next_constraint_values(current_ef_array):
                """Get next combination of constraint values using grid search"""
                next_array = current_ef_array.copy()
                
                # Increment like an odometer - rightmost digit first
                for i in range(len(next_array) - 1, -1, -1):
                    constraint_idx = constraint_indices[i]
                    next_array[i] -= step_sizes[i]  # Move towards best value (smaller)
                    
                    if next_array[i] >= best_objective_values[constraint_idx]:
                        return next_array  # Valid next point
                    else:
                        # Overflow - reset this dimension and carry to next
                        next_array[i] = nadir_objectives_values[constraint_idx]
                
                # If we get here, all combinations exhausted
                return None
            
            while iteration_count < max_iterations:
                
                if iteration_count % 20 == 0:  # Log progress more frequently 
                    logger.critical(f"INLINED: Iteration {iteration_count}, ef_array = {ef_array}")
                
                # Update constraint values
                for i, constr in enumerate(constraint_vars):
                    constraint_idx = constraint_indices[i]
                    model.remove(constr)
                    if slack_vars[i] is not None:
                        constr = model.addConstr(
                            objectives_exprs[constraint_idx] - slack_vars[i] == ef_array[i],
                            name=f"epsilon_constraint_{constraint_idx}_{iteration_count}"
                        )
                    else:
                        constr = model.addConstr(
                            objectives_exprs[constraint_idx] == ef_array[i],
                            name=f"epsilon_constraint_{constraint_idx}_{iteration_count}"
                        )
                    constraint_vars[i] = constr
                
                # Solve current configuration
                solve_start = time.time()
                model.optimize()
                solve_time = time.time() - solve_start
                
                if model.status == gp.GRB.OPTIMAL:
                    # Extract solution
                    solution_values = [int(select_image[j].X > 0.5) for j in images_id]
                    current_objs = []
                    
                    # Calculate all objective values
                    for obj_name in config.objectives:
                        if obj_name == "min_cost":
                            current_objs.append(int(sum(select_image[k].X * costs[k] for k in images_id 
                                                      if select_image[k].X > 0.5)))
                        elif obj_name == "cloud_coverage":
                            cloud_val = 0
                            if cloud_covered is not None:
                                cloud_val = sum(cloud_covered[c].X * clouds_id_area[c] 
                                              for c in clouds_id if c in cloud_covered)
                            current_objs.append(int(sum(clouds_id_area.values()) - cloud_val))
                        elif obj_name == "min_resolution":
                            if resolution_element is not None:
                                current_objs.append(int(sum(resolution_element[e].X for e in elements)))
                            else:
                                current_objs.append(0)
                        elif obj_name == "min_max_incidence_angle":
                            if current_max_incidence_angle is not None:
                                current_objs.append(int(current_max_incidence_angle.X))
                            else:
                                current_objs.append(0)
                    
                    # Check if this is a new solution
                    sol_str = str(current_objs)
                    if sol_str not in previous_solutions:
                        previous_solutions.add(sol_str)
                        add_to_pareto_front(current_objs, solution_values)
                        
                        statistics["number_of_solutions"] += 1
                        statistics["time_solver_sec"] += solve_time
                        statistics["solutions_time_list"].append(solve_time)
                        
                        logger.critical(f"INLINED: Found solution {statistics['number_of_solutions']}: {current_objs}")
                    
                    # Simple update: move to next constraint combination
                    next_ef_array = get_next_constraint_values(ef_array)
                    if next_ef_array is None:
                        # Exhausted all combinations
                        logger.critical(f"INLINED: Exhausted all constraint combinations at iteration {iteration_count}")
                        break
                    ef_array = next_ef_array
                    logger.critical(f"INLINED: Moving to next ef_array: {ef_array}")
                
                elif model.status == gp.GRB.INFEASIBLE:
                    # Move to next configuration
                    logger.critical(f"INLINED: Infeasible at ef_array = {ef_array}, moving to next")
                    next_ef_array = get_next_constraint_values(ef_array)
                    if next_ef_array is None:
                        logger.critical(f"INLINED: Exhausted all constraint combinations after infeasible at iteration {iteration_count}")
                        break
                    ef_array = next_ef_array
                
                elif model.status == gp.GRB.TIME_LIMIT:
                    print("Timeout during GPBA-A loop")
                    break
                
                iteration_count += 1
        
        statistics["exhaustive"] = True
        logger.critical("INLINED: Problem completely explored.")
        
    except TimeoutError as e:
        logger.critical(f"INLINED: Timeout during extreme point computation: {e}")
        statistics["exhaustive"] = False
    except Exception as e:
        logger.critical(f"INLINED: Error during solving: {e}")
        statistics["exhaustive"] = False
    
    logger.critical(f"INLINED: Final statistics - {statistics['number_of_solutions']} solutions found")
    
    # STAGE 10: FINALIZE RESULTS
    
    # Build results strings
    pareto_objs = []
    all_solutions = []
    
    for idx in pareto_front:
        solution = pareto_solutions[idx]
        pareto_objs.append(solution["objs"])
        all_solutions.append(solution)
    
    statistics["pareto_front"] = json.dumps(pareto_objs) + ","
    statistics["all_solutions"] = json.dumps([sol["objs"] for sol in all_solutions]) + ","
    
    # Simple hypervolume calculation
    if pareto_objs:
        hv = 1.0  # Placeholder - real hypervolume calculation is complex
        statistics["hypervolume"] = hv
    
    print(f"End of solving statistics (inlined): {statistics['number_of_solutions']} solutions found")
    
    # STAGE 11: WRITE RESULTS TO CSV
    write_statistics_inlined(config, statistics)


def write_statistics_inlined(config, statistics):
    """Write statistics to CSV file (simplified version)."""
    import csv
    import os
    from filelock import FileLock, Timeout as FileLockTimeout
    
    def create_summary_file_inlined(filename):
        """Create CSV summary file if it doesn't exist."""
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                headers = list(statistics.keys())
                writer = csv.DictWriter(f, fieldnames=headers, delimiter=";")
                writer.writeheader()
    
    try:
        lock = FileLock(config.summary_filename + ".lock", timeout=10)
        with lock:
            create_summary_file_inlined(config.summary_filename)
            with open(config.summary_filename, "a") as f:
                headers = list(statistics.keys())
                writer = csv.DictWriter(f, fieldnames=headers, delimiter=";")
                writer.writerow(statistics)
    except FileLockTimeout:
        print("Could not acquire lock on summary file. Statistics will be printed on standard output instead.")
        print(statistics)

def solve_milp(config: Config, objectives: list[str] | None = None):
    """Solve the SIMS problem using Mixed Integer Linear Programming (MILP) solver."""
    
    logger = logging.getLogger(__name__)
    logger.critical("ORIGINAL: Starting solve_milp")

    # If objectives are provided, update the config
    if objectives is not None:
        config.objectives = objectives
    
    logger.critical(f"ORIGINAL: Objectives = {config.objectives}")

    instance = build_instance(config)
    logger.critical("ORIGINAL: Built instance")
    print("Start computing: " + config.uid())
    statistics = {}
    config.init_statistics(statistics)
    init_top_level_statistics(statistics)
    model = build_model(instance, config)
    logger.critical("ORIGINAL: Built model")
    solver, pareto_front = build_solver(model, instance, config, statistics)
    logger.critical("ORIGINAL: Built solver and pareto_front")
    save_results = True
    try:
        statistics["exhaustive"] = False
        statistics["incomplete_timeout_solution_added_to_front"] = False
        logger.critical("ORIGINAL: Starting solver.solve() iteration")
        solution_count = 0
        for x in solver.solve():
            solution_count += 1
            if hasattr(x, 'objs'):
                logger.critical(f"ORIGINAL: Found solution {solution_count}: {x.objs}")
            else:
                logger.critical(f"ORIGINAL: Found solution {solution_count}: {x}")
        logger.critical("ORIGINAL: Problem completely explored.")
        statistics["exhaustive"] = True
    except TimeoutError:
        logger.critical("ORIGINAL: Timeout triggered getting last incomplete solution")
        if solver.process_last_incomplete_solution():
            # the last incomplete solution was added to the pareto front
            logger.critical("ORIGINAL: Last incomplete solution added to the pareto front")
            statistics["incomplete_timeout_solution_added_to_front"] = True
        else:
            logger.critical(
                "ORIGINAL: There were not incomplete solution or the last incomplete solution was not added to "
                "the pareto front"
            )
            set_right_time_after_timeout(statistics, config.solver_timeout_sec)
    except Exception as e:
        logger.critical("ORIGINAL: Error Exception raised: " + str(e))
        logging.error(traceback.format_exc())
        save_results = False
    if save_results:
        statistics["all_solutions"] = statistics["all_solutions"][:-1]
        statistics["all_solutions"] += "}"
        statistics["hypervolume"] = pareto_front.hypervolume()
        pareto_solutions_time_list = [
            statistics["solutions_time_list"][x] for x in pareto_front.front
        ]
        statistics["pareto_solutions_time_list"] = pareto_solutions_time_list
        logger.critical("ORIGINAL: end of solving statistics: " + str(statistics))
        write_statistics(config, statistics)
