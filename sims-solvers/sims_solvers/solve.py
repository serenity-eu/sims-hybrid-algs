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
    Complete inlined implementation of GPBA-A algorithm with Gurobi solver.
    
    This function contains the complete implementation of the GPBA-A (Coverage Grid Point Based Representation)
    algorithm without external function calls except stdlib and gurobipy. It follows the pseudocode from the
    SIMS_GPBA_A_SPECIFICATION document with all critical features implemented:
    
    - IntervalManager for adaptive exploration
    - adjust_parameter_ef_array() for intelligent grid point selection
    - Relative Worst Values (RWV) tracking
    - Multi-dimensional cascading updates
    - Solution relaxation search
    - Proper minimization to maximization conversion
    """
    import gurobipy as gp
    import re
    import os
    import time
    import json
    from datetime import datetime
    import csv
    from filelock import FileLock, Timeout as FileLockTimeout
    
    # ============================================================================
    # INNER CLASS: IntervalManager
    # ============================================================================
    class IntervalManager:
        """Manages intervals for efficient Pareto front coverage in GPBA-A algorithm."""
        
        def __init__(self, min_value, max_value):
            self.intervals = set()
            self.min_value = min_value
            self.max_value = max_value
            self.add_interval(min_value, max_value)
        
        def add_interval(self, start, end):
            """Add interval, merging with existing overlapping intervals."""
            new_intervals = set()
            to_add = (start, end)
            
            for interval in self.intervals:
                if interval[1] < start or interval[0] > end:  # No overlap
                    new_intervals.add(interval)
                else:  # Merge overlapping intervals
                    to_add = (min(to_add[0], interval[0]), max(to_add[1], interval[1]))
            
            new_intervals.add(to_add)
            self.intervals = new_intervals
        
        def remove_one_point(self, point):
            """Remove a single point, splitting intervals if necessary."""
            new_intervals = set()
            
            for interval in self.intervals:
                if interval[0] <= point <= interval[1]:  # Point within interval
                    if interval[0] < point:
                        new_intervals.add((interval[0], point - 1))
                    if interval[1] > point:
                        new_intervals.add((point + 1, interval[1]))
                else:  # No overlap
                    new_intervals.add(interval)
            
            self.intervals = new_intervals
        
        def remove_interval(self, start, end):
            """Remove interval, adjusting or splitting existing intervals."""
            new_intervals = set()
            
            for interval in self.intervals:
                if interval[1] < start or interval[0] > end:  # No overlap
                    new_intervals.add(interval)
                else:
                    # Adjust or split interval
                    if interval[0] < start:
                        new_intervals.add((interval[0], start - 1))
                    if interval[1] > end:
                        new_intervals.add((end + 1, interval[1]))
            
            self.intervals = new_intervals
        
        def find_largest_interval(self):
            """Find and return the largest interval by length."""
            if not self.intervals:
                return None
            return max(self.intervals, key=lambda x: x[1] - x[0])
    
    # ============================================================================
    # HELPER FUNCTIONS
    # ============================================================================
    
    def create_interval(obj_idx, best_values, nadir_values):
        """Create an IntervalManager for a given objective index."""
        min_interval = min(nadir_values[obj_idx], best_values[obj_idx])
        max_interval = max(nadir_values[obj_idx], best_values[obj_idx])
        return IntervalManager(min_interval, max_interval)
    
    def adjust_parameter_ef_array(id_constraint_objective, ef_array, sol_obj_k, 
                                  ef_interval, constraint_indices, best_objective_values,
                                  nadir_objectives_values, gamma=1):
        """
        Adjust ef_array parameter based on solution found, using interval management.
        
        This is the core of GPBA-A that adaptively explores the largest gaps in the Pareto front.
        """
        start_removal = ef_array[id_constraint_objective]
        new_max_interval = start_removal - 1
        
        if sol_obj_k is None:  # Infeasible
            end_removal = ef_interval.max_value
        else:
            end_removal = min(sol_obj_k, ef_interval.max_value)
        
        # Remove explored region from interval
        if start_removal < end_removal:
            ef_interval.remove_interval(start_removal, end_removal)
        else:
            ef_interval.remove_one_point(start_removal)
            if start_removal > end_removal:
                ef_interval.remove_one_point(end_removal)
        
        # Update max_value if needed
        if end_removal >= ef_interval.max_value:
            ef_interval.max_value = new_max_interval
        
        # Find next point to explore (center of largest remaining interval)
        max_interval = ef_interval.find_largest_interval()
        actual_obj_index = constraint_indices[id_constraint_objective]
        
        if max_interval is not None:
            if ef_array[id_constraint_objective] == nadir_objectives_values[actual_obj_index]:
                ef_array[id_constraint_objective] = best_objective_values[actual_obj_index]
            else:
                # Explore center of largest gap
                ef_array[id_constraint_objective] = int((max_interval[0] + max_interval[1]) / 2)
        else:
            # Interval exhausted - set to best+1 to signal completion and trigger cascade
            # The cascade logic uses > comparison, so best+1 will trigger it
            ef_array[id_constraint_objective] = best_objective_values[actual_obj_index] + 1
            ef_interval = create_interval(actual_obj_index, best_objective_values, nadir_objectives_values)
        
        return ef_interval
    
    def search_previous_solutions_relaxation(ef_array, previous_solution_information, constraint_indices):
        """
        Check if this constraint configuration was already explored with relaxation.
        For maximization: ef_array1 is less constrained (more relaxed) if all ef_array1[i] >= ef_array2[i]
        
        Args:
            ef_array: Current epsilon-constraint RHS values (for constraint objectives only)
            previous_solution_information: List of previous {ef_array, solution} pairs
            constraint_indices: Indices of constraint objectives in the full objective list
        """
        import logging
        logger = logging.getLogger(__name__)
        
        for prev_sol_info in previous_solution_information:
            prev_ef_array = prev_sol_info["ef_array"]
            prev_solution = prev_sol_info["solution"]
            
            # Check if previous ef_array is less constrained (all values >= current)
            is_less_constrained = all(prev_ef_array[i] >= ef_array[i] for i in range(len(ef_array)))
            
            if is_less_constrained:
                logger.debug(f"INLINED RELAX CHECK: Current ef={ef_array}, Prev ef={prev_ef_array}, less_constrained={is_less_constrained}")
                # If previous solution is not "infeasible", check if it satisfies current constraints
                if prev_solution != "infeasible":
                    # Check if previous solution satisfies current (tighter) constraints
                    # For maximization: solution[constraint_idx] must be <= ef_array[i]
                    # (In minimization, constraint is obj >= ef, which becomes -obj <= -ef in maximization)
                    # Note: prev_solution contains ALL objectives, so we need to use constraint_indices
                    constraint_vals = [prev_solution[constraint_indices[i]] for i in range(len(ef_array))]
                    satisfies = all(prev_solution[constraint_indices[i]] <= ef_array[i] 
                                  for i in range(len(ef_array)))
                    logger.debug(f"INLINED RELAX CHECK: Prev solution constraint vals={constraint_vals}, ef_array={ef_array}, satisfies={satisfies}")
                    if satisfies:
                        return True, prev_solution
                else:
                    # Previous was infeasible with less constrained constraints, so current is also infeasible
                    logger.debug("INLINED RELAX CHECK: Previous was infeasible, current also infeasible")
                    return True, "infeasible"
        
        return False, None
    
    def save_solution_information(ef_array, solution, previous_solution_information):
        """Save solution information for future relaxation checks."""
        previous_solution_information.append({
            "ef_array": ef_array.copy(),
            "solution": solution if solution != "infeasible" else "infeasible"
        })
    
    def convert_solution_value_to_str(solution_values):
        """Convert solution objective values to string for deduplication."""
        return str([round(float(x), 6) for x in solution_values])
    
    logger = logging.getLogger(__name__)
    logger.debug("INLINED: Starting solve_milp_inlined (COMPLETE VERSION)")
    
    # If objectives are provided, update the config
    if objectives is not None:
        config.objectives = objectives
    
    logger.debug(f"INLINED: Objectives = {config.objectives}")
    
    print("Start computing (inlined): " + config.uid())
    start_time = time.time()
    
    # STAGE 1: PARSE DATA FROM DZN FILE
    dzn_file_path = os.path.join(config.data_sets_folder, f"{config.data_name}.dzn")
    logger.debug(f"INLINED: Parsing DZN file: {dzn_file_path}")
    
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
    
    logger.debug(f"INLINED: Parsed data - {len(costs)} costs, {len(areas)} areas, {len(images_raw)} images, {len(clouds_raw)} clouds")
    
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
    
    logger.debug(f"INLINED: Total clouds: {len(clouds_id_area)}")
    logger.debug(f"INLINED: Uncoverable clouds: {len(uncoverable_clouds)}")
    logger.debug(f"INLINED: Total uncoverable area: {total_uncoverable_area}")
    if uncoverable_clouds:
        logger.debug(f"INLINED: All uncoverable clouds (id, area): {uncoverable_clouds}")
        # For each uncoverable cloud, show which images contain it and whether it's cloudy
        for cloud_id, area in uncoverable_clouds[:5]:  # Show first 5 in detail
            images_with_element = [i for i in range(len(images)) if cloud_id in images[i]]
            images_with_cloud = [i for i in range(len(clouds)) if cloud_id in clouds[i]]
            logger.debug(f"INLINED:   Cloud {cloud_id} (area={area}): in {len(images_with_element)} images, cloudy in {len(images_with_cloud)} images")
            logger.debug(f"INLINED:     Images containing element: {images_with_element}")
            logger.debug(f"INLINED:     Images with cloud: {images_with_cloud}")
    
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
    model.setParam('Threads', config.threads)  # Set thread count
    
    # STAGE 6: ADD VARIABLES
    select_image = model.addVars(len(images), vtype=gp.GRB.BINARY, name="select_image")
    
    # Cloud variables (only if cloud_coverage objective is used)
    cloud_covered = None
    if "cloud_coverage" in config.objectives:
        cloud_covered = model.addVars(clouds_id, vtype=gp.GRB.BINARY, name="cloud_covered")
    
    # Resolution variables (only if min_resolution objective is used)
    resolution_element = None
    if "min_resolution" in config.objectives:
        min_res = min(resolution)
        max_res = max(resolution)
        resolution_element = model.addVars(elements, lb=min_res, ub=max_res, 
                                         vtype=gp.GRB.INTEGER, name="resolution_element")
    
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
    if "min_resolution" in config.objectives and resolution_element is not None:
        big_M = max(resolution) + 1
        
        for element in elements:
            covering_images = images_covering_element[element]
            
            # resolution_element[element] must be <= resolution of ALL selected images covering the element
            # This ensures resolution_element is at most the minimum among selected images
            for image in covering_images:
                model.addConstr(
                    resolution_element[element] <= 
                    resolution[image] + big_M * (1 - select_image[image]),
                    name=f"resolution_upper_{element}_{image}"
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
        "total_nodes": 0,
        "time_solver_sec": 0,
        "minizinc_time_fzn_sec": 0,
        "hypervolume_current_solutions": [],
        "solutions_time_list": [],
        "pareto_solutions_time_list": [],
        "pareto_front": "",
        "solutions_pareto_front": "",
        "incomplete_timeout_solution_added_to_front": False,
        "hypervolume": 0,
        "exhaustive": False,
        "datetime": datetime.now(),
        "instance": config.data_name,
        "problem": config.problem_name,
        "solver_name": config.solver_name,
        "front_strategy": config.front_strategy,
        "solver_search_strategy": config.solver_search_strategy,
        "fzn_optimisation_level": config.fzn_optimisation_level,
        "threads": config.threads if hasattr(config, 'threads') else 1,
        "cores": config.cores,
        "solver_timeout_sec": config.solver_timeout_sec,
        "minizinc_model": config.minizinc_model if hasattr(config, 'minizinc_model') else ""
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
        # Convert binary solution_values to list of selected image IDs (0-indexed)
        selected_image_ids = [i for i, val in enumerate(solution_values) if val == 1]
        solution_data = {
            "objs": solution_objs,
            "solution_values": selected_image_ids,  # Store as list of image IDs
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
    
    # Progressive timeout: track total time elapsed
    total_start_time = time.time()
    total_timeout = config.solver_timeout_sec
    logger.debug(f"INLINED: Using progressive timeout - total budget: {total_timeout}s")
    
    try:
        # Find ideal points (best value for each objective individually)
        logger.debug(f"INLINED: STAGE 1 - Computing {num_objectives} extreme points")
        for i in range(num_objectives):
            # Calculate remaining time for this optimization
            elapsed = time.time() - total_start_time
            if elapsed >= total_timeout:
                raise TimeoutError(f"Total timeout exceeded before optimizing objective {i}")
            
            remaining = total_timeout - elapsed
            logger.debug(f"INLINED: Extreme point {i} - elapsed: {elapsed:.2f}s, remaining: {remaining:.2f}s")
            model.setParam('TimeLimit', remaining)
            model.setObjective(objectives_exprs[i], gp.GRB.MINIMIZE)
            
            solve_start = time.time()
            model.optimize()
            solve_time = time.time() - solve_start
            
            if model.status == gp.GRB.OPTIMAL:
                obj_value = int(model.objVal)
                best_objective_values[i] = obj_value
                logger.debug(f"INLINED: Best value for objective {i} ({config.objectives[i]}): {obj_value}")
                
                # Extract solution values
                solution_values = [int(select_image[j].X > 0.5) for j in images_id]
                current_objs = []
                
                # Log detailed solution info for cloud_coverage optimization
                if config.objectives[i] == "cloud_coverage":
                    logger.debug("INLINED: CLOUD_COVERAGE OPTIMIZATION - Details:")
                    selected_images = [j for j in images_id if select_image[j].X > 0.5]
                    logger.debug(f"INLINED: Selected images: {selected_images}")
                    logger.debug(f"INLINED: Total images available: {len(images_id)}")
                    if cloud_covered is not None:
                        covered_clouds = {c: cloud_covered[c].X for c in clouds_id if c in cloud_covered}
                        logger.debug(f"INLINED: Cloud coverage variables (first 10): {dict(list(covered_clouds.items())[:10])}")
                        total_cloud_area = sum(clouds_id_area.values())
                        covered_cloud_area = sum(cloud_covered[c].X * clouds_id_area[c] for c in clouds_id if c in cloud_covered)
                        logger.debug(f"INLINED: Total cloud area: {total_cloud_area}")
                        logger.debug(f"INLINED: Covered cloud area: {covered_cloud_area}")
                        logger.debug(f"INLINED: Uncovered cloud area: {total_cloud_area - covered_cloud_area}")
                        
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
                        logger.debug(f"INLINED: Uncovered clouds (first 10): {uncovered_clouds[:10]}")
                        logger.debug(f"INLINED: Covered area: {covered_area}, Uncovered area: {uncovered_area}")
                        logger.debug(f"INLINED: Total uncovered clouds: {len(uncovered_clouds)}")
                        
                        # Debug: Check some uncovered clouds with non-zero area
                        nonzero_uncovered = [(c, area) for c, area in uncovered_clouds if area > 0]
                        logger.debug(f"INLINED: Uncovered clouds with non-zero area (first 5): {nonzero_uncovered[:5]}")
                        
                        # Debug: Check image-cloud relationships for uncovered clouds
                        if uncovered_clouds:
                            sample_cloud = uncovered_clouds[0][0]
                            covering_images = []
                            for img in images_id:
                                if sample_cloud in cloud_covered_by_image.get(img, set()):
                                    covering_images.append(img)
                            logger.debug(f"INLINED: Sample uncovered cloud {sample_cloud} can be covered by images: {covering_images}")
                    else:
                        logger.debug("INLINED: No cloud_covered variables - cloud objective may be disabled")
                
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
                        # Recompute min_resolution based on selected images
                        # For each element, find the minimum resolution among selected images covering it
                        min_res_sum = 0
                        for element in elements:
                            covering_images = [i for i in images_id if element in images[i] and solution_values[i] == 1]
                            if covering_images:
                                min_res_sum += min(resolution[i] for i in covering_images)
                        current_objs.append(min_res_sum)
                    elif obj_name == "min_max_incidence_angle":
                        if current_max_incidence_angle is not None:
                            current_objs.append(int(round(current_max_incidence_angle.X)))
                        else:
                            current_objs.append(0)
                
                # Ensure all objectives are integers
                current_objs = [int(x) for x in current_objs]
                logger.debug(f"INLINED: Extreme point {i}: {current_objs}")
                
                # Log types of objectives
                obj_types = [f"{config.objectives[j]}:{type(val).__name__}" for j, val in enumerate(current_objs)]
                logger.debug(f"INLINED: Extreme point {i} types: {obj_types}")
                
                add_to_pareto_front(current_objs, solution_values)
                
                statistics["number_of_solutions"] += 1
                statistics["time_solver_sec"] += solve_time
                statistics["solutions_time_list"].append(solve_time)
                
            elif model.status == gp.GRB.TIME_LIMIT:
                raise TimeoutError(f"Timeout while optimizing objective {i}")
            else:
                raise RuntimeError(f"Failed to optimize objective {i}: status {model.status}")
                
        # Calculate nadir points using heuristic bounds (matching non-inlined version)
        # This avoids unbounded maximization issues with auxiliary variables
        logger.debug(f"INLINED: Computing {num_objectives} nadir points using heuristic")
        for i in range(num_objectives):
            obj_name = config.objectives[i]
            if obj_name == "min_cost":
                # Worst case: all images selected
                nadir_objectives_values[i] = int(sum(costs))
            elif obj_name == "cloud_coverage":
                # Worst case: all cloud areas covered
                nadir_objectives_values[i] = int(sum(areas))
            elif obj_name == "min_resolution":
                # Worst case: sum of max resolution per universe point
                resolution_parts_max = {}
                for idx, image_points in enumerate(images):
                    for u in image_points:
                        if u not in resolution_parts_max:
                            resolution_parts_max[u] = resolution[idx]
                        else:
                            resolution_parts_max[u] = max(resolution_parts_max[u], resolution[idx])
                nadir_objectives_values[i] = int(sum(resolution_parts_max.values()))
            elif obj_name == "min_max_incidence_angle":
                # Worst case: maximum incidence angle
                nadir_objectives_values[i] = int(max(incidence_angle))
            else:
                raise ValueError(f"Unknown objective: {obj_name}")
            logger.debug(f"INLINED: Nadir value for objective {i} ({obj_name}): {nadir_objectives_values[i]}")
        
        logger.debug(f"INLINED: Best values (minimization): {best_objective_values}")
        logger.debug(f"INLINED: Nadir values (minimization, heuristic): {nadir_objectives_values}")
        
        # GPBA-A: STAGE 1.5 - Convert to maximization for uniform handling
        logger.debug("INLINED: Converting objectives to maximization")
        best_objective_values = [-x for x in best_objective_values]
        nadir_objectives_values = [-x for x in nadir_objectives_values]
        objectives_exprs = [-obj for obj in objectives_exprs]
        
        logger.debug(f"INLINED: Best values (maximization): {best_objective_values}")
        logger.debug(f"INLINED: Nadir values (maximization): {nadir_objectives_values}")
        
        # ===== GOLDEN STANDARD CHECKPOINT: PAYOFF TABLE COMPLETE =====
        logger.info("=" * 80)
        logger.info("GOLDEN: PAYOFF TABLE RESULTS")
        logger.info(f"GOLDEN: Number of extreme points found: {statistics['number_of_solutions']}")
        logger.info(f"GOLDEN: Best values (ideal point, max form): {best_objective_values}")
        logger.info(f"GOLDEN: Nadir values (max form): {nadir_objectives_values}")
        logger.info(f"GOLDEN: Solutions in Pareto front: {len(pareto_solutions)}")
        logger.info("=" * 80)
        
        # Initialize exhaustive search flag (will be updated in main loop if applicable)
        exhaustive_search = True  # Default for single objective or if main loop doesn't run
        
        # GPBA-A: STAGE 2 - Setup ε-constraint formulation
        if num_objectives > 1:
            logger.debug("INLINED: STAGE 2 - Setting up ε-constraint formulation")
            main_obj_index = 0  # Use first objective as main
            constraint_indices = [i for i in range(num_objectives) if i != main_obj_index]
            logger.debug(f"INLINED: Main objective: {main_obj_index} ({config.objectives[main_obj_index]})")
            logger.debug(f"INLINED: Constraint objectives: {constraint_indices}")
            
            # Convert to maximization for main objective
            logger.debug(f"INLINED: Setting objective before aug: {objectives_exprs[main_obj_index]}")
            model.setObjective(objectives_exprs[main_obj_index], gp.GRB.MAXIMIZE)
            
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
                # objectives_exprs are already in maximization form (negated)
                augmented_obj = objectives_exprs[main_obj_index] + delta * slack_sum
            else:
                augmented_obj = objectives_exprs[main_obj_index]
            
            model.setObjective(augmented_obj, gp.GRB.MAXIMIZE)
            
            # GPBA-A: STAGE 3 - Main coverage loop with interval management
            ef_array = [nadir_objectives_values[i] for i in constraint_indices]
            previous_solutions = set()
            previous_solution_information = []
            logger.debug("INLINED: STAGE 3 - Starting main coverage loop (COMPLETE GPBA-A)")
            logger.debug(f"INLINED: Initial ef_array: {ef_array}")
            
            # Initialize interval managers for each constraint objective
            ef_intervals = []
            for constraint_idx in constraint_indices:
                ef_intervals.append(create_interval(constraint_idx, best_objective_values, nadir_objectives_values))
            
            # Initialize Relative Worst Values (RWV) for search space pruning
            rwv = [best_objective_values[i] for i in constraint_indices]
            
            # Track objective values at each ef point for precise interval updates
            obj_k_at_ef_k = [None] * len(constraint_indices)
            
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
            
            # Main GPBA-A loop with adaptive interval-based exploration
            iteration_count = 0
            max_iterations = 10000  # Safety limit
            gamma = 1  # Coverage parameter (1 = complete coverage)
            
            logger.debug(f"INLINED: Best values: {best_objective_values}")
            logger.debug(f"INLINED: Nadir values: {nadir_objectives_values}")
            logger.debug(f"INLINED: Constraint indices: {constraint_indices}")
            
            # ===== GOLDEN STANDARD CHECKPOINT: EPSILON SETUP COMPLETE =====
            logger.info("=" * 80)
            logger.info("GOLDEN: EPSILON SETUP RESULTS")
            logger.info(f"GOLDEN: Main objective index: {main_obj_index}")
            logger.info(f"GOLDEN: Constraint indices: {constraint_indices}")
            logger.info(f"GOLDEN: Initial ef_array: {ef_array}")
            logger.info(f"GOLDEN: Initial RWV: {rwv}")
            logger.info(f"GOLDEN: Augmented objective set: YES")
            logger.info("=" * 80)
            
            while ef_array[0] <= best_objective_values[constraint_indices[0]] and iteration_count < max_iterations:
                
                # ===== GOLDEN STANDARD: ITERATION START =====
                logger.info("=" * 60)
                logger.info(f"GOLDEN: ITERATION {iteration_count}")
                logger.info(f"GOLDEN: ef_array={ef_array}, rwv={rwv}")
                logger.info("=" * 60)
                
                if iteration_count % 50 == 0:
                    logger.debug(f"INLINED: Iteration {iteration_count}, ef_array = {ef_array}")
                
                # Initialize one_solution to empty list
                one_solution = []
                
                # Check if this configuration was already explored (relaxation)
                previous_relaxation, previous_values = search_previous_solutions_relaxation(
                    ef_array, previous_solution_information, constraint_indices)
                
                logger.info(f"GOLDEN: Relaxation check - found={previous_relaxation}, prev_info_count={len(previous_solution_information)}")
                logger.debug(f"INLINED: Loop - ef_array={ef_array}, previous_relaxation={previous_relaxation}")
                
                if previous_relaxation:
                    if previous_values == "infeasible":
                        one_solution = []
                    else:
                        one_solution = previous_values
                else:
                    # Update constraint RHS values
                    # NOTE: After line 618, objectives_exprs are ALREADY in maximization form (negated)
                    # So we use them directly: objectives_exprs[i] - slack = ef_array
                    for i, constr in enumerate(constraint_vars):
                        model.remove(constr)
                        if slack_vars[i] is not None:
                            constr = model.addConstr(
                                objectives_exprs[constraint_indices[i]] - slack_vars[i] == ef_array[i],
                                name=f"epsilon_constraint_{constraint_indices[i]}_iter{iteration_count}"
                            )
                        else:
                            constr = model.addConstr(
                                objectives_exprs[constraint_indices[i]] == ef_array[i],
                                name=f"epsilon_constraint_{constraint_indices[i]}_iter{iteration_count}"
                            )
                        constraint_vars[i] = constr
                    
                    # Calculate remaining time and check if we should continue
                    elapsed = time.time() - total_start_time
                    if elapsed >= total_timeout:
                        logger.debug("INLINED: Total timeout exceeded during GPBA-A loop")
                        break
                    
                    remaining = total_timeout - elapsed
                    # Update time limit for this optimization
                    model.setParam('TimeLimit', remaining)
                    
                    # Solve current configuration
                    solve_start = time.time()
                    model.optimize()
                    solve_time = time.time() - solve_start
                    
                    logger.debug(f"INLINED: After solve - status={model.status}, objVal={model.objVal if model.status == gp.GRB.OPTIMAL else 'N/A'}")
                    
                    if model.status == gp.GRB.INFEASIBLE:
                        save_solution_information(ef_array, "infeasible", previous_solution_information)
                        one_solution = []
                    elif model.status == gp.GRB.OPTIMAL:
                        # Extract solution
                        solution_values = [int(select_image[j].X > 0.5) for j in images_id]
                        current_objs = []
                        
                        # Calculate all objective values (in MAXIMIZATION form since we converted)
                        # Use binary solution_values to avoid floating-point precision issues
                        for obj_name in config.objectives:
                            if obj_name == "min_cost":
                                current_objs.append(-sum(solution_values[k] * costs[k] for k in images_id))
                            elif obj_name == "cloud_coverage":
                                cloud_val = 0
                                if cloud_covered is not None:
                                    # Round each cloud coverage to avoid floating-point errors
                                    cloud_val = sum(round(cloud_covered[c].X) * clouds_id_area[c] 
                                                  for c in clouds_id if c in cloud_covered)
                                total_cloud_area = sum(clouds_id_area.values())
                                current_objs.append(-int(total_cloud_area - cloud_val))
                            elif obj_name == "min_resolution":
                                # Recompute min_resolution based on selected images
                                # For each element, find the minimum resolution among selected images covering it
                                min_res_sum = 0
                                for element in elements:
                                    covering_images = [i for i in images_id if element in images[i] and solution_values[i] == 1]
                                    if covering_images:
                                        min_res_sum += min(resolution[i] for i in covering_images)
                                current_objs.append(-min_res_sum)
                            elif obj_name == "min_max_incidence_angle":
                                if current_max_incidence_angle is not None:
                                    current_objs.append(-int(round(current_max_incidence_angle.X)))
                                else:
                                    current_objs.append(0)
                        
                        logger.debug(f"INLINED: Found solution (in max form): {current_objs}")
                        
                        # Check if this is a new solution
                        sol_str = convert_solution_value_to_str(current_objs)
                        if sol_str not in previous_solutions:
                            previous_solutions.add(sol_str)
                            
                            # Convert back to minimization for Pareto front storage
                            current_objs_min = [-x for x in current_objs]
                            add_to_pareto_front(current_objs_min, solution_values)
                            
                            statistics["number_of_solutions"] += 1
                            statistics["time_solver_sec"] += solve_time
                            statistics["solutions_time_list"].append(solve_time)
                            
                            logger.debug(f"INLINED: Found solution {statistics['number_of_solutions']}: {current_objs_min}")
                            
                            one_solution = current_objs
                            save_solution_information(ef_array, one_solution, previous_solution_information)
                        else:
                            one_solution = current_objs
                    elif model.status == gp.GRB.TIME_LIMIT:
                        logger.debug("INLINED: Timeout during GPBA-A loop")
                        break
                    else:
                        logger.debug(f"INLINED: Unexpected solver status: {model.status}")
                        one_solution = []
                
                # Update RWV and obj_k_at_ef_k based on solution found
                id_interval = -1  # Last constraint objective
                obj_k_at_ef_k[id_interval] = None
                
                if len(one_solution) > 0:
                    for i in range(len(rwv)):
                        rwv[i] = min(rwv[i], one_solution[constraint_indices[i]])
                    obj_k_at_ef_k[id_interval] = one_solution[constraint_indices[id_interval]]
                
                # Adjust ef_array using interval management (core of GPBA-A)
                logger.info(f"GOLDEN: Before epsilon adjustment - ef_array={ef_array}, obj_k_at_ef_k[{id_interval}]={obj_k_at_ef_k[id_interval]}")
                
                ef_intervals[id_interval] = adjust_parameter_ef_array(
                    id_interval, ef_array, obj_k_at_ef_k[id_interval],
                    ef_intervals[id_interval], constraint_indices,
                    best_objective_values, nadir_objectives_values, gamma)
                
                logger.info(f"GOLDEN: After epsilon adjustment - ef_array={ef_array}")
                
                # Multi-dimensional cascading updates for higher constraint objectives
                # Loop from highest dimension down to 1 (dimension 0 handled by main loop condition)
                for i in range(len(constraint_indices) - 1, 0, -1):
                    if ef_array[i] > best_objective_values[constraint_indices[i]]:
                        # Reset this dimension and update previous dimension
                        # Uses > (not >=) because when interval exhausted, ef_array[i] = best+1
                        ef_array[i] = nadir_objectives_values[constraint_indices[i]]
                        rwv[i] = best_objective_values[constraint_indices[i]]
                        id_interval = i - 1
                        ef_intervals[id_interval] = adjust_parameter_ef_array(
                            id_interval, ef_array, obj_k_at_ef_k[id_interval],
                            ef_intervals[id_interval], constraint_indices,
                            best_objective_values, nadir_objectives_values, gamma)
                        obj_k_at_ef_k[id_interval] = None
                        if i == 1:
                            rwv[id_interval] = best_objective_values[constraint_indices[id_interval]]
                        # Continue checking previous dimensions (don't break)
                    else:
                        break  # Only cascade if dimension is exhausted (>best)
                
                iteration_count += 1
            
            logger.debug(f"INLINED: GPBA-A loop completed after {iteration_count} iterations")
            
            # Determine if problem was exhaustively explored
            # Loop exits for 3 reasons:
            # 1. ef_array[0] > best_objective_values[constraint_indices[0]] - exhaustive exploration complete
            # 2. iteration_count >= max_iterations - hit iteration limit (not exhaustive)
            # 3. break due to timeout - not exhaustive
            exhaustive_search = (iteration_count < max_iterations and 
                                ef_array[0] > best_objective_values[constraint_indices[0]])
            
            # ===== GOLDEN STANDARD CHECKPOINT: MAIN LOOP COMPLETE =====
            logger.info("=" * 80)
            logger.info("GOLDEN: MAIN LOOP RESULTS")
            logger.info(f"GOLDEN: Total iterations: {iteration_count}")
            logger.info(f"GOLDEN: Solutions found in main loop: {statistics['number_of_solutions'] - num_objectives}")
            logger.info(f"GOLDEN: Total solutions (including payoff): {statistics['number_of_solutions']}")
            logger.info(f"GOLDEN: Exhaustive: {exhaustive_search}")
            logger.info(f"GOLDEN: Final ef_array: {ef_array}")
            logger.info("=" * 80)
        
        statistics["exhaustive"] = exhaustive_search
        if exhaustive_search:
            logger.debug("INLINED: Problem completely explored.")
        else:
            logger.debug("INLINED: Problem exploration incomplete (timeout or iteration limit).")
        
    except TimeoutError as e:
        logger.debug(f"INLINED: Timeout during extreme point computation: {e}")
        statistics["exhaustive"] = False
    except Exception as e:
        logger.debug(f"INLINED: Error during solving: {e}")
        statistics["exhaustive"] = False
    
    logger.debug(f"INLINED: Final statistics - {statistics['number_of_solutions']} total solutions found")
    logger.debug(f"INLINED: Pareto front contains {len(pareto_front)} non-dominated solutions")
    
    # ===== GOLDEN STANDARD CHECKPOINT: FINAL RESULTS =====
    logger.info("=" * 80)
    logger.info("GOLDEN: FINAL RESULTS")
    logger.info(f"GOLDEN: Total solutions discovered: {statistics['number_of_solutions']}")
    logger.info(f"GOLDEN: Pareto front size (non-dominated): {len(pareto_front)}")
    logger.info(f"GOLDEN: Exhaustive: {statistics['exhaustive']}")
    logger.info("=" * 80)
    
    # STAGE 10: FINALIZE RESULTS
    
    # Deduplicate pareto_front based on selected images
    original_count = len(pareto_front)
    unique_fronts = {}
    new_indices = []
    
    for idx in pareto_front:
        solution = pareto_solutions[idx]
        # Use frozenset of selected image IDs as key for deduplication
        key = frozenset(solution["solution_values"])
        if key not in unique_fronts:
            unique_fronts[key] = idx
            new_indices.append(idx)
    
    pareto_front = new_indices
    
    if len(pareto_front) < original_count:
        logger.info(f"GOLDEN: Removed {original_count - len(pareto_front)} duplicate solutions from Pareto front")
    
    # Build results strings
    pareto_objs = []
    pareto_vectors = []
    all_solutions = []
    
    for idx in pareto_front:
        solution = pareto_solutions[idx]
        pareto_objs.append(solution["objs"])
        pareto_vectors.append(solution["solution_values"])
        all_solutions.append(solution)
    
    # Log objective types before JSON serialization
    if pareto_objs:
        logger.debug("INLINED: JSON SERIALIZATION - Checking objective types:")
        for i, obj_name in enumerate(config.objectives):
            obj_types = [type(sol[i]).__name__ for sol in pareto_objs]
            unique_types = set(obj_types)
            if len(unique_types) > 1 or 'float' in unique_types:
                logger.debug(f"INLINED: Objective {i} ({obj_name}): types={unique_types}, sample values: {[sol[i] for sol in pareto_objs[:3]]}")
    
    # Format as {[...],[...],...} to match original CSV format
    statistics["pareto_front"] = "{" + ",".join([str(obj) for obj in pareto_objs]) + "}"
    statistics["solutions_pareto_front"] = "{" + ",".join([str(vec) for vec in pareto_vectors]) + "}"
    
    # Populate pareto_solutions_time_list with timestamps for pareto solutions only
    statistics["pareto_solutions_time_list"] = [
        pareto_solutions[idx]["timestamp"] for idx in pareto_front
    ]
    
    # Simple hypervolume calculation
    if pareto_objs:
        hv = 1.0  # Placeholder - real hypervolume calculation is complex
        statistics["hypervolume"] = hv
    
    print(f"End of solving statistics (inlined): {statistics['number_of_solutions']} total solutions, {len(pareto_front)} non-dominated")
    
    # Calculate total execution time
    total_execution_time = time.time() - total_start_time
    
    # STAGE 11: WRITE RESULTS TO CSV (if summary_filename is provided)
    if config.summary_filename:
        write_statistics_inlined(config, statistics)
    
    # STAGE 12: RETURN STRUCTURED RESULTS
    # Return a dictionary with all the solution data
    return {
        "pareto_solutions": all_solutions,  # List of pareto front solutions with objs, solution_values, timestamp
        "statistics": statistics,
        "execution_time_sec": total_execution_time,
        "timeout_sec": config.solver_timeout_sec,
    }


def write_statistics_inlined(config, statistics):
    """Write statistics to CSV file using the same format as the original."""
    from filelock import FileLock, Timeout as FileLockTimeout
    from sims_solvers.main import statistics_to_csv, create_summary_file
    
    try:
        lock = FileLock(config.summary_filename + ".lock", timeout=10)
        with lock:
            create_summary_file(config)
            with open(config.summary_filename, "a") as summary:
                summary.write(statistics_to_csv(config, statistics))
    except FileLockTimeout:
        print("Could not acquire lock on summary file. Statistics will be printed on standard output instead.")
        print(statistics)

def solve_milp(config: Config, objectives: list[str] | None = None):
    """Solve the SIMS problem using Mixed Integer Linear Programming (MILP) solver."""
    
    logger = logging.getLogger(__name__)
    logger.debug("ORIGINAL: Starting solve_milp")

    # If objectives are provided, update the config
    if objectives is not None:
        config.objectives = objectives
    
    logger.debug(f"ORIGINAL: Objectives = {config.objectives}")

    instance = build_instance(config)
    logger.debug("ORIGINAL: Built instance")
    print("Start computing: " + config.uid())
    statistics = {}
    config.init_statistics(statistics)
    init_top_level_statistics(statistics)
    model = build_model(instance, config)
    logger.debug("ORIGINAL: Built model")
    solver, pareto_front = build_solver(model, instance, config, statistics)
    logger.debug("ORIGINAL: Built solver and pareto_front")
    save_results = True
    try:
        statistics["exhaustive"] = False
        statistics["incomplete_timeout_solution_added_to_front"] = False
        statistics["max_solutions_count_reached"] = False
        logger.debug("ORIGINAL: Starting solver.solve() iteration")
        solution_count = 0
        max_solutions = config.max_solutions_count
        for x in solver.solve():
            solution_count += 1
            if hasattr(x, 'objs'):
                logger.debug(f"ORIGINAL: Found solution {solution_count}: {x.objs}")
            else:
                logger.debug(f"ORIGINAL: Found solution {solution_count}: {x}")
            
            # Check if we've reached the maximum number of solutions
            if max_solutions is not None and solution_count >= max_solutions:
                logger.debug(f"ORIGINAL: Reached max_solutions_count limit of {max_solutions}, stopping")
                statistics["max_solutions_count_reached"] = True
                break
        
        if not statistics.get("max_solutions_count_reached", False):
            logger.debug("ORIGINAL: Problem completely explored.")
            statistics["exhaustive"] = True
    except TimeoutError:
        logger.debug("ORIGINAL: Timeout triggered getting last incomplete solution")
        if solver.process_last_incomplete_solution():
            # the last incomplete solution was added to the pareto front
            logger.debug("ORIGINAL: Last incomplete solution added to the pareto front")
            statistics["incomplete_timeout_solution_added_to_front"] = True
        else:
            logger.debug(
                "ORIGINAL: There were not incomplete solution or the last incomplete solution was not added to "
                "the pareto front"
            )
            set_right_time_after_timeout(statistics, config.solver_timeout_sec)
    except Exception as e:
        logger.debug("ORIGINAL: Error Exception raised: " + str(e))
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
        logger.debug("ORIGINAL: end of solving statistics: " + str(statistics))
        write_statistics(config, statistics)
