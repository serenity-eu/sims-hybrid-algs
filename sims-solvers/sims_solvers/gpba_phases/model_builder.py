"""Model builder helper for GPBA phases testing.

Extracts model building logic from solve_milp_inlined() to allow tests to work
with real SIMS problems without duplicating complex model construction code.

This is extracted from solve.py lines 215-450 for reuse in phase tests.
"""

import gurobipy as gp
import re
import os
import logging

logger = logging.getLogger(__name__)


def build_gurobi_model_from_config(config):
    """
    Build Gurobi model from Config object pointing to a .dzn file.
    
    Args:
        config: Config object with instance path, objectives, threads, etc.
        
    Returns:
        Tuple of (model, objective_exprs, problem_data_dict)
        
        - model: Configured Gurobi model with variables and constraints
        - objective_exprs: List of objective expressions in order
        - problem_data_dict: Dict with all problem data (costs, areas, images, etc.)
    
    Extracted from solve_milp_inlined lines 215-450.
    """
    # Get DZN path from config
    if hasattr(config, 'instance'):
        dzn_file_path = config.instance
    else:
        dzn_file_path = os.path.join(config.data_sets_folder, f"{config.data_name}.dzn")
    
    logger.info(f"Building Gurobi model from: {dzn_file_path}")
    
    if not os.path.exists(dzn_file_path):
        raise FileNotFoundError(f"DZN file not found: {dzn_file_path}")
    
    # Parse .dzn file
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
    
    # Convert 1-indexed to 0-indexed
    images = [[x - 1 for x in img_set] for img_set in images_raw]
    clouds = [[x - 1 for x in cloud_set] for cloud_set in clouds_raw]
    
    # Build cloud coverage relationships
    cloud_covered_by_image = {}
    clouds_id_area = {}
    
    for i in range(len(clouds)):
        image_cloud_set = set(clouds[i])
        for cloud_id in image_cloud_set:
            clouds_id_area[cloud_id] = areas[cloud_id]
            for j in range(len(images)):
                if i != j and cloud_id in images[j] and cloud_id not in clouds[j]:
                    if j in cloud_covered_by_image:
                        cloud_covered_by_image[j].add(cloud_id)
                    else:
                        cloud_covered_by_image[j] = {cloud_id}
    
    # Debug: Check for uncoverable clouds
    coverable_clouds = set()
    for img_clouds in cloud_covered_by_image.values():
        coverable_clouds.update(img_clouds)
    
    uncoverable_clouds_list = []
    for cloud_id in clouds_id_area:
        if cloud_id not in coverable_clouds:
            uncoverable_clouds_list.append((cloud_id, clouds_id_area[cloud_id]))
    
    logger.critical(f"MODEL_BUILDER: Total clouds with area: {len(clouds_id_area)}")
    logger.critical(f"MODEL_BUILDER: Coverable clouds: {len(coverable_clouds)}")
    logger.critical(f"MODEL_BUILDER: Uncoverable clouds: {len(uncoverable_clouds_list)}")
    logger.critical(f"MODEL_BUILDER: Uncoverable cloud IDs: {[c for c, a in uncoverable_clouds_list]}")
    
    # For first 3 uncoverable clouds, show detail
    for cloud_id, area in uncoverable_clouds_list[:3]:
        images_with_element = [i for i in range(len(images)) if cloud_id in images[i]]
        images_with_cloud = [i for i in range(len(clouds)) if cloud_id in clouds[i]]
        logger.critical(f"MODEL_BUILDER:   Cloud {cloud_id} (area={area}): in {len(images_with_element)} images, cloudy in {len(images_with_cloud)} images")
        logger.critical(f"MODEL_BUILDER:     Images with element: {images_with_element}")
        logger.critical(f"MODEL_BUILDER:     Images with cloud: {images_with_cloud}")
    
    # Build image-element mappings
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
    
    # Create Gurobi model
    model = gp.Model("SIMSModelPhases")
    model.setParam('OutputFlag', 0)
    if hasattr(config, 'threads'):
        model.setParam('Threads', config.threads)
    
    # Add variables
    select_image = model.addVars(len(images), vtype=gp.GRB.BINARY, name="select_image")
    
    # Cloud variables
    cloud_covered = None
    if "cloud_coverage" in config.objectives:
        cloud_covered = model.addVars(clouds_id, vtype=gp.GRB.BINARY, name="cloud_covered")
    
    # Resolution variables
    resolution_element = None
    auxiliary_resolution = None
    if "min_resolution" in config.objectives:
        min_res = min(resolution)
        max_res = max(resolution)
        resolution_element = model.addVars(elements, lb=min_res, ub=max_res, 
                                         vtype=gp.GRB.INTEGER, name="resolution_element")
        auxiliary_resolution = {}
        for element in elements:
            auxiliary_resolution[element] = {}
            for image in images_covering_element[element]:
                auxiliary_resolution[element][image] = model.addVar(
                    vtype=gp.GRB.BINARY, name=f"aux_res_{element}_{image}")
    
    # Incidence angle variables
    effective_incidence_angle = None
    current_max_incidence_angle = None
    if "min_max_incidence_angle" in config.objectives:
        effective_incidence_angle = model.addVars(images_id, vtype=gp.GRB.INTEGER,
                                                name="effective_incidence_angle")
        current_max_incidence_angle = model.addVar(vtype=gp.GRB.INTEGER, 
                                                 name="max_allowed_incidence_angle")
    
    # Add constraints
    # Coverage constraint
    for element in elements:
        model.addConstr(
            gp.quicksum(select_image[i] for i in images_id if element in images[i]) >= 1,
            name=f"coverage_{element}"
        )
    
    # Cloud constraints
    if "cloud_coverage" in config.objectives and cloud_covered is not None:
        for cloud in clouds_id:
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
                model.addConstr(cloud_covered[cloud] == 0, name=f"cloud_uncoverable_{cloud}")
    
    # Resolution constraints
    if "min_resolution" in config.objectives and resolution_element is not None and auxiliary_resolution is not None:
        big_M = max(resolution) + 1
        for element in elements:
            covering_images = images_covering_element[element]
            if element in auxiliary_resolution:
                total_aux = len(auxiliary_resolution[element])
                model.addConstr(
                    gp.quicksum(auxiliary_resolution[element][i] for i in covering_images) == total_aux - 1,
                    name=f"aux_resolution_sum_{element}"
                )
                for image in covering_images:
                    model.addConstr(
                        resolution_element[element] >= 
                        resolution[image] * select_image[image] +
                        big_M * (1 - select_image[image]) -
                        2 * big_M * auxiliary_resolution[element][image],
                        name=f"resolution_linearization_{element}_{image}"
                    )
    
    # Incidence angle constraints
    if "min_max_incidence_angle" in config.objectives and effective_incidence_angle is not None:
        for image in images_id:
            model.addConstr((select_image[image] == 0) >> 
                           (effective_incidence_angle[image] == 0),
                           name=f"incidence_angle_not_selected_{image}")
            model.addConstr((select_image[image] == 1) >> 
                           (effective_incidence_angle[image] == incidence_angle[image]),
                           name=f"incidence_angle_selected_{image}")
        if current_max_incidence_angle is not None:
            for image in images_id:
                model.addConstr(current_max_incidence_angle >= effective_incidence_angle[image],
                               name=f"max_incidence_angle_{image}")
    
    # Define objectives
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
    
    objectives_exprs = []
    for obj_name in config.objectives:
        if obj_name in available_objectives:
            objectives_exprs.append(available_objectives[obj_name]())
        else:
            raise ValueError(f"Unknown objective: {obj_name}")
    
    # Prepare problem data dict for phases
    problem_data = {
        'images_id': images_id,
        'select_image': select_image,
        'costs': costs,
        'areas': areas,
        'clouds': clouds,
        'resolution': resolution,
        'incidence_angle': incidence_angle,
        'cloud_covered': cloud_covered,
        'clouds_id_area': clouds_id_area,
        'cloud_covered_by_image': cloud_covered_by_image,
        'resolution_element': resolution_element,
        'current_max_incidence_angle': current_max_incidence_angle,
        'images': images,
        'elements': elements,
    }
    
    logger.info(f"Model built: {len(images)} images, {len(elements)} elements, {len(config.objectives)} objectives")
    
    return model, objectives_exprs, problem_data
