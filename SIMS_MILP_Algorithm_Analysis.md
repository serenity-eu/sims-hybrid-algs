# SIMS Multi-Objective MILP Algorithm Analysis

## Abstract

This document provides an in-depth analysis of the Satellite Image Mosaic Selection (SIMS) multi-objective Mixed Integer Linear Programming (MILP) solver implementation. The solver employs the Coverage Grid Point Based Representation Algorithm (GPBA-A) to generate complete Pareto front representations for the multi-objective optimization problem. This analysis includes detailed pseudocode, algorithmic stages, and implementation details necessary for replicating the algorithm in different frameworks.

## 1. Introduction

The SIMS problem is a multi-objective optimization challenge that involves selecting optimal subsets of satellite images while balancing multiple conflicting objectives such as cost minimization, cloud coverage reduction, resolution optimization, and incidence angle minimization. The solver implements the GPBA-A algorithm using the ε-constraint method with augmentation for efficient Pareto front exploration.

## 2. Problem Formulation

### 2.1 SIMS Problem Definition

The SIMS problem is formulated as a multi-objective integer linear programming problem with the following components:

#### Decision Variables:
- `select_image[i]` ∈ {0,1}: Binary variable indicating whether image i is selected
- `cloud_covered[c]` ∈ {0,1}: Binary variable indicating whether cloud c is covered
- `resolution_element[e]` ∈ Z⁺: Integer variable representing resolution for element e
- `effective_incidence_angle[i]` ∈ Z⁺: Integer variable for effective incidence angle of image i
- `current_max_incidence_angle` ∈ Z⁺: Integer variable for maximum incidence angle across all selected images

#### Objectives:
1. **Cost Minimization**: min ∑ᵢ select_image[i] × costs[i]
2. **Cloud Coverage**: min (total_cloud_area - ∑c cloud_covered[c] × area_clouds[c])
3. **Resolution**: min ∑e resolution_element[e]
4. **Max Incidence Angle**: min current_max_incidence_angle

#### Constraints:
1. **Coverage Constraint**: ∀e ∈ elements: ∑ᵢ∈images_covering[e] select_image[i] ≥ 1
2. **Cloud Constraints**: ∀c ∈ clouds: cloud_covered[c] ≤ ∑ᵢ∈cloud_covering_images[c] select_image[i]
3. **Resolution Constraints**: Complex linearization involving auxiliary binary variables
4. **Incidence Angle Constraints**: Using indicator constraints for effective angle calculation

## 3. Algorithm Architecture Overview

The SIMS MILP solver consists of several interconnected components working together to generate the complete Pareto front:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Config       │    │   Instance      │    │     Model       │
│  - Objectives   │───▶│   - Data        │───▶│  - Variables    │
│  - Solver Type  │    │   - Images      │    │  - Constraints  │
│  - Timeout      │    │   - Clouds      │    │  - Objectives   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              MOWithFrontGenerator                               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │ FrontGenerator  │───▶│  GurobiSolver   │───▶│ ParetoFront  │ │
│  │ (GPBA-A/CGP)    │    │                 │    │              │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 4. Detailed Algorithm Analysis

### 4.1 Main Solving Process

The main solving process follows this high-level structure:

```python
def solve_milp(config: Config, objectives: list[str] | None = None):
    """Main MILP solving function"""
    # STAGE 1: Initialization
    instance = build_instance(config)
    statistics = initialize_statistics()
    
    # STAGE 2: Model Construction  
    model = build_model(instance, config)
    
    # STAGE 3: Solver Setup
    solver, pareto_front = build_solver(model, instance, config, statistics)
    
    # STAGE 4: Pareto Front Generation
    try:
        for solution in solver.solve():
            pass  # Solutions are processed internally
        statistics["exhaustive"] = True
    except TimeoutError:
        handle_timeout(solver, statistics, config.solver_timeout_sec)
    
    # STAGE 5: Results Processing
    finalize_results(statistics, pareto_front)
```

### 4.2 Instance Building Process

The instance building process transforms raw data into structured format:

```python
def build_instance(config: Config) -> InstanceSIMS:
    """Build SIMS instance from data files"""
    if config.minizinc_data:
        # Parse .dzn files
        sims_data = parse_dzn_to_sims_dict(dzn_file_path)
    else:
        # Handle direct text data
        sims_data = build_instance_text_data(config)
    
    instance = InstanceSIMS(sims_data)
    
    # Transform data structures:
    # 1. Convert 1-indexed to 0-indexed
    # 2. Process cloud coverage relationships
    # 3. Build image-element mappings
    return instance
```

#### Data Structure Transformation:
```python
def correct_starting_indexes(images, clouds):
    """Convert from 1-indexed (MiniZinc) to 0-indexed (Python)"""
    for i in range(len(images)):
        images[i] = {x - 1 for x in images[i]}  # Elements covered by image i
    for i in range(len(clouds)):
        clouds[i] = {x - 1 for x in clouds[i]}  # Cloudy elements in image i
    return images, clouds

def get_clouds_covered_by_image():
    """Build cloud coverage relationships"""
    cloud_covered_by_image = {}  # image_id -> set of clouds it can cover
    clouds_id_area = {}          # cloud_id -> area
    
    for i in range(len(clouds)):
        image_cloud_set = clouds[i]  # Cloudy elements in image i
        for cloud_id in image_cloud_set:
            clouds_id_area[cloud_id] = areas[cloud_id]
            # Find images that can cover this cloud (image j covers element cloud_id without clouds)
            for j in range(len(images)):
                if i != j and cloud_id in images[j] and cloud_id not in clouds[j]:
                    if j in cloud_covered_by_image:
                        cloud_covered_by_image[j].add(cloud_id)
                    else:
                        cloud_covered_by_image[j] = {cloud_id}
```

### 4.3 Model Building Process

The model building creates the MILP formulation:

```python
def build_gurobi_model(instance: InstanceSIMS, config: Config):
    """Build Gurobi MILP model for SIMS"""
    model = gp.Model("SIMSModel")
    
    # STAGE 1: Data Processing
    elements = list(range(len(instance.areas)))
    images_id = list(range(len(instance.images)))
    
    # STAGE 2: Variable Creation
    select_image = model.addVars(len(images), vtype=gp.GRB.BINARY, name="select_image")
    cloud_covered = model.addVars(clouds_id, vtype=gp.GRB.BINARY, name="cloud_covered")
    
    if "min_resolution" in config.objectives:
        resolution_element = model.addVars(elements, 
                                         lb=min_resolution, ub=max_resolution,
                                         vtype=gp.GRB.INTEGER, name="resolution_element")
        # Auxiliary variables for linearization
        auxiliary_resolution = create_auxiliary_resolution_vars(model, elements, images)
    
    if "min_max_incidence_angle" in config.objectives:
        effective_incidence_angle = model.addVars(images_id, vtype=gp.GRB.INTEGER,
                                                name="effective_incidence_angle")
        current_max_incidence_angle = model.addVar(vtype=gp.GRB.INTEGER, 
                                                 name="max_allowed_incidence_angle")
    
    # STAGE 3: Constraint Addition
    add_constraints(model, config.objectives)
    
    # STAGE 4: Objective Definition
    define_objectives(model, config.objectives)
    
    return model
```

#### Constraint Details:

```python
def add_constraints(model, objectives):
    """Add constraints based on selected objectives"""
    
    # MANDATORY: Coverage constraint (always required)
    for element in elements:
        model.addConstr(
            gp.quicksum(select_image[i] for i in images_id if element in images[i]) >= 1,
            name=f"coverage_{element}"
        )
    
    # CONDITIONAL: Cloud constraints (only if cloud_coverage objective)
    if "cloud_coverage" in objectives:
        for cloud in clouds_id:
            # If cloud is covered, at least one covering image must be selected
            model.addConstr(
                gp.quicksum(select_image[i] for i in cloud_covered_by_image.keys()
                          if cloud in cloud_covered_by_image[i]) >= cloud_covered[cloud],
                name=f"cloud_coverage_lower_{cloud}"
            )
            # Linearization: if no covering image selected, cloud cannot be covered
            model.addConstr(
                gp.quicksum(select_image[i] for i in cloud_covered_by_image.keys()
                          if cloud in cloud_covered_by_image[i]) 
                <= cloud_covered[cloud] * len(images),
                name=f"cloud_coverage_upper_{cloud}"
            )
    
    # CONDITIONAL: Resolution constraints (only if min_resolution objective)
    if "min_resolution" in objectives:
        add_resolution_constraints(model)
    
    # CONDITIONAL: Incidence angle constraints (only if min_max_incidence_angle objective)
    if "min_max_incidence_angle" in objectives:
        add_incidence_angle_constraints(model)
```

#### Resolution Constraint Linearization:

The resolution constraint is the most complex, requiring linearization of the min operation:

```python
def add_resolution_constraints(model):
    """Add resolution constraints with linearization"""
    big_M = max(resolution.values()) + 1
    
    for element in elements:
        covering_images = images_covering_element[element]
        
        # Each element must have exactly (n-1) auxiliary variables set to 1
        # where n is the number of images covering the element
        total_aux = len(auxiliary_variables_for_resolution[element])
        model.addConstr(
            gp.quicksum(auxiliary_variables_for_resolution[element][i] 
                       for i in covering_images) == total_aux - 1,
            name=f"aux_resolution_sum_{element}"
        )
        
        # Linearization of: resolution_element[e] = min{resolution[i] : select_image[i] = 1, i covers e}
        for image in covering_images:
            model.addConstr(
                resolution_element[element] >= 
                resolution[image] * select_image[image] +
                big_M * (1 - select_image[image]) -
                2 * big_M * auxiliary_variables_for_resolution[element][image],
                name=f"resolution_linearization_{element}_{image}"
            )
```

#### Incidence Angle Constraints:

```python
def add_incidence_angle_constraints(model):
    """Add incidence angle constraints using indicator constraints"""
    
    # Effective incidence angle is 0 if image not selected, actual angle if selected
    for image in images_id:
        model.addConstr((select_image[image] == 0) >> 
                       (effective_incidence_angle[image] == 0),
                       name=f"incidence_angle_not_selected_{image}")
        model.addConstr((select_image[image] == 1) >> 
                       (effective_incidence_angle[image] == incidence_angle[image]),
                       name=f"incidence_angle_selected_{image}")
    
    # Maximum incidence angle across all selected images
    model.addConstr(current_max_incidence_angle == 
                   gp.max_(effective_incidence_angle[i] for i in images_id),
                   name="max_incidence_angle")
```

### 4.4 GPBA-A (Coverage Grid Point) Algorithm Implementation

The core of the Pareto front generation is the GPBA-A algorithm:

```python
def solve_gpba_a():
    """GPBA-A Algorithm Implementation"""
    
    # STAGE 1: Extreme Point Computation
    yield from get_best_worst_values()
    
    # STAGE 2: Model Conversion to Maximization
    convert_model_to_maximization()
    
    # STAGE 3: Main Objective Selection (could rotate through all objectives)
    main_obj_index = 0  # Currently uses first objective as main
    yield from solve_with_main_objective(main_obj_index)

def get_best_worst_values():
    """Compute extreme points for each objective"""
    num_objectives = len(solver.model.objectives)
    best_objective_values = [0] * num_objectives
    nadir_objectives_values = [0] * num_objectives
    
    # Find ideal points (best value for each objective individually)
    for i in range(num_objectives):
        solver.set_single_objective(solver.model.objectives[i])
        solver.set_optimization_sense(model_optimization_sense)
        
        try:
            solution_sec = get_solver_solution_for_timeout(optimize_not_satisfy=True)
            formatted_solution = process_feasible_solution(solution_sec)
            best_objective_values[i] = int(formatted_solution['objs'][i])
            yield formatted_solution  # Add to Pareto front
        except TimeoutError:
            raise TimeoutError(f"Timeout optimizing objective {i}")
    
    # Find nadir points (worst value for each objective)
    nadir_sense = "min" if model_optimization_sense == "max" else "max"
    for i in range(num_objectives):
        solver.set_single_objective(solver.model.objectives[i])
        solver.set_optimization_sense(nadir_sense)
        
        try:
            solution_sec = get_solver_solution_for_timeout(optimize_not_satisfy=True)
            formatted_solution = process_feasible_solution(solution_sec)
            nadir_objectives_values[i] = int(formatted_solution['objs'][i])
        except TimeoutError:
            raise TimeoutError(f"Timeout finding nadir for objective {i}")
```

#### Main GPBA-A Loop:

```python
def solve_with_main_objective(main_obj_index):
    """Main GPBA-A algorithm with specified main objective"""
    
    # STAGE 1: Setup ε-constraint formulation
    set_augmecon2_objective_model(main_obj_index)
    
    # STAGE 2: Initialize constraint objective indices
    num_objectives = len(solver.model.objectives)
    constraint_indices = [i for i in range(num_objectives) if i != main_obj_index]
    
    # STAGE 3: Initialize control arrays
    ef_array = [nadir_objectives_values[i] for i in constraint_indices]
    constraint_objectives = [0] * len(constraint_indices)
    
    # STAGE 4: Setup interval managers for grid point coverage
    ef_intervals = [create_interval(i) for i in constraint_indices]
    
    # STAGE 5: Initialize relative worst values
    rwv = [best_objective_values[constraint_indices[i]] for i in range(len(constraint_indices))]
    
    # STAGE 6: Main coverage loop
    yield from coverage_loop(ef_array, rwv, previous_solutions, 
                           previous_solution_information, ef_intervals, constraint_indices)
```

#### Coverage Loop Implementation:

```python
def coverage_loop(ef_array, rwv, previous_solutions, previous_solution_information,
                  ef_intervals, constraint_indices):
    """Main coverage loop implementing grid point algorithm"""
    
    iteration_count = 0
    max_iterations = 1000  # Safety limit
    
    while (ef_array[0] <= best_objective_values[constraint_indices[0]] and 
           iteration_count < max_iterations):
        
        yield from coverage_most_inner_loop(ef_array, rwv, previous_solutions,
                                          previous_solution_information, 
                                          ef_intervals, constraint_indices)
        iteration_count += 1

def coverage_most_inner_loop(ef_array, rwv, previous_solutions, 
                           previous_solution_information, ef_intervals, constraint_indices):
    """Inner loop solving each constraint configuration"""
    
    # STAGE 1: Check for previous solution at this point
    gamma = 1  # Coverage parameter
    previous_solution_relaxation, previous_solution_values = \
        search_previous_solutions_relaxation(ef_array, previous_solution_information)
    
    if previous_solution_relaxation:
        if isinstance(previous_solution_values, str):
            one_solution = []  # Infeasible
        else:
            one_solution = previous_solution_values
    else:
        # STAGE 2: Solve new constraint configuration
        update_objective_constraints(ef_array)
        
        try:
            solution_sec = get_solver_solution_for_timeout(optimize_not_satisfy=True)
            
            if solver.status_infeasible():
                save_solution_information(ef_array, "infeasible", previous_solution_information)
                one_solution = []
            else:
                objectives_solution_values = solver.get_solution_objective_values()
                str_objectives_solution_values = convert_solution_value_to_str(objectives_solution_values)
                
                if str_objectives_solution_values not in previous_solutions:
                    # New solution found
                    previous_solutions.add(str_objectives_solution_values)
                    formatted_solution = process_feasible_solution(solution_sec)
                    one_solution = formatted_solution["objs"]
                    
                    save_solution_information(ef_array, one_solution, previous_solution_information)
                    yield formatted_solution  # Add to Pareto front
                else:
                    one_solution = solver.get_solution_objective_values()
        
        except TimeoutError:
            raise TimeoutError("Solver timeout in inner loop")
    
    # STAGE 3: Update control structures
    update_ef_array_and_intervals(ef_array, one_solution, rwv, ef_intervals, constraint_indices, gamma)
```

#### ε-Constraint Setup with Augmentation:

```python
def set_augmecon2_objective_model(main_obj_index=0):
    """Setup augmented ε-constraint formulation"""
    constraint_objectives_lhs = build_objective_e_constraint_augmecon2(
        best_objective_values, nadir_objectives_values, 
        augmentation=True, main_obj_index=main_obj_index)
    
    solver.set_optimization_sense("max")
    return constraint_objectives_lhs

def build_objective_e_constraint_augmecon2(best_values, nadir_values, augmentation, main_obj_index):
    """Build ε-constraint with slack variable augmentation"""
    num_objectives = len(solver.model.objectives)
    constraint_indices = [i for i in range(num_objectives) if i != main_obj_index]
    
    if augmentation:
        delta = 0.01  # Augmentation parameter
        slack_vars = []
        
        # Create slack variables for constraint objectives
        for i in constraint_indices:
            max_s = abs(best_values[i] - nadir_values[i])
            s = solver.model.solver_model.addVar(vtype=gp.GRB.INTEGER, 
                                                lb=0, ub=max_s, name=f"s{i+1}")
            slack_vars.append(s)
        
        # Objective ranges for normalization
        obj_range = [abs(best_values[i] - nadir_values[i]) for i in constraint_indices]
        
        # Augmented objective: main_obj + δ * Σ(10^(k-1) * slack[k] / range[k])
        slack_range_sum = sum(slack_vars[i] / (10**i * obj_range[i]) 
                            for i in range(len(constraint_indices)))
        
        augmented_obj = solver.model.objectives[main_obj_index] + delta * slack_range_sum
        solver.set_single_objective(augmented_obj)
        
        # Constraint objectives with slack variables
        constraint_objectives = [solver.model.objectives[i] - slack_vars[j] 
                               for j, i in enumerate(constraint_indices)]
    else:
        # Standard ε-constraint without augmentation
        solver.set_single_objective(solver.model.objectives[main_obj_index])
        constraint_objectives = [solver.model.objectives[i] for i in constraint_indices]
    
    return constraint_objectives
```

#### Interval Management for Grid Coverage:

```python
class IntervalManager:
    """Manages intervals for grid point coverage"""
    
    def __init__(self, min_value, max_value):
        self.intervals = set()
        self.min_value = min_value
        self.max_value = max_value
        self.add_interval(min_value, max_value)
    
    def add_interval(self, start, end):
        """Add interval, merging with existing overlapping intervals"""
        new_intervals = set()
        to_add = (start, end)
        
        for interval in self.intervals:
            if interval[1] < start or interval[0] > end:  # No overlap
                new_intervals.add(interval)
            else:  # Merge overlapping intervals
                to_add = (min(to_add[0], interval[0]), max(to_add[1], interval[1]))
        
        new_intervals.add(to_add)
        self.intervals = new_intervals
    
    def remove_interval(self, start, end):
        """Remove interval, splitting existing intervals if necessary"""
        new_intervals = set()
        
        for interval in self.intervals:
            if interval[1] < start or interval[0] > end:  # No overlap
                new_intervals.add(interval)
            else:
                # Split interval if there's overlap
                if interval[0] < start:
                    new_intervals.add((interval[0], start - 1))
                if interval[1] > end:
                    new_intervals.add((end + 1, interval[1]))
        
        self.intervals = new_intervals
    
    def find_largest_interval(self):
        """Find the largest remaining interval"""
        if not self.intervals:
            return None
        return max(self.intervals, key=lambda x: x[1] - x[0])

def adjust_parameter_ef_array(id_constraint_objective, ef_array, sol_obj_k, 
                            ef_interval, constraint_indices, gamma=1):
    """Update ef_array based on solution found and manage intervals"""
    
    # Remove covered region from interval
    start_removal = ef_array[id_constraint_objective]
    new_max_interval = start_removal - 1
    
    if sol_obj_k is None:  # Infeasible
        end_removal = ef_interval.max_value
    else:
        end_removal = min(sol_obj_k, ef_interval.max_value)
    
    ef_interval.remove_interval(start_removal, end_removal)
    
    # Update max_value if needed
    if end_removal >= ef_interval.max_value:
        ef_interval.max_value = new_max_interval
    
    # Find next point to explore (center of largest remaining interval)
    max_interval = ef_interval.find_largest_interval()
    actual_obj_index = constraint_indices[id_constraint_objective]
    
    if ef_array[id_constraint_objective] == nadir_objectives_values[actual_obj_index]:
        ef_array[id_constraint_objective] = best_objective_values[actual_obj_index]
    else:
        if max_interval is not None:
            # Explore center of largest remaining interval
            ef_array[id_constraint_objective] = int((max_interval[0] + max_interval[1]) / 2)
        else:
            # No intervals left, reinitialize
            ef_array[id_constraint_objective] = best_objective_values[actual_obj_index] + 1
            ef_interval = create_interval(actual_obj_index)
    
    return ef_interval
```

### 4.5 Pareto Front Management

```python
class ParetoFront:
    """Pareto front management with dominance checking"""
    
    def __init__(self):
        self.minimize_objs = []
        self.solutions = []
        self.front = []  # Indices of non-dominated solutions
    
    def join(self, solution):
        """Add solution if non-dominated, remove dominated solutions"""
        idx = len(self.solutions)
        self.solutions.append(solution)
        
        return self.join_front(idx)
    
    def join_front(self, idx):
        """Update Pareto front with solution at index idx"""
        list_to_remove = []
        
        # Check if new solution is dominated
        for front_idx in self.front:
            if self.dominates(self.solutions[front_idx], self.solutions[idx]):
                return False  # New solution is dominated
        
        # Find solutions dominated by new solution
        for i, front_idx in enumerate(self.front):
            if self.dominates(self.solutions[idx], self.solutions[front_idx]):
                list_to_remove.append(i)
        
        # Remove dominated solutions from front
        for i in reversed(list_to_remove):
            self.front.pop(i)
        
        # Add new solution to front
        self.front.append(idx)
        return True
    
    def dominates(self, solution1, solution2):
        """Check if solution1 dominates solution2"""
        return all(self.compare(obj1, obj2, minimize) 
                  for obj1, obj2, minimize in zip(
                      solution1["objs"], solution2["objs"], 
                      solution1["minimize_objs"]))
    
    def compare(self, obj1, obj2, minimize):
        """Compare two objective values"""
        if minimize:
            return obj1 <= obj2
        else:
            return obj1 >= obj2
    
    def hypervolume(self):
        """Compute hypervolume indicator"""
        if not self.front:
            return 0
        
        ref_point = np.array(self.solutions[0]["ref_point"])
        front = np.array([self.solutions[f]["objs"] for f in self.front])
        
        if not self.solutions[0]["minimize_objs"][0]:
            # Convert maximization to minimization for hypervolume calculation
            ref_point = -ref_point
            front = -front
        
        hv = HV(ref_point=ref_point)(front)
        return float(hv) if hv is not None else 0.0
```

### 4.6 Solution Processing and Statistics

```python
def process_solution(solver, solution_time):
    """Process a feasible solution and update statistics"""
    
    # Extract objective values
    objective_values = solver.get_solution_objective_values()
    solution_values = solver.model.get_solution_values()
    
    # Create solution object
    ref_points = solver.model.get_ref_points_for_hypervolume()
    minimize_objs = [solver.model.is_a_minimization_model()] * len(objective_values)
    
    solution = Solution(
        objs=objective_values,
        solution_values=solution_values,
        minimize_objs=minimize_objs,
        ref_point=ref_points
    )
    
    # Update statistics
    solver.update_statistics(solution_time)
    
    # Format for Pareto front
    formatted_solution = MinizincResultFormat(
        status=solver.get_status(),
        solution=solution,
        statistics=None
    )
    
    return formatted_solution

def update_statistics(statistics, solution_time):
    """Update solving statistics"""
    statistics["number_of_solutions"] += 1
    statistics["time_solver_sec"] += solution_time
    statistics["solutions_time_list"].append(solution_time)
    # Additional statistics as needed
```

## 5. Key Algorithmic Stages for Implementation

### Stage 1: Data Preprocessing and Instance Creation
1. Parse problem data from files (.dzn format or text)
2. Convert from 1-indexed to 0-indexed representation
3. Build cloud coverage relationships between images
4. Create element-image mapping structures
5. Initialize data structures for efficient access

### Stage 2: MILP Model Construction
1. Create decision variables based on selected objectives
2. Add mandatory coverage constraints
3. Add objective-specific constraints conditionally
4. Linearize complex constraints (especially resolution constraints)
5. Define objective functions

### Stage 3: Extreme Point Computation
1. Optimize each objective individually to find ideal points
2. Optimize each objective in opposite direction for nadir points
3. Add extreme solutions to initial Pareto front
4. Handle timeout scenarios during extreme point computation

### Stage 4: ε-Constraint Setup with Augmentation
1. Convert problem to maximization if needed
2. Select main objective (typically first objective)
3. Create slack variables for constraint objectives
4. Build augmented objective function with δ parameter
5. Setup constraint objective expressions

### Stage 5: Grid Coverage Algorithm (GPBA-A)
1. Initialize control arrays (ef_array, rwv)
2. Create interval managers for each constraint objective
3. Setup previous solution tracking
4. Execute main coverage loop
5. Update intervals based on solutions found

### Stage 6: Solution Processing and Front Management
1. Check solution dominance relationships
2. Update Pareto front with new solutions
3. Remove dominated solutions from front
4. Calculate hypervolume indicator
5. Store solution statistics

### Stage 7: Termination and Results
1. Handle timeout scenarios
2. Process incomplete solutions
3. Calculate final hypervolume
4. Export Pareto front and statistics
5. Clean up solver resources

## 6. Critical Implementation Details

### 6.1 Numerical Considerations
- Use integer variables for objectives to avoid floating-point precision issues
- Set appropriate big-M values for linearization constraints
- Handle numerical tolerances in Gurobi solver parameters
- Round objective values to ensure integer consistency

### 6.2 Memory Management
- Efficiently store interval structures using sets
- Limit solution storage to essential information
- Use appropriate data structures for large-scale problems
- Clean up solver constraints when updating ε-constraints

### 6.3 Timeout Handling
- Implement hierarchical timeout checking
- Save best incomplete solution when timeout occurs
- Gracefully handle partial results
- Update statistics correctly for timeout scenarios

### 6.4 Performance Optimization
- Use indicator constraints for logical relationships
- Minimize constraint modifications during solving
- Employ lazy constraint generation when appropriate
- Utilize solver-specific performance parameters

## 7. Complete Algorithm Pseudocode

```
ALGORITHM: SIMS Multi-Objective MILP Solver with GPBA-A

INPUT: 
  - config: Configuration containing objectives, timeout, solver settings
  - instance_data: SIMS problem data (images, clouds, costs, areas, etc.)

OUTPUT:
  - pareto_front: Complete Pareto front representation
  - statistics: Solving statistics and performance metrics

BEGIN
  // STAGE 1: INITIALIZATION
  instance ← build_instance(instance_data, config)
  statistics ← initialize_empty_statistics()
  model ← build_gurobi_model(instance, config)
  solver ← initialize_gurobi_solver(model, config)
  pareto_front ← initialize_empty_pareto_front()
  front_generator ← CoverageGridPoint(solver, timer)
  
  // STAGE 2: EXTREME POINT COMPUTATION
  FOR each objective i in model.objectives DO
    solver.set_single_objective(model.objectives[i])
    solver.set_optimization_sense("min" or "max")
    TRY
      solution ← solver.solve_with_timeout(config.timeout)
      best_values[i] ← solution.objectives[i]
      pareto_front.add(solution)
    CATCH TimeoutError
      THROW "Extreme point computation failed"
    END TRY
  END FOR
  
  // Compute nadir points
  FOR each objective i in model.objectives DO
    solver.set_single_objective(model.objectives[i])
    solver.set_optimization_sense(opposite_sense)
    solution ← solver.solve_with_timeout(config.timeout)
    nadir_values[i] ← solution.objectives[i]
  END FOR
  
  // STAGE 3: CONVERT TO MAXIMIZATION
  IF model.is_minimization_model() THEN
    FOR each objective i DO
      model.objectives[i] ← -model.objectives[i]
      best_values[i] ← -best_values[i]
      nadir_values[i] ← -nadir_values[i]
    END FOR
  END IF
  
  // STAGE 4: SETUP ε-CONSTRAINT FORMULATION
  main_obj_index ← 0
  constraint_indices ← [1, 2, ..., num_objectives-1] // All except main
  
  // Create slack variables for augmentation
  slack_vars ← []
  FOR each i in constraint_indices DO
    max_s ← abs(best_values[i] - nadir_values[i])
    slack_vars[i] ← solver.add_integer_var(0, max_s)
  END FOR
  
  // Setup augmented objective
  delta ← 0.01
  obj_ranges ← [abs(best_values[i] - nadir_values[i]) for i in constraint_indices]
  augmented_term ← sum(slack_vars[i] / (10^i * obj_ranges[i]) for i in constraint_indices)
  augmented_objective ← model.objectives[main_obj_index] + delta * augmented_term
  solver.set_objective(augmented_objective)
  
  // Setup constraint objectives
  constraint_objectives ← [model.objectives[i] - slack_vars[j] for j,i in enumerate(constraint_indices)]
  
  // STAGE 5: INITIALIZE GPBA-A CONTROL STRUCTURES
  ef_array ← [nadir_values[i] for i in constraint_indices]
  ef_intervals ← [IntervalManager(nadir_values[i]+1, best_values[i]-1) for i in constraint_indices]
  rwv ← [best_values[i] for i in constraint_indices]
  previous_solutions ← empty_set()
  previous_solution_info ← empty_list()
  
  // STAGE 6: MAIN GPBA-A COVERAGE LOOP
  iteration ← 0
  max_iterations ← 1000
  
  WHILE ef_array[0] ≤ best_values[constraint_indices[0]] AND iteration < max_iterations DO
    
    // Check for previous solution at this configuration
    previous_relaxation, previous_values ← search_previous_solutions(ef_array, previous_solution_info)
    
    IF previous_relaxation THEN
      IF previous_values == "infeasible" THEN
        solution ← empty_solution
      ELSE
        solution ← previous_values
      END IF
    ELSE
      // Solve new configuration
      FOR j, i in enumerate(constraint_indices) DO
        IF constraint_constraints[j] != 0 THEN
          solver.remove_constraint(constraint_constraints[j])
        END IF
        constraint_constraints[j] ← solver.add_constraint(constraint_objectives[j] == ef_array[j])
      END FOR
      
      TRY
        solution ← solver.solve_with_timeout(remaining_timeout)
        
        IF solver.status == INFEASIBLE THEN
          save_solution_info(ef_array, "infeasible", previous_solution_info)
          solution ← empty_solution
        ELSE
          objective_values ← solver.get_objective_values()
          solution_string ← convert_to_string(objective_values)
          
          IF solution_string NOT IN previous_solutions THEN
            previous_solutions.add(solution_string)
            formatted_solution ← format_solution(solution, objective_values)
            save_solution_info(ef_array, objective_values, previous_solution_info)
            pareto_front.add(formatted_solution)
            solution ← objective_values
          ELSE
            solution ← objective_values
          END IF
        END IF
        
      CATCH TimeoutError
        handle_timeout_and_exit()
      END TRY
    END IF
    
    // Update control structures
    IF solution is not empty THEN
      FOR i in range(len(rwv)) DO
        rwv[i] ← min(rwv[i], solution[constraint_indices[i]])
      END FOR
    END IF
    
    // Update ef_array using interval management
    id_interval ← -1  // Last constraint objective
    actual_obj_index ← constraint_indices[id_interval]
    
    // Remove covered region from interval
    start_removal ← ef_array[id_interval]
    IF solution is empty THEN
      end_removal ← ef_intervals[id_interval].max_value
      sol_obj_value ← None
    ELSE
      end_removal ← min(solution[actual_obj_index], ef_intervals[id_interval].max_value)
      sol_obj_value ← solution[actual_obj_index]
    END IF
    
    ef_intervals[id_interval].remove_interval(start_removal, end_removal)
    
    // Find next point to explore
    largest_interval ← ef_intervals[id_interval].find_largest_interval()
    IF largest_interval != None THEN
      ef_array[id_interval] ← (largest_interval.start + largest_interval.end) / 2
    ELSE
      ef_array[id_interval] ← best_values[actual_obj_index] + 1
      ef_intervals[id_interval] ← create_new_interval(actual_obj_index)
    END IF
    
    // Update other constraint objectives
    FOR i in reverse(range(len(constraint_indices)-1, 0)) DO
      IF ef_array[i] > best_values[constraint_indices[i]] THEN
        ef_array[i] ← nadir_values[constraint_indices[i]]
        rwv[i] ← best_values[constraint_indices[i]]
        // Update ef_array for previous constraint objective
        prev_id ← i - 1
        ef_intervals[prev_id] ← update_ef_array_parameter(prev_id, ef_array, sol_obj_value, ef_intervals[prev_id], constraint_indices)
      ELSE
        BREAK
      END IF
    END FOR
    
    iteration ← iteration + 1
  END WHILE
  
  // STAGE 7: FINALIZATION
  statistics["exhaustive"] ← True
  statistics["hypervolume"] ← pareto_front.hypervolume()
  statistics["pareto_solutions_count"] ← pareto_front.size()
  
  RETURN pareto_front, statistics
END ALGORITHM
```

## 8. Conclusion

The SIMS multi-objective MILP solver implements a sophisticated algorithm combining the GPBA-A (Coverage Grid Point Based Representation) method with augmented ε-constraint techniques. The key innovations include:

1. **Systematic Pareto Front Coverage**: The GPBA-A algorithm ensures comprehensive exploration of the Pareto front through intelligent grid point management.

2. **Augmented ε-Constraint Method**: Slack variable augmentation improves solution diversity and prevents premature termination.

3. **Efficient Constraint Management**: Dynamic constraint addition/removal and linearization techniques enable scalable solving.

4. **Robust Solution Processing**: Comprehensive dominance checking and hypervolume calculation provide quality metrics.

5. **Adaptive Timeout Handling**: Graceful degradation under time pressure with partial solution preservation.

This implementation provides a complete framework for multi-objective satellite image selection problems and can be adapted for similar multi-objective integer linear programming challenges. The modular design allows for easy extension to additional objectives and constraints while maintaining algorithmic efficiency.

The algorithm successfully balances exploration completeness with computational efficiency, making it suitable for real-world satellite mission planning scenarios where multiple conflicting objectives must be optimized simultaneously.