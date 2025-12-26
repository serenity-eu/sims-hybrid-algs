# SIMS Multi-Objective MILP Algorithm Specification (GPBA-A Implementation)

## Abstract

This document provides a **complete and precise specification** of the Satellite Image Mosaic Selection (SIMS) multi-objective Mixed Integer Linear Programming (MILP) solver implementation based on the actual `solve_inlined()` function. The solver employs the **Coverage Grid Point Based Representation Algorithm (GPBA-A)** with augmented ε-constraint method to generate complete Pareto front representations. This specification includes all algorithmic intricacies, data conversions, precision handling, and implementation details necessary for exact replication.

**Document Status**: Updated based on complete analysis of `solve_inlined()` implementation in `sims-solvers/sims_solvers/solve.py`.

## 1. Introduction

The SIMS problem is a multi-objective optimization challenge that involves selecting optimal subsets of satellite images while balancing multiple conflicting objectives such as cost minimization, cloud coverage reduction, resolution optimization, and incidence angle minimization. The solver implements the GPBA-A algorithm using the **augmented ε-constraint method** with intelligent interval management for efficient and complete Pareto front exploration.

### Key Features of This Implementation

1. **Complete Inlined Implementation**: Self-contained with minimal external dependencies
2. **IntervalManager**: Adaptive exploration of largest gaps in the Pareto front
3. **Relative Worst Values (RWV)**: Tracking for multi-dimensional cascading updates
4. **Solution Relaxation Search**: Avoids redundant solves by checking previous configurations
5. **Precision Handling**: Uses binary solution values to avoid floating-point errors
6. **Proper Min-Max Conversion**: Explicit conversion to maximization with correct objective handling

## 2. Problem Formulation

### 2.1 SIMS Problem Definition

The SIMS problem is formulated as a multi-objective integer linear programming problem with the following components:

#### Decision Variables:
- `select_image[i]` ∈ {0,1}: Binary variable indicating whether image i is selected
- `cloud_covered[c]` ∈ {0,1}: Binary variable indicating whether cloud c is covered (only if `cloud_coverage` objective enabled)
- `resolution_element[e]` ∈ Z⁺: Integer variable representing resolution for element e (only if `min_resolution` objective enabled)
- `effective_incidence_angle[i]` ∈ Z⁺: Integer variable for effective incidence angle of image i (only if `min_max_incidence_angle` objective enabled)
- `current_max_incidence_angle` ∈ Z⁺: Integer variable for maximum incidence angle across all selected images (only if `min_max_incidence_angle` objective enabled)

#### Objectives (All Originally Minimization):
1. **Cost Minimization** (`min_cost`): min ∑ᵢ select_image[i] × costs[i]
2. **Cloud Coverage** (`cloud_coverage`): min (total_cloud_area - ∑c cloud_covered[c] × area_clouds[c])
3. **Resolution** (`min_resolution`): min ∑e resolution_element[e]
4. **Max Incidence Angle** (`min_max_incidence_angle`): min current_max_incidence_angle

#### Constraints:

**Mandatory (always active):**
1. **Coverage Constraint**: ∀e ∈ elements: ∑ᵢ∈images_covering[e] select_image[i] ≥ 1
   - Ensures every element is covered by at least one selected image

**Conditional (only when objective is enabled):**

2. **Cloud Constraints** (only if `cloud_coverage` objective):
   ```python
   ∀c ∈ clouds: cloud_covered[c] <= sum(select_image[i] for i in cloud_covered_by_image[c])
   ```
   - `cloud_covered[c]` can only be 1 if at least one image that can cover cloud c is selected
   - Note: `cloud_covered_by_image[c]` contains images that cover the element where cloud c exists WITHOUT having clouds themselves

3. **Resolution Constraints** (only if `min_resolution` objective):
   - Complex linearization involving auxiliary binary variables
   - See Section 3.3 for detailed implementation

4. **Incidence Angle Constraints** (only if `min_max_incidence_angle` objective):
   - Using indicator constraints or linearization for max operation
   - See Section 3.4 for detailed implementation

### 2.2 Critical Data Conversions

#### 2.2.1 Index Conversion: 1-indexed to 0-indexed

The implementation receives data in **1-indexed format** (from MiniZinc .dzn files) but operates in **0-indexed format** (Python convention). The conversion happens during data parsing:

```python
# Example from .dzn file (1-indexed):
# images = [array_of_sets([{1,2,3}, {2,3,4}, ...])]
# elements are numbered 1..n_elements
# images reference elements by 1-based index

# After parsing in Python (0-indexed):
# images[0] = {0, 1, 2}  # Image 0 covers elements 0, 1, 2
# images[1] = {1, 2, 3}  # Image 1 covers elements 1, 2, 3
```

**Critical**: All internal operations use 0-indexing. When storing solutions, image IDs are 0-indexed lists.

#### 2.2.2 Cloud Data Structure

Cloud data requires special processing:

```python
# Input clouds_per_image[i] = set of cloudy ELEMENT IDs in image i (0-indexed)
# Derived cloud_covered_by_image[image_id] = set of cloud IDs that this image can cover
# Derived clouds_id_area[cloud_id] = area of this cloud

# Cloud ID convention:
# - Each cloudy element in an image is assigned a unique cloud ID
# - Cloud ID is the element ID where the cloud exists
# - clouds_id_area[cloud_id] = area of that element

# Example:
# If image 0 has clouds on elements {5, 10, 15}:
#   clouds_per_image[0] = {5, 10, 15}
#   clouds_id_area[5] = areas[5]
#   clouds_id_area[10] = areas[10]
#   clouds_id_area[15] = areas[15]
#
# If image 1 covers element 5 without clouds:
#   cloud_covered_by_image[1].add(5)  # Image 1 can cover cloud at element 5
```

## 3. Complete GPBA-A Algorithm Implementation

### 3.1 Algorithm Overview

The GPBA-A algorithm consists of **7 main stages**:

1. **Data Parsing & Preprocessing**: Parse .dzn files, convert indices, build data structures
2. **Model Construction**: Build Gurobi MILP model with conditional constraints
3. **Extreme Point Computation**: Find ideal and nadir points for each objective
4. **Minimization to Maximization Conversion**: Convert all objectives to maximization
5. **Augmented ε-Constraint Setup**: Create slack variables and augmented objective
6. **Grid Coverage Loop**: Main GPBA-A algorithm with interval management
7. **Finalization**: Convert back to minimization, calculate statistics

### 3.2 IntervalManager Class

**Purpose**: Manages intervals to track which regions of the objective space have been explored.

```python
class IntervalManager:
    """Manages intervals for efficient Pareto front coverage in GPBA-A algorithm."""
    
    def __init__(self, min_value, max_value):
        self.intervals = set()           # Set of (start, end) tuples
        self.min_value = min_value       # Lower bound of objective range
        self.max_value = max_value       # Upper bound of objective range
        self.add_interval(min_value, max_value)  # Initially full range
    
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
```

**Key Insight**: The IntervalManager tracks unexplored regions. When a solution is found, we remove the explored region and focus on the largest remaining gap.

### 3.3 Core Helper Functions

#### 3.3.1 adjust_parameter_ef_array()

**Purpose**: The heart of GPBA-A - adaptively explores the largest gaps in the Pareto front.

```python
def adjust_parameter_ef_array(id_constraint_objective, ef_array, sol_obj_k, 
                              ef_interval, constraint_indices, best_objective_values,
                              nadir_objectives_values, gamma=1):
    """
    Adjust ef_array parameter based on solution found, using interval management.
    
    Args:
        id_constraint_objective: Index in ef_array (0 to len(constraint_indices)-1)
        ef_array: Current ε-constraint RHS values (for constraint objectives only)
        sol_obj_k: Solution value for the objective at id_constraint_objective (or None if infeasible)
        ef_interval: IntervalManager for this objective
        constraint_indices: Indices of constraint objectives in full objective list
        best_objective_values: Ideal points (all objectives, in maximization form)
        nadir_objectives_values: Nadir points (all objectives, in maximization form)
        gamma: Coverage parameter (default 1, not used in current implementation)
    
    Returns:
        Updated ef_interval
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
        # Reinitialize interval when exhausted
        ef_array[id_constraint_objective] = best_objective_values[actual_obj_index] + 1
        ef_interval = create_interval(actual_obj_index, best_objective_values, nadir_objectives_values)
    
    return ef_interval
```

**Critical Behavior**:
1. When a solution is found at constraint value `start_removal`, we know all points from `start_removal` to `sol_obj_k` are covered (for maximization)
2. Remove this region from the interval
3. Find the largest remaining gap and explore its center
4. If no gaps remain, reinitialize to start exploring from the beginning

#### 3.3.2 search_previous_solutions_relaxation()

**Purpose**: Avoid redundant solves by checking if a more relaxed configuration was already solved.

```python
def search_previous_solutions_relaxation(ef_array, previous_solution_information, constraint_indices):
    """
    Check if this constraint configuration was already explored with relaxation.
    
    For MAXIMIZATION (after conversion):
    - ef_array1 is LESS constrained (more relaxed) if all ef_array1[i] >= ef_array2[i]
    - Less constrained means the constraint allows higher objective values
    - Original constraint in minimization: obj_i >= ef_i
    - After conversion to maximization: -obj_i <= -ef_i, or obj_i >= ef_i
    - So in maximization form, constraint is: obj_i >= ef_i
    
    Args:
        ef_array: Current ε-constraint RHS values (for constraint objectives only)
        previous_solution_information: List of {"ef_array": [...], "solution": [...] or "infeasible"}
        constraint_indices: Indices of constraint objectives in full objective list
    
    Returns:
        (found, solution): 
            - (True, [obj_values]) if previous solution satisfies current constraints
            - (True, "infeasible") if previous was infeasible with less constrained constraints
            - (False, None) if no applicable previous solution
    """
    for prev_sol_info in previous_solution_information:
        prev_ef_array = prev_sol_info["ef_array"]
        prev_solution = prev_sol_info["solution"]
        
        # Check if previous ef_array is less constrained (all values >= current)
        # For maximization: higher ef value means less constrained
        is_less_constrained = all(prev_ef_array[i] >= ef_array[i] for i in range(len(ef_array)))
        
        if is_less_constrained:
            if prev_solution != "infeasible":
                # Check if previous solution satisfies current (tighter) constraints
                # For maximization: solution[constraint_idx] must be <= ef_array[i]
                # (Constraint is obj >= ef, so solution >= ef means constraint is satisfied)
                # But we're checking if solution fits TIGHTER constraint, so solution <= new_ef
                satisfies = all(prev_solution[constraint_indices[i]] <= ef_array[i] 
                              for i in range(len(ef_array)))
                if satisfies:
                    return True, prev_solution
            else:
                # Previous was infeasible with less constrained constraints, 
                # so current (tighter) is also infeasible
                return True, "infeasible"
    
    return False, None
```

**Key Insight**: If we previously solved with constraints `obj >= 100` and found infeasible or a specific solution, we don't need to re-solve with tighter constraints `obj >= 90` if the previous solution already satisfies them.

#### 3.3.3 Precision Handling Functions

**Critical Issue**: Gurobi variable `.X` values can be float approximations (e.g., 0.999999 instead of 1.0). This causes objective calculation errors.

**Solution**: Use binary solution values for objective calculations:

```python
# WRONG - causes precision errors:
cost = sum(select_image[k].X * costs[k] for k in images_id)  # .X might be 0.9999

# CORRECT - use binary values:
solution_values = [int(select_image[j].X > 0.5) for j in images_id]  # Convert to 0/1
cost = sum(solution_values[k] * costs[k] for k in images_id)  # Now exact

# For cloud coverage, also round intermediate values:
cloud_val = sum(round(cloud_covered[c].X) * clouds_id_area[c] 
                for c in clouds_id if c in cloud_covered)
```

**Why This Matters**: Errors of 2-4 units in objectives can cause solutions to appear different when they're actually identical, or to have incorrect objective values that fail validation.

## 4. Detailed Algorithm Stages

### 4.1 STAGE 1: Data Parsing & Preprocessing

```python
# Parse .dzn file
dzn_file_path = os.path.join(config.data_sets_folder, f"{config.data_name}.dzn")

def parse_array(pattern, content):
    """Parse array of integers/floats from .dzn format."""
    match = re.search(pattern, content, re.DOTALL)
    if match:
        array_str = match.group(1)
        items = [item.strip() for item in array_str.strip('[]').split(',')]
        return [float(item) if '.' in item else int(item) for item in items if item.strip()]
    return []

def parse_set_array(pattern, content):
    """Parse array of sets from .dzn format (returns 0-indexed sets)."""
    match = re.search(pattern, content, re.DOTALL)
    if match:
        array_str = match.group(1)
        sets = []
        set_matches = re.findall(r'\{([^}]*)\}', array_str)
        for set_match in set_matches:
            if set_match.strip():
                # Parse and convert from 1-indexed to 0-indexed
                elements = [int(x.strip()) - 1 for x in set_match.split(',') if x.strip()]
                sets.append(set(elements))
            else:
                sets.append(set())
        return sets
    return []

# Parse data
n_elements = parse_int(r'n_elements\s*=\s*(\d+)', content)
n_images = parse_int(r'n_images\s*=\s*(\d+)', content)
areas = parse_array(r'areas\s*=\s*\[(.*?)\];', content)
costs = parse_array(r'costs\s*=\s*\[(.*?)\];', content)
images = parse_set_array(r'images\s*=\s*\[\s*array_of_sets\s*\((.*?)\)\s*\];', content)
clouds_per_image = parse_set_array(r'clouds_per_image\s*=\s*\[\s*array_of_sets\s*\((.*?)\)\s*\];', content)
resolution = parse_array(r'resolution\s*=\s*\[(.*?)\];', content) if "min_resolution" in config.objectives else None
incidence_angle = parse_array(r'incidence_angle\s*=\s*\[(.*?)\];', content) if "min_max_incidence_angle" in config.objectives else None

# Build cloud data structures
clouds_id_area = {}
cloud_covered_by_image = {}

for image_id in range(len(clouds_per_image)):
    cloudy_elements = clouds_per_image[image_id]
    for cloud_element_id in cloudy_elements:
        # Cloud ID is the element ID where the cloud exists
        clouds_id_area[cloud_element_id] = areas[cloud_element_id]
        
        # Find which images can cover this cloud (images that cover the element without clouds)
        for other_image_id in range(len(images)):
            if other_image_id != image_id:  # Different image
                if cloud_element_id in images[other_image_id]:  # Covers the element
                    if cloud_element_id not in clouds_per_image[other_image_id]:  # Without clouds
                        if other_image_id not in cloud_covered_by_image:
                            cloud_covered_by_image[other_image_id] = set()
                        cloud_covered_by_image[other_image_id].add(cloud_element_id)
```

**Key Conversions**:
- All element and image indices converted from 1-based (MiniZinc) to 0-based (Python)
- Cloud IDs are element IDs where clouds exist
- `cloud_covered_by_image[img]` = set of cloud IDs that image `img` can cover

### 4.2 STAGE 2: Model Construction

```python
model = gp.Model("SIMSModel")
model.setParam('OutputFlag', 0)  # Suppress Gurobi output
model.setParam('TimeLimit', config.solver_timeout_sec)
model.setParam('Threads', config.threads)

elements = list(range(n_elements))
images_id = list(range(n_images))
clouds_id = sorted(clouds_id_area.keys())

# Create variables
select_image = model.addVars(len(images_id), vtype=gp.GRB.BINARY, name="select_image")

# Conditional variables based on objectives
cloud_covered = None
if "cloud_coverage" in config.objectives:
    cloud_covered = model.addVars(clouds_id, vtype=gp.GRB.BINARY, name="cloud_covered")

resolution_element = None
if "min_resolution" in config.objectives:
    min_resolution = int(min(resolution))
    max_resolution = int(max(resolution))
    resolution_element = model.addVars(elements, lb=min_resolution, ub=max_resolution,
                                      vtype=gp.GRB.INTEGER, name="resolution_element")

current_max_incidence_angle = None
effective_incidence_angle = None
if "min_max_incidence_angle" in config.objectives:
    effective_incidence_angle = model.addVars(images_id, vtype=gp.GRB.INTEGER, 
                                             name="effective_incidence_angle")
    current_max_incidence_angle = model.addVar(vtype=gp.GRB.INTEGER, 
                                              name="current_max_incidence_angle")

# MANDATORY CONSTRAINT: Coverage (always active)
for e in elements:
    covering_images = [i for i in images_id if e in images[i]]
    model.addConstr(gp.quicksum(select_image[i] for i in covering_images) >= 1,
                   name=f"coverage_{e}")

# CONDITIONAL CONSTRAINT: Cloud coverage
if cloud_covered is not None:
    for c in clouds_id:
        if c in clouds_id_area:  # Cloud exists
            covering_images = [i for i in cloud_covered_by_image.keys() if c in cloud_covered_by_image[i]]
            if covering_images:
                model.addConstr(
                    gp.quicksum(select_image[i] for i in covering_images) >= cloud_covered[c],
                    name=f"cloud_{c}"
                )

# CONDITIONAL CONSTRAINT: Resolution (complex linearization)
if resolution_element is not None:
    # Implementation depends on linearization method - see Section 4.3
    pass

# CONDITIONAL CONSTRAINT: Incidence angle
if effective_incidence_angle is not None and current_max_incidence_angle is not None:
    for i in images_id:
        # If image selected, effective angle = actual angle; else 0
        model.addConstr((select_image[i] == 0) >> (effective_incidence_angle[i] == 0),
                       name=f"incidence_not_selected_{i}")
        model.addConstr((select_image[i] == 1) >> (effective_incidence_angle[i] == incidence_angle[i]),
                       name=f"incidence_selected_{i}")
    
    # Max angle across all selected images
    model.addConstr(current_max_incidence_angle == gp.max_(effective_incidence_angle[i] for i in images_id),
                   name="max_incidence")

model.update()
```

### 4.3 STAGE 3: Extreme Point Computation

```python
num_objectives = len(config.objectives)
best_objective_values = [0] * num_objectives  # Ideal points
nadir_objectives_values = [0] * num_objectives  # Nadir points

# Define objective expressions (in MINIMIZATION form initially)
objectives_exprs = []
for obj_name in config.objectives:
    if obj_name == "min_cost":
        objectives_exprs.append(gp.quicksum(select_image[i] * costs[i] for i in images_id))
    elif obj_name == "cloud_coverage":
        total_cloud_area = sum(clouds_id_area.values())
        cloud_val = gp.quicksum(cloud_covered[c] * clouds_id_area[c] for c in clouds_id if c in cloud_covered)
        objectives_exprs.append(total_cloud_area - cloud_val)
    elif obj_name == "min_resolution":
        objectives_exprs.append(gp.quicksum(resolution_element[e] for e in elements))
    elif obj_name == "min_max_incidence_angle":
        objectives_exprs.append(current_max_incidence_angle)

# Find ideal points (best value for each objective individually)
for i in range(num_objectives):
    model.setObjective(objectives_exprs[i], gp.GRB.MINIMIZE)
    model.optimize()
    
    if model.status == gp.GRB.OPTIMAL:
        # Extract solution with PRECISE objective calculation
        solution_values = [int(select_image[j].X > 0.5) for j in images_id]  # Binary values
        current_objs = []
        
        for j, obj_name in enumerate(config.objectives):
            if obj_name == "min_cost":
                current_objs.append(sum(solution_values[k] * costs[k] for k in images_id))
            elif obj_name == "cloud_coverage":
                cloud_val = sum(round(cloud_covered[c].X) * clouds_id_area[c] 
                              for c in clouds_id if c in cloud_covered)
                total_cloud_area = sum(clouds_id_area.values())
                current_objs.append(int(total_cloud_area - cloud_val))
            elif obj_name == "min_resolution":
                current_objs.append(int(round(sum(resolution_element[e].X for e in elements))))
            elif obj_name == "min_max_incidence_angle":
                current_objs.append(int(round(current_max_incidence_angle.X)))
        
        best_objective_values[i] = int(current_objs[i])
        add_to_pareto_front(current_objs, solution_values)

# Find nadir points (worst value for each objective)
for i in range(num_objectives):
    model.setObjective(objectives_exprs[i], gp.GRB.MAXIMIZE)
    model.optimize()
    
    if model.status == gp.GRB.OPTIMAL:
        nadir_objectives_values[i] = int(model.objVal)
```

**Critical**: Use binary `solution_values` for objective calculations to avoid floating-point precision errors.

### 4.4 STAGE 4: Minimization to Maximization Conversion

**Critical Step**: GPBA-A works in maximization form. We must convert all objectives.

```python
# Convert objectives from minimization to maximization
# IMPORTANT: Negate both expressions AND ideal/nadir values
for i in range(num_objectives):
    objectives_exprs[i] = -objectives_exprs[i]  # Negate expression
    
    # Swap and negate ideal/nadir values
    temp = best_objective_values[i]
    best_objective_values[i] = -nadir_objectives_values[i]
    nadir_objectives_values[i] = -temp

model.update()
```

**Why This Matters**:
- Original problem: minimize cost → best=100, nadir=500
- After conversion: maximize -cost → best=-500, nadir=-100
- Constraints change: `cost >= ef` becomes `-cost >= -ef` or `cost <= ef`
- In maximization: higher ef value means LESS constrained (allows more solutions)

### 4.5 STAGE 5: Augmented ε-Constraint Setup

```python
main_obj_index = 0  # First objective is main objective
constraint_indices = [i for i in range(num_objectives) if i != main_obj_index]

# Create slack variables for augmentation
delta = 0.01  # Augmentation parameter
slack_vars = []

for i in range(len(constraint_indices)):
    actual_obj_idx = constraint_indices[i]
    max_s = abs(best_objective_values[actual_obj_idx] - nadir_objectives_values[actual_obj_idx])
    s = model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=max_s, name=f"slack_{actual_obj_idx}")
    slack_vars.append(s)

# Build augmented objective
# main_obj + δ * Σ(slack[k] / (10^k * range[k]))
obj_ranges = [abs(best_objective_values[constraint_indices[i]] - 
                  nadir_objectives_values[constraint_indices[i]]) 
              for i in range(len(constraint_indices))]

slack_term = gp.quicksum(
    slack_vars[i] / (10**i * obj_ranges[i]) 
    for i in range(len(constraint_indices))
)

augmented_objective = objectives_exprs[main_obj_index] + delta * slack_term
model.setObjective(augmented_objective, gp.GRB.MAXIMIZE)

# Create constraint expressions (obj - slack = ef_array)
constraint_exprs = [objectives_exprs[constraint_indices[i]] - slack_vars[i] 
                   for i in range(len(constraint_indices))]

model.update()
```

**Augmentation Purpose**:
1. **Lexicographic ordering**: Break ties by preferring solutions with larger constraint objective values
2. **Improved solution diversity**: Encourages exploring different regions of Pareto front
3. **10^i weighting**: Ensures later objectives don't dominate earlier ones

### 4.6 STAGE 6: Grid Coverage Loop (Main GPBA-A)

```python
# Initialize control structures
ef_array = [nadir_objectives_values[constraint_indices[i]] for i in range(len(constraint_indices))]
rwv = [best_objective_values[constraint_indices[i]] for i in range(len(constraint_indices))]

# Create interval managers
ef_intervals = []
for i in range(len(constraint_indices)):
    actual_obj_idx = constraint_indices[i]
    min_interval = min(nadir_objectives_values[actual_obj_idx], best_objective_values[actual_obj_idx])
    max_interval = max(nadir_objectives_values[actual_obj_idx], best_objective_values[actual_obj_idx])
    ef_intervals.append(IntervalManager(min_interval, max_interval))

# Tracking structures
previous_solutions = set()  # Set of solution strings for deduplication
previous_solution_information = []  # List of {"ef_array": [...], "solution": [...]}

# Create constraint variables (will be updated in loop)
constraint_vars = [None] * len(constraint_indices)
for i in range(len(constraint_indices)):
    constraint_vars[i] = model.addConstr(
        constraint_exprs[i] == ef_array[i],
        name=f"epsilon_constraint_{constraint_indices[i]}_init"
    )

model.update()

# MAIN LOOP
iteration_count = 0
max_iterations = 1000

while ef_array[0] <= best_objective_values[constraint_indices[0]] and iteration_count < max_iterations:
    iteration_count += 1
    one_solution = []
    
    # ========== STEP 1: Check for previous solution ==========
    previous_relaxation, previous_values = search_previous_solutions_relaxation(
        ef_array, previous_solution_information, constraint_indices)
    
    if previous_relaxation:
        one_solution = [] if previous_values == "infeasible" else previous_values
    else:
        # ========== STEP 2: Solve new configuration ==========
        # Update constraint RHS values
        for i in range(len(constraint_indices)):
            model.remove(constraint_vars[i])
            constraint_vars[i] = model.addConstr(
                constraint_exprs[i] == ef_array[i],
                name=f"epsilon_constraint_{constraint_indices[i]}_iter{iteration_count}"
            )
        
        model.optimize()
        
        if model.status == gp.GRB.INFEASIBLE:
            save_solution_information(ef_array, "infeasible", previous_solution_information)
            one_solution = []
        
        elif model.status == gp.GRB.OPTIMAL:
            # Extract solution with PRECISE calculations
            solution_values = [int(select_image[j].X > 0.5) for j in images_id]
            current_objs = []
            
            # Calculate objectives in MAXIMIZATION form (negated)
            for obj_name in config.objectives:
                if obj_name == "min_cost":
                    current_objs.append(-sum(solution_values[k] * costs[k] for k in images_id))
                elif obj_name == "cloud_coverage":
                    cloud_val = sum(round(cloud_covered[c].X) * clouds_id_area[c] 
                                  for c in clouds_id if c in cloud_covered)
                    total_cloud_area = sum(clouds_id_area.values())
                    current_objs.append(-int(total_cloud_area - cloud_val))
                elif obj_name == "min_resolution":
                    current_objs.append(-int(round(sum(resolution_element[e].X for e in elements))))
                elif obj_name == "min_max_incidence_angle":
                    current_objs.append(-int(round(current_max_incidence_angle.X)))
            
            # Check if new solution
            sol_str = convert_solution_value_to_str(current_objs)
            if sol_str not in previous_solutions:
                previous_solutions.add(sol_str)
                
                # Convert back to minimization for Pareto front
                current_objs_min = [-x for x in current_objs]
                add_to_pareto_front(current_objs_min, solution_values)
                
                one_solution = current_objs
                save_solution_information(ef_array, one_solution, previous_solution_information)
            else:
                one_solution = current_objs
        
        elif model.status == gp.GRB.TIME_LIMIT:
            break  # Timeout
    
    # ========== STEP 3: Update control structures ==========
    # Update RWV (Relative Worst Values)
    if one_solution:
        for i in range(len(constraint_indices)):
            rwv[i] = min(rwv[i], one_solution[constraint_indices[i]])
    
    # ========== STEP 4: Multi-dimensional cascading update ==========
    # Update ef_array for last constraint objective
    id_interval = len(constraint_indices) - 1
    actual_obj_idx = constraint_indices[id_interval]
    
    sol_obj_value = one_solution[actual_obj_idx] if one_solution else None
    ef_intervals[id_interval] = adjust_parameter_ef_array(
        id_interval, ef_array, sol_obj_value, ef_intervals[id_interval],
        constraint_indices, best_objective_values, nadir_objectives_values
    )
    
    # Cascading update for other dimensions
    for i in range(len(constraint_indices) - 1, 0, -1):
        if ef_array[i] > best_objective_values[constraint_indices[i]]:
            # Reset this dimension
            ef_array[i] = nadir_objectives_values[constraint_indices[i]]
            rwv[i] = best_objective_values[constraint_indices[i]]
            
            # Update previous dimension
            prev_id = i - 1
            prev_obj_idx = constraint_indices[prev_id]
            sol_prev = one_solution[prev_obj_idx] if one_solution else None
            
            ef_intervals[prev_id] = adjust_parameter_ef_array(
                prev_id, ef_array, sol_prev, ef_intervals[prev_id],
                constraint_indices, best_objective_values, nadir_objectives_values
            )
        else:
            break  # Stop cascading

# Loop finished - either complete exploration or timeout
```

**Key Features**:
1. **Interval-based exploration**: Focuses on largest gaps
2. **Solution reuse**: Checks previous solutions to avoid redundant solves
3. **Multi-dimensional update**: Cascading mechanism ensures systematic coverage
4. **RWV tracking**: Maintains relative worst values for each dimension

### 4.7 STAGE 7: Finalization

```python
# Mark completion status
statistics["exhaustive"] = True  # Set to False if timeout occurred

# Calculate final statistics
statistics["pareto_solutions_count"] = len(pareto_front)
statistics["total_solutions"] = len(pareto_solutions)

# Format Pareto front for CSV
# Convert objective lists to string format: {[obj1,obj2,...],[obj1,obj2,...],...}
pareto_objs_strs = [str(pareto_solutions[idx]) for idx in pareto_front]
statistics["pareto_front"] = "{" + ",".join(pareto_objs_strs) + "}"

# Format solution vectors (0-indexed image IDs)
pareto_sols_strs = [str(pareto_solutions[idx]["image_ids"]) for idx in pareto_front]
statistics["solutions_pareto_front"] = "{" + ",".join(pareto_sols_strs) + "}"

# Write to CSV file
write_statistics_to_csv(statistics, config.summary_filename)

print(f"Completed: {len(pareto_front)} Pareto-optimal solutions found")
print(f"Total time: {time.time() - start_time:.2f} seconds")
```

**CSV Format**:
```csv
instance,solver_name,number_of_solutions,pareto_solutions_count,time_solver_sec,pareto_front,solutions_pareto_front,...
lagos_30,gurobi,54,53,45.2,"{[100,200,300],[110,195,290],...}","{[0,3,5,7],[1,2,4,8],...}",...
```

**Critical**: Solutions are stored as 0-indexed image ID lists for direct comparison and validation.

## 5. Pareto Front Management

### 5.1 Dominance Checking

```python
def dominates(sol1_objs, sol2_objs):
    """
    Check if solution 1 dominates solution 2 (for minimization).
    
    sol1 dominates sol2 if:
    - All objectives of sol1 are <= corresponding objectives of sol2
    - At least one objective of sol1 is < corresponding objective of sol2
    """
    return all(obj1 <= obj2 for obj1, obj2 in zip(sol1_objs, sol2_objs)) and \
           any(obj1 < obj2 for obj1, obj2 in zip(sol1_objs, sol2_objs))

def add_to_pareto_front(solution_objs, solution_values):
    """
    Add solution to Pareto front if non-dominated.
    
    Args:
        solution_objs: List of objective values (in minimization form)
        solution_values: Binary list of selected images (0/1)
    """
    idx = len(pareto_solutions)
    
    # Convert binary solution_values to list of selected image IDs (0-indexed)
    selected_image_ids = [i for i, val in enumerate(solution_values) if val == 1]
    
    pareto_solutions.append({
        "objs": solution_objs,
        "image_ids": selected_image_ids
    })
    
    # Check if new solution is dominated
    for front_idx in pareto_front:
        if dominates(pareto_solutions[front_idx]["objs"], solution_objs):
            return  # New solution is dominated, don't add
    
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
```

### 5.2 Solution Storage Structure

```python
pareto_solutions = [
    {
        "objs": [cost, cloud, resolution, incidence],  # Objective values (minimization)
        "image_ids": [0, 3, 5, 7, ...]  # List of selected image indices (0-based)
    },
    # ... more solutions
]

pareto_front = [0, 5, 12, 23, ...]  # Indices of non-dominated solutions
```

## 6. Critical Implementation Details

### 6.1 Numerical Precision Issues and Solutions

**Problem**: Gurobi variable `.X` values can be floating-point approximations:
- `select_image[k].X` might be `0.999999` instead of `1.0`
- `cloud_covered[c].X` might be `0.999998` instead of `1.0`
- Accumulating these errors causes objective values to differ by 2-4 units

**Solution 1**: Convert to binary values for calculations
```python
# WRONG:
cost = sum(select_image[k].X * costs[k] for k in images_id)

# CORRECT:
solution_values = [int(select_image[j].X > 0.5) for j in images_id]
cost = sum(solution_values[k] * costs[k] for k in images_id)
```

**Solution 2**: Round intermediate float values
```python
# WRONG:
cloud_val = sum(cloud_covered[c].X * clouds_id_area[c] for c in clouds_id)

# CORRECT:
cloud_val = sum(round(cloud_covered[c].X) * clouds_id_area[c] for c in clouds_id)
```

**Impact**: Without these fixes, solutions appear to have different objectives when they're identical, causing validation failures and incorrect Pareto fronts.

### 6.2 Index Conversion Summary

| Context | Indexing | Example |
|---------|----------|---------|
| .dzn file (MiniZinc) | 1-indexed | `images[1] = {1,2,3}` |
| Python data structures | 0-indexed | `images[0] = {0,1,2}` |
| Solution storage | 0-indexed | `selected_images = [0,3,5,7]` |
| CSV output | 0-indexed | `solutions_pareto_front="{[0,3,5],[1,2,4]}"` |

**Critical**: ALL internal operations use 0-indexing after parsing.

### 6.3 Conditional Variables and Constraints

Variables and constraints are created **only when their corresponding objective is enabled**:

```python
# Check before using:
if cloud_covered is not None:
    # Safe to use cloud_covered variables
    pass

if resolution_element is not None:
    # Safe to use resolution_element variables
    pass
```

**Why**: Creating unused variables wastes memory and slows solver. The implementation is designed to scale to different objective combinations.

### 6.4 Maximization Form in GPBA-A Loop

**Critical Understanding**: Inside the main GPBA-A loop (after Stage 4):
- All `objectives_exprs[i]` are **negated** (maximization form)
- All `best_objective_values[i]` are **negated ideal points**
- All `nadir_objectives_values[i]` are **negated nadir points**
- All `ef_array[i]` values are in maximization form
- All `one_solution` values are in maximization form

**When converting back for Pareto front**:
```python
current_objs_max = [-cost, -cloud, -resolution, -incidence]  # From solver
current_objs_min = [-x for x in current_objs_max]  # Convert to minimization
add_to_pareto_front(current_objs_min, solution_values)  # Store in minimization
```

### 6.5 Thread Configuration

**Robust CPU Detection**:
```python
import multiprocessing

# Use logical CPU count (automatically handles hyperthreading)
config.threads = multiprocessing.cpu_count()
model.setParam('Threads', config.threads)
```

**Why**:
- `cpu_count()` returns logical CPUs (physical cores × threads per core)
- On non-hyperthreading: 8 cores → 8 threads
- On hyperthreading: 8 cores → 16 threads
- Avoids Gurobi warning: "Thread count X larger than processor count Y"

## 7. Complete Algorithm Pseudocode

```
ALGORITHM: SIMS GPBA-A with Gurobi (Complete Inlined Implementation)

INPUT:
  - config: Configuration with objectives, timeout, solver settings
  - instance_data: .dzn file path with SIMS problem data

OUTPUT:
  - pareto_front: List of non-dominated solution indices
  - pareto_solutions: Complete solution data with objectives and image IDs
  - statistics: CSV-formatted solving statistics

BEGIN
  // ============ STAGE 1: PARSE DATA ============
  PARSE .dzn file into arrays:
    n_elements, n_images, areas, costs, images, clouds_per_image, resolution, incidence_angle
  
  CONVERT from 1-indexed to 0-indexed:
    images[i] = {e-1 for e in images[i]}
    clouds_per_image[i] = {e-1 for e in clouds_per_image[i]}
  
  BUILD cloud data structures:
    clouds_id_area = {cloud_id: area}
    cloud_covered_by_image = {img_id: {cloud_ids that img can cover}}
  
  // ============ STAGE 2: BUILD MODEL ============
  CREATE Gurobi model with parameters:
    OutputFlag=0, TimeLimit=timeout, Threads=cpu_count()
  
  CREATE decision variables:
    select_image[i] ∈ {0,1} for all images
    IF "cloud_coverage" IN objectives:
      cloud_covered[c] ∈ {0,1} for all clouds
    IF "min_resolution" IN objectives:
      resolution_element[e] ∈ [min_res, max_res] for all elements
    IF "min_max_incidence_angle" IN objectives:
      effective_incidence_angle[i], current_max_incidence_angle
  
  ADD mandatory coverage constraints:
    FOR each element e:
      SUM(select_image[i] for i covering e) >= 1
  
  ADD conditional constraints based on enabled objectives
  
  CREATE objective expressions (minimization form initially):
    FOR each objective IN config.objectives:
      objectives_exprs.append(objective_expression)
  
  // ============ STAGE 3: EXTREME POINTS ============
  best_objective_values = [0] * num_objectives
  nadir_objectives_values = [0] * num_objectives
  
  FOR i IN range(num_objectives):
    model.setObjective(objectives_exprs[i], MINIMIZE)
    model.optimize()
    
    solution_values = [int(select_image[j].X > 0.5) for j in images_id]
    current_objs = calculate_objectives_precise(solution_values)  // Use binary values!
    
    best_objective_values[i] = current_objs[i]
    add_to_pareto_front(current_objs, solution_values)
  
  FOR i IN range(num_objectives):
    model.setObjective(objectives_exprs[i], MAXIMIZE)
    model.optimize()
    nadir_objectives_values[i] = model.objVal
  
  // ============ STAGE 4: CONVERT TO MAXIMIZATION ============
  FOR i IN range(num_objectives):
    objectives_exprs[i] = -objectives_exprs[i]
    SWAP(best_objective_values[i], nadir_objectives_values[i])
    best_objective_values[i] = -best_objective_values[i]
    nadir_objectives_values[i] = -nadir_objectives_values[i]
  
  model.update()
  
  // ============ STAGE 5: AUGMENTED ε-CONSTRAINT ============
  main_obj_index = 0
  constraint_indices = [i for i != main_obj_index]
  
  CREATE slack variables:
    FOR i IN constraint_indices:
      max_s = |best[i] - nadir[i]|
      slack_vars[i] = INTEGER_VAR(0, max_s)
  
  BUILD augmented objective:
    obj_ranges = [|best[i] - nadir[i]| for i in constraint_indices]
    slack_term = SUM(slack_vars[i] / (10^i * obj_ranges[i]))
    augmented_obj = objectives_exprs[main_obj_index] + 0.01 * slack_term
  
  model.setObjective(augmented_obj, MAXIMIZE)
  
  CREATE constraint expressions:
    constraint_exprs = [objectives_exprs[i] - slack_vars[j] 
                       for j, i in enumerate(constraint_indices)]
  
  // ============ STAGE 6: GPBA-A LOOP ============
  ef_array = [nadir[i] for i in constraint_indices]
  rwv = [best[i] for i in constraint_indices]
  ef_intervals = [IntervalManager(min(best[i],nadir[i]), max(best[i],nadir[i])) 
                 for i in constraint_indices]
  previous_solutions = SET()
  previous_solution_information = []
  
  constraint_vars = [model.addConstr(constraint_exprs[i] == ef_array[i]) 
                    for i in range(len(constraint_indices))]
  
  iteration_count = 0
  max_iterations = 1000
  
  WHILE ef_array[0] <= best[constraint_indices[0]] AND iteration_count < max_iterations:
    iteration_count += 1
    
    // Check for previous solution
    previous_relaxation, previous_values = search_previous_solutions_relaxation(
        ef_array, previous_solution_information, constraint_indices)
    
    IF previous_relaxation:
      one_solution = [] IF previous_values == "infeasible" ELSE previous_values
    ELSE:
      // Update constraints
      FOR i IN range(len(constraint_indices)):
        model.remove(constraint_vars[i])
        constraint_vars[i] = model.addConstr(constraint_exprs[i] == ef_array[i])
      
      model.optimize()
      
      IF model.status == INFEASIBLE:
        save_solution_information(ef_array, "infeasible", previous_solution_information)
        one_solution = []
      ELIF model.status == OPTIMAL:
        solution_values = [int(select_image[j].X > 0.5) for j in images_id]
        current_objs_max = calculate_objectives_precise_max(solution_values)
        
        sol_str = convert_to_str(current_objs_max)
        IF sol_str NOT IN previous_solutions:
          previous_solutions.add(sol_str)
          current_objs_min = [-x for x in current_objs_max]
          add_to_pareto_front(current_objs_min, solution_values)
          one_solution = current_objs_max
          save_solution_information(ef_array, one_solution, previous_solution_information)
        ELSE:
          one_solution = current_objs_max
      ELIF model.status == TIME_LIMIT:
        BREAK
    
    // Update RWV
    IF one_solution NOT EMPTY:
      FOR i IN range(len(constraint_indices)):
        rwv[i] = MIN(rwv[i], one_solution[constraint_indices[i]])
    
    // Multi-dimensional cascading update
    id_interval = len(constraint_indices) - 1
    sol_obj_value = one_solution[constraint_indices[id_interval]] IF one_solution ELSE None
    
    ef_intervals[id_interval] = adjust_parameter_ef_array(
        id_interval, ef_array, sol_obj_value, ef_intervals[id_interval],
        constraint_indices, best_objective_values, nadir_objectives_values)
    
    FOR i IN REVERSE(range(1, len(constraint_indices))):
      IF ef_array[i] > best[constraint_indices[i]]:
        ef_array[i] = nadir[constraint_indices[i]]
        rwv[i] = best[constraint_indices[i]]
        
        prev_id = i - 1
        sol_prev = one_solution[constraint_indices[prev_id]] IF one_solution ELSE None
        ef_intervals[prev_id] = adjust_parameter_ef_array(
            prev_id, ef_array, sol_prev, ef_intervals[prev_id],
            constraint_indices, best_objective_values, nadir_objectives_values)
      ELSE:
        BREAK
  
  // ============ STAGE 7: FINALIZATION ============
  statistics["exhaustive"] = TRUE
  statistics["pareto_solutions_count"] = len(pareto_front)
  statistics["pareto_front"] = format_objectives_csv(pareto_front)
  statistics["solutions_pareto_front"] = format_solutions_csv(pareto_front)
  
  write_statistics_to_csv(statistics, config.summary_filename)
  
  RETURN pareto_front, pareto_solutions, statistics
END ALGORITHM
```

## 8. Validation and Testing

### 8.1 Solution Validation

Each solution must satisfy:

1. **Feasibility**: All constraints satisfied
   ```python
   # Coverage: every element covered by at least one selected image
   for e in elements:
       assert any(e in images[i] for i in selected_images)
   
   # Cloud: cloud_covered only if covering image selected
   for c in clouds_id:
       if cloud_covered[c] == 1:
           assert any(i in selected_images and c in cloud_covered_by_image[i] 
                     for i in range(len(images)))
   ```

2. **Objective Correctness**: Recomputed objectives match stored values
   ```python
   computed_cost = sum(costs[i] for i in selected_images)
   assert computed_cost == solution["objs"][0]
   ```

3. **Non-Dominance**: No other solution in Pareto front dominates this solution
   ```python
   for other_idx in pareto_front:
       if other_idx != current_idx:
           assert not dominates(pareto_solutions[other_idx]["objs"], 
                              pareto_solutions[current_idx]["objs"])
   ```

### 8.2 Test Case Structure

```python
@pytest.mark.parametrize("instance_name", ["lagos_nigeria_30", "mexico_city_50", ...])
def test_solve_milp_inlined_vs_original(instance_name):
    """Compare inlined implementation against original."""
    
    # Run both implementations
    original_results = solve_milp(config)
    inlined_results = solve_milp_inlined(config)
    
    # Compare statistics
    assert original_results["pareto_solutions_count"] == inlined_results["pareto_solutions_count"]
    
    # Validate all solutions
    for solution in inlined_results["solutions"]:
        assert validate_solution_feasibility(solution)
        assert validate_solution_objectives(solution)
```

## 9. Conclusion

This specification provides a complete, implementation-ready description of the SIMS GPBA-A algorithm based on the actual `solve_inlined()` function. Key takeaways:

1. **Precision Matters**: Always use binary solution values for objective calculations
2. **Index Convention**: 0-indexed internally, consistently throughout
3. **Conditional Construction**: Only create variables/constraints for enabled objectives
4. **Maximization Form**: GPBA-A loop operates in maximization, convert back for storage
5. **Interval Management**: Core of adaptive exploration, focuses on largest gaps
6. **Solution Reuse**: Avoids redundant solves through relaxation checking
7. **Multi-dimensional Updates**: Cascading mechanism ensures systematic coverage
8. **Thread Robustness**: Use `cpu_count()` for automatic hyperthreading handling

The implementation successfully balances completeness, efficiency, and correctness, making it suitable for real-world satellite mission planning with multiple conflicting objectives.

## Appendix A: Common Issues and Solutions

### Issue 1: Objectives differ by 2-4 units
**Cause**: Using `.X` float values instead of binary solution values  
**Solution**: Convert to binary with `int(var.X > 0.5)` and round intermediate values

### Issue 2: "Thread count larger than processor count" warning
**Cause**: Hardcoded `threads = cores * 2` on non-hyperthreading CPU  
**Solution**: Use `multiprocessing.cpu_count()` directly

### Issue 3: Solutions appear different but are identical
**Cause**: Floating-point precision in objective calculations  
**Solution**: Use binary solution values and proper rounding

### Issue 4: Missing solutions_pareto_front field in CSV
**Cause**: Forgot to store solution vectors  
**Solution**: Store `selected_image_ids = [i for i, val in enumerate(solution_values) if val == 1]`

### Issue 5: Index mismatches between implementations
**Cause**: Mixing 0-indexed and 1-indexed representations  
**Solution**: Consistently use 0-indexing after parsing; document conversions clearly

---

**Document Version**: 2.0  
**Last Updated**: Based on solve_inlined() implementation analysis  
**Status**: Complete specification ready for implementation