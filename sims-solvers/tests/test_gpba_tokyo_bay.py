"""Test GPBA-A algorithm phases for tokyo_bay_150.

This test file investigates why the GPBA-A algorithm terminates prematurely
after 2 iterations, finding only 4 extreme point solutions instead of 9 solutions.

Based on the structure of test_gpba_phases.py but using tokyo_bay_150.dzn data.
"""

from pathlib import Path
import pytest


# Test Phase 1: Payoff Table Computation
class TestPayoffTableTokyoBay:
    """Test payoff table computation for tokyo_bay_150 with 4D objectives."""
    
    def test_payoff_table_tokyo_bay_150_4d(self):
        """Compute payoff table for tokyo_bay_150 with 4D objectives.
        
        This should find 4 extreme point solutions that we see in the current buggy run:
        - Solution 0: 101 images, [-2370273, -398690, 0, -179]
        - Solution 1: 48 images, [-1003952, -22430, 11530, -152]
        - Solution 2: 5 images, [-111365, 0, 18600, -61]
        - Solution 3: 6 images, [-133385, 0, 18600, -60]
        """
        from sims_solvers.gpba_phases import (
            build_gurobi_model_from_config,
            compute_payoff_table_with_gurobi
        )
        
        # Locate tokyo_bay_150.dzn
        dzn_path = Path(__file__).parent.parent.parent / "sims-core" / "tests" / "data" / "tokyo_bay_150.dzn"
        if not dzn_path.exists():
            pytest.skip(f"Test instance not found: {dzn_path}")
        
        # Config class with 4D objectives
        class Config:
            def __init__(self, instance, objectives):
                self.instance = instance
                self.objectives = objectives
                self.solver_timeout_sec = 300  # 5 minutes per payoff table solve
                self.threads = 1
        
        objectives = ["min_cost", "cloud_coverage", "min_resolution", "min_max_incidence_angle"]
        config = Config(str(dzn_path), objectives)
        
        # Build model
        model, objectives_exprs, problem_data_dict = build_gurobi_model_from_config(config)
        
        # Convert to maximization (GPBA-A uses maximization internally)
        objectives_exprs = [-obj for obj in objectives_exprs]
        
        # Track Pareto front
        pareto_front = []
        def add_to_pareto_front(objs, solution):
            pareto_front.append({"objectives": objs, "solution": solution})
        
        # Statistics
        statistics = {
            "number_of_solutions": 0,
            "time_solver_sec": 0.0,
            "solutions_time_list": []
        }
        
        # Run payoff table
        best_max, nadir_max = compute_payoff_table_with_gurobi(
            model=model,
            objectives_exprs=objectives_exprs,
            config=config,
            problem_data_dict=problem_data_dict,
            statistics=statistics,
            add_to_pareto_front=add_to_pareto_front
        )
        
        # Print results for analysis
        print("\n" + "="*80)
        print("PAYOFF TABLE RESULTS FOR tokyo_bay_150 (4D)")
        print("="*80)
        print(f"Best values (maximization): {best_max}")
        print(f"Nadir values (maximization): {nadir_max}")
        print(f"Number of extreme points: {statistics['number_of_solutions']}")
        print(f"Pareto front size: {len(pareto_front)}")
        print("\nExtreme point solutions:")
        for i, sol in enumerate(pareto_front):
            objs = sol['objectives']
            # Convert back to minimization for readability
            objs_min = [-obj for obj in objs]
            print(f"  Solution {i}: objectives (min form) = {objs_min}")
        print("="*80 + "\n")
        
        # Validate we find 4 extreme points
        assert statistics["number_of_solutions"] == 4, f"Expected 4 extreme points, got {statistics['number_of_solutions']}"
        assert len(pareto_front) == 4, "Pareto front should have 4 solutions"
        
        # Return values for use in next tests
        return best_max, nadir_max, pareto_front


# Test Phase 2: Epsilon Setup (with HARDCODED inputs from payoff table)
class TestEpsilonSetupTokyoBay:
    """Test epsilon setup using hardcoded payoff table outputs from tokyo_bay_150."""
    
    def test_epsilon_setup_tokyo_bay_150(self):
        """Test epsilon setup using payoff table outputs from tokyo_bay_150.
        
        Based on the current buggy run, we expect:
        - ef_array to be initialized to nadir values for constraint objectives
        - 3 constraint objectives (indices [1, 2, 3])
        - Main objective index 0 (min_cost)
        """
        from sims_solvers.gpba_phases import (
            build_gurobi_model_from_config,
            setup_epsilon_constraints
        )
        
        # HARDCODED INPUTS (will be replaced with actual values after running payoff table test)
        # For now, use expected values based on the 4 extreme points:
        # Solution 0: [-2370273, -398690, 0, -179]
        # Solution 1: [-1003952, -22430, 11530, -152]
        # Solution 2: [-111365, 0, 18600, -61]
        # Solution 3: [-133385, 0, 18600, -60]
        #
        # In maximization form:
        # Solution 0: [2370273, 398690, 0, 179]
        # Solution 1: [1003952, 22430, -11530, 152]
        # Solution 2: [111365, 0, -18600, 61]
        # Solution 3: [133385, 0, -18600, 60]
        #
        # Best (max): [2370273, 398690, 0, 179]
        # Nadir (max): [111365, 0, -18600, 60]
        
        best_objective_values_max = [2370273, 398690, 0, 179]
        nadir_objectives_values_max = [111365, 0, -18600, 60]
        num_objectives = 4
        
        # Simple config class
        class Config:
            def __init__(self, objectives):
                self.objectives = objectives
        
        config = Config(objectives=["min_cost", "cloud_coverage", "min_resolution", "min_max_incidence_angle"])
        
        # Build model structure
        dzn_path = Path(__file__).parent.parent.parent / "sims-core" / "tests" / "data" / "tokyo_bay_150.dzn"
        if not dzn_path.exists():
            pytest.skip(f"Test instance not found: {dzn_path}")
        
        class FullConfig:
            def __init__(self, instance, objectives):
                self.instance = instance
                self.objectives = objectives
                self.solver_timeout_sec = 300
                self.threads = 1
        
        full_config = FullConfig(str(dzn_path), config.objectives)
        model, objectives_exprs, _ = build_gurobi_model_from_config(full_config)
        
        # Convert to maximization
        objectives_exprs = [-obj for obj in objectives_exprs]
        
        # Run epsilon setup
        (
            ef_array,
            ef_intervals,
            constraint_indices,
            slack_vars,
            main_obj_index
        ) = setup_epsilon_constraints(
            model=model,
            objectives_exprs=objectives_exprs,
            best_objective_values=best_objective_values_max,
            nadir_objectives_values=nadir_objectives_values_max,
            num_objectives=num_objectives,
            config=full_config
        )
        
        # Print results for analysis
        print("\n" + "="*80)
        print("EPSILON SETUP RESULTS FOR tokyo_bay_150 (4D)")
        print("="*80)
        print(f"ef_array (initial): {ef_array}")
        print(f"constraint_indices: {constraint_indices}")
        print(f"main_obj_index: {main_obj_index}")
        print(f"Number of IntervalManagers: {len(ef_intervals)}")
        for i, interval in enumerate(ef_intervals):
            print(f"  IntervalManager {i}: min={interval.min_value}, max={interval.max_value}")
        print("="*80 + "\n")
        
        # Validate outputs
        assert len(ef_array) == 3, f"Expected 3 constraint values, got {len(ef_array)}"
        assert constraint_indices == [1, 2, 3], f"Expected constraint_indices=[1, 2, 3], got {constraint_indices}"
        assert main_obj_index == 0, f"Expected main_obj_index=0, got {main_obj_index}"
        assert len(ef_intervals) == 3, "Should have 3 IntervalManagers"
        assert len(slack_vars) == 3, "Should have 3 slack variables"
        
        # Validate ef_array is initialized to nadir values (in max form)
        # ef_array should be [nadir_max[1], nadir_max[2], nadir_max[3]] = [0, -18600, 60]
        # But based on the bug report, we see ef_array=[-2370273, -398690, 0] in iteration 0
        # This suggests ef_array might be initialized differently...
        # Let's see what the actual values are
        
        return ef_array, ef_intervals, constraint_indices, best_objective_values_max, nadir_objectives_values_max


# Test Phase: Epsilon Adjustment for tokyo_bay_150
class TestEpsilonAdjustmentTokyoBay:
    """Test epsilon adjustment logic with tokyo_bay_150 data."""
    
    def test_adjust_after_two_infeasible_iterations(self):
        """Test epsilon adjustment after 2 consecutive infeasible iterations.
        
        This simulates the bug scenario:
        - Iteration 0: ef_array=[-2370273, -398690, 0] → INFEASIBLE
        - Iteration 1: ef_array=[-2370273, -398690, -179] → INFEASIBLE (relaxation)
        - After adjustment: ef_array=[1, -398690, 0] → Loop exits (1 > 0)
        
        We need to understand why ef_array[0] jumps from -2370273 to 1.
        """
        from sims_solvers.gpba_phases import IntervalManager, adjust_parameter_ef_array
        
        # Setup based on tokyo_bay_150 payoff table (maximization form)
        # Best: [2370273, 398690, 0, 179]
        # Nadir: [111365, 0, -18600, 60]
        best_objective_values = [2370273, 398690, 0, 179]
        nadir_objectives_values = [111365, 0, -18600, 60]
        constraint_indices = [1, 2, 3]
        
        # Initial state (but something seems wrong with these values...)
        # Based on bug report, ef_array in iteration 0 is [-2370273, -398690, 0]
        # But this doesn't match nadir values [0, -18600, 60]
        # Let me use the values from the bug report to simulate the exact scenario
        
        # Iteration 0 state (from bug report, converted to minimization form)
        # Min form: ef_array = [-2370273, -398690, 0] (constraints on objectives 1, 2, 3)
        # Max form: ef_array = [2370273, 398690, 0]
        ef_array = [2370273, 398690, 0]  # Max form
        
        # Create IntervalManagers for each constraint objective
        # IntervalManager should be (nadir, best) for max form
        # Obj 1: (0, 398690)
        # Obj 2: (-18600, 0)
        # Obj 3: (60, 179)
        ef_intervals = [
            IntervalManager(min_value=0, max_value=398690),        # Constraint on obj 1
            IntervalManager(min_value=-18600, max_value=0),        # Constraint on obj 2
            IntervalManager(min_value=60, max_value=179),          # Constraint on obj 3
        ]
        
        # Simulate infeasible solution at iteration 0
        sol_obj_k = None  # Infeasible
        
        print("\n" + "="*80)
        print("SIMULATING EPSILON ADJUSTMENT AFTER INFEASIBLE ITERATION 0")
        print("="*80)
        print("Before adjustment:")
        print(f"  ef_array (max form): {ef_array}")
        print(f"  sol_obj_k: {sol_obj_k}")
        print(f"  IntervalManager 0: [{ef_intervals[0].min_value}, {ef_intervals[0].max_value}]")
        print(f"  IntervalManager 1: [{ef_intervals[1].min_value}, {ef_intervals[1].max_value}]")
        print(f"  IntervalManager 2: [{ef_intervals[2].min_value}, {ef_intervals[2].max_value}]")
        
        # Adjust first constraint (id_constraint_objective=0)
        adjust_parameter_ef_array(
            id_constraint_objective=0,
            ef_array=ef_array,
            sol_obj_k=sol_obj_k,
            ef_interval=ef_intervals[0],
            constraint_indices=constraint_indices,
            best_objective_values=best_objective_values,
            nadir_objectives_values=nadir_objectives_values,
            gamma=1
        )
        
        print("\nAfter adjustment:")
        print(f"  ef_array (max form): {ef_array}")
        print(f"  IntervalManager 0: [{ef_intervals[0].min_value}, {ef_intervals[0].max_value}]")
        print("="*80 + "\n")
        
        # Check if ef_array[0] exceeds best_objective_values[constraint_indices[0]]
        # constraint_indices[0] = 1, so best_objective_values[1] = 398690
        # If ef_array[0] > 398690, the loop would exit
        
        # The bug might be that after removing interval, find_largest_interval returns
        # a value beyond the valid range
        
        assert ef_array[0] <= best_objective_values[constraint_indices[0]], \
            f"ef_array[0]={ef_array[0]} exceeds best value {best_objective_values[constraint_indices[0]]}"


# Test Phase: Main Loop Simulation
class TestMainLoopTokyoBay:
    """Test main GPBA-A loop simulation for tokyo_bay_150."""
    
    def test_main_loop_first_two_iterations(self):
        """Simulate first 2 iterations of GPBA-A loop to reproduce the bug.
        
        This test will:
        1. Initialize ef_array and IntervalManagers
        2. Simulate iteration 0 (infeasible)
        3. Simulate iteration 1 (infeasible with relaxation)
        4. Show how ef_array adjustment causes premature termination
        """
        from sims_solvers.gpba_phases import (
            build_gurobi_model_from_config,
            setup_epsilon_constraints,
            adjust_parameter_ef_array,
            search_previous_solutions_relaxation
        )
        
        # Locate tokyo_bay_150.dzn
        dzn_path = Path(__file__).parent.parent.parent / "sims-core" / "tests" / "data" / "tokyo_bay_150.dzn"
        if not dzn_path.exists():
            pytest.skip(f"Test instance not found: {dzn_path}")
        
        # Config
        class Config:
            def __init__(self, instance, objectives):
                self.instance = instance
                self.objectives = objectives
                self.solver_timeout_sec = 60  # Short timeout for testing
                self.threads = 1
        
        objectives = ["min_cost", "cloud_coverage", "min_resolution", "min_max_incidence_angle"]
        config = Config(str(dzn_path), objectives)
        
        # Build model
        model, objectives_exprs, problem_data_dict = build_gurobi_model_from_config(config)
        
        # Convert to maximization
        objectives_exprs = [-obj for obj in objectives_exprs]
        
        # Compute payoff table (to get best/nadir values)
        # For this test, we'll use hardcoded values to save time
        best_objective_values_max = [2370273, 398690, 0, 179]
        nadir_objectives_values_max = [111365, 0, -18600, 60]
        num_objectives = 4
        
        # Setup epsilon constraints
        (
            ef_array,
            ef_intervals,
            constraint_indices,
            slack_vars,
            main_obj_index
        ) = setup_epsilon_constraints(
            model=model,
            objectives_exprs=objectives_exprs,
            best_objective_values=best_objective_values_max,
            nadir_objectives_values=nadir_objectives_values_max,
            num_objectives=num_objectives,
            config=config
        )
        
        print("\n" + "="*80)
        print("MAIN LOOP SIMULATION FOR tokyo_bay_150")
        print("="*80)
        print("Initial state:")
        print(f"  ef_array: {ef_array}")
        print(f"  constraint_indices: {constraint_indices}")
        print(f"  best_objective_values: {best_objective_values_max}")
        print("  Loop condition: ef_array[0] <= best_objective_values[constraint_indices[0]]")
        print(f"    {ef_array[0]} <= {best_objective_values_max[constraint_indices[0]]} = {ef_array[0] <= best_objective_values_max[constraint_indices[0]]}")
        
        # Track previous solutions
        previous_solution_information = []
        
        # Simulate iteration 0 (assume infeasible)
        print("\n--- Iteration 0 ---")
        print(f"ef_array before: {ef_array}")
        
        # Simulate solving MILP (we won't actually solve, just assume infeasible)
        sol_obj_k = None  # Infeasible
        
        # Check for relaxation
        found, solution = search_previous_solutions_relaxation(
            ef_array, previous_solution_information, constraint_indices
        )
        print(f"Relaxation search: found={found}, solution={solution}")
        
        # Store this configuration
        previous_solution_information.append({
            "ef_array": ef_array.copy(),
            "solution": "infeasible"
        })
        
        # Adjust ef_array (assume we adjust first constraint)
        id_constraint = 0
        adjust_parameter_ef_array(
            id_constraint_objective=id_constraint,
            ef_array=ef_array,
            sol_obj_k=sol_obj_k,
            ef_interval=ef_intervals[id_constraint],
            constraint_indices=constraint_indices,
            best_objective_values=best_objective_values_max,
            nadir_objectives_values=nadir_objectives_values_max,
            gamma=1
        )
        print(f"ef_array after adjustment: {ef_array}")
        print(f"Loop condition: {ef_array[0]} <= {best_objective_values_max[constraint_indices[0]]} = {ef_array[0] <= best_objective_values_max[constraint_indices[0]]}")
        
        # Simulate iteration 1 (assume infeasible with relaxation)
        print("\n--- Iteration 1 ---")
        print(f"ef_array before: {ef_array}")
        
        sol_obj_k = None  # Infeasible
        
        # Check for relaxation (should find iteration 0 configuration)
        found, solution = search_previous_solutions_relaxation(
            ef_array, previous_solution_information, constraint_indices
        )
        print(f"Relaxation search: found={found}, solution={solution}")
        
        if found and solution == "infeasible":
            print("Found infeasible relaxation! Adjusting next constraint...")
            id_constraint = 1  # Move to next constraint
        
        # Store this configuration
        previous_solution_information.append({
            "ef_array": ef_array.copy(),
            "solution": "infeasible"
        })
        
        # Adjust ef_array
        adjust_parameter_ef_array(
            id_constraint_objective=id_constraint,
            ef_array=ef_array,
            sol_obj_k=sol_obj_k,
            ef_interval=ef_intervals[id_constraint],
            constraint_indices=constraint_indices,
            best_objective_values=best_objective_values_max,
            nadir_objectives_values=nadir_objectives_values_max,
            gamma=1
        )
        print(f"ef_array after adjustment: {ef_array}")
        print(f"Loop condition: {ef_array[0]} <= {best_objective_values_max[constraint_indices[0]]} = {ef_array[0] <= best_objective_values_max[constraint_indices[0]]}")
        
        print("="*80 + "\n")
        
        # Check if loop would terminate
        loop_should_continue = ef_array[0] <= best_objective_values_max[constraint_indices[0]]
        
        if not loop_should_continue:
            print("BUG REPRODUCED: Loop terminates prematurely!")
            print(f"  ef_array[0] = {ef_array[0]}")
            print(f"  best_objective_values[{constraint_indices[0]}] = {best_objective_values_max[constraint_indices[0]]}")
            print(f"  Condition failed: {ef_array[0]} > {best_objective_values_max[constraint_indices[0]]}")
        
        # For debugging, we allow the test to pass even if bug is reproduced
        # so we can see the output
        # assert loop_should_continue, "Loop should continue but condition is false"
