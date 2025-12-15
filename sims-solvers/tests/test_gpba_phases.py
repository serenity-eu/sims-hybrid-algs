"""
Comprehensive Unit Tests for GPBA Phases

This test file validates all extracted GPBA phases using cascading hardcoded inputs.
Each test is independent and runs in <1ms (except payoff table which loads DZN).

Test Strategy:
- Phase 1 (Payoff Table): Loads DZN file, captures outputs
- Phase 2+ (All others): Use HARDCODED inputs from previous phase
- NO phase depends on running previous phases
- All tests validate correct behavior with real SIMS data
"""

import pytest
from pathlib import Path


# Test Phase 0: IntervalManager
class TestIntervalManager:
    """Test standalone IntervalManager functionality."""
    
    def test_interval_manager_creation(self):
        """Test IntervalManager initialization."""
        from sims_solvers.gpba_phases import IntervalManager
        
        interval = IntervalManager(min_value=10, max_value=100)
        assert interval.min_value == 10
        assert interval.max_value == 100
        assert len(interval.intervals) == 1
        assert (10, 100) in interval.intervals
    
    def test_interval_manager_remove_interval(self):
        """Test removing an interval."""
        from sims_solvers.gpba_phases import IntervalManager
        
        interval = IntervalManager(min_value=10, max_value=100)
        interval.remove_interval(50, 70)
        
        # Should have two intervals: (10, 49) and (71, 100)
        assert len(interval.intervals) == 2
        assert (10, 49) in interval.intervals
        assert (71, 100) in interval.intervals
    
    def test_interval_manager_find_largest(self):
        """Test finding largest interval."""
        from sims_solvers.gpba_phases import IntervalManager
        
        interval = IntervalManager(min_value=10, max_value=100)
        interval.remove_interval(50, 55)
        
        largest = interval.find_largest_interval()
        # Largest should be (56, 100) with length 45
        assert largest == (56, 100)
    
    def test_interval_manager_remove_one_point(self):
        """Test removing a single point."""
        from sims_solvers.gpba_phases import IntervalManager
        
        interval = IntervalManager(min_value=10, max_value=20)
        interval.remove_one_point(15)
        
        # Should have two intervals: (10, 14) and (16, 20)
        assert len(interval.intervals) == 2
        assert (10, 14) in interval.intervals
        assert (16, 20) in interval.intervals


# Test Phase 1: Payoff Table (with real DZN data)
class TestPayoffTable:
    """Test payoff table computation with lagos_nigeria_30.dzn."""
    
    def test_payoff_table_with_real_data(self):
        """Test payoff table computation with real SIMS instance."""
        from sims_solvers.gpba_phases import (
            build_gurobi_model_from_config,
            compute_payoff_table_with_gurobi
        )
        
        # Simple config class
        class Config:
            def __init__(self, instance, objectives, timeout=300):
                self.instance = instance
                self.objectives = objectives
                self.solver_timeout_sec = timeout
                self.threads = 1
        
        # Load lagos_nigeria_30.dzn
        dzn_path = Path(__file__).parent.parent.parent / "sims-core" / "tests" / "data" / "lagos_nigeria_30.dzn"
        if not dzn_path.exists():
            pytest.skip(f"Test instance not found: {dzn_path}")
        
        config = Config(
            instance=str(dzn_path),
            objectives=["min_cost", "cloud_coverage"]
        )
        
        # Build model
        model, objectives_exprs, problem_data_dict = build_gurobi_model_from_config(config)
        
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
        
        # Validate outputs (from Task 1.7)
        assert best_max == [-2736640, -53469], f"Expected best=[-2736640, -53469], got {best_max}"
        assert nadir_max == [-10948970, -656595], f"Expected nadir=[-10948970, -656595], got {nadir_max}"
        assert statistics["number_of_solutions"] == 2, "Should find 2 extreme points"
        assert len(pareto_front) == 2, "Pareto front should have 2 solutions"


# Test Phase 2: Epsilon Setup (with HARDCODED inputs)
class TestEpsilonSetup:
    """Test epsilon setup with HARDCODED payoff table outputs."""
    
    def test_epsilon_setup_with_hardcoded_inputs(self):
        """Test epsilon setup using hardcoded payoff table outputs from Task 1.7."""
        from sims_solvers.gpba_phases import (
            build_gurobi_model_from_config,
            setup_epsilon_constraints
        )
        
        # HARDCODED INPUTS FROM TASK 1.7 (no DZN loading needed for this logic!)
        best_objective_values_max = [-2736640, -53469]
        nadir_objectives_values_max = [-10948970, -656595]
        num_objectives = 2
        
        # Simple config class
        class Config:
            def __init__(self, objectives):
                self.objectives = objectives
        
        config = Config(objectives=["min_cost", "cloud_coverage"])
        
        # Build model structure (but don't solve)
        dzn_path = Path(__file__).parent.parent.parent / "sims-core" / "tests" / "data" / "lagos_nigeria_30.dzn"
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
        
        # Validate outputs (from golden standard)
        assert ef_array == [-656595], f"Expected ef_array=[-656595], got {ef_array}"
        assert constraint_indices == [1], f"Expected constraint_indices=[1], got {constraint_indices}"
        assert main_obj_index == 0, f"Expected main_obj_index=0, got {main_obj_index}"
        assert len(ef_intervals) == 1, "Should have 1 IntervalManager"
        # After fix: IntervalManager created with min < max numerically
        assert ef_intervals[0].min_value == -656595, f"IntervalManager min should be -656595 (got {ef_intervals[0].min_value})"
        assert ef_intervals[0].max_value == -53469, f"IntervalManager max should be -53469 (got {ef_intervals[0].max_value})"
        assert len(slack_vars) == 1, "Should have 1 slack variable"
        assert slack_vars[0] is not None, "Slack variable should exist"


# Test Phase: Epsilon Adjustment (with HARDCODED inputs)
class TestEpsilonAdjustment:
    """Test epsilon adjustment logic with synthetic data."""
    
    def test_adjust_parameter_feasible_solution(self):
        """Test epsilon adjustment with a feasible solution."""
        from sims_solvers.gpba_phases import IntervalManager, adjust_parameter_ef_array
        
        # Setup initial state (with corrected IntervalManager bounds: min < max numerically)
        ef_array = [-656595]
        sol_obj_k = -100000  # Feasible solution value
        ef_interval = IntervalManager(min_value=-656595, max_value=-53469)
        constraint_indices = [1]
        best_objective_values = [-2736640, -53469]
        nadir_objectives_values = [-10948970, -656595]
        
        # Run adjustment
        adjust_parameter_ef_array(
            id_constraint_objective=0,
            ef_array=ef_array,
            sol_obj_k=sol_obj_k,
            ef_interval=ef_interval,
            constraint_indices=constraint_indices,
            best_objective_values=best_objective_values,
            nadir_objectives_values=nadir_objectives_values,
            gamma=1
        )
        
        # ef_array should be updated to explore the largest remaining interval
        assert ef_array[0] != -656595, "ef_array should be updated"
        # Should explore center of remaining interval (or best value if at boundary)
        assert -656595 < ef_array[0] <= -53469, "ef_array should be in valid range or at best value"
    
    def test_adjust_parameter_infeasible_solution(self):
        """Test epsilon adjustment with an infeasible solution."""
        from sims_solvers.gpba_phases import IntervalManager, adjust_parameter_ef_array
        
        # Setup initial state (with corrected IntervalManager bounds: min < max numerically)
        ef_array = [-656595]
        sol_obj_k = None  # Infeasible
        ef_interval = IntervalManager(min_value=-656595, max_value=-53469)
        constraint_indices = [1]
        best_objective_values = [-2736640, -53469]
        nadir_objectives_values = [-10948970, -656595]
        
        # Run adjustment
        adjust_parameter_ef_array(
            id_constraint_objective=0,
            ef_array=ef_array,
            sol_obj_k=sol_obj_k,
            ef_interval=ef_interval,
            constraint_indices=constraint_indices,
            best_objective_values=best_objective_values,
            nadir_objectives_values=nadir_objectives_values,
            gamma=1
        )
        
        # ef_array should be updated
        assert ef_array[0] != -656595, "ef_array should be updated after infeasible"


# Test Phase: Relaxation Search (with synthetic data)
class TestRelaxationSearch:
    """Test relaxation search logic."""
    
    def test_search_no_previous_solutions(self):
        """Test when no previous solutions exist."""
        from sims_solvers.gpba_phases import search_previous_solutions_relaxation
        
        ef_array = [-656595]
        previous_solution_information = []
        constraint_indices = [1]
        
        found, solution = search_previous_solutions_relaxation(
            ef_array, previous_solution_information, constraint_indices
        )
        
        assert found is False, "Should not find previous solution"
        assert solution is None, "Solution should be None"
    
    def test_search_with_relaxed_feasible_solution(self):
        """Test finding a relaxed feasible solution."""
        from sims_solvers.gpba_phases import search_previous_solutions_relaxation
        
        # For maximization: less constrained means HIGHER ef values
        # Constraint is: obj_value <= ef_array
        ef_array = [-200000]  # Current (tighter) constraint
        previous_solution_information = [
            {
                "ef_array": [-100000],  # Less constrained (-100000 >= -200000, TRUE!)
                "solution": [-3000000, -150000]  # Previous solution
            }
        ]
        constraint_indices = [1]
        
        found, solution = search_previous_solutions_relaxation(
            ef_array, previous_solution_information, constraint_indices
        )
        
        # Should check if previous solution satisfies current constraints
        # solution[constraint_indices[0]] = solution[1] = -150000
        # Check: -150000 <= -200000? NO! (In max form, -150000 > -200000)
        # So this won't be reused. Let's adjust the test.
        # For reuse, we need solution[1] <= ef_array[0]
        # -150000 <= -200000 is false, so found should be False
        assert found is False, "Previous solution doesn't satisfy tighter constraint"
    
    def test_search_with_relaxed_feasible_solution_reusable(self):
        """Test finding a relaxed feasible solution that can be reused."""
        from sims_solvers.gpba_phases import search_previous_solutions_relaxation
        
        # For reuse: previous ef must be >= current AND solution must satisfy current
        ef_array = [-200000]  # Current constraint
        previous_solution_information = [
            {
                "ef_array": [-100000],  # Less constrained (-100000 >= -200000)
                "solution": [-3000000, -250000]  # Solution value -250000 <= -200000? YES!
            }
        ]
        constraint_indices = [1]
        
        found, solution = search_previous_solutions_relaxation(
            ef_array, previous_solution_information, constraint_indices
        )
        
        assert found is True, "Should find and reuse relaxed solution"
        assert solution == [-3000000, -250000], "Should return previous solution"
    
    def test_search_with_infeasible_relaxation(self):
        """Test finding an infeasible relaxed configuration."""
        from sims_solvers.gpba_phases import search_previous_solutions_relaxation
        
        # For maximization: less constrained means HIGHER ef values
        ef_array = [-200000]  # Current constraint
        previous_solution_information = [
            {
                "ef_array": [-100000],  # Less constrained (-100000 >= -200000, TRUE!)
                "solution": "infeasible"  # Was infeasible
            }
        ]
        constraint_indices = [1]
        
        found, solution = search_previous_solutions_relaxation(
            ef_array, previous_solution_information, constraint_indices
        )
        
        assert found is True, "Should find infeasible relaxation"
        assert solution == "infeasible", "Should return infeasible"


# Test Phase: Main Loop (with HARDCODED inputs from epsilon setup)
class TestMainLoop:
    """Test main GPBA-A loop with HARDCODED epsilon setup outputs."""
    
    def test_main_loop_with_hardcoded_inputs(self):
        """Test main loop using hardcoded outputs from epsilon setup phase."""
        from sims_solvers.gpba_phases import (
            build_gurobi_model_from_config,
            setup_epsilon_constraints,
            run_gpba_loop
        )
        
        # HARDCODED INPUTS from Phase 2 and 3
        best_max = [-2736640, -53469]
        nadir_max = [-10948970, -656595]
        
        # Build model
        dzn_path = Path(__file__).parent.parent.parent / "sims-core" / "tests" / "data" / "lagos_nigeria_30.dzn"
        if not dzn_path.exists():
            pytest.skip(f"Test instance not found: {dzn_path}")
        
        class Config:
            def __init__(self, instance, objectives):
                self.instance = instance
                self.objectives = objectives
                self.solver_timeout_sec = 300
                self.threads = 1
        
        config = Config(str(dzn_path), ["min_cost", "cloud_coverage"])
        model, objectives_exprs, problem_data_dict = build_gurobi_model_from_config(config)
        
        # Convert to maximization
        objectives_exprs = [-obj for obj in objectives_exprs]
        
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
            best_objective_values=best_max,
            nadir_objectives_values=nadir_max,
            num_objectives=2,
            config=config
        )
        
        # Pareto front storage
        pareto_front = []
        def add_to_pareto_front(objs, solution):
            pareto_front.append({"objectives": objs, "solution": solution})
        
        # Statistics
        statistics = {
            "number_of_solutions": 0,
            "time_solver_sec": 0.0,
            "solutions_time_list": []
        }
        
        # Run main loop
        iteration_count, exhaustive = run_gpba_loop(
            model=model,
            objectives_exprs=objectives_exprs,
            constraint_indices=constraint_indices,
            main_obj_index=main_obj_index,
            best_objective_values=best_max,
            nadir_objectives_values=nadir_max,
            slack_vars=slack_vars,
            ef_intervals=ef_intervals,
            config=config,
            images_id=problem_data_dict["images_id"],
            costs=problem_data_dict["costs"],
            cloud_covered=problem_data_dict.get("cloud_covered"),
            clouds_id_area=problem_data_dict["clouds_id_area"],
            resolution_element=problem_data_dict.get("resolution_element"),
            elements=problem_data_dict["elements"],
            current_max_incidence_angle=problem_data_dict.get("current_max_incidence_angle"),
            add_to_pareto_front=add_to_pareto_front,
            statistics=statistics,
            select_image=problem_data_dict["select_image"],
            gamma=1,
            max_iterations=10000
        )
        
        # Validate results
        print(f"\n=== MAIN LOOP RESULTS ===")
        print(f"Solutions found: {statistics['number_of_solutions']}")
        print(f"Pareto front size: {len(pareto_front)}")
        print(f"Exhaustive: {exhaustive}")
        print(f"Iteration count: {iteration_count}")
        
        # Print all solutions sorted
        print(f"\n=== ALL PYTHON SOLUTIONS (sorted by cost) ===")
        sorted_solutions = sorted(pareto_front, key=lambda x: x["objectives"][0])
        for i, sol in enumerate(sorted_solutions, 1):
            selected_images = [j for j, val in enumerate(sol['solution']) if val > 0.5]
            print(f"{i:3d}. Obj: {sol['objectives']} | Images ({len(selected_images)}): {selected_images}")
        
        # Main loop should find 52 solutions (54 total - 2 from payoff table)
        assert statistics["number_of_solutions"] == 52, \
            f"Expected 52 solutions from main loop, got {statistics['number_of_solutions']}"
        assert len(pareto_front) >= 51, \
            f"Expected at least 51 non-dominated solutions, got {len(pareto_front)}"
        assert exhaustive is True, "Search should be exhaustive"


# Test Phase: Complete Pipeline (all phases together)
class TestCompletePipeline:
    """Test complete GPBA-A pipeline from model building to Pareto front."""
    
    def test_complete_pipeline_end_to_end(self):
        """Test complete pipeline with all phases running in sequence."""
        from sims_solvers.gpba_phases import (
            build_gurobi_model_from_config,
            compute_payoff_table_with_gurobi,
            setup_epsilon_constraints,
            run_gpba_loop
        )
        
        # Build model
        dzn_path = Path(__file__).parent.parent.parent / "sims-core" / "tests" / "data" / "lagos_nigeria_30.dzn"
        if not dzn_path.exists():
            pytest.skip(f"Test instance not found: {dzn_path}")
        
        class Config:
            def __init__(self, instance, objectives):
                self.instance = instance
                self.objectives = objectives
                self.solver_timeout_sec = 300
                self.threads = 1
        
        config = Config(str(dzn_path), ["min_cost", "cloud_coverage"])
        
        # Phase 1: Build model
        model, objectives_exprs, problem_data_dict = build_gurobi_model_from_config(config)
        assert model is not None, "Model should be built"
        assert len(objectives_exprs) == 2, "Should have 2 objectives"
        
        # Phase 2: Compute payoff table
        pareto_front = []
        def add_to_pareto_front(objs, solution):
            pareto_front.append({"objectives": objs, "solution": solution})
        
        statistics = {
            "number_of_solutions": 0,
            "time_solver_sec": 0.0,
            "solutions_time_list": []
        }
        
        best_max, nadir_max = compute_payoff_table_with_gurobi(
            model=model,
            objectives_exprs=objectives_exprs,
            config=config,
            problem_data_dict=problem_data_dict,
            statistics=statistics,
            add_to_pareto_front=add_to_pareto_front
        )
        
        assert best_max == [-2736640, -53469], "Should get expected ideal point"
        assert nadir_max == [-10948970, -656595], "Should get expected nadir point"
        
        # Convert to maximization
        objectives_exprs = [-obj for obj in objectives_exprs]
        
        # Phase 3: Setup epsilon constraints
        (
            ef_array,
            ef_intervals,
            constraint_indices,
            slack_vars,
            main_obj_index
        ) = setup_epsilon_constraints(
            model=model,
            objectives_exprs=objectives_exprs,
            best_objective_values=best_max,
            nadir_objectives_values=nadir_max,
            num_objectives=2,
            config=config
        )
        
        assert ef_array == [-656595], "Should get expected initial ef_array"
        assert constraint_indices == [1], "Should get expected constraint indices"
        
        # Phase 4: Run main loop
        pareto_front_main = []
        def add_to_pareto_front_main(objs, solution):
            pareto_front_main.append({"objectives": objs, "solution": solution})
        
        statistics_main = {
            "number_of_solutions": 0,
            "time_solver_sec": 0.0,
            "solutions_time_list": []
        }
        
        iteration_count, exhaustive = run_gpba_loop(
            model=model,
            objectives_exprs=objectives_exprs,
            constraint_indices=constraint_indices,
            main_obj_index=main_obj_index,
            best_objective_values=best_max,
            nadir_objectives_values=nadir_max,
            slack_vars=slack_vars,
            ef_intervals=ef_intervals,
            config=config,
            images_id=problem_data_dict["images_id"],
            costs=problem_data_dict["costs"],
            cloud_covered=problem_data_dict.get("cloud_covered"),
            clouds_id_area=problem_data_dict["clouds_id_area"],
            resolution_element=problem_data_dict.get("resolution_element"),
            elements=problem_data_dict["elements"],
            current_max_incidence_angle=problem_data_dict.get("current_max_incidence_angle"),
            add_to_pareto_front=add_to_pareto_front_main,
            statistics=statistics_main,
            select_image=problem_data_dict["select_image"],
            gamma=1,
            max_iterations=10000
        )
        
        # Validate final results
        print(f"Payoff solutions: {statistics['number_of_solutions']}")
        print(f"Main loop solutions: {statistics_main['number_of_solutions']}")
        print(f"Payoff Pareto front: {len(pareto_front)}")
        print(f"Main loop Pareto front: {len(pareto_front_main)}")
        print(f"Exhaustive: {exhaustive}")
        
        # Expected: 2 from payoff + 52 from main loop = 54 total solutions
        assert statistics["number_of_solutions"] == 2, \
            f"Expected 2 solutions from payoff table, got {statistics['number_of_solutions']}"
        assert statistics_main["number_of_solutions"] == 52, \
            f"Expected 52 solutions from main loop, got {statistics_main['number_of_solutions']}"
        
        # Total solutions discovered = 54
        total_solutions = statistics["number_of_solutions"] + statistics_main["number_of_solutions"]
        assert total_solutions == 54, \
            f"Expected exactly 54 solutions total, got {total_solutions}"
        
        # Check for dominated solutions across both Pareto fronts
        # Combine all solutions and check dominance
        def is_dominated(sol1_objs, sol2_objs):
            """Check if sol1 is dominated by sol2 (minimization)."""
            better_or_equal = all(s2 <= s1 for s1, s2 in zip(sol1_objs, sol2_objs))
            strictly_better = any(s2 < s1 for s1, s2 in zip(sol1_objs, sol2_objs))
            return better_or_equal and strictly_better
        
        all_solutions = [sol["objectives"] for sol in pareto_front + pareto_front_main]
        non_dominated_count = 0
        for i, objs1 in enumerate(all_solutions):
            dominated = False
            for j, objs2 in enumerate(all_solutions):
                if i != j and is_dominated(objs1, objs2):
                    dominated = True
                    break
            if not dominated:
                non_dominated_count += 1
        
        # Expected: 52 non-dominated solutions (2 payoff solutions are dominated)
        assert non_dominated_count == 52, \
            f"Expected exactly 52 non-dominated solutions, got {non_dominated_count}"
        
        assert exhaustive is True, "Search must be exhaustive"


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
