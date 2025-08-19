"""
Comprehensive pytest tests for 4-objective SIMS solver implementation.
Tests cover correctness, integration, and validation of the extended solver.
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path
import json

# Add the sims-solvers directory to the Python path
sims_solvers_path = Path(__file__).parent.parent
sys.path.insert(0, str(sims_solvers_path))

from sims_solvers.Config import Config
from sims_solvers.main import build_instance, build_model, build_solver
from sims_solvers.solve import solve_milp
from sims_solvers.FrontGenerators.CoverageGridPoint import CoverageGridPoint
from sims_solvers.ParetoFront import ParetoFront
from conftest import TEST_INSTANCES_30, TEST_INSTANCES_50, TEST_INSTANCES_DIR, MZN_MODEL_PATH


class TestFourObjectiveCorrectness:
    """Test correctness of 4-objective implementation."""
    
    def test_coverage_grid_point_accepts_4_objectives(self, test_data_dir, mzn_model_path):
        """Test that CoverageGridPoint no longer rejects 4-objective problems."""
        config = self._create_test_config(
            os.path.join(test_data_dir, "lagos_nigeria_30.dzn"),
            mzn_model_path,
            timeout=30
        )
        
        instance = build_instance(config)
        model = build_model(instance, config)
        
        # This should not raise an exception anymore
        solver, pareto_front = build_solver(model, instance, config, {})
        
        # Verify it's a CoverageGridPoint instance
        assert isinstance(solver.front_generator, CoverageGridPoint), \
            f"Expected CoverageGridPoint, got {type(solver.front_generator)}"
        
        # Verify it accepts 4 objectives
        assert len(model.objectives) == 4, "Model should have 4 objectives"
    
    def test_all_4_objectives_are_minimization(self, test_data_dir, mzn_model_path):
        """Test that all 4 objectives are correctly set to minimization."""
        config = self._create_test_config(
            os.path.join(test_data_dir, "lagos_nigeria_30.dzn"),
            mzn_model_path
        )
        
        instance = build_instance(config)
        model = build_model(instance, config)
        
        # All objectives should be minimization for SIMS problem
        assert model.is_a_minimization_model(), "SIMS should be a minimization problem"
        
        # Test nadir bounds for all 4 objectives
        nadir_bounds = model.get_nadir_bound_estimation()
        assert len(nadir_bounds) == 4, f"Expected 4 nadir bounds, got {len(nadir_bounds)}"
        
        # All nadir bounds should be positive
        for i, bound in enumerate(nadir_bounds):
            assert bound > 0, f"Nadir bound {i+1} should be positive, got {bound}"
    
    def test_hypervolume_reference_points_4d(self, test_data_dir, mzn_model_path):
        """Test that hypervolume reference points are set for 4D space."""
        config = self._create_test_config(
            os.path.join(test_data_dir, "lagos_nigeria_30.dzn"),
            mzn_model_path
        )
        
        instance = build_instance(config)
        model = build_model(instance, config)
        
        ref_points = model.get_ref_points_for_hypervolume()
        assert len(ref_points) == 4, f"Expected 4 reference points, got {len(ref_points)}"
        
        # All reference points should be positive
        for i, ref_point in enumerate(ref_points):
            assert ref_point > 0, f"Reference point {i+1} should be positive, got {ref_point}"
    
    def test_4_objective_solution_generation(self, test_data_dir, mzn_model_path):
        """Test that solutions are generated with all 4 objective values."""
        config = self._create_test_config(
            os.path.join(test_data_dir, "lagos_nigeria_30.dzn"),
            mzn_model_path,
            timeout=45
        )
        
        instance = build_instance(config)
        model = build_model(instance, config)
        solver, pareto_front = build_solver(model, instance, config, {})
        
        solution_count = 0
        solutions = []
        
        try:
            for solution in solver.solve():
                if solution_count >= 5:  # Test first 5 solutions
                    break
                
                # Verify solution structure
                assert "objs" in solution, "Solution should contain 'objs' key"
                assert "solution_values" in solution, "Solution should contain 'solution_values' key"
                
                objs = solution["objs"]
                assert len(objs) == 4, f"Solution should have 4 objectives, got {len(objs)}"
                
                # Verify all objective values are valid
                for i, obj_val in enumerate(objs):
                    assert isinstance(obj_val, (int, float)), \
                        f"Objective {i+1} should be numeric, got {type(obj_val)}"
                    assert obj_val >= 0, f"Objective {i+1} should be non-negative, got {obj_val}"
                
                solutions.append(objs)
                solution_count += 1
                
            assert solution_count > 0, "Should find at least one solution"
            
            # Test that solutions show variation in objectives
            if len(solutions) > 1:
                # Check that not all solutions are identical
                first_solution = solutions[0]
                has_variation = any(
                    any(sol[i] != first_solution[i] for i in range(4))
                    for sol in solutions[1:]
                )
                assert has_variation, "Solutions should show variation in objective values"
                
        except Exception as e:
            pytest.fail(f"Failed to generate 4-objective solutions: {e}")
    
    def test_pareto_dominance_4d(self, test_data_dir, mzn_model_path):
        """Test that Pareto dominance works correctly in 4D space."""
        # Create test solutions
        solution1 = {"objs": [100, 200, 300, 400], "minimize_objs": [True, True, True, True]}
        solution2 = {"objs": [90, 190, 290, 390], "minimize_objs": [True, True, True, True]}  # Dominates solution1
        solution3 = {"objs": [110, 190, 290, 390], "minimize_objs": [True, True, True, True]}  # Non-dominated
        
        pareto_front = ParetoFront()
        
        # Test dominance relationships
        assert pareto_front.dominates(solution2, solution1), \
            "Solution2 should dominate Solution1 (better in all objectives)"
        
        assert not pareto_front.dominates(solution1, solution2), \
            "Solution1 should not dominate Solution2"
        
        assert not pareto_front.dominates(solution1, solution3), \
            "Solution1 should not dominate Solution3 (trade-offs exist)"
        
        assert not pareto_front.dominates(solution3, solution1), \
            "Solution3 should not dominate Solution1 (trade-offs exist)"
    
    def _create_test_config(self, dzn_file, mzn_file, timeout=60):
        """Helper method to create test configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_file = os.path.join(temp_dir, "test_summary.csv")
            
            # Create minimal config for testing
            config = Config()
            config.input_dzn = dzn_file
            config.input_mzn = mzn_file
            config.solver_name = "gurobi"
            config.front_strategy = "gpba-a"  # CoverageGridPoint
            config.solver_search_strategy = "free_search"
            config.solver_timeout_sec = timeout
            config.summary_filename = summary_file
            config.minizinc_data = True
            config.problem_name = "sims"
            
            return config


class TestFourObjectiveIntegration:
    """Test integration of 4-objective components."""
    
    @pytest.mark.parametrize("instance_file", TEST_INSTANCES_30)
    def test_complete_solve_pipeline_30(self, instance_file, test_data_dir, mzn_model_path):
        """Test complete solve pipeline with 30-size instances."""
        dzn_file = os.path.join(test_data_dir, instance_file)
        
        if not os.path.exists(dzn_file):
            pytest.skip(f"Test instance {dzn_file} not found")
        
        config = self._create_test_config(dzn_file, mzn_model_path, timeout=60)
        
        # Test complete pipeline
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                config.summary_filename = os.path.join(temp_dir, "test_results.csv")
                solve_milp(config)
                
                # Verify results file was created
                assert os.path.exists(config.summary_filename), \
                    f"Results file should be created at {config.summary_filename}"
                    
        except Exception as e:
            pytest.fail(f"Complete pipeline failed for {instance_file}: {e}")
    
    def test_solution_validation_4_objectives(self, test_data_dir, mzn_model_path):
        """Test that solutions can be validated with 4 objectives."""
        config = self._create_test_config(
            os.path.join(test_data_dir, "lagos_nigeria_30.dzn"),
            mzn_model_path,
            timeout=45
        )
        
        instance = build_instance(config)
        model = build_model(instance, config)
        solver, pareto_front = build_solver(model, instance, config, {})
        
        # Get a solution and validate it
        for solution in solver.solve():
            objs = solution["objs"]
            selected_images = solution["solution_values"]
            
            # This should not raise an exception with 4 objectives
            try:
                model.assert_solution(objs, selected_images)
                break  # Test passed, exit loop
            except Exception as e:
                pytest.fail(f"Solution validation failed with 4 objectives: {e}")
        else:
            pytest.fail("No solutions generated for validation test")
    
    def test_objective_value_consistency(self, test_data_dir, mzn_model_path):
        """Test that objective values are consistent between calculation methods."""
        config = self._create_test_config(
            os.path.join(test_data_dir, "lagos_nigeria_30.dzn"),
            mzn_model_path,
            timeout=30
        )
        
        instance = build_instance(config)
        model = build_model(instance, config)
        solver, pareto_front = build_solver(model, instance, config, {})
        
        for solution in solver.solve():
            objs = solution["objs"]
            selected_images = solution["solution_values"]
            
            # Test objective 1: cost
            calculated_cost = model.calculate_cost(selected_images)
            assert abs(calculated_cost - objs[0]) < 1e-6, \
                f"Cost calculation inconsistent: solver={objs[0]}, manual={calculated_cost}"
            
            # Test objective 2: cloud coverage
            calculated_cloud = model.calculate_cloud_uncovered(selected_images)
            assert abs(calculated_cloud - objs[1]) < 1e-6, \
                f"Cloud calculation inconsistent: solver={objs[1]}, manual={calculated_cloud}"
            
            # Test objective 3: resolution
            calculated_resolution = model.calculate_resolution(selected_images)
            assert abs(calculated_resolution - objs[2]) < 1e-6, \
                f"Resolution calculation inconsistent: solver={objs[2]}, manual={calculated_resolution}"
            
            break  # Test first solution only
    
    def _create_test_config(self, dzn_file, mzn_file, timeout=60):
        """Helper method to create test configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_file = os.path.join(temp_dir, "test_summary.csv")
            
            config = Config()
            config.input_dzn = dzn_file
            config.input_mzn = mzn_file
            config.solver_name = "gurobi"
            config.front_strategy = "gpba-a"
            config.solver_search_strategy = "free_search"
            config.solver_timeout_sec = timeout
            config.summary_filename = summary_file
            config.minizinc_data = True
            config.problem_name = "sims"
            
            return config


class TestFourObjectivePerformance:
    """Test performance characteristics of 4-objective implementation."""
    
    def test_reasonable_solution_time_30(self, test_data_dir, mzn_model_path):
        """Test that 30-size instances solve in reasonable time."""
        config = self._create_test_config(
            os.path.join(test_data_dir, "lagos_nigeria_30.dzn"),
            mzn_model_path,
            timeout=120  # 2 minutes should be reasonable for 30-size
        )
        
        import time
        start_time = time.time()
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                config.summary_filename = os.path.join(temp_dir, "performance_test.csv")
                solve_milp(config)
                
            elapsed_time = time.time() - start_time
            
            # Should complete within timeout
            assert elapsed_time < 120, f"Solving took too long: {elapsed_time:.2f} seconds"
            
            # Should find solutions reasonably quickly
            assert elapsed_time < 90, f"Expected faster solving: {elapsed_time:.2f} seconds"
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            pytest.fail(f"Performance test failed after {elapsed_time:.2f}s: {e}")
    
    def test_memory_usage_reasonable(self, test_data_dir, mzn_model_path):
        """Test that memory usage remains reasonable for 4-objective problems."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        config = self._create_test_config(
            os.path.join(test_data_dir, "lagos_nigeria_30.dzn"),
            mzn_model_path,
            timeout=60
        )
        
        instance = build_instance(config)
        model = build_model(instance, config)
        solver, pareto_front = build_solver(model, instance, config, {})
        
        # Process several solutions
        solution_count = 0
        for solution in solver.solve():
            if solution_count >= 10:
                break
            solution_count += 1
            
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for this test)
        assert memory_increase < 500, \
            f"Memory usage increased too much: {memory_increase:.2f} MB"
    
    def _create_test_config(self, dzn_file, mzn_file, timeout=60):
        """Helper method to create test configuration."""
        config = Config()
        config.input_dzn = dzn_file
        config.input_mzn = mzn_file
        config.solver_name = "gurobi"
        config.front_strategy = "gpba-a"
        config.solver_search_strategy = "free_search"
        config.solver_timeout_sec = timeout
        config.minizinc_data = True
        config.problem_name = "sims"
        
        return config


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
