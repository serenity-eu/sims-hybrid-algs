"""
Integration tests for PLS algorithm using real data instances.

This module tests the Pareto Local Search (PLS) algorithm on real satellite 
image scheduling data instances of various sizes.
"""
import pytest
import sims_problem
from .real_data_utils import (
    AlgorithmResult,
    PLSConfig,
    SMALL_INSTANCES,
    MEDIUM_INSTANCES,
    LARGE_INSTANCES,
    validate_solutions,
    analyze_problem_coverage,
    log_solution_statistics,
    create_plot_path,
    run_algorithm_with_timing
)


class TestPLSIntegrationWithRealData:
    """Integration tests for PLS algorithm using real data instances."""

    def run_pls_with_validation(
        self, 
        problem, 
        filename: str, 
        config: PLSConfig, 
        plot_path, 
        logger
    ) -> AlgorithmResult:
        """Run PLS algorithm and validate results."""
        # Validate the problem instance
        problem.validate()
        
        logger.info(f"Testing {filename}: {problem.num_images} images, {problem.universe} universe")
        
        # Run PLS algorithm with timing
        solutions, execution_time = run_algorithm_with_timing(
            sims_problem.solve_with_pls,
            problem,
            plots=True,
            plot_output_path=str(plot_path),
            timeout_seconds=config.timeout_seconds,
            max_iterations=config.max_iterations,
            is_deterministic=config.is_deterministic,
            initial_population_size=config.initial_population_size,
            neighborhood_size_min=config.neighborhood_size_min,
            neighborhood_size_max=config.neighborhood_size_max
        )
        
        logger.info(f"Execution time: {execution_time:.2f} seconds")
        logger.info(f"Found {len(solutions)} solutions")
        
        # Validate solutions
        valid_solutions = validate_solutions(solutions, problem, logger)
        
        return AlgorithmResult(
            solutions=solutions,
            execution_time=execution_time,
            valid_solutions=valid_solutions,
            plot_path=plot_path,
            algorithm_name="pls"
        )

    @pytest.mark.parametrize("filename", SMALL_INSTANCES)
    def test_pls_on_small_instances(self, filename, all_test_instances, plots_directory, 
                                   logger, small_pls_config):
        """Test PLS on small instances (30 and 50 images) with reasonable timeout."""
        if filename not in all_test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = all_test_instances[filename]
        plot_path = create_plot_path(plots_directory, "pls", "small", filename)
        
        # Run PLS with validation
        result = self.run_pls_with_validation(
            problem, filename, small_pls_config, plot_path, logger
        )
        
        # Basic assertions
        assert len(result.solutions) > 0, f"Should find at least one solution for {filename}"
        assert len(result.solutions) <= 200, f"Should not return excessive number of solutions for {filename}"
        
        # At least one solution should be valid
        if not result.valid_solutions:
            analyze_problem_coverage(problem, logger, filename)
        
        assert len(result.valid_solutions) > 0, f"Should find at least one valid solution for {filename}"

        # Log solution statistics
        log_solution_statistics(result, logger)

    @pytest.mark.parametrize("filename", MEDIUM_INSTANCES)
    def test_pls_on_medium_instances(self, filename, all_test_instances, plots_directory,
                                    logger, medium_pls_config):
        """Test PLS on medium instances (100 images) with longer timeout."""
        if filename not in all_test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = all_test_instances[filename]
        plot_path = create_plot_path(plots_directory, "pls", "medium", filename)
        
        # Run PLS with validation
        result = self.run_pls_with_validation(
            problem, filename, medium_pls_config, plot_path, logger
        )
        
        # Basic assertions
        assert len(result.solutions) > 0, f"Should find at least one solution for {filename}"
        
        # Should have at least some valid solutions
        assert len(result.valid_solutions) > 0, f"Should find at least one valid solution for {filename}"
        
        logger.info(f"Valid solutions: {len(result.valid_solutions)} out of {len(result.solutions)} total")

        # Print solution statistics
        costs = [sol.cost for sol in result.solutions]
        cloudy_areas = [sol.cloudy_area for sol in result.solutions]
        
        print(f"Cost range: {min(costs)} - {max(costs)}")
        print(f"Cloudy area range: {min(cloudy_areas)} - {max(cloudy_areas)}")
        print(f"Plots saved to: {result.plot_path}")

    @pytest.mark.parametrize("filename", LARGE_INSTANCES)
    @pytest.mark.slow
    def test_pls_on_large_instances(self, filename, all_test_instances, plots_directory,
                                   large_pls_config):
        """Test PLS on large instances (150+ images) with extended timeout."""
        if filename not in all_test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = all_test_instances[filename]
        plot_path = create_plot_path(plots_directory, "pls", "large", filename)
        
        # Validate the problem instance
        problem.validate()
        
        print(f"\nTesting {filename}: {problem.num_images} images, {problem.universe} universe")
        
        # Run PLS with extended parameters for large instances
        solutions, execution_time = run_algorithm_with_timing(
            sims_problem.solve_with_pls,
            problem,
            plots=True,  # Enable built-in plotting
            plot_output_path=str(plot_path),  # Path for saving plots
            timeout_seconds=large_pls_config.timeout_seconds,
            max_iterations=large_pls_config.max_iterations,
            is_deterministic=large_pls_config.is_deterministic,
            initial_population_size=large_pls_config.initial_population_size,
            neighborhood_size_min=large_pls_config.neighborhood_size_min,
            neighborhood_size_max=large_pls_config.neighborhood_size_max
        )
        
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Found {len(solutions)} solutions")
        
        # Basic assertions
        assert len(solutions) > 0, f"Should find at least one solution for {filename}"
        
        # For large instances, only validate first few solutions
        solutions_to_check = solutions[:min(5, len(solutions))]
        
        for i, solution in enumerate(solutions_to_check):
            # Check basic properties
            assert solution.cost >= 0, f"Solution {i} should have non-negative cost"
            assert solution.cloudy_area >= 0, f"Solution {i} should have non-negative cloudy area"
            
            # Use the built-in validate method to check coverage and constraints
            assert solution.validate(problem), \
                f"Solution {i} should be valid (coverage and constraints)"

        # Print solution statistics
        costs = [sol.cost for sol in solutions]
        cloudy_areas = [sol.cloudy_area for sol in solutions]
        
        print(f"Cost range: {min(costs)} - {max(costs)}")
        print(f"Cloudy area range: {min(cloudy_areas)} - {max(cloudy_areas)}")
        print(f"Plots saved to: {plot_path}")

    def test_deterministic_behavior_real_data(self, all_test_instances):
        """Test deterministic behavior on a real data instance."""
        # Use a small instance for deterministic testing
        filename = "lagos_nigeria_30.dzn"
        if filename not in all_test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = all_test_instances[filename]
        
        # Run the same deterministic configuration twice
        common_params = {
            'timeout_seconds': 30.0,
            'max_iterations': 1000,
            'is_deterministic': True,
            'initial_population_size': 20,
            'neighborhood_size_min': 1,
            'neighborhood_size_max': 3
        }
        
        solutions1 = sims_problem.solve_with_pls(problem, **common_params)
        solutions2 = sims_problem.solve_with_pls(problem, **common_params)
        
        # Should produce same number of solutions
        assert len(solutions1) == len(solutions2), \
            "Deterministic runs should produce same number of solutions"
        
        # Convert to sets of (cost, cloudy_area) for comparison
        objectives1 = {(sol.cost, sol.cloudy_area) for sol in solutions1}
        objectives2 = {(sol.cost, sol.cloudy_area) for sol in solutions2}
        
        assert objectives1 == objectives2, \
            "Deterministic runs should produce solutions with same objectives"

    def test_solution_diversity(self, all_test_instances, plots_directory):
        """Test that PLS produces diverse solutions."""
        # Use a medium-sized instance
        filename = "paris_50.dzn"
        if filename not in all_test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = all_test_instances[filename]
        plot_path = create_plot_path(plots_directory, "pls", "diversity", filename)
        
        solutions = sims_problem.solve_with_pls(
            problem,
            plots=True,  # Enable built-in plotting
            plot_output_path=str(plot_path),  # Path for saving plots
            timeout_seconds=60.0,
            max_iterations=5000,
            is_deterministic=False,
            initial_population_size=50,
            neighborhood_size_min=1,
            neighborhood_size_max=4
        )
        
        assert len(solutions) > 1, "Should find multiple solutions"
        
        # Check that we have diverse objectives
        objectives = [(sol.cost, sol.cloudy_area) for sol in solutions]
        unique_objectives = set(objectives)
        
        # Should have multiple different objective combinations
        assert len(unique_objectives) > 1, "Should produce solutions with different objectives"
        
        # Check cost diversity
        costs = [sol.cost for sol in solutions]
        cost_range = max(costs) - min(costs)
        assert cost_range > 0, "Should have diverse costs"
        
        print(f"Found {len(unique_objectives)} unique objective combinations")
        print(f"Cost range: {min(costs)} - {max(costs)} (range: {cost_range})")
        print(f"Plots saved to: {plot_path}")
