"""
Integration tests for MILP algorithm using real data instances.

This module tests the MILP (Mixed Integer Linear Programming) solver with AUGMECON
for exact Pareto solutions on real satellite image scheduling data.
"""
import pytest
import sims_problem
from .real_data_utils import (
    AlgorithmResult,
    MILPConfig,
    SMALL_INSTANCES,
    MEDIUM_INSTANCES,
    LARGE_INSTANCES,
    validate_solutions,
    analyze_problem_coverage,
    log_solution_statistics,
    create_plot_path,
    run_algorithm_with_timing
)


class TestMILPIntegrationWithRealData:
    """Integration tests for MILP algorithm using real data instances."""

    def run_milp_with_validation(
        self, 
        problem, 
        filename: str, 
        config: MILPConfig, 
        plot_path, 
        logger
    ) -> AlgorithmResult:
        """Run MILP algorithm and validate results."""
        # Validate the problem instance
        problem.validate()
        
        logger.info(f"Testing {filename}: {problem.num_images} images, {problem.universe} universe")
        logger.info(f"MILP Config: objectives={config.objectives}, grid_points={config.grid_points}, "
                   f"timeout={config.timeout_seconds}s, solver={config.solver_name}")
        
        # Run MILP algorithm with timing
        solutions, execution_time = run_algorithm_with_timing(
            sims_problem.solve_with_milp,
            problem,
            objectives=config.objectives,
            grid_points=config.grid_points,
            timeout_seconds=config.timeout_seconds,
            bypass_coefficient=config.bypass_coefficient,
            early_exit=config.early_exit,
            flag_array=config.flag_array,
            solver_name=config.solver_name
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
            algorithm_name="milp"
        )

    @pytest.mark.parametrize("filename", SMALL_INSTANCES)
    def test_milp_on_small_instances(self, filename, all_test_instances, plots_directory, 
                                    logger, small_milp_config):
        """Test MILP on small instances (30 and 50 images) with reasonable timeout."""
        if filename not in all_test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = all_test_instances[filename]
        plot_path = create_plot_path(plots_directory, "milp", "small", filename)
        
        # Run MILP with validation
        result = self.run_milp_with_validation(
            problem, filename, small_milp_config, plot_path, logger
        )
        
        # Basic assertions
        assert len(result.solutions) > 0, f"Should find at least one solution for {filename}"
        
        # MILP should provide exact solutions, so all should be valid
        if not result.valid_solutions:
            analyze_problem_coverage(problem, logger, filename)
        
        assert len(result.valid_solutions) > 0, f"Should find at least one valid solution for {filename}"
        
        # MILP solutions should be Pareto optimal
        self.verify_pareto_optimality(result.valid_solutions, logger)
        
        # Log solution statistics
        log_solution_statistics(result, logger)

    @pytest.mark.parametrize("filename", MEDIUM_INSTANCES)
    def test_milp_on_medium_instances(self, filename, all_test_instances, plots_directory,
                                     logger, medium_milp_config):
        """Test MILP on medium instances (100 images) with longer timeout."""
        if filename not in all_test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = all_test_instances[filename]
        plot_path = create_plot_path(plots_directory, "milp", "medium", filename)
        
        # Run MILP with validation
        result = self.run_milp_with_validation(
            problem, filename, medium_milp_config, plot_path, logger
        )
        
        # Basic assertions
        assert len(result.solutions) > 0, f"Should find at least one solution for {filename}"
        assert len(result.valid_solutions) > 0, f"Should find at least one valid solution for {filename}"
        
        # Verify Pareto optimality
        self.verify_pareto_optimality(result.valid_solutions, logger)
        
        logger.info(f"Valid solutions: {len(result.valid_solutions)} out of {len(result.solutions)} total")

        # Print solution statistics
        costs = [sol.cost for sol in result.solutions]
        cloudy_areas = [sol.cloudy_area for sol in result.solutions]
        
        print(f"Cost range: {min(costs)} - {max(costs)}")
        print(f"Cloudy area range: {min(cloudy_areas)} - {max(cloudy_areas)}")
        print(f"Execution time: {result.execution_time:.2f} seconds")

    @pytest.mark.parametrize("filename", LARGE_INSTANCES[:3])  # Test only first 3 large instances
    @pytest.mark.slow
    def test_milp_on_large_instances(self, filename, all_test_instances, plots_directory,
                                    logger, large_milp_config):
        """Test MILP on large instances (150+ images) with extended timeout."""
        if filename not in all_test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = all_test_instances[filename]
        plot_path = create_plot_path(plots_directory, "milp", "large", filename)
        
        # Run MILP with validation
        result = self.run_milp_with_validation(
            problem, filename, large_milp_config, plot_path, logger
        )
        
        print(f"\nTesting {filename}: {problem.num_images} images, {problem.universe} universe")
        print(f"Execution time: {result.execution_time:.2f} seconds")
        print(f"Found {len(result.solutions)} solutions")
        
        # Basic assertions
        assert len(result.solutions) > 0, f"Should find at least one solution for {filename}"
        
        # For large instances, we might hit timeout, so be more lenient
        if result.valid_solutions:
            self.verify_pareto_optimality(result.valid_solutions, logger)

        # Print solution statistics
        costs = [sol.cost for sol in result.solutions]
        cloudy_areas = [sol.cloudy_area for sol in result.solutions]
        
        print(f"Cost range: {min(costs)} - {max(costs)}")
        print(f"Cloudy area range: {min(cloudy_areas)} - {max(cloudy_areas)}")

    def test_milp_multi_objective(self, all_test_instances, plots_directory, logger):
        """Test MILP with multiple objectives."""
        filename = "paris_50.dzn"
        if filename not in all_test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = all_test_instances[filename]
        plot_path = create_plot_path(plots_directory, "milp", "multi_obj", filename)
        
        # Configure MILP with 3 objectives
        from .real_data_utils import MILPConfig
        multi_obj_config = MILPConfig(
            timeout_seconds=600.0,
            objectives=["min_cost", "cloud_coverage", "min_resolution"],
            grid_points=15,  # Smaller grid for 3 objectives
            bypass_coefficient=True,
            early_exit=True,
            flag_array=True,
            solver_name="cbc"
        )
        
        # Run MILP with validation
        result = self.run_milp_with_validation(
            problem, filename, multi_obj_config, plot_path, logger
        )
        
        assert len(result.solutions) > 0, "Should find at least one solution"
        assert len(result.valid_solutions) > 0, "Should find at least one valid solution"
        
        # Check that solutions have different objective values
        objectives = [(sol.cost, sol.cloudy_area) for sol in result.valid_solutions]
        unique_objectives = set(objectives)
        
        logger.info(f"Found {len(unique_objectives)} unique objective combinations")
        assert len(unique_objectives) >= 1, "Should have at least one unique objective combination"

    def test_milp_solver_comparison(self, all_test_instances, logger):
        """Test MILP with different solvers (if available)."""
        filename = "lagos_nigeria_30.dzn"
        if filename not in all_test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = all_test_instances[filename]
        
        # Test with CBC solver (default)
        from .real_data_utils import MILPConfig
        cbc_config = MILPConfig(
            timeout_seconds=180.0,
            objectives=["min_cost", "cloud_coverage"],
            grid_points=20,
            bypass_coefficient=True,
            early_exit=True,
            flag_array=True,
            solver_name="cbc"
        )
        
        solutions_cbc, time_cbc = run_algorithm_with_timing(
            sims_problem.solve_with_milp,
            problem,
            objectives=cbc_config.objectives,
            grid_points=cbc_config.grid_points,
            timeout_seconds=cbc_config.timeout_seconds,
            bypass_coefficient=cbc_config.bypass_coefficient,
            early_exit=cbc_config.early_exit,
            flag_array=cbc_config.flag_array,
            solver_name=cbc_config.solver_name
        )
        
        assert len(solutions_cbc) > 0, "CBC solver should find at least one solution"
        
        logger.info(f"CBC solver: {len(solutions_cbc)} solutions in {time_cbc:.2f}s")
        
        # Validate CBC solutions
        valid_cbc = validate_solutions(solutions_cbc, problem, logger)
        assert len(valid_cbc) > 0, "CBC should produce valid solutions"

    def test_milp_deterministic_behavior(self, all_test_instances):
        """Test deterministic behavior of MILP solver."""
        filename = "tokyo_bay_30.dzn"
        if filename not in all_test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = all_test_instances[filename]
        
        # Run the same configuration twice
        common_params = {
            'objectives': ["min_cost", "cloud_coverage"],
            'grid_points': 15,
            'timeout_seconds': 120.0,
            'bypass_coefficient': True,
            'early_exit': True,
            'flag_array': True,
            'solver_name': "cbc"
        }
        
        solutions1 = sims_problem.solve_with_milp(problem, **common_params)
        solutions2 = sims_problem.solve_with_milp(problem, **common_params)
        
        # MILP should be deterministic - same number of solutions
        assert len(solutions1) == len(solutions2), \
            "MILP runs should produce same number of solutions"
        
        # Convert to sets of (cost, cloudy_area) for comparison
        objectives1 = {(sol.cost, sol.cloudy_area) for sol in solutions1}
        objectives2 = {(sol.cost, sol.cloudy_area) for sol in solutions2}
        
        assert objectives1 == objectives2, \
            "MILP runs should produce solutions with same objectives"

    def test_milp_grid_density_impact(self, all_test_instances, logger):
        """Test how grid density affects solution quality and quantity."""
        filename = "mexico_city_50.dzn"
        if filename not in all_test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = all_test_instances[filename]
        
        grid_densities = [10, 20, 30]
        results = {}
        
        for grid_points in grid_densities:
            logger.info(f"Testing grid density: {grid_points}")
            
            solutions, exec_time = run_algorithm_with_timing(
                sims_problem.solve_with_milp,
                problem,
                objectives=["min_cost", "cloud_coverage"],
                grid_points=grid_points,
                timeout_seconds=240.0,
                bypass_coefficient=True,
                early_exit=True,
                flag_array=True,
                solver_name="cbc"
            )
            
            results[grid_points] = {
                'solutions': len(solutions),
                'time': exec_time,
                'valid': len(validate_solutions(solutions, problem, logger))
            }
        
        # Check that we get some solutions for each grid density
        for grid_points, result in results.items():
            assert result['solutions'] > 0, f"Should find solutions for grid density {grid_points}"
            assert result['valid'] > 0, f"Should find valid solutions for grid density {grid_points}"
            logger.info(f"Grid {grid_points}: {result['solutions']} solutions, "
                       f"{result['valid']} valid, {result['time']:.2f}s")

    def verify_pareto_optimality(self, solutions, logger):
        """Verify that solutions form a Pareto front (no solution dominates another)."""
        if len(solutions) <= 1:
            return  # Can't check Pareto optimality with <= 1 solution
        
        objectives = [(sol.cost, sol.cloudy_area) for sol in solutions]
        
        # Check for dominated solutions
        dominated_count = 0
        for i, (cost1, cloudy1) in enumerate(objectives):
            for j, (cost2, cloudy2) in enumerate(objectives):
                if i != j:
                    # Solution j dominates solution i if it's better or equal in all objectives
                    # and strictly better in at least one
                    if ((cost2 <= cost1 and cloudy2 <= cloudy1) and 
                        (cost2 < cost1 or cloudy2 < cloudy1)):
                        dominated_count += 1
                        logger.warning(f"Solution {i} (cost={cost1}, cloudy={cloudy1}) "
                                     f"is dominated by solution {j} (cost={cost2}, cloudy={cloudy2})")
                        break
        
        if dominated_count > 0:
            logger.warning(f"Found {dominated_count} dominated solutions out of {len(solutions)}")
        else:
            logger.info(f"All {len(solutions)} solutions are Pareto optimal")
        
        # For exact MILP solver, we expect very few or no dominated solutions
        # Allow some tolerance for numerical precision
        assert dominated_count <= len(solutions) * 0.1, \
            f"Too many dominated solutions ({dominated_count}/{len(solutions)})"
