import pytest
import sims_problem
from .test_data_loader import get_all_test_instances, load_test_instance_as_problem
import time
import logging
from pathlib import Path

# Configure logging to capture Rust logs
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class TestMultiobjectivePLSIntegrationWithRealData:
    """Integration tests for multiobjective PLS algorithm (4D) using real data instances."""

    @pytest.fixture(scope="class")
    def test_instances(self):
        """Load all test instances once for the class."""
        instances = {}
        for filename in get_all_test_instances():
            try:
                # Use the new Rust-based loader for better performance
                problem = load_test_instance_as_problem(filename)
                instances[filename] = problem
            except Exception as e:
                pytest.fail(f"Failed to load test instance {filename}: {e}")
        return instances

    @pytest.mark.parametrize("filename", [
        "lagos_nigeria_30.dzn",
        "lagos_nigeria_50.dzn",
        "mexico_city_30.dzn",
        "mexico_city_50.dzn",
        "paris_30.dzn",
        "paris_50.dzn",
        "rio_de_janeiro_30.dzn",
        "rio_de_janeiro_50.dzn",
        "tokyo_bay_30.dzn",
        "tokyo_bay_50.dzn",
    ])
    def test_multiobjective_pls_on_small_instances(self, filename, test_instances):
        """Test multiobjective PLS on small instances (30 and 50 images) with reasonable timeout."""
        if filename not in test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = test_instances[filename]
        logger = logging.getLogger(__name__)
        
        # Validate the problem instance
        problem.validate()
        
        logger.info(f"Testing multiobjective PLS on {filename}: {problem.num_images} images, {problem.universe} universe")
        
        start_time = time.time()
        
        # Run multiobjective PLS with moderate parameters for small instances
        logger.info("Starting multiobjective PLS algorithm...")
        solutions = sims_problem.solve_with_pls(
            problem,
            objectives=["min_cost", "cloud_coverage", "min_resolution", "max_incidence_angle"],
            timeout_seconds=120.0,  # 2 minute timeout for debugging
            max_iterations=10000,   # More iterations
            is_deterministic=True,  # Deterministic for debugging
            initial_population_size=100,  # Larger population
            neighborhood_size_min=1,
            neighborhood_size_max=8   # Larger neighborhood for better coverage
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.info(f"Execution time: {execution_time:.2f} seconds")
        logger.info(f"Found {len(solutions)} solutions")
        
        # Basic assertions
        assert len(solutions) > 0, f"Should find at least one solution for {filename}"
        assert len(solutions) <= 500, f"Should not return excessive number of solutions for {filename}"
        
        # Analyze and validate solutions
        valid_solutions = []
        for i, solution in enumerate(solutions):
            # Check 4D objective properties
            assert solution.cost >= 0, f"Solution {i} should have non-negative cost"
            assert solution.cloudy_area >= 0, f"Solution {i} should have non-negative cloudy area"
            assert solution.min_resolutions_sum is None or solution.min_resolutions_sum >= 0, f"Solution {i} should have non-negative min_resolutions_sum"
            assert solution.max_incidence_angle is None or solution.max_incidence_angle >= 0, f"Solution {i} should have non-negative max_incidence_angle"
            assert solution.timestamp.total_seconds() >= 0, f"Solution {i} should have non-negative timestamp"
            
            # Check that selected images are valid
            selected_images = solution.get_selected_images_list()
            assert all(0 <= img < problem.num_images for img in selected_images), \
                f"Solution {i} should have valid image indices"
            
            # Detailed validation
            is_valid = solution.validate(problem)
            if is_valid:
                valid_solutions.append(solution)
                logger.debug(f"Solution {i} is valid: cost={solution.cost}, cloudy_area={solution.cloudy_area}, "
                           f"min_res_sum={solution.min_resolutions_sum}, max_angle={solution.max_incidence_angle}, "
                           f"images={len(selected_images)}")
            else:
                logger.warning(f"Solution {i} is invalid: cost={solution.cost}, cloudy_area={solution.cloudy_area}, "
                             f"min_res_sum={solution.min_resolutions_sum}, max_angle={solution.max_incidence_angle}, "
                             f"images={selected_images}")
                
                # Debug coverage for invalid solutions
                coverage = set()
                for img_idx in selected_images:
                    coverage.update(problem.images[img_idx])
                uncovered = set(range(problem.universe)) - coverage
                logger.warning(f"Solution {i} missing coverage for {len(uncovered)} elements: {sorted(list(uncovered))[:20]}...")
        
        # At least one solution should be valid
        if not valid_solutions:
            logger.error(f"No valid solutions found for {filename}")
            # Print detailed problem analysis
            logger.error("Problem coverage analysis:")
            all_elements = set()
            for i, img in enumerate(problem.images):
                all_elements.update(img)
                logger.debug(f"Image {i}: covers {len(img)} elements")
            logger.error(f"All images together cover {len(all_elements)} out of {problem.universe} universe elements")
            missing_from_all = set(range(problem.universe)) - all_elements
            if missing_from_all:
                logger.error(f"Elements never covered by any image: {sorted(list(missing_from_all))}")
        
        assert len(valid_solutions) > 0, f"Should find at least one valid solution for {filename}"

        # Print some solution statistics for all 4 objectives
        costs = [sol.cost for sol in valid_solutions]
        cloudy_areas = [sol.cloudy_area for sol in valid_solutions]
        min_res_sums = [sol.min_resolutions_sum for sol in valid_solutions]
        max_angles = [sol.max_incidence_angle for sol in valid_solutions]
        
        logger.info(f"Valid solutions: {len(valid_solutions)} out of {len(solutions)} total")
        if valid_solutions:
            logger.info(f"Cost range: {min(costs)} - {max(costs)}")
            logger.info(f"Cloudy area range: {min(cloudy_areas)} - {max(cloudy_areas)}")
            logger.info(f"Min resolution sum range: {min(min_res_sums)} - {max(min_res_sums)}")
            logger.info(f"Max incidence angle range: {min(max_angles)} - {max(max_angles)}")

    @pytest.mark.parametrize("filename", [
        "lagos_nigeria_100.dzn",
        "mexico_city_100.dzn",
        "paris_100.dzn",
        "rio_de_janeiro_100.dzn",
        "tokyo_bay_100.dzn",
    ])
    def test_multiobjective_pls_on_medium_instances(self, filename, test_instances):
        """Test multiobjective PLS on medium instances (100 images) with longer timeout."""
        if filename not in test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = test_instances[filename]
        logger = logging.getLogger(__name__)
        
        # Validate the problem instance
        problem.validate()
        
        logger.info(f"Testing multiobjective PLS on {filename}: {problem.num_images} images, {problem.universe} universe")
        
        start_time = time.time()
        
        # Run multiobjective PLS with more generous parameters for medium instances
        logger.info("Starting multiobjective PLS algorithm for medium instance...")
        solutions = sims_problem.solve_with_pls(
            problem,
            objectives=["min_cost", "cloud_coverage", "min_resolution", "max_incidence_angle"],
            timeout_seconds=300.0,  # 5 minutes timeout
            max_iterations=15000,   # More iterations
            is_deterministic=True,  # Deterministic for debugging
            initial_population_size=150,  # Larger population
            neighborhood_size_min=1,
            neighborhood_size_max=10  # Larger neighborhood
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.info(f"Execution time: {execution_time:.2f} seconds")
        logger.info(f"Found {len(solutions)} solutions")
        
        # Basic assertions
        assert len(solutions) > 0, f"Should find at least one solution for {filename}"
        
        # Validate solutions and analyze coverage
        valid_solutions = []
        for i, solution in enumerate(solutions):
            # Check 4D objective properties
            assert solution.cost >= 0, f"Solution {i} should have non-negative cost"
            assert solution.cloudy_area >= 0, f"Solution {i} should have non-negative cloudy area"
            assert solution.min_resolutions_sum is None or solution.min_resolutions_sum >= 0, f"Solution {i} should have non-negative min_resolutions_sum"
            assert solution.max_incidence_angle is None or solution.max_incidence_angle >= 0, f"Solution {i} should have non-negative max_incidence_angle"
            
            # Validate solution
            is_valid = solution.validate(problem)
            if is_valid:
                valid_solutions.append(solution)
            else:
                logger.warning(f"Medium instance solution {i} is invalid")
        
        # Should have at least some valid solutions
        assert len(valid_solutions) > 0, f"Should find at least one valid solution for {filename}"
        
        logger.info(f"Valid solutions: {len(valid_solutions)} out of {len(solutions)} total")

        # Print solution statistics for all 4 objectives
        costs = [sol.cost for sol in solutions]
        cloudy_areas = [sol.cloudy_area for sol in solutions]
        min_res_sums = [sol.min_resolutions_sum for sol in solutions if sol.min_resolutions_sum is not None]
        max_angles = [sol.max_incidence_angle for sol in solutions if sol.max_incidence_angle is not None]
        
        print(f"Cost range: {min(costs)} - {max(costs)}")
        print(f"Cloudy area range: {min(cloudy_areas)} - {max(cloudy_areas)}")
        if min_res_sums:
            print(f"Min resolution sum range: {min(min_res_sums)} - {max(min_res_sums)}")
        if max_angles:
            print(f"Max incidence angle range: {min(max_angles)} - {max(max_angles)}")

    @pytest.mark.parametrize("filename", [
        "lagos_nigeria_145.dzn",
        "mexico_city_150.dzn",
        "mexico_city_200.dzn",
        "paris_150.dzn",
        "paris_200.dzn",
        "rio_de_janeiro_150.dzn",
        "rio_de_janeiro_200.dzn",
        "tokyo_bay_150.dzn",
        "tokyo_bay_200.dzn",
    ])
    @pytest.mark.slow
    def test_multiobjective_pls_on_large_instances(self, filename, test_instances):
        """Test multiobjective PLS on large instances (150+ images) with extended timeout."""
        if filename not in test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = test_instances[filename]
        
        # Validate the problem instance
        problem.validate()
        
        print(f"\nTesting multiobjective PLS on {filename}: {problem.num_images} images, {problem.universe} universe")
        
        start_time = time.time()
        
        # Run multiobjective PLS with extended parameters for large instances
        solutions = sims_problem.solve_with_pls(
            problem,
            objectives=["min_cost", "cloud_coverage", "min_resolution", "max_incidence_angle"],
            timeout_seconds=300.0,  # 5 minutes timeout
            max_iterations=20000,
            is_deterministic=False,
            initial_population_size=150,
            neighborhood_size_min=1,
            neighborhood_size_max=8
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Found {len(solutions)} solutions")
        
        # Basic assertions
        assert len(solutions) > 0, f"Should find at least one solution for {filename}"
        
        # For large instances, only validate first few solutions
        solutions_to_check = solutions[:min(5, len(solutions))]
        
        for i, solution in enumerate(solutions_to_check):
            # Check 4D objective properties
            assert solution.cost >= 0, f"Solution {i} should have non-negative cost"
            assert solution.cloudy_area >= 0, f"Solution {i} should have non-negative cloudy area"
            assert solution.min_resolutions_sum is None or solution.min_resolutions_sum >= 0, f"Solution {i} should have non-negative min_resolutions_sum"
            assert solution.max_incidence_angle is None or solution.max_incidence_angle >= 0, f"Solution {i} should have non-negative max_incidence_angle"
            
            # Use the built-in validate method to check coverage and constraints
            assert solution.validate(problem), \
                f"Solution {i} should be valid (coverage and constraints)"

        # Print solution statistics for all 4 objectives
        costs = [sol.cost for sol in solutions]
        cloudy_areas = [sol.cloudy_area for sol in solutions]
        min_res_sums = [sol.min_resolutions_sum for sol in solutions if sol.min_resolutions_sum is not None]
        max_angles = [sol.max_incidence_angle for sol in solutions if sol.max_incidence_angle is not None]
        
        print(f"Cost range: {min(costs)} - {max(costs)}")
        print(f"Cloudy area range: {min(cloudy_areas)} - {max(cloudy_areas)}")
        if min_res_sums:
            print(f"Min resolution sum range: {min(min_res_sums)} - {max(min_res_sums)}")
        if max_angles:
            print(f"Max incidence angle range: {min(max_angles)} - {max(max_angles)}")

    def test_multiobjective_deterministic_behavior_real_data(self, test_instances):
        """Test deterministic behavior on a real data instance with multiobjective PLS."""
        # Use a small instance for deterministic testing
        filename = "lagos_nigeria_30.dzn"
        if filename not in test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = test_instances[filename]
        
        # Run the same deterministic configuration twice
        common_params = {
            'timeout_seconds': 30.0,
            'max_iterations': 1000,
            'is_deterministic': True,
            'initial_population_size': 20,
            'neighborhood_size_min': 1,
            'neighborhood_size_max': 3
        }
        
        solutions1 = sims_problem.solve_with_pls(
            problem, 
            objectives=["min_cost", "cloud_coverage", "min_resolution", "max_incidence_angle"],
            **common_params
        )
        solutions2 = sims_problem.solve_with_pls(
            problem, 
            objectives=["min_cost", "cloud_coverage", "min_resolution", "max_incidence_angle"],
            **common_params
        )
        
        # Should produce same number of solutions
        assert len(solutions1) == len(solutions2), \
            "Deterministic runs should produce same number of solutions"
        
        # Convert to sets of 4D objectives for comparison
        objectives1 = {(sol.cost, sol.cloudy_area, sol.min_resolutions_sum, sol.max_incidence_angle) for sol in solutions1}
        objectives2 = {(sol.cost, sol.cloudy_area, sol.min_resolutions_sum, sol.max_incidence_angle) for sol in solutions2}
        
        assert objectives1 == objectives2, \
            "Deterministic runs should produce solutions with same objectives"

    def test_multiobjective_solution_diversity(self, test_instances):
        """Test that multiobjective PLS produces diverse solutions across all 4 objectives."""
        # Use a medium-sized instance
        filename = "paris_50.dzn"
        if filename not in test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = test_instances[filename]
        
        solutions = sims_problem.solve_with_pls(
            problem,
            objectives=["min_cost", "cloud_coverage", "min_resolution", "max_incidence_angle"],
            timeout_seconds=60.0,
            max_iterations=5000,
            is_deterministic=False,
            initial_population_size=50,
            neighborhood_size_min=1,
            neighborhood_size_max=4
        )
        
        assert len(solutions) > 1, "Should find multiple solutions"
        
        # Check that we have diverse 4D objectives
        objectives = [(sol.cost, sol.cloudy_area, sol.min_resolutions_sum, sol.max_incidence_angle) for sol in solutions]
        unique_objectives = set(objectives)
        
        # Should have multiple different objective combinations
        assert len(unique_objectives) > 1, "Should produce solutions with different 4D objectives"
        
        # Check diversity across all 4 objectives
        costs = [sol.cost for sol in solutions]
        cloudy_areas = [sol.cloudy_area for sol in solutions]
        min_res_sums = [sol.min_resolutions_sum for sol in solutions]
        max_angles = [sol.max_incidence_angle for sol in solutions]
        
        cost_range = max(costs) - min(costs)
        cloudy_range = max(cloudy_areas) - min(cloudy_areas)
        
        # Filter out None values for min_res_sums and max_angles
        valid_min_res_sums = [x for x in min_res_sums if x is not None]
        valid_max_angles = [x for x in max_angles if x is not None]
        
        resolution_range = max(valid_min_res_sums) - min(valid_min_res_sums) if valid_min_res_sums else 0
        angle_range = max(valid_max_angles) - min(valid_max_angles) if valid_max_angles else 0
        
        assert cost_range > 0, "Should have diverse costs"
        # Note: Other objectives might have 0 range on small instances, which is acceptable
        
        print(f"Found {len(unique_objectives)} unique 4D objective combinations")
        print(f"Cost range: {min(costs)} - {max(costs)} (range: {cost_range})")
        print(f"Cloudy area range: {min(cloudy_areas)} - {max(cloudy_areas)} (range: {cloudy_range})")
        min_res_display = f"{min(valid_min_res_sums)} - {max(valid_min_res_sums)}" if valid_min_res_sums else "N/A"
        max_angle_display = f"{min(valid_max_angles)} - {max(valid_max_angles)}" if valid_max_angles else "N/A"
        print(f"Min resolution sum range: {min_res_display} (range: {resolution_range})")
        print(f"Max incidence angle range: {max_angle_display} (range: {angle_range})")

    def test_multiobjective_pareto_optimality_real_data(self, test_instances):
        """Test that multiobjective PLS produces valid Pareto optimal solutions on real data."""
        # Use a small instance for Pareto testing
        filename = "mexico_city_30.dzn"
        if filename not in test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = test_instances[filename]
        
        solutions = sims_problem.solve_with_pls(
            problem,
            objectives=["min_cost", "cloud_coverage", "min_resolution", "max_incidence_angle"],
            timeout_seconds=45.0,
            max_iterations=3000,
            is_deterministic=True,
            initial_population_size=30,
            neighborhood_size_min=1,
            neighborhood_size_max=4
        )
        
        assert len(solutions) > 0, "Should find at least one solution"
        
        # Check Pareto optimality: no solution should dominate another
        objectives = [(sol.cost, sol.cloudy_area, sol.min_resolutions_sum, sol.max_incidence_angle) for sol in solutions]
        
        for i, obj1 in enumerate(objectives):
            for j, obj2 in enumerate(objectives):
                if i != j:
                    # Check if obj1 dominates obj2 (all objectives are minimization)
                    # Handle None values properly
                    dominates_checks = []
                    strictly_better_checks = []
                    
                    for k in range(4):
                        if obj1[k] is None and obj2[k] is None:
                            dominates_checks.append(True)  # Equal (None == None)
                            strictly_better_checks.append(False)  # Not strictly better
                        elif obj1[k] is None:
                            dominates_checks.append(False)  # None doesn't dominate a value
                            strictly_better_checks.append(False)
                        elif obj2[k] is None:
                            dominates_checks.append(True)  # Value dominates None
                            strictly_better_checks.append(True)  # Strictly better
                        else:
                            # Both are not None, safe to compare
                            dominates_checks.append(obj1[k] <= obj2[k])  # type: ignore
                            strictly_better_checks.append(obj1[k] < obj2[k])  # type: ignore
                    
                    dominates = all(dominates_checks) and any(strictly_better_checks)
                    assert not dominates, f"Solution {i} {obj1} dominates solution {j} {obj2}, violating Pareto optimality"
        
        print(f"Verified Pareto optimality for {len(solutions)} solutions")
        
        # Print some sample solutions to show diversity
        if len(solutions) > 1:
            print("Sample solutions (cost, cloudy_area, min_res_sum, max_angle):")
            for i, obj in enumerate(objectives[:min(5, len(objectives))]):
                print(f"  Solution {i}: {obj}")

    def test_multiobjective_performance_comparison(self, test_instances):
        """Compare multiobjective vs biobjective performance on real data."""
        # Use a medium instance for performance comparison
        filename = "rio_de_janeiro_50.dzn"
        if filename not in test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = test_instances[filename]
        
        # Common parameters
        common_params = {
            'timeout_seconds': 60.0,
            'max_iterations': 3000,
            'is_deterministic': True,
            'initial_population_size': 40,
            'neighborhood_size_min': 1,
            'neighborhood_size_max': 4
        }
        
        # Run biobjective PLS
        start_time = time.time()
        biobjective_solutions = sims_problem.solve_with_pls(problem, objectives=["min_cost", "cloud_coverage"], **common_params)
        biobjective_time = time.time() - start_time
        
        # Run multiobjective PLS
        start_time = time.time()
        multiobjective_solutions = sims_problem.solve_with_pls(problem, objectives=["min_cost", "cloud_coverage", "min_resolution", "max_incidence_angle"], **common_params)
        multiobjective_time = time.time() - start_time
        
        # Both should find solutions
        assert len(biobjective_solutions) > 0, "Biobjective should find solutions"
        assert len(multiobjective_solutions) > 0, "Multiobjective should find solutions"
        
        # Multiobjective might find more diverse solutions due to additional objectives
        # Extract common objectives (cost, cloudy_area) for comparison
        bio_objectives = {(sol.cost, sol.cloudy_area) for sol in biobjective_solutions}
        multi_objectives = {(sol.cost, sol.cloudy_area) for sol in multiobjective_solutions}
        
        print(f"Biobjective found {len(biobjective_solutions)} solutions in {biobjective_time:.2f}s")
        print(f"Multiobjective found {len(multiobjective_solutions)} solutions in {multiobjective_time:.2f}s")
        print(f"Unique biobjective (cost, cloudy_area) pairs: {len(bio_objectives)}")
        print(f"Unique multiobjective (cost, cloudy_area) pairs: {len(multi_objectives)}")
        
        # Performance ratio should be reasonable (multiobjective may be slower but not excessively)
        time_ratio = multiobjective_time / biobjective_time if biobjective_time > 0 else float('inf')
        print(f"Multiobjective/biobjective time ratio: {time_ratio:.2f}")
        
        # Both approaches should be reasonably efficient
        assert biobjective_time < 120, "Biobjective should complete within reasonable time"
        assert multiobjective_time < 180, "Multiobjective should complete within reasonable time"

    @pytest.mark.parametrize("filename", [
        "lagos_nigeria_30.dzn",
        "paris_50.dzn",
        "tokyo_bay_30.dzn"
    ])
    def test_multiobjective_pls_with_2d_plot_generation(self, filename, test_instances):
        """Test multiobjective PLS with 2D plot generation as artifacts."""
        if filename not in test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = test_instances[filename]
        logger = logging.getLogger(__name__)
        
        # Create output directory for plot artifacts
        test_dir = Path(__file__).parent
        output_dir = test_dir / "plot_artifacts"
        output_dir.mkdir(exist_ok=True)
        
        instance_name = filename.replace('.dzn', '')
        plot_path = str(output_dir / f"{instance_name}_2d_pareto_front.svg")
        
        logger.info(f"Testing multiobjective PLS with 2D plot generation on {filename}")
        
        # Run PLS with plot generation
        start_time = time.time()
        solutions = sims_problem.solve_with_pls(
            problem,
            objectives=["min_cost", "cloud_coverage"],
            plots=True,
            plot_output_path=plot_path,
            timeout_seconds=60.0,
            max_iterations=5000,
            is_deterministic=True,
            initial_population_size=50,
            neighborhood_size_min=1,
            neighborhood_size_max=5
        )
        
        execution_time = time.time() - start_time
        
        logger.info(f"PLS with 2D plotting completed in {execution_time:.2f} seconds")
        logger.info(f"Found {len(solutions)} solutions")
        
        # Basic assertions
        assert len(solutions) > 0, f"Should find at least one solution for {filename}"
        
        # Verify plot file was created
        plot_file = Path(plot_path)
        assert plot_file.exists(), f"Plot file should be created at {plot_path}"
        assert plot_file.stat().st_size > 0, "Plot file should not be empty"
        
        logger.info(f"Successfully generated 2D plot artifact: {plot_path}")

    @pytest.mark.parametrize("filename", [
        "mexico_city_30.dzn",
        "rio_de_janeiro_50.dzn"
    ])
    def test_multiobjective_pls_with_4d_plot_generation(self, filename, test_instances):
        """Test multiobjective PLS with 4D plot grid generation as artifacts."""
        if filename not in test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = test_instances[filename]
        logger = logging.getLogger(__name__)
        
        # Create output directory for plot artifacts
        test_dir = Path(__file__).parent
        output_dir = test_dir / "plot_artifacts"
        output_dir.mkdir(exist_ok=True)
        
        instance_name = filename.replace('.dzn', '')
        plot_path = str(output_dir / f"{instance_name}_4d_pareto_grid.svg")
        
        logger.info(f"Testing multiobjective PLS with 4D plot grid generation on {filename}")
        
        # Run 4D PLS with plot generation
        start_time = time.time()
        solutions = sims_problem.solve_with_pls(
            problem,
            objectives=["min_cost", "cloud_coverage", "min_resolution", "max_incidence_angle"],
            plots=True,
            plot_output_path=plot_path,
            timeout_seconds=90.0,
            max_iterations=8000,
            is_deterministic=True,
            initial_population_size=80,
            neighborhood_size_min=1,
            neighborhood_size_max=6
        )
        
        execution_time = time.time() - start_time
        
        logger.info(f"4D PLS with plotting completed in {execution_time:.2f} seconds")
        logger.info(f"Found {len(solutions)} solutions")
        
        # Basic assertions for 4D objectives
        assert len(solutions) > 0, f"Should find at least one solution for {filename}"
        
        # Verify all solutions have 4D objectives
        for i, solution in enumerate(solutions[:5]):  # Check first 5
            assert solution.cost >= 0, f"Solution {i} should have non-negative cost"
            assert solution.cloudy_area >= 0, f"Solution {i} should have non-negative cloudy area"
            assert solution.min_resolutions_sum is None or solution.min_resolutions_sum >= 0, f"Solution {i} should have non-negative min_resolutions_sum"
            assert solution.max_incidence_angle is None or solution.max_incidence_angle >= 0, f"Solution {i} should have non-negative max_incidence_angle"
        
        # Verify plot file was created
        plot_file = Path(plot_path)
        assert plot_file.exists(), f"4D plot grid file should be created at {plot_path}"
        assert plot_file.stat().st_size > 0, "4D plot file should not be empty"
        
        # Verify we have diversity in 4D space
        objectives_4d = set((sol.cost, sol.cloudy_area, sol.min_resolutions_sum, sol.max_incidence_angle) 
                           for sol in solutions)
        assert len(objectives_4d) > 1, "Should have diverse 4D objectives"
        
        # Print solution statistics for all 4 objectives
        costs = [sol.cost for sol in solutions]
        cloudy_areas = [sol.cloudy_area for sol in solutions]
        min_res_sums = [sol.min_resolutions_sum for sol in solutions]
        max_angles = [sol.max_incidence_angle for sol in solutions]
        
        # Filter out None values for display
        valid_min_res = [x for x in min_res_sums if x is not None]
        valid_max_angles = [x for x in max_angles if x is not None]
        
        logger.info("4D Solution diversity:")
        logger.info(f"  Cost range: {min(costs)} - {max(costs)}")
        logger.info(f"  Cloudy area range: {min(cloudy_areas)} - {max(cloudy_areas)}")
        min_res_range = f"{min(valid_min_res)} - {max(valid_min_res)}" if valid_min_res else "N/A (all None)"
        max_angle_range = f"{min(valid_max_angles)} - {max(valid_max_angles)}" if valid_max_angles else "N/A (all None)"
        logger.info(f"  Min resolution sum range: {min_res_range}")
        logger.info(f"  Max incidence angle range: {max_angle_range}")
        logger.info(f"  Unique 4D objective combinations: {len(objectives_4d)}")
        
        logger.info(f"Successfully generated 4D plot grid artifact: {plot_path}")

    def test_plot_artifacts_directory_creation(self):
        """Test that plot artifacts directory is properly created and accessible."""
        test_dir = Path(__file__).parent
        output_dir = test_dir / "plot_artifacts"
        
        # Ensure directory exists
        output_dir.mkdir(exist_ok=True)
        
        assert output_dir.exists(), "Plot artifacts directory should exist"
        assert output_dir.is_dir(), "Plot artifacts path should be a directory"
        
        # Create a README file explaining the artifacts
        readme_path = output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write("""# PLS Multi-Objective Plot Artifacts

This directory contains plot artifacts generated by the PLS multi-objective tests.

## Files

### 2D Plots
- `*_2d_pareto_front.svg`: 2D scatter plots showing cost vs cloudy area

### 4D Plot Grids  
- `*_4d_pareto_grid.svg`: Grid of 6 subplots showing all pairwise combinations of 4 objectives:
  - Cost vs Cloudy Area
  - Cost vs Min Resolution Sum
  - Cost vs Max Incidence Angle
  - Cloudy Area vs Min Resolution Sum
  - Cloudy Area vs Max Incidence Angle
  - Min Resolution Sum vs Max Incidence Angle

## Objectives

1. **Cost**: Total monetary cost of selected satellite images
2. **Cloudy Area**: Total area covered by clouds (to be minimized)
3. **Min Resolution Sum**: Sum of minimum resolutions across all universe elements  
4. **Max Incidence Angle**: Maximum incidence angle among selected images

## Plot Elements

- **Green circles**: Non-dominated (Pareto optimal) solutions
- **Blue triangles**: Initial solutions

## Generated by

Multi-objective PLS integration tests using real satellite image selection data.
""")
        
        assert readme_path.exists(), "README file should be created"
        print(f"Created plot artifacts directory with README: {output_dir}")
