import pytest
import sims_problem
from .test_data_loader import get_all_test_instances, load_test_instance_as_problem
import time
import logging

# Configure logging to capture Rust logs
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class TestPLSIntegrationWithRealData:
    """Integration tests for PLS algorithm using real data instances."""

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
    def test_pls_on_small_instances(self, filename, test_instances):
        """Test PLS on small instances (30 and 50 images) with reasonable timeout."""
        if filename not in test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = test_instances[filename]
        logger = logging.getLogger(__name__)
        
        # Validate the problem instance
        problem.validate()
        
        logger.info(f"Testing {filename}: {problem.num_images} images, {problem.universe} universe")
        
        start_time = time.time()
        
        # Run PLS with moderate parameters for small instances
        logger.info("Starting PLS algorithm...")
        solutions = sims_problem.solve_with_pls_advanced(
            problem,
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
        assert len(solutions) <= 200, f"Should not return excessive number of solutions for {filename}"
        
        # Analyze and validate solutions
        valid_solutions = []
        for i, solution in enumerate(solutions):
            # Check basic properties
            assert solution.cost >= 0, f"Solution {i} should have non-negative cost"
            assert solution.cloudy_area >= 0, f"Solution {i} should have non-negative cloudy area"
            assert solution.timestamp_us >= 0, f"Solution {i} should have non-negative timestamp"
            
            # Check that selected images are valid
            selected_images = solution.get_selected_images_list()
            assert all(0 <= img < problem.num_images for img in selected_images), \
                f"Solution {i} should have valid image indices"
            
            # Detailed validation
            is_valid = solution.validate(problem)
            if is_valid:
                valid_solutions.append(solution)
                logger.debug(f"Solution {i} is valid: cost={solution.cost}, cloudy_area={solution.cloudy_area}, images={len(selected_images)}")
            else:
                logger.warning(f"Solution {i} is invalid: cost={solution.cost}, cloudy_area={solution.cloudy_area}, images={selected_images}")
                
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

        # Print some solution statistics
        costs = [sol.cost for sol in valid_solutions]
        cloudy_areas = [sol.cloudy_area for sol in valid_solutions]
        
        logger.info(f"Valid solutions: {len(valid_solutions)} out of {len(solutions)} total")
        if valid_solutions:
            logger.info(f"Cost range: {min(costs)} - {max(costs)}")
            logger.info(f"Cloudy area range: {min(cloudy_areas)} - {max(cloudy_areas)}")

    @pytest.mark.parametrize("filename", [
        "lagos_nigeria_100.dzn",
        "mexico_city_100.dzn",
        "paris_100.dzn",
        "rio_de_janeiro_100.dzn",
        "tokyo_bay_100.dzn",
    ])
    def test_pls_on_medium_instances(self, filename, test_instances):
        """Test PLS on medium instances (100 images) with longer timeout."""
        if filename not in test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = test_instances[filename]
        logger = logging.getLogger(__name__)
        
        # Validate the problem instance
        problem.validate()
        
        logger.info(f"Testing {filename}: {problem.num_images} images, {problem.universe} universe")
        
        start_time = time.time()
        
        # Run PLS with more generous parameters for medium instances
        logger.info("Starting PLS algorithm for medium instance...")
        solutions = sims_problem.solve_with_pls_advanced(
            problem,
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
            # Check basic properties
            assert solution.cost >= 0, f"Solution {i} should have non-negative cost"
            assert solution.cloudy_area >= 0, f"Solution {i} should have non-negative cloudy area"
            
            # Validate solution
            is_valid = solution.validate(problem)
            if is_valid:
                valid_solutions.append(solution)
            else:
                logger.warning(f"Medium instance solution {i} is invalid")
        
        # Should have at least some valid solutions
        assert len(valid_solutions) > 0, f"Should find at least one valid solution for {filename}"
        
        logger.info(f"Valid solutions: {len(valid_solutions)} out of {len(solutions)} total")

        # Print solution statistics
        costs = [sol.cost for sol in solutions]
        cloudy_areas = [sol.cloudy_area for sol in solutions]
        
        print(f"Cost range: {min(costs)} - {max(costs)}")
        print(f"Cloudy area range: {min(cloudy_areas)} - {max(cloudy_areas)}")

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
    def test_pls_on_large_instances(self, filename, test_instances):
        """Test PLS on large instances (150+ images) with extended timeout."""
        if filename not in test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = test_instances[filename]
        
        # Validate the problem instance
        problem.validate()
        
        print(f"\nTesting {filename}: {problem.num_images} images, {problem.universe} universe")
        
        start_time = time.time()
        
        # Run PLS with extended parameters for large instances
        solutions = sims_problem.solve_with_pls_advanced(
            problem,
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

    def test_deterministic_behavior_real_data(self, test_instances):
        """Test deterministic behavior on a real data instance."""
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
        
        solutions1 = sims_problem.solve_with_pls_advanced(problem, **common_params)
        solutions2 = sims_problem.solve_with_pls_advanced(problem, **common_params)
        
        # Should produce same number of solutions
        assert len(solutions1) == len(solutions2), \
            "Deterministic runs should produce same number of solutions"
        
        # Convert to sets of (cost, cloudy_area) for comparison
        objectives1 = {(sol.cost, sol.cloudy_area) for sol in solutions1}
        objectives2 = {(sol.cost, sol.cloudy_area) for sol in solutions2}
        
        assert objectives1 == objectives2, \
            "Deterministic runs should produce solutions with same objectives"

    def test_solution_diversity(self, test_instances):
        """Test that PLS produces diverse solutions."""
        # Use a medium-sized instance
        filename = "paris_50.dzn"
        if filename not in test_instances:
            pytest.skip(f"Test instance {filename} not available")
        
        problem = test_instances[filename]
        
        solutions = sims_problem.solve_with_pls_advanced(
            problem,
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
