"""
Shared utilities and fixtures for real data integration tests.

This module contains common configuration classes, fixtures, and helper methods
used across different algorithm integration tests (PLS, MILP, etc.).
"""
import pytest
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Optional
from .test_data_loader import get_all_test_instances, load_test_instance_as_problem

# Configure logging to capture Rust logs
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@dataclass
class AlgorithmConfig:
    """Base configuration for algorithm parameters."""
    timeout_seconds: float


@dataclass
class PLSConfig(AlgorithmConfig):
    """Configuration for PLS algorithm parameters."""
    max_iterations: int
    is_deterministic: bool
    initial_population_size: int
    neighborhood_size_min: int
    neighborhood_size_max: int


@dataclass
class MILPConfig(AlgorithmConfig):
    """Configuration for MILP algorithm parameters.
    
    Note: The timeout_seconds parameter is passed to the solver but may not be 
    fully enforced in the current AUGMECON implementation. The solver will attempt 
    to respect the timeout but this is dependent on the underlying solver backend.
    """
    objectives: List[str]
    grid_points: int
    bypass_coefficient: bool
    early_exit: bool
    flag_array: bool
    solver_name: str


@dataclass
class AlgorithmResult:
    """Result from running any algorithm."""
    solutions: List[Any]
    execution_time: float
    valid_solutions: List[Any]
    plot_path: Optional[Path] = None
    algorithm_name: str = "unknown"


# Common fixtures for all integration tests
@pytest.fixture(scope="session")
def all_test_instances():
    """Load all test instances once for the entire test session."""
    instances = {}
    for filename in get_all_test_instances():
        try:
            # Use the new Rust-based loader for better performance
            problem = load_test_instance_as_problem(filename)
            instances[filename] = problem
        except Exception as e:
            pytest.fail(f"Failed to load test instance {filename}: {e}")
    return instances


@pytest.fixture
def plots_directory():
    """Create and return plots directory."""
    plots_dir = Path(__file__).parent / 'plot_artifacts'
    plots_dir.mkdir(exist_ok=True)
    return plots_dir


@pytest.fixture
def logger():
    """Get logger instance."""
    return logging.getLogger(__name__)


# PLS Configuration fixtures
@pytest.fixture
def small_pls_config():
    """Configuration for small instances with PLS."""
    return PLSConfig(
        timeout_seconds=120.0,
        max_iterations=10000,
        is_deterministic=True,
        initial_population_size=100,
        neighborhood_size_min=1,
        neighborhood_size_max=8
    )


@pytest.fixture
def medium_pls_config():
    """Configuration for medium instances with PLS."""
    return PLSConfig(
        timeout_seconds=300.0,
        max_iterations=15000,
        is_deterministic=True,
        initial_population_size=150,
        neighborhood_size_min=1,
        neighborhood_size_max=10
    )


@pytest.fixture
def large_pls_config():
    """Configuration for large instances with PLS."""
    return PLSConfig(
        timeout_seconds=300.0,
        max_iterations=20000,
        is_deterministic=False,
        initial_population_size=150,
        neighborhood_size_min=1,
        neighborhood_size_max=8
    )


# MILP Configuration fixtures
@pytest.fixture
def small_milp_config():
    """Configuration for small instances with MILP."""
    return MILPConfig(
        timeout_seconds=300.0,  # 5 minutes for small instances
        objectives=["min_cost", "cloud_coverage"],
        grid_points=25,  # Fewer grid points for faster execution
        bypass_coefficient=True,
        early_exit=True,
        flag_array=True,
        solver_name="cbc"
    )


@pytest.fixture
def medium_milp_config():
    """Configuration for medium instances with MILP."""
    return MILPConfig(
        timeout_seconds=600.0,  # 10 minutes for medium instances
        objectives=["min_cost", "cloud_coverage"],
        grid_points=30,
        bypass_coefficient=True,
        early_exit=True,
        flag_array=True,
        solver_name="cbc"
    )


@pytest.fixture
def large_milp_config():
    """Configuration for large instances with MILP."""
    return MILPConfig(
        timeout_seconds=900.0,  # 15 minutes for large instances
        objectives=["min_cost", "cloud_coverage"],
        grid_points=20,  # Fewer grid points for very large instances
        bypass_coefficient=True,
        early_exit=True,
        flag_array=True,
        solver_name="cbc"
    )


# Test instance categories
SMALL_INSTANCES = [
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
]

MEDIUM_INSTANCES = [
    "lagos_nigeria_100.dzn",
    "mexico_city_100.dzn",
    "paris_100.dzn",
    "rio_de_janeiro_100.dzn",
    "tokyo_bay_100.dzn",
]

LARGE_INSTANCES = [
    "lagos_nigeria_145.dzn",
    "mexico_city_150.dzn",
    "mexico_city_200.dzn",
    "paris_150.dzn",
    "paris_200.dzn",
    "rio_de_janeiro_150.dzn",
    "rio_de_janeiro_200.dzn",
    "tokyo_bay_150.dzn",
    "tokyo_bay_200.dzn",
]


# Common helper functions
def validate_solutions(solutions, problem, logger) -> List[Any]:
    """Validate solutions and return list of valid ones."""
    valid_solutions = []
    
    for i, solution in enumerate(solutions):
        # Check basic properties
        assert solution.cost >= 0, f"Solution {i} should have non-negative cost"
        assert solution.cloudy_area >= 0, f"Solution {i} should have non-negative cloudy area"
        assert solution.timestamp.seconds >= 0, f"Solution {i} should have non-negative timestamp"
        
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
            debug_invalid_solution(solution, problem, logger, i)
    
    return valid_solutions


def debug_invalid_solution(solution, problem, logger, solution_index: int):
    """Debug an invalid solution by analyzing coverage."""
    selected_images = solution.get_selected_images_list()
    coverage = set()
    for img_idx in selected_images:
        coverage.update(problem.images[img_idx])
    uncovered = set(range(problem.universe)) - coverage
    logger.warning(f"Solution {solution_index} missing coverage for {len(uncovered)} elements: {sorted(list(uncovered))[:20]}...")


def analyze_problem_coverage(problem, logger, filename: str):
    """Analyze problem coverage and log detailed information."""
    logger.error(f"No valid solutions found for {filename}")
    logger.error("Problem coverage analysis:")
    all_elements = set()
    for i, img in enumerate(problem.images):
        all_elements.update(img)
        logger.debug(f"Image {i}: covers {len(img)} elements")
    logger.error(f"All images together cover {len(all_elements)} out of {problem.universe} universe elements")
    missing_from_all = set(range(problem.universe)) - all_elements
    if missing_from_all:
        logger.error(f"Elements never covered by any image: {sorted(list(missing_from_all))}")


def log_solution_statistics(result: AlgorithmResult, logger):
    """Log solution statistics."""
    if not result.valid_solutions:
        logger.warning("No valid solutions to analyze")
        return
        
    costs = [sol.cost for sol in result.valid_solutions]
    cloudy_areas = [sol.cloudy_area for sol in result.valid_solutions]
    
    logger.info(f"Valid solutions: {len(result.valid_solutions)} out of {len(result.solutions)} total")
    logger.info(f"Cost range: {min(costs)} - {max(costs)}")
    logger.info(f"Cloudy area range: {min(cloudy_areas)} - {max(cloudy_areas)}")

    if result.plot_path:
        print(f"Plots saved to: {result.plot_path}")


def create_plot_path(plots_directory: Path, algorithm_name: str, instance_size: str, filename: str) -> Path:
    """Create a standardized plot path for algorithm results."""
    clean_filename = filename.replace(".dzn", "")
    return plots_directory / f'{algorithm_name}_{instance_size}_{clean_filename}.svg'


def run_algorithm_with_timing(algorithm_func, *args, **kwargs) -> tuple[Any, float]:
    """Run an algorithm function and return results with execution time."""
    start_time = time.time()
    result = algorithm_func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time
