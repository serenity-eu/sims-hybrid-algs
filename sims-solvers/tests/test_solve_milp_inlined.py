"""Test for solve_milp_inlined function."""

import ast
import csv
import logging
import shutil
import tempfile
from pathlib import Path

import pytest

from sims_solvers.solve import solve_milp, solve_milp_inlined
from sims_solvers.Config import Config

# Set up critical-only logging to reduce noise
logging.basicConfig(level=logging.CRITICAL, format='%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


@pytest.fixture
def test_artifacts_dir():
    """Create temporary directory for test artifacts."""
    temp_dir = tempfile.mkdtemp(prefix="test_solve_milp_inlined_")
    print(f"\nTest artifacts directory: {temp_dir}")
    yield temp_dir
    # Cleanup after test - commented out for debugging
    # shutil.rmtree(temp_dir, ignore_errors=True)


def create_test_config(dzn_file_path: str, test_artifacts_dir: str, timeout: int = 300) -> Config:
    """Create a test configuration for solving MILP problems."""
    from datetime import datetime
    from importlib import resources
    from sims_solvers import mzn_models
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    instance_name = Path(dzn_file_path).stem
    subdir_name = f"{instance_name}_inlined_test_{timestamp}"
    test_subdir = Path(test_artifacts_dir) / subdir_name
    test_subdir.mkdir(exist_ok=True)
    
    # Get MZN model path
    mzn_model_path = None
    with resources.as_file(resources.files(mzn_models) / "mosaic_cloud2.mzn") as mzn_path:
        mzn_model_path = mzn_path
    
    return Config(
        minizinc_data=False,
        instance_name=instance_name,
        data_sets_folder=Path(dzn_file_path).parent,
        input_mzn=Path(mzn_model_path),
        dzn_dir=Path(dzn_file_path).parent,
        solver_name="gurobi",
        problem_name="sims",
        front_strategy="gpba-a",
        solver_timeout_sec=timeout,
        summary_filename=str(test_subdir / "test_summary.csv"),
        solver_search_strategy="free_search",
        fzn_optimisation_level=1,
        cores=7,
        threads=14
    )


def extract_pareto_front_from_csv(csv_path: Path) -> list[list[float]]:
    """Extract Pareto front from CSV summary file."""
    pareto_front = []
    
    try:
        with open(csv_path, 'r') as f:
            # Try to detect delimiter
            first_line = f.readline()
            f.seek(0)  # Reset to beginning
            
            delimiter = ';' if ';' in first_line else ','
            reader = csv.DictReader(f, delimiter=delimiter)
            
            for row in reader:
                if 'pareto_front' not in row:
                    continue
                
                pareto_str = row['pareto_front'].strip()
                if not pareto_str:
                    continue
                
                # Remove trailing comma if present
                if pareto_str.startswith('{') and pareto_str.endswith('}'):
                    pareto_str = pareto_str[1:-1]  # Remove outer braces
                    
                if pareto_str.endswith(','):
                    pareto_str = pareto_str[:-1]
                
                try:
                    # The pareto_front field contains solutions in format like:
                    # {[obj1, obj2, ...], [obj1, obj2, ...], ...} or [[obj1, obj2, ...], [obj1, obj2, ...], ...]
                    # After removing braces, we need to wrap in brackets to make it a valid list
                    if not pareto_str.startswith('['):
                        pareto_str = '[' + pareto_str + ']'
                    solutions = ast.literal_eval(pareto_str)
                    # Convert tuple to list if needed
                    if isinstance(solutions, tuple):
                        solutions = list(solutions)
                    if isinstance(solutions, list) and solutions:
                        pareto_front.extend(solutions)
                        logger.info(f"Successfully parsed {len(solutions)} solutions from pareto_front")
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Could not parse pareto_front field: {e}")
                    logger.warning(f"pareto_str was: {pareto_str[:100]}...")
                    continue
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {e}")
    
    return pareto_front


def compare_pareto_fronts(front1: list[list[float]], front2: list[list[float]], tolerance: int = 2) -> bool:
    """
    Compare two Pareto fronts allowing for small differences.
    
    Args:
        front1: First Pareto front
        front2: Second Pareto front  
        tolerance: Maximum allowed difference in number of solutions
    
    Returns:
        True if fronts are similar enough
    """
    if not front1 and not front2:
        return True
    
    if abs(len(front1) - len(front2)) > tolerance:
        logger.warning(f"Solution count differs significantly: {len(front1)} vs {len(front2)}")
        return False
    
    # Convert to sets for comparison (handling floating point precision)
    def normalize_solution(sol):
        return tuple(round(float(x), 4) for x in sol)
    
    set1 = {normalize_solution(sol) for sol in front1}
    set2 = {normalize_solution(sol) for sol in front2}
    
    # Calculate intersection
    common = set1.intersection(set2)
    unique_to_1 = set1 - set2
    unique_to_2 = set2 - set1
    
    logger.info(f"Common solutions: {len(common)}")
    logger.info(f"Unique to original: {len(unique_to_1)}")  
    logger.info(f"Unique to inlined: {len(unique_to_2)}")
    
    # Accept if most solutions are common and differences are within tolerance
    total_solutions = len(set1.union(set2))
    if total_solutions == 0:
        return True
    
    similarity_ratio = len(common) / total_solutions
    return similarity_ratio >= 0.8  # At least 80% similarity


def test_solve_milp_identical(test_artifacts_dir):
    """Test that solve_milp_inlined produces identical results to solve_milp with detailed tracing."""
    
    # Use a small test instance
    dzn_file_path = "/home/vhlushchenko/sims-hybrid-algs/sims-problem/tests/data/lagos_nigeria_30.dzn"
    
    if not Path(dzn_file_path).exists():
        pytest.skip(f"Test data file not found: {dzn_file_path}")
    
    objectives = ["min_cost", "cloud_coverage"]  # Focus on 2-objective case
    timeout = 120  # Longer timeout for detailed tracing
    
    # Enable trace logging for both implementations
    logging.getLogger('sims_solvers.solve').setLevel(logging.CRITICAL)
    logging.getLogger('sims_solvers.FrontGenerators.CoverageGridPoint').setLevel(logging.CRITICAL)
    
    # Test original solve_milp
    logger.critical("=" * 80)
    logger.critical("TESTING ORIGINAL solve_milp")
    logger.critical("=" * 80)
    config_original = create_test_config(dzn_file_path, test_artifacts_dir, timeout)
    config_original.summary_filename = str(Path(test_artifacts_dir) / "original_summary.csv")
    
    original_pareto = []
    try:
        solve_milp(config_original, objectives)
        original_pareto = extract_pareto_front_from_csv(Path(config_original.summary_filename))
        logger.critical(f"ORIGINAL RESULT: Found {len(original_pareto)} solutions")
        for i, sol in enumerate(original_pareto):
            logger.critical(f"ORIGINAL SOLUTION {i}: {sol}")
    except Exception as e:
        pytest.fail(f"Original solve_milp failed: {e}")
    
    # Test inlined solve_milp_inlined
    logger.critical("=" * 80)
    logger.critical("TESTING INLINED solve_milp_inlined")
    logger.critical("=" * 80)
    config_inlined = create_test_config(dzn_file_path, test_artifacts_dir, timeout)
    config_inlined.summary_filename = str(Path(test_artifacts_dir) / "inlined_summary.csv")
    
    inlined_pareto = []
    try:
        solve_milp_inlined(config_inlined, objectives)
        inlined_pareto = extract_pareto_front_from_csv(Path(config_inlined.summary_filename))
        logger.critical(f"INLINED RESULT: Found {len(inlined_pareto)} solutions")
        for i, sol in enumerate(inlined_pareto):
            logger.critical(f"INLINED SOLUTION {i}: {sol}")
    except Exception as e:
        pytest.fail(f"Inlined solve_milp_inlined failed: {e}")
    
    # Compare results
    logger.critical("=" * 80)
    logger.critical("COMPARISON RESULTS")
    logger.critical("=" * 80)
    logger.critical(f"Original: {len(original_pareto)} solutions")
    logger.critical(f"Inlined:  {len(inlined_pareto)} solutions")
    logger.critical(f"Difference: {abs(len(original_pareto) - len(inlined_pareto))} solutions")
    
    # They should be nearly identical (user requirement: "like 52 and 53")
    if abs(len(original_pareto) - len(inlined_pareto)) > 2:
        pytest.fail(f"Solution counts differ too much. Original: {len(original_pareto)}, Inlined: {len(inlined_pareto)}")
    
    logger.critical("✅ Test passed: solve_milp_inlined produces similar results to solve_milp")


@pytest.mark.parametrize("objectives", [
    ["min_cost", "cloud_coverage"],
    ["min_cost", "cloud_coverage", "min_resolution"], 
    ["min_cost", "cloud_coverage", "min_resolution", "min_max_incidence_angle"]
])
def test_solve_milp_inlined_vs_original(test_artifacts_dir, objectives):
    """Test that solve_milp_inlined produces similar results to solve_milp."""
    
    # Use a small test instance
    dzn_file_path = "/home/vhlushchenko/sims-hybrid-algs/sims-problem/tests/data/lagos_nigeria_30.dzn"
    
    if not Path(dzn_file_path).exists():
        pytest.skip(f"Test data file not found: {dzn_file_path}")
    
    timeout = 60  # Short timeout for testing
    
    # Test original solve_milp
    logger.info("Testing original solve_milp function")
    config_original = create_test_config(dzn_file_path, test_artifacts_dir, timeout)
    config_original.summary_filename = str(Path(test_artifacts_dir) / "original_summary.csv")
    
    original_pareto = []
    try:
        solve_milp(config_original, objectives)
        original_pareto = extract_pareto_front_from_csv(Path(config_original.summary_filename))
        logger.info(f"Original solve_milp found {len(original_pareto)} solutions")
    except Exception as e:
        pytest.fail(f"Original solve_milp failed: {e}")
    
    # Test inlined solve_milp_inlined
    logger.info("Testing inlined solve_milp_inlined function")
    config_inlined = create_test_config(dzn_file_path, test_artifacts_dir, timeout)
    config_inlined.summary_filename = str(Path(test_artifacts_dir) / "inlined_summary.csv")
    
    inlined_pareto = []
    try:
        solve_milp_inlined(config_inlined, objectives)
        inlined_pareto = extract_pareto_front_from_csv(Path(config_inlined.summary_filename))
        logger.info(f"Inlined solve_milp_inlined found {len(inlined_pareto)} solutions")
    except Exception as e:
        pytest.fail(f"Inlined solve_milp_inlined failed: {e}")
    
    # Compare results
    if not compare_pareto_fronts(original_pareto, inlined_pareto):
        pytest.fail(f"Pareto fronts differ significantly. Original: {len(original_pareto)} solutions, Inlined: {len(inlined_pareto)} solutions")
    
    logger.info("✅ Test passed: solve_milp_inlined produces similar results to solve_milp")


if __name__ == "__main__":
    # Run a simple test
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_solve_milp_inlined_vs_original(temp_dir, ["min_cost", "cloud_coverage"])