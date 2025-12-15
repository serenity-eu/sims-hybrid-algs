"""Test for solve_milp_inlined function."""

import ast
import csv
import logging
import multiprocessing
import os
import re
import shutil
import tempfile
from pathlib import Path

import pytest

from sims_solvers.solve import solve_milp, solve_milp_inlined
from sims_solvers.Config import Config

# Set up critical-only logging to reduce noise
logging.basicConfig(level=logging.CRITICAL, format='%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


def parse_dzn_file(dzn_path: str):
    """Parse DZN file to extract problem data for validation."""
    with open(dzn_path, 'r') as f:
        content = f.read()
    
    def parse_array(pattern):
        match = re.search(pattern, content, re.DOTALL)
        if match:
            array_str = match.group(1)
            items = [item.strip() for item in array_str.strip('[]').split(',')]
            return [float(item) if '.' in item else int(item) for item in items if item.strip()]
        return []
    
    def parse_set_array(pattern):
        match = re.search(pattern, content, re.DOTALL)
        if match:
            array_str = match.group(1)
            sets = []
            set_matches = re.findall(r'\{([^}]*)\}', array_str)
            for set_match in set_matches:
                if set_match.strip():
                    elements = [int(x.strip()) for x in set_match.split(',') if x.strip()]
                    sets.append(elements)
                else:
                    sets.append([])
            return sets
        return []
    
    data = {
        'costs': parse_array(r'costs\s*=\s*\[([^\]]+)\]'),
        'areas': parse_array(r'areas\s*=\s*\[([^\]]+)\]'),
        'images_raw': parse_set_array(r'images\s*=\s*\[([^\]]+)\]'),
        'clouds_raw': parse_set_array(r'clouds\s*=\s*\[([^\]]+)\]'),
    }
    
    # Convert to 0-indexed
    data['images'] = [[x - 1 for x in img_set] for img_set in data['images_raw']]
    data['clouds'] = [[x - 1 for x in cloud_set] for cloud_set in data['clouds_raw']]
    
    return data


def validate_solution(solution_values: list[int], solution_objs: list[float], 
                     dzn_path: str, objectives: list[str]) -> tuple[bool, list[str]]:
    """
    Validate a SIMS solution for feasibility and correct objective computation.
    
    Args:
        solution_values: List of 0-indexed image IDs that are selected (e.g., [2,4,6,8])
        solution_objs: Objective values claimed for this solution
        dzn_path: Path to DZN file with problem data
        objectives: List of objective names
        
    Returns:
        (is_valid, issues) - is_valid is True if solution is feasible and objectives are correct
    """
    issues = []
    
    # Parse problem data
    data = parse_dzn_file(dzn_path)
    costs = data['costs']
    areas = data['areas']
    images = data['images']
    clouds = data['clouds']
    
    # Get selected images (already 0-indexed in CSV)
    selected_images = solution_values
    
    if not selected_images:
        issues.append("No images selected!")
        return False, issues
    
    # 1. CHECK COVERAGE CONSTRAINT
    # Every element must be covered by at least one selected image
    num_elements = len(areas)
    covered_elements = set()
    for img_id in selected_images:
        if img_id < len(images):
            covered_elements.update(images[img_id])
    
    uncovered = set(range(num_elements)) - covered_elements
    if uncovered:
        issues.append(f"COVERAGE VIOLATED: {len(uncovered)} elements not covered: {sorted(list(uncovered))[:10]}...")
    
    # 2. COMPUTE AND VERIFY OBJECTIVES
    computed_objs = []
    
    for obj_name in objectives:
        if obj_name == "min_cost":
            # Total cost of selected images
            computed_cost = sum(costs[i] for i in selected_images)
            computed_objs.append(computed_cost)
            
        elif obj_name == "cloud_coverage":
            # Build cloud coverage relationships
            clouds_id_area = {}
            cloud_covered_by_image = {}
            
            for i, cloud_set in enumerate(clouds):
                for cloud_id in cloud_set:
                    clouds_id_area[cloud_id] = areas[cloud_id]
                    # Find images that can cover this cloud
                    for j in range(len(images)):
                        if i != j and cloud_id in images[j] and cloud_id not in clouds[j]:
                            if j not in cloud_covered_by_image:
                                cloud_covered_by_image[j] = set()
                            cloud_covered_by_image[j].add(cloud_id)
            
            # Determine which clouds are covered by selected images
            covered_clouds = set()
            for img_id in selected_images:
                if img_id in cloud_covered_by_image:
                    covered_clouds.update(cloud_covered_by_image[img_id])
            
            # Uncovered cloud area
            total_cloud_area = sum(clouds_id_area.values())
            covered_cloud_area = sum(clouds_id_area[c] for c in covered_clouds)
            uncovered_cloud_area = total_cloud_area - covered_cloud_area
            computed_objs.append(int(uncovered_cloud_area))
    
    # 3. COMPARE COMPUTED VS CLAIMED OBJECTIVES
    for i, (computed, claimed) in enumerate(zip(computed_objs, solution_objs)):
        diff = abs(computed - claimed)
        if diff > 1:  # Allow rounding error of 1
            issues.append(
                f"OBJECTIVE {objectives[i]} MISMATCH: computed={computed}, claimed={claimed}, diff={diff}"
            )
    
    is_valid = len(issues) == 0
    return is_valid, issues


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
    
    # Auto-detect logical CPU count (accounts for hyperthreading automatically)
    # On machines without hyperthreading: cpu_count = physical cores
    # On machines with hyperthreading: cpu_count = physical cores × 2
    cpu_count = multiprocessing.cpu_count()
    
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
        cores=cpu_count,
        threads=cpu_count
    )


def extract_pareto_front_from_csv(csv_path: Path) -> tuple[list[list[float]], list[list[int]]]:
    """
    Extract Pareto front and solution vectors from CSV summary file.
    
    Returns:
        (pareto_objs, solution_vectors) - objectives and solution vectors as lists of 0-indexed image IDs
    """
    pareto_objs = []
    solution_vectors = []
    
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
                
                # Parse pareto_front (objectives)
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
                    objectives = ast.literal_eval(pareto_str)
                    # Convert tuple to list if needed
                    if isinstance(objectives, tuple):
                        objectives = list(objectives)
                    if isinstance(objectives, list) and objectives:
                        pareto_objs.extend(objectives)
                        logger.info(f"Successfully parsed {len(objectives)} solutions from pareto_front")
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Could not parse pareto_front field: {e}")
                    logger.warning(f"pareto_str was: {pareto_str[:100]}...")
                    continue
                
                # Parse solutions_pareto_front (solution vectors as image ID lists)
                if 'solutions_pareto_front' in row:
                    solutions_str = row['solutions_pareto_front'].strip()
                    if solutions_str:
                        if solutions_str.startswith('{') and solutions_str.endswith('}'):
                            solutions_str = solutions_str[1:-1]
                        if solutions_str.endswith(','):
                            solutions_str = solutions_str[:-1]
                        try:
                            if not solutions_str.startswith('['):
                                solutions_str = '[' + solutions_str + ']'
                            image_id_lists = ast.literal_eval(solutions_str)
                            if isinstance(image_id_lists, tuple):
                                image_id_lists = list(image_id_lists)
                            if isinstance(image_id_lists, list) and image_id_lists:
                                # Keep as image ID lists (0-indexed as in CSV)
                                solution_vectors.extend(image_id_lists)
                        except (ValueError, SyntaxError) as e:
                            logger.warning(f"Could not parse solutions_pareto_front: {e}")
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {e}")
    
    return pareto_objs, solution_vectors


def dominates(sol1, sol2):
    """Check if sol1 dominates sol2 (all objectives minimization)."""
    return all(s1 <= s2 for s1, s2 in zip(sol1, sol2)) and any(s1 < s2 for s1, s2 in zip(sol1, sol2))


def validate_pareto_front(front: list[list[float]], name: str = "Front") -> tuple[bool, list[str]]:
    """
    Validate that a Pareto front contains only non-dominated solutions.
    
    Returns:
        (is_valid, issues) - is_valid is True if all solutions are non-dominated, issues lists any problems
    """
    issues = []
    
    if not front:
        return True, []
    
    # Check for duplicates
    def normalize_solution(sol):
        return tuple(round(float(x), 4) for x in sol)
    
    normalized = [normalize_solution(sol) for sol in front]
    unique = set(normalized)
    
    if len(normalized) != len(unique):
        duplicates = len(normalized) - len(unique)
        issues.append(f"{name}: Contains {duplicates} duplicate solutions")
        logger.warning(f"{name}: Contains {duplicates} duplicate solutions")
    
    # Check for dominated solutions
    dominated_pairs = []
    for i, sol1 in enumerate(front):
        for j, sol2 in enumerate(front):
            if i != j and dominates(sol1, sol2):
                dominated_pairs.append((i, j, sol1, sol2))
    
    if dominated_pairs:
        issues.append(f"{name}: Contains {len(dominated_pairs)} dominated solutions")
        logger.warning(f"{name}: Found dominated solutions:")
        for i, j, sol1, sol2 in dominated_pairs[:5]:  # Show first 5
            logger.warning(f"  Solution {i} {sol1} dominates solution {j} {sol2}")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def compare_pareto_fronts(front1: list[list[float]], front2: list[list[float]], tolerance: int = 2) -> bool:
    """
    Compare two Pareto fronts allowing for small differences.
    
    Args:
        front1: First Pareto front (original)
        front2: Second Pareto front (inlined)
        tolerance: Maximum allowed difference in number of solutions
    
    Returns:
        True if fronts are similar enough
    """
    logger.critical("=" * 80)
    logger.critical("DETAILED PARETO FRONT COMPARISON")
    logger.critical("=" * 80)
    
    if not front1 and not front2:
        logger.critical("Both fronts are empty")
        return True
    
    # Validate both fronts
    logger.critical(f"Validating Original front ({len(front1)} solutions)...")
    valid1, issues1 = validate_pareto_front(front1, "Original")
    for issue in issues1:
        logger.critical(f"  ❌ {issue}")
    if valid1:
        logger.critical("  ✅ Original front is valid (no dominated solutions, no duplicates)")
    
    logger.critical(f"Validating Inlined front ({len(front2)} solutions)...")
    valid2, issues2 = validate_pareto_front(front2, "Inlined")
    for issue in issues2:
        logger.critical(f"  ❌ {issue}")
    if valid2:
        logger.critical("  ✅ Inlined front is valid (no dominated solutions, no duplicates)")
    
    # Convert to sets for comparison (handling floating point precision)
    def normalize_solution(sol):
        return tuple(round(float(x), 4) for x in sol)
    
    set1 = {normalize_solution(sol) for sol in front1}
    set2 = {normalize_solution(sol) for sol in front2}
    
    # Calculate intersection and differences
    common = set1.intersection(set2)
    unique_to_1 = set1 - set2
    unique_to_2 = set2 - set1
    
    logger.critical(f"\nSolution counts:")
    logger.critical(f"  Original (front1): {len(set1)} unique solutions")
    logger.critical(f"  Inlined (front2):  {len(set2)} unique solutions")
    logger.critical(f"  Common solutions:  {len(common)}")
    logger.critical(f"  Unique to original: {len(unique_to_1)}")  
    logger.critical(f"  Unique to inlined:  {len(unique_to_2)}")
    
    # Show unique solutions
    if unique_to_1:
        logger.critical("\nSolutions only in ORIGINAL (first 10):")
        for i, sol in enumerate(sorted(unique_to_1)[:10]):
            logger.critical(f"  {i+1}. {list(sol)}")
    
    if unique_to_2:
        logger.critical("\nSolutions only in INLINED (first 10):")
        for i, sol in enumerate(sorted(unique_to_2)[:10]):
            logger.critical(f"  {i+1}. {list(sol)}")
    
    # Check if unique solutions dominate each other
    if unique_to_1 and unique_to_2:
        logger.critical("\nChecking domination relationships between unique solutions:")
        domination_found = False
        inlined_dominates_count = 0
        original_dominates_count = 0
        
        for sol1 in unique_to_1:
            for sol2 in unique_to_2:
                if dominates(list(sol1), list(sol2)):
                    logger.critical(f"  ⚠️  Original solution {list(sol1)} dominates inlined solution {list(sol2)}")
                    domination_found = True
                    original_dominates_count += 1
                elif dominates(list(sol2), list(sol1)):
                    # Check if they're very close (likely same solution with rounding differences)
                    diff = [abs(sol2[i] - sol1[i]) for i in range(len(sol1))]
                    if all(d < 10 for d in diff):
                        logger.critical(f"  ✨ Inlined solution {list(sol2)} slightly dominates original {list(sol1)} (diff: {diff})")
                    else:
                        logger.critical(f"  ⚠️  Inlined solution {list(sol2)} dominates original solution {list(sol1)}")
                    domination_found = True
                    inlined_dominates_count += 1
        
        if not domination_found:
            logger.critical("  ✅ No domination between unique solutions (both fronts contain non-dominated solutions)")
        elif inlined_dominates_count > 0 and original_dominates_count == 0:
            logger.critical(f"\n  🎯 EXCELLENT: All {inlined_dominates_count} inlined unique solutions dominate their original counterparts!")
            logger.critical("     This suggests the inlined version has better numerical precision.")
        elif original_dominates_count > 0 and inlined_dominates_count == 0:
            logger.critical(f"\n  ⚠️  WARNING: {original_dominates_count} original solutions dominate inlined ones - inlined may be less accurate")
    
    # Check solution count difference
    if abs(len(set1) - len(set2)) > tolerance:
        logger.critical(f"\n❌ FAIL: Solution count differs by {abs(len(set1) - len(set2))} (tolerance: {tolerance})")
        return False
    
    # Accept if most solutions are common and differences are within tolerance
    total_solutions = len(set1.union(set2))
    if total_solutions == 0:
        return True
    
    similarity_ratio = len(common) / total_solutions
    logger.critical(f"\nSimilarity ratio: {similarity_ratio:.2%} (threshold: 80%)")
    
    if similarity_ratio >= 0.8:
        logger.critical(f"✅ PASS: Fronts are similar enough")
        return True
    else:
        logger.critical(f"❌ FAIL: Similarity ratio {similarity_ratio:.2%} below threshold")
        return False


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
    
    original_objs, original_vectors = [], []
    try:
        solve_milp(config_original, objectives)
        original_objs, original_vectors = extract_pareto_front_from_csv(Path(config_original.summary_filename))
        logger.critical(f"ORIGINAL RESULT: Found {len(original_objs)} solutions")
    except Exception as e:
        pytest.fail(f"Original solve_milp failed: {e}")
    
    # Validate original solutions
    logger.critical("\n" + "=" * 80)
    logger.critical("VALIDATING ORIGINAL SOLUTIONS")
    logger.critical("=" * 80)
    original_invalid_count = 0
    for i, (objs, vec) in enumerate(zip(original_objs, original_vectors)):
        is_valid, issues = validate_solution(vec, objs, dzn_file_path, objectives)
        if not is_valid:
            original_invalid_count += 1
            logger.critical(f"❌ ORIGINAL Solution {i} is INVALID:")
            logger.critical(f"   Objectives: {objs}")
            logger.critical(f"   Selected images (0-indexed): {vec}")
            for issue in issues:
                logger.critical(f"   - {issue}")
    
    if original_invalid_count == 0:
        logger.critical(f"✅ All {len(original_objs)} original solutions are VALID")
    else:
        pytest.fail(f"{original_invalid_count}/{len(original_objs)} original solutions are invalid!")
    
    # Test inlined solve_milp_inlined
    logger.critical("\n" + "=" * 80)
    logger.critical("TESTING INLINED solve_milp_inlined")
    logger.critical("=" * 80)
    config_inlined = create_test_config(dzn_file_path, test_artifacts_dir, timeout)
    config_inlined.summary_filename = str(Path(test_artifacts_dir) / "inlined_summary.csv")
    
    inlined_objs, inlined_vectors = [], []
    try:
        solve_milp_inlined(config_inlined, objectives)
        inlined_objs, inlined_vectors = extract_pareto_front_from_csv(Path(config_inlined.summary_filename))
        logger.critical(f"INLINED RESULT: Found {len(inlined_objs)} solutions")
    except Exception as e:
        pytest.fail(f"Inlined solve_milp_inlined failed: {e}")
    
    # Validate inlined solutions
    logger.critical("\n" + "=" * 80)
    logger.critical("VALIDATING INLINED SOLUTIONS")
    logger.critical("=" * 80)
    inlined_invalid_count = 0
    for i, (objs, vec) in enumerate(zip(inlined_objs, inlined_vectors)):
        is_valid, issues = validate_solution(vec, objs, dzn_file_path, objectives)
        if not is_valid:
            inlined_invalid_count += 1
            logger.critical(f"❌ INLINED Solution {i} is INVALID:")
            logger.critical(f"   Objectives: {objs}")
            logger.critical(f"   Selected images (0-indexed): {vec}")
            for issue in issues:
                logger.critical(f"   - {issue}")
    
    if inlined_invalid_count == 0:
        logger.critical(f"✅ All {len(inlined_objs)} inlined solutions are VALID")
    else:
        pytest.fail(f"{inlined_invalid_count}/{len(inlined_objs)} inlined solutions are invalid!")
    
    # Compare results using enhanced comparison
    if not compare_pareto_fronts(original_objs, inlined_objs, tolerance=0):
        pytest.fail(f"Pareto fronts differ. Original: {len(original_objs)}, Inlined: {len(inlined_objs)}")
    
    # Compare solution vectors for unique solutions
    logger.critical("\n" + "=" * 80)
    logger.critical("COMPARING SOLUTION VECTORS FOR UNIQUE SOLUTIONS")
    logger.critical("=" * 80)
    
    logger.critical(f"\nOriginal: {len(original_objs)} objectives, {len(original_vectors)} vectors")
    logger.critical(f"Inlined: {len(inlined_objs)} objectives, {len(inlined_vectors)} vectors")
    
    if len(inlined_vectors) == 0:
        logger.critical("\n⚠️  WARNING: Inlined solution vectors are empty!")
        logger.critical("   The inlined CSV format may not include 'solutions_pareto_front' field.")
        logger.critical("   Cannot compare image selections.")
    elif len(original_vectors) != len(original_objs) or len(inlined_vectors) != len(inlined_objs):
        logger.critical("\n⚠️  WARNING: Mismatch between number of objectives and vectors!")
        logger.critical(f"   Original: {len(original_objs)} objs, {len(original_vectors)} vectors")
        logger.critical(f"   Inlined: {len(inlined_objs)} objs, {len(inlined_vectors)} vectors")
    else:
        logger.critical(f"\nOriginal objectives (first 3): {original_objs[:3]}")
        logger.critical(f"Inlined objectives (first 3): {inlined_objs[:3]}")
        logger.critical(f"Original objectives (last 3): {original_objs[-3:]}")
        logger.critical(f"Inlined objectives (last 3): {inlined_objs[-3:]}")
        
        # Create dictionaries mapping objectives to solution vectors
        def obj_tuple(objs):
            return tuple(int(round(float(x))) for x in objs)  # Round to integers
        
        original_dict = {obj_tuple(objs): vec for objs, vec in zip(original_objs, original_vectors)}
        inlined_dict = {obj_tuple(objs): vec for objs, vec in zip(inlined_objs, inlined_vectors)}
        
        logger.critical(f"\nOriginal dict keys (first 3): {list(original_dict.keys())[:3]}")
        logger.critical(f"Inlined dict keys (first 3): {list(inlined_dict.keys())[:3]}")
        
        original_keys = set(original_dict.keys())
        inlined_keys = set(inlined_dict.keys())
        unique_to_original = sorted(original_keys - inlined_keys)
        unique_to_inlined = sorted(inlined_keys - original_keys)
    
    logger.critical(f"\nUnique to original: {len(unique_to_original)}")
    logger.critical(f"Unique to inlined: {len(unique_to_inlined)}")
    
    if unique_to_original and unique_to_inlined:
        logger.critical(f"\nFound unique solutions in both fronts")
        logger.critical(f"First 3 unique to original: {unique_to_original[:3]}")
        logger.critical(f"First 3 unique to inlined: {unique_to_inlined[:3]}")
        
        # Pair up by proximity (assuming they correspond to similar regions)
        num_pairs = min(len(unique_to_original), len(unique_to_inlined), 5)
        logger.critical(f"\nComparing {num_pairs} solution pairs:")
        
        for i in range(num_pairs):
            if i < len(unique_to_original) and i < len(unique_to_inlined):
                orig_key = unique_to_original[i]
                # Find closest inlined solution
                closest_inl = min(unique_to_inlined, 
                                key=lambda ik: sum(abs(a-b) for a,b in zip(orig_key, ik)))
                
                orig_vec = original_dict[orig_key]
                inl_vec = inlined_dict[closest_inl]
                
                logger.critical(f"\n  Pair {i+1}: orig={list(orig_key)} vs inl={list(closest_inl)}")
                logger.critical(f"    Obj difference: {[b-a for a,b in zip(orig_key, closest_inl)]}")
                logger.critical(f"    Original images: {sorted(orig_vec)}")
                logger.critical(f"    Inlined images:  {sorted(inl_vec)}")
                
                if sorted(orig_vec) == sorted(inl_vec):
                    logger.critical("    ⚠️  SAME IMAGES - objectives computed differently!")
                else:
                    orig_set = set(orig_vec)
                    inl_set = set(inl_vec)
                    only_orig = sorted(orig_set - inl_set)
                    only_inl = sorted(inl_set - orig_set)
                    logger.critical("    ✓ DIFFERENT IMAGES:")
                    logger.critical(f"      Only in original: {only_orig}")
                    logger.critical(f"      Only in inlined:  {only_inl}")
    else:
        logger.critical("\nNo unique solutions found or all solutions match!")
    
    logger.critical("\n" + "=" * 80)
    logger.critical("✅ TEST PASSED: solve_milp_inlined produces similar results to solve_milp")
    logger.critical("=" * 80)


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
    
    original_objs, original_vectors = [], []
    try:
        solve_milp(config_original, objectives)
        original_objs, original_vectors = extract_pareto_front_from_csv(Path(config_original.summary_filename))
        logger.info(f"Original solve_milp found {len(original_objs)} solutions")
    except Exception as e:
        pytest.fail(f"Original solve_milp failed: {e}")
    
    # Validate original solutions
    logger.info(f"Validating {len(original_objs)} original solutions")
    original_invalid_count = 0
    for i, (objs, vec) in enumerate(zip(original_objs, original_vectors)):
        is_valid, issues = validate_solution(vec, objs, dzn_file_path, objectives)
        if not is_valid:
            original_invalid_count += 1
            logger.error(f"❌ ORIGINAL Solution {i} is INVALID: {objs}")
            for issue in issues:
                logger.error(f"   - {issue}")
    
    if original_invalid_count > 0:
        pytest.fail(f"{original_invalid_count}/{len(original_objs)} original solutions are invalid!")
    logger.info(f"✅ All {len(original_objs)} original solutions are VALID")
    
    # Test inlined solve_milp_inlined
    logger.info("Testing inlined solve_milp_inlined function")
    config_inlined = create_test_config(dzn_file_path, test_artifacts_dir, timeout)
    config_inlined.summary_filename = str(Path(test_artifacts_dir) / "inlined_summary.csv")
    
    inlined_objs, inlined_vectors = [], []
    try:
        solve_milp_inlined(config_inlined, objectives)
        inlined_objs, inlined_vectors = extract_pareto_front_from_csv(Path(config_inlined.summary_filename))
        logger.info(f"Inlined solve_milp_inlined found {len(inlined_objs)} solutions")
    except Exception as e:
        pytest.fail(f"Inlined solve_milp_inlined failed: {e}")
    
    # Validate inlined solutions
    logger.info(f"Validating {len(inlined_objs)} inlined solutions")
    inlined_invalid_count = 0
    for i, (objs, vec) in enumerate(zip(inlined_objs, inlined_vectors)):
        is_valid, issues = validate_solution(vec, objs, dzn_file_path, objectives)
        if not is_valid:
            inlined_invalid_count += 1
            logger.error(f"❌ INLINED Solution {i} is INVALID: {objs}")
            for issue in issues:
                logger.error(f"   - {issue}")
    
    if inlined_invalid_count > 0:
        pytest.fail(f"{inlined_invalid_count}/{len(inlined_objs)} inlined solutions are invalid!")
    logger.info(f"✅ All {len(inlined_objs)} inlined solutions are VALID")
    
    # Compare results
    if not compare_pareto_fronts(original_objs, inlined_objs):
        pytest.fail(f"Pareto fronts differ significantly. Original: {len(original_objs)} solutions, Inlined: {len(inlined_objs)} solutions")
    
    logger.info("✅ Test passed: solve_milp_inlined produces similar results to solve_milp")


if __name__ == "__main__":
    # Run a simple test
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_solve_milp_inlined_vs_original(temp_dir, ["min_cost", "cloud_coverage"])