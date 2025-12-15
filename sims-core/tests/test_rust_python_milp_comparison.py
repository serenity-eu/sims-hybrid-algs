"""Test comparing Rust and Python MILP implementations.

This test verifies that the Rust sims-problem MILP solver and Python sims-solvers MILP solver
produce identical results on the same problem instances.

The test uses two wrappers:
- python_milp: Wraps sims_solvers.solve_milp_inlined() via CSV parsing infrastructure
- rust_milp: Wraps sims_problem.solve_with_milp() via direct Rust call

Both wrappers follow the standard sims-core solver interface and return SolverResult objects.
"""

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from sims.core.sims.problem import ProblemInstance
from sims.core.sims.solver_config import FrontStrategy
from sims.core.sims.solvers import python_milp, rust_milp
from solver_test_utils import (
    SMALL_INSTANCES,
    create_problem_instance,
    validate_solver_result,
)

log = logging.getLogger(__name__)


def normalize_objective_name(obj: str) -> str:
    """Normalize objective names between Rust and Python conventions."""
    mapping = {
        "min_cost": "min_cost",
        "cloud_coverage": "cloud_coverage",
        "min_resolution": "min_resolution",
        "min_max_incidence_angle": "min_max_incidence_angle",
    }
    return mapping.get(obj, obj)


def compare_pareto_fronts(rust_result, python_result, objectives, tolerance=1e-6):
    """
    Compare Rust and Python SolverResult Pareto fronts for equivalence.
    
    Args:
        rust_result: SolverResult from Rust solver wrapper
        python_result: SolverResult from Python solver wrapper
        objectives: List of objective names
        tolerance: Numerical tolerance for objective value comparison
    
    Returns:
        Tuple of (is_equal, error_message)
    """
    rust_solutions = rust_result.pareto_front
    python_solutions = python_result.pareto_front
    
    # Define helper function early for use in error messages
    def get_obj_value(sol, obj_name):
        if obj_name == "min_cost":
            return sol.cost
        elif obj_name == "cloud_coverage":
            return sol.cloudy_area
        elif obj_name == "min_resolution":
            return sol.min_resolutions_sum or 0
        elif obj_name == "min_max_incidence_angle":
            return sol.max_incidence_angle or 0
        return 0
    
    # Check if both have same number of solutions
    if len(rust_solutions) != len(python_solutions):
        # Build detailed error message with both fronts
        error_msg = [
            f"Different number of solutions: Rust={len(rust_solutions)}, Python={len(python_solutions)}",
            "",
            f"Full Rust Pareto Front ({len(rust_solutions)} solutions):"
        ]
        for i, sol in enumerate(rust_solutions):
            obj_vals = [get_obj_value(sol, obj) for obj in objectives]
            error_msg.append(f"  [{i}] objectives={obj_vals}, images={sorted(sol.selected_images)[:10]}...")
        
        error_msg.append("")
        error_msg.append(f"Full Python Pareto Front ({len(python_solutions)} solutions):")
        for i, sol in enumerate(python_solutions):
            obj_vals = [get_obj_value(sol, obj) for obj in objectives]
            error_msg.append(f"  [{i}] objectives={obj_vals}, images={sorted(sol.selected_images)[:10]}...")
        
        return False, "\n".join(error_msg)
    
    if len(rust_solutions) == 0:
        return True, ""  # Both empty, consider equal
    
    def get_sort_key(sol):
        """Create tuple of all objective values for stable sorting"""
        return tuple(get_obj_value(sol, obj) for obj in objectives)
    
    rust_sorted = sorted(rust_solutions, key=get_sort_key)
    python_sorted = sorted(python_solutions, key=get_sort_key)
    
    # Compare each solution pair
    for idx, (rust_sol, python_sol) in enumerate(zip(rust_sorted, python_sorted)):
        # Compare selected images
        rust_images = set(rust_sol.selected_images)
        python_images = set(python_sol.selected_images)
        
        if rust_images != python_images:
            # Build detailed error message with both fronts
            error_msg = [
                f"Solution {idx}: Different selected images.",
                f"  Rust images:   {sorted(rust_images)[:15]}...",
                f"  Python images: {sorted(python_images)[:15]}...",
                "",
                f"Full Rust Pareto Front ({len(rust_sorted)} solutions):"
            ]
            for i, sol in enumerate(rust_sorted):
                obj_vals = [get_obj_value(sol, obj) for obj in objectives]
                error_msg.append(f"  [{i}] objectives={obj_vals}, images={sorted(sol.selected_images)[:10]}...")
            
            error_msg.append("")
            error_msg.append(f"Full Python Pareto Front ({len(python_sorted)} solutions):")
            for i, sol in enumerate(python_sorted):
                obj_vals = [get_obj_value(sol, obj) for obj in objectives]
                error_msg.append(f"  [{i}] objectives={obj_vals}, images={sorted(sol.selected_images)[:10]}...")
            
            return False, "\n".join(error_msg)
        
        # Compare objective values
        for obj in objectives:
            rust_val = get_obj_value(rust_sol, obj)
            python_val = get_obj_value(python_sol, obj)
            
            if abs(rust_val - python_val) > tolerance:
                # Build detailed error message with both fronts
                error_msg = [
                    f"Solution {idx}: Different {obj} values.",
                    f"  Rust={rust_val}, Python={python_val}, diff={abs(rust_val - python_val)}",
                    f"  Rust solution:   objectives={[get_obj_value(rust_sol, o) for o in objectives]}, images={sorted(rust_images)[:10]}...",
                    f"  Python solution: objectives={[get_obj_value(python_sol, o) for o in objectives]}, images={sorted(python_images)[:10]}...",
                    "",
                    f"Full Rust Pareto Front ({len(rust_sorted)} solutions):"
                ]
                for i, sol in enumerate(rust_sorted):
                    obj_vals = [get_obj_value(sol, obj) for obj in objectives]
                    error_msg.append(f"  [{i}] objectives={obj_vals}, images={sorted(sol.selected_images)[:10]}...")
                
                error_msg.append("")
                error_msg.append(f"Full Python Pareto Front ({len(python_sorted)} solutions):")
                for i, sol in enumerate(python_sorted):
                    obj_vals = [get_obj_value(sol, obj) for obj in objectives]
                    error_msg.append(f"  [{i}] objectives={obj_vals}, images={sorted(sol.selected_images)[:10]}...")
                
                return False, "\n".join(error_msg)
    
    return True, ""


def run_rust_milp(problem_instance: ProblemInstance, dzn_path: Path, objectives: list[str], timeout_s: int):
    """Run Rust MILP solver via rust_milp wrapper."""
    log.info("Running Rust MILP solver (via rust_milp wrapper)")
    
    # Create temporary CSV file for output (required by interface but not used by Rust)
    with NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        summary_path = Path(tmp.name)
    
    try:
        result = rust_milp.solve(
            problem_instance=problem_instance,
            problem_path=dzn_path,
            timeout_s=timeout_s,
            summary_path=summary_path,
            front_strategy=FrontStrategy.GPBA_A,
            objectives=objectives,
            enable_trace=False,
            include_dominated=False,
        )
        return result
    except AttributeError as e:
        pytest.skip(
            f"Rust MILP solver not available. "
            f"Ensure sims-problem was built with 'milp' feature. Error: {e}"
        )
    except ValueError as e:
        if "not available" in str(e):
            pytest.skip(f"Rust MILP solver not available: {e}")
        raise
    finally:
        # Clean up temp file
        if summary_path.exists():
            summary_path.unlink()


def run_python_milp(problem_instance: ProblemInstance, dzn_path: Path, objectives: list[str], timeout_s: int):
    """Run Python MILP solver via python_milp wrapper."""
    log.info("Running Python MILP solver (via python_milp wrapper with CSV parsing)")
    
    # Create temporary CSV file for output
    with NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        summary_path = Path(tmp.name)
    
    try:
        result = python_milp.solve(
            problem_instance=problem_instance,
            problem_path=dzn_path,
            timeout_s=timeout_s,
            summary_path=summary_path,
            front_strategy=FrontStrategy.GPBA_A,
            objectives=objectives,
            enable_trace=False,
            include_dominated=False,
        )
        return result
    finally:
        # Clean up temp file
        if summary_path.exists():
            summary_path.unlink()


@pytest.mark.parametrize("instance_name", SMALL_INSTANCES[:2])  # Test first 2 small instances
@pytest.mark.parametrize(
    "objectives",
    [
        ["min_cost", "cloud_coverage"],
        ["min_cost", "min_resolution"],
    ],
)
def test_rust_python_milp_identical_results(instance_name, objectives, test_data_dir):
    """
    Test that Rust and Python MILP implementations produce identical results.
    
    This test:
    1. Loads the same problem instance
    2. Runs both Rust and Python MILP solvers via their wrappers
    3. Validates that both produce valid solutions
    4. Compares Pareto fronts for exact equivalence
    
    The wrappers handle:
    - Rust: Direct call to sims_problem.solve_with_milp()
    - Python: CSV-based call to sims_solvers.solve_milp_inlined() via _utils.run_sims_solver()
    """
    # Setup
    instance_path = Path(test_data_dir) / instance_name
    if not instance_path.exists():
        pytest.skip(f"Test data not found: {instance_path}")
    
    timeout = 120  # 2 minutes should be sufficient for small instances
    
    log.info(f"\n{'=' * 80}")
    log.info(f"Testing instance: {instance_name}")
    log.info(f"Objectives: {objectives}")
    log.info(f"Timeout: {timeout}s")
    log.info(f"{'=' * 80}\n")
    
    # Load problem instance
    problem_instance = create_problem_instance(str(instance_path))
    
    # Run Python MILP via wrapper FIRST
    import time
    python_start = time.time()
    python_result = run_python_milp(problem_instance, instance_path, objectives, timeout)
    python_time = time.time() - python_start
    log.info(f"Python MILP found {len(python_result.pareto_front)} solutions in {python_time:.2f}s")
    
    # Run Rust MILP via wrapper SECOND
    rust_start = time.time()
    rust_result = run_rust_milp(problem_instance, instance_path, objectives, timeout)
    rust_time = time.time() - rust_start
    log.info(f"Rust MILP found {len(rust_result.pareto_front)} solutions in {rust_time:.2f}s")
    
    # Print speedup comparison
    if rust_time > 0:
        speedup = python_time / rust_time
        faster = "Rust" if speedup > 1 else "Python"
        log.info(f"Performance: {faster} is {abs(speedup):.2f}x faster")
    
    # Validate Python solutions
    success, error_msg = validate_solver_result(python_result, problem_instance.problem, objectives)
    if not success:
        pytest.fail(f"Python solver result validation failed: {error_msg}")
    
    # Validate Rust solutions
    success, error_msg = validate_solver_result(rust_result, problem_instance.problem, objectives)
    if not success:
        pytest.fail(f"Rust solver result validation failed: {error_msg}")
    
    # Compare Pareto fronts
    is_equal, error_msg = compare_pareto_fronts(rust_result, python_result, objectives)
    
    if not is_equal:
        # Log detailed comparison for debugging
        log.error(f"\n{'=' * 80}")
        log.error("PARETO FRONT MISMATCH DETECTED")
        log.error(f"{'=' * 80}")
        log.error(error_msg)
        log.error(f"\nRust solutions ({len(rust_result.pareto_front)}):")
        for idx, sol in enumerate(rust_result.pareto_front[:5]):  # Show first 5
            log.error(f"  [{idx}] Images: {sorted(sol.selected_images)[:10]}...")
            log.error(f"       cost={sol.cost}, cloudy_area={sol.cloudy_area}")
        
        log.error(f"\nPython solutions ({len(python_result.pareto_front)}):")
        for idx, sol in enumerate(python_result.pareto_front[:5]):  # Show first 5
            log.error(f"  [{idx}] Images: {sorted(sol.selected_images)[:10]}...")
            log.error(f"       cost={sol.cost}, cloudy_area={sol.cloudy_area}")
        
        pytest.fail(error_msg)
    
    log.info(f"\n{'=' * 80}")
    log.info("✓ Pareto Fronts are IDENTICAL")
    log.info(f"✓ Both solvers found {len(rust_result.pareto_front)} valid Pareto optimal solutions")
    log.info("✓ All objective values match within tolerance")
    log.info("✓ All selected images match exactly")
    log.info(f"{'=' * 80}\n")


@pytest.mark.parametrize("instance_name", SMALL_INSTANCES[:1])  # Single instance for sanity check
def test_wrappers_work_independently(instance_name, test_data_dir):
    """
    Test that both wrappers work independently and produce valid results.
    
    This validates the integration layer for each wrapper.
    """
    instance_path = Path(test_data_dir) / instance_name
    if not instance_path.exists():
        pytest.skip(f"Test data not found: {instance_path}")
    
    objectives = ["min_cost", "cloud_coverage"]
    timeout = 120
    
    log.info(f"\n{'=' * 80}")
    log.info(f"Testing both wrappers independently with instance: {instance_name}")
    log.info(f"{'=' * 80}\n")
    
    # Load problem instance
    problem_instance = create_problem_instance(str(instance_path))
    
    # Test Rust wrapper
    log.info("Testing Rust MILP wrapper...")
    rust_result = run_rust_milp(problem_instance, instance_path, objectives, timeout)
    log.info(f"Rust wrapper returned {len(rust_result.pareto_front)} solutions")
    
    success, error_msg = validate_solver_result(rust_result, problem_instance.problem, objectives)
    if not success:
        pytest.fail(f"Rust wrapper validation failed: {error_msg}")
    
    assert len(rust_result.pareto_front) > 0, "Rust wrapper should return solutions"
    assert rust_result.front_strategy == FrontStrategy.GPBA_A
    log.info("✓ Rust wrapper works correctly")
    
    # Test Python wrapper
    log.info("\nTesting Python MILP wrapper...")
    python_result = run_python_milp(problem_instance, instance_path, objectives, timeout)
    log.info(f"Python wrapper returned {len(python_result.pareto_front)} solutions")
    
    success, error_msg = validate_solver_result(python_result, problem_instance.problem, objectives)
    if not success:
        pytest.fail(f"Python wrapper validation failed: {error_msg}")
    
    assert len(python_result.pareto_front) > 0, "Python wrapper should return solutions"
    assert python_result.front_strategy == FrontStrategy.GPBA_A
    log.info("✓ Python wrapper works correctly")
    
    log.info(f"\n{'=' * 80}")
    log.info("✓ Both wrappers work independently")
    log.info(f"{'=' * 80}\n")
