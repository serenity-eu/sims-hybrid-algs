"""Integration tests for two-phase solver instances using sims.core.sims.solver.solve_with_two_phases API.

Tests two-phase solver configurations (exact solver + PLS) on different instance groups
with various time allocation ratios and multiple objectives.
"""

import json
import logging
import time
from pathlib import Path

import pytest

from sims.core.sims.solver import solve_with_two_phases
from sims.core.sims.solver_config import TwoPhaseSolverConfig, SolverType, FrontStrategy
from sims.core.sims.solvers.pseudo_solver import get_pseudo_solver

from solver_test_utils import (
    SolverTestResult, SMALL_INSTANCES, MEDIUM_INSTANCES, LARGE_INSTANCES, HUGE_INSTANCES,
    TWO_PHASE_RATIOS, create_problem_instance, create_temp_output_dir,
    validate_two_phase_solver_result, log_solution_details, save_test_artifacts,
    format_ratio_string, get_timeout_for_instance_size
)

logger = logging.getLogger(__name__)

# Number of iterations to run for each test configuration
NUM_ITERATIONS = 1


# Load objective bounds from JSON file
def load_objective_bounds() -> tuple[list[str], dict[str, list[list[int]]]]:
    """Load objective bounds from tests/data/objective_bounds.json
    
    Returns:
        Tuple of (objective_names, bounds_dict)
    """
    bounds_file = Path(__file__).parent / "data" / "objective_bounds.json"
    if not bounds_file.exists():
        logger.warning(f"Objective bounds file not found: {bounds_file}")
        return [], {}
    
    with open(bounds_file, 'r') as f:
        data = json.load(f)
    
    return data.get("objectives", []), data.get("bounds", {})


# Load bounds once at module level
OBJECTIVE_NAMES, OBJECTIVE_BOUNDS = load_objective_bounds()


def get_objective_bounds_for_instance(instance_name: str, objectives: list[str]) -> list[list[int]] | None:
    """Get objective bounds for a specific instance, filtered by the specified objectives.
    
    Only returns bounds if ALL requested objectives are available in the bounds data.
    Otherwise returns None (solver will work without bounds).
    
    Args:
        instance_name: Name of the instance
        objectives: List of objective names being used (e.g., ["min_cost", "cloud_coverage"])
        
    Returns:
        List of bounds matching the specified objectives, or None if not all objectives have bounds
    """
    all_bounds = OBJECTIVE_BOUNDS.get(instance_name)
    if not all_bounds or not OBJECTIVE_NAMES:
        return None
    
    # Check if all requested objectives are in the bounds data
    filtered_bounds = []
    for obj in objectives:
        if obj not in OBJECTIVE_NAMES:
            # Objective not in bounds file - return None so solver works without bounds
            logger.info(f"Objective '{obj}' not found in bounds file, solver will run without bounds")
            return None
        
        idx = OBJECTIVE_NAMES.index(obj)
        if idx < len(all_bounds):
            filtered_bounds.append(all_bounds[idx])
        else:
            logger.warning(f"Objective {obj} index {idx} out of bounds for {instance_name}")
            return None
    
    return filtered_bounds


def run_two_phase_solver_with_validation(
    instance_path: str,
    objectives: list[str],
    ratio: tuple[int, int],
    timeout: int,
    logger: logging.Logger,
    test_type: str = "",
    artifacts_manager=None,
    test_name: str = "",
    iteration: int | None = None,
    max_solutions_count: int | None = None,
    pareto_archive: str = "nd-tree",
    use_pseudo_solver: bool = False
) -> SolverTestResult:
    """
    Run solver.solve_with_two_phases with the given configuration and validate results.
    
    Args:
        instance_path: Path to the DZN instance file
        objectives: List of objectives to optimize
        ratio: Time allocation ratio (exact_time, pls_time)
        timeout: Total solver timeout in seconds
        logger: Logger instance
        test_type: Test type identifier (2d/3d/4d)
        artifacts_manager: Manager for saving test artifacts
        test_name: Name of the test for artifact organization
        iteration: Iteration number for repeated runs
        max_solutions_count: Maximum number of solutions to generate (None = unlimited)
        pareto_archive: Pareto archive implementation ("nd-tree", "linked-list", "vector")
    
    Returns:
        SolverTestResult with execution details and success status
    """
    instance_name = Path(instance_path).stem
    ratio_str = format_ratio_string(ratio)
    
    # Get objective bounds for this instance, filtered by the objectives being used
    objective_bounds = get_objective_bounds_for_instance(instance_name, objectives)
    
    iter_str = f" (iteration {iteration})" if iteration is not None else ""
    logger.info(f"Running two-phase solver on instance {instance_name}{iter_str}")
    logger.info(f"Objectives: {objectives}")
    logger.info(f"Ratio: {ratio[0]}% exact, {ratio[1]}% PLS ({ratio_str})")
    logger.info(f"Timeout: {timeout}s")
    logger.info(f"Test type: {test_type}")
    if objective_bounds:
        logger.info(f"Using objective bounds: {objective_bounds}")
    else:
        logger.info("No objective bounds available for this instance")
    
    if use_pseudo_solver:
        logger.info("*** USING PSEUDO-SOLVER MODE ***")
    
    start_time = time.time()
    
    try:
        # Create problem instance from DZN file
        problem_instance = create_problem_instance(instance_path)
        
        # Create output directory for solver results
        output_dir = create_temp_output_dir(instance_name, f"{test_type}_{ratio_str}")
        
        # Create two-phase solver configuration
        # Note: ratio is in percentages (exact_percentage, pls_percentage) that sum to 100
        solver_config = TwoPhaseSolverConfig(
            exact_solver_type=SolverType.OR_TOOLS,
            front_strategy=FrontStrategy.GPBA_A,
            timeout_s=timeout,
            ratio=ratio
        )
        
        # Use pseudo-solver or real solver based on flag
        if use_pseudo_solver:
            pseudo_solver = get_pseudo_solver()
            result = pseudo_solver.solve_with_two_phases(
                problem_instance=problem_instance,
                problem_path=Path(instance_path),
                solver_config=solver_config,
                objectives=objectives,
                timeout_s=timeout,
                run_real_pls=True,  # Run real PLS after replaying exact solutions
                objective_bounds=objective_bounds,
                pareto_archive=pareto_archive
            )
        else:
            # Call solver.solve_with_two_phases with objective bounds
            result = solve_with_two_phases(
                problem_instance=problem_instance,
                problem_path=Path(instance_path),
                experiment_path=output_dir,
                solver_config=solver_config,
                objectives=objectives,
                dry_run=False,
                enable_pls_trace=False,  # Disabled: trace generation is O(n²) and takes hours for large solution sets
                enable_profiling_trace=True,  # Enable profiling trace for performance analysis
                objective_bounds=objective_bounds,
                max_solutions_count=max_solutions_count,
                pareto_archive=pareto_archive,
            )

        execution_time = time.time() - start_time
        logger.info(f"Two-phase execution completed successfully in {execution_time:.2f} seconds")
        
        # Validate that we got a valid result with semantic validation
        success, error_msg = validate_two_phase_solver_result(result, problem_instance.problem, objectives)
        if not success:
            return SolverTestResult(
                instance_name=instance_name,
                objectives=objectives,
                execution_time=execution_time,
                success=False,
                error_message=error_msg,
                test_type=test_type,
                ratio=ratio_str
            )
        
        # Log solution details
        log_solution_details(result, logger)
        
        # Count total solutions
        total_solutions = 0
        if result.exact_solver_result and result.exact_solver_result.pareto_front:
            total_solutions += len(result.exact_solver_result.pareto_front)
        if result.pls_result and result.pls_result.pareto_front:
            total_solutions += len(result.pls_result.pareto_front)
        
        if total_solutions == 0:
            logger.warning("No solutions found in either phase")
        
        # Save artifacts if manager is provided
        if artifacts_manager and test_name:
            save_test_artifacts(
                artifacts_manager=artifacts_manager,
                test_name=test_name,
                instance_name=instance_name,
                result=result,
                test_type=test_type,
                objectives=objectives,
                execution_time=execution_time,
                logger=logger,
                ratio=ratio_str,
                iteration=iteration
            )
        
        return SolverTestResult(
            instance_name=instance_name,
            objectives=objectives,
            execution_time=execution_time,
            success=True,
            num_solutions=total_solutions,
            test_type=test_type,
            ratio=ratio_str
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Error during two-phase solve: {str(e)}")
        logger.exception("Full traceback:")
        
        # Save failure artifacts if manager is provided
        if artifacts_manager and test_name:
            save_test_artifacts(
                artifacts_manager=artifacts_manager,
                test_name=test_name,
                instance_name=instance_name,
                result=None,
                test_type=test_type,
                objectives=objectives,
                execution_time=execution_time,
                logger=logger,
                ratio=ratio_str,
                iteration=iteration,
                test_failed=True,
                failure_exception=e
            )
        
        return SolverTestResult(
            instance_name=instance_name,
            objectives=objectives,
            execution_time=execution_time,
            success=False,
            error_message=str(e) if str(e) else f"Unknown error of type {type(e).__name__}",
            test_type=test_type,
            ratio=ratio_str
        )


class TestTwoPhaseInstances:
    """Test class for two-phase solver instance solving using sims-core solver.solve_with_two_phases API."""

    # 2D objectives tests with different ratios
    @pytest.mark.timeout(60)  # Solver timeout: 10-45s, pytest buffer included
    @pytest.mark.parametrize("pareto_archive", ["nd-tree", "linked-list", "vector"])
    @pytest.mark.parametrize("ratio", TWO_PHASE_RATIOS)
    @pytest.mark.parametrize("filename", SMALL_INSTANCES)
    def test_solve_two_phase_2d_on_small_instances(self, filename, ratio, pareto_archive, test_data_dir, mzn_model_path, artifacts_manager, caplog, use_pseudo_solver):
        """Test two-phase solver on small instances (30 images) with 2 objectives and different ratios."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            pytest.skip(f"Test instance {filename} not found at {instance_path}")
        
        # Configuration for small instances with 2 objectives
        objectives = ["min_cost", "cloud_coverage"]
        timeout = get_timeout_for_instance_size([filename])
        ratio_str = format_ratio_string(ratio)
        test_name = f"solve_two_phase_2d_small_{ratio_str}"
        
        # Run the test
        result = run_two_phase_solver_with_validation(
            instance_path=str(instance_path),
            objectives=objectives,
            ratio=ratio,
            timeout=timeout,
            logger=logger,
            test_type="2d",
            artifacts_manager=artifacts_manager,
            test_name=test_name,
            pareto_archive=pareto_archive,
            use_pseudo_solver=use_pseudo_solver
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"Two-phase solver failed for {filename} with ratio {ratio}: {result.error_message}")
        
        logger.info(f"Small 2D instance {filename} (ratio {ratio_str}) completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(4200)  # Solver timeout: 3600s (1h) + 600s (10min) buffer for size 100
    @pytest.mark.parametrize("pareto_archive", ["nd-tree", "linked-list", "vector"])
    @pytest.mark.parametrize("ratio", TWO_PHASE_RATIOS)
    @pytest.mark.parametrize("filename", MEDIUM_INSTANCES)
    def test_solve_two_phase_2d_on_medium_instances(self, filename, ratio, pareto_archive, test_data_dir, mzn_model_path, artifacts_manager, caplog, use_pseudo_solver):
        """Test two-phase solver on medium instances (50 images) with 2 objectives and different ratios."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            pytest.skip(f"Test instance {filename} not found at {instance_path}")
        
        # Configuration for medium instances with 2 objectives
        objectives = ["min_cost", "cloud_coverage"]
        timeout = get_timeout_for_instance_size([filename])
        ratio_str = format_ratio_string(ratio)
        test_name = f"solve_two_phase_2d_medium_{ratio_str}"
        
        # Run the test
        result = run_two_phase_solver_with_validation(
            instance_path=str(instance_path),
            objectives=objectives,
            ratio=ratio,
            timeout=timeout,
            logger=logger,
            test_type="2d",
            artifacts_manager=artifacts_manager,
            test_name=test_name,
            pareto_archive=pareto_archive,
            use_pseudo_solver=use_pseudo_solver
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"Two-phase solver failed for {filename} with ratio {ratio}: {result.error_message}")
        
        logger.info(f"Medium 2D instance {filename} (ratio {ratio_str}) completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(12600)  # Solver timeout: 12000s (200min) + 600s (10min) buffer for size 145-150
    @pytest.mark.parametrize("pareto_archive", ["nd-tree", "linked-list", "vector"])
    @pytest.mark.parametrize("ratio", TWO_PHASE_RATIOS)
    @pytest.mark.parametrize("filename", LARGE_INSTANCES)
    def test_solve_two_phase_2d_on_large_instances(self, filename, ratio, pareto_archive, test_data_dir, mzn_model_path, artifacts_manager, caplog, use_pseudo_solver):
        """Test two-phase solver on large instances (145-150 images) with 2 objectives and different ratios."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            pytest.skip(f"Test instance {filename} not found at {instance_path}")
        
        # Configuration for large instances with 2 objectives
        objectives = ["min_cost", "cloud_coverage"]
        timeout = get_timeout_for_instance_size([filename])
        ratio_str = format_ratio_string(ratio)
        test_name = f"solve_two_phase_2d_large_{ratio_str}"
        
        # Run the test
        result = run_two_phase_solver_with_validation(
            instance_path=str(instance_path),
            objectives=objectives,
            ratio=ratio,
            timeout=timeout,
            logger=logger,
            test_type="2d",
            artifacts_manager=artifacts_manager,
            test_name=test_name,
            pareto_archive=pareto_archive,
            use_pseudo_solver=use_pseudo_solver
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"Two-phase solver failed for {filename} with ratio {ratio}: {result.error_message}")
        
        logger.info(f"Large 2D instance {filename} (ratio {ratio_str}) completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(43800)  # Solver timeout: 43200s (12h) + 600s (10min) buffer for size 200
    @pytest.mark.parametrize("pareto_archive", ["nd-tree", "linked-list", "vector"])
    @pytest.mark.parametrize("ratio", TWO_PHASE_RATIOS)
    @pytest.mark.parametrize("filename", HUGE_INSTANCES)
    def test_solve_two_phase_2d_on_huge_instances(self, filename, ratio, pareto_archive, test_data_dir, mzn_model_path, artifacts_manager, caplog, use_pseudo_solver):
        """Test two-phase solver on huge instances (200 images) with 2 objectives and different ratios."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            pytest.skip(f"Test instance {filename} not found at {instance_path}")
        
        # Configuration for huge instances with 2 objectives
        objectives = ["min_cost", "cloud_coverage"]
        timeout = get_timeout_for_instance_size([filename])
        ratio_str = format_ratio_string(ratio)
        test_name = f"solve_two_phase_2d_huge_{ratio_str}"
        
        # Run the test
        result = run_two_phase_solver_with_validation(
            instance_path=str(instance_path),
            objectives=objectives,
            ratio=ratio,
            timeout=timeout,
            logger=logger,
            test_type="2d",
            artifacts_manager=artifacts_manager,
            test_name=test_name,
            pareto_archive=pareto_archive,
            use_pseudo_solver=use_pseudo_solver
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"Two-phase solver failed for {filename} with ratio {ratio}: {result.error_message}")
        
        logger.info(f"Huge 2D instance {filename} (ratio {ratio_str}) completed successfully in {result.execution_time:.2f}s")

    # 3D objectives tests with different ratios
    @pytest.mark.timeout(60)  # Solver timeout: 10-45s, pytest buffer included
    @pytest.mark.parametrize("pareto_archive", ["nd-tree", "linked-list", "vector"])
    @pytest.mark.parametrize("ratio", TWO_PHASE_RATIOS)
    @pytest.mark.parametrize("filename", SMALL_INSTANCES)
    def test_solve_two_phase_3d_on_small_instances(self, filename, ratio, pareto_archive, test_data_dir, mzn_model_path, artifacts_manager, caplog, use_pseudo_solver):
        """Test two-phase solver on small instances (30 images) with 3 objectives and different ratios."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            pytest.skip(f"Test instance {filename} not found at {instance_path}")
        
        # Configuration for small instances with 3 objectives
        objectives = ["min_cost", "cloud_coverage", "min_max_incidence_angle"]
        timeout = get_timeout_for_instance_size([filename])
        ratio_str = format_ratio_string(ratio)
        test_name = f"solve_two_phase_3d_small_{ratio_str}"
        
        # Run the test
        result = run_two_phase_solver_with_validation(
            instance_path=str(instance_path),
            objectives=objectives,
            ratio=ratio,
            timeout=timeout,
            logger=logger,
            test_type="3d",
            artifacts_manager=artifacts_manager,
            test_name=test_name,
            pareto_archive=pareto_archive,
            use_pseudo_solver=use_pseudo_solver
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"Two-phase solver failed for {filename} with ratio {ratio}: {result.error_message}")
        
        logger.info(f"Small 3D instance {filename} (ratio {ratio_str}) completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(4200)  # Solver timeout: 3600s (1h) + 600s (10min) buffer for size 100
    @pytest.mark.parametrize("pareto_archive", ["nd-tree", "linked-list", "vector"])
    @pytest.mark.parametrize("ratio", TWO_PHASE_RATIOS)
    @pytest.mark.parametrize("filename", MEDIUM_INSTANCES)
    def test_solve_two_phase_3d_on_medium_instances(self, filename, ratio, pareto_archive, test_data_dir, mzn_model_path, artifacts_manager, caplog, use_pseudo_solver):
        """Test two-phase solver on medium instances (50 images) with 3 objectives and different ratios."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            pytest.skip(f"Test instance {filename} not found at {instance_path}")
        
        # Configuration for medium instances with 3 objectives
        objectives = ["min_cost", "cloud_coverage", "min_max_incidence_angle"]
        timeout = get_timeout_for_instance_size([filename])
        ratio_str = format_ratio_string(ratio)
        test_name = f"solve_two_phase_3d_medium_{ratio_str}"
        
        # Run the test
        result = run_two_phase_solver_with_validation(
            instance_path=str(instance_path),
            objectives=objectives,
            ratio=ratio,
            timeout=timeout,
            logger=logger,
            test_type="3d",
            artifacts_manager=artifacts_manager,
            test_name=test_name,
            pareto_archive=pareto_archive,
            use_pseudo_solver=use_pseudo_solver
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"Two-phase solver failed for {filename} with ratio {ratio}: {result.error_message}")
        
        logger.info(f"Medium 3D instance {filename} (ratio {ratio_str}) completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(12600)  # Solver timeout: 12000s (200min) + 600s (10min) buffer for size 145-150
    @pytest.mark.parametrize("pareto_archive", ["nd-tree", "linked-list", "vector"])
    @pytest.mark.parametrize("ratio", TWO_PHASE_RATIOS)
    @pytest.mark.parametrize("filename", LARGE_INSTANCES)
    def test_solve_two_phase_3d_on_large_instances(self, filename, ratio, pareto_archive, test_data_dir, mzn_model_path, artifacts_manager, caplog, use_pseudo_solver):
        """Test two-phase solver on large instances (145-150 images) with 3 objectives and different ratios."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            pytest.skip(f"Test instance {filename} not found at {instance_path}")
        
        # Configuration for large instances with 3 objectives
        objectives = ["min_cost", "cloud_coverage", "min_max_incidence_angle"]
        timeout = get_timeout_for_instance_size([filename])
        ratio_str = format_ratio_string(ratio)
        test_name = f"solve_two_phase_3d_large_{ratio_str}"
        
        # Run the test
        result = run_two_phase_solver_with_validation(
            instance_path=str(instance_path),
            objectives=objectives,
            ratio=ratio,
            timeout=timeout,
            logger=logger,
            test_type="3d",
            artifacts_manager=artifacts_manager,
            test_name=test_name,
            pareto_archive=pareto_archive,
            use_pseudo_solver=use_pseudo_solver
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"Two-phase solver failed for {filename} with ratio {ratio}: {result.error_message}")
        
        logger.info(f"Large 3D instance {filename} (ratio {ratio_str}) completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(43800)  # Solver timeout: 43200s (12h) + 600s (10min) buffer for size 200
    @pytest.mark.parametrize("pareto_archive", ["nd-tree", "linked-list", "vector"])
    @pytest.mark.parametrize("ratio", TWO_PHASE_RATIOS)
    @pytest.mark.parametrize("filename", HUGE_INSTANCES)
    def test_solve_two_phase_3d_on_huge_instances(self, filename, ratio, pareto_archive, test_data_dir, mzn_model_path, artifacts_manager, caplog, use_pseudo_solver):
        """Test two-phase solver on huge instances (200 images) with 3 objectives and different ratios."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            pytest.skip(f"Test instance {filename} not found at {instance_path}")
        
        # Configuration for huge instances with 3 objectives
        objectives = ["min_cost", "cloud_coverage", "min_max_incidence_angle"]
        timeout = get_timeout_for_instance_size([filename])
        ratio_str = format_ratio_string(ratio)
        test_name = f"solve_two_phase_3d_huge_{ratio_str}"
        
        # Run the test
        result = run_two_phase_solver_with_validation(
            instance_path=str(instance_path),
            objectives=objectives,
            ratio=ratio,
            timeout=timeout,
            logger=logger,
            test_type="3d",
            artifacts_manager=artifacts_manager,
            test_name=test_name,
            pareto_archive=pareto_archive,
            use_pseudo_solver=use_pseudo_solver
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"Two-phase solver failed for {filename} with ratio {ratio}: {result.error_message}")
        
        logger.info(f"Huge 3D instance {filename} (ratio {ratio_str}) completed successfully in {result.execution_time:.2f}s")

    # 4D objectives tests with different ratios
    @pytest.mark.timeout(360)  # Solver timeout: 10-45s, pytest buffer included
    @pytest.mark.parametrize("pareto_archive", ["nd-tree", "linked-list", "vector"])
    @pytest.mark.parametrize("iteration", range(NUM_ITERATIONS))
    @pytest.mark.parametrize("filename", SMALL_INSTANCES)
    @pytest.mark.parametrize("ratio", TWO_PHASE_RATIOS)
    def test_solve_two_phase_4d_on_small_instances(self, filename, ratio, iteration, pareto_archive, test_data_dir, mzn_model_path, artifacts_manager, caplog, use_pseudo_solver):
        """Test two-phase solver on small instances (30 images) with 4 objectives and different ratios."""
        caplog.set_level(logging.DEBUG)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            pytest.skip(f"Test instance {filename} not found at {instance_path}")
        
        # Configuration for small instances with 4 objectives
        objectives = ["min_cost", "cloud_coverage", "min_max_incidence_angle", "min_resolution"]
        timeout = get_timeout_for_instance_size([filename])
        ratio_str = format_ratio_string(ratio)
        test_name = f"solve_two_phase_4d_small_{pareto_archive}"
        
        logger.info(f"Running iteration {iteration + 1}/{NUM_ITERATIONS} for {filename} with ratio {ratio_str}")
        
        # Run the test
        result = run_two_phase_solver_with_validation(
            instance_path=str(instance_path),
            objectives=objectives,
            ratio=ratio,
            timeout=timeout,
            logger=logger,
            test_type="4d",
            artifacts_manager=artifacts_manager,
            test_name=test_name,
            iteration=iteration,
            pareto_archive=pareto_archive,
            use_pseudo_solver=use_pseudo_solver
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"Two-phase solver failed for {filename} with ratio {ratio} (iteration {iteration + 1}): {result.error_message}")
        
        logger.info(f"Small 4D instance {filename} (ratio {ratio_str}, iteration {iteration + 1}/{NUM_ITERATIONS}) completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(4200)  # Solver timeout: 3600s (1h) + 600s (10min) buffer for size 100
    @pytest.mark.parametrize("pareto_archive", ["nd-tree", "linked-list", "vector"])
    @pytest.mark.parametrize("iteration", range(NUM_ITERATIONS))
    @pytest.mark.parametrize("filename", MEDIUM_INSTANCES)
    @pytest.mark.parametrize("ratio", TWO_PHASE_RATIOS)
    def test_solve_two_phase_4d_on_medium_instances(self, filename, ratio, iteration, pareto_archive, test_data_dir, mzn_model_path, artifacts_manager, caplog, use_pseudo_solver):
        """Test two-phase solver on medium instances (100 images) with 4 objectives and different ratios."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            pytest.skip(f"Test instance {filename} not found at {instance_path}")
        
        # Configuration for medium instances with 4 objectives
        objectives = ["min_cost", "cloud_coverage", "min_max_incidence_angle", "min_resolution"]
        timeout = get_timeout_for_instance_size([filename])
        ratio_str = format_ratio_string(ratio)
        test_name = f"solve_two_phase_4d_medium_{pareto_archive}"
        
        logger.info(f"Running iteration {iteration + 1}/{NUM_ITERATIONS} for {filename} with ratio {ratio_str}")
        
        # Run the test
        result = run_two_phase_solver_with_validation(
            instance_path=str(instance_path),
            objectives=objectives,
            ratio=ratio,
            timeout=timeout,
            logger=logger,
            test_type="4d",
            artifacts_manager=artifacts_manager,
            test_name=test_name,
            iteration=iteration,
            pareto_archive=pareto_archive,
            use_pseudo_solver=use_pseudo_solver
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"Two-phase solver failed for {filename} with ratio {ratio} (iteration {iteration + 1}): {result.error_message}")
        
        logger.info(f"Medium 4D instance {filename} (ratio {ratio_str}, iteration {iteration + 1}/{NUM_ITERATIONS}) completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(12600)  # Solver timeout: 12000s (200min) + 600s (10min) buffer for size 145-150
    @pytest.mark.parametrize("pareto_archive", ["nd-tree", "linked-list", "vector"])
    @pytest.mark.parametrize("iteration", range(NUM_ITERATIONS))
    @pytest.mark.parametrize("filename", LARGE_INSTANCES)
    @pytest.mark.parametrize("ratio", TWO_PHASE_RATIOS)
    def test_solve_two_phase_4d_on_large_instances(self, filename, ratio, iteration, pareto_archive, test_data_dir, mzn_model_path, artifacts_manager, caplog, use_pseudo_solver):
        """Test two-phase solver on large instances (145-150 images) with 4 objectives and different ratios."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            pytest.skip(f"Test instance {filename} not found at {instance_path}")
        
        # Configuration for large instances with 4 objectives
        objectives = ["min_cost", "cloud_coverage", "min_max_incidence_angle", "min_resolution"]
        timeout = get_timeout_for_instance_size([filename])
        ratio_str = format_ratio_string(ratio)
        test_name = f"solve_two_phase_4d_large_{pareto_archive}"
        
        logger.info(f"Running iteration {iteration + 1}/{NUM_ITERATIONS} for {filename} with ratio {ratio_str}")
        
        # Run the test
        result = run_two_phase_solver_with_validation(
            instance_path=str(instance_path),
            objectives=objectives,
            ratio=ratio,
            timeout=timeout,
            logger=logger,
            test_type="4d",
            artifacts_manager=artifacts_manager,
            test_name=test_name,
            iteration=iteration,
            max_solutions_count=10,  # Limit to 10 solutions for large instances
            pareto_archive=pareto_archive,
            use_pseudo_solver=use_pseudo_solver
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"Two-phase solver failed for {filename} with ratio {ratio} (iteration {iteration + 1}): {result.error_message}")
        
        logger.info(f"Large 4D instance {filename} (ratio {ratio_str}, iteration {iteration + 1}/{NUM_ITERATIONS}) completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(43800)  # Solver timeout: 43200s (12h) + 600s (10min) buffer for size 200
    @pytest.mark.parametrize("pareto_archive", ["nd-tree", "linked-list", "vector"])
    @pytest.mark.parametrize("iteration", range(NUM_ITERATIONS))
    @pytest.mark.parametrize("filename", HUGE_INSTANCES)
    @pytest.mark.parametrize("ratio", TWO_PHASE_RATIOS)
    def test_solve_two_phase_4d_on_huge_instances(self, filename, ratio, iteration, pareto_archive, test_data_dir, mzn_model_path, artifacts_manager, caplog, use_pseudo_solver):
        """Test two-phase solver on huge instances (200 images) with 4 objectives and different ratios."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            pytest.skip(f"Test instance {filename} not found at {instance_path}")
        
        # Configuration for huge instances with 4 objectives
        objectives = ["min_cost", "cloud_coverage", "min_max_incidence_angle", "min_resolution"]
        timeout = get_timeout_for_instance_size([filename])
        ratio_str = format_ratio_string(ratio)
        test_name = f"solve_two_phase_4d_huge_{pareto_archive}"
        
        logger.info(f"Running iteration {iteration + 1}/{NUM_ITERATIONS} for {filename} with ratio {ratio_str}")
        
        # Run the test
        result = run_two_phase_solver_with_validation(
            instance_path=str(instance_path),
            objectives=objectives,
            ratio=ratio,
            timeout=timeout,
            logger=logger,
            test_type="4d",
            artifacts_manager=artifacts_manager,
            test_name=test_name,
            iteration=iteration,
            max_solutions_count=10,  # Limit to 10 solutions for huge instances
            pareto_archive=pareto_archive,
            use_pseudo_solver=use_pseudo_solver
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"Two-phase solver failed for {filename} with ratio {ratio} (iteration {iteration + 1}): {result.error_message}")
        
        logger.info(f"Huge 4D instance {filename} (ratio {ratio_str}, iteration {iteration + 1}/{NUM_ITERATIONS}) completed successfully in {result.execution_time:.2f}s")