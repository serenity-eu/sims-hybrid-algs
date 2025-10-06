"""Integration tests for two-phase solver instances using sims.core.sims.solver.solve_with_two_phases API.

Tests two-phase solver configurations (exact solver + PLS) on different instance groups 
with various time allocation ratios and multiple objectives.
"""

import logging
import time
from pathlib import Path

import pytest

from sims.core.sims.solver import solve_with_two_phases
from sims.core.sims.solver_config import TwoPhaseSolverConfig, SolverType, FrontStrategy
from sims.core.sims.solver_result import TwoPhaseSolverResult

from solver_test_utils import (
    SolverTestResult, SMALL_INSTANCES, MEDIUM_INSTANCES, LARGE_INSTANCES,
    TWO_PHASE_RATIOS, create_problem_instance, create_temp_output_dir,
    validate_two_phase_solver_result, log_solution_details, save_test_artifacts,
    format_ratio_string, get_timeout_for_instance_size
)

logger = logging.getLogger(__name__)


def run_two_phase_solver_with_validation(
    instance_path: str,
    objectives: list[str],
    ratio: tuple[int, int],
    timeout: int,
    logger: logging.Logger,
    test_type: str = "",
    artifacts_manager=None,
    test_name: str = ""
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
    
    Returns:
        SolverTestResult with execution details and success status
    """
    instance_name = Path(instance_path).stem
    ratio_str = format_ratio_string(ratio)
    
    logger.info(f"Running two-phase solver on instance {instance_name}")
    logger.info(f"Objectives: {objectives}")
    logger.info(f"Ratio: {ratio[0]}% exact, {ratio[1]}% PLS ({ratio_str})")
    logger.info(f"Timeout: {timeout}s")
    logger.info(f"Test type: {test_type}")
    
    start_time = time.time()
    
    try:
        # Create problem instance from DZN file
        problem_instance = create_problem_instance(instance_path)
        
        # Create output directory for solver results
        output_dir = create_temp_output_dir(instance_name, f"{test_type}_{ratio_str}")
        
        # Create two-phase solver configuration
        # Note: ratio is in percentages (exact_percentage, pls_percentage) that sum to 100
        solver_config = TwoPhaseSolverConfig(
            exact_solver_type=SolverType.GUROBI,
            front_strategy=FrontStrategy.GPBA_A,
            timeout_s=timeout,
            ratio=ratio
        )
        
        # Call solver.solve_with_two_phases
        result = solve_with_two_phases(
            problem_instance=problem_instance,
            problem_path=Path(instance_path),
            experiment_path=output_dir,
            solver_config=solver_config,
            objectives=objectives,
            dry_run=False,
            enable_pls_trace=True
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
                ratio=ratio_str
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
    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", SMALL_INSTANCES)
    @pytest.mark.parametrize("ratio", TWO_PHASE_RATIOS)
    def test_solve_two_phase_2d_on_small_instances(self, filename, ratio, test_data_dir, mzn_model_path, artifacts_manager, caplog):
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
            test_name=test_name
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"Two-phase solver failed for {filename} with ratio {ratio}: {result.error_message}")
        
        logger.info(f"Small 2D instance {filename} (ratio {ratio_str}) completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", MEDIUM_INSTANCES)
    @pytest.mark.parametrize("ratio", TWO_PHASE_RATIOS)
    def test_solve_two_phase_2d_on_medium_instances(self, filename, ratio, test_data_dir, mzn_model_path, artifacts_manager, caplog):
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
            test_name=test_name
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"Two-phase solver failed for {filename} with ratio {ratio}: {result.error_message}")
        
        logger.info(f"Medium 2D instance {filename} (ratio {ratio_str}) completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", LARGE_INSTANCES)
    @pytest.mark.parametrize("ratio", TWO_PHASE_RATIOS)
    def test_solve_two_phase_2d_on_large_instances(self, filename, ratio, test_data_dir, mzn_model_path, artifacts_manager, caplog):
        """Test two-phase solver on large instances (100+ images) with 2 objectives and different ratios."""
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
            test_name=test_name
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"Two-phase solver failed for {filename} with ratio {ratio}: {result.error_message}")
        
        logger.info(f"Large 2D instance {filename} (ratio {ratio_str}) completed successfully in {result.execution_time:.2f}s")

    # 3D objectives tests with different ratios
    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", SMALL_INSTANCES)
    @pytest.mark.parametrize("ratio", TWO_PHASE_RATIOS)
    def test_solve_two_phase_3d_on_small_instances(self, filename, ratio, test_data_dir, mzn_model_path, artifacts_manager, caplog):
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
            test_name=test_name
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"Two-phase solver failed for {filename} with ratio {ratio}: {result.error_message}")
        
        logger.info(f"Small 3D instance {filename} (ratio {ratio_str}) completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", MEDIUM_INSTANCES)
    @pytest.mark.parametrize("ratio", TWO_PHASE_RATIOS)
    def test_solve_two_phase_3d_on_medium_instances(self, filename, ratio, test_data_dir, mzn_model_path, artifacts_manager, caplog):
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
            test_name=test_name
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"Two-phase solver failed for {filename} with ratio {ratio}: {result.error_message}")
        
        logger.info(f"Medium 3D instance {filename} (ratio {ratio_str}) completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", LARGE_INSTANCES)
    @pytest.mark.parametrize("ratio", TWO_PHASE_RATIOS)
    def test_solve_two_phase_3d_on_large_instances(self, filename, ratio, test_data_dir, mzn_model_path, artifacts_manager, caplog):
        """Test two-phase solver on large instances (100+ images) with 3 objectives and different ratios."""
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
            test_name=test_name
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"Two-phase solver failed for {filename} with ratio {ratio}: {result.error_message}")
        
        logger.info(f"Large 3D instance {filename} (ratio {ratio_str}) completed successfully in {result.execution_time:.2f}s")

    # 4D objectives tests with different ratios
    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", SMALL_INSTANCES)
    @pytest.mark.parametrize("ratio", TWO_PHASE_RATIOS)
    def test_solve_two_phase_4d_on_small_instances(self, filename, ratio, test_data_dir, mzn_model_path, artifacts_manager, caplog):
        """Test two-phase solver on small instances (30 images) with 4 objectives and different ratios."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            pytest.skip(f"Test instance {filename} not found at {instance_path}")
        
        # Configuration for small instances with 4 objectives
        objectives = ["min_cost", "cloud_coverage", "min_max_incidence_angle", "min_resolution"]
        timeout = get_timeout_for_instance_size([filename])
        ratio_str = format_ratio_string(ratio)
        test_name = f"solve_two_phase_4d_small_{ratio_str}"
        
        # Run the test
        result = run_two_phase_solver_with_validation(
            instance_path=str(instance_path),
            objectives=objectives,
            ratio=ratio,
            timeout=timeout,
            logger=logger,
            test_type="4d",
            artifacts_manager=artifacts_manager,
            test_name=test_name
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"Two-phase solver failed for {filename} with ratio {ratio}: {result.error_message}")
        
        logger.info(f"Small 4D instance {filename} (ratio {ratio_str}) completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", MEDIUM_INSTANCES)
    @pytest.mark.parametrize("ratio", TWO_PHASE_RATIOS)
    def test_solve_two_phase_4d_on_medium_instances(self, filename, ratio, test_data_dir, mzn_model_path, artifacts_manager, caplog):
        """Test two-phase solver on medium instances (50 images) with 4 objectives and different ratios."""
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
        test_name = f"solve_two_phase_4d_medium_{ratio_str}"
        
        # Run the test
        result = run_two_phase_solver_with_validation(
            instance_path=str(instance_path),
            objectives=objectives,
            ratio=ratio,
            timeout=timeout,
            logger=logger,
            test_type="4d",
            artifacts_manager=artifacts_manager,
            test_name=test_name
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"Two-phase solver failed for {filename} with ratio {ratio}: {result.error_message}")
        
        logger.info(f"Medium 4D instance {filename} (ratio {ratio_str}) completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", LARGE_INSTANCES)
    @pytest.mark.parametrize("ratio", TWO_PHASE_RATIOS)
    def test_solve_two_phase_4d_on_large_instances(self, filename, ratio, test_data_dir, mzn_model_path, artifacts_manager, caplog):
        """Test two-phase solver on large instances (100+ images) with 4 objectives and different ratios."""
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
        test_name = f"solve_two_phase_4d_large_{ratio_str}"
        
        # Run the test
        result = run_two_phase_solver_with_validation(
            instance_path=str(instance_path),
            objectives=objectives,
            ratio=ratio,
            timeout=timeout,
            logger=logger,
            test_type="4d",
            artifacts_manager=artifacts_manager,
            test_name=test_name
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"Two-phase solver failed for {filename} with ratio {ratio}: {result.error_message}")
        
        logger.info(f"Large 4D instance {filename} (ratio {ratio_str}) completed successfully in {result.execution_time:.2f}s")