"""Integration tests for Rust MILP solver using rust_milp.solve() API.

Tests multiple configurations on different instance groups with 2d, 3d, and 4d 
objectives using the direct Rust MILP solver implementation.
"""

import logging
import time
from pathlib import Path
from typing import List

import pytest

from sims.core.sims.solvers import rust_milp
from sims.core.sims.solver_config import FrontStrategy

from solver_test_utils import (
    SolverTestResult, SMALL_INSTANCES, MEDIUM_INSTANCES, LARGE_INSTANCES,
    create_problem_instance, create_temp_output_dir, validate_solver_result,
    log_solution_details, save_test_artifacts, get_timeout_for_instance_size
)

logger = logging.getLogger(__name__)


def run_rust_milp_with_validation(
    instance_path: str,
    mzn_model_path: str,
    objectives: List[str],
    timeout: int,
    logger: logging.Logger,
    test_type: str = "",
    artifacts_manager = None,
    test_name: str = ""
) -> SolverTestResult:
    """
    Run rust_milp.solve with the given configuration and validate results.
    
    Args:
        instance_path: Path to the DZN instance file
        mzn_model_path: Path to the MZN model file (unused but kept for compatibility)
        objectives: List of objectives to optimize
        timeout: Solver timeout in seconds
        logger: Logger instance
        test_type: Test type identifier (2d/3d/4d)
        artifacts_manager: Manager for saving test artifacts
        test_name: Name of the test for artifact organization
    
    Returns:
        SolverTestResult with execution details and success status
    """
    instance_name = Path(instance_path).stem
    logger.info(f"Running rust_milp.solve on instance {instance_name}")
    logger.info(f"Objectives: {objectives}")
    logger.info(f"Timeout: {timeout}s")
    logger.info(f"Test type: {test_type}")
    
    start_time = time.time()
    
    try:
        # Create problem instance from DZN file using utility function
        problem_instance = create_problem_instance(instance_path)
        
        # Create output directory using utility function
        output_dir = create_temp_output_dir(instance_name, test_type)
        summary_path = output_dir / "summary.csv"
        
        # Call rust_milp.solve with correct parameters
        result = rust_milp.solve(
            problem_instance=problem_instance,
            problem_path=Path(instance_path),
            timeout_s=timeout,
            summary_path=summary_path,
            front_strategy=FrontStrategy.GPBA_A,
            objectives=objectives,
            enable_trace=False,
            include_dominated=False
        )
        
        execution_time = time.time() - start_time
        logger.info(f"Execution completed successfully in {execution_time:.2f} seconds")
        
        # Validate result using utility function with semantic validation
        success, error_msg = validate_solver_result(result, problem_instance.problem, objectives)
        if not success:
            return SolverTestResult(
                instance_name=instance_name,
                objectives=objectives,
                execution_time=execution_time,
                success=False,
                error_message=error_msg,
                test_type=test_type
            )
        
        # Log solution details using utility function
        log_solution_details(result, logger)
        
        # Count solutions
        num_solutions = len(result.pareto_front) if result.pareto_front else 0
        
        # Save artifacts using utility function
        if artifacts_manager and test_name:
            save_test_artifacts(
                artifacts_manager=artifacts_manager,
                test_name=test_name,
                instance_name=instance_name,
                result=result,
                test_type=test_type,
                objectives=objectives,
                execution_time=execution_time,
                logger=logger
            )
        
        return SolverTestResult(
            instance_name=instance_name,
            objectives=objectives,
            execution_time=execution_time,
            success=True,
            num_solutions=num_solutions,
            test_type=test_type
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Error during rust_milp.solve: {str(e)}")
        logger.exception("Full traceback:")
        return SolverTestResult(
            instance_name=instance_name,
            objectives=objectives,
            execution_time=execution_time,
            success=False,
            error_message=str(e) if str(e) else f"Unknown error of type {type(e).__name__}",
            test_type=test_type
        )


class TestRustMilpInstances:
    """Test class for Rust MILP instance solving using rust_milp.solve() API."""

    # 2D objectives tests
    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", SMALL_INSTANCES)
    def test_rust_milp_2d_on_small_instances(self, filename, test_data_dir, mzn_model_path, artifacts_manager, caplog):
        """Test rust_milp.solve on small instances (30 images) with 2 objectives."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            pytest.skip(f"Test instance {filename} not found at {instance_path}")
        
        # Configuration for small instances with 2 objectives
        objectives = ["min_cost", "cloud_coverage"]
        timeout = get_timeout_for_instance_size([filename])
        test_name = "rust_milp_2d_small"
        
        # Run the test
        result = run_rust_milp_with_validation(
            instance_path=str(instance_path),
            mzn_model_path=mzn_model_path,
            objectives=objectives,
            timeout=timeout,
            logger=logger,
            test_type="2d",
            artifacts_manager=artifacts_manager,
            test_name=test_name
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"rust_milp.solve failed for {filename}: {result.error_message}")
        
        logger.info(f"Small 2D instance {filename} completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", MEDIUM_INSTANCES)
    def test_rust_milp_2d_on_medium_instances(self, filename, test_data_dir, mzn_model_path, artifacts_manager, caplog):
        """Test rust_milp.solve on medium instances (50 images) with 2 objectives."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            pytest.skip(f"Test instance {filename} not found at {instance_path}")
        
        # Configuration for medium instances with 2 objectives
        objectives = ["min_cost", "cloud_coverage"]
        timeout = get_timeout_for_instance_size([filename])
        test_name = "rust_milp_2d_medium"
        
        # Run the test
        result = run_rust_milp_with_validation(
            instance_path=str(instance_path),
            mzn_model_path=mzn_model_path,
            objectives=objectives,
            timeout=timeout,
            logger=logger,
            test_type="2d",
            artifacts_manager=artifacts_manager,
            test_name=test_name
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"rust_milp.solve failed for {filename}: {result.error_message}")
        
        logger.info(f"Medium 2D instance {filename} completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", LARGE_INSTANCES)
    def test_rust_milp_2d_on_large_instances(self, filename, test_data_dir, mzn_model_path, artifacts_manager, caplog):
        """Test rust_milp.solve on large instances (100+ images) with 2 objectives."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            pytest.skip(f"Test instance {filename} not found at {instance_path}")
        
        # Configuration for large instances with 2 objectives
        objectives = ["min_cost", "cloud_coverage"]
        timeout = get_timeout_for_instance_size([filename])
        test_name = "rust_milp_2d_large"
        
        # Run the test
        result = run_rust_milp_with_validation(
            instance_path=str(instance_path),
            mzn_model_path=mzn_model_path,
            objectives=objectives,
            timeout=timeout,
            logger=logger,
            test_type="2d",
            artifacts_manager=artifacts_manager,
            test_name=test_name
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"rust_milp.solve failed for {filename}: {result.error_message}")
        
        logger.info(f"Large 2D instance {filename} completed successfully in {result.execution_time:.2f}s")

    # 3D objectives tests
    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", SMALL_INSTANCES)
    def test_rust_milp_3d_on_small_instances(self, filename, test_data_dir, mzn_model_path, artifacts_manager, caplog):
        """Test rust_milp.solve on small instances (30 images) with 3 objectives."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            pytest.skip(f"Test instance {filename} not found at {instance_path}")
        
        # Configuration for small instances with 3 objectives
        objectives = ["min_cost", "cloud_coverage", "min_resolution"]
        timeout = get_timeout_for_instance_size([filename])
        test_name = "rust_milp_3d_small"
        
        # Run the test
        result = run_rust_milp_with_validation(
            instance_path=str(instance_path),
            mzn_model_path=mzn_model_path,
            objectives=objectives,
            timeout=timeout,
            logger=logger,
            test_type="3d",
            artifacts_manager=artifacts_manager,
            test_name=test_name
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"rust_milp.solve failed for {filename}: {result.error_message}")
        
        logger.info(f"Small 3D instance {filename} completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", MEDIUM_INSTANCES)
    def test_rust_milp_3d_on_medium_instances(self, filename, test_data_dir, mzn_model_path, artifacts_manager, caplog):
        """Test rust_milp.solve on medium instances (50 images) with 3 objectives."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            pytest.skip(f"Test instance {filename} not found at {instance_path}")
        
        # Configuration for medium instances with 3 objectives
        objectives = ["min_cost", "cloud_coverage", "min_resolution"]
        timeout = get_timeout_for_instance_size([filename])
        test_name = "rust_milp_3d_medium"
        
        # Run the test
        result = run_rust_milp_with_validation(
            instance_path=str(instance_path),
            mzn_model_path=mzn_model_path,
            objectives=objectives,
            timeout=timeout,
            logger=logger,
            test_type="3d",
            artifacts_manager=artifacts_manager,
            test_name=test_name
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"rust_milp.solve failed for {filename}: {result.error_message}")
        
        logger.info(f"Medium 3D instance {filename} completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", LARGE_INSTANCES)
    def test_rust_milp_3d_on_large_instances(self, filename, test_data_dir, mzn_model_path, artifacts_manager, caplog):
        """Test rust_milp.solve on large instances (100+ images) with 3 objectives."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            pytest.skip(f"Test instance {filename} not found at {instance_path}")
        
        # Configuration for large instances with 3 objectives
        objectives = ["min_cost", "cloud_coverage", "min_resolution"]
        timeout = get_timeout_for_instance_size([filename])
        test_name = "rust_milp_3d_large"
        
        # Run the test
        result = run_rust_milp_with_validation(
            instance_path=str(instance_path),
            mzn_model_path=mzn_model_path,
            objectives=objectives,
            timeout=timeout,
            logger=logger,
            test_type="3d",
            artifacts_manager=artifacts_manager,
            test_name=test_name
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"rust_milp.solve failed for {filename}: {result.error_message}")
        
        logger.info(f"Large 3D instance {filename} completed successfully in {result.execution_time:.2f}s")

    # 4D objectives tests
    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", SMALL_INSTANCES)
    def test_rust_milp_4d_on_small_instances(self, filename, test_data_dir, mzn_model_path, artifacts_manager, caplog):
        """Test rust_milp.solve on small instances (30 images) with 4 objectives."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            pytest.skip(f"Test instance {filename} not found at {instance_path}")
        
        # Configuration for small instances with 4 objectives
        objectives = ["min_cost", "cloud_coverage", "min_resolution", "min_max_incidence_angle"]
        timeout = get_timeout_for_instance_size([filename])
        test_name = "rust_milp_4d_small"
        
        # Run the test
        result = run_rust_milp_with_validation(
            instance_path=str(instance_path),
            mzn_model_path=mzn_model_path,
            objectives=objectives,
            timeout=timeout,
            logger=logger,
            test_type="4d",
            artifacts_manager=artifacts_manager,
            test_name=test_name
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"rust_milp.solve failed for {filename}: {result.error_message}")
        
        logger.info(f"Small 4D instance {filename} completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", MEDIUM_INSTANCES)
    def test_rust_milp_4d_on_medium_instances(self, filename, test_data_dir, mzn_model_path, artifacts_manager, caplog):
        """Test rust_milp.solve on medium instances (50 images) with 4 objectives."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            pytest.skip(f"Test instance {filename} not found at {instance_path}")
        
        # Configuration for medium instances with 4 objectives
        objectives = ["min_cost", "cloud_coverage", "min_resolution", "min_max_incidence_angle"]
        timeout = get_timeout_for_instance_size([filename])
        test_name = "rust_milp_4d_medium"
        
        # Run the test
        result = run_rust_milp_with_validation(
            instance_path=str(instance_path),
            mzn_model_path=mzn_model_path,
            objectives=objectives,
            timeout=timeout,
            logger=logger,
            test_type="4d",
            artifacts_manager=artifacts_manager,
            test_name=test_name
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"rust_milp.solve failed for {filename}: {result.error_message}")
        
        logger.info(f"Medium 4D instance {filename} completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", LARGE_INSTANCES)
    def test_rust_milp_4d_on_large_instances(self, filename, test_data_dir, mzn_model_path, artifacts_manager, caplog):
        """Test rust_milp.solve on large instances (100+ images) with 4 objectives."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        instance_path = Path(test_data_dir) / filename
        if not instance_path.exists():
            pytest.skip(f"Test instance {filename} not found at {instance_path}")
        
        # Configuration for large instances with 4 objectives
        objectives = ["min_cost", "cloud_coverage", "min_resolution", "min_max_incidence_angle"]
        timeout = get_timeout_for_instance_size([filename])
        test_name = "rust_milp_4d_large"
        
        # Run the test
        result = run_rust_milp_with_validation(
            instance_path=str(instance_path),
            mzn_model_path=mzn_model_path,
            objectives=objectives,
            timeout=timeout,
            logger=logger,
            test_type="4d",
            artifacts_manager=artifacts_manager,
            test_name=test_name
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"rust_milp.solve failed for {filename}: {result.error_message}")
        
        logger.info(f"Large 4D instance {filename} completed successfully in {result.execution_time:.2f}s")
