"""Integration tests for solve_milp functionality using real data instances.

Tests multiple solver configurations on different instance groups similar to
the MILP real data integration tests.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

import pytest
from dataclasses import dataclass

from sims_solvers import solve
from sims_solvers.Config import Config


logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a solve_milp test execution."""
    instance_name: str
    objectives: List[str]
    execution_time: float
    success: bool
    error_message: Optional[str] = None


# Instance groups for testing based on actual data in /home/hlvlad/code/sims-hybrid-algs/sims-problem/tests/data
SMALL_INSTANCES = [
    "lagos_nigeria_30.dzn", "rio_de_janeiro_30.dzn", "tokyo_bay_30.dzn", 
    "mexico_city_30.dzn", "paris_30.dzn"
]

MEDIUM_INSTANCES = [
    "lagos_nigeria_50.dzn", "rio_de_janeiro_50.dzn", "tokyo_bay_50.dzn",
    "mexico_city_50.dzn", "paris_50.dzn"
]

LARGE_INSTANCES = [
    "lagos_nigeria_100.dzn", "rio_de_janeiro_100.dzn", "tokyo_bay_100.dzn",
    "mexico_city_100.dzn", "paris_100.dzn", "lagos_nigeria_145.dzn"
]


def create_test_config(dzn_file_path: str, mzn_model_path: str, test_artifacts_dir: str, timeout: int = 300) -> Config:
    """Create a test configuration for solving MILP problems."""
    # Create a unique subdirectory for this test run
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # datetime-based timestamp
    instance_name = Path(dzn_file_path).stem
    test_subdir = Path(test_artifacts_dir) / f"{instance_name}_{timestamp}"
    test_subdir.mkdir(exist_ok=True)
    
    return Config(
        minizinc_data=False,
        instance_name=Path(dzn_file_path).stem,
        data_sets_folder=Path(dzn_file_path).parent,
        input_mzn=Path(mzn_model_path),
        dzn_dir=Path(dzn_file_path).parent,
        solver_name="gurobi",
        problem_name="sims",
        front_strategy="saugmecon",
        solver_timeout_sec=timeout,
        summary_filename=str(test_subdir / "test_summary.csv"),
        solver_search_strategy="free_search",
        fzn_optimisation_level=1,
        cores=1,
        threads=1
    )


def run_solve_milp_with_validation(
    dzn_file_path: str,
    mzn_model_path: str,
    test_artifacts_dir: str,
    objectives: List[str],
    timeout: int,
    logger: logging.Logger
) -> TestResult:
    """
    Run solve_milp with the given configuration and validate results.
    
    Args:
        dzn_file_path: Path to the DZN instance file
        mzn_model_path: Path to the MZN model file
        objectives: List of objectives to optimize
        timeout: Solver timeout in seconds
        logger: Logger instance
    
    Returns:
        TestResult with execution details and success status
    """
    instance_name = Path(dzn_file_path).stem
    logger.info(f"Running solve_milp on instance {instance_name}")
    logger.info(f"Objectives: {objectives}")
    logger.info(f"Timeout: {timeout}s")
    
    start_time = time.time()
    
    try:
        # Create configuration
        config = create_test_config(dzn_file_path, mzn_model_path, test_artifacts_dir, timeout)
        test_subdir = Path(config.summary_filename).parent
        
        logger.info(f"Test artifacts will be stored in: {test_subdir}")
        
        # Call solve_milp with the configuration
        solve.solve_milp(config=config, objectives=objectives)
        
        execution_time = time.time() - start_time
        logger.info(f"Execution completed successfully in {execution_time:.2f} seconds")
        
        # Validate that we got some result (solve_milp returns None)
        logger.info("solve_milp completed without exceptions")
        
        # Check if summary file was created (indicates successful execution)
        summary_file = Path(config.summary_filename)
        if summary_file.exists():
            logger.info(f"Summary file created: {config.summary_filename}")
            with open(config.summary_filename, 'r') as f:
                summary_content = f.read()
                if summary_content.strip():
                    logger.info("Summary file contains data")
                    logger.info(f"First few lines:\n{summary_content[:200]}...")
                else:
                    logger.warning("Summary file is empty")
        else:
            logger.warning(f"Summary file was not created at: {config.summary_filename}")
        
        # List all files in the test subdirectory
        if test_subdir.exists():
            artifacts = list(test_subdir.glob("*"))
            logger.info(f"Test artifacts created: {[str(f.name) for f in artifacts]}")
            logger.info(f"Artifacts location (will persist after test): {test_subdir}")
        
        return TestResult(
            instance_name=instance_name,
            objectives=objectives,
            execution_time=execution_time,
            success=True
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_message = str(e)
        logger.error(f"Execution failed after {execution_time:.2f} seconds: {error_message}")
        
        return TestResult(
            instance_name=instance_name,
            objectives=objectives,
            execution_time=execution_time,
            success=False,
            error_message=error_message
        )


class TestSolveMILPIntegration:
    """Integration tests for solve_milp function using real data instances."""

    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", SMALL_INSTANCES)
    def test_solve_milp_2d_on_small_instances(self, filename, test_data_dir, mzn_model_path, test_artifacts_dir, caplog):
        """Test solve_milp on small instances (30 images) with 2 objectives and reasonable timeout."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        dzn_file_path = Path(test_data_dir) / filename
        if not dzn_file_path.exists():
            pytest.skip(f"Test instance {filename} not found at {dzn_file_path}")
        
        # Configuration for small instances
        objectives = ["min_cost", "cloud_coverage"]
        timeout = 180  # 3 minutes for small instances
        
        # Run the test
        result = run_solve_milp_with_validation(
            dzn_file_path=str(dzn_file_path),
            mzn_model_path=mzn_model_path,
            test_artifacts_dir=test_artifacts_dir,
            objectives=objectives,
            timeout=timeout,
            logger=logger
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"solve_milp failed for {filename}: {result.error_message}")
        
        logger.info(f"Small instance {filename} completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", SMALL_INSTANCES)
    def test_solve_milp_3d_on_small_instances(self, filename, test_data_dir, mzn_model_path, test_artifacts_dir, caplog):
        """Test solve_milp on small instances (30 images) with 3 objectives and reasonable timeout."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        dzn_file_path = Path(test_data_dir) / filename
        if not dzn_file_path.exists():
            pytest.skip(f"Test instance {filename} not found at {dzn_file_path}")
        
        # Configuration for small instances with 3 objectives
        objectives = ["min_cost", "cloud_coverage", "min_resolution"]
        timeout = 240  # 4 minutes for small instances with 3 objectives
        
        # Run the test
        result = run_solve_milp_with_validation(
            dzn_file_path=str(dzn_file_path),
            mzn_model_path=mzn_model_path,
            test_artifacts_dir=test_artifacts_dir,
            objectives=objectives,
            timeout=timeout,
            logger=logger
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"solve_milp failed for {filename}: {result.error_message}")
        
        logger.info(f"Small instance {filename} (3D) completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", SMALL_INSTANCES)
    def test_solve_milp_4d_on_small_instances(self, filename, test_data_dir, mzn_model_path, test_artifacts_dir, caplog):
        """Test solve_milp on small instances (30 images) with 4 objectives and reasonable timeout."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        dzn_file_path = Path(test_data_dir) / filename
        if not dzn_file_path.exists():
            pytest.skip(f"Test instance {filename} not found at {dzn_file_path}")
        
        # Configuration for small instances with 4 objectives
        objectives = ["min_cost", "cloud_coverage", "min_resolution", "min_max_incidence_angle"]
        timeout = 300  # 5 minutes for small instances with 4 objectives
        
        # Run the test
        result = run_solve_milp_with_validation(
            dzn_file_path=str(dzn_file_path),
            mzn_model_path=mzn_model_path,
            test_artifacts_dir=test_artifacts_dir,
            objectives=objectives,
            timeout=timeout,
            logger=logger
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"solve_milp failed for {filename}: {result.error_message}")
        
        logger.info(f"Small instance {filename} (4D) completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", MEDIUM_INSTANCES)
    def test_solve_milp_2d_on_medium_instances(self, filename, test_data_dir, mzn_model_path, test_artifacts_dir, caplog):
        """Test solve_milp on medium instances (around 50 images) with 2 objectives and moderate timeout."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        dzn_file_path = Path(test_data_dir) / filename
        if not dzn_file_path.exists():
            pytest.skip(f"Test instance {filename} not found at {dzn_file_path}")
        
        # Configuration for medium instances
        objectives = ["min_cost", "cloud_coverage"]
        timeout = 300  # 5 minutes for medium instances
        
        # Run the test
        result = run_solve_milp_with_validation(
            dzn_file_path=str(dzn_file_path),
            mzn_model_path=mzn_model_path,
            test_artifacts_dir=test_artifacts_dir,
            objectives=objectives,
            timeout=timeout,
            logger=logger
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"solve_milp failed for {filename}: {result.error_message}")
        
        logger.info(f"Medium instance {filename} completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", MEDIUM_INSTANCES)
    def test_solve_milp_3d_on_medium_instances(self, filename, test_data_dir, mzn_model_path, test_artifacts_dir, caplog):
        """Test solve_milp on medium instances (around 50 images) with 3 objectives and moderate timeout."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        dzn_file_path = Path(test_data_dir) / filename
        if not dzn_file_path.exists():
            pytest.skip(f"Test instance {filename} not found at {dzn_file_path}")
        
        # Configuration for medium instances with 3 objectives
        objectives = ["min_cost", "cloud_coverage", "min_resolution"]
        timeout = 400  # 6.5 minutes for medium instances with 3 objectives
        
        # Run the test
        result = run_solve_milp_with_validation(
            dzn_file_path=str(dzn_file_path),
            mzn_model_path=mzn_model_path,
            test_artifacts_dir=test_artifacts_dir,
            objectives=objectives,
            timeout=timeout,
            logger=logger
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"solve_milp failed for {filename}: {result.error_message}")
        
        logger.info(f"Medium instance {filename} (3D) completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", MEDIUM_INSTANCES)
    def test_solve_milp_4d_on_medium_instances(self, filename, test_data_dir, mzn_model_path, test_artifacts_dir, caplog):
        """Test solve_milp on medium instances (around 50 images) with 4 objectives and moderate timeout."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        dzn_file_path = Path(test_data_dir) / filename
        if not dzn_file_path.exists():
            pytest.skip(f"Test instance {filename} not found at {dzn_file_path}")
        
        # Configuration for medium instances with 4 objectives
        objectives = ["min_cost", "cloud_coverage", "min_resolution", "min_max_incidence_angle"]
        timeout = 480  # 8 minutes for medium instances with 4 objectives
        
        # Run the test
        result = run_solve_milp_with_validation(
            dzn_file_path=str(dzn_file_path),
            mzn_model_path=mzn_model_path,
            test_artifacts_dir=test_artifacts_dir,
            objectives=objectives,
            timeout=timeout,
            logger=logger
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"solve_milp failed for {filename}: {result.error_message}")
        
        logger.info(f"Medium instance {filename} (4D) completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", LARGE_INSTANCES)
    def test_solve_milp_2d_on_large_instances(self, filename, test_data_dir, mzn_model_path, test_artifacts_dir, caplog):
        """Test solve_milp on large instances (100+ images) with 2 objectives and extended timeout."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        dzn_file_path = Path(test_data_dir) / filename
        if not dzn_file_path.exists():
            pytest.skip(f"Test instance {filename} not found at {dzn_file_path}")
        
        # Configuration for large instances
        objectives = ["min_cost", "cloud_coverage"]
        timeout = 600  # 10 minutes for large instances
        
        # Run the test
        result = run_solve_milp_with_validation(
            dzn_file_path=str(dzn_file_path),
            mzn_model_path=mzn_model_path,
            test_artifacts_dir=test_artifacts_dir,
            objectives=objectives,
            timeout=timeout,
            logger=logger
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"solve_milp failed for {filename}: {result.error_message}")
        
        logger.info(f"Large instance {filename} completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", LARGE_INSTANCES)
    def test_solve_milp_3d_on_large_instances(self, filename, test_data_dir, mzn_model_path, test_artifacts_dir, caplog):
        """Test solve_milp on large instances (100+ images) with 3 objectives and extended timeout."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        dzn_file_path = Path(test_data_dir) / filename
        if not dzn_file_path.exists():
            pytest.skip(f"Test instance {filename} not found at {dzn_file_path}")
        
        # Configuration for large instances with 3 objectives
        objectives = ["min_cost", "cloud_coverage", "min_resolution"]
        timeout = 900  # 15 minutes for large instances with 3 objectives
        
        # Run the test
        result = run_solve_milp_with_validation(
            dzn_file_path=str(dzn_file_path),
            mzn_model_path=mzn_model_path,
            test_artifacts_dir=test_artifacts_dir,
            objectives=objectives,
            timeout=timeout,
            logger=logger
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"solve_milp failed for {filename}: {result.error_message}")
        
        logger.info(f"Large instance {filename} (3D) completed successfully in {result.execution_time:.2f}s")

    @pytest.mark.timeout(3600)  # 1 hour timeout
    @pytest.mark.parametrize("filename", LARGE_INSTANCES)
    def test_solve_milp_4d_on_large_instances(self, filename, test_data_dir, mzn_model_path, test_artifacts_dir, caplog):
        """Test solve_milp on large instances (100+ images) with 4 objectives and extended timeout."""
        caplog.set_level(logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Skip if test instance doesn't exist
        dzn_file_path = Path(test_data_dir) / filename
        if not dzn_file_path.exists():
            pytest.skip(f"Test instance {filename} not found at {dzn_file_path}")
        
        # Configuration for large instances with 4 objectives
        objectives = ["min_cost", "cloud_coverage", "min_resolution", "min_max_incidence_angle"]
        timeout = 1200  # 20 minutes for large instances with 4 objectives
        
        # Run the test
        result = run_solve_milp_with_validation(
            dzn_file_path=str(dzn_file_path),
            mzn_model_path=mzn_model_path,
            test_artifacts_dir=test_artifacts_dir,
            objectives=objectives,
            timeout=timeout,
            logger=logger
        )
        
        # Assert success
        if not result.success:
            pytest.fail(f"solve_milp failed for {filename}: {result.error_message}")
        
        logger.info(f"Large instance {filename} (4D) completed successfully in {result.execution_time:.2f}s")
