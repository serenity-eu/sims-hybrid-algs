import pytest
from pathlib import Path
from datetime import datetime
import json
import logging
import os
from typing import Optional

# Test configuration
TIMEOUT_SECONDS = 60

# Use relative paths from the sims-core directory
sims_core_dir = Path(__file__).parent.parent
TEST_INSTANCES_DIR = str(sims_core_dir / "tests" / "data")
MZN_MODEL_PATH = str(sims_core_dir.parent / "sims-solvers" / "sims_solvers" / "mzn_models" / "mosaic_cloud2.mzn")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--name",
        action="store",
        default=None,
        help="Name for the test run to organize artifacts"
    )
    parser.addoption(
        "--use-pseudo-solver",
        action="store_true",
        default=False,
        help="Use pseudo-solver with pre-recorded solutions instead of real solver"
    )


@pytest.fixture(scope="session")
def test_artifacts_dir(request):
    """Fixture providing the test artifacts directory path."""
    # Get the name parameter from command line
    run_name = request.config.getoption("--name")
    
    if run_name is None:
        run_name = "unnamed"
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create artifacts directory
    artifacts_dir = sims_core_dir / "test_artifacts" / f"{run_name}_{timestamp}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    return artifacts_dir


@pytest.fixture
def test_data_dir():
    """Fixture providing the test data directory path."""
    return TEST_INSTANCES_DIR


@pytest.fixture
def mzn_model_path():
    """Fixture providing the MiniZinc model path."""
    return MZN_MODEL_PATH


@pytest.fixture
def timeout():
    """Fixture providing the timeout for tests."""
    return TIMEOUT_SECONDS


@pytest.fixture
def use_pseudo_solver(request):
    """Fixture providing whether to use pseudo-solver mode."""
    return request.config.getoption("--use-pseudo-solver")


class TestArtifactsManager:
    """Helper class for managing test artifacts."""
    
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup file logging for the test run."""
        # Create a log file for this test session
        log_file = self.artifacts_dir / "test_session.log"
        
        # Configure file handler for all logs
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)8s] %(name)s: %(message)s (%(filename)s:%(lineno)d)',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Get root logger and add file handler without changing existing console handlers
        root_logger = logging.getLogger()
        
        # Only add file handler if it doesn't already exist
        file_handler_exists = any(
            isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_file)
            for handler in root_logger.handlers
        )
        
        if not file_handler_exists:
            root_logger.addHandler(file_handler)
        
        # Ensure root logger level allows DEBUG for file logging
        if root_logger.level > logging.DEBUG:
            root_logger.setLevel(logging.DEBUG)
    
    def save_test_result(
        self,
        test_name: str,
        instance_name: str,
        result_data: dict,
        ratio: Optional[str] = None,
        iteration: Optional[int] = None,
        trace_data: Optional[bytes] = None,
        profiling_trace_data: Optional[bytes] = None,
        test_failed: bool = False,
        failure_info: Optional[dict] = None
    ):
        """Save test results and trace data to artifacts directory.
        
        Creates directory structure:
        - test_artifacts/test_name/instance_name/ratio/iterN/
        
        Args:
            profiling_trace_data: Chrome profiling trace data in JSON format
        """
        # Build path: test_name/instance_name/ratio/iterN
        test_dir = self.artifacts_dir / test_name / instance_name
        
        if ratio:
            # Add ratio subfolder (e.g., "100_0", "20_80")
            test_dir = test_dir / ratio
        
        if iteration is not None:
            # Add iteration subfolder (e.g., "iter0", "iter1")
            test_dir = test_dir / f"iter{iteration}"
        
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON result
        result_file = test_dir / "result.json"
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        # Save trace data if available
        if trace_data:
            trace_file = test_dir / "trace.tar.gz"
            # trace_data is already a compressed tar.gz archive from the Rust merge_traces function
            # Just write it directly to the file
            with open(trace_file, 'wb') as f:
                f.write(trace_data)
        
        # Save profiling trace data if available
        if profiling_trace_data:
            profiling_trace_file = test_dir / "profiling_trace.json"
            # profiling_trace_data is Chrome trace JSON format
            with open(profiling_trace_file, 'wb') as f:
                f.write(profiling_trace_data)
        
        # Save failure information if test failed
        if test_failed and failure_info:
            failure_file = test_dir / "failure.log"
            with open(failure_file, 'w') as f:
                f.write("=== TEST FAILURE DETAILS ===\n")
                f.write(f"Test: {test_name}\n")
                f.write(f"Instance: {instance_name}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
                
                if 'exception' in failure_info:
                    f.write("=== EXCEPTION ===\n")
                    f.write(str(failure_info['exception']))
                    f.write("\n\n")
                
                if 'traceback' in failure_info:
                    f.write("=== TRACEBACK ===\n")
                    f.write(failure_info['traceback'])
                    f.write("\n\n")
                
                if 'error_message' in failure_info:
                    f.write("=== ERROR MESSAGE ===\n")
                    f.write(failure_info['error_message'])
                    f.write("\n\n")
                
                if 'logs' in failure_info:
                    f.write("=== CAPTURED LOGS ===\n")
                    f.write(failure_info['logs'])
                    f.write("\n")
        
        return test_dir


@pytest.fixture
def artifacts_manager(test_artifacts_dir):
    """Fixture providing a test artifacts manager."""
    return TestArtifactsManager(test_artifacts_dir)