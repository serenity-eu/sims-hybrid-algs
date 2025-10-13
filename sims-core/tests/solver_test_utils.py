"""Common utilities for MILP and solver testing.

This module contains shared functionality for testing different solver configurations
including artifact management, result validation, and instance data organization.
"""

import logging
import time
import traceback
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from sims.core.sims.problem import ProblemInstance, SimsDiscreteProblem
from sims.core.sims.solver_result import SolverResult, TwoPhaseSolverResult


@dataclass
class SolverTestResult:
    """Result of a solver test execution."""
    instance_name: str
    objectives: List[str]
    execution_time: float
    success: bool
    num_solutions: Optional[int] = None
    error_message: Optional[str] = None
    test_type: str = ""
    ratio: Optional[str] = None  # For two-phase tests


# Instance groups for testing based on actual data
SMALL_INSTANCES = [
    "lagos_nigeria_30.dzn", "mexico_city_30.dzn", "paris_30.dzn", 
    "rio_de_janeiro_30.dzn", "tokyo_bay_30.dzn"
]

MEDIUM_INSTANCES = [
    "lagos_nigeria_50.dzn", "mexico_city_50.dzn", "paris_50.dzn",
    "rio_de_janeiro_50.dzn", "tokyo_bay_50.dzn"
]

LARGE_INSTANCES = [
    "lagos_nigeria_100.dzn", "mexico_city_100.dzn", "paris_100.dzn",
    "rio_de_janeiro_100.dzn", "tokyo_bay_100.dzn",
    "lagos_nigeria_145.dzn", "mexico_city_150.dzn", "paris_150.dzn", 
    "rio_de_janeiro_150.dzn", "tokyo_bay_150.dzn",
    "mexico_city_200.dzn", "paris_200.dzn", "rio_de_janeiro_200.dzn", "tokyo_bay_200.dzn"
]

# Common ratios for two-phase testing (exact_percentage, pls_percentage)
TWO_PHASE_RATIOS = [
    (100, 0),   # 100% exact, 0% PLS
    (50, 50),   # 50% exact, 50% PLS
    (20, 80),   # 20% exact, 80% PLS
    (0, 100),   # 0% exact, 100% PLS
]


def create_problem_instance(instance_path: str) -> ProblemInstance:
    """Create a ProblemInstance from a DZN file path."""
    return ProblemInstance.from_dzn(Path(instance_path))


def create_temp_output_dir(instance_name: str, test_type: str) -> Path:
    """Create a temporary output directory for solver results."""
    output_dir = Path("/tmp") / f"test_output_{instance_name}_{test_type}_{int(time.time())}"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def validate_solver_result(result: SolverResult, problem: SimsDiscreteProblem, objectives: list[str] | None = None) -> tuple[bool, str]:
    """
    Validate a single solver result with semantic validation of solutions.
    
    Args:
        result: The solver result to validate
        problem: SimsDiscreteProblem for semantic validation of solutions
        objectives: Optional list of objective names that were optimized (for validation)
    
    Returns:
        Tuple of (success, error_message)
    """
    if not result.pareto_front:
        return True, ""  # No solutions to validate
    
    # Semantic validation
    for idx, solution in enumerate(result.pareto_front):
        if not solution.validate(problem):
            return False, f"Solution {idx} failed validation (constraint violation or incorrect coverage)"
        
        # Validate objective values
        if not solution.validate_objectives(problem, objectives):
            return False, f"Solution {idx} has incorrect objective values"
    
    return True, ""


def validate_two_phase_solver_result(result: TwoPhaseSolverResult, problem: SimsDiscreteProblem, objectives: list[str] | None = None) -> tuple[bool, str]:
    """
    Validate a two-phase solver result with semantic validation of solutions.
    
    Args:
        result: The two-phase solver result to validate
        problem: SimsDiscreteProblem for semantic validation of solutions
        objectives: Optional list of objective names that were optimized (for validation)
    
    Returns:
        Tuple of (success, error_message)
    """
    # Validate exact solver result
    if result.exact_solver_result:
        if not result.exact_solver_result.pareto_front:
            logging.info("No solutions found in exact solver phase")
        else:
            for idx, solution in enumerate(result.exact_solver_result.pareto_front):
                if not solution.validate(problem):
                    return False, f"Solution {idx} from exact solver failed validation (constraint violation or incorrect coverage)"
                
                # Validate objective values
                if not solution.validate_objectives(problem, objectives):
                    return False, f"Solution {idx} from exact solver has incorrect objective values"
    
    # Validate PLS result
    if result.pls_result:
        if not result.pls_result.pareto_front:
            logging.info("No solutions found in PLS phase")
        else:
            for idx, solution in enumerate(result.pls_result.pareto_front):
                if not solution.validate(problem):
                    return False, f"Solution {idx} from pls failed validation (constraint violation or incorrect coverage)"
                
                # Validate objective values
                if not solution.validate_objectives(problem, objectives):
                    return False, f"Solution {idx} from pls has incorrect objective values"
    
    return True, ""


def log_solution_details(result, logger: logging.Logger):
    """Log details about solutions found in solver result."""
    if isinstance(result, TwoPhaseSolverResult):
        exact_solutions = 0
        pls_solutions = 0
        
        if result.exact_solver_result and result.exact_solver_result.pareto_front:
            exact_solutions = len(result.exact_solver_result.pareto_front)
            first_exact = result.exact_solver_result.pareto_front[0]
            logger.info(f"Exact solver: {exact_solutions} solutions, first - cost: {first_exact.cost}, cloudy_area: {first_exact.cloudy_area}")
        
        if result.pls_result and result.pls_result.pareto_front:
            pls_solutions = len(result.pls_result.pareto_front)
            first_pls = result.pls_result.pareto_front[0]
            logger.info(f"PLS: {pls_solutions} solutions, first - cost: {first_pls.cost}, cloudy_area: {first_pls.cloudy_area}")
        
        logger.info(f"Total solutions: {exact_solutions + pls_solutions}")
        
    elif hasattr(result, 'pareto_front') and result.pareto_front:
        num_solutions = len(result.pareto_front)
        first_solution = result.pareto_front[0]
        logger.info(f"Found {num_solutions} solutions")
        logger.info(f"First solution - cost: {first_solution.cost}, cloudy_area: {first_solution.cloudy_area}, selected_images: {len(first_solution.selected_images)}")
    else:
        logger.warning("No solutions found")


def save_test_artifacts(artifacts_manager, test_name: str, instance_name: str, 
                       result, test_type: str, objectives: List[str], 
                       execution_time: float, logger: logging.Logger, 
                       ratio: Optional[str] = None, test_failed: bool = False,
                       failure_exception: Optional[Exception] = None):
    """Save test artifacts (JSON result and trace data) for a test run."""
    if not artifacts_manager:
        return
    
    # Prepare result data for JSON serialization
    result_data = {
        "instance_name": instance_name,
        "test_type": test_type,
        "objectives": objectives,
        "execution_time": execution_time,
        "success": not test_failed,
        "timestamp": time.time(),
        "solutions": []
    }
    
    if ratio:
        result_data["ratio"] = ratio
    
    # Extract trace data
    trace_data = None
    failure_info = None
    
    # Prepare failure information if test failed
    if test_failed:
        failure_info = {}
        if failure_exception:
            failure_info["exception"] = str(failure_exception)
            failure_info["traceback"] = traceback.format_exc()
            failure_info["error_message"] = str(failure_exception)
        
        # Capture recent log records (this is a simple approach)
        # Note: For more sophisticated log capture, you'd need a custom log handler
        failure_info["logs"] = f"Test failed for {test_name}/{instance_name}"
    
    if isinstance(result, TwoPhaseSolverResult):
        result_data["solutions"] = result.solutions
        result_data["num_solutions"] = len(result.solutions)
        
        # Get trace data from TwoPhaseSolverResult (which handles merging automatically)
        if hasattr(result, 'trace_data') and result.trace_data:
            trace_data = result.trace_data
        
    elif result is not None:
        # Handle regular solver results
        num_solutions = len(result.pareto_front) if result.pareto_front else 0
        result_data["num_solutions"] = num_solutions
        
        if result.pareto_front:
            for i, solution in enumerate(result.pareto_front):
                solution_data = {
                    "index": i,
                    "selected_images": list(solution.selected_images),
                    "cost": solution.cost,
                    "cloudy_area": solution.cloudy_area,
                    "max_incidence_angle": solution.max_incidence_angle,
                    "min_resolutions_sum": solution.min_resolutions_sum,
                    "timestamp_s": solution.timestamp_s.total_seconds() if solution.timestamp_s else None
                }
                result_data["solutions"].append(solution_data)
        
        # Get trace data
        if hasattr(result, 'trace_data') and result.trace_data:
            trace_data = result.trace_data
    
    # Save artifacts with failure information
    artifacts_manager.save_test_result(
        test_name=test_name,
        instance_name=instance_name,
        result_data=result_data,
        trace_data=trace_data,
        test_failed=test_failed,
        failure_info=failure_info
    )
    logger.info(f"Artifacts saved for {test_name}/{instance_name} (failed: {test_failed})")


def format_ratio_string(ratio: tuple[int, int]) -> str:
    """Format a ratio tuple as a string for display and file naming."""
    return f"{ratio[0]}_{ratio[1]}"


def get_timeout_for_instance_size(instances: List[str], base_timeout: int = 3300) -> int:
    """
    Get appropriate timeout based on instance size.
    
    Args:
        instances: List of instance filenames to determine size category
        base_timeout: Base timeout in seconds (default 55 minutes)
    
    Returns:
        Timeout in seconds
    """
    # Check if any instance is in LARGE_INSTANCES
    # For large instances, use a reduced timeout to ensure completion before pytest timeout (3600s)
    for instance in instances:
        if instance in LARGE_INSTANCES:
            # 50 minutes for large instances (leaves 10 min buffer for pytest's 60 min timeout)
            return 3000
    
    return base_timeout  