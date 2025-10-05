#!/usr/bin/env python3

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

"""
SIMS Core CLI Script

A standalone command-line interface for running SIMS experiments in hybrid mode with configurable parameters.
This script provides a simplified interface to run hybrid solver experiments using the sims-core package.

Features:
- Hybrid solver execution with configurable ratio-step
- Experiment preparation and solving
- Results processing and analysis
- Support for filtering experiments by regex patterns
- Dry-run mode for testing configurations

Usage:
    """
"""
SIMS Hybrid Algorithm CLI

This script provides command-line interface for running SIMS hybrid experiments
using test instances from sims-problem/tests/data directory.

Features:
- Hybrid MILP+PLS solving with configurable ratio steps
- Experiment preparation and solving
- Results processing and analysis
- Support for filtering experiments by regex patterns
- Dry-run mode for testing configurations

Usage:
    # Run hybrid experiments on all test instances
    uv run sims_cli.py run-hybrid
    
    # Run with custom ratio step
    uv run sims_cli.py run-hybrid --ratio-step 10
    
    # Filter specific instances
    uv run sims_cli.py run-hybrid --filter lagos_nigeria
    
    # Process experiment results
    uv run sims_cli.py process --experiments-dir /path/to/experiments
"""

# Add the src directory to Python path to import sims.core
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Add sims-problem directory to path
sims_problem_dir = script_dir.parent.parent / "sims-problem"
sys.path.insert(0, str(sims_problem_dir))

try:
    from sims.core.sims import solver
    from sims.core.sims.solver_config import FrontStrategy, SolverConfig, SolverType, TwoPhaseSolverConfig
    from sims.core.sims.problem import ProblemInstance
    import sims_problem
except ImportError as e:
    print(f"Error importing sims.core modules: {e}")
    print("Make sure you're running this script from the sims-core directory")
    print("and that the sims-core package is properly installed.")
    sys.exit(1)

# Configure logging
log_format = "%(asctime)s %(name)-10s [%(levelname)-7s] %(message)s"


def setup_logging(verbose: bool = False, log_file: Path | None = None):
    """
    Configure logging to output to both console and optionally to a file.
    
    Args:
        verbose: Enable debug level logging
        log_file: Optional path to log file
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set base level
    level = logging.INFO if verbose else logging.ERROR
    root_logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)  # Always use DEBUG for file
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)
        
        logging.info(f"Logging to file: {log_file}")


# Initial basic configuration (will be reconfigured in main)
logging.basicConfig(level=logging.ERROR, format=log_format)
log = logging.getLogger("sims-cli")
log.setLevel(logging.INFO)  # Keep sims-cli messages visible


def run_experiments(
    instances_dir: Path | None,
    ratio_step: int,
    timeout_s: int,
    solver_type: SolverType,
    front_strategy: FrontStrategy,
    dry_run: bool,
    iter_count: int,
    instance_regex: str | None = None,
    skip_solved: bool = False,
    results_dir: Path | None = None,
    enable_trace: bool = False,
) -> int:
    """
    Run hybrid experiments with the specified configuration.
    
    Args:
        instances_dir: Directory containing .dzn instance files (default: sims-problem/tests/data)
        ratio_step: Step size for ratio configurations (default: 25, gives ratios 100:0, 75:25, 50:50, 25:75, 0:100)
        timeout_s: Timeout in seconds for each solver run
        solver_type: Type of solver to use (GUROBI, PLS, OR_TOOLS)
        front_strategy: Front generation strategy (GPBA_A, SAUGMECON, etc.)
        dry_run: If True, only simulate the run without actual execution
        iter_count: Number of iterations per configuration
        instance_regex: Regex pattern to filter instances by name
        skip_solved: Skip already solved experiments
        results_dir: Directory to save results (if different from default)
        enable_trace: Enable PLS trace data collection for debugging and analysis
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Use default test data directory if not specified
    if instances_dir is None:
        instances_dir = script_dir.parent.parent / "sims-problem" / "tests" / "data"
    
    if not instances_dir.exists():
        log.error(f"Instances directory does not exist: {instances_dir}")
        return 1
    
    if not instances_dir.is_dir():
        log.error(f"Instances path is not a directory: {instances_dir}")
        return 1
    
    # Configure solver for hybrid mode
    solver_config = SolverConfig(
        solver_type=solver_type,
        front_strategy=front_strategy,
        timeout_s=timeout_s,
        ratio_step=ratio_step,
    )
    
    log.info("Running hybrid experiments with configuration:")
    log.info(f"  Solver Type: {solver_config.solver_type}")
    log.info(f"  Front Strategy: {solver_config.front_strategy}")
    log.info(f"  Timeout: {solver_config.timeout_s}s")
    log.info(f"  Ratio Step: {solver_config.ratio_step}")
    log.info(f"  Dry Run: {dry_run}")
    log.info(f"  Iterations: {iter_count}")
    log.info(f"  Instances Directory: {instances_dir}")
    
    # Find .dzn instance files
    dzn_files = []
    if instance_regex is not None:
        import re
        pattern = re.compile(instance_regex)
        dzn_files = [
            dzn_file for dzn_file in instances_dir.glob("*.dzn")
            if pattern.search(dzn_file.stem)
        ]
        log.info(f'Selected {len(dzn_files)} instances matching regex "{instance_regex}": {[f.stem for f in dzn_files]}')
    else:
        dzn_files = list(instances_dir.glob("*.dzn"))
        log.info(f'Found {len(dzn_files)} .dzn instance files')
    
    if not dzn_files:
        log.warning("No .dzn instance files found")
        return 0
    
    # Set up results directory
    if results_dir is None:
        results_dir = Path.cwd() / "hybrid_results"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments for each instance
    success_count = 0
    for instance_idx, dzn_file in enumerate(dzn_files):
        instance_name = dzn_file.stem
        log.info(f"~~~~~~ Processing instance {instance_name} ({instance_idx + 1}/{len(dzn_files)}) ~~~~~~")
        
        try:
            if dry_run:
                log.info(f"[DRY RUN] Would process instance: {instance_name}")
                log.info(f"[DRY RUN] Instance file: {dzn_file}")
                ratios = list(range(100, -1, -ratio_step))
                log.info(f"[DRY RUN] Would run {len(ratios)} ratio configurations: {ratios}")
                for ratio in ratios:
                    log.info(f"[DRY RUN]   Ratio {ratio}:{100-ratio} (MILP:PLS) - timeout {timeout_s}s")
                success_count += 1
            else:
                # Load the problem instance
                problem = sims_problem.SimsDiscreteProblem.from_dzn(str(dzn_file))
                problem_instance = ProblemInstance(
                    name=instance_name,
                    problem=problem,
                    path=dzn_file
                )
                
                # Create experiment directory for this instance
                experiment_dir = results_dir / instance_name
                experiment_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy the .dzn file to the experiment directory
                import shutil
                dzn_dest = experiment_dir / f"{instance_name}.dzn"
                shutil.copy2(dzn_file, dzn_dest)
                
                # Generate ratio configurations and solve directly
                ratios = list(range(100, -1, -ratio_step))
                
                for ratio in ratios:
                    log.info(f"  Running with ratio {ratio}:{100-ratio} (MILP:PLS)")
                    
                    # Create solver config for this ratio
                    two_phase_config = TwoPhaseSolverConfig(
                        exact_solver_type=solver_type.value,
                        front_strategy=front_strategy,
                        timeout_s=timeout_s,
                        ratio=(ratio, 100-ratio)
                    )
                    
                    # Create results directory for this ratio
                    ratio_dir = experiment_dir / f"{ratio}_{100-ratio}"
                    ratio_dir.mkdir(parents=True, exist_ok=True)
                    
                    if not dry_run:
                        try:
                            two_phase_result = solver.solve_with_two_phases(
                                problem_instance=problem_instance,
                                problem_path=dzn_dest,
                                experiment_path=ratio_dir,
                                solver_config=two_phase_config,
                                objectives=["min_cost", "cloud_coverage", "min_max_incidence_angle"],
                                dry_run=dry_run,
                                enable_pls_trace=enable_trace
                            )

                            if two_phase_result.pls_result is not None:
                                log.error(f"PLS Pareto front has {len(two_phase_result.pls_result.pareto_front)} solutions")

                            # Save the result to JSON file in the ratio directory
                            result_file = ratio_dir / "two_phase_result.json"
                            with open(result_file, 'w') as f:
                                import json
                                json.dump(two_phase_result.to_dict(), f, indent=2)

                            if two_phase_result.pls_result is not None and two_phase_result.pls_result.trace_data is not None:
                                trace_file = ratio_dir / "pls_trace.tar.gz"
                                with open(trace_file, 'wb') as tf:
                                    tf.write(two_phase_result.pls_result.trace_data)
                                log.info(f"    Saved PLS trace to {trace_file}")
                            log.info(f"    Ratio {ratio}:{100-ratio} completed successfully")
                        except Exception as e:
                            log.error(f"    Ratio {ratio}:{100-ratio} failed: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        log.info(f"    [DRY RUN] Would solve with ratio {ratio}:{100-ratio}")
                
                success_count += 1
                log.info(f"Successfully processed instance: {instance_name}")
        
        except Exception as e:
            log.error(f"Failed to process instance {instance_name}. Reason: {e}")
            import traceback
            traceback.print_exc()
    
    log.info(f"Completed {success_count}/{len(dzn_files)} instances successfully")
    return 0 if success_count == len(dzn_files) else 1


def run_milp_experiments(
    instances_dir: Path | None,
    timeout_s: int,
    solver_type: SolverType,
    front_strategy: FrontStrategy,
    dry_run: bool,
    iter_count: int,
    instance_regex: str | None = None,
    skip_solved: bool = False,
    results_dir: Path | None = None,
) -> int:
    """
    Run MILP-only experiments (first phase only, no PLS heuristic).
    
    This function runs pure MILP optimization using the specified front strategy,
    which is perfect for testing our objective rotation implementation in GPBA-A.
    """
    import sims_problem
    import re
    from pathlib import Path
    
    # Set default instances directory
    if instances_dir is None:
        instances_dir = Path(__file__).parent.parent.parent / "sims-problem" / "tests" / "data"
    
    if not instances_dir.exists():
        log.error(f"Instances directory does not exist: {instances_dir}")
        return 1
    
    # Set default results directory
    if results_dir is None:
        results_dir = Path("sims_milp_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("Running MILP-only experiments with configuration:")
    log.info(f"  Solver Type: {solver_type.value}")
    log.info(f"  Front Strategy: {front_strategy.value}")
    log.info(f"  Timeout: {timeout_s}s")
    log.info(f"  Dry Run: {dry_run}")
    log.info(f"  Iterations: {iter_count}")
    log.info(f"  Instances Directory: {instances_dir}")
    
    # Find all .dzn files
    dzn_files = list(instances_dir.glob("*.dzn"))
    if not dzn_files:
        log.error(f"No .dzn files found in {instances_dir}")
        return 1
    
    # Filter files if regex pattern provided
    if instance_regex:
        pattern = re.compile(instance_regex)
        dzn_files = [f for f in dzn_files if pattern.search(f.stem)]
        log.info(f"Selected {len(dzn_files)} instances matching regex \"{instance_regex}\": {[f.stem for f in dzn_files]}")
    else:
        log.info(f"Found {len(dzn_files)} instances: {[f.stem for f in dzn_files]}")
    
    if not dzn_files:
        log.warning("No instances to process after filtering")
        return 0
    
    success_count = 0
    
    for i, dzn_file in enumerate(dzn_files, 1):
        instance_name = dzn_file.stem
        log.info(f"~~~~~~ Processing instance {instance_name} ({i}/{len(dzn_files)}) ~~~~~~")
        
        try:
            if dry_run:
                log.info(f"[DRY RUN] Would process instance: {instance_name}")
                log.info(f"[DRY RUN] Instance file: {dzn_file}")
                log.info(f"[DRY RUN] Would run MILP solver with {front_strategy.value} strategy - timeout {timeout_s}s")
                success_count += 1
                continue
            
            # Load the problem instance
            problem = sims_problem.SimsDiscreteProblem.from_dzn(str(dzn_file))
            problem_instance = ProblemInstance(
                name=instance_name,
                problem=problem,
                path=dzn_file
            )
            
            # Create experiment directory for this instance
            experiment_dir = results_dir / instance_name
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy the .dzn file to the experiment directory
            import shutil
            dzn_dest = experiment_dir / f"{instance_name}.dzn"
            shutil.copy2(dzn_file, dzn_dest)
            
            # Create MILP-only solver config (100% MILP, 0% PLS)
            log.info(f"  Running MILP-only optimization with {front_strategy.value} strategy")
            
            # Define objectives for SIMS problem
            objectives = ["min_cost", "cloud_coverage", "min_max_incidence_angle"]
            
            # Execute experiment iterations
            successful_iterations = 0
            for iteration in range(iter_count):
                log.info(f"    Iteration {iteration + 1}/{iter_count}")
                
                result_file = experiment_dir / f"milp_result_iter_{iteration + 1}.json"
                if skip_solved and result_file.exists():
                    log.info(f"    Skipping solved iteration {iteration + 1}")
                    successful_iterations += 1  # Count skipped as successful
                    continue
                
                # Run the single-phase MILP solver
                try:
                    # Create output file path for solver results
                    solver_output_file = experiment_dir / f"solver_output_iter_{iteration + 1}.json"
                    
                    result = solver.solve(
                        solver_type=solver_type,
                        problem_instance=problem_instance,
                        problem_path=dzn_dest,
                        timeout_s=timeout_s,
                        output_path=solver_output_file,
                        objectives=objectives,
                        front_strategy=front_strategy,
                        initial_population=None
                    )
                    
                    # Save result
                    with open(result_file, 'w') as f:
                        import json
                        json.dump(result.to_dict(), f, indent=2)
                    
                    # Count solutions found - solver.solve() returns SolverResult with pareto_front
                    solution_count = len(result.pareto_front)
                    explored_count = len(result.explored_solutions)
                    
                    log.info(f"    ✅ Completed iteration {iteration + 1} - found {solution_count} solutions, explored {explored_count} solutions")
                    successful_iterations += 1
                    
                except Exception as e:
                    log.error(f"    ❌ Failed iteration {iteration + 1}: {e}")
                    continue
            
            # Only count as successful if ALL iterations succeeded
            if successful_iterations == iter_count:
                success_count += 1
                log.info(f"✅ Successfully completed instance {instance_name} (all {iter_count} iterations)")
            else:
                log.error(f"❌ Failed to complete instance {instance_name} - only {successful_iterations}/{iter_count} iterations succeeded")
        
        except Exception as e:
            log.error(f"Failed to process instance {instance_name}. Reason: {e}")
            import traceback
            traceback.print_exc()
    
    log.info(f"Completed {success_count}/{len(dzn_files)} instances successfully")
    return 0 if success_count == len(dzn_files) else 1


def prepare_experiments(
    aois_dir: Path,
    images_dir: Path,
    output_dir: Path,
    satellite_data_dir: Path | None = None,
) -> int:
    """
    Prepare experiments from AOI and image data.
    
    Args:
        aois_dir: Directory containing AOI files
        images_dir: Directory containing preselected image files
        output_dir: Directory to create experiment folders
        satellite_data_dir: Optional satellite data directory
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not output_dir.exists():
            output_dir = output_dir / f"experiments_{timestamp}"
        
        log.info(f"Preparing experiments in: {output_dir}")
        
        # Use the experiment.prepare function
        # This is a placeholder - the actual implementation may vary
        # based on the exact interface of the experiment module
        log.info("Experiment preparation functionality needs to be implemented")
        log.info("Please refer to the sims.core.experiment module for preparation methods")
        
        return 0
    
    except Exception as e:
        log.error(f"Failed to prepare experiments: {e}")
        return 1


def process_results(experiments_dir: Path, output_dir: Path) -> int:
    """
    Process experiment results and generate analysis.
    
    Args:
        experiments_dir: Directory containing solved experiments
        output_dir: Directory to save processed results
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        log.info(f"Processing results from: {experiments_dir}")
        log.info(f"Output directory: {output_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        experiments_output_dir = output_dir / experiments_dir.name
        experiments_output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        for experiment_dir in experiments_dir.iterdir():
            if not experiment_dir.is_dir():
                continue
            
            try:
                # TODO: Update this to work with new simplified experiment structure
                log.warning("Process command not yet updated for simplified experiment structure")
                log.warning(f"Skipping experiment directory: {experiment_dir.name}")
                # Check if experiment is solved
                # experiment_obj = Experiment.from_dir(experiment_dir)
                # solver_config = SolverConfig.from_json(experiment_dir / "solver_config.json")
                
                # if experiment_obj.is_solved(experiment_dir, solver_config):
                #     log.info(f"Processing experiment: {experiment_dir.name}")
                #     experiment_results = experiment_obj.parse_results(experiment_dir=experiment_dir)
                #     experiment_results.process(output_dir=experiments_output_dir)
                #     processed_count += 1
                # else:
                #     log.warning(f"Experiment {experiment_dir.name} is not solved, skipping")
            
            except Exception as e:
                log.error(f"Failed to process experiment {experiment_dir.name}: {e}")
        
        log.info(f"Successfully processed {processed_count} experiments")
        return 0
    
    except Exception as e:
        log.error(f"Failed to process results: {e}")
        return 1


def compress_traces(artifacts_dir: Path, filter_pattern: str | None = None, overwrite: bool = False) -> int:
    """
    Compress result.json files from test artifacts to trace.tar.gz format.
    
    This function scans test_artifacts directory structure and converts all result.json 
    files to compressed trace.tar.gz archives in the same locations.
    
    Args:
        artifacts_dir: Directory containing test artifacts with result.json files
        filter_pattern: Regex pattern to filter artifact directories
        overwrite: Overwrite existing trace.tar.gz files
    
    Returns:
        0 on success, 1 on failure
    """
    artifacts_dir = artifacts_dir.resolve()
    if not artifacts_dir.exists():
        log.error(f"Artifacts directory does not exist: {artifacts_dir}")
        return 1
    
    log.info(f"🔍 Scanning test artifacts directory: {artifacts_dir}")
    
    # Recursively find all result.json files
    result_files: list[Path] = list(artifacts_dir.rglob("result.json"))
    
    # Apply filter if provided
    if filter_pattern:
        pattern = re.compile(filter_pattern)
        filtered_files = []
        for result_file in result_files:
            # Check if any parent directory matches the pattern
            if any(pattern.search(parent.name) for parent in result_file.parents):
                filtered_files.append(result_file)
        result_files = filtered_files
        log.info(f"📦 Filtered to {len(result_files)} result.json files matching pattern: {filter_pattern}")
    else:
        log.info(f"📦 Found {len(result_files)} result.json files to process")
    
    if not result_files:
        raise ValueError("No result.json files found matching filter pattern - ensure files exist and pattern is correct")
    
    converted_count = 0
    skipped_count = 0
    
    for result_file in sorted(result_files):
        try:
            trace_file = result_file.parent / "trace.tar.gz"
            
            # Skip if trace file already exists and not overwriting
            if trace_file.exists() and not overwrite:
                log.info(f"⏭️  Trace already exists: {trace_file.relative_to(artifacts_dir)}")
                skipped_count += 1
                continue
            
            log.info(f"🔄 Processing: {result_file.relative_to(artifacts_dir)}")
            
            # Load and parse the JSON file
            with open(result_file, 'r') as f:
                result_data: dict[str, Any] = json.load(f)
            
            # Extract instance name from result data - fail if not found
            if "instance_name" not in result_data or not result_data["instance_name"]:
                raise ValueError(f"Missing or empty 'instance_name' field in result.json: {result_file.relative_to(artifacts_dir)}")
            
            instance_name = result_data["instance_name"]
            log.info(f"Processing instance: {instance_name}")
            
            # Parse different JSON formats and extract solutions - fail fast on unknown format
            solutions: list[dict[str, Any]] = []
            
            # Format 1: From save_test_artifacts function (test results)
            if "solutions" in result_data and isinstance(result_data["solutions"], list):
                if not result_data["solutions"]:
                    raise ValueError(f"Empty solutions list in result.json: {result_file.relative_to(artifacts_dir)}")
                
                for i, sol in enumerate(result_data["solutions"]):
                    # Strict validation of required solution fields
                    required_fields = ["selected_images", "cost", "cloudy_area", "max_incidence_angle", "min_resolutions_sum", "timestamp_s"]
                    for field in required_fields:
                        if field not in sol:
                            raise ValueError(f"Solution {i} missing required field '{field}' in result.json")
                    
                    solutions.append({
                        "selected_images": sol["selected_images"],
                        "cost": sol["cost"], 
                        "cloudy_area": sol["cloudy_area"],
                        "max_incidence_angle": sol["max_incidence_angle"],
                        "min_resolutions_sum": sol["min_resolutions_sum"],
                        "timestamp_s": sol["timestamp_s"]
                    })
                    
            # Format 2: TwoPhaseSolverResult format (experiment files)
            elif "pls_result" in result_data or "exact_solver_result" in result_data:
                found_solutions = False
                
                if result_data.get("exact_solver_result") and result_data["exact_solver_result"].get("pareto_front"):
                    found_solutions = True
                    for i, sol in enumerate(result_data["exact_solver_result"]["pareto_front"]):
                        # Strict validation
                        required_fields = ["selected_images", "cost", "cloudy_area", "max_incidence_angle", "min_resolutions_sum", "timestamp_s"]
                        for field in required_fields:
                            if field not in sol:
                                raise ValueError(f"exact_solver_result solution {i} missing required field '{field}'")
                        
                        solutions.append({
                            "selected_images": sol["selected_images"],
                            "cost": sol["cost"],
                            "cloudy_area": sol["cloudy_area"], 
                            "max_incidence_angle": sol["max_incidence_angle"],
                            "min_resolutions_sum": sol["min_resolutions_sum"],
                            "timestamp_s": sol["timestamp_s"]
                        })
                
                if result_data.get("pls_result") and result_data["pls_result"].get("pareto_front"):
                    found_solutions = True
                    for i, sol in enumerate(result_data["pls_result"]["pareto_front"]):
                        # Strict validation
                        required_fields = ["selected_images", "cost", "cloudy_area", "max_incidence_angle", "min_resolutions_sum", "timestamp_s"]
                        for field in required_fields:
                            if field not in sol:
                                raise ValueError(f"pls_result solution {i} missing required field '{field}'")
                        
                        solutions.append({
                            "selected_images": sol["selected_images"],
                            "cost": sol["cost"],
                            "cloudy_area": sol["cloudy_area"],
                            "max_incidence_angle": sol["max_incidence_angle"],
                            "min_resolutions_sum": sol["min_resolutions_sum"],
                            "timestamp_s": sol["timestamp_s"]
                        })
                
                if not found_solutions:
                    raise ValueError(f"No solutions found in pls_result or exact_solver_result in: {result_file.relative_to(artifacts_dir)}")
            
            # Format 3: Direct SolverResult format
            elif "pareto_front" in result_data:
                if not result_data["pareto_front"]:
                    raise ValueError(f"Empty pareto_front in result.json: {result_file.relative_to(artifacts_dir)}")
                
                for i, sol in enumerate(result_data["pareto_front"]):
                    # Strict validation
                    required_fields = ["selected_images", "cost", "cloudy_area", "max_incidence_angle", "min_resolutions_sum", "timestamp_s"]
                    for field in required_fields:
                        if field not in sol:
                            raise ValueError(f"pareto_front solution {i} missing required field '{field}'")
                    
                    solutions.append({
                        "selected_images": sol["selected_images"],
                        "cost": sol["cost"],
                        "cloudy_area": sol["cloudy_area"],
                        "max_incidence_angle": sol["max_incidence_angle"],
                        "min_resolutions_sum": sol["min_resolutions_sum"],
                        "timestamp_s": sol["timestamp_s"]
                    })
            
            else:
                raise ValueError(f"Unknown JSON format in result.json: {result_file.relative_to(artifacts_dir)}. Expected 'solutions', 'pls_result'/'exact_solver_result', or 'pareto_front' fields")
            
            if not solutions:
                raise ValueError(f"No valid solutions found in: {result_file.relative_to(artifacts_dir)}")
            
            log.info(f"Found {len(solutions)} validated solutions for instance: {instance_name}")
            
            # Calculate variable bounds from solutions - fail if no variables found
            all_variables = set()
            for i, solution in enumerate(solutions):
                if not solution["selected_images"]:
                    raise ValueError(f"Solution {i} has empty selected_images list")
                all_variables.update(solution["selected_images"])
            
            if not all_variables:
                raise ValueError(f"No variables found in any solution in: {result_file.relative_to(artifacts_dir)}")
            
            variable_bounds = {
                "min_var": min(all_variables),
                "max_var": max(all_variables),
                "num_variables": len(all_variables)
            }
            
            log.info(f"Variable bounds: {variable_bounds['min_var']}-{variable_bounds['max_var']} ({variable_bounds['num_variables']} variables)")
            
            # Get objectives from result data - no defaults, fail if missing
            if "objectives" not in result_data:
                raise ValueError(f"Missing 'objectives' field in result.json: {result_file.relative_to(artifacts_dir)}")
            
            result_objectives = result_data["objectives"]
            if not result_objectives or not isinstance(result_objectives, list):
                raise ValueError(f"Invalid 'objectives' field: expected non-empty list, got {result_objectives}")
            
            # Map to the expected objective names for sims-problem - fail if unknown objective
            objective_mapping = {
                "min_cost": "min_cost",
                "cloud_coverage": "cloud_area", 
                "min_resolution": "min_resolution"
            }
            
            objectives = []
            for obj in result_objectives:
                if obj not in objective_mapping:
                    raise ValueError(f"Unknown objective '{obj}' in result.json. Supported: {list(objective_mapping.keys())}")
                objectives.append(objective_mapping[obj])
            
            obj_count = len(objectives)
            
            log.info(f"Using {obj_count} objectives: {objectives} (mapped from {result_objectives})")
            
            # Calculate objective bounds from solutions - no defaults, fail if missing data
            cost_values = []
            cloudy_area_values = []
            min_res_values = []
            
            for i, sol in enumerate(solutions):
                # Strict validation of all required fields
                if "cost" not in sol or sol["cost"] is None:
                    raise ValueError(f"Solution {i} missing required 'cost' field")
                if "cloudy_area" not in sol or sol["cloudy_area"] is None:
                    raise ValueError(f"Solution {i} missing required 'cloudy_area' field")
                if "min_resolutions_sum" not in sol or sol["min_resolutions_sum"] is None:
                    raise ValueError(f"Solution {i} missing required 'min_resolutions_sum' field")
                
                cost_values.append(int(sol["cost"]))
                cloudy_area_values.append(int(sol["cloudy_area"]))
                min_res_values.append(int(sol["min_resolutions_sum"]))
            
            log.info(f"Validated {len(cost_values)} complete solutions with all objective values")
            
            cost_bound = max(cost_values)
            cloudy_area_bound = max(cloudy_area_values)
            min_res_bound = max(min_res_values)
            
            objective_bounds = [[0, cost_bound], [0, cloudy_area_bound], [0, min_res_bound]]
            reference_point = [cost_bound + 1, cloudy_area_bound + 1, min_res_bound + 1]
            
            # Convert solutions to sims_problem.Solution objects - strict validation
            formatted_solutions = []
            for i, sol in enumerate(solutions):
                # Strict timestamp validation
                if "timestamp_s" not in sol or sol["timestamp_s"] is None:
                    raise ValueError(f"Solution {i} missing required 'timestamp_s' field")
                
                timestamp_us = int(sol["timestamp_s"] * 1_000_000)
                
                # Strict max_incidence_angle validation - -1 means None (not set), otherwise must be >= 0
                max_incidence_angle = None
                if sol["max_incidence_angle"] is not None:
                    if sol["max_incidence_angle"] == -1:
                        max_incidence_angle = None  # -1 explicitly means "not set"
                    elif sol["max_incidence_angle"] < 0:
                        raise ValueError(f"Solution {i} has invalid max_incidence_angle: {sol['max_incidence_angle']} (must be >= 0 or -1 for None)")
                    else:
                        max_incidence_angle = int(sol["max_incidence_angle"])
                
                # Strict selected_images validation
                if not isinstance(sol["selected_images"], list) or not sol["selected_images"]:
                    raise ValueError(f"Solution {i} has invalid selected_images: must be non-empty list")
                
                solution_obj = sims_problem.Solution.create(
                    selected_images=sol["selected_images"],
                    cost=int(sol["cost"]),
                    cloudy_area=int(sol["cloudy_area"]),
                    timestamp_us=timestamp_us,
                    max_incidence_angle=max_incidence_angle,
                    min_resolutions_sum=int(sol["min_resolutions_sum"]) if sol["min_resolutions_sum"] != 0 else 0
                )
                formatted_solutions.append(solution_obj)
            
            # Generate trace data using sims-problem
            trace_data = sims_problem.generate_trace(
                solutions=formatted_solutions,
                objectives=objectives,
                algorithm="MILP",
                num_objectives=obj_count,
                objective_bounds=objective_bounds,
                reference_point=reference_point
            )
            
            # Write trace file
            with open(trace_file, 'wb') as f:
                f.write(trace_data)
            
            log.info(f"✅ Generated trace file: {trace_file.relative_to(artifacts_dir)}")
            converted_count += 1
            
        except json.JSONDecodeError as e:
            log.error(f"Invalid JSON in {result_file.relative_to(artifacts_dir)}: {e}")
        except Exception as e:
            log.error(f"Failed to convert {result_file.relative_to(artifacts_dir)}: {e}")
            raise  # Re-raise the exception instead of continuing
    
    log.info(f"🎉 Successfully converted {converted_count} result.json files to traces")
    if skipped_count > 0:
        log.info(f"⏭️  Skipped {skipped_count} existing trace files")
    return 0


def convert_traces(artifacts_dir: Path, output_dir: Path | None = None, filter_pattern: str | None = None) -> int:
    """
    Convert CSV summary files to trace.tar.gz archives.
    
    Args:
        artifacts_dir: Directory containing test artifacts with CSV files
        output_dir: Directory to save trace files (default: same as artifacts_dir)
        filter_pattern: Regex pattern to filter artifact directories
    
    Returns:
        0 on success, 1 on failure
    """
    import re
    
    # Import the trace generation function from sims-problem
    try:
        sims_problem.generate_trace  # Just check if it exists
    except AttributeError as e:
        log.error(f"Failed to access generate_trace from sims_problem: {e}")
        log.error("Make sure sims-problem is built with the generate_trace function")
        return 1
    
    try:
        artifacts_dir = artifacts_dir.resolve()
        if not artifacts_dir.exists():
            log.error(f"Artifacts directory does not exist: {artifacts_dir}")
            return 1
        
        if output_dir is None:
            output_dir = artifacts_dir
        else:
            output_dir = output_dir.resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"🔍 Scanning artifacts directory: {artifacts_dir}")
        log.info(f"📁 Output directory: {output_dir}")
        
        # Find all directories containing test_summary.csv
        artifact_dirs = []
        for item in artifacts_dir.iterdir():
            if item.is_dir():
                csv_file = item / "test_summary.csv"
                if csv_file.exists():
                    # Apply filter if provided
                    if filter_pattern is None or re.search(filter_pattern, item.name):
                        artifact_dirs.append(item)
                    else:
                        log.debug(f"Skipping {item.name} (doesn't match filter: {filter_pattern})")
        
        if not artifact_dirs:
            log.warning("No artifact directories with test_summary.csv found")
            return 0
        
        log.info(f"📦 Found {len(artifact_dirs)} artifact directories to process")
        
        converted_count = 0
        for artifact_dir in sorted(artifact_dirs):
            try:
                log.info(f"🔄 Processing: {artifact_dir.name}")
                
                csv_file = artifact_dir / "test_summary.csv"
                trace_file = output_dir / f"{artifact_dir.name}_trace.tar.gz"
                
                # Skip if trace file already exists
                if trace_file.exists():
                    log.info(f"⏭️  Trace file already exists: {trace_file.name}")
                    continue
                
                # TODO: Implement CSV parsing for convert_traces
                log.warning(f"CSV parsing not yet implemented for: {csv_file.relative_to(artifacts_dir)}")
                
            except Exception as e:
                log.error(f"Failed to convert {artifact_dir.name}: {e}")
                import traceback
                log.debug(traceback.format_exc())
        
        log.info(f"🎉 Successfully converted {converted_count} artifact directories")
        return 0
    
    except Exception as e:
        log.error(f"Failed to convert traces: {e}")
        return 1


def main():
    """Main entry point for the SIMS Core CLI."""
    parser = argparse.ArgumentParser(
        description="SIMS Core CLI - Run hybrid experiments and process results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='COMMAND'
    )
    
    # Hybrid experiment command
    hybrid_parser = subparsers.add_parser(
        'run-hybrid',
        help='Run hybrid experiments with configurable ratio-step',
        description='Execute hybrid solver experiments with MILP/PLS ratio configurations'
    )
    
    hybrid_parser.add_argument(
        '--instances-dir',
        type=Path,
        default=None,
        help='Directory containing .dzn instance files (default: sims-problem/tests/data)'
    )
    
    hybrid_parser.add_argument(
        '--ratio-step',
        type=int,
        default=25,
        help='Step size for ratio configurations (default: 25, gives ratios 100:0, 75:25, 50:50, 25:75, 0:100)'
    )
    
    hybrid_parser.add_argument(
        '--timeout',
        type=int,
        default=600,
        help='Timeout in seconds for each solver run (default: 600)'
    )
    
    hybrid_parser.add_argument(
        '--solver-type',
        type=str,
        default='gurobi',
        choices=['gurobi', 'pls', 'ortools-py'],
        help='Type of solver to use (default: gurobi)'
    )
    
    hybrid_parser.add_argument(
        '--front-strategy',
        type=str,
        default='gpba-a',
        choices=['gpba-a', 'saugmecon', 'gavanelli', 'aneja-nair'],
        help='Front generation strategy (default: gpba-a)'
    )
    
    hybrid_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate the run without actual execution'
    )
    
    hybrid_parser.add_argument(
        '--iterations',
        type=int,
        default=1,
        help='Number of iterations per configuration (default: 1)'
    )
    
    hybrid_parser.add_argument(
        '--filter',
        type=str,
        help='Filter experiments by name using regex pattern (e.g., "lagos.*30")'
    )
    
    hybrid_parser.add_argument(
        '--skip-solved',
        action='store_true',
        help='Skip already solved experiments'
    )
    
    hybrid_parser.add_argument(
        '--results-dir',
        type=Path,
        help='Directory to save results (if different from experiment dirs)'
    )
    
    hybrid_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    hybrid_parser.add_argument(
        '--log-file',
        type=Path,
        help='Save logs to specified file (in addition to console output)'
    )
    
    hybrid_parser.add_argument(
        '--enable-trace',
        action='store_true',
        help='Enable PLS trace data collection for debugging and analysis'
    )
    
    # MILP-only experiment command
    milp_parser = subparsers.add_parser(
        'run-milp',
        help='Run MILP-only experiments (first phase only)',
        description='Execute pure MILP solver experiments without PLS heuristic phase'
    )
    
    milp_parser.add_argument(
        '--instances-dir',
        type=Path,
        help='Directory containing .dzn instance files (default: sims-problem/tests/data)'
    )
    
    milp_parser.add_argument(
        '--timeout',
        type=int,
        default=600,
        help='Timeout in seconds for each solver run (default: 600)'
    )
    
    milp_parser.add_argument(
        '--solver-type',
        choices=['gurobi', 'ortools-py'],
        default='gurobi',
        help='Type of MILP solver to use (default: gurobi)'
    )
    
    milp_parser.add_argument(
        '--front-strategy',
        choices=['gpba-a', 'saugmecon', 'gavanelli', 'aneja-nair'],
        default='gpba-a',
        help='Front generation strategy (default: gpba-a)'
    )
    
    milp_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate the run without actual execution'
    )
    
    milp_parser.add_argument(
        '--iterations',
        type=int,
        default=1,
        help='Number of iterations per configuration (default: 1)'
    )
    
    milp_parser.add_argument(
        '--filter',
        help='Filter experiments by name using regex pattern (e.g., "lagos.*30")'
    )
    
    milp_parser.add_argument(
        '--skip-solved',
        action='store_true',
        help='Skip already solved experiments'
    )
    
    milp_parser.add_argument(
        '--results-dir',
        type=Path,
        help='Directory to save results (if different from experiment dirs)'
    )
    
    milp_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    milp_parser.add_argument(
        '--log-file',
        type=Path,
        help='Save logs to specified file (in addition to console output)'
    )
    
    # Prepare experiments command
    prepare_parser = subparsers.add_parser(
        'prepare',
        help='Prepare experiments from AOI and image data',
        description='Create experiment folders from AOI and preselected image data'
    )
    
    prepare_parser.add_argument(
        '--aois-dir',
        type=Path,
        required=True,
        help='Directory containing AOI files'
    )
    
    prepare_parser.add_argument(
        '--images-dir',
        type=Path,
        required=True,
        help='Directory containing preselected image files'
    )
    
    prepare_parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directory to create experiment folders'
    )
    
    prepare_parser.add_argument(
        '--satellite-data-dir',
        type=Path,
        help='Optional satellite data directory'
    )
    
    prepare_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    prepare_parser.add_argument(
        '--log-file',
        type=Path,
        help='Save logs to specified file (in addition to console output)'
    )
    
    # Process results command
    process_parser = subparsers.add_parser(
        'process',
        help='Process experiment results',
        description='Process solved experiments and generate analysis'
    )
    
    process_parser.add_argument(
        'experiments_dir',
        type=Path,
        help='Directory containing solved experiments'
    )
    
    process_parser.add_argument(
        'output_dir',
        type=Path,
        help='Directory to save processed results'
    )
    
    process_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    process_parser.add_argument(
        '--log-file',
        type=Path,
        help='Save logs to specified file (in addition to console output)'
    )

    # Compress traces command
    compress_traces_parser = subparsers.add_parser(
        'compress-traces',
        help='Compress result.json files from test artifacts to trace.tar.gz format',
        description='Convert result.json files from test artifacts to binary trace archives'
    )
    
    compress_traces_parser.add_argument(
        '--artifacts-dir',
        type=Path,
        default=Path('test_artifacts'),
        help='Directory containing test artifacts with result.json files (default: test_artifacts)'
    )
    
    compress_traces_parser.add_argument(
        '--filter',
        type=str,
        help='Filter artifacts by name using regex pattern (e.g., "lagos.*30")'
    )
    
    compress_traces_parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing trace.tar.gz files'
    )
    
    compress_traces_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    compress_traces_parser.add_argument(
        '--log-file',
        type=Path,
        help='Save logs to specified file (in addition to console output)'
    )

    # Convert traces command
    convert_traces_parser = subparsers.add_parser(
        'convert-traces',
        help='Convert CSV summary files to trace.tar.gz archives',
        description='Convert CSV summary files from test artifacts to binary trace archives'
    )
    
    convert_traces_parser.add_argument(
        '--artifacts-dir',
        type=Path,
        default=Path('sims-solvers/test_artifacts'),
        help='Directory containing test artifacts with CSV files (default: sims-solvers/test_artifacts)'
    )
    
    convert_traces_parser.add_argument(
        '--output-dir',
        type=Path,
        help='Directory to save trace.tar.gz files (default: same as artifacts-dir)'
    )
    
    convert_traces_parser.add_argument(
        '--filter',
        type=str,
        help='Filter artifacts by name using regex pattern (e.g., "lagos.*30")'
    )
    
    convert_traces_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    convert_traces_parser.add_argument(
        '--log-file',
        type=Path,
        help='Save logs to specified file (in addition to console output)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging
    verbose = hasattr(args, 'verbose') and args.verbose
    log_file = getattr(args, 'log_file', None)
    setup_logging(verbose=verbose, log_file=log_file)
    
    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute the specified command
    try:
        if args.command == 'run-hybrid':
            # Convert string enums to proper types
            solver_type = SolverType.from_str(args.solver_type)
            front_strategy = FrontStrategy.from_str(args.front_strategy)
            
            # If no log file specified, create one automatically for hybrid experiments
            if not log_file and not args.dry_run:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = Path(f"sims_hybrid_{timestamp}.log")
                setup_logging(verbose=verbose, log_file=log_file)
                log.info(f"📝 Automatically created log file: {log_file}")
            
            return run_experiments(
                instances_dir=args.instances_dir,
                ratio_step=args.ratio_step,
                timeout_s=args.timeout,
                solver_type=solver_type,
                front_strategy=front_strategy,
                dry_run=args.dry_run,
                iter_count=args.iterations,
                instance_regex=args.filter,
                skip_solved=args.skip_solved,
                results_dir=args.results_dir,
                enable_trace=args.enable_trace,
            )
        
        elif args.command == 'run-milp':
            # Convert string enums to proper types
            solver_type = SolverType.from_str(args.solver_type)
            front_strategy = FrontStrategy.from_str(args.front_strategy)
            
            # If no log file specified, create one automatically for MILP experiments
            if not log_file and not args.dry_run:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = Path(f"sims_milp_{timestamp}.log")
                setup_logging(verbose=verbose, log_file=log_file)
                log.info(f"📝 Automatically created log file: {log_file}")
            
            return run_milp_experiments(
                instances_dir=args.instances_dir,
                timeout_s=args.timeout,
                solver_type=solver_type,
                front_strategy=front_strategy,
                dry_run=args.dry_run,
                iter_count=args.iterations,
                instance_regex=args.filter,
                skip_solved=args.skip_solved,
                results_dir=args.results_dir,
            )
        
        elif args.command == 'prepare':
            return prepare_experiments(
                aois_dir=args.aois_dir,
                images_dir=args.images_dir,
                output_dir=args.output_dir,
                satellite_data_dir=args.satellite_data_dir,
            )
        
        elif args.command == 'process':
            return process_results(
                experiments_dir=args.experiments_dir,
                output_dir=args.output_dir,
            )
        
        elif args.command == 'compress-traces':
            return compress_traces(
                artifacts_dir=args.artifacts_dir,
                filter_pattern=args.filter,
                overwrite=args.overwrite,
            )
        
        elif args.command == 'convert-traces':
            return convert_traces(
                artifacts_dir=args.artifacts_dir,
                output_dir=args.output_dir,
                filter_pattern=args.filter,
            )
        
        else:
            log.error(f"Unknown command: {args.command}")
            return 1
    
    except KeyboardInterrupt:
        log.info("Operation interrupted by user")
        return 1
    
    except Exception as e:
        log.error(f"Command failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
