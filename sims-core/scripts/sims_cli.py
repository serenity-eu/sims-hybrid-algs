#!/usr/bin/env python3
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
    
    # Run on specific instance size
    uv run sims_cli.py run-hybrid --size 30
    
    # Filter specific instances
    uv run sims_cli.py run-hybrid --filter lagos_nigeria
    
    # Process experiment results
    uv run sims_cli.py process --experiments-dir /path/to/experiments
"""
"""
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

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


def setup_logging(verbose: bool = False, log_file: Optional[Path] = None):
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
    level = logging.DEBUG if verbose else logging.INFO
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


def run_hybrid_experiments(
    instances_dir: Optional[Path],
    ratio_step: int = 25,
    timeout_s: int = 600,
    solver_type: SolverType = SolverType.GUROBI,
    front_strategy: FrontStrategy = FrontStrategy.GPBA_A,
    dry_run: bool = False,
    iter_count: int = 1,
    instance_regex: Optional[str] = None,
    skip_solved: bool = False,
    results_dir: Optional[Path] = None,
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
                            solver.solve_with_two_phases(
                                problem_instance=problem_instance,
                                problem_path=dzn_dest,
                                experiment_path=ratio_dir,
                                solver_config=two_phase_config,
                                objectives=["min_cost", "cloud_coverage", "min_max_incidence_angle"],
                                dry_run=dry_run
                            )
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


def prepare_experiments(
    aois_dir: Path,
    images_dir: Path,
    output_dir: Path,
    satellite_data_dir: Optional[Path] = None,
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
            
            return run_hybrid_experiments(
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
