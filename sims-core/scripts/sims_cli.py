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
    # Run hybrid experiment with ratio-step 25 (default)
    ./scripts/sims_cli.py run-hybrid --experiments-dir /path/to/experiments

    # Run with custom ratio-step
    ./scripts/sims_cli.py run-hybrid --experiments-dir /path/to/experiments --ratio-step 10

    # Run with timeout and instance filtering
    ./scripts/sims_cli.py run-hybrid --experiments-dir /path/to/experiments --timeout 300 --filter "lagos.*30"

    # Dry run to test configuration
    ./scripts/sims_cli.py run-hybrid --experiments-dir /path/to/experiments --dry-run

    # Prepare experiments from data
    ./scripts/sims_cli.py prepare --aois-dir /path/to/aois --images-dir /path/to/images --output-dir /path/to/experiments
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add the src directory to Python path to import sims.core
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

try:
    from sims.core.sims import experiment
    from sims.core.sims.experiment import Experiment
    from sims.core.sims.solver_config import FrontStrategy, SolverConfig, SolverType
except ImportError as e:
    print(f"Error importing sims.core modules: {e}")
    print("Make sure you're running this script from the sims-core directory")
    print("and that the sims-core package is properly installed.")
    sys.exit(1)

# Configure logging
log_format = "%(asctime)s %(name)-10s [%(levelname)-7s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
log = logging.getLogger("sims-cli")


def run_hybrid_experiments(
    experiments_dir: Path,
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
        experiments_dir: Directory containing experiment folders
        ratio_step: Step size for ratio configurations (default: 25, gives ratios 100:0, 75:25, 50:50, 25:75, 0:100)
        timeout_s: Timeout in seconds for each solver run
        solver_type: Type of solver to use (GUROBI, PLS, OR_TOOLS)
        front_strategy: Front generation strategy (GPBA_A, SAUGMECON, etc.)
        dry_run: If True, only simulate the run without actual execution
        iter_count: Number of iterations per configuration
        instance_regex: Regex pattern to filter experiments by name
        skip_solved: Skip already solved experiments
        results_dir: Directory to save results (if different from experiment dirs)
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    if not experiments_dir.exists():
        log.error(f"Experiments directory does not exist: {experiments_dir}")
        return 1
    
    if not experiments_dir.is_dir():
        log.error(f"Experiments path is not a directory: {experiments_dir}")
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
    
    # Find experiment directories
    experiment_dirs = []
    if instance_regex is not None:
        experiment_dirs = [
            experiment_dir
            for experiment_dir in experiments_dir.glob(instance_regex)
            if experiment_dir.is_dir()
        ]
        log.info(f'Selected {len(experiment_dirs)} instances matching regex "{instance_regex}": {[d.name for d in experiment_dirs]}')
    else:
        experiment_dirs = [experiment_dir for experiment_dir in experiments_dir.iterdir() if experiment_dir.is_dir()]
        log.info(f'Found {len(experiment_dirs)} experiment directories')
    
    if not experiment_dirs:
        log.warning("No experiment directories found")
        return 0
    
    # Prepare result directories if specified
    result_dirs = []
    if results_dir is not None:
        if results_dir.exists():
            log.error("Results directory already exists. Remove it before continuing.")
            return 1
        
        results_dir.mkdir(parents=True, exist_ok=True)
        result_dirs = [results_dir / experiment_dir.name for experiment_dir in experiment_dirs]
    else:
        result_dirs = [None for _ in experiment_dirs]
        # Clean old results if not skipping solved
        if not skip_solved:
            for experiment_dir in experiment_dirs:
                for subdir in experiment_dir.iterdir():
                    if subdir.is_dir() and subdir.name.startswith("solver_results_"):
                        log.info(f"Removing old results directory: {subdir}")
                        if not dry_run:
                            import shutil
                            shutil.rmtree(subdir)
    
    # Run experiments
    success_count = 0
    for experiment_idx, (experiment_dir, result_dir) in enumerate(zip(experiment_dirs, result_dirs)):
        log.info(f"~~~~~~ Solving experiment {experiment_dir.name} ({experiment_idx + 1}/{len(experiment_dirs)}) ~~~~~~")
        
        try:
            if dry_run:
                log.info(f"[DRY RUN] Would solve experiment: {experiment_dir.name}")
                log.info(f"[DRY RUN] Solver config: {solver_config.to_dict()}")
                success_count += 1
            else:
                experiment.solve(
                    experiment_dir=experiment_dir,
                    result_dir=result_dir,
                    solver_config=solver_config,
                    dry_run=dry_run,
                    iter_count=iter_count,
                    skip_solved=skip_solved,
                )
                success_count += 1
                log.info(f"Successfully solved experiment: {experiment_dir.name}")
        
        except Exception as e:
            log.error(f"Failed to solve experiment {experiment_dir.name}. Reason: {e}")
            if log.isEnabledFor(logging.DEBUG):
                import traceback
                traceback.print_exc()
    
    log.info(f"Completed {success_count}/{len(experiment_dirs)} experiments successfully")
    return 0 if success_count == len(experiment_dirs) else 1


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
                # Check if experiment is solved
                experiment_obj = Experiment.from_dir(experiment_dir)
                solver_config = SolverConfig.from_json(experiment_dir / "solver_config.json")
                
                if experiment_obj.is_solved(experiment_dir, solver_config):
                    log.info(f"Processing experiment: {experiment_dir.name}")
                    experiment_results = experiment_obj.parse_results(experiment_dir=experiment_dir)
                    experiment_results.process(output_dir=experiments_output_dir)
                    processed_count += 1
                else:
                    log.warning(f"Experiment {experiment_dir.name} is not solved, skipping")
            
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
        '--experiments-dir',
        type=Path,
        required=True,
        help='Directory containing experiment folders'
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
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging level
    if hasattr(args, 'verbose') and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
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
            
            return run_hybrid_experiments(
                experiments_dir=args.experiments_dir,
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
        if log.isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
