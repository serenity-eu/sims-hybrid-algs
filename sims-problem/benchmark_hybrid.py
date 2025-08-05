#!/usr/bin/env python3
"""
Hybrid Algorithm Benchmark Script

This script runs comprehensive benchmarks on the hybrid SIMS solver,
testing different MILP/PLS ratio configurations across multiple test instances.

Features:
- Tests ratio steps from (100,0) to (0,100) in increments of 10
- Runs multiple iterations for statistical significance
- Fancy TUI with nested progress bars
- Detailed data collection including timestamps and execution times
- JSON output with comprehensive results

Usage:
    # Recommended: auto-install dependencies with uv
    uv run --with rich benchmark_hybrid.py [--instances-dir DIR] [--output-dir DIR] [--iterations N]
    
    # Or install benchmark dependencies first
    uv pip install ".[benchmark]"
    python benchmark_hybrid.py [--instances-dir DIR] [--output-dir DIR] [--iterations N]
"""

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
import statistics

import sims_problem
from rich.console import Console
from rich.progress import (
    Progress, 
    BarColumn, 
    TextColumn, 
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    SpinnerColumn
)
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich import box


class BenchmarkResult:
    """Container for benchmark results of a single solver run."""
    
    def __init__(self):
        self.instance_name: str = ""
        self.iteration: int = 0
        self.ratio: Tuple[int, int] = (0, 0)
        self.total_runtime_seconds: float = 0.0
        self.milp_runtime_seconds: float = 0.0
        self.pls_runtime_seconds: float = 0.0
        self.milp_solutions: List[Dict[str, Any]] = []
        self.final_solutions: List[Dict[str, Any]] = []
        self.num_milp_solutions: int = 0
        self.num_final_solutions: int = 0
        self.error: str = ""
        self.timestamp: str = ""
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "instance_name": self.instance_name,
            "iteration": self.iteration,
            "ratio": self.ratio,
            "total_runtime_seconds": self.total_runtime_seconds,
            "milp_runtime_seconds": self.milp_runtime_seconds,
            "pls_runtime_seconds": self.pls_runtime_seconds,
            "milp_solutions": self.milp_solutions,
            "final_solutions": self.final_solutions,
            "num_milp_solutions": self.num_milp_solutions,
            "num_final_solutions": self.num_final_solutions,
            "error": self.error,
            "timestamp": self.timestamp
        }


class BenchmarkRunner:
    """Main benchmark runner with TUI progress tracking."""
    
    def __init__(self, instances_dir: Path, output_dir: Path, iterations: int = 10):
        self.instances_dir = instances_dir
        self.output_dir = output_dir
        self.iterations = iterations
        self.console = Console()
        
        # Generate ratio configurations (MILP%, PLS%)
        self.ratios = [(i, 100-i) for i in range(100, -1, -10)]
        
        # Find all .dzn instance files
        self.instance_files = list(instances_dir.glob("*.dzn"))
        if not self.instance_files:
            raise ValueError(f"No .dzn files found in {instances_dir}")
            
        self.results: List[BenchmarkResult] = []
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_instance(self, instance_file: Path) -> sims_problem.SimsDiscreteProblem:
        """Load a SIMS problem instance from .dzn file."""
        try:
            return sims_problem.SimsDiscreteProblem.from_dzn(str(instance_file))
        except Exception as e:
            raise RuntimeError(f"Failed to load instance {instance_file}: {e}")
    
    def run_hybrid_solver(
        self, 
        instance: sims_problem.SimsDiscreteProblem,
        ratio: Tuple[int, int],
        timeout_seconds: float = 300.0
    ) -> Tuple[List[sims_problem.Solution], float, float]:
        """
        Run the hybrid solver and return solutions with timing information.
        
        Returns:
            Tuple of (solutions, milp_time, pls_time)
        """
        # Configure solvers
        milp_config = sims_problem.MilpConfig(
            objectives=["min_cost", "cloud_coverage"],
            grid_points=50,
            bypass_coefficient=True,
            early_exit=True,
            flag_array=True,
            solver_name="cbc"
        )
        
        pls_config = sims_problem.PlsConfig(
            objectives=["min_cost", "cloud_coverage"],
            max_iterations=50000,
            is_deterministic=False,
            initial_population_size=100,
            neighborhood_size_min=1,
            neighborhood_size_max=6,
            plots=False
        )
        
        start_time = time.time()
        
        # Run hybrid solver
        solutions = sims_problem.solve_with_hybrid(
            instance,
            milp_config,
            pls_config,
            ratio,
            timedelta(seconds=timeout_seconds)
        )
        
        total_time = time.time() - start_time
        
        # Calculate phase times based on ratio
        milp_time = total_time * (ratio[0] / 100.0)
        pls_time = total_time * (ratio[1] / 100.0)
        
        return solutions, milp_time, pls_time
    
    def run_single_benchmark(
        self, 
        instance: sims_problem.SimsDiscreteProblem,
        instance_name: str,
        iteration: int,
        ratio: Tuple[int, int]
    ) -> BenchmarkResult:
        """Run a single benchmark and collect detailed results."""
        result = BenchmarkResult()
        result.instance_name = instance_name
        result.iteration = iteration
        result.ratio = ratio
        result.timestamp = datetime.now().isoformat()
        
        try:
            start_time = time.time()
            solutions, milp_time, pls_time = self.run_hybrid_solver(instance, ratio)
            result.total_runtime_seconds = time.time() - start_time
            result.milp_runtime_seconds = milp_time
            result.pls_runtime_seconds = pls_time
            
            # Convert solutions to dictionaries
            result.final_solutions = [sol.to_json() for sol in solutions]
            result.num_final_solutions = len(solutions)
            
            # For now, we don't have access to intermediate MILP solutions
            # This would require modifying the solver to return them
            result.milp_solutions = []
            result.num_milp_solutions = 0
            
        except Exception as e:
            result.error = str(e)
            
        return result
    
    def create_progress_layout(self, progress: Progress) -> Layout:
        """Create the main layout with progress bars and statistics."""
        
        # Create statistics table
        stats_table = Table(title="Benchmark Statistics", box=box.ROUNDED)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        completed_runs = len([r for r in self.results if not r.error])
        failed_runs = len([r for r in self.results if r.error])
        
        if completed_runs > 0:
            avg_runtime = statistics.mean([r.total_runtime_seconds for r in self.results if not r.error])
            stats_table.add_row("Completed Runs", str(completed_runs))
            stats_table.add_row("Failed Runs", str(failed_runs))
            stats_table.add_row("Avg Runtime", f"{avg_runtime:.2f}s")
        else:
            stats_table.add_row("Status", "Starting...")
        
        # Create layout
        layout = Layout()
        layout.split_row(
            Layout(Panel(progress, title="Benchmark Progress", border_style="blue"), size=80),
            Layout(Panel(stats_table, title="Statistics", border_style="green"), size=40)
        )
        
        return layout
    
    def run_benchmarks(self):
        """Run all benchmarks with fancy TUI progress tracking."""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            expand=True
        ) as progress:
            
            # Main progress tasks
            main_task = progress.add_task(
                f"[bold blue]Benchmarking {len(self.instance_files)} instances", 
                total=len(self.instance_files)
            )
            
            iteration_task = progress.add_task(
                "[bold green]Iterations", 
                total=self.iterations,
                visible=False
            )
            
            ratio_task = progress.add_task(
                "[bold yellow]Ratios", 
                total=len(self.ratios),
                visible=False
            )
            
            current_task = progress.add_task(
                "[bold white]Current Run", 
                total=1,
                visible=False
            )
            
            # Create live layout
            layout = self.create_progress_layout(progress)
            
            with Live(layout, console=self.console, refresh_per_second=4):
                
                for instance_idx, instance_file in enumerate(self.instance_files):
                    instance_name = instance_file.stem
                    
                    # Update main progress
                    progress.update(
                        main_task, 
                        description=f"[bold blue]Instance: {instance_name}",
                        completed=instance_idx
                    )
                    
                    # Load instance
                    try:
                        instance = self.load_instance(instance_file)
                    except Exception as e:
                        self.console.print(f"[red]Failed to load {instance_file}: {e}")
                        continue
                    
                    # Show iteration progress
                    progress.update(iteration_task, visible=True, completed=0)
                    
                    for iteration in range(self.iterations):
                        progress.update(
                            iteration_task,
                            description=f"[bold green]Iteration {iteration + 1}/{self.iterations}",
                            completed=iteration
                        )
                        
                        # Show ratio progress
                        progress.update(ratio_task, visible=True, completed=0)
                        
                        for ratio_idx, ratio in enumerate(self.ratios):
                            progress.update(
                                ratio_task,
                                description=f"[bold yellow]Ratio MILP:PLS = {ratio[0]}:{ratio[1]}",
                                completed=ratio_idx
                            )
                            
                            # Show current run
                            progress.update(
                                current_task,
                                visible=True,
                                description=f"[bold white]Running {instance_name} iter={iteration+1} ratio={ratio}",
                                completed=0
                            )
                            
                            # Run benchmark
                            result = self.run_single_benchmark(instance, instance_name, iteration, ratio)
                            self.results.append(result)
                            
                            # Update current run as completed
                            progress.update(current_task, completed=1)
                            
                            # Update layout with new statistics
                            layout = self.create_progress_layout(progress)
                        
                        # Complete ratio progress
                        progress.update(ratio_task, completed=len(self.ratios))
                    
                    # Complete iteration progress
                    progress.update(iteration_task, completed=self.iterations)
                    
                    # Hide sub-progress bars for next instance
                    progress.update(iteration_task, visible=False)
                    progress.update(ratio_task, visible=False)
                    progress.update(current_task, visible=False)
                
                # Complete main progress
                progress.update(main_task, completed=len(self.instance_files))
    
    def save_results(self):
        """Save benchmark results to JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detailed_file = self.output_dir / f"hybrid_benchmark_detailed_{timestamp}.json"
        with open(detailed_file, 'w') as f:
            json.dump(
                [result.to_dict() for result in self.results],
                f,
                indent=2,
                default=str
            )
        
        # Create summary statistics
        summary = self.create_summary_statistics()
        summary_file = self.output_dir / f"hybrid_benchmark_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.console.print("\n[green]Results saved:")
        self.console.print(f"  Detailed: {detailed_file}")
        self.console.print(f"  Summary:  {summary_file}")
    
    def create_summary_statistics(self) -> Dict[str, Any]:
        """Create summary statistics from benchmark results."""
        summary = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "instances_tested": len(self.instance_files),
                "iterations_per_ratio": self.iterations,
                "ratios_tested": self.ratios,
                "total_runs": len(self.results),
                "successful_runs": len([r for r in self.results if not r.error]),
                "failed_runs": len([r for r in self.results if r.error])
            },
            "performance_by_ratio": {},
            "performance_by_instance": {},
            "overall_statistics": {}
        }
        
        # Group results by ratio
        successful_results = [r for r in self.results if not r.error]
        
        if successful_results:
            for ratio in self.ratios:
                ratio_results = [r for r in successful_results if r.ratio == ratio]
                if ratio_results:
                    ratio_key = f"{ratio[0]}_{ratio[1]}"
                    summary["performance_by_ratio"][ratio_key] = {
                        "ratio": ratio,
                        "num_runs": len(ratio_results),
                        "avg_total_runtime": statistics.mean([r.total_runtime_seconds for r in ratio_results]),
                        "avg_solutions": statistics.mean([r.num_final_solutions for r in ratio_results]),
                        "std_runtime": statistics.stdev([r.total_runtime_seconds for r in ratio_results]) if len(ratio_results) > 1 else 0,
                        "min_runtime": min([r.total_runtime_seconds for r in ratio_results]),
                        "max_runtime": max([r.total_runtime_seconds for r in ratio_results])
                    }
            
            # Group results by instance
            for instance_file in self.instance_files:
                instance_name = instance_file.stem
                instance_results = [r for r in successful_results if r.instance_name == instance_name]
                if instance_results:
                    summary["performance_by_instance"][instance_name] = {
                        "num_runs": len(instance_results),
                        "avg_total_runtime": statistics.mean([r.total_runtime_seconds for r in instance_results]),
                        "avg_solutions": statistics.mean([r.num_final_solutions for r in instance_results]),
                        "best_ratio": max(instance_results, key=lambda x: x.num_final_solutions).ratio,
                        "fastest_ratio": min(instance_results, key=lambda x: x.total_runtime_seconds).ratio
                    }
            
            # Overall statistics
            summary["overall_statistics"] = {
                "avg_runtime": statistics.mean([r.total_runtime_seconds for r in successful_results]),
                "avg_solutions": statistics.mean([r.num_final_solutions for r in successful_results]),
                "total_runtime": sum([r.total_runtime_seconds for r in successful_results]),
                "best_performing_ratio": max(successful_results, key=lambda x: x.num_final_solutions).ratio,
                "fastest_ratio": min(successful_results, key=lambda x: x.total_runtime_seconds).ratio
            }
        
        return summary
    
    def print_summary(self):
        """Print a summary of benchmark results."""
        successful = [r for r in self.results if not r.error]
        failed = [r for r in self.results if r.error]
        
        self.console.print("\n[bold green]Benchmark Complete!")
        self.console.print(f"Total runs: {len(self.results)}")
        self.console.print(f"Successful: {len(successful)}")
        self.console.print(f"Failed: {len(failed)}")
        
        if successful:
            avg_runtime = statistics.mean([r.total_runtime_seconds for r in successful])
            avg_solutions = statistics.mean([r.num_final_solutions for r in successful])
            
            self.console.print(f"Average runtime: {avg_runtime:.2f}s")
            self.console.print(f"Average solutions found: {avg_solutions:.1f}")
            
            # Find best performing ratio
            best_result = max(successful, key=lambda x: x.num_final_solutions)
            self.console.print(f"Best ratio: {best_result.ratio} ({best_result.num_final_solutions} solutions)")
        
        if failed:
            self.console.print("\n[red]Failed runs:")
            for result in failed[:5]:  # Show first 5 failures
                self.console.print(f"  {result.instance_name} iter={result.iteration} ratio={result.ratio}: {result.error}")


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(description="Benchmark hybrid SIMS solver")
    parser.add_argument(
        "--instances-dir",
        type=Path,
        default=Path("tests/data"),
        help="Directory containing .dzn instance files"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path,
        default=Path("benchmark_results"),
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations per ratio configuration"
    )
    
    args = parser.parse_args()
    
    if not args.instances_dir.exists():
        print(f"Error: Instances directory {args.instances_dir} does not exist")
        return 1
    
    try:
        runner = BenchmarkRunner(args.instances_dir, args.output_dir, args.iterations)
        runner.run_benchmarks()
        runner.save_results()
        runner.print_summary()
        return 0
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
