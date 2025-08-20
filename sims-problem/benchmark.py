#!/usr/bin/env python3
"""
Multi-Algorithm Benchmark Script

This script runs comprehensive benchmarks on SIMS solvers with support for multiple algorithms:
- Hybrid SIMS solver (MILP + PLS with configurable ratios)
- Pure Pareto Local Search (PLS)
- Pure MILP solver with epsilon-constraint method

Features:
- Subcommand architecture for different algorithms
- Rich TUI with nested progress bars
- 3D Plotly visualizations of Pareto fronts
- Detailed data collection including timestamps and execution times
- JSON output with comprehensive results
- Solution validation and Pareto front quality assurance

Validation Features:
- Individual solution validation (checks solution structure, objective values, image selection validity)
- Pareto front validation (detects dominated solutions that shouldn't be in the front)
- Comprehensive validation reporting with statistics
- Optional validation (can be disabled for performance)

Usage:
    # Hybrid algorithm with ratio testing and validation
    python benchmark.py hybrid --instances-dir tests/data --ratio-step 20 --validate-solutions
    
    # Pure PLS algorithm with validation disabled for speed
    python benchmark.py pls --instances-dir tests/data --max-iterations 100000 --no-validate-solutions
    
    # Pure MILP algorithm with epsilon-constraint method
    python benchmark.py milp --instances-dir tests/data --grid-points 100 --solver-name gurobi --validate-solutions
    
    # Quick validation test on small instances
    python benchmark.py pls --size 30 --iterations 1 --timeout 10.0 --validate-solutions
"""

import argparse
import json
import re
import time
import statistics
import traceback
import sims_problem
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from abc import ABC, abstractmethod
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
from sims_problem import Solution

@dataclass
class SolverResult:
    """Dataclass for individual solver results."""
    solutions: list = field(default_factory=list)
    num_solutions: int = 0
    runtime_seconds: float = 0.0
    error: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "solutions": self.solutions,
            "num_solutions": self.num_solutions,
            "runtime_seconds": self.runtime_seconds,
            "error": self.error
        }

@dataclass
class ValidationResult:
    """Dataclass for individual solution validation results."""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    solution_index: Optional[int] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        result = {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings
        }
        if self.solution_index is not None:
            result["solution_index"] = self.solution_index
        return result


@dataclass 
class DominationDetail:
    """Dataclass for domination relationship details."""
    dominating_solution_index: int
    dominated_solution_index: int
    dominating_objectives: dict[str, int | None]
    dominated_objectives: dict[str, int | None]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "dominating_solution_index": self.dominating_solution_index,
            "dominated_solution_index": self.dominated_solution_index,
            "dominating_objectives": self.dominating_objectives,
            "dominated_objectives": self.dominated_objectives
        }


@dataclass
class ParetoValidationResult:
    """Dataclass for Pareto front validation results."""
    is_valid_pareto: bool
    dominated_solutions: list[int]
    domination_pairs: list[tuple[int, int]]
    num_solutions: int
    sample_domination_details: list[DominationDetail] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "is_valid_pareto": self.is_valid_pareto,
            "dominated_solutions": self.dominated_solutions,
            "domination_pairs": self.domination_pairs,
            "num_solutions": self.num_solutions,
            "sample_domination_details": [detail.to_dict() for detail in self.sample_domination_details]
        }


@dataclass
class ValidationSummary:
    """Dataclass for validation summary statistics."""
    total_solutions: int
    valid_solutions: int
    invalid_solutions: int
    solutions_with_warnings: int
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "total_solutions": self.total_solutions,
            "valid_solutions": self.valid_solutions,
            "invalid_solutions": self.invalid_solutions,
            "solutions_with_warnings": self.solutions_with_warnings
        }


@dataclass
class ComprehensiveValidationReport:
    """Dataclass for comprehensive validation report."""
    overall_valid: bool
    solution_validation: list[ValidationResult]
    pareto_validation: ParetoValidationResult
    summary: ValidationSummary
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "overall_valid": self.overall_valid,
            "solution_validation": [val.to_dict() for val in self.solution_validation],
            "pareto_validation": self.pareto_validation.to_dict(),
            "summary": self.summary.to_dict()
        }


@dataclass
class BenchmarkInfo:
    """Dataclass for benchmark information."""
    timestamp: str
    instances_tested: int
    total_runs: int
    successful_runs: int
    failed_runs: int
    # Hybrid-specific fields
    iterations_per_ratio: Optional[int] = None
    ratios_tested: Optional[list[tuple[int, int]]] = None
    # PLS-specific fields
    algorithm: Optional[str] = None
    iterations_per_instance: Optional[int] = None
    max_pls_iterations: Optional[int] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        result = {
            "timestamp": self.timestamp,
            "instances_tested": self.instances_tested,
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs
        }
        if self.iterations_per_ratio is not None:
            result["iterations_per_ratio"] = self.iterations_per_ratio
        if self.ratios_tested is not None:
            result["ratios_tested"] = self.ratios_tested
        if self.algorithm is not None:
            result["algorithm"] = self.algorithm
        if self.iterations_per_instance is not None:
            result["iterations_per_instance"] = self.iterations_per_instance
        if self.max_pls_iterations is not None:
            result["max_pls_iterations"] = self.max_pls_iterations
        return result


@dataclass
class PerformanceByRatio:
    """Dataclass for performance statistics by ratio."""
    ratio: tuple[int, int]
    num_runs: int
    avg_total_runtime: float
    avg_solutions: float
    std_runtime: float
    min_runtime: float
    max_runtime: float
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "ratio": self.ratio,
            "num_runs": self.num_runs,
            "avg_total_runtime": self.avg_total_runtime,
            "avg_solutions": self.avg_solutions,
            "std_runtime": self.std_runtime,
            "min_runtime": self.min_runtime,
            "max_runtime": self.max_runtime
        }


@dataclass
class PerformanceByInstance:
    """Dataclass for performance statistics by instance."""
    num_runs: int
    avg_runtime: float
    avg_solutions: float
    std_runtime: float
    min_runtime: float
    max_runtime: float
    min_solutions: int
    max_solutions: int
    # Hybrid-specific fields
    best_ratio: Optional[tuple[int, int]] = None
    fastest_ratio: Optional[tuple[int, int]] = None
    # Alternative names for hybrid compatibility
    avg_total_runtime: Optional[float] = None
    
    def __post_init__(self):
        # Set avg_total_runtime as alias for avg_runtime for hybrid compatibility
        if self.avg_total_runtime is None:
            self.avg_total_runtime = self.avg_runtime
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        result = {
            "num_runs": self.num_runs,
            "avg_runtime": self.avg_runtime,
            "avg_solutions": self.avg_solutions,
            "std_runtime": self.std_runtime,
            "min_runtime": self.min_runtime,
            "max_runtime": self.max_runtime,
            "min_solutions": self.min_solutions,
            "max_solutions": self.max_solutions
        }
        if self.best_ratio is not None:
            result["best_ratio"] = self.best_ratio
        if self.fastest_ratio is not None:
            result["fastest_ratio"] = self.fastest_ratio
        if self.avg_total_runtime is not None:
            result["avg_total_runtime"] = self.avg_total_runtime
        return result


@dataclass
class OverallStatistics:
    """Dataclass for overall benchmark statistics."""
    avg_runtime: float
    avg_solutions: float
    total_runtime: float
    # Hybrid-specific fields
    best_performing_ratio: Optional[tuple[int, int]] = None
    fastest_ratio: Optional[tuple[int, int]] = None
    # PLS-specific fields
    min_solutions: Optional[int] = None
    max_solutions: Optional[int] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        result: dict[str, Any] = {
            "avg_runtime": self.avg_runtime,
            "avg_solutions": self.avg_solutions,
            "total_runtime": self.total_runtime
        }
        if self.best_performing_ratio is not None:
            result["best_performing_ratio"] = self.best_performing_ratio
        if self.fastest_ratio is not None:
            result["fastest_ratio"] = self.fastest_ratio
        if self.min_solutions is not None:
            result["min_solutions"] = self.min_solutions
        if self.max_solutions is not None:
            result["max_solutions"] = self.max_solutions
        return result


@dataclass
class BenchmarkSummaryStatistics:
    """Dataclass for complete benchmark summary statistics."""
    benchmark_info: BenchmarkInfo
    overall_statistics: OverallStatistics
    performance_by_instance: dict[str, PerformanceByInstance]
    validation_statistics: dict[str, Any]
    # Hybrid-specific field
    performance_by_ratio: Optional[dict[str, PerformanceByRatio]] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        result = {
            "benchmark_info": self.benchmark_info.to_dict(),
            "overall_statistics": self.overall_statistics.to_dict(),
            "performance_by_instance": {k: v.to_dict() for k, v in self.performance_by_instance.items()},
            "validation_statistics": self.validation_statistics
        }
        if self.performance_by_ratio is not None:
            result["performance_by_ratio"] = {k: v.to_dict() for k, v in self.performance_by_ratio.items()}
        return result


@dataclass
class BenchmarkResult:
    """Dataclass for benchmark results of a single solver run."""
    instance_name: str = ""
    iteration: int = 0
    ratio: tuple[int, int] = (0, 0)
    total_runtime_seconds: float = 0.0
    milp_runtime_seconds: float = 0.0
    pls_runtime_seconds: float = 0.0
    milp_solutions: list[sims_problem.Solution] = field(default_factory=list)
    final_solutions: list[Solution] = field(default_factory=list)
    explored_solutions: list[Solution] = field(default_factory=list)
    num_milp_solutions: int = 0
    num_final_solutions: int = 0
    num_explored_solutions: int = 0
    error: str = ""
    finished_at: datetime = field(default_factory=datetime.now)
    validation_results: Optional[ComprehensiveValidationReport] = None
    trace_data: Optional[bytes] = None  # Optimization trace archive data
        
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization with compact solution format."""
        
        def convert_solution_to_compact(solution: Solution) -> dict:
            """Convert a solution to compact format: {"s":[indices],"o":[objectives],"t":timestamp_us}"""
            return {
                "s": list(solution.selected_images),  # selected_images as array of indices
                "o": [solution.cost, solution.cloudy_area, solution.max_incidence_angle or 0],  # objectives array
                "t": int(solution.timestamp.total_seconds() * 1_000_000)  # timestamp in microseconds
            }
        
        return {
            "instance_name": self.instance_name,
            "iteration": self.iteration,
            "ratio": self.ratio,
            "total_runtime_seconds": self.total_runtime_seconds,
            "milp_runtime_seconds": self.milp_runtime_seconds,
            "pls_runtime_seconds": self.pls_runtime_seconds,
            "objective_names": ["min_cost", "cloud_coverage", "max_incidence_angle"],  # Top-level field with objective names
            "milp_solutions": [convert_solution_to_compact(sol) for sol in self.milp_solutions],
            "final_solutions": [convert_solution_to_compact(sol) for sol in self.final_solutions],
            "explored_solutions": [convert_solution_to_compact(sol) for sol in self.explored_solutions],
            "num_milp_solutions": self.num_milp_solutions,
            "num_final_solutions": self.num_final_solutions,
            "num_explored_solutions": self.num_explored_solutions,
            "error": self.error,
            "timestamp": self.finished_at,
            "validation_results": self.validation_results.to_dict() if self.validation_results else {}
        }


class SolutionValidator:
    """Validates solutions and Pareto fronts for benchmark quality assurance."""
    
    def __init__(self, console=None):
        self.console = console or Console()
    
    def validate_solution(self, solution: Solution, instance: sims_problem.SimsDiscreteProblem) -> ValidationResult:
        """
        Validate a single solution against the problem instance.
        
        Returns a ValidationResult with validation details.
        """
        validation_result = ValidationResult(is_valid=True)
        
        try:
            # Get selected images as a list using the proper method
            # Note: solution.selected_images returns a set, but we need a list for validation
            try:
                selected_images = solution.get_selected_images_list()
            except AttributeError:
                # Fallback: convert set to list if get_selected_images_list() doesn't exist
                selected_images = list(solution.selected_images)
            
            if not isinstance(selected_images, list):
                validation_result.errors.append(f"selected_images must be a list, got {type(selected_images)}")
                validation_result.is_valid = False
                return validation_result
            
            # Check if selected images are within valid range
            for img_idx in selected_images:
                if not isinstance(img_idx, int) or img_idx < 0 or img_idx >= instance.num_images:
                    validation_result.errors.append(f"Invalid image index: {img_idx} (must be 0 <= idx < {instance.num_images})")
                    validation_result.is_valid = False
            
            # Check for duplicate images
            if len(selected_images) != len(set(selected_images)):
                validation_result.errors.append("Duplicate images in selection")
                validation_result.is_valid = False
            
            # Validate objective values are non-negative
            if solution.cost < 0:
                validation_result.errors.append(f"Cost cannot be negative: {solution.cost}")
                validation_result.is_valid = False
            
            if solution.cloudy_area < 0:
                validation_result.errors.append(f"Cloudy area cannot be negative: {solution.cloudy_area}")
                validation_result.is_valid = False

            if solution.max_incidence_angle is not None and solution.max_incidence_angle < 0:
                validation_result.errors.append(f"Max incidence angle cannot be negative: {solution.max_incidence_angle}")
                validation_result.is_valid = False
            
            # Warn if solution is empty
            if len(selected_images) == 0:
                validation_result.warnings.append("Solution contains no selected images")
            
        except Exception as e:
            validation_result.errors.append(f"Exception during validation: {str(e)}")
            validation_result.is_valid = False
        
        return validation_result
    
    def dominates(self, sol1: Solution, sol2: Solution) -> bool:
        """
        Check if solution 1 dominates solution 2 (for minimization objectives).
        Sol1 dominates sol2 if sol1 is better or equal in all objectives and strictly better in at least one.
        """
        objectives = [(sol1.cost, sol2.cost), 
                      (sol1.cloudy_area, sol2.cloudy_area), 
                      (sol1.max_incidence_angle, sol2.max_incidence_angle)]
        
        better_in_all = True
        better_in_at_least_one = False
        
        for val1, val2 in objectives:
            if val1 > val2:  # Sol1 is worse in this objective
                better_in_all = False
                break
            elif val1 < val2:  # Sol1 is better in this objective
                better_in_at_least_one = True
        
        return better_in_all and better_in_at_least_one
    
    def validate_pareto_front(self, solutions: list[Solution]) -> ParetoValidationResult:
        """
        Validate that the solutions form a valid Pareto front (no dominated solutions).
        """
        validation_result = ParetoValidationResult(
            is_valid_pareto=True,
            dominated_solutions=[],
            domination_pairs=[],
            num_solutions=len(solutions)
        )
        
        if len(solutions) <= 1:
            return validation_result
        
        # Check each pair of solutions for domination
        domination_count = 0
        for i in range(len(solutions)):
            for j in range(len(solutions)):
                if i != j and self.dominates(solutions[i], solutions[j]):
                    validation_result.domination_pairs.append((i, j))
                    if j not in validation_result.dominated_solutions:
                        validation_result.dominated_solutions.append(j)
                        validation_result.is_valid_pareto = False

                    # Collect detailed information for first few domination cases for debugging
                    if domination_count < 5:
                        # Throw exception if max_incidence_angle is None for either solution
                        if solutions[i].max_incidence_angle is None or solutions[j].max_incidence_angle is None:
                            raise ValueError(
                                f"max_incidence_angle is None for solution {i if solutions[i].max_incidence_angle is None else j} "
                                f"in Pareto front validation"
                            )
                        domination_detail = DominationDetail(
                            dominating_solution_index=i,
                            dominated_solution_index=j,
                            dominating_objectives={
                                "cost": solutions[i].cost,
                                "cloudy_area": solutions[i].cloudy_area,
                                "max_incidence_angle": solutions[i].max_incidence_angle
                            },
                            dominated_objectives={
                                "cost": solutions[j].cost,
                                "cloudy_area": solutions[j].cloudy_area,
                                "max_incidence_angle": solutions[j].max_incidence_angle
                            }
                        )
                        validation_result.sample_domination_details.append(domination_detail)
                        domination_count += 1
        
        return validation_result
    
    def validate_benchmark_result(self, result: BenchmarkResult, instance: sims_problem.SimsDiscreteProblem) -> ComprehensiveValidationReport:
        """
        Comprehensive validation of a benchmark result.
        
        Returns complete validation report including individual solution validation
        and Pareto front validation.
        """
        solution_validations = []
        
        # Validate each individual solution
        for i, solution in enumerate(result.final_solutions):
            sol_validation = self.validate_solution(solution, instance)
            sol_validation.solution_index = i
            solution_validations.append(sol_validation)
        
        # Calculate summary statistics
        valid_solutions = [val for val in solution_validations if val.is_valid]
        invalid_solutions = [val for val in solution_validations if not val.is_valid]
        solutions_with_warnings = [val for val in solution_validations if val.warnings]
        
        summary = ValidationSummary(
            total_solutions=len(result.final_solutions),
            valid_solutions=len(valid_solutions),
            invalid_solutions=len(invalid_solutions),
            solutions_with_warnings=len(solutions_with_warnings)
        )
        
        # Validate Pareto front if we have valid solutions
        valid_solution_data = [result.final_solutions[i] for i, val in enumerate(solution_validations) if val.is_valid]
        pareto_validation = self.validate_pareto_front(valid_solution_data)
        
        # Overall validity
        overall_valid = len(invalid_solutions) == 0 and pareto_validation.is_valid_pareto
        
        return ComprehensiveValidationReport(
            overall_valid=overall_valid,
            solution_validation=solution_validations,
            pareto_validation=pareto_validation,
            summary=summary
        )


class BenchmarkRunner(ABC):
    """Abstract base class for benchmark runners."""
    
    def __init__(self, instances_dir: Path, output_dir: Path, iterations: int = 1, use_tui: bool = False, size_limit: Optional[int] = None, name_filter: Optional[str] = None, timeout: float = 300.0, validate_solutions: bool = True, include_perfect: bool = False):
        self.instances_dir = instances_dir
        self.output_dir = output_dir
        self.iterations = iterations
        self.use_tui = use_tui
        self.size_limit = size_limit
        self.name_filter = name_filter
        self.timeout = timeout
        self.validate_solutions = validate_solutions
        self.include_perfect = include_perfect
        self.console = Console()
        self.validator = SolutionValidator(self.console) if validate_solutions else None
        
        # Find all .dzn instance files
        self.instance_files = list(instances_dir.glob("*.dzn"))
        if not self.instance_files:
            raise ValueError(f"No .dzn files found in {instances_dir}")
        
        # Filter instances by size if specified
        if self.size_limit is not None:
            self.instance_files = self._filter_instances_by_size()
            
        # Filter instances by name pattern if specified
        if self.name_filter is not None:
            self.instance_files = self._filter_instances_by_name()
            
        self.results: list[BenchmarkResult] = []
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def _filter_instances_by_size(self) -> list[Path]:
        """Filter instance files by size limit."""
        filtered_files = []
        for instance_file in self.instance_files:
            # Extract size from filename (e.g., "tokyo_bay_30.dzn" -> 30)
            try:
                # Get the stem (filename without extension) and split by underscore
                name_parts = instance_file.stem.split('_')
                # Try to find a number at the end of the filename
                size_from_filename = None
                for part in reversed(name_parts):
                    if part.isdigit():
                        size_from_filename = int(part)
                        break
                
                if size_from_filename is not None and self.size_limit is not None and size_from_filename <= self.size_limit:
                    filtered_files.append(instance_file)
                elif size_from_filename is not None:
                    self.console.print(f"Skipping {instance_file.name}: {size_from_filename} images > {self.size_limit}")
                else:
                    # Fallback: load instance if we can't parse size from filename
                    self.console.print(f"Warning: Could not parse size from filename {instance_file.name}, checking file...")
                    try:
                        instance = self.load_instance(instance_file)
                        if self.size_limit is not None and instance.num_images <= self.size_limit:
                            filtered_files.append(instance_file)
                        else:
                            self.console.print(f"Skipping {instance_file.name}: {instance.num_images} images > {self.size_limit}")
                    except Exception as e:
                        self.console.print(f"Warning: Could not check size of {instance_file.name}: {e}")
            except Exception as e:
                self.console.print(f"Warning: Error processing {instance_file.name}: {e}")
        
        if not filtered_files:
            raise ValueError(f"No .dzn files found with num_images <= {self.size_limit}")
        
        self.console.print(f"Found {len(filtered_files)} instances with <= {self.size_limit} images (based on filename)")
        return filtered_files
    
    def _filter_instances_by_name(self) -> list[Path]:
        """Filter instance files by name pattern using regex."""
        if self.name_filter is None:
            return self.instance_files
            
        filtered_files = []
        try:
            pattern = re.compile(self.name_filter, re.IGNORECASE)
            for instance_file in self.instance_files:
                if pattern.search(instance_file.stem):
                    filtered_files.append(instance_file)
                else:
                    self.console.print(f"Skipping {instance_file.name}: doesn't match filter '{self.name_filter}'")
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{self.name_filter}': {e}")
        
        if not filtered_files:
            raise ValueError(f"No .dzn files found matching pattern '{self.name_filter}'")
        
        self.console.print(f"Found {len(filtered_files)} instances matching pattern '{self.name_filter}'")
        return filtered_files
    
    
    def extract_solution_objectives(self, solutions: list[Solution]) -> tuple[list[float], list[float], list[float]]:
        """Extract the three objectives from solutions data."""
        costs = []
        cloudy_areas = []
        incidence_angles = []
        
        for solution in solutions:
            costs.append(solution.cost)
            cloudy_areas.append(solution.cloudy_area)
            incidence_angles.append(solution.max_incidence_angle)
        
        return costs, cloudy_areas, incidence_angles
    
    def extract_solution_objectives_4d(self, solutions: list[Solution]) -> tuple[list[float], list[float], list[float], list[float]]:
        """Extract all four objectives from solutions data for 4D problems."""
        costs = []
        cloudy_areas = []
        resolutions = []
        incidence_angles = []
        
        for solution in solutions:
            costs.append(solution.cost)
            cloudy_areas.append(solution.cloudy_area)
            resolutions.append(solution.min_resolutions_sum if solution.min_resolutions_sum is not None else 0)
            incidence_angles.append(solution.max_incidence_angle if solution.max_incidence_angle is not None else 0)
        
        return costs, cloudy_areas, resolutions, incidence_angles
    
    def extract_solution_timestamps(self, solutions: list[Solution]) -> list[timedelta]:
        """Extract timestamps from solutions data."""
        timestamps = []
        for solution in solutions:
            timestamps.append(solution.timestamp)
        return timestamps
    
    def load_instance(self, instance_file: Path) -> sims_problem.SimsDiscreteProblem:
        """Load a SIMS problem instance from .dzn file."""
        try:
            return sims_problem.SimsDiscreteProblem.from_dzn(str(instance_file))
        except Exception as e:
            raise RuntimeError(f"Failed to load instance {instance_file}: {e}")
    
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
    
    def _calculate_validation_statistics(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """Calculate validation statistics from results."""
        total_with_validation = len([r for r in results if r.validation_results])
        total_valid = len([r for r in results 
                          if r.validation_results and r.validation_results.overall_valid])
        
        total_solutions = sum([r.num_final_solutions for r in results])
        total_invalid_solutions = sum([
            r.validation_results.summary.invalid_solutions 
            for r in results if r.validation_results
        ])
        total_warned_solutions = sum([
            r.validation_results.summary.solutions_with_warnings 
            for r in results if r.validation_results
        ])
        
        total_pareto_violations = len([
            r for r in results 
            if r.validation_results and 
            not r.validation_results.pareto_validation.is_valid_pareto
        ])
        
        return {
            "runs_with_validation": total_with_validation,
            "valid_runs": total_valid,
            "invalid_runs": total_with_validation - total_valid,
            "total_solutions": total_solutions,
            "invalid_solutions": total_invalid_solutions,
            "solutions_with_warnings": total_warned_solutions,
            "pareto_front_violations": total_pareto_violations,
            "run_validation_rate": (total_valid / total_with_validation * 100) if total_with_validation > 0 else 0,
            "solution_validity_rate": ((total_solutions - total_invalid_solutions) / total_solutions * 100) if total_solutions > 0 else 0
        }
    
    def print_validation_summary(self, successful_results: list[BenchmarkResult]):
        """Print a summary of validation results."""
        if not successful_results:
            return
        
        self.console.print("\n[bold cyan]🔍 Validation Summary:")
        
        # Count validation results
        total_with_validation = len([r for r in successful_results if r.validation_results])
        total_valid = len([r for r in successful_results 
                          if r.validation_results and r.validation_results.overall_valid])
        total_invalid = total_with_validation - total_valid
        
        total_solutions = sum([r.num_final_solutions for r in successful_results])
        total_invalid_solutions = sum([
            r.validation_results.summary.invalid_solutions 
            for r in successful_results if r.validation_results
        ])
        total_warned_solutions = sum([
            r.validation_results.summary.solutions_with_warnings 
            for r in successful_results if r.validation_results
        ])
        
        # Count Pareto front violations
        total_pareto_violations = len([
            r for r in successful_results 
            if r.validation_results and 
            not r.validation_results.pareto_validation.is_valid_pareto
        ])
        
        self.console.print(f"  📊 Runs validated: {total_with_validation}/{len(successful_results)}")
        self.console.print(f"  ✅ Valid runs: {total_valid}")
        self.console.print(f"  ❌ Invalid runs: {total_invalid}")
        self.console.print(f"  📈 Total solutions: {total_solutions}")
        self.console.print(f"  🚫 Invalid solutions: {total_invalid_solutions}")
        self.console.print(f"  ⚠️  Solutions with warnings: {total_warned_solutions}")
        self.console.print(f"  🎯 Pareto front violations: {total_pareto_violations}")
        
        if total_invalid > 0:
            self.console.print(f"\n[yellow]⚠️  {total_invalid} runs had validation issues - check detailed results for specifics")
        
        if total_pareto_violations > 0:
            self.console.print(f"[red]❌ {total_pareto_violations} runs had non-Pareto fronts (dominated solutions present)")
        
        if total_invalid == 0 and total_pareto_violations == 0:
            self.console.print("[green]✅ All validated runs passed quality checks!")
            
        # Calculate validation statistics
        validation_rate = (total_valid / total_with_validation * 100) if total_with_validation > 0 else 0
        solution_validity_rate = ((total_solutions - total_invalid_solutions) / total_solutions * 100) if total_solutions > 0 else 0
        
        self.console.print(f"\n  📊 Run validation rate: {validation_rate:.1f}%")
        self.console.print(f"  📊 Solution validity rate: {solution_validity_rate:.1f}%")
    
    @abstractmethod
    def run_benchmarks(self):
        """Run all benchmarks. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def run_single_benchmark(self, instance: sims_problem.SimsDiscreteProblem, instance_name: str, iteration: int, *args) -> BenchmarkResult:
        """Run a single benchmark. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def create_summary_statistics(self) -> BenchmarkSummaryStatistics:
        """Create summary statistics. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def save_results(self):
        """Save benchmark results. Must be implemented by subclasses."""
        pass
    
    def save_instance_results(self, instance_name: str, instance_results: list[BenchmarkResult]):
        """
        Save results for a single instance (trace archives, summary, and visualization).
        Default implementation - subclasses can override for custom behavior.
        
        Args:
            instance_name: Name of the instance
            instance_results: List of benchmark results for this instance only
        """
        if not instance_results:
            return
            
        finished_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save trace archives for each result that has trace data
        trace_files = []
        for i, result in enumerate(instance_results):
            if result.trace_data:
                trace_file = self.output_dir / f"{instance_name}_trace_{i}_{finished_at}.tar.gz"
                with open(trace_file, 'wb') as f:
                    f.write(result.trace_data)
                trace_files.append(trace_file)
        
        # Create and save summary statistics for this instance
        summary = self.create_instance_summary_statistics(instance_name, instance_results)
        summary_file = self.output_dir / f"{instance_name}_summary_{finished_at}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)
        
        # Create visualization for this instance
        self.create_instance_3d_visualization(instance_name, instance_results)
        
        self.console.print(f"\n[green]Instance {instance_name} results saved:")
        if trace_files:
            for trace_file in trace_files:
                self.console.print(f"  Trace:    {trace_file}")
        self.console.print(f"  Summary:  {summary_file}")
    
    @abstractmethod
    def create_instance_summary_statistics(self, instance_name: str, instance_results: list[BenchmarkResult]) -> BenchmarkSummaryStatistics:
        """Create summary statistics for a single instance. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def create_instance_3d_visualization(self, instance_name: str, instance_results: list[BenchmarkResult]) -> bool:
        """Create 3D visualization for a single instance. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Get the name of the algorithm for this runner."""
        pass
    
    @abstractmethod
    def get_visualization_filename_suffix(self) -> str:
        """Get the filename suffix for visualizations (e.g., 'hybrid', 'pls')."""
        pass
    
    def _collect_visualization_data(self, successful_results: list[BenchmarkResult], instance_name: Optional[str] = None) -> tuple:
        """
        Collect and organize solutions data for visualization.
        
        Args:
            successful_results: List of successful benchmark results
            instance_name: Name of the instance (used to load JSONL solutions)
        
        Returns:
            Tuple of (explored_data, final_data, jsonl_data, all_timestamps_range) where each data is
            (costs, cloudy_areas, incidence_angles, timestamps, solution_details, colors, solver_types)
        """
        # Collect all explored solutions and final solutions
        all_explored_costs = []
        all_explored_cloudy_areas = []
        all_explored_incidence_angles = []
        all_explored_timestamps = []
        explored_solution_details = []
        explored_colors = []
        explored_solver_types = []
        
        all_final_costs = []
        all_final_cloudy_areas = []
        all_final_incidence_angles = []
        all_final_timestamps = []
        final_solution_details = []
        final_colors = []
        final_solver_types = []
        
        # Collect JSONL solutions data
        all_jsonl_costs = []
        all_jsonl_cloudy_areas = []
        all_jsonl_incidence_angles = []
        jsonl_solution_details = []
        
        # Determine algorithm type for coloring
        algorithm_name = self.get_algorithm_name().lower()
        is_hybrid = 'hybrid' in algorithm_name
        is_gurobi = 'gurobi' in algorithm_name
        is_pls = 'pls' in algorithm_name and not is_hybrid
        
        for result in successful_results:
            # Process explored solutions
            if result.explored_solutions:
                exp_costs, exp_cloudy_areas, exp_incidence_angles = self.extract_solution_objectives(result.explored_solutions)
                exp_timestamps = self.extract_solution_timestamps(result.explored_solutions)
                
                all_explored_costs.extend(exp_costs)
                all_explored_cloudy_areas.extend(exp_cloudy_areas)
                all_explored_incidence_angles.extend(exp_incidence_angles)
                all_explored_timestamps.extend(exp_timestamps)
                
                # Color explored solutions based on algorithm type
                if is_pls:
                    exp_colors = ['blue'] * len(exp_costs)
                    exp_types = ['PLS'] * len(exp_costs)
                elif is_gurobi and not is_hybrid:
                    exp_colors = ['red'] * len(exp_costs)
                    exp_types = ['Gurobi'] * len(exp_costs)
                else:
                    exp_colors = ['purple'] * len(exp_costs)  # Hybrid or other
                    exp_types = ['Mixed'] * len(exp_costs)
                
                explored_colors.extend(exp_colors)
                explored_solver_types.extend(exp_types)
                
                # Create hover details for explored solutions
                for i, (cost, cloudy, incidence) in enumerate(zip(exp_costs, exp_cloudy_areas, exp_incidence_angles)):
                    detail = self._create_explored_solution_detail(result, cost, cloudy, incidence)
                    detail += f"<br>Solver: {exp_types[i]}"
                    explored_solution_details.append(detail)
            
            # Process final solutions - distinguish MILP vs PLS for hybrid algorithms
            final_costs, final_cloudy_areas, final_incidence_angles = self.extract_solution_objectives(result.final_solutions)
            final_timestamps = self.extract_solution_timestamps(result.final_solutions)
            
            all_final_costs.extend(final_costs)
            all_final_cloudy_areas.extend(final_cloudy_areas)
            all_final_incidence_angles.extend(final_incidence_angles)
            all_final_timestamps.extend(final_timestamps)
            
            # Color final solutions based on algorithm and whether they're from MILP
            if is_hybrid and result.milp_solutions:
                # For hybrid algorithms, check if solutions are in milp_solutions
                milp_costs, milp_cloudy, milp_incidence = self.extract_solution_objectives(result.milp_solutions)
                milp_coords = set(zip(milp_costs, milp_cloudy, milp_incidence))
                
                for i, (cost, cloudy, incidence) in enumerate(zip(final_costs, final_cloudy_areas, final_incidence_angles)):
                    if (cost, cloudy, incidence) in milp_coords:
                        final_colors.append('red')
                        final_solver_types.append('Gurobi')
                    else:
                        final_colors.append('blue')
                        final_solver_types.append('PLS')
            else:
                # For non-hybrid algorithms, color all solutions based on algorithm type
                if is_pls:
                    colors = ['blue'] * len(final_costs)
                    types = ['PLS'] * len(final_costs)
                elif is_gurobi:
                    colors = ['red'] * len(final_costs)
                    types = ['Gurobi'] * len(final_costs)
                else:
                    colors = ['purple'] * len(final_costs)  # Other MILP solvers
                    types = ['MILP'] * len(final_costs)
                
                final_colors.extend(colors)
                final_solver_types.extend(types)
            
            # Create hover details for final solutions
            for i, (cost, cloudy, incidence) in enumerate(zip(final_costs, final_cloudy_areas, final_incidence_angles)):
                detail = self._create_final_solution_detail(result, cost, cloudy, incidence)
                detail += f"<br>Solver: {final_solver_types[i]}"
                final_solution_details.append(detail)
        
        # Load and process JSONL solutions if instance name is provided and include_perfect is enabled
        if instance_name and self.include_perfect:
            import json  # Import here to avoid issues with error handling
            
            # Fix path construction - instances_dir is already tests/data, so go up to project root
            jsonl_file = self.instances_dir.parent.parent / "tests" / "data" / "manuels_results" / f"{instance_name}.jsonl"
            
            if jsonl_file.exists():
                # Load the corresponding instance to compute objectives
                instance_file = self.instances_dir / f"{instance_name}.dzn"
                if not instance_file.exists():
                    raise FileNotFoundError(f"Instance file not found: {instance_file}")
                
                import sims_problem
                try:
                    problem = sims_problem.SimsDiscreteProblem.from_dzn(str(instance_file))
                except Exception as e:
                    raise RuntimeError(f"Failed to load instance {instance_file}: {e}")
                
                jsonl_solutions_count = 0
                try:
                    with open(jsonl_file, 'r') as f:
                        for line_idx, line in enumerate(f):
                            line = line.strip()
                            if line:
                                try:
                                    selected_images = json.loads(line)
                                    if not isinstance(selected_images, list) or not all(isinstance(i, int) for i in selected_images):
                                        raise ValueError(f"Line {line_idx + 1}: Invalid format, expected list of integers, got: {type(selected_images)}")
                                    
                                    # Create Solution object and compute objectives
                                    try:
                                        solution = sims_problem.Solution.create(
                                            selected_images=selected_images,
                                            cost=0,  # Will be computed
                                            cloudy_area=0,  # Will be computed
                                            timestamp_us=0,  # Not relevant
                                            max_incidence_angle=None,  # Will be computed
                                            min_resolutions_sum=None   # Will be computed
                                        )
                                    except Exception as e:
                                        raise RuntimeError(f"Failed to create Solution object for line {line_idx + 1}: {e}")
                                    
                                    # Compute objectives using the problem
                                    try:
                                        cost, cloudy_area, max_angle, min_res = solution.compute_objectives(problem)
                                        solution.cost = cost
                                        solution.cloudy_area = cloudy_area
                                        solution.max_incidence_angle = max_angle
                                        solution.min_resolutions_sum = min_res
                                    except Exception as e:
                                        raise RuntimeError(f"Failed to compute objectives for line {line_idx + 1}: {e}")
                                    
                                    all_jsonl_costs.append(cost)
                                    all_jsonl_cloudy_areas.append(cloudy_area)
                                    all_jsonl_incidence_angles.append(max_angle)
                                    jsonl_solutions_count += 1
                                    
                                    # Create hover detail for JSONL solution
                                    detail = (
                                        f"Instance: {instance_name}<br>"
                                        f"Source: Manuel's Results<br>"
                                        f"Solution #{line_idx + 1}<br>"
                                        f"Type: Reference Solution<br>"
                                        f"Cost: {cost}<br>"
                                        f"Cloudy Area: {cloudy_area}<br>"
                                        f"Max Incidence Angle: {max_angle}<br>"
                                        f"Selected Images: {len(selected_images)}"
                                    )
                                    jsonl_solution_details.append(detail)
                                    
                                except json.JSONDecodeError as e:
                                    raise ValueError(f"Failed to parse JSON on line {line_idx + 1}: {e}")
                                except Exception as e:
                                    raise RuntimeError(f"Error processing JSONL line {line_idx + 1}: {e}")
                    
                    self.console.print(f"✅ DEBUG: Successfully loaded {jsonl_solutions_count} JSONL solutions")
                except IOError as e:
                    raise IOError(f"Failed to read JSONL file {jsonl_file}: {e}")
            else:
                manuels_dir = self.instances_dir / "manuels_results"
                if manuels_dir.exists():
                    available_files = list(manuels_dir.glob("*.jsonl"))
                    for f in available_files:
                        self.console.print(f"   - {f.name}")
                else:
                    self.console.print(f"🔍 ERROR: Manuels results directory does not exist: {manuels_dir}")
        
        # Calculate normalized timestamps for coloring (combine all timestamps)
        all_timestamps = all_explored_timestamps + all_final_timestamps
        if all_timestamps:
            min_timestamp = min(all_timestamps)
            max_timestamp = max(all_timestamps)
            timestamp_range = max_timestamp - min_timestamp if max_timestamp > min_timestamp else timedelta(seconds=1)
        else:
            min_timestamp, max_timestamp, timestamp_range = timedelta(0), timedelta(seconds=1), timedelta(seconds=1)
        
        # Normalize timestamps
        explored_norm_timestamps = []
        for ts in all_explored_timestamps:
            explored_norm_timestamps.append((ts - min_timestamp) / timestamp_range if timestamp_range.total_seconds() > 0 else 0.5)
        
        final_norm_timestamps = []
        for ts in all_final_timestamps:
            final_norm_timestamps.append((ts - min_timestamp) / timestamp_range if timestamp_range.total_seconds() > 0 else 0.5)
        
        # JSONL solutions get a neutral timestamp color (middle of range)
        jsonl_norm_timestamps = [0.5] * len(all_jsonl_costs)
        
        explored_data = (all_explored_costs, all_explored_cloudy_areas, all_explored_incidence_angles, 
                        explored_norm_timestamps, explored_solution_details, explored_colors, explored_solver_types)
        final_data = (all_final_costs, all_final_cloudy_areas, all_final_incidence_angles, 
                     final_norm_timestamps, final_solution_details, final_colors, final_solver_types)
        jsonl_data = (all_jsonl_costs, all_jsonl_cloudy_areas, all_jsonl_incidence_angles,
                     jsonl_norm_timestamps, jsonl_solution_details)
        
        return explored_data, final_data, jsonl_data, (min_timestamp, max_timestamp, timestamp_range)
    
    def _create_explored_solution_detail(self, result: BenchmarkResult, cost: float, cloudy: float, incidence: float) -> str:
        """Create hover detail text for explored solutions. Can be overridden by subclasses."""
        return (
            f"Instance: {result.instance_name}<br>"
            f"Iteration: {result.iteration}<br>"
            f"Type: Explored Solution<br>"
            f"Cost: {cost}<br>"
            f"Cloudy Area: {cloudy}<br>"
            f"Max Incidence Angle: {incidence}<br>"
            f"Runtime: {result.total_runtime_seconds:.2f}s"
        )
    
    def _create_final_solution_detail(self, result: BenchmarkResult, cost: float, cloudy: float, incidence: float) -> str:
        """Create hover detail text for final solutions. Can be overridden by subclasses."""
        return (
            f"Instance: {result.instance_name}<br>"
            f"Iteration: {result.iteration}<br>"
            f"Type: Final Pareto Solution<br>"
            f"Cost: {cost}<br>"
            f"Cloudy Area: {cloudy}<br>"
            f"Max Incidence Angle: {incidence}<br>"
            f"Runtime: {result.total_runtime_seconds:.2f}s"
        )
    
    def _create_3d_visualization_base(self, instance_name: str, instance_results: list[BenchmarkResult], 
                                     title_context: str = "for this instance") -> bool:
        """
        Base implementation for 3D visualization creation.
        
        Args:
            instance_name: Name of the instance
            instance_results: List of benchmark results
            title_context: Context string for the subtitle (e.g., "for this instance", "exploration")
            
        Returns:
            True if visualization was created successfully, False otherwise
        """
        # Filter successful results
        successful_results = [r for r in instance_results if not r.error and r.final_solutions]
        
        if not successful_results:
            self.console.print(f"❌ No successful results for {instance_name} to visualize")
            return False
        
        self.console.print(f"📊 Creating 3D visualization for instance {instance_name}...")
        
        # Collect visualization data
        explored_data, final_data, jsonl_data, timestamp_range = self._collect_visualization_data(successful_results, instance_name)
        
        # Unpack data with new color information (backward compatible)
        if len(explored_data) >= 7:
            all_explored_costs, all_explored_cloudy_areas, all_explored_incidence_angles, explored_norm_timestamps, explored_solution_details, explored_colors, explored_solver_types = explored_data
        else:
            # Fallback for compatibility
            all_explored_costs, all_explored_cloudy_areas, all_explored_incidence_angles, explored_norm_timestamps, explored_solution_details = explored_data[:5]
            explored_colors = ['gray'] * len(all_explored_costs)
            explored_solver_types = ['Unknown'] * len(all_explored_costs)
            
        if len(final_data) >= 7:
            all_final_costs, all_final_cloudy_areas, all_final_incidence_angles, final_norm_timestamps, final_solution_details, final_colors, final_solver_types = final_data
        else:
            # Fallback for compatibility
            all_final_costs, all_final_cloudy_areas, all_final_incidence_angles, final_norm_timestamps, final_solution_details = final_data[:5]
            # Default colors based on algorithm type
            algorithm_name = self.get_algorithm_name().lower()
            if 'pls' in algorithm_name and 'hybrid' not in algorithm_name:
                default_color = 'blue'
                default_type = 'PLS'
            elif 'gurobi' in algorithm_name:
                default_color = 'red'
                default_type = 'Gurobi'
            else:
                default_color = 'purple'
                default_type = 'MILP'
            final_colors = [default_color] * len(all_final_costs)
            final_solver_types = [default_type] * len(all_final_costs)
            
        all_jsonl_costs, all_jsonl_cloudy_areas, all_jsonl_incidence_angles, jsonl_norm_timestamps, jsonl_solution_details = jsonl_data
        
        if not all_final_costs and not all_explored_costs and not all_jsonl_costs:
            self.console.print(f"❌ No solutions found for instance {instance_name}")
            return False
        
        # Check if plotly is available 
        try:
            import plotly.graph_objects as go
        except ImportError:
            self.console.print("❌ Plotly not available for creating plots")
            return False
        
        traces = []
        
        # Explored solutions (X markers)
        if all_explored_costs:
            traces.append(go.Scatter3d(
                x=all_explored_costs,
                y=all_explored_cloudy_areas,
                z=all_explored_incidence_angles,
                mode='markers',
                marker=dict(
                    size=2,
                    color=explored_colors,
                    opacity=0.4,
                    line=dict(color='black', width=0.5),
                    symbol='x'
                ),
                text=explored_solution_details,
                hovertemplate='%{text}<extra></extra>',
                name=f'🔍 Explored Solutions ({len(all_explored_costs)})',
                legendgroup='explored'
            ))
        
        # Final Pareto solutions (Circle markers) - colored by solver type
        if all_final_costs:
            traces.append(go.Scatter3d(
                x=all_final_costs,
                y=all_final_cloudy_areas,
                z=all_final_incidence_angles,
                mode='markers',
                marker=dict(
                    size=4,
                    color=final_colors,
                    opacity=0.9,
                    line=dict(color='black', width=1),
                    symbol='circle'
                ),
                text=final_solution_details,
                hovertemplate='%{text}<extra></extra>',
                name=f'⭐ Final Pareto Solutions ({len(all_final_costs)})',
                legendgroup='final'
            ))
        
        # JSONL Reference solutions (Diamond markers)
        if all_jsonl_costs:
            print("Adding JSONL reference solutions to visualization...")
            traces.append(go.Scatter3d(
                x=all_jsonl_costs,
                y=all_jsonl_cloudy_areas,
                z=all_jsonl_incidence_angles,
                mode='markers',
                marker=dict(
                    size=4,
                    color='red',  # Fixed red color for reference solutions
                    opacity=0.8,
                    line=dict(color='darkred', width=1),
                    symbol='diamond'
                ),
                text=jsonl_solution_details,
                hovertemplate='%{text}<extra></extra>',
                name=f'💎 Reference Solutions ({len(all_jsonl_costs)})',
                legendgroup='reference'
            ))
        
        fig = go.Figure(data=traces)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{self.get_algorithm_name()} Benchmark Results - {instance_name}<br>'
                     f'<sub>Interactive 3D plot showing explored solutions (X), final Pareto solutions (○), and reference solutions (◆)</sub>',
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis=dict(title=dict(text='Cost (minimize)'), tickfont=dict(size=12)),
                yaxis=dict(title=dict(text='Cloudy Area Coverage (minimize)'), tickfont=dict(size=12)),
                zaxis=dict(title=dict(text='Max Incidence Angle (minimize)'), tickfont=dict(size=12)),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1200,
            height=800,
            margin=dict(l=0, r=0, b=0, t=100)
        )
        
        # Add statistics annotation
        total_solutions = len(all_explored_costs) + len(all_final_costs) + len(all_jsonl_costs)
        all_combined_costs = all_explored_costs + all_final_costs + all_jsonl_costs
        all_combined_cloudy = all_explored_cloudy_areas + all_final_cloudy_areas + all_jsonl_cloudy_areas
        all_combined_incidence = all_explored_incidence_angles + all_final_incidence_angles + all_jsonl_incidence_angles
        
        annotation_text = (
            f"📊 {instance_name} Statistics:<br>"
            f"• Total Solutions: {total_solutions}<br>"
            f"• Explored Solutions: {len(all_explored_costs)}<br>"
            f"• Final Pareto Solutions: {len(all_final_costs)}<br>"
            f"• Reference Solutions: {len(all_jsonl_costs)}<br>"
            f"• Successful Runs: {len(successful_results)}<br>"
        )
        
        if all_combined_costs:
            annotation_text += (
                f"• Cost Range: {min(all_combined_costs):.1f} - {max(all_combined_costs):.1f}<br>"
                f"• Cloudy Area Range: {min(all_combined_cloudy):.1f} - {max(all_combined_cloudy):.1f}<br>"
                f"• Incidence Angle Range: {min(all_combined_incidence):.1f} - {max(all_combined_incidence):.1f}<br>"
            )
        
        annotation_text += f"• Avg Runtime: {statistics.mean([r.total_runtime_seconds for r in successful_results]):.2f}s"
        
        fig.add_annotation(
            text=annotation_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=11)
        )
        
        # Save visualization
        finished_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{instance_name}_{self.get_visualization_filename_suffix()}_3d_{finished_at}.html"
        fig.write_html(output_file)
        
        self.console.print(f"✓ 3D visualization saved: {output_file}")
        return True
    
    @abstractmethod
    def print_summary(self):
        """Print summary of results. Must be implemented by subclasses."""
        pass


class HybridBenchmarkRunner(BenchmarkRunner):
    """Hybrid algorithm benchmark runner with TUI progress tracking."""
    
    def __init__(self, instances_dir: Path, output_dir: Path, iterations: int = 1, ratio_step: int = 10, use_tui: bool = False, size_limit: Optional[int] = None, name_filter: Optional[str] = None, timeout: float = 300.0, validate_solutions: bool = True, include_perfect: bool = False):
        super().__init__(instances_dir, output_dir, iterations, use_tui, size_limit, name_filter, timeout, validate_solutions, include_perfect)
        
        # Generate ratio configurations (MILP%, PLS%)
        self.ratios = [(i, 100-i) for i in range(100, -1, -ratio_step)]
    
    def load_instance(self, instance_file: Path) -> sims_problem.SimsDiscreteProblem:
        """Load a SIMS problem instance from .dzn file."""
        try:
            return sims_problem.SimsDiscreteProblem.from_dzn(str(instance_file))
        except Exception as e:
            raise RuntimeError(f"Failed to load instance {instance_file}: {e}")
    
    def run_hybrid_solver(
        self, 
        instance: sims_problem.SimsDiscreteProblem,
        ratio: tuple[int, int],
        timeout_seconds: Optional[float] = None
    ) -> tuple[sims_problem.SolvingResult, float, float]:
        """
        Run the hybrid solver and return SolvingResult with timing information.
        
        Returns:
            tuple of (solving_result, milp_time, pls_time)
        """
        if timeout_seconds is None:
            timeout_seconds = self.timeout
        # Configure solvers
        milp_config = sims_problem.MilpConfig(
            objectives=["min_cost", "cloud_coverage", "max_incidence_angle"],
            grid_points=50,
            bypass_coefficient=True,
            early_exit=True,
            flag_array=True,
            solver_name="cbc"
        )
        
        pls_config = sims_problem.PlsConfig(
            objectives=["min_cost", "cloud_coverage", "max_incidence_angle"],
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
            timedelta(seconds=timeout_seconds),
            trace=True
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
        *args
    ) -> BenchmarkResult:
        """Run a single benchmark and collect detailed results."""
        ratio: tuple[int, int] = args[0]  # Extract ratio from args
        result = BenchmarkResult()
        result.instance_name = instance_name
        result.iteration = iteration
        result.ratio = ratio
        
        try:
            solving_result, milp_time, pls_time = self.run_hybrid_solver(instance, ratio)
            result.milp_runtime_seconds = milp_time
            result.pls_runtime_seconds = pls_time
            result.total_runtime_seconds = milp_time + pls_time
            
            # Extract solutions from SolvingResult
            result.final_solutions = solving_result.final_solutions
            result.explored_solutions = solving_result.explored_solutions
            result.num_final_solutions = len(solving_result.final_solutions)
            result.num_explored_solutions = len(solving_result.explored_solutions)
            
            # Store trace data for saving
            result.trace_data = solving_result.trace
            
            # For now, we don't have access to intermediate MILP solutions
            # This would require modifying the solver to return them
            result.milp_solutions = []
            result.num_milp_solutions = 0
            
            # Validate solutions and Pareto front
            if result.final_solutions and self.validate_solutions and self.validator:
                validation_report = self.validator.validate_benchmark_result(result, instance)
                result.validation_results = validation_report
                
                # Log validation issues if any
                if not validation_report.overall_valid:
                    self.console.print(f"[yellow]⚠️  Validation issues found for {instance_name} iter={iteration} ratio={ratio}")
                    
                    # Log invalid solutions
                    if validation_report.summary.invalid_solutions > 0:
                        self.console.print(f"   - {validation_report.summary.invalid_solutions} invalid solutions")
                    
                    # Log Pareto front issues
                    if not validation_report.pareto_validation.is_valid_pareto:
                        dominated_count = len(validation_report.pareto_validation.dominated_solutions)
                        self.console.print(f"   - {dominated_count} dominated solutions in Pareto front")
                
                # Log warnings
                if validation_report.summary.solutions_with_warnings > 0:
                    self.console.print(f"[blue]ℹ️  {validation_report.summary.solutions_with_warnings} solutions have warnings")
            
        except Exception as e:
            result.error = str(e)
            
        return result
    
    def run_benchmarks(self):
        """Run all benchmarks with optional TUI progress tracking."""
        if self.use_tui:
            self._run_benchmarks_with_tui()
        else:
            self._run_benchmarks_simple()
    
    def _run_benchmarks_simple(self):
        """Run benchmarks with simple CLI output."""
        total_runs = len(self.instance_files) * self.iterations * len(self.ratios)
        current_run = 0
        
        print(f"Starting hybrid benchmark with {len(self.instance_files)} instances, {self.iterations} iterations each, {len(self.ratios)} ratios")
        print(f"Total runs: {total_runs}")
        print(f"Ratios to test: {[f'{r[0]}:{r[1]}' for r in self.ratios]}")
        print("=" * 80)
        
        for instance_idx, instance_file in enumerate(self.instance_files):
            instance_name = instance_file.stem
            print(f"\nProcessing instance {instance_idx + 1}/{len(self.instance_files)}: {instance_name}")
            
            # Load instance
            try:
                instance = self.load_instance(instance_file)
                print(f"  ✓ Loaded instance: {instance.num_images} images, {instance.universe} universe elements")
            except Exception as e:
                print(f"  ❌ Failed to load {instance_file}: {e}")
                continue
            
            # Track instance start position for per-instance saving
            instance_start_idx = len(self.results)
            
            for iteration in range(self.iterations):
                print(f"  Iteration {iteration + 1}/{self.iterations}")
                
                for ratio_idx, ratio in enumerate(self.ratios):
                    current_run += 1
                    progress_pct = (current_run / total_runs) * 100
                    
                    print(f"    [{current_run:3d}/{total_runs}] ({progress_pct:5.1f}%) Ratio {ratio[0]:3d}:{ratio[1]:2d} - ", end="", flush=True)
                    
                    # Run benchmark
                    result = self.run_single_benchmark(instance, instance_name, iteration, ratio)
                    
                    self.results.append(result)
                    
                    if result.error:
                        print(f"FAILED ({result.total_runtime_seconds:.1f}s) - {result.error}")
                    else:
                        print(f"SUCCESS ({result.total_runtime_seconds:.1f}s) - {result.num_final_solutions} solutions")
            
            # Save results for this instance
            instance_results = self.results[instance_start_idx:]
            self.save_instance_results(instance_name, instance_results)
        
        print("\n" + "=" * 80)
        print("Benchmark completed!")
    
    def _run_benchmarks_with_tui(self):
        """Run benchmarks with fancy TUI progress tracking."""
        
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
                    
                    # Track instance start position for per-instance saving
                    instance_start_idx = len(self.results)
                    
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
                    
                    # Save results for this instance
                    instance_results = self.results[instance_start_idx:]
                    self.save_instance_results(instance_name, instance_results)
                    
                    # Hide sub-progress bars for next instance
                    progress.update(iteration_task, visible=False)
                    progress.update(ratio_task, visible=False)
                    progress.update(current_task, visible=False)
                
                # Complete main progress
                progress.update(main_task, completed=len(self.instance_files))
    
    def save_results(self):
        """Save benchmark results with trace archives and create 3D visualization."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save trace archives for results that have trace data
        trace_files = []
        for i, result in enumerate(self.results):
            if result.trace_data:
                trace_file = self.output_dir / f"hybrid_benchmark_trace_{i}_{timestamp}.tar.gz"
                with open(trace_file, 'wb') as f:
                    f.write(result.trace_data)
                trace_files.append(trace_file)
        
        # Create summary statistics
        summary = self.create_summary_statistics()
        summary_file = self.output_dir / f"hybrid_benchmark_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)
        
        self.console.print("\n[green]Results saved:")
        if trace_files:
            for trace_file in trace_files:
                self.console.print(f"  Trace:    {trace_file}")
        self.console.print(f"  Summary:  {summary_file}")
    
    def create_summary_statistics(self) -> BenchmarkSummaryStatistics:
        """Create summary statistics from benchmark results."""
        successful_results = [r for r in self.results if not r.error]
        
        # Create benchmark info
        benchmark_info = BenchmarkInfo(
            timestamp=datetime.now().isoformat(),
            instances_tested=len(self.instance_files),
            iterations_per_ratio=self.iterations,
            ratios_tested=self.ratios,
            total_runs=len(self.results),
            successful_runs=len(successful_results),
            failed_runs=len([r for r in self.results if r.error])
        )
        
        # Initialize collections
        performance_by_ratio: dict[str, PerformanceByRatio] = {}
        performance_by_instance: dict[str, PerformanceByInstance] = {}
        overall_statistics: Optional[OverallStatistics] = None
        
        if successful_results:
            # Group results by ratio
            for ratio in self.ratios:
                ratio_results = [r for r in successful_results if r.ratio == ratio]
                if ratio_results:
                    ratio_key = f"{ratio[0]}_{ratio[1]}"
                    performance_by_ratio[ratio_key] = PerformanceByRatio(
                        ratio=ratio,
                        num_runs=len(ratio_results),
                        avg_total_runtime=statistics.mean([r.total_runtime_seconds for r in ratio_results]),
                        avg_solutions=statistics.mean([r.num_final_solutions for r in ratio_results]),
                        std_runtime=statistics.stdev([r.total_runtime_seconds for r in ratio_results]) if len(ratio_results) > 1 else 0,
                        min_runtime=min([r.total_runtime_seconds for r in ratio_results]),
                        max_runtime=max([r.total_runtime_seconds for r in ratio_results])
                    )
            
            # Group results by instance
            for instance_file in self.instance_files:
                instance_name = instance_file.stem
                instance_results = [r for r in successful_results if r.instance_name == instance_name]
                if instance_results:
                    performance_by_instance[instance_name] = PerformanceByInstance(
                        num_runs=len(instance_results),
                        avg_runtime=statistics.mean([r.total_runtime_seconds for r in instance_results]),
                        avg_solutions=statistics.mean([r.num_final_solutions for r in instance_results]),
                        std_runtime=0,  # Not calculated for hybrid
                        min_runtime=0,  # Not calculated for hybrid
                        max_runtime=0,  # Not calculated for hybrid
                        min_solutions=0,  # Not calculated for hybrid
                        max_solutions=0,  # Not calculated for hybrid
                        best_ratio=max(instance_results, key=lambda x: x.num_final_solutions).ratio,
                        fastest_ratio=min(instance_results, key=lambda x: x.total_runtime_seconds).ratio
                    )
            
            # Overall statistics
            overall_statistics = OverallStatistics(
                avg_runtime=statistics.mean([r.total_runtime_seconds for r in successful_results]),
                avg_solutions=statistics.mean([r.num_final_solutions for r in successful_results]),
                total_runtime=sum([r.total_runtime_seconds for r in successful_results]),
                best_performing_ratio=max(successful_results, key=lambda x: x.num_final_solutions).ratio,
                fastest_ratio=min(successful_results, key=lambda x: x.total_runtime_seconds).ratio
            )
        else:
            # Default overall statistics if no successful results
            overall_statistics = OverallStatistics(
                avg_runtime=0.0,
                avg_solutions=0.0,
                total_runtime=0.0
            )
        
        # Add validation statistics
        validation_stats = self._calculate_validation_statistics(successful_results)
        
        return BenchmarkSummaryStatistics(
            benchmark_info=benchmark_info,
            overall_statistics=overall_statistics,
            performance_by_instance=performance_by_instance,
            validation_statistics=validation_stats,
            performance_by_ratio=performance_by_ratio
        )
    
    def _calculate_validation_statistics(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """Calculate validation statistics from results."""
        total_with_validation = len([r for r in results if r.validation_results])
        total_valid = len([r for r in results 
                          if r.validation_results and r.validation_results.overall_valid])
        
        total_solutions = sum([r.num_final_solutions for r in results])
        total_invalid_solutions = sum([
            r.validation_results.summary.invalid_solutions 
            for r in results if r.validation_results
        ])
        total_warned_solutions = sum([
            r.validation_results.summary.solutions_with_warnings 
            for r in results if r.validation_results
        ])
        
        total_pareto_violations = len([
            r for r in results 
            if r.validation_results and 
            not r.validation_results.pareto_validation.is_valid_pareto
        ])
        
        return {
            "runs_with_validation": total_with_validation,
            "valid_runs": total_valid,
            "invalid_runs": total_with_validation - total_valid,
            "total_solutions": total_solutions,
            "invalid_solutions": total_invalid_solutions,
            "solutions_with_warnings": total_warned_solutions,
            "pareto_front_violations": total_pareto_violations,
            "run_validation_rate": (total_valid / total_with_validation * 100) if total_with_validation > 0 else 0,
            "solution_validity_rate": ((total_solutions - total_invalid_solutions) / total_solutions * 100) if total_solutions > 0 else 0
        }
    
    def print_summary(self):
        """Print a summary of benchmark results."""
        successful = [r for r in self.results if not r.error]
        failed = [r for r in self.results if r.error]
        
        self.console.print("\n[bold green]Hybrid Benchmark Complete!")
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
            
            # Print validation summary
            self.print_validation_summary(successful)
        
        if failed:
            self.console.print("\n[red]Failed runs:")
            for result in failed[:5]:  # Show first 5 failures
                self.console.print(f"  {result.instance_name} iter={result.iteration} ratio={result.ratio}: {result.error}")
    
    def print_validation_summary(self, successful_results: list[BenchmarkResult]):
        """Print a summary of validation results."""
        if not successful_results:
            return
        
        self.console.print("\n[bold cyan]🔍 Validation Summary:")
        
        # Count validation results
        total_with_validation = len([r for r in successful_results if r.validation_results])
        total_valid = len([r for r in successful_results 
                          if r.validation_results and r.validation_results.overall_valid])
        total_invalid = total_with_validation - total_valid
        
        total_solutions = sum([r.num_final_solutions for r in successful_results])
        total_invalid_solutions = sum([
            r.validation_results.summary.invalid_solutions 
            for r in successful_results if r.validation_results
        ])
        total_warned_solutions = sum([
            r.validation_results.summary.solutions_with_warnings 
            for r in successful_results if r.validation_results
        ])
        
        # Count Pareto front violations
        total_pareto_violations = len([
            r for r in successful_results 
            if r.validation_results and 
            not r.validation_results.pareto_validation.is_valid_pareto
        ])
        
        self.console.print(f"  📊 Runs validated: {total_with_validation}/{len(successful_results)}")
        self.console.print(f"  ✅ Valid runs: {total_valid}")
        self.console.print(f"  ❌ Invalid runs: {total_invalid}")
        self.console.print(f"  📈 Total solutions: {total_solutions}")
        self.console.print(f"  🚫 Invalid solutions: {total_invalid_solutions}")
        self.console.print(f"  ⚠️  Solutions with warnings: {total_warned_solutions}")
        self.console.print(f"  🎯 Pareto front violations: {total_pareto_violations}")
        
        if total_invalid > 0:
            self.console.print(f"\n[yellow]⚠️  {total_invalid} runs had validation issues - check detailed results for specifics")
        
        if total_pareto_violations > 0:
            self.console.print(f"[red]❌ {total_pareto_violations} runs had non-Pareto fronts (dominated solutions present)")
        
        if total_invalid == 0 and total_pareto_violations == 0:
            self.console.print("[green]✅ All validated runs passed quality checks!")
            
        # Calculate validation statistics
        validation_rate = (total_valid / total_with_validation * 100) if total_with_validation > 0 else 0
        solution_validity_rate = ((total_solutions - total_invalid_solutions) / total_solutions * 100) if total_solutions > 0 else 0
        
        self.console.print(f"\n  📊 Run validation rate: {validation_rate:.1f}%")
        self.console.print(f"  📊 Solution validity rate: {solution_validity_rate:.1f}%")
    
    def create_instance_summary_statistics(self, instance_name: str, instance_results: list[BenchmarkResult]) -> BenchmarkSummaryStatistics:
        """Create summary statistics for a single instance."""
        successful_results = [r for r in instance_results if not r.error]
        
        # Create benchmark info
        benchmark_info = BenchmarkInfo(
            timestamp=datetime.now().isoformat(),
            instances_tested=1,  # Just this instance
            iterations_per_ratio=self.iterations,
            ratios_tested=self.ratios,
            total_runs=len(instance_results),
            successful_runs=len(successful_results),
            failed_runs=len([r for r in instance_results if r.error])
        )
        
        # Initialize collections
        performance_by_ratio: dict[str, PerformanceByRatio] = {}
        performance_by_instance: dict[str, PerformanceByInstance] = {}
        overall_statistics: Optional[OverallStatistics] = None
        
        if successful_results:
            # Group results by ratio for this instance
            for ratio in self.ratios:
                ratio_results = [r for r in successful_results if r.ratio == ratio]
                if ratio_results:
                    ratio_key = f"{ratio[0]}_{ratio[1]}"
                    performance_by_ratio[ratio_key] = PerformanceByRatio(
                        ratio=ratio,
                        num_runs=len(ratio_results),
                        avg_total_runtime=statistics.mean([r.total_runtime_seconds for r in ratio_results]),
                        avg_solutions=statistics.mean([r.num_final_solutions for r in ratio_results]),
                        std_runtime=statistics.stdev([r.total_runtime_seconds for r in ratio_results]) if len(ratio_results) > 1 else 0,
                        min_runtime=min([r.total_runtime_seconds for r in ratio_results]),
                        max_runtime=max([r.total_runtime_seconds for r in ratio_results])
                    )
            
            # Performance for this single instance
            performance_by_instance[instance_name] = PerformanceByInstance(
                num_runs=len(successful_results),
                avg_runtime=statistics.mean([r.total_runtime_seconds for r in successful_results]),
                avg_solutions=statistics.mean([r.num_final_solutions for r in successful_results]),
                std_runtime=statistics.stdev([r.total_runtime_seconds for r in successful_results]) if len(successful_results) > 1 else 0,
                min_runtime=min([r.total_runtime_seconds for r in successful_results]),
                max_runtime=max([r.total_runtime_seconds for r in successful_results]),
                min_solutions=min([r.num_final_solutions for r in successful_results]),
                max_solutions=max([r.num_final_solutions for r in successful_results]),
                best_ratio=max(successful_results, key=lambda x: x.num_final_solutions).ratio,
                fastest_ratio=min(successful_results, key=lambda x: x.total_runtime_seconds).ratio
            )
            
            # Overall statistics for this instance
            overall_statistics = OverallStatistics(
                avg_runtime=statistics.mean([r.total_runtime_seconds for r in successful_results]),
                avg_solutions=statistics.mean([r.num_final_solutions for r in successful_results]),
                total_runtime=sum([r.total_runtime_seconds for r in successful_results]),
                best_performing_ratio=max(successful_results, key=lambda x: x.num_final_solutions).ratio,
                fastest_ratio=min(successful_results, key=lambda x: x.total_runtime_seconds).ratio
            )
        else:
            # Default overall statistics if no successful results
            overall_statistics = OverallStatistics(
                avg_runtime=0.0,
                avg_solutions=0.0,
                total_runtime=0.0
            )
        
        # Add validation statistics
        validation_stats = self._calculate_validation_statistics(successful_results)
        
        return BenchmarkSummaryStatistics(
            benchmark_info=benchmark_info,
            overall_statistics=overall_statistics,
            performance_by_instance=performance_by_instance,
            validation_statistics=validation_stats,
            performance_by_ratio=performance_by_ratio
        )
    
    def get_algorithm_name(self) -> str:
        """Get the name of the algorithm for this runner."""
        return "Hybrid"
    
    def get_visualization_filename_suffix(self) -> str:
        """Get the filename suffix for visualizations."""
        return "hybrid"
    
    def _create_explored_solution_detail(self, result: BenchmarkResult, cost: float, cloudy: float, incidence: float) -> str:
        """Create hover detail text for explored solutions with ratio information."""
        return (
            f"Instance: {result.instance_name}<br>"
            f"Iteration: {result.iteration}<br>"
            f"Ratio: {result.ratio[0]}:{result.ratio[1]}<br>"
            f"Type: Explored Solution<br>"
            f"Cost: {cost}<br>"
            f"Cloudy Area: {cloudy}<br>"
            f"Max Incidence Angle: {incidence}<br>"
            f"Runtime: {result.total_runtime_seconds:.2f}s"
        )
    
    def _create_final_solution_detail(self, result: BenchmarkResult, cost: float, cloudy: float, incidence: float) -> str:
        """Create hover detail text for final solutions with ratio information."""
        return (
            f"Instance: {result.instance_name}<br>"
            f"Iteration: {result.iteration}<br>"
            f"Ratio: {result.ratio[0]}:{result.ratio[1]}<br>"
            f"Type: Final Pareto Solution<br>"
            f"Cost: {cost}<br>"
            f"Cloudy Area: {cloudy}<br>"
            f"Max Incidence Angle: {incidence}<br>"
            f"Runtime: {result.total_runtime_seconds:.2f}s"
        )
    
    def create_instance_3d_visualization(self, instance_name: str, instance_results: list[BenchmarkResult]) -> bool:
        """Create 3D visualization for a single instance using the base implementation."""
        return self._create_3d_visualization_base(instance_name, instance_results)


class PlsBenchmarkRunner(BenchmarkRunner):
    """PLS-specific benchmark runner."""
    
    def __init__(self, instances_dir: Path, output_dir: Path, iterations: int = 10, max_iterations: int = 50000, use_tui: bool = False, size_limit: Optional[int] = None, name_filter: Optional[str] = None, timeout: float = 300.0, validate_solutions: bool = True, include_perfect: bool = False):
        super().__init__(instances_dir, output_dir, iterations, use_tui, size_limit, name_filter, timeout, validate_solutions, include_perfect)
        self.max_iterations = max_iterations
    
    def get_algorithm_name(self) -> str:
        """Get the name of the algorithm for this runner."""
        return "PLS"
    
    def get_visualization_filename_suffix(self) -> str:
        """Get the filename suffix for visualizations."""
        return "pls"
    
    def _create_explored_solution_detail(self, result: BenchmarkResult, cost: float, cloudy: float, incidence: float) -> str:
        """Create hover detail text for explored solutions (PLS doesn't use ratios)."""
        return (
            f"Instance: {result.instance_name}<br>"
            f"Iteration: {result.iteration}<br>"
            f"Type: Explored Solution<br>"
            f"Cost: {cost}<br>"
            f"Cloudy Area: {cloudy}<br>"
            f"Max Incidence Angle: {incidence}<br>"
            f"Runtime: {result.total_runtime_seconds:.2f}s"
        )
    
    def _create_final_solution_detail(self, result: BenchmarkResult, cost: float, cloudy: float, incidence: float) -> str:
        """Create hover detail text for final solutions (PLS doesn't use ratios)."""
        return (
            f"Instance: {result.instance_name}<br>"
            f"Iteration: {result.iteration}<br>"
            f"Type: Final Pareto Solution<br>"
            f"Cost: {cost}<br>"
            f"Cloudy Area: {cloudy}<br>"
            f"Max Incidence Angle: {incidence}<br>"
            f"Runtime: {result.total_runtime_seconds:.2f}s"
        )
    
    def load_instance(self, instance_file: Path) -> sims_problem.SimsDiscreteProblem:
        """Load a SIMS problem instance from .dzn file."""
        try:
            return sims_problem.SimsDiscreteProblem.from_dzn(str(instance_file))
        except Exception as e:
            raise RuntimeError(f"Failed to load instance {instance_file}: {e}")
    
    def run_pls_solver(
        self, 
        instance: sims_problem.SimsDiscreteProblem,
        timeout: timedelta = timedelta(seconds=300)
    ) -> Any:
        """Run pure PLS solver and return SolvingResult."""
        return sims_problem.solve_with_pls(
            instance,
            objectives=["min_cost", "cloud_coverage", "min_resolution", "max_incidence_angle"],
            plots=False,
            plot_output_path=None,
            timeout=timeout,
            max_iterations=self.max_iterations,
            is_deterministic=False,
            initial_population_size=100,
            neighborhood_size_min=1,
            neighborhood_size_max=6,
            trace=True
        )
    
    def run_single_benchmark(
        self, 
        instance: sims_problem.SimsDiscreteProblem,
        instance_name: str,
        iteration: int,
        *args
    ) -> BenchmarkResult:
        """Run a single PLS benchmark and collect results."""
        # PLS doesn't use additional args like ratio, so we ignore them
        result = BenchmarkResult()
        result.instance_name = instance_name
        result.iteration = iteration
        result.ratio = (0, 100)  # Pure PLS
        
        try:
            start_time = time.time()
            solving_result = self.run_pls_solver(instance)
            result.total_runtime_seconds = time.time() - start_time
            result.milp_runtime_seconds = 0.0
            result.pls_runtime_seconds = result.total_runtime_seconds
            
            # Convert solutions to dictionaries
            result.final_solutions = solving_result.final_solutions
            result.explored_solutions = solving_result.explored_solutions
            result.num_final_solutions = len(solving_result.final_solutions)
            result.num_explored_solutions = len(solving_result.explored_solutions)
            result.milp_solutions = []
            result.num_milp_solutions = 0
            
            # Store trace data for saving
            result.trace_data = solving_result.trace

            print(f"DEBUG: EXPLORED SOLUTIONS COUNT: {result.num_explored_solutions}")
            
            # Validate solutions and Pareto front
            if result.final_solutions and self.validate_solutions and self.validator:
                validation_report = self.validator.validate_benchmark_result(result, instance)
                result.validation_results = validation_report
                
                # Log validation issues if any
                if not validation_report.overall_valid:
                    self.console.print(f"[yellow]⚠️  Validation issues found for {instance_name} iter={iteration}")
                    
                    # Log invalid solutions
                    if validation_report.summary.invalid_solutions > 0:
                        self.console.print(f"   - {validation_report.summary.invalid_solutions} invalid solutions")
                    
                    # Log Pareto front issues
                    if not validation_report.pareto_validation.is_valid_pareto:
                        dominated_count = len(validation_report.pareto_validation.dominated_solutions)
                        self.console.print(f"   - {dominated_count} dominated solutions in Pareto front")
                
                # Log warnings
                if validation_report.summary.solutions_with_warnings > 0:
                    self.console.print(f"[blue]ℹ️  {validation_report.summary.solutions_with_warnings} solutions have warnings")
            
        except Exception as e:
            result.error = str(e)
            
        return result
    
    def run_benchmarks(self):
        """Run PLS benchmarks with optional TUI progress tracking."""
        if self.use_tui:
            self._run_benchmarks_with_tui()
        else:
            self._run_benchmarks_simple()
    
    def _run_benchmarks_simple(self):
        """Run PLS benchmarks with simple CLI output."""
        total_runs = len(self.instance_files) * self.iterations
        current_run = 0
        
        print(f"Starting PLS benchmark with {len(self.instance_files)} instances, {self.iterations} iterations each")
        print(f"Total runs: {total_runs}")
        print(f"Max PLS iterations per run: {self.max_iterations}")
        print("=" * 80)
        
        for instance_idx, instance_file in enumerate(self.instance_files):
            instance_name = instance_file.stem
            print(f"\nProcessing instance {instance_idx + 1}/{len(self.instance_files)}: {instance_name}")
            
            try:
                instance = self.load_instance(instance_file)
                print(f"  ✓ Loaded instance: {instance.num_images} images, {instance.universe} universe elements")
            except Exception as e:
                print(f"  ❌ Failed to load {instance_file}: {e}")
                continue
            
            # Track instance start position for per-instance saving
            instance_start_idx = len(self.results)
            
            for iteration in range(self.iterations):
                current_run += 1
                progress_pct = (current_run / total_runs) * 100
                
                print(f"  [{current_run:3d}/{total_runs}] ({progress_pct:5.1f}%) Iteration {iteration + 1} - ", end="", flush=True)
                
                result = self.run_single_benchmark(instance, instance_name, iteration)
                
                self.results.append(result)
                
                if result.error:
                    print(f"FAILED ({result.total_runtime_seconds:.1f}s) - {result.error}")
                else:
                    print(f"SUCCESS ({result.total_runtime_seconds:.1f}s) - {result.num_final_solutions} solutions")
            
            # Save results for this instance
            instance_results = self.results[instance_start_idx:]
            self.save_instance_results(instance_name, instance_results)
        
        print("\n" + "=" * 80)
        print("PLS benchmark completed!")
    
    def _run_benchmarks_with_tui(self):
        """Run PLS benchmarks with TUI progress tracking."""
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
            
            main_task = progress.add_task(
                f"[bold blue]PLS Benchmarking {len(self.instance_files)} instances", 
                total=len(self.instance_files)
            )
            
            iteration_task = progress.add_task(
                "[bold green]Iterations", 
                total=self.iterations,
                visible=False
            )
            
            current_task = progress.add_task(
                "[bold white]Current Run", 
                total=1,
                visible=False
            )
            
            for instance_idx, instance_file in enumerate(self.instance_files):
                instance_name = instance_file.stem
                
                progress.update(
                    main_task, 
                    description=f"[bold blue]Instance: {instance_name}",
                    completed=instance_idx
                )
                
                try:
                    instance = self.load_instance(instance_file)
                except Exception as e:
                    self.console.print(f"[red]Failed to load {instance_file}: {e}")
                    continue
                
                # Track instance start position for per-instance saving
                instance_start_idx = len(self.results)
                
                progress.update(iteration_task, visible=True, completed=0)
                
                for iteration in range(self.iterations):
                    progress.update(
                        iteration_task,
                        description=f"[bold green]Iteration {iteration + 1}/{self.iterations}",
                        completed=iteration
                    )
                    
                    progress.update(
                        current_task,
                        visible=True,
                        description=f"[bold white]Running PLS on {instance_name} iter={iteration+1}",
                        completed=0
                    )
                    
                    result = self.run_single_benchmark(instance, instance_name, iteration)
                    self.results.append(result)
                    
                    progress.update(current_task, completed=1)
                
                # Save results for this instance
                instance_results = self.results[instance_start_idx:]
                self.save_instance_results(instance_name, instance_results)
                
                progress.update(iteration_task, completed=self.iterations, visible=False)
                progress.update(current_task, visible=False)
            
            progress.update(main_task, completed=len(self.instance_files))
    
    def save_results(self):
        """Save PLS benchmark results with trace archives and create 3D visualization."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save trace archives for results that have trace data
        trace_files = []
        for i, result in enumerate(self.results):
            if result.trace_data:
                trace_file = self.output_dir / f"pls_benchmark_trace_{i}_{timestamp}.tar.gz"
                with open(trace_file, 'wb') as f:
                    f.write(result.trace_data)
                trace_files.append(trace_file)
        
        summary = self.create_summary_statistics()
        summary_file = self.output_dir / f"pls_benchmark_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)
        
        self.console.print("\n[green]PLS Results saved:")
        if trace_files:
            for trace_file in trace_files:
                self.console.print(f"  Trace:    {trace_file}")
        self.console.print(f"  Summary:  {summary_file}")
    
    def create_summary_statistics(self) -> BenchmarkSummaryStatistics:
        """Create summary statistics from PLS benchmark results."""
        successful_results = [r for r in self.results if not r.error]
        
        # Create benchmark info
        benchmark_info = BenchmarkInfo(
            algorithm="PLS",
            timestamp=datetime.now().isoformat(),
            instances_tested=len(self.instance_files),
            iterations_per_instance=self.iterations,
            max_pls_iterations=self.max_iterations,
            total_runs=len(self.results),
            successful_runs=len(successful_results),
            failed_runs=len([r for r in self.results if r.error])
        )
        
        # Initialize collections
        performance_by_instance: dict[str, PerformanceByInstance] = {}
        overall_statistics: Optional[OverallStatistics] = None
        
        if successful_results:
            # Group results by instance
            for instance_file in self.instance_files:
                instance_name = instance_file.stem
                instance_results = [r for r in successful_results if r.instance_name == instance_name]
                if instance_results:
                    performance_by_instance[instance_name] = PerformanceByInstance(
                        num_runs=len(instance_results),
                        avg_runtime=statistics.mean([r.total_runtime_seconds for r in instance_results]),
                        avg_solutions=statistics.mean([r.num_final_solutions for r in instance_results]),
                        std_runtime=statistics.stdev([r.total_runtime_seconds for r in instance_results]) if len(instance_results) > 1 else 0,
                        min_runtime=min([r.total_runtime_seconds for r in instance_results]),
                        max_runtime=max([r.total_runtime_seconds for r in instance_results]),
                        min_solutions=min([r.num_final_solutions for r in instance_results]),
                        max_solutions=max([r.num_final_solutions for r in instance_results])
                    )
            
            # Overall statistics
            overall_statistics = OverallStatistics(
                avg_runtime=statistics.mean([r.total_runtime_seconds for r in successful_results]),
                avg_solutions=statistics.mean([r.num_final_solutions for r in successful_results]),
                total_runtime=sum([r.total_runtime_seconds for r in successful_results]),
                min_solutions=min([r.num_final_solutions for r in successful_results]),
                max_solutions=max([r.num_final_solutions for r in successful_results])
            )
        else:
            # Default overall statistics if no successful results
            overall_statistics = OverallStatistics(
                avg_runtime=0.0,
                avg_solutions=0.0,
                total_runtime=0.0
            )
        
        # Add validation statistics
        validation_stats = self._calculate_validation_statistics(successful_results)
        
        return BenchmarkSummaryStatistics(
            benchmark_info=benchmark_info,
            overall_statistics=overall_statistics,
            performance_by_instance=performance_by_instance,
            validation_statistics=validation_stats
        )
    
    def _calculate_validation_statistics(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """Calculate validation statistics from results."""
        total_with_validation = len([r for r in results if r.validation_results])
        total_valid = len([r for r in results 
                          if r.validation_results and r.validation_results.overall_valid])
        
        total_solutions = sum([r.num_final_solutions for r in results])
        total_invalid_solutions = sum([
            r.validation_results.summary.invalid_solutions 
            for r in results if r.validation_results
        ])
        total_warned_solutions = sum([
            r.validation_results.summary.solutions_with_warnings 
            for r in results if r.validation_results
        ])
        
        total_pareto_violations = len([
            r for r in results 
            if r.validation_results and 
            not r.validation_results.pareto_validation.is_valid_pareto
        ])
        
        return {
            "runs_with_validation": total_with_validation,
            "valid_runs": total_valid,
            "invalid_runs": total_with_validation - total_valid,
            "total_solutions": total_solutions,
            "invalid_solutions": total_invalid_solutions,
            "solutions_with_warnings": total_warned_solutions,
            "pareto_front_violations": total_pareto_violations,
            "run_validation_rate": (total_valid / total_with_validation * 100) if total_with_validation > 0 else 0,
            "solution_validity_rate": ((total_solutions - total_invalid_solutions) / total_solutions * 100) if total_solutions > 0 else 0
        }
    
    def print_summary(self):
        """Print a summary of PLS benchmark results."""
        successful = [r for r in self.results if not r.error]
        failed = [r for r in self.results if r.error]
        
        self.console.print("\n[bold green]PLS Benchmark Complete!")
        self.console.print(f"Total runs: {len(self.results)}")
        self.console.print(f"Successful: {len(successful)}")
        self.console.print(f"Failed: {len(failed)}")
        
        if successful:
            avg_runtime = statistics.mean([r.total_runtime_seconds for r in successful])
            avg_solutions = statistics.mean([r.num_final_solutions for r in successful])
            
            self.console.print(f"Average runtime: {avg_runtime:.2f}s")
            self.console.print(f"Average solutions found: {avg_solutions:.1f}")
            
            best_result = max(successful, key=lambda x: x.num_final_solutions)
            self.console.print(f"Best run: {best_result.instance_name} ({best_result.num_final_solutions} solutions)")
            
            # Print validation summary
            self.print_validation_summary(successful)
    
    def print_validation_summary(self, successful_results: list[BenchmarkResult]):
        """Print a summary of validation results."""
        if not successful_results:
            return
        
        self.console.print("\n[bold cyan]🔍 Validation Summary:")
        
        # Count validation results
        total_with_validation = len([r for r in successful_results if r.validation_results])
        total_valid = len([r for r in successful_results 
                          if r.validation_results and r.validation_results.overall_valid])
        total_invalid = total_with_validation - total_valid
        
        total_solutions = sum([r.num_final_solutions for r in successful_results])
        total_invalid_solutions = sum([
            r.validation_results.summary.invalid_solutions 
            for r in successful_results if r.validation_results
        ])
        total_warned_solutions = sum([
            r.validation_results.summary.solutions_with_warnings 
            for r in successful_results if r.validation_results
        ])
        
        # Count Pareto front violations
        total_pareto_violations = len([
            r for r in successful_results 
            if r.validation_results and 
            not r.validation_results.pareto_validation.is_valid_pareto
        ])
        
        self.console.print(f"  📊 Runs validated: {total_with_validation}/{len(successful_results)}")
        self.console.print(f"  ✅ Valid runs: {total_valid}")
        self.console.print(f"  ❌ Invalid runs: {total_invalid}")
        self.console.print(f"  📈 Total solutions: {total_solutions}")
        self.console.print(f"  🚫 Invalid solutions: {total_invalid_solutions}")
        self.console.print(f"  ⚠️  Solutions with warnings: {total_warned_solutions}")
        self.console.print(f"  🎯 Pareto front violations: {total_pareto_violations}")
        
        if total_invalid > 0:
            self.console.print(f"\n[yellow]⚠️  {total_invalid} runs had validation issues - check detailed results for specifics")
        
        if total_pareto_violations > 0:
            self.console.print(f"[red]❌ {total_pareto_violations} runs had non-Pareto fronts (dominated solutions present)")
        
        if total_invalid == 0 and total_pareto_violations == 0:
            self.console.print("[green]✅ All validated runs passed quality checks!")
            
        # Calculate validation statistics
        validation_rate = (total_valid / total_with_validation * 100) if total_with_validation > 0 else 0
        solution_validity_rate = ((total_solutions - total_invalid_solutions) / total_solutions * 100) if total_solutions > 0 else 0
        
        self.console.print(f"\n  📊 Run validation rate: {validation_rate:.1f}%")
        self.console.print(f"  📊 Solution validity rate: {solution_validity_rate:.1f}%")
    
    def create_instance_summary_statistics(self, instance_name: str, instance_results: list[BenchmarkResult]) -> BenchmarkSummaryStatistics:
        """Create summary statistics for a single instance."""
        successful_results = [r for r in instance_results if not r.error]
        
        # Create benchmark info
        benchmark_info = BenchmarkInfo(
            algorithm="PLS",
            timestamp=datetime.now().isoformat(),
            instances_tested=1,  # Just this instance
            iterations_per_instance=self.iterations,
            max_pls_iterations=self.max_iterations,
            total_runs=len(instance_results),
            successful_runs=len(successful_results),
            failed_runs=len([r for r in instance_results if r.error])
        )
        
        # Initialize collections
        performance_by_instance: dict[str, PerformanceByInstance] = {}
        overall_statistics: Optional[OverallStatistics] = None
        
        if successful_results:
            # Performance for this single instance
            performance_by_instance[instance_name] = PerformanceByInstance(
                num_runs=len(successful_results),
                avg_runtime=statistics.mean([r.total_runtime_seconds for r in successful_results]),
                avg_solutions=statistics.mean([r.num_final_solutions for r in successful_results]),
                std_runtime=statistics.stdev([r.total_runtime_seconds for r in successful_results]) if len(successful_results) > 1 else 0,
                min_runtime=min([r.total_runtime_seconds for r in successful_results]),
                max_runtime=max([r.total_runtime_seconds for r in successful_results]),
                min_solutions=min([r.num_final_solutions for r in successful_results]),
                max_solutions=max([r.num_final_solutions for r in successful_results])
            )
            
            # Overall statistics for this instance
            overall_statistics = OverallStatistics(
                avg_runtime=statistics.mean([r.total_runtime_seconds for r in successful_results]),
                avg_solutions=statistics.mean([r.num_final_solutions for r in successful_results]),
                total_runtime=sum([r.total_runtime_seconds for r in successful_results]),
                min_solutions=min([r.num_final_solutions for r in successful_results]),
                max_solutions=max([r.num_final_solutions for r in successful_results])
            )
        else:
            # Default overall statistics if no successful results
            overall_statistics = OverallStatistics(
                avg_runtime=0.0,
                avg_solutions=0.0,
                total_runtime=0.0
            )
        
        # Add validation statistics
        validation_stats = self._calculate_validation_statistics(successful_results)
        
        return BenchmarkSummaryStatistics(
            benchmark_info=benchmark_info,
            overall_statistics=overall_statistics,
            performance_by_instance=performance_by_instance,
            validation_statistics=validation_stats
        )
    
    def create_instance_3d_visualization(self, instance_name: str, instance_results: list[BenchmarkResult]) -> bool:
        """Create 3D visualization for a single instance using the base implementation."""
        return self._create_3d_visualization_base(instance_name, instance_results)


class MilpBenchmarkRunner(BenchmarkRunner):
    """MILP-specific benchmark runner."""
    
    def __init__(self, instances_dir: Path, output_dir: Path, iterations: int = 10, grid_points: int = 50, 
                 use_tui: bool = False, size_limit: Optional[int] = None, name_filter: Optional[str] = None, 
                 timeout: float = 300.0, validate_solutions: bool = True, bypass_coefficient: bool = True,
                 early_exit: bool = True, flag_array: bool = True, solver_name: str = "cbc", include_perfect: bool = False):
        super().__init__(instances_dir, output_dir, iterations, use_tui, size_limit, name_filter, timeout, validate_solutions, include_perfect)
        self.grid_points = grid_points
        self.bypass_coefficient = bypass_coefficient
        self.early_exit = early_exit
        self.flag_array = flag_array
        self.solver_name = solver_name
    
    def get_algorithm_name(self) -> str:
        """Get the name of the algorithm for this runner."""
        return "MILP"
    
    def get_visualization_filename_suffix(self) -> str:
        """Get the filename suffix for visualizations."""
        return "milp"
    
    def _create_explored_solution_detail(self, result: BenchmarkResult, cost: float, cloudy: float, incidence: float) -> str:
        """Create hover detail text for explored solutions (MILP doesn't use explored solutions)."""
        return (
            f"Instance: {result.instance_name}<br>"
            f"Iteration: {result.iteration}<br>"
            f"Type: MILP Solution<br>"
            f"Cost: {cost}<br>"
            f"Cloudy Area: {cloudy}<br>"
            f"Max Incidence Angle: {incidence}<br>"
            f"Runtime: {result.total_runtime_seconds:.2f}s<br>"
            f"Grid Points: {self.grid_points}"
        )
    
    def _create_final_solution_detail(self, result: BenchmarkResult, cost: float, cloudy: float, incidence: float) -> str:
        """Create hover detail text for final solutions."""
        return (
            f"Instance: {result.instance_name}<br>"
            f"Iteration: {result.iteration}<br>"
            f"Type: MILP Pareto Solution<br>"
            f"Cost: {cost}<br>"
            f"Cloudy Area: {cloudy}<br>"
            f"Max Incidence Angle: {incidence}<br>"
            f"Runtime: {result.total_runtime_seconds:.2f}s<br>"
            f"Grid Points: {self.grid_points}<br>"
            f"Solver: {self.solver_name}"
        )
    
    def load_instance(self, instance_file: Path) -> sims_problem.SimsDiscreteProblem:
        """Load a SIMS problem instance from .dzn file."""
        try:
            return sims_problem.SimsDiscreteProblem.from_dzn(str(instance_file))
        except Exception as e:
            raise RuntimeError(f"Failed to load instance {instance_file}: {e}")
    
    def run_milp_solver(
        self, 
        instance: sims_problem.SimsDiscreteProblem,
        timeout: timedelta = timedelta(seconds=300)
    ) -> sims_problem.SolvingResult:
        """Run MILP solver and return SolvingResult."""
        return sims_problem.solve_with_milp(
            instance,
            objectives=["min_cost", "cloud_coverage", "max_incidence_angle"],
            grid_points=self.grid_points,
            timeout=timeout,
            bypass_coefficient=self.bypass_coefficient,
            early_exit=self.early_exit,
            flag_array=self.flag_array,
            solver_name=self.solver_name,
            trace=True
        )
    
    def run_single_benchmark(
        self, 
        instance: sims_problem.SimsDiscreteProblem,
        instance_name: str,
        iteration: int,
        *args
    ) -> BenchmarkResult:
        """Run a single MILP benchmark and collect results."""
        # MILP doesn't use additional args like ratio, so we ignore them
        result = BenchmarkResult()
        result.instance_name = instance_name
        result.iteration = iteration
        result.ratio = (100, 0)  # Pure MILP
        
        try:
            start_time = time.time()
            solving_result = self.run_milp_solver(instance, timedelta(seconds=self.timeout))
            result.total_runtime_seconds = time.time() - start_time
            result.milp_runtime_seconds = result.total_runtime_seconds
            result.pls_runtime_seconds = 0.0
            
            # Extract solutions from SolvingResult
            solutions = solving_result.final_solutions
            result.final_solutions = solutions
            result.explored_solutions = solving_result.explored_solutions
            result.num_final_solutions = len(solutions)
            result.num_explored_solutions = len(solving_result.explored_solutions)
            result.milp_solutions = solutions  # Store MILP solutions separately
            result.num_milp_solutions = len(solutions)
            
            # Store trace data for saving
            result.trace_data = solving_result.trace

            print(f"DEBUG: MILP SOLUTIONS COUNT: {result.num_milp_solutions}")
            
            # Validate solutions and Pareto front
            if result.final_solutions and self.validate_solutions and self.validator:
                validation_report = self.validator.validate_benchmark_result(result, instance)
                result.validation_results = validation_report
                
                # Log validation issues if any
                if not validation_report.overall_valid:
                    self.console.print(f"[yellow]⚠️  Validation issues found for {instance_name} iter={iteration}")
                    
                    # Log invalid solutions
                    if validation_report.summary.invalid_solutions > 0:
                        self.console.print(f"   - {validation_report.summary.invalid_solutions} invalid solutions")
                    
                    # Log Pareto front issues
                    if not validation_report.pareto_validation.is_valid_pareto:
                        dominated_count = len(validation_report.pareto_validation.dominated_solutions)
                        self.console.print(f"   - {dominated_count} dominated solutions in Pareto front")
                
                # Log warnings
                if validation_report.summary.solutions_with_warnings > 0:
                    self.console.print(f"[blue]ℹ️  {validation_report.summary.solutions_with_warnings} solutions have warnings")
            
        except Exception as e:
            result.error = str(e)
            
        return result
    
    def run_benchmarks(self):
        """Run MILP benchmarks with optional TUI progress tracking."""
        if self.use_tui:
            self._run_benchmarks_with_tui()
        else:
            self._run_benchmarks_simple()
    
    def _run_benchmarks_simple(self):
        """Run MILP benchmarks with simple CLI output."""
        total_runs = len(self.instance_files) * self.iterations
        current_run = 0
        
        print(f"Starting MILP benchmark with {len(self.instance_files)} instances, {self.iterations} iterations each")
        print(f"Total runs: {total_runs}")
        print(f"Grid points per run: {self.grid_points}")
        print(f"Solver: {self.solver_name}")
        print("=" * 80)
        
        for instance_idx, instance_file in enumerate(self.instance_files):
            instance_name = instance_file.stem
            print(f"\nProcessing instance {instance_idx + 1}/{len(self.instance_files)}: {instance_name}")
            
            try:
                instance = self.load_instance(instance_file)
                print(f"  ✓ Loaded instance: {instance.num_images} images, {instance.universe} universe elements")
            except Exception as e:
                print(f"  ❌ Failed to load {instance_file}: {e}")
                continue
            
            # Track instance start position for per-instance saving
            instance_start_idx = len(self.results)
            
            for iteration in range(self.iterations):
                current_run += 1
                progress_pct = (current_run / total_runs) * 100
                
                print(f"  [{current_run:3d}/{total_runs}] ({progress_pct:5.1f}%) Iteration {iteration + 1} - ", end="", flush=True)
                
                result = self.run_single_benchmark(instance, instance_name, iteration)
                
                self.results.append(result)
                
                if result.error:
                    print(f"FAILED ({result.total_runtime_seconds:.1f}s) - {result.error}")
                else:
                    print(f"SUCCESS ({result.total_runtime_seconds:.1f}s) - {result.num_final_solutions} solutions")
            
            # Save results for this instance
            instance_results = self.results[instance_start_idx:]
            self.save_instance_results(instance_name, instance_results)
        
        print("\n" + "=" * 80)
        print("MILP benchmark completed!")
    
    def _run_benchmarks_with_tui(self):
        """Run MILP benchmarks with TUI progress tracking."""
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
            
            main_task = progress.add_task(
                f"[bold blue]MILP Benchmarking {len(self.instance_files)} instances", 
                total=len(self.instance_files)
            )
            
            iteration_task = progress.add_task(
                "[bold green]Iterations", 
                total=self.iterations,
                visible=False
            )
            
            current_task = progress.add_task(
                "[bold white]Current Run", 
                total=1,
                visible=False
            )
            
            for instance_idx, instance_file in enumerate(self.instance_files):
                instance_name = instance_file.stem
                
                progress.update(
                    main_task, 
                    description=f"[bold blue]Instance: {instance_name}",
                    completed=instance_idx
                )
                
                progress.update(iteration_task, completed=0, visible=True)
                
                try:
                    instance = self.load_instance(instance_file)
                except Exception as e:
                    progress.console.print(f"[red]❌ Failed to load {instance_file}: {e}")
                    continue
                
                # Track instance start position for per-instance saving
                instance_start_idx = len(self.results)
                
                for iteration in range(self.iterations):
                    progress.update(
                        iteration_task,
                        description=f"[bold green]Iteration {iteration + 1}/{self.iterations}",
                        completed=iteration
                    )
                    
                    progress.update(current_task, completed=0, visible=True, 
                                  description=f"[bold white]Running MILP (grid: {self.grid_points})")
                    
                    result = self.run_single_benchmark(instance, instance_name, iteration)
                    
                    self.results.append(result)
                    
                    progress.update(current_task, completed=1)
                    
                    # Show result in console
                    if result.error:
                        progress.console.print(f"[red]❌ {instance_name} iter={iteration + 1}: {result.error}")
                    else:
                        progress.console.print(f"[green]✅ {instance_name} iter={iteration + 1}: {result.num_final_solutions} solutions ({result.total_runtime_seconds:.1f}s)")
                
                progress.update(iteration_task, completed=self.iterations, visible=False)
                progress.update(current_task, visible=False)
                
                # Save results for this instance
                instance_results = self.results[instance_start_idx:]
                self.save_instance_results(instance_name, instance_results)
                
            progress.update(main_task, completed=len(self.instance_files))
    
    def calculate_summary_statistics(self, results: list[BenchmarkResult]) -> BenchmarkSummaryStatistics:
        """Calculate summary statistics for MILP results."""
        
        # Filter successful results
        successful_results: list[BenchmarkResult] = [r for r in results if r.error is None]
        failed_results = [r for r in results if r.error is not None]
        
        # Get unique instance names
        instance_names = list(set(r.instance_name for r in results))
        
        benchmark_info = BenchmarkInfo(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            instances_tested=len(instance_names),
            total_runs=len(results),
            successful_runs=len(successful_results),
            failed_runs=len(failed_results),
            algorithm="MILP"
        )
        
        # Initialize collections
        performance_by_instance: dict[str, PerformanceByInstance] = {}
        overall_statistics: Optional[OverallStatistics] = None
        
        if successful_results:
            # Performance by instance
            for instance_name in instance_names:
                instance_results = [r for r in successful_results if r.instance_name == instance_name]
                if instance_results:
                    performance_by_instance[instance_name] = PerformanceByInstance(
                        num_runs=len(instance_results),
                        avg_runtime=statistics.mean([r.total_runtime_seconds for r in instance_results]),
                        avg_solutions=statistics.mean([r.num_final_solutions for r in instance_results]),
                        std_runtime=statistics.stdev([r.total_runtime_seconds for r in instance_results]) if len(instance_results) > 1 else 0,
                        min_runtime=min([r.total_runtime_seconds for r in instance_results]),
                        max_runtime=max([r.total_runtime_seconds for r in instance_results]),
                        min_solutions=min([r.num_final_solutions for r in instance_results]),
                        max_solutions=max([r.num_final_solutions for r in instance_results])
                    )
            
            # Overall statistics
            overall_statistics = OverallStatistics(
                avg_runtime=statistics.mean([r.total_runtime_seconds for r in successful_results]),
                avg_solutions=statistics.mean([r.num_final_solutions for r in successful_results]),
                total_runtime=sum([r.total_runtime_seconds for r in successful_results]),
                min_solutions=min([r.num_final_solutions for r in successful_results]),
                max_solutions=max([r.num_final_solutions for r in successful_results])
            )
        else:
            # Default overall statistics if no successful results
            overall_statistics = OverallStatistics(
                avg_runtime=0.0,
                avg_solutions=0.0,
                total_runtime=0.0
            )
        
        # Add validation statistics
        validation_stats = self._calculate_validation_statistics(successful_results) if successful_results else {}
        
        return BenchmarkSummaryStatistics(
            benchmark_info=benchmark_info,
            overall_statistics=overall_statistics,
            performance_by_instance=performance_by_instance,
            validation_statistics=validation_stats
        )
    
    def create_summary_statistics(self) -> BenchmarkSummaryStatistics:
        """Create summary statistics for all MILP results."""
        return self.calculate_summary_statistics(self.results)
    
    def save_results(self):
        """Save MILP benchmark results to files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        finished_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save trace archives for results that have trace data
        trace_files = []
        for i, result in enumerate(self.results):
            if result.trace_data:
                trace_file = self.output_dir / f"milp_trace_{i}_{finished_at}.tar.gz"
                with open(trace_file, 'wb') as f:
                    f.write(result.trace_data)
                trace_files.append(trace_file)
        
        # Create and save summary statistics
        summary = self.create_summary_statistics()
        summary_file = self.output_dir / f"milp_summary_{finished_at}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)
        
        # Create comprehensive 3D visualization
        self.create_comprehensive_3d_visualization()
        
        self.console.print("\n[green]All MILP results saved:")
        if trace_files:
            for trace_file in trace_files:
                self.console.print(f"  Trace:    {trace_file}")
        self.console.print(f"  Summary:  {summary_file}")
    
    def create_comprehensive_3d_visualization(self) -> bool:
        """Create comprehensive 3D visualization for all instances."""
        try:
            finished_at = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_file = self.output_dir / f"milp_comprehensive_3d_{finished_at}.html"
            
            self.console.print("[blue]📊 Creating comprehensive 3D visualization...")
            
            # Group results by instance for separate plotting
            instance_groups: dict[str, list[BenchmarkResult]] = {}
            for result in self.results:
                if result.error is None:  # Only include successful results
                    if result.instance_name not in instance_groups:
                        instance_groups[result.instance_name] = []
                    instance_groups[result.instance_name].append(result)
            
            if not instance_groups:
                self.console.print("[yellow]⚠️  No successful results to visualize")
                return False
            
            # Create subplots for multiple instances
            from plotly.subplots import make_subplots
            
            num_instances = len(instance_groups)
            fig = make_subplots(
                rows=1, 
                cols=num_instances,
                subplot_titles=list(instance_groups.keys()),
                specs=[[{"type": "scatter3d"} for _ in range(num_instances)]]
            )
            
            for col_idx, (instance_name, instance_results) in enumerate(instance_groups.items(), 1):
                self._add_instance_to_subplot(fig, instance_results, instance_name, col_idx)
            
            fig.update_layout(
                title=f"MILP Algorithm - 3D Pareto Fronts (Grid: {self.grid_points}, Solver: {self.solver_name})",
                scene=dict(
                    xaxis_title="Cost",
                    yaxis_title="Cloudy Area",
                    zaxis_title="Max Incidence Angle"
                ),
                showlegend=True,
                height=600
            )
            
            fig.write_html(str(viz_file))
            self.console.print(f"[green]✅ 3D visualization saved: {viz_file}")
            return True
            
        except Exception as e:
            self.console.print(f"[red]❌ Failed to create 3D visualization: {e}")
            return False
    
    def _add_instance_to_subplot(self, fig, instance_results: list[BenchmarkResult], instance_name: str, col_idx: int):
        """Add instance data to subplot."""
        import plotly.graph_objects as go
        
        # Collect all solutions from all iterations for this instance
        all_costs = []
        all_cloudy = []
        all_incidence = []
        hover_texts = []
        
        for result in instance_results:
            if result.final_solutions:
                for solution in result.final_solutions:
                    all_costs.append(solution.cost)
                    all_cloudy.append(solution.cloudy_area)
                    all_incidence.append(solution.max_incidence_angle or 0)
                    hover_texts.append(self._create_final_solution_detail(result, 
                                                                         solution.cost,
                                                                         solution.cloudy_area,
                                                                         solution.max_incidence_angle or 0))
        
        if all_costs:
            fig.add_trace(
                go.Scatter3d(
                    x=all_costs,
                    y=all_cloudy,
                    z=all_incidence,
                    mode='markers',
                    marker=dict(
                        size=6,
                        color='red',
                        symbol='circle'
                    ),
                    name=f"{instance_name} MILP",
                    text=hover_texts,
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=True
                ),
                row=1, col=col_idx
            )
    
    def save_instance_summary_statistics(self, instance_name: str, instance_results: list[BenchmarkResult]) -> BenchmarkSummaryStatistics:
        """Calculate and save summary statistics for a single instance with MILP-specific information."""
        
        # Filter successful results
        successful_results: list[BenchmarkResult] = [r for r in instance_results if r.error is None]
        failed_results = [r for r in instance_results if r.error is not None]
        
        benchmark_info = BenchmarkInfo(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            instances_tested=1,
            total_runs=len(instance_results),
            successful_runs=len(successful_results),
            failed_runs=len(failed_results),
            algorithm="MILP"
        )
        
        # Initialize collections
        performance_by_instance: dict[str, PerformanceByInstance] = {}
        overall_statistics: Optional[OverallStatistics] = None
        
        if successful_results:
            # Performance for this single instance
            performance_by_instance[instance_name] = PerformanceByInstance(
                num_runs=len(successful_results),
                avg_runtime=statistics.mean([r.total_runtime_seconds for r in successful_results]),
                avg_solutions=statistics.mean([r.num_final_solutions for r in successful_results]),
                std_runtime=statistics.stdev([r.total_runtime_seconds for r in successful_results]) if len(successful_results) > 1 else 0,
                min_runtime=min([r.total_runtime_seconds for r in successful_results]),
                max_runtime=max([r.total_runtime_seconds for r in successful_results]),
                min_solutions=min([r.num_final_solutions for r in successful_results]),
                max_solutions=max([r.num_final_solutions for r in successful_results])
            )
            
            # Overall statistics for this instance
            overall_statistics = OverallStatistics(
                avg_runtime=statistics.mean([r.total_runtime_seconds for r in successful_results]),
                avg_solutions=statistics.mean([r.num_final_solutions for r in successful_results]),
                total_runtime=sum([r.total_runtime_seconds for r in successful_results]),
                min_solutions=min([r.num_final_solutions for r in successful_results]),
                max_solutions=max([r.num_final_solutions for r in successful_results])
            )
        else:
            # Default overall statistics if no successful results
            overall_statistics = OverallStatistics(
                avg_runtime=0.0,
                avg_solutions=0.0,
                total_runtime=0.0
            )
        
        # Add validation statistics
        validation_stats = self._calculate_validation_statistics(successful_results)
        
        return BenchmarkSummaryStatistics(
            benchmark_info=benchmark_info,
            overall_statistics=overall_statistics,
            performance_by_instance=performance_by_instance,
            validation_statistics=validation_stats
        )
    
    def create_instance_3d_visualization(self, instance_name: str, instance_results: list[BenchmarkResult]) -> bool:
        """Create 3D visualization for a single instance using the base implementation."""
        return self._create_3d_visualization_base(instance_name, instance_results)
    
    def create_instance_summary_statistics(self, instance_name: str, instance_results: list[BenchmarkResult]) -> BenchmarkSummaryStatistics:
        """Create summary statistics for a single instance with MILP-specific information."""
        return self.save_instance_summary_statistics(instance_name, instance_results)
    
    def print_summary(self):
        """Print comprehensive benchmark summary with MILP-specific information."""
        summary = self.create_summary_statistics()
        
        self.console.print("\n" + "="*80)
        self.console.print("[bold blue]📊 MILP Benchmark Summary")
        self.console.print("="*80)
        
        # Basic info
        info = summary.benchmark_info
        self.console.print(f"[bold]Algorithm:[/bold] {info.algorithm}")
        self.console.print(f"[bold]Grid Points:[/bold] {self.grid_points}")
        self.console.print(f"[bold]Solver:[/bold] {self.solver_name}")
        self.console.print(f"[bold]Instances Tested:[/bold] {info.instances_tested}")
        self.console.print(f"[bold]Total Runs:[/bold] {info.total_runs}")
        self.console.print(f"[bold]Successful Runs:[/bold] {info.successful_runs}")
        self.console.print(f"[bold]Failed Runs:[/bold] {info.failed_runs}")
        
        success_rate = (info.successful_runs / info.total_runs * 100) if info.total_runs > 0 else 0
        self.console.print(f"[bold]Success Rate:[/bold] {success_rate:.1f}%")
        
        # Overall performance
        if summary.overall_statistics and info.successful_runs > 0:
            stats = summary.overall_statistics
            self.console.print("\n[bold blue]⚡ Overall Performance")
            self.console.print(f"  Average Runtime: {stats.avg_runtime:.2f}s")
            self.console.print(f"  Total Runtime: {stats.total_runtime:.2f}s")
            self.console.print(f"  Average Solutions: {stats.avg_solutions:.1f}")
            self.console.print(f"  Solution Range: {stats.min_solutions}-{stats.max_solutions}")
        
        # Performance by instance
        if summary.performance_by_instance:
            self.console.print("\n[bold blue]📋 Performance by Instance")
            for instance_name, perf in summary.performance_by_instance.items():
                self.console.print(f"  [bold]{instance_name}:[/bold]")
                self.console.print(f"    Runtime: {perf.avg_runtime:.2f}s ± {perf.std_runtime:.2f}s ({perf.min_runtime:.2f}-{perf.max_runtime:.2f}s)")
                self.console.print(f"    Solutions: {perf.avg_solutions:.1f} ({perf.min_solutions}-{perf.max_solutions})")
                self.console.print(f"    Runs: {perf.num_runs}")
        
        # Validation results
        if self.validate_solutions:
            successful_results: list[BenchmarkResult] = [r for r in self.results if r.error is None]
            if successful_results:
                self.print_validation_summary(successful_results)
        
        self.console.print("="*80)


class GurobiBenchmarkRunner(BenchmarkRunner):
    """Gurobi-based benchmark runner using solve_milp method from sims-solvers."""
    
    def __init__(self, instances_dir: Path, output_dir: Path, iterations: int = 10, 
                 use_tui: bool = False, size_limit: Optional[int] = None, name_filter: Optional[str] = None, 
                 timeout: float = 300.0, validate_solutions: bool = True, solver_name: str = "gurobi",
                 front_strategy: str = "saugmecon", include_perfect: bool = False):
        super().__init__(instances_dir, output_dir, iterations, use_tui, size_limit, name_filter, timeout, validate_solutions, include_perfect)
        self.solver_name = solver_name
        self.front_strategy = front_strategy
        
        # Import sims-solvers modules
        try:
            import sys
            from pathlib import Path
            # Add sims-solvers to path
            sims_solvers_path = Path(__file__).parent.parent / "sims-solvers"
            if str(sims_solvers_path) not in sys.path:
                sys.path.insert(0, str(sims_solvers_path))
            
            from sims_solvers.solve import solve_milp
            from sims_solvers.Config import Config
            from sims_solvers.solve import MZN_MODEL_PATH
            from sims_solvers.Instances.InstanceSIMS import InstanceSIMS
            self.solve_milp = solve_milp
            self.Config = Config
            self.MZN_MODEL_PATH = MZN_MODEL_PATH
            self.InstanceSIMS = InstanceSIMS
        except ImportError as e:
            raise ImportError(f"Failed to import sims-solvers modules: {e}")
    
    def get_algorithm_name(self) -> str:
        """Get the name of the algorithm for this runner."""
        return f"Gurobi-{self.front_strategy.upper()}"
    
    def get_visualization_filename_suffix(self) -> str:
        """Get the filename suffix for visualizations."""
        return f"gurobi_{self.front_strategy}"
    
    def _create_explored_solution_detail(self, result: BenchmarkResult, cost: float, cloudy: float, incidence: float) -> str:
        """Create hover detail text for explored solutions."""
        return (
            f"Instance: {result.instance_name}<br>"
            f"Iteration: {result.iteration}<br>"
            f"Type: Gurobi Solution<br>"
            f"Cost: {cost}<br>"
            f"Cloudy Area: {cloudy}<br>"
            f"Max Incidence Angle: {incidence}<br>"
            f"Runtime: {result.total_runtime_seconds:.2f}s<br>"
            f"Solver: {self.solver_name}<br>"
            f"Strategy: {self.front_strategy}"
        )
    
    def _create_final_solution_detail(self, result: BenchmarkResult, cost: float, cloudy: float, incidence: float) -> str:
        """Create hover detail text for final solutions."""
        return (
            f"Instance: {result.instance_name}<br>"
            f"Iteration: {result.iteration}<br>"
            f"Type: Gurobi Pareto Solution<br>"
            f"Cost: {cost}<br>"
            f"Cloudy Area: {cloudy}<br>"
            f"Max Incidence Angle: {incidence}<br>"
            f"Runtime: {result.total_runtime_seconds:.2f}s<br>"
            f"Solver: {self.solver_name}<br>"
            f"Strategy: {self.front_strategy}"
        )
    
    def load_instance(self, instance_file: Path) -> sims_problem.SimsDiscreteProblem:
        """Load a SIMS problem instance from .dzn file."""
        try:
            return sims_problem.SimsDiscreteProblem.from_dzn(str(instance_file))
        except Exception as e:
            raise RuntimeError(f"Failed to load instance {instance_file}: {e}")
    
    def parse_dzn_data(self, dzn_file_path: Path) -> dict:
        """Parse DZN file data for InstanceSIMS."""
        data = {}
        
        with open(dzn_file_path, 'r') as f:
            content = f.read()
        
        # Simple parsing for basic data structures
        import re
        
        # Parse arrays
        def parse_array(pattern, content):
            match = re.search(pattern, content, re.DOTALL)
            if match:
                array_str = match.group(1)
                # Remove brackets and split by comma, then convert to appropriate type
                items = [item.strip() for item in array_str.strip('[]').split(',')]
                return [float(item) if '.' in item else int(item) for item in items if item.strip()]
            return []
        
        # Parse set arrays (for images and clouds)
        def parse_set_array(pattern, content):
            match = re.search(pattern, content, re.DOTALL)
            if match:
                array_str = match.group(1)
                sets = []
                # Find all sets in the array
                set_matches = re.findall(r'\{([^}]*)\}', array_str)
                for set_match in set_matches:
                    if set_match.strip():
                        # Convert to set of integers (1-indexed in DZN, will be corrected later)
                        elements = [int(x.strip()) for x in set_match.split(',') if x.strip()]
                        sets.append(set(elements))
                    else:
                        sets.append(set())
                return sets
            return []
        
        # Extract data
        data['costs'] = parse_array(r'costs\s*=\s*\[([^\]]+)\]', content)
        data['areas'] = parse_array(r'areas\s*=\s*\[([^\]]+)\]', content)
        data['resolution'] = parse_array(r'resolution\s*=\s*\[([^\]]+)\]', content)
        data['incidence_angle'] = parse_array(r'incidence_angle\s*=\s*\[([^\]]+)\]', content)
        
        data['images'] = parse_set_array(r'images\s*=\s*\[([^\]]+)\]', content)
        data['clouds'] = parse_set_array(r'clouds\s*=\s*\[([^\]]+)\]', content)
        
        # Extract max_cloud_area
        match = re.search(r'max_cloud_area\s*=\s*(\d+)', content)
        data['max_cloud_area'] = int(match.group(1)) if match else 0
        
        return data
    
    def create_gurobi_config(self, instance_name: str, dzn_file: Path):
        """Create Gurobi configuration for solve_milp."""
        import tempfile
        
        # Create temporary directory for output
        temp_dir = tempfile.mkdtemp()
        
        config = self.Config(
            minizinc_data=False,  # Use direct Gurobi model instead of MiniZinc
            instance_name=instance_name,
            data_sets_folder=dzn_file.parent,
            input_mzn=self.MZN_MODEL_PATH,  # Required but not used for direct Gurobi
            dzn_dir=dzn_file.parent,
            solver_name=self.solver_name,
            problem_name="sims",
            front_strategy=self.front_strategy,
            solver_timeout_sec=int(self.timeout),
            summary_filename=Path(temp_dir) / f"{instance_name}_summary.csv",
            solver_search_strategy="free_search",
            fzn_optimisation_level=1,
            cores=1,
            threads=1
        )
        return config
    
    def convert_solutions_to_benchmark_format(self, solver_solutions: list) -> list[sims_problem.Solution]:
        """Convert solutions from solve_milp format to benchmark format."""
        benchmark_solutions = []
        
        for i, sol_data in enumerate(solver_solutions):
            if sol_data is None:
                continue
                
            # Extract objective values and solution values
            if hasattr(sol_data, 'solution') and hasattr(sol_data.solution, 'objs'):
                # MinizincResultFormat style
                objs = sol_data.solution.objs
                solution_values = sol_data.solution.solution_values
            elif isinstance(sol_data, dict):
                # Dictionary style
                objs = sol_data.get("objs", [])
                solution_values = sol_data.get("solution_values", [])
            else:
                # Direct solution object
                objs = getattr(sol_data, 'objs', [])
                solution_values = getattr(sol_data, 'solution_values', [])
            
            # Ensure we have the right number of objectives (4 for SIMS)
            if len(objs) < 4:
                # Pad with default values if missing
                while len(objs) < 4:
                    objs.append(0)
            
            # Convert boolean list to indices of selected images (1-based indexing)
            if isinstance(solution_values, list) and len(solution_values) > 0:
                if isinstance(solution_values[0], bool):
                    # Boolean array - convert to set of selected indices
                    selected_images = {idx + 1 for idx, selected in enumerate(solution_values) if selected}
                else:
                    # Assume it's already indices
                    selected_images = set(solution_values)
            else:
                selected_images = set()
            
            # Create benchmark Solution object
            # Convert set to list for Solution.create()
            selected_images_list = list(selected_images)
            
            # Convert timedelta to microseconds
            timestamp_us = int(i * 100_000)  # 0.1 seconds in microseconds
            
            benchmark_solution = Solution.create(
                selected_images=selected_images_list,
                cost=int(objs[0]) if len(objs) > 0 else 0,
                cloudy_area=int(objs[1]) if len(objs) > 1 else 0,
                timestamp_us=timestamp_us,
                max_incidence_angle=int(objs[3]) if len(objs) >= 4 and objs[3] is not None else (
                    int(objs[2]) if len(objs) > 2 and objs[2] is not None else None
                ),
                min_resolutions_sum=int(objs[2]) if len(objs) >= 4 and objs[2] is not None else None
            )
            
            benchmark_solutions.append(benchmark_solution)
        
        return benchmark_solutions
    
    def run_single_benchmark(
        self, 
        instance: sims_problem.SimsDiscreteProblem,
        instance_name: str,
        iteration: int,
        *args
    ) -> BenchmarkResult:
        """Run a single Gurobi benchmark and collect results."""
        result = BenchmarkResult()
        result.instance_name = instance_name
        result.iteration = iteration
        result.ratio = (0, 100)  # Pure Gurobi MILP
        
        try:
            # Find the corresponding .dzn file
            dzn_file = None
            for potential_file in self.instance_files:
                if potential_file.stem == instance_name:
                    dzn_file = potential_file
                    break
            
            if dzn_file is None:
                raise RuntimeError(f"Could not find .dzn file for instance {instance_name}")
            
            # Parse DZN data and create InstanceSIMS
            dzn_data = self.parse_dzn_data(dzn_file)
            
            # Create configuration for solver
            config = self.create_gurobi_config(instance_name, dzn_file)
            
            # Capture solutions during solve_milp
            captured_solutions = []
            
            # Build the solver pipeline using InstanceSIMS data
            from sims_solvers.main import build_model, build_solver
            from sims_solvers.Instances.InstanceSIMS import InstanceSIMS
            
            start_time = time.time()
            
            # Create InstanceSIMS from parsed data
            sims_instance = InstanceSIMS(dzn_data)
            
            # Build model and solver using the SIMS instance
            model = build_model(sims_instance, config)
            solver, pareto_front = build_solver(model, sims_instance, config, {})
            
            # Collect solutions
            try:
                for solution in solver.solve():
                    if solution is not None:
                        captured_solutions.append(solution)
            except Exception as e:
                print(f"Warning: Error during solving: {e}")
            
            result.total_runtime_seconds = time.time() - start_time
            result.milp_runtime_seconds = result.total_runtime_seconds
            result.pls_runtime_seconds = 0.0
            
            # Convert solutions to benchmark format
            benchmark_solutions = self.convert_solutions_to_benchmark_format(captured_solutions)
            
            result.final_solutions = benchmark_solutions
            result.explored_solutions = []  # Gurobi doesn't track explored solutions the same way
            result.num_final_solutions = len(benchmark_solutions)
            result.num_explored_solutions = 0
            result.milp_solutions = benchmark_solutions
            result.num_milp_solutions = len(benchmark_solutions)
            
            print(f"DEBUG: GUROBI SOLUTIONS COUNT: {result.num_milp_solutions}")
            
            # Validate solutions and Pareto front
            if result.final_solutions and self.validate_solutions and self.validator:
                validation_report = self.validator.validate_benchmark_result(result, instance)
                result.validation_results = validation_report
                
                # Log validation issues if any
                if not validation_report.overall_valid:
                    self.console.print(f"[yellow]⚠️  Validation issues found for {instance_name} iter={iteration}")
                    
                    # Log invalid solutions
                    if validation_report.summary.invalid_solutions > 0:
                        self.console.print(f"   - {validation_report.summary.invalid_solutions} invalid solutions")
                    
                    # Log dominated solutions
                    if not validation_report.pareto_validation.is_valid_pareto:
                        dominated_count = len(validation_report.pareto_validation.dominated_solutions)
                        self.console.print(f"   - {dominated_count} dominated solutions in Pareto front")
        
        except Exception as e:
            result.error = str(e)
            print(f"Error in Gurobi benchmark for {instance_name}: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def run_benchmarks(self):
        """Run all Gurobi benchmarks."""
        if self.use_tui:
            self._run_benchmarks_with_tui()
        else:
            self._run_benchmarks_simple()
    
    def _run_benchmarks_simple(self):
        """Run Gurobi benchmarks with simple CLI output."""
        total_runs = len(self.instance_files) * self.iterations
        current_run = 0
        
        print(f"Starting Gurobi benchmark with {len(self.instance_files)} instances, {self.iterations} iterations each")
        print(f"Total runs: {total_runs}")
        print(f"Solver: {self.solver_name}")
        print(f"Strategy: {self.front_strategy}")
        print("=" * 80)
        
        for instance_idx, instance_file in enumerate(self.instance_files):
            instance_name = instance_file.stem
            print(f"\nProcessing instance {instance_idx + 1}/{len(self.instance_files)}: {instance_name}")
            
            try:
                instance = self.load_instance(instance_file)
                print(f"  ✓ Loaded instance: {instance.num_images} images, {instance.universe} universe elements")
            except Exception as e:
                print(f"  ❌ Failed to load {instance_file}: {e}")
                continue
            
            # Track instance start position for per-instance saving
            instance_start_idx = len(self.results)
            
            for iteration in range(self.iterations):
                current_run += 1
                progress_pct = (current_run / total_runs) * 100
                
                print(f"    Iteration {iteration + 1}/{self.iterations} (Run {current_run}/{total_runs}, {progress_pct:.1f}%)")
                
                try:
                    result = self.run_single_benchmark(instance, instance_name, iteration)
                    self.results.append(result)
                    
                    if result.error:
                        print(f"      ❌ Failed: {result.error}")
                    else:
                        print(f"      ✓ Success: {result.num_final_solutions} solutions in {result.total_runtime_seconds:.2f}s")
                        
                except KeyboardInterrupt:
                    print(f"\n  ⏹️  Interrupted during {instance_name}")
                    raise
                except Exception as e:
                    print(f"      ❌ Error: {e}")
                    # Create error result
                    error_result = BenchmarkResult()
                    error_result.instance_name = instance_name
                    error_result.iteration = iteration
                    error_result.error = str(e)
                    self.results.append(error_result)
            
            # Save results for this instance
            instance_results = self.results[instance_start_idx:]
            if instance_results:
                self.save_instance_results(instance_name, instance_results)
        
        print("\n✓ Completed all benchmark runs!")
    
    def _run_benchmarks_with_tui(self):
        """Run benchmarks with TUI (reuse base implementation)."""
        # For TUI, we can use the base implementation since it's generic
        total_runs = len(self.instance_files) * self.iterations
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console
        ) as progress:
            
            # Create main task
            main_task = progress.add_task("Running Gurobi Benchmarks", total=total_runs)
            
            for instance_idx, instance_file in enumerate(self.instance_files):
                instance_name = instance_file.stem
                
                try:
                    instance = self.load_instance(instance_file)
                except Exception as e:
                    self.console.print(f"[red]Failed to load {instance_file}: {e}")
                    continue
                
                # Track instance start position for per-instance saving
                instance_start_idx = len(self.results)
                
                for iteration in range(self.iterations):
                    progress.update(main_task, description=f"[bold blue]Instance: {instance_name} | Iteration: {iteration + 1}")
                    
                    try:
                        result = self.run_single_benchmark(instance, instance_name, iteration)
                        self.results.append(result)
                        
                    except KeyboardInterrupt:
                        progress.update(main_task, description="[red]Interrupted by user")
                        raise
                    except Exception as e:
                        # Create error result
                        error_result = BenchmarkResult()
                        error_result.instance_name = instance_name
                        error_result.iteration = iteration
                        error_result.error = str(e)
                        self.results.append(error_result)
                    
                    progress.advance(main_task)
                
                # Save results for this instance
                instance_results = self.results[instance_start_idx:]
                if instance_results:
                    self.save_instance_results(instance_name, instance_results)
    
    def create_summary_statistics(self) -> BenchmarkSummaryStatistics:
        """Create summary statistics for Gurobi benchmarks."""
        # Filter successful results
        successful_results: list[BenchmarkResult] = [r for r in self.results if r.error is None]
        failed_results = [r for r in self.results if r.error is not None]
        
        # Calculate overall statistics
        overall_stats = None
        if successful_results:
            runtimes = [r.total_runtime_seconds for r in successful_results]
            solution_counts = [r.num_final_solutions for r in successful_results]
            
            overall_stats = OverallStatistics(
                avg_runtime=statistics.mean(runtimes),
                avg_solutions=statistics.mean(solution_counts),
                total_runtime=sum(runtimes),
                min_solutions=min(solution_counts),
                max_solutions=max(solution_counts)
            )
        
        # Calculate performance by instance
        performance_by_instance = {}
        instances = set(r.instance_name for r in successful_results)
        
        for instance_name in instances:
            instance_results = [r for r in successful_results if r.instance_name == instance_name]
            if instance_results:
                instance_runtimes = [r.total_runtime_seconds for r in instance_results]
                instance_solutions = [r.num_final_solutions for r in instance_results]
                
                performance_by_instance[instance_name] = PerformanceByInstance(
                    num_runs=len(instance_results),
                    avg_runtime=statistics.mean(instance_runtimes),
                    std_runtime=statistics.stdev(instance_runtimes) if len(instance_runtimes) > 1 else 0.0,
                    min_runtime=min(instance_runtimes),
                    max_runtime=max(instance_runtimes),
                    avg_solutions=statistics.mean(instance_solutions),
                    min_solutions=min(instance_solutions),
                    max_solutions=max(instance_solutions)
                )
        
        # Calculate validation statistics
        validation_stats = {}
        if self.validate_solutions and successful_results:
            results_with_validation = [r for r in successful_results if hasattr(r, 'validation_results') and r.validation_results]
            
            if results_with_validation:
                total_solutions = sum(r.num_final_solutions for r in results_with_validation)
                total_invalid = sum(len(getattr(r.validation_results, 'invalid_solutions', [])) for r in results_with_validation)
                
                validation_stats = {
                    'total_solutions': total_solutions,
                    'total_invalid_solutions': total_invalid,
                    'validation_rate': len(results_with_validation) / len(successful_results) * 100,
                    'overall_valid_rate': (total_solutions - total_invalid) / total_solutions * 100 if total_solutions > 0 else 0
                }
        
        # Create benchmark info
        from datetime import datetime
        info = BenchmarkInfo(
            timestamp=datetime.now().isoformat(),
            algorithm=self.get_algorithm_name(),
            instances_tested=len(set(r.instance_name for r in self.results)),
            total_runs=len(self.results),
            successful_runs=len(successful_results),
            failed_runs=len(failed_results)
        )
        
        return BenchmarkSummaryStatistics(
            benchmark_info=info,
            overall_statistics=overall_stats or OverallStatistics(0.0, 0.0, 0.0),
            performance_by_instance=performance_by_instance,
            validation_statistics=validation_stats
        )
    
    def save_results(self):
        """Save Gurobi benchmark results."""
        finished_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results file
        results_file = self.output_dir / f"gurobi_{self.front_strategy}_results_{finished_at}.json"
        
        results_data = {
            "algorithm": self.get_algorithm_name(),
            "solver": self.solver_name,
            "front_strategy": self.front_strategy,
            "finished_at": finished_at,
            "instances": len(set(r.instance_name for r in self.results)),
            "total_runs": len(self.results),
            "results": [result.to_dict() for result in self.results]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.console.print(f"\n[green]Results saved to: {results_file}")
        
        # Also save summary statistics
        summary = self.create_summary_statistics()
        summary_file = self.output_dir / f"gurobi_{self.front_strategy}_summary_{finished_at}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)
        
        self.console.print(f"[green]Summary saved to: {summary_file}")
    
    def create_instance_3d_visualization(self, instance_name: str, instance_results: list[BenchmarkResult]) -> bool:
        """Create 3D visualization for a single instance - for 4D problems, create 3 plots."""
        return self._create_4d_visualization_multiple_plots(instance_name, instance_results)
    
    def _create_4d_visualization_multiple_plots(self, instance_name: str, instance_results: list[BenchmarkResult]) -> bool:
        """Create separate 4D visualizations for each ratio configuration."""
        # Filter successful results
        successful_results = [r for r in instance_results if not r.error and r.final_solutions]
        
        if not successful_results:
            self.console.print(f"❌ No successful results for {instance_name} to visualize")
            return False
        
        # Group results by ratio configuration
        results_by_ratio = {}
        for result in successful_results:
            ratio_key = f"{result.ratio[0]}_{result.ratio[1]}"
            if ratio_key not in results_by_ratio:
                results_by_ratio[ratio_key] = []
            results_by_ratio[ratio_key].append(result)
        
        all_created = True
        for ratio_key, ratio_results in results_by_ratio.items():
            self.console.print(f"📊 Creating 4D visualization for {instance_name} ratio {ratio_key}...")
            if not self._create_4d_visualization_for_ratio(instance_name, ratio_results, ratio_key):
                all_created = False
        
        return all_created
    
    def _create_4d_visualization_for_ratio(self, instance_name: str, ratio_results: list[BenchmarkResult], ratio_key: str) -> bool:
        """Create a single HTML with 3 subplots showing different objective combinations for 4D problems for one ratio."""
        
        # Check if plotly is available 
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            self.console.print("❌ Plotly not available for creating plots")
            return False
        
        # Define the 4 objective combinations to visualize
        combinations = [
            {
                'indices': (0, 1, 2),  # Cost, Cloud, Resolution
                'labels': ('Cost (minimize)', 'Cloudy Area Coverage (minimize)', 'Resolution Sum (minimize)'),
                'suffix': 'cost_cloud_resolution',
                'title': 'Cost vs Cloud vs Resolution',
                'row': 1, 'col': 1
            },
            {
                'indices': (0, 1, 3),  # Cost, Cloud, Incidence Angle
                'labels': ('Cost (minimize)', 'Cloudy Area Coverage (minimize)', 'Max Incidence Angle (minimize)'),
                'suffix': 'cost_cloud_angle',
                'title': 'Cost vs Cloud vs Incidence Angle',
                'row': 1, 'col': 2
            },
            {
                'indices': (0, 2, 3),  # Cost, Resolution, Incidence Angle
                'labels': ('Cost (minimize)', 'Resolution Sum (minimize)', 'Max Incidence Angle (minimize)'),
                'suffix': 'cost_resolution_angle',
                'title': 'Cost vs Resolution vs Incidence Angle',
                'row': 2, 'col': 1
            },
            {
                'indices': (1, 2, 3),  # Cloud, Resolution, Incidence Angle
                'labels': ('Cloudy Area Coverage (minimize)', 'Resolution Sum (minimize)', 'Max Incidence Angle (minimize)'),
                'suffix': 'cloud_resolution_angle',
                'title': 'Cloud vs Resolution vs Incidence Angle',
                'row': 2, 'col': 2
            }
        ]
        
        try:
            # Create subplots with 3D scenes in a 2x2 grid
            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{'type': 'scene'}, {'type': 'scene'}], 
                       [{'type': 'scene'}, {'type': 'scene'}]],
                subplot_titles=[combo['title'] for combo in combinations],
                horizontal_spacing=0.05
            )
            
            # Collect and process visualization data using the enhanced method
            explored_data, final_data, jsonl_data, timestamp_range = self._collect_visualization_data(ratio_results)
            
            # Unpack final data
            if len(final_data) >= 7:
                final_costs, final_cloudy_areas, final_incidence_angles, final_norm_timestamps, final_solution_details, final_colors, final_solver_types = final_data
            else:
                # Fallback for compatibility - should not happen with hybrid solver
                final_costs, final_cloudy_areas, final_incidence_angles, final_norm_timestamps, final_solution_details = final_data[:5]
                self.console.print(f"⚠️ Warning: Using fallback coloring for {instance_name} ratio {ratio_key}")
                # For hybrid solver, try to determine based on milp_solutions
                final_colors = []
                final_solver_types = []
                for result in ratio_results:
                    result_final_costs, result_final_cloudy, result_final_incidence = self.extract_solution_objectives(result.final_solutions)
                    if result.milp_solutions:
                        milp_costs, milp_cloudy, milp_incidence = self.extract_solution_objectives(result.milp_solutions)
                        milp_coords = set(zip(milp_costs, milp_cloudy, milp_incidence))
                        for cost, cloudy, incidence in zip(result_final_costs, result_final_cloudy, result_final_incidence):
                            if (cost, cloudy, incidence) in milp_coords:
                                final_colors.append('red')
                                final_solver_types.append('Gurobi')
                            else:
                                final_colors.append('blue')
                                final_solver_types.append('PLS')
                    else:
                        # No MILP solutions available, assume all are from the primary solver
                        final_colors.extend(['red'] * len(result_final_costs))
                        final_solver_types.extend(['Gurobi'] * len(result_final_costs))
            
            # Extract 4D objectives
            final_objectives_4d = self.extract_solution_objectives_4d([sol for result in ratio_results for sol in result.final_solutions])
            
            if not final_objectives_4d or not final_objectives_4d[0]:
                self.console.print(f"❌ No 4D solutions found for {instance_name} ratio {ratio_key}")
                return False
            
            # Add traces for each subplot
            for combo in combinations:
                # Extract the 3 objectives for this combination
                x_data = final_objectives_4d[combo['indices'][0]]
                y_data = final_objectives_4d[combo['indices'][1]]
                z_data = final_objectives_4d[combo['indices'][2]]
                
                # Debug: Print solver type distribution
                solver_type_counts = {}
                for solver_type in final_solver_types:
                    solver_type_counts[solver_type] = solver_type_counts.get(solver_type, 0) + 1
                print(f"Debug: Solver types in {instance_name} ratio {ratio_key}: {solver_type_counts}")
                
                # Separate solutions by solver type for different symbols and colors
                gurobi_indices = [i for i, solver_type in enumerate(final_solver_types) if solver_type == 'Gurobi']
                pls_indices = [i for i, solver_type in enumerate(final_solver_types) if solver_type == 'PLS']
                
                print(f"Debug: Found {len(gurobi_indices)} Gurobi solutions, {len(pls_indices)} PLS solutions")
                
                # Add Gurobi solutions as red diamonds
                if gurobi_indices:
                    fig.add_trace(
                        go.Scatter3d(
                            x=[x_data[i] for i in gurobi_indices],
                            y=[y_data[i] for i in gurobi_indices],
                            z=[z_data[i] for i in gurobi_indices],
                            mode='markers',
                            marker=dict(
                                size=6,
                                color='red',
                                opacity=0.8,
                                line=dict(color='black', width=0.5),
                                symbol='diamond'
                            ),
                            text=[final_solution_details[i] for i in gurobi_indices],
                            hovertemplate='%{text}<extra></extra>',
                            name=f'Gurobi Solutions ({len(gurobi_indices)})',
                            legendgroup='gurobi',
                            showlegend=(combo == combinations[0])  # Only show legend for first plot
                        ),
                        row=combo['row'], col=combo['col']
                    )
                
                # Add PLS solutions as blue circles
                if pls_indices:
                    fig.add_trace(
                        go.Scatter3d(
                            x=[x_data[i] for i in pls_indices],
                            y=[y_data[i] for i in pls_indices],
                            z=[z_data[i] for i in pls_indices],
                            mode='markers',
                            marker=dict(
                                size=5,
                                color='blue',
                                opacity=0.8,
                                line=dict(color='black', width=0.5),
                                symbol='circle'
                            ),
                            text=[final_solution_details[i] for i in pls_indices],
                            hovertemplate='%{text}<extra></extra>',
                            name=f'PLS Solutions ({len(pls_indices)})',
                            legendgroup='pls',
                            showlegend=(combo == combinations[0])  # Only show legend for first plot
                        ),
                        row=combo['row'], col=combo['col']
                    )
                
                # Update scene for this subplot
                scene_num = (combo['row'] - 1) * 2 + combo['col']
                scene_attr = f'scene{scene_num}' if scene_num > 1 else 'scene'
                fig.update_layout(**{
                    scene_attr: dict(
                        xaxis=dict(title=dict(text=combo['labels'][0], font=dict(size=10)), tickfont=dict(size=8)),
                        yaxis=dict(title=dict(text=combo['labels'][1], font=dict(size=10)), tickfont=dict(size=8)),
                        zaxis=dict(title=dict(text=combo['labels'][2], font=dict(size=10)), tickfont=dict(size=8)),
                        camera=dict(eye=dict(x=1.3, y=1.3, z=1.3)),
                        aspectmode='cube'
                    )
                })
            
            # Get ratio for title
            ratio = ratio_results[0].ratio if ratio_results else (0, 0)
            
            # Update overall layout
            fig.update_layout(
                title=dict(
                    text=f'{self.get_algorithm_name()} - {instance_name} (Ratio {ratio[0]}:{ratio[1]}): 4D Pareto Front Analysis<br>'
                         f'<sub>Four 3D projections showing different objective combinations ({len(final_objectives_4d[0])} solutions)</sub>',
                    x=0.5,
                    font=dict(size=16)
                ),
                width=1600,
                height=1000,
                margin=dict(l=50, r=50, b=50, t=120),
                showlegend=True,
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor="black",
                    borderwidth=1
                )
            )
            
            # Add statistics annotation
            annotation_text = (
                f"📊 {instance_name} (Ratio {ratio[0]}:{ratio[1]}) Statistics:<br>"
                f"• Total Solutions: {len(final_objectives_4d[0])}<br>"
                f"• Successful Runs: {len(ratio_results)}<br>"
                f"• Avg Runtime: {statistics.mean([r.total_runtime_seconds for r in ratio_results]):.2f}s<br>"
                f"• Objective Ranges:<br>"
            )
            
            objective_names = ['Cost', 'Cloud Coverage', 'Resolution Sum', 'Incidence Angle']
            for i in range(4):
                if final_objectives_4d[i]:
                    annotation_text += f"  - {objective_names[i]}: {min(final_objectives_4d[i]):.0f} - {max(final_objectives_4d[i]):.0f}<br>"
            
            fig.add_annotation(
                text=annotation_text,
                xref="paper", yref="paper",
                x=0.98, y=0.98,
                showarrow=False,
                align="left",
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10)
            )
            
            # Save visualization with ratio in filename
            finished_at = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"{instance_name}_{self.get_visualization_filename_suffix()}_4d_ratio_{ratio_key}_{finished_at}.html"
            fig.write_html(output_file)
            
            self.console.print(f"✓ 4D visualization saved for ratio {ratio[0]}:{ratio[1]}: {output_file}")
            return True
            
        except Exception as e:
            self.console.print(f"❌ Failed to create 4D visualization for ratio {ratio_key}: {e}")
            return False
    
    
    def create_instance_summary_statistics(self, instance_name: str, instance_results: list[BenchmarkResult]) -> BenchmarkSummaryStatistics:
        """Create summary statistics for a single instance."""
        # Filter successful results for this instance
        successful_results = [r for r in instance_results if not r.error]
        
        if not successful_results:
            # Create minimal summary for failed instance
            from datetime import datetime
            info = BenchmarkInfo(
                timestamp=datetime.now().isoformat(),
                algorithm=self.get_algorithm_name(),
                instances_tested=1,
                total_runs=len(instance_results),
                successful_runs=0,
                failed_runs=len(instance_results)
            )
            return BenchmarkSummaryStatistics(
                benchmark_info=info,
                overall_statistics=OverallStatistics(0.0, 0.0, 0.0),
                performance_by_instance={},
                validation_statistics={}
            )
        
        # Calculate instance performance
        runtimes = [r.total_runtime_seconds for r in successful_results]
        solution_counts = [r.num_final_solutions for r in successful_results]
        
        overall_stats = OverallStatistics(
            avg_runtime=statistics.mean(runtimes),
            avg_solutions=statistics.mean(solution_counts),
            total_runtime=sum(runtimes),
            min_solutions=min(solution_counts),
            max_solutions=max(solution_counts)
        )
        
        # Single instance performance
        performance_by_instance = {
            instance_name: PerformanceByInstance(
                num_runs=len(successful_results),
                avg_runtime=statistics.mean(runtimes),
                std_runtime=statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0,
                min_runtime=min(runtimes),
                max_runtime=max(runtimes),
                avg_solutions=statistics.mean(solution_counts),
                min_solutions=min(solution_counts),
                max_solutions=max(solution_counts)
            )
        }
        
        # Validation statistics
        validation_stats = self._calculate_validation_statistics(successful_results)
        
        # Create benchmark info
        from datetime import datetime
        info = BenchmarkInfo(
            timestamp=datetime.now().isoformat(),
            algorithm=self.get_algorithm_name(),
            instances_tested=1,
            total_runs=len(instance_results),
            successful_runs=len(successful_results),
            failed_runs=len(instance_results) - len(successful_results)
        )
        
        return BenchmarkSummaryStatistics(
            benchmark_info=info,
            overall_statistics=overall_stats,
            performance_by_instance=performance_by_instance,
            validation_statistics=validation_stats
        )
    
    def print_summary(self):
        """Print comprehensive benchmark summary."""
        summary = self.create_summary_statistics()
        
        self.console.print("\n" + "="*80)
        self.console.print(f"[bold blue]📊 {self.get_algorithm_name()} Benchmark Summary")
        self.console.print("="*80)
        
        # Basic info
        info = summary.benchmark_info
        self.console.print(f"[bold]Algorithm:[/bold] {info.algorithm}")
        self.console.print(f"[bold]Solver:[/bold] {self.solver_name}")
        self.console.print(f"[bold]Front Strategy:[/bold] {self.front_strategy}")
        self.console.print(f"[bold]Instances Tested:[/bold] {info.instances_tested}")
        self.console.print(f"[bold]Total Runs:[/bold] {info.total_runs}")
        self.console.print(f"[bold]Successful Runs:[/bold] {info.successful_runs}")
        self.console.print(f"[bold]Failed Runs:[/bold] {info.failed_runs}")
        
        success_rate = (info.successful_runs / info.total_runs * 100) if info.total_runs > 0 else 0
        self.console.print(f"[bold]Success Rate:[/bold] {success_rate:.1f}%")
        
        # Overall performance
        if summary.overall_statistics and info.successful_runs > 0:
            stats = summary.overall_statistics
            self.console.print("\n[bold blue]⚡ Overall Performance")
            self.console.print(f"  Average Runtime: {stats.avg_runtime:.2f}s")
            self.console.print(f"  Total Runtime: {stats.total_runtime:.2f}s")
            self.console.print(f"  Average Solutions: {stats.avg_solutions:.1f}")
            self.console.print(f"  Solution Range: {stats.min_solutions}-{stats.max_solutions}")
        
        # Performance by instance
        if summary.performance_by_instance:
            self.console.print("\n[bold blue]📋 Performance by Instance")
            for instance_name, perf in summary.performance_by_instance.items():
                self.console.print(f"  [bold]{instance_name}:[/bold]")
                self.console.print(f"    Runtime: {perf.avg_runtime:.2f}s ± {perf.std_runtime:.2f}s ({perf.min_runtime:.2f}-{perf.max_runtime:.2f}s)")
                self.console.print(f"    Solutions: {perf.avg_solutions:.1f} ({perf.min_solutions}-{perf.max_solutions})")
                self.console.print(f"    Runs: {perf.num_runs}")
        
        # Validation results
        if self.validate_solutions:
            successful_results: list[BenchmarkResult] = [r for r in self.results if r.error is None]
            if successful_results:
                self.print_validation_summary(successful_results)
        
        self.console.print("="*80)


class GurobiHybridBenchmarkRunner(GurobiBenchmarkRunner):
    """Hybrid benchmark runner that combines Gurobi MILP solver with PLS algorithm."""
    
    def __init__(self, instances_dir: Path, output_dir: Path, iterations: int,
                 tui: bool, size: Optional[int], filter_str: Optional[str],
                 timeout: int, validate_solutions: bool, solver_name: str,
                 front_strategy: str, ratio_step: int, include_perfect: bool = False):
        """Initialize Gurobi Hybrid benchmark runner.
        
        Args:
            instances_dir: Directory containing problem instances
            output_dir: Directory to save results
            iterations: Number of iterations per instance
            tui: Whether to use text UI
            size: Filter instances by size
            filter_str: Filter instances by name pattern
            timeout: Total timeout in seconds
            validate_solutions: Whether to validate solutions
            solver_name: Gurobi solver name
            front_strategy: Front generation strategy
            ratio_step: Step size for ratio configurations
            include_perfect: Whether to include perfect solutions from JSONL
        """
        super().__init__(instances_dir, output_dir, iterations, tui, size, filter_str, timeout, validate_solutions, include_perfect=include_perfect)
        self.solver_name = solver_name
        self.front_strategy = front_strategy
        self.ratio_step = ratio_step
        self.ratio_configs = [(i, 100-i) for i in range(100, -1, -ratio_step)]
        
    def get_algorithm_name(self) -> str:
        """Get algorithm name for display."""
        return f"Gurobi-Hybrid (step={self.ratio_step})"
    
    def get_visualization_filename_suffix(self) -> str:
        """Get the filename suffix for visualizations."""
        return f"gurobi_hybrid_{self.ratio_step}"
    
    def run_single_benchmark(
        self, 
        instance: sims_problem.SimsDiscreteProblem,
        instance_name: str,
        iteration: int,
        ratio: tuple[int, int]
    ) -> BenchmarkResult:
        """Run a single hybrid benchmark combining Gurobi MILP with PLS."""
        import time
        
        # Find the corresponding .dzn file
        dzn_file = None
        for potential_file in self.instance_files:
            if potential_file.stem == instance_name:
                dzn_file = potential_file
                break
        
        if dzn_file is None:
            raise RuntimeError(f"Could not find .dzn file for instance {instance_name}")
        
        if self.use_tui:
            self.console.print(f"  Solving {instance_name} with Gurobi-Hybrid (ratio {ratio[0]}:{ratio[1]})...")
        
        start_time = time.time()
        
        try:
            # Run hybrid solver with the specified ratio
            result_data = self.run_gurobi_hybrid_solver(dzn_file, self.timeout, ratio)
            
            runtime = time.time() - start_time
            solutions = result_data.solutions
            
            # Solutions are already Solution objects, no conversion needed
            benchmark_solutions = solutions
            
            if self.use_tui:
                if benchmark_solutions:
                    self.console.print(f"    Found {len(benchmark_solutions)} solutions in {runtime:.2f}s")
                    if result_data.error:
                        self.console.print(f"    [yellow]With warnings: {result_data.error}[/yellow]")
                else:
                    self.console.print(f"    [red]No solutions found in {runtime:.2f}s[/red]")
                    if result_data.error:
                        self.console.print(f"    [red]Errors: {result_data.error}[/red]")
            
            # Create BenchmarkResult - include partial results even if there are errors
            result = BenchmarkResult()
            result.instance_name = instance_name
            result.iteration = iteration
            result.final_solutions = benchmark_solutions
            result.explored_solutions = []
            
            # Set MILP solutions for proper visualization coloring
            if hasattr(result_data, 'milp_solutions') and result_data.milp_solutions:
                result.milp_solutions = result_data.milp_solutions
                result.num_milp_solutions = len(result_data.milp_solutions)
            else:
                # Fallback: assume solutions from ratio split
                milp_count = int(len(benchmark_solutions) * ratio[0] / 100)
                result.milp_solutions = benchmark_solutions[:milp_count] if milp_count > 0 else []
                result.num_milp_solutions = milp_count
            
            result.num_final_solutions = len(benchmark_solutions)
            result.num_explored_solutions = 0
            result.total_runtime_seconds = runtime
            result.milp_runtime_seconds = runtime * ratio[0] / 100  # Proportional estimate
            result.pls_runtime_seconds = runtime * ratio[1] / 100   # Proportional estimate
            result.ratio = ratio  # Use the actual ratio parameter
            # Only set error if we have no solutions - partial success is okay
            result.error = result_data.error if not benchmark_solutions else ""
            
            # Validate solutions if requested
            if benchmark_solutions and self.validate_solutions and self.validator:
                validation_report = self.validator.validate_benchmark_result(result, instance)
                result.validation_results = validation_report
                
                if not validation_report.overall_valid:
                    self.console.print(f"[yellow]⚠️  Validation issues found for {instance_name} iter={iteration}")
            
            return result
            
        except Exception as e:
            runtime = time.time() - start_time
            error_msg = f"Gurobi hybrid solver error: {str(e)}"
            
            if self.use_tui:
                self.console.print(f"    [red]Error: {error_msg}[/red]")
            
            # Create error result
            result = BenchmarkResult()
            result.instance_name = instance_name
            result.iteration = iteration
            result.final_solutions = []
            result.explored_solutions = []
            result.num_final_solutions = 0
            result.num_explored_solutions = 0
            result.total_runtime_seconds = runtime
            result.milp_runtime_seconds = 0
            result.pls_runtime_seconds = 0
            result.ratio = (50, 50)
            result.error = error_msg
            
            return result
    
    def run_benchmarks(self):
        """Run Gurobi hybrid benchmarks with multiple ratio configurations."""
        if self.use_tui:
            self._run_hybrid_benchmarks_with_tui()
        else:
            self._run_hybrid_benchmarks_simple()
    
    def _run_hybrid_benchmarks_simple(self):
        """Run Gurobi hybrid benchmarks with simple CLI output."""
        total_runs = len(self.instance_files) * self.iterations * len(self.ratio_configs)
        current_run = 0
        
        print(f"Starting Gurobi Hybrid benchmark with {len(self.instance_files)} instances, {self.iterations} iterations each, {len(self.ratio_configs)} ratio configs")
        print(f"Total runs: {total_runs}")
        print(f"Ratio configs: {[f'{r[0]}:{r[1]}' for r in self.ratio_configs]}")
        print("=" * 80)
        
        for instance_idx, instance_file in enumerate(self.instance_files):
            print(f"\nProcessing instance {instance_idx + 1}/{len(self.instance_files)}: {instance_file.stem}")
            
            try:
                instance = self.load_instance(instance_file)
                instance_name = instance_file.stem
                print(f"  ✓ Loaded instance: {instance.num_images} images")
                
                # Run multiple iterations for this instance
                for iteration in range(self.iterations):
                    print(f"    Iteration {iteration + 1}/{self.iterations}")
                    
                    # Run multiple ratio configurations
                    for ratio_idx, ratio in enumerate(self.ratio_configs):
                        current_run += 1
                        progress = (current_run / total_runs) * 100
                        print(f"      Ratio {ratio[0]}:{ratio[1]} (Run {current_run}/{total_runs}, {progress:.1f}%)")
                        
                        result = self.run_single_benchmark(instance, instance_name, iteration, ratio)
                        self.results.append(result)
                        
                        if result.error:
                            print(f"        ❌ Error: {result.error}")
                        else:
                            print(f"        ✓ Success: {result.num_final_solutions} solutions in {result.total_runtime_seconds:.2f}s")
                
                # Create visualization for this instance if requested
                instance_results = [r for r in self.results if r.instance_name == instance_name]
                if instance_results and not all(r.error for r in instance_results):
                    self.create_instance_3d_visualization(instance_name, instance_results)
                
                # Save results for this instance
                self.save_instance_results(instance_name, instance_results)
                
            except Exception as e:
                print(f"  ❌ Failed to load or process {instance_file.stem}: {e}")
                # Create error results for all ratios and iterations
                for iteration in range(self.iterations):
                    for ratio in self.ratio_configs:
                        current_run += 1
                        error_result = BenchmarkResult()
                        error_result.instance_name = instance_file.stem
                        error_result.iteration = iteration
                        error_result.ratio = ratio
                        error_result.error = str(e)
                        self.results.append(error_result)
        
        # Save final summary
        self.save_results()
        
    def _run_hybrid_benchmarks_with_tui(self):
        """Run Gurobi hybrid benchmarks with TUI progress tracking."""
        # Similar to simple but with rich progress bars
        # For now, fall back to simple method
        self._run_hybrid_benchmarks_simple()
    
    def run_gurobi_solver(self, instance_file: Path, timeout: int) -> SolverResult:
        """Run Gurobi MILP solver on an instance.
        
        Args:
            instance_file: Path to instance file
            timeout: Timeout in seconds
            
        Returns:
            SolverResult with Gurobi solver results
        """
        import time
        start_time = time.time()
        
        # Import inside the method to avoid module loading issues
        try:
            from sims_solvers.main import build_model, build_solver
            from sims_solvers.Instances.InstanceSIMS import InstanceSIMS
        except ImportError as e:
            return SolverResult(
                solutions=[],
                num_solutions=0,
                runtime_seconds=time.time() - start_time,
                error=f"Failed to import sims_solvers modules: {e}"
            )
        
        try:
            if self.use_tui:
                self.console.print(f"      Running Gurobi solver for {timeout}s")
            
            # Parse DZN data and create InstanceSIMS
            dzn_data = self.parse_dzn_data(instance_file)
            instance = InstanceSIMS(dzn_data)
            
            # Create configuration for solver
            config = self.create_gurobi_config(instance_file.stem, instance_file)
            config.solver_timeout_sec = timeout
            
            # Build model and solver using the correct approach
            model = build_model(instance, config)
            solver, pareto_front = build_solver(model, instance, config, {})
            
            # Collect solutions from solver
            gurobi_solutions = []
            try:
                for solution in solver.solve():
                    if solution is not None:
                        # Convert to expected format
                        if hasattr(solution, 'objs'):
                            objs = solution.objs
                        elif hasattr(solution, 'objectives'):
                            objs = solution.objectives
                        else:
                            continue
                            
                        solution_values = getattr(solution, 'solution_values', [])
                        
                        # Convert to Solution object
                        try:
                            import sims_problem
                            sol_obj = sims_problem.Solution.create(
                                selected_images=solution_values,
                                cost=int(objs[0]) if len(objs) > 0 else 0,
                                cloudy_area=int(objs[1]) if len(objs) > 1 else 0,
                                min_resolutions_sum=int(objs[2]) if len(objs) > 2 else 0,
                                max_incidence_angle=int(objs[3]) if len(objs) > 3 else 0,
                                timestamp_us=0
                            )
                            gurobi_solutions.append(sol_obj)
                        except Exception as e:
                            if self.use_tui:
                                self.console.print(f"[yellow]Warning: Could not convert Gurobi solution: {e}[/yellow]")
                        
                    # Check timeout
                    if time.time() - start_time >= timeout:
                        break
                        
            except Exception as solver_error:
                # Still return any solutions we collected before the error
                error_msg = f"Gurobi solver error: {solver_error}" if str(solver_error) else "Gurobi solver encountered an unknown error"
                return SolverResult(
                    solutions=gurobi_solutions,
                    num_solutions=len(gurobi_solutions),
                    runtime_seconds=time.time() - start_time,
                    error=error_msg
                )
            
            return SolverResult(
                solutions=gurobi_solutions,
                num_solutions=len(gurobi_solutions),
                runtime_seconds=time.time() - start_time,
                error=""
            )
            
        except Exception as e:
            error_msg = f"Gurobi setup error: {e}" if str(e) else "Gurobi setup encountered an unknown error"
            return SolverResult(
                solutions=[],
                num_solutions=0,
                runtime_seconds=time.time() - start_time,
                error=error_msg
            )

    def run_gurobi_hybrid_solver(self, instance_file: Path, timeout: int, ratio: tuple[int, int]) -> SolverResult:
        """Run Gurobi hybrid solver on an instance with a specific ratio configuration.
        
        Args:
            instance_file: Path to instance file
            timeout: Total timeout in seconds
            ratio: Tuple of (gurobi_percentage, pls_percentage)
            
        Returns:
            SolverResult with hybrid solver results for the specific ratio
        """
        import time
        start_time = time.time()
        
        all_solutions = set()
        best_solutions = []  # Contains Solution objects
        error_messages = []
        milp_solutions = []  # Track MILP solutions separately
        
        gurobi_ratio, pls_ratio = ratio
        
        if self.use_tui:
            self.console.print(f"  Running ratio config: Gurobi {gurobi_ratio}% + PLS {pls_ratio}%")
        
        # Track MILP solutions for this config to pass to PLS
        milp_population = []
        
        try:
            # Phase 1: Gurobi MILP
            if gurobi_ratio > 0:
                gurobi_time = timeout * (gurobi_ratio / 100.0)

                if self.use_tui:
                    self.console.print(f"    Running Gurobi solver for {gurobi_time:.1f}s")
                
                gurobi_result = self.run_gurobi_solver(instance_file, int(gurobi_time))
                
                if gurobi_result.solutions:
                    milp_solutions.extend(gurobi_result.solutions)  # Store MILP solutions
                    milp_population.extend(gurobi_result.solutions)
                    for sol in gurobi_result.solutions:
                        sol_tuple = self.solution_to_tuple(sol)
                        if sol_tuple not in all_solutions:
                            all_solutions.add(sol_tuple)
                            best_solutions.append(sol)
                    
                    if self.use_tui:
                        self.console.print(f"    Gurobi found {len(gurobi_result.solutions)} solutions")
                else:
                    if self.use_tui:
                        self.console.print(f"    Gurobi found no solutions")
                
                if gurobi_result.error:
                    error_messages.append(f"Gurobi: {gurobi_result.error}")
            
            # Phase 2: PLS
            if pls_ratio > 0:
                pls_time = timeout * (pls_ratio / 100.0)
                
                if self.use_tui:
                    self.console.print(f"    Running PLS solver for {pls_time:.1f}s")
                
                pls_result = self.run_pls_solver(instance_file, int(pls_time), milp_population)
                
                if pls_result.solutions:
                    for sol in pls_result.solutions:
                        sol_tuple = self.solution_to_tuple(sol)
                        if sol_tuple not in all_solutions:
                            all_solutions.add(sol_tuple)
                            best_solutions.append(sol)
                    
                    if self.use_tui:
                        self.console.print(f"    PLS found {len(pls_result.solutions)} new solutions")
                else:
                    if self.use_tui:
                        self.console.print(f"    PLS found no new solutions")
                
                if pls_result.error:
                    error_messages.append(f"PLS: {pls_result.error}")
                    
        except Exception as e:
            error_messages.append(f"Hybrid solver error: {str(e)}")
        
        runtime = time.time() - start_time
        
        # Create result with MILP solutions tracked separately
        result = SolverResult(
            solutions=best_solutions,
            runtime_seconds=runtime,
            error="; ".join(error_messages) if error_messages else None
        )
        
        # Add MILP solutions as a custom attribute
        result.milp_solutions = milp_solutions
        
        return result
    
    def run_pls_solver(self, instance_file: Path, timeout: int, initial_population=None) -> SolverResult:
        """Run PLS solver using existing infrastructure.
        
        Args:
            instance_file: Path to instance file
            timeout: Timeout in seconds
            initial_population: List of Solution objects to use as initial population
            
        Returns:
            SolverResult with PLS solver results
        """
        import time
        start_time = time.time()
        
        try:
            import sims_problem
            
            if self.use_tui:
                self.console.print(f"      Running PLS solver for {timeout}s")
            
            # Load problem instance using correct API
            problem = sims_problem.SimsDiscreteProblem.from_dzn(str(instance_file))
            
            # Run PLS solver with correct parameter format
            from datetime import timedelta
            solving_result = sims_problem.solve_with_pls(
                problem,
                objectives=["min_cost", "cloud_coverage", "min_resolution", "max_incidence_angle"],
                plots=False,
                plot_output_path=None,
                timeout=timedelta(seconds=timeout),
                max_iterations=10000,
                is_deterministic=False,
                initial_population_size=100,
                initial_population=initial_population,
                neighborhood_size_min=1,
                neighborhood_size_max=6,
                trace=False
            )
            
            return SolverResult(
                solutions=solving_result.final_solutions,
                num_solutions=len(solving_result.final_solutions),
                runtime_seconds=time.time() - start_time,
                error=""
            )
            
        except Exception as e:
            return SolverResult(
                solutions=[],
                num_solutions=0,
                runtime_seconds=time.time() - start_time,
                error=f"PLS solver error: {e}"
            )
    
    def _create_visualization(self, successful_results: list[BenchmarkResult], output_path: Path):
        """Create visualization for Gurobi hybrid results."""
        # Check if we have 4D problems
        if successful_results:
            # Check number of objectives from first solution
            first_result = successful_results[0]
            if first_result.solutions and len(first_result.solutions[0].objectives) == 4:
                # Get instance name from first result
                instance_name = first_result.instance_name if hasattr(first_result, 'instance_name') else "unknown_instance"
                self._create_4d_visualization_multiple_plots(instance_name, successful_results)
            else:
                # Fall back to regular visualization
                super()._create_visualization(successful_results, output_path)
    
    def print_summary(self):
        """Print comprehensive Gurobi hybrid benchmark summary."""
        summary = self.create_summary_statistics()
        
        self.console.print("\n" + "="*80)
        self.console.print(f"[bold blue]📊 {self.get_algorithm_name()} Benchmark Summary")
        self.console.print("="*80)
        
        # Basic info
        info = summary.benchmark_info
        self.console.print(f"[bold]Algorithm:[/bold] {info.algorithm}")
        self.console.print(f"[bold]Solver:[/bold] {self.solver_name}")
        self.console.print(f"[bold]Front Strategy:[/bold] {self.front_strategy}")
        self.console.print(f"[bold]Ratio Step:[/bold] {self.ratio_step}%")
        self.console.print(f"[bold]Ratio Configs:[/bold] {len(self.ratio_configs)}")
        self.console.print(f"[bold]Instances Tested:[/bold] {info.instances_tested}")
        self.console.print(f"[bold]Total Runs:[/bold] {info.total_runs}")
        self.console.print(f"[bold]Successful Runs:[/bold] {info.successful_runs}")
        self.console.print(f"[bold]Failed Runs:[/bold] {info.failed_runs}")
        
        success_rate = (info.successful_runs / info.total_runs * 100) if info.total_runs > 0 else 0
        self.console.print(f"[bold]Success Rate:[/bold] {success_rate:.1f}%")
        
        # Show ratio configurations
        self.console.print("\n[bold blue]🔄 Ratio Configurations")
        for i, (gurobi_ratio, pls_ratio) in enumerate(self.ratio_configs):
            self.console.print(f"  Config {i+1}: Gurobi {gurobi_ratio}% + PLS {pls_ratio}%")
        
        # Overall performance
        if summary.overall_statistics and info.successful_runs > 0:
            stats = summary.overall_statistics
            self.console.print("\n[bold blue]⚡ Overall Performance")
            self.console.print(f"  Average Runtime: {stats.avg_runtime:.2f}s")
            self.console.print(f"  Total Runtime: {stats.total_runtime:.2f}s")
            self.console.print(f"  Average Solutions: {stats.avg_solutions:.1f}")
            self.console.print(f"  Solution Range: {stats.min_solutions}-{stats.max_solutions}")
        
        # Performance by instance
        if summary.performance_by_instance:
            self.console.print("\n[bold blue]📋 Performance by Instance")
            for instance_name, perf in summary.performance_by_instance.items():
                self.console.print(f"  [bold]{instance_name}:[/bold]")
                self.console.print(f"    Runtime: {perf.avg_runtime:.2f}s ± {perf.std_runtime:.2f}s ({perf.min_runtime:.2f}-{perf.max_runtime:.2f}s)")
                self.console.print(f"    Solutions: {perf.avg_solutions:.1f} ({perf.min_solutions}-{perf.max_solutions})")
                self.console.print(f"    Runs: {perf.num_runs}")
        
        # Validation results
        if self.validate_solutions:
            successful_results: list[BenchmarkResult] = [r for r in self.results if r.error is None]
            if successful_results:
                self.print_validation_summary(successful_results)
        
        self.console.print("="*80)


def run_milp_benchmark(args):
    """Run MILP algorithm benchmark."""
    if not args.instances_dir.exists():
        print(f"Error: Instances directory {args.instances_dir} does not exist")
        return 1
    
    # Determine validation setting
    validate_solutions = True  # Default
    if hasattr(args, 'no_validate_solutions') and args.no_validate_solutions:
        validate_solutions = False
    elif hasattr(args, 'validate_solutions') and args.validate_solutions:
        validate_solutions = True
    
    try:
        runner = MilpBenchmarkRunner(
            args.instances_dir, 
            args.output_dir, 
            args.iterations,
            args.grid_points,
            args.tui,
            args.size,
            getattr(args, 'filter', None),
            args.timeout,
            validate_solutions,
            args.bypass_coefficient,
            args.early_exit,
            args.flag_array,
            args.solver_name,
            getattr(args, 'include_perfect', False)
        )
        runner.run_benchmarks()
        runner.save_results()
        runner.print_summary()
        return 0
        
    except KeyboardInterrupt:
        print("\nMILP Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running MILP benchmark: {e}")
        return 1


def run_gurobi_benchmark(args):
    """Run Gurobi algorithm benchmark."""
    if not args.instances_dir.exists():
        print(f"Error: Instances directory {args.instances_dir} does not exist")
        return 1
    
    # Determine validation setting
    validate_solutions = True  # Default
    if hasattr(args, 'no_validate_solutions') and args.no_validate_solutions:
        validate_solutions = False
    elif hasattr(args, 'validate_solutions') and args.validate_solutions:
        validate_solutions = True
    
    try:
        runner = GurobiBenchmarkRunner(
            args.instances_dir, 
            args.output_dir, 
            args.iterations,
            args.tui,
            getattr(args, 'size', None),
            getattr(args, 'filter', None),
            args.timeout,
            validate_solutions,
            args.solver_name,
            args.front_strategy,
            getattr(args, 'include_perfect', False)
        )
        runner.run_benchmarks()
        runner.save_results()
        runner.print_summary()
        return 0
        
    except KeyboardInterrupt:
        print("\nGurobi Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running Gurobi benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_hybrid_benchmark(args):
    """Run hybrid algorithm benchmark."""
    if not args.instances_dir.exists():
        print(f"Error: Instances directory {args.instances_dir} does not exist")
        return 1
    
    # Determine validation setting
    validate_solutions = True  # Default
    if hasattr(args, 'no_validate_solutions') and args.no_validate_solutions:
        validate_solutions = False
    elif hasattr(args, 'validate_solutions') and args.validate_solutions:
        validate_solutions = True
    
    try:
        runner = HybridBenchmarkRunner(
            args.instances_dir, 
            args.output_dir, 
            args.iterations,
            args.ratio_step,
            args.tui,
            args.size,
            getattr(args, 'filter', None),
            args.timeout,
            validate_solutions,
            getattr(args, 'include_perfect', False)
        )
        runner.run_benchmarks()
        runner.save_results()
        runner.print_summary()
        return 0
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running hybrid benchmark: {e}")
        return 1


def run_pls_benchmark(args):
    """Run PLS algorithm benchmark."""
    if not args.instances_dir.exists():
        print(f"Error: Instances directory {args.instances_dir} does not exist")
        return 1
    
    # Determine validation setting
    validate_solutions = True  # Default
    if hasattr(args, 'no_validate_solutions') and args.no_validate_solutions:
        validate_solutions = False
    elif hasattr(args, 'validate_solutions') and args.validate_solutions:
        validate_solutions = True
    
    try:
        runner = PlsBenchmarkRunner(
            args.instances_dir, 
            args.output_dir, 
            args.iterations,
            args.max_iterations,
            args.tui,
            args.size,
            getattr(args, 'filter', None),
            args.timeout,
            validate_solutions,
            getattr(args, 'include_perfect', False)
        )
        runner.run_benchmarks()
        runner.save_results()
        runner.print_summary()
        return 0
        
    except KeyboardInterrupt:
        print("\nPLS Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running PLS benchmark: {e}")
        return 1


def run_gurobi_hybrid_benchmark(args):
    """Run Gurobi hybrid algorithm benchmark."""
    if not args.instances_dir.exists():
        print(f"Error: Instances directory {args.instances_dir} does not exist")
        return 1
    
    # Determine validation setting
    validate_solutions = True  # Default
    if hasattr(args, 'no_validate_solutions') and args.no_validate_solutions:
        validate_solutions = False
    elif hasattr(args, 'validate_solutions') and args.validate_solutions:
        validate_solutions = True
    
    try:
        runner = GurobiHybridBenchmarkRunner(
            args.instances_dir, 
            args.output_dir, 
            args.iterations,
            args.tui,
            getattr(args, 'size', None),
            getattr(args, 'filter', None),
            args.timeout,
            validate_solutions,
            args.solver_name,
            args.front_strategy,
            args.ratio_step,
            getattr(args, 'include_perfect', False)
        )
        runner.run_benchmarks()
        runner.save_results()
        runner.print_summary()
        return 0
        
    except KeyboardInterrupt:
        print("\nGurobi Hybrid Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running Gurobi hybrid benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point for the benchmark script."""
    print("DEBUG: Main function called")
    
    parser = argparse.ArgumentParser(
        description="Benchmark SIMS solvers",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    print("DEBUG: Parser created")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available benchmark commands',
        metavar='COMMAND'
    )
    
    print("DEBUG: Subparsers created")
    
    # Common arguments for all subcommands
    def add_common_args(subparser):
        subparser.add_argument(
            "--instances-dir",
            type=Path,
            default=Path("tests/data"),
            help="Directory containing .dzn instance files"
        )
        subparser.add_argument(
            "--output-dir", 
            type=Path,
            default=Path("benchmark_results"),
            help="Directory to save benchmark results"
        )
        subparser.add_argument(
            "--iterations",
            type=int,
            default=1,
            help="Number of iterations per configuration"
        )
        subparser.add_argument(
            "--tui",
            action="store_true",
            help="Use rich TUI interface (default: simple CLI)"
        )
        subparser.add_argument(
            "--size",
            type=int,
            help="Limit instances to those with <= N images (optional filter)"
        )
        subparser.add_argument(
            "--filter",
            type=str,
            help="Filter instances by name using regex pattern (e.g., 'lagos_nigeria_30' or 'tokyo.*30')"
        )
        subparser.add_argument(
            "--timeout",
            type=float,
            default=300.0,
            help="Timeout in seconds for each solver run (default: 300.0)"
        )
        subparser.add_argument(
            "--validate-solutions",
            action="store_true",
            help="Enable solution validation and Pareto front checking (adds computational overhead)"
        )
        subparser.add_argument(
            "--no-validate-solutions",
            action="store_true",
            help="Disable solution validation (default: validation enabled)"
        )
        subparser.add_argument(
            "--include-perfect",
            action="store_true",
            help="Include perfect solutions from JSONL files in visualizations"
        )
    
    # Hybrid algorithm subcommand
    hybrid_parser = subparsers.add_parser(
        'hybrid',
        help='Benchmark hybrid SIMS solver with different MILP/PLS ratios',
        description='Run benchmarks on hybrid solver testing different MILP/PLS ratio configurations'
    )
    add_common_args(hybrid_parser)
    hybrid_parser.add_argument(
        "--ratio-step",
        type=int,
        default=10,
        help="Step size for ratio configurations (default: 10, gives ratios 100:0, 90:10, ..., 0:100)"
    )
    hybrid_parser.set_defaults(func=run_hybrid_benchmark)
    
    # PLS algorithm subcommand
    pls_parser = subparsers.add_parser(
        'pls',
        help='Benchmark pure Pareto Local Search algorithm',
        description='Run benchmarks on pure PLS solver with 3 objectives (cost, cloud_coverage, max_incidence_angle)'
    )
    add_common_args(pls_parser)
    pls_parser.add_argument(
        "--max-iterations",
        type=int,
        default=50000,
        help="Maximum number of PLS iterations per run (default: 50000)"
    )
    pls_parser.set_defaults(func=run_pls_benchmark)
    
    # MILP algorithm subcommand
    milp_parser = subparsers.add_parser(
        'milp',
        help='Benchmark pure MILP solver using epsilon-constraint method',
        description='Run benchmarks on pure MILP solver with 3 objectives (cost, cloud_coverage, max_incidence_angle)'
    )
    add_common_args(milp_parser)
    milp_parser.add_argument(
        "--grid-points",
        type=int,
        default=50,
        help="Number of grid points for epsilon-constraint method (default: 50)"
    )
    milp_parser.add_argument(
        "--bypass-coefficient",
        action="store_true",
        default=True,
        help="Use bypass coefficient optimization (default: enabled)"
    )
    milp_parser.add_argument(
        "--no-bypass-coefficient",
        action="store_true",
        help="Disable bypass coefficient optimization"
    )
    milp_parser.add_argument(
        "--early-exit",
        action="store_true",
        default=True,
        help="Enable early exit optimization (default: enabled)"
    )
    milp_parser.add_argument(
        "--no-early-exit",
        action="store_true",
        help="Disable early exit optimization"
    )
    milp_parser.add_argument(
        "--flag-array",
        action="store_true",
        default=True,
        help="Use flag array optimization (default: enabled)"
    )
    milp_parser.add_argument(
        "--no-flag-array",
        action="store_true",
        help="Disable flag array optimization"
    )
    milp_parser.add_argument(
        "--solver-name",
        type=str,
        default="cbc",
        choices=["cbc", "gurobi", "cplex", "scip"],
        help="MILP solver to use (default: cbc)"
    )
    
    # Handle conflicting flags for MILP parser
    def process_milp_args(args):
        if args.no_bypass_coefficient:
            args.bypass_coefficient = False
        if args.no_early_exit:
            args.early_exit = False
        if args.no_flag_array:
            args.flag_array = False
        return args
    
    milp_parser.set_defaults(func=lambda args: run_milp_benchmark(process_milp_args(args)))
    
    # Gurobi algorithm subcommand
    gurobi_parser = subparsers.add_parser(
        'gurobi',
        help='Benchmark pure Gurobi solver using sims-solvers with 4 objectives',
        description='Run benchmarks on pure Gurobi solver with 4 objectives (cost, cloud_coverage, resolution, incidence_angle) using sims-solvers'
    )
    add_common_args(gurobi_parser)
    gurobi_parser.add_argument(
        "--solver-name",
        type=str,
        default="gurobi",
        choices=["gurobi"],
        help="Solver to use (currently only gurobi supported)"
    )
    gurobi_parser.add_argument(
        "--front-strategy",
        type=str,
        default="saugmecon",
        choices=["saugmecon", "gpba-a"],
        help="Front generation strategy to use (default: saugmecon)"
    )
    gurobi_parser.set_defaults(func=run_gurobi_benchmark)
    
    # Gurobi hybrid algorithm subcommand
    gurobi_hybrid_parser = subparsers.add_parser(
        'gurobi-hybrid',
        help='Benchmark Gurobi+PLS hybrid solver using sims-solvers with 4 objectives',
        description='Run benchmarks on Gurobi+PLS hybrid solver with 4 objectives (cost, cloud_coverage, resolution, incidence_angle) using sims-solvers and PLS in combination'
    )
    add_common_args(gurobi_hybrid_parser)
    gurobi_hybrid_parser.add_argument(
        "--solver-name",
        type=str,
        default="gurobi",
        choices=["gurobi"],
        help="Gurobi solver to use (currently only gurobi supported)"
    )
    gurobi_hybrid_parser.add_argument(
        "--front-strategy",
        type=str,
        default="saugmecon",
        choices=["saugmecon", "gpba-a"],
        help="Front generation strategy for Gurobi phase (default: saugmecon)"
    )
    gurobi_hybrid_parser.add_argument(
        "--ratio-step",
        type=int,
        default=50,
        help="Step size for ratio configurations (default: 50, meaning 100/0, 50/50, 0/100)"
    )
    gurobi_hybrid_parser.set_defaults(func=run_gurobi_hybrid_benchmark)
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command specified, show help
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1
    
    # Run the specified command
    return args.func(args)


if __name__ == "__main__":
    print("Starting benchmark script...")
    try:
        exit_code = main()
        print(f"Script completed with exit code: {exit_code}")
        exit(exit_code)
    except Exception as e:
        print(f"Script failed with exception: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
