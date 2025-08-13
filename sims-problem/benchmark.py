#!/usr/bin/env python3
"""
Multi-Algorithm Benchmark Script

This script runs comprehensive benchmarks on SIMS solvers with support for multiple algorithms:
- Hybrid SIMS solver (MILP + PLS with configurable ratios)
- Pure Pareto Local Search (PLS)

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
        
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "instance_name": self.instance_name,
            "iteration": self.iteration,
            "ratio": self.ratio,
            "total_runtime_seconds": self.total_runtime_seconds,
            "milp_runtime_seconds": self.milp_runtime_seconds,
            "pls_runtime_seconds": self.pls_runtime_seconds,
            "milp_solutions": [sol.to_json() for sol in self.milp_solutions],
            "final_solutions": [sol.to_json() for sol in self.final_solutions],
            "explored_solutions": [sol.to_json() for sol in self.explored_solutions],
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
    
    def __init__(self, instances_dir: Path, output_dir: Path, iterations: int = 1, use_tui: bool = False, size_limit: Optional[int] = None, name_filter: Optional[str] = None, timeout: float = 300.0, validate_solutions: bool = True):
        self.instances_dir = instances_dir
        self.output_dir = output_dir
        self.iterations = iterations
        self.use_tui = use_tui
        self.size_limit = size_limit
        self.name_filter = name_filter
        self.timeout = timeout
        self.validate_solutions = validate_solutions
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
        Save results for a single instance (detailed, summary, and visualization).
        Default implementation - subclasses can override for custom behavior.
        
        Args:
            instance_name: Name of the instance
            instance_results: List of benchmark results for this instance only
        """
        if not instance_results:
            return
            
        finished_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results for this instance
        detailed_file = self.output_dir / f"{instance_name}_detailed_{finished_at}.json"
        with open(detailed_file, 'w') as f:
            json.dump(
                [result.to_dict() for result in instance_results],
                f,
                indent=2,
                default=str
            )
        
        # Create and save summary statistics for this instance
        summary = self.create_instance_summary_statistics(instance_name, instance_results)
        summary_file = self.output_dir / f"{instance_name}_summary_{finished_at}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)
        
        # Create visualization for this instance
        self.create_instance_3d_visualization(instance_name, instance_results)
        
        self.console.print(f"\n[green]Instance {instance_name} results saved:")
        self.console.print(f"  Detailed: {detailed_file}")
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
            (costs, cloudy_areas, incidence_angles, timestamps, solution_details)
        """
        # Collect all explored solutions and final solutions
        all_explored_costs = []
        all_explored_cloudy_areas = []
        all_explored_incidence_angles = []
        all_explored_timestamps = []
        explored_solution_details = []
        
        all_final_costs = []
        all_final_cloudy_areas = []
        all_final_incidence_angles = []
        all_final_timestamps = []
        final_solution_details = []
        
        # Collect JSONL solutions data
        all_jsonl_costs = []
        all_jsonl_cloudy_areas = []
        all_jsonl_incidence_angles = []
        jsonl_solution_details = []
        
        for result in successful_results:
            # Process explored solutions
            if result.explored_solutions:
                exp_costs, exp_cloudy_areas, exp_incidence_angles = self.extract_solution_objectives(result.explored_solutions)
                exp_timestamps = self.extract_solution_timestamps(result.explored_solutions)
                
                all_explored_costs.extend(exp_costs)
                all_explored_cloudy_areas.extend(exp_cloudy_areas)
                all_explored_incidence_angles.extend(exp_incidence_angles)
                all_explored_timestamps.extend(exp_timestamps)
                
                # Create hover details for explored solutions
                for (cost, cloudy, incidence) in zip(exp_costs, exp_cloudy_areas, exp_incidence_angles):
                    detail = self._create_explored_solution_detail(result, cost, cloudy, incidence)
                    explored_solution_details.append(detail)
            
            # Process final solutions
            final_costs, final_cloudy_areas, final_incidence_angles = self.extract_solution_objectives(result.final_solutions)
            final_timestamps = self.extract_solution_timestamps(result.final_solutions)
            
            all_final_costs.extend(final_costs)
            all_final_cloudy_areas.extend(final_cloudy_areas)
            all_final_incidence_angles.extend(final_incidence_angles)
            all_final_timestamps.extend(final_timestamps)
            
            # Create hover details for final solutions
            for i, (cost, cloudy, incidence) in enumerate(zip(final_costs, final_cloudy_areas, final_incidence_angles)):
                detail = self._create_final_solution_detail(result, cost, cloudy, incidence)
                final_solution_details.append(detail)
        
        # Load and process JSONL solutions if instance name is provided
        if instance_name:
            import json  # Import here to avoid issues with error handling
            
            # Debug: Print directory structure
            self.console.print(f"🔍 DEBUG: instances_dir = {self.instances_dir}")
            self.console.print(f"🔍 DEBUG: instances_dir.parent = {self.instances_dir.parent}")
            
            # Fix path construction - instances_dir is already tests/data, so go up to project root
            jsonl_file = self.instances_dir.parent.parent / "tests" / "data" / "manuels_results" / f"{instance_name}.jsonl"
            self.console.print(f"🔍 DEBUG: Looking for JSONL at: {jsonl_file}")
            self.console.print(f"🔍 DEBUG: JSONL file exists: {jsonl_file.exists()}")
            
            if jsonl_file.exists():
                self.console.print(f"🔍 DEBUG: Found JSONL file, starting to process...")
                
                # Load the corresponding instance to compute objectives
                instance_file = self.instances_dir / f"{instance_name}.dzn"
                if not instance_file.exists():
                    raise FileNotFoundError(f"Instance file not found: {instance_file}")
                
                import sims_problem
                try:
                    problem = sims_problem.SimsDiscreteProblem.from_dzn(str(instance_file))
                    self.console.print(f"🔍 DEBUG: Loaded problem instance successfully")
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
                # Only log missing JSONL file, don't raise error (it's optional)
                self.console.print(f"ℹ️ No JSONL reference file found for {instance_name} at {jsonl_file}")
                
                # Debug: List what files are actually in the manuels_results directory
                manuels_dir = self.instances_dir / "manuels_results"
                if manuels_dir.exists():
                    available_files = list(manuels_dir.glob("*.jsonl"))
                    self.console.print(f"🔍 DEBUG: Available JSONL files in {manuels_dir}:")
                    for f in available_files:
                        self.console.print(f"   - {f.name}")
                else:
                    self.console.print(f"🔍 DEBUG: Manuels results directory does not exist: {manuels_dir}")
        
        self.console.print(f"🔍 DEBUG: Final JSONL counts - costs: {len(all_jsonl_costs)}, areas: {len(all_jsonl_cloudy_areas)}, angles: {len(all_jsonl_incidence_angles)}")
        
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
                        explored_norm_timestamps, explored_solution_details)
        final_data = (all_final_costs, all_final_cloudy_areas, all_final_incidence_angles, 
                     final_norm_timestamps, final_solution_details)
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
        all_explored_costs, all_explored_cloudy_areas, all_explored_incidence_angles, explored_norm_timestamps, explored_solution_details = explored_data
        all_final_costs, all_final_cloudy_areas, all_final_incidence_angles, final_norm_timestamps, final_solution_details = final_data
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
                    color=explored_norm_timestamps,
                    colorscale='Rainbow_r',  # Reversed colorscale: early=yellow, late=purple
                    opacity=0.4,
                    line=dict(color='black', width=0.5),
                    symbol='x'
                ),
                text=explored_solution_details,
                hovertemplate='%{text}<extra></extra>',
                name=f'🔍 Explored Solutions ({len(all_explored_costs)})',
                legendgroup='explored'
            ))
        
        # Final Pareto solutions (Circle markers)
        if all_final_costs:
            traces.append(go.Scatter3d(
                x=all_final_costs,
                y=all_final_cloudy_areas,
                z=all_final_incidence_angles,
                mode='markers',
                marker=dict(
                    size=4,
                    color=final_norm_timestamps,
                    colorscale='Rainbow_r',  # Reversed colorscale: early=yellow, late=purple
                    colorbar=dict(title="Final Time", x=0.92),
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
                xaxis=dict(title='Cost (minimize)', tickfont=dict(size=12)),
                yaxis=dict(title='Cloudy Area Coverage (minimize)', tickfont=dict(size=12)),
                zaxis=dict(title='Max Incidence Angle (minimize)', tickfont=dict(size=12)),
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
    
    def __init__(self, instances_dir: Path, output_dir: Path, iterations: int = 1, ratio_step: int = 10, use_tui: bool = False, size_limit: Optional[int] = None, name_filter: Optional[str] = None, timeout: float = 300.0, validate_solutions: bool = True):
        super().__init__(instances_dir, output_dir, iterations, use_tui, size_limit, name_filter, timeout, validate_solutions)
        
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
    ) -> tuple[list[sims_problem.Solution], float, float]:
        """
        Run the hybrid solver and return solutions with timing information.
        
        Returns:
            tuple of (solutions, milp_time, pls_time)
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
        *args
    ) -> BenchmarkResult:
        """Run a single benchmark and collect detailed results."""
        ratio: tuple[int, int] = args[0]  # Extract ratio from args
        result = BenchmarkResult()
        result.instance_name = instance_name
        result.iteration = iteration
        result.ratio = ratio
        
        try:
            solutions, milp_time, pls_time = self.run_hybrid_solver(instance, ratio)
            result.milp_runtime_seconds = milp_time
            result.pls_runtime_seconds = pls_time
            result.total_runtime_seconds = milp_time + pls_time
            
            # Convert solutions to Solution objects
            result.final_solutions = solutions
            result.num_final_solutions = len(solutions)
            
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
        """Save benchmark results to JSON files and create 3D visualization."""
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
            json.dump(summary.to_dict(), f, indent=2, default=str)
        
        self.console.print("\n[green]Results saved:")
        self.console.print(f"  Detailed: {detailed_file}")
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
    
    def __init__(self, instances_dir: Path, output_dir: Path, iterations: int = 10, max_iterations: int = 50000, use_tui: bool = False, size_limit: Optional[int] = None, name_filter: Optional[str] = None, timeout: float = 300.0, validate_solutions: bool = True):
        super().__init__(instances_dir, output_dir, iterations, use_tui, size_limit, name_filter, timeout, validate_solutions)
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
            objectives=["min_cost", "cloud_coverage", "max_incidence_angle"],
            plots=False,
            plot_output_path=None,
            timeout=timeout,
            max_iterations=self.max_iterations,
            is_deterministic=False,
            initial_population_size=100,
            neighborhood_size_min=1,
            neighborhood_size_max=6
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
        """Save PLS benchmark results to JSON files and create 3D visualization."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        detailed_file = self.output_dir / f"pls_benchmark_detailed_{timestamp}.json"
        with open(detailed_file, 'w') as f:
            json.dump(
                [result.to_dict() for result in self.results],
                f,
                indent=2,
                default=str
            )
        
        summary = self.create_summary_statistics()
        summary_file = self.output_dir / f"pls_benchmark_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)
        
        self.console.print("\n[green]PLS Results saved:")
        self.console.print(f"  Detailed: {detailed_file}")
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
            validate_solutions
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
            validate_solutions
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
