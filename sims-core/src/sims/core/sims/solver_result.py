from __future__ import annotations

import ast
import csv
import dataclasses
import logging
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from os import PathLike
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from pymoo.indicators.hv import HV

from .problem import ProblemInstance, SimsDiscreteProblem
from .solver_config import FrontStrategy, SolverType, TwoPhaseSolverConfig

csv.field_size_limit(sys.maxsize // 10)

log = logging.getLogger(Path(__file__).stem)


@dataclass
class Solution:
    selected_images: frozenset[int]
    cost: int
    cloudy_area: int
    timestamp_s: timedelta
    max_incidence_angle: int | None = None
    min_resolutions_sum: int | None = None

    def to_json(self) -> dict:
        return {
            "selected_images": list(self.selected_images),
            "cost": self.cost,
            "cloudy_area": self.cloudy_area,
            "max_incidence_angle": self.max_incidence_angle,
            "min_resolutions_sum": self.min_resolutions_sum,
            "timestamp_s": self.timestamp_s.total_seconds(),
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Solution):
            return False
        return (
            self.selected_images == other.selected_images
            and self.cost == other.cost
            and self.cloudy_area == other.cloudy_area
            and self.max_incidence_angle == other.max_incidence_angle
            and self.min_resolutions_sum == other.min_resolutions_sum
        )

    def __hash__(self) -> int:
        return self.selected_images.__hash__()

    def validate(self, problem: SimsDiscreteProblem) -> bool:
        if not self.selected_images:
            coverage = frozenset()
        else:
            # Use frozenset.union correctly with an iterable of sets
            coverage_sets = [frozenset(problem.images[i]) for i in self.selected_images]
            coverage = frozenset().union(*coverage_sets)
        is_valid = len(coverage) == problem.universe
        if not is_valid:
            uncovered_elements = sorted(set(range(problem.universe)) - coverage)
            print(
                f"Error: the selected images do not cover the whole universe, uncovered elements: {uncovered_elements}"
            )

        # return is_valid and self.validate_objectives(problem)
        return is_valid

    def compute_objectives(self, problem: SimsDiscreteProblem, objectives: list[str]) -> dict[str, int]:
        """
        Compute only the specified objectives for efficiency.
        
        Args:
            problem: The problem instance
            objectives: List of objective names to compute (e.g., ["min_cost", "cloud_coverage"])
            
        Returns:
            Dictionary mapping objective names to their computed values
        """
        result = {}
        
        # Map of objective names to computation functions
        objective_computations = {
            "min_cost": lambda: sum(problem.costs[i] for i in self.selected_images),
            "cloud_coverage": lambda: self._compute_cloudy_area(problem),
            "min_max_incidence_angle": lambda: max(problem.incidence_angle[i] for i in self.selected_images),
            "min_resolution": lambda: self._compute_min_resolutions_sum(problem),
        }
        
        # Compute only the requested objectives
        for obj_name in objectives:
            if obj_name in objective_computations:
                result[obj_name] = objective_computations[obj_name]()
            else:
                log.warning(f"Unknown objective: {obj_name}")
                
        return result
    
    def _compute_cloudy_area(self, problem: SimsDiscreteProblem) -> int:
        """Helper method to compute cloudy area."""
        clear_parts = frozenset.union(
            *(frozenset(problem.images[i]) - frozenset(problem.clouds[i]) for i in self.selected_images)
        )
        return sum(problem.areas[u] for u in range(problem.universe) if u not in clear_parts)
    
    def _compute_min_resolutions_sum(self, problem: SimsDiscreteProblem) -> int:
        """Helper method to compute minimum resolutions sum."""
        return sum(
            map(
                lambda u: min(
                    problem.resolution[i] for i in self.selected_images if u in frozenset(problem.images[i])
                ),
                range(problem.universe),
            )
        )

    def validate_objectives(self, problem: SimsDiscreteProblem, objectives: list[str] | None = None) -> bool:
        """
        Validate the objectives of the solution
        
        Args:
            problem: The problem instance to validate against
            objectives: Optional list of objective names to validate. If None, validates all non-None objectives.
                       Valid objective names: "min_cost", "cloud_coverage", "min_max_incidence_angle", "min_resolution"
        """
        # If objectives not specified, validate all objectives that are set (not None)
        if objectives is None:
            all_objectives = ["min_cost", "cloud_coverage", "min_max_incidence_angle", "min_resolution"]
        else:
            all_objectives = objectives
            
        computed_objectives = self.compute_objectives(problem, all_objectives)

        # Validate only the objectives that were requested
        validations = {}
        
        if "min_cost" in all_objectives:
            validations["cost"] = self.cost == computed_objectives.get("min_cost", self.cost)
        
        if "cloud_coverage" in all_objectives:
            validations["cloudy_area"] = self.cloudy_area == computed_objectives.get("cloud_coverage", self.cloudy_area)
        
        if "min_max_incidence_angle" in all_objectives:
            validations["max_incidence_angle"] = (
                self.max_incidence_angle == computed_objectives.get("min_max_incidence_angle", self.max_incidence_angle)
                if self.max_incidence_angle is not None and self.max_incidence_angle != -1
                else True
            )
        
        if "min_resolution" in all_objectives:
            validations["min_resolutions_sum"] = (
                self.min_resolutions_sum == computed_objectives.get("min_resolution", self.min_resolutions_sum)
                if self.min_resolutions_sum is not None and self.min_resolutions_sum != -1
                else True
            )

        all_valid = all(validations.values())
        
        # Print differences if invalid
        if not all_valid:
            print(f"Objective validation failed for solution with images: {sorted(self.selected_images)}")
            if "cost" in validations and not validations["cost"]:
                print(f"  cost: expected {computed_objectives.get('min_cost')}, got {self.cost}")
            if "cloudy_area" in validations and not validations["cloudy_area"]:
                print(f"  cloudy_area: expected {computed_objectives.get('cloud_coverage')}, got {self.cloudy_area}")
            if "max_incidence_angle" in validations and not validations["max_incidence_angle"]:
                print(f"  max_incidence_angle: expected {computed_objectives.get('min_max_incidence_angle')}, got {self.max_incidence_angle}")
            if "min_resolutions_sum" in validations and not validations["min_resolutions_sum"]:
                print(f"  min_resolutions_sum: expected {computed_objectives.get('min_resolution')}, got {self.min_resolutions_sum}")
        
        return all_valid

    def fix_objectives(self, problem: SimsDiscreteProblem, objectives: list[str]):
        """
        Fix the objectives of the solution to handle negative values and missing objectives.
        
        This method computes or fixes the specified objectives to ensure they are valid
        for compatibility with Rust PLS solver that expects unsigned integers.
        
        Args:
            problem: The problem instance
            objectives: List of objective names to fix/compute (e.g., ["min_cost", "cloud_coverage"])
        """
        log.debug(f"fix_objectives called with objectives: {objectives}")
        log.debug(f"Initial values - cost: {self.cost}, cloudy_area: {self.cloudy_area}, "
                 f"max_incidence_angle: {self.max_incidence_angle}, min_resolutions_sum: {self.min_resolutions_sum}")
        
        # Compute objectives that need fixing
        computed_objectives = self.compute_objectives(problem, objectives)
        log.debug(f"Computed objectives: {computed_objectives}")
        
        # Fix each objective explicitly with proper typing
        for obj_name in objectives:
            if obj_name == "min_cost":
                computed_value = computed_objectives.get("min_cost")
                if computed_value is not None:
                    log.debug(f"Processing min_cost: current={self.cost}, computed={computed_value}")
                    if self.cost < 0:
                        log.warning(f"Found negative cost {self.cost}, setting to computed value {computed_value}")
                        self.cost = max(0, computed_value)
                        log.info(f"Fixed cost: {self.cost}")
                    elif self.cost != computed_value:
                        log.warning(f"Cost mismatch: current={self.cost}, computed={computed_value}, using computed")
                        self.cost = max(0, computed_value)
                        log.info(f"Updated cost: {self.cost}")
            
            elif obj_name == "cloud_coverage":
                computed_value = computed_objectives.get("cloud_coverage")
                if computed_value is not None:
                    log.debug(f"Processing cloud_coverage: current={self.cloudy_area}, computed={computed_value}")
                    if self.cloudy_area < 0:
                        log.warning(f"Found negative cloudy_area {self.cloudy_area}, setting to computed value {computed_value}")
                        self.cloudy_area = max(0, computed_value)
                        log.info(f"Fixed cloudy_area: {self.cloudy_area}")
                    elif self.cloudy_area != computed_value:
                        log.warning(f"Cloudy area mismatch: current={self.cloudy_area}, computed={computed_value}, using computed")
                        self.cloudy_area = max(0, computed_value)
                        log.info(f"Updated cloudy_area: {self.cloudy_area}")
            
            elif obj_name == "min_max_incidence_angle":
                computed_value = computed_objectives.get("min_max_incidence_angle")
                if computed_value is not None:
                    log.debug(f"Processing min_max_incidence_angle: current={self.max_incidence_angle}, computed={computed_value}")
                    # Handle -1 as "not set" sentinel value, don't treat as negative error
                    if self.max_incidence_angle is not None and self.max_incidence_angle < 0 and self.max_incidence_angle != -1:
                        log.warning(f"Found negative max_incidence_angle {self.max_incidence_angle}, setting to computed value {computed_value}")
                        self.max_incidence_angle = max(0, computed_value)
                        log.info(f"Fixed max_incidence_angle: {self.max_incidence_angle}")
                    elif self.max_incidence_angle != computed_value and self.max_incidence_angle != -1:
                        log.warning(f"Max incidence angle mismatch: current={self.max_incidence_angle}, computed={computed_value}, using computed")
                        self.max_incidence_angle = max(0, computed_value)
                        log.info(f"Updated max_incidence_angle: {self.max_incidence_angle}")
            
            elif obj_name == "min_resolution":
                computed_value = computed_objectives.get("min_resolution")
                if computed_value is not None:
                    log.debug(f"Processing min_resolution: current={self.min_resolutions_sum}, computed={computed_value}")
                    # Handle -1 as "not set" sentinel value, don't treat as negative error
                    if self.min_resolutions_sum is not None and self.min_resolutions_sum < 0 and self.min_resolutions_sum != -1:
                        log.warning(f"Found negative min_resolutions_sum {self.min_resolutions_sum}, setting to computed value {computed_value}")
                        self.min_resolutions_sum = max(0, computed_value)
                        log.info(f"Fixed min_resolutions_sum: {self.min_resolutions_sum}")
                    elif self.min_resolutions_sum != computed_value and self.min_resolutions_sum != -1:
                        log.warning(f"Min resolutions sum mismatch: current={self.min_resolutions_sum}, computed={computed_value}, using computed")
                        self.min_resolutions_sum = max(0, computed_value)
                        log.info(f"Updated min_resolutions_sum: {self.min_resolutions_sum}")
            
            else:
                log.warning(f"Unknown objective: {obj_name}")
        
        log.debug(f"Final values - cost: {self.cost}, cloudy_area: {self.cloudy_area}, "
                 f"max_incidence_angle: {self.max_incidence_angle}, min_resolutions_sum: {self.min_resolutions_sum}")


def compute_hypervolume(
    pareto_front: list[Solution], max_objectives: tuple[float, float], scaled=True
) -> float:
    if len(pareto_front) == 0:
        return 0
    ref_point = None
    if scaled:
        scaled_pareto_front = np.array(
            [
                [
                    solution.cost / max_objectives[0],
                    solution.cloudy_area / max_objectives[1],
                ]
                for solution in pareto_front
            ]
        )
        ref_point = (1, 1)
        return HV(ref_point=ref_point)(scaled_pareto_front)  # type: ignore # pymoo is not typed
    else:
        pareto_front_ndarr = np.array(
            [[solution.cost, solution.cloudy_area] for solution in pareto_front]
        )
        ref_point = (max_objectives[0] + 1, max_objectives[1] + 1)
        return HV(ref_point=ref_point)(pareto_front_ndarr)  # type: ignore # pymoo is not typed


@dataclass
class ParetoFrontSnapshot:
    solutions: list[Solution]
    timestamp_s: timedelta


@dataclass
class SolverResult:
    pareto_front: list[Solution]
    timeout_sec: int
    execution_time_sec: float
    hypervolume: float
    solver_type: SolverType
    problem_instance: ProblemInstance
    front_strategy: Optional[FrontStrategy] = None
    pareto_front_snapshots: list[ParetoFrontSnapshot] = field(default_factory=list)
    trace_data: Optional[bytes] = None  # trace data for debugging/analysis (raw bytes)
    profiling_trace_data: Optional[bytes] = None  # Chrome profiling trace data in JSON format

    def to_dict(self, full=False) -> dict:
        result_dict = {
            "pareto_front": [solution.to_json() for solution in self.pareto_front],
            "timeout_sec": self.timeout_sec,
            "execution_time_sec": self.execution_time_sec,
            "hypervolume": self.hypervolume,
        }

        if full:
            result_dict["problem_instance"] = self.problem_instance.to_dict()
            result_dict["solver_type"] = str(self.solver_type)
            result_dict["front_strategy"] = str(self.front_strategy)

        return result_dict

    @staticmethod
    def from_summary_csv(
        input_path: PathLike,
        problem_instance: ProblemInstance,
        objectives: list[str],
        row_index: int = -1,
        no_headers=False,
        trace_data: Optional[bytes] = None,
    ) -> SolverResult:
        data = Path(input_path).read_text().splitlines()
        if not no_headers:
            summary_data = list(csv.DictReader(data, delimiter=";"))
        else:
            fieldnames = [
                "instance",
                "problem",
                "solver_name",
                "front_strategy",
                "solver_search_strategy",
                "fzn_optimisation_level",
                "threads",
                "cores",
                "solver_timeout_sec",
                "minizinc_model",
                "exhaustive",
                "hypervolume",
                "datetime",
                "number_of_solutions",
                "total_nodes",
                "time_solver_sec",
                "minizinc_time_fzn_sec",
                "hypervolume_current_solutions",
                "solutions_time_list",
                "pareto_solutions_time_list",
                "pareto_front",
                "solutions_pareto_front",
                "incomplete_timeout_solution_added_to_front",
            ]
            summary_data = list(csv.DictReader(data, fieldnames=fieldnames, delimiter=";"))

        # Print the Exhaustive flag
        if summary_data:
            exhaustive_flag = summary_data[row_index].get("exhaustive", "unknown")
            print(f"Exhaustive flag: {exhaustive_flag}", flush=True)

        # Parse the given row of the summary data and return the result object
        return SolverResult.from_summary_dict(summary_data[row_index], problem_instance, objectives=objectives, trace_data=trace_data)

    @staticmethod
    def from_summary_dict(
        summary_dict: dict,
        problem_instance: ProblemInstance,
        objectives: list[str],
        pareto_front_snapshots: list[ParetoFrontSnapshot] | None = None,
        trace_data: Optional[bytes] = None,
    ) -> SolverResult:
        expected_name = problem_instance.name
        # Check if the instance name is the expected one
        if summary_dict["instance"] != expected_name:
            raise ValueError(
                f"Expected instance name {expected_name}, got {summary_dict['instance']}"
            )

        # Preprocess the data
        for key in [
            "pareto_front",
            "solutions_pareto_front",
            "solutions_time_list",
            "pareto_solutions_time_list",
        ]:
            if summary_dict[key] == "":
                summary_dict[key] = []
                continue
            # Convert set to list to preserve order
            value_str = summary_dict[key].replace("{", "[").replace("}", "]")
            summary_dict[key] = ast.literal_eval(value_str)

        # Parse solutions pareto front
        pareto_front = []
        for objective_values, selected_images, timestamp_s in zip(
            summary_dict["pareto_front"],
            summary_dict["solutions_pareto_front"],
            summary_dict["pareto_solutions_time_list"],
        ):
            if not selected_images:
                continue
                
            # Create a mapping of objective names to values
            obj_map = {obj_name: objective_values[i] for i, obj_name in enumerate(objectives) if i < len(objective_values)}
            
            # Ensure all objective values are converted to integers (CSV parsing may return floats)
            solution = Solution(
                selected_images=frozenset(selected_images),
                cost=int(obj_map.get("min_cost", -1)),
                cloudy_area=int(obj_map.get("cloud_coverage", -1)),
                min_resolutions_sum=int(obj_map.get("min_resolution", -1)) if obj_map.get("min_resolution", -1) != -1 else None,
                max_incidence_angle=int(obj_map.get("min_max_incidence_angle", -1)) if obj_map.get("min_max_incidence_angle", -1) != -1 else None,
                timestamp_s=timedelta(seconds=float(timestamp_s)),
            )
            # Only fix objectives if we have a problem instance
            if problem_instance.problem is not None:
                solution.fix_objectives(problem_instance.problem, objectives)
            pareto_front.append(solution)

        # Derive the pareto front snapshots from current pareto front
        # if pareto_front_snapshots is None:
        if False:
            pareto_front_snapshots = []
            sorted_solutions = sorted(pareto_front, key=lambda s: s.timestamp_s)
            current_time = timedelta(seconds=10)
            snapshot_solutions = []

            for solution in sorted_solutions:
                while solution.timestamp_s >= current_time:
                    pareto_front_snapshots.append(
                        ParetoFrontSnapshot(
                            solutions=list(snapshot_solutions), timestamp_s=current_time
                        )
                    )
                    current_time += timedelta(seconds=10)
                snapshot_solutions.append(solution)

            # Add the last snapshot if there are remaining solutions
            if snapshot_solutions:
                pareto_front_snapshots.append(
                    ParetoFrontSnapshot(
                        solutions=list(snapshot_solutions), timestamp_s=current_time
                    )
                )

        # Return the experiment object
        return SolverResult(
            problem_instance=problem_instance,
            pareto_front=pareto_front,
            timeout_sec=int(summary_dict["solver_timeout_sec"]),
            execution_time_sec=float(summary_dict["time_solver_sec"]),
            hypervolume=float(summary_dict["hypervolume"]),
            solver_type=SolverType.from_str(summary_dict["solver_name"]),
            front_strategy=FrontStrategy.from_str(summary_dict["front_strategy"]),
            pareto_front_snapshots=pareto_front_snapshots or [],
            trace_data=trace_data,
        )

    def validate(self) -> bool:
        is_valid = True
        if self.problem_instance is not None:
            for i, solution in enumerate(self.pareto_front):
                if not solution.validate(self.problem_instance.problem):
                    logging.error(f"Solution {i} is invalid")
                    is_valid = False
        else:
            logging.error("Cannot validate solutions without a problem instance")
            return False
        
        return is_valid

    def filter_invalid(self):
        if self.problem_instance is not None:
            self.pareto_front = [
                solution
                for solution in self.pareto_front
                if solution.validate(self.problem_instance.problem)
            ]
        else:
            logging.warning("Cannot filter invalid solutions without a problem instance")

    def scatter_plot(
        self,
        ax: plt.Axes,  # type: ignore # matplotlib.pyplot.Axes is not exported
        max_values: tuple[int, int],
        unique_solutions: list[Solution] | None = None,
    ):
        """Add scatter data to the plot"""
        if unique_solutions is not None:
            x_data = [solution.cost for solution in unique_solutions]
            y_data = [solution.cloudy_area for solution in unique_solutions]
        else:
            x_data = [solution.cost for solution in self.pareto_front]
            y_data = [solution.cloudy_area for solution in self.pareto_front]
        scaled_x_data = [x / max_values[0] for x in x_data]
        scaled_y_data = [y / max_values[1] for y in y_data]
        label = (
            f"{repr(self.solver_type)} {repr(self.front_strategy)}"
            if self.solver_type == SolverType.OR_TOOLS
            else f"{repr(self.solver_type)}"
        )
        color = "red" if self.solver_type == SolverType.PLS else "blue"
        ax.scatter(scaled_x_data, scaled_y_data, label=label, marker="^", color=color)

    def render_plot(self, plot_path: PathLike):
        fig, axes = plt.subplots()
        axes.grid(True)
        axes.set_axisbelow(True)
        axes.set_xlabel("Cost")
        axes.set_ylabel("Cloudy Area")
        axes.set_title(f"Pareto Front for {self.problem_instance.name}")

        max_objectives = self.max_objectives()
        self.scatter_plot(axes, max_objectives)

        axes.legend(loc="upper right")

        # Adjust the space on the left side of the plot
        fig.subplots_adjust(left=0.15)

        fig.savefig(plot_path, dpi=600)
        plt.close(fig)

    def compute_hypervolume(self, max_objectives: tuple[float, float], scaled=True):
        self.hypervolume = compute_hypervolume(self.pareto_front, max_objectives, scaled)

    def max_objectives(self):
        if len(self.pareto_front) == 0:
            return (0, 0)
        x_max = max(solution.cost for solution in self.pareto_front)
        y_max = max(solution.cloudy_area for solution in self.pareto_front)
        return (x_max, y_max)


@dataclass
class TwoPhaseSolverResult:
    problem_instance: ProblemInstance
    solver_config: TwoPhaseSolverConfig
    total_time_sec: float
    exact_solver_result: Optional[SolverResult] | None = None
    pls_result: Optional[SolverResult] | None = None
    discarded_solutions: list[Solution] | None = None
    added_solutions: list[Solution] | None = None
    _trace_data: bytes | None = field(default=None, init=False)
    _profiling_trace_data: bytes | None = field(default=None, init=False)

    def __post_init__(self):
        """Compute merged trace data and profiling trace data after initialization."""
        self._trace_data = self._compute_merged_trace_data()
        self._profiling_trace_data = self._compute_merged_profiling_trace_data()

    @property
    def trace_data(self) -> bytes | None:
        """Get the merged trace data from both phases."""
        return self._trace_data

    @property
    def profiling_trace_data(self) -> bytes | None:
        """Get the merged profiling trace data from both phases."""
        return self._profiling_trace_data

    def _compute_merged_trace_data(self) -> bytes | None:
        """Compute merged trace data from exact and PLS solver results."""
        exact_has_trace = (self.exact_solver_result is not None and 
                          hasattr(self.exact_solver_result, 'trace_data') and 
                          self.exact_solver_result.trace_data is not None)
        pls_has_trace = (self.pls_result is not None and 
                        hasattr(self.pls_result, 'trace_data') and 
                        self.pls_result.trace_data is not None)
        
        if exact_has_trace and pls_has_trace and self.exact_solver_result and self.pls_result:
            # Import merge_traces function from sims_problem
            import sims_problem
            
            # Calculate objective bounds and reference point from combined pareto fronts
            objective_bounds, reference_point = self._calculate_objective_bounds_and_ref_point()
            
            # Merge traces with proper timestamp offsetting
            combined_algorithm = f"two-phase-{self.solver_config.ratio[0]}-{self.solver_config.ratio[1]}"
            return sims_problem.merge_traces(
                self.exact_solver_result.trace_data,
                self.pls_result.trace_data, 
                combined_algorithm,
                objective_bounds,
                reference_point
            )
        elif exact_has_trace and self.exact_solver_result:
            return self.exact_solver_result.trace_data
        elif pls_has_trace and self.pls_result:
            return self.pls_result.trace_data
        else:
            return None

    def _compute_merged_profiling_trace_data(self) -> bytes | None:
        """Compute merged profiling trace data from exact and PLS solver results."""
        exact_has_profiling_trace = (self.exact_solver_result is not None and
                                    self.exact_solver_result.profiling_trace_data is not None)
        pls_has_profiling_trace = (self.pls_result is not None and
                                  self.pls_result.profiling_trace_data is not None)
        
        if exact_has_profiling_trace and pls_has_profiling_trace:
            # Both phases have profiling traces - merge them
            assert self.exact_solver_result is not None and self.exact_solver_result.profiling_trace_data is not None
            assert self.pls_result is not None and self.pls_result.profiling_trace_data is not None
            
            import json
            
            try:
                exact_trace = json.loads(self.exact_solver_result.profiling_trace_data.decode('utf-8'))
                pls_trace = json.loads(self.pls_result.profiling_trace_data.decode('utf-8'))
                
                # Merge trace events - both should be arrays
                if isinstance(exact_trace, list) and isinstance(pls_trace, list):
                    combined_trace = exact_trace + pls_trace
                    return json.dumps(combined_trace).encode('utf-8')
                else:
                    log.warning("Profiling traces are not in expected array format, returning exact trace only")
                    return self.exact_solver_result.profiling_trace_data
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                log.warning(f"Failed to merge profiling traces: {e}, returning exact trace only")
                return self.exact_solver_result.profiling_trace_data
        elif exact_has_profiling_trace:
            assert self.exact_solver_result is not None
            return self.exact_solver_result.profiling_trace_data
        elif pls_has_profiling_trace:
            assert self.pls_result is not None
            return self.pls_result.profiling_trace_data
        else:
            return None

    def _calculate_objective_bounds_and_ref_point(self):
        """Calculate objective bounds and reference point from combined pareto fronts."""
        # Collect all solutions from both phases
        all_solutions: list[Solution] = []
        if self.exact_solver_result and self.exact_solver_result.pareto_front:
            all_solutions.extend(self.exact_solver_result.pareto_front)
        if self.pls_result and self.pls_result.pareto_front:
            all_solutions.extend(self.pls_result.pareto_front)
        
        if not all_solutions:
            # Default bounds if no solutions
            return [[0, 0], [0, 0]], [1, 1]
        
        # Determine number of objectives based on solution attributes
        objectives = []
        if all_solutions[0].cost is not None and all_solutions[0].cost >= 0:
            objectives.append('cost')
        if all_solutions[0].cloudy_area is not None and all_solutions[0].cloudy_area >= 0:
            objectives.append('cloudy_area')
        if all_solutions[0].max_incidence_angle is not None and all_solutions[0].max_incidence_angle >= 0:
            objectives.append('max_incidence_angle')
        if all_solutions[0].min_resolutions_sum is not None and all_solutions[0].min_resolutions_sum >= 0:
            objectives.append('min_resolutions_sum')
        
        # Calculate bounds for each objective
        bounds = []
        ref_point = []
        
        for obj_name in objectives:
            match obj_name:
                case 'cost':
                    values = [sol.cost for sol in all_solutions if sol.cost is not None]
                case 'cloudy_area':
                    values = [sol.cloudy_area for sol in all_solutions if sol.cloudy_area is not None]
                case 'max_incidence_angle':
                    values = [sol.max_incidence_angle for sol in all_solutions if sol.max_incidence_angle is not None]
                case 'min_resolutions_sum':
                    values = [sol.min_resolutions_sum for sol in all_solutions if sol.min_resolutions_sum is not None]
                case _:
                    raise ValueError(f"Unknown objective name: {obj_name}")
            # Raise error if any objective has sentinel value (-1)
            if any(v == -1 for v in values):
                raise ValueError(f"Found sentinel value (-1) in objective '{obj_name}' values: {values}")
            min_val = min(values)
            max_val = max(values)
            bounds.append([min_val, max_val])
            ref_point.append(max_val + 1)
        
        return bounds, ref_point

    def to_dict(self, full=False):
        result_dict = {
            "ratio": list(self.solver_config.ratio),
            "total_time_sec": self.total_time_sec,
            "solutions": self.solutions,
            "exact_solver_result": self.exact_solver_result.to_dict()
            if self.exact_solver_result is not None
            else None,
            "pls_result": self.pls_result.to_dict() if self.pls_result is not None else None,
        }

        if full:
            result_dict["problem_instance"] = self.problem_instance.to_dict()
            result_dict["solver_config"] = dataclasses.asdict(self.solver_config)
            if self.discarded_solutions is not None:
                result_dict["discarded_solutions"] = [
                    solution.to_json() for solution in self.discarded_solutions
                ]
            if self.added_solutions is not None:
                result_dict["added_solutions"] = [
                    solution.to_json() for solution in self.added_solutions
                ]
        return result_dict

    def heuristic_unique_solutions(self):
        if self.added_solutions is not None:
            return self.added_solutions
        elif self.pls_result is not None:
            return self.pls_result.pareto_front
        else:
            return None

    @staticmethod
    def from_results_pair(
        exact_solver_result: SolverResult | None,
        pls_result: SolverResult | None,
        solver_config: TwoPhaseSolverConfig,
        filter_invalid: bool = True,
    ) -> TwoPhaseSolverResult:
        if pls_result is None and exact_solver_result is None:
            raise ValueError("At least one of exact_solver_result or pls_result must be provided")
        
        # Initialize problem_instance from whichever result is available
        problem_instance = (
            exact_solver_result.problem_instance if exact_solver_result is not None
            else pls_result.problem_instance if pls_result is not None
            else None
        )
        
        if problem_instance is None:
            raise ValueError("No problem instance found in either result")
        
        total_time_sec = 0
        if exact_solver_result is not None:
            total_time_sec += exact_solver_result.execution_time_sec

        if pls_result is not None:
            total_time_sec += pls_result.execution_time_sec

        if exact_solver_result is not None and pls_result is not None:
            discarded_solutions, added_solutions = (
                TwoPhaseSolverResult._detect_discarded_and_added_solutions(
                    exact_solver_result.pareto_front, pls_result.pareto_front
                )
            )
        elif exact_solver_result is None and pls_result is not None:
            # PLS-only case (0:100 ratio): all PLS solutions are "added" solutions
            discarded_solutions, added_solutions = None, pls_result.pareto_front
        else:
            discarded_solutions, added_solutions = None, None


        if filter_invalid:
            if exact_solver_result is not None:
                exact_solver_result.filter_invalid()
            if pls_result is not None:
                pls_result.filter_invalid()

        two_phase_solver_result = TwoPhaseSolverResult(
            problem_instance=problem_instance,
            solver_config=solver_config,
            total_time_sec=total_time_sec,
            exact_solver_result=exact_solver_result,
            pls_result=pls_result,
            discarded_solutions=discarded_solutions,
            added_solutions=added_solutions,
        )

        return two_phase_solver_result

    @staticmethod
    def _parse_pls_pareto_front_snapshots(
        pareto_snapshots_path: Path, problem: SimsDiscreteProblem, objectives: list[str]
    ):
        pareto_fronts = []

        with pareto_snapshots_path.open("r") as file:
            lines = file.readlines()

            i = 0
            while i < len(lines):
                # Parse the snapshot header
                _, elapsed_time, num_solutions = map(int, lines[i].strip().split())
                i += 1

                # Parse the solutions
                solutions = []
                for _ in range(num_solutions):
                    selected_images = list(map(int, lines[i].strip().split()))
                    solution = Solution(
                        selected_images=frozenset(selected_images),
                        cost=-1,
                        cloudy_area=-1,
                        timestamp_s=timedelta(seconds=elapsed_time),
                    )
                    solution.fix_objectives(problem, objectives)
                    solutions.append(solution)
                    i += 1

                # Create a ParetoFrontSnapshot and add it to the list
                snapshot = ParetoFrontSnapshot(
                    solutions=solutions, timestamp_s=timedelta(seconds=elapsed_time)
                )
                pareto_fronts.append(snapshot)

    @staticmethod
    def from_summary_csv(
        input_path: Path,
        two_phase_solver_config: TwoPhaseSolverConfig,
        problem_instance: ProblemInstance,
        objectives: list[str],
    ) -> TwoPhaseSolverResult:
        data = Path(input_path).read_text().splitlines()

        # Dummy check to see if the data has headers
        no_headers = not data[0].startswith("instance")

        if no_headers:
            fieldnames = [
                "instance",
                "problem",
                "solver_name",
                "front_strategy",
                "solver_search_strategy",
                "fzn_optimisation_level",
                "threads",
                "cores",
                "solver_timeout_sec",
                "minizinc_model",
                "exhaustive",
                "hypervolume",
                "datetime",
                "number_of_solutions",
                "total_nodes",
                "time_solver_sec",
                "minizinc_time_fzn_sec",
                "hypervolume_current_solutions",
                "solutions_time_list",
                "pareto_solutions_time_list",
                "pareto_front",
                "solutions_pareto_front",
                "incomplete_timeout_solution_added_to_front",
            ]
            summary_data = list(csv.DictReader(data, fieldnames=fieldnames, delimiter=";"))
        else:
            summary_data = list(csv.DictReader(data, delimiter=";"))

        ratio = two_phase_solver_config.ratio
        # if ratio in [(100, 0), (0, 100)]:
        #     if len(summary_data) != 1:
        #         raise ValueError(
        #             f"Invalid number of rows in the summary CSV file {input_path} for ratio {ratio}: {len(summary_data)}"
        #         )
        # else:
        #     if len(summary_data) != 2:
        #         raise ValueError(
        #             f"Invalid number of rows in the summary CSV file {input_path} for ratio {ratio}: {len(summary_data)}"
        #         )

        # For PLS results, parse the pareto front snapshots
        # if ratio != (100, 0):
        if False:
            pls_pareto_front_snapshots_path = input_path.parent / "pareto_front_snapshots.txt"
            pls_pareto_front_snapshots = TwoPhaseSolverResult._parse_pls_pareto_front_snapshots(
                pls_pareto_front_snapshots_path, problem=problem_instance.problem, objectives=objectives
            )
        else:
            pls_pareto_front_snapshots = None

        if ratio == (100, 0):
            exact_solver_result = SolverResult.from_summary_dict(summary_data[0], problem_instance, objectives)
            pls_solver_result = None
        elif ratio == (0, 100):
            exact_solver_result = None
            pls_solver_result = SolverResult.from_summary_dict(
                summary_data[0], problem_instance, objectives, pls_pareto_front_snapshots
            )
        else:
            exact_solver_result = SolverResult.from_summary_dict(summary_data[0], problem_instance, objectives)
            pls_solver_result = SolverResult.from_summary_dict(
                summary_data[-1], problem_instance, objectives, pls_pareto_front_snapshots
            )

        return TwoPhaseSolverResult.from_results_pair(
            exact_solver_result, pls_solver_result, two_phase_solver_config, filter_invalid=False
        )

    def title(self) -> str:
        ratio = self.solver_config.ratio
        return f"Pareto Front {ratio[0]}% {repr(self.solver_config.exact_solver_type)} - {ratio[1]}% PLS"

    def scatter_plot(self, ax: plt.Axes, max_objectives: tuple[int, int]):  # type: ignore # matplotlib.pyplot.Axes is not exported
        # Configure ax styling
        ax.set_aspect("auto", adjustable="datalim")
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_xlabel("Cost")
        ax.set_ylabel("Cloudy Area")
        ax.set_xlim((-0.05, 1.1))
        ax.set_ylim((-0.05, 1.1))
        ax.set_title(self.title())

        if self.exact_solver_result is not None:
            self.exact_solver_result.scatter_plot(ax, max_objectives)
        elif self.pls_result is not None:
            self.pls_result.scatter_plot(ax, max_objectives)

        if (
            self.discarded_solutions is not None
            and self.added_solutions is not None
            and self.pls_result is not None
        ):
            scaled_x_data = [
                solution.cost / max_objectives[0] for solution in self.discarded_solutions
            ]
            scaled_y_data = [
                solution.cloudy_area / max_objectives[1] for solution in self.discarded_solutions
            ]
            ax.scatter(
                scaled_x_data,
                scaled_y_data,
                label="Discarded Solutions",
                marker="x",
                color="black",
            )

            self.pls_result.scatter_plot(ax, max_objectives, unique_solutions=self.added_solutions)

        ax.legend()

    @staticmethod
    def _detect_discarded_and_added_solutions(
        exact_solver_solutions: list[Solution], pls_solutions: list[Solution]
    ) -> tuple[list[Solution], list[Solution]]:
        discarded_solutions = [
            solution for solution in exact_solver_solutions if solution not in pls_solutions
        ]
        added_solutions = [
            solution for solution in pls_solutions if solution not in exact_solver_solutions
        ]

        return discarded_solutions, added_solutions

    def max_objectives(self):
        if self.pls_result is not None and self.exact_solver_result is not None:
            first_max = self.exact_solver_result.max_objectives()
            second_max = self.pls_result.max_objectives()
            x_max = max(first_max[0], second_max[0])
            y_max = max(first_max[1], second_max[1])
            return (x_max, y_max)
        elif self.pls_result is not None:
            return self.pls_result.max_objectives()
        elif self.exact_solver_result is not None:
            return self.exact_solver_result.max_objectives()
        else:
            # Should never happen
            raise ValueError("Both exact and PLS results are None")

    @property
    def solutions(self) -> list[dict]:
        """
        Get a combined list of solutions from both phases with correct indexing and phase information.
        This property ensures that solutions from the PLS phase that were also in the exact phase are not duplicated.
        """
        all_solutions = []
        
        # Add exact solutions
        if self.exact_solver_result:
            for i, sol in enumerate(self.exact_solver_result.pareto_front):
                sol_dict = sol.to_json()
                sol_dict["phase"] = "exact"
                sol_dict["index"] = i
                all_solutions.append(sol_dict)
        
        # Add unique solutions from PLS phase
        if self.added_solutions:
            start_index = len(self.exact_solver_result.pareto_front) if self.exact_solver_result else 0
            for i, sol in enumerate(self.added_solutions):
                sol_dict = sol.to_json()
                sol_dict["phase"] = "heuristic"
                sol_dict["index"] = start_index + i
                all_solutions.append(sol_dict)
                
        return all_solutions
