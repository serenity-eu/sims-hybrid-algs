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
        coverage = frozenset.union(*(frozenset(problem.images[i]) for i in self.selected_images))
        is_valid = len(coverage) == problem.universe
        if not is_valid:
            uncovered_elements = sorted(set(range(problem.universe)) - coverage)
            print(
                f"Error: the selected images do not cover the whole universe, uncovered elements: {uncovered_elements}"
            )

        return is_valid and self.validate_objectives(problem)

    def compute_objectives(self, problem: SimsDiscreteProblem) -> tuple[int, int, int, int]:
        total_cost = sum(problem.costs[i] for i in self.selected_images)

        clear_parts = frozenset.union(
            *(frozenset(problem.images[i]) - frozenset(problem.clouds[i]) for i in self.selected_images)
        )
        cloudy_area = sum(problem.areas[u] for u in range(problem.universe) if u not in clear_parts)

        max_incidence_angle = max(problem.incidence_angle[i] for i in self.selected_images)

        min_resolutions_sum = sum(
            map(
                lambda u: min(
                    problem.resolution[i] for i in self.selected_images if u in frozenset(problem.images[i])
                ),
                range(problem.universe),
            )
        )

        return total_cost, cloudy_area, max_incidence_angle, min_resolutions_sum

    def validate_objectives(self, problem: SimsDiscreteProblem) -> bool:
        """
        Validate the objectives of the solution
        """

        (
            total_cost,
            cloudy_area,
            max_incidence_angle,
            min_resolutions_sum,
        ) = self.compute_objectives(problem)

        is_cost_valid = self.cost == total_cost
        is_cloudy_area_valid = self.cloudy_area == cloudy_area
        is_max_incidence_angle_valid = (
            self.max_incidence_angle == max_incidence_angle
            if self.max_incidence_angle != -1
            else True
        )
        is_min_resolutions_sum_valid = (
            self.min_resolutions_sum == min_resolutions_sum
            if self.min_resolutions_sum != -1
            else True
        )

        return (
            is_cost_valid
            and is_cloudy_area_valid
            and is_max_incidence_angle_valid
            and is_min_resolutions_sum_valid
        )

    def fix_objectives(self, problem: SimsDiscreteProblem):
        """
        Fix the objectives of the solution - only fix objectives that need fixing
        """
        # Check which objectives need fixing
        need_cost = self.cost < 0
        need_cloudy_area = self.cloudy_area < 0
        need_max_incidence = self.max_incidence_angle == -1
        need_min_resolutions = self.min_resolutions_sum == -1
        
        # Only compute objectives if any need fixing
        if need_cost or need_cloudy_area or need_max_incidence or need_min_resolutions:
            cost, cloudy_area, max_incidence_angle, min_resolutions_sum = self.compute_objectives(problem)
            
            if need_cost:
                log.warning(f"Fixing cost from {self.cost} to {cost}")
                self.cost = cost
            if need_cloudy_area:
                log.warning(f"Fixing cloudy area from {self.cloudy_area} to {cloudy_area}")
                self.cloudy_area = cloudy_area
            if need_max_incidence:
                log.warning(f"Fixing max incidence angle from {self.max_incidence_angle} to {max_incidence_angle}")
                self.max_incidence_angle = max_incidence_angle
            if need_min_resolutions:
                log.warning(f"Fixing min resolutions sum from {self.min_resolutions_sum} to {min_resolutions_sum}")
                self.min_resolutions_sum = min_resolutions_sum


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

        # Parse the given row of the summary data and return the result object
        return SolverResult.from_summary_dict(summary_data[row_index], problem_instance, objectives=objectives)

    @staticmethod
    def from_summary_dict(
        summary_dict: dict,
        problem_instance: ProblemInstance,
        objectives: list[str],
        pareto_front_snapshots: list[ParetoFrontSnapshot] | None = None,
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
            
            solution = Solution(
                selected_images=frozenset(selected_images),
                cost=obj_map.get("min_cost", -1),
                cloudy_area=obj_map.get("cloud_coverage", -1),
                min_resolutions_sum=obj_map.get("min_resolution", -1),
                max_incidence_angle=obj_map.get("min_max_incidence_angle", -1),
                timestamp_s=timedelta(seconds=float(timestamp_s)),
            )
            solution.fix_objectives(problem_instance.problem)
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

    def to_dict(self, full=False):
        result_dict = {
            "ratio": list(self.solver_config.ratio),
            "total_time_sec": self.total_time_sec,
            "exact_solver_result": self.exact_solver_result.to_dict()
            if self.exact_solver_result is not None
            else None,
            "pls_result": self.pls_result.to_dict() if self.pls_result is not None else None,
        }

        if full:
            result_dict["problem_instance"] = self.problem_instance.to_dict()
            result_dict["solver_config"] = dataclasses.asdict(self.solver_config)
            result_dict["pls_result"] = (
                self.pls_result.to_dict() if self.pls_result is not None else None
            )
            if self.discarded_solutions is not None:
                result_dict["discarded_solutions"] = [
                    dataclasses.asdict(solution) for solution in self.discarded_solutions
                ]
            if self.added_solutions is not None:
                result_dict["added_solutions"] = [
                    dataclasses.asdict(solution) for solution in self.added_solutions
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
        total_time_sec = 0
        if exact_solver_result is not None:
            problem_instance = exact_solver_result.problem_instance
            total_time_sec += exact_solver_result.execution_time_sec

        if pls_result is not None:
            problem_instance = pls_result.problem_instance
            total_time_sec += pls_result.execution_time_sec

        if exact_solver_result is not None and pls_result is not None:
            discarded_solutions, added_solutions = (
                TwoPhaseSolverResult._detect_discarded_and_added_solutions(
                    exact_solver_result.pareto_front, pls_result.pareto_front
                )
            )
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
        pareto_snapshots_path: Path, problem: SimsDiscreteProblem
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
                    solution.fix_objectives(problem)
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
                pls_pareto_front_snapshots_path, problem=problem_instance.problem
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
