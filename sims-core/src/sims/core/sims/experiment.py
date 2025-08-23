from __future__ import annotations

import itertools
import json
import logging
import warnings
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from textwrap import wrap
from typing import Iterator, Optional

import pandas as pd
from geopandas import GeoDataFrame
from matplotlib import pyplot as plt

from .. import geometry, image_set
from ..data_providers import up42_provider as up42
from ..data_providers.up42_provider import SearchParameters
from ..image_set import PreprocessedData
from . import solver
from .geodata import Geodata
from .problem import ProblemInstance, SimsProblem
from .solver_config import SolverConfig, TwoPhaseSolverConfig
from .solver_result import Solution, TwoPhaseSolverResult

pd.options.mode.chained_assignment = None  # Suppress the SettingWithCopyWarning warning

plt.rcParams["text.antialiased"] = True  # Enable antialiasing for text

log = logging.getLogger(Path(__file__).stem)


def add_thousands_separator(number_str: str, sep=" ") -> str:
    whole, frac = number_str.split(".")
    frac = [frac[i : i + 3] for i in range(0, len(frac), 3)]
    return whole + "." + sep.join(frac)


@dataclass
class ExperimentResults:
    experiment: Experiment
    solver_config: SolverConfig
    solver_results: list[TwoPhaseSolverResult]

    def to_geodata(self, geodata_dir: Path, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)

        geodata = Geodata.load(geodata_dir)

        for solver_result in self.solver_results:
            ratio_dir = (
                output_dir
                / f"{solver_result.solver_config.ratio[0]}_{solver_result.solver_config.ratio[1]}"
            )
            ratio_dir.mkdir(parents=True, exist_ok=True)

            if solver_result.pls_result is not None:
                pareto_front = solver_result.pls_result.pareto_front
            elif solver_result.exact_solver_result is not None:
                pareto_front = solver_result.exact_solver_result.pareto_front
            else:
                # Should never happen
                raise ValueError("Both PLS and exact solver results are None")

            for solution_idx, solution in enumerate(pareto_front):
                solution_gs = geodata.original_images_gdf["geometry"].iloc[solution.selected_images]  # type: ignore # geopandas is poorly typed
                result_gs = pd.concat([solution_gs, geodata.aoi_gdf["geometry"]])

                result_gs.to_file(ratio_dir / f"solution{solution_idx}.geojson")  # type: ignore # geopandas is poorly typed

    def to_dict(self):
        return {
            "experiment": self.experiment.to_dict(),
            "solver_config": self.solver_config.to_dict(),
            "solver_results": [solver_result.to_dict() for solver_result in self.solver_results],
        }

    def to_json(self, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=4))

    def get_max_objectives(self, problem_based=False) -> tuple[int, int]:
        if problem_based:
            return self.experiment._problem_instance.problem.get_max_values()
        else:
            max_objectives = [
                solver_result.max_objectives() for solver_result in self.solver_results
            ]
            max_x = max(max_objectives, key=lambda x: x[0])[0]
            max_y = max(max_objectives, key=lambda x: x[1])[1]
            return (max_x, max_y)

    def compute_missing_hypervolumes(self, problem_based=False, scaled=True):
        max_objectives = self.get_max_objectives(problem_based)
        for solver_result in self.solver_results:
            if solver_result.pls_result is not None and solver_result.pls_result.hypervolume == 0:
                solver_result.pls_result.compute_hypervolume(
                    max_objectives=max_objectives, scaled=scaled
                )

    def recompute_all_hypervolumes(
        self, problem_based=False, scaled=True, max_objectives: Optional[tuple[int, int]] = None
    ):
        if max_objectives is None:
            max_objectives = self.get_max_objectives(problem_based)

        log.debug(f"Max objectives for {self.experiment._problem_instance.name}: {max_objectives}")
        for solver_result in self.solver_results:
            if solver_result.exact_solver_result is not None:
                solver_result.exact_solver_result.compute_hypervolume(
                    max_objectives=max_objectives, scaled=scaled
                )
            if solver_result.pls_result is not None:
                solver_result.pls_result.compute_hypervolume(
                    max_objectives=max_objectives, scaled=scaled
                )

    def generate_plot_report(
        self,
        output_dir: Path,
        max_objectives: Optional[tuple[int, int]] = None,
    ):
        fig = plt.figure(figsize=(25, 30), constrained_layout=True)
        fig.suptitle(
            f"Two-Phase Pareto Local Search ({repr(self.solver_config.solver_type)} {repr(self.solver_config.front_strategy)} and PLS) for SIMS problem",
            fontsize=25,
        )
        num_instances = 1
        subfigs = fig.subfigures(
            nrows=num_instances + 1, ncols=1, height_ratios=[1] * num_instances + [3]
        )
        subfig = subfigs[0]
        instance_name = self.experiment._problem_instance.name

        subfig.suptitle(f"Pareto front for instance {instance_name}")
        axes = subfig.subplots(nrows=1, ncols=len(self.solver_results))

        if max_objectives is None:
            max_objectives = self.get_max_objectives()

        for ax, solver_result in zip(axes, self.solver_results):
            solver_result.scatter_plot(ax, max_objectives)

        table_ax = subfigs[-1].add_subplot()

        self.generate_table_report(table_ax)

        fig.savefig(
            output_dir
            / f"{self.solver_config.solver_type}_{self.solver_config.front_strategy}_{instance_name}_report.svg",
            format="svg",
        )
        plt.close(fig)

    def _generate_metadata_dataframe(self) -> pd.DataFrame:
        data = [
            {
                "ratio": f"{two_phase_solver_result.solver_config.ratio[0]}% : {two_phase_solver_result.solver_config.ratio[1]}%",
                "exact_solver_timeout": two_phase_solver_result.exact_solver_result.timeout_sec
                if two_phase_solver_result.exact_solver_result is not None
                else None,
                "heuristic_timeout": two_phase_solver_result.pls_result.timeout_sec
                if two_phase_solver_result.pls_result is not None
                else None,
                "exact_solver_exec_time": two_phase_solver_result.exact_solver_result.execution_time_sec
                if two_phase_solver_result.exact_solver_result is not None
                else None,
                "heuristic_exec_time": two_phase_solver_result.pls_result.execution_time_sec
                if two_phase_solver_result.pls_result is not None
                else None,
                "exact_solver_new_solutions_count": len(
                    two_phase_solver_result.exact_solver_result.pareto_front
                )
                if two_phase_solver_result.exact_solver_result is not None
                else None,
                "heuristic_new_solutions_count": len(
                    two_phase_solver_result.heuristic_unique_solutions()
                )
                if two_phase_solver_result.heuristic_unique_solutions() is not None
                else None,  # type: ignore # pylance cannot infer that heuristic_unique_solutions() is not None
                "exact_solver_hypervolume": two_phase_solver_result.exact_solver_result.hypervolume
                if two_phase_solver_result.exact_solver_result is not None
                else None,
                "heuristic_hypervolume": two_phase_solver_result.pls_result.hypervolume
                if two_phase_solver_result.pls_result is not None
                else None,
            }
            for two_phase_solver_result in self.solver_results
        ]

        df = pd.DataFrame(data)

        df.insert(
            df.columns.get_loc("heuristic_exec_time") + 1,  # type: ignore # get_loc return type is steered by argument, int by defalt
            "total_exec_time",
            df["exact_solver_exec_time"].add(df["heuristic_exec_time"], fill_value=0),
        )
        df.insert(
            df.columns.get_loc("heuristic_new_solutions_count") + 1,  # type: ignore # get_loc return type is steered by argument, int by defalt
            "total_solutions_count",
            df["exact_solver_new_solutions_count"].add(
                df["heuristic_new_solutions_count"], fill_value=0
            ),
        )
        df.insert(
            df.columns.get_loc("heuristic_hypervolume") + 1,  # type: ignore # get_loc return type is steered by argument, int by defalt
            "final_hypervolume",
            df["heuristic_hypervolume"].combine_first(df["exact_solver_hypervolume"]),
        )

        return df

    def generate_table_report(self, ax: plt.Axes):  # type: ignore # pyplot considered not to export this module: https://github.com/matplotlib/matplotlib/issues/26812
        """Add table data to the plot"""
        # Create a table
        col_labels = {
            "exact_solver_timeout": "Exact Solver timeout, sec",
            "heuristic_timeout": "Heuristic timeout, sec",
            "exact_solver_exec_time": "Exact Solver execution time, sec",
            "heuristic_exec_time": "Heuristic execution time, sec",
            "total_exec_time": "Total execution time, sec",
            "exact_solver_new_solutions_count": "Exact Solver new solutions count",
            "discarded_solutions_count": "Discarded solutions count",
            "heuristic_new_solutions_count": "Heuristic new solutions count",
            "total_solutions_count": "Total solutions count",
            "exact_solver_hypervolume": "Exact Solver Hypervolume",
            "heuristic_hypervolume": "Heuristic Hypervolume",
            "final_hypervolume": "Final Hypervolume",
        }

        ax.axis("off")
        wrapped_col_labels = ["\n".join(wrap(col_label, 10)) for col_label in col_labels.values()]
        row_labels = []
        row_colors = []
        table_data = []
        cell_color_data = []
        ROW_COLORS = ["lightgrey", "white"]
        ROW_COLOR_ITER = cycle(ROW_COLORS)

        instance_name = self.experiment._problem_instance.name

        # Add empty row to separate instances
        table_data.append(["" for _ in range(len(col_labels))])
        cell_color_data.append(["white" for _ in range(len(col_labels))])
        row_labels.append(instance_name)
        row_colors.append("white")

        for two_phase_solver_result in self.solver_results:
            solver_config = two_phase_solver_result.solver_config
            row_color = next(ROW_COLOR_ITER)
            cell_colors = [row_color for _ in range(len(col_labels))]
            ratio = f"{solver_config.ratio[0]}% : {solver_config.ratio[1]}%"
            per_solver_stats = []

            for solver_result in [
                two_phase_solver_result.exact_solver_result,
                two_phase_solver_result.pls_result,
            ]:
                if solver_result is None:
                    per_solver_stats.append(
                        {
                            "timeout": "N/A",
                            "exec_time": "N/A",
                            "solutions_count": "N/A",
                            "hypervolume": "N/A",
                        }
                    )
                    continue

                per_solver_stats.append(
                    {
                        "timeout": solver_result.timeout_sec,
                        "exec_time": f"{solver_result.execution_time_sec:.1f}",
                        "solutions_count": len(solver_result.pareto_front),
                        "hypervolume": add_thousands_separator(
                            f"{solver_result.hypervolume:_.8f}", sep="'"
                        ),
                    }
                )

            # Provide only unique solutions count for the second phase if first phase is not None
            if two_phase_solver_result.added_solutions is not None:
                per_solver_stats[1]["solutions_count"] = len(
                    two_phase_solver_result.added_solutions
                )

            total_exec_time = 0
            total_exec_time += (
                two_phase_solver_result.exact_solver_result.execution_time_sec
                if two_phase_solver_result.exact_solver_result is not None
                else 0
            )
            total_exec_time += (
                two_phase_solver_result.pls_result.execution_time_sec
                if two_phase_solver_result.pls_result is not None
                else 0
            )

            if two_phase_solver_result.pls_result is not None:
                total_solutions_count = len(two_phase_solver_result.pls_result.pareto_front)
                final_hypervolume = two_phase_solver_result.pls_result.hypervolume
            elif two_phase_solver_result.exact_solver_result is not None:
                total_solutions_count = len(
                    two_phase_solver_result.exact_solver_result.pareto_front
                )
                final_hypervolume = two_phase_solver_result.exact_solver_result.hypervolume
            else:
                # Should never happen
                raise ValueError("Both PLS and exact solver results are None")

            discarded_solutions_count = (
                len(two_phase_solver_result.discarded_solutions)
                if two_phase_solver_result.discarded_solutions is not None
                else 0
            )

            table_row = [
                per_solver_stats[0]["timeout"],
                per_solver_stats[1]["timeout"],
                per_solver_stats[0]["exec_time"],
                per_solver_stats[1]["exec_time"],
                f"{total_exec_time:.1f}",
                per_solver_stats[0]["solutions_count"],
                str(discarded_solutions_count),
                per_solver_stats[1]["solutions_count"],
                str(total_solutions_count),
                per_solver_stats[0]["hypervolume"],
                per_solver_stats[1]["hypervolume"],
                add_thousands_separator(f"{final_hypervolume:_.8f}", sep="'"),
            ]

            table_data.append(table_row)
            cell_color_data.append(cell_colors)
            row_labels.append(ratio)
            row_colors.append("white")

        table = ax.table(
            cellText=table_data,
            colLabels=wrapped_col_labels,
            cellColours=cell_color_data,
            rowLabels=row_labels,
            rowColours=row_colors,
            loc="center",
        )

        for i in range(0, len(col_labels)):
            table[0, i].set_height(0.07)

        table.auto_set_column_width(col=list(range(1, len(col_labels))))  # Adjusts width to content
        table.auto_set_font_size(False)  # Disables automatic font size adjustment
        table.set_fontsize(20)  # Sets the font size
        table.scale(1, 1.5)  # Scales the table

    def render_table(self, table_path: Path):
        fig, ax = plt.subplots()
        self.generate_table_report(ax)
        fig.savefig(table_path, dpi=600)
        plt.close(fig)

    def process(self, output_dir: Path, recompute_hypervolumes: bool = True):
        output_dir.mkdir(parents=True, exist_ok=True)

        if recompute_hypervolumes:
            log.info("Recomputing the hypervolumes")
            self.recompute_all_hypervolumes()

        log.info("Storing the experiment results to json")
        self.to_json(output_dir / "experiment_results.json")

        log.info("Generating the plot report")
        self.generate_plot_report(output_dir)

    def compare_pareto_fronts(self, lhs_index: int, rhs_index: int):
        lhs_pareto_front: set[Solution] = set(
            self.solver_results[lhs_index].pls_result.pareto_front
        )
        rhs_pareto_front: set[Solution] = set(
            self.solver_results[rhs_index].pls_result.pareto_front
        )
        common_solutions = lhs_pareto_front.intersection(rhs_pareto_front)
        lhs_unique_solutions = lhs_pareto_front.difference(common_solutions)
        rhs_unique_solutions = rhs_pareto_front.difference(common_solutions)
        print(f"Common solutions count: {len(common_solutions)}")
        print(f"LHS unique solutions count: {len(lhs_unique_solutions)}")
        print(f"RHS unique solutions count: {len(rhs_unique_solutions)}")

    def has_no_solutions(self) -> bool:
        # TODO(hlvlad): Check all solutions and not only first one
        if self.solver_results[0].exact_solver_result is not None and len(self.solver_results[0].exact_solver_result.pareto_front) > 0:
            return False
        else:
            return True

        for solver_result in self.solver_results:
            if solver_result.exact_solver_result is not None and len(solver_result.exact_solver_result.pareto_front) > 0:
                return False
            if solver_result.pls_result is not None and len(solver_result.pls_result.pareto_front) > 0:
                return False
        return True

@dataclass
class ExperimentResultsSeries:
    experiment_series_results: list[ExperimentResults]

    def generate_metadata_table(self):
        results_dataframes = [
            experiment_results._generate_metadata_dataframe()
            for experiment_results in self.experiment_series_results
        ]
        for df in results_dataframes:
            df.set_index("ratio", inplace=True)

        return (
            pd.concat(
                results_dataframes,
                axis=0,
                keys=range(len(results_dataframes)),
                names=["iteration", "ratio"],
            )
            .swaplevel("iteration", "ratio")
            .reset_index()
        )

    def generate_mean_metadata_table(self, humanize_labels: bool = True):
        combined_dataframe = self.generate_metadata_table()

        for ratio, group in combined_dataframe.groupby("ratio"):
            log.critical(
                f"Ratio {ratio}: Final Hypervolumes: {group['final_hypervolume'].tolist()}"
            )

        combined_df_sems = (
            combined_dataframe.groupby("ratio")
            .sem()
            .rename(
                columns={col_name: f"{col_name}_sem" for col_name in combined_dataframe.columns}
            )
        )
        combined_df_means = combined_dataframe.groupby("ratio").mean()
        interleaved_columns = list(
            itertools.chain(*zip(combined_df_means.columns, combined_df_sems.columns))
        )
        sorted_ratios = [
            f"{ratio}% : {100 - ratio}%"
            for ratio in range(100, -1, -self.experiment_series_results[0].solver_config.ratio_step)
        ]

        combined_df_with_sems = pd.concat([combined_df_means, combined_df_sems], axis=1).reindex(
            columns=interleaved_columns, index=sorted_ratios
        )
        combined_df_with_sems.reset_index(inplace=True)

        if humanize_labels:
            combined_df_with_sems.rename(
                columns={
                    "ratio": "Ratio",
                    "exact_solver_timeout": "Exact Solver Timeout",
                    "exact_solver_timeout_sem": "Exact Solver Timeout Error",
                    "heuristic_timeout": "Heuristic Timeout",
                    "heuristic_timeout_sem": "Heuristic Timeout Error",
                    "exact_solver_exec_time": "Exact Solver Execution Time",
                    "exact_solver_exec_time_sem": "Exact Solver Execution Time Error",
                    "heuristic_exec_time": "Heuristic Execution Time",
                    "heuristic_exec_time_sem": "Heuristic Execution Time Error",
                    "total_exec_time": "Total Execution Time",
                    "total_exec_time_sem": "Total Execution Time Error",
                    "exact_solver_new_solutions_count": "Exact Solver New Solutions Count",
                    "exact_solver_new_solutions_count_sem": "Exact Solver New Solutions Count Error",
                    "heuristic_new_solutions_count": "Heuristic New Solutions Count",
                    "heuristic_new_solutions_count_sem": "Heuristic New Solutions Count Error",
                    "total_solutions_count": "Total Solutions Count",
                    "total_solutions_count_sem": "Total Solutions Count Error",
                    "exact_solver_hypervolume": "Exact Solver Hypervolume",
                    "exact_solver_hypervolume_sem": "Exact Solver Hypervolume Error",
                    "heuristic_hypervolume": "Heuristic Hypervolume",
                    "heuristic_hypervolume_sem": "Heuristic Hypervolume Error",
                    "final_hypervolume": "Final Hypervolume",
                    "final_hypervolume_sem": "Final Hypervolume Error",
                },
                inplace=True,
            )

        return combined_df_with_sems

    def __getitem__(self, index: int) -> ExperimentResults:
        return self.experiment_series_results[index]

    def max_objectives(self, problem_based=False) -> tuple[int, int]:
        max_objectives = [
            experiment_results.get_max_objectives(problem_based)
            for experiment_results in self.experiment_series_results
        ]
        max_x = max(max_objectives, key=lambda x: x[0])[0]
        max_y = max(max_objectives, key=lambda x: x[1])[1]
        return (max_x, max_y)

    def recompute_all_hypervolumes(self, problem_based=False, scaled=True, max_objectives: tuple[int, int] | None = None):
        max_objectives = max_objectives or self.max_objectives(problem_based)
        for experiment_results in self.experiment_series_results:
            experiment_results.recompute_all_hypervolumes(
                problem_based, scaled, max_objectives=max_objectives
            )


@dataclass
class Experiment:
    _problem_instance: ProblemInstance
    _preprocessed_data: PreprocessedData

    def to_dict(self):
        return {
            "problem_instance": self._problem_instance.to_dict(),
        }

    @staticmethod
    def from_dir(experiment_dir: Path, legacy_geodata: bool = False) -> Experiment:
        log.debug(f"Loading the experiment from {experiment_dir}")

        # Parse the problem instance
        dzn_path = next(experiment_dir.glob("*.dzn"))
        problem_instance = ProblemInstance.from_dzn(dzn_path)

        # Parse the geodata
        if legacy_geodata:
            geodata = Geodata.load(experiment_dir / "geodata")
            preprocessed_data = PreprocessedData(
                geodata.preprocessed_images_gdf,
                geodata.clipped_images_gdf.geometry,
                geodata.fragments_gdf.geometry,
                geodata.images_to_fragments_map
            )
        else:
            preprocessed_data = PreprocessedData.load(experiment_dir / "geodata")

        return Experiment(problem_instance, _preprocessed_data=preprocessed_data)

    def is_solved(self, experiment_dir: Path, solver_config: SolverConfig) -> bool:
        try:
            results_dir = next(experiment_dir.glob("solver_results*"))
        except StopIteration:
            log.warn(
                f"Experiment {experiment_dir.name} is not solved, no solved results folder found."
            )
            return False

        ratio_dirs = [
            f"{ratio}_{100 - ratio}" for ratio in range(100, -1, -solver_config.ratio_step)
        ]

        for ratio_dir in ratio_dirs:
            ratio_results_dir = results_dir / ratio_dir
            if not ratio_results_dir.exists():
                log.error(
                    f"Experiment {experiment_dir.name} is not solved for ratio {ratio_dir}, no results folder found."
                )
                return False

        log.info(f"Experiment {experiment_dir.name} is solved.")
        return True

    def solve(
        self,
        solver_results_dir: Path,
        solver_config: SolverConfig,
        dry_run: bool = False,
        iter_count: int = 1,
    ):
        solver_results_dir.mkdir(parents=True, exist_ok=True)
        solver_config.to_json(solver_results_dir / "solver_config.json")

        problem_path = self._problem_instance.path
        if problem_path is None:
            raise ValueError(f"Problem instance {self._problem_instance.name} has no path set")

        ratios = [(ratio, 100 - ratio) for ratio in range(100, -1, -solver_config.ratio_step)]

        for ratio in ratios:
            log.info(
                f"~~ Solving the instance {self._problem_instance.name} with ratio {ratio[0]}%:{ratio[1]}%, timeout {solver_config.timeout_s} sec ~~"
            )
            two_phase_solver_config = TwoPhaseSolverConfig(
                solver_config.solver_type,
                solver_config.front_strategy,
                timeout_s=solver_config.timeout_s,
                ratio=ratio,
            )

            for i in range(iter_count):
                log.info(f"~~~ Iteration {i + 1}/{iter_count} ~~~")

                ratio_solver_result_dir = solver_results_dir / f"{ratio[0]}-{ratio[1]}_iter{i}"
                ratio_solver_result_dir.mkdir(parents=True, exist_ok=True)

                try:
                    solver.solve_with_two_phases(
                        self._problem_instance,
                        problem_path,
                        ratio_solver_result_dir,
                        two_phase_solver_config,
                        objectives=["min_cost", "cloud_coverage", "min_max_incidence_angle"],
                        dry_run=dry_run,
                    )
                except Exception as e:
                    log.exception(
                        f"Failed to solve the problem instance with ratio {ratio}. Reason: {e}"
                    )
                    raise e

    def _parse_results_from_dir(
        self,
        solver_results_dir: Path,
        solver_config: SolverConfig | None = None,
        recompute_hypervolumes: bool = True,
        iter_count: int = 1,
    ) -> ExperimentResults:
        solver_config = solver_config or SolverConfig.from_json(
            solver_results_dir / "solver_config.json"
        )

        ratios = [(ratio, 100 - ratio) for ratio in range(100, -1, -solver_config.ratio_step)]

        solver_results = []
        for ratio in ratios:
            two_phase_solver_config = TwoPhaseSolverConfig(
                solver_config.solver_type,
                solver_config.front_strategy,
                timeout_s=solver_config.timeout_s,
                ratio=ratio,
            )

            if iter_count == 1:
                dir_name = f"{ratio[0]}-{ratio[1]}"
                result_csv_path = (
                    solver_results_dir
                    / dir_name
                    / f"{self._problem_instance.name.rsplit('_', maxsplit=1)[0]}.csv"
                )

                log.debug(f"Parsing the results for ratio {ratio}")

                solver_results.append(
                    TwoPhaseSolverResult.from_summary_csv(
                        result_csv_path, two_phase_solver_config, self._problem_instance
                    )
                )
            else:
                for i in range(iter_count):
                    dir_name = f"{ratio[0]}-{ratio[1]}_iter{i}"

                    log.debug(f"Parsing the results for ratio {ratio}")

                    result_csv_path = (
                        solver_results_dir
                        / dir_name
                        / f"{self._problem_instance.name.rsplit('_', maxsplit=1)[0]}.csv"
                    )

                    solver_results.append(
                        TwoPhaseSolverResult.from_summary_csv(
                            result_csv_path, two_phase_solver_config, self._problem_instance
                        )
                    )

        experiment_results = ExperimentResults(self, solver_config, solver_results)

        if recompute_hypervolumes:
            experiment_results.recompute_all_hypervolumes()

        return experiment_results

    def _parse_results_series_from_dir(
        self,
        solver_results_dir: Path,
        iter_count: int,
        solver_config: SolverConfig | None = None,
        recompute_hypervolumes: bool = True,
    ) -> ExperimentResultsSeries:
        solver_config = solver_config or SolverConfig.from_json(
            solver_results_dir / "solver_config.json"
        )

        ratios = [(ratio, 100 - ratio) for ratio in range(100, -1, -solver_config.ratio_step)]

        solver_results_series = [[] for _ in range(iter_count)]
        for ratio in ratios:
            two_phase_solver_config = TwoPhaseSolverConfig(
                solver_config.solver_type,
                solver_config.front_strategy,
                timeout_s=solver_config.timeout_s,
                ratio=ratio,
            )

            for i in range(iter_count):
                dir_name = f"{ratio[0]}-{ratio[1]}_iter{i}"

                log.debug(f"Parsing the results for ratio {ratio}")

                result_csv_path = (
                    solver_results_dir
                    / dir_name
                    / f"{self._problem_instance.name.rsplit('_', maxsplit=1)[0]}.csv"
                )
                try:
                    two_phase_solver_result = TwoPhaseSolverResult.from_summary_csv(
                        result_csv_path, two_phase_solver_config, self._problem_instance
                    )
                except Exception as e:
                    log.error(f"Error parsing results for {result_csv_path}: {e}")
                    continue

                solver_results_series[i].append(two_phase_solver_result)

        experiment_results_series = ExperimentResultsSeries(
            [
                ExperimentResults(self, solver_config, solver_results)
                for solver_results in solver_results_series
            ]
        )

        if recompute_hypervolumes:
            experiment_results_series.recompute_all_hypervolumes(scaled=True)

        return experiment_results_series

    def parse_results(
        self,
        experiment_dir: Path,
        solver_config: SolverConfig | None = None,
        recompute_hypervolumes: bool = True,
    ) -> ExperimentResults:
        if solver_config is None:
            warnings.warn(
                "Solver config is not provided, parsing results for first solver results dir."
            )
            solver_results_dir_name = next(experiment_dir.glob("solver_results*"))
            solver_config = SolverConfig.from_json(
                experiment_dir / solver_results_dir_name / "solver_config.json"
            )
        else:
            solver_results_dir_name = _get_solver_results_dir_name(solver_config)

        solver_results_dir = experiment_dir / solver_results_dir_name
        return self._parse_results_from_dir(
            solver_results_dir, solver_config, recompute_hypervolumes
        )

    def parse_results_series(
        self,
        result_dir: Path,
        iter_count: int,
        solver_config: SolverConfig | None = None,
        recompute_hypervolumes: bool = True,
    ) -> ExperimentResultsSeries:
        if solver_config is None:
            warnings.warn(
                "Solver config is not provided, parsing results for first solver results dir."
            )
            solver_config = SolverConfig.from_json(result_dir / "solver_config.json")

        return self._parse_results_series_from_dir(
            result_dir, iter_count, solver_config, recompute_hypervolumes
        )


def _change_log_handler(log_file: Path):
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s %(name)-17s [%(levelname)-6s] %(message)s")
    )
    root_logger.addHandler(file_handler)


def _get_solver_results_dir_name(solver_config: SolverConfig) -> str:
    return f"solver_results_{solver_config.solver_type}_{solver_config.front_strategy}_{solver_config.timeout_s}sec"


def _prepare_preprocessed_data(
    aoi_gdf: GeoDataFrame,
    image_count: int,
    max_intersect_percentage: Optional[float] = None,
    preselected_images_gdf: Optional[GeoDataFrame] = None,
) -> PreprocessedData:
    if preselected_images_gdf is not None:
        log.info("Using preselected images.")
        original_images_gdf = preselected_images_gdf
    else:
        log.info("Fetching images from UP42 data provider.")
        original_images_gdf = up42.fetch(
            aoi_gdf, search_params=SearchParameters(max_image_count=image_count)
        )

    if "cost" not in original_images_gdf.columns:
        log.info("Estimating costs for the images")
        costs = up42.estimate_missing_costs(original_images_gdf, aoi_gdf)
        original_images_gdf["cost"] = costs

    log.info("Estimating UTM CRS")
    projected_crs = str(original_images_gdf.estimate_utm_crs())

    log.info("Normalizing the images")
    normalized_images_gdf = up42.normalize(original_images_gdf)

    log.info("Preprocessing the images")
    preprocessed_data: PreprocessedData = image_set.preprocess(
        normalized_images_gdf,
        aoi_gdf,
        projected_crs=projected_crs,
        max_intersect_percentage=max_intersect_percentage,
    )

    return preprocessed_data


def _generate_problem_instance_from_aoi(
    aoi_path: Path,
    result_dir: Path,
    image_count: int,
    preselected_images_path: Optional[Path] = None,
    max_intersect_percentage: Optional[float] = None,
) -> ProblemInstance:
    result_dir.mkdir(parents=True, exist_ok=True)

    log.info("Parsing the Area Of Interest from the GeoJSON file.")
    aoi_gdf = GeoDataFrame.from_file(aoi_path)

    if preselected_images_path is not None:
        preselected_images_gdf = GeoDataFrame.from_file(preselected_images_path)
        image_count = len(preselected_images_gdf)
    else:
        preselected_images_gdf = None

    log.info("Preparing preprocessed data")
    preprocessed_data = _prepare_preprocessed_data(
        aoi_gdf,
        image_count,
        preselected_images_gdf=preselected_images_gdf,
        max_intersect_percentage=max_intersect_percentage,
    )

    log.info("Saving the preprocessed data to the result directory")
    preprocessed_data.save(result_dir / "geodata")

    log.info("Preparing the SIMS discrete problem instance")
    sims_discrete_problem = SimsProblem.from_preprocessed_data(preprocessed_data).discretize()
    instance_name = f"{aoi_path.stem}_{image_count}"
    problem_instance = ProblemInstance(name=instance_name, problem=sims_discrete_problem)

    log.info("Saving the problem instance to the result directory")
    problem_instance.to_dzn(result_dir / f"{instance_name}.dzn")

    return problem_instance


def prepare(
    aoi_path: Path,
    images_count: int,
    result_dir: Path,
    preselected_images_path: Optional[Path] = None,
    max_intersect_percentage: Optional[float] = None,
) -> ProblemInstance:
    result_dir.mkdir(parents=True, exist_ok=True)

    _change_log_handler(result_dir / "prepare.log")

    log.info("Generating the problem instance")
    return _generate_problem_instance_from_aoi(
        aoi_path,
        result_dir,
        images_count,
        preselected_images_path,
        max_intersect_percentage=max_intersect_percentage,
    )


def solve(experiment_dir: Path, solver_config: SolverConfig, result_dir: Path | None = None, dry_run: bool = False, iter_count: int = 1, skip_solved: bool = False):
    if result_dir is None:
        result_dir = experiment_dir / _get_solver_results_dir_name(solver_config)

    result_dir.mkdir(parents=True, exist_ok=True)

    _change_log_handler(result_dir / "solve.log")

    experiment = Experiment.from_dir(experiment_dir)
    if skip_solved and experiment.is_solved(result_dir, solver_config):
        log.info(f"Experiment {experiment_dir.name} is already solved, skipping.")
        return
    experiment.solve(result_dir, solver_config, dry_run=dry_run, iter_count=iter_count)


def parse_results(experiment_dir: Path, result_dir: Path) -> ExperimentResults:
    _change_log_handler(result_dir / "results_analysis.log")
    experiment = Experiment.from_dir(experiment_dir)

    log.info("Parsing the results")
    experiment_results = experiment.parse_results(result_dir)

    log.info("Recomputing the hypervolumes")
    experiment_results.recompute_all_hypervolumes()

    return experiment_results


def parse_results_series(
    experiment_dirs: list[Path], result_dirs: list[Path]
) -> ExperimentResultsSeries:
    _change_log_handler(result_dirs[0] / "results_analysis.log")
    experiment_series_results = []
    for experiment_dir, result_dir in zip(experiment_dirs, result_dirs):
        experiment = Experiment.from_dir(experiment_dir)

        log.debug("Parsing the results")
        experiment_results = experiment.parse_results(result_dir)
        experiment_series_results.append(experiment_results)

    return ExperimentResultsSeries(experiment_series_results)


def process_results(
    experiment_dir: Path,
    experiment_results_dir: Path,
    output_dir: Path,
):
    _change_log_handler(output_dir / f"{experiment_results_dir.name}_results_processing.log")
    experiment = Experiment.from_dir(experiment_dir)

    log.info("Parsing the results")
    experiment_results = experiment.parse_results(experiment_results_dir)

    log.info("Recomputing the hypervolumes")
    experiment_results.recompute_all_hypervolumes()

    log.info("Storing the experiment results to json")
    experiment_results.to_json(output_dir / f"{experiment_results_dir.name}_results.json")

    log.info("Generating the plot report")
    experiment_results.generate_plot_report(output_dir)
