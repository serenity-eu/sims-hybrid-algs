#!/usr/bin/env python3

import argparse
import csv
import datetime
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from geopandas import GeoDataFrame, GeoSeries
from matplotlib.gridspec import GridSpec
from scipy.stats import wilcoxon, friedmanchisquare
import scikit_posthocs as sp
from sims.core import (
    Experiment,
    ExperimentResults,
    FrontStrategy,
    Solution,
    SolverConfig,
    SolverType,
    experiment,
    SimsDiscreteProblem,
)

ROOT_DIR = Path(__file__).parent.parent.resolve()
RESULTS_DIR = ROOT_DIR / "results"
TEST_RESULT_DIR = RESULTS_DIR / "test_result"
TEST_DATA_DIR = ROOT_DIR / "test_data"
AOIS_DIR = TEST_DATA_DIR / "original" / "aois"
PRESELECTED_IMAGES_DIR = TEST_DATA_DIR / "original" / "preselected_images"

EXAMPLE_DATA_DIR = TEST_DATA_DIR / "new_1.5x_reduced"
EXAMPLE_AOI_DIR = EXAMPLE_DATA_DIR / "aois"
EXAMPLE_PRESELECTED_IMAGES_DIR = EXAMPLE_DATA_DIR / "preselected_images"

dotenv.load_dotenv()

log_format = "%(asctime)s %(name)-5s [%(levelname)-7s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

log = logging.getLogger("main")

TEST_INSTANCE_NAME = "paris"
TEST_INSTANCE_IMAGES_PATH = PRESELECTED_IMAGES_DIR / f"{TEST_INSTANCE_NAME}_150_image_set.geojson"
TEST_AOI_PATH = AOIS_DIR / f"{TEST_INSTANCE_NAME}.geojson"


def scale_gdf(gdf: GeoDataFrame, scale_factor: float) -> GeoDataFrame:
    result_gdf = gdf.copy()
    result_gdf["geometry"] = GeoSeries(gdf["geometry"]).scale(
        xfact=scale_factor, yfact=scale_factor, zfact=1.0, origin="center"
    )
    return GeoDataFrame(result_gdf)


def generate_image_set_subsets(
    aoi_gdf: GeoDataFrame, images_gdf: GeoDataFrame, images_counts: list[int]
):
    images_gdf_random_order = images_gdf.sample(frac=1).copy()

    desired_count_iter = iter(images_counts)

    output_gdfs = []
    try:
        desired_count = next(desired_count_iter)

        for index, _ in images_gdf_random_order.iterrows():
            temp_images_gdf = images_gdf_random_order.drop(index)
            if temp_images_gdf.unary_union.contains(aoi_gdf.geometry[0]):
                images_gdf_random_order.drop(index, inplace=True)
                if len(temp_images_gdf) == desired_count:
                    output_gdfs.append(temp_images_gdf)
                    desired_count = next(desired_count_iter)
        log.error(
            f"Failed to generate all image subsets. Requested subsets: {images_counts}, generated subsets: {images_counts[-len(output_gdfs) :]}"
        )
    except StopIteration:
        pass

    return output_gdfs


def solutions_to_geo_data(
    solutions: list[Solution], images_gdf: GeoDataFrame, aoi_gdf: GeoDataFrame
):
    result_gdfs = []
    for solution in solutions:
        selected_images_gdf = images_gdf.iloc[list(solution.selected_images)]
        selected_images_gs = pd.concat([selected_images_gdf["geometry"], aoi_gdf["geometry"]])
        images_with_aoi_gdf = GeoDataFrame(geometry=selected_images_gs, crs=images_gdf.crs)  # type: ignore # geopandas is poorly typed
        result_gdfs.append(images_with_aoi_gdf)
    return result_gdfs


_ANY_IMAGES_COUNT = 0

def prepare_experiments(experiments_dir: Path | None = None, satellite_data_dir: Path | None = None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Use default results directory if not provided
    experiments_dir = experiments_dir or RESULTS_DIR / f"real_experiment_{timestamp}"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    INPUT_DATA_DIR = TEST_DATA_DIR / "new_1.5x_reduced"
    if satellite_data_dir is not None:
        aoi_dir = satellite_data_dir / "aois"
        preselected_images_dir = satellite_data_dir / "images"
    else:
        aoi_dir = INPUT_DATA_DIR / "aois"
        preselected_images_dir = INPUT_DATA_DIR / "preselected_images"

    found_aois = list(aoi_dir.glob("*.geojson"))

    if not found_aois:
        log.error(f"No AOIs found in {aoi_dir}.")
        return

    for aoi_path in found_aois:
        found_preselected_images = list(preselected_images_dir.glob(f"{aoi_path.stem}*.geojson"))

        if not found_preselected_images:
            log.error(
                f"No preselected images found for {aoi_path.stem} in {preselected_images_dir}."
            )
            return

        for preselected_images_path in preselected_images_dir.glob(f"{aoi_path.stem}*.geojson"):
            experiment_name = preselected_images_path.stem[: -len("_images")]

            experiment_dir = experiments_dir / experiment_name
            log.info(f"Preparing experiment {experiment_name} under path: {experiment_dir}")
            experiment.prepare(aoi_path, _ANY_IMAGES_COUNT, experiment_dir, preselected_images_path)


def solve_experiments(
    experiments_dir: Path,
    modified_solver_config: SolverConfig,
    dry_run: bool = False,
    iter_count: int = 1,
    instance_regex: str | None = None,
    skip_solved: bool | None = False,
    results_dir: Path | None = None,
):
    solver_config = SolverConfig(
        solver_type=modified_solver_config.solver_type or SolverType.GUROBI,
        front_strategy=modified_solver_config.front_strategy or FrontStrategy.GBPA_A,
        timeout_s=modified_solver_config.timeout_s or 600,
        ratio_step=modified_solver_config.ratio_step or 20,
    )

    experiment_dirs = None
    if instance_regex is not None:
        experiment_dirs = [
            experiment_dir
            for experiment_dir in experiments_dir.glob(instance_regex)
            if experiment_dir.is_dir()
        ]
        log.info(f'Selected instances matching regex "{instance_regex}": {experiment_dirs}')
    else:
        experiment_dirs = [experiment_dir for experiment_dir in experiments_dir.iterdir()]

    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        result_dirs = [
            results_dir / experiment_dir.name for experiment_dir in experiment_dirs
        ]
        for experiment_dir, result_dir in zip(experiment_dirs, result_dirs):
            if result_dir.exists():
                log.error("Result directory already exists. Remove it before continue.")
                sys.exit(1)

            shutil.copytree(experiment_dir, result_dir)
            # Remove old solver results directories
            for subdir in result_dir.iterdir():
                if subdir.is_dir() and subdir.name.startswith("solver_results_"):
                    shutil.rmtree(subdir)
        
        experiment_dirs = result_dirs


    for experiment_idx, experiment_dir in enumerate(experiment_dirs):
        if not experiment_dir.is_dir():
            continue
        log.info(
            f"~~~~~~ Solving experiment {experiment_dir.name} ({experiment_idx + 1}/{len(experiment_dirs)}) ~~~~~~"
        )
        try:
            experiment.solve(
                experiment_dir=experiment_dir,
                result_dir=experiment_dir,
                solver_config=solver_config,
                dry_run=dry_run,
                iter_count=iter_count,
                skip_solved=skip_solved,
            )
        except Exception as e:
            log.error(f"Failed to solve experiment {experiment_dir.name}. Reason: {e}")


def process_experiments_results(experiments_dir: Path, output_dir: Path):
    experiments_output_dir = output_dir / experiments_dir.name
    experiments_output_dir.mkdir(parents=True, exist_ok=True)

    for experiment_dir in experiments_dir.iterdir():
        if not experiment_dir.is_dir():
            continue
        experiment = Experiment.from_dir(experiment_dir)
        solver_config = SolverConfig.from_json(experiment_dir / "solver_config.json")
        if experiment.is_solved(experiment_dir, solver_config):
            log.info(f"Experiment {experiment_dir.name} is solved.")
            experiment_results = experiment.parse_results(experiment_dir=experiment_dir)
            experiment_results.process(output_dir=experiments_output_dir)
        else:
            log.error(f"Experiment {experiment_dir.name} is not solved.")


def generate_reports(experiments_dir: Path, reports_dir: Path, instance_regex: str | None = None):
    notebook_path = ROOT_DIR / "serenity-cli" / "experiments_analysis.ipynb"
    report_params_path = ROOT_DIR / "serenity-cli" / "experiments_analysis_input.py"
    log.critical(f"Reports dir: {reports_dir}")
    reports_dir.mkdir(parents=True, exist_ok=True)

    experiment_dirs = (
        experiments_dir.iterdir()
        if instance_regex is None
        else experiments_dir.glob(instance_regex)
    )
    if instance_regex is not None:
        experiment_dirs = [
            experiment_dir
            for experiment_dir in experiments_dir.glob(instance_regex)
            if experiment_dir.is_dir()
        ]
        log.info(f'Selected instances matching regex "{instance_regex}": {experiment_dirs}')
    else:
        experiment_dirs = [experiment_dir for experiment_dir in experiments_dir.iterdir()]
    for experiment_dir in experiment_dirs:
        if not experiment_dir.is_dir():
            continue

        report_params = f"""
from pathlib import Path
EXPERIMENT_DIR = Path(\'{experiment_dir}\')
    """
        report_params_path.write_text(report_params)
        report_path = reports_dir / f"{experiment_dir.name}_report.html"
        try:
            subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    notebook_path,
                    "--output",
                    report_path,
                    "--to",
                    "html",
                    "--no-input",
                    "--execute",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            log.error(
                f"Failed to generate report for experiment {experiment_dir.name}. Reason: {e}"
            )


def _generate_pareto_front_plots_sns(
    experiment_results: ExperimentResults,
    metadata_stats_df: pd.DataFrame,
    output_dir: Path,
    large_barplot: bool = False,
    horizontal: bool = True,
    show_bar_labels: bool = False,
    experiment_results2: ExperimentResults | None = None,
    metadata_stats_df2: pd.DataFrame | None = None,
):
    is_comparison = experiment_results2 is not None

    solutions_df = pd.DataFrame()
    instance_name = experiment_results.experiment._problem_instance.name
    max_objectives = experiment_results.get_max_objectives()
    experiments_solver_results = [experiment_results.solver_results]

    if is_comparison:
        max_objectives2 = experiment_results2.get_max_objectives()
        max_objectives = (
            max(max_objectives[0], max_objectives2[0]),
            max(max_objectives[1], max_objectives2[1]),
        )
        experiments_solver_results.append(experiment_results2.solver_results)

    for experiment_num, solver_results in enumerate(experiments_solver_results):
        for two_phase_result in solver_results:
            if two_phase_result.solver_config.ratio not in [(0, 100), (50, 50), (100, 0)]:
                continue

            for result in [two_phase_result.exact_solver_result, two_phase_result.pls_result]:
                if result is None:
                    continue

                pareto_front = (
                    result.pareto_front
                    if result.solver_type != SolverType.PLS
                    else two_phase_result.heuristic_unique_solutions()
                )

                if result.solver_type == SolverType.GUROBI:
                    if experiment_num == 0:
                        solver_name = "GPBA-A"
                    else:
                        solver_name = "Anytime Aneja & Nair"
                else:
                    solver_name = repr(result.solver_type)

                result_solutions_data = [
                    {
                        "cost": solution.cost,
                        "experiment_num": experiment_num,
                        "cloudy_area": solution.cloudy_area,
                        "solver": solver_name,
                        "ratio": f"{two_phase_result.solver_config.ratio[0]}%:{two_phase_result.solver_config.ratio[1]}%",
                    }
                    for solution in pareto_front
                ]
                result_solutions_df = pd.DataFrame(result_solutions_data)

                solutions_df = pd.concat([solutions_df, result_solutions_df], ignore_index=True)

            # final_pareto_front = (
            #     two_phase_result.pls_result.pareto_front
            #     if two_phase_result.pls_result is not None
            #     else two_phase_result.exact_solver_result.pareto_front
            # )

            # final_solutions_data = [
            #     {
            #         "cost": solution.cost,
            #         "cloudy_area": solution.cloudy_area,
            #         "solver": "Final",
            #         "ratio": f"{two_phase_result.solver_config.ratio[0]}%:{two_phase_result.solver_config.ratio[1]}%",
            #     }
            #     for solution in final_pareto_front
            # ]

            # final_solutions_df = pd.DataFrame(final_solutions_data)

            # solutions_df = pd.concat([solutions_df, final_solutions_df], ignore_index=True)

            # solutions_df = solutions_df[solutions_df["solver"] != "Final"]

            # Scale objectives

    solutions_df["cost"] /= max_objectives[0]
    solutions_df["cloudy_area"] /= max_objectives[1]

    solutions_df["solver"] = pd.Categorical(
        solutions_df["solver"],
        categories=["Pareto Local Search", "GPBA-A", "Anytime Aneja & Nair"],
        ordered=True,
    )

    solutions_df["ratio"] = pd.Categorical(
        solutions_df["ratio"],
        categories=["100%:0%", "50%:50%", "0%:100%"],
        ordered=True,
    )

    # Sort the solutions by the solver
    solutions_df.sort_values(["ratio", "solver"], inplace=True)

    labels = {
        "cost": "Cost",
        "cloudy_area": "Cloudy area",
        "solver": "Pareto front",
        "ratio": "Ratio",
    }
    solutions_df.rename(columns=labels, inplace=True)

    metadata_stats_df.rename(
        columns={
            "Final Hypervolume": "Hypervolume",
            "Total Solutions Count": "Solutions Count",
        },
        inplace=True,
    )
    if is_comparison:
        metadata_stats_df2.rename(
            columns={
                "Final Hypervolume": "Hypervolume",
                "Total Solutions Count": "Solutions Count",
            },
            inplace=True,
        )

    sns.set_context("paper", font_scale=4)
    sns.set_style("darkgrid")

    if not is_comparison:
        fig = plt.figure(figsize=(15, 18))
        if large_barplot:
            gs = GridSpec(2, 3, height_ratios=[3, 1], hspace=0.4, wspace=0.15)
            ax0 = plt.subplot(gs[0, :])
            ax1 = plt.subplot(gs[1, 0])
            ax2 = plt.subplot(gs[1, 1])
            ax3 = plt.subplot(gs[1, 2])
        else:
            axes = fig.subplots(nrows=2, ncols=2, gridspec_kw={"hspace": 0.4})
            [ax0, ax1, ax2, ax3] = axes.flatten()

        experiments_axes = [(ax0, ax1, ax2, ax3)]
        experiments_solutions_dfs = [solutions_df]
        experiments_metadata_dfs = [metadata_stats_df]
    else:
        fig = plt.figure(figsize=(45, 27))
        # fig.suptitle("Comparison of Front Strategies GBPA-A (left) and Aneja-Nair (right)")
        if large_barplot:
            gs = GridSpec(2, 6, height_ratios=[3, 1], hspace=0.4, wspace=0.15)
            ex1ax0 = plt.subplot(gs[0, :3])
            ex1ax1 = plt.subplot(gs[1, 0])
            ex1ax2 = plt.subplot(gs[1, 1])
            ex1ax3 = plt.subplot(gs[1, 2])
            # Second plot set
            ex2ax0 = plt.subplot(gs[0, 3:])
            ex2ax1 = plt.subplot(gs[1, 3])
            ex2ax2 = plt.subplot(gs[1, 4])
            ex2ax3 = plt.subplot(gs[1, 5])
        elif horizontal:
            gs = GridSpec(2, 4, hspace=0.4, wspace=0.25)
            ex1ax0 = plt.subplot(gs[0, 0])
            ex1ax1 = plt.subplot(gs[0, 1])
            ex1ax2 = plt.subplot(gs[0, 2])
            ex1ax3 = plt.subplot(gs[0, 3])
            # Second plot set
            ex2ax0 = plt.subplot(gs[1, 0])
            ex2ax1 = plt.subplot(gs[1, 1])
            ex2ax2 = plt.subplot(gs[1, 2])
            ex2ax3 = plt.subplot(gs[1, 3])
        else:
            # Nested Gridspecs
            main_gs = GridSpec(1, 2, figure=fig)
            gs_left = main_gs[0].subgridspec(2, 2, hspace=0.4, wspace=0.3)
            gs_right = main_gs[1].subgridspec(2, 2, hspace=0.4, wspace=0.3)

            ex1ax0 = fig.add_subplot(gs_left[0, 0])
            ex1ax1 = fig.add_subplot(gs_left[0, 1])
            ex1ax2 = fig.add_subplot(gs_left[1, 0])
            ex1ax3 = fig.add_subplot(gs_left[1, 1])

            ex2ax0 = fig.add_subplot(gs_right[0, 0])
            ex2ax1 = fig.add_subplot(gs_right[0, 1])
            ex2ax2 = fig.add_subplot(gs_right[1, 0])
            ex2ax3 = fig.add_subplot(gs_right[1, 1])

        experiments_axes = [(ex1ax0, ex1ax1, ex1ax2, ex1ax3), (ex2ax0, ex2ax1, ex2ax2, ex2ax3)]
        experiments_solutions_dfs = [
            solutions_df[solutions_df["experiment_num"] == 0],
            solutions_df[solutions_df["experiment_num"] == 1],
        ]
        experiments_metadata_dfs = [metadata_stats_df, metadata_stats_df2]

    cost_range = tuple(solutions_df["Cost"].agg(["min", "max"]))
    cost_margin = (cost_range[1] - cost_range[0]) * 0.1
    cloudy_area_range = tuple(solutions_df["Cloudy area"].agg(["min", "max"]))
    cloudy_area_margin = (cloudy_area_range[1] - cloudy_area_range[0]) * 0.1
    final_hypervolume_range = tuple(metadata_stats_df["Hypervolume"].agg(["min", "max"]))
    if is_comparison:
        final_hypervolume_range2 = tuple(
            metadata_stats_df2["Hypervolume"].agg(["min", "max"])
        )
        final_hypervolume_range = (
            min(final_hypervolume_range[0], final_hypervolume_range2[0]),
            max(final_hypervolume_range[1], final_hypervolume_range2[1]),
        )
    final_hypervolume_margin = (final_hypervolume_range[1] - final_hypervolume_range[0]) * 0.1

    for (ax0, ax1, ax2, ax3), solutions_df, metadata_df in zip(
        experiments_axes, experiments_solutions_dfs, experiments_metadata_dfs
    ):
        sns.barplot(data=metadata_df, x="Ratio", y="Hypervolume", color="#2ca02c", ax=ax0)

        if show_bar_labels:
            ax0.bar_label(
                ax0.containers[0],
                label_type="edge",
                fontsize=12,
                padding=3,
                color="black",
                rotation=30,
            )

        # ax0.tick_params(axis="x", labelrotation=30)
        plt.setp(ax0.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax0.set_title("a)")

        ax1.set_xlim(cost_range[0] - cost_margin, cost_range[1] + cost_margin)
        ax1.set_ylim(
            cloudy_area_range[0] - cloudy_area_margin, cloudy_area_range[1] + cloudy_area_margin
        )
        ax2.set_xlim(cost_range[0] - cost_margin, cost_range[1] + cost_margin)
        ax2.set_ylim(
            cloudy_area_range[0] - cloudy_area_margin, cloudy_area_range[1] + cloudy_area_margin
        )
        ax3.set_xlim(cost_range[0] - cost_margin, cost_range[1] + cost_margin)
        ax3.set_ylim(
            cloudy_area_range[0] - cloudy_area_margin, cloudy_area_range[1] + cloudy_area_margin
        )
        # ax1.set_xlim(-0.1, 1.1)
        # ax1.set_ylim(-0.1, 1.1)
        # ax2.set_xlim(-0.1, 1.1)
        # ax2.set_ylim(-0.1, 1.1)
        # ax3.set_xlim(-0.1, 1.1)
        # ax3.set_ylim(-0.1, 1.1)
        ax0.set_ylim(
            final_hypervolume_range[0] - final_hypervolume_margin,
            final_hypervolume_range[1] + final_hypervolume_margin,
        )

        ratios = ["100%:0%", "50%:50%", "0%:100%"]
        axes = [ax1, ax2, ax3]
        plot_labels = ["b)", "c)", "d)"]
        pallete = sns.color_palette("tab10")
        custom_palette = {
            "GPBA-A": pallete[1],
            "Anytime Aneja & Nair": pallete[3],
            "Pareto Local Search": pallete[0],
        }
        markers = {
            "GPBA-A": "X",
            "Anytime Aneja & Nair": "X",
            "Pareto Local Search": "^",
        }
        for ratio, ax, label in zip(ratios, axes, plot_labels):
            sns.scatterplot(
                data=solutions_df[solutions_df["Ratio"] == ratio],
                x="Cost",
                y="Cloudy area",
                hue="Pareto front",
                hue_order=["GPBA-A", "Anytime Aneja & Nair", "Pareto Local Search"],
                style="Pareto front",
                palette=custom_palette,
                markers=markers,
                s=200,
                edgecolors="none",
                ax=ax,
            )
            ax.set_title(f"{label} Ratio: {ratio}")
            ax.legend().set_visible(False)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center")

    log.info(f"Saving to path {output_dir / f'{instance_name}_pareto_fronts.png'}")
    fig.savefig(output_dir / f"{instance_name}_pareto_fronts.png", bbox_inches="tight", dpi=300)


def generate_plots(
    experiments_dir: Path,
    output_dir: Path,
    instance_regex: str | None = None,
    experiments_dir2: Path | None = None,
):
    comparing = experiments_dir2 is not None

    experiment_dirs = None
    if instance_regex is not None:
        experiment_dirs = [
            experiment_dir
            for experiment_dir in experiments_dir.glob(instance_regex)
            if experiment_dir.is_dir()
        ]
        log.info(f'Selected instances matching regex "{instance_regex}": {experiment_dirs}')
        if comparing:
            experiment_dirs2 = [
                experiment_dir
                for experiment_dir in experiments_dir2.glob(instance_regex)
                if experiment_dir.is_dir()
            ]
            log.info(f'Selected instances matching regex "{instance_regex}": {experiment_dirs2}')
    else:
        experiment_dirs = list(experiments_dir.iterdir())
        if comparing:
            experiment_dirs2 = list(experiments_dir2.iterdir())

    # Generate plots
    output_dir.mkdir(parents=True, exist_ok=True)

    for dir_num, experiment_dir in enumerate(experiment_dirs):
        # Parse experiment results
        try:
            experiment = Experiment.from_dir(experiment_dir)
            experiment_results_series = experiment.parse_results_series(
                experiment_dir, 10, recompute_hypervolumes=False
            )
            experiment_results = experiment_results_series[0]
            if experiment_results.has_no_solutions():
                log.warning(f"Experiment {experiment_dir.name} has no solutions. Skipping.")
                continue
        except Exception as e:
            log.error(f"Failed to parse experiment {experiment_dir.name}. Reason: {e}")
            continue

        # Parse second experiment if present:
        if comparing:
            try:
                experiment_dir2 = experiment_dirs2[dir_num]
                experiment2 = Experiment.from_dir(experiment_dir2)
                experiment_results_series2 = experiment2.parse_results_series(
                    experiment_dir2, 10, recompute_hypervolumes=False
                )
                experiment_results2 = experiment_results_series2[0]
                if experiment_results2.has_no_solutions():
                    log.warning(f"Experiment2 {experiment_dir2.name} has no solutions. Skipping.")
                    continue

                max_objectives = experiment_results.get_max_objectives()
                max_objectives2 = experiment_results2.get_max_objectives()

                max_objectives = (
                    max(max_objectives[0], max_objectives2[0]),
                    max(max_objectives[1], max_objectives2[1]),
                )
                experiment_results_series.recompute_all_hypervolumes(max_objectives=max_objectives)
                experiment_results_series2.recompute_all_hypervolumes(max_objectives=max_objectives)

                log.info("Generating metadata stats tables")
                metadata_stats_df = experiment_results_series.generate_mean_metadata_table()
                metadata_stats_df2 = experiment_results_series2.generate_mean_metadata_table()
                all_metadata_df = experiment_results_series.generate_metadata_table()
                all_metadata_df2 = experiment_results_series2.generate_metadata_table()
                all_metadata_df["method"] = "aneja"
                all_metadata_df2["method"] = "gpba"
            except Exception as e:
                log.exception(f"Failed to parse experiment {experiment_dir.name}. Reason: {e}")
                continue
        else:
            experiment_results.recompute_all_hypervolumes()

            log.info("Generating metadata stats table")
            metadata_stats_df = experiment_results_series.generate_mean_metadata_table()

            experiment_results2 = None
            metadata_stats_df2 = None

        # Compute hypervolumes based on max objectives, either from first experiment or from both, if compared
        try:
            _generate_pareto_front_plots_sns(
                experiment_results_series[0],
                metadata_stats_df,
                output_dir,
                show_bar_labels=False,
                horizontal=True,
                experiment_results2=experiment_results2,
                metadata_stats_df2=metadata_stats_df2,
            )

            # Save concatenated metadata stats tables
            if comparing:
                all_metadata_stats_df = pd.concat([all_metadata_df, all_metadata_df2], axis=0)
                metadata_dir = output_dir / "metadata"
                metadata_dir.mkdir(parents=True, exist_ok=True)
                all_metadata_stats_df.to_csv(
                    metadata_dir / f"{experiment_dir.name}_metadata.csv",
                    quoting=csv.QUOTE_NONNUMERIC,
                )
        except Exception as e:
            log.exception(
                f"Failed to generate pareto front plots for {experiment_dir.name}. Reason: {e}"
            )
            continue


def validate_solutions(experiments_dir: Path) -> bool:
    is_valid = True
    for experiment_dir in experiments_dir.iterdir():
        if not experiment_dir.is_dir():
            continue
        try:
            experiment = Experiment.from_dir(experiment_dir)
            experiment_results_series = experiment.parse_results_series(experiment_dir, 1)
        except Exception as e:
            log.error(f"Failed to parse experiment {experiment_dir.name}. Reason: {e}")
            continue

        for result in experiment_results_series[0].solver_results:
            if result.exact_solver_result is not None:
                if not result.exact_solver_result.validate():
                    log.error(
                        f"Exact result for experiment {experiment_dir.name}, ratio {result.solver_config.ratio} is invalid."
                    )
                    is_valid = False

            if result.pls_result is not None:
                if not result.pls_result.validate():
                    log.error(
                        f"PLS result for experiment {experiment_dir.name}, ratio {result.solver_config.ratio} is invalid."
                    )
                    is_valid = False

        log.info(f"Validated {experiment_dir.name} experiment")

    if is_valid:
        log.info(f"All experiments in {experiments_dir.name} are valid ")
    else:
        log.error(f"Some experiments in {experiments_dir.name} are invalid")

    return is_valid


def generate_metadata(
    experiments_dir: Path,
    output_dir: Path,
    instance_regex: str | None = None,
):
    if instance_regex is not None:
        experiment_dirs = [
            experiment_dir
            for experiment_dir in experiments_dir.glob(instance_regex)
            if experiment_dir.is_dir()
        ]
        log.info(f'Selected instances matching regex "{instance_regex}": {experiments_dir}')
    else:
        experiment_dirs = [experiment_dir for experiment_dir in experiments_dir.iterdir()]

    output_dir.mkdir(parents=True, exist_ok=True)

    for experiment_dir in experiment_dirs:
        log.info(f"Processing metadata for instance: {experiment_dir.stem}")
        try:
            experiment = Experiment.from_dir(experiment_dir)
            experiment_results_series = experiment.parse_results_series(experiment_dir, 10)
            all_metadata_df = experiment_results_series.generate_metadata_table()
        except Exception as e:
            log.error(f"Failed to parse experiment {experiment_dir.name}. Reason: {e}")
            continue
        plot_metadata_stats = all_metadata_df[
            ["ratio", "iteration", "final_hypervolume", "total_solutions_count"]
        ]
        # Escape % character in the ratio column
        plot_metadata_stats["ratio"] = plot_metadata_stats["ratio"].str.replace("%", r"\%")
        plot_metadata_stats.to_csv(
            output_dir / f"{experiment._problem_instance.name}_all_metadata.csv",
            index=True,
            quoting=csv.QUOTE_NONNUMERIC,
            index_label="Index",
        )


def _wilcoxon_test(
    results_left: np.ndarray,
    results_right: np.ndarray,
    alternative: str = "greater",
    alpha: float = 0.05,
) -> tuple[bool, float, float]:
    stat, p_value = wilcoxon(results_left, results_right, alternative=alternative)
    log.debug(f"Wilcoxon test statistic: {stat:.7f}")
    log.debug(f"P-value: {p_value:.7f}")

    is_statistically_significant = p_value < alpha

    if is_statistically_significant:
        log.debug("Reject null hypothesis. Hybrid is statistically better than MILP")
    else:
        log.debug(
            "Fail to reject null hypothesis. No statistical difference between Hybrid and MILP"
        )

    return is_statistically_significant, stat, p_value


def batch_wilcoxon_test(metadata_dir: Path):
    for csv_file in metadata_dir.glob("*all_metadata.csv"):
        milp_results = []
        hybrid_results = []
        pls_results = []
        log.info(f"Processing metadata for instance: {csv_file.stem}")
        with open(csv_file, mode="r") as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        if not rows:
            log.error("Metadata file is emplty")

        for row in rows:
            match row["ratio"]:
                case "100\\% : 0\\%":
                    milp_result = float(row["final_hypervolume"])
                    milp_results.append(milp_result)
                case "50\\% : 50\\%":
                    hybrid_result = float(row["final_hypervolume"])
                    hybrid_results.append(hybrid_result)
                case "0\\% : 100\\%":
                    pls_result = float(row["final_hypervolume"])
                    pls_results.append(pls_result)

        if not milp_results or not hybrid_results or not pls_results:
            log.error("No records found in the metadata file for given ratios")
            continue

        log.debug("MILP results:" + str(milp_results))
        log.debug("Hybrid results:" + str(hybrid_results))
        log.debug("PLS results:" + str(pls_results))
        milp_results = np.array(milp_results)
        hybrid_results = np.array(hybrid_results)
        pls_results = np.array(pls_results)

        alpha = 0.05

        log.info(f"Hybrid results: {hybrid_results}")
        log.info(f"PLS results:    {pls_results}")

        log.debug("Wilcoxon test for Hybrid and PLS")
        is_statistically_significant, stat, hybrid_pls_p_value = _wilcoxon_test(
            hybrid_results, pls_results, alternative="greater", alpha=alpha
        )
        log.info(
            f"Is Hybrid statistically better then PLS  (alpha == {alpha}): {is_statistically_significant} [P-value: {hybrid_pls_p_value:.7f}, Wilcoxon Statistic: {stat:.7f}]"
        )

        log.info(f"Hybrid results: {hybrid_results}")
        log.info(f"MILP results:   {milp_results}")

        log.debug("Wilcoxon test for Hybrid and MILP")
        is_statistically_significant, stat, hybrid_milp_p_value = _wilcoxon_test(
            hybrid_results, milp_results, alternative="greater", alpha=alpha
        )
        log.info(
            f"Is Hybrid statistically better then MILP (alpha == {alpha}): {is_statistically_significant} [P-value: {hybrid_milp_p_value:.7f}, Wilcoxon Statistic: {stat:.7f}]"
        )


def batch_friedman_test(metadata_dir1: Path, metadata_dir2: Path):
    if "aneja" in metadata_dir1.name:
        aneja_dir = metadata_dir1
        gpba_dir = metadata_dir2
    else:
        gpba_dir = metadata_dir1
        aneja_dir = metadata_dir2
    aneja_metadata_files = set(csv_file.name for csv_file in aneja_dir.glob("*all_metadata.csv"))
    gpba_metadata_files = set(csv_file.name for csv_file in gpba_dir.glob("*all_metadata.csv"))
    metadata_files_set = aneja_metadata_files.intersection(gpba_metadata_files)
    metadata_files_set = {f for f in metadata_files_set if "300" not in f}

    gpba_100_results = pd.DataFrame()
    gpba_50_50_results = pd.DataFrame()
    aneja_100_results = pd.DataFrame()
    aneja_50_50_results = pd.DataFrame()
    pls_results = pd.DataFrame()
    print(sorted(metadata_files_set))

    for metadata_file in metadata_files_set:
        gpba_file_path = gpba_dir / metadata_file
        aneja_file_path = aneja_dir / metadata_file

        gpba_df = pd.read_csv(gpba_file_path)
        aneja_df = pd.read_csv(aneja_file_path)

        current_gpba_100_results = gpba_df[gpba_df["ratio"] == "100\\% : 0\\%"]
        current_gpba_50_50_results = gpba_df[gpba_df["ratio"] == "50\\% : 50\\%"]
        current_aneja_100_results = aneja_df[aneja_df["ratio"] == "100\\% : 0\\%"]
        current_aneja_50_50_results = aneja_df[aneja_df["ratio"] == "50\\% : 50\\%"]
        current_pls_results = gpba_df[gpba_df["ratio"] == "0\\% : 100\\%"]
        if (
            current_gpba_100_results.empty
            or current_gpba_50_50_results.empty
            or current_aneja_100_results.empty
            or current_aneja_50_50_results.empty
            or current_pls_results.empty
        ):
            log.error(f"No records found in the metadata file {metadata_file} for given ratios")
            continue

        current_gpba_100_results["instance"] = metadata_file.rstrip("_all_metadata.csv")
        current_gpba_50_50_results["instance"] = metadata_file.rstrip("_all_metadata.csv")
        current_aneja_100_results["instance"] = metadata_file.rstrip("_all_metadata.csv")
        current_aneja_50_50_results["instance"] = metadata_file.rstrip("_all_metadata.csv")
        current_pls_results["instance"] = metadata_file.rstrip("_all_metadata.csv")

        gpba_100_results = pd.concat(
            [gpba_100_results, current_gpba_100_results], ignore_index=True
        )
        gpba_50_50_results = pd.concat(
            [gpba_50_50_results, current_gpba_50_50_results], ignore_index=True
        )
        aneja_100_results = pd.concat(
            [aneja_100_results, current_aneja_100_results], ignore_index=True
        )
        aneja_50_50_results = pd.concat(
            [aneja_50_50_results, current_aneja_50_50_results], ignore_index=True
        )
        pls_results = pd.concat([pls_results, current_pls_results], ignore_index=True)

    alpha = 0.05

    log.debug("Performing Friedman test")

    combined_results_df = pd.DataFrame(
        {
            "Instance": gpba_100_results["instance"],
            "GPBA-A 100%": gpba_100_results["final_hypervolume"],
            "GPBA-A 50%:50%": gpba_50_50_results["final_hypervolume"],
            "Aneja & Nair 100%": aneja_100_results["final_hypervolume"],
            "Aneja & Nair 50%:50%": aneja_50_50_results["final_hypervolume"],
            "PLS 100%": pls_results["final_hypervolume"],
        }
    )

    combined_results_df.sort_values("Instance").to_csv(metadata_dir1.parent / "combined_df.csv", index=False)

    combined_results = [
        combined_results_df["GPBA-A 100%"],
        combined_results_df["GPBA-A 50%:50%"],
        combined_results_df["Aneja & Nair 100%"],
        combined_results_df["Aneja & Nair 50%:50%"],
        combined_results_df["PLS 100%"],
    ]

    stat, p_value = friedmanchisquare(*combined_results)

    log.debug(f"Friedman test statistic: {stat:.7f}")
    log.debug(f"P-value: {p_value:.7f}")

    is_statistically_significant = p_value < alpha

    log.info(
        f"Are methods statistically different (alpha == {alpha}): {is_statistically_significant} [P-value: {p_value}, Friedman Statistic: {stat:.7f}]"
    )

    if is_statistically_significant:
        log.debug("Reject null hypothesis. One of the algorithms is statistically better")
    else:
        log.debug("Fail to reject null hypothesis. No statistical difference between algorithms")

    log.info("Performing post-hoc Nemenyi test")
    combined_results_ndarray = np.array([np.array(r) for r in combined_results])
    nemenyi_results = sp.posthoc_nemenyi_friedman(combined_results_ndarray.T)
    print(nemenyi_results)


def print_latex_table(experiments_dir: Path, output_dir: Path, instance_regex: str | None = None):
    if instance_regex is not None:
        experiment_dirs = [
            experiment_dir
            for experiment_dir in experiments_dir.glob(instance_regex)
            if experiment_dir.is_dir()
        ]
        log.info(f'Selected instances matching regex "{instance_regex}": {experiments_dir}')
    else:
        experiment_dirs = [experiment_dir for experiment_dir in experiments_dir.iterdir()]

    output_dir.mkdir(parents=True, exist_ok=True)

    all_instances_metadata = pd.DataFrame()

    for experiment_dir in experiment_dirs:
        instance_name = experiment_dir.stem
        if instance_name.endswith("300"):
            continue
        log.info(f"Parsing metadata for instance: {instance_name}")
        try:
            experiment = Experiment.from_dir(experiment_dir)
            experiment_results_series = experiment.parse_results_series(experiment_dir, 10)
            all_metadata_df = experiment_results_series.generate_mean_metadata_table(
                humanize_labels=True
            )
        except Exception as e:
            log.error(f"Failed to parse experiment {experiment_dir.name}. Reason: {e}")
            continue
        metadata_df = all_metadata_df[["Ratio", "Final Hypervolume", "Total Solutions Count"]]
        metadata_df.set_index("Ratio", inplace=True)
        metadata_df.index = pd.MultiIndex.from_product(
            [[instance_name], metadata_df.index], names=["Instance", "Ratio"]
        )
        all_instances_metadata = pd.concat([all_instances_metadata, metadata_df])

    log.info("Generating LaTeX table for all instances")
    all_instances_metadata = all_instances_metadata.rename(
        columns={
            "Final Hypervolume": "Hypervolume",
            "Total Solutions Count": "Solutions Count",
        }
    )
    sorted_ratios = [f"{ratio}% : {100 - ratio}%" for ratio in range(100, -1, -10)]
    all_instances_metadata.sort_index().reindex(sorted_ratios, level=1)
    all_instances_metadata.to_latex(
        output_dir / f"{experiments_dir.name}_all_metadata.tex",
        longtable=True,
        multirow=True,
        multicolumn=False,
        escape=True,
        column_format="llrr",
        caption="Experiment results",
        label="tab:experiment_results",
        position="H",
        formatters={
            "Solutions Count": lambda x: f"{x:.1f}",
        },
    )


def prepare_command(args):
    if args.experiments_dir is not None:
        experiments_dir = Path(args.experiments_dir).resolve()
    if args.aois_dir is not None:
        aois_dir = Path(args.aois_dir).resolve()
    if args.satellite_images_dir is not None:
        satellite_images_dir = Path(args.satellite_images_dir).resolve()
    prepare_experiments(
        experiments_dir=experiments_dir,
        aois_dir=aois_dir,
        satellite_images_dir=satellite_images_dir,
    )


def solve_command(args):
    iter_count = args.iter_count or 1
    solve_experiments(
        experiments_dir=Path(args.experiments_dir).resolve(),
        modified_solver_config=SolverConfig(
            solver_type=args.exact_solver,
            front_strategy=args.front_strategy,
            timeout_s=args.timeout_s,
            ratio_step=args.ratio_step,
        ),
        dry_run=args.dry_run,
        iter_count=iter_count,
        instance_regex=args.instance_regex,
        skip_solved=True,
        results_dir=Path(args.results_dir).resolve() if args.results_dir is not None else None,
    )


def process_command(args):
    process_experiments_results(
        experiments_dir=Path(args.experiments_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
    )


def reports_command(args):
    generate_reports(
        experiments_dir=Path(args.experiments_dir).resolve(),
        reports_dir=Path(args.reports_dir).resolve(),
        instance_regex=args.instance_regex,
    )


def plots_command(args):
    if len(args.experiments_dir) > 2:
        log.error("Too many experiments directories provided. Only one or two are allowed.")
        sys.exit(1)

    generate_plots(
        experiments_dir=Path(args.experiments_dir[0]).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        instance_regex=args.instance_regex,
        experiments_dir2=Path(args.experiments_dir[1]).resolve()
        if len(args.experiments_dir) == 2
        else None,
    )


def validate_command(args):
    validate_solutions(
        experiments_dir=Path(args.experiments_dir).resolve(),
    )


def metadata_command(args):
    experiments_dir = Path(args.experiments_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    generate_metadata(
        experiments_dir=experiments_dir, output_dir=output_dir, instance_regex=args.instance_regex
    )


def wilcoxon_command(args):
    metadata_dir = Path(args.metadata_dir).resolve()
    batch_wilcoxon_test(metadata_dir)


def friedman_command(args):
    metadata_dir1 = Path(args.metadata_dir1).resolve()
    metadata_dir2 = Path(args.metadata_dir2).resolve()
    batch_friedman_test(metadata_dir1, metadata_dir2)


def dzn2json_command(args):
    if args.input_dir:
        input_dir = Path(args.input_dir).resolve()
        for dzn_path in input_dir.glob("**/*.dzn"):
            logging.info(f"Converting {dzn_path} to JSON")
            json_path = dzn_path.with_suffix(".json")
            problem_instance = SimsDiscreteProblem.from_dzn(dzn_path)
            json_path.write_text(json.dumps(problem_instance.to_dict()))
    else:
        dzn_path = Path(args.dzn_path).resolve()
        json_path = Path(args.json_path).resolve()

        problem_instance = SimsDiscreteProblem.from_dzn(dzn_path)
        json_path.write_text(json.dumps(problem_instance.to_dict()))


def latex_table_command(args):
    experiments_dir = Path(args.experiments_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    print_latex_table(experiments_dir, output_dir, instance_regex=args.instance_regex)


def main():
    parser = argparse.ArgumentParser(prog="serenity.py")
    subparsers = parser.add_subparsers(dest="command")

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--experiments-dir",
        type=str,
        help="Path where folder with experiments should be created",
    )
    prepare_parser.add_argument(
        "--experiment-name",
        type=str,
        help="Name of the experiment to be created",
    )
    prepare_parser.set_defaults(func=prepare_command)

    solve_parser = subparsers.add_parser("solve")
    solve_parser.add_argument(
        "--experiments-dir",
        type=str,
        help="Path to directory where processed results should be stored",
        required=True,
    )
    solve_parser.add_argument(
        "--results-dir",
        type=str,
        help="Path to directory where results should be stored",
    )
    solve_parser.add_argument("--timeout-s", type=int, help="Timeout in seconds for the solver")
    solve_parser.add_argument(
        "--ratio-step",
        type=int,
        help='Step between ratios for multi-ratio experiments. Step "20" means ratios 100%%:0%%, 80%%:20%%, 60%%:40%%, 40%%:60%%, 20%%:80%%, 0%%:100%% will be used',
    )
    solve_parser.add_argument("--exact-solver", type=str, help="Name of the exact solver to use")
    solve_parser.add_argument(
        "--front-strategy", type=str, help="Name of the pareto front strategy to use"
    )
    solve_parser.add_argument(
        "--dry-run", action="store_true", help="Do not run the solver, only show what would be done"
    )
    solve_parser.add_argument("--iter-count", type=int, help="Number of iterations for the solver")
    solve_parser.add_argument(
        "--instance-regex", type=str, help="Regex to filter instances to solve"
    )

    solve_parser.set_defaults(func=solve_command)

    process_parser = subparsers.add_parser("process")
    process_parser.add_argument(
        "--experiments-dir", type=str, help="Path to directory with experiments", required=True
    )
    process_parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to directory where processed results should be stored",
        required=True,
    )
    process_parser.set_defaults(func=process_command)

    reports_parser = subparsers.add_parser("reports")
    reports_parser.add_argument(
        "--experiments-dir", type=str, help="Path to directory with experiments", required=True
    )
    reports_parser.add_argument(
        "--reports-dir",
        type=str,
        help="Path to directory where reports should be generated",
        required=True,
    )
    reports_parser.add_argument(
        "--instance-regex", type=str, help="Regex to filter instances to generate reports"
    )
    reports_parser.set_defaults(func=reports_command)

    plots_parser = subparsers.add_parser("plots")
    plots_parser.add_argument(
        "--experiments-dir",
        type=str,
        help="Path to directory with experiments. Pass to directories to generate comparison plots",
        required=True,
        nargs="+",
    )
    plots_parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to directory where plots should be generated",
        required=True,
    )
    plots_parser.add_argument(
        "--instance-regex", type=str, help="Regex to filter instances to generate plots"
    )
    plots_parser.set_defaults(func=plots_command)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument(
        "--experiments-dir", type=str, help="Path to directory with experiments", required=True
    )
    validate_parser.set_defaults(func=validate_command)

    metadata_parser = subparsers.add_parser("metadata")
    metadata_parser.add_argument(
        "--experiments-dir", type=str, help="Path to directory with experiments", required=True
    )
    metadata_parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to directory where metadata should be generated",
        required=True,
    )
    metadata_parser.add_argument(
        "--instance-regex", type=str, help="Regex to filter instances to generate metadata"
    )
    metadata_parser.set_defaults(func=metadata_command)

    wilcoxon_parser = subparsers.add_parser("wilcoxon")
    wilcoxon_parser.add_argument(
        "--metadata-dir", type=str, help="Path to directory with metadata files", required=True
    )
    wilcoxon_parser.set_defaults(func=wilcoxon_command)

    friedman_parser = subparsers.add_parser("friedman")
    friedman_parser.add_argument(
        "--metadata-dir1", type=str, help="Path to directory with metadata files", required=True
    )
    friedman_parser.add_argument(
        "--metadata-dir2", type=str, help="Path to directory with metadata files", required=True
    )
    friedman_parser.set_defaults(func=friedman_command)

    dzn2json_parser = subparsers.add_parser("dzn2json")
    # TODO: Make this work
    # mode_group = dzn2json_parser.add_mutually_exclusive_group(required=True)
    single_file_group = dzn2json_parser.add_argument_group("Single file conversion")
    single_file_group.add_argument("--dzn-path", type=str, help="Path to DZN file")
    single_file_group.add_argument("--json-path", type=str, help="Path to JSON file")
    batch_group = dzn2json_parser.add_argument_group("Batch conversion")
    batch_group.add_argument(
        "--input-dir",
        type=str,
        help="Path to directory with DZN files. All files will be converted to JSON",
    )
    dzn2json_parser.set_defaults(func=dzn2json_command)

    latex_table_parser = subparsers.add_parser("latex-table")
    latex_table_parser.add_argument(
        "--experiments-dir",
        type=str,
        help="Path to directory with experiments",
        required=True,
    )
    latex_table_parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to directory where LaTeX table should be generated",
        required=True,
    )
    latex_table_parser.add_argument(
        "--instance-regex", type=str, help="Regex to filter instances to generate LaTeX table"
    )
    latex_table_parser.set_defaults(func=latex_table_command)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
