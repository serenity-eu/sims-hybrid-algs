import csv
import logging
import os
import sys
import traceback
from datetime import datetime

from filelock import FileLock, Timeout
from minizinc import Instance, Model, Solver

from sims_solvers import constants
from sims_solvers.Config import Config
from sims_solvers.FrontGenerators.AnejaNair import AnejaNair
from sims_solvers.FrontGenerators.CoverageGridPoint import CoverageGridPoint
from sims_solvers.FrontGenerators.Gavanelli import Gavanelli
from sims_solvers.FrontGenerators.Saugmecon import Saugmecon
from sims_solvers.Instances.InstanceGeneric import InstanceGeneric
from sims_solvers.Instances.InstanceMinizinc import InstanceMinizinc
from sims_solvers.Instances.InstanceMIPMatrix import InstanceMIPMatrix
from sims_solvers.Instances.InstanceSIMS import InstanceSIMS
from sims_solvers.MOCP import MOCP
from sims_solvers.Models.GenericModel import GenericModel
from sims_solvers.Models.GurobiModels.MultiobjectiveKnapsackGurobiModel import (
    MultiobjectiveKnapsackGurobiModel,
)
from sims_solvers.Models.GurobiModels.SatelliteImageMosaicSelectionGurobiModel import (
    SatelliteImageMosaicSelectionGurobiModel,
)
from sims_solvers.Models.MinizincPseudoModel import MinizincPseudoModel
from sims_solvers.Models.OrtoolsCPModels.MultiobjectiveKnapsackOrtoolsCPModel import (
    MultiobjectiveKnapsackOrtoolsCPModel,
)
from sims_solvers.Models.OrtoolsCPModels.SatelliteImageMosaicSelectionOrtoolsCPModel import (
    SatelliteImageMosaicSelectionOrtoolsCPModel,
)
from sims_solvers.MOWithFrontGenerator import MOWithFrontGenerator
from sims_solvers.ParetoFront import ParetoFront
from sims_solvers.Solvers.GurobiSolver import GurobiSolver
from sims_solvers.Solvers.MinizincSolver import MinizincSolver
from sims_solvers.Solvers.OrtoolsCPSolver import OrtoolsCPSolver
from sims_solvers.Timer import Timer


def init_top_level_statistics(statistics: dict):
    statistics["exhaustive"] = False
    statistics["hypervolume"] = 0
    statistics["datetime"] = datetime.now()


def init_solution_details_statistics(statistics):
    statistics["number_of_solutions"] = 0
    statistics["total_nodes"] = 0
    statistics["time_solver_sec"] = 0  # Time spent in the solver.
    statistics["minizinc_time_fzn_sec"] = 0
    statistics["hypervolume_current_solutions"] = []
    statistics["solutions_time_list"] = []
    statistics["pareto_solutions_time_list"] = []
    statistics["pareto_front"] = ""
    statistics["solutions_pareto_front"] = ""
    statistics["incomplete_timeout_solution_added_to_front"] = False


def main():
    config = Config.from_args()
    # check_already_computed(config)
    instance = build_instance(config)
    print("Start computing: " + config.uid())
    statistics = {}
    config.init_statistics(statistics)
    init_top_level_statistics(statistics)
    model = build_model(instance, config)
    solver, pareto_front = build_solver(model, instance, config, statistics)
    save_results = True
    try:
        statistics["exhaustive"] = False
        statistics["incomplete_timeout_solution_added_to_front"] = False
        for x in solver.solve():
            pass
        print("Problem completely explored.")
        statistics["exhaustive"] = True
    except TimeoutError:
        print("Timeout triggered getting last incomplete solution")
        if solver.process_last_incomplete_solution():
            # the last incomplete solution was added to the pareto front
            print("Last incomplete solution added to the pareto front")
            statistics["incomplete_timeout_solution_added_to_front"] = True
        else:
            print(
                "There were not incomplete solution or the last incomplete solution was not added to "
                "the pareto front"
            )
            set_right_time_after_timeout(statistics, config.solver_timeout_sec)
    except Exception as e:
        print("Error Execption raised: " + str(e))
        logging.error(traceback.format_exc())
        save_results = False
    if save_results:
        statistics["all_solutions"] = statistics["all_solutions"][:-1]
        statistics["all_solutions"] += "}"
        statistics["hypervolume"] = pareto_front.hypervolume()
        pareto_solutions_time_list = [
            statistics["solutions_time_list"][x] for x in pareto_front.front
        ]
        statistics["pareto_solutions_time_list"] = pareto_solutions_time_list
        print("end of solving statistics: " + str(statistics))
        write_statistics(config, statistics)


def set_right_time_after_timeout(statistics, config_timeout_sec):
    statistics["time_solver_sec"] = config_timeout_sec


def build_instance(config: Config) -> InstanceGeneric:
    instance = None
    if use_minizinc_data(config):
        instance = build_instance_minizinc_data(config)
    else:
        config.initialize_cores(solver=None, check_solver=False)
        instance = build_instance_text_data(config)

    return instance


def use_minizinc_data(config: Config) -> bool:
    return config.minizinc_data


def build_instance_minizinc_data(config: Config) -> InstanceGeneric:
    # here we build the instance, it could from minizinc or from other format
    model = Model(config.input_mzn)
    model.add_file(config.input_dzn, parse_data=True)
    if config.solver_name == "ortools-py":
        mzn_solver = Solver.lookup("gurobi")
    else:
        mzn_solver = Solver.lookup(config.solver_name)
    config.initialize_cores(mzn_solver)
    instance = Instance(mzn_solver, model)
    problem_name = config.problem_name
    if config.solver_name == "gurobi" or config.solver_name == "ortools-py":
        if problem_name == constants.Problem.SATELLITE_IMAGE_SELECTION_PROBLEM.value:
            instance = InstanceSIMS(instance)
    else:
        instance = InstanceMinizinc(mzn_solver, model, problem_name)

    return instance


def build_instance_text_data(config: Config) -> InstanceMIPMatrix:
    objective_matrix = []
    constraints_matrix = []
    rhs_constraints_vector = []
    if config.problem_name == constants.Problem.MULTI_OBJECTIVE_KNAPSACK_PROBLEM.value:
        datasets_folder = os.path.join(
            config.data_sets_folder, "mokp", config.data_name
        )
        path_objective_matrix, path_constraints_matrix, path_rhs_constraints_vector = (
            mokp_get_text_data_files(datasets_folder, config.data_name)
        )
        objective_matrix = mokp_get_matrix_from_file(path_objective_matrix)
        constraints_matrix = mokp_get_matrix_from_file(path_constraints_matrix)
        rhs_constraints_vector = mokp_get_rhs_vector_from_file(
            path_rhs_constraints_vector
        )
    else:
        print("Error: problem name not recognized")
        sys.exit(1)
    instance = InstanceMIPMatrix(
        config.problem_name,
        objective_matrix,
        constraints_matrix,
        rhs_constraints_vector,
    )
    return instance


def mokp_get_text_data_files(datasets_folder, data_name):
    # data_name is in the form 3kp40, get first character and the ones after kp two characters
    kp_index = data_name.find("kp")
    number_objs = data_name[:kp_index]
    number_items = data_name[kp_index + 2 :]

    path_objective_matrix = os.path.join(
        datasets_folder, f"z{number_objs}_{number_items}ctt.txt"
    )
    path_constraints_matrix = os.path.join(
        datasets_folder, f"z{number_objs}_{number_items}att.txt"
    )
    path_rhs_constraints_vector = os.path.join(
        datasets_folder, f"z{number_objs}_{number_items}btt.txt"
    )
    return path_objective_matrix, path_constraints_matrix, path_rhs_constraints_vector


def mokp_get_matrix_from_file(file_path, file_has_headers=True):
    # Read the file and extract data
    with open(file_path, "r") as file:
        lines = file.readlines()
    if file_has_headers:
        # Remove the first row and first column headers
        rows = [line.strip().split("\t")[1:] for line in lines[1:]]
    else:
        rows = [line.strip().split("\t") for line in lines]
    rows = [[float(value) for value in row] for row in rows]
    number_of_columns = len(rows[0])
    matrix = []
    for column in range(number_of_columns):
        complete_column = [row[column] for row in rows]
        matrix.append(complete_column)
    return matrix


def mokp_get_rhs_vector_from_file(file_path):
    # Read the file and extract data
    with open(file_path, "r") as file:
        lines = file.readlines()
    # Remove the first column headers
    # rows = [line.strip().split('\t')[1:] for line in lines[1:]]
    rows = [line.strip().split("\t")[1:] for line in lines]
    rhs_vector = [float(row[0]) for row in rows]
    return rhs_vector


def check_already_computed(config):
    if os.path.exists(config.summary_filename):
        csv.field_size_limit(sys.maxsize)
        with open(config.summary_filename, "r") as fsummary:
            summary = csv.DictReader(fsummary, delimiter=";")
            for row in summary:
                if (
                    row["instance"] == config.data_name
                    and row["problem"] == config.problem_name
                    and row["solver_name"] == config.solver_name
                    and row["front_strategy"] == config.front_strategy
                    and row["solver_search_strategy"] == config.solver_search_strategy
                    and row["fzn_optimisation_level"]
                    == str(config.fzn_optimisation_level)
                    and row["cores"] == str(config.cores)
                    and row["solver_timeout_sec"] == str(config.solver_timeout_sec)
                ):
                    print(
                        f"Skipping {config.uid()} because it is already in {config.summary_filename}"
                    )
                    sys.exit(0)


def build_solver(model: GenericModel, instance: InstanceGeneric, config: Config, statistics: dict) -> tuple[MOWithFrontGenerator, ParetoFront]:
    osolve = build_osolver(model, instance, config, statistics)
    front_generator_strategy = set_front_strategy(config, osolve)
    osolve_mo = build_MO(instance, statistics, front_generator_strategy, osolve)
    return osolve_mo, osolve_mo.pareto_front


def build_model(instance: InstanceGeneric, config: Config) -> GenericModel:
    if instance.is_minizinc:
        return MinizincPseudoModel()
    problem = instance.problem_name
    if not instance.is_minizinc:
        return get_model_by_problem_and_solver_name_(
            problem, config.solver_name, instance
        )
    else:
        print(
            "Error. You're trying to build a model from a Minizinc instance. Minizinc instances already have the model"
        )
        sys.exit(1)


def get_model_by_problem_and_solver_name_(problem_name, solver_name, instance):
    if problem_name == constants.Problem.SATELLITE_IMAGE_SELECTION_PROBLEM.value:
        if solver_name == "gurobi":
            return SatelliteImageMosaicSelectionGurobiModel(instance)
        elif solver_name == "ortools-py":
            return SatelliteImageMosaicSelectionOrtoolsCPModel(instance)
    elif problem_name == constants.Problem.MULTI_OBJECTIVE_KNAPSACK_PROBLEM.value:
        if solver_name == "gurobi":
            return MultiobjectiveKnapsackGurobiModel(instance)
        elif solver_name == "ortools-py":
            return MultiobjectiveKnapsackOrtoolsCPModel(instance)


def build_osolver(model: GenericModel, instance: InstanceGeneric, config: Config, statistics: dict) -> Solver:
    free_search = config.solver_search_strategy == "free_search"
    if instance.is_minizinc:
        # todo maybe change the name to indicate that this is using MiniZinc
        # return OSolveCP(instance, statistics, config.threads, free_search,
        #                 config.fzn_optimisation_level)
        solver = MinizincSolver(
            model,
            instance,
            statistics,
            config.threads,
            free_search,
            config.fzn_optimisation_level,
        )
        model.set_solver(solver)
        return solver
    else:
        if config.solver_name == "gurobi":
            return GurobiSolver(model, statistics, config.threads, free_search)
        elif config.solver_name == "ortools-py":
            return OrtoolsCPSolver(model, statistics, config.threads, free_search)
        else:
            raise ValueError("Unknown solver name: " + config.solver_name)


def set_front_strategy(config, solver):
    if config.front_strategy == "saugmecon":
        return Saugmecon(solver, Timer(config.solver_timeout_sec))
    elif config.front_strategy == "gavanelli":
        return Gavanelli(solver, Timer(config.solver_timeout_sec))
    elif config.front_strategy == "gavanelli-opt":
        return Gavanelli(solver, Timer(config.solver_timeout_sec), optimize=True)
    elif config.front_strategy == "augmecon-coverage":
        return CoverageGridPoint(solver, Timer(config.solver_timeout_sec))
    elif config.front_strategy == "aneja-nair":
        return AnejaNair(solver, Timer(config.solver_timeout_sec))
    else:
        raise ValueError("Unknown front strategy: " + config.front_strategy)


def build_MO(instance, statistics, front_generator, osolve):
    return MOWithFrontGenerator(instance, statistics, front_generator)
    # MOCP is used to compute the pareto front with Gavanelli and Minizinc, is not used for the other solvers. It is
    # a bit faster than the current implementation of MOWithFrontGenerator when the solver is accessed through Minizinc
    if instance.is_minizinc:
        return MOCP(instance, statistics, osolve)
    else:
        return MOWithFrontGenerator(instance, statistics, front_generator)


def csv_header(config):
    statistics = {}
    config.init_statistics(statistics)
    init_top_level_statistics(statistics)
    init_solution_details_statistics(statistics)
    return list(statistics.keys())


def create_summary_file(config):
    """We create the CSV summary file if it does not exist yet.
    The header of the summary file is the list of all the statistics the combinators can collect, even if the current solving algorithm do not use all of them.
    This is to be able to compare the statistics of different algorithms.
    """
    if not os.path.exists(config.summary_filename):
        with open(config.summary_filename, "w") as summary:
            writer = csv.DictWriter(
                summary, fieldnames=csv_header(config), delimiter=";"
            )
            writer.writeheader()


def statistics_to_csv(config, statistics):
    stats_keys = csv_header(config)
    csv_entry = ""
    for k in stats_keys:
        if k in statistics:
            csv_entry += str(statistics[k])
        csv_entry += ";"
    return csv_entry[:-1] + "\n"


def write_statistics(config, statistics):
    try:
        lock = FileLock(config.summary_filename + ".lock", timeout=10)
        with lock:
            create_summary_file(config)
            with open(config.summary_filename, "a") as summary:
                summary.write(statistics_to_csv(config, statistics))
    except Timeout:
        print(
            "Could not acquire lock on summary file. Statistics will be printed on standard output instead."
        )
        print(statistics_to_csv(config, statistics))


if __name__ == "__main__":
    main()
