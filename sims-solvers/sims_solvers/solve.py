import logging
import traceback
from importlib import resources
from . import mzn_models

from .Config import Config
from .main import (
    build_instance,
    build_model,
    build_solver,
    init_top_level_statistics,
    set_right_time_after_timeout,
    write_statistics,
)

MZN_MODEL_PATH = resources.as_file(resources.files(mzn_models) / "mozaic_cloud2.mzn")

def solve_milp(config: Config):
    """Solve the SIMS problem using Mixed Integer Linear Programming (MILP) solver."""

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
        print("Error Exception raised: " + str(e))
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
