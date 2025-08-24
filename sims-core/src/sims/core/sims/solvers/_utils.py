import logging
import subprocess
from pathlib import Path
from subprocess import PIPE, STDOUT, CompletedProcess, Popen

import sims_solvers
from sims_solvers import Config, MZN_MODEL_PATH

from ..problem import ProblemInstance
from ..solver_config import FrontStrategy, SolverType
from ..solver_result import SolverResult

log = logging.getLogger(__name__)


def run_command(cmd, log: logging.Logger, realtime_output: bool = False):
    if realtime_output:
        process = Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True)
        with process.stdout as pipe:
            for line in iter(pipe.readline, ""):
                log.debug(line.strip())
            log.debug("Closing command's stdout pipe.")
        returncode = process.wait()
        stderr = process.stderr.read()
        completed_process = CompletedProcess(args=cmd, returncode=returncode, stderr=stderr)
    else:
        completed_process = subprocess.run(cmd, capture_output=True, text=True)

    completed_process.check_returncode()


def run_sims_solver(
    problem_instance: ProblemInstance,
    problem_path: Path,
    timeout_s: int,
    summary_path: Path,
    solver_type: SolverType,
    front_strategy: FrontStrategy,
    objectives: list[str],
):
    DZN_DIR = problem_path.parent

    if not DZN_DIR.exists():
        raise FileNotFoundError(f"DZN directory {DZN_DIR} does not exist.")

    if not problem_path.exists():
        raise FileNotFoundError(f"Problem file {problem_path} does not exist.")

    config = Config(
        minizinc_data=True,
        instance_name=problem_path.stem,
        data_sets_folder=DZN_DIR,
        input_mzn=MZN_MODEL_PATH,
        dzn_dir=DZN_DIR,
        problem_name="sims",
        solver_name=str(solver_type),
        front_strategy=str(front_strategy),
        solver_timeout_sec=timeout_s,
        summary_filename=str(summary_path),
        solver_search_strategy="free",
        fzn_optimisation_level=1,
        cores=4,
        threads=8,
        objectives=objectives,
    )

    log.debug("Running command SIMS solver.")
    try:
        sims_solvers.solve_milp(config)
    except Exception as e:
        log.error(e)
        raise e

    log.debug(f"Reading summary from {summary_path}")
    return SolverResult.from_summary_csv(summary_path, problem_instance)
