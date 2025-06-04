import logging
from pathlib import Path

from ..problem import ProblemInstance
from ..solver_config import FrontStrategy, SolverType
from . import _utils

log = logging.getLogger(Path(__file__).stem)


def solve(
    problem_instance: ProblemInstance,
    problem_path: Path,
    timeout_s: int,
    summary_path: Path,
    front_strategy: FrontStrategy,
):
    return _utils.run_sims_solver(
        problem_instance,
        problem_path,
        timeout_s,
        summary_path,
        SolverType.GUROBI,
        front_strategy,
    )
