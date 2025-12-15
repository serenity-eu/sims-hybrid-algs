import logging
from pathlib import Path

from ..problem import ProblemInstance
from ..solver_config import FrontStrategy, SolverType
from . import _utils

log = logging.getLogger(__name__)


def solve(
    problem_instance: ProblemInstance,
    problem_path: Path,
    timeout_s: int,
    summary_path: Path,
    front_strategy: FrontStrategy,
    objectives: list[str],
    enable_trace: bool = False,
    include_dominated: bool = False,
    max_solutions_count: int | None = None,
):
    return _utils.run_sims_solver(
        problem_instance,
        problem_path,
        timeout_s,
        summary_path,
        SolverType.OR_TOOLS,
        front_strategy,
        objectives,
        enable_trace,
        include_dominated,
        max_solutions_count,
    )
