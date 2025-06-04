import logging
import subprocess
from pathlib import Path
import os

from ..problem import ProblemInstance
from ..solver_result import SolverResult
from . import _utils

log = logging.getLogger(Path(__file__).stem)


def solve(
    problem_instance: ProblemInstance,
    problem_path: Path,
    timeout_s: int,
    output_path: Path,
    initial_population_csv: Path | None = None,
):
    if "PLS_PATH" not in os.environ:
        raise EnvironmentError(
            "PLS_PATH environment variable is not set. Please set it to the path of the PLS executable."
        )

    PLS_PATH = Path(os.environ["PLS_PATH"])

    command = [
        PLS_PATH,
        "--problem",
        problem_path,
        "--output",
        output_path,
        "--timeout",
        f"{timeout_s}s",
    ]
    if initial_population_csv is not None:
        command.extend(["--initial-population", initial_population_csv])

    log.debug(f"Running command: {' '.join(map(str, command))}")
    try:
        _utils.run_command(command, log)
    except subprocess.CalledProcessError as e:
        log.error(e)
        log.error(e.stderr)
        raise e

    log.debug(f"Reading summary from {output_path}")
    return SolverResult.from_summary_csv(output_path, problem_instance, no_headers=True)
