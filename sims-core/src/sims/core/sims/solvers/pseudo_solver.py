"""
Pseudo-solver that replays pre-recorded solutions for the exact phase.

Provides a low-level building block (PseudoSolver.solve_exact) that can be
passed to solve_with_two_phases as ``exact_solver_fn``, replacing the real
MILP/CP exact solver with replayed data from JSON files.
"""

import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Optional

from ..problem import ProblemInstance
from ..solver_result import Solution, SolverResult
from ..solver_config import SolverType

logger = logging.getLogger(__name__)


class PseudoSolver:
    """
    Pseudo-solver that replays pre-recorded solutions for the exact phase only.

    This is a low-level building block: it acts as a drop-in replacement for
    the exact (MILP/CP) solver and returns a SolverResult from replayed data.
    Two-phase orchestration (time splitting, PLS seeding, hypervolume, parallel
    execution) is handled by solve_with_two_phases in solver.py, exactly as for
    real solvers.
    """

    def __init__(self, solutions_data_dir: Path):
        """
        Initialize the pseudo-solver.

        Args:
            solutions_data_dir: Directory containing JSON files with pre-recorded solutions
        """
        self.solutions_data_dir = solutions_data_dir
        self._solutions_cache = {}
    
    def _load_solutions(self, instance_name: str) -> dict:
        """Load solutions for a specific instance from JSON file."""
        if instance_name in self._solutions_cache:
            return self._solutions_cache[instance_name]
        
        solution_file = self.solutions_data_dir / f"{instance_name}.json"
        
        if not solution_file.exists():
            logger.warning(f"No pre-recorded solutions found for instance {instance_name}")
            return {"solutions": [], "objectives": [], "test_type": "unknown"}
        
        with open(solution_file, 'r') as f:
            data = json.load(f)
        
        self._solutions_cache[instance_name] = data
        return data
    
    def solve_exact(
        self,
        problem_instance: ProblemInstance,
        problem_path: Path,
        timeout_s: int,
        objectives: list[str],
        **_kwargs,
    ) -> SolverResult:
        """
        Replay pre-recorded exact-phase solutions as a SolverResult.

        This method has the same role as ortools.solve / gurobi.solve and is
        intended to be passed as ``exact_solver_fn`` to solve_with_two_phases.
        Two-phase orchestration (PLS seeding, hypervolume, parallel flag, etc.)
        stays in solve_with_two_phases.

        Args:
            problem_instance: The problem instance
            problem_path: Path to the problem file (unused, kept for API symmetry)
            timeout_s: Time budget for the exact phase
            objectives: List of objectives being optimized

        Returns:
            SolverResult whose pareto_front contains the replayed exact solutions
        """
        instance_name = problem_instance.name
        logger.info(f"[PseudoSolver] Loading solutions for {instance_name}")
        logger.info(f"[PseudoSolver] Timeout: {timeout_s}s")

        data = self._load_solutions(instance_name)

        exact_solutions = [
            sol_data
            for sol_data in data.get("solutions", [])
            if sol_data.get("phase", "exact") == "exact"
            and sol_data.get("timestamp_s", 0) <= timeout_s
        ]

        logger.info(f"[PseudoSolver] Found {len(exact_solutions)} exact solutions from replay")

        def _convert(sol_data: dict) -> Solution:
            return Solution(
                selected_images=frozenset(sol_data["selected_images"]),
                cost=sol_data["cost"],
                cloudy_area=sol_data["cloudy_area"],
                max_incidence_angle=sol_data.get("max_incidence_angle"),
                min_resolutions_sum=sol_data.get("min_resolutions_sum"),
                timestamp_s=timedelta(seconds=sol_data["timestamp_s"]),
            )

        pareto_front = [_convert(s) for s in exact_solutions]

        return SolverResult(
            pareto_front=pareto_front,
            timeout_sec=timeout_s,
            execution_time_sec=float(timeout_s),
            hypervolume=0.0,
            solver_type=SolverType.OR_TOOLS,
            problem_instance=problem_instance,
        )


# Global pseudo-solver instance
_pseudo_solver_instance: Optional[PseudoSolver] = None


def get_pseudo_solver(solutions_data_dir: Optional[Path] = None) -> PseudoSolver:
    """
    Get or create the global pseudo-solver instance.
    
    Args:
        solutions_data_dir: Directory with solution data (if None, uses default location)
    
    Returns:
        PseudoSolver instance
    """
    global _pseudo_solver_instance
    
    if _pseudo_solver_instance is None:
        if solutions_data_dir is None:
            # Default location: tests/data/pseudo_solver_solutions
            # Navigate from sims-core/src/sims/core/sims/solvers/ to sims-core/tests/
            sims_core_dir = Path(__file__).parent.parent.parent.parent.parent.parent
            solutions_data_dir = sims_core_dir / "tests" / "data" / "pseudo_solver_solutions"
        
        _pseudo_solver_instance = PseudoSolver(solutions_data_dir)
    
    return _pseudo_solver_instance