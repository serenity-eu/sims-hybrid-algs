"""
Pseudo-solver that replays pre-recorded solutions for the exact phase and optionally runs real PLS.

This solver loads solutions from JSON files (extracted from real solver runs) for the exact phase,
then optionally runs real PLS using those solutions as initial population.
"""

import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Optional

from ..problem import ProblemInstance
from ..solver_result import Solution, SolverResult, TwoPhaseSolverResult
from ..solver_config import TwoPhaseSolverConfig, SolverType, FrontStrategy
from ..solver import solve

logger = logging.getLogger(__name__)


class PseudoSolver:
    """Pseudo-solver that replays pre-recorded solutions."""
    
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
    
    def solve_with_two_phases(
        self,
        problem_instance: ProblemInstance,
        problem_path: Path,
        solver_config: TwoPhaseSolverConfig,
        objectives: list[str],
        timeout_s: Optional[int] = None,
        run_real_pls: bool = False,
        objective_bounds: Optional[list[list[int]]] = None,
        pareto_archive: str = "nd-tree",
    ) -> TwoPhaseSolverResult:
        """
        Hybrid solver: use pre-recorded solutions for exact phase, optionally run real PLS.
        
        Args:
            problem_instance: The problem instance
            problem_path: Path to the problem file (needed for real PLS)
            solver_config: Two-phase solver configuration (includes timeout and ratio)
            objectives: List of objectives being optimized
            timeout_s: Override timeout (if None, uses solver_config.timeout_s)
            run_real_pls: If True, runs real PLS instead of replaying solutions
            objective_bounds: Objective bounds for PLS
            pareto_archive: Pareto archive implementation for PLS
        
        Returns:
            TwoPhaseSolverResult with exact solutions from replay and optionally real PLS results
        """
        instance_name = problem_instance.name
        timeout = timeout_s if timeout_s is not None else solver_config.timeout_s
        ratio = solver_config.ratio
        
        logger.info(f"[PseudoSolver] Loading solutions for {instance_name}")
        logger.info(f"[PseudoSolver] Timeout: {timeout}s, Ratio: {ratio}")
        
        # Load pre-recorded solutions
        data = self._load_solutions(instance_name)
        
        if not data["solutions"]:
            logger.warning(f"[PseudoSolver] No solutions available for {instance_name}")
            # Return empty result
            return TwoPhaseSolverResult(
                problem_instance=problem_instance,
                solver_config=solver_config,
                total_time_sec=0.0,
                exact_solver_result=None,
                pls_result=None,
            )
        
        # Calculate time allocation for each phase
        exact_time = timeout * ratio[0] / 100
        pls_time = timeout * ratio[1] / 100
        
        # Filter exact solutions by timestamp
        exact_solutions = []
        
        for sol_data in data["solutions"]:
            timestamp = sol_data.get("timestamp_s", 0)
            phase = sol_data.get("phase", "exact")
            
            # Only include exact phase solutions within the exact time window
            if phase == "exact" and timestamp <= exact_time:
                exact_solutions.append(sol_data)
        
        logger.info(f"[PseudoSolver] Found {len(exact_solutions)} exact solutions from replay")
        
        # Convert solution data to Solution objects
        def convert_solution(sol_data: dict) -> Solution:
            return Solution(
                selected_images=frozenset(sol_data["selected_images"]),
                cost=sol_data["cost"],
                cloudy_area=sol_data["cloudy_area"],
                max_incidence_angle=sol_data.get("max_incidence_angle"),
                min_resolutions_sum=sol_data.get("min_resolutions_sum"),
                timestamp_s=timedelta(seconds=sol_data["timestamp_s"])
            )
        
        # Create exact solver result from replayed solutions
        exact_pareto_front = [convert_solution(sol) for sol in exact_solutions] if exact_solutions else []
        
        exact_solver_result = None
        if ratio[0] > 0:
            # Always create exact solver result if ratio > 0, even if empty
            exact_solver_result = SolverResult(
                pareto_front=exact_pareto_front,
                timeout_sec=int(exact_time),
                execution_time_sec=exact_time,
                hypervolume=0.0,  # Not computed for pseudo-solver
                solver_type=solver_config.exact_solver_type,
                problem_instance=problem_instance,
                front_strategy=solver_config.front_strategy,
            )
        
        pls_result = None
        if ratio[1] > 0:
            if run_real_pls:
                # Run real PLS using exact solutions as initial population
                logger.info(f"[PseudoSolver] Running REAL PLS for {pls_time}s with {len(exact_pareto_front)} initial solutions")
                
                pls_result = solve(
                    solver_type=SolverType.PLS,
                    problem_instance=problem_instance,
                    problem_path=problem_path,
                    timeout_s=int(pls_time),
                    output_path=problem_path.parent / "pls_output",
                    objectives=objectives,
                    initial_population=exact_pareto_front if exact_pareto_front else None,
                    enable_trace=False,
                    enable_profiling_trace=False,
                    objective_bounds=objective_bounds,
                    include_dominated=True,
                    pareto_archive=pareto_archive,
                )
                
                logger.info(f"[PseudoSolver] Real PLS completed with {len(pls_result.pareto_front)} solutions")
            else:
                # Use replayed PLS solutions (original behavior)
                logger.info(f"[PseudoSolver] Using replayed PLS solutions (not running real PLS)")
                pls_pareto_front = []
                pls_result = SolverResult(
                    pareto_front=pls_pareto_front,
                    timeout_sec=int(pls_time),
                    execution_time_sec=pls_time,
                    hypervolume=0.0,
                    solver_type=SolverType.PLS,
                    problem_instance=problem_instance,
                )
        
        # Create two-phase result
        total_time = exact_time + pls_time
        
        return TwoPhaseSolverResult.from_results_pair(
            exact_solver_result=exact_solver_result,
            pls_result=pls_result,
            solver_config=solver_config,
            filter_invalid=False  # Assume pre-recorded solutions are valid
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