"""
Phase: Relaxation Search

This module implements the relaxation search optimization for GPBA-A algorithm.
Before solving a new epsilon-constraint configuration, check if a previously explored
configuration was less constrained (relaxed) and its solution satisfies current constraints.

This avoids redundant MILP solves by reusing previous solutions when applicable.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def search_previous_solutions_relaxation(
    ef_array: List[int],
    previous_solution_information: List[Dict[str, Any]],
    constraint_indices: List[int]
) -> Tuple[bool, Optional[Union[List[int], str]]]:
    """
    Check if this constraint configuration was already explored with relaxation.
    For maximization: ef_array1 is less constrained (more relaxed) if all ef_array1[i] >= ef_array2[i]
    
    Args:
        ef_array: Current epsilon-constraint RHS values (for constraint objectives only)
        previous_solution_information: List of previous {ef_array, solution} pairs
        constraint_indices: Indices of constraint objectives in the full objective list
    
    Returns:
        Tuple of (found: bool, solution: List[int] | "infeasible" | None)
        - If found=True and solution=list: Previous solution satisfies current constraints
        - If found=True and solution="infeasible": Previous was infeasible, current is too
        - If found=False: Must solve current configuration
    """
    for prev_sol_info in previous_solution_information:
        prev_ef_array = prev_sol_info["ef_array"]
        prev_solution = prev_sol_info["solution"]
        
        # Check if previous ef_array is less constrained (all values >= current)
        is_less_constrained = all(prev_ef_array[i] >= ef_array[i] for i in range(len(ef_array)))
        
        if is_less_constrained:
            logger.critical(f"INLINED RELAX CHECK: Current ef={ef_array}, Prev ef={prev_ef_array}, less_constrained={is_less_constrained}")
            # If previous solution is not "infeasible", check if it satisfies current constraints
            if prev_solution != "infeasible":
                # Check if previous solution satisfies current (tighter) constraints
                # For maximization: solution[constraint_idx] must be <= ef_array[i]
                # (In minimization, constraint is obj >= ef, which becomes -obj <= -ef in maximization)
                # Note: prev_solution contains ALL objectives, so we need to use constraint_indices
                constraint_vals = [prev_solution[constraint_indices[i]] for i in range(len(ef_array))]
                satisfies = all(prev_solution[constraint_indices[i]] <= ef_array[i] 
                              for i in range(len(ef_array)))
                logger.critical(f"INLINED RELAX CHECK: Prev solution constraint vals={constraint_vals}, ef_array={ef_array}, satisfies={satisfies}")
                if satisfies:
                    return True, prev_solution
            else:
                # Previous was infeasible with less constrained constraints, so current is also infeasible
                logger.critical("INLINED RELAX CHECK: Previous was infeasible, current also infeasible")
                return True, "infeasible"
    
    return False, None


def save_solution_information(
    ef_array: List[int],
    solution: Union[List[int], str],
    previous_solution_information: List[Dict[str, Any]]
) -> None:
    """
    Save solution information for future relaxation checks.
    
    Args:
        ef_array: Current epsilon-constraint RHS values
        solution: Solution found (list of objective values) or "infeasible"
        previous_solution_information: List to append to (modified in-place)
    """
    logger.critical(f"SAVE SOL INFO: Saving ef_array={ef_array}, solution={'infeasible' if solution == 'infeasible' else solution}, prev_count={len(previous_solution_information)}")
    previous_solution_information.append({
        "ef_array": ef_array.copy(),  # Copy to avoid mutation issues
        "solution": solution.copy() if isinstance(solution, list) else solution
    })
