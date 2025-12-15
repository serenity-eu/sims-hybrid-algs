"""
Phase: Epsilon Adjustment

This module implements the epsilon adjustment logic for GPBA-A algorithm.
This is the core of GPBA-A that adaptively explores the largest gaps in the Pareto front.

After solving an epsilon-constraint problem, this phase updates the epsilon parameters
to focus on the largest unexplored regions of the objective space.
"""

from typing import Optional, List
from .interval_manager import IntervalManager
import logging

logger = logging.getLogger(__name__)


def adjust_parameter_ef_array(
    id_constraint_objective: int,
    ef_array: List[int],
    sol_obj_k: Optional[int],
    ef_interval: IntervalManager,
    constraint_indices: List[int],
    best_objective_values: List[int],
    nadir_objectives_values: List[int],
    gamma: int = 1
) -> Optional[IntervalManager]:
    """
    Adjust ef_array parameter based on solution found, using interval management.
    
    This is the core of GPBA-A that adaptively explores the largest gaps in the Pareto front.
    
    Args:
        id_constraint_objective: Index of the objective being constrained (0 to n_objectives-2)
        ef_array: Current epsilon values for each constrained objective (modified in-place)
        sol_obj_k: Objective value found for the constrained objective, or None if infeasible
        ef_interval: IntervalManager tracking unexplored regions for this objective
        constraint_indices: Mapping from constraint index to actual objective index
        best_objective_values: Ideal point (best value for each objective)
        nadir_objectives_values: Nadir point (worst value for each objective)
        gamma: Unused parameter (kept for API compatibility)
    
    Returns:
        Updated IntervalManager (or newly created one if exhausted)
    """
    actual_obj_index = constraint_indices[id_constraint_objective]
    logger.critical(
        f"EPS ADJUST: ENTRY - ef_array[{id_constraint_objective}]={ef_array[id_constraint_objective]}, "
        f"sol_obj_k={sol_obj_k}, interval.max={ef_interval.max_value}, "
        f"best[{actual_obj_index}]={best_objective_values[actual_obj_index]}, "
        f"nadir[{actual_obj_index}]={nadir_objectives_values[actual_obj_index]}"
    )
    
    start_removal = ef_array[id_constraint_objective]
    new_max_interval = start_removal - 1
    
    if sol_obj_k is None:  # Infeasible
        end_removal = ef_interval.max_value
    else:
        end_removal = min(sol_obj_k, ef_interval.max_value)
    
    logger.critical(
        f"EPS ADJUST: REMOVAL - start={start_removal}, end={end_removal}, "
        f"new_max={new_max_interval}"
    )
    
    # Remove explored region from interval
    if start_removal < end_removal:
        ef_interval.remove_interval(start_removal, end_removal)
    else:
        ef_interval.remove_one_point(start_removal)
        if start_removal > end_removal:
            ef_interval.remove_one_point(end_removal)
    
    # Update max_value if needed
    if end_removal >= ef_interval.max_value:
        ef_interval.max_value = new_max_interval
    
    # Find next point to explore (center of largest remaining interval)
    max_interval = ef_interval.find_largest_interval()
    actual_obj_index = constraint_indices[id_constraint_objective]
    
    logger.critical(
        f"EPS ADJUST: FIND LARGEST - max_interval={max_interval}, "
        f"intervals={ef_interval.intervals}"
    )
    
    if max_interval is not None:
        if ef_array[id_constraint_objective] == nadir_objectives_values[actual_obj_index]:
            ef_array[id_constraint_objective] = best_objective_values[actual_obj_index]
            logger.critical(f"EPS ADJUST: At nadir, jump to best: ef_array[{id_constraint_objective}]={ef_array[id_constraint_objective]}")
        else:
            # Explore center of largest gap
            ef_array[id_constraint_objective] = int((max_interval[0] + max_interval[1]) / 2)
            logger.critical(f"EPS ADJUST: Explore center of gap: ef_array[{id_constraint_objective}]={ef_array[id_constraint_objective]}")
    else:
        # Interval exhausted - set to best+1 to signal completion and trigger cascade
        # The cascade logic uses > comparison, so best+1 will trigger it
        logger.critical("EPS ADJUST: Interval exhausted! Setting to best+1 to trigger cascade...")
        ef_array[id_constraint_objective] = best_objective_values[actual_obj_index] + 1
        logger.critical(f"EPS ADJUST: Set ef_array[{id_constraint_objective}]={ef_array[id_constraint_objective]}")
        # Recreate interval - CRITICAL: Ensure min < max numerically
        min_interval = min(nadir_objectives_values[actual_obj_index], best_objective_values[actual_obj_index])
        max_interval_val = max(nadir_objectives_values[actual_obj_index], best_objective_values[actual_obj_index])
        ef_interval = IntervalManager(min_interval, max_interval_val)
        logger.critical(f"EPS ADJUST: Recreated interval: min={min_interval}, max={max_interval_val}")
    
    logger.critical(f"EPS ADJUST: EXIT - ef_array[{id_constraint_objective}]={ef_array[id_constraint_objective]}")
    return ef_interval
