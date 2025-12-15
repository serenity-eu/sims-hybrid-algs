"""
Phase: Epsilon-Constraint Setup

This module implements the epsilon-constraint method setup phase of GPBA-A algorithm.
It takes the payoff table outputs (ideal and nadir points) and initializes the
epsilon-constraint formulation with augmented objective and interval managers.

This phase does NOT require solving the model - it only sets up structures and constraints.
"""

from typing import List, Tuple, Any
import gurobipy as gp
import logging
from .interval_manager import IntervalManager

logger = logging.getLogger(__name__)


def setup_epsilon_constraints(
    model: gp.Model,
    objectives_exprs: List[gp.LinExpr],
    best_objective_values: List[int],
    nadir_objectives_values: List[int],
    num_objectives: int,
    config: Any
) -> Tuple[List[int], List[IntervalManager], List[int], List[gp.Var], int]:
    """
    Setup epsilon-constraint formulation for GPBA-A algorithm.
    
    This phase:
    - Selects main objective (first one) and constraint objectives (rest)
    - Creates slack variables for augmented objective method
    - Sets augmented objective on the model
    - Initializes ef_array to nadir values (starting point)
    - Creates interval managers for adaptive exploration
    
    NOTE: Does NOT create epsilon-constraints - that's done in main loop!
    
    Returns:
        Tuple containing:
        - ef_array: Initial epsilon-constraint RHS values (starts at nadir)
        - ef_intervals: List of IntervalManager objects for each constraint objective
        - constraint_indices: Indices of constraint objectives
        - slack_vars: Slack variables for augmented constraint method (or None)
        - main_obj_index: Index of the main objective (0)
    """
    if num_objectives <= 1:
        raise ValueError("Epsilon-constraint setup requires at least 2 objectives")
    
    logger.critical("INLINED: STAGE 2 - Setting up ε-constraint formulation")
    main_obj_index = 0  # Use first objective as main
    constraint_indices = [i for i in range(num_objectives) if i != main_obj_index]
    logger.critical(f"INLINED: Main objective: {main_obj_index} ({config.objectives[main_obj_index]})")
    logger.critical(f"INLINED: Constraint objectives: {constraint_indices}")
    
    # Convert to maximization for main objective
    logger.critical(f"INLINED: Setting objective before aug: {objectives_exprs[main_obj_index]}")
    model.setObjective(objectives_exprs[main_obj_index], gp.GRB.MAXIMIZE)
    
    # Create slack variables for augmentation
    delta = 0.01
    slack_vars = []
    for i, constraint_idx in enumerate(constraint_indices):
        max_s = abs(best_objective_values[constraint_idx] - nadir_objectives_values[constraint_idx])
        if max_s > 0:
            s = model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=max_s, name=f"s{constraint_idx+1}")
            slack_vars.append(s)
        else:
            slack_vars.append(None)
    
    # Setup augmented objective
    obj_ranges = [abs(best_objective_values[i] - nadir_objectives_values[i]) for i in constraint_indices]
    slack_terms = []
    for i in range(len(constraint_indices)):
        if obj_ranges[i] > 0 and slack_vars[i] is not None:
            slack_terms.append(slack_vars[i] / (10**i * obj_ranges[i]))
    
    if slack_terms:
        slack_sum = gp.quicksum(slack_terms)
        # objectives_exprs are already in maximization form (negated)
        augmented_obj = objectives_exprs[main_obj_index] + delta * slack_sum
    else:
        augmented_obj = objectives_exprs[main_obj_index]
    
    model.setObjective(augmented_obj, gp.GRB.MAXIMIZE)
    
    # Initialize ef_array to nadir values (starting point for exploration)
    ef_array = [nadir_objectives_values[i] for i in constraint_indices]
    logger.critical(f"INLINED: Initial ef_array: {ef_array}")
    
    # Initialize interval managers for each constraint objective
    ef_intervals = []
    for constraint_idx in constraint_indices:
        # CRITICAL: Ensure min < max numerically for interval logic to work
        # For maximization (negative values), nadir is more negative (smaller) than best
        min_interval = min(nadir_objectives_values[constraint_idx], best_objective_values[constraint_idx])
        max_interval = max(nadir_objectives_values[constraint_idx], best_objective_values[constraint_idx])
        
        interval = IntervalManager(
            min_value=min_interval,
            max_value=max_interval
        )
        ef_intervals.append(interval)
    
    logger.critical(f"INLINED: Best values: {best_objective_values}")
    logger.critical(f"INLINED: Nadir values: {nadir_objectives_values}")
    logger.critical(f"INLINED: Constraint indices: {constraint_indices}")
    
    return (
        ef_array,
        ef_intervals,
        constraint_indices,
        slack_vars,
        main_obj_index
    )
