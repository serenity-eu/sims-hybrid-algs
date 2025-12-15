"""GPBA-A Algorithm Phases.

This package contains extracted phases from the GPBA-A algorithm for testing and validation.
Each phase is extracted from solve_milp_inlined() as a standalone, testable component.

Phases:
- interval_manager: IntervalManager class for adaptive exploration
- payoff_table: Compute ideal and nadir points
- epsilon_setup: Initialize epsilon-constraint method
- epsilon_adjustment: Adjust epsilon values using intervals
- relaxation_search: Check for previously explored configurations
- epsilon_solve: Solve single epsilon-constraint configuration
- cascading: Handle multi-dimensional cascading updates
- model_builder: Helper to build Gurobi models from DZN files

All phases are designed to work with real SIMS problem instances (e.g., lagos_nigeria_30.dzn).
"""

__version__ = "0.1.0"

# Exports
from .interval_manager import IntervalManager
from .model_builder import build_gurobi_model_from_config
from .epsilon_adjustment import adjust_parameter_ef_array
from .relaxation_search import search_previous_solutions_relaxation, save_solution_information
from .payoff_table import compute_payoff_table_with_gurobi
from .epsilon_setup import setup_epsilon_constraints
from .main_loop import run_gpba_loop, convert_solution_value_to_str

__all__ = [
    'IntervalManager',
    'build_gurobi_model_from_config',
    'adjust_parameter_ef_array',
    'search_previous_solutions_relaxation',
    'save_solution_information',
    'compute_payoff_table_with_gurobi',
    'setup_epsilon_constraints',
    'run_gpba_loop',
    'convert_solution_value_to_str',
]
