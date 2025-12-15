"""Generate cascading test data for GPBA phases.

Uses lagos_nigeria_30.dzn as starting point, then cascades outputs through phases.
Each phase test can run independently with hardcoded inputs from previous phase.

Usage:
    python -m sims_solvers.gpba_phases.test_data_generator
"""

from pathlib import Path
import sys


def get_lagos_nigeria_30_path():
    """Get path to lagos_nigeria_30.dzn test instance."""
    # Path from sims-solvers to sims-core/tests/data
    base = Path(__file__).parent.parent.parent.parent
    return base / "sims-core" / "tests" / "data" / "lagos_nigeria_30.dzn"


# ============================================================================
# PHASE 0: IntervalManager (standalone, no DZN needed)
# ============================================================================
def generate_interval_manager_test_data():
    """
    Generate test data for IntervalManager.
    
    This phase doesn't need lagos_nigeria_30.dzn - it's standalone.
    """
    from .interval_manager import IntervalManager
    
    print("\n" + "=" * 80)
    print("PHASE 0: INTERVAL MANAGER (standalone)")
    print("=" * 80)
    
    # Test case 1: Basic operations
    im = IntervalManager(100, 200)
    print("\n--- Test Case 1: Basic interval operations ---")
    print(f"Initial: {im}")
    
    im.remove_interval(120, 150)
    largest = im.find_largest_interval()
    
    print(f"After remove_interval(120, 150): {im}")
    print(f"Largest interval: {largest}")
    
    print("\n--- Python test code ---")
    print("im = IntervalManager(100, 200)")
    print("im.remove_interval(120, 150)")
    print(f"assert im.find_largest_interval() == {largest}")
    
    print("\n--- Rust test code ---")
    print("let mut im = IntervalManager::new(100, 200);")
    print("im.remove_interval(120, 150);")
    print(f"assert_eq!(im.find_largest_interval(), Some({largest}));")
    
    return {'largest_after_remove': largest}


# ============================================================================
# PHASE 1: Payoff Table (needs actual DZN file - expensive)
# ============================================================================
def generate_payoff_table_output():
    """
    Run payoff table on lagos_nigeria_30.dzn and print outputs for next phase.
    
    This is the ONLY test that needs to load the actual DZN file.
    All subsequent phases use hardcoded outputs from this step.
    """
    print("\n" + "=" * 80)
    print("PHASE 1: PAYOFF TABLE (loads lagos_nigeria_30.dzn)")
    print("=" * 80)
    print("\nNOTE: This phase will be implemented in Task 1.6-1.7")
    print("It requires extracting compute_payoff_table_with_gurobi() first.")
    
    # TODO: Implement after Task 1.6
    # from .payoff_table import compute_payoff_table_with_gurobi
    # from .model_builder import build_gurobi_model_from_config
    # from ..Config import Config
    # 
    # config = Config()
    # config.instance = str(get_lagos_nigeria_30_path())
    # config.objectives = ["min_cost", "cloud_coverage"]
    # config.timeout = 120
    # 
    # model, objective_exprs, problem_data = build_gurobi_model_from_config(config)
    # result = compute_payoff_table_with_gurobi(model, objective_exprs, ...)
    # 
    # print(f"PAYOFF_IDEAL_MIN = {result['ideal_min']}")
    # ... etc
    
    return None


# ============================================================================
# PHASE 2: Epsilon Setup (uses hardcoded payoff outputs - fast)
# ============================================================================
def generate_epsilon_setup_output(payoff_result):
    """
    Run epsilon setup with HARDCODED payoff outputs and print for next phase.
    
    NO DZN FILE NEEDED - uses outputs from Phase 1.
    """
    print("\n" + "=" * 80)
    print("PHASE 2: EPSILON SETUP (uses hardcoded payoff outputs)")
    print("=" * 80)
    print("\nNOTE: This phase will be implemented in Task 1.8-1.9")
    print("It requires extracting setup_epsilon_constraints() first.")
    
    # TODO: Implement after Task 1.8
    return None


# ============================================================================
# PHASE 3+: Additional phases...
# ============================================================================


def generate_all_cascading_test_data():
    """Generate all cascading test data in sequence."""
    print("=" * 80)
    print("GENERATING CASCADING TEST DATA")
    print("Each phase uses hardcoded outputs from previous phase")
    print("=" * 80)
    
    lagos_path = get_lagos_nigeria_30_path()
    if not lagos_path.exists():
        print(f"\nERROR: Test instance not found: {lagos_path}")
        print("Please ensure sims-core/tests/data/lagos_nigeria_30.dzn exists")
        sys.exit(1)
    
    print(f"\nUsing test instance: {lagos_path}")
    
    # Phase 0: IntervalManager (standalone)
    _interval_result = generate_interval_manager_test_data()
    
    # Phase 1: Payoff table (loads DZN)
    payoff_result = generate_payoff_table_output()
    
    # Phase 2: Epsilon setup (uses payoff outputs)
    if payoff_result:
        _epsilon_setup_result = generate_epsilon_setup_output(payoff_result)
    else:
        print("\n[Skipping Phase 2 - Phase 1 not yet implemented]")
    
    # Phase 3: Epsilon solve (uses epsilon setup outputs)
    # TODO: Add after Task 1.10-1.11
    
    # Phase 4: Cascading (uses epsilon solve outputs)
    # TODO: Add after Task 1.12-1.13
    
    print("\n" + "=" * 80)
    print("CASCADING TEST DATA GENERATION COMPLETE")
    print("=" * 80)
    print("\nAs more phases are extracted, this script will generate their test data.")
    print("Copy printed values into test files for hardcoded inputs.")


if __name__ == "__main__":
    generate_all_cascading_test_data()
