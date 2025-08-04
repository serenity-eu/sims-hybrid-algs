#!/usr/bin/env python3
"""
Simple test to verify that the timeout parameter is properly accepted by solve_with_milp.
"""

import sys
sys.path.insert(0, '/home/hlvlad/code/serenity/sims-hybrid-algs/sims-problem')

import sims_problem

def test_milp_timeout_parameter():
    """Test that solve_with_milp accepts timeout parameter without errors."""
    
    # Create a simple test problem
    test_problem = sims_problem.SimsDiscreteProblem(
        num_images=3,
        universe=4,
        images=[[0, 1], [1, 2], [2, 3]],
        costs=[10, 15, 20],
        clouds=[[0], [1], [2]],
        areas=[1, 1, 1, 1],
        resolution=[5, 4, 3],
        incidence_angle=[10, 15, 20],
        max_cloud_area=10
    )
    
    print("Testing MILP with timeout parameter...")
    
    try:
        # Test with a short timeout to verify parameter is accepted
        solutions = sims_problem.solve_with_milp(
            test_problem,
            objectives=["min_cost", "cloud_coverage"],
            grid_points=5,  # Small number for quick test
            timeout_seconds=30.0,  # 30 second timeout
            bypass_coefficient=True,
            early_exit=True,
            flag_array=True,
            solver_name="cbc"
        )
        
        print(f"✅ SUCCESS: MILP solved with timeout parameter. Found {len(solutions)} solutions.")
        
        # Print basic info about solutions
        if solutions:
            print("Sample solution:")
            sol = solutions[0]
            print(f"  Cost: {sol.cost}")
            print(f"  Cloudy area: {sol.cloudy_area}")
            print(f"  Selected images: {sol.get_selected_images_list()}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_milp_timeout_parameter()
    sys.exit(0 if success else 1)
