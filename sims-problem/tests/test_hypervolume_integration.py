"""
Integration tests for hypervolume functionality with real SIMS problem scenarios.

This module demonstrates how to use hypervolume computation in practical
multi-objective optimization scenarios for the SIMS problem.
"""

import pytest
import numpy as np
from sims_problem import (
    compute_hypervolume, 
    compute_hypervolume_solutions, 
    Solution,
    SimsDiscreteProblem
)

# Try to import pymoo for cross-validation
try:
    from pymoo.indicators.hv import HV
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    HV = None


def compute_pymoo_hypervolume(points, reference):
    """Helper function to compute hypervolume using pymoo for cross-validation."""
    if not PYMOO_AVAILABLE or not points or HV is None:
        return None
    
    points_array = np.array(points, dtype=np.float64)
    reference_array = np.array(reference, dtype=np.float64)
    
    hv_indicator = HV(ref_point=reference_array)
    result = hv_indicator(points_array)
    return float(result) if result is not None else None


def assert_hypervolume_matches_pymoo(points, reference, our_hv, tolerance=1e-10):
    """Assert that our hypervolume matches pymoo's computation."""
    if not PYMOO_AVAILABLE:
        return  # Skip validation if pymoo not available
    
    pymoo_hv = compute_pymoo_hypervolume(points, reference)
    if pymoo_hv is not None:
        assert abs(our_hv - pymoo_hv) < tolerance, \
            f"Hypervolume mismatch: ours={our_hv}, pymoo={pymoo_hv}, diff={abs(our_hv - pymoo_hv)}"


class TestHypervolumeIntegrationAdvanced:
    """Advanced integration tests for hypervolume in real scenarios."""
    
    def create_realistic_problem(self) -> SimsDiscreteProblem:
        """Create a realistic SIMS problem for testing."""
        return SimsDiscreteProblem(
            num_images=8,
            universe=15,
            images=[
                [0, 1, 2],      # Image 0: covers areas 0,1,2
                [2, 3, 4, 5],   # Image 1: covers areas 2,3,4,5  
                [4, 5, 6],      # Image 2: covers areas 4,5,6
                [6, 7, 8, 9],   # Image 3: covers areas 6,7,8,9
                [8, 9, 10],     # Image 4: covers areas 8,9,10
                [10, 11, 12],   # Image 5: covers areas 10,11,12
                [12, 13, 14],   # Image 6: covers areas 12,13,14
                [0, 5, 10, 14], # Image 7: sparse coverage
            ],
            costs=[100, 150, 120, 180, 140, 160, 130, 200],  # Different costs
            clouds=[
                [0],           # Image 0: cloud on area 0
                [3],           # Image 1: cloud on area 3  
                [6],           # Image 2: cloud on area 6
                [7, 8],        # Image 3: clouds on areas 7,8
                [9],           # Image 4: cloud on area 9
                [11],          # Image 5: cloud on area 11
                [13],          # Image 6: cloud on area 13
                [5, 14],       # Image 7: clouds on areas 5,14
            ],
            areas=[80, 90, 100, 110, 95, 105, 85, 120, 75, 95, 88, 92, 98, 102, 78],
            resolution=[150, 180, 160, 200, 170, 190, 140, 220],
            incidence_angle=[30, 35, 40, 25, 45, 28, 38, 42],
            max_cloud_area=400
        )
    
    def test_hypervolume_pareto_front_analysis(self):
        """Test hypervolume computation for analyzing Pareto fronts."""
        # Create several solutions representing different trade-offs
        solutions = [
            # Low cost, high cloud coverage
            Solution.create(
                selected_images=[0, 2],
                cost=220,  # 100 + 120
                cloudy_area=180,  # areas[0] + areas[6] = 80 + 85
                timestamp_us=1000000,
                max_incidence_angle=40,
                min_resolutions_sum=310  # 150 + 160
            ),
            
            # Medium cost, medium cloud coverage
            Solution.create(
                selected_images=[1, 4, 6],
                cost=430,  # 150 + 140 + 130
                cloudy_area=275,  # areas[3] + areas[9] + areas[13] = 110 + 95 + 102
                timestamp_us=2000000,
                max_incidence_angle=45,
                min_resolutions_sum=550  # 180 + 170 + 140
            ),
            
            # High cost, low cloud coverage
            Solution.create(
                selected_images=[3, 5, 7],
                cost=540,  # 180 + 160 + 200
                cloudy_area=350,  # areas[7] + areas[8] + areas[11] + areas[5] + areas[14] = 120 + 75 + 92 + 105 + 78
                timestamp_us=3000000,
                max_incidence_angle=42,
                min_resolutions_sum=610  # 200 + 190 + 220
            ),
            
            # Optimal balance solution
            Solution.create(
                selected_images=[0, 3, 6],
                cost=410,  # 100 + 180 + 130  
                cloudy_area=300,  # areas[0] + areas[7] + areas[8] + areas[13] = 80 + 120 + 75 + 102
                timestamp_us=4000000,
                max_incidence_angle=42,
                min_resolutions_sum=480  # 150 + 200 + 130
            )
        ]
        
        # Test 2D hypervolume (cost vs cloudy_area)
        reference_2d = [600, 400]
        hv_2d = compute_hypervolume_solutions(solutions, reference_2d)
        assert hv_2d > 0, f"2D hypervolume should be positive, got {hv_2d}"
        
        # Cross-validate with pymoo for 2D
        points_2d = [[s.cost, s.cloudy_area] for s in solutions]
        assert_hypervolume_matches_pymoo(points_2d, reference_2d, hv_2d)
        
        # Test 3D hypervolume (cost, cloudy_area, max_incidence_angle)
        reference_3d = [600, 400, 50]
        hv_3d = compute_hypervolume_solutions(solutions, reference_3d)
        assert hv_3d > 0, f"3D hypervolume should be positive, got {hv_3d}"
        assert hv_3d <= hv_2d * 50, f"3D HV should be reasonable extension of 2D: {hv_3d} vs {hv_2d}"
        
        # Cross-validate with pymoo for 3D
        points_3d = [[s.cost, s.cloudy_area, s.max_incidence_angle] for s in solutions]
        assert_hypervolume_matches_pymoo(points_3d, reference_3d, hv_3d)
        
        # Test 4D hypervolume (all objectives)
        reference_4d = [600, 400, 50, 700]
        hv_4d = compute_hypervolume_solutions(solutions, reference_4d)
        assert hv_4d > 0, f"4D hypervolume should be positive, got {hv_4d}"
        assert hv_4d <= hv_3d * 700, f"4D HV should be reasonable extension of 3D: {hv_4d} vs {hv_3d}"
        
        # Cross-validate with pymoo for 4D
        points_4d = [[s.cost, s.cloudy_area, s.max_incidence_angle, s.min_resolutions_sum] for s in solutions]
        assert_hypervolume_matches_pymoo(points_4d, reference_4d, hv_4d)
    
    def test_hypervolume_solution_quality_comparison(self):
        """Test using hypervolume to compare solution quality."""
        # Create clearly dominated solutions
        dominated_solution = Solution.create(
            selected_images=[0, 1, 2, 3, 4, 5, 6, 7],  # All images - very expensive
            cost=1180,  # Sum of all costs
            cloudy_area=390,  # High cloud coverage
            timestamp_us=1000000,
            max_incidence_angle=45,
            min_resolutions_sum=1440  # Sum of all resolutions
        )
        
        efficient_solution = Solution.create(
            selected_images=[0, 3],  # Minimal selection
            cost=280,  # 100 + 180
            cloudy_area=200,  # areas[0] + areas[7] + areas[8] = 80 + 120 + 75
            timestamp_us=2000000,
            max_incidence_angle=30,
            min_resolutions_sum=350  # 150 + 200
        )
        
        reference = [1200, 400, 50, 1500]
        
        # Single dominated solution
        hv_dominated = compute_hypervolume_solutions([dominated_solution], reference)
        # Cross-validate dominated solution
        points_dominated = [[dominated_solution.cost, dominated_solution.cloudy_area, 
                           dominated_solution.max_incidence_angle, dominated_solution.min_resolutions_sum]]
        assert_hypervolume_matches_pymoo(points_dominated, reference, hv_dominated)
        
        # Single efficient solution  
        hv_efficient = compute_hypervolume_solutions([efficient_solution], reference)
        # Cross-validate efficient solution
        points_efficient = [[efficient_solution.cost, efficient_solution.cloudy_area,
                           efficient_solution.max_incidence_angle, efficient_solution.min_resolutions_sum]]
        assert_hypervolume_matches_pymoo(points_efficient, reference, hv_efficient)
        
        # Combined solutions
        hv_combined = compute_hypervolume_solutions([dominated_solution, efficient_solution], reference)
        # Cross-validate combined solutions
        points_combined = [points_dominated[0], points_efficient[0]]
        assert_hypervolume_matches_pymoo(points_combined, reference, hv_combined)
        
        # The efficient solution should contribute more to hypervolume
        assert hv_efficient > 0, f"Efficient solution should have positive HV: {hv_efficient}"
        assert hv_dominated >= 0, f"Dominated solution HV should be non-negative: {hv_dominated}"
        
        # Combined HV should be at least as large as the better individual solution
        assert hv_combined >= max(hv_efficient, hv_dominated), \
            f"Combined HV {hv_combined} should be >= max individual HV {max(hv_efficient, hv_dominated)}"
    
    def test_hypervolume_convergence_tracking(self):
        """Test using hypervolume to track optimization convergence."""
        # Simulate an optimization process with improving solutions
        iteration_solutions = [
            # Iteration 1: Random initial solution
            [Solution.create(
                selected_images=[7],
                cost=200,
                cloudy_area=183,  # areas[5] + areas[14] = 105 + 78
                timestamp_us=1000000,
                max_incidence_angle=42,
                min_resolutions_sum=220
            )],
            
            # Iteration 2: Found better cost solution
            [Solution.create(
                selected_images=[0],
                cost=100,
                cloudy_area=80,  # areas[0] = 80
                timestamp_us=2000000,
                max_incidence_angle=30,
                min_resolutions_sum=150
            ),
            Solution.create(
                selected_images=[7],
                cost=200,
                cloudy_area=183,
                timestamp_us=1000000,
                max_incidence_angle=42,
                min_resolutions_sum=220
            )],
            
            # Iteration 3: Found better cloud coverage solution
            [Solution.create(
                selected_images=[0],
                cost=100,
                cloudy_area=80,
                timestamp_us=2000000,
                max_incidence_angle=30,
                min_resolutions_sum=150
            ),
            Solution.create(
                selected_images=[2],
                cost=120,
                cloudy_area=85,  # areas[6] = 85
                timestamp_us=3000000,
                max_incidence_angle=40,
                min_resolutions_sum=160
            ),
            Solution.create(
                selected_images=[7],
                cost=200,
                cloudy_area=183,
                timestamp_us=1000000,
                max_incidence_angle=42,
                min_resolutions_sum=220
            )]
        ]
        
        reference = [250, 200]
        hypervolumes = []
        
        for i, solutions in enumerate(iteration_solutions):
            hv = compute_hypervolume_solutions(solutions, reference)
            hypervolumes.append(hv)
            print(f"Iteration {i+1}: HV = {hv}, {len(solutions)} solutions")
            
            # Cross-validate each iteration with pymoo
            points_2d = [[s.cost, s.cloudy_area] for s in solutions]
            assert_hypervolume_matches_pymoo(points_2d, reference, hv)
        
        # Hypervolume should generally increase as we find better solutions
        assert all(hv >= 0 for hv in hypervolumes), "All hypervolumes should be non-negative"
        assert hypervolumes[-1] >= hypervolumes[0], \
            f"Final HV {hypervolumes[-1]} should be >= initial HV {hypervolumes[0]}"
        
        # At least one improvement should occur
        improvements = sum(1 for i in range(1, len(hypervolumes)) if hypervolumes[i] > hypervolumes[i-1])
        assert improvements > 0, f"Should see at least one HV improvement, saw {improvements}"
    
    def test_hypervolume_dimension_scaling(self):
        """Test how hypervolume behaves when adding dimensions."""
        solution = Solution.create(
            selected_images=[0, 1],
            cost=250,  # 100 + 150
            cloudy_area=190,  # areas[0] + areas[3] = 80 + 110
            timestamp_us=1000000,
            max_incidence_angle=35,
            min_resolutions_sum=330  # 150 + 180
        )
        
        # Test same solution in different dimensional spaces
        hv_2d = compute_hypervolume_solutions([solution], [300, 250])
        hv_3d = compute_hypervolume_solutions([solution], [300, 250, 40])  
        hv_4d = compute_hypervolume_solutions([solution], [300, 250, 40, 400])
        
        # Cross-validate each dimension with pymoo
        point_2d = [[solution.cost, solution.cloudy_area]]
        assert_hypervolume_matches_pymoo(point_2d, [300, 250], hv_2d)
        
        point_3d = [[solution.cost, solution.cloudy_area, solution.max_incidence_angle]]
        assert_hypervolume_matches_pymoo(point_3d, [300, 250, 40], hv_3d)
        
        point_4d = [[solution.cost, solution.cloudy_area, solution.max_incidence_angle, solution.min_resolutions_sum]]
        assert_hypervolume_matches_pymoo(point_4d, [300, 250, 40, 400], hv_4d)
        
        assert hv_2d > 0, f"2D hypervolume should be positive: {hv_2d}"
        assert hv_3d > 0, f"3D hypervolume should be positive: {hv_3d}"
        assert hv_4d > 0, f"4D hypervolume should be positive: {hv_4d}"
        
        # Higher dimensions should have larger absolute values
        # (since we're multiplying by additional volume)
        assert hv_3d >= hv_2d, f"3D HV {hv_3d} should be >= 2D HV {hv_2d}"
        assert hv_4d >= hv_3d, f"4D HV {hv_4d} should be >= 3D HV {hv_3d}"


class TestHypervolumeUtilityFunctions:
    """Test utility functions for hypervolume analysis."""
    
    def test_hypervolume_contribution_analysis(self):
        """Test analyzing individual solution contributions to hypervolume."""
        # Base solutions
        solution1 = Solution.create(
            selected_images=[0, 1],
            cost=100,
            cloudy_area=150,
            timestamp_us=1000000,
            max_incidence_angle=30,
            min_resolutions_sum=200
        )
        
        solution2 = Solution.create(
            selected_images=[2, 3],
            cost=150,
            cloudy_area=100,
            timestamp_us=2000000,
            max_incidence_angle=35,
            min_resolutions_sum=250
        )
        
        solution3 = Solution.create(
            selected_images=[4],
            cost=120,
            cloudy_area=130,
            timestamp_us=3000000,
            max_incidence_angle=25,
            min_resolutions_sum=180
        )
        
        reference = [200, 200, 40, 300]
        
        # Calculate hypervolume with all solutions
        all_solutions = [solution1, solution2, solution3]
        hv_all = compute_hypervolume_solutions(all_solutions, reference)
        
        # Cross-validate the full solution set
        points_all = [[s.cost, s.cloudy_area, s.max_incidence_angle, s.min_resolutions_sum] for s in all_solutions]
        assert_hypervolume_matches_pymoo(points_all, reference, hv_all)
        
        # Calculate hypervolume without each solution (to find contribution)
        hv_without_1 = compute_hypervolume_solutions([solution2, solution3], reference)
        hv_without_2 = compute_hypervolume_solutions([solution1, solution3], reference)
        hv_without_3 = compute_hypervolume_solutions([solution1, solution2], reference)
        
        # Cross-validate the subset computations
        points_without_1 = [points_all[1], points_all[2]]
        assert_hypervolume_matches_pymoo(points_without_1, reference, hv_without_1)
        
        points_without_2 = [points_all[0], points_all[2]]
        assert_hypervolume_matches_pymoo(points_without_2, reference, hv_without_2)
        
        points_without_3 = [points_all[0], points_all[1]]
        assert_hypervolume_matches_pymoo(points_without_3, reference, hv_without_3)
        
        # Contribution = HV(all) - HV(all without solution)
        contrib_1 = hv_all - hv_without_1
        contrib_2 = hv_all - hv_without_2
        contrib_3 = hv_all - hv_without_3
        
        assert contrib_1 >= 0, f"Solution 1 contribution should be non-negative: {contrib_1}"
        assert contrib_2 >= 0, f"Solution 2 contribution should be non-negative: {contrib_2}"
        assert contrib_3 >= 0, f"Solution 3 contribution should be non-negative: {contrib_3}"
        
        # At least one solution should contribute positively
        total_contribution = contrib_1 + contrib_2 + contrib_3
        assert total_contribution > 0, f"Total contribution should be positive: {total_contribution}"
        
        print(f"HV contributions - Sol1: {contrib_1}, Sol2: {contrib_2}, Sol3: {contrib_3}")
    
    def test_reference_point_sensitivity(self):
        """Test how hypervolume changes with different reference points."""
        solution = Solution.create(
            selected_images=[0],
            cost=100,
            cloudy_area=150,
            timestamp_us=1000000,
            max_incidence_angle=30,
            min_resolutions_sum=200
        )
        
        # Test different reference points
        ref_points = [
            [150, 200],    # Close to solution
            [200, 250],    # Moderate distance
            [300, 300],    # Far from solution
            [500, 500],    # Very far from solution
        ]
        
        hypervolumes = []
        point_2d = [[solution.cost, solution.cloudy_area]]
        
        for ref in ref_points:
            hv = compute_hypervolume_solutions([solution], ref)
            hypervolumes.append(hv)
            print(f"Reference {ref}: HV = {hv}")
            
            # Cross-validate each reference point computation
            assert_hypervolume_matches_pymoo(point_2d, ref, hv)
        
        # Hypervolume should increase with larger reference points
        for i in range(1, len(hypervolumes)):
            assert hypervolumes[i] >= hypervolumes[i-1], \
                f"HV should increase with larger reference: {hypervolumes[i]} < {hypervolumes[i-1]}"
    
    def test_empty_and_single_solution_edge_cases(self):
        """Test edge cases with empty and single solution sets."""
        solution = Solution.create(
            selected_images=[0],
            cost=100,
            cloudy_area=150,
            timestamp_us=1000000,
            max_incidence_angle=30,
            min_resolutions_sum=200
        )
        
        reference = [200, 200, 40, 300]
        
        # Empty solution set
        hv_empty = compute_hypervolume_solutions([], reference)
        assert hv_empty == 0, f"Empty set should have HV = 0, got {hv_empty}"
        # Cross-validate empty case
        assert_hypervolume_matches_pymoo([], reference, hv_empty)
        
        # Single solution
        hv_single = compute_hypervolume_solutions([solution], reference)
        assert hv_single > 0, f"Single solution should have positive HV, got {hv_single}"
        # Cross-validate single solution
        point_4d = [[solution.cost, solution.cloudy_area, solution.max_incidence_angle, solution.min_resolutions_sum]]
        assert_hypervolume_matches_pymoo(point_4d, reference, hv_single)
        
        # Two identical solutions (should be same as single)
        hv_duplicate = compute_hypervolume_solutions([solution, solution], reference)
        assert hv_duplicate == hv_single, \
            f"Duplicate solutions should have same HV as single: {hv_duplicate} != {hv_single}"
        # Cross-validate duplicate case
        points_duplicate = [point_4d[0], point_4d[0]]
        assert_hypervolume_matches_pymoo(points_duplicate, reference, hv_duplicate)


if __name__ == "__main__":
    # Run tests directly if script is executed
    pytest.main([__file__, "-v"])
