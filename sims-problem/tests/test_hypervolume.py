"""
Comprehensive test suite for hypervolume functionality in sims-problem.

This module tests:
1. Core hypervolume computation algorithms (2D, 3D, 4D)
2. Python bindings for raw points and Solution objects
3. Edge cases and error handling
4. Cross-validation with pymoo library
5. Performance characteristics
6. Integration with real SIMS problem data
"""

import pytest
import numpy as np
from typing import List, Tuple
import time
from sims_problem import (
    compute_hypervolume,
    Solution,
    SimsDiscreteProblem
)

# Try to import pymoo for cross-validation, but make it optional
try:
    from pymoo.indicators.hv import HV
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    HV = None


class TestHypervolumeCore:
    """Test core hypervolume computation functionality."""
    
    def test_2d_hypervolume_basic(self):
        """Test basic 2D hypervolume computation."""
        points = [[1, 2], [2, 1]]
        reference = [3, 3]
        hv = compute_hypervolume(points, reference)
        assert hv == 3, f"Expected hypervolume 3, got {hv}"
    
    def test_3d_hypervolume_basic(self):
        """Test basic 3D hypervolume computation."""
        points = [[1, 2, 3], [2, 1, 3]]
        reference = [4, 4, 4]
        hv = compute_hypervolume(points, reference)
        assert hv == 8, f"Expected hypervolume 8, got {hv}"
    
    def test_4d_hypervolume_basic(self):
        """Test basic 4D hypervolume computation."""
        points = [[1, 2, 3, 4]]
        reference = [5, 6, 7, 8]
        hv = compute_hypervolume(points, reference)
        assert hv == 256, f"Expected hypervolume 256, got {hv}"
    
    def test_empty_front(self):
        """Test hypervolume with empty Pareto front."""
        points = []
        reference = [5, 5, 5, 5]
        hv = compute_hypervolume(points, reference)
        assert hv == 0, f"Expected hypervolume 0 for empty front, got {hv}"
    
    def test_single_point(self):
        """Test hypervolume with single point in different dimensions."""
        # 2D
        hv_2d = compute_hypervolume([[1, 2]], [3, 4])
        assert hv_2d == 4, f"2D single point: expected 4, got {hv_2d}"
        
        # 3D  
        hv_3d = compute_hypervolume([[1, 2, 3]], [4, 5, 6])
        assert hv_3d == 27, f"3D single point: expected 27, got {hv_3d}"
        
        # 4D
        hv_4d = compute_hypervolume([[1, 2, 3, 4]], [5, 6, 7, 8])
        assert hv_4d == 256, f"4D single point: expected 256, got {hv_4d}"
    
    def test_identical_points(self):
        """Test hypervolume with identical points (should handle duplicates)."""
        points = [[2, 2], [2, 2], [2, 2]]
        reference = [3, 3]
        hv = compute_hypervolume(points, reference)
        assert hv == 1, f"Expected hypervolume 1 for identical points, got {hv}"
    
    def test_points_on_reference(self):
        """Test hypervolume when points are on the reference boundary."""
        points = [[3, 3], [2, 3], [3, 2]]
        reference = [3, 3]
        hv = compute_hypervolume(points, reference)
        assert hv == 0, f"Expected hypervolume 0 for points on reference, got {hv}"
    
    def test_points_outside_reference(self):
        """Test hypervolume when points are outside reference bounds."""
        points = [[4, 4], [5, 5]]
        reference = [3, 3]
        hv = compute_hypervolume(points, reference)
        assert hv == 0, f"Expected hypervolume 0 for points outside reference, got {hv}"
    
    def test_dominated_points(self):
        """Test that dominated points don't affect hypervolume."""
        # Point [2, 2] dominates [3, 3] in minimization
        points_with_dominated = [[1, 2], [2, 1], [3, 3]]
        points_pareto_only = [[1, 2], [2, 1]]
        reference = [4, 4]
        
        hv_with_dominated = compute_hypervolume(points_with_dominated, reference)
        hv_pareto_only = compute_hypervolume(points_pareto_only, reference)
        
        assert hv_with_dominated == hv_pareto_only, \
            f"Dominated points should not affect hypervolume: {hv_with_dominated} != {hv_pareto_only}"


class TestHypervolumeDimensionality:
    """Test hypervolume computation across different dimensions."""
    
    def test_2d_multiple_points(self):
        """Test 2D hypervolume with multiple non-dominated points."""
        points = [[1, 4], [2, 3], [3, 2], [4, 1]]
        reference = [5, 5]
        hv = compute_hypervolume(points, reference)
        assert hv > 0, f"2D multiple points should have positive hypervolume, got {hv}"
    
    def test_3d_multiple_points(self):
        """Test 3D hypervolume with multiple non-dominated points."""
        points = [
            [1, 3, 3],
            [2, 2, 3], 
            [3, 1, 3],
            [2, 2, 2]
        ]
        reference = [4, 4, 4]
        hv = compute_hypervolume(points, reference)
        assert hv > 0, f"3D multiple points should have positive hypervolume, got {hv}"
    
    def test_4d_multiple_points(self):
        """Test 4D hypervolume with multiple non-dominated points."""
        points = [
            [1, 2, 3, 4],
            [2, 1, 3, 4],
            [1, 1, 2, 3],
        ]
        reference = [5, 5, 5, 5]
        hv = compute_hypervolume(points, reference)
        assert hv > 0, f"4D multiple points should have positive hypervolume, got {hv}"
    
    def test_dimension_consistency(self):
        """Test that point and reference dimensions must match."""
        with pytest.raises(Exception):
            # Mismatched dimensions should raise an error
            compute_hypervolume([[1, 2]], [3, 4, 5])
        
        with pytest.raises(Exception):
            # Inconsistent point dimensions should raise an error
            compute_hypervolume([[1, 2], [3, 4, 5]], [6, 7, 8])


class TestHypervolumeSolutions:
    """Test hypervolume computation with Solution objects."""
    
    def create_test_solution(self, cost: int, cloudy_area: int, 
                           max_incidence_angle: int = 30, 
                           min_resolutions_sum: int = 150) -> Solution:
        """Helper to create a test solution."""
        return Solution.create(
            selected_images=[0, 1, 2],
            cost=cost,
            cloudy_area=cloudy_area,
            timestamp_us=1000000,
            max_incidence_angle=max_incidence_angle,
            min_resolutions_sum=min_resolutions_sum
        )
    
    def test_solution_2d_hypervolume(self):
        """Test 2D hypervolume with Solution objects (cost, cloudy_area)."""
        solutions = [
            self.create_test_solution(100, 200),
            self.create_test_solution(150, 100),
            self.create_test_solution(120, 180)
        ]
        
        reference = [200, 300]
        hv = compute_hypervolume(solutions, reference)
        assert hv > 0, f"Solution 2D hypervolume should be positive, got {hv}"
        
        # Verify consistency with raw points
        points = [[s.cost, s.cloudy_area] for s in solutions]
        hv_raw = compute_hypervolume(points, reference)
        assert hv == hv_raw, f"Solution and raw point hypervolumes should match: {hv} != {hv_raw}"
    
    def test_solution_3d_hypervolume(self):
        """Test 3D hypervolume with Solution objects (cost, cloudy_area, max_incidence_angle)."""
        solutions = [
            self.create_test_solution(100, 200, 30),
            self.create_test_solution(150, 100, 25),
            self.create_test_solution(120, 180, 35)
        ]
        
        reference = [200, 300, 40]
        hv = compute_hypervolume(solutions, reference)
        assert hv > 0, f"Solution 3D hypervolume should be positive, got {hv}"
    
    def test_solution_4d_hypervolume(self):
        """Test 4D hypervolume with Solution objects (cost, cloudy_area, max_incidence_angle, min_resolutions_sum)."""
        solutions = [
            self.create_test_solution(100, 200, 30, 150),
            self.create_test_solution(150, 100, 25, 200),
            self.create_test_solution(120, 180, 35, 180)
        ]
        
        reference = [200, 300, 40, 250]
        hv = compute_hypervolume(solutions, reference)
        assert hv > 0, f"Solution 4D hypervolume should be positive, got {hv}"
    
    def test_solution_empty_list(self):
        """Test hypervolume with empty solution list."""
        solutions = []
        reference = [200, 300]
        hv = compute_hypervolume(solutions, reference)
        assert hv == 0, f"Empty solution list should have hypervolume 0, got {hv}"
    
    def test_solution_single(self):
        """Test hypervolume with single solution."""
        solution = self.create_test_solution(100, 200)
        solutions = [solution]
        reference = [200, 300]
        hv = compute_hypervolume(solutions, reference)
        assert hv == 10000, f"Single solution hypervolume should be 10000, got {hv}"


@pytest.mark.skipif(not PYMOO_AVAILABLE, reason="pymoo not available")
class TestHypervolumeCrossValidation:
    """Cross-validate hypervolume results with pymoo library."""
    
    def compute_pymoo_hypervolume(self, points: List[List[int]], reference: List[int]) -> float:
        """Compute hypervolume using pymoo for comparison."""
        if not points or HV is None:
            return 0.0
        
        points_array = np.array(points, dtype=np.float64)
        reference_array = np.array(reference, dtype=np.float64)
        
        hv_indicator = HV(ref_point=reference_array)
        result = hv_indicator(points_array)
        return float(result) if result is not None else 0.0
    
    def test_2d_cross_validation(self):
        """Cross-validate 2D hypervolume with pymoo."""
        points = [[1, 2], [2, 1], [1, 1]]
        reference = [3, 3]
        
        our_hv = compute_hypervolume(points, reference)
        pymoo_hv = self.compute_pymoo_hypervolume(points, reference)
        
        assert abs(our_hv - pymoo_hv) < 1e-10, \
            f"2D hypervolume mismatch: ours={our_hv}, pymoo={pymoo_hv}"
    
    def test_3d_cross_validation(self):
        """Cross-validate 3D hypervolume with pymoo."""
        points = [[1, 2, 3], [2, 1, 3], [1, 1, 2]]
        reference = [4, 4, 4]
        
        our_hv = compute_hypervolume(points, reference)
        pymoo_hv = self.compute_pymoo_hypervolume(points, reference)
        
        assert abs(our_hv - pymoo_hv) < 1e-10, \
            f"3D hypervolume mismatch: ours={our_hv}, pymoo={pymoo_hv}"
    
    def test_4d_cross_validation(self):
        """Cross-validate 4D hypervolume with pymoo."""
        points = [[1, 2, 3, 4], [2, 1, 3, 4], [1, 1, 2, 3]]
        reference = [5, 5, 5, 5]
        
        our_hv = compute_hypervolume(points, reference)
        pymoo_hv = self.compute_pymoo_hypervolume(points, reference)
        
        assert abs(our_hv - pymoo_hv) < 1e-10, \
            f"4D hypervolume mismatch: ours={our_hv}, pymoo={pymoo_hv}"


class TestHypervolumeEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_invalid_dimensions(self):
        """Test error handling for invalid dimensions."""
        # Empty points should return 0, not raise an error
        hv = compute_hypervolume([], [1, 2])
        assert hv == 0, f"Empty points should give hypervolume 0, got {hv}"
        
        # Mismatched dimensions should raise an error
        with pytest.raises(Exception):
            compute_hypervolume([[1, 2]], [3, 4, 5])
    
    def test_unsupported_dimensions(self):
        """Test error handling for unsupported dimensions (>4D)."""
        points = [[1, 2, 3, 4, 5]]  # 5D point
        reference = [6, 7, 8, 9, 10]
        
        with pytest.raises(Exception):
            compute_hypervolume(points, reference)
    
    def test_1d_dimension(self):
        """Test error handling for 1D (unsupported)."""
        points = [[1], [2], [3]]
        reference = [4]
        
        with pytest.raises(Exception):
            compute_hypervolume(points, reference)
    
    def test_negative_coordinates(self):
        """Test handling of negative coordinates."""
        # The current implementation doesn't support negative coordinates
        # because it uses u64 internally. This test verifies the error handling.
        points = [[-1, -2], [-2, -1]]
        reference = [0, 0]
        
        with pytest.raises(TypeError):
            compute_hypervolume(points, reference)
    
    def test_large_coordinates(self):
        """Test handling of large coordinate values."""
        points = [[1000000, 2000000], [2000000, 1000000]]
        reference = [3000000, 3000000]
        hv = compute_hypervolume(points, reference)
        assert hv > 0, f"Large coordinates should work, got hypervolume {hv}"
    
    def test_reference_point_validation(self):
        """Test that reference point must dominate all points."""
        points = [[1, 2], [2, 1]]
        reference = [1, 1]  # Reference doesn't dominate points
        hv = compute_hypervolume(points, reference)
        assert hv == 0, f"Reference that doesn't dominate should give 0 hypervolume, got {hv}"


@pytest.mark.slow
class TestHypervolumePerformance:
    """Test performance characteristics of hypervolume computation."""
    
    def generate_random_front(self, n_points: int, dimension: int, 
                             max_coord: int = 1000) -> Tuple[List[List[int]], List[int]]:
        """Generate a random Pareto front for testing."""
        np.random.seed(42)  # For reproducible tests
        
        points = []
        for _ in range(n_points):
            point = np.random.randint(1, max_coord, dimension).tolist()
            points.append(point)
        
        # Generate reference point that dominates all points
        reference = [max_coord + 100] * dimension
        
        return points, reference
    
    def test_2d_performance_scaling(self):
        """Test 2D hypervolume performance with increasing point count."""
        dimensions = 2
        point_counts = [10, 50, 100, 200]
        times = []
        
        for n_points in point_counts:
            points, reference = self.generate_random_front(n_points, dimensions)
            
            start_time = time.time()
            hv = compute_hypervolume(points, reference)
            end_time = time.time()
            
            times.append(end_time - start_time)
            assert hv >= 0, f"Hypervolume should be non-negative for {n_points} points"
        
        # Performance should be reasonable (less than 1 second for 200 points in 2D)
        assert times[-1] < 1.0, f"2D performance too slow: {times[-1]:.3f}s for {point_counts[-1]} points"
    
    def test_3d_performance_scaling(self):
        """Test 3D hypervolume performance with increasing point count."""
        dimensions = 3
        point_counts = [10, 25, 50, 100]
        times = []
        
        for n_points in point_counts:
            points, reference = self.generate_random_front(n_points, dimensions)
            
            start_time = time.time()
            hv = compute_hypervolume(points, reference)
            end_time = time.time()
            
            times.append(end_time - start_time)
            assert hv >= 0, f"Hypervolume should be non-negative for {n_points} points"
        
        # 3D should be slower but still reasonable
        assert times[-1] < 5.0, f"3D performance too slow: {times[-1]:.3f}s for {point_counts[-1]} points"
    
    def test_4d_performance_scaling(self):
        """Test 4D hypervolume performance with increasing point count."""
        dimensions = 4
        point_counts = [5, 10, 20, 30]  # Smaller counts for 4D
        times = []
        
        for n_points in point_counts:
            points, reference = self.generate_random_front(n_points, dimensions)
            
            start_time = time.time()
            hv = compute_hypervolume(points, reference)
            end_time = time.time()
            
            times.append(end_time - start_time)
            assert hv >= 0, f"Hypervolume should be non-negative for {n_points} points"
        
        # 4D will be slowest but should still complete
        assert times[-1] < 10.0, f"4D performance too slow: {times[-1]:.3f}s for {point_counts[-1]} points"
    
    def test_solution_performance(self):
        """Test performance of Solution-based hypervolume computation."""
        n_solutions = 50
        
        solutions = []
        for i in range(n_solutions):
            solution = Solution.create(
                selected_images=[0, 1, 2],
                cost=100 + i * 10,
                cloudy_area=200 - i * 3,
                timestamp_us=1000000 + i * 1000,
                max_incidence_angle=30 + i,
                min_resolutions_sum=150 + i * 2
            )
            solutions.append(solution)
        
        reference = [1000, 300, 100, 300]
        
        start_time = time.time()
        hv = compute_hypervolume(solutions, reference)
        end_time = time.time()
        
        computation_time = end_time - start_time
        assert hv >= 0, f"Solution hypervolume should be non-negative, got {hv}"
        assert computation_time < 1.0, f"Solution performance too slow: {computation_time:.3f}s for {n_solutions} solutions"


class TestHypervolumeIntegration:
    """Test hypervolume integration with SIMS problem solving."""
    
    def create_test_problem(self) -> SimsDiscreteProblem:
        """Create a test SIMS problem."""
        return SimsDiscreteProblem(
            num_images=5,
            universe=10,
            images=[[0, 1], [1, 2, 3], [3, 4], [5, 6, 7], [7, 8, 9]],
            costs=[10, 20, 15, 25, 30],
            clouds=[[0], [2], [4], [5], [8]],
            areas=[100, 150, 200, 120, 80, 90, 110, 170, 130, 160],
            resolution=[100, 200, 150, 180, 220],
            incidence_angle=[45, 30, 60, 35, 50],
            max_cloud_area=500
        )
    
    def test_hypervolume_with_problem_solutions(self):
        """Test hypervolume computation with actual SIMS problem solutions."""
        # Create some feasible solutions manually
        solutions = [
            Solution.create(
                selected_images=[0, 2],
                cost=25,  # 10 + 15
                cloudy_area=100,  # areas[0] = 100 (cloud coverage)
                timestamp_us=1000000,
                max_incidence_angle=60,
                min_resolutions_sum=250
            ),
            Solution.create(
                selected_images=[1, 3],
                cost=45,  # 20 + 25
                cloudy_area=270,  # areas[2] + areas[5] = 200 + 90
                timestamp_us=2000000,
                max_incidence_angle=35,
                min_resolutions_sum=380
            ),
            Solution.create(
                selected_images=[4],
                cost=30,
                cloudy_area=130,  # areas[8] = 130
                timestamp_us=3000000,
                max_incidence_angle=50,
                min_resolutions_sum=220
            )
        ]
        
        # Test 2D hypervolume (cost vs cloudy_area trade-off)
        reference_2d = [100, 400]  # Reference point for cost and cloudy area
        hv_2d = compute_hypervolume(solutions, reference_2d)
        assert hv_2d > 0, f"Problem-based 2D hypervolume should be positive, got {hv_2d}"
        
        # Test 3D hypervolume 
        reference_3d = [100, 400, 70]
        hv_3d = compute_hypervolume(solutions, reference_3d)
        assert hv_3d > 0, f"Problem-based 3D hypervolume should be positive, got {hv_3d}"
        
        # Test 4D hypervolume
        reference_4d = [100, 400, 70, 500]
        hv_4d = compute_hypervolume(solutions, reference_4d)
        assert hv_4d > 0, f"Problem-based 4D hypervolume should be positive, got {hv_4d}"
    
    def test_hypervolume_comparison(self):
        """Test that better solutions have higher hypervolume contribution."""
        # Create a clearly better solution (lower cost, lower cloudy area)
        better_solution = Solution.create(
            selected_images=[0],
            cost=10,
            cloudy_area=100,
            timestamp_us=1000000,
            max_incidence_angle=45,
            min_resolutions_sum=100
        )
        
        # Create a worse solution (higher cost, higher cloudy area)  
        worse_solution = Solution.create(
            selected_images=[1, 3, 4],
            cost=75,  # 20 + 25 + 30
            cloudy_area=400,
            timestamp_us=2000000,
            max_incidence_angle=50,
            min_resolutions_sum=600
        )
        
        reference = [100, 500]
        
        # Single better solution should have higher HV than single worse solution
        hv_better = compute_hypervolume([better_solution], reference)
        hv_worse = compute_hypervolume([worse_solution], reference)
        
        assert hv_better > hv_worse, \
            f"Better solution should have higher hypervolume: {hv_better} <= {hv_worse}"
        
        # Combined front should have higher HV than either individual solution
        hv_combined = compute_hypervolume([better_solution, worse_solution], reference)
        assert hv_combined >= hv_better, \
            f"Combined front should have at least as high HV as better solution: {hv_combined} < {hv_better}"


if __name__ == "__main__":
    # Run tests directly if script is executed
    pytest.main([__file__, "-v"])
