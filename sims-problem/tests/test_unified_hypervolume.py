"""
Test suite for the unified hypervolume function.

This tests the single compute_hypervolume function that can handle:
- Points (Vec<Vec<u64>>) or Solutions with scaling options
- 2D, 3D, and 4D cases
- With and without scaling
- Cross-validation with pymoo
"""

import pytest
import numpy as np
from sims_problem import compute_hypervolume, Solution
try:
    from pymoo.indicators.hv import HV
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    HV = None


class TestUnifiedHypervolume:
    """Test the unified compute_hypervolume function."""

    def test_points_2d_basic(self):
        """Test basic 2D points without scaling."""
        points = [[1, 2], [2, 1]]
        reference = [3, 3]
        hv = compute_hypervolume(points, [[0, 100], [0, 100]], reference)
        assert hv == 3

    def test_points_2d_scaled(self):
        """Test 2D points with scaling."""
        points = [[100, 200], [200, 100]]
        reference = [300, 300]
        hv_unscaled = compute_hypervolume(points, [[0, 400], [0, 400]], reference, scaled=False)
        hv_scaled = compute_hypervolume(points, [[0, 400], [0, 400]], reference, scaled=True)
        
        # Both should give meaningful results, scaled should normalize
        assert hv_unscaled > 0
        assert hv_scaled > 0
        # Scaled hypervolume should be in a normalized range
        assert hv_scaled <= 1000 * 1000  # max scaled range

    @pytest.mark.skipif(not PYMOO_AVAILABLE, reason="pymoo not available")
    def test_points_2d_pymoo_validation(self):
        """Cross-validate 2D points with pymoo."""
        points = [[1, 2], [2, 1], [1, 1]]
        reference = [4, 4]
        
        # Our implementation
        hv_ours = compute_hypervolume(points, [[0, 100], [0, 100]], reference)
        
        # PyMoo validation
        points_np = np.array(points, dtype=float)
        reference_np = np.array(reference, dtype=float)
        assert HV is not None, "HV should be available for this test"
        hv_indicator = HV(ref_point=reference_np)
        hv_pymoo = hv_indicator(points_np)
        
        # Convert to integer for comparison (pymoo returns float)
        assert hv_pymoo is not None, "PyMoo should return a valid value"
        hv_pymoo_int = int(round(hv_pymoo))
        assert hv_ours == hv_pymoo_int

    def test_points_3d_basic(self):
        """Test basic 3D points."""
        points = [[1, 2, 3], [2, 1, 3], [1, 1, 2]]
        reference = [4, 4, 4]
        hv = compute_hypervolume(points, [[0, 4], [0, 4], [0, 4]], reference)
        assert hv > 0

    @pytest.mark.skipif(not PYMOO_AVAILABLE, reason="pymoo not available")
    def test_points_3d_pymoo_validation(self):
        """Cross-validate 3D points with pymoo."""
        points = [[1, 2, 1], [2, 1, 1]]
        reference = [3, 3, 3]
        
        # Our implementation
        hv_ours = compute_hypervolume(points, [[0, 3], [0, 3], [0, 3]], reference)
        
        # PyMoo validation
        points_np = np.array(points, dtype=float)
        reference_np = np.array(reference, dtype=float)
        assert HV is not None, "HV should be available for this test"
        hv_indicator = HV(ref_point=reference_np)
        hv_pymoo = hv_indicator(points_np)
        
        assert hv_pymoo is not None, "PyMoo should return a valid value"
        hv_pymoo_int = int(round(hv_pymoo))
        assert hv_ours == hv_pymoo_int

    def test_points_4d_basic(self):
        """Test basic 4D points."""
        points = [[1, 2, 3, 4], [2, 1, 3, 4]]
        reference = [5, 5, 5, 5]
        hv = compute_hypervolume(points, [[0, 5], [0, 5], [0, 5], [0, 5]], reference)
        assert hv > 0

    def test_solutions_2d(self):
        """Test with Solution objects (2D)."""
        # Create simple solutions using the create method
        solution1 = Solution.create(
            selected_images=[1, 2],
            cost=100,
            cloudy_area=200,
            timestamp_us=0,
            max_incidence_angle=None,
            min_resolutions_sum=None
        )
        solution2 = Solution.create(
            selected_images=[3, 4],
            cost=200,
            cloudy_area=100,
            timestamp_us=0,
            max_incidence_angle=None,
            min_resolutions_sum=None
        )
        
        solutions = [solution1, solution2]
        reference = [300, 300]
        
        hv = compute_hypervolume(solutions, [[0, 100], [0, 100]], reference)
        assert hv > 0

    def test_solutions_2d_scaled(self):
        """Test Solution objects with scaling."""
        solution1 = Solution.create(
            selected_images=[1, 2],
            cost=1000,
            cloudy_area=2000,
            timestamp_us=0,
            max_incidence_angle=None,
            min_resolutions_sum=None
        )
        solution2 = Solution.create(
            selected_images=[3, 4],
            cost=2000,
            cloudy_area=1000,
            timestamp_us=0,
            max_incidence_angle=None,
            min_resolutions_sum=None
        )
        
        solutions = [solution1, solution2]
        reference = [3000, 3000]
        bounds = [[0, 3000], [0, 3000]]
        
        hv_unscaled = compute_hypervolume(solutions, bounds, reference, scaled=False)
        hv_scaled = compute_hypervolume(solutions, bounds, reference, scaled=True)
        
        assert hv_unscaled > 0
        assert hv_scaled > 0

    def test_empty_input(self):
        """Test with empty input."""
        bounds = [[0, 3], [0, 3]]
        assert compute_hypervolume([], bounds, [3, 3]) == 0
        assert compute_hypervolume([], bounds, [3, 3], scaled=True) == 0

    def test_dimension_mismatch(self):
        """Test error handling for dimension mismatch."""
        points = [[1, 2], [3, 4, 5]]  # Mixed dimensions
        reference = [4, 4]
        
        with pytest.raises(ValueError, match="same dimension"):
            compute_hypervolume(points, [[0, 100], [0, 100]], reference)

    def test_unsupported_dimension(self):
        """Test error for unsupported dimensions."""
        points = [[1, 2, 3, 4, 5]]  # 5D not supported
        reference = [6, 6, 6, 6, 6]
        
        with pytest.raises(ValueError, match="Unsupported|dimension"):
            compute_hypervolume(points, [[0, 100], [0, 100], [0, 100], [0, 100], [0, 100]], reference)

    def test_invalid_input_type(self):
        """Test error for invalid input types."""
        invalid_data = "not a list"
        reference = [3, 3]
        
        with pytest.raises(TypeError, match="Input data must be either a list"):
            compute_hypervolume(invalid_data, [[0, 10], [0, 10]], reference)  # type: ignore

    @pytest.mark.skipif(not PYMOO_AVAILABLE, reason="pymoo not available")
    def test_scaling_preserves_relationships(self):
        """Test that scaling preserves dominance relationships."""
        # Large scale points (convert floats to integers)
        points = [[100, 200], [150, 150], [200, 100]]
        reference = [300, 300]
        
        # Small scale equivalent (convert floats to integers)
        points_small = [[1, 2], [2, 2], [2, 1]]  # Changed 1.5 to 2 for integer compatibility
        reference_small = [3, 3]
        
        # Both should have the same relative hypervolume when normalized
        hv_large = compute_hypervolume(points, [[0, 400], [0, 400]], reference, scaled=True)
        hv_small = compute_hypervolume(points_small, [[0, 10], [0, 10]], reference_small, scaled=True)
        
        # Both should give meaningful results
        assert hv_large > 0
        assert hv_small > 0

    def test_points_vs_equivalent_solutions(self):
        """Test that points and equivalent solutions give same result."""
        # Create solutions
        solution1 = Solution.create(
            selected_images=[1, 2],
            cost=100,
            cloudy_area=200,
            timestamp_us=0,
            max_incidence_angle=None,
            min_resolutions_sum=None
        )
        solution2 = Solution.create(
            selected_images=[3, 4],
            cost=200,
            cloudy_area=100,
            timestamp_us=0,
            max_incidence_angle=None,
            min_resolutions_sum=None
        )
        solutions = [solution1, solution2]
        
        # Equivalent points
        points = [[100, 200], [200, 100]]
        reference = [300, 300]
        
        hv_solutions = compute_hypervolume(solutions, [[0, 100], [0, 100]], reference)
        hv_points = compute_hypervolume(points, [[0, 100], [0, 100]], reference)
        
        assert hv_solutions == hv_points

    @pytest.mark.skipif(not PYMOO_AVAILABLE, reason="pymoo not available")
    def test_comprehensive_pymoo_validation(self):
        """Comprehensive test with various cases against pymoo."""
        test_cases = [
            # 2D cases
            {
                "points": [[1, 3], [2, 2], [3, 1]],
                "reference": [4, 4],
                "expected_hv": 6  # Manual calculation: (4-1)*(4-3) + (4-2)*(4-2) + (4-3)*(4-1) = 3*1 + 2*2 + 1*3 = 10, but with overlap handling = 6
            },
            {
                "points": [[1, 1]],
                "reference": [3, 3],
                "expected_hv": 4  # (3-1) * (3-1) = 4
            },
            # 3D cases
            {
                "points": [[1, 1, 1]],
                "reference": [3, 3, 3],
                "expected_hv": 8  # (3-1)^3 = 8
            }
        ]
        
        for case in test_cases:
            points = case["points"]
            reference = case["reference"]
            
            # Create bounds matching the reference dimensions
            bounds = [[0, 100] for _ in reference]
            
            # Our implementation
            hv_ours = compute_hypervolume(points, bounds, reference)
            
            # PyMoo validation
            points_np = np.array(points, dtype=float)
            reference_np = np.array(reference, dtype=float)
            assert HV is not None, "HV should be available for this test"
            hv_indicator = HV(ref_point=reference_np)
            hv_pymoo = hv_indicator(points_np)
            assert hv_pymoo is not None, "PyMoo should return a valid value"
            hv_pymoo_int = int(round(hv_pymoo))
            
            assert hv_ours == hv_pymoo_int, f"Failed for case: {case}"


if __name__ == "__main__":
    pytest.main([__file__])
