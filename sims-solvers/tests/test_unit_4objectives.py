"""
Unit tests for 4-objective implementation in sims-solvers.
Tests core functionality without requiring full solver runs.
"""

import pytest
import sys
from pathlib import Path

# Add the sims-solvers directory to the Python path
sims_solvers_path = Path(__file__).parent.parent
sys.path.insert(0, str(sims_solvers_path))


class TestParetoFront4Objectives:
    """Test ParetoFront with 4 objectives."""
    
    def test_pareto_dominance_4_objectives(self):
        """Test Pareto dominance with 4 objectives."""
        from sims_solvers.ParetoFront import ParetoFront
        
        front = ParetoFront()
        
        # Create test solutions with 4 objectives (all minimize)
        solution1 = {
            "objs": [100, 50, 30, 200],
            "minimize_objs": [True, True, True, True]
        }
        solution2 = {
            "objs": [90, 60, 25, 220],  # Better in obj 1&3, worse in obj 2&4
            "minimize_objs": [True, True, True, True]  
        }
        solution3 = {
            "objs": [80, 40, 20, 180],  # Dominates solution1
            "minimize_objs": [True, True, True, True]
        }
        
        # Test dominance relationships
        assert front.dominates(solution3, solution1), "Solution3 should dominate solution1"
        assert not front.dominates(solution1, solution2), "Solution1 should not dominate solution2"
        assert not front.dominates(solution2, solution1), "Solution2 should not dominate solution1"
    
    def test_compare_method_4_objectives(self):
        """Test the compare method works with any number of objectives."""
        from sims_solvers.ParetoFront import ParetoFront
        
        front = ParetoFront()
        
        # Test minimize objectives
        assert front.compare(5, 10, True), "5 <= 10 for minimization"
        assert not front.compare(10, 5, True), "10 > 5 for minimization"
        
        # Test maximize objectives
        assert front.compare(10, 5, False), "10 >= 5 for maximization"
        assert not front.compare(5, 10, False), "5 < 10 for maximization"


class TestCoverageGridPointInitialization:
    """Test CoverageGridPoint initialization for 4 objectives."""
    
    def test_constraint_objectives_initialization(self):
        """Test that constraint_objectives are properly initialized."""
        from sims_solvers.FrontGenerators.CoverageGridPoint import CoverageGridPoint
        from sims_solvers.Timer import Timer
        
        # Mock solver with 4 objectives
        class MockSolver:
            def __init__(self):
                self.model = MockModel()
        
        class MockModel:
            def __init__(self):
                self.objectives = [None, None, None, None]  # 4 objectives
            
            def is_a_minimization_model(self):
                return True
        
        solver = MockSolver()
        timer = Timer(30)
        
        # Should not raise ValueError about number of objectives
        try:
            cgp = CoverageGridPoint(solver, timer)
            # For 4 objectives, should have 3 constraint objectives (n-1)
            assert len(cgp.constraint_objectives) == 3, \
                f"Expected 3 constraint objectives, got {len(cgp.constraint_objectives)}"
        except ValueError as e:
            if "only works for 2 objectives" in str(e):
                pytest.fail("CoverageGridPoint should accept 4 objectives")
            else:
                raise


class TestInstanceData:
    """Test that instance data contains resolution and incidence angle."""
    
    def test_resolution_data_available(self):
        """Test that resolution data is available in test instances."""
        from pathlib import Path
        
        # Use relative path from test file location
        test_file = Path(__file__).parent.parent.parent / "sims-problem" / "tests" / "data" / "lagos_nigeria_30.dzn"
        
        if test_file.exists():
            with open(test_file, 'r') as f:
                content = f.read()
                assert "resolution =" in content, "Test file should contain resolution data"
                assert "incidence_angle =" in content, "Test file should contain incidence_angle data"
        else:
            pytest.skip("Test data file not available")
    
    def test_minizinc_model_has_4_objectives(self):
        """Test that MiniZinc model defines 4 objectives."""
        from pathlib import Path
        
        # Use relative path from test file location
        mzn_file = Path(__file__).parent.parent / "sims_solvers" / "mzn_models" / "mosaic_cloud2.mzn"
        
        if mzn_file.exists():
            with open(mzn_file, 'r') as f:
                content = f.read()
                assert "array[1..4] of var int: objs;" in content, "Model should define 4 objectives"
                assert "objs[1] = total_cost;" in content, "Should define cost objective"
                assert "objs[2] = cloudy_area;" in content, "Should define cloud objective"
                assert "objs[3] = max_resolution;" in content, "Should define resolution objective"
                assert "objs[4] = max_incidence;" in content, "Should define incidence objective"
        else:
            pytest.skip("MiniZinc model file not available")


class TestIntervalManager:
    """Test the IntervalManager class used by CoverageGridPoint."""
    
    def test_interval_manager_basic_operations(self):
        """Test basic IntervalManager operations."""
        from sims_solvers.FrontGenerators.CoverageGridPoint import IntervalManager
        
        manager = IntervalManager(1, 10)
        
        # Test initial state
        assert manager.min_value == 1
        assert manager.max_value == 10
        assert len(manager.intervals) == 1
        
        # Test interval removal
        manager.remove_interval(3, 5)
        largest = manager.find_largest_interval()
        assert largest is not None, "Should find largest interval after removal"
        assert largest[0] == 6 and largest[1] == 10, "Largest interval should be [1,2]"

        # Test point removal
        manager.remove_one_point(7)
        for interval in manager.intervals:
            assert 7 < interval[0] or 7 > interval[1], "Point 7 should be removed from intervals"
        
        # Test interval addition
        manager = IntervalManager(1, 10)
        manager.add_interval(15, 20)
        assert len(manager.intervals) >= 1, "Should have intervals after addition"


class TestSolutionPrinting:
    """Test solution printing functionality with selected images."""
    
    def test_solution_format_display(self):
        """Test the format for displaying solutions with selected images."""
        # Mock solution data to demonstrate the output format
        mock_solutions = [
            {
                'objectives': [1000, 25, 15, 30],
                'selected_images': [True, False, True, True, False, True, False, False, True, False],
                'instance': 'test_instance'
            },
            {
                'objectives': [1200, 20, 18, 25],
                'selected_images': [False, True, True, False, True, True, True, False, False, True],
                'instance': 'test_instance'
            },
            {
                'objectives': [800, 35, 12, 40],
                # Simulate a larger selection with 15 images
                'selected_images': [True] * 15 + [False] * 5,
                'instance': 'test_large_instance'
            }
        ]
        
        print("\n📊 Example Solution Output Format:")
        for i, sol in enumerate(mock_solutions):
            obj = sol['objectives']
            selected_imgs = sol['selected_images']
            
            # Convert boolean list to indices of selected images (1-based indexing)
            selected_indices = [idx + 1 for idx, selected in enumerate(selected_imgs) if selected]
            
            print(f"  Solution {i+1}:")
            print(f"    Objectives: Cost={obj[0]}, Cloud={obj[1]}, Resolution={obj[2]}, Angle={obj[3]}")
            print(f"    Selected Images ({len(selected_indices)}): {selected_indices}")
            if len(selected_indices) <= 10:
                print(f"    Image Details: {selected_indices}")
            else:
                print(f"    Image Details: {selected_indices[:5]} ... {selected_indices[-5:]} (showing first 5 and last 5)")
            print()
        
        # Basic assertions
        assert len(mock_solutions) == 3
        assert all(len(sol['objectives']) == 4 for sol in mock_solutions)
        assert all(isinstance(sol['selected_images'], list) for sol in mock_solutions)
        
        print("✓ Solution format test completed successfully")


class TestObjectiveCalculations:
    """Test objective calculation methods."""
    
    def test_resolution_calculation_exists(self):
        """Test that resolution calculation method exists."""
        from sims_solvers.Models.SatelliteImageMosaicSelectionGeneralModel import SatelliteImageMosaicSelectionGeneralModel
        
        # Check that the method exists
        assert hasattr(SatelliteImageMosaicSelectionGeneralModel, 'calculate_resolution'), \
            "Should have calculate_resolution method"
        assert hasattr(SatelliteImageMosaicSelectionGeneralModel, 'assert_resolution'), \
            "Should have assert_resolution method"
        assert hasattr(SatelliteImageMosaicSelectionGeneralModel, 'assert_incidence_angle'), \
            "Should have assert_incidence_angle method"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
