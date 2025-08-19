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
        import os
        
        test_file = "/home/hlvlad/code/serenity/sims-hybrid-algs/sims-problem/tests/data/lagos_nigeria_30.dzn"
        
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                content = f.read()
                assert "resolution =" in content, "Test file should contain resolution data"
                assert "incidence_angle =" in content, "Test file should contain incidence_angle data"
        else:
            pytest.skip("Test data file not available")
    
    def test_minizinc_model_has_4_objectives(self):
        """Test that MiniZinc model defines 4 objectives."""
        import os
        
        mzn_file = "/home/hlvlad/code/serenity/sims-hybrid-algs/sims-solvers/sims_solvers/mzn_models/mosaic_cloud2.mzn"
        
        if os.path.exists(mzn_file):
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
        
        # Test interval addition
        manager.add_interval(15, 20)
        assert len(manager.intervals) >= 1, "Should have intervals after addition"


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
