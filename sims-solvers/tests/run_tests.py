#!/usr/bin/env python3
"""
Simple test runner for 4-objective implementation tests.
Runs without requiring pytest installation.
"""

import sys
import traceback
from pathlib import Path

# Add the sims-solvers directory to the Python path
sims_solvers_path = Path(__file__).parent.parent
sys.path.insert(0, str(sims_solvers_path))


class TestRunner:
    """Simple test runner."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def run_test(self, test_func, test_name):
        """Run a single test function."""
        try:
            print(f"Running {test_name}...", end=" ")
            test_func()
            print("PASSED")
            self.passed += 1
        except Exception as e:
            print("FAILED")
            self.failed += 1
            self.errors.append(f"{test_name}: {str(e)}")
            traceback.print_exc()
    
    def summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"Test Summary: {self.passed}/{total} tests passed")
        if self.errors:
            print("\nFailed tests:")
            for error in self.errors:
                print(f"  - {error}")
        print(f"{'='*50}")


def test_pareto_front_4_objectives():
    """Test ParetoFront with 4 objectives."""
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


def test_compare_method():
    """Test the compare method."""
    from sims_solvers.ParetoFront import ParetoFront
    
    front = ParetoFront()
    
    # Test minimize objectives
    assert front.compare(5, 10, True), "5 <= 10 for minimization"
    assert not front.compare(10, 5, True), "10 > 5 for minimization"
    
    # Test maximize objectives
    assert front.compare(10, 5, False), "10 >= 5 for maximization"
    assert not front.compare(5, 10, False), "5 < 10 for maximization"


def test_coverage_grid_point_initialization():
    """Test CoverageGridPoint accepts 4 objectives."""
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
    cgp = CoverageGridPoint(solver, timer)
    # For 4 objectives, should have 3 constraint objectives (n-1)
    assert len(cgp.constraint_objectives) == 3, \
        f"Expected 3 constraint objectives, got {len(cgp.constraint_objectives)}"


def test_interval_manager():
    """Test IntervalManager operations."""
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


def test_minizinc_model_4_objectives():
    """Test MiniZinc model has 4 objectives."""
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


def test_instance_data():
    """Test instance data has required fields."""
    import os
    
    test_file = "/home/hlvlad/code/serenity/sims-hybrid-algs/sims-problem/tests/data/lagos_nigeria_30.dzn"
    
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            content = f.read()
            assert "resolution =" in content, "Test file should contain resolution data"
            assert "incidence_angle =" in content, "Test file should contain incidence_angle data"


def parse_simple_dzn_data(dzn_file_path):
    """Simple DZN parser to extract data for InstanceSIMS."""
    data = {}
    
    with open(dzn_file_path, 'r') as f:
        content = f.read()
    
    # Simple parsing for basic data structures
    import re
    
    # Parse arrays
    def parse_array(pattern, content):
        match = re.search(pattern, content, re.DOTALL)
        if match:
            array_str = match.group(1)
            # Remove brackets and split by comma, then convert to appropriate type
            items = [item.strip() for item in array_str.strip('[]').split(',')]
            return [float(item) if '.' in item else int(item) for item in items if item.strip()]
        return []
    
    # Parse set arrays (for images and clouds)
    def parse_set_array(pattern, content):
        match = re.search(pattern, content, re.DOTALL)
        if match:
            array_str = match.group(1)
            sets = []
            # Find all sets in the array
            set_matches = re.findall(r'\{([^}]*)\}', array_str)
            for set_match in set_matches:
                if set_match.strip():
                    # Convert to set of integers (1-indexed in DZN, will be corrected later)
                    elements = [int(x.strip()) for x in set_match.split(',') if x.strip()]
                    sets.append(set(elements))
                else:
                    sets.append(set())
            return sets
        return []
    
    # Extract data
    data['costs'] = parse_array(r'costs\s*=\s*\[([^\]]+)\]', content)
    data['areas'] = parse_array(r'areas\s*=\s*\[([^\]]+)\]', content)
    data['resolution'] = parse_array(r'resolution\s*=\s*\[([^\]]+)\]', content)
    data['incidence_angle'] = parse_array(r'incidence_angle\s*=\s*\[([^\]]+)\]', content)
    
    data['images'] = parse_set_array(r'images\s*=\s*\[([^\]]+)\]', content)
    data['clouds'] = parse_set_array(r'clouds\s*=\s*\[([^\]]+)\]', content)
    
    # Extract max_cloud_area
    match = re.search(r'max_cloud_area\s*=\s*(\d+)', content)
    data['max_cloud_area'] = int(match.group(1)) if match else 0
    
    return data


def test_gurobi_model_4_objectives():
    """Test Gurobi model with 4 objectives using real DZN data."""
    import os
    import tempfile
    from sims_solvers.main import build_model
    from sims_solvers.Config import Config
    from sims_solvers.Instances.InstanceSIMS import InstanceSIMS
    from sims_solvers.Models.GurobiModels.SatelliteImageMosaicSelectionGurobiModel import SatelliteImageMosaicSelectionGurobiModel
    
    test_file = "/home/hlvlad/code/serenity/sims-hybrid-algs/sims-problem/tests/data/lagos_nigeria_30.dzn"
    
    if os.path.exists(test_file):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config for direct Gurobi model (not MiniZinc)
            config = Config(
                minizinc_data=False,  # Use direct Gurobi model instead of MiniZinc
                instance_name="lagos_nigeria_30",
                data_sets_folder=Path("/tmp"),
                input_mzn=Path("dummy.mzn"),  # Not used for direct Gurobi
                dzn_dir=Path("/tmp"),
                solver_name="gurobi",
                problem_name="sims",
                front_strategy="gpba-a",
                solver_timeout_sec=60,
                summary_filename=os.path.join(temp_dir, "test_summary.csv"),
                solver_search_strategy="free_search",
                fzn_optimisation_level=1,
                cores=1,
                threads=1
            )
            
            # Parse real DZN data
            dzn_data = parse_simple_dzn_data(test_file)
            
            # Create instance from real data
            instance = InstanceSIMS(dzn_data)
            model = build_model(instance, config)
            
            # Should be SatelliteImageMosaicSelectionGurobiModel
            assert isinstance(model, SatelliteImageMosaicSelectionGurobiModel), \
                f"Expected SatelliteImageMosaicSelectionGurobiModel, got {type(model)}"
            
            # Should have 4 objectives
            assert len(model.objectives) == 4, f"Expected 4 objectives, got {len(model.objectives)}"
            
            # Verify we have real data
            assert len(dzn_data['resolution']) > 0, "Should have resolution data from real DZN"
            assert len(dzn_data['incidence_angle']) > 0, "Should have incidence_angle data from real DZN"
            
            print(f"✓ Direct Gurobi model with real data loaded: {type(model).__name__}")
    else:
        print("✓ Lagos Nigeria test file not found, skipping")


def test_rio_de_janeiro_4_objectives():
    """Test 4 objectives on rio_de_janeiro_30 instance with real data."""
    import os
    import tempfile
    from sims_solvers.main import build_model
    from sims_solvers.Config import Config
    from sims_solvers.Instances.InstanceSIMS import InstanceSIMS
    from sims_solvers.Models.GurobiModels.SatelliteImageMosaicSelectionGurobiModel import SatelliteImageMosaicSelectionGurobiModel
    
    test_file = "/home/hlvlad/code/serenity/sims-hybrid-algs/sims-problem/tests/data/rio_de_janeiro_30.dzn"
    
    if os.path.exists(test_file):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config for direct Gurobi model
            config = Config(
                minizinc_data=False,
                instance_name="rio_de_janeiro_30",
                data_sets_folder=Path("/tmp"),
                input_mzn=Path("dummy.mzn"),
                dzn_dir=Path("/tmp"),
                solver_name="gurobi",
                problem_name="sims",
                front_strategy="gpba-a",
                solver_timeout_sec=60,
                summary_filename=os.path.join(temp_dir, "test_summary.csv"),
                solver_search_strategy="free_search",
                fzn_optimisation_level=1,
                cores=1,
                threads=1
            )
            
            # Parse real DZN data
            dzn_data = parse_simple_dzn_data(test_file)
            
            # Create instance from real data
            instance = InstanceSIMS(dzn_data)
            model = build_model(instance, config)
            
            # Should be SatelliteImageMosaicSelectionGurobiModel
            assert isinstance(model, SatelliteImageMosaicSelectionGurobiModel), 
                f"Expected SatelliteImageMosaicSelectionGurobiModel, got {type(model)}"
            
            # Should have 4 objectives
            assert len(model.objectives) == 4, f"Expected 4 objectives, got {len(model.objectives)}"
            
            print(f"✓ Rio de Janeiro Gurobi model with real data: {type(model).__name__}")
    else:
        print("✓ Rio de Janeiro test file not found, skipping")


def test_tokyo_bay_4_objectives():
    """Test 4 objectives on tokyo_bay_30 instance with real data."""
    import os
    import tempfile
    from sims_solvers.main import build_model
    from sims_solvers.Config import Config
    from sims_solvers.Instances.InstanceSIMS import InstanceSIMS
    from sims_solvers.Models.GurobiModels.SatelliteImageMosaicSelectionGurobiModel import SatelliteImageMosaicSelectionGurobiModel
    
    test_file = "/home/hlvlad/code/serenity/sims-hybrid-algs/sims-problem/tests/data/tokyo_bay_30.dzn"
    
    if os.path.exists(test_file):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config for direct Gurobi model
            config = Config(
                minizinc_data=False,
                instance_name="tokyo_bay_30",
                data_sets_folder=Path("/tmp"),
                input_mzn=Path("dummy.mzn"),
                dzn_dir=Path("/tmp"),
                solver_name="gurobi",
                problem_name="sims",
                front_strategy="gpba-a",
                solver_timeout_sec=60,
                summary_filename=os.path.join(temp_dir, "test_summary.csv"),
                solver_search_strategy="free_search",
                fzn_optimisation_level=1,
                cores=1,
                threads=1
            )
            
            # Parse real DZN data
            dzn_data = parse_simple_dzn_data(test_file)
            
            # Create instance from real data
            instance = InstanceSIMS(dzn_data)
            model = build_model(instance, config)
            
            # Should be SatelliteImageMosaicSelectionGurobiModel
            assert isinstance(model, SatelliteImageMosaicSelectionGurobiModel), 
                f"Expected SatelliteImageMosaicSelectionGurobiModel, got {type(model)}"
            
            # Should have 4 objectives
            assert len(model.objectives) == 4, f"Expected 4 objectives, got {len(model.objectives)}"
            
            print(f"✓ Tokyo Bay Gurobi model with real data: {type(model).__name__}")
    else:
        print("✓ Tokyo Bay test file not found, skipping")


def test_tokyo_bay_4_objectives():
    """Test 4 objectives on tokyo_bay_30 instance."""
    test_file = "/home/hlvlad/code/serenity/sims-hybrid-algs/sims-problem/tests/data/tokyo_bay_30.dzn"
    mzn_file = "/home/hlvlad/code/serenity/sims-hybrid-algs/augmecon-rs/src/mosaic_cloud2.mzn"
    
    import os
    import tempfile
    from sims_solvers.main import build_instance, build_model
    from sims_solvers.Config import Config
    
    if os.path.exists(test_file) and os.path.exists(mzn_file):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config with all required parameters
            config = Config(
                minizinc_data=True,
                instance_name="tokyo_bay_30",
                data_sets_folder=Path("/home/hlvlad/code/serenity/sims-hybrid-algs/sims-problem/tests/data"),
                input_mzn=Path(mzn_file),
                dzn_dir=Path("/home/hlvlad/code/serenity/sims-hybrid-algs/sims-problem/tests/data"),
                solver_name="gurobi",
                problem_name="sims",
                front_strategy="gpba-a",
                solver_timeout_sec=60,
                summary_filename=os.path.join(temp_dir, "test_summary.csv"),
                solver_search_strategy="free_search",
                fzn_optimisation_level=1,
                cores=1,
                threads=1
            )
            
            instance = build_instance(config)
            model = build_model(instance, config)
            
            # Should have 4 objectives
            assert len(model.objectives) == 4, f"Expected 4 objectives, got {len(model.objectives)}"
            print(f"✓ Tokyo Bay instance loaded: {type(instance).__name__}")
    else:
        print("✓ Tokyo Bay test file not found, skipping")


def test_objective_calculations():
    """Test objective calculation methods exist."""
    from sims_solvers.Models.SatelliteImageMosaicSelectionGeneralModel import SatelliteImageMosaicSelectionGeneralModel
    
    # Check that the methods exist
    assert hasattr(SatelliteImageMosaicSelectionGeneralModel, 'calculate_resolution'), \
        "Should have calculate_resolution method"
    assert hasattr(SatelliteImageMosaicSelectionGeneralModel, 'assert_resolution'), \
        "Should have assert_resolution method"
    assert hasattr(SatelliteImageMosaicSelectionGeneralModel, 'assert_incidence_angle'), \
        "Should have assert_incidence_angle method"


def main():
    """Run all tests."""
    runner = TestRunner()
    
    print("Running 4-Objective Implementation Tests")
    print("=" * 50)
    
    # Run all test functions
    test_functions = [
        (test_pareto_front_4_objectives, "ParetoFront 4 objectives"),
        (test_compare_method, "Compare method"),
        (test_coverage_grid_point_initialization, "CoverageGridPoint initialization"),
        (test_interval_manager, "IntervalManager operations"),
        (test_minizinc_model_4_objectives, "MiniZinc model 4 objectives"),
        (test_instance_data, "Instance data availability"),
        (test_gurobi_model_4_objectives, "Gurobi model 4 objectives - Lagos Nigeria"),
        (test_rio_de_janeiro_4_objectives, "Rio de Janeiro 4 objectives"),
        (test_tokyo_bay_4_objectives, "Tokyo Bay 4 objectives"),
        (test_objective_calculations, "Objective calculation methods"),
    ]
    
    for test_func, test_name in test_functions:
        runner.run_test(test_func, test_name)
    
    runner.summary()
    return runner.failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
