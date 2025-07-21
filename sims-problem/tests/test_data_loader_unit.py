import pytest
from .test_data_loader import load_test_instance, get_all_test_instances, parse_dzn_file, load_test_instance_as_problem
import sims_problem
from pathlib import Path


class TestDataLoader:
    """Test the data loader functionality."""

    def test_get_all_test_instances(self):
        """Test that we can discover all test instance files."""
        instances = get_all_test_instances()
        assert len(instances) > 0, "Should find at least one test instance"
        assert all(filename.endswith('.dzn') for filename in instances), \
            "All instances should be .dzn files"

    def test_parse_small_instance_python(self):
        """Test parsing a small instance file using Python implementation."""
        data = load_test_instance("lagos_nigeria_30.dzn")
        
        # Check required fields are present
        required_fields = [
            'num_images', 'universe', 'images', 'costs', 'clouds', 
            'areas', 'resolution', 'incidence_angle', 'max_cloud_area'
        ]
        
        for field in required_fields:
            assert field in data, f"Field {field} should be present in parsed data"
        
        # Check data types and basic constraints
        assert isinstance(data['num_images'], int) and data['num_images'] > 0
        assert isinstance(data['universe'], int) and data['universe'] > 0
        assert isinstance(data['max_cloud_area'], int) and data['max_cloud_area'] > 0
        
        assert len(data['images']) == data['num_images']
        assert len(data['costs']) == data['num_images']
        assert len(data['clouds']) == data['num_images']
        assert len(data['areas']) == data['universe']
        assert len(data['resolution']) == data['num_images']
        assert len(data['incidence_angle']) == data['num_images']
        
        # Check that images use 0-based indexing after parsing
        for image in data['images']:
            for fragment in image:
                assert 0 <= fragment < data['universe'], \
                    f"Fragment index {fragment} should be in range [0, {data['universe']})"

    def test_parse_small_instance_rust(self):
        """Test parsing a small instance file using Rust implementation."""
        problem = load_test_instance_as_problem("lagos_nigeria_30.dzn")
        
        # Check basic properties
        assert problem.num_images > 0
        assert problem.universe > 0
        assert problem.max_cloud_area > 0
        
        assert len(problem.images) == problem.num_images
        assert len(problem.costs) == problem.num_images
        assert len(problem.clouds) == problem.num_images
        assert len(problem.areas) == problem.universe
        assert len(problem.resolution) == problem.num_images
        assert len(problem.incidence_angle) == problem.num_images
        
        # Check that images use 0-based indexing
        for image in problem.images:
            for fragment in image:
                assert 0 <= fragment < problem.universe, \
                    f"Fragment index {fragment} should be in range [0, {problem.universe})"

    def test_rust_vs_python_equivalence(self):
        """Test that Rust and Python implementations produce equivalent results."""
        filename = "lagos_nigeria_30.dzn"
        
        # Load with Python implementation
        python_data = load_test_instance(filename)
        python_problem = sims_problem.SimsDiscreteProblem(
            num_images=python_data['num_images'],
            universe=python_data['universe'],
            images=python_data['images'],
            costs=python_data['costs'],
            clouds=python_data['clouds'],
            areas=python_data['areas'],
            resolution=python_data['resolution'],
            incidence_angle=python_data['incidence_angle'],
            max_cloud_area=python_data['max_cloud_area']
        )
        
        # Load with Rust implementation
        rust_problem = load_test_instance_as_problem(filename)
        
        # Compare all fields
        assert python_problem.num_images == rust_problem.num_images
        assert python_problem.universe == rust_problem.universe
        assert python_problem.max_cloud_area == rust_problem.max_cloud_area
        assert python_problem.images == rust_problem.images
        assert python_problem.costs == rust_problem.costs
        assert python_problem.clouds == rust_problem.clouds
        assert python_problem.areas == rust_problem.areas
        assert python_problem.resolution == rust_problem.resolution
        assert python_problem.incidence_angle == rust_problem.incidence_angle

    def test_create_sims_problem_from_parsed_data(self):
        """Test creating a SimsDiscreteProblem from parsed data."""
        data = load_test_instance("lagos_nigeria_30.dzn")
        
        problem = sims_problem.SimsDiscreteProblem(
            num_images=data['num_images'],
            universe=data['universe'],
            images=data['images'],
            costs=data['costs'],
            clouds=data['clouds'],
            areas=data['areas'],
            resolution=data['resolution'],
            incidence_angle=data['incidence_angle'],
            max_cloud_area=data['max_cloud_area']
        )
        
        # Should be able to validate without errors
        problem.validate()
        
        # Check basic properties
        assert problem.num_images == data['num_images']
        assert problem.universe == data['universe']
        assert problem.max_cloud_area == data['max_cloud_area']

    def test_from_dzn_static_method(self):
        """Test the new from_dzn static method directly."""
        test_dir = Path(__file__).parent / "data"
        file_path = test_dir / "lagos_nigeria_30.dzn"
        
        problem = sims_problem.SimsDiscreteProblem.from_dzn(str(file_path))
        
        # Should be able to validate without errors
        problem.validate()
        
        # Check basic properties
        assert problem.num_images > 0
        assert problem.universe > 0
        assert problem.max_cloud_area > 0

    @pytest.mark.parametrize("filename", [
        "lagos_nigeria_30.dzn",
        "mexico_city_50.dzn",
        "paris_100.dzn",
    ])
    def test_parse_multiple_instances(self, filename):
        """Test parsing multiple different instance files."""
        try:
            # Test both implementations
            data = load_test_instance(filename)
            problem_from_data = sims_problem.SimsDiscreteProblem(**data)
            problem_from_dzn = load_test_instance_as_problem(filename)
            
            # Basic validation
            assert data['num_images'] > 0
            assert data['universe'] > 0
            assert len(data['images']) == data['num_images']
            assert len(data['costs']) == data['num_images']
            
            # Should be able to create valid problems
            problem_from_data.validate()
            problem_from_dzn.validate()
            
            # Both should be equivalent
            assert problem_from_data.num_images == problem_from_dzn.num_images
            assert problem_from_data.universe == problem_from_dzn.universe
            
        except FileNotFoundError:
            pytest.skip(f"Test file {filename} not found")

    def test_indexing_conversion(self):
        """Test that 1-based indexing in .dzn files is correctly converted to 0-based."""
        # Create a minimal test data string with 1-based indexing
        test_dzn_content = """
        num_images = 2;
        universe = 4;
        images = [{1, 2}, {3, 4}];
        clouds = [{1}, {3}];
        costs = [10, 20];
        areas = [1, 1, 1, 1];
        resolution = [30, 50];
        incidence_angle = [10, 20];
        max_cloud_area = 100;
        """
        
        # Write to a temporary file and parse
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dzn', delete=False) as f:
            f.write(test_dzn_content)
            temp_path = f.name
        
        try:
            # Test Python implementation
            data = parse_dzn_file(temp_path)
            
            # Check that indices were converted to 0-based
            assert data['images'] == [[0, 1], [2, 3]]
            assert data['clouds'] == [[0], [2]]
            
            # Test Rust implementation
            rust_problem = sims_problem.SimsDiscreteProblem.from_dzn(temp_path)
            assert rust_problem.images == [[0, 1], [2, 3]]
            assert rust_problem.clouds == [[0], [2]]
            
        finally:
            import os
            os.unlink(temp_path)
