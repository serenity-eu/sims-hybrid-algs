import sims_problem


def test_solve_with_pls_basic():
    """Test the basic solve_with_pls function."""
    # Create a test problem instance
    test_problem = sims_problem.SimsDiscreteProblem(
        num_images=5,
        universe=10,
        images=[[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9, 0]],
        costs=[10, 15, 20, 12, 18],
        clouds=[[0], [2], [4], [6], [8]],
        areas=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        resolution=[5, 4, 3, 4, 5],
        incidence_angle=[10, 15, 20, 12, 18],
        max_cloud_area=50
    )
    
    # Test the simple solve_with_pls function
    solutions = sims_problem.solve_with_pls(test_problem)
    
    # Assertions
    assert len(solutions) > 0, "Should return at least one solution"
    
    for solution in solutions:
        # Check that solutions have valid structure
        assert hasattr(solution, 'cost'), "Solution should have cost attribute"
        assert hasattr(solution, 'cloudy_area'), "Solution should have cloudy_area attribute"
        assert hasattr(solution, 'timestamp_us'), "Solution should have timestamp_us attribute"
        
        # Check that selected images are valid
        selected_images = solution.get_selected_images_list()
        assert isinstance(selected_images, list), "Selected images should be a list"
        assert all(0 <= img < test_problem.num_images for img in selected_images), \
            "All selected image indices should be valid"
        
        # Check that cost and cloudy_area are non-negative
        assert solution.cost >= 0, "Cost should be non-negative"
        assert solution.cloudy_area >= 0, "Cloudy area should be non-negative"


def test_solve_with_pls_advanced():
    """Test the advanced solve_with_pls_advanced function with custom parameters."""
    # Create a test problem instance
    test_problem = sims_problem.SimsDiscreteProblem(
        num_images=4,
        universe=8,
        images=[[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 0]],
        costs=[10, 15, 20, 12],
        clouds=[[0], [2], [4], [6]],
        areas=[1, 1, 1, 1, 1, 1, 1, 1],
        resolution=[5, 4, 3, 4],
        incidence_angle=[10, 15, 20, 12],
        max_cloud_area=40
    )
    
    # Test the advanced solve_with_pls_advanced function
    solutions = sims_problem.solve_with_pls_advanced(
        test_problem,
        timeout_seconds=2.0,
        max_iterations=500,
        is_deterministic=True,
        initial_population_size=10,
        neighborhood_size_min=1,
        neighborhood_size_max=2
    )
    
    # Assertions
    assert len(solutions) > 0, "Should return at least one solution"
    
    for solution in solutions:
        # Check that solutions have valid structure
        selected_images = solution.get_selected_images_list()
        assert isinstance(selected_images, list), "Selected images should be a list"
        assert all(0 <= img < test_problem.num_images for img in selected_images), \
            "All selected image indices should be valid"
        
        # Check that cost and cloudy_area are non-negative
        assert solution.cost >= 0, "Cost should be non-negative"
        assert solution.cloudy_area >= 0, "Cloudy area should be non-negative"
        
        # Check timestamp is present
        assert solution.timestamp_us >= 0, "Timestamp should be non-negative"


def test_deterministic_behavior():
    """Test that deterministic runs produce consistent results."""
    # Create a test problem instance
    test_problem = sims_problem.SimsDiscreteProblem(
        num_images=3,
        universe=6,
        images=[[0, 1, 2], [2, 3, 4], [4, 5, 0]],
        costs=[10, 15, 20],
        clouds=[[0], [2], [4]],
        areas=[1, 1, 1, 1, 1, 1],
        resolution=[5, 4, 3],
        incidence_angle=[10, 15, 20],
        max_cloud_area=30
    )
    
    # Run deterministic solver twice
    solutions1 = sims_problem.solve_with_pls_advanced(
        test_problem,
        timeout_seconds=1.0,
        max_iterations=100,
        is_deterministic=True,
        initial_population_size=5,
        neighborhood_size_min=1,
        neighborhood_size_max=2
    )
    
    solutions2 = sims_problem.solve_with_pls_advanced(
        test_problem,
        timeout_seconds=1.0,
        max_iterations=100,
        is_deterministic=True,
        initial_population_size=5,
        neighborhood_size_min=1,
        neighborhood_size_max=2
    )
    
    # Should produce same number of solutions
    assert len(solutions1) == len(solutions2), \
        "Deterministic runs should produce same number of solutions"
    
    # Convert to sets of (cost, cloudy_area) for comparison
    objectives1 = {(sol.cost, sol.cloudy_area) for sol in solutions1}
    objectives2 = {(sol.cost, sol.cloudy_area) for sol in solutions2}
    
    assert objectives1 == objectives2, \
        "Deterministic runs should produce solutions with same objectives"


def test_parameter_validation():
    """Test that invalid parameters are handled properly."""
    # Create a test problem instance
    test_problem = sims_problem.SimsDiscreteProblem(
        num_images=3,
        universe=6,
        images=[[0, 1, 2], [2, 3, 4], [4, 5, 0]],
        costs=[10, 15, 20],
        clouds=[[0], [2], [4]],
        areas=[1, 1, 1, 1, 1, 1],
        resolution=[5, 4, 3],
        incidence_angle=[10, 15, 20],
        max_cloud_area=30
    )
    
    # Test with very small timeout - should still return solutions
    solutions = sims_problem.solve_with_pls_advanced(
        test_problem,
        timeout_seconds=0.1,
        max_iterations=10,
        is_deterministic=True,
        initial_population_size=2,
        neighborhood_size_min=1,
        neighborhood_size_max=1
    )
    
    # Should still return at least initial population
    assert len(solutions) >= 1, "Should return at least one solution even with small timeout"
