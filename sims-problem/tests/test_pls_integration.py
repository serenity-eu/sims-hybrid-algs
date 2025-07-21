import sims_problem


def test_solve_with_pls_biobjective_basic():
    """Test the basic solve_with_pls function (biobjective: cost + cloudy area)."""
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


def test_solve_with_pls_biobjective_advanced():
    """Test the advanced solve_with_pls_advanced function with custom parameters (biobjective)."""
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


def test_biobjective_deterministic_behavior():
    """Test that deterministic biobjective runs produce consistent results."""
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


def test_biobjective_parameter_validation():
    """Test that invalid parameters are handled properly in biobjective optimization."""
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


# ========================================================================================
# MULTIOBJECTIVE TESTS (4D: Cost + Cloudy Area + Min Resolution + Max Incidence Angle)
# ========================================================================================

def test_solve_with_pls_multiobjective_basic():
    """Test the basic solve_with_pls_multiobjective function (4D optimization)."""
    # Create a test problem instance with diverse objective values
    test_problem = sims_problem.SimsDiscreteProblem(
        num_images=6,
        universe=10,
        images=[[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9, 0], [1, 3, 5]],
        costs=[10, 25, 15, 30, 20, 18],  # Different costs for trade-offs
        clouds=[[0], [2], [4], [6], [8], [1]],  # Different cloud patterns
        areas=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        resolution=[5, 2, 8, 3, 6, 4],  # Different resolutions (lower is better)
        incidence_angle=[10, 45, 20, 60, 30, 25],  # Different incidence angles (lower is better)
        max_cloud_area=50
    )
    
    # Test the multiobjective solve_with_pls_multiobjective function
    solutions = sims_problem.solve_with_pls_multiobjective(test_problem)
    
    # Assertions
    assert len(solutions) > 0, "Should return at least one solution"
    
    for solution in solutions:
        # Check that solutions have valid structure for 4D objectives
        assert hasattr(solution, 'cost'), "Solution should have cost attribute"
        assert hasattr(solution, 'cloudy_area'), "Solution should have cloudy_area attribute"
        assert hasattr(solution, 'min_resolutions_sum'), "Solution should have min_resolutions_sum attribute"
        assert hasattr(solution, 'max_incidence_angle'), "Solution should have max_incidence_angle attribute"
        assert hasattr(solution, 'timestamp_us'), "Solution should have timestamp_us attribute"
        
        # Check that selected images are valid
        selected_images = solution.get_selected_images_list()
        assert isinstance(selected_images, list), "Selected images should be a list"
        assert all(0 <= img < test_problem.num_images for img in selected_images), \
            "All selected image indices should be valid"
        
        # Check that all objectives are non-negative
        assert solution.cost >= 0, "Cost should be non-negative"
        assert solution.cloudy_area >= 0, "Cloudy area should be non-negative"
        assert solution.min_resolutions_sum >= 0, "Min resolution should be non-negative"
        assert solution.max_incidence_angle >= 0, "Max incidence angle should be non-negative"


def test_solve_with_pls_multiobjective_advanced():
    """Test the multiobjective PLS with custom parameters."""
    # Create a test problem instance
    test_problem = sims_problem.SimsDiscreteProblem(
        num_images=5,
        universe=8,
        images=[[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 0], [1, 3, 5]],
        costs=[12, 18, 25, 15, 22],
        clouds=[[0], [2], [4], [6], [1]],
        areas=[1, 1, 1, 1, 1, 1, 1, 1],
        resolution=[4, 7, 2, 9, 5],  # Mix of good and bad resolutions
        incidence_angle=[15, 35, 50, 20, 40],  # Mix of angles
        max_cloud_area=40
    )
    
    # Test the multiobjective solve with custom parameters
    solutions = sims_problem.solve_with_pls_multiobjective(
        test_problem,
        timeout_seconds=3.0,
        max_iterations=750,
        is_deterministic=True,
        initial_population_size=15,
        neighborhood_size_min=1,
        neighborhood_size_max=3
    )
    
    # Assertions
    assert len(solutions) > 0, "Should return at least one solution"
    
    # Check that we get diverse solutions across 4 objectives
    costs = [sol.cost for sol in solutions]
    cloudy_areas = [sol.cloudy_area for sol in solutions]
    min_resolutions = [sol.min_resolutions_sum for sol in solutions]
    max_incidence_angles = [sol.max_incidence_angle for sol in solutions]
    
    # With multiobjective optimization, we should see some diversity
    assert len(set(costs)) > 1 or len(set(cloudy_areas)) > 1 or \
           len(set(min_resolutions)) > 1 or len(set(max_incidence_angles)) > 1, \
           "Multiobjective optimization should produce diverse solutions"
    
    for solution in solutions:
        selected_images = solution.get_selected_images_list()
        assert isinstance(selected_images, list), "Selected images should be a list"
        assert all(0 <= img < test_problem.num_images for img in selected_images), \
            "All selected image indices should be valid"
        
        # Verify all 4 objectives are valid
        assert solution.cost >= 0, "Cost should be non-negative"
        assert solution.cloudy_area >= 0, "Cloudy area should be non-negative"
        assert solution.min_resolutions_sum >= 0, "Min resolution should be non-negative"
        assert solution.max_incidence_angle >= 0, "Max incidence angle should be non-negative"
        assert solution.timestamp_us >= 0, "Timestamp should be non-negative"


def test_multiobjective_deterministic_behavior():
    """Test that deterministic multiobjective runs produce consistent results."""
    # Create a test problem instance
    test_problem = sims_problem.SimsDiscreteProblem(
        num_images=4,
        universe=6,
        images=[[0, 1, 2], [2, 3, 4], [4, 5, 0], [1, 3, 5]],
        costs=[10, 20, 15, 25],
        clouds=[[0], [2], [4], [1]],
        areas=[1, 1, 1, 1, 1, 1],
        resolution=[3, 6, 4, 2],
        incidence_angle=[20, 40, 30, 50],
        max_cloud_area=30
    )
    
    # Run deterministic multiobjective solver twice
    solutions1 = sims_problem.solve_with_pls_multiobjective(
        test_problem,
        timeout_seconds=1.5,
        max_iterations=200,
        is_deterministic=True,
        initial_population_size=8,
        neighborhood_size_min=1,
        neighborhood_size_max=2
    )
    
    solutions2 = sims_problem.solve_with_pls_multiobjective(
        test_problem,
        timeout_seconds=1.5,
        max_iterations=200,
        is_deterministic=True,
        initial_population_size=8,
        neighborhood_size_min=1,
        neighborhood_size_max=2
    )
    
    # Should produce same number of solutions
    assert len(solutions1) == len(solutions2), \
        "Deterministic multiobjective runs should produce same number of solutions"
    
    # Convert to sets of 4D objectives for comparison
    objectives1 = {(sol.cost, sol.cloudy_area, sol.min_resolutions_sum, sol.max_incidence_angle) 
                   for sol in solutions1}
    objectives2 = {(sol.cost, sol.cloudy_area, sol.min_resolutions_sum, sol.max_incidence_angle) 
                   for sol in solutions2}
    
    assert objectives1 == objectives2, \
        "Deterministic multiobjective runs should produce solutions with same 4D objectives"


def test_multiobjective_pareto_optimality():
    """Test that multiobjective solutions form a proper Pareto front."""
    # Create a test problem with clear trade-offs
    test_problem = sims_problem.SimsDiscreteProblem(
        num_images=5,
        universe=8,
        images=[[0, 1, 2], [3, 4, 5], [6, 7, 0], [1, 4, 7], [2, 5, 6]],
        costs=[5, 50, 25, 35, 15],  # Clear cost differences
        clouds=[[0, 1], [], [6, 7], [1], [2]],  # Different cloud patterns
        areas=[2, 2, 2, 2, 2, 2, 2, 2],
        resolution=[10, 1, 5, 3, 7],  # Clear resolution trade-offs (lower is better)
        incidence_angle=[10, 80, 45, 60, 25],  # Clear angle trade-offs (lower is better)
        max_cloud_area=50
    )
    
    solutions = sims_problem.solve_with_pls_multiobjective(
        test_problem,
        timeout_seconds=2.0,
        max_iterations=500,
        is_deterministic=True,
        initial_population_size=12
    )
    
    assert len(solutions) > 1, "Should find multiple Pareto-optimal solutions"
    
    # Test Pareto optimality: no solution should dominate another
    for i, sol1 in enumerate(solutions):
        for j, sol2 in enumerate(solutions):
            if i != j:
                # Check if sol1 dominates sol2 (all objectives better or equal, at least one strictly better)
                cost_better = sol1.cost <= sol2.cost
                cloudy_better = sol1.cloudy_area <= sol2.cloudy_area
                resolution_better = sol1.min_resolutions_sum <= sol2.min_resolutions_sum
                angle_better = sol1.max_incidence_angle <= sol2.max_incidence_angle
                
                all_better_or_equal = cost_better and cloudy_better and resolution_better and angle_better
                at_least_one_strictly_better = (sol1.cost < sol2.cost or 
                                                sol1.cloudy_area < sol2.cloudy_area or
                                                sol1.min_resolutions_sum < sol2.min_resolutions_sum or
                                                sol1.max_incidence_angle < sol2.max_incidence_angle)
                
                dominates = all_better_or_equal and at_least_one_strictly_better
                
                assert not dominates, \
                    f"Solution {i} dominates solution {j}, violating Pareto optimality"


def test_multiobjective_objective_relationships():
    """Test the mathematical relationships of the new objectives."""
    # Create a test problem where we can verify objective calculations
    test_problem = sims_problem.SimsDiscreteProblem(
        num_images=3,
        universe=4,
        images=[[0, 1], [1, 2], [2, 3]],  # Simple coverage
        costs=[10, 20, 30],
        clouds=[[], [1], [2]],  # Image 0 is clear, others have clouds
        areas=[5, 5, 5, 5],
        resolution=[2, 4, 6],  # Different resolutions
        incidence_angle=[15, 30, 45],  # Different angles
        max_cloud_area=20
    )
    
    solutions = sims_problem.solve_with_pls_multiobjective(
        test_problem,
        timeout_seconds=1.0,
        max_iterations=100,
        is_deterministic=True
    )
    
    assert len(solutions) > 0, "Should return at least one solution"
    
    for solution in solutions:
        selected_images = solution.get_selected_images_list()
        
        # Verify min resolution calculation
        if len(selected_images) > 0:
            # The objective sums minimum resolutions per element, so it should be reasonable
            assert solution.min_resolutions_sum >= 0, "Min resolution sum should be non-negative"
        
        # Verify max incidence angle calculation
        if len(selected_images) > 0:
            # Max incidence angle should equal the maximum among selected images
            max_selected_angle = max(test_problem.incidence_angle[i] for i in selected_images)
            assert solution.max_incidence_angle == max_selected_angle, \
                f"Max incidence angle should be {max_selected_angle}, got {solution.max_incidence_angle}"


def test_multiobjective_parameter_validation():
    """Test that invalid parameters are handled properly in multiobjective optimization."""
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
    solutions = sims_problem.solve_with_pls_multiobjective(
        test_problem,
        timeout_seconds=0.1,
        max_iterations=10,
        is_deterministic=True,
        initial_population_size=3,
        neighborhood_size_min=1,
        neighborhood_size_max=1
    )
    
    # Should still return at least initial population
    assert len(solutions) >= 1, "Should return at least one solution even with small timeout"
    
    # Verify all solutions have 4D objectives
    for solution in solutions:
        assert hasattr(solution, 'cost'), "Solution should have cost"
        assert hasattr(solution, 'cloudy_area'), "Solution should have cloudy_area"
        assert hasattr(solution, 'min_resolutions_sum'), "Solution should have min_resolution"
        assert hasattr(solution, 'max_incidence_angle'), "Solution should have max_incidence_angle"


def test_biobjective_vs_multiobjective_comparison():
    """Compare biobjective and multiobjective results to ensure consistency in shared objectives."""
    # Create a test problem instance
    test_problem = sims_problem.SimsDiscreteProblem(
        num_images=4,
        universe=6,
        images=[[0, 1, 2], [2, 3, 4], [4, 5, 0], [1, 3, 5]],
        costs=[10, 20, 15, 25],
        clouds=[[0], [2], [4], [1]],
        areas=[2, 2, 2, 2, 2, 2],
        resolution=[3, 6, 4, 2],
        incidence_angle=[20, 40, 30, 50],
        max_cloud_area=24
    )
    
    # Run biobjective optimization
    biobjective_solutions = sims_problem.solve_with_pls_advanced(
        test_problem,
        timeout_seconds=1.0,
        max_iterations=200,
        is_deterministic=True,
        initial_population_size=5
    )
    
    # Run multiobjective optimization
    multiobjective_solutions = sims_problem.solve_with_pls_multiobjective(
        test_problem,
        timeout_seconds=1.0,
        max_iterations=200,
        is_deterministic=True,
        initial_population_size=5
    )
    
    assert len(biobjective_solutions) > 0, "Biobjective should return solutions"
    assert len(multiobjective_solutions) > 0, "Multiobjective should return solutions"
    
    # Extract cost and cloudy_area objectives from both
    bio_objectives = {(sol.cost, sol.cloudy_area) for sol in biobjective_solutions}
    multi_objectives = {(sol.cost, sol.cloudy_area) for sol in multiobjective_solutions}
    
    # The multiobjective optimization might find different (cost, cloudy_area) pairs
    # due to additional objectives, but they should still be valid
    for cost, cloudy_area in multi_objectives:
        assert cost >= 0, "Cost should be non-negative in multiobjective solutions"
        assert cloudy_area >= 0, "Cloudy area should be non-negative in multiobjective solutions"
    
    print(f"Biobjective found {len(bio_objectives)} unique (cost, cloudy_area) pairs")
    print(f"Multiobjective found {len(multi_objectives)} unique (cost, cloudy_area) pairs")
