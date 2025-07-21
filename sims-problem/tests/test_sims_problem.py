"""
Test module for SIMS problem core functionality
"""
import pytest
import sims_problem
from sims_problem import solve_with_pls, SimsDiscreteProblem, Solution


def create_test_problem():
    """Helper function to create a test problem"""
    return SimsDiscreteProblem(
        num_images=3,
        universe=5,
        images=[[0, 1], [1, 2, 3], [3, 4]],
        costs=[10, 20, 15],
        clouds=[[0], [2], [4]],
        areas=[100, 150, 200, 120, 80],
        resolution=[100, 200, 150],
        incidence_angle=[45, 30, 60],
        max_cloud_area=500
    )


class TestSimsDiscreteProblem:
    """Test cases for SimsDiscreteProblem class"""
    
    def test_create_problem(self):
        """Test creating a new SIMS discrete problem"""
        problem = create_test_problem()
        assert problem.num_images == 3
        assert problem.universe == 5
        assert problem.max_cloud_area == 500
    
    def test_problem_properties(self):
        """Test problem property access"""
        problem = create_test_problem()
        
        # Test getters
        assert len(problem.costs) == 3
        assert len(problem.areas) == 5
        assert len(problem.resolution) == 3
        assert len(problem.incidence_angle) == 3
        
        # Test setters
        problem.max_cloud_area = 600
        assert problem.max_cloud_area == 600
    
    def test_to_dict_method(self):
        """Test the to_dict method"""
        problem = create_test_problem()
        result = problem.to_dict()
        
        assert isinstance(result, dict)
        assert "num_images" in result
        assert "universe" in result
        assert "images" in result
        assert "costs" in result
        assert "clouds" in result
        assert "areas" in result
        assert "resolution" in result
        assert "incidence_angle" in result
        assert "max_cloud_area" in result
        
        # Check values
        assert result["num_images"] == 3
        assert result["universe"] == 5
        assert result["max_cloud_area"] == 500


class TestSolution:
    """Test cases for Solution class"""
    
    def test_create_solution(self):
        """Test creating a new solution"""
        objectives = [1.5, 2.5, 3.5]
        decisions = [1, 2, 3, 4]
        solution = Solution(objectives, decisions)
        
        assert solution.objective_values == objectives
        assert solution.decision_variables == decisions
    
    def test_solution_setters(self):
        """Test setting solution properties"""
        solution = Solution([1.0, 2.0], [1, 2])
        
        new_objectives = [3.0, 4.0, 5.0]
        new_decisions = [5, 6, 7]
        
        solution.objective_values = new_objectives
        solution.decision_variables = new_decisions
        
        assert solution.objective_values == new_objectives
        assert solution.decision_variables == new_decisions
    
    def test_empty_solution(self):
        """Test creating solution with empty lists"""
        solution = Solution([], [])
        assert solution.objective_values == []
        assert solution.decision_variables == []


class TestSolveWithPls:
    """Test cases for solve_with_pls function"""
    
    def test_solve_basic(self):
        """Test basic solve functionality"""
        problem = create_test_problem()
        solutions = solve_with_pls(problem)
        
        # Check that we get a list of solutions
        assert isinstance(solutions, list)
        assert len(solutions) > 0
        
        # Check that all returned items are Solution objects
        for solution in solutions:
            assert isinstance(solution, Solution)
            assert isinstance(solution.objective_values, list)
            assert isinstance(solution.decision_variables, list)
    
    def test_solve_multiple_calls(self):
        """Test that multiple calls work consistently"""
        problem = create_test_problem()
        
        solutions1 = solve_with_pls(problem)
        solutions2 = solve_with_pls(problem)
        
        # Both calls should return solutions
        assert len(solutions1) > 0
        assert len(solutions2) > 0
        
        # For the placeholder implementation, results should be the same
        assert len(solutions1) == len(solutions2)
    
    def test_solve_different_problems(self):
        """Test solving different problem instances"""
        problem1 = create_test_problem()
        problem2 = SimsDiscreteProblem(
            num_images=2,
            universe=3,
            images=[[0], [1, 2]],
            costs=[5, 10],
            clouds=[[], [1]],
            areas=[50, 75, 100],
            resolution=[150, 300],
            incidence_angle=[30, 45],
            max_cloud_area=200
        )
        
        solutions1 = solve_with_pls(problem1)
        solutions2 = solve_with_pls(problem2)
        
        # Both should return valid results
        assert len(solutions1) > 0
        assert len(solutions2) > 0
    
    def test_solution_structure(self):
        """Test the structure of returned solutions"""
        problem = create_test_problem()
        solutions = solve_with_pls(problem)
        
        for solution in solutions:
            # Check that objectives are numeric
            for obj_val in solution.objective_values:
                assert isinstance(obj_val, (int, float))
            
            # Check that decision variables are integers
            for dec_var in solution.decision_variables:
                assert isinstance(dec_var, int)


class TestModuleStructure:
    """Test cases for module structure and imports"""
    
    def test_top_level_imports(self):
        """Test that top-level imports work correctly"""
        # These should not raise import errors
        from sims_problem import solve_with_pls, SimsDiscreteProblem, Solution
        
        # Check that the function is callable
        assert callable(solve_with_pls)
        assert callable(SimsDiscreteProblem)
        assert callable(Solution)
    
    def test_direct_access(self):
        """Test accessing functionality through direct imports"""
        problem = sims_problem.SimsDiscreteProblem(
            num_images=1,
            universe=2,
            images=[[0, 1]],
            costs=[10],
            clouds=[[]],
            areas=[50, 60],
            resolution=[100],
            incidence_angle=[45],
            max_cloud_area=100
        )
        solutions = sims_problem.solve_with_pls(problem)
        
        assert len(solutions) > 0
        assert isinstance(solutions[0], sims_problem.Solution)


class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_full_workflow(self):
        """Test a complete workflow from problem creation to solution"""
        # Create problem
        problem = create_test_problem()
        assert problem.num_images == 3
        
        # Convert to dict
        problem_dict = problem.to_dict()
        assert isinstance(problem_dict, dict)
        assert problem_dict["num_images"] == 3
        
        # Solve the problem
        solutions = solve_with_pls(problem)
        assert len(solutions) > 0
        
        # Verify solution properties
        for solution in solutions:
            assert len(solution.objective_values) > 0
            assert len(solution.decision_variables) > 0
            
            # For the placeholder implementation, we know the structure
            assert len(solution.objective_values) == 2  # Based on current implementation (cost, cloud_area)
            # Decision variables length depends on the problem size and algorithm
            # For our test problem with 3 images, we get 1 selected image per solution
            assert len(solution.decision_variables) >= 1  # At least 1 decision variable
    
    def test_multiple_problems_workflow(self):
        """Test workflow with multiple different problems"""
        problems = [
            create_test_problem(),
            SimsDiscreteProblem(
                num_images=2,
                universe=4,
                images=[[0, 1], [2, 3]],
                costs=[15, 25],
                clouds=[[0], [2]],
                areas=[80, 90, 100, 110],
                resolution=[120, 180],
                incidence_angle=[35, 55],
                max_cloud_area=300
            ),
            SimsDiscreteProblem(
                num_images=1,
                universe=3,
                images=[[1, 2]],
                costs=[30],
                clouds=[[]],
                areas=[60, 70, 80],
                resolution=[200],
                incidence_angle=[40],
                max_cloud_area=150
            ),
        ]
        
        all_solutions = []
        for problem in problems:
            solutions = solve_with_pls(problem)
            all_solutions.extend(solutions)
        
        # Should have solutions from all problems
        assert len(all_solutions) >= len(problems)
        
        # All should be valid Solution objects
        for solution in all_solutions:
            assert isinstance(solution, Solution)


if __name__ == "__main__":
    pytest.main([__file__])
