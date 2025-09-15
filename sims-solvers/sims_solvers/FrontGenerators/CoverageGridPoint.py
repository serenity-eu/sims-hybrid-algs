from sims_solvers.FrontGenerators.FrontGeneratorStrategy import FrontGeneratorStrategy
from sims_solvers.FrontGenerators.Saugmecon import Saugmecon


class CoverageGridPoint(FrontGeneratorStrategy):
    """
    This class implements coverage grid point based representation (GPBA-A) algorithm described in the paper
    'New ϵ−constraint methods for multi-objective integer linear programming: A Pareto front representation approach'
    DOI: 10.1016/j.ejor.2022.07.044
    This algorithm tries to represent all the areas of the Pareto front by minimizing the maximum distance between two
    consecutive solutions in the Pareto front.
    """
    def __init__(self, solver, timer):
        super().__init__(solver, timer)
        self.constraint_objectives_lhs = None
        self.constraint_objectives = [0] * (len(self.solver.model.objectives) - 1)
        self.is_a_minimization_model_originally = False

    def set_multiply_solution_by_minus_one(self):
        if self.model_optimization_sense == "min":
            return True
        return False

    def solve(self):
        """
        Implements GPBA-A algorithm with objective rotation as described in the paper:
        'performs a single run for every iteration and for each objective function'
        """
        # get the best and worst values for each objective. todo consider computing best and worst only for the objective variables, e.g. all except main_obj_index
        yield from self.get_best_worst_values()
        # convert problem to maximization problem
        self.convert_model_to_maximization()
        
        # todo in the original paper only one objective is optimized, rotation is tricky, could lead to missing some points
        # num_objectives = len(self.solver.model.objectives)
        # Run GPBA-A for each objective as the main one (complete objective rotation)
        # for main_obj_index in range(num_objectives):
        #     print(f"🎯 GPBA-A: Running iteration {main_obj_index + 1}/{num_objectives} with objective {main_obj_index} as main objective")
        #     yield from self.solve_with_main_objective(main_obj_index)
        yield from self.solve_with_main_objective(0)
    
    def solve_with_main_objective(self, main_obj_index):
        """
        Run GPBA-A algorithm with specified objective as the main one.
        
        Args:
            main_obj_index: Index of objective to optimize (others become constraints)
        """
        # declare the model with the specified main objective
        self.set_augmecon2_objective_model(main_obj_index)
        
        # Determine constraint objective indices (all except main_obj_index)
        num_objectives = len(self.solver.model.objectives)
        constraint_indices = [i for i in range(num_objectives) if i != main_obj_index]
        
        # Initializes the loop control variable for all constraint objectives
        ef_array = []
        for i in constraint_indices:
            ef_array.append(self.nadir_objectives_values[i])
        
        # Update constraint_objectives array to match number of constraint objectives
        self.constraint_objectives = [0] * len(constraint_indices)
        
        # save previous solutions
        previous_solutions = set()
        previous_solution_information = []
        for solutions in self.front_solutions:
            objs_solution_values = solutions['objs']
            str_objs_solution_values = Saugmecon.convert_solution_value_to_str(objs_solution_values)
            previous_solutions.add(str_objs_solution_values)
        
        # Create interval managers for each constraint objective
        ef_intervals = []
        for i in constraint_indices:
            ef_intervals.append(self.create_interval(i))

        # Initialize ef_array with the nadir points for all constraint objectives
        for j, i in enumerate(constraint_indices):
            ef_array[j] = int(self.nadir_objectives_values[i])

        # Initialize the relative worst values
        rwv = [0] * len(constraint_indices)
        for i in range(len(constraint_indices)):
            rwv[i] = self.best_objective_values[constraint_indices[i]]

        yield from self.coverage_loop(ef_array, rwv, previous_solutions, previous_solution_information, ef_intervals, constraint_indices)

    def coverage_loop(self, ef_array, rwv, previous_solutions, previous_solution_information, ef_intervals, constraint_indices):
        """
        Main coverage loop for GPBA-A algorithm.
        
        Args:
            ef_array: Array of constraint values
            rwv: Relative worst values for constraint objectives
            previous_solutions: Set of previous solution strings
            previous_solution_information: List of previous solution information
            ef_intervals: List of interval managers for constraint objectives
            constraint_indices: Indices of objectives that are constraints (not main objective)
        """
        # For multi-objective, continue while any constraint objective has valid intervals
        iteration_count = 0
        # todo the algorithm should finish when all the points are found, max_iterations is only useful if you want to use them as a stopping condition
        max_iterations = 1000  # Safety limit to prevent infinite loops
        
        while ef_array[0] <= self.best_objective_values[constraint_indices[0]]:
            yield from self.coverage_most_inner_loop(ef_array, rwv, previous_solutions, previous_solution_information,
                                                     ef_intervals, constraint_indices)
            iteration_count += 1

    def coverage_most_inner_loop(self, ef_array, rwv, previous_solutions, previous_solution_information, ef_intervals, constraint_indices):
        """
        Inner loop of coverage algorithm that solves each constraint configuration.
        
        Args:
            ef_array: Array of constraint values
            rwv: Relative worst values for constraint objectives
            previous_solutions: Set of previous solution strings
            previous_solution_information: List of previous solution information
            ef_intervals: List of interval managers for constraint objectives
            constraint_indices: Indices of objectives that are constraints (not main objective)
        """
        gamma = 1  # with the value of 1, the algorithm will find the whole Pareto front if run enough time
        previous_solution_relaxation, previous_solution_values = \
            Saugmecon.search_previous_solutions_relaxation(ef_array, previous_solution_information, min_sense=False)
        if previous_solution_relaxation:
            if type(previous_solution_values) is str:
                # the previous solution is infeasible
                one_solution = []
            else:
                one_solution = previous_solution_values
        else:
            # solve the problem
            # update right-hand side values (rhs) for the objective constraints
            self.update_objective_constraints(ef_array)
            solution_sec = self.get_solver_solution_for_timeout(optimize_not_satisfy=True)
            if self.solver.status_infeasible():
                Saugmecon.save_solution_information(ef_array, "infeasible", previous_solution_information,
                                                    min_sense=False)
                one_solution = []
            else:
                objectives_solution_values = self.solver.get_solution_objective_values()
                str_objectives_solution_values = Saugmecon.convert_solution_value_to_str(objectives_solution_values)
                if str_objectives_solution_values in previous_solutions:
                    one_solution = self.solver.get_solution_objective_values()
                else:
                    # update previous_solutions
                    previous_solutions.add(str_objectives_solution_values)
                    formatted_solution = self.process_feasible_solution(solution_sec)
                    one_solution = formatted_solution["objs"]
                    Saugmecon.save_solution_information(ef_array, one_solution, previous_solution_information,
                                                        min_sense=False)

                    if self.is_a_minimization_model_originally:
                        formatted_solution.solution.objs = [-1 * i for i in formatted_solution.solution.objs]
                    yield formatted_solution
                    # todo comment below the line below is for testing purposes, uncomment when necessary
                    try:
                        self.solver.model.assert_solution([abs(obj) for obj in one_solution], formatted_solution["solution_values"])
                    except Exception as e:
                        print(e)
                        self.solver.model.print_solution_values_model_values()
                        calculated_cost = self.solver.model.calculate_cost(formatted_solution["solution_values"])
                        if calculated_cost != abs(one_solution[0]):
                            print(f"Cost error. Expected: {one_solution[0]}, calculated: {calculated_cost}")

                        calculated_cloud_uncovered = self.solver.model.calculate_cloud_uncovered(formatted_solution["solution_values"])
                        if calculated_cloud_uncovered != abs(one_solution[1]):
                            print(f"Cloud covered error. Expected: {one_solution[1]}, calculated: {calculated_cloud_uncovered}")
        # Update relative worst values array
        sol_obj_id = None
        id_interval = -1
        if len(one_solution) > 0:
            for i in range(len(rwv)):
                rwv[i] = min(rwv[i], one_solution[constraint_indices[i]])
            sol_obj_id = one_solution[constraint_indices[id_interval]]

        # Update all constraint objectives. NOTE: An objective x (with 0 < x < p-1, where p-1 is the index of the last objective) is only updated when the ef_array[x-1] > best_value[x-1]
        ef_intervals[id_interval] = self.adjust_parameter_ef_array(id_interval, ef_array, sol_obj_id, ef_intervals[id_interval], constraint_indices, gamma)
        for i in range(len(constraint_indices)-1, 0, -1):
            if ef_array[i] > self.best_objective_values[constraint_indices[i]]:
                ef_array[i] = self.nadir_objectives_values[constraint_indices[i]]
                rwv[i] = self.best_objective_values[constraint_indices[i]]
                id_interval = i - 1
                if sol_obj_id is not None:
                    sol_obj_id = one_solution[constraint_indices[id_interval]]
                ef_intervals[id_interval] = self.adjust_parameter_ef_array(id_interval, ef_array, sol_obj_id,
                                               ef_intervals[id_interval], constraint_indices, gamma)
            else:
                break

    def adjust_parameter_ef_array(self, id_constraint_objective, ef_array, sol_obj_k, ef_interval, constraint_indices, gamma=1):
        """
        Adjust the ef_array parameter based on the solution found.
        
        Args:
            id_constraint_objective: Index in the constraint objective array (0-based)
            ef_array: Array of constraint values
            sol_obj_k: Objective k value for the solution found (or None if infeasible)
            ef_interval: Interval manager for this constraint objective
            constraint_indices: Indices of objectives that are constraints (not main objective)
            gamma: Coverage parameter
        """
        # check if the list one_solution is empty
        start_removal = ef_array[id_constraint_objective]
        new_max_interval = start_removal - 1
        if sol_obj_k is None:
            end_removal = ef_interval.max_value
        else:
            # Map from constraint objective index to actual objective index
            end_removal = min(sol_obj_k, ef_interval.max_value)
        ef_interval.remove_interval(start_removal, end_removal)
        # update max_value
        if end_removal >= ef_interval.max_value:
            ef_interval.max_value = new_max_interval
        max_interval = ef_interval.find_largest_interval()
        actual_obj_index = constraint_indices[id_constraint_objective]
        if ef_array[id_constraint_objective] == self.nadir_objectives_values[actual_obj_index]:
            ef_array[id_constraint_objective] = self.best_objective_values[actual_obj_index]
        else:
            if max_interval is not None:
                ef_array[id_constraint_objective] = int((max_interval[0] + max_interval[1]) / 2)
            else:
                ef_array[id_constraint_objective] = self.best_objective_values[actual_obj_index] + 1
                # reinitialize the interval manager to avoid stopping the algorithm
                ef_interval = self.create_interval(actual_obj_index)
        return ef_interval

    def convert_model_to_maximization(self):
        if not self.solver.model.is_a_minimization_model():
            return  # the model is already a maximization model
        self.is_a_minimization_model_originally = True
        # multiply nadir and best values by -1
        self.best_objective_values = [-1 * x for x in self.best_objective_values]
        self.nadir_objectives_values = [-1 * x for x in self.nadir_objectives_values]
        # multiply objectives by -1
        for i in range(len(self.solver.model.objectives)):
            self.solver.change_objective_sense(i)

    def get_best_worst_values(self):
        """Get extreme points by optimizing each objective individually for any number of objectives."""
        num_objectives = len(self.solver.model.objectives)
        self.best_objective_values = [0] * num_objectives
        self.nadir_objectives_values = [0] * num_objectives
        formatted_solutions = []
        
        # For each objective, find its extreme value (ideal point)
        for i in range(num_objectives):
            formatted_solution, objective_val = self.optimize_single_objectives(self.model_optimization_sense, i)
            if formatted_solution is not None and objective_val is not None:
                self.best_objective_values[i] = int(objective_val)
                formatted_solutions.append(formatted_solution)
                self.front_solutions.append(formatted_solution)
            else:
                raise TimeoutError(f"Timeout while optimizing objective {i}")
        # Get the nadir values by optimizing each objective in the opposite sense. Do it after getting the best values,
        # because the best values are potential Pareto points, but the nadir values are not. If the time is short, it is better to get the best values.
        nadir_optimization_sense = "min" if self.model_optimization_sense == "max" else "max"
        for i in range(num_objectives):
            sols, objective_val = self.optimize_single_objectives(nadir_optimization_sense, i)
            if objective_val is not None:
                self.nadir_objectives_values[i] = int(objective_val)
            else:
                raise TimeoutError(f"Timeout while optimizing objective {i}")

        # Yield all found extreme solutions
        for formatted_solution in formatted_solutions:
            if formatted_solution is not None:
                yield formatted_solution
            else:
                raise TimeoutError()

    def set_augmecon2_objective_model(self, main_obj_index=0):
        """
        Set up the augmecon2 objective model with specified main objective.
        
        Args:
            main_obj_index: Index of objective to optimize (others become constraints)
        """
        self.constraint_objectives_lhs = self.solver.build_objective_e_constraint_augmecon2(
            self.best_objective_values, self.nadir_objectives_values, True, main_obj_index)
        self.solver.set_optimization_sense("max")

    def update_objective_constraints(self, ef_array):
        for i in range(len(ef_array)):
            if self.constraint_objectives[i] != 0:
                self.solver.remove_constraint(self.constraint_objectives[i])
            self.constraint_objectives[i] = self.solver.add_constraints_eq(self.constraint_objectives_lhs[i],
                                                                           ef_array[i])

    def always_add_new_solutions_to_front(self):
        return False

    def create_interval(self, i):
        min_interval = min(self.nadir_objectives_values[i], self.best_objective_values[i])
        max_interval = max(self.nadir_objectives_values[i], self.best_objective_values[i])
        # todo Vlad review if it should be min_interval or min_interval+1 and max_interval or max_interval-1
        return IntervalManager(min_interval + 1, max_interval - 1)


class IntervalManager:
    def __init__(self, min_value, max_value):
        self.intervals = set()  # Using set to manage unique intervals
        self.min_value = min_value
        self.max_value = max_value
        self.add_interval(min_value, max_value)

    def add_interval(self, start, end):
        """
        Adds a new interval, merging with existing ones if overlapping.
        """
        new_intervals = set()
        to_add = (start, end)
        for interval in self.intervals:
            if interval[1] < start or interval[0] > end:  # No overlap
                new_intervals.add(interval)
            else:  # Merge overlapping intervals
                to_add = (min(to_add[0], interval[0]), max(to_add[1], interval[1]))
        new_intervals.add(to_add)
        self.intervals = new_intervals

    def remove_interval(self, start, end):
        """
        Removes an interval, adjusting or splitting existing intervals as necessary.
        """
        new_intervals = set()
        for interval in self.intervals:
            if interval[1] < start or interval[0] > end:  # No overlap, keep interval
                new_intervals.add(interval)
            else:
                # Adjust or split interval if there's any overlap
                if interval[0] < start:
                    new_intervals.add((interval[0], start - 1))
                if interval[1] > end:
                    new_intervals.add((end + 1, interval[1]))
        self.intervals = new_intervals  # Update intervals

    def find_largest_interval(self):
        """
        Finds and returns the largest interval by length.
        """
        if not self.intervals:
            return None  # No intervals to compare
        return max(self.intervals, key=lambda x: x[1] - x[0])

    def print_intervals(self):
        """
        Prints all intervals sorted by their start value.
        """
        for interval in sorted(list(self.intervals)):
            print(interval)
