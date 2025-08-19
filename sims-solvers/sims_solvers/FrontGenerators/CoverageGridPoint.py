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
        # get the best and worst values for each objective
        yield from self.get_best_worst_values()
        # convert problem to maximization problem
        self.convert_model_to_maximization()
        # declare the model
        self.set_augmecon2_objective_model()
        # Initializes the loop control variable for all constraint objectives (all except the first one)
        ef_array = []
        for i in range(1, len(self.nadir_objectives_values)):
            ef_array.append(self.nadir_objectives_values[i])
        # save previous solutions
        previous_solutions = set()
        previous_solution_information = []
        for solutions in self.front_solutions:
            objs_solution_values = solutions['objs']
            str_objs_solution_values = Saugmecon.convert_solution_value_to_str(objs_solution_values)
            previous_solutions.add(str_objs_solution_values)
        
        # Create interval managers for each constraint objective (starting from objective 1, not 0)
        ef_intervals = []
        for i in range(1, len(self.nadir_objectives_values)):
            min_interval = min(self.nadir_objectives_values[i], self.best_objective_values[i])
            max_interval = max(self.nadir_objectives_values[i], self.best_objective_values[i])
            ef_intervals.append(IntervalManager(min_interval+1, max_interval-1))

        # Initialize ef_array with middle points for all constraint objectives
        for i in range(len(ef_array)):
            ef_array[i] = int((self.best_objective_values[i+1] + self.nadir_objectives_values[i+1]) / 2)
        
        yield from self.coverage_loop(ef_array, previous_solutions, previous_solution_information, ef_intervals)

    def coverage_loop(self, ef_array, previous_solutions, previous_solution_information, ef_intervals):
        # For multi-objective, continue while any constraint objective has valid intervals
        iteration_count = 0
        max_iterations = 1000  # Safety limit to prevent infinite loops
        
        while (any(len(interval.intervals) > 0 for interval in ef_intervals) and 
               iteration_count < max_iterations):
            yield from self.coverage_most_inner_loop(ef_array, previous_solutions, previous_solution_information,
                                                     ef_intervals)
            iteration_count += 1
            
            # Update ef_array to next valid point from intervals
            updated = False
            for i, interval_manager in enumerate(ef_intervals):
                if len(interval_manager.intervals) > 0:
                    largest_interval = interval_manager.find_largest_interval()
                    if largest_interval is not None:
                        ef_array[i] = int((largest_interval[0] + largest_interval[1]) / 2)
                        updated = True
                        break
            
            if not updated:
                break

    def coverage_most_inner_loop(self, ef_array, previous_solutions, previous_solution_information, ef_intervals):
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
        # Update all constraint objectives
        for i in range(len(self.constraint_objectives)):
            self.adjust_parameter_ef_array(gamma, i, ef_array, one_solution, ef_intervals[i])

    def adjust_parameter_ef_array(self, gamma, id_constraint_objective, ef_array, one_solution, ef_interval):
        # check if the list one_solution is empty
        start_removal = ef_array[id_constraint_objective]
        new_max_interval = start_removal - 1
        if not one_solution:
            end_removal = ef_interval.max_value
        else:
            end_removal = min(one_solution[id_constraint_objective + 1], ef_interval.max_value)
        ef_interval.remove_interval(start_removal, end_removal)
        # update max_value
        if end_removal >= ef_interval.max_value:
            ef_interval.max_value = new_max_interval
        max_interval = ef_interval.find_largest_interval()
        if max_interval is not None:
            ef_array[id_constraint_objective] = int((max_interval[0] + max_interval[1]) / 2)
        else:
            ef_array[id_constraint_objective] = self.best_objective_values[id_constraint_objective + 1] + 1

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
                
                # For nadir calculation, use the worst values from other objectives when optimizing objective i
                obj_values = formatted_solution['objs']
                for j in range(num_objectives):
                    if i != j:
                        # Update nadir if this is worse than current nadir for objective j
                        if self.nadir_objectives_values[j] == 0 or obj_values[j] > self.nadir_objectives_values[j]:
                            self.nadir_objectives_values[j] = obj_values[j]
            else:
                raise TimeoutError(f"Timeout while optimizing objective {i}")
        
        # Yield all found extreme solutions
        for formatted_solution in formatted_solutions:
            if formatted_solution is not None:
                yield formatted_solution
            else:
                raise TimeoutError()

    def set_augmecon2_objective_model(self):
        self.constraint_objectives_lhs = self.solver.build_objective_e_constraint_augmecon2(
            self.best_objective_values, self.nadir_objectives_values, True)
        self.solver.set_optimization_sense("max")

    def update_objective_constraints(self, ef_array):
        for i in range(len(ef_array)):
            if self.constraint_objectives[i] != 0:
                self.solver.remove_constraint(self.constraint_objectives[i])
            self.constraint_objectives[i] = self.solver.add_constraints_eq(self.constraint_objectives_lhs[i],
                                                                           ef_array[i])

    def always_add_new_solutions_to_front(self):
        return False


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
