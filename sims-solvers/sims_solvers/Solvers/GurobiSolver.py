import gurobipy as gp

from sims_solvers import constants
from sims_solvers.Solvers.Solver import Solver


class GurobiSolver(Solver):

    def __init__(self, model, statistics, threads, free_search=True):
        super().__init__(model, statistics, threads, free_search)
        self.latest_solution = None
        self.latest_values_decision_variables = None
        self.use_lazy_constraints = False
        self.auxiliary_variables_objs_smaller_equal_at_least_one_smaller = None

    def assert_right_solver(self, model):
        if model.solver_name != constants.Solver.GUROBI.value:
            raise Exception(self.message_incorrect_solver())

    def set_solver(self):
        return None

    def set_threads(self, threads):
        self.model.solver_model.Params.Threads = threads

    def get_complete_solution(self):
        return self.model.solver_model

    def get_nodes_solution(self, solution):
        return solution.NodeCount
        # return solution.SolCount

    def get_solution_objective_values(self):
        one_solution = []
        for i in range(len(self.model.objectives)):
            if type(self.model.objectives[i]) is gp.Var:
                one_solution.append(self.model.objectives[i].x)
            else:
                one_solution.append(self.model.objectives[i].getValue())
        # make sure the values of the objectives are rounded down to the nearest integer
        one_solution = [int(round(x, 0)) for x in one_solution]
        self.model.review_objective_values(one_solution)
        return one_solution

    def set_minimization(self):
        self.model.solver_model.ModelSense = gp.GRB.MINIMIZE

    def set_maximization(self):
        self.model.solver_model.ModelSense = gp.GRB.MAXIMIZE

    def set_time_limit(self, timeout_seconds):
        self.model.solver_model.Params.TimeLimit = timeout_seconds

    def reset(self):
        self.model.solver_model.reset(1)

    def get_status(self):
        return self.model.solver_model.Status

    def status_time_limit(self):
        return self.model.solver_model.Status == gp.GRB.TIME_LIMIT

    def status_infeasible(self):
        return self.model.solver_model.Status == gp.GRB.INFEASIBLE

    def build_objective_e_constraint_saugmecon(self, range_array, augmentation):
        obj = self.model.objectives[0]
        delta = 0.001  # delta should be between 0.001 and 0.000001
        rest_obj = 0
        for i in range(len(range_array)):
            rest_obj += self.model.objectives[i + 1] / range_array[i]
        if augmentation:
            obj = obj + (delta * rest_obj)
        self.set_single_objective(obj)

    def build_objective_e_constraint_augmecon2(self, best_constrain_obj_list, nadir_constrain_obj_list, augmentation, main_obj_index=0):
        """
        Build ε-constraint formulation with specified main objective.
        
        Args:
            best_constrain_obj_list: Best values for constraint objectives
            nadir_constrain_obj_list: Nadir values for constraint objectives  
            augmentation: Whether to use slack variable augmentation
            main_obj_index: Index of objective to optimize (others become constraints)
        
        Returns:
            List of constraint objective expressions
        """
        num_objectives = len(self.model.objectives)
        if num_objectives < 2:
            raise Exception("The augmecon2 requires at least 2 objectives.")
        
        if main_obj_index < 0 or main_obj_index >= num_objectives:
            raise Exception(f"main_obj_index {main_obj_index} is out of range [0, {num_objectives-1}]")
        
        constraint_objectives = []
        if augmentation:
            delta = 0.01  # delta should be between 0.001 and 0.000001. In the paper of  Mesquita-Cunha et al. 2021 it is set to 0.01 for computational experiments.
            
            # Create slack variables for all constraint objectives (all except main_obj_index)
            slack_vars = []
            constraint_indices = [i for i in range(num_objectives) if i != main_obj_index]
            
            for i in constraint_indices:
                max_s = abs(best_constrain_obj_list[i] - nadir_constrain_obj_list[i])
                s = self.model.solver_model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=max_s, name=f"s{i+1}")
                slack_vars.append(s)
            # objective ranges
            obj_range = []
            for i in constraint_indices:
                range_val = abs(best_constrain_obj_list[i] - nadir_constrain_obj_list[i])
                if range_val == 0:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.critical(f"DIVISION BY ZERO: Objective {i} has zero range! "
                                  f"best={best_constrain_obj_list[i]}, nadir={nadir_constrain_obj_list[i]}")
                obj_range.append(range_val)
            
            # Main objective: selected objective + delta * sum_{k=2,..p}(10^(k-1)*slack[k] / range[k])
            slack_range_sum = 0
            for i in range(len(constraint_indices)):
                if obj_range[i] == 0:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.critical(f"SKIPPING objective {constraint_indices[i]} in slack calculation due to zero range")
                    continue  # Skip this objective entirely to avoid division by zero
                slack_range_sum += slack_vars[i]/(10 ** i * obj_range[i])

            obj = self.model.objectives[main_obj_index] + (delta * slack_range_sum)
            self.set_single_objective(obj)
            
            # Constraint objectives: each constraint objective minus its corresponding slack variable
            for j, i in enumerate(constraint_indices):
                constraint_objectives.append(self.model.objectives[i] - slack_vars[j])
        else:
            # Without augmentation: specified objective is main, others are constraints
            self.set_single_objective(self.model.objectives[main_obj_index])
            constraint_indices = [i for i in range(num_objectives) if i != main_obj_index]
            for i in constraint_indices:
                constraint_objectives.append(self.model.objectives[i])
        
        return constraint_objectives


    def change_objective_sense(self, id_objective):
        self.model.objectives[id_objective] = -self.model.objectives[id_objective]

    def set_single_objective(self, objective_expression):
        self.model.solver_model.setObjective(objective_expression)

    def set_weighted_sum_objective(self, weights):
        weighted_sum = sum(w * o for w, o in zip(weights, self.model.objectives))
        self.set_single_objective(weighted_sum)

    def add_constraints_eq(self, constraint, rhs):
        new_constraint = self.model.solver_model.addConstr(constraint == rhs)
        return new_constraint

    def add_constraints_leq(self, constraint, rhs):
        new_constraint = self.model.solver_model.addConstr(constraint <= rhs)
        return new_constraint

    def add_constraints_geq(self, constraint, rhs):
        new_constraint = self.model.solver_model.addConstr(constraint >= rhs)
        return new_constraint

    def remove_constraint(self, constraint):
        self.model.solver_model.remove(constraint)

    def opt_one_objective_or_satisfy(self, optimize_not_satisfy=True, verbose=False):
        if optimize_not_satisfy:
            self.model.solver_model.optimize()
        else:
            self.run_satisfaction()

    def run_satisfaction(self):
        self.model.solver_model.Params.solutionLimit = 1
        self.model.solver_model.Params.MIPFocus = 1
        self.model.solver_model.setObjective(0)
        # self.model.solver_model.Params.Cuts = 3
        if self.use_lazy_constraints:
            self.model.solver_model.Params.lazyConstraints = 1
            self.latest_solution = None
            self.latest_values_decision_variables = None
            self.model.solver_model._objectives_val = self.model.objectives_val
            self.model.solver_model._solution_vars = self.model.solution_variables
            # todo add code to differentiate between possible different lazy constraints
            self.model.solver_model.optimize(
                lambda model, where: self.add_unsatisfaction_constraint_callback(model, where))
        else:
            self.model.solver_model.optimize()

    def perform_lexicographic_optimization(self, verbose=False):
        priority = len(self.model.objectives)
        if self.model.solver_model.IsMultiObj == 0:
            # set the objectives
            for i in range(len(self.model.objectives)):
                self.model.solver_model.setObjectiveN(self.model.objectives[i], i,
                                                      priority - self.lexicographic_obj_order[i])
        else:
            # update the priorities
            for i in range(len(self.model.objectives)):
                self.model.solver_model.params.ObjNumber = i
                self.model.solver_model.ObjNPriority = priority - self.lexicographic_obj_order[i]
        self.model.solver_model.optimize()

    def deactivate_lexicographic_optimization(self):
        if self.model.solver_model.IsMultiObj != 0:
            self.lexicographic_obj_order = []
            self.model.solver_model.NumObj = 0
            self.model.solver_model.update()

    def add_or_all_objectives_constraint(self, rhs, id_constraint=0):
        y = self.model.solver_model.addVars(len(self.model.objectives), vtype=gp.GRB.BINARY,
                                            name=f"temp_y_{id_constraint}")
        self.model.solver_model.addConstr(gp.quicksum(y) == 1)
        if self.model.is_a_minimization_model():
            rhs = [rhs[i] - 1 for i in range(len(rhs))]
        else:
            rhs = [rhs[i] + 1 for i in range(len(rhs))]
        big_m = self.get_big_m_for_or_all_objectives(rhs)
        for i in range(len(self.model.objectives)):
            if self.model.is_a_minimization_model():
                if self.can_big_m_introduce_problems(big_m[i]):
                    self.model.solver_model.addConstr((y[i] == 1) >> (self.model.objectives[i] <= rhs[i]),
                                                      name=f"indicator_const{id_constraint}_{i}")
                    self.model.solver_model.addConstr((y[i] == 0) >> (self.model.objectives[i] <= rhs[i] + big_m[i]),
                                                      name=f"indicator_const{id_constraint}_{i}")
                else:
                    self.model.solver_model.addConstr(self.model.objectives[i] <=
                                                      rhs[i] + (big_m[i] * (1 - y[i])))
            else:
                if self.can_big_m_introduce_problems(big_m[i]):
                    self.model.solver_model.addConstr((y[i] == 1) >> (self.model.objectives[i] >= rhs[i]),
                                                      name=f"indicator_const{id_constraint}_{i}")
                    self.model.solver_model.addConstr((y[i] == 0) >> (self.model.objectives[i] >= rhs[i] - big_m[i]),
                                                      name=f"indicator_const{id_constraint}_{i}")
                else:
                    self.model.solver_model.addConstr(self.model.objectives[i] >=
                                                      rhs[i] - (big_m[i] * (1 - y[i])))

    def objs_smaller_equal_at_least_one_smaller(self, obj_constraints_lhs, rhs, id_constraint=0):
        new_constraints = []
        for i in range(len(obj_constraints_lhs)):
            new_constraints.append(self.model.solver_model.addConstr(
                obj_constraints_lhs[i] <= rhs[i] - self.auxiliary_variables_objs_smaller_equal_at_least_one_smaller[i]))
        return new_constraints

    # todo create this method for all solvers in constraint solvers it will be empty
    def create_variable_for_constraint_objs_smaller_equal_at_least_one_smaller(self):
        if self.auxiliary_variables_objs_smaller_equal_at_least_one_smaller is None:
            self.auxiliary_variables_objs_smaller_equal_at_least_one_smaller = self.model.solver_model.addVars(
                len(self.model.objectives), vtype=gp.GRB.BINARY)
            auxiliary_constraint = self.model.solver_model.addConstr(gp.quicksum(
                self.auxiliary_variables_objs_smaller_equal_at_least_one_smaller) >= 1)

    def get_big_m_for_or_all_objectives(self, rhs):
        big_m = []
        nadir_objectives = self.model.get_nadir_bound_estimation()
        for i in range(len(rhs)):
            big_m.append(abs(nadir_objectives[i] - rhs[i]))
        return big_m

    def can_big_m_introduce_problems(self, big_m):
        if big_m * self.model.solver_model.Params.IntFeasTol >= 1:
            return True
        return False

    def add_unsatisfaction_constraint_callback(self, model, where):
        if where == gp.GRB.Callback.MIPSOL:
            # Get the solution
            solution_variables_value = model.cbGetSolution(model._solution_vars)
            obj_model = model.cbGetSolution(model._objectives_val)
            bad_solution_generated = False
            for i in range(len(obj_model)):
                if not obj_model[i].is_integer():
                    obj_model[i] = round(obj_model[i])
            if self.latest_solution is None:
                self.latest_solution = self.model.get_nadir_bound_estimation()
            for i in range(len(obj_model)):
                if obj_model[i] > self.latest_solution[i]:
                    bad_solution_generated = True
                    break
            if not bad_solution_generated:
                self.latest_solution = obj_model
                self.latest_values_decision_variables = solution_variables_value
            for i in range(len(obj_model)):
                model.cbLazy(model._objectives_val[i] <= self.latest_solution[i] -
                             self.auxiliary_variables_objs_smaller_equal_at_least_one_smaller[i])

    def lazy_constraints_possible(self):
        return True
