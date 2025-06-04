from abc import ABC

from sims_solvers import constants
from sims_solvers.Models.GenericModel import GenericModel


class MultiobjectiveKnapsackGenericModel(GenericModel, ABC):

    def __init__(self, instance):
        self.select_item = None
        self.number_items = 0
        super().__init__(instance)

    def problem_name(self):
        return constants.Problem.MULTI_OBJECTIVE_KNAPSACK_PROBLEM.value

    def is_a_minimization_model(self):
        return False

    def get_nadir_bound_estimation(self):
        nadir_objectives = [0]*len(self.objectives)
        return nadir_objectives

    def get_ref_points_for_hypervolume(self):
        ref_points = [0]*len(self.objectives)
        return ref_points
