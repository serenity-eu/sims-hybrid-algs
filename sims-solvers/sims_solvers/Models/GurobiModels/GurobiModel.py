from abc import ABC, abstractmethod

from sims_solvers import constants
from sims_solvers.Models.GenericModel import GenericModel


class GurobiModel(GenericModel, ABC):

    def set_solver_name(self):
        self.solver_name = constants.Solver.GUROBI.value

    @abstractmethod
    def review_objective_values(self, objective_values):
        pass
