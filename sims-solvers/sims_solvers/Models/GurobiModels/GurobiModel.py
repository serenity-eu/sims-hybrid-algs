from abc import ABC, abstractmethod

import gurobipy as gp

from sims_solvers import constants
from sims_solvers.Models.GenericModel import GenericModel


class GurobiModel(GenericModel, ABC):

    @property
    def gurobi_solver_model(self) -> gp.Model:
        """Return the solver_model cast to gp.Model type."""
        assert isinstance(self.solver_model, gp.Model), f"Expected gp.Model, got {type(self.solver_model)}"
        return self.solver_model

    def set_solver_name(self):
        self.solver_name = constants.Solver.GUROBI.value

    @abstractmethod
    def review_objective_values(self, objective_values):
        pass
