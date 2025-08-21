from abc import ABC

from ortools.sat.python import cp_model

from sims_solvers import constants
from sims_solvers.Models.GenericModel import GenericModel


class OrtoolsCPModel(GenericModel, ABC):

    def __init__(self):
        self.solution_variables = []
        self.solver_values = []

    @property
    def ortools_solver_model(self) -> cp_model.CpModel:
        """Return the solver_model cast to cp_model.CpModel type."""
        assert isinstance(self.solver_model, cp_model.CpModel), f"Expected cp_model.CpModel, got {type(self.solver_model)}"
        return self.solver_model

    def set_solver_name(self):
        self.solver_name = constants.Solver.ORTOOLS_PY.value

    def create_model(self):
        return cp_model.CpModel()