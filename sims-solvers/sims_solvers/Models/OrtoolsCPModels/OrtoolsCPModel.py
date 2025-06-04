from abc import ABC

from ortools.sat.python import cp_model

from sims_solvers import constants
from sims_solvers.Models.GenericModel import GenericModel


class OrtoolsCPModel(GenericModel, ABC):

    def __init__(self):
        self.solution_variables = []
        self.solver_values = []

    def set_solver_name(self):
        self.solver_name = constants.Solver.ORTOOLS_PY.value

    def create_model(self):
        return cp_model.CpModel()