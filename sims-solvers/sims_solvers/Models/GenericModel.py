from __future__ import annotations

from abc import ABC, abstractmethod

from sims_solvers.Instances.InstanceGeneric import InstanceGeneric


class GenericModel(ABC):

    def __init__(self, instance: InstanceGeneric | None = None) -> None:
        self.instance: InstanceGeneric | None = instance
        self.solver_model: object  # Will be cp_model.CpModel or gp.Model or similar
        self.solver_name: str = ""
        self.objectives: list[object] = []  # Solver-specific objective variables/expressions
        self.constraints: list[object] = []  # Solver-specific constraint objects
        
        self.assert_right_instance(instance)
        self.solver_model = self.create_model()
        self.set_solver_name()
        self.get_data_from_instance()
        self.add_variables_to_model()
        self.define_objectives()
        self.add_constraints_to_model()
        self.add_necessary_solver_configuration()

    @abstractmethod
    def create_model(self) -> object:
        """Create and return the solver-specific model object."""
        pass

    @abstractmethod
    def set_solver_name(self) -> None:
        """Set the solver name for this model."""
        pass

    @abstractmethod
    def problem_name(self) -> str:
        """Return the name of the problem this model solves."""
        pass

    def assert_right_instance(self, instance: InstanceGeneric | None) -> None:
        """Assert that the provided instance is compatible with this model."""
        if self.instance is not None and self.instance.problem_name != self.problem_name():
            raise Exception(self.message_incorrect_instance())

    def message_incorrect_instance(self) -> str:
        """Return an error message for incorrect instance type."""
        return f"Incorrect instance {self.instance} for model {self}"

    @abstractmethod
    def get_data_from_instance(self) -> None:
        """Extract and process data from the problem instance."""
        pass

    @abstractmethod
    def add_variables_to_model(self) -> None:
        """Add decision variables to the solver model."""
        pass

    @abstractmethod
    def add_constraints_to_model(self) -> None:
        """Add constraints to the solver model."""
        pass

    @abstractmethod
    def define_objectives(self) -> None:
        """Define the objective functions for the model."""
        pass

    @abstractmethod
    def is_a_minimization_model(self) -> bool:
        """Return True if all objectives should be minimized, False if maximized."""
        # all objectives should be minimized or maximized
        pass

    @abstractmethod
    def get_solution_values(self) -> object:
        """Extract and return the solution values from the solved model."""
        pass

    @abstractmethod
    def get_nadir_bound_estimation(self) -> list[float]:
        """Return estimated nadir bounds for the objectives."""
        pass

    @abstractmethod
    def get_ref_points_for_hypervolume(self) -> list[float]:
        """Return reference points for hypervolume calculation."""
        pass

    @abstractmethod
    def is_numerically_possible_augment_objective(self) -> bool:
        """Return True if the solver supports augmenting objectives numerically."""
        pass

    @abstractmethod
    def add_necessary_solver_configuration(self) -> None:
        """Apply any solver-specific configuration needed for the model."""
        pass




