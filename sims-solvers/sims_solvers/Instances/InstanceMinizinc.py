from typing import Optional

from minizinc import Instance, Solver

from sims_solvers.Instances.InstanceGeneric import InstanceGeneric


class InstanceMinizinc(Instance, InstanceGeneric):
    def __init__(self, solver: Solver, model: Optional = None, problem_name: Optional[str] = ""):
        super().__init__(solver, model)
        InstanceGeneric.__init__(self, is_minizinc=True, problem_name=problem_name)
