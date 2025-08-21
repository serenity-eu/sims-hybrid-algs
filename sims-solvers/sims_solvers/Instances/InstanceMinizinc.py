from __future__ import annotations

from minizinc import Instance, Solver

from sims_solvers.Instances.InstanceGeneric import InstanceGeneric


class InstanceMinizinc(Instance, InstanceGeneric):
    def __init__(self, solver: Solver, model=None, problem_name: str | None = ""):
        super().__init__(solver, model)
        InstanceGeneric.__init__(self, is_minizinc=True, problem_name=problem_name)
