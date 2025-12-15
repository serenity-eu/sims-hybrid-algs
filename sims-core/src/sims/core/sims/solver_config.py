from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path


class SolverType(StrEnum):
    OR_TOOLS = "ortools-py"
    PLS = "pls"
    GUROBI = "gurobi"
    PYTHON_MILP = "python-milp"
    RUST_MILP = "rust-milp"
    CBC = "coin_cbc"
    HIGHS = "highs"
    SCIP = "scip"

    def __repr__(self) -> str:
        match self:
            case SolverType.OR_TOOLS:
                return "OR-tools"
            case SolverType.PLS:
                return "Pareto Local Search"
            case SolverType.GUROBI:
                return "Gurobi"
            case SolverType.PYTHON_MILP:
                return "Python MILP (Inlined)"
            case SolverType.RUST_MILP:
                return "Rust MILP"
            case SolverType.CBC:
                return "COIN-OR CBC"
            case SolverType.HIGHS:
                return "HiGHS"
            case SolverType.SCIP:
                return "SCIP"
            case _:
                return "Unknown"

    @staticmethod
    def from_str(solver_name: str):
        return SolverType(solver_name)


class FrontStrategy(StrEnum):
    GAVANELLI = "gavanelli"
    SAUGMECON = "saugmecon"
    GPBA_A = "gpba-a"
    ANEJA_NAIR = "aneja-nair"
    NON_APLICABLE = "None"

    def __repr__(self) -> str:
        match self:
            case FrontStrategy.GAVANELLI:
                return "Gavanelli"
            case FrontStrategy.SAUGMECON:
                return "SAUGMECON"
            case FrontStrategy.GPBA_A:
                return "GBPA-A"
            case FrontStrategy.ANEJA_NAIR:
                return "Aneja-Nair"
            case FrontStrategy.NON_APLICABLE:
                return "Non-Aplicable"
            case _:
                # Should never happen
                return "Unknown"

    @staticmethod
    def from_str(front_strategy_name: str):
        return FrontStrategy(front_strategy_name)


@dataclass(frozen=True)
class SolverConfig:
    solver_type: SolverType
    front_strategy: FrontStrategy
    timeout_s: int
    ratio_step: int = 20

    def to_dict(self) -> dict:
        return {
            "solver_type": str(self.solver_type),
            "front_strategy": str(self.front_strategy),
            "timeout_s": self.timeout_s,
            "ratio_step": self.ratio_step,
        }

    def to_json(self, json_path: Path):
        json_path.write_text(json.dumps(self.to_dict(), indent=4))

    @staticmethod
    def from_json(json_path: Path):
        json_data = json.loads(json_path.read_text())
        return SolverConfig(
            solver_type=SolverType.from_str(json_data["solver_type"]),
            front_strategy=FrontStrategy.from_str(json_data["front_strategy"]),
            timeout_s=json_data["timeout_s"],
            ratio_step=json_data["ratio_step"],
        )


@dataclass(frozen=True)
class TwoPhaseSolverConfig:
    exact_solver_type: SolverType
    front_strategy: FrontStrategy
    timeout_s: int
    ratio: tuple[int, int]
