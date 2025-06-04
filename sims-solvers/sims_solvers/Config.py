import argparse
import multiprocessing
import os
from pathlib import Path


class Config:
    """Configuration class for the multi-objective constraint programming.
    It parses the commandline arguments and initializes the temporary and result directories."""

    def __init__(
        self,
        minizinc_data: bool,
        instance_name: str,
        data_sets_folder: Path,
        input_mzn: Path,
        dzn_dir: Path,
        solver_name: str,
        problem_name: str,
        front_strategy: str,
        solver_timeout_sec: int,
        summary_filename: str,
        solver_search_strategy: str,
        fzn_optimisation_level: int,
        cores: int,
        threads: int,
    ):
        self.minizinc_data: bool = minizinc_data
        self.data_name: str = instance_name
        self.data_sets_folder: str = str(data_sets_folder)
        self.input_mzn: str = str(input_mzn)
        self.minizinc_model: str = input_mzn.stem
        self.input_dzn: str = str(dzn_dir / f"{instance_name}.dzn")
        self.solver_name: str = solver_name
        self.problem_name: str = problem_name
        self.front_strategy: str = front_strategy
        self.solver_timeout_sec: int = solver_timeout_sec
        self.summary_filename: str = summary_filename
        self.solver_search_strategy: str = solver_search_strategy
        self.fzn_optimisation_level: int = fzn_optimisation_level
        self.cores: int = cores
        self.threads: int = threads

    @staticmethod
    def from_args(cls) -> "Config":
        parser = argparse.ArgumentParser(
            prog="sims_solvers",
            description="Multi-objective constraint programming. This program computes a Pareto front of an optimization constraint problem.",
        )
        parser.add_argument("--instance_name", required=True)
        parser.add_argument("--minizinc_data", required=True)
        parser.add_argument("--model_mzn", required=False)
        parser.add_argument("--dzn_dir", required=False)
        parser.add_argument("--problem_name", required=True)
        parser.add_argument("--solver_name", required=True)
        parser.add_argument("--front_strategy", required=True)
        parser.add_argument("--solver_timeout_sec", required=True, type=int)
        parser.add_argument("--summary", required=True)
        parser.add_argument("--cores", type=int)
        parser.add_argument(
            "--solver_search_strategy", required=True
        )  # Must be "free" or the name of a CP strategy (only for information purposes, the strategy must be described in the model).
        parser.add_argument("--fzn_optimisation_level", required=True, type=int)
        args = parser.parse_args()
        args.dzn_dir = cls.clean_dir_name(args.dzn_dir)
        return cls(
            minizinc_data=bool(int(args.minizinc_data)),
            data_name=args.instance_name,
            data_sets_folder=args.dzn_dir,
            input_mzn=args.model_mzn,
            minizinc_model=os.path.basename(args.model_mzn)[:-4],
            input_dzn=args.dzn_dir + "/" + args.instance_name + ".dzn",
            solver_name=args.solver_name,
            problem_name=args.problem_name,
            front_strategy=args.front_strategy,
            solver_timeout_sec=args.solver_timeout_sec,
            summary_filename=args.summary,
            solver_search_strategy=args.solver_search_strategy,
            fzn_optimisation_level=args.fzn_optimisation_level,
            cores=args.cores,
            threads=None,
        )

    @staticmethod
    def clean_dir_name(directory):
        """Remove the last '/' if it exists."""
        if directory[-1] == "/":
            directory = directory[:-1]
        return directory

    def init_statistics(self, statistics):
        statistics["instance"] = self.data_name
        statistics["problem"] = self.problem_name
        statistics["solver_name"] = self.solver_name
        statistics["front_strategy"] = self.front_strategy
        statistics["solver_search_strategy"] = self.solver_search_strategy
        statistics["fzn_optimisation_level"] = self.fzn_optimisation_level
        statistics["threads"] = self.threads
        statistics["cores"] = self.cores
        statistics["solver_timeout_sec"] = self.solver_timeout_sec
        statistics["minizinc_model"] = self.minizinc_model

    def uid(self):
        """Unique identifier for this experiment."""
        return (
            self.data_name
            + "_"
            + self.problem_name
            + "_"
            + self.solver_name
            + "_"
            + self.front_strategy
            + "_"
            + self.minizinc_model
            + "_"
            + self.solver_search_strategy
            + "_"
            + str(self.solver_timeout_sec)
            + "_"
            + str(self.fzn_optimisation_level)
            + "_"
            + str(self.cores)
        )

    def initialize_cores(self, solver, check_solver=True):
        """If the solver supports parallelization, use twice the number of available cores. By default, we assume that it does."""
        if not check_solver:
            print(
                "Check if the solver supports parallelization. By default, we assume that it does."
            )
        if not check_solver or "-p" in solver.stdFlags:
            if self.cores is None:
                self.cores = multiprocessing.cpu_count()
            self.threads = self.cores * 2
        else:
            self.cores = 1
            self.threads = 1
