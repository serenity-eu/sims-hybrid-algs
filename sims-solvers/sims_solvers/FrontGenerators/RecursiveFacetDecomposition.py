import logging
from collections import deque

import numpy as np

from sims_solvers.FrontGenerators.FrontGeneratorStrategy import FrontGeneratorStrategy

log = logging.getLogger(__name__)


class RecursiveFacetDecomposition(FrontGeneratorStrategy):
    """
    Dichotomic search for multi-objective MILP nondominated extreme points.

    Implements the weighted-sum facet-exploration approach from:
      Przybylski, Klamroth & Lacour, "A Simple and Efficient Dichotomic Search Algorithm
      for Multi-Objective Mixed Integer Linear Programmes", arXiv:1911.08937 (2019).

    The algorithm (Algorithm 3 / Bd_Dichotomy in the paper) iteratively:
      1. Initialises with nondominated extreme points of all p (p-1)-objective subproblems.
      2. Maintains the convex hull of known supported extreme points S.
      3. For each unexplored nondominated facet F (with strictly-positive normal λ ∈ R^p_>),
         solves the weighted-sum scalarisation with objective λ^T z.
      4. If λ^T y = λ^T y' (y the new solution, y' any vertex of F) the facet is confirmed
         as a facet of conv(YSN); otherwise the new point is added to S and the convex hull
         is updated.
      5. Terminates when all nondominated facets of conv(S) have been explored.

    For p=2 this reduces exactly to the standard Aneja-Nair / dichotomic-search method.

    IMPLEMENTATION SIMPLIFICATIONS (compared to the paper):
    - Initialisation: we solve each of the p objectives independently to obtain p extreme
      solutions, rather than recursively solving all 2^p (p-k)-objective subproblems.
      This is sufficient for p=2 but may miss boundary extreme points for p≥3 when the
      (p-1)-objective frontier contains points not dominated individually.
    - Convex hull: we maintain an explicit list of k-simplices (cells) via BFS rather than
      running an incremental convex hull algorithm (e.g. CGAL).  For problems in general
      position every nondominated facet of conv(S) is a k-simplex, so the behaviour is
      equivalent to the paper for generic instances.
    - These simplifications do not affect the p=2 case (exact), and are practical
      approximations for p=3 and p=4.

    Only works with solvers that implement set_weighted_sum_objective (e.g. Gurobi,
    OR-Tools CP-SAT via integer-scaled weights).
    """

    def set_multiply_solution_by_minus_one(self):
        return False if self.solver.model.is_a_minimization_model() else True

    def always_add_new_solutions_to_front(self):
        return True

    @staticmethod
    def compute_weight_for_cell(solutions):
        """
        Given k solutions in R^k, find w (>=0, sums to 1) such that w·z_i = w·z_j for all i,j.

        Solves the k×k linear system:
            (z_i - z_0) · w = 0   for i = 1 .. k-1
            sum(w)              = 1

        Returns None if the matrix is near-singular or any weight component is negative
        (cell lies outside the valid weight simplex).
        """
        k = len(solutions)
        A = [
            [float(solutions[i][j] - solutions[0][j]) for j in range(k)]
            for i in range(1, k)
        ]
        A.append([1.0] * k)
        b = [0.0] * (k - 1) + [1.0]

        try:
            A_np = np.array(A, dtype=float)
            b_np = np.array(b, dtype=float)
            if np.linalg.cond(A_np) > 1e12:
                return None  # near-singular: skip
            w = np.linalg.solve(A_np, b_np)
            if np.any(w < -1e-9):
                return None  # cell outside weight simplex: skip
            w = np.maximum(w, 0.0)
            total = w.sum()
            if total < 1e-12:
                return None
            w /= total
            return tuple(float(wi) for wi in w)
        except np.linalg.LinAlgError:
            return None

    def solve(self):
        k = len(self.solver.model.objectives)
        if k < 2:
            return

        # Phase 1: k extreme solutions at the simplex corners
        extreme_objs = []
        for sol in self._get_extreme_solutions():
            extreme_objs.append(tuple(int(v) for v in sol["objs"]))
            yield sol

        distinct = list(
            dict.fromkeys(extreme_objs)
        )  # deduplicated, insertion-order preserved
        if len(distinct) < k:
            log.warning(
                "RecursiveFacetDecomposition: only %d distinct extreme solutions for %d objectives; "
                "degenerate initial cell skipped.",
                len(distinct),
                k,
            )
            return

        # Phase 2: BFS over k-simplices
        found = set(extreme_objs)
        explored = set()
        queue = deque()

        initial_cell = tuple(sorted(distinct))
        if self.compute_weight_for_cell([list(s) for s in initial_cell]) is not None:
            queue.append(initial_cell)
        elif k >= 3:
            # The p extreme solutions don't form a visible facet (negative normal).
            # Seed the BFS with a uniform-weight solve, then form k sub-cells from
            # it paired with k-1 of the extreme solutions.
            uniform_w = [1.0 / k] * k
            self.solver.set_weighted_sum_objective(uniform_w)
            try:
                elapsed = self.get_solver_solution_for_timeout(
                    optimize_not_satisfy=True
                )
            except TimeoutError:
                raise
            if not self.solver.status_infeasible():
                seed_sol = self.process_feasible_solution(elapsed)
                seed_obj = tuple(int(v) for v in seed_sol["objs"])
                if seed_obj not in found:
                    found.add(seed_obj)
                    self.front_solutions.append(seed_sol)
                    yield seed_sol
                distinct_list = list(distinct)
                for i in range(k):
                    candidate = tuple(
                        sorted(distinct_list[:i] + [seed_obj] + distinct_list[i + 1 :])
                    )
                    if len(set(candidate)) == k:
                        queue.append(candidate)

        while queue:
            cell = queue.popleft()
            if cell in explored:
                continue
            explored.add(cell)

            weights = self.compute_weight_for_cell([list(s) for s in cell])
            if weights is None:
                log.debug(
                    "RecursiveFacetDecomposition: degenerate cell skipped: %s", cell
                )
                continue

            self.solver.set_weighted_sum_objective(weights)
            elapsed = self.get_solver_solution_for_timeout(optimize_not_satisfy=True)

            if self.solver.status_infeasible():
                continue

            new_sol = self.process_feasible_solution(elapsed)
            new_obj = tuple(int(v) for v in new_sol["objs"])

            if new_obj in set(cell):
                continue  # terminal: no new solution in this region

            if new_obj not in found:
                found.add(new_obj)
                yield new_sol

            # Split: replace each of the k vertices with new_obj
            cell_list = list(cell)
            for i in range(k):
                candidate = tuple(
                    sorted(cell_list[:i] + [new_obj] + cell_list[i + 1 :])
                )
                if len(set(candidate)) == k and candidate not in explored:
                    queue.append(candidate)

    def _get_extreme_solutions(self):
        """Solve each objective independently to obtain extreme supported solutions."""
        k = len(self.solver.model.objectives)
        for i in range(k):
            fmt_sol, obj_val = self.optimize_single_objectives(
                self.model_optimization_sense, i
            )
            if fmt_sol is None:
                raise TimeoutError(f"Timeout while optimizing objective {i}")
            self.front_solutions.append(fmt_sol)
            yield fmt_sol
