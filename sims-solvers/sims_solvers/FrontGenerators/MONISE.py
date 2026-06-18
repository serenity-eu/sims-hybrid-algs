import logging

from sims_solvers.FrontGenerators.FrontGeneratorStrategy import FrontGeneratorStrategy

log = logging.getLogger(__name__)


class MONISE(FrontGeneratorStrategy):
    """
    Many-Objective NISE (MONISE): exact implementation per Raimundo, Ferreira & Von Zuben (2017).

    Reference:
      Raimundo, Ferreira & Von Zuben, "An Extension of the Non-Inferior Set Estimation
      Algorithm for Many Objectives", arXiv:1709.00797.

    The algorithm has three phases:

    1. Initialisation (Section 4.4.1):
       Solve each of the m individual single-objective problems to obtain m extreme solutions
       x^1, ..., x^m together with their unit weighting vectors w^1 = e_1, ..., w^m = e_m.
       The utopian point z^utopian = (min_i f_1(x^i), ..., min_i f_m(x^i)).

    2. Iterative step (Section 4.4.2 / Definition 9):
       At each iteration, solve the auxiliary MILP (Definition 9) to obtain the next
       weight vector w^{L+1} and the current estimation gap µ.  Then solve the main
       weighted-sum problem with w^{L+1} to obtain solution x^{L+1}.

    3. Stopping criterion (Section 4.4.3):
       Stop when µ ≤ mu_stop (gap below threshold) or timeout.

    The auxiliary MILP (Definition 9) maximises µ = v - w^T r, where:
      - w ∈ R^m   : new weight vector (Σw_j = 1, w ≥ 0)
      - r ∈ R^m   : outer approximation of the Pareto frontier
      - v ∈ R     : scalar representing w^T r_inner (inner approx value)
      - κ ∈ R^L   : convex-combination weights (Σκ_i = 1, κ ≥ 0)
      - κ^B ∈{0,1}^L, ν ∈ R^m, ν^B ∈{0,1}^m : complementary-slackness binary variables
      - µ ∈ R     : gap (objective)

    Constraints (using w_i, f(x^i) as known constants from previous iterations):
      [1]  w_i^T r ≥ w_i^T f(x^i)          ∀i  (outer approx)
      [2]  r - Σ_i κ_i f(x^i) - ν + µ·1 = 0   (KKT stationarity)
      [3]  r ≥ z^utopian
      [4]  0 ≤ w_j ≤ ν_j^B                  ∀j  (comp-slack: w or ν active)
      [5]  0 ≤ ν_j ≤ (1 - ν_j^B)·ν̄_j       ∀j  (big-M)
      [6]  w^T f(x^i) - v ≥ 0               ∀i  (inner approx)
      [7]  w^T f(x^i) - v ≤ κ_i^B·κ̄_i      ∀i  (comp-slack: κ or gap active, big-M)
      [8]  κ_i ≤ 1 - κ_i^B                  ∀i
      [9]  κ_i ≥ 0                           ∀i
      [10] Σ_j w_j = 1
      [11] Σ_i κ_i = 1

    The auxiliary MILP (Definition 9) is solved via OR-Tools pywraplp (SCIP or CBC
    backend).  The main weighted-sum problems may be solved by any supported solver.
    """

    # Gap threshold: iterations stop when µ ≤ mu_stop.
    # For integer-objective MILPs a value < 1 makes the method exact.
    MU_STOP_DEFAULT = 1e-4

    # Whether to enumerate co-optimal solutions on each convex-hull face.
    # Disabled by default: co-optionals add depth on one face but waste budget
    # that could be spent exploring new faces — bad for anytime/hybrid use.
    ENUMERATE_COOPTIMAL = False

    def set_multiply_solution_by_minus_one(self):
        return False if self.solver.model.is_a_minimization_model() else True

    def always_add_new_solutions_to_front(self):
        return True

    # ------------------------------------------------------------------
    # Public solve generator
    # ------------------------------------------------------------------

    def solve(self, mu_stop=None):
        if mu_stop is None:
            mu_stop = self.MU_STOP_DEFAULT

        m = len(self.solver.model.objectives)
        if m < 2:
            return

        # Phase 1: m single-objective solutions + utopian point
        solutions = []  # list[list[float]]  — objective vectors f(x^i)
        weight_vecs = []  # list[list[float]]  — weight vectors w^i used for each x^i

        for i in range(m):
            weights = [0.0] * m
            weights[i] = 1.0
            self.solver.set_weighted_sum_objective(weights)
            elapsed = self.get_solver_solution_for_timeout(optimize_not_satisfy=True)
            if self.solver.status_infeasible():
                log.warning("MONISE: single-objective problem %d infeasible", i)
                return
            sol = self.process_feasible_solution(elapsed)
            self.front_solutions.append(sol)
            yield sol
            solutions.append([float(v) for v in sol["objs"]])
            weight_vecs.append(weights)

        # utopian = component-wise minimum across all known solutions
        z_utopian = [
            min(solutions[i][j] for i in range(len(solutions))) for j in range(m)
        ]

        found = {tuple(int(v) for v in s) for s in solutions}

        # Phase 2: iterative MONISE
        while True:
            w_next, mu = self._solve_definition9(solutions, weight_vecs, z_utopian)

            if w_next is None:
                log.debug("MONISE: auxiliary MILP infeasible/error — stopping")
                break

            log.debug("MONISE: µ=%.6f  w=%s", mu, w_next)

            if mu <= mu_stop:
                log.debug("MONISE: gap µ=%.6f ≤ µ_stop=%.6f — converged", mu, mu_stop)
                break

            self.solver.set_weighted_sum_objective(w_next)
            elapsed = self.get_solver_solution_for_timeout(optimize_not_satisfy=True)
            if self.solver.status_infeasible():
                log.debug("MONISE: main problem infeasible for w=%s — stopping", w_next)
                break

            new_sol = self.process_feasible_solution(elapsed)
            new_obj = tuple(int(v) for v in new_sol["objs"])

            solutions.append([float(v) for v in new_sol["objs"]])
            weight_vecs.append(list(w_next))

            if new_obj not in found:
                found.add(new_obj)
                yield new_sol

            if self.ENUMERATE_COOPTIMAL:
                # Enumerate all co-optimal solutions on the same convex hull face.
                # w_next defines the supporting hyperplane; any solution with the same
                # w_next · z value is co-optimal and belongs to the same face.
                v_star = sum(w_next[j] * new_sol["objs"][j] for j in range(m))
                nogoods = [self.solver.add_objective_nogood(new_sol["objs"])]
                try:
                    while True:
                        elapsed = self.get_solver_solution_for_timeout(
                            optimize_not_satisfy=True
                        )
                        if self.solver.status_infeasible():
                            break
                        raw_objs = self.solver.get_solution_objective_values()
                        v_co = sum(w_next[j] * raw_objs[j] for j in range(m))
                        if v_co > v_star + 1e-6:
                            break
                        co_sol = self.process_feasible_solution(elapsed)
                        co_obj = tuple(int(v) for v in co_sol["objs"])
                        solutions.append([float(v) for v in co_sol["objs"]])
                        weight_vecs.append(list(w_next))
                        if co_obj not in found:
                            found.add(co_obj)
                            yield co_sol
                        nogoods.append(self.solver.add_objective_nogood(co_sol["objs"]))
                finally:
                    for ng in nogoods:
                        self.solver.remove_constraint(ng)

    # ------------------------------------------------------------------
    # Auxiliary MILP (Definition 9)
    # ------------------------------------------------------------------

    def _solve_definition9(self, solutions, weight_vecs, z_utopian):
        """
        Solve the auxiliary MILP from Definition 9 of Raimundo et al. (2017).

        Parameters
        ----------
        solutions  : list of L objective vectors f(x^i) (minimisation form)
        weight_vecs: list of L weight vectors w^i used to generate each solution
        z_utopian  : list of m component-wise minima (utopian point)

        Returns
        -------
        (w, mu) : next weight vector and gap, or (None, 0.0) on failure
        """
        from ortools.linear_solver import pywraplp

        m = len(z_utopian)
        L = len(solutions)

        # Big-M constants (generous, derived from observed objective ranges)
        f_flat = [solutions[i][j] for i in range(L) for j in range(m)]
        f_max_global = max(f_flat)
        f_min_global = min(f_flat)
        obj_range = max(f_max_global - f_min_global, 1.0)

        kappa_bar = obj_range  # upper bound for w^T f(x^i) - v
        nu_bar = [
            max(solutions[i][j] for i in range(L)) - z_utopian[j] + obj_range
            for j in range(m)
        ]

        # pywraplp uses SCIP for MIP (continuous + binary variables)
        slv = pywraplp.Solver.CreateSolver("SCIP")
        if slv is None:
            slv = pywraplp.Solver.CreateSolver("CBC")
        if slv is None:
            raise RuntimeError(
                "MONISE: no suitable MIP backend found in OR-Tools (tried SCIP, CBC)"
            )
        slv.SuppressOutput()
        inf = slv.infinity()

        # Decision variables
        w = [slv.NumVar(0.0, 1.0, f"w_{j}") for j in range(m)]
        r = [
            slv.NumVar(float(z_utopian[j]), float(f_max_global + obj_range), f"r_{j}")
            for j in range(m)
        ]
        v = slv.NumVar(float(f_min_global) - obj_range, float(f_max_global), "v")
        kappa = [slv.NumVar(0.0, 1.0, f"kappa_{i}") for i in range(L)]
        kappa_B = [slv.BoolVar(f"kappaB_{i}") for i in range(L)]
        nu = [slv.NumVar(0.0, float(nu_bar[j]), f"nu_{j}") for j in range(m)]
        nu_B = [slv.BoolVar(f"nuB_{j}") for j in range(m)]
        mu = slv.NumVar(-obj_range, obj_range, "mu")

        # Objective: maximize µ
        obj = slv.Objective()
        obj.SetCoefficient(mu, 1.0)
        obj.SetMaximization()

        # [1] Outer approximation: Σ_j w_i[j]*r[j] ≥ Σ_j w_i[j]*f_i[j]  ∀i  (w_i are constants)
        for i in range(L):
            wi = weight_vecs[i]
            fi = solutions[i]
            rhs = float(sum(wi[j] * fi[j] for j in range(m)))
            ct = slv.Constraint(rhs, inf)
            for j in range(m):
                ct.SetCoefficient(r[j], float(wi[j]))

        # [2] Stationarity: r_j - Σ_i κ_i*f_i[j] - ν_j + µ = 0  ∀j
        for j in range(m):
            ct = slv.Constraint(0.0, 0.0)
            ct.SetCoefficient(r[j], 1.0)
            for i in range(L):
                ct.SetCoefficient(kappa[i], -float(solutions[i][j]))
            ct.SetCoefficient(nu[j], -1.0)
            ct.SetCoefficient(mu, 1.0)

        # [4] w_j ≤ ν_j^B  (i.e. w_j - ν_j^B ≤ 0)
        for j in range(m):
            ct = slv.Constraint(-inf, 0.0)
            ct.SetCoefficient(w[j], 1.0)
            ct.SetCoefficient(nu_B[j], -1.0)

        # [5] ν_j ≤ (1 - ν_j^B)*ν̄_j  →  ν_j + ν̄_j*ν_j^B ≤ ν̄_j
        for j in range(m):
            ct = slv.Constraint(-inf, float(nu_bar[j]))
            ct.SetCoefficient(nu[j], 1.0)
            ct.SetCoefficient(nu_B[j], float(nu_bar[j]))

        # [6] w^T f(x^i) - v ≥ 0  ∀i
        for i in range(L):
            fi = solutions[i]
            ct = slv.Constraint(0.0, inf)
            for j in range(m):
                ct.SetCoefficient(w[j], float(fi[j]))
            ct.SetCoefficient(v, -1.0)

        # [7] w^T f(x^i) - v ≤ κ_i^B * κ̄_i  →  w^T f(x^i) - v - κ̄_i*κ_i^B ≤ 0  ∀i
        for i in range(L):
            fi = solutions[i]
            ct = slv.Constraint(-inf, 0.0)
            for j in range(m):
                ct.SetCoefficient(w[j], float(fi[j]))
            ct.SetCoefficient(v, -1.0)
            ct.SetCoefficient(kappa_B[i], -float(kappa_bar))

        # [8] κ_i ≤ 1 - κ_i^B  →  κ_i + κ_i^B ≤ 1  ∀i
        for i in range(L):
            ct = slv.Constraint(-inf, 1.0)
            ct.SetCoefficient(kappa[i], 1.0)
            ct.SetCoefficient(kappa_B[i], 1.0)

        # [10] Σ_j w_j = 1
        ct = slv.Constraint(1.0, 1.0)
        for j in range(m):
            ct.SetCoefficient(w[j], 1.0)

        # [11] Σ_i κ_i = 1
        ct = slv.Constraint(1.0, 1.0)
        for i in range(L):
            ct.SetCoefficient(kappa[i], 1.0)

        status = slv.Solve()

        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            log.debug("MONISE: auxiliary MILP status %d", status)
            return None, 0.0

        w_val = [max(0.0, w[j].solution_value()) for j in range(m)]
        total = sum(w_val)
        if total < 1e-12:
            return None, 0.0
        w_val = [wi / total for wi in w_val]  # re-normalise after small numerical drift
        mu_val = float(mu.solution_value())

        return w_val, mu_val


class MONISEWithCoopt(MONISE):
    """MONISE with co-optimal face enumeration enabled.

    Use when a complete or near-complete Pareto front is needed.
    For hybrid/anytime use prefer plain MONISE (co-opt disabled).
    """

    ENUMERATE_COOPTIMAL = True
