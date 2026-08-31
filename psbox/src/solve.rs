//! The scalarisations the search issues, and the deadline that bounds them.
//!
//! Everything funnels through [`scalarise`] so the backend, the time limit and
//! the status classification are handled in one place. Status is the part that
//! carries the guarantee: a solve stopped at its time limit yields a feasible
//! but *unproven* point, and the algorithm may only admit points whose solve
//! proved optimality.

use crate::model::{Config, Feasibility, Problem, SolveResult, SolveStatus, objective_to_f64};
use good_lp::{Expression, Variable};
// The bound rows a session installs are built with this; without a backend
// there is no model to install them in.
#[cfg(feature = "gurobi")]
use good_lp::constraint;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Wall-clock budget shared by every solve in a run.
pub struct Budget {
    start: Instant,
    total: Option<Duration>,
    per_solve: Option<Duration>,
}

impl Budget {
    #[must_use]
    pub fn new(config: &Config) -> Self {
        Self::new_with(config.deadline, config.per_solve)
    }

    /// A budget from an explicit deadline and per-solve cap.
    #[must_use]
    pub fn new_with(total: Option<Duration>, per_solve: Option<Duration>) -> Self {
        Self {
            start: Instant::now(),
            total,
            per_solve,
        }
    }

    #[must_use]
    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// Time left overall; `None` when unbounded, zero when the run is over.
    #[must_use]
    pub fn remaining(&self) -> Option<Duration> {
        self.total
            .map(|t| t.checked_sub(self.start.elapsed()).unwrap_or_default())
    }

    #[must_use]
    pub fn exhausted(&self) -> bool {
        self.remaining().is_some_and(|r| r.is_zero())
    }

    /// Budget for the next solve: the lesser of what remains and the per-solve cap.
    #[must_use]
    pub fn next_solve(&self) -> Option<Duration> {
        match (self.remaining(), self.per_solve) {
            (Some(r), Some(p)) => Some(r.min(p)),
            (Some(r), None) => Some(r),
            (None, p) => p,
        }
    }
}

/// A Gurobi model held across many solves.
///
/// Every scalarisation this crate issues differs from the last only in its
/// objective and in a few bounds, yet building a fresh model for each one
/// measured at 0.65s on `lagos_nigeria_150` -- against solves that often finish
/// in three. Nearly all of that is marshalling rather than anything Gurobi
/// does: 27905 constraints cloned out of the problem and rebuilt term by term,
/// roughly 22us each. Batching the rows into one API call changed nothing,
/// which is what established the cost was on this side of the boundary.
///
/// So the model is built once and edited in place. The structural constraints
/// go in at construction; every objective bound the search can impose has a row
/// of its own, added non-binding and given a real right-hand side only when a
/// solve needs it. Re-solving an edited model also lets Gurobi keep its
/// presolved copy, its basis and its cut pool, which a fresh model discards.
///
/// The rows are, for each objective `j`: `f_j <= u_j` and `-f_j <= -l_j`. Both
/// are relaxed to infinity between solves, so a solve sees exactly the bounds
/// it asked for and none of the previous one's.
#[cfg(feature = "gurobi")]
pub struct Session<'a> {
    problem: &'a Problem,
    model: good_lp::solvers::gurobi::GurobiProblem,
    upper: Vec<good_lp::constraint::ConstraintReference>,
    lower: Vec<good_lp::constraint::ConstraintReference>,
    /// Each objective's constant term, which its bound rows do not carry.
    ///
    /// A row built from `f_j <= x` is stored by the solver as
    /// `linear_j <= x - c_j`: the constant moves to the right-hand side at
    /// construction. Setting that right-hand side later therefore bounds the
    /// *linear part*, not the objective, and a bound of `x` silently becomes
    /// `f_j <= x + c_j`. These objectives carry constants -- a maximised one
    /// is negated, which puts its whole range there -- so the offset has to be
    /// reapplied on every set.
    constants: Vec<f64>,
}

/// A bound wide enough that Gurobi treats the row as absent.
///
/// Gurobi's own infinity threshold; anything at or beyond it is not a
/// restriction.
#[cfg(feature = "gurobi")]
const FREE: f64 = 1e30;

#[cfg(feature = "gurobi")]
impl<'a> Session<'a> {
    /// Build the model once.
    #[must_use]
    pub fn new(problem: &'a Problem) -> Self {
        use good_lp::solvers::SolverModel;
        let mut model = problem
            .variables
            .clone()
            .minimise(problem.objectives[0].clone())
            .using(good_lp::solvers::gurobi::gurobi)
            .with_all(problem.constraints.iter().cloned());
        tighten_tolerances(&mut model);

        // One relaxed row per bound the search can impose. Added here so their
        // right-hand sides can be set later without touching the model's
        // structure, which is what keeps Gurobi's warm start usable.
        let mut upper = Vec::with_capacity(problem.num_objectives());
        let mut lower = Vec::with_capacity(problem.num_objectives());
        let mut constants = Vec::with_capacity(problem.num_objectives());
        for objective in &problem.objectives {
            upper.push(model.add_constraint(constraint!(objective.clone() <= FREE)));
            lower.push(model.add_constraint(constraint!(-objective.clone() <= FREE)));
            constants.push(good_lp::IntoAffineExpression::constant(objective));
        }
        Self {
            problem,
            model,
            upper,
            lower,
            constants,
        }
    }

    /// The problem this session was built from.
    #[must_use]
    pub const fn problem(&self) -> &'a Problem {
        self.problem
    }

    /// Minimise `objective` subject to `lower[j] <= f_j <= upper[j]`.
    ///
    /// A bound of infinity means "no bound"; pass [`f64::INFINITY`] for the
    /// objectives a particular solve does not constrain.
    pub fn solve(
        &mut self,
        objective: Expression,
        lower: &[f64],
        upper: &[f64],
        timeout: Option<Duration>,
    ) -> SolveResult {
        use good_lp::Solution as _;
        use good_lp::solvers::{
            ModelWithMutableObjective, ModelWithMutableRhs, ObjectiveDirection, ReusableModel,
            SolutionStatus,
        };

        // A zero budget means the deadline already passed; do not start a solve.
        if timeout.is_some_and(|d| d.is_zero()) {
            return SolveResult {
                status: SolveStatus::Deadline,
                values: None,
                objective_value: None,
            };
        }

        self.model
            .set_objective(objective, ObjectiveDirection::Minimisation);
        for index in 0..self.problem.num_objectives() {
            let hi = upper.get(index).copied().unwrap_or(f64::INFINITY);
            let lo = lower.get(index).copied().unwrap_or(f64::NEG_INFINITY);
            // `linear_j <= hi - c_j` is `f_j <= hi`.
            let c = self.constants[index];
            self.model.set_rhs(
                self.upper[index],
                if hi.is_finite() { hi - c } else { FREE },
            );
            // `-linear_j <= c_j - lo` is `f_j >= lo`.
            self.model.set_rhs(
                self.lower[index],
                if lo.is_finite() { c - lo } else { FREE },
            );
        }
        if let Some(limit) = timeout {
            let _ = self
                .model
                .as_inner_mut()
                .set_param(grb::parameter::DoubleParam::TimeLimit, limit.as_secs_f64());
        }

        match self.model.solve_mut() {
            Ok(solution) => {
                let status = match solution.status() {
                    SolutionStatus::Optimal => SolveStatus::Optimal,
                    // A gap limit is not proof of optimality any more than a
                    // time limit is, and this crate's guarantee rests on the
                    // difference.
                    SolutionStatus::TimeLimit | SolutionStatus::GapLimit => SolveStatus::Deadline,
                };
                let objective_value = self.model.as_inner().get_attr(grb::attr::ObjVal).ok();
                let values = self
                    .problem
                    .var_names
                    .values()
                    .map(|v| (*v, solution.value(*v)))
                    .collect::<HashMap<_, _>>();
                SolveResult {
                    status,
                    values: Some(values),
                    objective_value,
                }
            }
            Err(good_lp::ResolutionError::Infeasible) => SolveResult {
                status: SolveStatus::Infeasible,
                values: None,
                objective_value: None,
            },
            // Everything else -- a limit that expired before any incumbent was
            // found included -- leaves nothing proven.
            Err(_) => SolveResult {
                status: SolveStatus::Deadline,
                values: None,
                objective_value: None,
            },
        }
    }
}

/// A session without a solver backend.
///
/// Mirrors the real one so the rest of the crate compiles and its tests run
/// without Gurobi, and refuses every solve. That refusal is the honest answer:
/// this algorithm's guarantee rests on distinguishing a proven optimum from an
/// unfinished solve, and a backend that cannot report why it stopped cannot
/// establish it. Silently degrading to weakly nondominated output would be
/// worse than declining.
#[cfg(not(feature = "gurobi"))]
pub struct Session<'a> {
    problem: &'a Problem,
}

#[cfg(not(feature = "gurobi"))]
impl<'a> Session<'a> {
    #[must_use]
    pub const fn new(problem: &'a Problem) -> Self {
        Self { problem }
    }

    #[must_use]
    pub const fn problem(&self) -> &'a Problem {
        self.problem
    }

    pub fn solve(
        &mut self,
        _objective: Expression,
        _lower: &[f64],
        _upper: &[f64],
        _timeout: Option<Duration>,
    ) -> SolveResult {
        SolveResult {
            status: SolveStatus::Deadline,
            values: None,
            objective_value: None,
        }
    }
}

/// First stage `P(eps)`: minimise objective 0 within the rectangle.
///
/// Its optimum is the best attainable value of objective 0 there, but the
/// solution attaining it may be only weakly nondominated -- any of several
/// optima may be returned. [`stage_two`] resolves that.
#[must_use]
pub fn stage_one(session: &mut Session, upper: &[i64], timeout: Option<Duration>) -> SolveResult {
    let objective = session.problem().objectives[0].clone();
    let mut bounds = vec![f64::INFINITY; session.problem().num_objectives()];
    for (slot, &value) in bounds[1..].iter_mut().zip(upper) {
        *slot = objective_to_f64(value);
    }
    session.solve(objective, &[], &bounds, timeout)
}

/// Second stage `Q(eps, z)`: among the optima of the first stage, take one that
/// is nondominated.
///
/// Minimising a strictly positively weighted sum of all objectives, subject to
/// objective 0 holding at its first-stage optimum, cannot return a dominated
/// point: any point dominating the result would score strictly lower. This is
/// what makes the output nondominated rather than weakly nondominated, and it
/// needs only inequalities -- an equality pinning objective 0 to its optimum
/// expresses the same set but is markedly harder for the solver, and in this
/// crate's predecessor it timed out without finding any feasible point.
///
/// The weights are the reference implementation's plain sum, rescaled by each
/// objective's range. Any strictly positive weights preserve the argument
/// above, and equal weights on objectives that differ by two orders of
/// magnitude -- cost against area here -- would let the larger one decide the
/// tie-break alone.
///
/// `first_optimum` is the *unrounded* first-stage optimum, and it is relaxed by
/// a whisker before being imposed. Both matter: the criterion values are sums
/// of floating-point coefficients, so a bound rounded to the nearest integer
/// can fall below the value it came from, and even the exact value can be
/// rejected by a solver working to a feasibility tolerance. Either way this
/// stage would report infeasible on a region its own first stage just proved
/// non-empty -- which the caller cannot distinguish from an empty region.
#[must_use]
pub fn stage_two(
    session: &mut Session,
    weights: &[f64],
    first_optimum: f64,
    upper: &[i64],
    timeout: Option<Duration>,
) -> SolveResult {
    let objective = weighted(session.problem(), weights);
    let mut bounds = vec![f64::INFINITY; session.problem().num_objectives()];
    bounds[0] = relax(first_optimum);
    for (slot, &value) in bounds[1..].iter_mut().zip(upper) {
        *slot = objective_to_f64(value);
    }
    session.solve(objective, &[], &bounds, timeout)
}

/// A strictly positively weighted sum of every objective.
fn weighted(problem: &Problem, weights: &[f64]) -> Expression {
    problem
        .objectives
        .iter()
        .zip(weights)
        .fold(Expression::from(0.0), |acc, (f, &w)| acc + w * f.clone())
}

/// Tighten the solver's tolerances so a reported solution is genuinely integral.
///
/// The algorithm bounds one solve by another's optimum, which is only sound if
/// that optimum belongs to a solution the next solve can reproduce. Under the
/// default integrality tolerance of `1e-5`, a few hundred binaries at
/// coefficients around `1e6` can be "integer feasible" while sitting thousands
/// of units from any truly integral point -- so the bound excludes every real
/// solution and the next solve reports infeasible on a region its predecessor
/// just proved non-empty. Observed exactly that way on `lagos_nigeria_150`.
///
/// Failures are ignored deliberately: a solver that rejects a tolerance still
/// solves the model, and its status remains the thing the guarantee rests on.
#[cfg(feature = "gurobi")]
fn tighten_tolerances(model: &mut good_lp::solvers::gurobi::GurobiProblem) {
    use grb::parameter::{DoubleParam, IntParam};
    let inner = model.as_inner_mut();
    let _ = inner.set_param(DoubleParam::IntFeasTol, 1e-9);
    let _ = inner.set_param(DoubleParam::FeasibilityTol, 1e-9);
    let _ = inner.set_param(DoubleParam::OptimalityTol, 1e-9);
    // Ask for extra care with the wide coefficient ranges these models carry.
    let _ = inner.set_param(IntParam::NumericFocus, 2);
}

/// Widen a bound by the solver's feasibility tolerance, scaled to the value.
///
/// A fixed `1e-6` -- Gurobi's default absolute feasibility tolerance -- is
/// meaningless against criterion values around `1e9`; the relative term is what
/// covers those. The widening is far below `delta`, so it cannot admit a
/// neighbouring point.
fn relax(value: f64) -> f64 {
    value + value.abs().mul_add(1e-9, 1e-6)
}

/// Minimise one objective over the whole feasible set.
#[must_use]
pub fn minimise(session: &mut Session, index: usize, timeout: Option<Duration>) -> SolveResult {
    let objective = session.problem().objectives[index].clone();
    session.solve(objective, &[], &[], timeout)
}

/// Maximise one objective over the whole feasible set.
///
/// Used only to bound the nondominated set from above when the caller supplies
/// no bounds. This is the anti-ideal, which is a safe over-estimate of the
/// nadir rather than the nadir itself.
#[must_use]
pub fn maximise(session: &mut Session, index: usize, timeout: Option<Duration>) -> SolveResult {
    let objective = -session.problem().objectives[index].clone();
    session.solve(objective, &[], &[], timeout)
}

/// Minimise a weighted sum of the objectives.
///
/// Used to populate a front cheaply before a search begins. The caller decides
/// what to do with a solve that did not finish: a merely feasible point still
/// prunes, but only a *proven* optimum carries the two guarantees that make
/// these solves worth more than their cost --
///
/// * with strictly positive weights, an optimal point is nondominated. Any
///   point dominating it would score strictly lower, so there is none. Weights
///   are the caller's to choose, and a zero weight forfeits this: the optimum
///   is then only weakly nondominated.
/// * the optimal value `z` proves `w . f >= z` for every feasible `f`, a
///   supporting half-plane of the whole criterion space. That bounds regions
///   the search has not looked at, and costs no further solve.
///
/// Both are why this returns the full [`SolveResult`] rather than just a point.
///
/// `weights` should already account for the objectives' scales -- an unweighted
/// sum over objectives differing by orders of magnitude is decided by the
/// largest alone, and would return the same corner for every setting.
#[must_use]
pub fn weighted_sum(
    session: &mut Session,
    weights: &[f64],
    timeout: Option<Duration>,
) -> SolveResult {
    let objective = weighted(session.problem(), weights);
    session.solve(objective, &[], &[], timeout)
}

/// Is there any feasible point inside the box `[lower, upper]`?
///
/// The question the quadtree search asks, and it is deliberately weaker than
/// the one an optimality-based method asks. Proving that *some* point exists in
/// a region is far cheaper than proving a particular one is optimal, and a
/// solver stopped early still answers it whenever it has an incumbent -- so the
/// time limit costs information only when nothing at all was found.
///
/// Returns the point found, when there is one.
#[must_use]
pub fn feasibility_check(
    session: &mut Session,
    lower: &[i64],
    upper: &[i64],
    timeout: Option<Duration>,
) -> (Feasibility, Option<HashMap<Variable, f64>>) {
    let lo: Vec<f64> = lower.iter().map(|v| objective_to_f64(*v)).collect();
    let hi: Vec<f64> = upper.iter().map(|v| objective_to_f64(*v)).collect();
    // A constant objective: any feasible point answers the question, so the
    // solver may stop at the first one it finds.
    let result = session.solve(Expression::from(0.0), &lo, &hi, timeout);
    match (result.status, result.values) {
        // An incumbent settles the question whether or not the solve finished.
        (SolveStatus::Optimal | SolveStatus::Deadline, Some(values)) => {
            (Feasibility::Feasible, Some(values))
        }
        (SolveStatus::Infeasible, _) => (Feasibility::Infeasible, None),
        (SolveStatus::Deadline | SolveStatus::Optimal, None) => (Feasibility::Undecided, None),
    }
}

#[cfg(test)]
mod tests {
    use good_lp::{IntoAffineExpression, ProblemVariables, variable};

    /// Regression: a bound row does not carry its objective's constant term.
    ///
    /// good_lp normalises `f <= x` to `linear <= x - c`, moving the constant to
    /// the right-hand side at construction. Setting that right-hand side later
    /// therefore bounds the linear part alone, and a bound of `x` silently
    /// becomes `f <= x + c`. A maximised objective is negated, which puts its
    /// entire range into the constant, so the error is the size of the
    /// objective -- not a rounding difference.
    ///
    /// It surfaced as a rectangle the search could not clear: the epsilon bound
    /// admitted points outside the rectangle it was meant to restrict, so the
    /// region was never exhausted and the run's own termination guard fired.
    #[test]
    fn a_bound_row_must_reapply_the_objective_constant() {
        let mut vars = ProblemVariables::new();
        let x = vars.add(variable().binary());
        // A negated maximisation, as `to_psbox_problem` builds: max 7x becomes
        // min -7x + 7, carrying a constant of 7.
        let negated = 7.0 - 7.0 * x;
        let constant = IntoAffineExpression::constant(&negated);
        assert!(
            (constant - 7.0).abs() < 1e-12,
            "the constant is where the range went: {constant}"
        );

        // Bounding this objective at 3 must bound the row at 3 - 7 = -4.
        let requested = 3.0;
        assert!(
            (requested - constant - -4.0).abs() < 1e-12,
            "the row's right-hand side carries the offset"
        );
        // Setting the row to the requested value instead admits everything up
        // to 3 + 7 = 10, which is the whole feasible range here.
        assert!(
            requested + constant > 7.0,
            "unoffset, the bound stops restricting anything"
        );
    }
}
