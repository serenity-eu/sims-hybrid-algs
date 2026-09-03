//! Single-objective optimization solver
//!
//! This module provides functionality for solving individual objective functions
//! from multi-objective problems, used primarily for bounds calculation and payoff tables.

use crate::{
    error::{AugmeconError, Result},
    model::MultiObjectiveProblem,
    options::Options,
    solution::Solution,
};
use good_lp::constraint;
#[cfg(feature = "coin_cbc")]
use good_lp::solvers::coin_cbc;
#[cfg(feature = "highs")]
use good_lp::solvers::highs;
use good_lp::solvers::lp_solvers::{GurobiSolver, WithMaxSeconds};
#[cfg(feature = "scip")]
use good_lp::solvers::scip;
#[cfg(feature = "gurobi")]
use good_lp::solvers::{gurobi::gurobi, WithTimeLimit};
use good_lp::{Solution as GoodLpSolution, SolverModel};
use std::time::Duration;

/// Create the default solver (Gurobi via lp-solvers)
fn create_gurobi_solver() -> good_lp::solvers::lp_solvers::LpSolver<GurobiSolver> {
    println!("DEBUG: Creating Gurobi solver via lp-solvers");
    let gurobi = GurobiSolver::new();
    good_lp::solvers::lp_solvers::LpSolver(gurobi)
}

/// Create solver with time limit
fn create_gurobi_solver_with_timeout(
    timeout: Duration,
) -> good_lp::solvers::lp_solvers::LpSolver<GurobiSolver> {
    println!(
        "DEBUG: Creating Gurobi solver via lp-solvers with {}s timeout",
        timeout.as_secs()
    );
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Timeout duration in seconds is expected to fit in u32 for Gurobi solver API - values over 4.2 billion seconds (136 years) are not realistic"
    )]
    let seconds = timeout.as_secs() as u32;
    let gurobi = GurobiSolver::new().with_max_seconds(seconds);
    good_lp::solvers::lp_solvers::LpSolver(gurobi)
}

/// A Gurobi model held across a sweep of scalarised solves.
///
/// Both callers issue dozens of solves that differ only in their objective and
/// in a bound or two: the variables, the constraints and the feasible set are
/// identical every time. Aneja & Nair varies the weights; GPBA-A's
/// lexicographic post-pass pins one objective and minimises the other. Rebuilding the model for each one costs about 0.65s on the
/// larger instances -- roughly 22us per constraint to clone it out of the
/// problem and rebuild it term by term -- which at thirty solves is a tenth of
/// a 200-second budget spent re-describing a model the solver already has.
///
/// Batching the rows into a single Gurobi call was measured and changed
/// nothing, which is what established the cost is marshalling on this side
/// rather than anything the solver does. So the model is built once and only
/// its objective is replaced. Gurobi keeps its presolved copy, its basis and
/// its cut pool across the change.
///
/// Restricted to the native Gurobi backend: it is the only one this crate has
/// in-place editing for, and the only one the comparison runs on.
#[cfg(feature = "gurobi")]
pub struct ScalarisationSession<'a> {
    problem: &'a MultiObjectiveProblem,
    model: good_lp::solvers::gurobi::GurobiProblem,
    /// `f_j <= u_j` and `-f_j <= -l_j`, one pair per objective, relaxed unless
    /// a solve asks for them.
    upper: Vec<good_lp::constraint::ConstraintReference>,
    lower: Vec<good_lp::constraint::ConstraintReference>,
    /// Each objective's constant term, which its bound rows do not carry.
    ///
    /// good_lp normalises `f <= x` to `linear <= x - c`, moving the constant to
    /// the right-hand side when the row is built, so setting that right-hand
    /// side later bounds the linear part alone. Reapplied on every set.
    constants: Vec<f64>,
}

/// A bound wide enough that Gurobi treats the row as absent.
#[cfg(feature = "gurobi")]
const FREE: f64 = 1e30;

#[cfg(feature = "gurobi")]
impl<'a> ScalarisationSession<'a> {
    /// Build the model once, with every structural constraint in place.
    ///
    /// The objective is a placeholder; [`Self::solve`] replaces it outright on
    /// every call, and `set_objective` zeroes any coefficient absent from the
    /// replacement, so no term of one sweep step survives into the next.
    #[must_use]
    pub fn new(problem: &'a MultiObjectiveProblem, options: &Options) -> Self {
        let mut model = problem
            .variables
            .clone()
            .minimise(good_lp::Expression::from(0.0))
            .using(gurobi);
        crate::options::apply_gurobi_options(&mut model, options);
        for constraint in &problem.constraints {
            model.add_constraint(constraint.clone());
        }

        // One relaxed row per bound a solve can impose, added here so their
        // right-hand sides can move later without touching the model's
        // structure -- which is what keeps Gurobi's warm start usable.
        let mut upper = Vec::with_capacity(problem.num_objectives());
        let mut lower = Vec::with_capacity(problem.num_objectives());
        let mut constants = Vec::with_capacity(problem.num_objectives());
        for (obj_expr, _) in &problem.objectives {
            upper.push(model.add_constraint(constraint!(obj_expr.clone() <= FREE)));
            lower.push(model.add_constraint(constraint!(-obj_expr.clone() <= FREE)));
            constants.push(good_lp::IntoAffineExpression::constant(obj_expr));
        }

        Self {
            problem,
            model,
            upper,
            lower,
            constants,
        }
    }

    /// Relax every bound row, so a solve sees only the bounds it asks for.
    fn release_bounds(&mut self) {
        use good_lp::solvers::ModelWithMutableRhs;
        for index in 0..self.problem.num_objectives() {
            self.model.set_rhs(self.upper[index], FREE);
            self.model.set_rhs(self.lower[index], FREE);
        }
    }

    /// Minimise objective `secondary` with objective `pinned` held at `value`.
    ///
    /// The second half of a lexicographic refinement: the epsilon-constraint
    /// solve fixes what the primary objective can reach, and this establishes
    /// the best the other objective can do there -- turning a merely weakly
    /// efficient point into an efficient one.
    ///
    /// Pinning is two bound rows rather than an equality constraint, so it is a
    /// right-hand-side edit on the model already built. That matters: this pass
    /// costs one extra solve per emitted point, and when each of those solves
    /// also rebuilt the model it was too expensive to keep on -- measured at
    /// six emitted points falling to two.
    ///
    /// # Errors
    /// [`AugmeconError::OptimizationError`] if the solve fails.
    pub fn solve_pinned(
        &mut self,
        secondary: usize,
        pinned: usize,
        value: f64,
        timeout: Option<Duration>,
    ) -> Result<Solution> {
        use good_lp::solvers::{
            ModelWithMutableObjective, ModelWithMutableRhs, ObjectiveDirection, ReusableModel,
            SolverModel as _,
        };

        self.release_bounds();
        let constant = self.constants[pinned];
        self.model.set_rhs(self.upper[pinned], value - constant);
        self.model.set_rhs(self.lower[pinned], constant - value);

        let (secondary_expr, direction) = &self.problem.objectives[secondary];
        let objective = match direction {
            crate::model::ObjectiveDirection::Minimize => secondary_expr.clone(),
            crate::model::ObjectiveDirection::Maximize => -secondary_expr.clone(),
        };
        self.model
            .set_objective(objective, ObjectiveDirection::Minimisation);
        if let Some(limit) = timeout {
            let _ = self
                .model
                .as_inner_mut()
                .set_param(grb::parameter::DoubleParam::TimeLimit, limit.as_secs_f64());
        }

        let started = std::time::Instant::now();
        let solved = self.model.solve_mut().map_err(|e| {
            AugmeconError::OptimizationError(format!("Lexicographic refinement failed: {e:?}"))
        })?;
        log::info!(
            "PINNED obj{secondary} with obj{pinned} fixed at {value}: {:.2}s (reused model)",
            started.elapsed().as_secs_f64()
        );
        let solution = self.extract(&solved);
        self.release_bounds();
        Ok(solution)
    }

    /// Minimise `sum w_i f_i` over the model, maximised objectives negated.
    ///
    /// # Errors
    /// [`AugmeconError::InvalidObjectiveCount`] if `weights` does not match the
    /// problem's objectives, or [`AugmeconError::OptimizationError`] if the
    /// solve fails.
    pub fn solve(&mut self, weights: &[f64], timeout: Option<Duration>) -> Result<Solution> {
        use good_lp::solvers::{
            ModelWithMutableObjective, ObjectiveDirection, ReusableModel, SolverModel as _,
        };

        if weights.len() != self.problem.num_objectives() {
            return Err(AugmeconError::InvalidObjectiveCount(weights.len()));
        }
        let expr: good_lp::Expression = self
            .problem
            .objectives
            .iter()
            .zip(weights)
            .map(|((obj_expr, dir), &w)| match dir {
                crate::model::ObjectiveDirection::Minimize => w * obj_expr.clone(),
                crate::model::ObjectiveDirection::Maximize => (-w) * obj_expr.clone(),
            })
            .sum();
        // A weighted sum is unconstrained beyond the problem's own rows; drop
        // anything a pinned solve left behind.
        self.release_bounds();
        self.model
            .set_objective(expr, ObjectiveDirection::Minimisation);
        if let Some(limit) = timeout {
            let _ = self
                .model
                .as_inner_mut()
                .set_param(grb::parameter::DoubleParam::TimeLimit, limit.as_secs_f64());
        }

        let solution = self.model.solve_mut().map_err(|e| {
            AugmeconError::OptimizationError(format!("Weighted-sum optimization failed: {e:?}"))
        })?;
        // A weighted sum stopped at its time limit returns an incumbent, not an
        // optimum. Aneja & Nair's construction assumes the optimum -- a
        // suboptimal answer here does not just cost a point, it misplaces the
        // segment the next weight is derived from -- so say when that happens.
        if !matches!(
            good_lp::solvers::Solution::status(&solution),
            good_lp::solvers::SolutionStatus::Optimal
        ) {
            log::warn!(
                "Weighted-sum solve for weights {weights:?} stopped before proving optimality"
            );
        }
        Ok(self.extract(&solution))
    }

    /// Read a solved model back into a [`Solution`].
    fn extract<S: GoodLpSolution>(&self, solution: &S) -> Solution {
        let variable_values = self
            .problem
            .var_map
            .iter()
            .map(|(name, &var)| (name.clone(), solution.value(var)))
            .collect();
        let objective_values = self
            .problem
            .objectives
            .iter()
            .map(|(obj_expr, _)| obj_expr.eval_with(solution))
            .collect();
        Solution::new(objective_values, variable_values)
    }
}

/// Solver for single-objective optimization problems
pub struct SingleObjectiveSolver<'a> {
    problem: &'a MultiObjectiveProblem,
    options: &'a Options,
}

impl<'a> SingleObjectiveSolver<'a> {
    /// Create a new single-objective solver for the given problem
    #[must_use]
    pub const fn new(problem: &'a MultiObjectiveProblem, options: &'a Options) -> Self {
        Self { problem, options }
    }

    /// Solve single-objective optimization for the specified objective index
    ///
    /// This simply optimizes the specified objective without any constraints
    /// on other objectives.
    ///
    /// # Errors
    /// Returns error if optimization fails or problem is infeasible
    pub fn solve_objective(
        &self,
        objective_index: usize,
        timeout: Option<Duration>,
    ) -> Result<Solution> {
        println!("DEBUG: Solving single objective problem for objective {objective_index} with timeout: {timeout:?}");
        log::debug!("Solving single objective problem for objective {objective_index} with timeout: {timeout:?}");

        if objective_index >= self.problem.num_objectives() {
            println!(
                "DEBUG: Invalid objective index {} >= {}",
                objective_index,
                self.problem.num_objectives()
            );
            return Err(AugmeconError::InvalidObjectiveCount(objective_index));
        }

        // Use the problem's existing variables instead of recreating them
        let prob_vars = self.problem.variables.clone();
        println!(
            "DEBUG: Using {} problem variables",
            self.problem.var_map.len()
        );
        log::debug!(
            "Using problem variables: {:?}",
            self.problem.var_map.keys().collect::<Vec<_>>()
        );

        // Build the objective expression
        let objective_expr = if objective_index < self.problem.objectives.len() {
            let (obj_expr, _obj_direction) = &self.problem.objectives[objective_index];
            log::debug!("Objective expression for index {objective_index}: {obj_expr:?}");
            obj_expr.clone()
        } else {
            good_lp::Expression::from(0.0)
        };

        // Determine optimization direction
        let (_, direction) = &self.problem.objectives[objective_index];

        println!(
            "DEBUG: Solver parameters: {:?}",
            self.options.solver_parameters
        );
        println!("DEBUG: Using solver: {}", self.options.solver.name());

        // Combine timeout parameter with options timeout, preferring the parameter
        let effective_timeout =
            timeout.or_else(|| self.options.process_timeout.map(Duration::from_secs));

        println!(
            "DEBUG: Using {} with {} parameters and effective timeout: {:?}",
            self.options.solver.name(),
            self.options.solver_parameters.len(),
            effective_timeout
        );

        // Create and solve with the selected solver - each branch handles its own type
        match self.options.solver {
            crate::solver_enum::Solver::Default => {
                let model = if let Some(timeout_duration) = effective_timeout {
                    if matches!(direction, crate::model::ObjectiveDirection::Minimize) {
                        prob_vars
                            .minimise(objective_expr)
                            .using(create_gurobi_solver_with_timeout(timeout_duration))
                    } else {
                        prob_vars
                            .maximise(objective_expr)
                            .using(create_gurobi_solver_with_timeout(timeout_duration))
                    }
                } else if matches!(direction, crate::model::ObjectiveDirection::Minimize) {
                    prob_vars
                        .minimise(objective_expr)
                        .using(create_gurobi_solver())
                } else {
                    prob_vars
                        .maximise(objective_expr)
                        .using(create_gurobi_solver())
                };
                // Note: Gurobi via lp-solvers doesn't expose set_parameter method
                if !self.options.solver_parameters.is_empty() {
                    println!(
                        "DEBUG: Solver {} does not support parameters, ignoring {} parameters",
                        self.options.solver.name(),
                        self.options.solver_parameters.len()
                    );
                }
                self.solve_with_model_common(model, objective_index)
            }
            #[cfg(feature = "coin_cbc")]
            crate::solver_enum::Solver::CoinCbc => {
                let mut model = if matches!(direction, crate::model::ObjectiveDirection::Minimize) {
                    prob_vars.minimise(objective_expr).using(coin_cbc::coin_cbc)
                } else {
                    prob_vars.maximise(objective_expr).using(coin_cbc::coin_cbc)
                };
                // Enforce the solve budget so a hard single-objective (e.g. the payoff
                // table on large instances) can't run unbounded.
                if let Some(t) = effective_timeout {
                    // Integer ceil of the duration in whole seconds (no float cast).
                    let secs = t.as_secs() + u64::from(t.subsec_nanos() > 0);
                    model.set_parameter("sec", &secs.to_string());
                }
                self.solve_with_model_common(model, objective_index)
            }
            #[cfg(not(feature = "coin_cbc"))]
            crate::solver_enum::Solver::CoinCbc => Err(AugmeconError::UnsupportedSolver(
                "CoinCbc solver is not available. Enable the 'coin_cbc' feature to use it."
                    .to_string(),
            )),
            #[cfg(feature = "highs")]
            crate::solver_enum::Solver::HiGHS => {
                let model = if matches!(direction, crate::model::ObjectiveDirection::Minimize) {
                    prob_vars.minimise(objective_expr).using(highs::highs)
                } else {
                    prob_vars.maximise(objective_expr).using(highs::highs)
                };
                // HiGHS has no wall-clock stop by default; wire the solve budget in so the
                // payoff-table / ideal-bounds solve on large instances can't hang forever.
                let model = match effective_timeout {
                    Some(t) => model.set_time_limit(t.as_secs_f64()),
                    None => model,
                };
                // Disable presolve: on the large SIMS models (tens of thousands of rows)
                // HiGHS presolve does not poll the wall-clock and runs past the time limit
                // (observed: a 30s budget solve running >120s, a 300s budget → hours).
                // Skipping presolve keeps the solve budget-respecting — branch-and-bound
                // itself does check the limit — so every instance completes on one backend.
                let model = model.set_option("presolve", "off");
                self.solve_with_model_common(model, objective_index)
            }
            #[cfg(not(feature = "highs"))]
            crate::solver_enum::Solver::HiGHS => Err(AugmeconError::UnsupportedSolver(
                "HiGHS solver is not available. Enable the 'highs' feature to use it.".to_string(),
            )),
            #[cfg(feature = "scip")]
            crate::solver_enum::Solver::SCIP => {
                let model = if matches!(direction, crate::model::ObjectiveDirection::Minimize) {
                    prob_vars.minimise(objective_expr).using(scip::scip)
                } else {
                    prob_vars.maximise(objective_expr).using(scip::scip)
                };
                // Enforce the solve budget (SCIP takes an integer-second limit).
                let model = match effective_timeout {
                    Some(t) => model.set_time_limit(t.as_secs_f64().ceil() as usize),
                    None => model,
                };
                self.solve_with_model_common(model, objective_index)
            }
            #[cfg(not(feature = "scip"))]
            crate::solver_enum::Solver::SCIP => Err(AugmeconError::UnsupportedSolver(
                "SCIP solver is not available. Enable the 'scip' feature to use it.".to_string(),
            )),
            #[cfg(feature = "gurobi")]
            crate::solver_enum::Solver::Gurobi => {
                let model = if matches!(direction, crate::model::ObjectiveDirection::Minimize) {
                    {
                        let mut m = prob_vars.minimise(objective_expr).using(gurobi);
                        crate::options::apply_gurobi_options(&mut m, self.options);
                        m
                    }
                } else {
                    {
                        let mut m = prob_vars.maximise(objective_expr).using(gurobi);
                        crate::options::apply_gurobi_options(&mut m, self.options);
                        m
                    }
                };
                // Gurobi honours TimeLimit natively (unlike HiGHS presolve), so
                // just wire the solve budget in.
                let model = match effective_timeout {
                    Some(t) => model.with_time_limit(t.as_secs_f64()),
                    None => model,
                };
                self.solve_with_model_common(model, objective_index)
            }
            #[cfg(not(feature = "gurobi"))]
            crate::solver_enum::Solver::Gurobi => Err(AugmeconError::UnsupportedSolver(
                "Gurobi solver is not available. Enable the 'gurobi' feature to use it."
                    .to_string(),
            )),
        }
    }

    /// Minimise an arbitrary linear expression over the problem's feasible set
    /// (the problem's own constraints plus `extra_constraints`), using the
    /// configured solver, timeout, and — for `HiGHS` — presolve disabled.
    ///
    /// This is the primitive behind the weighted-sum scalarisation used by the
    /// Anytime Aneja & Nair method (`aneja_nair`) and by lexicographic extreme
    /// solves. It shares the exact same model-build + solve + extract path as
    /// [`Self::solve_objective`]; only the objective and extra constraints differ.
    ///
    /// # Errors
    /// Returns an error if the optimisation fails or the problem is infeasible.
    pub fn solve_minimize_with_constraints(
        &self,
        expr: good_lp::Expression,
        extra_constraints: &[good_lp::constraint::Constraint],
        timeout: Option<Duration>,
    ) -> Result<Solution> {
        let prob_vars = self.problem.variables.clone();
        let effective_timeout =
            timeout.or_else(|| self.options.process_timeout.map(Duration::from_secs));

        match self.options.solver {
            crate::solver_enum::Solver::Default => {
                let model = match effective_timeout {
                    Some(t) => prob_vars
                        .minimise(expr)
                        .using(create_gurobi_solver_with_timeout(t)),
                    None => prob_vars.minimise(expr).using(create_gurobi_solver()),
                };
                self.solve_with_extra(model, extra_constraints)
            }
            #[cfg(feature = "coin_cbc")]
            crate::solver_enum::Solver::CoinCbc => {
                let mut model = prob_vars.minimise(expr).using(coin_cbc::coin_cbc);
                if let Some(t) = effective_timeout {
                    // Integer ceil of the duration in whole seconds (no float cast).
                    let secs = t.as_secs() + u64::from(t.subsec_nanos() > 0);
                    model.set_parameter("sec", &secs.to_string());
                }
                self.solve_with_extra(model, extra_constraints)
            }
            #[cfg(not(feature = "coin_cbc"))]
            crate::solver_enum::Solver::CoinCbc => Err(AugmeconError::UnsupportedSolver(
                "CoinCbc solver is not available. Enable the 'coin_cbc' feature to use it."
                    .to_string(),
            )),
            #[cfg(feature = "highs")]
            crate::solver_enum::Solver::HiGHS => {
                let model = prob_vars.minimise(expr).using(highs::highs);
                let model = match effective_timeout {
                    Some(t) => model.set_time_limit(t.as_secs_f64()),
                    None => model,
                };
                // Same rationale as solve_objective: HiGHS presolve ignores the
                // wall-clock limit on the large SIMS models, so disable it.
                let model = model.set_option("presolve", "off");
                self.solve_with_extra(model, extra_constraints)
            }
            #[cfg(not(feature = "highs"))]
            crate::solver_enum::Solver::HiGHS => Err(AugmeconError::UnsupportedSolver(
                "HiGHS solver is not available. Enable the 'highs' feature to use it.".to_string(),
            )),
            #[cfg(feature = "scip")]
            crate::solver_enum::Solver::SCIP => {
                let model = prob_vars.minimise(expr).using(scip::scip);
                let model = match effective_timeout {
                    Some(t) => model.set_time_limit(t.as_secs_f64().ceil() as usize),
                    None => model,
                };
                self.solve_with_extra(model, extra_constraints)
            }
            #[cfg(not(feature = "scip"))]
            crate::solver_enum::Solver::SCIP => Err(AugmeconError::UnsupportedSolver(
                "SCIP solver is not available. Enable the 'scip' feature to use it.".to_string(),
            )),
            #[cfg(feature = "gurobi")]
            crate::solver_enum::Solver::Gurobi => {
                let mut model = prob_vars.minimise(expr).using(gurobi);
                crate::options::apply_gurobi_options(&mut model, self.options);
                let model = match effective_timeout {
                    Some(t) => model.with_time_limit(t.as_secs_f64()),
                    None => model,
                };
                self.solve_with_extra(model, extra_constraints)
            }
            #[cfg(not(feature = "gurobi"))]
            crate::solver_enum::Solver::Gurobi => Err(AugmeconError::UnsupportedSolver(
                "Gurobi solver is not available. Enable the 'gurobi' feature to use it."
                    .to_string(),
            )),
        }
    }

    /// Minimise a non-negative weighted sum `Σ wₗ·fₗ(x)` (Maximise objectives are
    /// negated first). Weights should be pre-normalised by the caller to keep the
    /// scalarised coefficients small (avoids `f64` precision loss on the large
    /// integer objectives — see `aneja_nair`).
    ///
    /// # Errors
    /// Returns an error on a weight/objective count mismatch or if the solve fails.
    pub fn solve_weighted_sum(
        &self,
        weights: &[f64],
        timeout: Option<Duration>,
    ) -> Result<Solution> {
        if weights.len() != self.problem.num_objectives() {
            return Err(AugmeconError::InvalidObjectiveCount(weights.len()));
        }
        let expr: good_lp::Expression = self
            .problem
            .objectives
            .iter()
            .zip(weights)
            .map(|((obj_expr, dir), &w)| match dir {
                crate::model::ObjectiveDirection::Minimize => w * obj_expr.clone(),
                crate::model::ObjectiveDirection::Maximize => (-w) * obj_expr.clone(),
            })
            .sum();
        self.solve_minimize_with_constraints(expr, &[], timeout)
    }

    /// Add `extra` constraints to a built model, then run the shared solve+extract.
    fn solve_with_extra<T: SolverModel>(
        &self,
        mut model: T,
        extra: &[good_lp::constraint::Constraint],
    ) -> Result<Solution> {
        for c in extra {
            model.add_constraint(c.clone());
        }
        self.solve_with_model_common(model, 0)
    }

    /// Common solving logic for all solver types
    fn solve_with_model_common<T: SolverModel>(
        &self,
        mut model: T,
        _objective_index: usize,
    ) -> Result<Solution> {
        // Add original constraints
        println!(
            "DEBUG: Adding {} constraints to single objective solver",
            self.problem.constraints.len()
        );
        log::debug!(
            "Adding {} constraints to single objective solver",
            self.problem.constraints.len()
        );
        for (idx, constraint) in self.problem.constraints.iter().enumerate() {
            // NB: was an unconditional println! — on a 44k-constraint instance ×
            // dozens of solves that dumped ~450 MB of stdout and filled the disk.
            log::debug!("Adding constraint {idx}: {constraint:?}");
            model.add_constraint(constraint.clone());
        }

        // Solve the model
        println!("DEBUG: Solving the model...");
        let solution = model.solve().map_err(|e| {
            println!("DEBUG: Single objective optimization failed: {e:?}");
            AugmeconError::OptimizationError(format!("Single objective optimization failed: {e:?}"))
        })?;
        log::debug!("Single-objective problem solved successfully");

        // Extract variable values (don't log individual variables, too verbose)
        let mut variable_values = std::collections::HashMap::new();
        for (name, &var) in &self.problem.var_map {
            let val = solution.value(var);
            variable_values.insert(name.clone(), val);
        }

        // Calculate objective values by evaluating expressions with the solution
        let mut objective_values = Vec::with_capacity(self.problem.num_objectives());
        for i in 0..self.problem.num_objectives() {
            if i < self.problem.objectives.len() {
                let (obj_expr, _) = &self.problem.objectives[i];
                let obj_value = obj_expr.eval_with(&solution);
                log::debug!("Objective {i}: {obj_value}");
                objective_values.push(obj_value);
            } else {
                objective_values.push(0.0);
            }
        }

        println!("DEBUG: Single objective solve completed successfully");
        Ok(Solution::new(objective_values, variable_values))
    }

    /// Solve single-objective optimization with MAXIMIZATION
    ///
    /// This is used to compute nadir points by finding the worst (maximum) value for each objective.
    /// Matches Python's behavior: `model.setObjective(objectives_exprs[i], gp.GRB.MAXIMIZE)`
    ///
    /// # Errors
    /// Returns error if optimization fails or problem is infeasible
    pub fn solve_objective_maximized(
        &self,
        objective_index: usize,
        timeout: Option<Duration>,
    ) -> Result<Solution> {
        log::debug!("Solving single objective (MAXIMIZED) for objective {objective_index} with timeout: {timeout:?}");

        if objective_index >= self.problem.num_objectives() {
            return Err(AugmeconError::InvalidObjectiveCount(objective_index));
        }

        // Use the problem's existing variables
        let prob_vars = self.problem.variables.clone();

        // Build the objective expression
        let objective_expr = if objective_index < self.problem.objectives.len() {
            let (obj_expr, _obj_direction) = &self.problem.objectives[objective_index];
            obj_expr.clone()
        } else {
            good_lp::Expression::from(0.0)
        };

        // Combine timeout parameter with options timeout
        let effective_timeout =
            timeout.or_else(|| self.options.process_timeout.map(Duration::from_secs));

        // Create and solve with the selected solver - ALWAYS MAXIMIZING
        match self.options.solver {
            crate::solver_enum::Solver::Default => {
                let model = if let Some(timeout_duration) = effective_timeout {
                    // FORCE MAXIMIZATION regardless of original direction
                    prob_vars
                        .maximise(objective_expr)
                        .using(create_gurobi_solver_with_timeout(timeout_duration))
                } else {
                    // FORCE MAXIMIZATION regardless of original direction
                    prob_vars
                        .maximise(objective_expr)
                        .using(create_gurobi_solver())
                };

                if !self.options.solver_parameters.is_empty() {
                    println!(
                        "DEBUG: Solver {} does not support parameters, ignoring {} parameters",
                        self.options.solver.name(),
                        self.options.solver_parameters.len()
                    );
                }
                self.solve_with_model_common(model, objective_index)
            }
            #[cfg(feature = "coin_cbc")]
            crate::solver_enum::Solver::CoinCbc => {
                let model = prob_vars.maximise(objective_expr).using(coin_cbc::coin_cbc);
                self.solve_with_model_common(model, objective_index)
            }
            #[cfg(not(feature = "coin_cbc"))]
            crate::solver_enum::Solver::CoinCbc => Err(AugmeconError::UnsupportedSolver(
                "CoinCbc solver is not available. Enable the 'coin_cbc' feature to use it."
                    .to_string(),
            )),
            #[cfg(feature = "highs")]
            crate::solver_enum::Solver::HiGHS => {
                let model = prob_vars.maximise(objective_expr).using(highs::highs);
                self.solve_with_model_common(model, objective_index)
            }
            #[cfg(not(feature = "highs"))]
            crate::solver_enum::Solver::HiGHS => Err(AugmeconError::UnsupportedSolver(
                "HiGHS solver is not available. Enable the 'highs' feature to use it.".to_string(),
            )),
            #[cfg(feature = "scip")]
            crate::solver_enum::Solver::SCIP => {
                let model = prob_vars.maximise(objective_expr).using(scip::scip);
                self.solve_with_model_common(model, objective_index)
            }
            #[cfg(not(feature = "scip"))]
            crate::solver_enum::Solver::SCIP => Err(AugmeconError::UnsupportedSolver(
                "SCIP solver is not available. Enable the 'scip' feature to use it.".to_string(),
            )),
            #[cfg(feature = "gurobi")]
            crate::solver_enum::Solver::Gurobi => {
                // FORCE MAXIMIZATION regardless of original direction
                let mut model = prob_vars.maximise(objective_expr).using(gurobi);
                crate::options::apply_gurobi_options(&mut model, self.options);
                let model = match effective_timeout {
                    Some(t) => model.with_time_limit(t.as_secs_f64()),
                    None => model,
                };
                self.solve_with_model_common(model, objective_index)
            }
            #[cfg(not(feature = "gurobi"))]
            crate::solver_enum::Solver::Gurobi => Err(AugmeconError::UnsupportedSolver(
                "Gurobi solver is not available. Enable the 'gurobi' feature to use it."
                    .to_string(),
            )),
        }
    }

    /// Solve two objectives lexicographically in a single Gurobi call.
    ///
    /// Declares a hierarchical multi-objective model -- `primary` at the higher
    /// `ObjNPriority`, `secondary` below it -- and lets Gurobi run the passes
    /// internally, "only from among those that would not degrade the solution
    /// quality for higher-priority objectives". That is the same guarantee a
    /// pin-and-reminimise pair of solves gives, without the second cold solve:
    /// measured here, the manual version cost ~95s of a 200s budget.
    ///
    /// It also sidesteps the augmentation entirely. The epsilon-constraint
    /// augmentation only makes a solution efficient when its term stays above
    /// the solver's sensitivity (GPBA-A paper, Section 4.2); a priority is a
    /// structural statement instead of a numerical one, so no rho, no range
    /// scaling, no tolerance interaction.
    ///
    /// `ObjN` is a per-variable attribute with no `good_lp` equivalent, so this
    /// reaches through `GurobiProblem::var_map` and `as_inner_mut`.
    ///
    /// # Errors
    /// Returns error if the model cannot be built or the solve fails
    #[cfg(feature = "gurobi")]
    pub fn solve_lexicographic(
        &self,
        primary: usize,
        secondary: usize,
        timeout: Option<Duration>,
    ) -> Result<Solution> {
        use grb::prelude as grbp;

        let prob_vars = self.problem.variables.clone();
        let (primary_expr, primary_dir) = &self.problem.objectives[primary];
        // Build with the primary objective so good_lp lays out the model as
        // usual; the ObjN attributes below define both objectives explicitly.
        let mut model = match primary_dir {
            crate::model::ObjectiveDirection::Minimize => {
                prob_vars.minimise(primary_expr.clone()).using(gurobi)
            }
            crate::model::ObjectiveDirection::Maximize => {
                prob_vars.maximise(primary_expr.clone()).using(gurobi)
            }
        };
        crate::options::apply_gurobi_options(&mut model, self.options);
        let model = match timeout {
            Some(t) => model.with_time_limit(t.as_secs_f64()),
            None => model,
        };
        // Constraints are added by solve_with_model_common below.
        let mut model = model;

        // Gurobi minimises every objective of a hierarchical model in the
        // model's own sense, so flip the sign of any objective whose direction
        // differs from the model's.
        let sense_flip = |dir: &crate::model::ObjectiveDirection| -> f64 {
            if std::mem::discriminant(dir) == std::mem::discriminant(primary_dir) {
                1.0
            } else {
                -1.0
            }
        };

        let objectives: [(usize, i32); 2] = [(primary, 2), (secondary, 1)];
        {
            let var_map = model.var_map().clone();
            let inner = model.as_inner_mut();
            inner
                .set_attr(grbp::attr::NumObj, 2)
                .map_err(|e| AugmeconError::OptimizationError(format!("NumObj: {e}")))?;
            for (obj_idx, priority) in objectives {
                inner
                    .set_param(
                        grbp::param::ObjNumber,
                        if obj_idx == primary { 0 } else { 1 },
                    )
                    .map_err(|e| AugmeconError::OptimizationError(format!("ObjNumber: {e}")))?;
                let (expr, dir) = &self.problem.objectives[obj_idx];
                let flip = sense_flip(dir);
                let coeffs: Vec<(grb::Var, f64)> =
                    good_lp::IntoAffineExpression::linear_coefficients(expr)
                        .filter_map(|(v, c)| var_map.get(&v).map(|gv| (*gv, flip * c)))
                        .collect();
                inner
                    .set_obj_attr_batch(grbp::attr::ObjN, coeffs)
                    .map_err(|e| AugmeconError::OptimizationError(format!("ObjN: {e}")))?;
                inner
                    .set_attr(grbp::attr::ObjNPriority, priority)
                    .map_err(|e| AugmeconError::OptimizationError(format!("ObjNPriority: {e}")))?;
            }
        }

        self.solve_with_model_common(model, primary)
    }

    /// Minimise `objective_index` with `pin_index` fixed at `pin_value`.
    ///
    /// The second stage of the two-stage payoff-table construction: it turns an
    /// arbitrary optimum of one objective into the *lexicographic* optimum, and
    /// so an efficient rather than merely weakly efficient extreme point.
    ///
    /// Exists so callers outside this crate can do that without depending on
    /// `good_lp` directly to build the equality constraint.
    ///
    /// # Errors
    /// Returns error if the optimization fails or the pinned problem is infeasible
    pub fn solve_objective_pinned(
        &self,
        objective_index: usize,
        pin_index: usize,
        pin_value: f64,
        timeout: Option<Duration>,
    ) -> Result<Solution> {
        let (pin_expr, _) = &self.problem.objectives[pin_index];
        let pinned = constraint!(pin_expr.clone() == pin_value);
        self.solve_objective_with_constraints(objective_index, &[pinned], timeout)
    }

    /// Solve single-objective optimization with additional constraints
    ///
    /// This allows solving an objective with extra constraints beyond the original problem.
    /// Used for calculating payoff table entries where one objective is fixed at its optimal value.
    ///
    /// # Errors
    /// Returns error if optimization fails or problem is infeasible
    pub fn solve_objective_with_constraints(
        &self,
        objective_index: usize,
        additional_constraints: &[good_lp::constraint::Constraint],
        timeout: Option<Duration>,
    ) -> Result<Solution> {
        log::debug!(
            "Solving single objective problem for objective {} with {} additional constraints and timeout: {:?}",
            objective_index,
            additional_constraints.len(),
            timeout
        );

        if objective_index >= self.problem.num_objectives() {
            return Err(AugmeconError::InvalidObjectiveCount(objective_index));
        }

        // Use the problem's existing variables
        let prob_vars = self.problem.variables.clone();

        // Build the objective expression
        let objective_expr = if objective_index < self.problem.objectives.len() {
            let (obj_expr, _obj_direction) = &self.problem.objectives[objective_index];
            obj_expr.clone()
        } else {
            good_lp::Expression::from(0.0)
        };

        // Determine optimization direction
        let (_, direction) = &self.problem.objectives[objective_index];

        println!(
            "DEBUG: [solve_objective_with_constraints] Solver parameters: {:?}",
            self.options.solver_parameters
        );
        println!(
            "DEBUG: [solve_objective_with_constraints] Using solver: {}",
            self.options.solver.name()
        );

        // Combine timeout parameter with options timeout, preferring the parameter
        let effective_timeout =
            timeout.or_else(|| self.options.process_timeout.map(Duration::from_secs));

        println!(
            "DEBUG: [solve_objective_with_constraints] Using {} with {} parameters and effective timeout: {:?}",
            self.options.solver.name(),
            self.options.solver_parameters.len(),
            effective_timeout
        );

        // Create and solve with the selected solver - each branch handles its own type
        match self.options.solver {
            crate::solver_enum::Solver::Default => {
                let model = if let Some(timeout_duration) = effective_timeout {
                    if matches!(direction, crate::model::ObjectiveDirection::Minimize) {
                        prob_vars
                            .minimise(objective_expr)
                            .using(create_gurobi_solver_with_timeout(timeout_duration))
                    } else {
                        prob_vars
                            .maximise(objective_expr)
                            .using(create_gurobi_solver_with_timeout(timeout_duration))
                    }
                } else if matches!(direction, crate::model::ObjectiveDirection::Minimize) {
                    prob_vars
                        .minimise(objective_expr)
                        .using(create_gurobi_solver())
                } else {
                    prob_vars
                        .maximise(objective_expr)
                        .using(create_gurobi_solver())
                };
                // Note: Gurobi via lp-solvers doesn't expose set_parameter method
                if !self.options.solver_parameters.is_empty() {
                    println!(
                        "DEBUG: [solve_objective_with_constraints] Solver {} does not support parameters, ignoring {} parameters",
                        self.options.solver.name(),
                        self.options.solver_parameters.len()
                    );
                }
                self.solve_with_constraints_common(model, objective_index, additional_constraints)
            }
            #[cfg(feature = "coin_cbc")]
            crate::solver_enum::Solver::CoinCbc => {
                let model = if matches!(direction, crate::model::ObjectiveDirection::Minimize) {
                    prob_vars.minimise(objective_expr).using(coin_cbc::coin_cbc)
                } else {
                    prob_vars.maximise(objective_expr).using(coin_cbc::coin_cbc)
                };
                self.solve_with_constraints_common(model, objective_index, additional_constraints)
            }
            #[cfg(not(feature = "coin_cbc"))]
            crate::solver_enum::Solver::CoinCbc => Err(AugmeconError::UnsupportedSolver(
                "CoinCbc solver is not available. Enable the 'coin_cbc' feature to use it."
                    .to_string(),
            )),
            #[cfg(feature = "highs")]
            crate::solver_enum::Solver::HiGHS => {
                let model = if matches!(direction, crate::model::ObjectiveDirection::Minimize) {
                    prob_vars.minimise(objective_expr).using(highs::highs)
                } else {
                    prob_vars.maximise(objective_expr).using(highs::highs)
                };
                self.solve_with_constraints_common(model, objective_index, additional_constraints)
            }
            #[cfg(not(feature = "highs"))]
            crate::solver_enum::Solver::HiGHS => Err(AugmeconError::UnsupportedSolver(
                "HiGHS solver is not available. Enable the 'highs' feature to use it.".to_string(),
            )),
            #[cfg(feature = "scip")]
            crate::solver_enum::Solver::SCIP => {
                let model = if matches!(direction, crate::model::ObjectiveDirection::Minimize) {
                    prob_vars.minimise(objective_expr).using(scip::scip)
                } else {
                    prob_vars.maximise(objective_expr).using(scip::scip)
                };
                self.solve_with_constraints_common(model, objective_index, additional_constraints)
            }
            #[cfg(not(feature = "scip"))]
            crate::solver_enum::Solver::SCIP => Err(AugmeconError::UnsupportedSolver(
                "SCIP solver is not available. Enable the 'scip' feature to use it.".to_string(),
            )),
            #[cfg(feature = "gurobi")]
            crate::solver_enum::Solver::Gurobi => {
                let model = if matches!(direction, crate::model::ObjectiveDirection::Minimize) {
                    {
                        let mut m = prob_vars.minimise(objective_expr).using(gurobi);
                        crate::options::apply_gurobi_options(&mut m, self.options);
                        m
                    }
                } else {
                    {
                        let mut m = prob_vars.maximise(objective_expr).using(gurobi);
                        crate::options::apply_gurobi_options(&mut m, self.options);
                        m
                    }
                };
                let model = match effective_timeout {
                    Some(t) => model.with_time_limit(t.as_secs_f64()),
                    None => model,
                };
                self.solve_with_constraints_common(model, objective_index, additional_constraints)
            }
            #[cfg(not(feature = "gurobi"))]
            crate::solver_enum::Solver::Gurobi => Err(AugmeconError::UnsupportedSolver(
                "Gurobi solver is not available. Enable the 'gurobi' feature to use it."
                    .to_string(),
            )),
        }
    }

    /// Common solving logic with additional constraints for all solver types
    fn solve_with_constraints_common<T: SolverModel>(
        &self,
        mut model: T,
        _objective_index: usize,
        additional_constraints: &[good_lp::Constraint],
    ) -> Result<Solution> {
        // Add original constraints
        for constraint in &self.problem.constraints {
            model.add_constraint(constraint.clone());
        }

        // Add additional constraints
        for constraint in additional_constraints {
            model.add_constraint(constraint.clone());
        }

        // Solve the model
        let solution = model.solve().map_err(|e| {
            AugmeconError::OptimizationError(format!(
                "Constrained single objective optimization failed: {e:?}"
            ))
        })?;

        // Extract variable values
        let mut variable_values = std::collections::HashMap::new();
        for (name, &var) in &self.problem.var_map {
            let val = solution.value(var);
            variable_values.insert(name.clone(), val);
        }

        // Calculate objective values
        let mut objective_values = Vec::with_capacity(self.problem.num_objectives());
        for i in 0..self.problem.num_objectives() {
            if i < self.problem.objectives.len() {
                let (obj_expr, _) = &self.problem.objectives[i];
                let obj_value = obj_expr.eval_with(&solution);
                objective_values.push(obj_value);
            } else {
                objective_values.push(0.0);
            }
        }

        Ok(Solution::new(objective_values, variable_values))
    }
}
