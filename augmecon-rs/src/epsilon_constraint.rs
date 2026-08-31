//! Epsilon-constraint problem builder and solver
//!
//! This module provides functionality for solving epsilon-constraint problems,
//! which are fundamental to multi-objective optimization algorithms like AUGMECON and GPBA.
//!
//! The epsilon-constraint method transforms a multi-objective optimization problem into
//! a series of single-objective problems by constraining all but one objective to specific
//! threshold values (epsilon values).
//!
//! ### Mathematical Formulation
//!
//! For a multi-objective problem with objectives z₁(x), z₂(x), ..., zₙ(x):
//!
//! ```text
//! minimize/maximize z_q(x) + ρ * Σ(10^(k-1) * s_k / r_k)
//! subject to: z_k(x) - s_k = ε_k, s_k ≥ 0, x ∈ X
//! ```
//!
//! Where:
//! - `z_q(x)` is the primary objective to optimize
//! - `ε_k` are the epsilon constraint values for objectives k ≠ q
//! - `s_k` are slack variables for numerical stability
//! - `r_k` are objective ranges for proper scaling
//! - `ρ` is a small augmentation parameter
//! - `X` represents the feasible region defined by constraints

use crate::{
    error::{AugmeconError, Result},
    model::MultiObjectiveProblem,
    options::Options,
    solution::{self, HasObjectives, Solution},
};
#[cfg(feature = "coin_cbc")]
use good_lp::solvers::coin_cbc;
#[cfg(feature = "highs")]
use good_lp::solvers::highs;
use good_lp::solvers::lp_solvers::{GurobiSolver, WithMaxSeconds};
#[cfg(feature = "scip")]
use good_lp::solvers::scip;
#[cfg(feature = "gurobi")]
use good_lp::solvers::{gurobi::gurobi, WithTimeLimit};
use good_lp::{constraint, variable, Expression, Solution as GoodLpSolution, SolverModel};
use std::collections::HashMap;
use std::time::Duration;

/// Create the default solver (Gurobi via lp-solvers)
fn create_solver() -> good_lp::solvers::lp_solvers::LpSolver<GurobiSolver> {
    let gurobi = GurobiSolver::new();
    good_lp::solvers::lp_solvers::LpSolver(gurobi)
}

/// Create solver with time limit
fn create_solver_with_timeout(
    timeout: Duration,
) -> good_lp::solvers::lp_solvers::LpSolver<GurobiSolver> {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Timeout duration in seconds is expected to fit in u32 for Gurobi solver API - values over 4.2 billion seconds (136 years) are not realistic"
    )]
    let seconds = timeout.as_secs() as u32;
    let gurobi = GurobiSolver::new().with_max_seconds(seconds);
    good_lp::solvers::lp_solvers::LpSolver(gurobi)
}

/// Solution with additional slack variable values for bypass coefficient optimization
#[derive(Debug, Clone)]
pub struct SolutionWithSlack {
    /// The main solution
    pub solution: Solution,
    /// Slack variable values indexed by objective index
    pub slack_values: HashMap<usize, f64>,
    /// Other feasible solutions this same solve found and would otherwise
    /// discard (see `Options::solution_pool_size`). Pareto *candidates*, not
    /// optima: they must go through the front's dominance filter like any
    /// other point.
    pub pool: Vec<Solution>,
}

impl SolutionWithSlack {
    /// Create a new solution with slack values
    #[must_use]
    pub const fn new(solution: Solution, slack_values: HashMap<usize, f64>) -> Self {
        Self {
            solution,
            slack_values,
            pool: Vec::new(),
        }
    }
}

/// Outcome of attempting to solve one epsilon-constrained subproblem.
///
/// Distinguishing `Infeasible` from `Inconclusive` matters: GPBA's
/// interval-pruning logic treats a proven-infeasible epsilon value (and,
/// via its cascade rules, everything beyond it) as permanently excluded from
/// the search. That's only valid when the solver has actually *proven*
/// infeasibility. A solver timeout with no incumbent yet found is not proof
/// of anything — the subproblem may well be feasible and just need more
/// time. Collapsing both cases into "no solution" (as this code used to do)
/// causes GPBA to silently prune feasible regions and can converge with zero
/// solutions on an instance that is demonstrably solvable, purely because a
/// single sub-solve ran out of time.
#[derive(Debug)]
pub enum EpsilonSolveOutcome<T> {
    /// A solution was found (possibly non-optimal, e.g. a time-limited
    /// incumbent — `good_lp`'s Gurobi/HiGHS/CoinCbc backends surface those as
    /// `Ok` rather than erroring).
    Solved(T),
    /// The solver proved the subproblem has no feasible solution.
    Infeasible,
    /// The solver did not return a solution, but *not* because infeasibility
    /// was proven — most commonly a timeout before any incumbent was found.
    /// Carries the underlying error's message for diagnostics.
    Inconclusive(String),
}

/// Classifies a `good_lp` solve error into the epsilon-solve outcome that's
/// safe to act on. Shared by every solver backend below so they can't drift
/// out of sync on what counts as a proven infeasibility.
fn classify_solve_error<T>(error: &good_lp::ResolutionError) -> EpsilonSolveOutcome<T> {
    match error {
        good_lp::ResolutionError::Infeasible => EpsilonSolveOutcome::Infeasible,
        other => EpsilonSolveOutcome::Inconclusive(other.to_string()),
    }
}

/// A Gurobi model held across an epsilon-constraint sweep.
///
/// Every subproblem GPBA-A issues has the same variables, the same structural
/// constraints and the same slack variables; only the right-hand sides of the
/// epsilon rows move. Rebuilding for each one costs about 0.65s on the larger
/// instances, almost all of it cloning constraints out of the problem and
/// rebuilding them term by term, which at thirty solves is a tenth of a
/// 200-second budget.
///
/// So the model is built once, with one epsilon row per constrained objective,
/// and a sweep step sets their right-hand sides. Gurobi keeps its presolved
/// copy, its basis and its cut pool across the change.
///
/// # The constant term
///
/// good_lp normalises `expr == eps` to `linear == eps - c`, moving the
/// objective's constant to the right-hand side when the row is built. Setting
/// that right-hand side later therefore constrains the *linear part*, and an
/// epsilon of `e` silently becomes `f == e + c`. These objectives carry
/// constants, so [`Self::solve`] reapplies the offset on every set.
///
/// Restricted to the native Gurobi backend, the only one with in-place editing
/// and the only one the comparison runs on.
#[cfg(feature = "gurobi")]
pub struct EpsilonSession<'a> {
    problem: &'a MultiObjectiveProblem,
    model: good_lp::solvers::gurobi::GurobiProblem,
    /// Slack variable and epsilon row per constrained objective.
    rows: HashMap<usize, (good_lp::Variable, good_lp::constraint::ConstraintReference)>,
    /// Each constrained objective's constant term, absent from its row.
    constants: HashMap<usize, f64>,
    penalty_sum: Expression,
    augmented_primary: Expression,
    epsilon_augmentation: f64,
    primary_objective: usize,
}

#[cfg(feature = "gurobi")]
impl<'a> EpsilonSession<'a> {
    /// Build the model once for a sweep constraining `constrained` objectives.
    ///
    /// `ranges` scales each slack's penalty exactly as the rebuild path does.
    #[must_use]
    pub fn new(
        problem: &'a MultiObjectiveProblem,
        options: &Options,
        primary_objective: usize,
        constrained: &[usize],
        ranges: &HashMap<usize, f64>,
        epsilon_augmentation: f64,
    ) -> Self {
        let mut prob_vars = problem.variables.clone();

        // Slack variables and the penalty term, mirroring
        // `create_slack_variables_and_penalty` so both paths solve the same
        // model -- see the derivation there for the weighting.
        let mut slack = HashMap::new();
        let mut penalty_sum = Expression::from(0.0);
        for &obj_idx in constrained {
            if obj_idx >= problem.objectives.len() {
                continue;
            }
            let slack_var = prob_vars.add(variable().min(0.0));
            slack.insert(obj_idx, slack_var);
            let weight = 10_f64.powi(i32::try_from(obj_idx).unwrap_or_default());
            let range = ranges.get(&obj_idx).copied().unwrap_or(1000.0);
            let normalized_weight = if range.abs() < 1e-10 {
                weight
            } else {
                weight / range
            };
            penalty_sum += normalized_weight * slack_var;
        }

        let (primary_expr, direction) = &problem.objectives[primary_objective];
        // The augmented objective is fixed for the sweep: only the epsilon
        // right-hand sides move.
        let augmented_primary = primary_expr.clone() + epsilon_augmentation * penalty_sum.clone();
        let mut model = match direction {
            crate::model::ObjectiveDirection::Minimize => {
                prob_vars.minimise(augmented_primary.clone())
            }
            crate::model::ObjectiveDirection::Maximize => {
                prob_vars.maximise(augmented_primary.clone())
            }
        }
        .using(gurobi);
        crate::options::apply_gurobi_options(&mut model, options);

        for constraint in &problem.constraints {
            model.add_constraint(constraint.clone());
        }

        // One epsilon row per constrained objective, with a placeholder
        // right-hand side that `solve` replaces before the first solve.
        let mut rows = HashMap::new();
        let mut constants = HashMap::new();
        for (&obj_idx, &slack_var) in &slack {
            let (obj_expr, obj_direction) = &problem.objectives[obj_idx];
            let mut expr = obj_expr.clone();
            match obj_direction {
                crate::model::ObjectiveDirection::Maximize => expr -= slack_var,
                crate::model::ObjectiveDirection::Minimize => expr += slack_var,
            }
            constants.insert(obj_idx, good_lp::IntoAffineExpression::constant(&expr));
            rows.insert(
                obj_idx,
                (slack_var, model.add_constraint(constraint!(expr == 0.0))),
            );
        }

        Self {
            problem,
            model,
            rows,
            constants,
            penalty_sum,
            augmented_primary,
            epsilon_augmentation,
            primary_objective,
        }
    }

    /// Solve for one set of epsilon values, reusing the model.
    ///
    /// Extraction is delegated to a throwaway [`EpsilonConstraintBuilder`], so
    /// the reused path and the rebuild path report solutions through identical
    /// code -- the point of reuse is to reach the same answer sooner, and a
    /// second copy of this logic would be free to drift.
    pub fn solve(
        &mut self,
        epsilon_values: &HashMap<usize, f64>,
        ranges: &HashMap<usize, f64>,
        options: &Options,
        timeout: Option<Duration>,
    ) -> EpsilonSolveOutcome<SolutionWithSlack> {
        use good_lp::solvers::{ModelWithMutableRhs, ReusableModel, SolverModel as _};

        crate::verify::bump(&crate::verify::SLACK_SOLVES);
        for (&obj_idx, &epsilon) in epsilon_values {
            if let Some(&(_, row)) = self.rows.get(&obj_idx) {
                // The row carries the linear part alone; see the type's note.
                let constant = self.constants.get(&obj_idx).copied().unwrap_or(0.0);
                self.model.set_rhs(row, epsilon - constant);
            }
        }
        if let Some(limit) = timeout {
            let _ = self
                .model
                .as_inner_mut()
                .set_param(grb::parameter::DoubleParam::TimeLimit, limit.as_secs_f64());
        }

        log::info!(
            "Solving epsilon-constraint: optimize obj[{}], constraints: {epsilon_values:?} using Gurobi (reused model)",
            self.primary_objective
        );

        match self.model.solve_mut() {
            Ok(solution) => {
                let mut builder =
                    EpsilonConstraintBuilder::new(self.problem, options, self.primary_objective);
                for (&obj_idx, &epsilon) in epsilon_values {
                    let range = ranges.get(&obj_idx).copied().unwrap_or(1000.0);
                    builder = builder.add_constraint_with_range(obj_idx, epsilon, range);
                }
                let slack_vars: HashMap<usize, good_lp::Variable> = self
                    .rows
                    .iter()
                    .map(|(&obj_idx, &(slack, _))| (obj_idx, slack))
                    .collect();
                let mut sol = builder.extract_solution_with_slack(
                    &solution,
                    &self.penalty_sum,
                    &self.augmented_primary,
                    self.epsilon_augmentation,
                    &slack_vars,
                );
                // Free Pareto candidates this solve already found; read from the
                // model, which a reused solve leaves intact.
                if options.solution_pool_size > 0 {
                    sol.pool = builder.harvest_pool_values(self.model.solution_pool());
                    if !sol.pool.is_empty() {
                        log::debug!("Harvested {} pooled candidates", sol.pool.len());
                    }
                }
                EpsilonSolveOutcome::Solved(sol)
            }
            Err(good_lp::ResolutionError::Infeasible) => EpsilonSolveOutcome::Infeasible,
            Err(e) => EpsilonSolveOutcome::Inconclusive(format!("{e:?}")),
        }
    }

    /// Whether this session constrains exactly `constrained`.
    ///
    /// A sweep that changes which objectives it bounds needs a different model,
    /// since the epsilon rows and slack variables are structural.
    #[must_use]
    pub fn covers(&self, constrained: &[usize]) -> bool {
        constrained.len() == self.rows.len()
            && constrained.iter().all(|k| self.rows.contains_key(k))
    }
}

/// Builder for epsilon-constraint problems used by optimization algorithms
pub struct EpsilonConstraintBuilder<'a> {
    problem: &'a MultiObjectiveProblem,
    options: &'a Options,
    primary_objective: usize,
    epsilon_values: HashMap<usize, f64>,
    objective_ranges: HashMap<usize, f64>, // New field for objective ranges
}

impl<'a> EpsilonConstraintBuilder<'a> {
    /// Create a new epsilon-constraint builder
    #[must_use]
    pub fn new(
        problem: &'a MultiObjectiveProblem,
        options: &'a Options,
        primary_objective: usize,
    ) -> Self {
        Self {
            problem,
            options,
            primary_objective,
            epsilon_values: HashMap::new(),
            objective_ranges: HashMap::new(), // Initialize the new field
        }
    }

    /// Add epsilon constraint for an objective: `z_k(x)` >= `epsilon_k`
    #[must_use]
    pub fn add_constraint(mut self, objective_index: usize, epsilon: f64) -> Self {
        if objective_index != self.primary_objective {
            self.epsilon_values.insert(objective_index, epsilon);
        }
        self
    }

    /// Add epsilon constraint with range for proper augmentation scaling
    #[must_use]
    pub fn add_constraint_with_range(
        mut self,
        objective_index: usize,
        epsilon: f64,
        range: f64,
    ) -> Self {
        if objective_index != self.primary_objective {
            self.epsilon_values.insert(objective_index, epsilon);
            self.objective_ranges.insert(objective_index, range);
        }
        self
    }

    /// Creates the augmented primary objective with penalty terms
    fn create_augmented_primary_objective(
        &self,
        primary_objective_expr: Expression,
        penalty_sum: Expression,
        epsilon_augmentation: f64,
    ) -> Expression {
        crate::verify::bump(&crate::verify::AUGMENTATION_BUILT);
        let mut augmented_primary = primary_objective_expr;
        match self.problem.objectives[self.primary_objective].1 {
            crate::model::ObjectiveDirection::Maximize => {
                augmented_primary += epsilon_augmentation * penalty_sum;
            }
            crate::model::ObjectiveDirection::Minimize => {
                augmented_primary -= epsilon_augmentation * penalty_sum;
            }
        }
        augmented_primary
    }

    /// Calculates objective values from the solution
    fn calculate_objective_values<S: GoodLpSolution>(&self, solution: &S) -> Vec<f64> {
        let mut objective_values = Vec::with_capacity(self.problem.num_objectives());
        for i in 0..self.problem.num_objectives() {
            if i < self.problem.objectives.len() {
                let (obj_expr, _) = &self.problem.objectives[i];
                let obj_value = obj_expr.eval_with(solution);
                objective_values.push(obj_value);
            } else {
                objective_values.push(0.0);
            }
        }
        objective_values
    }

    /// Recalculates the primary objective value without penalty terms
    fn recalculate_primary_objective<S: GoodLpSolution>(
        &self,
        penalty_sum: &Expression,
        augmented_primary: &Expression,
        epsilon_augmentation: f64,
        solution: &S,
        objective_values: &mut [f64],
    ) {
        let penalty_value = penalty_sum.eval_with(solution);
        let augmented_obj_value = augmented_primary.eval_with(solution);
        let direction = &self.problem.objectives[self.primary_objective].1;

        let primary_obj_value = match direction {
            crate::model::ObjectiveDirection::Minimize => {
                f64::mul_add(epsilon_augmentation, penalty_value, augmented_obj_value)
            }
            crate::model::ObjectiveDirection::Maximize => {
                f64::mul_add(epsilon_augmentation, -penalty_value, augmented_obj_value)
            }
        };

        // Replace the primary objective value in the objectives vector
        if self.primary_objective < objective_values.len() {
            objective_values[self.primary_objective] = primary_obj_value;
        }
    }

    /// Solve the epsilon-constraint problem
    ///
    /// Optimizes: max/min `z_primary(x)`
    /// Subject to: `z_k(x)` >= `epsilon_k` for all k != primary, x ∈ X
    ///
    /// # Errors
    /// Returns error if optimization fails
    pub fn solve(self) -> Result<Option<Solution>> {
        self.validate_primary_objective()?;

        // Build the primary objective expression directly from the stored expression
        let primary_objective_expr = self.build_primary_objective_expression();

        // Add augmentation term based on GPBA paper formulation (Problem 5)
        // Primary objective += ρ * Σ(10^(k-1) * s_k / r_k)
        // This ensures proper solutions and prevents weak efficiency
        // Theorem 3 (Mavrotas 2009, restated as Thm 3 in the GPBA-A paper): rho
        // must be "sufficiently small", usually 1e-3 to 1e-6, for the optimum of
        // (P5) to be efficient. rho was 1e2 here, which makes the augmentation
        // term reach 1.0 -- the same size as the smallest gap between two
        // integer cost values, so it can reorder the primary objective rather
        // than merely break ties in it.
        let epsilon_augmentation = self.options.epsilon_augmentation;

        // Work directly with the problem's variables and add slack variables to it
        // Clone the problem variables to avoid mutating the original
        let mut prob_vars = self.problem.variables.clone();

        // Create slack variables for epsilon constraints and add them to the same variable set
        let (slack_vars, penalty_sum) = self.create_slack_variables_and_penalty(&mut prob_vars);

        let augmented_primary = self.create_augmented_primary_objective(
            primary_objective_expr,
            penalty_sum.clone(),
            epsilon_augmentation,
        );

        // Determine optimization direction for primary objective
        let (_, direction) = &self.problem.objectives[self.primary_objective];

        // Create the model with the selected solver
        log::debug!(
            "Epsilon-constraint solver: Using {} with {} parameters",
            self.options.solver.name(),
            self.options.solver_parameters.len()
        );

        match self.options.solver {
            crate::solver_enum::Solver::Default => Ok(self.solve_with_default_solver_impl(
                prob_vars,
                &augmented_primary,
                *direction,
                &slack_vars,
                &penalty_sum,
                epsilon_augmentation,
            )),
            #[cfg(feature = "coin_cbc")]
            crate::solver_enum::Solver::CoinCbc => Ok(self.solve_with_coin_cbc_impl(
                prob_vars,
                &augmented_primary,
                *direction,
                &slack_vars,
                &penalty_sum,
                epsilon_augmentation,
            )),
            #[cfg(not(feature = "coin_cbc"))]
            crate::solver_enum::Solver::CoinCbc => Err(AugmeconError::UnsupportedSolver(
                "CoinCbc solver is not available. Enable the 'coin_cbc' feature to use it."
                    .to_string(),
            )),
            #[cfg(feature = "highs")]
            crate::solver_enum::Solver::HiGHS => Ok(self.solve_with_highs_impl(
                prob_vars,
                &augmented_primary,
                *direction,
                &slack_vars,
                &penalty_sum,
                epsilon_augmentation,
            )),
            #[cfg(not(feature = "highs"))]
            crate::solver_enum::Solver::HiGHS => Err(AugmeconError::UnsupportedSolver(
                "HiGHS solver is not available. Enable the 'highs' feature to use it.".to_string(),
            )),
            #[cfg(feature = "scip")]
            crate::solver_enum::Solver::SCIP => Ok(self.solve_with_scip_impl(
                prob_vars,
                &augmented_primary,
                *direction,
                &slack_vars,
                &penalty_sum,
                epsilon_augmentation,
            )),
            #[cfg(not(feature = "scip"))]
            crate::solver_enum::Solver::SCIP => Err(AugmeconError::UnsupportedSolver(
                "SCIP solver is not available. Enable the 'scip' feature to use it.".to_string(),
            )),
            #[cfg(feature = "gurobi")]
            crate::solver_enum::Solver::Gurobi => Ok(self.solve_with_gurobi_impl(
                prob_vars,
                &augmented_primary,
                *direction,
                &slack_vars,
                &penalty_sum,
                epsilon_augmentation,
            )),
            #[cfg(not(feature = "gurobi"))]
            crate::solver_enum::Solver::Gurobi => Err(AugmeconError::UnsupportedSolver(
                "Gurobi solver is not available. Enable the 'gurobi' feature to use it."
                    .to_string(),
            )),
        }
    }

    /// Solve the epsilon-constraint problem and return slack values
    ///
    /// This version returns both the solution and the slack variable values
    /// which are needed for bypass coefficient optimization.
    ///
    /// # Errors
    /// Returns error if optimization fails
    pub fn solve_with_slack(
        self,
        timeout: Option<Duration>,
    ) -> Result<EpsilonSolveOutcome<SolutionWithSlack>> {
        self.validate_primary_objective()?;

        // Build the primary objective expression directly from the stored expression
        let primary_objective_expr = self.build_primary_objective_expression();

        // Add augmentation term based on GPBA paper formulation (Problem 5)
        // Primary objective += ρ * Σ(10^(k-1) * s_k / r_k)
        // Same rho as the non-slack path: this is the solve the main GPBA-A
        // loop actually uses, so it is the one that has to satisfy Theorem 3.
        let epsilon_augmentation = self.options.epsilon_augmentation;

        // Work directly with the problem's variables and add slack variables to it
        // Clone the problem variables to avoid mutating the original
        let mut prob_vars = self.problem.variables.clone();

        // Create slack variables for epsilon constraints and add them to the same variable set
        let (slack_vars, penalty_sum) = self.create_slack_variables_and_penalty(&mut prob_vars);

        let augmented_primary = self.create_augmented_primary_objective(
            primary_objective_expr,
            penalty_sum.clone(),
            epsilon_augmentation,
        );

        // Determine optimization direction for primary objective
        let (_, direction) = &self.problem.objectives[self.primary_objective];

        // Create the model with the selected solver
        log::debug!(
            "Epsilon-constraint slack solver: Using {} with {} parameters",
            self.options.solver.name(),
            self.options.solver_parameters.len()
        );

        match self.options.solver {
            crate::solver_enum::Solver::Default => Ok(self.solve_with_slack_default_solver_impl(
                prob_vars,
                &augmented_primary,
                *direction,
                &slack_vars,
                &penalty_sum,
                epsilon_augmentation,
                timeout,
            )),
            #[cfg(feature = "coin_cbc")]
            crate::solver_enum::Solver::CoinCbc => Ok(self.solve_with_slack_coin_cbc_impl(
                prob_vars,
                &augmented_primary,
                *direction,
                &slack_vars,
                &penalty_sum,
                epsilon_augmentation,
                timeout,
            )),
            #[cfg(not(feature = "coin_cbc"))]
            crate::solver_enum::Solver::CoinCbc => Err(AugmeconError::UnsupportedSolver(
                "CoinCbc solver is not available. Enable the 'coin_cbc' feature to use it."
                    .to_string(),
            )),
            #[cfg(feature = "highs")]
            crate::solver_enum::Solver::HiGHS => Ok(self.solve_with_slack_highs_impl(
                prob_vars,
                &augmented_primary,
                *direction,
                &slack_vars,
                &penalty_sum,
                epsilon_augmentation,
                timeout,
            )),
            #[cfg(not(feature = "highs"))]
            crate::solver_enum::Solver::HiGHS => Err(AugmeconError::UnsupportedSolver(
                "HiGHS solver is not available. Enable the 'highs' feature to use it.".to_string(),
            )),
            #[cfg(feature = "scip")]
            crate::solver_enum::Solver::SCIP => Ok(self.solve_with_slack_scip_impl(
                prob_vars,
                &augmented_primary,
                *direction,
                &slack_vars,
                &penalty_sum,
                epsilon_augmentation,
                timeout,
            )),
            #[cfg(not(feature = "scip"))]
            crate::solver_enum::Solver::SCIP => Err(AugmeconError::UnsupportedSolver(
                "SCIP solver is not available. Enable the 'scip' feature to use it.".to_string(),
            )),
            #[cfg(feature = "gurobi")]
            crate::solver_enum::Solver::Gurobi => Ok(self.solve_with_slack_gurobi_impl(
                prob_vars,
                &augmented_primary,
                *direction,
                &slack_vars,
                &penalty_sum,
                epsilon_augmentation,
                timeout,
            )),
            #[cfg(not(feature = "gurobi"))]
            crate::solver_enum::Solver::Gurobi => Err(AugmeconError::UnsupportedSolver(
                "Gurobi solver is not available. Enable the 'gurobi' feature to use it."
                    .to_string(),
            )),
        }
    }

    /// Create slack variables and penalty sum for epsilon constraints
    fn create_slack_variables_and_penalty(
        &self,
        prob_vars: &mut good_lp::ProblemVariables,
    ) -> (HashMap<usize, good_lp::Variable>, Expression) {
        let mut slack_vars = HashMap::new();
        let mut penalty_sum = Expression::from(0.0);

        for (&obj_idx, &_epsilon_val) in &self.epsilon_values {
            if obj_idx < self.problem.objectives.len() {
                let slack_var = prob_vars.add(variable().min(0.0)); // Non-negative slack
                slack_vars.insert(obj_idx, slack_var);

                // Problem (P5) of Mesquita-Cunha et al. (EJOR 306, 2023):
                //   max z_q(x) + rho * sum_{k != q} 10^(k-1) * s_k / r_k
                // with k the 1-based objective index, so for the 0-based
                // `obj_idx` the exponent is obj_idx itself. This previously read
                // 10^-(obj_idx+1), i.e. 10^-2 where the paper asks for 10^+1 --
                // a factor of 1000 in the wrong direction, which combined with
                // rho = 1e-6 in the main loop left the augmentation 1e6 times
                // weaker than Theorem 3 requires for efficiency.
                let weight = 10_f64.powi(i32::try_from(obj_idx).unwrap_or_default());
                let range = self
                    .objective_ranges
                    .get(&obj_idx)
                    .copied()
                    .unwrap_or(1000.0);

                // Handle the case where range is 0 (all values in payoff table are the same)
                // This happens when an objective has the same optimal value regardless of other objectives
                let normalized_weight = if range.abs() < 1e-10 {
                    // If range is effectively zero, use just the weight without normalization
                    // This prevents division by zero while maintaining the hierarchical weighting
                    weight
                } else {
                    weight / range
                };

                penalty_sum += normalized_weight * slack_var;
            }
        }

        (slack_vars, penalty_sum)
    }

    /// Build the primary objective expression
    fn build_primary_objective_expression(&self) -> Expression {
        if self.primary_objective < self.problem.objectives.len() {
            let (obj_expr, _) = &self.problem.objectives[self.primary_objective];
            obj_expr.clone()
        } else {
            Expression::from(0.0)
        }
    }

    /// Validate that the primary objective index is valid
    const fn validate_primary_objective(&self) -> Result<()> {
        if self.primary_objective >= self.problem.num_objectives() {
            return Err(AugmeconError::InvalidObjectiveCount(self.primary_objective));
        }
        Ok(())
    }

    /// Add common constraints to the model
    fn add_constraints_to_model<T: SolverModel>(
        &self,
        model: &mut T,
        slack_vars: &HashMap<usize, good_lp::Variable>,
    ) {
        // Add original constraints - these should work since we use the same variable set
        for constraint in &self.problem.constraints {
            model.add_constraint(constraint.clone());
        }

        // Add epsilon constraints: objective_k(x) - slack_k = epsilon_k
        // This is the standard formulation for augmented ε-constraint methods
        log::debug!("Adding {} epsilon constraints", self.epsilon_values.len());
        for (&obj_idx, &epsilon_val) in &self.epsilon_values {
            if obj_idx < self.problem.objectives.len() {
                let (obj_expr, direction) = &self.problem.objectives[obj_idx];
                let mut constraint_expr = obj_expr.clone();

                if let Some(&slack_var) = slack_vars.get(&obj_idx) {
                    match direction {
                        crate::model::ObjectiveDirection::Maximize => {
                            constraint_expr -= slack_var;
                        }
                        crate::model::ObjectiveDirection::Minimize => {
                            constraint_expr += slack_var;
                        }
                    }
                }

                log::trace!("Adding constraint for objective {obj_idx}: obj_expr +/- slack = {epsilon_val} (direction: {direction:?})");

                model.add_constraint(constraint!(constraint_expr == epsilon_val));
            }
        }
    }

    /// Extract solution and create Solution object
    fn extract_solution<S: GoodLpSolution>(
        &self,
        solution: &S,
        penalty_sum: &Expression,
        augmented_primary: &Expression,
        epsilon_augmentation: f64,
        slack_vars: &HashMap<usize, good_lp::Variable>,
    ) -> Solution {
        // Extract variable values using the original variable map
        let mut variable_values = HashMap::new();
        for (name, &var) in &self.problem.var_map {
            variable_values.insert(name.clone(), solution.value(var));
        }

        // Calculate objective values by evaluating expressions with the solution
        let mut objective_values = self.calculate_objective_values(solution);

        // Recalculate the primary objective value without the penalty
        self.recalculate_primary_objective(
            penalty_sum,
            augmented_primary,
            epsilon_augmentation,
            solution,
            &mut objective_values,
        );

        log::debug!("Solution found with objectives: {objective_values:?}");

        // Debug: Check if the solution satisfies epsilon constraints (only in trace level)
        log::trace!("Verifying epsilon constraints:");
        for (&obj_idx, &epsilon_val) in &self.epsilon_values {
            if obj_idx < self.problem.objectives.len() && obj_idx < objective_values.len() {
                let actual_value = objective_values[obj_idx];
                let slack_value = if let Some(&slack_var) = slack_vars.get(&obj_idx) {
                    solution.value(slack_var)
                } else {
                    0.0
                };
                log::trace!("obj{obj_idx} = {actual_value}, epsilon = {epsilon_val}, slack = {slack_value}, constraint: {actual_value} - {slack_value} == {epsilon_val} -> diff: {}", 
                    (actual_value - slack_value - epsilon_val).abs());
            }
        }

        Solution::new(objective_values, variable_values)
    }

    /// Extract solution with slack values
    fn extract_solution_with_slack<S: GoodLpSolution>(
        &self,
        solution: &S,
        penalty_sum: &Expression,
        augmented_primary: &Expression,
        epsilon_augmentation: f64,
        slack_vars: &HashMap<usize, good_lp::Variable>,
    ) -> SolutionWithSlack {
        // Extract variable values using the original variable map
        let mut variable_values = HashMap::new();
        for (name, &var) in &self.problem.var_map {
            variable_values.insert(name.clone(), solution.value(var));
        }

        // Extract slack variable values
        let mut extracted_slack_values = HashMap::new();
        for (&obj_idx, slack_var) in slack_vars {
            let slack_value = solution.value(*slack_var);
            extracted_slack_values.insert(obj_idx, slack_value);
            log::trace!("Slack for objective {obj_idx}: {slack_value}");
        }

        // Calculate objective values by evaluating expressions with the solution
        let mut objective_values = self.calculate_objective_values(solution);

        // Recalculate the primary objective value without the penalty
        self.recalculate_primary_objective(
            penalty_sum,
            augmented_primary,
            epsilon_augmentation,
            solution,
            &mut objective_values,
        );

        log::debug!("Solution found with objectives: {objective_values:?}");
        log::debug!("Slack values: {extracted_slack_values:?}");

        // Debug: Check if the solution satisfies epsilon constraints (only in trace level)
        log::trace!("Verifying epsilon constraints:");
        for (&obj_idx, &epsilon_val) in &self.epsilon_values {
            if obj_idx < self.problem.objectives.len() && obj_idx < objective_values.len() {
                let actual_value = objective_values[obj_idx];
                let slack_value = extracted_slack_values.get(&obj_idx).copied().unwrap_or(0.0);
                log::trace!("obj{obj_idx} = {actual_value}, epsilon = {epsilon_val}, slack = {slack_value}, constraint: {actual_value} - {slack_value} == {epsilon_val} -> diff: {}", 
                    (actual_value - slack_value - epsilon_val).abs());
            }
        }

        let sol = Solution::new(objective_values, variable_values);
        SolutionWithSlack::new(sol, extracted_slack_values)
    }

    /// Solve with Default solver implementation
    fn solve_with_default_solver_impl(
        &self,
        prob_vars: good_lp::ProblemVariables,
        augmented_primary: &Expression,
        direction: crate::model::ObjectiveDirection,
        slack_vars: &HashMap<usize, good_lp::Variable>,
        penalty_sum: &Expression,
        epsilon_augmentation: f64,
    ) -> std::option::Option<solution::Solution> {
        let problem = match direction {
            crate::model::ObjectiveDirection::Minimize => {
                prob_vars.minimise(augmented_primary.clone())
            }
            crate::model::ObjectiveDirection::Maximize => {
                prob_vars.maximise(augmented_primary.clone())
            }
        };

        let mut model = problem.using(create_solver());

        // Note: LpSolver (Gurobi) doesn't expose set_parameter method
        // Apply solver parameters if the solver supports them
        // if self.options.solver.supports_parameters() {
        //     for (key, value) in &self.options.solver_parameters {
        //         log::debug!("Default solver: Setting parameter: {key} = {value}");
        //         model.set_parameter(key, value);
        //     }
        // } else if !self.options.solver_parameters.is_empty() {
        //     log::warn!(
        //         "Default solver does not support parameters, but {} parameters were specified",
        //         self.options.solver_parameters.len()
        //     );
        // }

        // Add constraints
        self.add_constraints_to_model(&mut model, slack_vars);

        // Solve the problem
        log::debug!(
            "Solving epsilon-constraint problem with {} epsilon constraints",
            self.epsilon_values.len()
        );

        match model.solve() {
            Ok(solution) => {
                let sol = self.extract_solution(
                    &solution,
                    penalty_sum,
                    augmented_primary,
                    epsilon_augmentation,
                    slack_vars,
                );
                Some(sol)
            }
            Err(e) => {
                log::debug!("Epsilon-constraint problem failed: {e:?}");
                None
            }
        }
    }

    /// Solve with slack - Default solver implementation (uses Gurobi via lp-solvers)
    fn solve_with_slack_default_solver_impl(
        &self,
        prob_vars: good_lp::ProblemVariables,
        augmented_primary: &Expression,
        direction: crate::model::ObjectiveDirection,
        slack_vars: &HashMap<usize, good_lp::Variable>,
        penalty_sum: &Expression,
        epsilon_augmentation: f64,
        timeout: Option<Duration>,
    ) -> EpsilonSolveOutcome<SolutionWithSlack> {
        let problem = match direction {
            crate::model::ObjectiveDirection::Minimize => {
                prob_vars.minimise(augmented_primary.clone())
            }
            crate::model::ObjectiveDirection::Maximize => {
                prob_vars.maximise(augmented_primary.clone())
            }
        };

        let mut model = if let Some(timeout_duration) = timeout {
            problem.using(create_solver_with_timeout(timeout_duration))
        } else {
            problem.using(create_solver())
        };

        // Add constraints
        self.add_constraints_to_model(&mut model, slack_vars);

        // Solve the problem - log what we're optimizing and the constraints
        log::info!(
            "Solving ε-constraint: optimize obj[{}], constraints: {:?}, timeout: {}s",
            self.primary_objective,
            self.epsilon_values,
            timeout.map_or(0, |t| t.as_secs())
        );

        match model.solve() {
            Ok(solution) => {
                let sol = self.extract_solution_with_slack(
                    &solution,
                    penalty_sum,
                    augmented_primary,
                    epsilon_augmentation,
                    slack_vars,
                );
                log::info!(
                    "ε-constraint solved: obj[{}]={:.2}, feasible={}, slacks present={}",
                    self.primary_objective,
                    sol.solution.objectives()[self.primary_objective],
                    sol.solution.feasible,
                    !sol.slack_values.is_empty()
                );
                EpsilonSolveOutcome::Solved(sol)
            }
            Err(e) => {
                log::info!(
                    "ε-constraint did not return a solution: obj[{}], constraints: {:?}: {e}",
                    self.primary_objective,
                    self.epsilon_values
                );
                classify_solve_error(&e)
            }
        }
    }

    /// Solve with `CoinCbc` solver implementation
    #[cfg(feature = "coin_cbc")]
    fn solve_with_coin_cbc_impl(
        &self,
        prob_vars: good_lp::ProblemVariables,
        augmented_primary: &Expression,
        direction: crate::model::ObjectiveDirection,
        slack_vars: &HashMap<usize, good_lp::Variable>,
        penalty_sum: &Expression,
        epsilon_augmentation: f64,
    ) -> Option<Solution> {
        let problem = match direction {
            crate::model::ObjectiveDirection::Minimize => {
                prob_vars.minimise(augmented_primary.clone())
            }
            crate::model::ObjectiveDirection::Maximize => {
                prob_vars.maximise(augmented_primary.clone())
            }
        };

        let mut model = problem.using(coin_cbc::coin_cbc);

        // Apply solver parameters if specified
        for (key, value) in &self.options.solver_parameters {
            log::debug!("CoinCbc solver: Setting parameter: {key} = {value}");
            model.set_parameter(key, value);
        }

        // Add constraints
        self.add_constraints_to_model(&mut model, slack_vars);

        // Solve the problem
        log::debug!(
            "Solving epsilon-constraint problem with {} epsilon constraints using CoinCbc",
            self.epsilon_values.len()
        );

        match model.solve() {
            Ok(solution) => {
                let sol = self.extract_solution(
                    &solution,
                    penalty_sum,
                    augmented_primary,
                    epsilon_augmentation,
                    slack_vars,
                );
                Some(sol)
            }
            Err(e) => {
                log::debug!("Epsilon-constraint problem failed with CoinCbc: {e:?}");
                None
            }
        }
    }

    /// Solve with `HiGHS` solver implementation
    #[cfg(feature = "highs")]
    fn solve_with_highs_impl(
        &self,
        prob_vars: good_lp::ProblemVariables,
        augmented_primary: &Expression,
        direction: crate::model::ObjectiveDirection,
        slack_vars: &HashMap<usize, good_lp::Variable>,
        penalty_sum: &Expression,
        epsilon_augmentation: f64,
    ) -> Option<Solution> {
        let problem = match direction {
            crate::model::ObjectiveDirection::Minimize => {
                prob_vars.minimise(augmented_primary.clone())
            }
            crate::model::ObjectiveDirection::Maximize => {
                prob_vars.maximise(augmented_primary.clone())
            }
        };

        let mut model = problem.using(highs::highs);

        // Note: HiGHS solver doesn't support generic parameter setting via set_parameter
        if !self.options.solver_parameters.is_empty() {
            log::warn!(
                "HiGHS solver does not support generic parameters, ignoring {} parameters",
                self.options.solver_parameters.len()
            );
        }

        // Add constraints
        self.add_constraints_to_model(&mut model, slack_vars);

        // Solve the problem
        log::debug!(
            "Solving epsilon-constraint problem with {} epsilon constraints using HiGHS",
            self.epsilon_values.len()
        );

        match model.solve() {
            Ok(solution) => {
                let sol = self.extract_solution(
                    &solution,
                    penalty_sum,
                    augmented_primary,
                    epsilon_augmentation,
                    slack_vars,
                );
                Some(sol)
            }
            Err(e) => {
                log::debug!("Epsilon-constraint problem failed with HiGHS: {e:?}");
                None
            }
        }
    }

    /// Solve with the native Gurobi (`grb`) backend.
    #[cfg(feature = "gurobi")]
    fn solve_with_gurobi_impl(
        &self,
        prob_vars: good_lp::ProblemVariables,
        augmented_primary: &Expression,
        direction: crate::model::ObjectiveDirection,
        slack_vars: &HashMap<usize, good_lp::Variable>,
        penalty_sum: &Expression,
        epsilon_augmentation: f64,
    ) -> Option<Solution> {
        let problem = match direction {
            crate::model::ObjectiveDirection::Minimize => {
                prob_vars.minimise(augmented_primary.clone())
            }
            crate::model::ObjectiveDirection::Maximize => {
                prob_vars.maximise(augmented_primary.clone())
            }
        };

        let mut model = problem.using(gurobi);
        crate::options::apply_gurobi_options(&mut model, self.options);

        // Note: generic parameter setting is not wired for the Gurobi backend.
        if !self.options.solver_parameters.is_empty() {
            log::warn!(
                "Gurobi backend does not support generic parameters, ignoring {} parameters",
                self.options.solver_parameters.len()
            );
        }

        // Add constraints
        self.add_constraints_to_model(&mut model, slack_vars);

        log::debug!(
            "Solving epsilon-constraint problem with {} epsilon constraints using Gurobi",
            self.epsilon_values.len()
        );

        match model.solve() {
            Ok(solution) => {
                let sol = self.extract_solution(
                    &solution,
                    penalty_sum,
                    augmented_primary,
                    epsilon_augmentation,
                    slack_vars,
                );
                Some(sol)
            }
            Err(e) => {
                log::debug!("Epsilon-constraint problem failed with Gurobi: {e:?}");
                None
            }
        }
    }

    /// Solve with SCIP solver implementation
    #[cfg(feature = "scip")]
    fn solve_with_scip_impl(
        &self,
        prob_vars: good_lp::ProblemVariables,
        augmented_primary: &Expression,
        direction: crate::model::ObjectiveDirection,
        slack_vars: &HashMap<usize, good_lp::Variable>,
        penalty_sum: &Expression,
        epsilon_augmentation: f64,
    ) -> Option<Solution> {
        let problem = match direction {
            crate::model::ObjectiveDirection::Minimize => {
                prob_vars.minimise(augmented_primary.clone())
            }
            crate::model::ObjectiveDirection::Maximize => {
                prob_vars.maximise(augmented_primary.clone())
            }
        };

        let mut model = problem.using(scip::scip);

        // Note: SCIP solver doesn't support generic parameter setting via set_parameter
        if !self.options.solver_parameters.is_empty() {
            log::warn!(
                "SCIP solver does not support generic parameters, ignoring {} parameters",
                self.options.solver_parameters.len()
            );
        }

        // Add constraints
        self.add_constraints_to_model(&mut model, slack_vars);

        // Solve the problem
        log::debug!(
            "Solving epsilon-constraint problem with {} epsilon constraints using SCIP",
            self.epsilon_values.len()
        );

        match model.solve() {
            Ok(solution) => {
                let sol = self.extract_solution(
                    &solution,
                    penalty_sum,
                    augmented_primary,
                    epsilon_augmentation,
                    slack_vars,
                );
                Some(sol)
            }
            Err(e) => {
                log::debug!("Epsilon-constraint problem failed with SCIP: {e:?}");
                None
            }
        }
    }

    /// Solve with slack - `CoinCbc` solver implementation
    #[cfg(feature = "coin_cbc")]
    fn solve_with_slack_coin_cbc_impl(
        &self,
        prob_vars: good_lp::ProblemVariables,
        augmented_primary: &Expression,
        direction: crate::model::ObjectiveDirection,
        slack_vars: &HashMap<usize, good_lp::Variable>,
        penalty_sum: &Expression,
        epsilon_augmentation: f64,
        timeout: Option<Duration>,
    ) -> EpsilonSolveOutcome<SolutionWithSlack> {
        let problem = match direction {
            crate::model::ObjectiveDirection::Minimize => {
                prob_vars.minimise(augmented_primary.clone())
            }
            crate::model::ObjectiveDirection::Maximize => {
                prob_vars.maximise(augmented_primary.clone())
            }
        };

        let mut model = problem.using(coin_cbc::coin_cbc);

        // Bound this subproblem's wall-clock; CBC returns its incumbent at the limit.
        if let Some(t) = timeout {
            // Integer ceil of the duration in whole seconds (no float cast).
            let secs = t.as_secs() + u64::from(t.subsec_nanos() > 0);
            model.set_parameter("sec", &secs.to_string());
        }

        // Apply solver parameters if specified
        for (key, value) in &self.options.solver_parameters {
            log::debug!("CoinCbc solver: Setting parameter: {key} = {value}");
            model.set_parameter(key, value);
        }

        // Add constraints
        self.add_constraints_to_model(&mut model, slack_vars);

        // Solve the problem
        log::info!(
            "Solving ε-constraint: optimize obj[{}], constraints: {:?} using CoinCbc",
            self.primary_objective,
            self.epsilon_values
        );

        match model.solve() {
            Ok(solution) => {
                let sol = self.extract_solution_with_slack(
                    &solution,
                    penalty_sum,
                    augmented_primary,
                    epsilon_augmentation,
                    slack_vars,
                );
                log::info!(
                    "ε-constraint solved: obj[{}]={:.2}, feasible={}, slacks present={}",
                    self.primary_objective,
                    sol.solution.objectives()[self.primary_objective],
                    sol.solution.feasible,
                    !sol.slack_values.is_empty()
                );
                EpsilonSolveOutcome::Solved(sol)
            }
            Err(e) => {
                log::info!(
                    "ε-constraint did not return a solution: obj[{}], constraints: {:?}: {e}",
                    self.primary_objective,
                    self.epsilon_values
                );
                classify_solve_error(&e)
            }
        }
    }

    /// Solve with slack - `HiGHS` solver implementation
    #[cfg(feature = "highs")]
    fn solve_with_slack_highs_impl(
        &self,
        prob_vars: good_lp::ProblemVariables,
        augmented_primary: &Expression,
        direction: crate::model::ObjectiveDirection,
        slack_vars: &HashMap<usize, good_lp::Variable>,
        penalty_sum: &Expression,
        epsilon_augmentation: f64,
        timeout: Option<Duration>,
    ) -> EpsilonSolveOutcome<SolutionWithSlack> {
        crate::verify::bump(&crate::verify::SLACK_SOLVES);
        let problem = match direction {
            crate::model::ObjectiveDirection::Minimize => {
                prob_vars.minimise(augmented_primary.clone())
            }
            crate::model::ObjectiveDirection::Maximize => {
                prob_vars.maximise(augmented_primary.clone())
            }
        };

        let mut model = problem.using(highs::highs);

        // Bound this subproblem's wall-clock. HiGHS has no default stop, so without this a
        // single hard ε-solve runs unbounded (blowing the GPBA budget on one point). On the
        // time limit HiGHS returns its incumbent (good_lp maps ReachedTimeLimit -> Ok), which
        // is a valid feasible Pareto candidate — the front dedups any weakly-dominated point.
        if let Some(t) = timeout {
            model = model.set_time_limit(t.as_secs_f64());
        }
        // Disable presolve so the time limit is actually honored on large models
        // (HiGHS presolve doesn't poll the clock — see single_objective.rs).
        model = model.set_option("presolve", "off");

        // Note: HiGHS solver doesn't support generic parameter setting via set_parameter
        if !self.options.solver_parameters.is_empty() {
            log::warn!(
                "HiGHS solver does not support generic parameters, ignoring {} parameters",
                self.options.solver_parameters.len()
            );
        }

        // Add constraints
        self.add_constraints_to_model(&mut model, slack_vars);

        // Solve the problem
        log::info!(
            "Solving ε-constraint: optimize obj[{}], constraints: {:?} using HiGHS",
            self.primary_objective,
            self.epsilon_values
        );

        match model.solve() {
            Ok(solution) => {
                let sol = self.extract_solution_with_slack(
                    &solution,
                    penalty_sum,
                    augmented_primary,
                    epsilon_augmentation,
                    slack_vars,
                );
                log::info!(
                    "ε-constraint solved: obj[{}]={:.2}, feasible={}, slacks present={}",
                    self.primary_objective,
                    sol.solution.objectives()[self.primary_objective],
                    sol.solution.feasible,
                    !sol.slack_values.is_empty()
                );
                EpsilonSolveOutcome::Solved(sol)
            }
            Err(e) => {
                log::info!(
                    "ε-constraint did not return a solution: obj[{}], constraints: {:?}: {e}",
                    self.primary_objective,
                    self.epsilon_values
                );
                classify_solve_error(&e)
            }
        }
    }

    /// Solve with slack - native Gurobi (`grb`) backend.
    #[cfg(feature = "gurobi")]
    fn solve_with_slack_gurobi_impl(
        &self,
        prob_vars: good_lp::ProblemVariables,
        augmented_primary: &Expression,
        direction: crate::model::ObjectiveDirection,
        slack_vars: &HashMap<usize, good_lp::Variable>,
        penalty_sum: &Expression,
        epsilon_augmentation: f64,
        timeout: Option<Duration>,
    ) -> EpsilonSolveOutcome<SolutionWithSlack> {
        crate::verify::bump(&crate::verify::SLACK_SOLVES);
        let problem = match direction {
            crate::model::ObjectiveDirection::Minimize => {
                prob_vars.minimise(augmented_primary.clone())
            }
            crate::model::ObjectiveDirection::Maximize => {
                prob_vars.maximise(augmented_primary.clone())
            }
        };

        let mut model = problem.using(gurobi);
        crate::options::apply_gurobi_options(&mut model, self.options);

        // Bound this subproblem's wall-clock. Gurobi honours TimeLimit natively
        // (no presolve workaround needed) and returns its incumbent on the limit,
        // which good_lp maps to Ok — a valid feasible Pareto candidate.
        if let Some(t) = timeout {
            model = model.with_time_limit(t.as_secs_f64());
        }

        if !self.options.solver_parameters.is_empty() {
            log::warn!(
                "Gurobi backend does not support generic parameters, ignoring {} parameters",
                self.options.solver_parameters.len()
            );
        }

        // Add constraints
        self.add_constraints_to_model(&mut model, slack_vars);

        log::info!(
            "Solving ε-constraint: optimize obj[{}], constraints: {:?} using Gurobi",
            self.primary_objective,
            self.epsilon_values
        );

        match model.solve() {
            Ok(mut solution) => {
                let mut sol = self.extract_solution_with_slack(
                    &solution,
                    penalty_sum,
                    augmented_primary,
                    epsilon_augmentation,
                    slack_vars,
                );
                // Free Pareto candidates: solutions this solve already
                // found and would otherwise throw away.
                if self.options.solution_pool_size > 0 {
                    sol.pool = self.harvest_pool(&mut solution);
                    if !sol.pool.is_empty() {
                        log::debug!("Harvested {} pooled candidates", sol.pool.len());
                    }
                }
                log::info!(
                    "ε-constraint solved: obj[{}]={:.2}, feasible={}, slacks present={}",
                    self.primary_objective,
                    sol.solution.objectives()[self.primary_objective],
                    sol.solution.feasible,
                    !sol.slack_values.is_empty()
                );
                EpsilonSolveOutcome::Solved(sol)
            }
            Err(e) => {
                log::info!(
                    "ε-constraint did not return a solution: obj[{}], constraints: {:?}: {e}",
                    self.primary_objective,
                    self.epsilon_values
                );
                classify_solve_error(&e)
            }
        }
    }

    /// Turn a solve's retained solution pool into Pareto candidates.
    ///
    /// Only the decision variables are read back; objective values are
    /// recomputed from the problem's own expressions rather than taken from
    /// Gurobi, because the model it solved carries the augmentation term and
    /// its objective value is therefore not the problem's.
    #[cfg(feature = "gurobi")]
    fn harvest_pool(&self, solved: &mut good_lp::solvers::gurobi::GurobiSolved) -> Vec<Solution> {
        self.harvest_pool_values(solved.solution_pool())
    }

    /// Turn a raw solution pool into `Solution`s.
    ///
    /// Split out so a reused model, which reads its pool from the model rather
    /// than from a consumed `GurobiSolved`, harvests through exactly the same
    /// code as the rebuild path.
    #[cfg(feature = "gurobi")]
    fn harvest_pool_values(&self, pool: Vec<HashMap<good_lp::Variable, f64>>) -> Vec<Solution> {
        let mut out = Vec::with_capacity(pool.len());
        // Skip index 0: that is the incumbent, already reported separately.
        for values in pool.into_iter().skip(1) {
            let objective_values: Vec<f64> = self
                .problem
                .objectives
                .iter()
                .map(|(expr, _)| {
                    good_lp::IntoAffineExpression::linear_coefficients(expr)
                        .map(|(v, c)| c * values.get(&v).copied().unwrap_or(0.0))
                        .sum::<f64>()
                        .round()
                })
                .collect();
            let mut decision_variables = HashMap::with_capacity(self.problem.var_map.len());
            for (name, var) in &self.problem.var_map {
                decision_variables.insert(name.clone(), values.get(var).copied().unwrap_or(0.0));
            }
            out.push(Solution::new(objective_values, decision_variables));
        }
        out
    }

    /// Solve with slack - SCIP solver implementation
    #[cfg(feature = "scip")]
    fn solve_with_slack_scip_impl(
        &self,
        prob_vars: good_lp::ProblemVariables,
        augmented_primary: &Expression,
        direction: crate::model::ObjectiveDirection,
        slack_vars: &HashMap<usize, good_lp::Variable>,
        penalty_sum: &Expression,
        epsilon_augmentation: f64,
        _timeout: Option<Duration>,
    ) -> EpsilonSolveOutcome<SolutionWithSlack> {
        let problem = match direction {
            crate::model::ObjectiveDirection::Minimize => {
                prob_vars.minimise(augmented_primary.clone())
            }
            crate::model::ObjectiveDirection::Maximize => {
                prob_vars.maximise(augmented_primary.clone())
            }
        };

        let mut model = problem.using(scip::scip);

        // Note: SCIP solver doesn't support generic parameter setting via set_parameter
        if !self.options.solver_parameters.is_empty() {
            log::warn!(
                "SCIP solver does not support generic parameters, ignoring {} parameters",
                self.options.solver_parameters.len()
            );
        }

        // Add constraints
        self.add_constraints_to_model(&mut model, slack_vars);

        // Solve the problem
        log::info!(
            "Solving ε-constraint: optimize obj[{}], constraints: {:?} using SCIP",
            self.primary_objective,
            self.epsilon_values
        );

        match model.solve() {
            Ok(solution) => {
                let sol = self.extract_solution_with_slack(
                    &solution,
                    penalty_sum,
                    augmented_primary,
                    epsilon_augmentation,
                    slack_vars,
                );
                log::info!(
                    "ε-constraint solved: obj[{}]={:.2}, feasible={}, slacks present={}",
                    self.primary_objective,
                    sol.solution.objectives()[self.primary_objective],
                    sol.solution.feasible,
                    !sol.slack_values.is_empty()
                );
                EpsilonSolveOutcome::Solved(sol)
            }
            Err(e) => {
                log::info!(
                    "ε-constraint did not return a solution: obj[{}], constraints: {:?}: {e}",
                    self.primary_objective,
                    self.epsilon_values
                );
                classify_solve_error(&e)
            }
        }
    }
}
