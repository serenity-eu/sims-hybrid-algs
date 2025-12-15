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
use good_lp::solvers::lp_solvers::{GurobiSolver, WithMaxSeconds};
#[cfg(feature = "coin_cbc")]
use good_lp::solvers::coin_cbc;
#[cfg(feature = "highs")]
use good_lp::solvers::highs;
#[cfg(feature = "scip")]
use good_lp::solvers::scip;
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
    #[allow(clippy::cast_possible_truncation, reason = "Timeout duration in seconds is expected to fit in u32 for Gurobi solver API - values over 4.2 billion seconds (136 years) are not realistic")]
    let seconds = timeout.as_secs() as u32;
    let gurobi = GurobiSolver::new().with_max_seconds(seconds);
    good_lp::solvers::lp_solvers::LpSolver(gurobi)
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
                let model = if matches!(direction, crate::model::ObjectiveDirection::Minimize) {
                    prob_vars.minimise(objective_expr).using(coin_cbc::coin_cbc)
                } else {
                    prob_vars.maximise(objective_expr).using(coin_cbc::coin_cbc)
                };
                self.solve_with_model_common(model, objective_index)
            }
            #[cfg(not(feature = "coin_cbc"))]
            crate::solver_enum::Solver::CoinCbc => Err(AugmeconError::UnsupportedSolver(
                "CoinCbc solver is not available. Enable the 'coin_cbc' feature to use it.".to_string(),
            )),
            #[cfg(feature = "highs")]
            crate::solver_enum::Solver::HiGHS => {
                let model = if matches!(direction, crate::model::ObjectiveDirection::Minimize) {
                    prob_vars.minimise(objective_expr).using(highs::highs)
                } else {
                    prob_vars.maximise(objective_expr).using(highs::highs)
                };
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
                self.solve_with_model_common(model, objective_index)
            }
            #[cfg(not(feature = "scip"))]
            crate::solver_enum::Solver::SCIP => Err(AugmeconError::UnsupportedSolver(
                "SCIP solver is not available. Enable the 'scip' feature to use it.".to_string(),
            )),
        }
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
            println!("DEBUG: Adding constraint {idx}: {constraint:?}");
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
                "CoinCbc solver is not available. Enable the 'coin_cbc' feature to use it.".to_string(),
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
        }
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
                "CoinCbc solver is not available. Enable the 'coin_cbc' feature to use it.".to_string(),
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
