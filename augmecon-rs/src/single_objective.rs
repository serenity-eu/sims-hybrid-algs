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
use good_lp::{
    coin_cbc, default_solver, highs, Solution as GoodLpSolution, SolverModel,
    WithTimeLimit,
};
use std::time::Duration;

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
                let model = if matches!(direction, crate::model::ObjectiveDirection::Minimize) {
                    prob_vars.minimise(objective_expr).using(default_solver)
                } else {
                    prob_vars.maximise(objective_expr).using(default_solver)
                };
                // Default solver doesn't support parameters or timeout
                if !self.options.solver_parameters.is_empty() {
                    println!(
                        "DEBUG: Solver {} does not support parameters, ignoring {} parameters",
                        self.options.solver.name(),
                        self.options.solver_parameters.len()
                    );
                }
                self.solve_with_model_common(model, objective_index)
            }
            crate::solver_enum::Solver::CoinCbc => {
                let mut model = if matches!(direction, crate::model::ObjectiveDirection::Minimize) {
                    prob_vars.minimise(objective_expr).using(coin_cbc)
                } else {
                    prob_vars.maximise(objective_expr).using(coin_cbc)
                };
                // Apply solver parameters
                for (key, value) in &self.options.solver_parameters {
                    println!(
                        "DEBUG: Setting {} parameter: {key} = {value}",
                        self.options.solver.name()
                    );
                    model.set_parameter(key, value);
                }
                // Apply timeout if specified
                if let Some(timeout_duration) = effective_timeout {
                    println!("DEBUG: Setting timeout: {timeout_duration:?}");
                    model = model.with_time_limit(timeout_duration.as_secs_f64());
                }
                self.solve_with_model_common(model, objective_index)
            }
            crate::solver_enum::Solver::HiGHS => {
                let mut model = if matches!(direction, crate::model::ObjectiveDirection::Minimize) {
                    prob_vars.minimise(objective_expr).using(highs)
                } else {
                    prob_vars.maximise(objective_expr).using(highs)
                };
                // Apply solver parameters if the solver supports them
                if self.options.solver.supports_parameters() {
                    for (key, value) in &self.options.solver_parameters {
                        println!(
                            "DEBUG: Setting {} parameter: {key} = {value}",
                            self.options.solver.name()
                        );
                        // Note: HiGHS doesn't support generic set_parameter method
                        // Parameters would need to be handled differently for HiGHS
                    }
                } else if !self.options.solver_parameters.is_empty() {
                    println!(
                        "WARNING: {} solver does not support parameters, but {} parameters were specified",
                        self.options.solver.name(),
                        self.options.solver_parameters.len()
                    );
                }
                // Apply timeout if specified
                if let Some(timeout_duration) = effective_timeout {
                    println!("DEBUG: Setting timeout: {timeout_duration:?}");
                    model = model.with_time_limit(timeout_duration.as_secs_f64());
                }
                self.solve_with_model_common(model, objective_index)
            }
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
        println!("DEBUG: Model solved successfully");

        // Extract variable values
        println!("DEBUG: Extracting variable values...");
        let mut variable_values = std::collections::HashMap::new();
        for (name, &var) in &self.problem.var_map {
            let val = solution.value(var);
            println!("DEBUG: Variable {name}: {val}");
            log::debug!("Variable {name}: {val}");
            variable_values.insert(name.clone(), val);
        }

        // Calculate objective values by evaluating expressions with the solution
        println!("DEBUG: Calculating objective values...");
        let mut objective_values = Vec::with_capacity(self.problem.num_objectives());
        for i in 0..self.problem.num_objectives() {
            if i < self.problem.objectives.len() {
                let (obj_expr, _) = &self.problem.objectives[i];
                let obj_value = obj_expr.eval_with(&solution);
                println!("DEBUG: Objective {i}: {obj_value}");
                log::debug!("Objective {i}: {obj_value}");
                objective_values.push(obj_value);
            } else {
                objective_values.push(0.0);
            }
        }

        println!("DEBUG: Single objective solve completed successfully");
        Ok(Solution::new(objective_values, variable_values))
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
                let model = if matches!(direction, crate::model::ObjectiveDirection::Minimize) {
                    prob_vars.minimise(objective_expr).using(default_solver)
                } else {
                    prob_vars.maximise(objective_expr).using(default_solver)
                };
                // Default solver doesn't support parameters or timeout
                if !self.options.solver_parameters.is_empty() {
                    println!(
                        "DEBUG: [solve_objective_with_constraints] Solver {} does not support parameters, ignoring {} parameters",
                        self.options.solver.name(),
                        self.options.solver_parameters.len()
                    );
                }
                self.solve_with_constraints_common(model, objective_index, additional_constraints)
            }
            crate::solver_enum::Solver::CoinCbc => {
                let mut model = if matches!(direction, crate::model::ObjectiveDirection::Minimize) {
                    prob_vars.minimise(objective_expr).using(coin_cbc)
                } else {
                    prob_vars.maximise(objective_expr).using(coin_cbc)
                };
                // Apply solver parameters
                for (key, value) in &self.options.solver_parameters {
                    println!(
                        "DEBUG: [solve_objective_with_constraints] Setting {} parameter: {key} = {value}",
                        self.options.solver.name()
                    );
                    model.set_parameter(key, value);
                }
                // Apply timeout if specified
                if let Some(timeout_duration) = effective_timeout {
                    println!("DEBUG: [solve_objective_with_constraints] Setting timeout: {timeout_duration:?}");
                    model = model.with_time_limit(timeout_duration.as_secs_f64());
                }
                self.solve_with_constraints_common(model, objective_index, additional_constraints)
            }
            crate::solver_enum::Solver::HiGHS => {
                let mut model = if matches!(direction, crate::model::ObjectiveDirection::Minimize) {
                    prob_vars.minimise(objective_expr).using(highs)
                } else {
                    prob_vars.maximise(objective_expr).using(highs)
                };
                // Apply solver parameters if the solver supports them
                if self.options.solver.supports_parameters() {
                    for (key, value) in &self.options.solver_parameters {
                        println!(
                            "DEBUG: [solve_objective_with_constraints] Setting {} parameter: {key} = {value}",
                            self.options.solver.name()
                        );
                        // Note: HiGHS doesn't support generic set_parameter method
                        // Parameters would need to be handled differently for HiGHS
                    }
                } else if !self.options.solver_parameters.is_empty() {
                    println!(
                        "WARNING: [solve_objective_with_constraints] {} solver does not support parameters, but {} parameters were specified",
                        self.options.solver.name(),
                        self.options.solver_parameters.len()
                    );
                }
                // Apply timeout if specified
                if let Some(timeout_duration) = effective_timeout {
                    println!("DEBUG: [solve_objective_with_constraints] Setting timeout: {timeout_duration:?}");
                    model = model.with_time_limit(timeout_duration.as_secs_f64());
                }
                self.solve_with_constraints_common(model, objective_index, additional_constraints)
            }
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
