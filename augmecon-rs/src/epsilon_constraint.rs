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
    solution::{self, Solution},
};
use good_lp::{
    constraint, variable, Expression, Solution as GoodLpSolution, SolverModel, WithTimeLimit,
    coin_cbc, default_solver, highs,
};
use std::collections::HashMap;
use std::time::Duration;

/// Solution with additional slack variable values for bypass coefficient optimization
#[derive(Debug, Clone)]
pub struct SolutionWithSlack {
    /// The main solution
    pub solution: Solution,
    /// Slack variable values indexed by objective index
    pub slack_values: HashMap<usize, f64>,
}

impl SolutionWithSlack {
    /// Create a new solution with slack values
    #[must_use]
    pub const fn new(solution: Solution, slack_values: HashMap<usize, f64>) -> Self {
        Self {
            solution,
            slack_values,
        }
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
        let epsilon_augmentation = 1e2; // Extremely large augmentation to prevent slack usage

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
            crate::solver_enum::Solver::Default => {
                Ok(self.solve_with_default_solver_impl(prob_vars, &augmented_primary, *direction, &slack_vars, &penalty_sum, epsilon_augmentation))
            }
            crate::solver_enum::Solver::CoinCbc => {
                Ok(self.solve_with_coin_cbc_solver_impl(prob_vars, &augmented_primary, *direction, &slack_vars, &penalty_sum, epsilon_augmentation))
            }
            crate::solver_enum::Solver::HiGHS => {
                Ok(self.solve_with_highs_solver_impl(prob_vars, &augmented_primary, *direction, &slack_vars, &penalty_sum, epsilon_augmentation))
            }
        }
    }

    /// Solve the epsilon-constraint problem and return slack values
    ///
    /// This version returns both the solution and the slack variable values
    /// which are needed for bypass coefficient optimization.
    ///
    /// # Errors
    /// Returns error if optimization fails
    pub fn solve_with_slack(self, timeout: Option<Duration>) -> Result<Option<SolutionWithSlack>> {
        self.validate_primary_objective()?;

        // Build the primary objective expression directly from the stored expression
        let primary_objective_expr = self.build_primary_objective_expression();

        // Add augmentation term based on GPBA paper formulation (Problem 5)
        // Primary objective += ρ * Σ(10^(k-1) * s_k / r_k)
        // Use small augmentation to encourage slack usage for bypass calculation
        let epsilon_augmentation = 1e-6; // Small augmentation to compute slack values

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
            crate::solver_enum::Solver::Default => {
                Ok(self.solve_with_slack_default_solver_impl(prob_vars, &augmented_primary, *direction, &slack_vars, &penalty_sum, epsilon_augmentation, timeout))
            }
            crate::solver_enum::Solver::CoinCbc => {
                Ok(self.solve_with_slack_coin_cbc_solver_impl(prob_vars, &augmented_primary, *direction, &slack_vars, &penalty_sum, epsilon_augmentation, timeout))
            }
            crate::solver_enum::Solver::HiGHS => {
                Ok(self.solve_with_slack_highs_solver_impl(prob_vars, &augmented_primary, *direction, &slack_vars, &penalty_sum, epsilon_augmentation, timeout))
            }
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

                // Add slack to primary objective with correct weight using standard formulation
                // Uses: eps * (10^(-o+1) * slack / range) where o is 1-based objective index
                let weight = 10_f64.powi(-(i32::try_from(obj_idx).unwrap_or_default() + 1));
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
        
        let mut model = default_solver(problem);

        // Apply solver parameters if the solver supports them
        if self.options.solver.supports_parameters() {
            for (key, value) in &self.options.solver_parameters {
                log::debug!("Default solver: Setting parameter: {key} = {value}");
                model.set_parameter(key, value);
            }
        } else if !self.options.solver_parameters.is_empty() {
            log::warn!(
                "Default solver does not support parameters, but {} parameters were specified",
                self.options.solver_parameters.len()
            );
        }

        // Add constraints
        self.add_constraints_to_model(&mut model, slack_vars);

        // Solve the problem
        log::debug!(
            "Solving epsilon-constraint problem with {} epsilon constraints",
            self.epsilon_values.len()
        );

        match model.solve() {
            Ok(solution) => {
                let sol = self.extract_solution(&solution, penalty_sum, augmented_primary, epsilon_augmentation, slack_vars);
                Some(sol)
            }
            Err(e) => {
                log::debug!("Epsilon-constraint problem failed: {e:?}");
                None
            }
        }
    }

    /// Solve with COIN-OR CBC solver implementation
    fn solve_with_coin_cbc_solver_impl(
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
        
        let mut model = coin_cbc(problem);

        // Apply solver parameters if the solver supports them
        if self.options.solver.supports_parameters() {
            for (key, value) in &self.options.solver_parameters {
                log::debug!("COIN-OR CBC solver: Setting parameter: {key} = {value}");
                model.set_parameter(key, value);
            }
        } else if !self.options.solver_parameters.is_empty() {
            log::warn!(
                "COIN-OR CBC solver does not support parameters, but {} parameters were specified",
                self.options.solver_parameters.len()
            );
        }

        // Add constraints
        self.add_constraints_to_model(&mut model, slack_vars);

        // Solve the problem
        log::debug!(
            "Solving epsilon-constraint problem with {} epsilon constraints",
            self.epsilon_values.len()
        );

        match model.solve() {
            Ok(solution) => {
                let sol = self.extract_solution(&solution, penalty_sum, augmented_primary, epsilon_augmentation, slack_vars);
                Some(sol)
            }
            Err(e) => {
                log::debug!("Epsilon-constraint problem failed: {e:?}");
                None
            }
        }
    }

    /// Solve with `HiGHS` solver implementation
    fn solve_with_highs_solver_impl(
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
        
        let mut model = highs(problem);

        // Apply solver parameters if the solver supports them (HiGHS doesn't support generic parameters)
        if self.options.solver.supports_parameters() {
            for (key, value) in &self.options.solver_parameters {
                log::debug!("HiGHS solver: Setting parameter: {key} = {value}");
                // Note: HiGHS doesn't have a generic set_parameter method
                // Parameters would need to be handled differently for HiGHS
            }
        } else if !self.options.solver_parameters.is_empty() {
            log::warn!(
                "HiGHS solver does not support parameters, but {} parameters were specified",
                self.options.solver_parameters.len()
            );
        }

        // Add constraints
        self.add_constraints_to_model(&mut model, slack_vars);

        // Solve the problem
        log::debug!(
            "Solving epsilon-constraint problem with {} epsilon constraints",
            self.epsilon_values.len()
        );

        match model.solve() {
            Ok(solution) => {
                let sol = self.extract_solution(&solution, penalty_sum, augmented_primary, epsilon_augmentation, slack_vars);
                Some(sol)
            }
            Err(e) => {
                log::debug!("Epsilon-constraint problem failed: {e:?}");
                None
            }
        }
    }

    /// Solve with slack - Default solver implementation
    fn solve_with_slack_default_solver_impl(
        &self,
        prob_vars: good_lp::ProblemVariables,
        augmented_primary: &Expression,
        direction: crate::model::ObjectiveDirection,
        slack_vars: &HashMap<usize, good_lp::Variable>,
        penalty_sum: &Expression,
        epsilon_augmentation: f64,
        timeout: Option<Duration>,
    ) -> Option<SolutionWithSlack> {
        let problem = match direction {
            crate::model::ObjectiveDirection::Minimize => {
                prob_vars.minimise(augmented_primary.clone())
            }
            crate::model::ObjectiveDirection::Maximize => {
                prob_vars.maximise(augmented_primary.clone())
            }
        };
        
        let mut model = default_solver(problem);

        // Apply solver parameters if the solver supports them
        if self.options.solver.supports_parameters() {
            for (key, value) in &self.options.solver_parameters {
                log::debug!("Default solver slack: Setting parameter: {key} = {value}");
                model.set_parameter(key, value);
            }
        } else if !self.options.solver_parameters.is_empty() {
            log::warn!(
                "Default solver does not support parameters, but {} parameters were specified",
                self.options.solver_parameters.len()
            );
        }

        // Add constraints
        self.add_constraints_to_model(&mut model, slack_vars);

        // Apply timeout if specified
        let final_model = if let Some(timeout_duration) = timeout {
            log::debug!("Applying timeout of {timeout_duration:?} to epsilon-constraint solver");
            model.with_time_limit(timeout_duration.as_secs_f64())
        } else {
            model
        };

        // Solve the problem
        log::debug!(
            "Solving epsilon-constraint problem with {} epsilon constraints and slack variables",
            self.epsilon_values.len()
        );

        match final_model.solve() {
            Ok(solution) => {
                let sol = self.extract_solution_with_slack(&solution, penalty_sum, augmented_primary, epsilon_augmentation, slack_vars);
                Some(sol)
            }
            Err(e) => {
                log::debug!("Epsilon-constraint problem failed: {e:?}");
                None
            }
        }
    }

    /// Solve with slack - COIN-OR CBC solver implementation
    fn solve_with_slack_coin_cbc_solver_impl(
        &self,
        prob_vars: good_lp::ProblemVariables,
        augmented_primary: &Expression,
        direction: crate::model::ObjectiveDirection,
        slack_vars: &HashMap<usize, good_lp::Variable>,
        penalty_sum: &Expression,
        epsilon_augmentation: f64,
        timeout: Option<Duration>,
    ) -> Option<SolutionWithSlack> {
        let problem = match direction {
            crate::model::ObjectiveDirection::Minimize => {
                prob_vars.minimise(augmented_primary.clone())
            }
            crate::model::ObjectiveDirection::Maximize => {
                prob_vars.maximise(augmented_primary.clone())
            }
        };
        
        let mut model = coin_cbc(problem);

        // Apply solver parameters if the solver supports them
        if self.options.solver.supports_parameters() {
            for (key, value) in &self.options.solver_parameters {
                log::debug!("COIN-OR CBC solver slack: Setting parameter: {key} = {value}");
                model.set_parameter(key, value);
            }
        } else if !self.options.solver_parameters.is_empty() {
            log::warn!(
                "COIN-OR CBC solver does not support parameters, but {} parameters were specified",
                self.options.solver_parameters.len()
            );
        }

        // Add constraints
        self.add_constraints_to_model(&mut model, slack_vars);

        // Apply timeout if specified
        let final_model = if let Some(timeout_duration) = timeout {
            log::debug!("Applying timeout of {timeout_duration:?} to epsilon-constraint solver");
            model.with_time_limit(timeout_duration.as_secs_f64())
        } else {
            model
        };

        // Solve the problem
        log::debug!(
            "Solving epsilon-constraint problem with {} epsilon constraints and slack variables",
            self.epsilon_values.len()
        );

        match final_model.solve() {
            Ok(solution) => {
                let sol = self.extract_solution_with_slack(&solution, penalty_sum, augmented_primary, epsilon_augmentation, slack_vars);
                Some(sol)
            }
            Err(e) => {
                log::debug!("Epsilon-constraint problem failed: {e:?}");
                None
            }
        }
    }

    /// Solve with slack - `HiGHS` solver implementation
    fn solve_with_slack_highs_solver_impl(
        &self,
        prob_vars: good_lp::ProblemVariables,
        augmented_primary: &Expression,
        direction: crate::model::ObjectiveDirection,
        slack_vars: &HashMap<usize, good_lp::Variable>,
        penalty_sum: &Expression,
        epsilon_augmentation: f64,
        timeout: Option<Duration>,
    ) -> Option<SolutionWithSlack> {
        let problem = match direction {
            crate::model::ObjectiveDirection::Minimize => {
                prob_vars.minimise(augmented_primary.clone())
            }
            crate::model::ObjectiveDirection::Maximize => {
                prob_vars.maximise(augmented_primary.clone())
            }
        };
        
        let mut model = highs(problem);

        // Apply solver parameters if the solver supports them (HiGHS doesn't support generic parameters)
        if self.options.solver.supports_parameters() {
            for (key, value) in &self.options.solver_parameters {
                log::debug!("HiGHS solver slack: Setting parameter: {key} = {value}");
                // Note: HiGHS doesn't have a generic set_parameter method
                // Parameters would need to be handled differently for HiGHS
            }
        } else if !self.options.solver_parameters.is_empty() {
            log::warn!(
                "HiGHS solver does not support parameters, but {} parameters were specified",
                self.options.solver_parameters.len()
            );
        }

        // Add constraints
        self.add_constraints_to_model(&mut model, slack_vars);

        // Apply timeout if specified
        let final_model = if let Some(timeout_duration) = timeout {
            log::debug!("Applying timeout of {timeout_duration:?} to epsilon-constraint solver");
            model.with_time_limit(timeout_duration.as_secs_f64())
        } else {
            model
        };

        // Solve the problem
        log::debug!(
            "Solving epsilon-constraint problem with {} epsilon constraints and slack variables",
            self.epsilon_values.len()
        );

        match final_model.solve() {
            Ok(solution) => {
                let sol = self.extract_solution_with_slack(&solution, penalty_sum, augmented_primary, epsilon_augmentation, slack_vars);
                Some(sol)
            }
            Err(e) => {
                log::debug!("Epsilon-constraint problem failed: {e:?}");
                None
            }
        }
    }
}
