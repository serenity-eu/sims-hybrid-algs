//! Bounds calculation for multi-objective optimization
//!
//! This module provides functionality for computing ideal and nadir points
//! by solving single-objective problems and constructing payoff tables.

use crate::{
    error::{AugmeconError, Result},
    model::MultiObjectiveProblem,
    options::Options,
    single_objective::SingleObjectiveSolver,
    timer::Timer,
};
use good_lp::constraint;

/// Calculator for optimization bounds (ideal and nadir points)
pub struct BoundsCalculator<'a> {
    problem: &'a MultiObjectiveProblem,
    options: &'a Options,
}

impl<'a> BoundsCalculator<'a> {
    /// Create a new bounds calculator for the given problem
    #[must_use]
    pub const fn new(problem: &'a MultiObjectiveProblem, options: &'a Options) -> Self {
        Self { problem, options }
    }

    /// Calculate ideal and nadir points by solving single-objective problems
    ///
    /// Returns (`ideal_point`, `nadir_point`) where:
    /// - `ideal_point`[i] = best possible value for objective i (minimizing it)
    /// - `nadir_point`[i] = worst possible value for objective i (maximizing it)
    ///
    /// This matches Python's GPBA implementation which computes nadir by maximizing each objective.
    ///
    /// # Errors
    /// Returns error if any single-objective optimization fails
    pub fn calculate_bounds(&self, timer: Option<&Timer>) -> Result<(Vec<f64>, Vec<f64>)> {
        let num_objectives = self.problem.num_objectives();
        let mut ideal = vec![f64::NEG_INFINITY; num_objectives];
        let mut nadir = vec![f64::INFINITY; num_objectives];

        // Calculate payoff table by optimizing each objective
        let payoff_table = self.calculate_payoff_table(timer)?;

        // Extract ideal from payoff table
        // Ideal point: best value each objective can achieve (diagonal elements = minimization result)
        for i in 0..num_objectives {
            ideal[i] = payoff_table[i][i];
        }

        // Nadir point: worst value for each objective by MAXIMIZING it
        // This matches Python's behavior: model.setObjective(objectives_exprs[i], gp.GRB.MAXIMIZE)
        log::debug!("Computing nadir points by maximizing each objective");
        for (i, nadir_value) in nadir.iter_mut().enumerate().take(num_objectives) {
            let timeout = timer.map(Timer::remaining);
            let solution = SingleObjectiveSolver::new(self.problem, self.options)
                .solve_objective_maximized(i, timeout)?;

            if !solution.feasible {
                return Err(AugmeconError::OptimizationError(format!(
                    "Nadir computation for objective {i} failed - infeasible"
                )));
            }

            *nadir_value = solution.objective_values[i];
            log::debug!("Nadir value for objective {i}: {nadir_value}");
        }

        log::debug!("Calculated ideal point: {ideal:?}");
        log::debug!("Calculated nadir point: {nadir:?}");

        Ok((ideal, nadir))
    }

    /// Calculate full payoff table using standard algorithm
    ///
    /// This implements the two-phase algorithm:
    /// 1. Optimize each objective independently (diagonal elements)
    /// 2. For each objective i, fix it at its optimal value and optimize all other objectives
    ///
    /// The `timer` reference is used to get the remaining timeout for each solver invocation.
    ///
    /// # Errors
    /// Returns error if any single-objective optimization fails or problem is infeasible
    pub fn calculate_payoff_table(&self, timer: Option<&Timer>) -> Result<Vec<Vec<f64>>> {
        let num_objectives = self.problem.num_objectives();
        let mut payoff_table = vec![vec![0.0; num_objectives]; num_objectives];

        println!("DEBUG: Starting payoff table calculation for {num_objectives} objectives");
        log::debug!("Starting payoff table calculation for {num_objectives} objectives");

        // Phase 1: Calculate diagonal elements (independent optimization)
        println!("DEBUG: Phase 1: Calculating diagonal elements");
        log::debug!("Phase 1: Calculating diagonal elements");
        for (i, row) in payoff_table.iter_mut().enumerate() {
            println!("DEBUG: Solving single objective {i}");
            let timeout = timer.map(Timer::remaining);
            let solution = SingleObjectiveSolver::new(self.problem, self.options)
                .solve_objective(i, timeout)?;

            if !solution.feasible {
                println!("DEBUG: Single objective {i} failed - infeasible");
                return Err(AugmeconError::OptimizationError(format!(
                    "Single objective optimization {i} failed - infeasible"
                )));
            }

            // Store the diagonal element (optimal value for objective i)
            row[i] = solution.objective_values[i];
            println!("DEBUG: Diagonal element [{}][{}] = {}", i, i, row[i]);
            log::debug!("Diagonal element [{i}][{i}] = {}", row[i]);
        }

        // Phase 2: Calculate off-diagonal elements (constrained optimization)
        log::debug!("Phase 2: Calculating off-diagonal elements");
        for (i, row) in payoff_table.iter_mut().enumerate() {
            // Create constraint: objective i == its optimal value
            let (obj_i_expr, _) = &self.problem.objectives[i];
            let optimal_value_i = row[i];

            // Create the constraint using the constraint! macro
            let constraint_i = constraint!(obj_i_expr.clone() == optimal_value_i);
            let additional_constraints = vec![constraint_i];

            log::debug!("Row {i}: Adding constraint obj_{i} == {optimal_value_i}");

            // For each other objective j, optimize it under the constraint
            #[expect(
                clippy::needless_range_loop,
                reason = "Index j is needed to identify which objective is being optimized and to maintain correspondence with payoff table structure"
            )]
            for j in 0..num_objectives {
                if i != j {
                    log::debug!("Optimizing objective {j} with constraint on objective {i}");

                    let timeout = timer.map(Timer::remaining);
                    let solution = SingleObjectiveSolver::new(self.problem, self.options)
                        .solve_objective_with_constraints(j, &additional_constraints, timeout)?;

                    if !solution.feasible {
                        return Err(AugmeconError::OptimizationError(format!(
                            "Constrained optimization failed for obj {j} with constraint on obj {i}"
                        )));
                    }

                    let obj_value = solution.objective_values[j];
                    row[j] = obj_value;
                    log::debug!("Payoff[{i}][{j}] = {obj_value}");
                }
            }
        }

        log::debug!("Final payoff table: {payoff_table:?}");
        Ok(payoff_table)
    }
}
