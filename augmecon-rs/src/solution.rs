//! # Solution Representation Module
//!
//! This module provides structures and traits for representing optimization solutions,
//! Pareto fronts, and performing multi-objective solution analysis.
//!
//! ## Overview
//!
//! The solution system includes:
//! - **Solution Trait**: Common interfaces for multi-objective solution operations
//! - **Pareto Solutions**: Individual solutions with objective values and variable assignments
//! - **Pareto Fronts**: Collections of non-dominated solutions with analysis capabilities
//! - **Dominance Analysis**: Methods for comparing and ranking solutions
//!
//! ## Key Components
//!
//! - [`MoSolution`]: Trait for multi-objective solution operations
//! - [`Solution`]: Individual solution in the Pareto front
//! - [`ParetoFront`]: Collection of Pareto-optimal solutions
//! - [`Solution`]: Basic solution structure with variable values
//!
//! ## Solution Analysis
//!
//! ### Dominance Relationships
//!
//! ```rust
//! use augmecon::{Solution, MoSolution};
//! use std::collections::HashMap;
//!
//! // Create two solutions for comparison
//! let solution1 = Solution {
//!     objectives: vec![100.0, 50.0],  // Higher profit, higher cost
//!     variables: HashMap::new(),
//!     is_dominated: false,
//! };
//!
//! let solution2 = Solution {
//!     objectives: vec![80.0, 30.0],   // Lower profit, lower cost
//!     variables: HashMap::new(),
//!     is_dominated: false,
//! };
//!
//! // Define optimization directions (true = maximize, false = minimize)
//! let directions = vec![true, false];  // Maximize profit, minimize cost
//!
//! // Check dominance relationships
//! if solution1.dominates(&solution2, &directions) {
//!     println!("Solution 1 dominates solution 2");
//! } else if solution2.dominates(&solution1, &directions) {
//!     println!("Solution 2 dominates solution 1");
//! } else {
//!     println!("Solutions are non-dominated (both Pareto-optimal)");
//! }
//! ```

use std::collections::HashMap;

/// Trait for types that have objective values
pub trait HasObjectives {
    /// Get the objective values for this solution
    fn objectives(&self) -> &[f64];
}

/// Trait for multi-objective solution operations
pub trait MoSolution: HasObjectives {
    /// Check if this solution dominates another solution
    /// A solution dominates another if it's at least as good in all objectives
    /// and strictly better in at least one objective
    /// `is_maximizing`: slice indicating whether each objective should be maximized (true) or minimized (false)
    fn dominates(&self, other: &Self, is_maximizing: &[bool]) -> bool {
        let self_objectives = self.objectives();
        let other_objectives = other.objectives();

        if self_objectives.len() != other_objectives.len()
            || self_objectives.len() != is_maximizing.len()
        {
            return false;
        }

        let mut at_least_as_good = true;
        let mut strictly_better_in_one = false;

        for ((self_val, other_val), &is_max) in self_objectives
            .iter()
            .zip(other_objectives.iter())
            .zip(is_maximizing.iter())
        {
            if self_val.is_nan() || other_val.is_nan() {
                return false; // Can't compare with NaN values
            }

            if is_max {
                // For maximization: self dominates other if self >= other in all and self > other in at least one
                if self_val < other_val {
                    at_least_as_good = false;
                    break;
                } else if self_val > other_val {
                    strictly_better_in_one = true;
                }
            } else {
                // For minimization: self dominates other if self <= other in all and self < other in at least one
                if self_val > other_val {
                    at_least_as_good = false;
                    break;
                } else if self_val < other_val {
                    strictly_better_in_one = true;
                }
            }
        }

        at_least_as_good && strictly_better_in_one
    }

    /// Check if this solution is dominated by another solution
    fn is_dominated_by(&self, other: &Self, is_maximizing: &[bool]) -> bool {
        other.dominates(self, is_maximizing)
    }

    /// Check if this solution covers another solution (dominates or equals)
    /// A solution covers another if it's at least as good in all objectives
    fn covers(&self, other: &Self, is_maximizing: &[bool]) -> bool {
        let self_objectives = self.objectives();
        let other_objectives = other.objectives();

        if self_objectives.len() != other_objectives.len()
            || self_objectives.len() != is_maximizing.len()
        {
            return false;
        }

        for ((self_val, other_val), &is_max) in self_objectives
            .iter()
            .zip(other_objectives.iter())
            .zip(is_maximizing.iter())
        {
            if self_val.is_nan() || other_val.is_nan() {
                return false; // Can't compare with NaN values
            }

            if is_max {
                // For maximization: self covers other if self >= other
                if self_val < other_val {
                    return false;
                }
            } else {
                // For minimization: self covers other if self <= other
                if self_val > other_val {
                    return false;
                }
            }
        }

        true
    }

    /// Check if this solution is covered by another solution
    fn is_covered_by(&self, other: &Self, is_maximizing: &[bool]) -> bool {
        other.covers(self, is_maximizing)
    }

    /// Convenience methods assuming minimization for all objectives
    /// Check if this solution dominates another solution (assuming minimization)
    fn dominates_min(&self, other: &Self) -> bool {
        let objectives = self.objectives();
        let is_minimizing = vec![false; objectives.len()];
        self.dominates(other, &is_minimizing)
    }

    /// Check if this solution is dominated by another solution (assuming minimization)
    fn is_dominated_by_min(&self, other: &Self) -> bool {
        other.dominates_min(self)
    }

    /// Convenience methods assuming maximization for all objectives
    /// Check if this solution dominates another solution (assuming maximization)
    fn dominates_max(&self, other: &Self) -> bool {
        let objectives = self.objectives();
        let is_maximizing = vec![true; objectives.len()];
        self.dominates(other, &is_maximizing)
    }

    /// Check if this solution is dominated by another solution (assuming maximization)
    fn is_dominated_by_max(&self, other: &Self) -> bool {
        other.dominates_max(self)
    }
}

/// Represents a single solution to the optimization problem
#[derive(Debug, Clone)]
pub struct Solution {
    /// Values of the objective functions for this solution
    pub objective_values: Vec<f64>,
    /// Values of the decision variables for this solution
    pub decision_variables: HashMap<String, f64>,
    /// Whether this solution is feasible
    pub feasible: bool,
    /// Additional metadata about the solution
    pub metadata: HashMap<String, String>,
}

impl Solution {
    /// Create a new solution
    #[must_use]
    pub fn new(objective_values: Vec<f64>, decision_variables: HashMap<String, f64>) -> Self {
        Self {
            objective_values,
            decision_variables,
            feasible: true,
            metadata: HashMap::new(),
        }
    }

    /// Create an infeasible solution
    #[must_use]
    pub fn infeasible(num_objectives: usize) -> Self {
        Self {
            objective_values: vec![f64::NAN; num_objectives],
            decision_variables: HashMap::new(),
            feasible: false,
            metadata: HashMap::new(),
        }
    }

    /// Get the value of a specific decision variable
    #[must_use]
    pub fn get_variable(&self, name: &str) -> Option<f64> {
        self.decision_variables.get(name).copied()
    }

    /// Get the value of a specific objective
    #[must_use]
    pub fn get_objective(&self, index: usize) -> Option<f64> {
        self.objective_values.get(index).copied()
    }

    /// Add metadata to the solution
    #[must_use]
    pub fn with_metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Collection of solutions representing the Pareto front
#[derive(Debug, Clone)]
pub struct ParetoFront {
    /// All Pareto-optimal solutions
    pub solutions: Vec<Solution>,
    /// The payoff table showing min/max values for each objective
    pub payoff_table: Vec<Vec<f64>>,
    /// Objective directions for proper dominance checking
    pub objective_directions: Vec<crate::model::ObjectiveDirection>,
}

impl ParetoFront {
    /// Create a new empty Pareto front
    #[must_use]
    pub const fn new(objective_directions: Vec<crate::model::ObjectiveDirection>) -> Self {
        Self {
            solutions: Vec::new(),
            payoff_table: Vec::new(),
            objective_directions,
        }
    }

    /// Add a solution to the Pareto front
    /// Only performs duplicate checking, defers dominance filtering
    pub fn add_solution(&mut self, solution: Solution) {
        self.add_solution_with_precision(solution, 9);
    }

    /// Add a solution to the Pareto front with configurable precision
    /// Only performs duplicate checking - call `filter_dominated_solutions()` when done adding
    pub fn add_solution_with_precision(&mut self, solution: Solution, decimal_places: u32) {
        // Round objective values to match Python's approach
        let rounded_objectives: Vec<f64> = solution
            .objective_values
            .iter()
            .map(|&val| {
                let factor = f64::from(10u32.pow(decimal_places));
                (val * factor).round() / factor
            })
            .collect();

        // Check if we already have a solution with these rounded objective values
        // Use epsilon based on decimal places: 0.5 * 10^(-decimal_places)
        let epsilon = 0.5 / f64::from(10u32.pow(decimal_places));
        let is_duplicate = self.solutions.iter().any(|existing| {
            if existing.objective_values.len() != rounded_objectives.len() {
                return false;
            }

            existing
                .objective_values
                .iter()
                .zip(rounded_objectives.iter())
                .all(|(&existing_val, &new_val)| (existing_val - new_val).abs() < epsilon)
        });

        if is_duplicate {
            log::trace!("Skipping duplicate solution: {rounded_objectives:?}");
            return;
        }

        // Create solution with rounded objectives
        let rounded_solution = Solution {
            objective_values: rounded_objectives.clone(),
            decision_variables: solution.decision_variables,
            feasible: solution.feasible,
            metadata: solution.metadata,
        };

        // Just add the solution - filtering will be done at the end
        self.solutions.push(rounded_solution);
        log::trace!("Added solution to collection: {rounded_objectives:?}");
    }

    /// Filter out dominated solutions for mixed objective directions
    /// Call this once after adding all solutions for better performance
    /// A solution dominates another if it's at least as good in all objectives and strictly better in at least one
    pub fn filter_dominated_solutions(&mut self) {
        let original_count = self.solutions.len();
        if original_count <= 1 {
            return;
        }

        log::debug!("Filtering Pareto front with {original_count} solutions");

        let mut is_dominated = vec![false; original_count];

        for i in 0..original_count {
            if is_dominated[i] {
                continue; // Already marked as dominated
            }

            let solution_i = &self.solutions[i];

            for j in (i + 1)..original_count {
                if is_dominated[j] {
                    continue; // Already marked as dominated
                }

                let solution_j = &self.solutions[j];

                // Check dominance relationships between i and j
                let i_dominates_j = self.solution_dominates(solution_i, solution_j);
                let j_dominates_i = self.solution_dominates(solution_j, solution_i);

                if i_dominates_j {
                    is_dominated[j] = true;
                    log::trace!(
                        "Solution {:?} dominated by {:?}",
                        solution_j.objective_values,
                        solution_i.objective_values
                    );
                } else if j_dominates_i {
                    is_dominated[i] = true;
                    log::trace!(
                        "Solution {:?} dominated by {:?}",
                        solution_i.objective_values,
                        solution_j.objective_values
                    );
                    break; // i is dominated, no need to check further
                }
            }
        }

        // Keep only non-dominated solutions
        self.solutions = self
            .solutions
            .drain(..)
            .enumerate()
            .filter_map(|(i, sol)| if is_dominated[i] { None } else { Some(sol) })
            .collect();

        let new_count = self.solutions.len();
        log::debug!("Filtered Pareto front: {original_count} -> {new_count} solutions");
    }

    /// Helper method to check if `solution_a` dominates `solution_b`
    fn solution_dominates(&self, solution_a: &Solution, solution_b: &Solution) -> bool {
        let mut at_least_as_good = true;
        let mut strictly_better_in_one = false;

        for k in 0..solution_a.objective_values.len() {
            let obj_a = solution_a.objective_values[k];
            let obj_b = solution_b.objective_values[k];

            // Handle NaN values
            if obj_a.is_nan() || obj_b.is_nan() {
                return false;
            }

            // Get objective direction from the problem
            let direction = if k < self.objective_directions.len() {
                &self.objective_directions[k]
            } else {
                &crate::model::ObjectiveDirection::Minimize // Default fallback
            };

            match direction {
                crate::model::ObjectiveDirection::Minimize => {
                    if obj_a > obj_b {
                        // a is worse than b in this minimization objective
                        at_least_as_good = false;
                        break;
                    } else if obj_a < obj_b {
                        // a is strictly better than b in this minimization objective
                        strictly_better_in_one = true;
                    }
                }
                crate::model::ObjectiveDirection::Maximize => {
                    if obj_a < obj_b {
                        // a is worse than b in this maximization objective
                        at_least_as_good = false;
                        break;
                    } else if obj_a > obj_b {
                        // a is strictly better than b in this maximization objective
                        strictly_better_in_one = true;
                    }
                }
            }
        }

        at_least_as_good && strictly_better_in_one
    }

    /// Set the payoff table
    pub fn set_payoff_table(&mut self, payoff_table: Vec<Vec<f64>>) {
        self.payoff_table = payoff_table;
    }

    /// Get the number of solutions in the Pareto front
    #[must_use]
    pub const fn len(&self) -> usize {
        self.solutions.len()
    }

    /// Check if the Pareto front is empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    /// Get objective values for all solutions as a matrix
    #[must_use]
    pub fn objective_matrix(&self) -> Vec<Vec<f64>> {
        self.solutions
            .iter()
            .map(|sol| sol.objective_values.clone())
            .collect()
    }
}

impl Default for ParetoFront {
    fn default() -> Self {
        Self::new(Vec::new()) // Empty objective directions for default
    }
}

/// Implementation of `HasObjectives` for Solution
impl HasObjectives for Solution {
    fn objectives(&self) -> &[f64] {
        &self.objective_values
    }
}

/// Implementation of `MoSolution` for Solution
impl MoSolution for Solution {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn init_test_logging() {
        INIT.call_once(|| {
            env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
                .is_test(true)
                .try_init()
                .ok();
        });
    }

    #[test]
    fn test_has_objectives_trait() {
        init_test_logging();
        let solution = Solution::new(vec![1.0, 2.0, 3.0], HashMap::new());
        assert_eq!(solution.objectives(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dominates_minimization() {
        init_test_logging();
        let solution1 = Solution::new(vec![1.0, 2.0], HashMap::new());
        let solution2 = Solution::new(vec![2.0, 3.0], HashMap::new());
        let is_maximizing = vec![false, false]; // minimization

        // solution1 dominates solution2 in minimization (1 < 2 and 2 < 3)
        assert!(solution1.dominates(&solution2, &is_maximizing));
        assert!(!solution2.dominates(&solution1, &is_maximizing));
    }

    #[test]
    fn test_dominates_maximization() {
        init_test_logging();
        let solution1 = Solution::new(vec![2.0, 3.0], HashMap::new());
        let solution2 = Solution::new(vec![1.0, 2.0], HashMap::new());
        let is_maximizing = vec![true, true]; // maximization

        // solution1 dominates solution2 in maximization (2 > 1 and 3 > 2)
        assert!(solution1.dominates(&solution2, &is_maximizing));
        assert!(!solution2.dominates(&solution1, &is_maximizing));
    }

    #[test]
    fn test_no_dominance() {
        init_test_logging();
        let solution1 = Solution::new(vec![1.0, 3.0], HashMap::new());
        let solution2 = Solution::new(vec![2.0, 2.0], HashMap::new());
        let is_maximizing = vec![true, true];

        // Neither solution dominates the other (solution1 is worse in obj1 but better in obj2)
        assert!(!solution1.dominates(&solution2, &is_maximizing));
        assert!(!solution2.dominates(&solution1, &is_maximizing));
    }

    #[test]
    fn test_covers() {
        init_test_logging();
        let solution1 = Solution::new(vec![1.0, 2.0], HashMap::new());
        let solution2 = Solution::new(vec![1.0, 3.0], HashMap::new());
        let is_maximizing = vec![false, false]; // minimization

        // solution1 covers solution2 (1 <= 1 and 2 <= 3)
        assert!(solution1.covers(&solution2, &is_maximizing));
        assert!(!solution2.covers(&solution1, &is_maximizing));
    }

    #[test]
    fn test_convenience_methods() {
        init_test_logging();
        let solution1 = Solution::new(vec![1.0, 2.0], HashMap::new());
        let solution2 = Solution::new(vec![2.0, 3.0], HashMap::new());

        // Test minimization convenience methods
        assert!(solution1.dominates_min(&solution2));
        assert!(solution2.is_dominated_by_min(&solution1));

        // Test maximization convenience methods
        assert!(solution2.dominates_max(&solution1));
        assert!(solution1.is_dominated_by_max(&solution2));
    }

    #[test]
    fn test_nan_handling() {
        init_test_logging();
        let solution1 = Solution::new(vec![1.0, f64::NAN], HashMap::new());
        let solution2 = Solution::new(vec![2.0, 3.0], HashMap::new());
        let is_maximizing = vec![true, true];

        // Should return false when NaN values are present
        assert!(!solution1.dominates(&solution2, &is_maximizing));
        assert!(!solution2.dominates(&solution1, &is_maximizing));
    }
}
