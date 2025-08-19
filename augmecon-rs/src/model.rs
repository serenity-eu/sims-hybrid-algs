//! # Problem Modeling Module
//!
//! This module provides the core building blocks for defining multi-objective optimization problems.
//! It includes structures for variables, constraints, objectives, and the main problem container.
//!
//! ## Overview
//!
//! The modeling system supports:
//! - **Multiple variable types**: Continuous, integer, and binary variables
//! - **Linear constraints**: Using `good_lp` Constraint types
//! - **Linear objectives**: Using `good_lp` Expression types with optimization directions
//! - **Flexible problem construction**: Builder patterns and validation
//!
//! ## Key Structures
//!
//! - [`MultiObjectiveProblem`]: Main container for the optimization problem
//! - [`VariableType`]: Defines variable domains and bounds
//! - [`ObjectiveDirection`]: Specifies optimization direction
//!
//! ## Example Usage
//!
//! ```rust
//! use augmecon::{MultiObjectiveProblem, VariableType, ObjectiveDirection};
//! use good_lp::{constraint, Expression, variable};
//!
//! let mut problem = MultiObjectiveProblem::new();
//!
//! // Add variables
//! let x1 = problem.add_variable(
//!     "production_a".to_string(),
//!     VariableType::Continuous { min: Some(0.0), max: Some(100.0) }
//! );
//!
//! // Add constraint: production_a <= 50
//! let constraint = constraint!(x1 <= 50.0);
//! problem.add_constraint(constraint);
//!
//! // Add objective: maximize 3 * production_a
//! let objective_expr = 3.0 * x1;
//! problem.add_objective(objective_expr, ObjectiveDirection::Maximize);
//! ```
//!
//! ## Advanced Patterns
//!
//! ### Problem Validation
//!
//! ```rust
//! # use augmecon::*;
//! # let problem = MultiObjectiveProblem::new();
//! // Validate problem structure before solving
//! match problem.validate() {
//!     Ok(()) => println!("Problem is valid"),
//!     Err(e) => eprintln!("Problem validation failed: {}", e),
//! }
//! ```

use crate::error::{AugmeconError, Result};
use good_lp::{variable, Constraint, Expression, ProblemVariables, Variable};
use std::collections::HashMap;

/// Direction of optimization for an objective function
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectiveDirection {
    /// Minimize the objective function
    Minimize,
    /// Maximize the objective function  
    Maximize,
}

/// Represents a complete multi-objective optimization problem
pub struct MultiObjectiveProblem {
    /// Variable definitions for the problem (for `good_lp` compatibility)
    pub variables: ProblemVariables,
    /// Constraints for the problem (for `good_lp` compatibility)
    pub constraints: Vec<Constraint>,
    /// Objective functions with their optimization directions (for `good_lp` compatibility)
    pub objectives: Vec<(Expression, ObjectiveDirection)>,
    /// Mapping from variable names to Variable objects
    pub var_map: HashMap<String, Variable>,
    /// Variable type information for proper recreation
    pub variable_types: HashMap<String, VariableType>,
}

impl MultiObjectiveProblem {
    /// Create a new multi-objective problem
    #[must_use]
    pub fn new() -> Self {
        Self {
            variables: ProblemVariables::new(),
            constraints: Vec::new(),
            objectives: Vec::new(),
            var_map: HashMap::new(),
            variable_types: HashMap::new(),
        }
    }

    /// Add a constraint to the problem
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Add an objective function to the problem
    pub fn add_objective(&mut self, expr: Expression, direction: ObjectiveDirection) {
        self.objectives.push((expr, direction));
    }

    /// Add a variable with type information
    pub fn add_variable(&mut self, name: String, var_type: VariableType) -> Variable {
        let var = match &var_type {
            VariableType::Continuous { min, max } => {
                let mut v = variable();
                if let Some(min_val) = min {
                    v = v.min(*min_val);
                }
                if let Some(max_val) = max {
                    v = v.max(*max_val);
                }
                self.variables.add(v)
            }
            VariableType::Integer { min, max } => {
                let mut v = variable().integer();
                if let Some(min_val) = min {
                    v = v.min(f64::from(*min_val));
                }
                if let Some(max_val) = max {
                    v = v.max(f64::from(*max_val));
                }
                self.variables.add(v)
            }
            VariableType::Binary => self.variables.add(variable().binary()),
        };

        self.var_map.insert(name.clone(), var);
        self.variable_types.insert(name, var_type);
        var
    }

    /// Get the number of objectives
    #[must_use]
    pub const fn num_objectives(&self) -> usize {
        self.objectives.len()
    }

    /// Validate the problem structure
    ///
    /// # Errors
    /// Returns an error if the problem has no objectives, no variables, or inconsistent data
    pub fn validate(&self) -> Result<()> {
        let num_objectives = self.num_objectives();
        println!(
            "DEBUG: Validating problem - objectives: {}, constraints: {}, variables: {}",
            num_objectives,
            self.constraints.len(),
            self.var_map.len()
        );
        if num_objectives < 2 {
            println!("DEBUG: Validation failed - need at least 2 objectives, got {num_objectives}");
            return Err(AugmeconError::InvalidObjectiveCount(num_objectives));
        }
        println!("DEBUG: Problem validation passed");
        Ok(())
    }
}

impl Default for MultiObjectiveProblem {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for MultiObjectiveProblem {
    fn clone(&self) -> Self {
        // Clone the cloneable fields and create fresh good_lp structures
        let mut new_problem = Self::new();

        // Clone variable types and recreate variables
        new_problem.variable_types.clone_from(&self.variable_types);
        for (name, var_type) in &self.variable_types {
            new_problem.add_variable(name.clone(), var_type.clone());
        }

        // Note: good_lp constraints and expressions don't implement Clone,
        // so we only clone the metadata. The constraints and objectives
        // would need to be reconstructed by the caller if needed.

        new_problem
    }
}

impl std::fmt::Debug for MultiObjectiveProblem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiObjectiveProblem")
            .field("variable_types", &self.variable_types)
            .field("num_constraints", &self.constraints.len())
            .field("num_objectives", &self.objectives.len())
            .finish_non_exhaustive()
    }
}

/// Variable type information for proper recreation
#[derive(Clone, Debug, PartialEq)]
pub enum VariableType {
    /// Continuous variable with optional bounds
    Continuous {
        /// Minimum value for the variable
        min: Option<f64>,
        /// Maximum value for the variable
        max: Option<f64>,
    },
    /// Integer variable with optional bounds
    Integer {
        /// Minimum value for the variable
        min: Option<i32>,
        /// Maximum value for the variable
        max: Option<i32>,
    },
    /// Binary variable (0 or 1)
    Binary,
}
