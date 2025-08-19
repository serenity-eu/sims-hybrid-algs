//! Utility functions for problem setup and variable creation
//!
//! This module provides common helper functions used across different optimization
//! algorithms for creating variables and setting up optimization problems.

use crate::model::MultiObjectiveProblem;
use good_lp::{variable, variables};
use std::collections::HashMap;

/// Create variables for a multi-objective problem
///
/// Returns a tuple containing the problem variables and a mapping from variable names to variables.
/// This function handles different variable types (continuous, integer, binary) with their bounds.
#[must_use]
pub fn create_problem_variables(
    problem: &MultiObjectiveProblem,
) -> (
    good_lp::ProblemVariables,
    HashMap<String, good_lp::Variable>,
) {
    let mut prob_vars = variables!();
    let mut var_map = HashMap::new();

    // Add variables with their bounds and types
    for (name, var_type) in &problem.variable_types {
        let var_def = match var_type {
            crate::model::VariableType::Continuous { min, max } => {
                let mut v = variable();
                if let Some(min_val) = min {
                    v = v.min(*min_val);
                }
                if let Some(max_val) = max {
                    v = v.max(*max_val);
                }
                v
            }
            crate::model::VariableType::Integer { min, max } => {
                let mut v = variable().integer();
                if let Some(min_val) = min {
                    v = v.min(f64::from(*min_val));
                }
                if let Some(max_val) = max {
                    v = v.max(f64::from(*max_val));
                }
                v
            }
            crate::model::VariableType::Binary => variable().binary(),
        };

        let var = prob_vars.add(var_def);
        var_map.insert(name.clone(), var);
    }

    (prob_vars, var_map)
}
