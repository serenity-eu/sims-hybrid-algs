//! Phase: Relaxation Search
//!
//! This module implements the relaxation search optimization for GPBA-A algorithm.
//! Before solving a new epsilon-constraint configuration, check if a previously explored
//! configuration was less constrained (relaxed) and its solution satisfies current constraints.
//!
//! This avoids redundant MILP solves by reusing previous solutions when applicable.

use std::fmt;

/// Information about a previously solved configuration
#[derive(Debug, Clone)]
pub struct PreviousSolutionInfo {
    /// Epsilon-constraint RHS values (for constraint objectives only)
    pub ef_array: Vec<i64>,
    /// Solution found: either objective values or "infeasible"
    pub solution: SolutionResult,
}

/// Result from solving an epsilon-constraint configuration
#[derive(Debug, Clone)]
pub enum SolutionResult {
    /// Feasible solution with objective values (all objectives, not just constraint ones)
    Feasible(Vec<i64>),
    /// Configuration was infeasible
    Infeasible,
}

impl fmt::Display for SolutionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Feasible(vals) => write!(f, "{vals:?}"),
            Self::Infeasible => write!(f, "infeasible"),
        }
    }
}

/// Check if this constraint configuration was already explored with relaxation.
///
/// For maximization: `ef_array1` is less constrained (more relaxed) if all `ef_array1[i]` >= `ef_array2[i]`
///
/// # Arguments
///
/// * `ef_array` - Current epsilon-constraint RHS values (for constraint objectives only)
/// * `previous_solution_information` - List of previous {`ef_array`, solution} pairs
/// * `constraint_indices` - Indices of constraint objectives in the full objective list
///
/// # Returns
///
/// Tuple of (found: bool, solution: Option<SolutionResult>)
/// - If found=true and solution=Some(Feasible): Previous solution satisfies current constraints
/// - If found=true and solution=Some(Infeasible): Previous was infeasible, current is too
/// - If found=false: Must solve current configuration
#[must_use]
pub fn search_previous_solutions_relaxation(
    ef_array: &[i64],
    previous_solution_information: &[PreviousSolutionInfo],
    constraint_indices: &[usize],
) -> (bool, Option<SolutionResult>) {
    for prev_sol_info in previous_solution_information {
        let prev_ef_array = &prev_sol_info.ef_array;
        let prev_solution = &prev_sol_info.solution;

        // Check if previous ef_array is less constrained (all values >= current)
        let is_less_constrained = prev_ef_array
            .iter()
            .zip(ef_array.iter())
            .all(|(prev, curr)| prev >= curr);

        if is_less_constrained {
            // If previous solution is not infeasible, check if it satisfies current constraints
            match prev_solution {
                SolutionResult::Feasible(prev_sol_vals) => {
                    // Check if previous solution satisfies current (tighter) constraints
                    // For maximization: solution[constraint_idx] must be <= ef_array[i]
                    // Note: prev_sol_vals contains ALL objectives, so we need to use constraint_indices
                    let satisfies = ef_array.iter().enumerate().all(|(i, &ef_val)| {
                        let constraint_idx = constraint_indices[i];
                        prev_sol_vals[constraint_idx] <= ef_val
                    });

                    if satisfies {
                        return (true, Some(prev_solution.clone()));
                    }
                }
                SolutionResult::Infeasible => {
                    // Previous was infeasible with less constrained constraints,
                    // so current is also infeasible
                    return (true, Some(SolutionResult::Infeasible));
                }
            }
        }
    }

    (false, None)
}

/// Save solution information for future relaxation checks.
///
/// # Arguments
///
/// * `ef_array` - Current epsilon-constraint RHS values
/// * `solution` - Solution found (objective values) or Infeasible
/// * `previous_solution_information` - List to append to (modified in-place)
pub fn save_solution_information(
    ef_array: &[i64],
    solution: SolutionResult,
    previous_solution_information: &mut Vec<PreviousSolutionInfo>,
) {
    previous_solution_information.push(PreviousSolutionInfo {
        ef_array: ef_array.to_vec(),
        solution,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relaxation_search_no_match() {
        // Test when no previous solution matches
        let ef_array = vec![100, 200];
        let constraint_indices = vec![1, 2]; // objectives 1 and 2 are constrained
        let previous_solutions = vec![];

        let (found, solution) = search_previous_solutions_relaxation(
            &ef_array,
            &previous_solutions,
            &constraint_indices,
        );

        assert!(!found);
        assert!(solution.is_none());
    }

    #[test]
    fn test_relaxation_search_reuse_solution() {
        // Test when a less constrained solution can be reused
        let ef_array = vec![100, 200]; // Current: more constrained
        let constraint_indices = vec![1, 2];

        // Previous solution with less constrained ef_array (120 >= 100, 250 >= 200)
        // Solution values: [300, 80, 150] (obj0=300, obj1=80, obj2=150)
        // Check: obj1=80 <= 100 ✓, obj2=150 <= 200 ✓
        let previous_solutions = vec![PreviousSolutionInfo {
            ef_array: vec![120, 250], // Less constrained (higher values)
            solution: SolutionResult::Feasible(vec![300, 80, 150]),
        }];

        let (found, solution) = search_previous_solutions_relaxation(
            &ef_array,
            &previous_solutions,
            &constraint_indices,
        );

        assert!(found);
        match solution {
            Some(SolutionResult::Feasible(vals)) => {
                assert_eq!(vals, vec![300, 80, 150]);
            }
            _ => panic!("Expected feasible solution"),
        }
    }

    #[test]
    fn test_relaxation_search_solution_violates_constraints() {
        // Test when previous solution exists but doesn't satisfy current constraints
        let ef_array = vec![100, 200]; // Current: more constrained
        let constraint_indices = vec![1, 2];

        // Previous solution with less constrained ef_array BUT solution violates current
        // Solution values: [300, 110, 150] (obj1=110 > 100, violates!)
        let previous_solutions = vec![PreviousSolutionInfo {
            ef_array: vec![120, 250],                                // Less constrained
            solution: SolutionResult::Feasible(vec![300, 110, 150]), // obj1=110 > 100
        }];

        let (found, solution) = search_previous_solutions_relaxation(
            &ef_array,
            &previous_solutions,
            &constraint_indices,
        );

        // Should not find a reusable solution
        assert!(!found);
        assert!(solution.is_none());
    }

    #[test]
    fn test_relaxation_search_infeasible_propagation() {
        // Test when previous configuration was infeasible with less constraints
        let ef_array = vec![100, 200]; // Current: more constrained
        let constraint_indices = vec![1, 2];

        // Previous was infeasible with less constrained ef_array
        // If less constrained config was infeasible, current (more constrained) is also infeasible
        let previous_solutions = vec![PreviousSolutionInfo {
            ef_array: vec![120, 250], // Less constrained
            solution: SolutionResult::Infeasible,
        }];

        let (found, solution) = search_previous_solutions_relaxation(
            &ef_array,
            &previous_solutions,
            &constraint_indices,
        );

        assert!(found);
        match solution {
            Some(SolutionResult::Infeasible) => {}
            _ => panic!("Expected infeasible result"),
        }
    }

    #[test]
    fn test_save_solution_information() {
        // Test saving solution information
        let mut previous_solutions = vec![];

        let ef_array = vec![100, 200];
        let solution = SolutionResult::Feasible(vec![300, 80, 150]);

        save_solution_information(&ef_array, solution, &mut previous_solutions);

        assert_eq!(previous_solutions.len(), 1);
        assert_eq!(previous_solutions[0].ef_array, vec![100, 200]);
        match &previous_solutions[0].solution {
            SolutionResult::Feasible(vals) => {
                assert_eq!(vals, &vec![300, 80, 150]);
            }
            SolutionResult::Infeasible => panic!("Expected feasible solution"),
        }

        // Save another (infeasible)
        let ef_array2 = vec![50, 100];
        save_solution_information(
            &ef_array2,
            SolutionResult::Infeasible,
            &mut previous_solutions,
        );

        assert_eq!(previous_solutions.len(), 2);
        match &previous_solutions[1].solution {
            SolutionResult::Infeasible => {}
            SolutionResult::Feasible(_) => panic!("Expected infeasible solution"),
        }
    }
}
