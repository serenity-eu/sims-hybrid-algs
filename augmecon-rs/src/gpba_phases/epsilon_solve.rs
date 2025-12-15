//! Phase: Epsilon-Constraint Solve
//!
//! This module implements the epsilon-constraint solving phase of GPBA-A algorithm.
//! It takes the current epsilon configuration and solves the epsilon-constraint MILP,
//! returning the solution (if feasible) and solve statistics.
//!
//! This is the core solving step that invokes the MILP solver.

use crate::epsilon_constraint::EpsilonConstraintBuilder;
use crate::model::MultiObjectiveProblem;
use crate::options::Options;
use crate::solution::Solution;
use std::collections::HashMap;
use std::time::Duration;

/// Result of epsilon-constraint solving phase.
#[derive(Debug, Clone)]
pub struct EpsilonSolveResult {
    /// Whether the problem was feasible.
    pub is_feasible: bool,

    /// Objective values in maximization form (negated from original).
    /// Empty if infeasible.
    pub objectives_max: Vec<f64>,

    /// Solution variable assignments.
    /// Empty if infeasible.
    pub solution: Solution,

    /// Time spent solving in seconds.
    pub solve_time_secs: f64,
}

/// Solve one epsilon-constraint MILP configuration.
///
/// This function:
/// - Builds epsilon-constraint problem with slack variables
/// - Solves the MILP
/// - Extracts solution and objective values if feasible
/// - Returns solve result and timing
///
/// # Arguments
///
/// * `problem` - Multi-objective problem to solve
/// * `epsilons` - Epsilon values for constraint objectives (in minimization form, negated)
/// * `ranges` - Objective ranges for slack variable scaling
/// * `primary_objective` - Index of the main objective to optimize
/// * `timeout` - Optional timeout for solver
///
/// # Returns
///
/// `EpsilonSolveResult` containing feasibility, objectives, solution, and timing.
///
/// # Example
///
/// ```ignore
/// use std::collections::HashMap;
/// use std::time::Duration;
///
/// let mut epsilons = HashMap::new();
/// epsilons.insert(1, -15.0);  // Constraint on objective 1
///
/// let mut ranges = HashMap::new();
/// ranges.insert(1, 13.0);  // Range for objective 1
///
/// let result = solve_epsilon_constraint_config(
///     &problem,
///     &epsilons,
///     &ranges,
///     0,  // Main objective index
///     Some(Duration::from_secs(60)),
/// ).unwrap();
///
/// if result.is_feasible {
///     println!("Found solution: {:?}", result.objectives_max);
/// }
/// ```
#[allow(clippy::implicit_hasher, reason = "Public API uses standard HashMap for simplicity - generics would complicate the interface")]
/// Solve an epsilon-constraint problem with the given configuration.
///
/// # Errors
///
/// Returns an error if the solver fails, the problem is invalid, or timeout is reached.
pub fn solve_epsilon_constraint_config(
    problem: &MultiObjectiveProblem,
    epsilons: &HashMap<usize, f64>,
    ranges: &HashMap<usize, f64>,
    primary_objective: usize,
    timeout: Option<Duration>,
) -> Result<EpsilonSolveResult, Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();

    // Build epsilon-constraint problem with slack variables
    let default_options = Options::default();
    let mut builder = EpsilonConstraintBuilder::new(problem, &default_options, primary_objective);

    for (&k, &epsilon) in epsilons {
        let range = ranges.get(&k).copied().unwrap_or(1000.0);
        builder = builder.add_constraint_with_range(k, epsilon, range);
    }

    // Solve with slack variables
    let solve_result = builder.solve_with_slack(timeout)?;

    let solve_time_secs = start.elapsed().as_secs_f64();

    match solve_result {
        Some(solution_with_slack) => {
            let solution = solution_with_slack.solution;

            // Extract objective values from solution
            // These are already in maximization form (negated)
            let objectives_max = solution.objective_values.clone();

            Ok(EpsilonSolveResult {
                is_feasible: true,
                objectives_max,
                solution,
                solve_time_secs,
            })
        }
        None => {
            // Infeasible
            Ok(EpsilonSolveResult {
                is_feasible: false,
                objectives_max: vec![],
                solution: Solution::infeasible(problem.num_objectives()),
                solve_time_secs,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::MultiObjectiveProblem;
    use crate::ObjectiveDirection;
    use good_lp::*;
    use std::collections::HashMap;

    fn create_simple_2obj_problem() -> MultiObjectiveProblem {
        // Simple 2-objective problem:
        // x1, x2 in [0,1]
        // obj1: minimize x1 + x2 (or maximize -(x1+x2))
        // obj2: minimize 2*x1 + x2 (or maximize -(2*x1+x2))
        // Constraint: x1 + x2 >= 1

        variables!(
            problem:
               0 <= x1 <= 1;
               0 <= x2 <= 1;
        );

        let var_map = HashMap::from([("x1".to_string(), x1), ("x2".to_string(), x2)]);

        let constraints = vec![constraint!(x1 + x2 >= 1.0)];

        let objectives = vec![
            (x1 + x2, ObjectiveDirection::Minimize),
            ((2.0 * x1) + x2, ObjectiveDirection::Minimize),
        ];

        MultiObjectiveProblem {
            variables: problem,
            constraints,
            objectives,
            var_map,
            variable_types: HashMap::new(),
        }
    }

    #[test]
    fn test_solve_feasible_configuration() {
        let problem = create_simple_2obj_problem();

        // epsilon configuration: obj2 >= -2 (in minimization: obj2 <= 2)
        // This allows solutions like x1=1,x2=0 (obj1=1, obj2=2)
        let mut epsilons = HashMap::new();
        epsilons.insert(1, -2.0); // Constraint: obj2_max >= -2

        let mut ranges = HashMap::new();
        ranges.insert(1, 2.0);

        let result = solve_epsilon_constraint_config(
            &problem,
            &epsilons,
            &ranges,
            0, // Primary objective
            Some(Duration::from_secs(10)),
        )
        .unwrap();

        assert!(result.is_feasible);
        assert_eq!(result.objectives_max.len(), 2);

        // Objectives should be negative (maximization form)
        assert!(result.objectives_max[0] <= 0.0);
        assert!(result.objectives_max[1] <= 0.0);

        // Solution should have feasible flag
        assert!(result.solution.feasible);

        // Solution should have decision variables
        assert!(!result.solution.decision_variables.is_empty());

        // Solve time should be reasonable
        assert!(result.solve_time_secs >= 0.0);
        assert!(result.solve_time_secs < 10.0);
    }

    #[test]
    fn test_solve_with_tight_epsilon() {
        let problem = create_simple_2obj_problem();

        // Tight but feasible: obj2 >= -2 (allows x1=1,x2=0)
        let mut epsilons = HashMap::new();
        epsilons.insert(1, -2.0);

        let mut ranges = HashMap::new();
        ranges.insert(1, 3.0);

        let result = solve_epsilon_constraint_config(
            &problem,
            &epsilons,
            &ranges,
            0,
            Some(Duration::from_secs(10)),
        )
        .unwrap();

        assert!(result.is_feasible);

        // With obj2 constrained to >= -2, and minimizing obj1,
        // the best solution is x1=1,x2=0: obj1=-1, obj2=-2
        // or x1=0,x2=1: obj1=-1, obj2=-1 (better, violates constraint)
        // So we expect x1=1,x2=0
        assert!(result.objectives_max[1] >= -2.0 - 1e-6);
    }

    #[test]
    fn test_solve_returns_timing() {
        let problem = create_simple_2obj_problem();

        let mut epsilons = HashMap::new();
        epsilons.insert(1, -2.0);

        let mut ranges = HashMap::new();
        ranges.insert(1, 3.0);

        let result = solve_epsilon_constraint_config(
            &problem,
            &epsilons,
            &ranges,
            0,
            Some(Duration::from_secs(10)),
        )
        .unwrap();

        // Verify timing information is collected
        assert!(result.solve_time_secs >= 0.0);
        assert!(result.solve_time_secs < 10.0);
    }
}
