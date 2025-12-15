//! Test example for `HiGHS` solver integration with the augmecon library.
//!
//! This example demonstrates how to use different solvers (Default, `CoinCbc`, `HiGHS`)
//! with the augmecon multi-objective optimization library.

use augmecon::{
    Augmecon, MultiObjectiveProblem, ObjectiveDirection, Options, Solver, VariableType,
};
use good_lp::constraint;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("Testing HiGHS solver integration...");

    // Test all three solvers
    let solvers = [Solver::Default, Solver::CoinCbc, Solver::HiGHS];

    for solver in &solvers {
        println!("\n--- Testing {} solver ---", solver.name());

        let problem = create_simple_problem();
        let options = Options::new()
            .with_name(format!("test_{}", solver.name().to_lowercase()))
            .with_solver(*solver)
            .with_grid_points(3);

        let mut augmecon_solver = Augmecon::try_new(problem, options)?;

        match augmecon_solver.solve() {
            Ok(solutions) => {
                println!(
                    "✅ {} solver: Found {} Pareto solutions",
                    solver.name(),
                    solutions.len()
                );

                // Print first few solutions
                for (i, solution) in solutions.iter().take(3).enumerate() {
                    println!(
                        "   Solution {}: Objectives = {:?}",
                        i + 1,
                        solution.objective_values
                    );
                }

                if solutions.len() > 3 {
                    println!("   ... and {} more solutions", solutions.len() - 3);
                }
            }
            Err(e) => {
                println!("❌ {} solver: Failed to solve: {}", solver.name(), e);
            }
        }
    }

    println!("\nHiGHS integration test completed!");
    Ok(())
}

fn create_simple_problem() -> MultiObjectiveProblem {
    let mut problem = MultiObjectiveProblem::new();

    // Add variables: x and y (continuous, non-negative)
    problem.add_variable(
        "x".to_string(),
        VariableType::Continuous {
            min: Some(0.0),
            max: Some(10.0),
        },
    );
    problem.add_variable(
        "y".to_string(),
        VariableType::Continuous {
            min: Some(0.0),
            max: Some(10.0),
        },
    );

    // Get variables and add constraints/objectives
    let x = *problem.var_map.get("x").unwrap();
    let y = *problem.var_map.get("y").unwrap();

    // Constraint: x + y <= 8
    problem.add_constraint(constraint!(x + y <= 8.0));

    // Objective 1: Maximize x + 2*y
    problem.add_objective(1.0 * x + 2.0 * y, ObjectiveDirection::Maximize);

    // Objective 2: Minimize x - y
    problem.add_objective(1.0 * x - 1.0 * y, ObjectiveDirection::Minimize);

    problem
}
