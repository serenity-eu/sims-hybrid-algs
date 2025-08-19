//! Test script to verify solver selection functionality
//!
//! This test creates a simple multi-objective problem and solves it with different solvers
//! to ensure the solver selection from the Solver enum works correctly.

use augmecon::{
    constraint, variable, MultiObjectiveProblem, ObjectiveDirection, Options, Solver, VariableType,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("Testing solver selection functionality...");

    // Create a simple 2-objective problem
    let mut problem = MultiObjectiveProblem::new();

    // Add variables: x1, x2 both continuous [0, 10]
    problem.add_variable(
        "x1",
        VariableType::Continuous {
            lower: 0.0,
            upper: 10.0,
        },
    );
    problem.add_variable(
        "x2",
        VariableType::Continuous {
            lower: 0.0,
            upper: 10.0,
        },
    );

    // Objective 1: maximize x1 + x2
    let x1 = variable!("x1");
    let x2 = variable!("x2");
    problem.add_objective(x1.clone() + x2.clone(), ObjectiveDirection::Maximize);

    // Objective 2: maximize 2*x1 - x2
    problem.add_objective(2.0 * x1.clone() - x2.clone(), ObjectiveDirection::Maximize);

    // Add constraint: x1 + x2 <= 8
    problem.add_constraint(constraint!(x1 + x2 <= 8.0));

    println!("Created test problem with 2 objectives and 2 variables");

    // Test different solvers
    let solvers_to_test = vec![
        (Solver::Default, "Default"),
        (Solver::CoinCbc, "COIN-OR CBC"),
        (Solver::HiGHS, "HiGHS"),
        (Solver::MicroLP, "MicroLP"),
        (Solver::LPSolve, "LPSolve"),
    ];

    for (solver, solver_name) in solvers_to_test {
        println!("\n--- Testing solver: {} ---", solver_name);

        // Create options with the specific solver
        let options = Options::new()
            .with_grid_points(5)
            .with_solver(solver)
            .with_solver_option("ratioGap", "0.01")
            .with_solver_option("seconds", "10");

        println!(
            "Solver supports parameters: {}",
            solver.supports_parameters()
        );
        println!("Solver supports timeout: {}", solver.supports_timeout());

        // Test single objective solving (this uses our updated SingleObjectiveSolver)
        println!("Testing single objective solving...");

        match test_single_objective(&problem, &options) {
            Ok(_) => println!("✓ Single objective solving successful with {}", solver_name),
            Err(e) => println!(
                "✗ Single objective solving failed with {}: {}",
                solver_name, e
            ),
        }
    }

    println!("\n=== Solver selection test completed ===");
    Ok(())
}

fn test_single_objective(
    problem: &MultiObjectiveProblem,
    options: &Options,
) -> Result<(), Box<dyn std::error::Error>> {
    use augmecon::single_objective::SingleObjectiveSolver;

    let solver = SingleObjectiveSolver::new(problem, options);

    // Try solving the first objective
    let solution = solver.solve_objective(0, None)?;

    println!("Solution found:");
    println!("  Feasible: {}", solution.feasible);
    if solution.feasible {
        println!("  Objective values: {:?}", solution.objective_values);
        for (var, value) in &solution.variable_values {
            println!("  {}: {:.4}", var, value);
        }
    }

    Ok(())
}
