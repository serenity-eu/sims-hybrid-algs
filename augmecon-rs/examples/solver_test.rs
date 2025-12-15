//! # Solver Selection Test
//!
//! This example demonstrates how to use different solvers with AUGMECON-RS.

use augmecon::{
    Augmecon, MultiObjectiveProblem, ObjectiveDirection, Options, Solver, VariableType,
};
use good_lp::constraint;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("🔧 AUGMECON-RS Solver Selection Test");
    println!("====================================");

    // Test with Default solver
    println!("\n📊 Testing with Default solver:");
    let problem1 = create_simple_problem();
    let options_default = Options::new()
        .with_name("test_default")
        .with_solver(Solver::Default)
        .with_grid_points(5);

    let mut solver_default = Augmecon::try_new(problem1, options_default)?;
    let solutions_default = solver_default.solve()?;
    println!(
        "✅ Default solver: Found {} solutions",
        solutions_default.len()
    );

    // Test with CoinCbc solver
    println!("\n📊 Testing with CoinCbc solver:");
    let problem2 = create_simple_problem();
    let options_cbc = Options::new()
        .with_name("test_cbc")
        .with_solver(Solver::CoinCbc)
        .with_grid_points(5);

    let mut solver_cbc = Augmecon::try_new(problem2, options_cbc)?;
    let solutions_cbc = solver_cbc.solve()?;
    println!("✅ CoinCbc solver: Found {} solutions", solutions_cbc.len());

    // Test with HiGHS solver
    println!("\n📊 Testing with HiGHS solver:");
    let problem3 = create_simple_problem();
    let options_highs = Options::new()
        .with_name("test_highs")
        .with_solver(Solver::HiGHS)
        .with_grid_points(5);

    let mut solver_highs = Augmecon::try_new(problem3, options_highs)?;
    let solutions_highs = solver_highs.solve()?;
    println!("✅ HiGHS solver: Found {} solutions", solutions_highs.len());

    println!("\n🎯 All solvers completed successfully!");
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

    // Constraint: x + y <= 10
    problem.add_constraint(constraint!(x + y <= 10.0));

    // Objective 1: Maximize x (profit)
    problem.add_objective(1.0 * x, ObjectiveDirection::Maximize);

    // Objective 2: Minimize y (cost)
    problem.add_objective(1.0 * y, ObjectiveDirection::Minimize);

    problem
}
