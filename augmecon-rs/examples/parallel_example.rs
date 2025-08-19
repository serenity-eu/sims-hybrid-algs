#!/usr/bin/env rust-script
//! Example demonstrating parallel execution of AUGMECON with Rayon
//!
//! This example shows how to:
//! 1. Create a multi-objective problem
//! 2. Configure AUGMECON with parallel execution enabled
//! 3. Compare performance between sequential and parallel execution
//! 4. Monitor thread usage and processing time

use augmecon::{Augmecon, MultiObjectiveProblem, ObjectiveDirection, Options, VariableType};
use good_lp::constraint;
use std::time::Instant;

fn main() -> augmecon::Result<()> {
    env_logger::init();

    println!("=== AUGMECON Parallel Execution Example ===");

    let grid_points = 20;
    println!("Problem: 3 objectives, 5 variables, {grid_points} grid points");
    println!("Available CPU cores: {}", rayon::current_num_threads());

    // Test sequential execution
    println!("\n--- Sequential Execution ---");
    let problem_sequential = create_knapsack_problem();
    let options_sequential = Options::new()
        .with_grid_points(grid_points)
        .with_flag_array(false)
        .with_parallel_execution(false); // Disable parallel execution

    let start_time = Instant::now();
    let mut solver_sequential = Augmecon::try_new(problem_sequential, options_sequential)?;
    solver_sequential.solve()?;
    let solutions_sequential = solver_sequential.get_pareto_solutions();
    let sequential_duration = start_time.elapsed();

    println!("Sequential time: {sequential_duration:?}");
    println!("Solutions found: {}", solutions_sequential.len());

    // Test parallel execution
    println!("\n--- Parallel Execution ---");
    let problem_parallel = create_knapsack_problem(); // Create a fresh problem
    let options_parallel = Options::new()
        .with_grid_points(grid_points)
        .with_flag_array(false)
        .with_parallel_execution(true); // Enable parallel execution

    let start_time = Instant::now();
    let mut solver_parallel = Augmecon::try_new(problem_parallel, options_parallel)?;
    solver_parallel.solve()?;
    let solutions_parallel = solver_parallel.get_pareto_solutions();
    let parallel_duration = start_time.elapsed();

    println!("Parallel time: {parallel_duration:?}");
    println!("Solutions found: {}", solutions_parallel.len());

    // Compare results
    println!("\n--- Performance Comparison ---");
    if sequential_duration > parallel_duration {
        let speedup = sequential_duration.as_secs_f64() / parallel_duration.as_secs_f64();
        println!("Speedup: {speedup:.2}x faster with parallel execution");
    } else {
        let slowdown = parallel_duration.as_secs_f64() / sequential_duration.as_secs_f64();
        println!("Slowdown: {slowdown:.2}x slower with parallel execution (overhead)");
    }

    // Verify that both approaches found the same number of solutions
    assert_eq!(
        solutions_sequential.len(),
        solutions_parallel.len(),
        "Sequential and parallel execution should find the same number of solutions"
    );

    println!("✅ Both sequential and parallel execution produced consistent results!");

    // Print a few example solutions
    println!("\n--- Example Pareto Solutions ---");
    for (i, solution) in solutions_parallel.iter().take(5).enumerate() {
        println!(
            "Solution {}: objectives = {:?}, feasible = {}",
            i + 1,
            solution.objective_values,
            solution.feasible
        );
    }

    if solutions_parallel.len() > 5 {
        println!("... and {} more solutions", solutions_parallel.len() - 5);
    }

    Ok(())
}

fn create_knapsack_problem() -> MultiObjectiveProblem {
    let mut problem = MultiObjectiveProblem::new();

    // Add variables: x1, x2, x3, x4, x5 (binary variables for items)
    problem.add_variable("x1".to_string(), VariableType::Binary);
    problem.add_variable("x2".to_string(), VariableType::Binary);
    problem.add_variable("x3".to_string(), VariableType::Binary);
    problem.add_variable("x4".to_string(), VariableType::Binary);
    problem.add_variable("x5".to_string(), VariableType::Binary);

    println!(
        "Variables added: {:?}",
        problem.var_map.keys().collect::<Vec<_>>()
    );

    // Get variable references for constraints and objectives
    if let (Some(&x1), Some(&x2), Some(&x3), Some(&x4), Some(&x5)) = (
        problem.var_map.get("x1"),
        problem.var_map.get("x2"),
        problem.var_map.get("x3"),
        problem.var_map.get("x4"),
        problem.var_map.get("x5"),
    ) {
        println!("All variables found, adding constraints and objectives...");

        // Add capacity constraint: sum of weights <= 15
        let capacity_constraint = 5.0 * x1 + 4.0 * x2 + 6.0 * x3 + 8.0 * x4 + 10.0 * x5;
        problem.add_constraint(constraint!(capacity_constraint <= 15.0));

        // Objective 1: maximize value
        let value_objective = 10.0 * x1 + 20.0 * x2 + 30.0 * x3 + 40.0 * x4 + 50.0 * x5;
        problem.add_objective(value_objective, ObjectiveDirection::Maximize);

        // Objective 2: minimize weight
        let weight_objective = 5.0 * x1 + 4.0 * x2 + 6.0 * x3 + 8.0 * x4 + 10.0 * x5;
        problem.add_objective(weight_objective, ObjectiveDirection::Minimize);

        // Objective 3: minimize cost
        let cost_objective = 1.0 * x1 + 2.0 * x2 + 3.0 * x3 + 4.0 * x4 + 5.0 * x5;
        problem.add_objective(cost_objective, ObjectiveDirection::Minimize);

        println!("Added {} objectives", problem.objectives.len());
    } else {
        println!("Failed to find all variables!");
    }

    problem
}
