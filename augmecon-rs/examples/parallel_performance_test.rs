#!/usr/bin/env rust-script
//! Performance test for parallel AUGMECON execution
//!
//! This example uses a larger grid size to demonstrate the benefits
//! of parallel execution with more substantial computational workload.

use augmecon::{Augmecon, MultiObjectiveProblem, ObjectiveDirection, Options, VariableType};
use good_lp::constraint;
use std::time::Instant;

#[allow(clippy::uninlined_format_args)]
fn main() -> augmecon::Result<()> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("=== AUGMECON Parallel Performance Test ===");

    let grid_points = 50; // Larger grid for better parallelization benefits
    println!(
        "Problem: 3 objectives, 8 variables, {} grid points",
        grid_points
    );
    println!(
        "Expected grid size: {}^2 = {} subproblems",
        grid_points,
        grid_points * grid_points
    );
    println!("Available CPU cores: {}", rayon::current_num_threads());

    // Test sequential execution
    println!("\n--- Sequential Execution ---");
    let problem_sequential = create_larger_knapsack_problem();
    let options_sequential = Options::new()
        .with_grid_points(grid_points)
        .with_flag_array(false)
        .with_parallel_execution(false);

    let start_time = Instant::now();
    let mut solver_sequential = Augmecon::try_new(problem_sequential, options_sequential)?;
    solver_sequential.solve()?;
    let solutions_sequential = solver_sequential.get_pareto_solutions();
    let sequential_duration = start_time.elapsed();

    println!("Sequential time: {:?}", sequential_duration);
    println!("Solutions found: {}", solutions_sequential.len());

    // Test parallel execution
    println!("\n--- Parallel Execution ---");
    let problem_parallel = create_larger_knapsack_problem();
    let options_parallel = Options::new()
        .with_grid_points(grid_points)
        .with_flag_array(false)
        .with_parallel_execution(true);

    let start_time = Instant::now();
    let mut solver_parallel = Augmecon::try_new(problem_parallel, options_parallel)?;
    solver_parallel.solve()?;
    let solutions_parallel = solver_parallel.get_pareto_solutions();
    let parallel_duration = start_time.elapsed();

    println!("Parallel time: {:?}", parallel_duration);
    println!("Solutions found: {}", solutions_parallel.len());

    // Performance analysis
    println!("\n--- Performance Analysis ---");
    println!(
        "Sequential: {:.2} seconds",
        sequential_duration.as_secs_f64()
    );
    println!("Parallel:   {:.2} seconds", parallel_duration.as_secs_f64());

    if sequential_duration > parallel_duration {
        let speedup = sequential_duration.as_secs_f64() / parallel_duration.as_secs_f64();
        println!("🚀 Speedup: {:.2}x faster with parallel execution", speedup);

        #[expect(
            clippy::cast_precision_loss,
            reason = "Calculating efficiency based on speedup and number of threads"
        )]
        let efficiency = speedup / rayon::current_num_threads() as f64;
        println!(
            "📊 Parallel efficiency: {:.1}% ({:.2}x speedup / {} cores)",
            efficiency * 100.0,
            speedup,
            rayon::current_num_threads()
        );
    } else {
        let slowdown = parallel_duration.as_secs_f64() / sequential_duration.as_secs_f64();
        println!(
            "⚠️  Slowdown: {:.2}x slower (parallelization overhead)",
            slowdown
        );
    }

    // Verify correctness
    assert_eq!(
        solutions_sequential.len(),
        solutions_parallel.len(),
        "Sequential and parallel execution should find the same number of solutions"
    );

    println!(
        "✅ Both approaches produced {} consistent solutions!",
        solutions_sequential.len()
    );

    // Show throughput
    let grid_size = grid_points * grid_points;
    #[expect(
        clippy::cast_precision_loss,
        reason = "Calculating throughput based on duration"
    )]
    let sequential_throughput = grid_size as f64 / sequential_duration.as_secs_f64();
    #[expect(
        clippy::cast_precision_loss,
        reason = "Calculating throughput based on duration"
    )]
    let parallel_throughput = grid_size as f64 / parallel_duration.as_secs_f64();

    println!("\n--- Throughput Analysis ---");
    println!(
        "Sequential: {:.1} subproblems/second",
        sequential_throughput
    );
    println!("Parallel:   {:.1} subproblems/second", parallel_throughput);

    Ok(())
}

fn create_larger_knapsack_problem() -> MultiObjectiveProblem {
    let mut problem = MultiObjectiveProblem::new();

    // Add more variables for a more substantial problem
    let var_names = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"];
    for name in &var_names {
        problem.add_variable((*name).to_string(), VariableType::Binary);
    }

    // Get variable references
    if let (
        Some(&x1),
        Some(&x2),
        Some(&x3),
        Some(&x4),
        Some(&x5),
        Some(&x6),
        Some(&x7),
        Some(&x8),
    ) = (
        problem.var_map.get("x1"),
        problem.var_map.get("x2"),
        problem.var_map.get("x3"),
        problem.var_map.get("x4"),
        problem.var_map.get("x5"),
        problem.var_map.get("x6"),
        problem.var_map.get("x7"),
        problem.var_map.get("x8"),
    ) {
        // Capacity constraint: total weight <= 25
        let capacity_constraint =
            5.0 * x1 + 4.0 * x2 + 6.0 * x3 + 8.0 * x4 + 10.0 * x5 + 3.0 * x6 + 7.0 * x7 + 9.0 * x8;
        problem.add_constraint(constraint!(capacity_constraint <= 25.0));

        // Additional constraint: at most 5 items
        let count_constraint = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8;
        problem.add_constraint(constraint!(count_constraint <= 5.0));

        // Objective 1: maximize value
        let value_objective = 10.0 * x1
            + 20.0 * x2
            + 30.0 * x3
            + 40.0 * x4
            + 50.0 * x5
            + 15.0 * x6
            + 25.0 * x7
            + 35.0 * x8;
        problem.add_objective(value_objective, ObjectiveDirection::Maximize);

        // Objective 2: minimize weight
        let weight_objective =
            5.0 * x1 + 4.0 * x2 + 6.0 * x3 + 8.0 * x4 + 10.0 * x5 + 3.0 * x6 + 7.0 * x7 + 9.0 * x8;
        problem.add_objective(weight_objective, ObjectiveDirection::Minimize);

        // Objective 3: minimize cost
        let cost_objective =
            1.0 * x1 + 2.0 * x2 + 3.0 * x3 + 4.0 * x4 + 5.0 * x5 + 1.5 * x6 + 2.5 * x7 + 3.5 * x8;
        problem.add_objective(cost_objective, ObjectiveDirection::Minimize);
    }

    problem
}
