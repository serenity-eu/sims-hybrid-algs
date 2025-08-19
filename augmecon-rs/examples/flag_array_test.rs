//! Flag Array Performance Test
//!
//! This example demonstrates the performance benefits of the flag array optimization
//! by comparing solving times with and without flag arrays enabled.

use augmecon::{Augmecon, MultiObjectiveProblem, ObjectiveDirection, Options, VariableType};
use good_lp::{constraint, Expression};
use std::error::Error;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    println!("🚀 Flag Array Performance Test");
    println!("==============================");

    // Create a test problem with multiple variables and objectives
    let problem = create_test_problem();
    println!(
        "Problem created with {} objectives",
        problem.num_objectives()
    );
    println!("Linear objectives: {}", problem.objectives.len());
    println!("Regular objectives: {}", problem.objectives.len());

    // Test with flag array disabled
    println!("\n🔴 Testing with Flag Array DISABLED");
    let options_no_flag = Options::new()
        .with_name("flag_test_disabled")
        .with_grid_points(30)
        .with_bypass_coefficient(true)
        .with_flag_array(false)
        .with_early_exit(true);

    let start = Instant::now();
    let cloned_problem = problem.clone();
    println!(
        "Cloned problem objectives: {}",
        cloned_problem.num_objectives()
    );
    println!(
        "Cloned linear objectives: {}",
        cloned_problem.objectives.len()
    );
    println!(
        "Cloned regular objectives: {}",
        cloned_problem.objectives.len()
    );
    let mut solver = Augmecon::try_new(cloned_problem, options_no_flag)?;
    solver.solve()?;
    let time_no_flag = start.elapsed();
    let solutions_no_flag = solver.get_pareto_solutions().len();

    println!("   ⏱️  Time: {time_no_flag:.2?}");
    println!("   📊 Solutions: {solutions_no_flag}");

    // Test with flag array enabled
    println!("\n🟢 Testing with Flag Array ENABLED");
    let options_with_flag = Options::new()
        .with_name("flag_test_enabled")
        .with_grid_points(30)
        .with_bypass_coefficient(true)
        .with_flag_array(true)
        .with_early_exit(true);

    let start = Instant::now();
    let mut solver = Augmecon::try_new(problem, options_with_flag)?;
    solver.solve()?;
    let time_with_flag = start.elapsed();
    let solutions_with_flag = solver.get_pareto_solutions().len();

    println!("   ⏱️  Time: {time_with_flag:.2?}");
    println!("   📊 Solutions: {solutions_with_flag}");

    // Performance comparison
    println!("\n📈 Performance Comparison");
    println!("=========================");
    match time_no_flag.cmp(&time_with_flag) {
        std::cmp::Ordering::Greater => {
            let improvement = time_no_flag.as_secs_f64() / time_with_flag.as_secs_f64();
            println!("🚀 Flag array provided {improvement:.2}x speedup!");
        }
        std::cmp::Ordering::Less => {
            let slowdown = time_with_flag.as_secs_f64() / time_no_flag.as_secs_f64();
            println!("⚠️  Flag array was {slowdown:.2}x slower (overhead for small problems)");
        }
        std::cmp::Ordering::Equal => {
            println!("⚖️  Similar performance (times too close to measure)");
        }
    }

    println!(
        "✅ Solutions found are consistent: {}",
        solutions_no_flag == solutions_with_flag
    );

    Ok(())
}

/// Create a test problem that benefits from flag array optimization
fn create_test_problem() -> MultiObjectiveProblem {
    let mut problem = MultiObjectiveProblem::new();

    // Add multiple decision variables (resource allocation)
    for i in 0..4 {
        let var_name = format!("resource_{i}");
        problem.add_variable(
            var_name,
            VariableType::Continuous {
                min: Some(0.0),
                max: Some(50.0),
            },
        );
    }

    // Add capacity constraints (create infeasibility regions)
    let mut capacity_constraint = Expression::from(0.0);
    for i in 0..4 {
        if let Some(&var) = problem.var_map.get(&format!("resource_{i}")) {
            capacity_constraint += var;
        }
    }
    problem.add_constraint(constraint!(capacity_constraint <= 100.0));

    // Add budget constraint (another constraint that creates infeasible regions)
    let coeffs = [2.0, 3.0, 1.5, 4.0];
    let mut budget_constraint = Expression::from(0.0);
    for (i, &coeff) in coeffs.iter().enumerate() {
        if let Some(&var) = problem.var_map.get(&format!("resource_{i}")) {
            budget_constraint += coeff * var;
        }
    }
    problem.add_constraint(constraint!(budget_constraint <= 200.0));

    // Add production constraint (creates more complex feasible region)
    let prod_coeffs = [0.5, 2.0, 1.0, 0.8];
    let mut production_constraint = Expression::from(0.0);
    for (i, &coeff) in prod_coeffs.iter().enumerate() {
        if let Some(&var) = problem.var_map.get(&format!("resource_{i}")) {
            production_constraint += coeff * var;
        }
    }
    problem.add_constraint(constraint!(production_constraint >= 20.0));

    // First objective: Maximize profit
    let profit_coeffs = [10.0, 15.0, 8.0, 12.0];
    let mut profit_objective = Expression::from(0.0);
    for (i, &coeff) in profit_coeffs.iter().enumerate() {
        if let Some(&var) = problem.var_map.get(&format!("resource_{i}")) {
            profit_objective += coeff * var;
        }
    }
    problem.add_objective(profit_objective, ObjectiveDirection::Maximize);

    // Second objective: Minimize cost
    let cost_coeffs = [5.0, 8.0, 3.0, 6.0];
    let mut cost_objective = Expression::from(0.0);
    for (i, &coeff) in cost_coeffs.iter().enumerate() {
        if let Some(&var) = problem.var_map.get(&format!("resource_{i}")) {
            cost_objective += coeff * var;
        }
    }
    problem.add_objective(cost_objective, ObjectiveDirection::Minimize);

    // Third objective: Minimize environmental impact
    let env_coeffs = [2.0, 1.0, 4.0, 3.0];
    let mut env_objective = Expression::from(0.0);
    for (i, &coeff) in env_coeffs.iter().enumerate() {
        if let Some(&var) = problem.var_map.get(&format!("resource_{i}")) {
            env_objective += coeff * var;
        }
    }
    problem.add_objective(env_objective, ObjectiveDirection::Minimize);

    problem
}
