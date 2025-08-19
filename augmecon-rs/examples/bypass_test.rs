//! Test to verify bypass coefficient optimization
//!
//! This test compares solving the same problem with and without bypass coefficient
//! to demonstrate the performance improvement.

use augmecon::{Augmecon, MultiObjectiveProblem, ObjectiveDirection, Options, VariableType};
use good_lp::constraint;
use std::error::Error;
use std::time::Instant;

fn create_simple_problem() -> MultiObjectiveProblem {
    let mut problem = MultiObjectiveProblem::new();

    // Add variables
    problem.add_variable(
        "x1".to_string(),
        VariableType::Continuous {
            min: Some(0.0),
            max: Some(100.0),
        },
    );
    problem.add_variable(
        "x2".to_string(),
        VariableType::Continuous {
            min: Some(0.0),
            max: Some(100.0),
        },
    );

    // Add constraint: x1 + x2 <= 100
    if let (Some(&x1), Some(&x2)) = (problem.var_map.get("x1"), problem.var_map.get("x2")) {
        let constraint_expr = x1 + x2;
        problem.add_constraint(constraint!(constraint_expr <= 100.0));
    }

    // Add objectives
    // Objective 1: maximize x1 + x2 (profit)
    if let (Some(&x1), Some(&x2)) = (problem.var_map.get("x1"), problem.var_map.get("x2")) {
        let obj1 = x1 + x2;
        problem.add_objective(obj1, ObjectiveDirection::Maximize);
    }

    // Objective 2: minimize 0.3*x1 + 0.3*x2 (cost)
    if let (Some(&x1), Some(&x2)) = (problem.var_map.get("x1"), problem.var_map.get("x2")) {
        let obj2 = 0.3 * x1 + 0.3 * x2;
        problem.add_objective(obj2, ObjectiveDirection::Minimize);
    }

    problem
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("🧪 Bypass Coefficient Test");
    println!("==========================");

    let problem = create_simple_problem();

    // Test WITH bypass coefficient (default)
    println!("\n📈 Testing WITH bypass coefficient optimization:");
    let options_with_bypass = Options::new()
        .with_name("test_with_bypass")
        .with_grid_points(50)
        .with_bypass_coefficient(true);

    let start = Instant::now();
    let mut solver_with_bypass = Augmecon::try_new(problem.clone(), options_with_bypass)?;
    solver_with_bypass.solve()?;
    let duration_with_bypass = start.elapsed();

    let pareto_solutions_with = solver_with_bypass.get_pareto_solutions();
    println!(
        "   ✅ Found {} Pareto solutions",
        pareto_solutions_with.len()
    );
    println!("   ⏱️  Time: {duration_with_bypass:?}");

    // Test WITHOUT bypass coefficient
    println!("\n📉 Testing WITHOUT bypass coefficient optimization:");
    let options_without_bypass = Options::new()
        .with_name("test_without_bypass")
        .with_grid_points(50)
        .with_bypass_coefficient(false);

    let start = Instant::now();
    let mut solver_without_bypass = Augmecon::try_new(problem, options_without_bypass)?;
    solver_without_bypass.solve()?;
    let duration_without_bypass = start.elapsed();

    let pareto_solutions_without = solver_without_bypass.get_pareto_solutions();
    println!(
        "   ✅ Found {} Pareto solutions",
        pareto_solutions_without.len()
    );
    println!("   ⏱️  Time: {duration_without_bypass:?}");

    // Compare results
    println!("\n📊 Comparison:");
    println!(
        "   With bypass:    {} solutions in {:?}",
        pareto_solutions_with.len(),
        duration_with_bypass
    );
    println!(
        "   Without bypass: {} solutions in {:?}",
        pareto_solutions_without.len(),
        duration_without_bypass
    );

    if duration_without_bypass > duration_with_bypass {
        let speedup = duration_without_bypass.as_secs_f64() / duration_with_bypass.as_secs_f64();
        println!("   🚀 Speedup: {speedup:.2}x faster with bypass coefficient!");
    } else {
        println!("   ⚠️  No significant speedup observed (problem too simple)");
    }

    // Verify same number of solutions
    if pareto_solutions_with.len() == pareto_solutions_without.len() {
        println!("   ✅ Both methods found the same number of Pareto solutions");
    } else {
        println!("   ⚠️  Different number of solutions found!");
    }

    Ok(())
}
