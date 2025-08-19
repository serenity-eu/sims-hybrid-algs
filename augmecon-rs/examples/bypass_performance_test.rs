//! Quick test of bypass coefficient disabled vs enabled

use augmecon::{Augmecon, MultiObjectiveProblem, ObjectiveDirection, Options, VariableType};
use good_lp::constraint;
use std::error::Error;
use std::time::Instant;

fn create_production_problem() -> MultiObjectiveProblem {
    let mut problem = MultiObjectiveProblem::new();

    // Add decision variables: production quantities for two products
    problem.add_variable(
        "product_a".to_string(),
        VariableType::Continuous {
            min: Some(0.0),
            max: Some(100.0),
        },
    );
    problem.add_variable(
        "product_b".to_string(),
        VariableType::Continuous {
            min: Some(0.0),
            max: Some(100.0),
        },
    );

    // Add capacity constraint: product_a + product_b <= 100
    if let (Some(&prod_a), Some(&prod_b)) = (
        problem.var_map.get("product_a"),
        problem.var_map.get("product_b"),
    ) {
        let capacity_constraint = prod_a + prod_b;
        problem.add_constraint(constraint!(capacity_constraint <= 100.0));
    }

    // Objective 1: Maximize profit = 5 * product_a + 3 * product_b
    if let (Some(&prod_a), Some(&prod_b)) = (
        problem.var_map.get("product_a"),
        problem.var_map.get("product_b"),
    ) {
        let profit_objective = 5.0 * prod_a + 3.0 * prod_b;
        problem.add_objective(profit_objective, ObjectiveDirection::Maximize);
    }

    // Objective 2: Minimize environmental impact = 3 * product_a + 3 * product_b
    if let (Some(&prod_a), Some(&prod_b)) = (
        problem.var_map.get("product_a"),
        problem.var_map.get("product_b"),
    ) {
        let impact_objective = 3.0 * prod_a + 3.0 * prod_b;
        problem.add_objective(impact_objective, ObjectiveDirection::Minimize);
    }

    problem
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("🧪 Bypass Coefficient Performance Test");
    println!("======================================");

    // Test WITHOUT bypass coefficient
    println!("\n📉 Testing WITHOUT bypass coefficient optimization:");
    let problem1 = create_production_problem();
    let options_without_bypass = Options::new()
        .with_name("test_without_bypass")
        .with_grid_points(50)
        .with_bypass_coefficient(false);

    let start = Instant::now();
    let mut solver_without_bypass = Augmecon::try_new(problem1, options_without_bypass)?;
    solver_without_bypass.solve()?;
    let duration_without_bypass = start.elapsed();

    let pareto_solutions_without = solver_without_bypass.get_pareto_solutions();
    println!(
        "   ✅ Found {} Pareto solutions",
        pareto_solutions_without.len()
    );
    println!("   ⏱️  Time: {duration_without_bypass:?}");

    // Test WITH bypass coefficient (default)
    println!("\n📈 Testing WITH bypass coefficient optimization:");
    let problem2 = create_production_problem();
    let options_with_bypass = Options::new()
        .with_name("test_with_bypass")
        .with_grid_points(50)
        .with_bypass_coefficient(true);

    let start = Instant::now();
    let mut solver_with_bypass = Augmecon::try_new(problem2, options_with_bypass)?;
    solver_with_bypass.solve()?;
    let duration_with_bypass = start.elapsed();

    let pareto_solutions_with = solver_with_bypass.get_pareto_solutions();
    println!(
        "   ✅ Found {} Pareto solutions",
        pareto_solutions_with.len()
    );
    println!("   ⏱️  Time: {duration_with_bypass:?}");

    // Compare results
    println!("\n📊 Performance Comparison:");
    println!(
        "   Without bypass: {} solutions in {:?}",
        pareto_solutions_without.len(),
        duration_without_bypass
    );
    println!(
        "   With bypass:    {} solutions in {:?}",
        pareto_solutions_with.len(),
        duration_with_bypass
    );

    if duration_without_bypass > duration_with_bypass {
        let speedup = duration_without_bypass.as_secs_f64() / duration_with_bypass.as_secs_f64();
        println!("   🚀 Speedup: {speedup:.2}x faster with bypass coefficient!");
    } else {
        println!("   ⚠️  Bypass coefficient didn't provide significant speedup for this problem");
    }

    // Verify same number of solutions
    if pareto_solutions_with.len() == pareto_solutions_without.len() {
        println!("   ✅ Both methods found the same number of Pareto solutions");
    } else {
        println!(
            "   ⚠️  Different number of solutions found! ({} vs {})",
            pareto_solutions_with.len(),
            pareto_solutions_without.len()
        );
    }

    Ok(())
}
