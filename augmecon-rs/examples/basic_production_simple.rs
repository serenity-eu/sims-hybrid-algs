//! # Basic AUGMECON Example: Production Planning
//!
//! This example demonstrates the fundamental usage of AUGMECON-RS for solving
//! a simple two-objective optimization problem.

use augmecon::solution::HasObjectives;
use augmecon::{Augmecon, MultiObjectiveProblem, ObjectiveDirection, Options, VariableType};
use good_lp::constraint;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("🏭 AUGMECON-RS Basic Example: Production Planning");
    println!("================================================");

    // Create the multi-objective problem
    let problem = create_production_problem();

    // Configure solver
    let options = Options::new()
        .with_name("production_planning")
        .with_grid_points(50)
        .with_penalty_weight(1e-3);

    println!("📊 Problem Configuration:");
    println!("  Variables: {}", problem.variable_types.len());
    println!("  Constraints: {}", problem.constraints.len());
    println!("  Objectives: {}", problem.num_objectives());
    println!();

    // Solve the problem
    println!("🚀 Starting optimization...");
    let start_time = std::time::Instant::now();

    let mut solver = Augmecon::try_new(problem, options)?;
    solver.solve()?;

    let elapsed = start_time.elapsed();
    println!("✅ Optimization completed in {elapsed:.2?}");
    println!();

    // Display results
    display_results(&solver);

    Ok(())
}

fn create_production_problem() -> MultiObjectiveProblem {
    let mut problem = MultiObjectiveProblem::new();

    // Add variables
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
            max: Some(150.0),
        },
    );

    // Add capacity constraint: product_a + product_b <= 200
    if let (Some(&prod_a), Some(&prod_b)) = (
        problem.var_map.get("product_a"),
        problem.var_map.get("product_b"),
    ) {
        let capacity_constraint = prod_a + prod_b;
        problem.add_constraint(constraint!(capacity_constraint <= 200.0));
    }

    // Add profit objective (maximize): 3*product_a + 2*product_b
    if let (Some(&prod_a), Some(&prod_b)) = (
        problem.var_map.get("product_a"),
        problem.var_map.get("product_b"),
    ) {
        let profit_objective = 3.0 * prod_a + 2.0 * prod_b;
        problem.add_objective(profit_objective, ObjectiveDirection::Maximize);
    }

    // Add environmental impact objective (minimize): 2*product_a + 1*product_b
    if let (Some(&prod_a), Some(&prod_b)) = (
        problem.var_map.get("product_a"),
        problem.var_map.get("product_b"),
    ) {
        let impact_objective = 2.0 * prod_a + 1.0 * prod_b;
        problem.add_objective(impact_objective, ObjectiveDirection::Minimize);
    }

    problem
}

fn display_results(solver: &Augmecon) {
    let pareto_solutions = solver.get_pareto_solutions();
    let payoff_table = solver.get_payoff_table();

    println!("🎯 Results Summary");
    println!("=================");
    println!("Found {} Pareto-optimal solutions", pareto_solutions.len());
    println!();

    // Display payoff table
    println!("📊 Payoff Table:");
    println!(
        "  Profit:  [{:.2}, {:.2}]",
        payoff_table[0][0], payoff_table[1][0]
    );
    println!(
        "  Impact:  [{:.2}, {:.2}]",
        payoff_table[0][1], payoff_table[1][1]
    );
    println!();

    // Display some solutions
    println!("🏆 Sample Pareto Solutions:");
    println!(
        "  {:>3} {:>10} {:>10} {:>10} {:>10}",
        "No.", "Profit", "Impact", "Prod_A", "Prod_B"
    );
    println!("  {}", "-".repeat(50));

    let num_to_show = 5.min(pareto_solutions.len());
    for i in 0..num_to_show {
        let idx = i * pareto_solutions.len() / num_to_show.max(1);
        let solution = &pareto_solutions[idx.min(pareto_solutions.len() - 1)];
        let objectives = solution.objectives();
        let vars = &solution.decision_variables;

        let prod_a = vars.get("product_a").expect("Missing product_a variable");
        let prod_b = vars.get("product_b").expect("Missing product_b variable");

        println!(
            "  {:>3} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            i + 1,
            objectives[0],
            objectives[1],
            prod_a,
            prod_b
        );
    }

    println!();
    println!("💡 Analysis:");
    if let Some(best_profit) = pareto_solutions.first() {
        let objectives = best_profit.objectives();
        println!(
            "  Best Profit: ${:.2} (Impact: {:.2})",
            objectives[0], objectives[1]
        );
    }

    if let Some(best_env) = pareto_solutions.last() {
        let objectives = best_env.objectives();
        println!(
            "  Best Environment: Impact {:.2} (Profit: ${:.2})",
            objectives[1], objectives[0]
        );
    }
}
