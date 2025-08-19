//! # Basic AUGMECON Example
//!
//! This example demonstrates the fundamental usage of AUGMECON-RS for solving
//! a simple two-objective optimization problem.
//!
//! ## Problem Description
//!
//! We have a manufacturing company that produces two products (A and B).
//! The company wants to:
//! 1. Maximize total profit
//! 2. Minimize environmental impact
//!
//! ### Problem Data
//! - Product A: Profit = $3/unit, Environmental impact = 2 units/unit
//! - Product B: Profit = $2/unit, Environmental impact = 1 unit/unit
//!
//! ### Constraints
//! - Maximum 100 units of Product A
//! - Maximum 150 units of Product B
//! - Total production capacity: 200 units
//!
//! ## Expected Output
//!
//! The solver will find multiple Pareto-optimal solutions representing different
//! trade-offs between profit maximization and environmental impact minimization.

use augmecon::solution::HasObjectives;
use augmecon::{Augmecon, MultiObjectiveProblem, ObjectiveDirection, Options, VariableType};
use good_lp::constraint;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging to see solver progress
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("🏭 AUGMECON-RS Basic Example: Production Planning");
    println!("================================================");

    // Create the multi-objective problem
    let problem = create_production_problem();

    // Configure solver with basic AUGMECON settings
    let options = Options::new()
        .with_name("production_planning")
        .with_grid_points(50) // 50 grid points for good resolution
        .with_penalty_weight(1e-3); // Standard precision

    println!("📊 Problem Configuration:");
    println!("  Variables: {}", problem.variable_types.len());
    println!("  Constraints: {}", problem.constraints.len());
    println!("  Objectives: {}", problem.num_objectives());
    println!("  Grid Points: {}", options.grid_points.unwrap());
    println!();

    // Create and solve the problem
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

    // Decision variables: production quantities
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

    // Constraint: Total production capacity
    // product_a + product_b <= 200
    if let (Some(&prod_a), Some(&prod_b)) = (
        problem.var_map.get("product_a"),
        problem.var_map.get("product_b"),
    ) {
        let capacity_constraint = prod_a + prod_b;
        problem.add_constraint(constraint!(capacity_constraint <= 200.0));
    }

    // Objective 1: Maximize profit
    // profit = 3 * product_a + 2 * product_b
    if let (Some(&prod_a), Some(&prod_b)) = (
        problem.var_map.get("product_a"),
        problem.var_map.get("product_b"),
    ) {
        let profit_objective = 3.0 * prod_a + 2.0 * prod_b;
        problem.add_objective(profit_objective, ObjectiveDirection::Maximize);
    }

    // Objective 2: Minimize environmental impact
    // impact = 2 * product_a + 1 * product_b
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
    let _pareto_front = solver.get_pareto_front();
    let pareto_solutions = solver.get_pareto_solutions();

    println!("🎯 Results Summary");
    println!("=================");
    println!("Found {} Pareto-optimal solutions", pareto_solutions.len());
    println!();

    // Display payoff table
    println!("📊 Payoff Table (Objective Ranges):");
    let payoff_table = solver.get_payoff_table();
    println!(
        "  Profit:    [{:.2}, {:.2}]",
        payoff_table[0][0], payoff_table[1][0]
    );
    println!(
        "  Impact:    [{:.2}, {:.2}]",
        payoff_table[0][1], payoff_table[1][1]
    );
    println!();

    // Display some representative solutions
    println!("🏆 Representative Pareto Solutions:");
    println!(
        "  {:>3} {:>12} {:>12} {:>12} {:>12}",
        "No.", "Profit ($)", "Impact", "Product A", "Product B"
    );
    println!("  {}", "-".repeat(60));

    let num_to_show = 10.min(pareto_solutions.len());
    let step = if pareto_solutions.len() > num_to_show {
        pareto_solutions.len() / num_to_show
    } else {
        1
    };

    for (i, solution) in pareto_solutions.iter().enumerate() {
        if i.is_multiple_of(step) && i / step < num_to_show {
            let solution_num = i / step + 1;
            let objectives = solution.objectives();
            let variables = &solution.decision_variables;

            let product_a = variables.get("product_a").unwrap_or(&0.0);
            let product_b = variables.get("product_b").unwrap_or(&0.0);

            println!(
                "  {:>3} {:>12.2} {:>12.2} {:>12.2} {:>12.2}",
                solution_num, objectives[0], objectives[1], product_a, product_b
            );
        }
    }
    println!();

    // Analysis and insights
    println!("💡 Analysis:");

    // Best profit solution
    if let Some(best_profit) = pareto_solutions.first() {
        let objectives = best_profit.objectives();
        println!(
            "  Best Profit: ${:.2} (Impact: {:.2})",
            objectives[0], objectives[1]
        );
    }

    // Best environmental solution
    if let Some(best_env) = pareto_solutions.last() {
        let objectives = best_env.objectives();
        println!(
            "  Best Environment: Impact {:.2} (Profit: ${:.2})",
            objectives[1], objectives[0]
        );
    }

    // Calculate trade-off ratio
    if pareto_solutions.len() >= 2 {
        let first = pareto_solutions.first().unwrap().objectives();
        let last = pareto_solutions.last().unwrap().objectives();

        let profit_diff = (first[0] - last[0]).abs();
        let impact_diff = (last[1] - first[1]).abs();

        if impact_diff > 0.0 {
            let trade_off_ratio = profit_diff / impact_diff;
            println!("  Trade-off Ratio: ${trade_off_ratio:.2} profit per unit impact reduction");
        }
    }

    println!();
    println!("📈 Recommendation:");
    println!("  Review the Pareto solutions above to select the best trade-off");
    println!("  between profit maximization and environmental impact minimization");
    println!("  based on your company's priorities and constraints.");
}
