//! # Portfolio Optimization Example
//!
//! This example demonstrates using AUGMECON-RS for financial portfolio optimization,
//! a classic multi-objective optimization problem balancing return and risk.
//!
//! ## Problem Description
//!
//! We want to construct an optimal investment portfolio with:
//! - **Objective 1**: Maximize expected return
//! - **Objective 2**: Minimize portfolio risk (simplified linear model)
//!
//! ### Assets Available
//! - Stocks: 12% expected return, 20% risk factor
//! - Bonds: 6% expected return, 5% risk factor  
//! - Commodities: 9% expected return, 15% risk factor
//! - Real Estate: 8% expected return, 12% risk factor
//!
//! ### Constraints
//! - Portfolio weights must sum to 100%
//! - No short selling (all weights ≥ 0%)
//! - Maximum 60% in any single asset class
//! - Minimum 5% in bonds (for stability)

use augmecon::solution::HasObjectives;
use augmecon::{Augmecon, MultiObjectiveProblem, ObjectiveDirection, Options, VariableType};
use good_lp::{constraint, Expression};
use std::collections::HashMap;
use std::error::Error;

// Asset data structure
#[derive(Debug)]
struct Asset {
    name: &'static str,
    expected_return: f64, // Annual expected return (as decimal)
    risk_factor: f64,     // Risk metric (simplified)
}

const ASSETS: &[Asset] = &[
    Asset {
        name: "stocks",
        expected_return: 0.12,
        risk_factor: 0.20,
    },
    Asset {
        name: "bonds",
        expected_return: 0.06,
        risk_factor: 0.05,
    },
    Asset {
        name: "commodities",
        expected_return: 0.09,
        risk_factor: 0.15,
    },
    Asset {
        name: "real_estate",
        expected_return: 0.08,
        risk_factor: 0.12,
    },
];

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("💰 AUGMECON-RS Portfolio Optimization Example");
    println!("==============================================");

    display_asset_information();

    // Create the portfolio optimization problem
    let problem = create_portfolio_problem();

    // Configure solver for financial precision
    let options = Options::new()
        .with_name("portfolio_optimization")
        .with_grid_points(100) // High resolution for finance
        .with_penalty_weight(1e-6) // High precision
        .with_round_decimals(8); // Financial precision

    println!("📊 Problem Configuration:");
    println!("  Assets: {}", ASSETS.len());
    println!("  Variables: {}", problem.variable_types.len());
    println!("  Constraints: {}", problem.constraints.len());
    println!("  Objectives: {}", problem.num_objectives());
    println!("  Grid Points: {}", options.grid_points.unwrap());
    println!();

    // Solve the optimization problem
    println!("🚀 Starting portfolio optimization...");
    let start_time = std::time::Instant::now();

    let mut solver = Augmecon::try_new(problem, options)?;
    solver.solve()?;

    let elapsed = start_time.elapsed();
    println!("✅ Optimization completed in {elapsed:.2?}");
    println!();

    // Analyze and display results
    analyze_portfolio_results(&solver);

    Ok(())
}

fn display_asset_information() {
    println!("📈 Available Assets:");
    println!(
        "  {:>12} {:>15} {:>12}",
        "Asset", "Expected Return", "Risk Factor"
    );
    println!("  {}", "-".repeat(42));

    for asset in ASSETS {
        println!(
            "  {:>12} {:>14.1}% {:>11.1}%",
            asset.name,
            asset.expected_return * 100.0,
            asset.risk_factor * 100.0
        );
    }
    println!();
}

fn create_portfolio_problem() -> MultiObjectiveProblem {
    let mut problem = MultiObjectiveProblem::new();

    // Add weight variables for each asset (0% to 60% allocation)
    for asset in ASSETS {
        problem.add_variable(
            format!("weight_{}", asset.name),
            VariableType::Continuous {
                min: Some(0.0),
                max: Some(0.60),
            },
        );
    }

    // Constraint 1: Portfolio weights must sum to 100%
    let mut budget_constraint = Expression::from(0.0);
    for asset in ASSETS {
        if let Some(&var) = problem.var_map.get(&format!("weight_{}", asset.name)) {
            budget_constraint += var;
        }
    }
    problem.add_constraint(constraint!(budget_constraint == 1.0));

    // Constraint 2: Minimum 5% in bonds for stability
    if let Some(&bonds_var) = problem.var_map.get("weight_bonds") {
        problem.add_constraint(constraint!(bonds_var >= 0.05));
    }

    // Objective 1: Maximize expected portfolio return
    let mut return_objective = Expression::from(0.0);
    for asset in ASSETS {
        if let Some(&var) = problem.var_map.get(&format!("weight_{}", asset.name)) {
            return_objective += asset.expected_return * var;
        }
    }
    problem.add_objective(return_objective, ObjectiveDirection::Maximize);

    // Objective 2: Minimize portfolio risk (simplified linear model)
    let mut risk_objective = Expression::from(0.0);
    for asset in ASSETS {
        if let Some(&var) = problem.var_map.get(&format!("weight_{}", asset.name)) {
            risk_objective += asset.risk_factor * var;
        }
    }
    problem.add_objective(risk_objective, ObjectiveDirection::Minimize);

    problem
}

fn analyze_portfolio_results(solver: &Augmecon) {
    let pareto_solutions = solver.get_pareto_solutions();
    let payoff_table = solver.get_payoff_table();

    println!("🎯 Portfolio Optimization Results");
    println!("=================================");
    println!("Found {} efficient portfolios", pareto_solutions.len());
    println!();

    // Display payoff table
    println!("📊 Efficient Frontier Bounds:");
    println!(
        "  Expected Return: [{:.2}%, {:.2}%]",
        payoff_table[1][0] * 100.0,
        payoff_table[0][0] * 100.0
    );
    println!(
        "  Portfolio Risk:  [{:.2}%, {:.2}%]",
        payoff_table[0][1] * 100.0,
        payoff_table[1][1] * 100.0
    );
    println!();

    // Display representative portfolios
    println!("🏆 Representative Efficient Portfolios:");
    display_portfolio_table(pareto_solutions);

    // Strategic analysis
    println!();
    println!("📈 Strategic Analysis:");
    analyze_portfolio_strategies(pareto_solutions);

    // Risk-Return recommendations
    println!();
    println!("💡 Investment Recommendations:");
    provide_investment_recommendations(pareto_solutions);
}

fn display_portfolio_table(solutions: &[augmecon::Solution]) {
    println!(
        "  {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Return%", "Risk%", "Stocks%", "Bonds%", "Comm.%", "RealEst%", "Profile"
    );
    println!("  {}", "-".repeat(70));

    let num_to_show = 8.min(solutions.len());
    let indices = select_representative_solutions(solutions, num_to_show);

    for &idx in &indices {
        let solution = &solutions[idx];
        let objectives = solution.objectives();
        let variables = &solution.decision_variables;

        let return_pct = objectives[0] * 100.0;
        let risk_pct = objectives[1] * 100.0;

        let stocks = variables.get("weight_stocks").unwrap_or(&0.0) * 100.0;
        let bonds = variables.get("weight_bonds").unwrap_or(&0.0) * 100.0;
        let commodities = variables.get("weight_commodities").unwrap_or(&0.0) * 100.0;
        let real_estate = variables.get("weight_real_estate").unwrap_or(&0.0) * 100.0;

        let profile = classify_portfolio_profile(return_pct, risk_pct);

        println!(
            "  {return_pct:>7.2} {risk_pct:>7.2} {stocks:>7.1} {bonds:>6.1} {commodities:>7.1} {real_estate:>8.1} {profile:>8}"
        );
    }
}

fn select_representative_solutions(solutions: &[augmecon::Solution], count: usize) -> Vec<usize> {
    if solutions.len() <= count {
        return (0..solutions.len()).collect();
    }

    let mut indices = Vec::new();
    #[expect(
        clippy::cast_precision_loss,
        reason = "Converting solution count to f64 for proportional selection - precision loss acceptable for display purposes"
    )]
    let step = solutions.len() as f64 / count as f64;

    for i in 0..count {
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "Converting index calculations for solution selection - precision loss and truncation acceptable for display indexing"
        )]
        let idx = (i as f64 * step) as usize;
        indices.push(idx.min(solutions.len() - 1));
    }

    indices
}

fn classify_portfolio_profile(return_pct: f64, risk_pct: f64) -> &'static str {
    match (return_pct, risk_pct) {
        (r, k) if r >= 10.0 && k >= 15.0 => "Aggressive",
        (r, k) if r >= 8.0 && k >= 10.0 => "Growth",
        (r, k) if r >= 7.0 && k <= 10.0 => "Balanced",
        (r, k) if r >= 6.0 && k <= 8.0 => "Conservative",
        _ => "Capital Preservation",
    }
}

fn analyze_portfolio_strategies(solutions: &[augmecon::Solution]) {
    if solutions.is_empty() {
        return;
    }

    // Analyze extreme portfolios
    let high_return = &solutions[0];
    let low_risk = &solutions[solutions.len() - 1];

    println!("  🚀 High Return Strategy:");
    print_portfolio_composition(&high_return.decision_variables, "    ");

    println!("  🛡️  Low Risk Strategy:");
    print_portfolio_composition(&low_risk.decision_variables, "    ");

    // Find balanced portfolio (closest to middle)
    if solutions.len() >= 3 {
        let mid_idx = solutions.len() / 2;
        let balanced = &solutions[mid_idx];
        println!("  ⚖️  Balanced Strategy:");
        print_portfolio_composition(&balanced.decision_variables, "    ");
    }
}

fn print_portfolio_composition(variables: &HashMap<String, f64>, indent: &str) {
    for asset in ASSETS {
        let weight = variables
            .get(&format!("weight_{}", asset.name))
            .unwrap_or(&0.0);
        if *weight > 0.001 {
            // Only show allocations > 0.1%
            println!("{}{}%: {:.1}%", indent, asset.name, weight * 100.0);
        }
    }
}

fn provide_investment_recommendations(solutions: &[augmecon::Solution]) {
    println!("  1. Conservative Investors (Risk-averse):");
    println!("     → Choose portfolios with risk < 8%");
    println!("     → Focus on bond-heavy allocations");
    println!();

    println!("  2. Moderate Investors (Balanced approach):");
    println!("     → Select middle-range portfolios (7-9% return, 8-12% risk)");
    println!("     → Diversify across all asset classes");
    println!();

    println!("  3. Aggressive Investors (Growth-focused):");
    println!("     → Consider high-return portfolios (>10% return)");
    println!("     → Accept higher volatility for potential gains");
    println!();

    println!("  4. Institutional Investors:");
    println!("     → Evaluate multiple portfolios based on liability structure");
    println!("     → Consider regulatory constraints and risk budgets");

    // Calculate Sharpe ratio approximation for top portfolios
    if solutions.len() >= 3 {
        println!();
        println!("  📊 Risk-Adjusted Performance (Top 3):");

        for (i, solution) in solutions.iter().take(3).enumerate() {
            let objectives = solution.objectives();
            let return_val = objectives[0];
            let risk_val = objectives[1];

            // Simplified Sharpe ratio (assuming risk-free rate = 3%)
            let risk_free_rate = 0.03;
            let sharpe_approx = if risk_val > 0.0 {
                (return_val - risk_free_rate) / risk_val
            } else {
                0.0
            };

            println!(
                "     Portfolio {}: Return {:.2}%, Risk {:.2}%, Sharpe≈{:.2}",
                i + 1,
                return_val * 100.0,
                risk_val * 100.0,
                sharpe_approx
            );
        }
    }
}
