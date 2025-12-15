//! # Solver Selection Test (Non-Gurobi)
//!
//! This example demonstrates how to use `HiGHS` and `CoinCbc` solvers with AUGMECON-RS.

#[cfg(any(feature = "coin_cbc", feature = "highs"))]
use augmecon::{MultiObjectiveProblem, ObjectiveDirection, VariableType};
#[cfg(any(feature = "coin_cbc", feature = "highs"))]
use good_lp::constraint;

fn main() {
    #[cfg(any(feature = "coin_cbc", feature = "highs"))]
    {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Info)
            .init();

        println!("🔧 AUGMECON-RS Solver Selection Test (HiGHS & CoinCbc)");
        println!("=====================================================");

        run_solver_if_enabled("CoinCbc", Solver::CoinCbc, "test_cbc");
        run_solver_if_enabled("HiGHS", Solver::HiGHS, "test_highs");

        println!("\n🎯 Test completed!");
    }

    #[cfg(not(any(feature = "coin_cbc", feature = "highs")))]
    {
        println!("⚠️  No solvers enabled. Enable 'coin_cbc' or 'highs' features.");
        println!("   Run with: cargo run --example solver_test_non_gurobi --features highs,coin_cbc");
    }
}

#[cfg(any(feature = "coin_cbc", feature = "highs"))]
fn run_solver_if_enabled(solver_name: &str, solver: Solver, test_name: &str) {
    // Only run if the feature for this solver is enabled
    match solver {
        Solver::CoinCbc => {
            #[cfg(feature = "coin_cbc")]
            run_solver(solver_name, solver, test_name);
        }
        Solver::HiGHS => {
            #[cfg(feature = "highs")]
            run_solver(solver_name, solver, test_name);
        }
        _ => {}
    }
}

#[cfg(any(feature = "coin_cbc", feature = "highs"))]
fn run_solver(solver_name: &str, solver: Solver, test_name: &str) {
    println!("\n📊 Testing with {solver_name} solver:");
    let problem = create_simple_problem();
    let options = Options::new()
        .with_name(test_name)
        .with_solver(solver)
        .with_grid_points(5);

    let mut solver_instance = Augmecon::try_new(problem, options)
        .unwrap_or_else(|e| panic!("Failed to create {solver_name} solver: {e}"));
    let solutions = solver_instance.solve()
        .unwrap_or_else(|e| panic!("{solver_name} solve failed: {e}"));
    println!("✅ {solver_name} solver: Found {} solutions", solutions.len());

    for (i, sol) in solutions.iter().take(3).enumerate() {
        println!("  Solution {}: {:?}", i + 1, sol.objectives());
    }
}
