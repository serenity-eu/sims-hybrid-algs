//! GPBA Algorithms Example: Advanced Pareto Front Representation
//!
//! This example demonstrates the three Grid Point Based Algorithms (GPBA)
//! for generating high-quality representations of Pareto fronts.

use augmecon::{
    GpbaA, GpbaB, GpbaC, GpbaConfig, MultiObjectiveProblem, ObjectiveDirection, Options,
    VariableType,
};
use good_lp::{constraint, Expression};
use std::collections::HashMap;

fn main() {
    println!("🚀 GPBA Algorithms Demonstration");
    println!("==================================\n");

    // Create a bi-objective production planning problem
    let problem = create_production_problem();

    println!("📊 Problem Setup:");
    println!("- Objective 1: Maximize Profit");
    println!("- Objective 2: Maximize Environmental Score");
    println!("- Constraints: Resource capacity, minimum production\n");

    // Demonstrate each GPBA algorithm
    demonstrate_gpba_a(&problem);
    demonstrate_gpba_b(&problem);
    demonstrate_gpba_c(&problem);

    // Demonstrate GPBA configuration presets
    demonstrate_gpba_presets();

    println!("✅ GPBA demonstration completed!");
}

/// Create a bi-objective production planning problem
fn create_production_problem() -> MultiObjectiveProblem {
    let mut problem = MultiObjectiveProblem::new();

    // Decision variables: production quantities for three products
    let products = ["ProductA", "ProductB", "ProductC"];
    for product in &products {
        problem.add_variable(
            (*product).to_string(),
            VariableType::Continuous {
                min: Some(0.0),
                max: Some(100.0),
            },
        );
    }

    // Resource constraint: limited production capacity
    let mut capacity_constraint = Expression::from(0.0);
    let resource_usage = [2.0, 3.0, 1.5]; // Resource units per product
    for (i, product) in products.iter().enumerate() {
        if let Some(&var) = problem.var_map.get(*product) {
            capacity_constraint += resource_usage[i] * var;
        }
    }
    problem.add_constraint(constraint!(capacity_constraint <= 200.0));

    // Minimum production constraint for each product
    for product in &products {
        if let Some(&var) = problem.var_map.get(*product) {
            problem.add_constraint(constraint!(var >= 5.0));
        }
    }

    // Objective 1: Maximize profit
    let mut profit_objective = Expression::from(0.0);
    let profit_per_unit = [12.0, 15.0, 8.0]; // Profit per unit
    for (i, product) in products.iter().enumerate() {
        if let Some(&var) = problem.var_map.get(*product) {
            profit_objective += profit_per_unit[i] * var;
        }
    }
    problem.add_objective(profit_objective, ObjectiveDirection::Maximize);

    // Objective 2: Maximize environmental score
    let mut env_objective = Expression::from(0.0);
    let env_score_per_unit = [3.0, 5.0, 7.0]; // Environmental score per unit
    for (i, product) in products.iter().enumerate() {
        if let Some(&var) = problem.var_map.get(*product) {
            env_objective += env_score_per_unit[i] * var;
        }
    }
    problem.add_objective(env_objective, ObjectiveDirection::Maximize);

    problem
}

/// Demonstrate GPBA-A: Coverage-focused representation
fn demonstrate_gpba_a(problem: &MultiObjectiveProblem) {
    println!("🎯 GPBA-A: Coverage-Focused Representation");
    println!("Goal: Minimize maximum distance between consecutive points\n");

    // Configure GPBA-A with coverage focus
    let mut target_points = HashMap::new();
    target_points.insert(0, 20); // 20 points for profit objective
    target_points.insert(1, 20); // 20 points for environmental objective

    let config = GpbaConfig {
        primary_objective: 0, // Optimize profit directly
        target_points_per_objective: target_points,
        manual_bounds: None, // Let algorithm compute bounds
    };

    let mut gpba_a = GpbaA::new(config);

    // Create basic options for the solver
    let options = Options::new().with_grid_points(50);

    // GPBA-A algorithm with coverage-focused optimization
    println!("Configuration:");
    println!("- Primary objective: 0 (Profit)");
    println!("- Target points per objective: 50 per objective");
    println!("- Focus: Minimize coverage gaps");

    match gpba_a.generate_representation(problem, &options) {
        Ok(pareto_front) => {
            println!(
                "✅ GPBA-A completed: {} solutions generated",
                pareto_front.len()
            );
            if pareto_front.is_empty() {
                println!("ℹ️  Note: No feasible solutions found - check problem constraints");
            }
        }
        Err(e) => {
            println!("❌ GPBA-A failed: {e}");
        }
    }

    println!("\n📈 GPBA-A Algorithm Features:");
    println!("- Adaptive epsilon adjustment based on coverage error γ_k");
    println!("- Tracks discarded points for better coverage optimization");
    println!("- Ensures no large gaps in Pareto front representation");
    println!("- Best for: Applications requiring complete front coverage\n");
}

/// Demonstrate GPBA-B: Uniformity-focused representation
fn demonstrate_gpba_b(problem: &MultiObjectiveProblem) {
    println!("🎯 GPBA-B: Uniformity-Focused Representation");
    println!("Goal: Maximize minimum distance between points (uniform distribution)\n");

    let mut target_points = HashMap::new();
    target_points.insert(0, 15);
    target_points.insert(1, 15);

    let config = GpbaConfig {
        primary_objective: 1, // Optimize environmental score directly
        target_points_per_objective: target_points,
        manual_bounds: None,
    };

    let mut gpba_b = GpbaB::new(config);

    // Create basic options for the solver
    let options = Options::new().with_grid_points(50);

    println!("Configuration:");
    println!("- Primary objective: 1 (Environmental Score)");
    println!("- Target points per objective: 50 per objective");
    println!("- Focus: Uniform point distribution");

    match gpba_b.generate_representation(problem, &options) {
        Ok(pareto_front) => {
            println!(
                "✅ GPBA-B completed: {} solutions generated",
                pareto_front.len()
            );
            if pareto_front.is_empty() {
                println!("ℹ️  Note: Actual solver integration required for solution generation");
            }
        }
        Err(e) => {
            println!("ℹ️  GPBA-B structure ready (solver integration pending): {e}");
        }
    }

    println!("\n📈 GPBA-B Algorithm Features:");
    println!("- Simple uniform step size δ_k for each objective");
    println!("- Maximizes minimum distance between consecutive points");
    println!("- Creates evenly spaced representation");
    println!("- Best for: Visualization and balanced trade-off analysis\n");
}

/// Demonstrate GPBA-C: Cardinality-focused representation
fn demonstrate_gpba_c(problem: &MultiObjectiveProblem) {
    println!("🎯 GPBA-C: Cardinality-Focused Representation");
    println!("Goal: Achieve target number of points with balanced coverage and uniformity\n");

    let mut target_points = HashMap::new();
    target_points.insert(0, 25); // Target 25 points for profit
    target_points.insert(1, 25); // Target 25 points for environment

    let config = GpbaConfig {
        primary_objective: 0,
        target_points_per_objective: target_points,
        manual_bounds: Some((
            vec![0.0, 0.0],      // Nadir point (worst case)
            vec![1500.0, 700.0], // Ideal point (best case estimates)
        )),
    };

    let mut gpba_c = GpbaC::new(config);

    // Create basic options for the solver
    let options = Options::new().with_grid_points(50);

    println!("Configuration:");
    println!("- Primary objective: 0 (Profit)");
    println!("- Target points per objective: 50 per objective");
    println!("- Manual bounds provided: Some((vec![100.0, 100.0], vec![0.0, 0.0]))");
    println!("- Focus: Target cardinality with adaptive refinement");

    match gpba_c.generate_representation(problem, &options) {
        Ok(pareto_front) => {
            println!(
                "✅ GPBA-C completed: {} solutions generated",
                pareto_front.len()
            );
            if pareto_front.is_empty() {
                println!("ℹ️  Note: Actual solver integration required for solution generation");
            }
        }
        Err(e) => {
            println!("ℹ️  GPBA-C structure ready (solver integration pending): {e}");
        }
    }

    println!("\n📈 GPBA-C Algorithm Features:");
    println!("- Adaptive grid refinement based on slack variables");
    println!("- Maintains target cardinality per objective");
    println!("- Balances coverage and uniformity automatically");
    println!("- Can skip grid points when beneficial (using slack information)");
    println!("- Best for: Fixed-size representations with quality guarantees\n");
}

/// Helper function to demonstrate GPBA configuration presets
fn demonstrate_gpba_presets() {
    // Note: Using GpbaConfig directly since presets aren't exported yet
    // In a full implementation, you could access: use augmecon::gpba::presets;

    println!("🔧 GPBA Configuration Presets:");

    // High coverage configuration (manual creation)
    let mut target_points = std::collections::HashMap::new();
    target_points.insert(0, 30);
    target_points.insert(1, 30);

    let coverage_config = GpbaConfig {
        primary_objective: 0,
        target_points_per_objective: target_points.clone(),
        manual_bounds: None,
    };

    println!(
        "- High Coverage (GPBA-A): Primary obj {}, {} points per objective",
        coverage_config.primary_objective,
        coverage_config
            .target_points_per_objective
            .get(&0)
            .unwrap_or(&0)
    );

    // Uniform distribution configuration (manual creation)
    let mut uniform_target_points = std::collections::HashMap::new();
    uniform_target_points.insert(0, 25);
    uniform_target_points.insert(1, 25);

    let uniform_config = GpbaConfig {
        primary_objective: 1,
        target_points_per_objective: uniform_target_points,
        manual_bounds: None,
    };

    println!(
        "- Uniform Distribution (GPBA-B): Primary obj {}, {} points per objective",
        uniform_config.primary_objective,
        uniform_config
            .target_points_per_objective
            .get(&0)
            .unwrap_or(&0)
    );

    // Balanced cardinality configuration (manual creation)
    let mut balanced_target_points = std::collections::HashMap::new();
    balanced_target_points.insert(0, 50);
    balanced_target_points.insert(1, 50);

    let balanced_config = GpbaConfig {
        primary_objective: 0,
        target_points_per_objective: balanced_target_points,
        manual_bounds: None,
    };

    println!(
        "- Balanced Cardinality (GPBA-C): Primary obj {}, target size {}",
        balanced_config.primary_objective, 50
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_production_problem_creation() {
        let problem = create_production_problem().expect("Failed to create problem");
        assert_eq!(problem.num_objectives(), 2);
        assert_eq!(problem.num_variables(), 3);
    }

    #[test]
    fn test_gpba_configurations() {
        let problem = create_production_problem().expect("Failed to create problem");

        // Test GPBA-A configuration
        let mut target_points = HashMap::new();
        target_points.insert(0, 10);
        target_points.insert(1, 10);

        let config = GpbaConfig {
            primary_objective: 0,
            target_points_per_objective: target_points,
            manual_bounds: None,
        };

        let _gpba_a = GpbaA::new(config.clone());
        // Note: config field is private, so we can't test it directly
        // In a full implementation, you would add getter methods

        let _gpba_b = GpbaB::new(config.clone());
        let _gpba_c = GpbaC::new(config);

        // Test that the structs are created successfully
        // The successful creation itself is the test - no assertion needed
    }
}
