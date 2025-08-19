//! Integration tests for general objective optimization problems
//! Tests two-objective, three-objective, and three-objective mixed problems

use augmecon::{
    constraint, variable, Augmecon, MultiObjectiveProblem, ObjectiveDirection, ProblemVariables,
};
use good_lp::variables;
use std::collections::HashMap;

// ============================================================================
// Helper Functions (specific to objective tests)
// ============================================================================

/// Initialize test logging
fn init_test_logging() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        env_logger::init();
    });
}

/// Compare two floating point numbers with tolerance
fn float_equal(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() < tolerance
}

/// Compare two 2D arrays with tolerance
fn array_equal(a: &[Vec<f64>], b: &[Vec<f64>], tolerance: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for (i, (row_a, row_b)) in a.iter().zip(b.iter()).enumerate() {
        if row_a.len() != row_b.len() {
            return false;
        }

        for (j, (val_a, val_b)) in row_a.iter().zip(row_b.iter()).enumerate() {
            if !float_equal(*val_a, *val_b, tolerance) {
                log::debug!(
                    "Mismatch at [{i}][{j}]: expected {val_b}, got {val_a}, diff: {}",
                    (val_a - val_b).abs()
                );
                return false;
            }
        }
    }

    true
}

/// Create options for two-objective test problem
fn options_for_two_objectives() -> augmecon::Options {
    augmecon::Options::new()
        .with_name("two_objective_model".to_string())
        .with_grid_points(10)
        .with_solver_option("log", "0")
}

/// Create options for three-objective test problem
fn options_for_three_objectives() -> augmecon::Options {
    augmecon::Options::new()
        .with_name("three_objective_model".to_string())
        .with_grid_points(10)
        .with_early_exit(false)
        .with_bypass_coefficient(false)
        .with_flag_array(false)
        .with_solver_option("log", "0")
}

/// Create options for three-objective mixed test problem
fn options_for_three_objectives_mixed() -> augmecon::Options {
    augmecon::Options::new()
        .with_name("three_objective_mixed_model".to_string())
        .with_grid_points(10)
        .with_early_exit(true)
        .with_bypass_coefficient(true)
        .with_flag_array(true)
        .with_solver_option("log", "0")
}

/// Create a two-objective test model
fn two_objective_model() -> MultiObjectiveProblem {
    variables!(
        problem:
           0 <= x1;
           0 <= x2;
    );

    let var_map = HashMap::from([("x1".to_string(), x1), ("x2".to_string(), x2)]);

    let constraints = vec![
        constraint!(x1 <= 20.0),
        constraint!(x2 <= 40.0),
        constraint!(5.0 * x1 + 4.0 * x2 <= 200.0),
    ];

    let objectives = vec![
        (x1.into(), ObjectiveDirection::Maximize),
        (3.0 * x1 + 4.0 * x2, ObjectiveDirection::Maximize),
    ];

    MultiObjectiveProblem {
        variables: problem,
        constraints,
        objectives,
        var_map,
        variable_types: HashMap::new(),
    }
}

/// Create a three-objective test model
fn three_objective_model() -> MultiObjectiveProblem {
    let mut variables = ProblemVariables::new();
    let mut var_map = HashMap::new();

    // Define all variables from test model
    let var_names = [
        "LIGN", "LIGN1", "LIGN2", "OIL", "OIL2", "OIL3", "NG", "NG1", "NG2", "NG3", "RES", "RES1",
        "RES3",
    ];
    for name in &var_names {
        let var = variables.add(variable().min(0));
        var_map.insert((*name).to_string(), var);
    }

    // Define constraints based on test model
    let constraints = vec![
        // Balance constraints
        constraint!(var_map["LIGN"] - var_map["LIGN1"] - var_map["LIGN2"] == 0.0),
        constraint!(var_map["OIL"] - var_map["OIL2"] - var_map["OIL3"] == 0.0),
        constraint!(var_map["NG"] - var_map["NG1"] - var_map["NG2"] - var_map["NG3"] == 0.0),
        constraint!(var_map["RES"] - var_map["RES1"] - var_map["RES3"] == 0.0),
        // Upper bounds
        constraint!(var_map["LIGN"] <= 31000.0),
        constraint!(var_map["OIL"] <= 15000.0),
        constraint!(var_map["NG"] <= 22000.0),
        constraint!(var_map["RES"] <= 10000.0),
        // Demand constraints
        constraint!(var_map["LIGN1"] + var_map["NG1"] + var_map["RES1"] >= 38400.0),
        constraint!(var_map["LIGN2"] + var_map["OIL2"] + var_map["NG2"] >= 19200.0),
        constraint!(var_map["OIL3"] + var_map["NG3"] + var_map["RES3"] >= 6400.0),
    ];

    // Define objectives based on test model (all minimize)
    let objectives = vec![
        (
            30.0 * var_map["LIGN"]
                + 75.0 * var_map["OIL"]
                + 60.0 * var_map["NG"]
                + 90.0 * var_map["RES"],
            ObjectiveDirection::Minimize,
        ),
        (
            1.44 * var_map["LIGN"] + 0.72 * var_map["OIL"] + 0.45 * var_map["NG"],
            ObjectiveDirection::Minimize,
        ),
        (var_map["OIL"] + var_map["NG"], ObjectiveDirection::Minimize),
    ];

    MultiObjectiveProblem {
        variables,
        constraints,
        objectives,
        var_map,
        variable_types: HashMap::new(),
    }
}

/// Create a three-objective mixed test model
fn three_objective_model_mixed() -> MultiObjectiveProblem {
    let mut model = three_objective_model();

    // Change the second objective to maximize (mixed optimization model)
    // Uses: maximize(-1 * (1.44*LIGN + 0.72*OIL + 0.45*NG))
    // Which is equivalent to: minimize(1.44*LIGN + 0.72*OIL + 0.45*NG) with negative sign
    model.objectives[1] = (
        -1.0 * (1.44 * model.var_map["LIGN"]
            + 0.72 * model.var_map["OIL"]
            + 0.45 * model.var_map["NG"]),
        ObjectiveDirection::Maximize,
    );

    model
}

// ============================================================================
// Tests for Two-Objective Model
// ============================================================================

mod test_two_objectives {
    use super::*;

    const MODEL_TYPE: &str = "two_objective_model";

    #[test]
    fn test_model_creation() {
        init_test_logging();
        // Test that we can create the model
        let problem = two_objective_model();

        // Check that we have the expected number of objectives (2)
        assert_eq!(
            problem.objectives.len(),
            2,
            "two_objective_model should have 2 objectives"
        );

        // For mixed objectives, check both maximize and minimize directions
        for (_, direction) in &problem.objectives {
            match direction {
                ObjectiveDirection::Maximize | ObjectiveDirection::Minimize => (), // Both are valid
            }
        }

        // Check that we have some variables
        assert!(!problem.var_map.is_empty(), "Should have variables");

        // Check that we have some constraints
        assert!(!problem.constraints.is_empty(), "Should have constraints");
    }

    #[test]
    fn test_options_creation() {
        init_test_logging();
        // Test that we can create appropriate options
        let options = options_for_two_objectives();

        // Verify that options are created successfully
        assert!(options.grid_points.is_some(), "Should have grid points set");
        assert_eq!(
            options.grid_points.unwrap(),
            10,
            "Should have correct grid points"
        );
        assert_eq!(options.name, MODEL_TYPE, "Should have correct name");
    }

    #[test]
    fn test_payoff_table() {
        init_test_logging();
        // Following the Python test pattern
        let model = two_objective_model();
        let options = options_for_two_objectives();

        let mut solver = Augmecon::try_new(model, options).expect("Failed to create solver");

        // Solve the problem
        solver.solve().expect("Solve should succeed");

        // Expected payoff table from Python test
        let expected_payoff = vec![vec![20.0, 160.0], vec![8.0, 184.0]];

        // Get actual payoff table from solver
        let actual_payoff = solver.get_payoff_table();

        // Debug: Print actual vs expected only if test fails
        if actual_payoff != expected_payoff {
            println!("Expected payoff table: {expected_payoff:?}");
            println!("Actual payoff table: {actual_payoff:?}");
        }

        // Compare with tolerance (2 decimal places as in Python)
        assert!(
            array_equal(actual_payoff, &expected_payoff, 0.01),
            "Payoff table does not match expected values. Expected: {expected_payoff:?}, Actual: {actual_payoff:?}"
        );
    }

    #[test]
    fn test_pareto_solutions() {
        init_test_logging();
        // Following the Python test pattern
        let model = two_objective_model();
        let options = options_for_two_objectives();
        let mut solver = Augmecon::try_new(model, options).expect("Failed to create solver");

        // Solve the problem
        solver.solve().expect("Solve should succeed");

        // Expected Pareto solutions from Python test
        let expected_pareto = vec![
            vec![8.0, 184.0],
            vec![9.33, 181.33],
            vec![10.67, 178.67],
            vec![12.0, 176.0],
            vec![13.33, 173.33],
            vec![14.67, 170.67],
            vec![16.0, 168.0],
            vec![17.33, 165.33],
            vec![18.67, 162.67],
            vec![20.0, 160.0],
        ];

        // Get actual Pareto solutions from solver
        let mut actual_pareto = solver.get_pareto_front().objective_matrix();

        // Sort both arrays by first objective value for consistent comparison
        let mut expected_pareto_sorted = expected_pareto;
        expected_pareto_sorted.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        actual_pareto.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());

        // Compare with tolerance (2 decimal places as in Python)
        assert!(
            array_equal(&actual_pareto, &expected_pareto_sorted, 0.01),
            "Pareto solutions do not match expected values. Expected: {expected_pareto_sorted:?}, Actual: {actual_pareto:?}"
        );
    }
}

// ============================================================================
// Tests for Three-Objective Model
// ============================================================================

mod test_three_objectives {
    use super::*;

    const MODEL_TYPE: &str = "three_objective_model";

    #[test]
    fn test_model_creation() {
        init_test_logging();
        // Test that we can create the model
        let problem = three_objective_model();

        // Check that we have the expected number of objectives (3)
        assert_eq!(
            problem.objectives.len(),
            3,
            "three_objective_model should have 3 objectives"
        );

        // Verify objectives are minimization (this is the three objective model)
        for (_, direction) in &problem.objectives {
            match direction {
                ObjectiveDirection::Minimize => (), // Good
                ObjectiveDirection::Maximize => panic!("Expected minimization objectives"),
            }
        }

        // Check that we have some variables
        assert!(!problem.var_map.is_empty(), "Should have variables");

        // Check that we have the correct constraints (test model has 11 constraints)
        assert_eq!(
            problem.constraints.len(),
            11,
            "Should have 11 constraints as per test model"
        );
    }

    #[test]
    fn test_options_creation() {
        init_test_logging();
        // Test that we can create appropriate options
        let options = options_for_three_objectives();

        // Verify that options are created successfully
        assert!(options.grid_points.is_some(), "Should have grid points set");
        assert_eq!(
            options.grid_points.unwrap(),
            10,
            "Should have correct grid points"
        );
        assert_eq!(options.name, MODEL_TYPE, "Should have correct name");
    }

    #[test]
    fn test_solver_creation_and_solve() {
        init_test_logging();
        // Create the model and options (following Python test pattern)
        let model = three_objective_model();
        let options = options_for_three_objectives();

        let mut solver = Augmecon::try_new(model, options).expect("Failed to create solver");

        // Basic solver state checks
        assert!(
            solver.get_payoff_table().is_empty(),
            "Payoff table should be empty initially"
        );
        assert!(
            solver.get_pareto_front().is_empty(),
            "Pareto front should be empty initially"
        );

        // Test that solve works and finds solutions
        let result = solver.solve();
        assert!(result.is_ok(), "solve() should succeed: {result:?}");

        // Test that we found some Pareto-optimal solutions
        let pareto_front = solver.get_pareto_front();
        assert!(
            !pareto_front.is_empty(),
            "Should find at least one Pareto-optimal solution"
        );

        println!("Found {} Pareto-optimal solutions", pareto_front.len());
    }

    #[test]
    fn test_payoff_table() {
        init_test_logging();
        // Following the Python test pattern
        let model = three_objective_model();
        let options = options_for_three_objectives();
        let mut solver = Augmecon::try_new(model, options).expect("Failed to create solver");

        // Solve the problem
        solver.solve().expect("Solve should succeed");

        // Expected payoff table from Python test
        let expected_payoff = vec![
            vec![3_075_000.0, 62460.0, 33000.0],
            vec![3_855_000.0, 45180.0, 37000.0],
            vec![3_225_000.0, 55260.0, 23000.0],
        ];

        // Get actual payoff table from solver
        let actual_payoff = solver.get_payoff_table();

        // Compare with tolerance (2 decimal places as in Python)
        assert!(
            array_equal(actual_payoff, &expected_payoff, 0.01),
            "Payoff table does not match expected values"
        );
    }

    #[test]
    fn test_pareto_solutions() {
        init_test_logging();
        // Following the Python test pattern
        let model = three_objective_model();
        let options = options_for_three_objectives();
        let mut solver = Augmecon::try_new(model, options).expect("Failed to create solver");

        // Solve the problem
        solver.solve().expect("Solve should succeed");

        // Expected Pareto solutions from Python test
        let expected_pareto = vec![
            vec![3_075_000.0, 62460.0, 33000.0],
            vec![3_085_000.0, 61980.0, 32333.33],
            vec![3_108_333.33, 60860.0, 30777.78],
            vec![3_115_000.0, 60540.0, 30333.33],
            vec![3_131_666.67, 59740.0, 29222.22],
            vec![3_155_000.0, 58620.0, 27666.67],
            vec![3_178_333.33, 57500.0, 26111.11],
            vec![3_195_000.0, 56700.0, 25000.0],
            vec![3_201_666.67, 56380.0, 24555.56],
            vec![3_225_000.0, 55260.0, 23000.0],
            vec![3_255_000.0, 54780.0, 23666.67],
            vec![3_375_000.0, 52860.0, 26333.33],
            vec![3_495_000.0, 50940.0, 29000.0],
            vec![3_615_000.0, 49020.0, 31666.67],
            vec![3_735_000.0, 47100.0, 34333.33],
            vec![3_855_000.0, 45180.0, 37000.0],
        ];

        // Get actual Pareto solutions from solver
        let mut actual_pareto = solver.get_pareto_front().objective_matrix();

        // Sort both arrays by first objective value for consistent comparison
        let mut expected_pareto_sorted = expected_pareto;
        expected_pareto_sorted.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        actual_pareto.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());

        // Debug: Print actual vs expected
        println!(
            "Expected {} Pareto solutions: {:?}",
            expected_pareto_sorted.len(),
            expected_pareto_sorted
        );
        println!(
            "Actual {} Pareto solutions: {:?}",
            actual_pareto.len(),
            actual_pareto
        );

        // Compare with tolerance (2 decimal places as in Python)
        assert!(
            array_equal(&actual_pareto, &expected_pareto_sorted, 0.01),
            "Pareto solutions do not match expected values. Expected: {} solutions, Got: {} solutions", 
            expected_pareto_sorted.len(), actual_pareto.len()
        );
    }
}

// ============================================================================
// Tests for Three-Objective Mixed Model
// ============================================================================

mod test_three_objectives_mixed {
    use super::*;

    const MODEL_TYPE: &str = "three_objective_mixed_model";

    #[test]
    fn test_model_creation() {
        init_test_logging();
        // Test that we can create the model
        let problem = three_objective_model_mixed();

        // Check that we have the expected number of objectives (3)
        assert_eq!(
            problem.objectives.len(),
            3,
            "three_objective_mixed_model should have 3 objectives"
        );

        // This model has mixed objectives (minimize, maximize, minimize)
        match problem.objectives[0].1 {
            ObjectiveDirection::Minimize => (), // Good
            ObjectiveDirection::Maximize => panic!("Expected first objective to be minimization"),
        }
        match problem.objectives[1].1 {
            ObjectiveDirection::Maximize => (), // Good
            ObjectiveDirection::Minimize => panic!("Expected second objective to be maximization"),
        }
        match problem.objectives[2].1 {
            ObjectiveDirection::Minimize => (), // Good
            ObjectiveDirection::Maximize => panic!("Expected third objective to be minimization"),
        }

        // Check that we have some variables
        assert!(!problem.var_map.is_empty(), "Should have variables");

        // Check that we have the correct constraints (test model has 11 constraints)
        assert_eq!(
            problem.constraints.len(),
            11,
            "Should have 11 constraints as per test model"
        );
    }

    #[test]
    fn test_options_creation() {
        init_test_logging();
        // Test that we can create appropriate options
        let options = options_for_three_objectives_mixed();

        // Verify that options are created successfully
        assert!(options.grid_points.is_some(), "Should have grid points set");
        assert_eq!(
            options.grid_points.unwrap(),
            10,
            "Should have correct grid points"
        );
        assert_eq!(options.name, MODEL_TYPE, "Should have correct name");
    }

    #[test]
    fn test_solver_creation_and_solve() {
        init_test_logging();
        // Create the model and options (following Python test pattern)
        let model = three_objective_model_mixed();
        let options = options_for_three_objectives_mixed();
        let mut solver = Augmecon::try_new(model, options).expect("Failed to create solver");

        // Basic solver state checks
        assert!(
            solver.get_payoff_table().is_empty(),
            "Payoff table should be empty initially"
        );
        assert!(
            solver.get_pareto_front().is_empty(),
            "Pareto front should be empty initially"
        );

        // Test that solve works and finds solutions
        let result = solver.solve();
        assert!(result.is_ok(), "solve() should succeed: {result:?}");

        // Test that we found some Pareto-optimal solutions
        let pareto_front = solver.get_pareto_front();
        assert!(
            !pareto_front.is_empty(),
            "Should find at least one Pareto-optimal solution"
        );

        println!("Found {} Pareto-optimal solutions", pareto_front.len());
    }

    #[test]
    fn test_payoff_table() {
        init_test_logging();
        // Following the Python test pattern
        let model = three_objective_model_mixed();
        let options = options_for_three_objectives_mixed();
        let mut solver = Augmecon::try_new(model, options).expect("Failed to create solver");

        // Solve the problem
        solver.solve().expect("Solve should succeed");

        // Expected payoff table from Python test (note negative values for minimization)
        let expected_payoff = vec![
            vec![3_075_000.0, -62460.0, 33000.0],
            vec![3_855_000.0, -45180.0, 37000.0],
            vec![3_225_000.0, -55260.0, 23000.0],
        ];

        // Get actual payoff table from solver
        let actual_payoff = solver.get_payoff_table();

        // Compare with tolerance (2 decimal places as in Python)
        assert!(
            array_equal(actual_payoff, &expected_payoff, 0.01),
            "Payoff table does not match expected values"
        );
    }

    #[test]
    fn test_pareto_solutions() {
        init_test_logging();
        // Following the Python test pattern
        let model = three_objective_model_mixed();
        let options = options_for_three_objectives_mixed();
        let mut solver = Augmecon::try_new(model, options).expect("Failed to create solver");

        // Solve the problem
        solver.solve().expect("Solve should succeed");

        // Expected Pareto solutions from Python test (note negative values for minimization)
        let expected_pareto = vec![
            vec![3_075_000.0, -62460.0, 33000.0],
            vec![3_085_000.0, -61980.0, 32333.33],
            vec![3_108_333.33, -60860.0, 30777.78],
            vec![3_115_000.0, -60540.0, 30333.33],
            vec![3_131_666.67, -59740.0, 29222.22],
            vec![3_155_000.0, -58620.0, 27666.67],
            vec![3_178_333.33, -57500.0, 26111.11],
            vec![3_195_000.0, -56700.0, 25000.0],
            vec![3_201_666.67, -56380.0, 24555.56],
            vec![3_225_000.0, -55260.0, 23000.0],
            vec![3_255_000.0, -54780.0, 23666.67],
            vec![3_375_000.0, -52860.0, 26333.33],
            vec![3_495_000.0, -50940.0, 29000.0],
            vec![3_615_000.0, -49020.0, 31666.67],
            vec![3_735_000.0, -47100.0, 34333.33],
            vec![3_855_000.0, -45180.0, 37000.0],
        ];

        // Get actual Pareto solutions from solver
        let mut actual_pareto = solver.get_pareto_front().objective_matrix();

        // Sort both arrays by first objective value for consistent comparison
        let mut expected_pareto_sorted = expected_pareto;
        expected_pareto_sorted.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        actual_pareto.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());

        // Debug: Print actual vs expected
        println!(
            "Expected {} Pareto solutions: {:?}",
            expected_pareto_sorted.len(),
            expected_pareto_sorted
        );
        println!(
            "Actual {} Pareto solutions: {:?}",
            actual_pareto.len(),
            actual_pareto
        );

        // Compare with tolerance (2 decimal places as in Python)
        assert!(
            array_equal(&actual_pareto, &expected_pareto_sorted, 0.01),
            "Pareto solutions do not match expected values. Expected: {} solutions, Got: {} solutions", 
            expected_pareto_sorted.len(), actual_pareto.len()
        );
    }
}
