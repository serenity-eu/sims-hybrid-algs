//! Integration tests for 4KP (4-objective knapsack) problems
//! Tests both 4KP40 and 4KP50 variants

use augmecon::{Augmecon, MultiObjectiveProblem, ObjectiveDirection, VariableType};
use calamine::{open_workbook, Data, Reader, Xlsx};
use good_lp::{constraint, Expression};

// ============================================================================
// Helper Functions (specific to 4KP tests)
// ============================================================================

/// Initialize test logging
fn init_test_logging() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        env_logger::init();
    });
}

/// Helper to convert `calamine::Data` to f64
fn cell_to_float(c: &Data) -> f64 {
    match c {
        Data::Float(f) => *f,
        Data::Int(i) => {
            #[expect(
                clippy::cast_precision_loss,
                reason = "Converting test data from i64 to f64 - precision loss acceptable for test verification"
            )]
            let f = *i as f64;
            f
        }
        Data::String(s) => s
            .trim()
            .parse::<f64>()
            .unwrap_or_else(|_| panic!("Cannot convert string '{s}' to f64")),
        Data::Empty => 0.0, // Handle empty cells as 0.0
        Data::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        _ => panic!("Cannot convert cell {c:?} to f64"),
    }
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

/// Read Excel data for testing
fn read_excel(file_path: &str, sheet_name: &str) -> Vec<Vec<f64>> {
    let mut workbook: Xlsx<_> = open_workbook(file_path).expect("Failed to open workbook");
    let range = workbook
        .worksheet_range(sheet_name)
        .expect("Failed to get worksheet");

    let mut data = Vec::new();
    for row in range.rows() {
        let mut row_data = Vec::new();
        for cell in row {
            row_data.push(cell_to_float(cell));
        }
        if !row_data.is_empty() {
            data.push(row_data);
        }
    }

    data
}

/// Create options for 4KP40 test problem
fn options_for_4kp40() -> augmecon::Options {
    augmecon::Options::new()
        .with_name("4kp40".to_string())
        .with_grid_points(141)
        .with_nadir_points(vec![138.0, 106.0, 121.0])
        .with_solver_option("log", "0")
}

/// Create options for 4KP50 test problem
fn options_for_4kp50() -> augmecon::Options {
    augmecon::Options::new()
        .with_name("4kp50".to_string())
        .with_grid_points(53)
        .with_nadir_points(vec![718.0, 717.0, 705.0])
        .with_solver_option("log", "0")
}

/// Create a 4-objective knapsack problem model
fn four_kp_model(model_name: &str) -> MultiObjectiveProblem {
    let file_path = format!("tests/input/knapsack/{model_name}.xlsx");
    let mut workbook: Xlsx<_> = open_workbook(&file_path).expect("Cannot open file");

    let a_range = workbook.worksheet_range("a").expect("Cannot find sheet a");
    let b_range = workbook.worksheet_range("b").expect("Cannot find sheet b");
    let c_range = workbook.worksheet_range("c").expect("Cannot find sheet c");

    let weights: Vec<Vec<f64>> = a_range
        .rows()
        .enumerate()
        .filter_map(|(row_idx, row)| {
            // Skip the first row (column indices)
            if row_idx == 0 {
                return None;
            }

            // Skip the first column (row indices) and convert the rest
            let row_data: Vec<f64> = row
                .iter()
                .skip(1) // Skip first column
                .map(cell_to_float)
                .collect();

            Some(row_data)
        })
        .collect();
    let capacities: Vec<f64> = b_range
        .rows()
        .enumerate()
        .filter_map(|(row_idx, row)| {
            // Skip the first row (column indices)
            if row_idx == 0 {
                return None;
            }

            // Skip the first column (row indices) and convert the rest
            row.iter()
                .skip(1) // Skip first column
                .map(cell_to_float)
                .next() // Take only the first value (capacity)
        })
        .collect();
    let profits: Vec<Vec<f64>> = c_range
        .rows()
        .enumerate()
        .filter_map(|(row_idx, row)| {
            // Skip the first row (column indices)
            if row_idx == 0 {
                return None;
            }

            // Skip the first column (row indices) and convert the rest
            let row_data: Vec<f64> = row
                .iter()
                .skip(1) // Skip first column
                .map(cell_to_float)
                .collect();

            Some(row_data)
        })
        .collect();

    let n_items = profits[0].len();

    let mut problem = MultiObjectiveProblem::new();

    // Add binary variables for each item
    for i in 0..n_items {
        problem.add_variable(format!("x{i}"), VariableType::Binary);
    }

    // Create constraints using the weights and capacities
    for (weight_row, capacity) in weights.iter().zip(capacities.iter()) {
        let mut total_weight = Expression::from(0.0);
        for (i, &weight_val) in weight_row.iter().enumerate().take(n_items) {
            if let Some(&var) = problem.var_map.get(&format!("x{i}")) {
                total_weight += weight_val * var;
            }
        }
        problem.add_constraint(constraint!(total_weight <= *capacity));
    }

    // Add objectives
    for profit in profits.iter().take(4) {
        let mut obj_expr = Expression::from(0.0);
        for (i, &coeff) in profit.iter().enumerate().take(n_items) {
            if let Some(&var) = problem.var_map.get(&format!("x{i}")) {
                obj_expr += coeff * var;
            }
        }
        problem.add_objective(obj_expr, ObjectiveDirection::Maximize);
    }

    problem
}

// ============================================================================
// Tests for 4KP40
// ============================================================================

mod test_4kp40 {
    use super::*;

    const MODEL_TYPE: &str = "4kp40";

    #[test]
    fn test_model_creation() {
        init_test_logging();
        // Test that we can create the model
        let problem = four_kp_model(MODEL_TYPE);

        // Check that we have the expected number of objectives (4 for 4kp problems)
        assert_eq!(
            problem.objectives.len(),
            4,
            "4kp40 should have 4 objectives"
        );

        // Verify objectives are maximization (standard for knapsack problems)
        for (_, direction) in &problem.objectives {
            match direction {
                ObjectiveDirection::Maximize => (), // Good
                ObjectiveDirection::Minimize => panic!("Expected maximization objectives"),
            }
        }

        // Check that we have some variables
        assert!(!problem.var_map.is_empty(), "Should have variables");

        // Check that we have the right number of constraints (4 for 4kp problems)
        assert_eq!(
            problem.constraints.len(),
            4,
            "4kp40 should have 4 constraints"
        );
    }

    #[test]
    fn test_options_creation() {
        init_test_logging();
        // Test that we can create appropriate options
        let options = options_for_4kp40();

        // Verify that options are created successfully
        assert!(options.grid_points.is_some(), "Should have grid points set");
        assert_eq!(
            options.grid_points.unwrap(),
            141,
            "Should have correct grid points"
        );
        assert_eq!(options.name, MODEL_TYPE, "Should have correct name");

        // Check nadir points
        assert!(
            options.nadir_points.is_some(),
            "Should have nadir points set"
        );
        let nadir = options.nadir_points.unwrap();
        assert_eq!(
            nadir.len(),
            3,
            "Should have 3 nadir points for 4 objectives"
        );
        assert!(
            float_equal(nadir[0], 138.0, 0.01),
            "First nadir point should be 138.0, got {}",
            nadir[0]
        );
        assert!(
            float_equal(nadir[1], 106.0, 0.01),
            "Second nadir point should be 106.0, got {}",
            nadir[1]
        );
        assert!(
            float_equal(nadir[2], 121.0, 0.01),
            "Third nadir point should be 121.0, got {}",
            nadir[2]
        );
    }

    #[test]
    fn test_payoff_table() {
        init_test_logging();
        // Following the Python test pattern
        let model = four_kp_model(MODEL_TYPE);
        let options = options_for_4kp40();
        let mut solver = Augmecon::try_new(model, options).expect("Failed to create solver");

        // Solve the problem
        solver.solve().expect("Solve should succeed");

        // Read expected payoff table from Excel
        let file_path = format!("tests/input/knapsack/{MODEL_TYPE}.xlsx");
        let expected_payoff = read_excel(&file_path, "payoff_table");

        // Get actual payoff table from solver
        let actual_payoff = solver.get_payoff_table();

        // Compare with tolerance (2 decimal places as in Python)
        assert!(
            array_equal(actual_payoff, &expected_payoff, 0.01),
            "Payoff table does not match expected values"
        );
    }

    #[test]
    fn test_pareto_sols() {
        init_test_logging();
        // Following the Python test pattern
        let model = four_kp_model(MODEL_TYPE);
        let options = options_for_4kp40();
        let mut solver = Augmecon::try_new(model, options).expect("Failed to create solver");

        // Solve the problem
        solver.solve().expect("Solve should succeed");

        // Read expected Pareto solutions from Excel
        let file_path = format!("tests/input/knapsack/{MODEL_TYPE}.xlsx");
        let expected_pareto = read_excel(&file_path, "pareto_sols");

        // Get actual Pareto solutions from solver
        let actual_pareto = solver.get_pareto_front().objective_matrix();

        // Compare with tolerance (2 decimal places as in Python)
        assert!(
            array_equal(&actual_pareto, &expected_pareto, 0.01),
            "Pareto solutions do not match expected values"
        );
    }
}

// ============================================================================
// Tests for 4KP50
// ============================================================================

mod test_4kp50 {
    use super::*;

    const MODEL_TYPE: &str = "4kp50";

    #[test]
    fn test_model_creation() {
        init_test_logging();
        // Test that we can create the model
        let problem = four_kp_model(MODEL_TYPE);

        // Check that we have the expected number of objectives (4 for 4kp problems)
        assert_eq!(
            problem.objectives.len(),
            4,
            "4kp50 should have 4 objectives"
        );

        // Verify objectives are maximization (standard for knapsack problems)
        for (_, direction) in &problem.objectives {
            match direction {
                ObjectiveDirection::Maximize => (), // Good
                ObjectiveDirection::Minimize => panic!("Expected maximization objectives"),
            }
        }

        // Check that we have some variables
        assert!(!problem.var_map.is_empty(), "Should have variables");

        // Check that we have the right number of constraints (4 for 4kp problems)
        assert_eq!(
            problem.constraints.len(),
            4,
            "4kp50 should have 4 constraints"
        );
    }

    #[test]
    fn test_options_creation() {
        init_test_logging();
        // Test that we can create appropriate options
        let options = options_for_4kp50();

        // Verify that options are created successfully
        assert!(options.grid_points.is_some(), "Should have grid points set");
        assert_eq!(
            options.grid_points.unwrap(),
            53,
            "Should have correct grid points"
        );
        assert_eq!(options.name, MODEL_TYPE, "Should have correct name");

        // Check nadir points
        assert!(
            options.nadir_points.is_some(),
            "Should have nadir points set"
        );
        let nadir = options.nadir_points.unwrap();
        assert_eq!(
            nadir.len(),
            3,
            "Should have 3 nadir points for 4 objectives"
        );
        assert!(
            float_equal(nadir[0], 718.0, 0.01),
            "First nadir point should be 718.0, got {}",
            nadir[0]
        );
        assert!(
            float_equal(nadir[1], 717.0, 0.01),
            "Second nadir point should be 717.0, got {}",
            nadir[1]
        );
        assert!(
            float_equal(nadir[2], 705.0, 0.01),
            "Third nadir point should be 705.0, got {}",
            nadir[2]
        );
    }

    #[test]
    fn test_payoff_table() {
        init_test_logging();
        // Following the Python test pattern
        let model = four_kp_model(MODEL_TYPE);
        let options = options_for_4kp50();
        let mut solver = Augmecon::try_new(model, options).expect("Failed to create solver");

        // Solve the problem
        solver.solve().expect("Solve should succeed");

        // Read expected payoff table from Excel
        let file_path = format!("tests/input/knapsack/{MODEL_TYPE}.xlsx");
        let expected_payoff = read_excel(&file_path, "payoff_table");

        // Get actual payoff table from solver
        let actual_payoff = solver.get_payoff_table();

        // Compare with tolerance (2 decimal places as in Python)
        assert!(
            array_equal(actual_payoff, &expected_payoff, 0.01),
            "Payoff table does not match expected values"
        );
    }

    #[test]
    fn test_pareto_sols() {
        init_test_logging();
        // Following the Python test pattern
        let model = four_kp_model(MODEL_TYPE);
        let options = options_for_4kp50();
        let mut solver = Augmecon::try_new(model, options).expect("Failed to create solver");

        // Solve the problem
        solver.solve().expect("Solve should succeed");

        // Read expected Pareto solutions from Excel
        let file_path = format!("tests/input/knapsack/{MODEL_TYPE}.xlsx");
        let expected_pareto = read_excel(&file_path, "pareto_sols");

        // Get actual Pareto solutions from solver
        let actual_pareto = solver.get_pareto_front().objective_matrix();

        // Compare with tolerance (2 decimal places as in Python)
        assert!(
            array_equal(&actual_pareto, &expected_pareto, 0.01),
            "Pareto solutions do not match expected values"
        );
    }
}
