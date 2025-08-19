//! Integration tests for 2KP (two-objective knapsack) problems
//!
//! This module contains tests for 2KP50, 2KP100, and 2KP250 variants

#![allow(clippy::needless_borrow)]

use augmecon::{Augmecon, MultiObjectiveProblem, ObjectiveDirection, VariableType};
use calamine::{open_workbook, Data, Reader};
use good_lp::{constraint, Expression};
use std::sync::Once;

static INIT: Once = Once::new();

/// Initialize logging for tests with debug level
fn init_test_logging() {
    INIT.call_once(|| {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
            .is_test(true)
            .try_init()
            .ok();
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
        Data::Empty => 0.0,
        Data::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        Data::DateTime(dt) => dt.as_f64(),
        Data::DateTimeIso(dt_iso) => dt_iso.parse::<f64>().unwrap_or(0.0),
        Data::DurationIso(dur_iso) => dur_iso.parse::<f64>().unwrap_or(0.0),
        Data::Error(err) => panic!("Error in cell data: {err:?}"),
    }
}

/// Read Excel file and return 2D vector of f64 values
fn read_excel(file_path: &str, sheet_name: &str) -> Vec<Vec<f64>> {
    println!("DEBUG: Reading Excel file '{file_path}', sheet '{sheet_name}'");
    let mut workbook: calamine::Xlsx<_> =
        open_workbook(file_path).unwrap_or_else(|_| panic!("Cannot open file {file_path}"));

    let range = workbook
        .worksheet_range(sheet_name)
        .unwrap_or_else(|_| panic!("Cannot find sheet {sheet_name}"));

    println!(
        "DEBUG: Sheet '{}' has {} rows, {} cols",
        sheet_name,
        range.height(),
        range.width()
    );

    let mut result = Vec::new();
    for (row_idx, row) in range.rows().skip(1).enumerate() {
        let mut row_data = Vec::new();
        for (col_idx, cell) in row.iter().skip(1).enumerate() {
            let val = cell_to_float(cell);
            row_data.push(val);
            if row_idx == 0 && col_idx < 5 {
                // Print first few values for debugging
                println!("DEBUG: Cell [{row_idx}][{col_idx}] = {cell:?} -> {val}");
            }
        }
        result.push(row_data);
    }
    println!(
        "DEBUG: Read {} rows from sheet '{}'",
        result.len(),
        sheet_name
    );
    result
}

/// Compare two 2D arrays of floats with tolerance
fn array_equal(a: &[Vec<f64>], b: &[Vec<f64>], tolerance: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (row_a, row_b) in a.iter().zip(b.iter()) {
        if row_a.len() != row_b.len() {
            return false;
        }
        for (val_a, val_b) in row_a.iter().zip(row_b.iter()) {
            if (val_a - val_b).abs() > tolerance {
                return false;
            }
        }
    }
    true
}

/// Create a 2KP model based on the problem name, reading data from Excel files
fn create_2kp_model(model_name: &str) -> MultiObjectiveProblem {
    println!("DEBUG: Creating 2KP model for {model_name}");
    // Read data from Excel file like the reference implementation does
    let excel_path = format!("tests/input/knapsack/{model_name}.xlsx");
    println!("DEBUG: Reading Excel file: {excel_path}");

    // Read weights matrix (a), capacities vector (b), and profits matrix (c)
    let weights = read_excel(&excel_path, "a"); // a[constraint][item]
    println!(
        "DEBUG: Weights matrix shape: {} x {}",
        weights.len(),
        weights.first().map_or(0, Vec::len)
    );
    println!("DEBUG: Weights: {weights:?}");

    let capacities_data = read_excel(&excel_path, "b"); // b[constraint][0]
    let capacities: Vec<f64> = capacities_data.iter().map(|row| row[0]).collect(); // Extract first column from each row
    println!("DEBUG: Capacities: {capacities:?}");

    let profits = read_excel(&excel_path, "c"); // c[objective][item]
    println!(
        "DEBUG: Profits matrix shape: {} x {}",
        profits.len(),
        profits.first().map_or(0, Vec::len)
    );
    println!("DEBUG: Profits: {profits:?}");

    // Determine number of items from the data
    let num_items = weights[0].len();
    println!("DEBUG: Number of items: {num_items}");

    assert!(
        num_items != 0,
        "No items found in Excel data for {model_name}"
    );

    // Create the problem
    let mut problem = MultiObjectiveProblem::new();
    println!("DEBUG: Created new MultiObjectiveProblem");

    // Add binary variables for each item
    for i in 0..num_items {
        problem.add_variable(format!("x{i}"), VariableType::Binary);
    }
    println!("DEBUG: Added {num_items} binary variables");

    // Add capacity constraints
    for (constraint_idx, &capacity) in capacities.iter().enumerate() {
        println!("DEBUG: Adding constraint {constraint_idx} with capacity {capacity}");
        let mut constraint_expr = Expression::from(0.0);
        // sum(a[constraint_idx][i] * model.x[i] for i in model.ITEMS) <= b[constraint_idx][0]
        for item_idx in 0..num_items {
            let weight = weights[constraint_idx][item_idx];
            if let Some(&var) = problem.var_map.get(&format!("x{item_idx}")) {
                constraint_expr += weight * var;
            }
        }
        problem.add_constraint(constraint!(constraint_expr <= capacity));
    }
    println!("DEBUG: Added {} constraints total", capacities.len());

    // Add objectives (both maximization for knapsack)
    for obj_idx in 0..profits.len() {
        println!(
            "DEBUG: Adding objective {} of {}",
            obj_idx + 1,
            profits.len()
        );
        // Ensure we only take 2 objectives for 2KP
        let mut objective_expr = Expression::from(0.0);
        for item_idx in 0..num_items {
            let profit = profits[obj_idx][item_idx];
            if let Some(&var) = problem.var_map.get(&format!("x{item_idx}")) {
                objective_expr += profit * var;
            }
        }
        problem.add_objective(objective_expr, ObjectiveDirection::Maximize);
    }
    println!("DEBUG: Added {} objectives total", profits.len());

    println!(
        "DEBUG: Final problem has {} variables, {} constraints, {} objectives",
        problem.var_map.len(),
        problem.constraints.len(),
        problem.objectives.len()
    );
    problem
}

/// Create options for specific 2KP problems
fn create_2kp_options(model_name: &str) -> augmecon::Options {
    match model_name {
        "2kp50" => augmecon::Options::new()
            .with_name("2kp50".to_string())
            .with_grid_points(492)
            .with_solver_option("log", "0"),
        "2kp100" => augmecon::Options::new()
            .with_name("2kp100".to_string())
            .with_grid_points(2535)
            .with_solver_option("log", "0"),
        "2kp250" => augmecon::Options::new()
            .with_name("2kp250".to_string())
            .with_grid_points(2534)
            .with_solver_option("log", "0"),
        _ => panic!("Unknown 2KP model: {model_name}"),
    }
}

// Test functions for 2KP50
mod test_2kp50 {
    use super::*;

    const MODEL_TYPE: &str = "2kp50";

    #[test]
    fn test_model_creation() {
        init_test_logging();
        let problem = create_2kp_model(MODEL_TYPE);

        assert_eq!(
            problem.objectives.len(),
            2,
            "2kp50 should have 2 objectives"
        );

        // Verify objectives are maximization
        for (_, direction) in &problem.objectives {
            match direction {
                ObjectiveDirection::Maximize => (),
                ObjectiveDirection::Minimize => panic!("Expected maximization objectives"),
            }
        }

        assert!(!problem.var_map.is_empty(), "Should have variables");
        assert_eq!(
            problem.constraints.len(),
            2,
            "2kp50 should have 2 constraints"
        );
    }

    #[test]
    fn test_options_creation() {
        init_test_logging();
        let options = create_2kp_options(MODEL_TYPE);

        assert!(options.grid_points.is_some(), "Should have grid points set");
        assert_eq!(
            options.grid_points.unwrap(),
            492,
            "Should have correct grid points"
        );
        assert_eq!(options.name, MODEL_TYPE, "Should have correct name");
    }

    #[test]
    fn test_problem_solving() {
        init_test_logging();
        let problem = create_2kp_model(MODEL_TYPE);
        let options = create_2kp_options(MODEL_TYPE);

        let mut solver = Augmecon::try_new(problem, options).expect("Failed to create solver");
        let result = solver.solve();
        assert!(result.is_ok(), "solve() should succeed: {result:?}");

        let pareto_solutions = solver.get_pareto_solutions();
        assert!(
            !pareto_solutions.is_empty(),
            "Should find some Pareto solutions"
        );
    }

    #[test]
    fn test_payoff_table() {
        init_test_logging();
        let problem = create_2kp_model(MODEL_TYPE);
        let options = create_2kp_options(MODEL_TYPE);

        let mut solver = Augmecon::try_new(problem, options).expect("Failed to create solver");
        solver.solve().expect("Failed to solve");
        let payoff_table = solver.get_payoff_table();

        assert_eq!(payoff_table.len(), 2, "Should have 2 rows in payoff table");
        for row in payoff_table {
            assert_eq!(row.len(), 2, "Each row should have 2 columns");
        }

        // If we have reference data, we can test against it
        let expected_payoff = read_excel(
            &format!("tests/input/knapsack/{MODEL_TYPE}.xlsx"),
            "payoff_table",
        );
        assert!(
            array_equal(&payoff_table, &expected_payoff, 0.01),
            "Payoff table should match expected values"
        );
    }

    #[test]
    fn test_pareto_solutions() {
        init_test_logging();
        let problem = create_2kp_model(MODEL_TYPE);
        let options = create_2kp_options(MODEL_TYPE);

        let mut solver = Augmecon::try_new(problem, options).expect("Failed to create solver");
        let result = solver.solve();
        assert!(result.is_ok(), "solve() should succeed");

        let actual_pareto = solver.get_pareto_solutions();
        assert!(!actual_pareto.is_empty(), "Should have Pareto solutions");

        // Convert to the format expected by array_equal
        let mut actual_pareto: Vec<Vec<f64>> = actual_pareto
            .iter()
            .map(|sol| sol.objective_values.clone())
            .collect();

        // If we have reference data, test against it
        let mut expected_pareto = read_excel(
            &format!("tests/input/knapsack/{MODEL_TYPE}.xlsx"),
            "pareto_sols",
        );

        // Sort both arrays to ensure consistent comparison since Pareto solution order doesn't matter
        actual_pareto.sort_by(|a, b| {
            for (val_a, val_b) in a.iter().zip(b.iter()) {
                let cmp = val_a
                    .partial_cmp(val_b)
                    .unwrap_or(std::cmp::Ordering::Equal);
                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
            }
            std::cmp::Ordering::Equal
        });
        expected_pareto.sort_by(|a, b| {
            for (val_a, val_b) in a.iter().zip(b.iter()) {
                let cmp = val_a
                    .partial_cmp(val_b)
                    .unwrap_or(std::cmp::Ordering::Equal);
                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
            }
            std::cmp::Ordering::Equal
        });

        assert!(
            array_equal(&actual_pareto, &expected_pareto, 0.01),
            "Pareto solutions should match expected values"
        );
    }
}

// Test functions for 2KP100
mod test_2kp100 {
    use super::*;

    const MODEL_TYPE: &str = "2kp100";

    #[test]
    fn test_model_creation() {
        init_test_logging();
        let problem = create_2kp_model(MODEL_TYPE);

        assert_eq!(
            problem.objectives.len(),
            2,
            "2kp100 should have 2 objectives"
        );
        assert!(!problem.var_map.is_empty(), "Should have variables");
        assert_eq!(
            problem.constraints.len(),
            2,
            "2kp100 should have 2 constraints"
        );
    }

    #[test]
    fn test_options_creation() {
        init_test_logging();
        let options = create_2kp_options(MODEL_TYPE);

        assert!(options.grid_points.is_some(), "Should have grid points set");
        assert_eq!(
            options.grid_points.unwrap(),
            2535,
            "Should have correct grid points"
        );
        assert_eq!(options.name, MODEL_TYPE, "Should have correct name");
    }

    #[test]
    fn test_problem_solving() {
        init_test_logging();
        let problem = create_2kp_model(MODEL_TYPE);
        let options = create_2kp_options(MODEL_TYPE);

        let mut solver = Augmecon::try_new(problem, options).expect("Failed to create solver");
        let result = solver.solve();
        assert!(result.is_ok(), "solve() should succeed: {result:?}");

        let pareto_solutions = solver.get_pareto_solutions();
        assert!(
            !pareto_solutions.is_empty(),
            "Should find some Pareto solutions"
        );
    }

    #[test]
    fn test_payoff_table() {
        init_test_logging();
        let problem = create_2kp_model(MODEL_TYPE);
        let options = create_2kp_options(MODEL_TYPE);

        let mut solver = Augmecon::try_new(problem, options).expect("Failed to create solver");
        solver.solve().expect("Failed to solve");
        let payoff_table = solver.get_payoff_table();

        assert_eq!(payoff_table.len(), 2, "Should have 2 rows in payoff table");
        for row in payoff_table {
            assert_eq!(row.len(), 2, "Each row should have 2 columns");
        }

        let file_path = format!("tests/input/knapsack/{MODEL_TYPE}.xlsx");
        if std::path::Path::new(&file_path).exists() {
            let expected_payoff = read_excel(&file_path, "payoff_table");
            assert!(
                array_equal(&payoff_table, &expected_payoff, 0.01),
                "Payoff table should match expected values"
            );
        }
    }

    #[test]
    fn test_pareto_solutions() {
        init_test_logging();
        let problem = create_2kp_model(MODEL_TYPE);
        let options = create_2kp_options(MODEL_TYPE);

        let mut solver = Augmecon::try_new(problem, options).expect("Failed to create solver");
        let result = solver.solve();
        assert!(result.is_ok(), "solve() should succeed");

        let actual_pareto = solver.get_pareto_solutions();
        let mut actual_pareto: Vec<Vec<f64>> = actual_pareto
            .iter()
            .map(|sol| sol.objective_values.clone())
            .collect();

        let file_path = format!("tests/input/knapsack/{MODEL_TYPE}.xlsx");
        if std::path::Path::new(&file_path).exists() {
            let mut expected_pareto = read_excel(&file_path, "pareto_sols");

            // Sort both arrays for consistent comparison
            actual_pareto.sort_by(|a, b| {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                    .find(|&ord| ord != std::cmp::Ordering::Equal)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            expected_pareto.sort_by(|a, b| {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                    .find(|&ord| ord != std::cmp::Ordering::Equal)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            assert!(
                array_equal(&actual_pareto, &expected_pareto, 0.01),
                "Pareto solutions should match expected values"
            );
        }
    }
}

// Test functions for 2KP250
mod test_2kp250 {
    use super::*;

    const MODEL_TYPE: &str = "2kp250";

    #[test]
    fn test_model_creation() {
        init_test_logging();
        let problem = create_2kp_model(MODEL_TYPE);

        assert_eq!(
            problem.objectives.len(),
            2,
            "2kp250 should have 2 objectives"
        );
        assert!(!problem.var_map.is_empty(), "Should have variables");
        assert_eq!(
            problem.constraints.len(),
            2,
            "2kp250 should have 2 constraints"
        );
    }

    #[test]
    fn test_options_creation() {
        init_test_logging();
        let options = create_2kp_options(MODEL_TYPE);

        assert!(options.grid_points.is_some(), "Should have grid points set");
        assert_eq!(
            options.grid_points.unwrap(),
            2534,
            "Should have correct grid points"
        );
        assert_eq!(options.name, MODEL_TYPE, "Should have correct name");
    }

    #[test]
    fn test_problem_solving() {
        init_test_logging();
        let problem = create_2kp_model(MODEL_TYPE);
        let options = create_2kp_options(MODEL_TYPE);

        let mut solver = Augmecon::try_new(problem, options).expect("Failed to create solver");
        let result = solver.solve();
        assert!(result.is_ok(), "solve() should succeed: {result:?}");

        let pareto_solutions = solver.get_pareto_solutions();
        assert!(
            !pareto_solutions.is_empty(),
            "Should find some Pareto solutions"
        );
    }

    #[test]
    fn test_payoff_table() {
        init_test_logging();
        let problem = create_2kp_model(MODEL_TYPE);
        let options = create_2kp_options(MODEL_TYPE);

        let mut solver = Augmecon::try_new(problem, options).expect("Failed to create solver");
        solver.solve().expect("Failed to solve");
        let payoff_table = solver.get_payoff_table();

        assert_eq!(payoff_table.len(), 2, "Should have 2 rows in payoff table");

        let file_path = format!("tests/input/knapsack/{MODEL_TYPE}.xlsx");
        if std::path::Path::new(&file_path).exists() {
            let expected_payoff = read_excel(&file_path, "payoff_table");
            assert!(
                array_equal(&payoff_table, &expected_payoff, 0.01),
                "Payoff table should match expected values"
            );
        }
    }

    #[test]
    fn test_pareto_solutions() {
        init_test_logging();
        let problem = create_2kp_model(MODEL_TYPE);
        let options = create_2kp_options(MODEL_TYPE);

        let mut solver = Augmecon::try_new(problem, options).expect("Failed to create solver");
        let result = solver.solve();
        assert!(result.is_ok(), "solve() should succeed");

        let actual_pareto = solver.get_pareto_solutions();
        let actual_pareto: Vec<Vec<f64>> = actual_pareto
            .iter()
            .map(|sol| sol.objective_values.clone())
            .collect();

        let file_path = format!("tests/input/knapsack/{MODEL_TYPE}.xlsx");
        if std::path::Path::new(&file_path).exists() {
            let expected_pareto = read_excel(&file_path, "pareto_sols");
            assert!(
                array_equal(&actual_pareto, &expected_pareto, 0.01),
                "Pareto solutions should match expected values"
            );
        }
    }
}
