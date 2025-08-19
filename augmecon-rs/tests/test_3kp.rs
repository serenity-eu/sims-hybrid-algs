//! Integration tests for 3KP (3-objective knapsack) problems
//! Tests both 3KP40 and 3KP50 variants

use augmecon::{
    grid::GridGenerator, Augmecon, MultiObjectiveProblem, ObjectiveDirection, VariableType,
};
use calamine::{open_workbook, Data, Reader, Xlsx};
use good_lp::{constraint, Expression};
use std::fmt::Write as _;

// ============================================================================
// Helper Functions (specific to 3KP tests)
// ============================================================================

/// Initialize test logging
fn init_test_logging() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
            .is_test(true)
            .try_init()
            .ok();
    });
}

/// Compare two floating point numbers with tolerance
fn float_equal(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() < tolerance
}

/// Compare two 2D arrays with tolerance
fn array_equal(actual: &[Vec<f64>], expected: &[Vec<f64>], tolerance: f64) -> bool {
    if actual.len() != expected.len() {
        return false;
    }

    for (actual_row, expected_row) in actual.iter().zip(expected.iter()) {
        if actual_row.len() != expected_row.len() {
            return false;
        }

        for (actual_val, expected_val) in actual_row.iter().zip(expected_row.iter()) {
            if !float_equal(*actual_val, *expected_val, tolerance) {
                return false;
            }
        }
    }

    true
}

/// Read Excel data for testing
fn read_excel(file_path: &str, sheet_name: &str) -> Vec<Vec<f64>> {
    use calamine::{open_workbook, Data, Reader, Xlsx};

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

/// Create options for 3KP40 test problem
fn options_for_3kp40() -> augmecon::Options {
    augmecon::Options::new()
        .with_name("3kp40".to_string())
        .with_grid_points(540)  // Same as PyAugmecon
        .with_nadir_points(vec![1031.0, 1069.0])
        .with_bypass_coefficient(true)  // Enable bypass optimization for speed
        .with_flag_array(true)         // Enable flag array optimization for speed
        .with_early_exit(true)         // Enable early exit optimization
        .with_solver_option("log", "0")
}

/// Create options for 3KP50 test problem
fn options_for_3kp50() -> augmecon::Options {
    augmecon::Options::new()
        .with_name("3kp50".to_string())
        .with_grid_points(847)  // Same as PyAugmecon
        .with_nadir_points(vec![1124.0, 1041.0])
        .with_bypass_coefficient(true)  // Enable bypass optimization for speed
        .with_flag_array(true)         // Enable flag array optimization for speed
        .with_early_exit(true)         // Enable early exit optimization
        .with_solver_option("log", "0")
}

/// Create a 3-objective knapsack problem model
fn three_kp_model(model_name: &str) -> MultiObjectiveProblem {
    #[allow(clippy::items_after_statements)]
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
    for (constraint_idx, capacity) in capacities.iter().enumerate() {
        let mut total_weight = Expression::from(0.0);
        for i in 0..n_items {
            if let Some(&var) = problem.var_map.get(&format!("x{i}")) {
                total_weight += weights[constraint_idx][i] * var;
            }
        }
        problem.add_constraint(constraint!(total_weight <= *capacity));
    }

    // Add objectives
    for profit in profits.iter().take(3) {
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
// Tests for 3KP40
// ============================================================================

mod test_3kp40 {
    use super::*;

    const MODEL_TYPE: &str = "3kp40";

    #[test]
    fn test_model_creation() {
        init_test_logging();
        // Test that we can create the model
        let problem = three_kp_model(MODEL_TYPE);

        // Check that we have the expected number of objectives (3 for 3kp problems)
        assert_eq!(
            problem.objectives.len(),
            3,
            "3kp40 should have 3 objectives"
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

        // Check that we have the right number of constraints (3 for 3kp problems)
        assert_eq!(
            problem.constraints.len(),
            3,
            "3kp40 should have 3 constraints"
        );
    }

    #[test]
    fn test_options_creation() {
        init_test_logging();
        // Test that we can create appropriate options
        let options = options_for_3kp40();

        // Verify that options are created successfully
        assert!(options.grid_points.is_some(), "Should have grid points set");
        assert_eq!(
            options.grid_points.unwrap(),
            540, // Back to original PyAugmecon value
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
            2,
            "Should have 2 nadir points for 3 objectives"
        );
        assert!(
            float_equal(nadir[0], 1031.0, 0.01),
            "First nadir point should be 1031.0, got {}",
            nadir[0]
        );
        assert!(
            float_equal(nadir[1], 1069.0, 0.01),
            "Second nadir point should be 1069.0, got {}",
            nadir[1]
        );
    }

    #[test]
    fn test_payoff_table() {
        init_test_logging();
        // Following the Python test pattern
        let model = three_kp_model(MODEL_TYPE);
        let options = options_for_3kp40();
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
    fn test_pareto_solutions() {
        init_test_logging();
        // Following the Python test pattern
        let model = three_kp_model(MODEL_TYPE);
        let options = options_for_3kp40();
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
// Tests for 3KP50
// ============================================================================

mod test_3kp50 {
    use super::*;

    const MODEL_TYPE: &str = "3kp50";

    #[test]
    fn test_model_creation() {
        init_test_logging();
        // Test that we can create the model
        let problem = three_kp_model(MODEL_TYPE);

        // Check that we have the expected number of objectives (3 for 3kp problems)
        assert_eq!(
            problem.objectives.len(),
            3,
            "3kp50 should have 3 objectives"
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

        // Check that we have the right number of constraints (3 for 3kp problems)
        assert_eq!(
            problem.constraints.len(),
            3,
            "3kp50 should have 3 constraints"
        );
    }

    #[test]
    fn test_options_creation() {
        init_test_logging();
        // Test that we can create appropriate options
        let options = options_for_3kp50();

        // Verify that options are created successfully
        assert!(options.grid_points.is_some(), "Should have grid points set");
        assert_eq!(
            options.grid_points.unwrap(),
            847, // Back to original PyAugmecon value
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
            2,
            "Should have 2 nadir points for 3 objectives"
        );
        assert!(
            float_equal(nadir[0], 1124.0, 0.01),
            "First nadir point should be 1124.0, got {}",
            nadir[0]
        );
        assert!(
            float_equal(nadir[1], 1041.0, 0.01),
            "Second nadir point should be 1041.0, got {}",
            nadir[1]
        );
    }

    #[test]
    fn test_payoff_table() {
        init_test_logging();
        // Following the Python test pattern
        let model = three_kp_model(MODEL_TYPE);
        let options = options_for_3kp50();
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
    fn test_pareto_solutions() {
        init_test_logging();
        // Following the Python test pattern
        let model = three_kp_model(MODEL_TYPE);
        let options = options_for_3kp50();
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

    #[test]
    fn test_payoff_table_fast() {
        init_test_logging();
        println!(
            "DEBUG: Starting FAST payoff table test for {MODEL_TYPE} with reduced grid points"
        );
        // Use much smaller grid points for fast testing
        let model = three_kp_model(MODEL_TYPE);
        let options = augmecon::Options::new()
            .with_name("3kp50_fast".to_string())  // Fixed name
            .with_grid_points(10)  // Much smaller for quick testing
            .with_nadir_points(vec![1124.0, 1041.0])  // Correct nadir for 3kp50
            .with_bypass_coefficient(false)  // DISABLE bypass to see if it affects results
            .with_flag_array(false)         // DISABLE flag array to see if it affects results
            .with_early_exit(false)         // DISABLE early exit to see if it affects results
            .with_solver_option("log", "0");
        let mut solver = Augmecon::try_new(model, options).expect("Failed to create solver");

        println!("DEBUG: About to solve the problem with 10 grid points...");
        // Solve the problem
        solver.solve().expect("Solve should succeed");
        println!("DEBUG: Problem solved successfully");

        // Get detailed statistics
        println!("DEBUG: Getting solver statistics");
        let actual_payoff = solver.get_payoff_table();
        let pareto_solutions = solver.get_pareto_front().objective_matrix();

        println!("DEBUG: ========== RUST RESULTS (NO OPTIMIZATIONS) ==========");
        println!("DEBUG: Payoff table: {actual_payoff:?}");
        println!(
            "DEBUG: Number of Pareto solutions: {}",
            pareto_solutions.len()
        );

        // Export solutions to file for comparison - simple format
        let mut solutions_text = String::new();
        for (i, solution) in pareto_solutions.iter().enumerate() {
            writeln!(&mut solutions_text, "Solution {i}: {solution:?}").ok();
        }
        std::fs::write("rust_solutions.txt", &solutions_text).ok();

        println!("DEBUG: Pareto solutions exported to rust_solutions.txt");
        println!("DEBUG: Pareto solutions: {pareto_solutions:?}");

        // Try to get solve statistics if available
        println!("DEBUG: ===================================");

        // Basic sanity checks
        assert_eq!(actual_payoff.len(), 3, "Should have 3 rows in payoff table");
        for row in actual_payoff {
            assert_eq!(row.len(), 3, "Each row should have 3 columns");
        }

        println!("DEBUG: Fast payoff table test completed successfully");
    }

    #[test]
    fn test_detailed_debug_3kp50() {
        init_test_logging();
        println!("DEBUG: Starting DETAILED DEBUG Rust test for 3kp50 with reduced grid points");

        // Create model
        let model = three_kp_model(MODEL_TYPE);

        // Configure options to exactly match Python
        let options = augmecon::Options::new()
            .with_name("3kp50_detailed_debug".to_string())
            .with_grid_points(10) // Same as Python
            .with_nadir_points(vec![1124.0, 1041.0]) // Same as Python
            .with_bypass_coefficient(false) // Disable all optimizations
            .with_flag_array(false)
            .with_early_exit(false)
            .with_solver_option("log", "0");

        println!("DEBUG: DETAILED ANALYSIS - Options configured to match Python test");
        println!("DEBUG: Grid points: {:?}", options.grid_points);
        println!("DEBUG: Nadir points: {:?}", options.nadir_points);
        println!("DEBUG: All optimizations disabled for fair comparison");

        println!("DEBUG: STEP 1 - Problem Analysis");
        println!("DEBUG: Number of variables: {}", model.var_map.len());
        println!("DEBUG: Number of constraints: {}", model.constraints.len());
        println!("DEBUG: Number of objectives: {}", model.objectives.len());

        println!("DEBUG: About to solve the problem with AUGMECON...");

        // Create solver and solve
        let mut solver = Augmecon::try_new(model, options).expect("Failed to create solver");

        println!("DEBUG: STEP 2 - Pre-solve Analysis");
        println!("DEBUG: Solver created successfully");

        solver.solve().expect("Failed to solve problem");

        println!("DEBUG: STEP 3 - Post-solve Analysis");
        println!("DEBUG: Problem solved successfully");

        // Get detailed results
        let pareto_front = solver.get_pareto_front();
        let payoff_table = solver.get_payoff_table();

        println!("DEBUG: ========== DETAILED RUST RESULTS ==========");
        println!("DEBUG: Payoff table: {payoff_table:?}");
        println!("DEBUG: Final Pareto solutions: {}", pareto_front.len());

        // Export all solutions for detailed comparison
        let solutions = pareto_front.objective_matrix();

        // Write to file for comparison
        let mut output = String::new();
        output.push_str("Rust Pareto solutions (detailed debug):\n");
        for (i, solution) in solutions.iter().enumerate() {
            writeln!(&mut output, "Solution {i}: {solution:?}").ok();
        }

        std::fs::write("rust_detailed_solutions.txt", output).expect("Failed to write solutions");

        println!("DEBUG: STEP 4 - Solution Analysis");
        println!("DEBUG: First 5 Pareto solutions:");
        for (i, solution) in solutions.iter().take(5).enumerate() {
            println!("DEBUG:   Rust solution {i}: {solution:?}");
        }

        println!("DEBUG: ===================================");
        println!("DEBUG: Detailed Rust analysis completed successfully");
    }

    #[test]
    fn test_super_detailed_debug_3kp50() {
        init_test_logging();
        println!("DEBUG: Starting SUPER DETAILED DEBUG Rust test for 3kp50");

        // Create model
        let model = three_kp_model(MODEL_TYPE);

        // Configure options to exactly match Python
        let options = augmecon::Options::new()
            .with_name("3kp50_super_detailed_debug".to_string())
            .with_grid_points(10) // Same as Python
            .with_nadir_points(vec![1124.0, 1041.0]) // Same as Python
            .with_bypass_coefficient(false) // Disable all optimizations
            .with_flag_array(false)
            .with_early_exit(false)
            .with_solver_option("log", "0");

        println!("DEBUG: SUPER DETAILED ANALYSIS - Options configured to match Python test");
        println!("DEBUG: Grid points: {:?}", options.grid_points);
        println!("DEBUG: Nadir points: {:?}", options.nadir_points);
        println!("DEBUG: All optimizations disabled for fair comparison");

        println!("DEBUG: STEP 1 - Problem Analysis");
        println!("DEBUG: Number of variables: {}", model.var_map.len());
        println!("DEBUG: Number of constraints: {}", model.constraints.len());
        println!("DEBUG: Number of objectives: {}", model.objectives.len());

        println!("DEBUG: About to solve the problem with AUGMECON...");

        // Create solver and solve
        let mut solver = Augmecon::try_new(model, options).expect("Failed to create solver");

        println!("DEBUG: STEP 2 - Pre-solve Analysis");
        println!("DEBUG: Solver created successfully");

        solver.solve().expect("Failed to solve problem");

        println!("DEBUG: STEP 3 - Post-solve Analysis");
        println!("DEBUG: Problem solved successfully");

        // Get detailed results
        let pareto_front = solver.get_pareto_front();
        let payoff_table = solver.get_payoff_table();

        println!("DEBUG: ========== SUPER DETAILED RUST RESULTS ==========");
        println!("DEBUG: Payoff table: {payoff_table:?}");
        println!("DEBUG: Final Pareto solutions: {}", pareto_front.len());

        // Export all solutions for detailed comparison
        let solutions = pareto_front.objective_matrix();

        // Write comprehensive analysis to file
        let mut output = String::new();
        output.push_str("RUST SUPER DETAILED ANALYSIS\n");
        output.push_str("===============================\n\n");

        writeln!(&mut output, "Payoff table: {payoff_table:?}").ok();
        writeln!(
            &mut output,
            "Final Pareto solutions: {}\n",
            pareto_front.len()
        )
        .ok();

        output.push_str("ALL PARETO SOLUTIONS:\n");
        for (i, solution) in solutions.iter().enumerate() {
            writeln!(&mut output, "Pareto solution {i}: {solution:?}").ok();
        }

        std::fs::write("rust_super_detailed_analysis.txt", output)
            .expect("Failed to write analysis");

        // Write simplified comparison file
        let mut comparison_output = String::new();
        comparison_output.push_str("Rust Pareto solutions (for direct comparison with Python):\n");
        for (i, solution) in solutions.iter().enumerate() {
            writeln!(&mut comparison_output, "Solution {i}: {solution:?}").ok();
        }

        std::fs::write("rust_pareto_solutions_comparison.txt", comparison_output)
            .expect("Failed to write comparison");
        println!("DEBUG: Super detailed analysis files exported");

        println!("DEBUG: STEP 4 - Solution Analysis");
        println!("DEBUG: First 10 Pareto solutions:");
        for (i, solution) in solutions.iter().take(10).enumerate() {
            println!("DEBUG:   Rust Pareto solution {i}: {solution:?}");
        }

        println!("DEBUG: ===================================");
        println!("DEBUG: Super detailed Rust analysis completed successfully");
    }

    #[test]
    fn test_grid_debug_3kp50() {
        init_test_logging();
        println!("DEBUG: Starting GRID-SPECIFIC DEBUG Rust test for 3kp50");

        // Create model
        let model = three_kp_model(MODEL_TYPE);

        // Configure options to exactly match Python
        let options = augmecon::Options::new()
            .with_name("3kp50_grid_debug".to_string())
            .with_grid_points(10) // Same as Python
            .with_nadir_points(vec![1124.0, 1041.0]) // Same as Python
            .with_bypass_coefficient(false) // Disable all optimizations
            .with_flag_array(false)
            .with_early_exit(false)
            .with_solver_option("log", "0");

        println!("DEBUG: GRID ANALYSIS - Options configured to match Python test");
        println!("DEBUG: Grid points: {:?}", options.grid_points);
        println!("DEBUG: Nadir points: {:?}", options.nadir_points);

        println!("DEBUG: STEP 1 - Pre-solve Grid Analysis");
        println!("DEBUG: Number of variables: {}", model.var_map.len());
        println!("DEBUG: Number of constraints: {}", model.constraints.len());
        println!("DEBUG: Number of objectives: {}", model.objectives.len());

        let num_objectives = model.objectives.len();
        let grid_points = options.grid_points.unwrap_or(10);
        let nadir_points = options.nadir_points.clone();

        println!(
            "DEBUG: Generating uniform grid for {num_objectives} objectives with {grid_points} points"
        );
        let grid = GridGenerator::generate_uniform_grid(num_objectives, grid_points);
        println!("DEBUG: Generated {} total grid points", grid.len());

        // Show first few grid points
        println!("DEBUG: First 10 grid points:");
        for (i, point) in grid.iter().take(10).enumerate() {
            println!("DEBUG:   Grid point {i}: {point:?}");
        }

        println!("DEBUG: About to solve the problem with AUGMECON...");

        // Create solver and solve
        let mut solver = Augmecon::try_new(model, options).expect("Failed to create solver");

        println!("DEBUG: STEP 2 - Solving with grid tracking");

        solver.solve().expect("Failed to solve problem");

        println!("DEBUG: STEP 3 - Post-solve Grid Analysis");
        println!("DEBUG: Problem solved successfully");

        // Get detailed results
        let pareto_front = solver.get_pareto_front();
        let payoff_table = solver.get_payoff_table();

        println!("DEBUG: ========== GRID-SPECIFIC RUST RESULTS ==========");
        println!("DEBUG: Payoff table: {payoff_table:?}");
        println!("DEBUG: Final Pareto solutions: {}", pareto_front.len());
        println!("DEBUG: Grid points generated: {}", grid.len());

        // Export grid and solution analysis
        let solutions = pareto_front.objective_matrix();

        let mut output = String::new();
        output.push_str("RUST GRID ANALYSIS\n");
        output.push_str("==================\n\n");

        writeln!(&mut output, "Payoff table: {payoff_table:?}").ok();
        writeln!(&mut output, "Grid points setting: {grid_points}").ok();
        writeln!(&mut output, "Nadir points setting: {nadir_points:?}").ok();
        writeln!(&mut output, "Total grid points generated: {}", grid.len()).ok();
        writeln!(
            &mut output,
            "Final Pareto solutions: {}\n",
            pareto_front.len()
        )
        .ok();

        output.push_str("GRID POINTS GENERATED:\n");
        for (i, point) in grid.iter().enumerate() {
            writeln!(&mut output, "Grid point {i}: {point:?}").ok();
        }

        output.push_str("\nPARETO SOLUTIONS FOUND:\n");
        for (i, solution) in solutions.iter().enumerate() {
            writeln!(&mut output, "Solution {i}: {solution:?}").ok();
        }

        std::fs::write("rust_grid_analysis.txt", output).expect("Failed to write grid analysis");
        println!("DEBUG: Grid analysis exported to rust_grid_analysis.txt");

        println!("DEBUG: STEP 4 - Grid Point Summary");
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Calculating total grid points based on grid_points and num_objectives"
        )]
        let grid_points_per_dimension = grid_points.pow((num_objectives - 1) as u32);
        println!(
            "DEBUG: Expected grid points for {num_objectives} objectives with {grid_points} points per dimension: {grid_points_per_dimension}"
        );
        println!("DEBUG: Actual grid points generated: {}", grid.len());

        println!("DEBUG: ===================================");
        println!("DEBUG: Grid-specific Rust analysis completed successfully");
    }
}
