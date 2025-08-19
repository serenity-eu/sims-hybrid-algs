//! Integration tests for SIMS (Satellite Image Mosaic Selection) problems
//! Tests solve real SIMS instances from the test data files

use augmecon::{
    sims_problem::{create_sims_problem, SimsInstance},
    Augmecon, HasObjectives, Options,
};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

// ============================================================================
// Helper Functions
// ============================================================================

/// Initialize test logging
fn init_test_logging() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .is_test(true)
            .try_init()
            .ok();
    });
}

/// Parse a DZN (`MiniZinc` data) file to extract SIMS instance data
fn parse_dzn_file(file_path: &Path) -> Result<SimsInstance, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(file_path)?;

    let mut num_images = 0;
    let mut universe = 0;
    let mut max_cloud_area = 0;
    let mut images_data: Vec<Vec<usize>> = Vec::new();
    let mut clouds_data: Vec<Vec<usize>> = Vec::new();
    let mut costs_data: Vec<f64> = Vec::new();
    let mut areas_data: Vec<f64> = Vec::new();
    let mut resolution_data: Vec<f64> = Vec::new();
    let mut incidence_angle_data: Vec<f64> = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('%') {
            continue;
        }

        if line.starts_with("num_images = ") {
            num_images = line
                .split('=')
                .nth(1)
                .unwrap()
                .trim()
                .trim_end_matches(';')
                .parse()?;
        } else if line.starts_with("universe = ") {
            universe = line
                .split('=')
                .nth(1)
                .unwrap()
                .trim()
                .trim_end_matches(';')
                .parse()?;
        } else if line.starts_with("max_cloud_area = ") {
            max_cloud_area = line
                .split('=')
                .nth(1)
                .unwrap()
                .trim()
                .trim_end_matches(';')
                .parse()?;
        } else if line.starts_with("images = ") {
            images_data = parse_array_of_sets(&content, "images");
        } else if line.starts_with("clouds = ") {
            clouds_data = parse_array_of_sets(&content, "clouds");
        } else if line.starts_with("costs = ") {
            costs_data = parse_numeric_array(&content, "costs")?;
        } else if line.starts_with("areas = ") {
            areas_data = parse_numeric_array(&content, "areas")?;
        } else if line.starts_with("resolution = ") {
            resolution_data = parse_numeric_array(&content, "resolution")?;
        } else if line.starts_with("incidence_angle = ") {
            incidence_angle_data = parse_numeric_array(&content, "incidence_angle")?;
        }
    }

    // Calculate the number of clouds from the cloud data
    let mut max_cloud_id = 0;
    for cloud_set in &clouds_data {
        for &cloud_id in cloud_set {
            if cloud_id > max_cloud_id {
                max_cloud_id = cloud_id;
            }
        }
    }
    let num_clouds = if max_cloud_id > 0 { max_cloud_id } else { 1 }; // At least 1 cloud

    let mut config = SimsInstance::new(num_images, universe, num_clouds, max_cloud_area);

    // Set image coverage
    for (i, image_set) in images_data.iter().enumerate() {
        let coverage: HashSet<usize> = image_set.iter().map(|&x| x - 1).collect(); // Convert 1-based to 0-based
        config.set_image_coverage(i, coverage);
    }

    // Set cloud coverage
    for (i, cloud_set) in clouds_data.iter().enumerate() {
        let clouds: HashSet<usize> = cloud_set.iter().map(|&x| x - 1).collect(); // Convert 1-based to 0-based
        config.set_cloud_coverage(i, clouds);
    }

    // Set costs
    for (i, &cost) in costs_data.iter().enumerate() {
        config.set_cost(i, cost);
    }

    // Set areas
    for (i, &area) in areas_data.iter().enumerate() {
        config.set_area(i, area);
    }

    // Set cloud areas (default to 1.0 for each cloud)
    for cloud_id in 0..num_clouds {
        config.set_cloud_area(cloud_id, 1.0);
    }

    // Set resolution
    for (i, &resolution) in resolution_data.iter().enumerate() {
        config.set_resolution(i, resolution);
    }

    // Set incidence angles
    for (i, &angle) in incidence_angle_data.iter().enumerate() {
        config.set_incidence_angle(i, angle);
    }

    Ok(config)
}

/// Parse an array of sets from DZN content
fn parse_array_of_sets(content: &str, array_name: &str) -> std::vec::Vec<std::vec::Vec<usize>> {
    let mut result = Vec::new();
    let mut inside_array = false;
    let mut current_set = String::new();
    let mut brace_count = 0;

    for line in content.lines() {
        let line = line.trim();

        if line.starts_with(&format!("{array_name} = [")) {
            inside_array = true;
            let after_equals = line.split('=').nth(1).unwrap().trim();
            current_set = after_equals.to_string();
        } else if inside_array {
            current_set.push(' ');
            current_set.push_str(line);
        }

        if inside_array && line.contains("];") {
            break;
        }
    }

    // Parse the array content
    let array_content = current_set
        .trim_start_matches('[')
        .trim_end_matches("];")
        .trim();

    let mut current_set_str = String::new();
    let mut in_set = false;

    for ch in array_content.chars() {
        match ch {
            '{' => {
                in_set = true;
                brace_count += 1;
            }
            '}' => {
                brace_count -= 1;
                if brace_count == 0 {
                    in_set = false;
                    // Parse the current set
                    if current_set_str.is_empty() {
                        result.push(Vec::new());
                    } else {
                        let numbers: Vec<usize> = current_set_str
                            .split(',')
                            .filter_map(|s| s.trim().parse().ok())
                            .collect();
                        result.push(numbers);
                    }
                    current_set_str.clear();
                }
            }
            ',' if !in_set => {
                // Skip commas between sets
            }
            _ if in_set => {
                current_set_str.push(ch);
            }
            _ => {}
        }
    }

    result
}

/// Parse a numeric array from DZN content
fn parse_numeric_array<T>(
    content: &str,
    array_name: &str,
) -> Result<Vec<T>, Box<dyn std::error::Error>>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let mut result = Vec::new();
    let mut inside_array = false;
    let mut array_content = String::new();

    for line in content.lines() {
        let line = line.trim();

        if line.starts_with(&format!("{array_name} = [")) {
            inside_array = true;
            let after_equals = line.split('=').nth(1).unwrap().trim();
            array_content = after_equals.to_string();
        } else if inside_array {
            array_content.push(' ');
            array_content.push_str(line);
        }

        if inside_array && line.contains("];") {
            break;
        }
    }

    // Parse the array content
    let content_str = array_content
        .trim_start_matches('[')
        .trim_end_matches("];")
        .trim();

    for item in content_str.split(',') {
        let item = item.trim();
        if !item.is_empty() {
            result.push(item.parse()?);
        }
    }

    Ok(result)
}

/// Create test options for SIMS problems
fn options_for_sims(name: &str, grid_points: usize) -> Options {
    Options::new()
        .with_name(name.to_string())
        .with_grid_points(grid_points)
        .with_bypass_coefficient(true)  // Enable bypass optimization for speed
        .with_flag_array(true)         // Enable flag array optimization for speed
        .with_early_exit(true)         // Enable early exit optimization
        .with_solver_option("log", "0") // Disable solver output for tests
}

/// Validate that a solution is feasible for the SIMS problem
fn validate_sims_solution(config: &SimsInstance, solution: &augmecon::Solution) -> bool {
    let objectives = solution.objectives();

    // Check that we have 4 objectives
    if objectives.len() != 4 {
        eprintln!("Expected 4 objectives, got {}", objectives.len());
        return false;
    }

    // Print objective values for debugging
    println!(
        "DEBUG: Objective values: [{:.2}, {:.2}, {:.2}, {:.2}]",
        objectives[0], objectives[1], objectives[2], objectives[3]
    );

    // All objectives should be finite
    for (i, &obj) in objectives.iter().enumerate() {
        if !obj.is_finite() {
            eprintln!("Objective {i} is invalid: {obj}");
            return false;
        }
        // Objectives should be non-negative (except for the resolution which might be negative to represent maximization)
        if obj < 0.0 && i != 2 {
            // Allow negative for resolution (objective 2)
            eprintln!("Objective {} is negative: {}", i + 1, obj);
            return false;
        }
    }

    // Extract variable values
    let variables = &solution.decision_variables;

    // Print some key variables for debugging
    let mut selected_images = Vec::new();
    for i in 0..config.num_images {
        let var_name = format!("taken_{i}");
        if let Some(&value) = variables.get(&var_name) {
            if value > 0.5 {
                selected_images.push(i);
            }
        }
    }
    println!("DEBUG: Selected images: {selected_images:?}");

    // Check that all universe points are covered
    for u in 0..config.universe_size {
        let mut covered = false;
        for i in 0..config.num_images {
            if config.images[i].contains(&u) {
                let var_name = format!("taken_{i}");
                if let Some(&value) = variables.get(&var_name) {
                    if value > 0.5 {
                        // Binary variable should be 0 or 1
                        covered = true;
                        break;
                    }
                }
            }
        }
        if !covered {
            eprintln!("Universe point {u} is not covered");
            return false;
        }
    }

    true
}

// ============================================================================
// Test Cases
// ============================================================================

#[test]
fn test_sims_parser_basic() {
    init_test_logging();

    let test_path = Path::new("tests/input/sims/paris_30.dzn");
    if !test_path.exists() {
        println!("Skipping test - file not found: {test_path:?}");
        return;
    }

    let config = parse_dzn_file(test_path).expect("Failed to parse DZN file");

    assert_eq!(config.num_images, 30);
    assert_eq!(config.universe_size, 475);
    assert_eq!(config.max_cloud_area, 1_480_038_297);

    // Verify some data was loaded
    assert!(!config.costs.is_empty());
    assert!(!config.areas.is_empty());
    assert!(!config.resolution.is_empty());
    assert!(!config.incidence_angle.is_empty());

    println!(
        "✅ Successfully parsed paris_30.dzn with {} images and {} universe points",
        config.num_images, config.universe_size
    );
}

#[test]
fn test_sims_small_paris() {
    init_test_logging();

    let test_path = Path::new("tests/input/sims/paris_30.dzn");
    if !test_path.exists() {
        println!("Skipping test - file not found: {test_path:?}");
        return;
    }

    println!("🛰️  Starting SIMS test for paris_30");

    let config = parse_dzn_file(test_path).expect("Failed to parse DZN file");
    println!("✅ Parsed DZN file successfully");

    let problem = create_sims_problem(&config);
    println!("✅ Created SIMS problem");

    let options = options_for_sims("paris_30", 3); // Extremely small grid for testing
    println!(
        "✅ Created options with {} grid points",
        options.grid_points.unwrap_or(0)
    );

    println!("📊 Problem Statistics:");
    println!("  - Images: {}", config.num_images);
    println!("  - Universe points: {}", config.universe_size);
    println!("  - Variables: {}", problem.var_map.len());
    println!("  - Constraints: {}", problem.constraints.len());
    println!("🚀 Starting to solve...");

    let mut augmecon = Augmecon::try_new(problem, options).expect("Failed to create Augmecon");
    let solutions = augmecon.solve().expect("Failed to solve problem");

    assert!(!solutions.is_empty(), "Should find at least one solution");

    // Validate all solutions
    for (i, solution) in solutions.iter().enumerate() {
        assert!(
            validate_sims_solution(&config, solution),
            "Solution {i} is not valid"
        );
    }

    println!("✅ Found {} valid solutions for paris_30", solutions.len());

    // Print some solution statistics
    if !solutions.is_empty() {
        let objectives = solutions[0].objectives();
        println!("📈 First solution objectives:");
        println!("  - Total cost: {:.2}", objectives[0]);
        println!("  - Cloudy area: {:.2}", objectives[1]);
        println!("  - Max resolution: {:.2}", objectives[2]);
        println!("  - Max incidence: {:.2}", objectives[3]);
    }
}

#[test]
fn test_sims_multiple_cities_small() {
    init_test_logging();

    let test_cases = [
        ("paris_30.dzn", 15),
        ("tokyo_bay_30.dzn", 15),
        ("lagos_nigeria_30.dzn", 15),
    ];

    for (filename, grid_points) in test_cases {
        let test_path = Path::new("tests/input/sims").join(filename);
        if !test_path.exists() {
            println!("Skipping test - file not found: {test_path:?}");
            continue;
        }

        println!("\n🛰️  Testing SIMS problem: {filename}");

        let config = parse_dzn_file(&test_path).expect("Failed to parse DZN file");
        let problem = create_sims_problem(&config);
        let options = options_for_sims(filename, grid_points);

        println!("📊 Problem Statistics:");
        println!("  - Images: {}", config.num_images);
        println!("  - Universe points: {}", config.universe_size);

        let mut augmecon = Augmecon::try_new(problem, options).expect("Failed to create Augmecon");
        let solutions = augmecon.solve().expect("Failed to solve problem");

        assert!(
            !solutions.is_empty(),
            "Should find at least one solution for {filename}"
        );

        // Validate first few solutions
        for (i, solution) in solutions.iter().take(3).enumerate() {
            assert!(
                validate_sims_solution(&config, solution),
                "Solution {i} is not valid for {filename}"
            );
        }

        println!(
            "✅ Found {} valid solutions for {}",
            solutions.len(),
            filename
        );
    }
}

#[test]
#[ignore = "This test takes a long time to run - use for comprehensive validation"]
fn test_sims_medium_instances() {
    init_test_logging();

    let test_cases = [
        ("paris_50.dzn", 25),
        ("tokyo_bay_50.dzn", 25),
        ("mexico_city_50.dzn", 25),
        ("rio_de_janeiro_50.dzn", 25),
    ];

    for (filename, grid_points) in test_cases {
        let test_path = Path::new("tests/input/sims").join(filename);
        if !test_path.exists() {
            println!("Skipping test - file not found: {test_path:?}");
            continue;
        }

        println!("\n🛰️  Testing SIMS problem: {filename}");

        let config = parse_dzn_file(&test_path).expect("Failed to parse DZN file");
        let problem = create_sims_problem(&config);
        let options = options_for_sims(filename, grid_points);

        println!("📊 Problem Statistics:");
        println!("  - Images: {}", config.num_images);
        println!("  - Universe points: {}", config.universe_size);

        let start_time = std::time::Instant::now();
        let mut augmecon = Augmecon::try_new(problem, options).expect("Failed to create Augmecon");
        let solutions = augmecon.solve().expect("Failed to solve problem");
        let solve_time = start_time.elapsed();

        assert!(
            !solutions.is_empty(),
            "Should find at least one solution for {filename}"
        );

        // Validate some solutions
        for (i, solution) in solutions.iter().take(5).enumerate() {
            assert!(
                validate_sims_solution(&config, solution),
                "Solution {i} is not valid for {filename}"
            );
        }

        println!(
            "✅ Found {} solutions for {} in {:.2}s",
            solutions.len(),
            filename,
            solve_time.as_secs_f64()
        );

        // Print payoff table if available
        let payoff_table = augmecon.get_payoff_table();
        if !payoff_table.is_empty() {
            println!("📈 Payoff Table:");
            for (i, row) in payoff_table.iter().enumerate() {
                println!(
                    "    Obj {}: [{:.2}, {:.2}, {:.2}, {:.2}]",
                    i + 1,
                    row[0],
                    row[1],
                    row[2],
                    row[3]
                );
            }
        }
    }
}

#[test]
#[ignore = "This test takes a very long time to run - use for full system validation"]
fn test_sims_large_instances() {
    init_test_logging();

    let test_cases = [("paris_100.dzn", 30), ("tokyo_bay_100.dzn", 30)];

    for (filename, grid_points) in test_cases {
        let test_path = Path::new("tests/input/sims").join(filename);
        if !test_path.exists() {
            println!("Skipping test - file not found: {test_path:?}");
            continue;
        }

        println!("\n🛰️  Testing large SIMS problem: {filename}");

        let config = parse_dzn_file(&test_path).expect("Failed to parse DZN file");
        let problem = create_sims_problem(&config);
        let options = options_for_sims(filename, grid_points);

        println!("📊 Problem Statistics:");
        println!("  - Images: {}", config.num_images);
        println!("  - Universe points: {}", config.universe_size);

        let start_time = std::time::Instant::now();
        let mut augmecon = Augmecon::try_new(problem, options).expect("Failed to create Augmecon");
        let solutions = augmecon.solve().expect("Failed to solve problem");
        let solve_time = start_time.elapsed();

        assert!(
            !solutions.is_empty(),
            "Should find at least one solution for {filename}"
        );

        // Validate first solution
        assert!(
            validate_sims_solution(&config, &solutions[0]),
            "First solution is not valid for {filename}"
        );

        println!(
            "✅ Found {} solutions for {} in {:.2}s",
            solutions.len(),
            filename,
            solve_time.as_secs_f64()
        );
    }
}

#[test]
fn test_all_small_sims_instances() {
    init_test_logging();

    let sims_dir = Path::new("tests/input/sims");
    if !sims_dir.exists() {
        println!("Skipping test - SIMS directory not found: {sims_dir:?}");
        return;
    }

    let entries = fs::read_dir(sims_dir).expect("Failed to read SIMS directory");

    for entry in entries {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();

        if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
            // Only test small instances (30 images) for speed
            if filename.ends_with("_30.dzn") {
                println!("\n🛰️  Testing SIMS problem: {filename}");

                let config = parse_dzn_file(&path).expect("Failed to parse DZN file");
                let problem = create_sims_problem(&config);
                let options = options_for_sims(filename, 10); // Very small grid for fast testing

                let mut augmecon =
                    Augmecon::try_new(problem, options).expect("Failed to create Augmecon");
                let solutions = augmecon.solve().expect("Failed to solve problem");

                assert!(
                    !solutions.is_empty(),
                    "Should find at least one solution for {filename}"
                );

                // Validate first solution
                assert!(
                    validate_sims_solution(&config, &solutions[0]),
                    "First solution is not valid for {filename}"
                );

                println!("✅ Found {} solutions for {}", solutions.len(), filename);
            }
        }
    }
}
