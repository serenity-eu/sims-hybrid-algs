//! Test demonstrating how grid point density affects Pareto front completeness
//! This test shows that insufficient grid points can lead to incomplete Pareto fronts

use augmecon::{
    sims_problem::{create_sims_problem, SimsInstance},
    Augmecon, HasObjectives, Options,
};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

/// Parse a DZN file (simplified version for this test)
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
        let coverage: HashSet<usize> = image_set.iter().map(|&x| x - 1).collect();
        config.set_image_coverage(i, coverage);
    }

    // Set cloud coverage
    for (i, cloud_set) in clouds_data.iter().enumerate() {
        let clouds: HashSet<usize> = cloud_set.iter().map(|&x| x - 1).collect();
        config.set_cloud_coverage(i, clouds);
    }

    // Set costs, areas, resolution, incidence angles
    for (i, &cost) in costs_data.iter().enumerate() {
        config.set_cost(i, cost);
    }
    for (i, &area) in areas_data.iter().enumerate() {
        config.set_area(i, area);
    }

    // Set cloud areas (default to 1.0 for each cloud)
    for cloud_id in 0..num_clouds {
        config.set_cloud_area(cloud_id, 1.0);
    }

    for (i, &resolution) in resolution_data.iter().enumerate() {
        config.set_resolution(i, resolution);
    }
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

/// Solve SIMS problem with specified grid points
fn solve_with_grid_points(
    config: &SimsInstance,
    grid_points: usize,
    name: &str,
) -> Vec<augmecon::Solution> {
    let problem = create_sims_problem(config);
    let options = Options::new()
        .with_name(name.to_string())
        .with_grid_points(grid_points)
        .with_bypass_coefficient(true)
        .with_flag_array(true)
        .with_early_exit(true)
        .with_solver_option("log", "0");

    let mut augmecon = Augmecon::try_new(problem, options).expect("Failed to create Augmecon");
    augmecon.solve().expect("Failed to solve problem")
}

/// Print solution statistics
fn print_solution_stats(solutions: &[augmecon::Solution], grid_points: usize) {
    println!(
        "Grid Points: {}, Solutions Found: {}",
        grid_points,
        solutions.len()
    );

    if !solutions.is_empty() {
        println!("  📊 Objective Ranges:");

        // Calculate ranges for each objective
        for obj_idx in 0..4 {
            let values: Vec<f64> = solutions.iter().map(|s| s.objectives()[obj_idx]).collect();

            let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            println!(
                "    Obj {}: [{:.2} → {:.2}] (range: {:.2})",
                obj_idx + 1,
                min_val,
                max_val,
                max_val - min_val
            );
        }

        println!("  🎯 Sample Solutions:");
        for (i, solution) in solutions.iter().take(3).enumerate() {
            let objs = solution.objectives();
            println!(
                "    Sol {}: [{:.2}, {:.2}, {:.2}, {:.2}]",
                i + 1,
                objs[0],
                objs[1],
                objs[2],
                objs[3]
            );
        }
    }
    println!();
}

#[test]
fn test_grid_density_impact() {
    // Initialize logging
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .is_test(true)
            .try_init()
            .ok();
    });

    let test_path = Path::new("tests/input/sims/tokyo_bay_30.dzn");
    if !test_path.exists() {
        println!("Skipping test - file not found: {test_path:?}");
        return;
    }

    println!("🔬 Testing Grid Density Impact on Pareto Front Completeness");
    println!("Instance: tokyo_bay_30.dzn\n");

    let config = parse_dzn_file(test_path).expect("Failed to parse DZN file");

    // Test different grid densities
    let grid_sizes = vec![2, 3, 5, 8, 10];

    for grid_points in grid_sizes {
        println!("🔍 Testing with {grid_points} grid points:");

        let start_time = std::time::Instant::now();
        let solutions = solve_with_grid_points(&config, grid_points, "tokyo_bay_grid_test");
        let solve_time = start_time.elapsed();

        print_solution_stats(&solutions, grid_points);
        println!("  ⏱️  Solve time: {:.2}s", solve_time.as_secs_f64());
        println!("  📈 Grid combinations: {}", grid_points.pow(3)); // 4 objectives = 3 constrained
        println!(
            "  🎲 Solutions per combination: {:.4}",
            solutions.len() / grid_points.pow(3)
        );
        println!("{}", "─".repeat(60));
    }
}

#[test]
fn test_convergence_analysis() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .is_test(true)
            .try_init()
            .ok();
    });

    let test_path = Path::new("tests/input/sims/tokyo_bay_30.dzn");
    if !test_path.exists() {
        println!("Skipping convergence test - file not found: {test_path:?}");
        return;
    }

    println!("📈 Convergence Analysis: How Pareto Front Quality Improves with Grid Density\n");

    let config = parse_dzn_file(test_path).expect("Failed to parse DZN file");

    let grid_sizes = [3, 5, 8, 12];
    let mut previous_solutions: Vec<augmecon::Solution> = Vec::new();

    for (i, grid_points) in grid_sizes.iter().enumerate() {
        println!("🎯 Grid Points: {grid_points}");

        let solutions = solve_with_grid_points(&config, *grid_points, "convergence_test");

        // Analyze solution diversity
        let mut obj1_values: Vec<f64> = solutions.iter().map(|s| s.objectives()[0]).collect();
        obj1_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        obj1_values.dedup_by(|a, b| (*a - *b).abs() < 1.0); // Remove near-duplicates

        println!("  Unique solutions (obj1): {}", obj1_values.len());

        if i > 0 {
            // Compare with previous run
            #[expect(
                clippy::cast_precision_loss,
                reason = "Calculating solution count improvement"
            )]
            let improvement = solutions.len() as f64 / previous_solutions.len() as f64;
            println!("  Solution count improvement: {improvement:.2}x");

            // Check for new extreme points
            let current_min = solutions
                .iter()
                .map(|s| s.objectives()[0])
                .reduce(f64::min)
                .unwrap_or(f64::INFINITY);
            let current_max = solutions
                .iter()
                .map(|s| s.objectives()[0])
                .reduce(f64::max)
                .unwrap_or(f64::NEG_INFINITY);
            let prev_min = previous_solutions
                .iter()
                .map(|s| s.objectives()[0])
                .reduce(f64::min)
                .unwrap_or(f64::INFINITY);
            let prev_max = previous_solutions
                .iter()
                .map(|s| s.objectives()[0])
                .reduce(f64::max)
                .unwrap_or(f64::NEG_INFINITY);

            println!(
                "  Objective 1 range: [{:.0}, {:.0}] (width: {:.0})",
                current_min,
                current_max,
                current_max - current_min
            );

            if current_min < prev_min || current_max > prev_max {
                println!("  ✨ New extreme points discovered!");
            }
        } else {
            let obj1_min = solutions
                .iter()
                .map(|s| s.objectives()[0])
                .reduce(f64::min)
                .unwrap_or(f64::INFINITY);
            let obj1_max = solutions
                .iter()
                .map(|s| s.objectives()[0])
                .reduce(f64::max)
                .unwrap_or(f64::NEG_INFINITY);
            println!("  Objective 1 range: [{obj1_min:.0}, {obj1_max:.0}]");
        }

        previous_solutions = solutions;
        println!();
    }

    println!("💡 Key Insights:");
    println!("   • More grid points typically reveal more Pareto-optimal solutions");
    println!("   • Computational cost grows exponentially: O(grid_points^(objectives-1))");
    println!("   • For 4 objectives: 3 grid → 27 solves, 12 grid → 1728 solves");
    println!("   • Diminishing returns: Eventually additional points find fewer new solutions");
}
