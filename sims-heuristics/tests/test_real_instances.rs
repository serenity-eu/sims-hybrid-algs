use std::{fs, path::Path, time::Duration};

use pareto::ParetoFront;
use pls::{
    objectives::ObjectiveType, pareto_local_search::ParetoLocalSearch,
    problem_bitset::ProblemBitset, solution::bitset_encoded_solution::BitsetEncodedSolution,
    solution_set_impl::NdTreeSolutionSet,
};
use serde::Deserialize;

/// Represents a solution loaded from JSON file (from exact solver results)
/// Note: We only use `selected_images`, `timestamp_s`, and `phase` for loading initial populations.
/// The objective values are re-computed by evaluating the solution against the problem instance.
#[derive(Debug, Deserialize)]
struct JsonSolution {
    selected_images: Vec<usize>,
    timestamp_s: f64,
    phase: String,
}

/// Container for JSON data file with pre-recorded solutions
#[derive(Debug, Deserialize)]
struct JsonSolutionData {
    solutions: Vec<JsonSolution>,
}

const OBJECTIVE_TYPES: [ObjectiveType; 4] = [
    ObjectiveType::TotalCost,
    ObjectiveType::CloudyArea,
    ObjectiveType::MinResolution,
    ObjectiveType::MaxIncidenceAngle,
];

const INSTANCES_PATH: &str = "data";

const SMALL_INSTANCES: [&str; 10] = [
    "lagos_nigeria_30.dzn",
    "rio_de_janeiro_30.dzn",
    "paris_30.dzn",
    "tokyo_bay_30.dzn",
    "mexico_city_30.dzn",
    "lagos_nigeria_50.dzn",
    "rio_de_janeiro_50.dzn",
    "paris_50.dzn",
    "tokyo_bay_50.dzn",
    "mexico_city_50.dzn",
];

const MEDIUM_INSTANCES: [&str; 5] = [
    "lagos_nigeria_100.dzn",
    "rio_de_janeiro_100.dzn",
    "paris_100.dzn",
    "tokyo_bay_100.dzn",
    "mexico_city_100.dzn",
];

const LARGE_INSTANCES: [&str; 5] = [
    "lagos_nigeria_145.dzn",
    "rio_de_janeiro_150.dzn",
    "paris_150.dzn",
    "tokyo_bay_150.dzn",
    "mexico_city_150.dzn",
];

const MAX_ITERATIONS: usize = 1000;
const MAX_DURATION: Duration = Duration::from_hours(1);

/// Load initial population from JSON file containing pre-recorded solutions
/// This simulates the two-phase approach where exact solver provides initial solutions
fn load_initial_population_from_json<const N: usize>(
    json_path: &Path,
    problem: &ProblemBitset<N>,
    exact_time_limit_s: f64,
) -> NdTreeSolutionSet<BitsetEncodedSolution<ProblemBitset<N>, N>, N> {
    let mut initial_population = NdTreeSolutionSet::new("initial_from_exact");

    // Read and parse JSON file
    let json_content = fs::read_to_string(json_path).expect("Failed to read JSON solution file");

    let solution_data: JsonSolutionData =
        serde_json::from_str(&json_content).expect("Failed to parse JSON solution data");

    // Filter solutions by exact phase and time limit
    let exact_solutions: Vec<&JsonSolution> = solution_data
        .solutions
        .iter()
        .filter(|sol| sol.phase == "exact" && sol.timestamp_s <= exact_time_limit_s)
        .collect();

    println!(
        "Loaded {} exact phase solutions from {} (time limit: {}s)",
        exact_solutions.len(),
        json_path.display(),
        exact_time_limit_s
    );

    // Convert JSON solutions to BitsetEncodedSolution and add to population
    let num_images = problem.num_images();
    let mut skipped_count = 0;

    for json_sol in exact_solutions {
        // Validate that all image indices are within bounds
        let max_index = json_sol.selected_images.iter().max().copied().unwrap_or(0);
        if max_index >= num_images {
            eprintln!(
                "WARNING: Skipping solution with invalid image index {max_index} (problem has {num_images} images)"
            );
            skipped_count += 1;
            continue;
        }

        // Create solution from selected images using the from_selected_images method
        let mut solution =
            BitsetEncodedSolution::from_selected_images(&json_sol.selected_images, problem);

        // Set the timestamp from the JSON data
        solution.timestamp = Duration::from_secs_f64(json_sol.timestamp_s);

        // Add to initial population (non-dominated solutions will be kept)
        initial_population.try_insert(&solution);
    }

    if skipped_count > 0 {
        eprintln!("WARNING: Skipped {skipped_count} solutions with invalid image indices");
    }

    println!(
        "Initial population size after filtering dominated: {}",
        initial_population.len()
    );

    initial_population
}

/// Two-phase test ratios: (`exact_percentage`, `pls_percentage`)
/// These represent time allocation between exact solver replay and PLS
const TWO_PHASE_RATIOS: [(u32, u32); 11] = [
    (0, 100),
    (10, 90),
    (20, 80),
    (30, 70),
    (40, 60),
    (50, 50),
    (60, 40),
    (70, 30),
    (80, 20),
    (90, 10),
    (100, 0),
];

#[test]
fn test_4d_small_instances() {
    for instance_file in SMALL_INSTANCES {
        let instance_path = Path::new(INSTANCES_PATH).join(instance_file);

        let problem = ProblemBitset::<4>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES)
            .expect("the instance file to be present");

        let initial_population: NdTreeSolutionSet<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4> =
            NdTreeSolutionSet::new("test_population");
        let is_deterministic = true;

        let mut pareto_local_search =
            ParetoLocalSearch::new(&problem, &initial_population, 1..=5, is_deterministic);

        let solutions = pareto_local_search.run(MAX_ITERATIONS, MAX_DURATION);
        assert!(
            !solutions.is_empty(),
            "expected to have at least one solution for instance {instance_file}"
        );
    }
}

#[test]
fn test_4d_medium_instances() {
    for instance_file in MEDIUM_INSTANCES {
        let instance_path = Path::new(INSTANCES_PATH).join(instance_file);

        let problem = ProblemBitset::<4>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES)
            .expect("the instance file to be present");

        let initial_population: NdTreeSolutionSet<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4> =
            NdTreeSolutionSet::new("test_population");
        let is_deterministic = true;

        let mut pareto_local_search =
            ParetoLocalSearch::new(&problem, &initial_population, 1..=5, is_deterministic);

        let solutions = pareto_local_search.run(MAX_ITERATIONS, MAX_DURATION);
        assert!(
            !solutions.is_empty(),
            "expected to have at least one solution for instance {instance_file}"
        );
    }
}

#[test]
fn test_4d_large_instances() {
    for instance_file in LARGE_INSTANCES {
        let instance_path = Path::new(INSTANCES_PATH).join(instance_file);

        let problem = ProblemBitset::<4>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES)
            .expect("the instance file to be present");

        let initial_population: NdTreeSolutionSet<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4> =
            NdTreeSolutionSet::new("test_population");
        let is_deterministic = true;

        let mut pareto_local_search =
            ParetoLocalSearch::new(&problem, &initial_population, 1..=5, is_deterministic);

        let solutions = pareto_local_search.run(MAX_ITERATIONS, MAX_DURATION);
        assert!(
            !solutions.is_empty(),
            "expected to have at least one solution for instance {instance_file}"
        );
    }
}

// ============================================================================
// TWO-PHASE TESTS: Load initial population from JSON and run PLS
// ============================================================================

/// Test two-phase solving on small instances with different time ratios
/// This simulates: exact solver (replay from JSON) -> PLS with initial population
#[test]
#[ignore = "Ignored by default due to longer runtime"]
fn test_two_phase_4d_small_instances() {
    const TOTAL_TIME_S: f64 = 300.0; // Total time budget for small instances (5 minutes)
    const JSON_DATA_PATH: &str = "tests/data"; // Path to JSON solution files

    for instance_file in SMALL_INSTANCES {
        let instance_name = instance_file.strip_suffix(".dzn").unwrap();
        let instance_path = Path::new(INSTANCES_PATH).join(instance_file);
        let json_path = Path::new(JSON_DATA_PATH).join(format!("{instance_name}.json"));

        // Skip if JSON file doesn't exist
        if !json_path.exists() {
            println!("Skipping {instance_name}: JSON file not found");
            continue;
        }

        let problem = ProblemBitset::<4>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES)
            .expect("the instance file to be present");

        println!(
            "Problem loaded: {} images, {} universe elements",
            problem.num_images(),
            problem.universe_size
        );

        // Test a subset of ratios for small instances
        // Start with 100:0 (no PLS) to verify loading works, then test 50:50
        for (exact_pct, pls_pct) in [(100, 0), (50, 50)] {
            println!("\n=== Testing {instance_name} with ratio {exact_pct}:{pls_pct} ===");

            let exact_time = TOTAL_TIME_S * (f64::from(exact_pct) / 100.0);
            let pls_time = TOTAL_TIME_S * (f64::from(pls_pct) / 100.0);

            // Phase 1: Load initial population from JSON (simulating exact solver)
            let initial_population = if exact_pct > 0 {
                load_initial_population_from_json(&json_path, &problem, exact_time)
            } else {
                NdTreeSolutionSet::new("empty_initial")
            };

            println!(
                "Phase 1 complete: {} initial solutions",
                initial_population.len()
            );

            // Phase 2: Run PLS with initial population
            if pls_pct > 0 {
                println!(
                    "Starting PLS with {} seconds timeout and {} initial solutions",
                    pls_time,
                    initial_population.len()
                );

                let is_deterministic = true;
                let mut pareto_local_search =
                    ParetoLocalSearch::new(&problem, &initial_population, 1..=5, is_deterministic);

                let pls_duration = Duration::from_secs_f64(pls_time);
                let max_pls_iterations = 100_000; // Much higher limit for PLS
                let solutions = pareto_local_search.run(max_pls_iterations, pls_duration);

                println!(
                    "Phase 2 complete: {} total solutions (added {} new)",
                    solutions.len(),
                    solutions.len().saturating_sub(initial_population.len())
                );

                assert!(
                    !solutions.is_empty(),
                    "expected to have at least one solution for {instance_name} (ratio {exact_pct}:{pls_pct})"
                );
            } else {
                // 100:0 ratio - only exact phase
                assert!(
                    !initial_population.is_empty(),
                    "expected to have solutions from exact phase for {instance_name} (ratio 100:0)"
                );
            }
        }
    }
}

/// Test two-phase solving on medium instances with different time ratios
#[test]
#[ignore = "Ignored by default due to longer runtime"]
fn test_two_phase_4d_medium_instances() {
    const TOTAL_TIME_S: f64 = 3600.0; // 1 hour for medium instances
    const JSON_DATA_PATH: &str = "tests/data";

    for instance_file in MEDIUM_INSTANCES {
        let instance_name = instance_file.strip_suffix(".dzn").unwrap();
        let instance_path = Path::new(INSTANCES_PATH).join(instance_file);
        let json_path = Path::new(JSON_DATA_PATH).join(format!("{instance_name}.json"));

        if !json_path.exists() {
            println!("Skipping {instance_name}: JSON file not found");
            continue;
        }

        let problem = ProblemBitset::<4>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES)
            .expect("the instance file to be present");

        // Test all ratios for medium instances
        for (exact_pct, pls_pct) in TWO_PHASE_RATIOS {
            println!("\n=== Testing {instance_name} with ratio {exact_pct}:{pls_pct} ===");

            let exact_time = TOTAL_TIME_S * (f64::from(exact_pct) / 100.0);
            let pls_time = TOTAL_TIME_S * (f64::from(pls_pct) / 100.0);

            let initial_population = if exact_pct > 0 {
                load_initial_population_from_json(&json_path, &problem, exact_time)
            } else {
                NdTreeSolutionSet::new("empty_initial")
            };

            println!(
                "Phase 1 complete: {} initial solutions",
                initial_population.len()
            );

            if pls_pct > 0 {
                let is_deterministic = true;
                let mut pareto_local_search =
                    ParetoLocalSearch::new(&problem, &initial_population, 1..=5, is_deterministic);

                let pls_duration = Duration::from_secs_f64(pls_time);
                let solutions = pareto_local_search.run(MAX_ITERATIONS, pls_duration);

                println!("Phase 2 complete: {} total solutions", solutions.len());

                assert!(
                    !solutions.is_empty(),
                    "expected to have at least one solution for {instance_name} (ratio {exact_pct}:{pls_pct})"
                );
            } else {
                assert!(
                    !initial_population.is_empty(),
                    "expected to have solutions from exact phase for {instance_name} (ratio 100:0)"
                );
            }
        }
    }
}

/// Test two-phase solving on large instances with different time ratios
#[test]
#[ignore = "Ignored by default due to very long runtime"]
fn test_two_phase_4d_large_instances() {
    const TOTAL_TIME_S: f64 = 12000.0; // ~3.3 hours for large instances
    const JSON_DATA_PATH: &str = "tests/data";

    for instance_file in LARGE_INSTANCES {
        let instance_name = instance_file.strip_suffix(".dzn").unwrap();
        let instance_path = Path::new(INSTANCES_PATH).join(instance_file);
        let json_path = Path::new(JSON_DATA_PATH).join(format!("{instance_name}.json"));

        if !json_path.exists() {
            println!("Skipping {instance_name}: JSON file not found");
            continue;
        }

        let problem = ProblemBitset::<4>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES)
            .expect("the instance file to be present");

        for (exact_pct, pls_pct) in TWO_PHASE_RATIOS {
            println!("\n=== Testing {instance_name} with ratio {exact_pct}:{pls_pct} ===");

            let exact_time = TOTAL_TIME_S * (f64::from(exact_pct) / 100.0);
            let pls_time = TOTAL_TIME_S * (f64::from(pls_pct) / 100.0);

            let initial_population = if exact_pct > 0 {
                load_initial_population_from_json(&json_path, &problem, exact_time)
            } else {
                NdTreeSolutionSet::new("empty_initial")
            };

            println!(
                "Phase 1 complete: {} initial solutions",
                initial_population.len()
            );

            if pls_pct > 0 {
                let is_deterministic = true;
                let mut pareto_local_search =
                    ParetoLocalSearch::new(&problem, &initial_population, 1..=5, is_deterministic);

                let pls_duration = Duration::from_secs_f64(pls_time);
                let solutions = pareto_local_search.run(MAX_ITERATIONS, pls_duration);

                println!("Phase 2 complete: {} total solutions", solutions.len());

                assert!(
                    !solutions.is_empty(),
                    "expected to have at least one solution for {instance_name} (ratio {exact_pct}:{pls_pct})"
                );
            } else {
                assert!(
                    !initial_population.is_empty(),
                    "expected to have solutions from exact phase for {instance_name} (ratio 100:0)"
                );
            }
        }
    }
}
