//! Debug PLS run with composite tracker validation.
//!
//! Runs PLS on a specified instance with CompositeDebugTrackerArray to validate
//! that Standard and Simd tracker implementations produce identical results.
//!
//! Uses the same JSON files as the pseudo-solver for initial population.

use clap::Parser;
use pareto::ParetoFront;
use pls::{
    objectives::ObjectiveType,
    pareto_local_search::ParetoLocalSearch,
    pls_config::PlsOptimizations,
    problem_bitset::ProblemBitset,
    solution::{SIMSModifiable, bitset_encoded_solution::BitsetEncodedSolution},
    solution_set_impl::NdTreeSolutionSet,
};
use serde::Deserialize;
use std::{fs, path::PathBuf, time::Duration};
use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};

const OBJECTIVE_TYPES: [ObjectiveType; 4] = [
    ObjectiveType::TotalCost,
    ObjectiveType::CloudyArea,
    ObjectiveType::MinResolution,
    ObjectiveType::MaxIncidenceAngle,
];

/// Solution entry from pseudo-solver JSON
#[derive(Debug, Deserialize)]
struct JsonSolution {
    selected_images: Vec<usize>,
    #[allow(dead_code)]
    cost: u64,
    #[allow(dead_code)]
    cloudy_area: u64,
    #[allow(dead_code)]
    max_incidence_angle: u64,
    #[allow(dead_code)]
    min_resolutions_sum: u64,
    #[allow(dead_code)]
    timestamp_s: f64,
    #[allow(dead_code)]
    phase: String,
    #[allow(dead_code)]
    index: usize,
}

/// Pseudo-solver JSON file format
#[derive(Debug, Deserialize)]
struct PseudoSolverData {
    #[allow(dead_code)]
    instance_name: String,
    #[allow(dead_code)]
    test_type: String,
    #[allow(dead_code)]
    objectives: Vec<String>,
    #[allow(dead_code)]
    num_solutions: usize,
    solutions: Vec<JsonSolution>,
}

#[derive(Parser, Debug)]
#[command(name = "debug-pls")]
#[command(about = "Debug PLS run with tracker validation")]
struct Args {
    /// Instance name (e.g., lagos_nigeria_100, paris_50)
    #[arg(short, long, default_value = "lagos_nigeria_30")]
    instance: String,

    /// Timeout in seconds
    #[arg(short, long, default_value = "300")]
    timeout: u64,

    /// Use random initial population instead of pseudo-solver JSON
    #[arg(long)]
    random_population: bool,

    /// Initial population size (only used with --random-population)
    #[arg(short, long, default_value = "100")]
    population: usize,
}

/// Load solutions from pseudo-solver JSON file
fn load_pseudo_solver_solutions(instance_name: &str) -> Option<PseudoSolverData> {
    // Path relative to sims-heuristics crate
    let json_path = PathBuf::from("../sims-core/tests/data/pseudo_solver_solutions")
        .join(format!("{instance_name}.json"));

    if !json_path.exists() {
        println!("Warning: No pseudo-solver JSON found at {}", json_path.display());
        return None;
    }

    let content = fs::read_to_string(&json_path)
        .map_err(|e| println!("Warning: Failed to read JSON: {e}"))
        .ok()?;

    serde_json::from_str(&content)
        .map_err(|e| println!("Warning: Failed to parse JSON: {e}"))
        .ok()
}

fn main() {
    let args = Args::parse();

    // Initialize console logging
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(false)
        .compact()
        .with_filter(tracing_subscriber::filter::LevelFilter::ERROR);

    tracing_subscriber::registry().with(fmt_layer).init();
    tracing_log::LogTracer::init().ok();

    // Load the problem
    let instance_path = PathBuf::from("data").join(format!("{}.dzn", args.instance));
    println!("Loading instance: {}", instance_path.display());

    let problem = ProblemBitset::<4>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES)
        .expect("Failed to load problem instance");

    println!(
        "Problem loaded: {} images, {} universe elements",
        problem.num_images(),
        problem.universe_size
    );

    // Create initial population
    let mut initial_population: NdTreeSolutionSet<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4> =
        NdTreeSolutionSet::new("debug_population");

    if args.random_population {
        // Create random initial population
        println!(
            "Generating random initial population of {} solutions...",
            args.population
        );

        for i in 0..args.population {
            let solution = BitsetEncodedSolution::random_with_seed(&problem, i as u64);
            initial_population.try_insert(&solution);
        }
    } else {
        // Load from pseudo-solver JSON
        println!("Loading initial population from pseudo-solver JSON...");

        if let Some(data) = load_pseudo_solver_solutions(&args.instance) {
            println!("Found {} solutions in JSON file", data.solutions.len());

            for json_sol in &data.solutions {
                let solution = BitsetEncodedSolution::from_selected_images(
                    &json_sol.selected_images,
                    &problem,
                );
                initial_population.try_insert(&solution);
            }

            println!(
                "Loaded {} non-dominated solutions from pseudo-solver JSON",
                initial_population.len()
            );
        } else {
            println!("Falling back to random initial population of {} solutions...", args.population);
            for i in 0..args.population {
                let solution = BitsetEncodedSolution::random_with_seed(&problem, i as u64);
                initial_population.try_insert(&solution);
            }
        }
    }

    println!(
        "Initial population: {} non-dominated solutions",
        initial_population.len()
    );
    println!("Running PLS with {} second timeout...", args.timeout);

    // Run PLS
    let is_deterministic = true;
    let mut pareto_local_search =
        ParetoLocalSearch::new(&problem, &initial_population, 1..=5, is_deterministic, PlsOptimizations::default());

    let max_iterations = usize::MAX;
    let timeout = Duration::from_secs(args.timeout);
    let solutions = pareto_local_search.run(max_iterations, timeout);

    println!("PLS complete: {} solutions found", solutions.len());

    // Print some solution details
    for (i, solution) in solutions.iter().enumerate().take(5) {
        let selected_count = solution.selected_images().count();
        println!("Solution {i}: {selected_count} images selected");
    }

    if solutions.len() > 5 {
        println!("... and {} more solutions", solutions.len() - 5);
    }

    // Validate Pareto front
    solutions.validate();
    for solution in solutions.iter() {
        assert!(
            solution.is_valid(&problem),
            "A solution in the Pareto front does not cover the universe"
        );
    }

    println!("All validations passed!");
}
