use pareto::ParetoFront;
use pls::{
    objectives::ObjectiveType, pareto_local_search::ParetoLocalSearch,
    pls_config::PlsOptimizations,
    problem_bitset::ProblemBitset, solution::{SIMSModifiable, bitset_encoded_solution::BitsetEncodedSolution},
    solution_set_impl::NdTreeSolutionSet,
};
use std::{path::Path, time::Duration};
use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};

const OBJECTIVE_TYPES: [ObjectiveType; 4] = [
    ObjectiveType::TotalCost,
    ObjectiveType::CloudyArea,
    ObjectiveType::MinResolution,
    ObjectiveType::MaxIncidenceAngle,
];

fn main() {
    const INITIAL_POPULATION_SIZE: usize = 100;

    // Initialize console logging only (Perfetto disabled temporarily)
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(false)
        .compact()
        .with_filter(tracing_subscriber::filter::LevelFilter::ERROR);

    tracing_subscriber::registry().with(fmt_layer).init();

    // Bridge log crate to tracing
    tracing_log::LogTracer::init().ok();
    println!("Loading lagos_nigeria_30 instance...");

    // Load the problem
    let instance_path = Path::new("data").join("lagos_nigeria_30.dzn");
    let problem = ProblemBitset::<4>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES)
        .expect("Failed to load problem instance");

    println!(
        "Problem loaded: {} images, {} universe elements",
        problem.num_images(),
        problem.universe_size
    );

    // Create random initial population
    println!("Generating random initial population of {INITIAL_POPULATION_SIZE} solutions...");

    // Generate random solutions using the problem instance with deterministic seeds
    let mut initial_population: NdTreeSolutionSet<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4> =
        NdTreeSolutionSet::new("debug_population");

    for i in 0..INITIAL_POPULATION_SIZE {
        let solution = BitsetEncodedSolution::random_with_seed(&problem, i as u64);
        initial_population.try_insert(&solution);
    }

    println!(
        "Initial population created with {} non-dominated solutions",
        initial_population.len()
    );
    println!("Running PLS with 5 minute timeout...");

    // Run PLS
    let is_deterministic = true;
    let mut pareto_local_search =
        ParetoLocalSearch::new(&problem, &initial_population, 1..=5, is_deterministic, PlsOptimizations::default());

    let max_iterations = usize::MAX;
    let timeout = Duration::from_secs(300);
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
}
