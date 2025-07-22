mod file_io;

use clap::{ArgAction, Parser};

use log::debug;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use std::{
    ops::RangeInclusive,
    path::PathBuf,
    time::{Duration, Instant},
};

use pls::pareto_local_search::ParetoLocalSearch;
use pls::solution_set_impl::BTreeSolutionSet;
use pls::{problem::Problem, solution::EncodedSolution, solution_set::SolutionSet};

const INITIAL_POPULATION_SIZE: usize = 100;
const MAX_ITERATIONS: usize = 50000;
const MAX_DURATION: &str = "240s";
const NEIGHBORHOOD_SIZE_RANGE: RangeInclusive<u32> = 1..=6;
const TEST_SEED: u64 = 1_234_567_890;
const NUM_OBJECTIVES: usize = 2;

#[derive(Parser)]
#[command(about = "Pareto Local Search solver for the Satellite Image Selection Problem", long_about = None)]
struct Cli {
    #[arg(
        name = "PROBLEM_PATH",
        short = 'p',
        long = "problem",
        help = "Path to the problem instance file in MiniZinc format"
    )]
    problem: PathBuf,
    #[arg(
        name = "INITIAL_POPULATION_DIR",
        short = 'i',
        long = "initial-population",
        help = "Path to the CSV file with initial population solutions"
    )]
    initial_population: Option<PathBuf>,
    #[arg(
        name = "OUTPUT_PATH",
        short = 'o',
        long = "output",
        help = "Path to output file where results will be written"
    )]
    output: PathBuf,
    #[arg(
        name = "TIMEOUT",
        short = 't',
        long = "timeout",
        help = "Timeout for the solver in human readable format (e.g. 300ms, 50s, 5m, 1h, 2d)",
        value_parser = humantime::parse_duration,
        default_value = MAX_DURATION
    )]
    timeout: Duration,
    #[arg(
        short = 'd',
        long = "deterministic",
        help = "Make the solver deterministic, i.e. eliminate all randomness",
        action = ArgAction::SetTrue,
        default_value_t = false
    )]
    is_deterministic: bool,
    #[arg(
        short = 'n',
        long = "max-iterations",
        help = "Maximum number of iterations for the solver",
        default_value_t = MAX_ITERATIONS
    )]
    max_iterations: usize,
    #[arg(
        long = "chrome-trace",
        help = "Path to output Chrome trace file for viewing in chrome://tracing"
    )]
    chrome_trace: Option<PathBuf>,
}

#[allow(clippy::too_many_lines)]
fn main() {
    let args = Cli::parse();
    let start_time = Instant::now();

    // Initialize tracing with logging handler as default
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,pls=debug"));

    // Always add the console logging layer
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(false)
        .with_level(true)
        .with_thread_ids(true)
        .with_line_number(true)
        .with_file(true)
        .compact();

    // Optionally add Chrome tracing layer
    if let Some(chrome_trace_path) = &args.chrome_trace {
        let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
            .file(chrome_trace_path)
            .build();

        tracing_subscriber::registry()
            .with(fmt_layer)
            .with(chrome_layer)
            .with(env_filter)
            .init();

        // Keep the guard alive for the duration of the program
        std::mem::forget(guard);
    } else {
        tracing_subscriber::registry()
            .with(fmt_layer)
            .with(env_filter)
            .init();
    }

    // Bridge log crate to tracing (ignore errors if already set)
    let _ = tracing_log::LogTracer::init();

    debug!("Starting Pareto Local Search solver");

    let output_dir = args.output.parent().unwrap();
    if !output_dir.exists() {
        eprintln!("Output directory does not exist: {}", output_dir.display());
        std::process::exit(1);
    }

    if let Some(initial_population_path) = &args.initial_population
        && !initial_population_path.exists()
    {
        eprintln!(
            "Initial population file does not exist: {}",
            initial_population_path.display()
        );
        std::process::exit(1);
    }

    debug!(
        "Loading problem instance from file: {}",
        args.problem.display()
    );
    let sims_problem_instance = Problem::<NUM_OBJECTIVES>::from_minizinc_datafile(&args.problem);

    debug!("Initializing initial solution set");
    let initial_solution_set: BTreeSolutionSet<EncodedSolution<NUM_OBJECTIVES>, NUM_OBJECTIVES> =
        if let Some(initial_population_csv) = &args.initial_population {
            debug!(
                "Loading initial solutions from file: {}",
                initial_population_csv.display()
            );
            let initial_solutions =
                file_io::solution_list_from_csv(initial_population_csv, &sims_problem_instance);

            // debug!("Checking validity of initial solutions");
            // let (valid_initial_solutions, invalid_initial_solutions): (Vec<_>, Vec<_>) =
            //     initial_solutions
            //         .into_iter()
            //         .partition(|solution| solution.is_valid(&sims_problem_instance));
            // if !invalid_initial_solutions.is_empty() {
            //     file_io::dump_invalid_initial_solutions(
            //         invalid_initial_solutions,
            //         &args.problem,
            //         &args.output,
            //         args.timeout,
            //     );
            // }
            // if !valid_initial_solutions.is_empty() {
            if !initial_solutions.is_empty() {
                BTreeSolutionSet::from_iter(initial_solutions)
            } else if args.is_deterministic {
                BTreeSolutionSet::random_with_seed(
                    INITIAL_POPULATION_SIZE,
                    &sims_problem_instance,
                    TEST_SEED,
                )
            } else {
                BTreeSolutionSet::random(INITIAL_POPULATION_SIZE, &sims_problem_instance)
            }
        } else if args.is_deterministic {
            BTreeSolutionSet::random_with_seed(
                INITIAL_POPULATION_SIZE,
                &sims_problem_instance,
                TEST_SEED,
            )
        } else {
            BTreeSolutionSet::random(INITIAL_POPULATION_SIZE, &sims_problem_instance)
        };

    debug!("Initial solution set:");

    for solution in initial_solution_set.iter() {
        debug!("Initial solution: {solution:?}");
    }

    let mut pareto_local_search = ParetoLocalSearch::new(
        &sims_problem_instance,
        &initial_solution_set,
        NEIGHBORHOOD_SIZE_RANGE,
        args.is_deterministic,
    );
    let final_solution_set = pareto_local_search.run(args.max_iterations, args.timeout);

    #[cfg(feature = "plotting")]
    pls::plotting::draw_solutions_plot_with_problem(
        &pareto_local_search.explored_solutions,
        &sims_problem_instance,
    );

    let final_solutions: Vec<EncodedSolution<NUM_OBJECTIVES>> =
        final_solution_set.into_iter().collect();

    let non_dominated_points = pareto_local_search.explored_solutions.non_dominated();
    debug_assert_eq!(final_solutions.len(), non_dominated_points.len());

    let time_list: Vec<f32> = final_solutions
        .iter()
        .map(|solution| {
            pareto_local_search
                .explored_solutions
                .get_solution_fingerprint(solution)
                .unwrap()
                .time
                .as_secs_f32()
        })
        .collect();

    file_io::dump_pareto_front_snapshots(
        pareto_local_search
            .explored_solutions
            .pareto_front_snapshots,
        &output_dir.join("pareto_front_snapshots.txt"),
    );

    let elapsed_time_s = start_time.elapsed().as_secs();
    file_io::append_solutions_to_csv(
        &args.output,
        &final_solutions,
        &sims_problem_instance,
        args.timeout.as_secs(),
        &time_list,
        elapsed_time_s,
    );
}
