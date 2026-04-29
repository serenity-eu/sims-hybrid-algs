mod file_io;

use clap::{ArgAction, Parser, ValueEnum};

use log::debug;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use std::{
    ops::RangeInclusive,
    path::PathBuf,
    time::{Duration, Instant},
};

use pareto::{ParetoFront, RandomCollection};
use pls::solution_set_impl::BTreeSolutionSet;
use pls::{
    PlsOptimizations,
    objectives::ObjectiveType,
    pareto_local_search::ParetoLocalSearch,
    pls_config::{ScalarizedSelectionSource, SolutionSelectionMode},
};
use pls::{
    problem_bitset::ProblemBitset, solution_impl::bitset_encoded_solution::BitsetEncodedSolution,
};

const INITIAL_POPULATION_SIZE: usize = 100;
const MAX_ITERATIONS: usize = 50000;
const MAX_DURATION: &str = "240s";
const NEIGHBORHOOD_SIZE_RANGE: RangeInclusive<u32> = 1..=6;
const TEST_SEED: u64 = 1_234_567_890;
const NUM_OBJECTIVES: usize = 2;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliSolutionSelectionMode {
    RandomShuffle,
    DiverseProbe,
    #[cfg(feature = "scalarized_selection")]
    ScalarizedChebycheff,
    #[cfg(feature = "scalarized_selection")]
    DiverseThenScalarizedChebycheff,
}

impl CliSolutionSelectionMode {
    const fn to_runtime(self) -> SolutionSelectionMode {
        match self {
            Self::RandomShuffle => SolutionSelectionMode::RandomShuffle,
            Self::DiverseProbe => SolutionSelectionMode::DiverseProbe,
            #[cfg(feature = "scalarized_selection")]
            Self::ScalarizedChebycheff => SolutionSelectionMode::ScalarizedChebycheff,
            #[cfg(feature = "scalarized_selection")]
            Self::DiverseThenScalarizedChebycheff => {
                SolutionSelectionMode::DiverseThenScalarizedChebycheff
            }
        }
    }
}

#[cfg(feature = "scalarized_selection")]
#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliScalarizedSelectionSource {
    Population,
    Archive,
}

#[cfg(feature = "scalarized_selection")]
impl CliScalarizedSelectionSource {
    const fn to_runtime(self) -> ScalarizedSelectionSource {
        match self {
            Self::Population => ScalarizedSelectionSource::Population,
            Self::Archive => ScalarizedSelectionSource::Archive,
        }
    }
}

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

    #[arg(
        long = "solution-selection",
        value_enum,
        default_value = "random-shuffle",
        help = "Parent-solution selection policy for neighborhood exploration"
    )]
    solution_selection: CliSolutionSelectionMode,

    #[arg(
        long = "diverse-probe-budget",
        help = "Number of parent solutions to probe when using diverse probing"
    )]
    diverse_probe_budget: Option<usize>,

    #[cfg(feature = "scalarized_selection")]
    #[arg(
        long = "use-nd-tree-scalarized-query",
        action = ArgAction::Set,
        default_value_t = true,
        help = "Use ND-tree accelerated branch-and-bound queries for scalarized archive selection"
    )]
    use_nd_tree_scalarized_query: bool,

    #[cfg(feature = "scalarized_selection")]
    #[arg(
        long = "scalarized-selection-source",
        value_enum,
        default_value = "population",
        help = "Source pool used by scalarized parent selection"
    )]
    scalarized_selection_source: CliScalarizedSelectionSource,

    #[cfg(feature = "scalarized_selection")]
    #[arg(
        long = "scalarized-parent-budget",
        help = "Maximum number of parent solutions selected per step by scalarized selection"
    )]
    scalarized_parent_budget: Option<usize>,

    #[cfg(feature = "scalarized_selection")]
    #[arg(
        long = "scalarized-weight-samples",
        default_value_t = 1,
        help = "Number of random weight vectors sampled per step for scalarized selection"
    )]
    scalarized_weight_samples: usize,

    #[cfg(feature = "scalarized_selection")]
    #[arg(
        long = "scalarized-rho",
        default_value_t = 1e-3,
        help = "Augmentation coefficient for weighted Chebycheff scalarization"
    )]
    scalarized_rho: f64,
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

    // Chrome tracing is not available in this version
    if args.chrome_trace.is_some() {
        eprintln!("Warning: Chrome tracing is not available. Ignoring --chrome-trace option.");
    }

    tracing_subscriber::registry()
        .with(fmt_layer)
        .with(env_filter)
        .init();

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
    let sims_problem_instance = ProblemBitset::<NUM_OBJECTIVES>::from_minizinc_datafile(
        &args.problem,
        [ObjectiveType::TotalCost, ObjectiveType::CloudyArea],
    )
    .expect("Failed to load problem instance");

    debug!("Initializing initial solution set");
    let initial_solution_set: BTreeSolutionSet<
        BitsetEncodedSolution<ProblemBitset<NUM_OBJECTIVES>, NUM_OBJECTIVES>,
        NUM_OBJECTIVES,
    > = if let Some(initial_population_csv) = &args.initial_population {
        debug!(
            "Loading initial solutions from file: {}",
            initial_population_csv.display()
        );
        let initial_solutions_indices =
            file_io::solution_list_from_csv::<NUM_OBJECTIVES>(initial_population_csv);
        let initial_solutions = initial_solutions_indices
            .into_iter()
            .map(|selected_images| {
                BitsetEncodedSolution::from_selected_images(
                    &selected_images,
                    &sims_problem_instance,
                )
            })
            .collect::<Vec<_>>();

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
            BTreeSolutionSet::random_with_seed(INITIAL_POPULATION_SIZE, TEST_SEED)
        } else {
            BTreeSolutionSet::random(INITIAL_POPULATION_SIZE)
        }
    } else if args.is_deterministic {
        BTreeSolutionSet::random_with_seed(INITIAL_POPULATION_SIZE, TEST_SEED)
    } else {
        BTreeSolutionSet::random(INITIAL_POPULATION_SIZE)
    };

    debug!("Initial solution set:");

    for solution in initial_solution_set.iter() {
        debug!("Initial solution: {solution:?}");
    }

    let mut optimizations = PlsOptimizations::default();
    optimizations.solution_selection_mode = args.solution_selection.to_runtime();
    optimizations.use_diverse_probing = matches!(
        args.solution_selection,
        CliSolutionSelectionMode::DiverseProbe
    );
    optimizations.diverse_probe_budget = args.diverse_probe_budget;
    #[cfg(feature = "scalarized_selection")]
    {
        optimizations.use_nd_tree_scalarized_query = args.use_nd_tree_scalarized_query;
        optimizations.scalarized_selection_source = args.scalarized_selection_source.to_runtime();
        optimizations.scalarized_parent_budget = args.scalarized_parent_budget;
        optimizations.scalarized_weight_samples = args.scalarized_weight_samples;
        optimizations.scalarized_rho = args.scalarized_rho;
    }

    let mut pareto_local_search = ParetoLocalSearch::new(
        &sims_problem_instance,
        &initial_solution_set,
        NEIGHBORHOOD_SIZE_RANGE,
        args.is_deterministic,
        optimizations,
    );
    let final_solution_set = pareto_local_search.run(args.max_iterations, args.timeout);

    #[cfg(feature = "plotting")]
    pls::plotting::draw_solutions_plot_with_problem(
        &pareto_local_search.explored_solutions,
        &sims_problem_instance,
    );

    let final_solutions: Vec<BitsetEncodedSolution<ProblemBitset<NUM_OBJECTIVES>, NUM_OBJECTIVES>> =
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
                .timestamp
                .as_secs_f32()
        })
        .collect();

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
