#[cfg(feature = "milp")]
use augmecon::{
    sims_problem::{SimsInstance, SimsObjective},
    GpbaA, GpbaConfig, HasObjectives, Options,
};
use log::{debug, error, info};
use pareto::ParetoFront;
use pls::explored_solutions_data::SolutionFingerprint;
use pls::pareto_local_search::ParetoLocalSearch;
use pls::pls_config::{PlsFlags, PlsOptimizations};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
#[cfg(feature = "milp")]
use std::collections::HashSet;
use std::str::FromStr;
use std::{iter::IntoIterator, ops::RangeInclusive, time::Duration};

use crate::problem::SimsDiscreteProblem;
#[cfg(feature = "milp")]
use crate::solution::Solution;
use crate::solution::SolvingResult;
use crate::trace;

/// Enum representing the type of solution set (Pareto archive) to use
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolutionSetType {
    /// ND-Tree based solution set (default, best for 3D/4D)
    NdTree,
    /// Linked list based solution set
    LinkedList,
    /// Vector based solution set
    Vector,
}

impl FromStr for SolutionSetType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "nd-tree" => Ok(SolutionSetType::NdTree),
            "linked-list" => Ok(SolutionSetType::LinkedList),
            "vector" => Ok(SolutionSetType::Vector),
            _ => Err(format!(
                "Invalid solution set type '{}'. Valid options: nd-tree, linked-list, vector",
                s
            )),
        }
    }
}

impl SolutionSetType {
    /// Returns the string representation of the solution set type
    pub fn as_str(&self) -> &'static str {
        match self {
            SolutionSetType::NdTree => "nd-tree",
            SolutionSetType::LinkedList => "linked-list",
            SolutionSetType::Vector => "vector",
        }
    }
}

/// Configuration for MILP solver
#[cfg(feature = "milp")]
#[pyclass]
#[derive(Debug, Clone)]
pub struct MilpConfig {
    #[pyo3(get, set)]
    pub objectives: Vec<String>,
    #[pyo3(get, set)]
    pub grid_points: usize,
    #[pyo3(get, set)]
    pub bypass_coefficient: bool,
    #[pyo3(get, set)]
    pub early_exit: bool,
    #[pyo3(get, set)]
    pub flag_array: bool,
    #[pyo3(get, set)]
    pub solver_name: String,
}

#[cfg(feature = "milp")]
#[pymethods]
impl MilpConfig {
    #[new]
    #[pyo3(signature = (
        objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string()],
        grid_points=50,
        bypass_coefficient=true,
        early_exit=true,
        flag_array=true,
        solver_name="highs".to_string()
    ))]
    pub fn new(
        objectives: Vec<String>,
        grid_points: usize,
        bypass_coefficient: bool,
        early_exit: bool,
        flag_array: bool,
        solver_name: String,
    ) -> Self {
        Self {
            objectives,
            grid_points,
            bypass_coefficient,
            early_exit,
            flag_array,
            solver_name,
        }
    }
}

/// Configuration for PLS solver
#[pyclass]
#[derive(Debug, Clone)]
pub struct PlsConfig {
    #[pyo3(get, set)]
    pub objectives: Vec<String>,
    #[pyo3(get, set)]
    pub max_iterations: usize,
    #[pyo3(get, set)]
    pub is_deterministic: bool,
    #[pyo3(get, set)]
    pub initial_population_size: usize,
    #[pyo3(get, set)]
    pub neighborhood_size_min: u32,
    #[pyo3(get, set)]
    pub neighborhood_size_max: u32,
    #[pyo3(get, set)]
    pub plots: bool,
    #[pyo3(get, set)]
    pub plot_output_path: Option<String>,
}

#[pymethods]
impl PlsConfig {
    #[new]
    #[expect(
        clippy::too_many_arguments,
        reason = "Configuration struct needs all these parameters"
    )]
    #[pyo3(signature = (
        objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string()],
        max_iterations=50000,
        is_deterministic=false,
        initial_population_size=100,
        neighborhood_size_min=1,
        neighborhood_size_max=6,
        plots=false,
        plot_output_path=None
    ))]
    pub fn new(
        objectives: Vec<String>,
        max_iterations: usize,
        is_deterministic: bool,
        initial_population_size: usize,
        neighborhood_size_min: u32,
        neighborhood_size_max: u32,
        plots: bool,
        plot_output_path: Option<String>,
    ) -> Self {
        Self {
            objectives,
            max_iterations,
            is_deterministic,
            initial_population_size,
            neighborhood_size_min,
            neighborhood_size_max,
            plots,
            plot_output_path,
        }
    }
}

/// Wrapper for shared Vec<u8> buffer that implements Write
struct SharedVecWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

impl std::io::Write for SharedVecWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.0.lock().unwrap().flush()
    }
}

/// Monolithic PLS solver that handles 2D, 3D, and 4D optimization in a single function
#[expect(
    clippy::too_many_arguments,
    reason = "It's okay for Python API to have many parameters"

)]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string()],
    plots=false,
    plot_output_path=None,
    timeout=Duration::from_secs(240),
    max_iterations=50000,
    is_deterministic=false,
    initial_population_size=100,
    initial_population=None,
    neighborhood_size_min=1,
    neighborhood_size_max=6,
    trace=true,
    objective_bounds=None,
    include_dominated=false,
    pareto_archive="nd-tree".to_string(),
    profiling_trace=false,
    parallel=false,
    num_parallel_threads=0usize,
    neighborhood_budget=None,
    use_checkpoint=true,
    use_ranked_candidates=true,
    max_k1_candidates=15usize,
    probing_budget=None,
    use_greedy_initial_population=true,
    // Default OFF. Perturbation restart keeps an exhausted search alive by
    // re-injecting perturbed archive solutions, which changes both the search
    // trajectory and how the wall-clock budget is consumed. Callers that want
    // it must ask for it, so an omitted argument cannot silently put a
    // measurement into a different regime than the experiments it is compared
    // against.
    use_perturbation_restart=false,
    use_diverse_probing=false,
    diverse_probe_budget=None,
    use_nd_tree_scalarized_query=true,
    solution_selection_mode=None,
    scalarized_selection_source=None,
    scalarized_parent_budget=None,
    scalarized_weight_samples=None,
    scalarized_rho=None
))]
#[allow(
    unused_variables,
    reason = "plot_output_path is used only when plotting feature is enabled"
)]
pub fn solve_with_pls(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: Vec<String>,
    plots: bool,
    plot_output_path: Option<String>,
    timeout: Duration,
    max_iterations: usize,
    is_deterministic: bool,
    initial_population_size: usize,
    initial_population: Option<Vec<crate::solution::Solution>>,
    neighborhood_size_min: u32,
    neighborhood_size_max: u32,
    trace: bool,
    objective_bounds: Option<Vec<Vec<u64>>>,
    include_dominated: bool,
    pareto_archive: String,
    profiling_trace: bool,
    parallel: bool,
    num_parallel_threads: usize,
    neighborhood_budget: Option<usize>,
    use_checkpoint: bool,
    use_ranked_candidates: bool,
    max_k1_candidates: usize,
    probing_budget: Option<usize>,
    use_greedy_initial_population: bool,
    use_perturbation_restart: bool,
    use_diverse_probing: bool,
    diverse_probe_budget: Option<usize>,
    use_nd_tree_scalarized_query: bool,
    solution_selection_mode: Option<String>,
    scalarized_selection_source: Option<String>,
    scalarized_parent_budget: Option<usize>,
    scalarized_weight_samples: Option<usize>,
    scalarized_rho: Option<f64>,
) -> PyResult<SolvingResult> {
    // Setup Chrome tracing if requested
    let (_chrome_guard, _profiling_buffer) = if profiling_trace {
        // Use Arc<Mutex<Vec<u8>>> for shared in-memory buffer
        use std::sync::{Arc, Mutex};
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let writer_buffer = Arc::clone(&buffer);

        let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
            .writer(SharedVecWriter(writer_buffer))
            .include_args(true)
            .build();

        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;
        let subscriber = tracing_subscriber::registry().with(chrome_layer);

        // Try to set subscriber, but don't fail if one is already set
        // This allows multiple tests to run in the same process
        match subscriber.try_init() {
            Ok(()) => info!("Chrome tracing profiling enabled"),
            Err(_) => {
                // Subscriber already set, just log a warning
                log::warn!(
                    "Global tracing subscriber already set, profiling may not work correctly"
                );
            }
        }
        (Some(guard), Some(buffer))
    } else {
        (None, None)
    };

    // Convert string to enum
    let solution_set_type = pareto_archive
        .parse::<SolutionSetType>()
        .map_err(PyValueError::new_err)?;

    debug!(
        "solve_with_pls called with {} objectives, solution_set_type={:?}, profiling_trace={}",
        objectives.len(),
        solution_set_type,
        profiling_trace
    );

    // Validate number of objectives first
    if objectives.len() < 2 {
        return Err(PyValueError::new_err(format!(
            "At least 2 objectives are required for multi-objective optimization. Found: {}",
            objectives.len()
        )));
    }

    // Validate objectives
    let valid_objectives = [
        "min_cost",
        "cloud_coverage",
        "min_resolution",
        "min_max_incidence_angle",
    ];
    for obj in &objectives {
        if !valid_objectives.contains(&obj.as_str()) {
            return Err(PyValueError::new_err(format!(
                "Invalid objective '{obj}'. Valid objectives are: {valid_objectives:?}"
            )));
        }
    }

    let timeout_seconds = timeout.as_secs_f64();
    let initial_pop_info = match &initial_population {
        Some(pop) => {
            info!(
                "PLS starting with {} solutions from initial population",
                pop.len()
            );
            format!("provided {} solutions", pop.len())
        }
        None => {
            info!(
                "PLS starting with random initial population of size {}",
                initial_population_size
            );
            format!("random generation size {}", initial_population_size)
        }
    };
    let bounds_info = match &objective_bounds {
        Some(bounds) => format!("provided bounds: {:?}", bounds),
        None => "no bounds provided".to_string(),
    };
    info!(
        "Starting PLS algorithm with {} objectives: {objectives:?}, plots: {plots}, timeout: {timeout_seconds}s, max_iterations: {max_iterations}, deterministic: {is_deterministic}, initial_population: {initial_pop_info}, neighborhood: {neighborhood_size_min}..{neighborhood_size_max}, objective_bounds: {bounds_info}, pareto_archive: '{}'",
        objectives.len(),
        pareto_archive
    );

    let neighborhood_size_range: RangeInclusive<u32> =
        neighborhood_size_min..=neighborhood_size_max;

    // Convert to PLS problem format - common for all dimensions
    // Note: sims_instance already has 0-based indices from Python, and
    // from_raw_with_objectives expects 0-based indices, so we pass them directly
    let raw_instance = pls::problem::SIMSProblemInstanceRaw {
        name: "python_instance".to_string(),
        num_images: sims_instance.num_images,
        universe_size: sims_instance.universe,
        images: sims_instance.images.clone(),
        costs: sims_instance.costs.iter().map(|&c| c as u64).collect(),
        clouds: sims_instance.clouds.clone(),
        areas: sims_instance.areas.iter().map(|&a| a as u64).collect(),
        max_cloud_area: sims_instance.max_cloud_area as u64,
        resolution: sims_instance.resolution.iter().map(|&r| r as u64).collect(),
        incidence_angle: sims_instance
            .incidence_angle
            .iter()
            .map(|&i| i as u64)
            .collect(),
    };

    debug!(
        "Created PLS problem: {} images, universe size {}",
        sims_instance.num_images, sims_instance.universe
    );

    // Branch based on number of objectives and handle each case inline
    match objectives.len() {
        2 => {
            use pls::problem_bitset::ProblemBitset;
            use pls::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;
            use pls::solution_set_impl::{BTreeSolutionSet, LinkedListSolutionSet, VecSolutionSet};

            debug!(
                "Using 2D optimization with objectives: {objectives:?}, solution_set_type: {:?}",
                solution_set_type
            );

            // Create 2D objective definitions inline
            let mut objective_definitions = [
                pls::objectives::ObjectiveType::TotalCost,
                pls::objectives::ObjectiveType::CloudyArea,
            ];
            for (i, obj_name) in objectives.iter().enumerate() {
                objective_definitions[i] = match obj_name.as_str() {
                    "min_cost" => pls::objectives::ObjectiveType::TotalCost,
                    "cloud_coverage" => pls::objectives::ObjectiveType::CloudyArea,
                    "min_resolution" => pls::objectives::ObjectiveType::MinResolution,
                    "min_max_incidence_angle" => pls::objectives::ObjectiveType::MaxIncidenceAngle,
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "Unknown objective: {}",
                            obj_name
                        )))
                    }
                };
            }

            let mut pls_problem =
                ProblemBitset::from_raw_with_objectives(&raw_instance, objective_definitions);

            // Set objective bounds if provided
            if let Some(ref bounds) = objective_bounds {
                let bounds_vec: Vec<[u64; 2]> = bounds
                    .iter()
                    .map(|b| {
                        if b.len() != 2 {
                            return Err(PyValueError::new_err(format!(
                                "Each objective bound must have exactly 2 elements [min, max], got {}",
                                b.len()
                            )));
                        }
                        Ok([b[0], b[1]])
                    })
                    .collect::<PyResult<Vec<[u64; 2]>>>()?;
                let bounds_array: [[u64; 2]; 2] = bounds_vec.try_into().map_err(|_| {
                    PyValueError::new_err("Expected exactly 2 objective bounds for 2D problem")
                })?;
                pls_problem.set_objective_bounds(bounds_array);
            }

            // Macro to handle different solution set types for 2D
            // Returns (final_solutions_vec, explored_solutions_vec)
            macro_rules! run_pls_2d_with_archive {
                ($SolutionSetType:ty, $archive_name:expr) => {{
                    // Create initial population manually for 2D
                    let mut initial_solution_set = <$SolutionSetType>::new("initial_2d_solutions");

                    if let Some(provided_population) = &initial_population {
                        // Always use provided population and generate additional random solutions
                        info!("Using provided initial population of {} solutions and generating {} random solutions for 2D PLS",
                              provided_population.len(), initial_population_size);
                        for solution in provided_population {
                            let selected_images: Vec<usize> = solution.selected_images.iter().cloned().collect();
                            let pls_solution = BitsetEncodedSolution::from_selected_images(&selected_images, &pls_problem);
                            initial_solution_set.try_insert(&pls_solution);
                        }
                        // Generate additional random solutions
                        for i in 0..initial_population_size {
                            let random_solution = if is_deterministic {
                                BitsetEncodedSolution::random_with_seed(&pls_problem, 1_234_567_890u64.wrapping_add(i as u64))
                            } else {
                                BitsetEncodedSolution::random(&pls_problem)
                            };
                            initial_solution_set.try_insert(&random_solution);
                        }
                    } else {
                        // Generate random initial population
                        info!("Generating random initial population of {} solutions for 2D PLS", initial_population_size);
                        for i in 0..initial_population_size {
                            let random_solution = if is_deterministic {
                                BitsetEncodedSolution::random_with_seed(&pls_problem, 1_234_567_890u64.wrapping_add(i as u64))
                            } else {
                                BitsetEncodedSolution::random(&pls_problem)
                            };
                            initial_solution_set.try_insert(&random_solution);
                        }
                    }

                    // Create and run 2D PLS
                    let mut optimizations = PlsOptimizations {
                        neighborhood_budget,
                        flags: PlsFlags::from_pairs([
                            (PlsFlags::USE_CHECKPOINT, use_checkpoint),
                            (PlsFlags::USE_RANKED_CANDIDATES, use_ranked_candidates),
                            (
                                PlsFlags::USE_GREEDY_INITIAL_POPULATION,
                                use_greedy_initial_population,
                            ),
                            (PlsFlags::USE_PERTURBATION_RESTART, use_perturbation_restart),
                            (PlsFlags::USE_DIVERSE_PROBING, use_diverse_probing),
                            (
                                PlsFlags::USE_ND_TREE_SCALARIZED_QUERY,
                                use_nd_tree_scalarized_query,
                            ),
                        ]),
                        max_k1_candidates,
                        probing_budget,
                        diverse_probe_budget,
                        ..PlsOptimizations::default()
                    };

                    if let Some(mode) = &solution_selection_mode {
                        optimizations.solution_selection_mode = match mode.as_str() {
                            "random-shuffle" => pls::pls_config::SolutionSelectionMode::RandomShuffle,
                            "diverse-probe" => pls::pls_config::SolutionSelectionMode::DiverseProbe,
                            #[cfg(feature = "scalarized_selection")]
                            "scalarized-chebycheff" => pls::pls_config::SolutionSelectionMode::ScalarizedChebycheff,
                            #[cfg(feature = "scalarized_selection")]
                            "diverse-then-scalarized-chebycheff" => {
                                pls::pls_config::SolutionSelectionMode::DiverseThenScalarizedChebycheff
                            }
                            _ => {
                                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                    "Unknown solution_selection_mode: {mode}"
                                )));
                            }
                        };
                    }

                    #[cfg(feature = "scalarized_selection")]
                    if let Some(source) = &scalarized_selection_source {
                        optimizations.scalarized_selection_source = match source.as_str() {
                            "population" => pls::pls_config::ScalarizedSelectionSource::Population,
                            "archive" => pls::pls_config::ScalarizedSelectionSource::Archive,
                            _ => {
                                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                    "Unknown scalarized_selection_source: {source}"
                                )));
                            }
                        };
                    }

                    #[cfg(feature = "scalarized_selection")]
                    if let Some(parent_budget) = scalarized_parent_budget {
                        optimizations.scalarized_parent_budget = Some(parent_budget);
                    }

                    #[cfg(feature = "scalarized_selection")]
                    if let Some(weight_samples) = scalarized_weight_samples {
                        optimizations.scalarized_weight_samples = weight_samples;
                    }

                    #[cfg(feature = "scalarized_selection")]
                    if let Some(rho) = scalarized_rho {
                        optimizations.scalarized_rho = rho;
                    }
                    let mut pareto_local_search = pls::pareto_local_search::ParetoLocalSearch::new(
                        &pls_problem,
                        &initial_solution_set,
                        neighborhood_size_range,
                        is_deterministic,
                        optimizations,
                    );

                    info!("Starting 2D PLS execution with {max_iterations} iterations timeout");
                    let final_solution_set = pareto_local_search.run(max_iterations, timeout);

                    info!("2D PLS completed, processing {} solutions", final_solution_set.len());

                    // Generate plot if requested
                    if plots {
                        #[cfg(feature = "plotting")]
                        {
                            let objective_names = pls_problem.objective_names();
                            pls::plotting::draw_solutions_plot(
                                &pareto_local_search.explored_solutions,
                                &objective_names,
                            );

                            if let Some(path) = plot_output_path {
                                if path != "pareto_solutions_2d.svg" {
                                    if let Err(e) = std::fs::rename("pareto_solutions_2d.svg", &path) {
                                        log::warn!("Failed to move plot to {path}: {e}");
                                    }
                                }
                            }
                        }
                        #[cfg(not(feature = "plotting"))]
                        {
                            log::warn!("Plotting requested but plotting feature is not enabled");
                        }
                    }

                    // Convert to Vec to have a uniform return type
                    let final_solutions_vec: Vec<BitsetEncodedSolution<ProblemBitset<2>, 2>> = final_solution_set.into_iter().collect();
                    let explored_solutions_vec: Vec<SolutionFingerprint<2>> = pareto_local_search.explored_solutions_fingerprints();

                    (final_solutions_vec, explored_solutions_vec)
                }};
            }

            // --- Concurrent PLS (parallel) path ---
            #[cfg(not(feature = "parallel"))]
            if parallel {
                return Err(PyValueError::new_err(
                    "parallel=True requires 'parallel' feature. Rebuild with: \
                     uv pip install -e . --reinstall-package sims-problem \
                     --config-settings=build-args='--features parallel'",
                ));
            }
            #[cfg(feature = "parallel")]
            if parallel {
                use pls::concurrent_pls::{ConcurrentPLS, ConcurrentPLSConfig};
                use pls::solution_set_impl::NdTreeSolutionSet;

                let num_threads = if num_parallel_threads == 0 {
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(4)
                } else {
                    num_parallel_threads
                };
                info!("Running 2D ConcurrentPLS with {num_threads} threads");

                let mut initial_nd = NdTreeSolutionSet::new("cpls_initial_2d");
                if let Some(provided_population) = &initial_population {
                    info!("Using provided initial population of {} solutions and generating {} random solutions for 2D ConcurrentPLS",
                          provided_population.len(), initial_population_size);
                    for solution in provided_population {
                        let selected_images: Vec<usize> =
                            solution.selected_images.iter().cloned().collect();
                        let pls_solution = BitsetEncodedSolution::from_selected_images(
                            &selected_images,
                            &pls_problem,
                        );
                        initial_nd.try_insert(&pls_solution);
                    }
                    for i in 0..initial_population_size {
                        let random_solution = if is_deterministic {
                            BitsetEncodedSolution::random_with_seed(
                                &pls_problem,
                                1_234_567_890u64.wrapping_add(i as u64),
                            )
                        } else {
                            BitsetEncodedSolution::random(&pls_problem)
                        };
                        initial_nd.try_insert(&random_solution);
                    }
                } else {
                    info!(
                        "Generating random initial population of {} solutions for 2D ConcurrentPLS",
                        initial_population_size
                    );
                    for i in 0..initial_population_size {
                        let random_solution = if is_deterministic {
                            BitsetEncodedSolution::random_with_seed(
                                &pls_problem,
                                1_234_567_890u64.wrapping_add(i as u64),
                            )
                        } else {
                            BitsetEncodedSolution::random(&pls_problem)
                        };
                        initial_nd.try_insert(&random_solution);
                    }
                }

                let config = ConcurrentPLSConfig {
                    max_iterations,
                    neighborhood_size_range: neighborhood_size_range.clone(),
                    is_deterministic,
                    ..ConcurrentPLSConfig::default_with_threads(num_threads, timeout)
                };

                info!("Starting 2D ConcurrentPLS execution ({num_threads} threads, {max_iterations} iterations timeout)");
                let cpls_result =
                    py.detach(|| ConcurrentPLS::new(&pls_problem, config).solve(&initial_nd));
                info!(
                    "2D ConcurrentPLS completed, processing {} solutions",
                    cpls_result.archive.len()
                );

                let mut python_final_solutions = Vec::new();
                for solution in cpls_result.archive {
                    let py_solution: crate::solution::Solution = (&solution, &pls_problem).into();
                    python_final_solutions.push(py_solution);
                }
                if trace {
                    log::warn!("trace=true is not supported with parallel=true (ConcurrentPLS does not track explored solutions); returning result without trace");
                }
                return Ok(crate::solution::SolvingResult::new(python_final_solutions));
            }
            // --- End concurrent PLS path ---

            // Select solution set type based on enum and run PLS
            let (final_solutions, explored_solutions) = match solution_set_type {
                SolutionSetType::LinkedList => {
                    info!("Using LinkedListSolutionSet for 2D PLS");
                    run_pls_2d_with_archive!(
                        LinkedListSolutionSet<BitsetEncodedSolution<ProblemBitset<2>, 2>, 2>,
                        "linked-list"
                    )
                }
                SolutionSetType::Vector => {
                    info!("Using VecSolutionSet for 2D PLS");
                    run_pls_2d_with_archive!(
                        VecSolutionSet<BitsetEncodedSolution<ProblemBitset<2>, 2>, 2>,
                        "vector"
                    )
                }
                SolutionSetType::NdTree => {
                    // For 2D, nd-tree uses BTreeSolutionSet
                    info!("Using BTreeSolutionSet for 2D PLS (nd-tree)");
                    run_pls_2d_with_archive!(
                        BTreeSolutionSet<BitsetEncodedSolution<ProblemBitset<2>, 2>, 2>,
                        "nd-tree"
                    )
                }
            };

            // Convert 2D solutions back to Python format
            let mut python_final_solutions = Vec::new();
            for solution in final_solutions.iter() {
                let py_solution: crate::solution::Solution = (solution, &pls_problem).into();
                python_final_solutions.push(py_solution);
            }

            info!(
                "Successfully converted {} 2D final solutions to Python format",
                python_final_solutions.len(),
            );

            // Generate trace if requested
            if trace {
                info!("Generating 2D optimization trace archive");

                // Compute dominance info (filtering + domination indices in one pass)
                let dominance_info = if include_dominated {
                    // Don't filter, but still compute domination indices
                    trace::compute_dominance_info(explored_solutions, false)
                } else {
                    info!("Filtering dominated solutions from trace");
                    // Filter and compute domination indices
                    trace::compute_dominance_info(explored_solutions, true)
                };

                let trace_solutions = dominance_info.solutions;
                let domination_indices = Some(dominance_info.domination_indices);

                // Use provided objective bounds or calculate from solutions
                let (trace_objective_bounds, reference_point) = if let Some(provided_bounds) =
                    &objective_bounds
                {
                    // Validate provided bounds
                    if provided_bounds.len() != objectives.len() {
                        return Err(PyValueError::new_err(format!(
                            "objective_bounds length ({}) does not match objectives length ({})",
                            provided_bounds.len(),
                            objectives.len()
                        )));
                    }

                    // Convert Vec<Vec<u64>> to Vec<[u64; 2]> and Vec<u64>
                    let mut bounds_vec = Vec::new();
                    let mut ref_point = Vec::new();

                    for bound in provided_bounds {
                        if bound.len() != 2 {
                            return Err(PyValueError::new_err(format!(
                                "Each objective bound must have exactly 2 elements [min, max], got {}",
                                bound.len()
                            )));
                        }
                        bounds_vec.push([bound[0], bound[1]]);
                        ref_point.push(bound[1] + 1); // Use max + 1 as reference point
                    }

                    info!("Using provided objective bounds: {:?}", bounds_vec);
                    (bounds_vec, ref_point)
                } else {
                    // Calculate from trace solutions (filtered or not)
                    trace::calculate_objective_bounds_from_solutions(&trace_solutions).map_err(
                        |e| {
                            PyValueError::new_err(format!(
                                "Failed to calculate objective bounds: {}",
                                e
                            ))
                        },
                    )?
                };

                let trace_archive = trace::create_optimization_trace_archive(
                    trace_solutions,
                    objectives,
                    timeout.as_micros() as u64,
                    "PLS-2D".to_string(),
                    trace_objective_bounds,
                    reference_point,
                    domination_indices,
                )
                .map_err(|e| {
                    PyValueError::new_err(format!("Failed to create trace archive: {}", e))
                })?;

                // Capture profiling data if enabled
                let profiling_data = read_profiling_trace_data(_chrome_guard, _profiling_buffer);

                if let Some(prof_data) = profiling_data {
                    Ok(crate::solution::SolvingResult::with_trace_and_profiling(
                        python_final_solutions,
                        trace_archive,
                        prof_data,
                    ))
                } else {
                    Ok(crate::solution::SolvingResult::with_trace(
                        python_final_solutions,
                        trace_archive,
                    ))
                }
            } else {
                // Capture profiling data if enabled
                let profiling_data = read_profiling_trace_data(_chrome_guard, _profiling_buffer);

                if let Some(prof_data) = profiling_data {
                    let mut result = crate::solution::SolvingResult::new(python_final_solutions);
                    result.profiling_trace_data = Some(prof_data);
                    Ok(result)
                } else {
                    Ok(crate::solution::SolvingResult::new(python_final_solutions))
                }
            }
        }
        3 => {
            use pls::problem_bitset::ProblemBitset;
            use pls::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;
            use pls::solution_set_impl::{
                LinkedListSolutionSet, NdTreeSolutionSet, VecSolutionSet,
            };

            debug!(
                "Using 3D optimization with objectives: {objectives:?}, solution_set_type: {:?}",
                solution_set_type
            );

            // Create 3D objective definitions inline
            let mut objective_definitions = [
                pls::objectives::ObjectiveType::TotalCost,
                pls::objectives::ObjectiveType::CloudyArea,
                pls::objectives::ObjectiveType::MinResolution,
            ];
            for (i, obj_name) in objectives.iter().enumerate() {
                objective_definitions[i] = match obj_name.as_str() {
                    "min_cost" => pls::objectives::ObjectiveType::TotalCost,
                    "cloud_coverage" => pls::objectives::ObjectiveType::CloudyArea,
                    "min_resolution" => pls::objectives::ObjectiveType::MinResolution,
                    "min_max_incidence_angle" => pls::objectives::ObjectiveType::MaxIncidenceAngle,
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "Unknown objective: {}",
                            obj_name
                        )))
                    }
                };
            }

            let mut pls_problem =
                ProblemBitset::from_raw_with_objectives(&raw_instance, objective_definitions);

            // Set objective bounds if provided
            if let Some(ref bounds) = objective_bounds {
                let bounds_vec: Vec<[u64; 2]> = bounds
                    .iter()
                    .map(|b| {
                        if b.len() != 2 {
                            return Err(PyValueError::new_err(format!(
                                "Each objective bound must have exactly 2 elements [min, max], got {}",
                                b.len()
                            )));
                        }
                        Ok([b[0], b[1]])
                    })
                    .collect::<PyResult<Vec<[u64; 2]>>>()?;
                let bounds_array: [[u64; 2]; 3] = bounds_vec.try_into().map_err(|_| {
                    PyValueError::new_err("Expected exactly 3 objective bounds for 3D problem")
                })?;
                pls_problem.set_objective_bounds(bounds_array);
            }

            // Macro to handle different solution set types for 3D
            macro_rules! run_pls_3d_with_archive {
                ($SolutionSetType:ty, $archive_name:expr) => {{
                    // Create initial population manually for 3D
                    let mut initial_solution_set = <$SolutionSetType>::new("initial_3d_solutions");

                    if let Some(provided_population) = &initial_population {
                        // Always use provided population and generate additional random solutions
                        info!("Using provided initial population of {} solutions and generating {} random solutions for 3D PLS",
                              provided_population.len(), initial_population_size);
                        for solution in provided_population {
                            let selected_images: Vec<usize> = solution.selected_images.iter().cloned().collect();
                            let pls_solution = BitsetEncodedSolution::from_selected_images(&selected_images, &pls_problem);
                            initial_solution_set.try_insert(&pls_solution);
                        }
                        // Generate additional random solutions
                        for i in 0..initial_population_size {
                            let random_solution = if is_deterministic {
                                BitsetEncodedSolution::random_with_seed(&pls_problem, 1_234_567_890u64.wrapping_add(i as u64))
                            } else {
                                BitsetEncodedSolution::random(&pls_problem)
                            };
                            initial_solution_set.try_insert(&random_solution);
                        }
                    } else {
                        // Generate random initial population
                        info!("Generating random initial population of {} solutions for 3D PLS", initial_population_size);
                        for i in 0..initial_population_size {
                            let random_solution = if is_deterministic {
                                BitsetEncodedSolution::random_with_seed(&pls_problem, 1_234_567_890u64.wrapping_add(i as u64))
                            } else {
                                BitsetEncodedSolution::random(&pls_problem)
                            };
                            initial_solution_set.try_insert(&random_solution);
                        }
                    }

                    // Create and run 3D PLS
                    let mut optimizations = PlsOptimizations {
                        neighborhood_budget,
                        flags: PlsFlags::from_pairs([
                            (PlsFlags::USE_CHECKPOINT, use_checkpoint),
                            (PlsFlags::USE_RANKED_CANDIDATES, use_ranked_candidates),
                            (
                                PlsFlags::USE_GREEDY_INITIAL_POPULATION,
                                use_greedy_initial_population,
                            ),
                            (PlsFlags::USE_PERTURBATION_RESTART, use_perturbation_restart),
                            (PlsFlags::USE_DIVERSE_PROBING, use_diverse_probing),
                            (
                                PlsFlags::USE_ND_TREE_SCALARIZED_QUERY,
                                use_nd_tree_scalarized_query,
                            ),
                        ]),
                        max_k1_candidates,
                        probing_budget,
                        diverse_probe_budget,
                        ..PlsOptimizations::default()
                    };

                    if let Some(mode) = &solution_selection_mode {
                        optimizations.solution_selection_mode = match mode.as_str() {
                            "random-shuffle" => pls::pls_config::SolutionSelectionMode::RandomShuffle,
                            "diverse-probe" => pls::pls_config::SolutionSelectionMode::DiverseProbe,
                            #[cfg(feature = "scalarized_selection")]
                            "scalarized-chebycheff" => pls::pls_config::SolutionSelectionMode::ScalarizedChebycheff,
                            #[cfg(feature = "scalarized_selection")]
                            "diverse-then-scalarized-chebycheff" => {
                                pls::pls_config::SolutionSelectionMode::DiverseThenScalarizedChebycheff
                            }
                            _ => {
                                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                    "Unknown solution_selection_mode: {mode}"
                                )));
                            }
                        };
                    }

                    #[cfg(feature = "scalarized_selection")]
                    if let Some(source) = &scalarized_selection_source {
                        optimizations.scalarized_selection_source = match source.as_str() {
                            "population" => pls::pls_config::ScalarizedSelectionSource::Population,
                            "archive" => pls::pls_config::ScalarizedSelectionSource::Archive,
                            _ => {
                                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                    "Unknown scalarized_selection_source: {source}"
                                )));
                            }
                        };
                    }

                    #[cfg(feature = "scalarized_selection")]
                    if let Some(parent_budget) = scalarized_parent_budget {
                        optimizations.scalarized_parent_budget = Some(parent_budget);
                    }

                    #[cfg(feature = "scalarized_selection")]
                    if let Some(weight_samples) = scalarized_weight_samples {
                        optimizations.scalarized_weight_samples = weight_samples;
                    }

                    #[cfg(feature = "scalarized_selection")]
                    if let Some(rho) = scalarized_rho {
                        optimizations.scalarized_rho = rho;
                    }
                    let mut pareto_local_search = ParetoLocalSearch::new(
                        &pls_problem,
                        &initial_solution_set,
                        neighborhood_size_range,
                        is_deterministic,
                        optimizations,
                    );

                    info!("Starting 3D PLS execution with {max_iterations} iterations timeout");
                    let final_solution_set = pareto_local_search.run(max_iterations, timeout);

                    info!("3D PLS completed, processing {} solutions", final_solution_set.len());

                    // Generate 3D plot if requested
                    if plots {
                        #[cfg(feature = "plotting")]
                        {
                            let objective_names = pls_problem.objective_names();
                            pls::plotting::draw_solutions_plot(
                                &pareto_local_search.explored_solutions,
                                &objective_names,
                            );

                            if let Some(path) = plot_output_path {
                                if path != "pareto_solutions_grid.svg" {
                                    if let Err(e) = std::fs::rename("pareto_solutions_grid.svg", &path) {
                                        log::warn!("Failed to move plot to {path}: {e}");
                                    }
                                }
                            }
                        }
                        #[cfg(not(feature = "plotting"))]
                        {
                            log::warn!("Plotting requested but plotting feature is not enabled");
                        }
                    }

                    // Convert to Vec to have a uniform return type
                    let final_solutions_vec: Vec<BitsetEncodedSolution<ProblemBitset<3>, 3>> = final_solution_set.into_iter().collect();
                    let explored_solutions_vec: Vec<SolutionFingerprint<3>> = pareto_local_search.explored_solutions_fingerprints();

                    (final_solutions_vec, explored_solutions_vec)
                }};
            }

            // --- Concurrent PLS (parallel) path ---
            #[cfg(not(feature = "parallel"))]
            if parallel {
                return Err(PyValueError::new_err(
                    "parallel=True requires 'parallel' feature. Rebuild with: \
                     uv pip install -e . --reinstall-package sims-problem \
                     --config-settings=build-args='--features parallel'",
                ));
            }
            #[cfg(feature = "parallel")]
            if parallel {
                use pls::concurrent_pls::{ConcurrentPLS, ConcurrentPLSConfig};
                use pls::solution_set_impl::NdTreeSolutionSet;

                let num_threads = if num_parallel_threads == 0 {
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(4)
                } else {
                    num_parallel_threads
                };
                info!("Running 3D ConcurrentPLS with {num_threads} threads");

                let mut initial_nd = NdTreeSolutionSet::new("cpls_initial_3d");
                if let Some(provided_population) = &initial_population {
                    info!("Using provided initial population of {} solutions and generating {} random solutions for 3D ConcurrentPLS",
                          provided_population.len(), initial_population_size);
                    for solution in provided_population {
                        let selected_images: Vec<usize> =
                            solution.selected_images.iter().cloned().collect();
                        let pls_solution = BitsetEncodedSolution::from_selected_images(
                            &selected_images,
                            &pls_problem,
                        );
                        initial_nd.try_insert(&pls_solution);
                    }
                    for i in 0..initial_population_size {
                        let random_solution = if is_deterministic {
                            BitsetEncodedSolution::random_with_seed(
                                &pls_problem,
                                1_234_567_890u64.wrapping_add(i as u64),
                            )
                        } else {
                            BitsetEncodedSolution::random(&pls_problem)
                        };
                        initial_nd.try_insert(&random_solution);
                    }
                } else {
                    info!(
                        "Generating random initial population of {} solutions for 3D ConcurrentPLS",
                        initial_population_size
                    );
                    for i in 0..initial_population_size {
                        let random_solution = if is_deterministic {
                            BitsetEncodedSolution::random_with_seed(
                                &pls_problem,
                                1_234_567_890u64.wrapping_add(i as u64),
                            )
                        } else {
                            BitsetEncodedSolution::random(&pls_problem)
                        };
                        initial_nd.try_insert(&random_solution);
                    }
                }

                let config = ConcurrentPLSConfig {
                    max_iterations,
                    neighborhood_size_range: neighborhood_size_range.clone(),
                    is_deterministic,
                    ..ConcurrentPLSConfig::default_with_threads(num_threads, timeout)
                };

                info!("Starting 3D ConcurrentPLS execution ({num_threads} threads, {max_iterations} iterations timeout)");
                let cpls_result =
                    py.detach(|| ConcurrentPLS::new(&pls_problem, config).solve(&initial_nd));
                info!(
                    "3D ConcurrentPLS completed, processing {} solutions",
                    cpls_result.archive.len()
                );

                let mut python_final_solutions = Vec::new();
                for solution in cpls_result.archive {
                    let py_solution: crate::solution::Solution = (&solution, &pls_problem).into();
                    python_final_solutions.push(py_solution);
                }
                if trace {
                    log::warn!("trace=true is not supported with parallel=true (ConcurrentPLS does not track explored solutions); returning result without trace");
                }
                return Ok(crate::solution::SolvingResult::new(python_final_solutions));
            }
            // --- End concurrent PLS path ---

            // Select solution set type based on enum and run PLS
            let (final_solutions, explored_solutions) = match solution_set_type {
                SolutionSetType::LinkedList => {
                    info!("Using LinkedListSolutionSet for 3D PLS");
                    run_pls_3d_with_archive!(
                        LinkedListSolutionSet<BitsetEncodedSolution<ProblemBitset<3>, 3>, 3>,
                        "linked-list"
                    )
                }
                SolutionSetType::Vector => {
                    info!("Using VecSolutionSet for 3D PLS");
                    run_pls_3d_with_archive!(
                        VecSolutionSet<BitsetEncodedSolution<ProblemBitset<3>, 3>, 3>,
                        "vector"
                    )
                }
                SolutionSetType::NdTree => {
                    info!("Using NdTreeSolutionSet for 3D PLS (nd-tree)");
                    run_pls_3d_with_archive!(
                        NdTreeSolutionSet<BitsetEncodedSolution<ProblemBitset<3>, 3>, 3>,
                        "nd-tree"
                    )
                }
            };

            // Convert 3D solutions back to Python format
            let mut python_final_solutions = Vec::new();
            for solution in final_solutions.iter() {
                let py_solution: crate::solution::Solution = (solution, &pls_problem).into();
                python_final_solutions.push(py_solution);
            }

            info!(
                "Successfully converted {} 3D final solutions to Python format",
                python_final_solutions.len(),
            );

            // Generate trace if requested
            if trace {
                info!("Generating 3D optimization trace archive");

                // Compute dominance info (filtering + domination indices in one pass)
                let dominance_info = if include_dominated {
                    // Don't filter, but still compute domination indices
                    trace::compute_dominance_info(explored_solutions, false)
                } else {
                    info!("Filtering dominated solutions from trace");
                    // Filter and compute domination indices
                    trace::compute_dominance_info(explored_solutions, true)
                };

                let trace_solutions = dominance_info.solutions;
                let domination_indices = Some(dominance_info.domination_indices);

                // Use provided objective bounds or calculate from solutions
                let (trace_objective_bounds, reference_point) = if let Some(provided_bounds) =
                    &objective_bounds
                {
                    // Validate provided bounds
                    if provided_bounds.len() != objectives.len() {
                        return Err(PyValueError::new_err(format!(
                            "objective_bounds length ({}) does not match objectives length ({})",
                            provided_bounds.len(),
                            objectives.len()
                        )));
                    }

                    // Convert Vec<Vec<u64>> to Vec<[u64; 2]> and Vec<u64>
                    let mut bounds_vec = Vec::new();
                    let mut ref_point = Vec::new();

                    for bound in provided_bounds {
                        if bound.len() != 2 {
                            return Err(PyValueError::new_err(format!(
                                "Each objective bound must have exactly 2 elements [min, max], got {}",
                                bound.len()
                            )));
                        }
                        bounds_vec.push([bound[0], bound[1]]);
                        ref_point.push(bound[1] + 1); // Use max + 1 as reference point
                    }

                    info!("Using provided objective bounds: {:?}", bounds_vec);
                    (bounds_vec, ref_point)
                } else {
                    // Calculate from trace solutions (filtered or not)
                    trace::calculate_objective_bounds_from_solutions(&trace_solutions).map_err(
                        |e| {
                            PyValueError::new_err(format!(
                                "Failed to calculate objective bounds: {}",
                                e
                            ))
                        },
                    )?
                };

                let trace_archive = trace::create_optimization_trace_archive(
                    trace_solutions,
                    objectives,
                    timeout.as_micros() as u64,
                    "PLS-3D".to_string(),
                    trace_objective_bounds,
                    reference_point,
                    domination_indices,
                )
                .map_err(|e| {
                    PyValueError::new_err(format!("Failed to create trace archive: {}", e))
                })?;

                // Capture profiling data if enabled
                let profiling_data = read_profiling_trace_data(_chrome_guard, _profiling_buffer);

                if let Some(prof_data) = profiling_data {
                    Ok(crate::solution::SolvingResult::with_trace_and_profiling(
                        python_final_solutions,
                        trace_archive,
                        prof_data,
                    ))
                } else {
                    Ok(crate::solution::SolvingResult::with_trace(
                        python_final_solutions,
                        trace_archive,
                    ))
                }
            } else {
                // Capture profiling data if enabled
                let profiling_data = read_profiling_trace_data(_chrome_guard, _profiling_buffer);

                if let Some(prof_data) = profiling_data {
                    let mut result = crate::solution::SolvingResult::new(python_final_solutions);
                    result.profiling_trace_data = Some(prof_data);
                    Ok(result)
                } else {
                    Ok(crate::solution::SolvingResult::new(python_final_solutions))
                }
            }
        }
        4 => {
            use pls::problem_bitset::ProblemBitset;
            use pls::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;
            use pls::solution_set_impl::{
                LinkedListSolutionSet, NdTreeSolutionSet, VecSolutionSet,
            };

            debug!(
                "Using 4D optimization with objectives: {objectives:?}, solution_set_type: {:?}",
                solution_set_type
            );

            // Create 4D objective definitions inline
            let mut objective_definitions = [
                pls::objectives::ObjectiveType::TotalCost,
                pls::objectives::ObjectiveType::CloudyArea,
                pls::objectives::ObjectiveType::MinResolution,
                pls::objectives::ObjectiveType::MaxIncidenceAngle,
            ];
            for (i, obj_name) in objectives.iter().enumerate() {
                objective_definitions[i] = match obj_name.as_str() {
                    "min_cost" => pls::objectives::ObjectiveType::TotalCost,
                    "cloud_coverage" => pls::objectives::ObjectiveType::CloudyArea,
                    "min_resolution" => pls::objectives::ObjectiveType::MinResolution,
                    "min_max_incidence_angle" => pls::objectives::ObjectiveType::MaxIncidenceAngle,
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "Unknown objective: {}",
                            obj_name
                        )))
                    }
                };
            }

            let mut pls_problem =
                ProblemBitset::from_raw_with_objectives(&raw_instance, objective_definitions);

            // Set objective bounds if provided
            if let Some(ref bounds) = objective_bounds {
                let bounds_vec: Vec<[u64; 2]> = bounds
                    .iter()
                    .map(|b| {
                        if b.len() != 2 {
                            return Err(PyValueError::new_err(format!(
                                "Each objective bound must have exactly 2 elements [min, max], got {}",
                                b.len()
                            )));
                        }
                        Ok([b[0], b[1]])
                    })
                    .collect::<PyResult<Vec<[u64; 2]>>>()?;
                let bounds_array: [[u64; 2]; 4] = bounds_vec.try_into().map_err(|_| {
                    PyValueError::new_err("Expected exactly 4 objective bounds for 4D problem")
                })?;
                pls_problem.set_objective_bounds(bounds_array);
            }

            // Macro to handle different solution set types for 4D
            macro_rules! run_pls_4d_with_archive {
                ($SolutionSetType:ty, $archive_name:expr) => {{
                    // Create initial population manually for 4D
                    let mut initial_solution_set = <$SolutionSetType>::new("initial_4d_solutions");

                    if let Some(provided_population) = &initial_population {
                        // Always use provided population and generate additional random solutions
                        info!("Using provided initial population of {} solutions and generating {} random solutions for 4D PLS",
                              provided_population.len(), initial_population_size);
                        for solution in provided_population {
                            let selected_images: Vec<usize> = solution.selected_images.iter().cloned().collect();
                            let pls_solution = BitsetEncodedSolution::from_selected_images(&selected_images, &pls_problem);
                            initial_solution_set.try_insert(&pls_solution);
                        }
                        // Generate additional random solutions
                        for i in 0..initial_population_size {
                            let random_solution = if is_deterministic {
                                BitsetEncodedSolution::random_with_seed(&pls_problem, 1_234_567_890u64.wrapping_add(i as u64))
                            } else {
                                BitsetEncodedSolution::random(&pls_problem)
                            };
                            initial_solution_set.try_insert(&random_solution);
                        }
                    } else {
                        // Generate random initial population
                        info!("Generating random initial population of {} solutions for 4D PLS", initial_population_size);
                        for i in 0..initial_population_size {
                            let random_solution = if is_deterministic {
                                BitsetEncodedSolution::random_with_seed(&pls_problem, 1_234_567_890u64.wrapping_add(i as u64))
                            } else {
                                BitsetEncodedSolution::random(&pls_problem)
                            };
                            initial_solution_set.try_insert(&random_solution);
                        }
                    }

                    // Create and run 4D PLS
                    let mut optimizations = PlsOptimizations {
                        neighborhood_budget,
                        flags: PlsFlags::from_pairs([
                            (PlsFlags::USE_CHECKPOINT, use_checkpoint),
                            (PlsFlags::USE_RANKED_CANDIDATES, use_ranked_candidates),
                            (
                                PlsFlags::USE_GREEDY_INITIAL_POPULATION,
                                use_greedy_initial_population,
                            ),
                            (PlsFlags::USE_PERTURBATION_RESTART, use_perturbation_restart),
                            (PlsFlags::USE_DIVERSE_PROBING, use_diverse_probing),
                            (
                                PlsFlags::USE_ND_TREE_SCALARIZED_QUERY,
                                use_nd_tree_scalarized_query,
                            ),
                        ]),
                        max_k1_candidates,
                        probing_budget,
                        diverse_probe_budget,
                        ..PlsOptimizations::default()
                    };

                    if let Some(mode) = &solution_selection_mode {
                        optimizations.solution_selection_mode = match mode.as_str() {
                            "random-shuffle" => pls::pls_config::SolutionSelectionMode::RandomShuffle,
                            "diverse-probe" => pls::pls_config::SolutionSelectionMode::DiverseProbe,
                            #[cfg(feature = "scalarized_selection")]
                            "scalarized-chebycheff" => pls::pls_config::SolutionSelectionMode::ScalarizedChebycheff,
                            #[cfg(feature = "scalarized_selection")]
                            "diverse-then-scalarized-chebycheff" => {
                                pls::pls_config::SolutionSelectionMode::DiverseThenScalarizedChebycheff
                            }
                            _ => {
                                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                    "Unknown solution_selection_mode: {mode}"
                                )));
                            }
                        };
                    }

                    #[cfg(feature = "scalarized_selection")]
                    if let Some(source) = &scalarized_selection_source {
                        optimizations.scalarized_selection_source = match source.as_str() {
                            "population" => pls::pls_config::ScalarizedSelectionSource::Population,
                            "archive" => pls::pls_config::ScalarizedSelectionSource::Archive,
                            _ => {
                                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                    "Unknown scalarized_selection_source: {source}"
                                )));
                            }
                        };
                    }

                    #[cfg(feature = "scalarized_selection")]
                    if let Some(parent_budget) = scalarized_parent_budget {
                        optimizations.scalarized_parent_budget = Some(parent_budget);
                    }

                    #[cfg(feature = "scalarized_selection")]
                    if let Some(weight_samples) = scalarized_weight_samples {
                        optimizations.scalarized_weight_samples = weight_samples;
                    }

                    #[cfg(feature = "scalarized_selection")]
                    if let Some(rho) = scalarized_rho {
                        optimizations.scalarized_rho = rho;
                    }
                    let mut pareto_local_search = pls::pareto_local_search::ParetoLocalSearch::new(
                        &pls_problem,
                        &initial_solution_set,
                        neighborhood_size_range,
                        is_deterministic,
                        optimizations,
                    );

                    info!("Starting 4D PLS execution with {max_iterations} iterations timeout");
                    let final_solution_set = pareto_local_search.run(max_iterations, timeout);

                    info!("4D PLS completed, processing {} solutions", final_solution_set.len());

                    // Generate 4D plot if requested
                    if plots {
                        #[cfg(feature = "plotting")]
                        {
                            let objective_names = pls_problem.objective_names();
                            pls::plotting::draw_solutions_plot(
                                &pareto_local_search.explored_solutions,
                                &objective_names,
                            );

                            if let Some(path) = plot_output_path {
                                if path != "pareto_solutions_grid.svg" {
                                    if let Err(e) = std::fs::rename("pareto_solutions_grid.svg", &path) {
                                        log::warn!("Failed to move plot to {path}: {e}");
                                    }
                                }
                            }
                        }
                        #[cfg(not(feature = "plotting"))]
                        {
                            log::warn!("Plotting requested but plotting feature is not enabled");
                        }
                    }

                    // Convert to Vec to have a uniform return type
                    let final_solutions_vec: Vec<BitsetEncodedSolution<ProblemBitset<4>, 4>> = final_solution_set.into_iter().collect();
                    let explored_solutions_vec: Vec<SolutionFingerprint<4>> = pareto_local_search.explored_solutions_fingerprints();

                    (final_solutions_vec, explored_solutions_vec)
                }};
            }

            // --- Concurrent PLS (parallel) path ---
            #[cfg(not(feature = "parallel"))]
            if parallel {
                return Err(PyValueError::new_err(
                    "parallel=True requires 'parallel' feature. Rebuild with: \
                     uv pip install -e . --reinstall-package sims-problem \
                     --config-settings=build-args='--features parallel'",
                ));
            }
            #[cfg(feature = "parallel")]
            if parallel {
                use pls::concurrent_pls::{ConcurrentPLS, ConcurrentPLSConfig};
                use pls::solution_set_impl::NdTreeSolutionSet;

                let num_threads = if num_parallel_threads == 0 {
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(4)
                } else {
                    num_parallel_threads
                };
                info!("Running 4D ConcurrentPLS with {num_threads} threads");

                let mut initial_nd = NdTreeSolutionSet::new("cpls_initial_4d");
                if let Some(provided_population) = &initial_population {
                    info!("Using provided initial population of {} solutions and generating {} random solutions for 4D ConcurrentPLS",
                          provided_population.len(), initial_population_size);
                    for solution in provided_population {
                        let selected_images: Vec<usize> =
                            solution.selected_images.iter().cloned().collect();
                        let pls_solution = BitsetEncodedSolution::from_selected_images(
                            &selected_images,
                            &pls_problem,
                        );
                        initial_nd.try_insert(&pls_solution);
                    }
                    for i in 0..initial_population_size {
                        let random_solution = if is_deterministic {
                            BitsetEncodedSolution::random_with_seed(
                                &pls_problem,
                                1_234_567_890u64.wrapping_add(i as u64),
                            )
                        } else {
                            BitsetEncodedSolution::random(&pls_problem)
                        };
                        initial_nd.try_insert(&random_solution);
                    }
                } else {
                    info!(
                        "Generating random initial population of {} solutions for 4D ConcurrentPLS",
                        initial_population_size
                    );
                    for i in 0..initial_population_size {
                        let random_solution = if is_deterministic {
                            BitsetEncodedSolution::random_with_seed(
                                &pls_problem,
                                1_234_567_890u64.wrapping_add(i as u64),
                            )
                        } else {
                            BitsetEncodedSolution::random(&pls_problem)
                        };
                        initial_nd.try_insert(&random_solution);
                    }
                }

                let config = ConcurrentPLSConfig {
                    max_iterations,
                    neighborhood_size_range: neighborhood_size_range.clone(),
                    is_deterministic,
                    ..ConcurrentPLSConfig::default_with_threads(num_threads, timeout)
                };

                info!("Starting 4D ConcurrentPLS execution ({num_threads} threads, {max_iterations} iterations timeout)");
                let cpls_result =
                    py.detach(|| ConcurrentPLS::new(&pls_problem, config).solve(&initial_nd));
                info!(
                    "4D ConcurrentPLS completed, processing {} solutions",
                    cpls_result.archive.len()
                );

                let mut python_final_solutions = Vec::new();
                for solution in cpls_result.archive {
                    let py_solution: crate::solution::Solution = (&solution, &pls_problem).into();
                    python_final_solutions.push(py_solution);
                }
                if trace {
                    log::warn!("trace=true is not supported with parallel=true (ConcurrentPLS does not track explored solutions); returning result without trace");
                }
                return Ok(crate::solution::SolvingResult::new(python_final_solutions));
            }
            // --- End concurrent PLS path ---

            // Select solution set type based on enum and run PLS
            let (final_solutions, explored_solutions) = match solution_set_type {
                SolutionSetType::LinkedList => {
                    info!("Using LinkedListSolutionSet for 4D PLS");
                    run_pls_4d_with_archive!(
                        LinkedListSolutionSet<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4>,
                        "linked-list"
                    )
                }
                SolutionSetType::Vector => {
                    info!("Using VecSolutionSet for 4D PLS");
                    run_pls_4d_with_archive!(
                        VecSolutionSet<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4>,
                        "vector"
                    )
                }
                SolutionSetType::NdTree => {
                    info!("Using NdTreeSolutionSet for 4D PLS (nd-tree)");
                    run_pls_4d_with_archive!(
                        NdTreeSolutionSet<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4>,
                        "nd-tree"
                    )
                }
            };

            // Convert 4D solutions back to Python format
            let mut python_final_solutions = Vec::new();
            for solution in final_solutions.iter() {
                let py_solution: crate::solution::Solution = (solution, &pls_problem).into();
                python_final_solutions.push(py_solution);
            }

            info!(
                "Successfully converted {} 4D final solutions to Python format",
                python_final_solutions.len(),
            );

            // Generate trace if requested
            if trace {
                info!(
                    "Generating 4D optimization trace archive ({} explored solutions)",
                    explored_solutions.len()
                );

                // Compute dominance info (filtering + domination indices in one pass)
                info!(
                    "Computing dominance info (filter_dominated={})...",
                    !include_dominated
                );
                let dominance_start = std::time::Instant::now();
                let dominance_info = if include_dominated {
                    // Don't filter, but still compute domination indices
                    trace::compute_dominance_info(explored_solutions, false)
                } else {
                    info!("Filtering dominated solutions from trace");
                    // Filter and compute domination indices
                    trace::compute_dominance_info(explored_solutions, true)
                };
                info!(
                    "Dominance computation done in {:.3}s ({} -> {} solutions)",
                    dominance_start.elapsed().as_secs_f64(),
                    dominance_info.domination_indices.len(),
                    dominance_info.solutions.len(),
                );

                let trace_solutions = dominance_info.solutions;
                let domination_indices = Some(dominance_info.domination_indices);

                // Use provided objective bounds or calculate from solutions
                info!(
                    "Computing objective bounds for {} trace solutions...",
                    trace_solutions.len()
                );
                let bounds_start = std::time::Instant::now();
                let (trace_objective_bounds, reference_point) = if let Some(provided_bounds) =
                    &objective_bounds
                {
                    // Validate provided bounds
                    if provided_bounds.len() != objectives.len() {
                        return Err(PyValueError::new_err(format!(
                            "objective_bounds length ({}) does not match objectives length ({})",
                            provided_bounds.len(),
                            objectives.len()
                        )));
                    }

                    // Convert Vec<Vec<u64>> to Vec<[u64; 2]> and Vec<u64>
                    let mut bounds_vec = Vec::new();
                    let mut ref_point = Vec::new();

                    for bound in provided_bounds {
                        if bound.len() != 2 {
                            return Err(PyValueError::new_err(format!(
                                "Each objective bound must have exactly 2 elements [min, max], got {}",
                                bound.len()
                            )));
                        }
                        bounds_vec.push([bound[0], bound[1]]);
                        ref_point.push(bound[1] + 1); // Use max + 1 as reference point
                    }

                    info!("Using provided objective bounds: {:?}", bounds_vec);
                    (bounds_vec, ref_point)
                } else {
                    // Calculate from trace solutions (filtered or not)
                    trace::calculate_objective_bounds_from_solutions(&trace_solutions).map_err(
                        |e| {
                            PyValueError::new_err(format!(
                                "Failed to calculate objective bounds: {}",
                                e
                            ))
                        },
                    )?
                };
                info!(
                    "Objective bounds computed in {:.3}s",
                    bounds_start.elapsed().as_secs_f64()
                );

                info!("Creating trace archive (binaries + hypervolume + compression)...");
                let archive_start = std::time::Instant::now();
                let trace_archive = trace::create_optimization_trace_archive(
                    trace_solutions,
                    objectives,
                    timeout.as_micros() as u64,
                    "PLS-4D".to_string(),
                    trace_objective_bounds,
                    reference_point,
                    domination_indices,
                )
                .map_err(|e| {
                    PyValueError::new_err(format!("Failed to create trace archive: {}", e))
                })?;
                info!(
                    "Trace archive created in {:.3}s ({} bytes)",
                    archive_start.elapsed().as_secs_f64(),
                    trace_archive.len()
                );

                // Capture profiling data if enabled
                let profiling_data = read_profiling_trace_data(_chrome_guard, _profiling_buffer);

                if let Some(prof_data) = profiling_data {
                    Ok(crate::solution::SolvingResult::with_trace_and_profiling(
                        python_final_solutions,
                        trace_archive,
                        prof_data,
                    ))
                } else {
                    Ok(crate::solution::SolvingResult::with_trace(
                        python_final_solutions,
                        trace_archive,
                    ))
                }
            } else {
                // Capture profiling data if enabled
                let profiling_data = read_profiling_trace_data(_chrome_guard, _profiling_buffer);

                if let Some(prof_data) = profiling_data {
                    let mut result = crate::solution::SolvingResult::new(python_final_solutions);
                    result.profiling_trace_data = Some(prof_data);
                    Ok(result)
                } else {
                    Ok(crate::solution::SolvingResult::new(python_final_solutions))
                }
            }
        }
        n => Err(PyValueError::new_err(format!(
            "Unsupported number of objectives: {n}. Supported: 2, 3, or 4 objectives."
        ))),
    }
}

/// Helper function to compute cloudy area for a set of selected images
/// This matches the Python implementation in solver_result.py::_compute_cloudy_area
#[cfg(feature = "milp")]
fn compute_cloudy_area(selected_images: &[usize], problem_data: &SimsDiscreteProblem) -> i64 {
    // Compute clear parts - universe elements that are covered by non-cloudy parts of images
    let mut clear_parts = HashSet::new();
    for &img_idx in selected_images {
        let image_set: HashSet<usize> = problem_data.images[img_idx].iter().copied().collect();
        let cloud_set: HashSet<usize> = problem_data.clouds[img_idx].iter().copied().collect();
        let clear_in_image: HashSet<usize> = image_set.difference(&cloud_set).copied().collect();
        clear_parts.extend(clear_in_image);
    }

    // Compute cloudy area - sum of areas for universe elements not in clear parts
    (0..problem_data.universe)
        .filter(|u| !clear_parts.contains(u))
        .map(|u| problem_data.areas[u])
        .sum()
}

/// Helper function to compute minimum resolutions sum
/// This matches the Python implementation in solver_result.py::_compute_min_resolutions_sum
#[cfg(feature = "milp")]
fn compute_min_resolutions_sum(
    selected_images: &[usize],
    problem_data: &SimsDiscreteProblem,
) -> i64 {
    // For each universe element, find minimum resolution among images that cover it
    (0..problem_data.universe)
        .map(|u| {
            selected_images
                .iter()
                .filter(|&&img_idx| problem_data.images[img_idx].contains(&u))
                .map(|&img_idx| problem_data.resolution[img_idx])
                .min()
                .unwrap_or(0)
        })
        .sum()
}

/// Solves the SIMS problem using MILP with AUGMECON for exact Pareto solutions
#[cfg(feature = "milp")]
#[expect(
    clippy::too_many_arguments,
    reason = "It's okay for Python API to have many parameters"
)]
#[allow(
    unused_variables,
    reason = "grid_points parameter reserved for future use"
)]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string()],
    grid_points=50,
    timeout=Duration::from_secs(300),
    bypass_coefficient=true,
    early_exit=true,
    flag_array=true,
    solver_name="highs".to_string(),
    method="gpba".to_string(),
))]
pub fn solve_with_milp(
    sims_instance: &SimsDiscreteProblem,
    objectives: Vec<String>,
    grid_points: usize,
    timeout: Duration,
    bypass_coefficient: bool,
    early_exit: bool,
    flag_array: bool,
    solver_name: String,
    method: String,
) -> PyResult<SolvingResult> {
    // Started here, not after the setup below, because the caller's budget is
    // wall clock: everything this function does is spent from it, including
    // building the model and computing the heuristic nadir. Starting the clock
    // after that work made it free, and it is not -- measured on
    // `lagos_nigeria_150` as a flat 8 seconds over budget whatever the budget
    // was, at 20s, 60s and 200s alike.
    let start_time = std::time::Instant::now();

    // Validate objectives
    let valid_objectives = [
        "min_cost",
        "cloud_coverage",
        "min_resolution",
        "min_max_incidence_angle",
    ];
    for obj in &objectives {
        if !valid_objectives.contains(&obj.as_str()) {
            return Err(PyValueError::new_err(format!(
                "Invalid objective '{obj}'. Valid objectives are: {valid_objectives:?}"
            )));
        }
    }

    // Convert objectives from strings to SimsObjective enum
    let mut objective_set = std::collections::HashSet::new();
    for obj in &objectives {
        let sims_obj = match obj.as_str() {
            "min_cost" => augmecon::sims_problem::SimsObjective::MinCost,
            "cloud_coverage" => augmecon::sims_problem::SimsObjective::CloudCoverage,
            "min_resolution" => augmecon::sims_problem::SimsObjective::MinResolution,
            "min_max_incidence_angle" => augmecon::sims_problem::SimsObjective::MaxIncidenceAngle,
            _ => unreachable!(), // Already validated above
        };
        objective_set.insert(sims_obj);
    }

    let timeout_seconds = timeout.as_secs_f64();
    info!(
        "Starting MILP algorithm with objectives: {objectives:?}, timeout: {timeout_seconds}s, solver: {solver_name}"
    );

    // Note: The timeout parameter is passed to AUGMECON but may not be fully enforced
    // in the current implementation. The solver will attempt to respect the timeout
    // but this is dependent on the underlying AUGMECON solver implementation.

    // Build the augmecon-facing SimsInstance directly from the raw parsed
    // arrays — including the image_clouds relation (which images can
    // substitute for a cloud-obscured fragment) — in a single optimized
    // pass. See `SimsInstance::from_raw_refs` for the complexity/perf
    // rationale; this used to be ~100 lines of duplicated, unoptimized
    // conversion logic inlined here.
    let sims_augmecon_instance = SimsInstance::from_raw_refs(
        &sims_instance.images,
        &sims_instance.clouds,
        &sims_instance.costs,
        &sims_instance.areas,
        &sims_instance.resolution,
        &sims_instance.incidence_angle,
        sims_instance.universe,
        sims_instance.max_cloud_area,
    );
    info!(
        "Found {} unique clouds across all images",
        sims_augmecon_instance.cloud_ids.len()
    );

    // Create MultiObjectiveProblem with only the requested objectives
    let problem = augmecon::sims_problem::create_sims_problem_with_objectives(
        &sims_augmecon_instance,
        Some(&objective_set),
    );

    // Verify problem was created correctly
    if problem.num_objectives() != objectives.len() {
        return Err(PyValueError::new_err(format!(
            "Internal error: Problem has {} objectives but {} were requested",
            problem.num_objectives(),
            objectives.len()
        )));
    }

    // Parse solver_name to Solver enum
    use augmecon::solver_enum::Solver;
    let solver = match solver_name.to_lowercase().as_str() {
        "default" => Solver::Default,
        "coin_cbc" => Solver::CoinCbc,
        "highs" => Solver::HiGHS,
        "scip" => Solver::SCIP,
        // Native Gurobi (grb) backend. The enum variant always exists; the
        // augmecon dispatch returns UnsupportedSolver at solve time unless
        // sims-problem was built with `--features gurobi`.
        "gurobi" => Solver::Gurobi,
        _ => {
            error!("Unknown solver_name '{}'", solver_name);
            return Err(PyValueError::new_err(format!(
                "Unknown solver_name '{}'. Valid options are: default, coin_cbc, highs, scip, gurobi",
                solver_name
            )));
        }
    };

    info!(
        "Using solver: {} (from solver_name='{}')",
        solver, solver_name
    );

    // Configure options for GPBA-A with Python parameters
    let mut options = Options::default()
        .with_solver(solver)
        .with_bypass_coefficient(bypass_coefficient)
        .with_early_exit(early_exit)
        .with_flag_array(flag_array);
    // Temporary A/B switch for the lexicographic post-pass, which trades one
    // extra solve per emitted point for a guarantee that the point is
    // efficient rather than only weakly so.
    options.lexicographic_refine = std::env::var("GPBA_LEX_REFINE").is_ok();
    // Efficiency from an objective priority rather than an augmentation term
    // or a second solve; see `Options::hierarchical_efficiency`.
    options.hierarchical_efficiency = std::env::var("GPBA_HIERARCHY").is_ok();

    info!(
        "GPBA-A Options: solver={}, bypass_coefficient={}, early_exit={}, flag_array={}",
        solver, bypass_coefficient, early_exit, flag_array
    );

    // Calculate heuristic nadir bounds (matching Python non-inlined version)
    // This avoids unbounded maximization issues with auxiliary variables
    let objectives_enum: Vec<SimsObjective> = objectives
        .iter()
        .map(|obj_name| {
            match obj_name.as_str() {
                "min_cost" => SimsObjective::MinCost,
                "cloud_coverage" => SimsObjective::CloudCoverage,
                "min_resolution" => SimsObjective::MinResolution,
                "min_max_incidence_angle" => SimsObjective::MaxIncidenceAngle,
                _ => panic!("Invalid objective: {}", obj_name), // Already validated earlier
            }
        })
        .collect();
    let nadir_heuristic = sims_augmecon_instance.calculate_nadir_heuristic(&objectives_enum);
    info!("Using heuristic nadir bounds: {:?}", nadir_heuristic);

    // Compute ideal bounds by minimizing each objective
    info!("Computing ideal bounds by minimizing each objective");
    let mut ideal_bounds = Vec::with_capacity(objectives.len());
    let mut lex_extremes: Vec<augmecon::solution::Solution> = Vec::new();

    // The first-phase method needs time of its own, so the preparation above it
    // is bounded to a share of the budget rather than to the whole of it.
    //
    // Without this the lexicographic solves can take everything: measured on
    // tokyo_bay_225 as 215s against a 200s budget, leaving the method zero
    // seconds and the run reporting nothing but the two extremes. Overrunning
    // the caller's limit to hide that is not an option -- a wall-clock bound
    // that is exceeded whenever preparation is slow is not a bound.
    let prework_budget = timeout.mul_f64(PREWORK_SHARE);

    for (i, _objective) in objectives.iter().enumerate() {
        let elapsed = start_time.elapsed();
        if elapsed >= timeout {
            return Err(PyValueError::new_err(
                "Timeout exceeded while computing ideal bounds",
            ));
        }
        let prework_remaining = prework_budget
            .checked_sub(elapsed)
            .unwrap_or(Duration::from_secs(0));

        // Prefer the *lexicographic* extreme, and take the ideal value from it
        // rather than solving for the ideal separately: Gurobi runs the two
        // priority passes internally, and its first pass is exactly the plain
        // single-objective solve, so its optimum for objective `i` is the ideal
        // value. Doing both is two redundant full solves per objective, which
        // on the larger instances consumed the entire budget before the
        // first-phase method was even called.
        //
        // The plain solve remains the fallback, because it is what defines
        // `ideal_bounds`, and those are required. It is also far cheaper --
        // seconds against the lexicographic solve's minutes on the larger
        // instances -- which is what makes it a usable fallback when the
        // preparation budget runs out. It returns an arbitrary optimum among
        // ties, which is only weakly efficient (measured on tokyo_bay_225 as
        // 3.6% excess cloud at the min-cost extreme, and 0.8% excess cost at
        // the zero-cloud extreme), so it never serves as an extreme point --
        // only as the ideal value.
        let lex = if objectives.len() == 2 && !prework_remaining.is_zero() {
            let other = 1 - i;
            match augmecon::single_objective::SingleObjectiveSolver::new(&problem, &options)
                .solve_lexicographic(i, other, Some(prework_remaining))
            {
                Ok(lex) if lex.feasible => Some(lex),
                Ok(_) => {
                    log::warn!("Hierarchical extreme {i} infeasible, falling back to plain solve");
                    None
                }
                Err(e) => {
                    log::warn!("Hierarchical extreme {i} failed: {e}, falling back to plain solve");
                    None
                }
            }
        } else {
            if objectives.len() == 2 {
                log::warn!(
                    "Preparation budget spent; taking objective {i}'s extreme from a plain solve"
                );
            }
            None
        };

        let ideal = if let Some(lex) = lex {
            let ideal = lex.objective_values[i];
            lex_extremes.push(lex);
            ideal
        } else {
            let timeout_remaining = timeout
                .checked_sub(start_time.elapsed())
                .unwrap_or(Duration::from_secs(0));
            let solution =
                augmecon::single_objective::SingleObjectiveSolver::new(&problem, &options)
                    .solve_objective(i, Some(timeout_remaining))
                    .map_err(|e| {
                        PyValueError::new_err(format!(
                            "Failed to compute ideal for objective {i}: {e}"
                        ))
                    })?;
            if !solution.feasible {
                return Err(PyValueError::new_err(format!(
                    "Problem infeasible when minimizing objective {i}"
                )));
            }
            solution.objective_values[i]
        };

        ideal_bounds.push(ideal);
        info!("Ideal value for objective {} ({}): {}", i, objectives[i], ideal);
    }

    // Cap each ε-subproblem so no single hard solve consumes the whole GPBA budget:
    // spread the budget across many solves for a denser coverage front. On the cap the
    // solver returns its incumbent (a valid feasible Pareto candidate). Bounded to a
    // sensible [10s, 45s] window regardless of total budget.
    // `GPBA_PER_SOLVE_CAP` overrides the cap for A/B testing: a value in
    // seconds sets it directly, and `0` removes it so every subproblem is
    // solved to optimality. The default below spreads the budget across many
    // solves, which raises the point count on instances where a subproblem
    // finishes well inside the cap -- but on hard instances every subproblem
    // hits it, and a capped solve still reports its incumbent as a Pareto
    // point, so the count falls *and* the points are no longer proven optimal.
    let per_solve_cap = match std::env::var("GPBA_PER_SOLVE_CAP")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
    {
        Some(0) => timeout,
        Some(secs) => Duration::from_secs(secs).min(timeout),
        None => timeout
            .checked_div(6)
            .unwrap_or(timeout)
            .clamp(Duration::from_secs(10), Duration::from_secs(45)),
    };
    info!("GPBA/A&N per-solve cap: {per_solve_cap:?} (global budget {timeout:?})");

    // Configure GPBA-A for Python-compatible dynamic interval exploration (gamma=1)
    let config = GpbaConfig {
        primary_objective: 0, // First objective is primary
        manual_bounds: Some((ideal_bounds, nadir_heuristic)), // Use computed ideal + heuristic nadir
        target_solutions: None,
        per_solve_timeout: Some(per_solve_cap),
    };

    // GPBA-A (ε-constraint coverage bisection) or Anytime Aneja & Nair
    // (weighted-sum dichotomic search). Both return a ParetoFront.
    let pareto_front = if method.eq_ignore_ascii_case("quadtree") || method.eq_ignore_ascii_case("qt")
    {
        let remaining = method_budget(timeout, start_time.elapsed());
        run_quadtree_dispatch(RunPsbox {
            problem: &problem,
            num_objectives: objectives.len(),
            timeout: remaining,
            per_solve_cap,
            lex_extremes: &lex_extremes,
        })?
    } else if method.eq_ignore_ascii_case("psbox") {
        // The box search is charged only the time that is left. The corners
        // above already cost four solves, and its guarantee is a bound on total
        // wall-clock, so handing it the full budget a second time would
        // overrun the limit by however long they took.
        let remaining = method_budget(timeout, start_time.elapsed());
        run_psbox_dispatch(RunPsbox {
            problem: &problem,
            num_objectives: objectives.len(),
            timeout: remaining,
            per_solve_cap,
            lex_extremes: &lex_extremes,
        })?
    } else if method.eq_ignore_ascii_case("aneja")
        || method.eq_ignore_ascii_case("aneja_nair")
        || method.eq_ignore_ascii_case("an")
    {
        info!("Using Anytime Aneja & Nair first-phase method");
        let an_config = augmecon::aneja_nair::AnejaNairConfig {
            per_solve_timeout: Some(per_solve_cap),
            target_solutions: None,
        };
        let mut an = augmecon::aneja_nair::AnejaNair::new(an_config);
        let remaining = method_budget(timeout, start_time.elapsed());
        if !remaining.is_zero() {
            an = an.with_timeout(remaining);
        }
        an.generate_representation(&problem, &options)
            .map_err(|e| PyValueError::new_err(format!("Aneja & Nair solving failed: {e}")))?
    } else {
        info!(
            "Using GPBA-A algorithm with Python-compatible dynamic interval exploration (gamma=1)"
        );
        let mut gpba_a = GpbaA::new(config);
        let remaining = method_budget(timeout, start_time.elapsed());
        if !remaining.is_zero() {
            gpba_a = gpba_a.with_timeout(remaining);
        }
        gpba_a
            .generate_representation(&problem, &options)
            .map_err(|e| PyValueError::new_err(format!("GPBA-A solving failed: {e}")))?
    };

    let mut pareto_front = pareto_front;

    for extreme in lex_extremes {
        pareto_front.add_solution_with_precision(extreme, 0);
    }

    let n_before = pareto_front.solutions.len();

    // Filter step of Algorithm 1 in the GPBA-A paper (line 23,
    // `N(Z) <- Filter(N̂(Z))`): the accumulated set "may contain weakly
    // non-dominated criterion vectors" and has to be filtered before it is
    // reported as the Pareto front. This matters here beyond weak efficiency:
    // each epsilon-subproblem is capped (see `per_solve_cap`) and returns its
    // incumbent on the cap, so the set can contain unproven points that a
    // later, better solve dominates.
    pareto_front.filter_dominated_solutions();
    info!(
        "Front after filtering: {} -> {} solutions",
        n_before,
        pareto_front.solutions.len()
    );

    let pareto_solutions = &pareto_front.solutions;

    info!(
        "GPBA-A solving completed with {} solutions, converting",
        pareto_solutions.len()
    );

    // Convert augmecon solutions to our Solution format
    let mut python_solutions = Vec::new();

    for (i, milp_solution) in pareto_solutions.iter().enumerate() {
        debug!(
            "Processing MILP solution {}: objectives = {:?}",
            i,
            milp_solution.objectives()
        );

        // Extract selected images from decision variables
        let mut selected_images = Vec::new();

        for img_idx in 0..sims_instance.num_images {
            let var_name = format!("x_{img_idx}");
            if let Some(value) = milp_solution.get_variable(&var_name) {
                if value > 0.5 {
                    // Binary variable is "true"
                    selected_images.push(img_idx);
                }
            }
        }

        // Skip invalid solutions with no selected images (zero coverage)
        if selected_images.is_empty() {
            debug!("Skipping invalid solution {} with no selected images", i);
            continue;
        }

        // Sort selected images for consistent ordering
        selected_images.sort_unstable();

        // Compute objective values directly from problem and selected images
        // This ensures exact integer values matching Python's computation
        let mut cost: Option<u64> = None;
        let mut cloudy_area: Option<u64> = None;
        let mut max_incidence_angle: Option<u64> = None;
        let mut min_resolution_sum: Option<u64> = None;

        for obj_name in objectives.iter() {
            match obj_name.as_str() {
                "min_cost" => {
                    let computed_cost: i64 = selected_images
                        .iter()
                        .map(|&img_idx| sims_instance.costs[img_idx])
                        .sum();
                    cost = Some(computed_cost as u64);
                }
                "cloud_coverage" => {
                    let cloudy = compute_cloudy_area(&selected_images, sims_instance);
                    cloudy_area = Some(cloudy as u64);
                }
                "max_incidence_angle" => {
                    if let Some(max_angle) = selected_images
                        .iter()
                        .map(|&img_idx| sims_instance.incidence_angle[img_idx])
                        .max()
                    {
                        max_incidence_angle = Some(max_angle as u64);
                    }
                }
                "min_resolution" => {
                    let res_sum = compute_min_resolutions_sum(&selected_images, sims_instance);
                    min_resolution_sum = Some(res_sum as u64);
                }
                _ => {}
            }
        }

        // Real GPBA-A discovery timestamp (µs since solve start), recorded in the
        // augmecon solution's metadata; fall back to index-based if absent.
        let timestamp_us = milp_solution
            .metadata
            .get("timestamp_us")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(i as u64 * 1000);

        // Create Python solution with computed objective values
        let py_solution = Solution::create(
            selected_images.clone(),
            cost,
            cloudy_area,
            timestamp_us,
            max_incidence_angle,
            min_resolution_sum,
        )?;

        debug!(
            "Converted MILP solution {}: selected_images={:?}",
            i,
            py_solution.get_selected_images_list()
        );

        python_solutions.push(py_solution);
    }

    let original_count = python_solutions.len();
    info!(
        "Converted {} MILP solutions to Python format",
        original_count
    );

    // Sort by the objectives that were actually optimized for consistent ordering
    python_solutions.sort_by(|a, b| {
        let mut cmp_result = std::cmp::Ordering::Equal;
        for obj_name in objectives.iter() {
            cmp_result = match obj_name.as_str() {
                "min_cost" => a.cost.cmp(&b.cost),
                "cloud_coverage" => a.cloudy_area.cmp(&b.cloudy_area),
                "max_incidence_angle" => a.max_incidence_angle.cmp(&b.max_incidence_angle),
                "min_resolution" => a.min_resolutions_sum.cmp(&b.min_resolutions_sum),
                _ => std::cmp::Ordering::Equal,
            };
            if cmp_result != std::cmp::Ordering::Equal {
                break;
            }
        }
        cmp_result
    });

    // Deduplicate consecutive identical solutions (based on selected_images)
    python_solutions.dedup_by(|a, b| a.selected_images == b.selected_images);

    if python_solutions.len() < original_count {
        info!(
            "Removed {} duplicate solutions, {} unique solutions remaining",
            original_count - python_solutions.len(),
            python_solutions.len()
        );
    }

    info!(
        "Successfully returning {} unique MILP solutions",
        python_solutions.len()
    );

    Ok(crate::solution::SolvingResult::new(python_solutions))
}

/// Solves the SIMS problem using a hybrid approach: MILP first, then PLS with MILP solutions as initial population
#[cfg(feature = "milp")]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    milp_config,
    pls_config,
    ratio,
    timeout=Duration::from_secs(300),
    trace=true
))]
pub fn solve_with_hybrid(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    milp_config: &MilpConfig,
    pls_config: &PlsConfig,
    ratio: (i32, i32),
    timeout: Duration,
    trace: bool,
) -> PyResult<SolvingResult> {
    // TODO: Implement trace support for hybrid algorithm
    // For now, trace parameter is accepted but not used
    let _trace = trace; // Silence unused variable warning

    let total_ratio = ratio.0 + ratio.1;
    if total_ratio != 100 {
        return Err(PyValueError::new_err(format!(
            "Ratio values must sum to 100 (representing percentages), got {} + {} = {}",
            ratio.0, ratio.1, total_ratio
        )));
    }
    if ratio.0 < 0 || ratio.1 < 0 {
        return Err(PyValueError::new_err("Ratio values cannot be negative"));
    }
    if ratio.0 == 0 && ratio.1 == 0 {
        return Err(PyValueError::new_err("Both ratio values cannot be zero"));
    }

    let milp_ratio = ratio.0 as f64 / 100.0;
    let pls_ratio = ratio.1 as f64 / 100.0;

    // Calculate timeouts for each phase based on the total timeout and ratio
    let milp_timeout = Duration::from_secs_f64(timeout.as_secs_f64() * milp_ratio);
    let pls_timeout = Duration::from_secs_f64(timeout.as_secs_f64() * pls_ratio);

    info!(
        "Starting hybrid algorithm: MILP for {:.1}s ({:.1}%), then PLS for {:.1}s ({:.1}%)",
        milp_timeout.as_secs_f64(),
        milp_ratio * 100.0,
        pls_timeout.as_secs_f64(),
        pls_ratio * 100.0
    );

    // Handle pure algorithm cases
    if ratio.0 == 0 {
        // Pure PLS case
        info!("Pure PLS algorithm (ratio 0:100)");
        let solving_result = solve_with_pls(
            py,
            sims_instance,
            pls_config.objectives.clone(),
            pls_config.plots,
            pls_config.plot_output_path.clone(),
            timeout, // Use full timeout for PLS
            pls_config.max_iterations,
            pls_config.is_deterministic,
            pls_config.initial_population_size,
            None, // No initial population provided
            pls_config.neighborhood_size_min,
            pls_config.neighborhood_size_max,
            false,                 // No trace for internal hybrid calls
            None,                  // No objective bounds
            false,                 // Don't include dominated solutions
            "nd-tree".to_string(), // Default pareto archive
            false,                 // No profiling trace for hybrid
            false,                 // parallel disabled for internal hybrid calls
            0,                     // num_parallel_threads (ignored)
            None,                  // neighborhood_budget
            true,                  // use_checkpoint
            true,                  // use_ranked_candidates
            15,                    // max_k1_candidates
            None,                  // probing_budget
            true,                  // use_greedy_initial_population
            false,                 // use_perturbation_restart (see the signature default)
            false,                 // use_diverse_probing
            None,                  // diverse_probe_budget
            true,                  // use_nd_tree_scalarized_query
            None,                  // solution_selection_mode
            None,                  // scalarized_selection_source
            None,                  // scalarized_parent_budget
            None,                  // scalarized_weight_samples
            None,                  // scalarized_rho
        )?;
        return Ok(solving_result);
    }

    if ratio.1 == 0 {
        // Pure MILP case
        info!("Pure MILP algorithm (ratio 100:0)");
        return solve_with_milp(
            sims_instance,
            milp_config.objectives.clone(),
            milp_config.grid_points,
            timeout, // Use full timeout for MILP
            milp_config.bypass_coefficient,
            milp_config.early_exit,
            milp_config.flag_array,
            milp_config.solver_name.clone(),
            "gpba".to_string(),
        );
    }

    // Phase 1: Run MILP to get initial solutions
    info!("Phase 1: Running MILP algorithm");
    let milp_solutions = solve_with_milp(
        sims_instance,
        milp_config.objectives.clone(),
        milp_config.grid_points,
        milp_timeout,
        milp_config.bypass_coefficient,
        milp_config.early_exit,
        milp_config.flag_array,
        milp_config.solver_name.clone(),
        "gpba".to_string(),
    )?;

    info!(
        "MILP phase completed with {} solutions",
        milp_solutions.final_solutions.len()
    );

    if milp_solutions.final_solutions.is_empty() {
        info!("MILP found no solutions, falling back to PLS only");
        let solving_result = solve_with_pls(
            py,
            sims_instance,
            pls_config.objectives.clone(),
            pls_config.plots,
            pls_config.plot_output_path.clone(),
            pls_timeout,
            pls_config.max_iterations,
            pls_config.is_deterministic,
            pls_config.initial_population_size,
            None, // No initial population provided
            pls_config.neighborhood_size_min,
            pls_config.neighborhood_size_max,
            false,                 // No trace for internal hybrid calls
            None,                  // No objective bounds
            false,                 // Don't include dominated solutions
            "nd-tree".to_string(), // Default pareto archive
            false,                 // No profiling trace for hybrid
            false,                 // parallel disabled for internal hybrid calls
            0,                     // num_parallel_threads (ignored)
            None,                  // neighborhood_budget
            true,                  // use_checkpoint
            true,                  // use_ranked_candidates
            15,                    // max_k1_candidates
            None,                  // probing_budget
            true,                  // use_greedy_initial_population
            false,                 // use_perturbation_restart (see the signature default)
            false,                 // use_diverse_probing
            None,                  // diverse_probe_budget
            true,                  // use_nd_tree_scalarized_query
            None,                  // solution_selection_mode
            None,                  // scalarized_selection_source
            None,                  // scalarized_parent_budget
            None,                  // scalarized_weight_samples
            None,                  // scalarized_rho
        )?;
        return Ok(solving_result);
    }

    // Phase 2: Run PLS with MILP solutions as initial population
    info!(
        "Phase 2: Running PLS with {} MILP solutions as initial population",
        milp_solutions.final_solutions.len()
    );

    // Validate objectives consistency
    if milp_config.objectives != pls_config.objectives {
        return Err(PyValueError::new_err(
            "MILP and PLS must use the same objectives for hybrid approach",
        ));
    }

    let objectives = &pls_config.objectives;
    let valid_objectives = [
        "min_cost",
        "cloud_coverage",
        "min_resolution",
        "max_incidence_angle",
    ];
    for obj in objectives {
        if !valid_objectives.contains(&obj.as_str()) {
            return Err(PyValueError::new_err(format!(
                "Invalid objective '{obj}'. Valid objectives are: {valid_objectives:?}"
            )));
        }
    }

    // Determine if multiobjective based on objectives
    let objectives = &pls_config.objectives;

    let neighborhood_size_range: RangeInclusive<u32> =
        pls_config.neighborhood_size_min..=pls_config.neighborhood_size_max;

    // Dispatch to the appropriate dimensional solver based on number of objectives
    match objectives.len() {
        2 => solve_hybrid_2d(
            sims_instance,
            milp_solutions.final_solutions,
            pls_config,
            pls_timeout,
            neighborhood_size_range,
        ),
        3 => solve_hybrid_3d(
            sims_instance,
            milp_solutions.final_solutions,
            pls_config,
            pls_timeout,
            neighborhood_size_range,
        ),
        4 => solve_hybrid_4d(
            sims_instance,
            milp_solutions.final_solutions,
            pls_config,
            pls_timeout,
            neighborhood_size_range,
        ),
        n => Err(PyValueError::new_err(format!(
            "Unsupported number of objectives: {n}. Supported: 2, 3, or 4 objectives."
        ))),
    }
}

/// 2D hybrid solver implementation
#[cfg(feature = "milp")]
fn solve_hybrid_2d(
    sims_instance: &SimsDiscreteProblem,
    milp_solutions: Vec<Solution>,
    pls_config: &PlsConfig,
    pls_timeout: Duration,
    neighborhood_size_range: RangeInclusive<u32>,
) -> PyResult<SolvingResult> {
    use pls::problem_bitset::ProblemBitset;
    use pls::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;
    use pls::solution_set_impl::BTreeSolutionSet;

    debug!("Using 2D hybrid optimization");

    // Convert to PLS problem format and create 2D problem with specified objectives
    // Note: sims_instance already has 0-based indices from Python, and
    // from_raw_with_objectives expects 0-based indices, so we pass them directly
    let raw_instance = pls::problem::SIMSProblemInstanceRaw {
        name: "python_hybrid_instance".to_string(),
        num_images: sims_instance.num_images,
        universe_size: sims_instance.universe,
        images: sims_instance.images.clone(),
        costs: sims_instance.costs.iter().map(|&c| c as u64).collect(),
        clouds: sims_instance.clouds.clone(),
        areas: sims_instance.areas.iter().map(|&a| a as u64).collect(),
        max_cloud_area: sims_instance.max_cloud_area as u64,
        resolution: sims_instance.resolution.iter().map(|&r| r as u64).collect(),
        incidence_angle: sims_instance
            .incidence_angle
            .iter()
            .map(|&i| i as u64)
            .collect(),
    };

    // Create 2D problem with specified objectives inline
    let mut objective_definitions = [
        pls::objectives::ObjectiveType::TotalCost,
        pls::objectives::ObjectiveType::CloudyArea,
    ];
    for (i, obj_name) in pls_config.objectives.iter().enumerate() {
        objective_definitions[i] = match obj_name.as_str() {
            "min_cost" => pls::objectives::ObjectiveType::TotalCost,
            "cloud_coverage" => pls::objectives::ObjectiveType::CloudyArea,
            "min_resolution" => pls::objectives::ObjectiveType::MinResolution,
            "max_incidence_angle" => pls::objectives::ObjectiveType::MaxIncidenceAngle,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown objective: {}",
                    obj_name
                )))
            }
        };
    }

    let pls_problem = ProblemBitset::from_raw_with_objectives(&raw_instance, objective_definitions);

    // Convert MILP solutions to PLS initial solutions
    let mut initial_solutions = Vec::new();
    for milp_sol in &milp_solutions {
        let selected_images = milp_sol.get_selected_images_list();

        // Create PLS solution from selected images
        let pls_solution =
            BitsetEncodedSolution::from_selected_images(&selected_images, &pls_problem);
        initial_solutions.push(pls_solution);
    }

    // Create additional random solutions if needed
    let initial_solution_set = if initial_solutions.len() < pls_config.initial_population_size {
        let remaining_size = pls_config.initial_population_size - initial_solutions.len();
        info!("Adding {remaining_size} random solutions to reach desired population size");

        // Create random solutions manually
        let mut random_solutions: BTreeSolutionSet<BitsetEncodedSolution<ProblemBitset<2>, 2>, 2> =
            BTreeSolutionSet::new("random_2d_solutions");
        for _ in 0..remaining_size {
            let random_solution = if pls_config.is_deterministic {
                BitsetEncodedSolution::random_with_seed(&pls_problem, 1_234_567_890)
            } else {
                BitsetEncodedSolution::random(&pls_problem)
            };
            random_solutions.try_insert(&random_solution);
        }

        let mut combined_set = BTreeSolutionSet::new("hybrid_2d_solutions");
        // Add MILP solutions
        for sol in initial_solutions {
            combined_set.try_insert(&sol);
        }
        // Add random solutions
        for sol in random_solutions.into_iter() {
            combined_set.try_insert(&sol);
        }
        combined_set
    } else {
        // Use only MILP solutions
        let mut solution_set = BTreeSolutionSet::new("hybrid_2d_milp_only");
        for sol in initial_solutions
            .into_iter()
            .take(pls_config.initial_population_size)
        {
            solution_set.try_insert(&sol);
        }
        solution_set
    };

    info!(
        "Created initial population of {} solutions for 2D PLS",
        initial_solution_set.len()
    );

    // Create and run PLS
    let mut pareto_local_search = ParetoLocalSearch::new(
        &pls_problem,
        &initial_solution_set,
        neighborhood_size_range,
        pls_config.is_deterministic,
        PlsOptimizations::default(),
    );

    info!(
        "Starting 2D PLS phase with {} iterations",
        pls_config.max_iterations
    );
    let final_solution_set = pareto_local_search.run(pls_config.max_iterations, pls_timeout);

    info!(
        "Hybrid 2D algorithm completed with {} final solutions",
        final_solution_set.len()
    );

    // Generate plots if requested
    if pls_config.plots {
        #[cfg(feature = "plotting")]
        {
            let objective_names = pls_problem.objective_names();
            pls::plotting::draw_solutions_plot(
                &pareto_local_search.explored_solutions,
                &objective_names,
            );

            if let Some(path) = &pls_config.plot_output_path {
                if path != "pareto_solutions_2d.svg" {
                    if let Err(e) = std::fs::rename("pareto_solutions_2d.svg", path) {
                        log::warn!("Failed to move plot to {path}: {e}");
                    }
                }
            }
        }
        #[cfg(not(feature = "plotting"))]
        {
            log::warn!("Plotting requested but plotting feature is not enabled");
        }
    }

    // Convert final solutions to Python format
    let final_solutions: Vec<BitsetEncodedSolution<ProblemBitset<2>, 2>> =
        final_solution_set.into_iter().collect();
    let mut python_solutions = Vec::new();

    for solution in final_solutions.iter() {
        let py_solution: Solution = (solution, &pls_problem).into();
        python_solutions.push(py_solution);
    }

    info!(
        "Successfully converted {} hybrid 2D solutions",
        python_solutions.len()
    );

    Ok(crate::solution::SolvingResult::new(python_solutions))
}

/// 3D hybrid solver implementation
#[cfg(feature = "milp")]
fn solve_hybrid_3d(
    sims_instance: &SimsDiscreteProblem,
    milp_solutions: Vec<Solution>,
    pls_config: &PlsConfig,
    pls_timeout: Duration,
    neighborhood_size_range: RangeInclusive<u32>,
) -> PyResult<SolvingResult> {
    use pls::problem_bitset::ProblemBitset;
    use pls::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;
    use pls::solution_set_impl::NdTreeSolutionSet;

    debug!("Using 3D hybrid optimization");

    // Convert to PLS problem format
    // Note: sims_instance already has 0-based indices from Python, and
    // from_raw_with_objectives expects 0-based indices, so we pass them directly
    let raw_instance = pls::problem::SIMSProblemInstanceRaw {
        name: "python_hybrid_instance".to_string(),
        num_images: sims_instance.num_images,
        universe_size: sims_instance.universe,
        images: sims_instance.images.clone(),
        costs: sims_instance.costs.iter().map(|&c| c as u64).collect(),
        clouds: sims_instance.clouds.clone(),
        areas: sims_instance.areas.iter().map(|&a| a as u64).collect(),
        max_cloud_area: sims_instance.max_cloud_area as u64,
        resolution: sims_instance.resolution.iter().map(|&r| r as u64).collect(),
        incidence_angle: sims_instance
            .incidence_angle
            .iter()
            .map(|&i| i as u64)
            .collect(),
    };

    // Create 3D problem with specified objectives inline
    let mut objective_definitions = [
        pls::objectives::ObjectiveType::TotalCost,
        pls::objectives::ObjectiveType::CloudyArea,
        pls::objectives::ObjectiveType::MinResolution,
    ];
    for (i, obj_name) in pls_config.objectives.iter().enumerate() {
        objective_definitions[i] = match obj_name.as_str() {
            "min_cost" => pls::objectives::ObjectiveType::TotalCost,
            "cloud_coverage" => pls::objectives::ObjectiveType::CloudyArea,
            "min_resolution" => pls::objectives::ObjectiveType::MinResolution,
            "max_incidence_angle" => pls::objectives::ObjectiveType::MaxIncidenceAngle,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown objective: {}",
                    obj_name
                )))
            }
        };
    }

    let pls_problem = ProblemBitset::from_raw_with_objectives(&raw_instance, objective_definitions);

    // Convert MILP solutions to PLS initial solutions
    let mut initial_solutions = Vec::new();
    for milp_sol in &milp_solutions {
        let selected_images = milp_sol.get_selected_images_list();

        // Create PLS solution from selected images
        let pls_solution =
            BitsetEncodedSolution::from_selected_images(&selected_images, &pls_problem);
        initial_solutions.push(pls_solution);
    }

    // Create additional random solutions if we have fewer MILP solutions than desired population size
    let initial_solution_set = if initial_solutions.len() < pls_config.initial_population_size {
        let remaining_size = pls_config.initial_population_size - initial_solutions.len();
        info!("Adding {remaining_size} random solutions to reach desired population size");

        // Create random solutions manually
        let mut random_solutions: NdTreeSolutionSet<BitsetEncodedSolution<ProblemBitset<3>, 3>, 3> =
            NdTreeSolutionSet::new("random_3d_solutions");
        for _ in 0..remaining_size {
            let random_solution = if pls_config.is_deterministic {
                BitsetEncodedSolution::random_with_seed(&pls_problem, 1_234_567_890)
            } else {
                BitsetEncodedSolution::random(&pls_problem)
            };
            random_solutions.try_insert(&random_solution);
        }

        let mut combined_set = NdTreeSolutionSet::new("hybrid_3d_solutions");
        // Add MILP solutions
        for sol in initial_solutions {
            combined_set.try_insert(&sol);
        }
        // Add random solutions
        for sol in random_solutions.into_iter() {
            combined_set.try_insert(&sol);
        }
        combined_set
    } else {
        // Use only MILP solutions (truncate if we have too many)
        let mut solution_set = NdTreeSolutionSet::new("hybrid_3d_milp_only");
        for sol in initial_solutions
            .into_iter()
            .take(pls_config.initial_population_size)
        {
            solution_set.try_insert(&sol);
        }
        solution_set
    };

    info!(
        "Created initial population of {} solutions for 3D PLS",
        initial_solution_set.len()
    );

    // Create and run PLS
    let mut pareto_local_search = ParetoLocalSearch::new(
        &pls_problem,
        &initial_solution_set,
        neighborhood_size_range,
        pls_config.is_deterministic,
        PlsOptimizations::default(),
    );

    info!(
        "Starting 3D PLS phase with {} iterations",
        pls_config.max_iterations
    );
    let final_solution_set = pareto_local_search.run(pls_config.max_iterations, pls_timeout);

    info!(
        "Hybrid 3D algorithm completed with {} final solutions",
        final_solution_set.len()
    );

    // Generate plots if requested
    if pls_config.plots {
        #[cfg(feature = "plotting")]
        {
            let objective_names = pls_problem.objective_names();
            pls::plotting::draw_solutions_plot(
                &pareto_local_search.explored_solutions,
                &objective_names,
            );

            if let Some(path) = &pls_config.plot_output_path {
                if path != "pareto_solutions_grid.svg" {
                    if let Err(e) = std::fs::rename("pareto_solutions_grid.svg", path) {
                        log::warn!("Failed to move plot to {path}: {e}");
                    }
                }
            }
        }
        #[cfg(not(feature = "plotting"))]
        {
            log::warn!("Plotting requested but plotting feature is not enabled");
        }
    }

    // Convert final solutions to Python format
    let final_solutions: Vec<BitsetEncodedSolution<ProblemBitset<3>, 3>> =
        final_solution_set.into_iter().collect();
    let mut python_solutions = Vec::new();

    for solution in final_solutions.iter() {
        let py_solution: Solution = (solution, &pls_problem).into();
        python_solutions.push(py_solution);
    }

    info!(
        "Successfully converted {} hybrid 3D solutions",
        python_solutions.len()
    );
    Ok(crate::solution::SolvingResult::new(python_solutions))
}

/// 4D hybrid solver implementation
#[cfg(feature = "milp")]
fn solve_hybrid_4d(
    sims_instance: &SimsDiscreteProblem,
    milp_solutions: Vec<Solution>,
    pls_config: &PlsConfig,
    pls_timeout: Duration,
    neighborhood_size_range: RangeInclusive<u32>,
) -> PyResult<SolvingResult> {
    use pls::problem_bitset::ProblemBitset;
    use pls::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;
    use pls::solution_set_impl::NdTreeSolutionSet;

    debug!("Using 4D hybrid optimization");

    // Convert to PLS problem format
    // Note: sims_instance already has 0-based indices from Python, and
    // from_raw_with_objectives expects 0-based indices, so we pass them directly
    let raw_instance = pls::problem::SIMSProblemInstanceRaw {
        name: "python_hybrid_instance".to_string(),
        num_images: sims_instance.num_images,
        universe_size: sims_instance.universe,
        images: sims_instance.images.clone(),
        costs: sims_instance.costs.iter().map(|&c| c as u64).collect(),
        clouds: sims_instance.clouds.clone(),
        areas: sims_instance.areas.iter().map(|&a| a as u64).collect(),
        max_cloud_area: sims_instance.max_cloud_area as u64,
        resolution: sims_instance.resolution.iter().map(|&r| r as u64).collect(),
        incidence_angle: sims_instance
            .incidence_angle
            .iter()
            .map(|&i| i as u64)
            .collect(),
    };

    // Create 4D problem with specified objectives inline
    let mut objective_definitions = [
        pls::objectives::ObjectiveType::TotalCost,
        pls::objectives::ObjectiveType::CloudyArea,
        pls::objectives::ObjectiveType::MinResolution,
        pls::objectives::ObjectiveType::MaxIncidenceAngle,
    ];
    for (i, obj_name) in pls_config.objectives.iter().enumerate() {
        objective_definitions[i] = match obj_name.as_str() {
            "min_cost" => pls::objectives::ObjectiveType::TotalCost,
            "cloud_coverage" => pls::objectives::ObjectiveType::CloudyArea,
            "min_resolution" => pls::objectives::ObjectiveType::MinResolution,
            "max_incidence_angle" => pls::objectives::ObjectiveType::MaxIncidenceAngle,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown objective: {}",
                    obj_name
                )))
            }
        };
    }

    let pls_problem = ProblemBitset::from_raw_with_objectives(&raw_instance, objective_definitions);

    // Convert MILP solutions to PLS initial solutions
    let mut initial_solutions = Vec::new();
    for milp_sol in &milp_solutions {
        let selected_images = milp_sol.get_selected_images_list();

        // Create PLS solution from selected images
        let pls_solution =
            BitsetEncodedSolution::from_selected_images(&selected_images, &pls_problem);
        initial_solutions.push(pls_solution);
    }

    // Create additional random solutions if we have fewer MILP solutions than desired population size
    let initial_solution_set = if initial_solutions.len() < pls_config.initial_population_size {
        let remaining_size = pls_config.initial_population_size - initial_solutions.len();
        info!("Adding {remaining_size} random solutions to reach desired population size");

        // Create random solutions manually
        let mut random_solutions: NdTreeSolutionSet<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4> =
            NdTreeSolutionSet::new("random_4d_solutions");
        for _ in 0..remaining_size {
            let random_solution = if pls_config.is_deterministic {
                BitsetEncodedSolution::random_with_seed(&pls_problem, 1_234_567_890)
            } else {
                BitsetEncodedSolution::random(&pls_problem)
            };
            random_solutions.try_insert(&random_solution);
        }

        let mut combined_set = NdTreeSolutionSet::new("hybrid_4d_solutions");
        // Add MILP solutions
        for sol in initial_solutions {
            combined_set.try_insert(&sol);
        }
        // Add random solutions
        for sol in random_solutions.into_iter() {
            combined_set.try_insert(&sol);
        }
        combined_set
    } else {
        // Use only MILP solutions (truncate if we have too many)
        let mut solution_set = NdTreeSolutionSet::new("hybrid_4d_milp_only");
        for sol in initial_solutions
            .into_iter()
            .take(pls_config.initial_population_size)
        {
            solution_set.try_insert(&sol);
        }
        solution_set
    };

    info!(
        "Created initial population of {} solutions for 4D PLS",
        initial_solution_set.len()
    );

    // Create and run PLS
    let mut pareto_local_search = ParetoLocalSearch::new(
        &pls_problem,
        &initial_solution_set,
        neighborhood_size_range,
        pls_config.is_deterministic,
        PlsOptimizations::default(),
    );

    info!(
        "Starting 4D PLS phase with {} iterations",
        pls_config.max_iterations
    );
    let final_solution_set = pareto_local_search.run(pls_config.max_iterations, pls_timeout);

    info!(
        "Hybrid 4D algorithm completed with {} final solutions",
        final_solution_set.len()
    );

    // Generate plots if requested
    if pls_config.plots {
        #[cfg(feature = "plotting")]
        {
            let objective_names = pls_problem.objective_names();
            pls::plotting::draw_solutions_plot(
                &pareto_local_search.explored_solutions,
                &objective_names,
            );

            if let Some(path) = &pls_config.plot_output_path {
                if path != "pareto_solutions_grid.svg" {
                    if let Err(e) = std::fs::rename("pareto_solutions_grid.svg", path) {
                        log::warn!("Failed to move plot to {path}: {e}");
                    }
                }
            }
        }
        #[cfg(not(feature = "plotting"))]
        {
            log::warn!("Plotting requested but plotting feature is not enabled");
        }
    }

    // Convert final solutions to Python format
    let final_solutions: Vec<BitsetEncodedSolution<ProblemBitset<4>, 4>> =
        final_solution_set.into_iter().collect();
    let mut python_solutions = Vec::new();

    for solution in final_solutions.iter() {
        let py_solution: Solution = (solution, &pls_problem).into();
        python_solutions.push(py_solution);
    }

    info!(
        "Successfully converted {} hybrid 4D solutions",
        python_solutions.len()
    );
    Ok(crate::solution::SolvingResult::new(python_solutions))
}

/// Helper function to extract Chrome tracing data from shared buffer
/// Solve the SIMS problem using the NSGA-II evolutionary algorithm.
///
/// Only supports 4D optimization (cost + cloud coverage + incidence angle + resolution).
#[expect(
    clippy::too_many_arguments,
    reason = "It's okay for Python API to have many parameters"
)]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string(), "min_max_incidence_angle".to_string(), "min_resolution".to_string()],
    timeout=Duration::from_secs(120),
    population_size=100usize,
    max_generations=50000usize,
    seed=42u64,
    trace=true,
    objective_bounds=None,
    include_dominated=false,
    crossover_rate=0.9,
    swap_mutation_rate=0.4,
    add_prune_mutation_rate=0.3,
    bitflip_mutation_rate=0.0,
    multi_swap_max_removals=3usize,
    multi_swap_rate=0.2,
    shift_mutation_rate=0.25,
    coverage_biased_crossover_fraction=0.5,
    ensure_mutation=true,
    stagnation_limit=50usize,
    use_contribution_distance=true
))]
pub fn solve_with_nsga2(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: Vec<String>,
    timeout: Duration,
    population_size: usize,
    max_generations: usize,
    seed: u64,
    trace: bool,
    objective_bounds: Option<Vec<Vec<u64>>>,
    include_dominated: bool,
    crossover_rate: f64,
    swap_mutation_rate: f64,
    add_prune_mutation_rate: f64,
    bitflip_mutation_rate: f64,
    multi_swap_max_removals: usize,
    multi_swap_rate: f64,
    shift_mutation_rate: f64,
    coverage_biased_crossover_fraction: f64,
    ensure_mutation: bool,
    stagnation_limit: usize,
    use_contribution_distance: bool,
) -> PyResult<SolvingResult> {
    let config = pls::evolutionary::nsga2::Nsga2Config {
        population_size,
        crossover_rate,
        swap_mutation_rate,
        add_prune_mutation_rate,
        bitflip_mutation_rate,
        multi_swap_max_removals,
        multi_swap_rate,
        shift_mutation_rate,
        coverage_biased_crossover_fraction,
        ensure_mutation,
        stagnation_limit,
        use_contribution_distance,
    };

    match objectives.len() {
        2 => run_nsga2_tailored_dim::<2>(
            py,
            sims_instance,
            &objectives,
            timeout,
            None,
            config,
            max_generations,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        3 => run_nsga2_tailored_dim::<3>(
            py,
            sims_instance,
            &objectives,
            timeout,
            None,
            config,
            max_generations,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        4 => run_nsga2_tailored_dim::<4>(
            py,
            sims_instance,
            &objectives,
            timeout,
            None,
            config,
            max_generations,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        n => Err(PyValueError::new_err(format!(
            "solve_with_nsga2 only supports 2, 3, or 4 objectives, got {n}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_nsga2_tailored_dim<const D: usize>(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: &[String],
    timeout: Duration,
    initial_population: Option<Vec<crate::solution::Solution>>,
    config: pls::evolutionary::nsga2::Nsga2Config,
    max_generations: usize,
    seed: u64,
    trace: bool,
    objective_bounds: &Option<Vec<Vec<u64>>>,
    include_dominated: bool,
) -> PyResult<SolvingResult> {
    use pls::evolutionary::nsga2::Nsga2;
    use pls::problem_bitset::ProblemBitset;

    let raw_instance = build_raw_instance(sims_instance);
    let objective_definitions = parse_objective_definitions::<D>(objectives)?;
    let mut pls_problem =
        ProblemBitset::<D>::from_raw_with_objectives(&raw_instance, objective_definitions);
    apply_objective_bounds(&mut pls_problem, objective_bounds)?;

    let seed_pop = initial_population
        .as_deref()
        .map(|p| py_solutions_to_bitset::<ProblemBitset<D>, D>(p, &pls_problem));

    info!(
        "NSGA-II (tailored) {D}D: {} seed solutions, pop_size={}, max_generations={max_generations}",
        seed_pop.as_ref().map(|p| p.len()).unwrap_or(0),
        config.population_size,
    );

    let (archive_solutions, explored_fingerprints) = py.detach(|| {
        let mut nsga2 = Nsga2::new(&pls_problem, config, seed_pop, seed);
        let archive = nsga2.run(max_generations, timeout);
        let fingerprints: Vec<SolutionFingerprint<D>> = nsga2
            .explored_solutions
            .solutions
            .values()
            .cloned()
            .collect();
        (archive, fingerprints)
    });

    info!(
        "NSGA-II (tailored) completed: {} archive solutions, {} explored solutions",
        archive_solutions.len(),
        explored_fingerprints.len()
    );

    let python_final_solutions: Vec<_> = archive_solutions
        .iter()
        .map(|s| {
            let sol: crate::solution::Solution = (s, &pls_problem).into();
            sol
        })
        .collect();

    build_solving_result(
        python_final_solutions,
        explored_fingerprints,
        objectives,
        objective_bounds,
        timeout,
        format!("NSGA2-{D}D"),
        trace,
        include_dominated,
    )
}

/// Solve the SIMS problem using the MOEA/D evolutionary algorithm.
///
/// Only supports 4D optimization (cost + cloud coverage + incidence angle + resolution).
#[expect(
    clippy::too_many_arguments,
    reason = "It's okay for Python API to have many parameters"
)]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string(), "min_max_incidence_angle".to_string(), "min_resolution".to_string()],
    timeout=Duration::from_secs(120),
    population_size=200usize,
    max_generations=50000usize,
    seed=42u64,
    trace=true,
    objective_bounds=None,
    include_dominated=false,
    num_divisions=99usize,
    neighbourhood_size=20usize,
    delta=0.9,
    max_replacements=3usize,
    crossover_rate=1.0,
    swap_mutation_rate=0.3,
    add_prune_mutation_rate=0.2,
    multi_swap_max_removals=3usize,
    multi_swap_rate=0.15,
    shift_mutation_rate=0.15,
    coverage_biased_crossover_fraction=0.5,
    ensure_mutation=false,
    auto_divisions=true,
    use_pbi=false,
    pbi_theta=5.0,
    stagnation_limit=80usize
))]
pub fn solve_with_moead(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: Vec<String>,
    timeout: Duration,
    population_size: usize,
    max_generations: usize,
    seed: u64,
    trace: bool,
    objective_bounds: Option<Vec<Vec<u64>>>,
    include_dominated: bool,
    num_divisions: usize,
    neighbourhood_size: usize,
    delta: f64,
    max_replacements: usize,
    crossover_rate: f64,
    swap_mutation_rate: f64,
    add_prune_mutation_rate: f64,
    multi_swap_max_removals: usize,
    multi_swap_rate: f64,
    shift_mutation_rate: f64,
    coverage_biased_crossover_fraction: f64,
    ensure_mutation: bool,
    auto_divisions: bool,
    use_pbi: bool,
    pbi_theta: f64,
    stagnation_limit: usize,
) -> PyResult<SolvingResult> {
    let config = pls::evolutionary::moead::MoeadConfig {
        population_size,
        num_divisions,
        target_pop_size: population_size,
        auto_divisions,
        neighbourhood_size,
        delta,
        max_replacements,
        crossover_rate,
        swap_mutation_rate,
        add_prune_mutation_rate,
        multi_swap_max_removals,
        multi_swap_rate,
        shift_mutation_rate,
        coverage_biased_crossover_fraction,
        ensure_mutation,
        use_pbi,
        pbi_theta,
        stagnation_limit,
    };

    match objectives.len() {
        2 => run_moead_tailored_dim::<2>(
            py,
            sims_instance,
            &objectives,
            timeout,
            None,
            config,
            max_generations,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        3 => run_moead_tailored_dim::<3>(
            py,
            sims_instance,
            &objectives,
            timeout,
            None,
            config,
            max_generations,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        4 => run_moead_tailored_dim::<4>(
            py,
            sims_instance,
            &objectives,
            timeout,
            None,
            config,
            max_generations,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        n => Err(PyValueError::new_err(format!(
            "solve_with_moead only supports 2, 3, or 4 objectives, got {n}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_moead_tailored_dim<const D: usize>(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: &[String],
    timeout: Duration,
    initial_population: Option<Vec<crate::solution::Solution>>,
    config: pls::evolutionary::moead::MoeadConfig,
    max_generations: usize,
    seed: u64,
    trace: bool,
    objective_bounds: &Option<Vec<Vec<u64>>>,
    include_dominated: bool,
) -> PyResult<SolvingResult> {
    use pls::evolutionary::moead::Moead;
    use pls::problem_bitset::ProblemBitset;

    let raw_instance = build_raw_instance(sims_instance);
    let objective_definitions = parse_objective_definitions::<D>(objectives)?;
    let mut pls_problem =
        ProblemBitset::<D>::from_raw_with_objectives(&raw_instance, objective_definitions);
    apply_objective_bounds(&mut pls_problem, objective_bounds)?;

    let seed_pop = initial_population
        .as_deref()
        .map(|p| py_solutions_to_bitset::<ProblemBitset<D>, D>(p, &pls_problem));

    info!(
        "MOEA/D (tailored) {D}D: {} seed solutions, target_pop_size={}, max_generations={max_generations}",
        seed_pop.as_ref().map(|p| p.len()).unwrap_or(0),
        config.target_pop_size,
    );

    let (archive_solutions, explored_fingerprints) = py.detach(|| {
        let mut moead = Moead::new(&pls_problem, config, seed_pop, seed);
        let archive = moead.run(max_generations, timeout);
        let fingerprints: Vec<SolutionFingerprint<D>> = moead
            .explored_solutions
            .solutions
            .values()
            .cloned()
            .collect();
        (archive, fingerprints)
    });

    info!(
        "MOEA/D (tailored) completed: {} archive solutions, {} explored solutions",
        archive_solutions.len(),
        explored_fingerprints.len()
    );

    let python_final_solutions: Vec<_> = archive_solutions
        .iter()
        .map(|s| {
            let sol: crate::solution::Solution = (s, &pls_problem).into();
            sol
        })
        .collect();

    build_solving_result(
        python_final_solutions,
        explored_fingerprints,
        objectives,
        objective_bounds,
        timeout,
        format!("MOEAD-{D}D"),
        trace,
        include_dominated,
    )
}

/// Solve the SIMS problem using NSGA-III with reference-point niching (Deb & Jain 2014).
///
/// Only supports 4D optimization (cost + cloud coverage + incidence angle + resolution).
/// NSGA-III is superior to NSGA-II for 4-objective problems: structured reference points
/// replace crowding distance for diversity preservation.
#[expect(
    clippy::too_many_arguments,
    reason = "It's okay for Python API to have many parameters"
)]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string(), "min_max_incidence_angle".to_string(), "min_resolution".to_string()],
    timeout=Duration::from_secs(120),
    target_pop_size=200usize,
    num_divisions=12usize,
    auto_divisions=true,
    max_generations=500000usize,
    seed=42u64,
    trace=true,
    objective_bounds=None,
    include_dominated=false,
    crossover_rate=0.9,
    swap_mutation_rate=0.4,
    add_prune_mutation_rate=0.3,
    bitflip_mutation_rate=0.0,
    multi_swap_max_removals=3usize,
    multi_swap_rate=0.2,
    shift_mutation_rate=0.25,
    coverage_biased_crossover_fraction=0.5,
    ensure_mutation=true,
    stagnation_limit=50usize
))]
pub fn solve_with_nsga3(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: Vec<String>,
    timeout: Duration,
    target_pop_size: usize,
    num_divisions: usize,
    auto_divisions: bool,
    max_generations: usize,
    seed: u64,
    trace: bool,
    objective_bounds: Option<Vec<Vec<u64>>>,
    include_dominated: bool,
    crossover_rate: f64,
    swap_mutation_rate: f64,
    add_prune_mutation_rate: f64,
    bitflip_mutation_rate: f64,
    multi_swap_max_removals: usize,
    multi_swap_rate: f64,
    shift_mutation_rate: f64,
    coverage_biased_crossover_fraction: f64,
    ensure_mutation: bool,
    stagnation_limit: usize,
) -> PyResult<SolvingResult> {
    let config = pls::evolutionary::nsga3::Nsga3Config {
        num_divisions,
        target_pop_size,
        auto_divisions,
        crossover_rate,
        swap_mutation_rate,
        add_prune_mutation_rate,
        bitflip_mutation_rate,
        multi_swap_max_removals,
        multi_swap_rate,
        shift_mutation_rate,
        coverage_biased_crossover_fraction,
        ensure_mutation,
        stagnation_limit,
    };

    match objectives.len() {
        2 => run_nsga3_tailored_dim::<2>(
            py,
            sims_instance,
            &objectives,
            timeout,
            None,
            config,
            max_generations,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        3 => run_nsga3_tailored_dim::<3>(
            py,
            sims_instance,
            &objectives,
            timeout,
            None,
            config,
            max_generations,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        4 => run_nsga3_tailored_dim::<4>(
            py,
            sims_instance,
            &objectives,
            timeout,
            None,
            config,
            max_generations,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        n => Err(PyValueError::new_err(format!(
            "solve_with_nsga3 only supports 2, 3, or 4 objectives, got {n}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_nsga3_tailored_dim<const D: usize>(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: &[String],
    timeout: Duration,
    initial_population: Option<Vec<crate::solution::Solution>>,
    config: pls::evolutionary::nsga3::Nsga3Config,
    max_generations: usize,
    seed: u64,
    trace: bool,
    objective_bounds: &Option<Vec<Vec<u64>>>,
    include_dominated: bool,
) -> PyResult<SolvingResult> {
    use pls::evolutionary::nsga3::Nsga3;
    use pls::problem_bitset::ProblemBitset;

    let raw_instance = build_raw_instance(sims_instance);
    let objective_definitions = parse_objective_definitions::<D>(objectives)?;
    let mut pls_problem =
        ProblemBitset::<D>::from_raw_with_objectives(&raw_instance, objective_definitions);
    apply_objective_bounds(&mut pls_problem, objective_bounds)?;

    let seed_pop = initial_population
        .as_deref()
        .map(|p| py_solutions_to_bitset::<ProblemBitset<D>, D>(p, &pls_problem));

    info!(
        "NSGA-III (tailored) {D}D: {} seed solutions, target_pop_size={}, max_generations={max_generations}",
        seed_pop.as_ref().map(|p| p.len()).unwrap_or(0),
        config.target_pop_size,
    );

    let (archive_solutions, explored_fingerprints) = py.detach(|| {
        let mut nsga3 = Nsga3::new(&pls_problem, config, seed_pop, seed);
        let archive = nsga3.run(max_generations, timeout);
        let fingerprints: Vec<SolutionFingerprint<D>> = nsga3
            .explored_solutions
            .solutions
            .values()
            .cloned()
            .collect();
        (archive, fingerprints)
    });

    info!(
        "NSGA-III (tailored) completed: {} archive solutions, {} explored solutions",
        archive_solutions.len(),
        explored_fingerprints.len()
    );

    let python_final_solutions: Vec<_> = archive_solutions
        .iter()
        .map(|s| {
            let sol: crate::solution::Solution = (s, &pls_problem).into();
            sol
        })
        .collect();

    build_solving_result(
        python_final_solutions,
        explored_fingerprints,
        objectives,
        objective_bounds,
        timeout,
        format!("NSGA3-{D}D"),
        trace,
        include_dominated,
    )
}

/// Solve the SIMS problem using the Memetic hybrid: PLS warm-start → NSGA-II.
///
/// Runs PLS for `pls_time_fraction` of the total budget to build an initial archive,
/// then uses NSGA-II with that archive as seed population for the remaining time.
#[expect(
    clippy::too_many_arguments,
    reason = "It's okay for Python API to have many parameters"
)]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string(), "min_max_incidence_angle".to_string(), "min_resolution".to_string()],
    timeout=Duration::from_secs(120),
    pls_time_fraction=0.3,
    pls_initial_pop_size=50usize,
    max_pls_seed_size=0usize,
    population_size=200usize,
    max_generations=500000usize,
    seed=42u64,
    trace=true,
    objective_bounds=None,
    include_dominated=false,
    crossover_rate=0.9,
    swap_mutation_rate=0.4,
    add_prune_mutation_rate=0.3,
    bitflip_mutation_rate=0.0,
    multi_swap_max_removals=3usize,
    multi_swap_rate=0.2,
    shift_mutation_rate=0.25,
    coverage_biased_crossover_fraction=0.5,
    ensure_mutation=true,
    stagnation_limit=50usize,
    use_contribution_distance=true
))]
pub fn solve_with_memetic_nsga2(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: Vec<String>,
    timeout: Duration,
    pls_time_fraction: f64,
    pls_initial_pop_size: usize,
    max_pls_seed_size: usize,
    population_size: usize,
    max_generations: usize,
    seed: u64,
    trace: bool,
    objective_bounds: Option<Vec<Vec<u64>>>,
    include_dominated: bool,
    crossover_rate: f64,
    swap_mutation_rate: f64,
    add_prune_mutation_rate: f64,
    bitflip_mutation_rate: f64,
    multi_swap_max_removals: usize,
    multi_swap_rate: f64,
    shift_mutation_rate: f64,
    coverage_biased_crossover_fraction: f64,
    ensure_mutation: bool,
    stagnation_limit: usize,
    use_contribution_distance: bool,
) -> PyResult<SolvingResult> {
    let _ = max_generations; // unused but kept for API symmetry with pure EA bindings
    let nsga2_config = pls::evolutionary::nsga2::Nsga2Config {
        population_size,
        crossover_rate,
        swap_mutation_rate,
        add_prune_mutation_rate,
        bitflip_mutation_rate,
        multi_swap_max_removals,
        multi_swap_rate,
        shift_mutation_rate,
        coverage_biased_crossover_fraction,
        ensure_mutation,
        stagnation_limit,
        use_contribution_distance,
    };

    match objectives.len() {
        2 => run_memetic_nsga2_dim::<2>(
            py,
            sims_instance,
            &objectives,
            timeout,
            pls_time_fraction,
            pls_initial_pop_size,
            max_pls_seed_size,
            nsga2_config,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        3 => run_memetic_nsga2_dim::<3>(
            py,
            sims_instance,
            &objectives,
            timeout,
            pls_time_fraction,
            pls_initial_pop_size,
            max_pls_seed_size,
            nsga2_config,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        4 => run_memetic_nsga2_dim::<4>(
            py,
            sims_instance,
            &objectives,
            timeout,
            pls_time_fraction,
            pls_initial_pop_size,
            max_pls_seed_size,
            nsga2_config,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        n => Err(PyValueError::new_err(format!(
            "solve_with_memetic_nsga2 only supports 2, 3, or 4 objectives, got {n}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_memetic_nsga2_dim<const D: usize>(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: &[String],
    timeout: Duration,
    pls_time_fraction: f64,
    pls_initial_pop_size: usize,
    max_pls_seed_size: usize,
    nsga2_config: pls::evolutionary::nsga2::Nsga2Config,
    seed: u64,
    trace: bool,
    objective_bounds: &Option<Vec<Vec<u64>>>,
    include_dominated: bool,
) -> PyResult<SolvingResult> {
    use pls::evolutionary::memetic::{EaBackend, MemeticAlgorithm, MemeticConfig};
    use pls::problem_bitset::ProblemBitset;

    let raw_instance = build_raw_instance(sims_instance);
    let objective_definitions = parse_objective_definitions::<D>(objectives)?;
    let mut pls_problem =
        ProblemBitset::<D>::from_raw_with_objectives(&raw_instance, objective_definitions);
    apply_objective_bounds(&mut pls_problem, objective_bounds)?;

    let config = MemeticConfig {
        pls_time_fraction,
        pls_initial_pop_size,
        max_pls_seed_size,
        ea_backend: EaBackend::Nsga2,
        nsga2_config,
        seed,
        ..MemeticConfig::default()
    };

    let (archive_solutions, explored_fingerprints) = py.detach(|| {
        let result = MemeticAlgorithm::run::<ProblemBitset<D>, D>(&pls_problem, &config, timeout);
        let fingerprints: Vec<SolutionFingerprint<D>> = result
            .explored_solutions
            .solutions
            .values()
            .cloned()
            .collect();
        (result.archive, fingerprints)
    });

    info!(
        "Memetic NSGA-II completed: {} archive solutions, {} explored solutions",
        archive_solutions.len(),
        explored_fingerprints.len()
    );

    let python_final_solutions: Vec<_> = archive_solutions
        .iter()
        .map(|s| {
            let sol: crate::solution::Solution = (s, &pls_problem).into();
            sol
        })
        .collect();

    build_solving_result(
        python_final_solutions,
        explored_fingerprints,
        objectives,
        objective_bounds,
        timeout,
        format!("Memetic-NSGA2-{D}D"),
        trace,
        include_dominated,
    )
}

/// Solve the SIMS problem using the Memetic hybrid: PLS warm-start → NSGA-III.
#[expect(
    clippy::too_many_arguments,
    reason = "It's okay for Python API to have many parameters"
)]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string(), "min_max_incidence_angle".to_string(), "min_resolution".to_string()],
    timeout=Duration::from_secs(120),
    pls_time_fraction=0.3,
    pls_initial_pop_size=50usize,
    max_pls_seed_size=0usize,
    target_pop_size=200usize,
    num_divisions=12usize,
    auto_divisions=true,
    max_generations=500000usize,
    seed=42u64,
    trace=true,
    objective_bounds=None,
    include_dominated=false,
    crossover_rate=0.9,
    swap_mutation_rate=0.4,
    add_prune_mutation_rate=0.3,
    bitflip_mutation_rate=0.0,
    multi_swap_max_removals=3usize,
    multi_swap_rate=0.2,
    shift_mutation_rate=0.25,
    coverage_biased_crossover_fraction=0.5,
    ensure_mutation=true,
    stagnation_limit=50usize
))]
pub fn solve_with_memetic_nsga3(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: Vec<String>,
    timeout: Duration,
    pls_time_fraction: f64,
    pls_initial_pop_size: usize,
    max_pls_seed_size: usize,
    target_pop_size: usize,
    num_divisions: usize,
    auto_divisions: bool,
    max_generations: usize,
    seed: u64,
    trace: bool,
    objective_bounds: Option<Vec<Vec<u64>>>,
    include_dominated: bool,
    crossover_rate: f64,
    swap_mutation_rate: f64,
    add_prune_mutation_rate: f64,
    bitflip_mutation_rate: f64,
    multi_swap_max_removals: usize,
    multi_swap_rate: f64,
    shift_mutation_rate: f64,
    coverage_biased_crossover_fraction: f64,
    ensure_mutation: bool,
    stagnation_limit: usize,
) -> PyResult<SolvingResult> {
    let _ = max_generations;
    let nsga3_config = pls::evolutionary::nsga3::Nsga3Config {
        num_divisions,
        target_pop_size,
        auto_divisions,
        crossover_rate,
        swap_mutation_rate,
        add_prune_mutation_rate,
        bitflip_mutation_rate,
        multi_swap_max_removals,
        multi_swap_rate,
        shift_mutation_rate,
        coverage_biased_crossover_fraction,
        ensure_mutation,
        stagnation_limit,
    };

    match objectives.len() {
        2 => run_memetic_nsga3_dim::<2>(
            py,
            sims_instance,
            &objectives,
            timeout,
            pls_time_fraction,
            pls_initial_pop_size,
            max_pls_seed_size,
            nsga3_config,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        3 => run_memetic_nsga3_dim::<3>(
            py,
            sims_instance,
            &objectives,
            timeout,
            pls_time_fraction,
            pls_initial_pop_size,
            max_pls_seed_size,
            nsga3_config,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        4 => run_memetic_nsga3_dim::<4>(
            py,
            sims_instance,
            &objectives,
            timeout,
            pls_time_fraction,
            pls_initial_pop_size,
            max_pls_seed_size,
            nsga3_config,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        n => Err(PyValueError::new_err(format!(
            "solve_with_memetic_nsga3 only supports 2, 3, or 4 objectives, got {n}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_memetic_nsga3_dim<const D: usize>(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: &[String],
    timeout: Duration,
    pls_time_fraction: f64,
    pls_initial_pop_size: usize,
    max_pls_seed_size: usize,
    nsga3_config: pls::evolutionary::nsga3::Nsga3Config,
    seed: u64,
    trace: bool,
    objective_bounds: &Option<Vec<Vec<u64>>>,
    include_dominated: bool,
) -> PyResult<SolvingResult> {
    use pls::evolutionary::memetic::{EaBackend, MemeticAlgorithm, MemeticConfig};
    use pls::problem_bitset::ProblemBitset;

    let raw_instance = build_raw_instance(sims_instance);
    let objective_definitions = parse_objective_definitions::<D>(objectives)?;
    let mut pls_problem =
        ProblemBitset::<D>::from_raw_with_objectives(&raw_instance, objective_definitions);
    apply_objective_bounds(&mut pls_problem, objective_bounds)?;

    let config = MemeticConfig {
        pls_time_fraction,
        pls_initial_pop_size,
        max_pls_seed_size,
        ea_backend: EaBackend::Nsga3,
        nsga3_config,
        seed,
        ..MemeticConfig::default()
    };

    let (archive_solutions, explored_fingerprints) = py.detach(|| {
        let result = MemeticAlgorithm::run::<ProblemBitset<D>, D>(&pls_problem, &config, timeout);
        let fingerprints: Vec<SolutionFingerprint<D>> = result
            .explored_solutions
            .solutions
            .values()
            .cloned()
            .collect();
        (result.archive, fingerprints)
    });

    info!(
        "Memetic NSGA-III completed: {} archive solutions, {} explored solutions",
        archive_solutions.len(),
        explored_fingerprints.len()
    );

    let python_final_solutions: Vec<_> = archive_solutions
        .iter()
        .map(|s| {
            let sol: crate::solution::Solution = (s, &pls_problem).into();
            sol
        })
        .collect();

    build_solving_result(
        python_final_solutions,
        explored_fingerprints,
        objectives,
        objective_bounds,
        timeout,
        format!("Memetic-NSGA3-{D}D"),
        trace,
        include_dominated,
    )
}

/// Solve the SIMS problem using the Memetic hybrid: PLS warm-start → MOEA/D.
#[expect(
    clippy::too_many_arguments,
    reason = "It's okay for Python API to have many parameters"
)]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string(), "min_max_incidence_angle".to_string(), "min_resolution".to_string()],
    timeout=Duration::from_secs(120),
    pls_time_fraction=0.3,
    pls_initial_pop_size=50usize,
    max_pls_seed_size=0usize,
    population_size=200usize,
    num_divisions=99usize,
    neighbourhood_size=20usize,
    delta=0.9,
    max_replacements=3usize,
    max_generations=500000usize,
    seed=42u64,
    trace=true,
    objective_bounds=None,
    include_dominated=false,
    crossover_rate=1.0,
    swap_mutation_rate=0.3,
    add_prune_mutation_rate=0.2,
    multi_swap_max_removals=3usize,
    multi_swap_rate=0.15,
    shift_mutation_rate=0.15,
    coverage_biased_crossover_fraction=0.5,
    ensure_mutation=false,
    auto_divisions=true,
    use_pbi=false,
    pbi_theta=5.0,
    stagnation_limit=80usize
))]
pub fn solve_with_memetic_moead(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: Vec<String>,
    timeout: Duration,
    pls_time_fraction: f64,
    pls_initial_pop_size: usize,
    max_pls_seed_size: usize,
    population_size: usize,
    num_divisions: usize,
    neighbourhood_size: usize,
    delta: f64,
    max_replacements: usize,
    max_generations: usize,
    seed: u64,
    trace: bool,
    objective_bounds: Option<Vec<Vec<u64>>>,
    include_dominated: bool,
    crossover_rate: f64,
    swap_mutation_rate: f64,
    add_prune_mutation_rate: f64,
    multi_swap_max_removals: usize,
    multi_swap_rate: f64,
    shift_mutation_rate: f64,
    coverage_biased_crossover_fraction: f64,
    ensure_mutation: bool,
    auto_divisions: bool,
    use_pbi: bool,
    pbi_theta: f64,
    stagnation_limit: usize,
) -> PyResult<SolvingResult> {
    let _ = max_generations;
    let moead_config = pls::evolutionary::moead::MoeadConfig {
        population_size,
        num_divisions,
        target_pop_size: population_size,
        auto_divisions,
        neighbourhood_size,
        delta,
        max_replacements,
        crossover_rate,
        swap_mutation_rate,
        add_prune_mutation_rate,
        multi_swap_max_removals,
        multi_swap_rate,
        shift_mutation_rate,
        coverage_biased_crossover_fraction,
        ensure_mutation,
        use_pbi,
        pbi_theta,
        stagnation_limit,
    };

    match objectives.len() {
        2 => run_memetic_moead_dim::<2>(
            py,
            sims_instance,
            &objectives,
            timeout,
            pls_time_fraction,
            pls_initial_pop_size,
            max_pls_seed_size,
            moead_config,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        3 => run_memetic_moead_dim::<3>(
            py,
            sims_instance,
            &objectives,
            timeout,
            pls_time_fraction,
            pls_initial_pop_size,
            max_pls_seed_size,
            moead_config,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        4 => run_memetic_moead_dim::<4>(
            py,
            sims_instance,
            &objectives,
            timeout,
            pls_time_fraction,
            pls_initial_pop_size,
            max_pls_seed_size,
            moead_config,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        n => Err(PyValueError::new_err(format!(
            "solve_with_memetic_moead only supports 2, 3, or 4 objectives, got {n}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_memetic_moead_dim<const D: usize>(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: &[String],
    timeout: Duration,
    pls_time_fraction: f64,
    pls_initial_pop_size: usize,
    max_pls_seed_size: usize,
    moead_config: pls::evolutionary::moead::MoeadConfig,
    seed: u64,
    trace: bool,
    objective_bounds: &Option<Vec<Vec<u64>>>,
    include_dominated: bool,
) -> PyResult<SolvingResult> {
    use pls::evolutionary::memetic::{EaBackend, MemeticAlgorithm, MemeticConfig};
    use pls::problem_bitset::ProblemBitset;

    let raw_instance = build_raw_instance(sims_instance);
    let objective_definitions = parse_objective_definitions::<D>(objectives)?;
    let mut pls_problem =
        ProblemBitset::<D>::from_raw_with_objectives(&raw_instance, objective_definitions);
    apply_objective_bounds(&mut pls_problem, objective_bounds)?;

    let config = MemeticConfig {
        pls_time_fraction,
        pls_initial_pop_size,
        max_pls_seed_size,
        ea_backend: EaBackend::Moead,
        moead_config,
        seed,
        ..MemeticConfig::default()
    };

    let (archive_solutions, explored_fingerprints) = py.detach(|| {
        let result = MemeticAlgorithm::run::<ProblemBitset<D>, D>(&pls_problem, &config, timeout);
        let fingerprints: Vec<SolutionFingerprint<D>> = result
            .explored_solutions
            .solutions
            .values()
            .cloned()
            .collect();
        (result.archive, fingerprints)
    });

    info!(
        "Memetic MOEA/D completed: {} archive solutions, {} explored solutions",
        archive_solutions.len(),
        explored_fingerprints.len()
    );

    let python_final_solutions: Vec<_> = archive_solutions
        .iter()
        .map(|s| {
            let sol: crate::solution::Solution = (s, &pls_problem).into();
            sol
        })
        .collect();

    build_solving_result(
        python_final_solutions,
        explored_fingerprints,
        objectives,
        objective_bounds,
        timeout,
        format!("Memetic-MOEAD-{D}D"),
        trace,
        include_dominated,
    )
}

/// Solve the SIMS problem using a literal NSGA-II baseline (Deb et al., 2002).
///
/// Unlike `solve_with_nsga2`, this uses plain crowding distance (no D>=3
/// fallback), no stagnation injection, no `ensure_mutation` guard, and a
/// single crossover + single mutation operator -- nothing beyond what the
/// original paper specifies. See `pls::evolutionary::nsga2_baseline` for the
/// exact correspondence to the paper's boxed procedures.
#[expect(
    clippy::too_many_arguments,
    reason = "It's okay for Python API to have many parameters"
)]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string(), "min_max_incidence_angle".to_string(), "min_resolution".to_string()],
    timeout=Duration::from_secs(120),
    population_size=100usize,
    max_generations=500000usize,
    seed=42u64,
    trace=true,
    objective_bounds=None,
    include_dominated=false,
    crossover_rate=0.9,
    mutation_rate=0.01
))]
pub fn solve_with_nsga2_baseline(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: Vec<String>,
    timeout: Duration,
    population_size: usize,
    max_generations: usize,
    seed: u64,
    trace: bool,
    objective_bounds: Option<Vec<Vec<u64>>>,
    include_dominated: bool,
    crossover_rate: f64,
    mutation_rate: f64,
) -> PyResult<SolvingResult> {
    let config = pls::evolutionary::nsga2_baseline::Nsga2BaselineConfig {
        population_size,
        crossover_rate,
        mutation_rate,
        seed,
    };

    match objectives.len() {
        2 => run_nsga2_baseline_dim::<2>(
            py,
            sims_instance,
            &objectives,
            timeout,
            config,
            max_generations,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        3 => run_nsga2_baseline_dim::<3>(
            py,
            sims_instance,
            &objectives,
            timeout,
            config,
            max_generations,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        4 => run_nsga2_baseline_dim::<4>(
            py,
            sims_instance,
            &objectives,
            timeout,
            config,
            max_generations,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        n => Err(PyValueError::new_err(format!(
            "solve_with_nsga2_baseline only supports 2, 3, or 4 objectives, got {n}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_nsga2_baseline_dim<const D: usize>(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: &[String],
    timeout: Duration,
    config: pls::evolutionary::nsga2_baseline::Nsga2BaselineConfig,
    max_generations: usize,
    trace: bool,
    objective_bounds: &Option<Vec<Vec<u64>>>,
    include_dominated: bool,
) -> PyResult<SolvingResult> {
    use pls::evolutionary::nsga2_baseline::run_nsga2_baseline;
    use pls::problem_bitset::ProblemBitset;

    let raw_instance = build_raw_instance(sims_instance);
    let objective_definitions = parse_objective_definitions::<D>(objectives)?;
    let mut pls_problem =
        ProblemBitset::<D>::from_raw_with_objectives(&raw_instance, objective_definitions);
    apply_objective_bounds(&mut pls_problem, objective_bounds)?;

    let (archive_solutions, explored) = py.detach(|| {
        run_nsga2_baseline::<ProblemBitset<D>, D>(&pls_problem, config, max_generations, timeout)
    });
    let explored_fingerprints: Vec<SolutionFingerprint<D>> =
        explored.solutions.values().cloned().collect();

    info!(
        "NSGA-II baseline completed: {} archive solutions, {} explored solutions",
        archive_solutions.len(),
        explored_fingerprints.len()
    );

    let python_final_solutions: Vec<_> = archive_solutions
        .iter()
        .map(|s| {
            let sol: crate::solution::Solution = (s, &pls_problem).into();
            sol
        })
        .collect();

    build_solving_result(
        python_final_solutions,
        explored_fingerprints,
        objectives,
        objective_bounds,
        timeout,
        format!("NSGA2-Baseline-{D}D"),
        trace,
        include_dominated,
    )
}

/// Solve the SIMS problem using a literal NSGA-III baseline (Deb & Jain, 2014).
///
/// Unlike `solve_with_nsga3`, parents are picked uniformly at random (the
/// paper explicitly does not use tournament selection), divisions are not
/// auto-sized, and there is no stagnation injection / `ensure_mutation` /
/// composite mutation suite. See `pls::evolutionary::nsga3_baseline`.
#[expect(
    clippy::too_many_arguments,
    reason = "It's okay for Python API to have many parameters"
)]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string(), "min_max_incidence_angle".to_string(), "min_resolution".to_string()],
    timeout=Duration::from_secs(120),
    num_divisions=12usize,
    max_generations=500000usize,
    seed=42u64,
    trace=true,
    objective_bounds=None,
    include_dominated=false,
    crossover_rate=0.9,
    mutation_rate=0.01
))]
pub fn solve_with_nsga3_baseline(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: Vec<String>,
    timeout: Duration,
    num_divisions: usize,
    max_generations: usize,
    seed: u64,
    trace: bool,
    objective_bounds: Option<Vec<Vec<u64>>>,
    include_dominated: bool,
    crossover_rate: f64,
    mutation_rate: f64,
) -> PyResult<SolvingResult> {
    let config = pls::evolutionary::nsga3_baseline::Nsga3BaselineConfig {
        num_divisions,
        crossover_rate,
        mutation_rate,
        seed,
    };

    match objectives.len() {
        2 => run_nsga3_baseline_dim::<2>(
            py,
            sims_instance,
            &objectives,
            timeout,
            config,
            max_generations,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        3 => run_nsga3_baseline_dim::<3>(
            py,
            sims_instance,
            &objectives,
            timeout,
            config,
            max_generations,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        4 => run_nsga3_baseline_dim::<4>(
            py,
            sims_instance,
            &objectives,
            timeout,
            config,
            max_generations,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        n => Err(PyValueError::new_err(format!(
            "solve_with_nsga3_baseline only supports 2, 3, or 4 objectives, got {n}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_nsga3_baseline_dim<const D: usize>(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: &[String],
    timeout: Duration,
    config: pls::evolutionary::nsga3_baseline::Nsga3BaselineConfig,
    max_generations: usize,
    trace: bool,
    objective_bounds: &Option<Vec<Vec<u64>>>,
    include_dominated: bool,
) -> PyResult<SolvingResult> {
    use pls::evolutionary::nsga3_baseline::run_nsga3_baseline;
    use pls::problem_bitset::ProblemBitset;

    let raw_instance = build_raw_instance(sims_instance);
    let objective_definitions = parse_objective_definitions::<D>(objectives)?;
    let mut pls_problem =
        ProblemBitset::<D>::from_raw_with_objectives(&raw_instance, objective_definitions);
    apply_objective_bounds(&mut pls_problem, objective_bounds)?;

    let (archive_solutions, explored) = py.detach(|| {
        run_nsga3_baseline::<ProblemBitset<D>, D>(&pls_problem, config, max_generations, timeout)
    });
    let explored_fingerprints: Vec<SolutionFingerprint<D>> =
        explored.solutions.values().cloned().collect();

    info!(
        "NSGA-III baseline completed: {} archive solutions, {} explored solutions",
        archive_solutions.len(),
        explored_fingerprints.len()
    );

    let python_final_solutions: Vec<_> = archive_solutions
        .iter()
        .map(|s| {
            let sol: crate::solution::Solution = (s, &pls_problem).into();
            sol
        })
        .collect();

    build_solving_result(
        python_final_solutions,
        explored_fingerprints,
        objectives,
        objective_bounds,
        timeout,
        format!("NSGA3-Baseline-{D}D"),
        trace,
        include_dominated,
    )
}

/// Solve the SIMS problem using a literal MOEA/D baseline (Zhang & Li, 2007).
///
/// Unlike `solve_with_moead`, mating is restricted to the neighbourhood
/// only (no `delta` whole-population option), every improving neighbour is
/// replaced (no `max_replacements` cap), and only Tchebycheff decomposition
/// is used. Those extras are from Li & Zhang (2009), not the original
/// paper. See `pls::evolutionary::moead_baseline`.
#[expect(
    clippy::too_many_arguments,
    reason = "It's okay for Python API to have many parameters"
)]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string(), "min_max_incidence_angle".to_string(), "min_resolution".to_string()],
    timeout=Duration::from_secs(120),
    num_divisions=99usize,
    neighbourhood_size=10usize,
    max_generations=500000usize,
    seed=42u64,
    trace=true,
    objective_bounds=None,
    include_dominated=false,
    crossover_rate=1.0,
    mutation_rate=0.01
))]
pub fn solve_with_moead_baseline(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: Vec<String>,
    timeout: Duration,
    num_divisions: usize,
    neighbourhood_size: usize,
    max_generations: usize,
    seed: u64,
    trace: bool,
    objective_bounds: Option<Vec<Vec<u64>>>,
    include_dominated: bool,
    crossover_rate: f64,
    mutation_rate: f64,
) -> PyResult<SolvingResult> {
    let config = pls::evolutionary::moead_baseline::MoeadBaselineConfig {
        num_divisions,
        neighbourhood_size,
        crossover_rate,
        mutation_rate,
        seed,
    };

    match objectives.len() {
        2 => run_moead_baseline_dim::<2>(
            py,
            sims_instance,
            &objectives,
            timeout,
            config,
            max_generations,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        3 => run_moead_baseline_dim::<3>(
            py,
            sims_instance,
            &objectives,
            timeout,
            config,
            max_generations,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        4 => run_moead_baseline_dim::<4>(
            py,
            sims_instance,
            &objectives,
            timeout,
            config,
            max_generations,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        n => Err(PyValueError::new_err(format!(
            "solve_with_moead_baseline only supports 2, 3, or 4 objectives, got {n}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_moead_baseline_dim<const D: usize>(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: &[String],
    timeout: Duration,
    config: pls::evolutionary::moead_baseline::MoeadBaselineConfig,
    max_generations: usize,
    trace: bool,
    objective_bounds: &Option<Vec<Vec<u64>>>,
    include_dominated: bool,
) -> PyResult<SolvingResult> {
    use pls::evolutionary::moead_baseline::run_moead_baseline;
    use pls::problem_bitset::ProblemBitset;

    let raw_instance = build_raw_instance(sims_instance);
    let objective_definitions = parse_objective_definitions::<D>(objectives)?;
    let mut pls_problem =
        ProblemBitset::<D>::from_raw_with_objectives(&raw_instance, objective_definitions);
    apply_objective_bounds(&mut pls_problem, objective_bounds)?;

    let (archive_solutions, explored) = py.detach(|| {
        run_moead_baseline::<ProblemBitset<D>, D>(&pls_problem, config, max_generations, timeout)
    });
    let explored_fingerprints: Vec<SolutionFingerprint<D>> =
        explored.solutions.values().cloned().collect();

    info!(
        "MOEA/D baseline completed: {} archive solutions, {} explored solutions",
        archive_solutions.len(),
        explored_fingerprints.len()
    );

    let python_final_solutions: Vec<_> = archive_solutions
        .iter()
        .map(|s| {
            let sol: crate::solution::Solution = (s, &pls_problem).into();
            sol
        })
        .collect();

    build_solving_result(
        python_final_solutions,
        explored_fingerprints,
        objectives,
        objective_bounds,
        timeout,
        format!("MOEAD-Baseline-{D}D"),
        trace,
        include_dominated,
    )
}

// ---------------------------------------------------------------------------
// Pseudosolver-seeded baseline EAs
// ---------------------------------------------------------------------------
//
// These are hybrids that use GPBA-A / pseudosolver solutions as the initial
// population for the three literal-paper baseline EAs, with the full timeout
// given to the EA. The pattern mirrors `Hybrid 50:50` (pseudosolver → PLS) but
// replaces PLS with a literal-paper baseline EA.

/// Helper: convert Python Solution list → Vec<BitsetEncodedSolution<P, D>>.
fn py_solutions_to_bitset<P, const D: usize>(
    solutions: &[crate::solution::Solution],
    problem: &P,
) -> Vec<pls::solution_impl::bitset_encoded_solution::BitsetEncodedSolution<P, D>>
where
    P: pls::problem::SetCoverProblem<D> + Clone + Send + Sync,
{
    solutions
        .iter()
        .map(|s| {
            let selected: Vec<usize> = s.selected_images.iter().cloned().collect();
            pls::solution_impl::bitset_encoded_solution::BitsetEncodedSolution::from_selected_images(
                &selected, problem,
            )
        })
        .collect()
}

/// Solve using a pseudosolver-seeded literal NSGA-II baseline.
///
/// The full timeout is given to the EA; no PLS warm-start phase is run.
#[expect(
    clippy::too_many_arguments,
    reason = "It's okay for Python API to have many parameters"
)]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string(), "min_max_incidence_angle".to_string(), "min_resolution".to_string()],
    timeout=Duration::from_secs(120),
    initial_population=None,
    population_size=100usize,
    max_generations=500000usize,
    seed=42u64,
    trace=true,
    objective_bounds=None,
    include_dominated=false,
    crossover_rate=0.9,
    mutation_rate=0.01
))]
pub fn solve_with_pseudo_seeded_nsga2_baseline(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: Vec<String>,
    timeout: Duration,
    initial_population: Option<Vec<crate::solution::Solution>>,
    population_size: usize,
    max_generations: usize,
    seed: u64,
    trace: bool,
    objective_bounds: Option<Vec<Vec<u64>>>,
    include_dominated: bool,
    crossover_rate: f64,
    mutation_rate: f64,
) -> PyResult<SolvingResult> {
    let config = pls::evolutionary::nsga2_baseline::Nsga2BaselineConfig {
        population_size,
        crossover_rate,
        mutation_rate,
        seed,
    };

    match objectives.len() {
        2 => run_pseudo_seeded_nsga2_baseline_dim::<2>(
            py,
            sims_instance,
            &objectives,
            timeout,
            initial_population,
            config,
            max_generations,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        3 => run_pseudo_seeded_nsga2_baseline_dim::<3>(
            py,
            sims_instance,
            &objectives,
            timeout,
            initial_population,
            config,
            max_generations,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        4 => run_pseudo_seeded_nsga2_baseline_dim::<4>(
            py,
            sims_instance,
            &objectives,
            timeout,
            initial_population,
            config,
            max_generations,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        n => Err(PyValueError::new_err(format!(
            "solve_with_pseudo_seeded_nsga2_baseline only supports 2, 3, or 4 objectives, got {n}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_pseudo_seeded_nsga2_baseline_dim<const D: usize>(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: &[String],
    timeout: Duration,
    initial_population: Option<Vec<crate::solution::Solution>>,
    config: pls::evolutionary::nsga2_baseline::Nsga2BaselineConfig,
    max_generations: usize,
    trace: bool,
    objective_bounds: &Option<Vec<Vec<u64>>>,
    include_dominated: bool,
) -> PyResult<SolvingResult> {
    use pls::evolutionary::nsga2_baseline::run_nsga2_baseline_seeded;
    use pls::problem_bitset::ProblemBitset;

    let raw_instance = build_raw_instance(sims_instance);
    let objective_definitions = parse_objective_definitions::<D>(objectives)?;
    let mut pls_problem =
        ProblemBitset::<D>::from_raw_with_objectives(&raw_instance, objective_definitions);
    apply_objective_bounds(&mut pls_problem, objective_bounds)?;

    let seed_pop = initial_population
        .as_deref()
        .map(|p| py_solutions_to_bitset::<ProblemBitset<D>, D>(p, &pls_problem))
        .unwrap_or_default();

    info!(
        "Pseudo-seeded NSGA-II baseline {D}D: {} seed solutions, pop_size={}",
        seed_pop.len(),
        config.population_size,
    );

    let (archive_solutions, explored) = py.detach(|| {
        run_nsga2_baseline_seeded::<ProblemBitset<D>, D>(
            &pls_problem,
            config,
            seed_pop,
            max_generations,
            timeout,
        )
    });
    let explored_fingerprints: Vec<SolutionFingerprint<D>> =
        explored.solutions.values().cloned().collect();

    let python_final_solutions: Vec<_> = archive_solutions
        .iter()
        .map(|s| {
            let sol: crate::solution::Solution = (s, &pls_problem).into();
            sol
        })
        .collect();

    build_solving_result(
        python_final_solutions,
        explored_fingerprints,
        objectives,
        objective_bounds,
        timeout,
        format!("PseudoSeeded-NSGA2-Baseline-{D}D"),
        trace,
        include_dominated,
    )
}

/// Solve using a pseudosolver-seeded literal NSGA-III baseline.
///
/// The full timeout is given to the EA; no PLS warm-start phase is run.
#[expect(
    clippy::too_many_arguments,
    reason = "It's okay for Python API to have many parameters"
)]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string(), "min_max_incidence_angle".to_string(), "min_resolution".to_string()],
    timeout=Duration::from_secs(120),
    initial_population=None,
    num_divisions=12usize,
    max_generations=500000usize,
    seed=42u64,
    trace=true,
    objective_bounds=None,
    include_dominated=false,
    crossover_rate=0.9,
    mutation_rate=0.01
))]
pub fn solve_with_pseudo_seeded_nsga3_baseline(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: Vec<String>,
    timeout: Duration,
    initial_population: Option<Vec<crate::solution::Solution>>,
    num_divisions: usize,
    max_generations: usize,
    seed: u64,
    trace: bool,
    objective_bounds: Option<Vec<Vec<u64>>>,
    include_dominated: bool,
    crossover_rate: f64,
    mutation_rate: f64,
) -> PyResult<SolvingResult> {
    let config = pls::evolutionary::nsga3_baseline::Nsga3BaselineConfig {
        num_divisions,
        crossover_rate,
        mutation_rate,
        seed,
    };

    match objectives.len() {
        2 => run_pseudo_seeded_nsga3_baseline_dim::<2>(
            py,
            sims_instance,
            &objectives,
            timeout,
            initial_population,
            config,
            max_generations,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        3 => run_pseudo_seeded_nsga3_baseline_dim::<3>(
            py,
            sims_instance,
            &objectives,
            timeout,
            initial_population,
            config,
            max_generations,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        4 => run_pseudo_seeded_nsga3_baseline_dim::<4>(
            py,
            sims_instance,
            &objectives,
            timeout,
            initial_population,
            config,
            max_generations,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        n => Err(PyValueError::new_err(format!(
            "solve_with_pseudo_seeded_nsga3_baseline only supports 2, 3, or 4 objectives, got {n}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_pseudo_seeded_nsga3_baseline_dim<const D: usize>(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: &[String],
    timeout: Duration,
    initial_population: Option<Vec<crate::solution::Solution>>,
    config: pls::evolutionary::nsga3_baseline::Nsga3BaselineConfig,
    max_generations: usize,
    trace: bool,
    objective_bounds: &Option<Vec<Vec<u64>>>,
    include_dominated: bool,
) -> PyResult<SolvingResult> {
    use pls::evolutionary::nsga3_baseline::run_nsga3_baseline_seeded;
    use pls::problem_bitset::ProblemBitset;

    let raw_instance = build_raw_instance(sims_instance);
    let objective_definitions = parse_objective_definitions::<D>(objectives)?;
    let mut pls_problem =
        ProblemBitset::<D>::from_raw_with_objectives(&raw_instance, objective_definitions);
    apply_objective_bounds(&mut pls_problem, objective_bounds)?;

    let seed_pop = initial_population
        .as_deref()
        .map(|p| py_solutions_to_bitset::<ProblemBitset<D>, D>(p, &pls_problem))
        .unwrap_or_default();

    info!(
        "Pseudo-seeded NSGA-III baseline {D}D: {} seed solutions, num_divisions={}",
        seed_pop.len(),
        config.num_divisions,
    );

    let (archive_solutions, explored) = py.detach(|| {
        run_nsga3_baseline_seeded::<ProblemBitset<D>, D>(
            &pls_problem,
            config,
            seed_pop,
            max_generations,
            timeout,
        )
    });
    let explored_fingerprints: Vec<SolutionFingerprint<D>> =
        explored.solutions.values().cloned().collect();

    let python_final_solutions: Vec<_> = archive_solutions
        .iter()
        .map(|s| {
            let sol: crate::solution::Solution = (s, &pls_problem).into();
            sol
        })
        .collect();

    build_solving_result(
        python_final_solutions,
        explored_fingerprints,
        objectives,
        objective_bounds,
        timeout,
        format!("PseudoSeeded-NSGA3-Baseline-{D}D"),
        trace,
        include_dominated,
    )
}

/// Solve using a pseudosolver-seeded literal MOEA/D baseline.
///
/// The full timeout is given to the EA; no PLS warm-start phase is run.
#[expect(
    clippy::too_many_arguments,
    reason = "It's okay for Python API to have many parameters"
)]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string(), "min_max_incidence_angle".to_string(), "min_resolution".to_string()],
    timeout=Duration::from_secs(120),
    initial_population=None,
    num_divisions=12usize,
    neighbourhood_size=5usize,
    max_generations=500000usize,
    seed=42u64,
    trace=true,
    objective_bounds=None,
    include_dominated=false,
    crossover_rate=1.0,
    mutation_rate=0.01
))]
pub fn solve_with_pseudo_seeded_moead_baseline(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: Vec<String>,
    timeout: Duration,
    initial_population: Option<Vec<crate::solution::Solution>>,
    num_divisions: usize,
    neighbourhood_size: usize,
    max_generations: usize,
    seed: u64,
    trace: bool,
    objective_bounds: Option<Vec<Vec<u64>>>,
    include_dominated: bool,
    crossover_rate: f64,
    mutation_rate: f64,
) -> PyResult<SolvingResult> {
    let config = pls::evolutionary::moead_baseline::MoeadBaselineConfig {
        num_divisions,
        neighbourhood_size,
        crossover_rate,
        mutation_rate,
        seed,
    };

    match objectives.len() {
        2 => run_pseudo_seeded_moead_baseline_dim::<2>(
            py,
            sims_instance,
            &objectives,
            timeout,
            initial_population,
            config,
            max_generations,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        3 => run_pseudo_seeded_moead_baseline_dim::<3>(
            py,
            sims_instance,
            &objectives,
            timeout,
            initial_population,
            config,
            max_generations,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        4 => run_pseudo_seeded_moead_baseline_dim::<4>(
            py,
            sims_instance,
            &objectives,
            timeout,
            initial_population,
            config,
            max_generations,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        n => Err(PyValueError::new_err(format!(
            "solve_with_pseudo_seeded_moead_baseline only supports 2, 3, or 4 objectives, got {n}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_pseudo_seeded_moead_baseline_dim<const D: usize>(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: &[String],
    timeout: Duration,
    initial_population: Option<Vec<crate::solution::Solution>>,
    config: pls::evolutionary::moead_baseline::MoeadBaselineConfig,
    max_generations: usize,
    trace: bool,
    objective_bounds: &Option<Vec<Vec<u64>>>,
    include_dominated: bool,
) -> PyResult<SolvingResult> {
    use pls::evolutionary::moead_baseline::run_moead_baseline_seeded;
    use pls::problem_bitset::ProblemBitset;

    let raw_instance = build_raw_instance(sims_instance);
    let objective_definitions = parse_objective_definitions::<D>(objectives)?;
    let mut pls_problem =
        ProblemBitset::<D>::from_raw_with_objectives(&raw_instance, objective_definitions);
    apply_objective_bounds(&mut pls_problem, objective_bounds)?;

    let seed_pop = initial_population
        .as_deref()
        .map(|p| py_solutions_to_bitset::<ProblemBitset<D>, D>(p, &pls_problem))
        .unwrap_or_default();

    info!(
        "Pseudo-seeded MOEA/D baseline {D}D: {} seed solutions, num_divisions={}",
        seed_pop.len(),
        config.num_divisions,
    );

    let (archive_solutions, explored) = py.detach(|| {
        run_moead_baseline_seeded::<ProblemBitset<D>, D>(
            &pls_problem,
            config,
            seed_pop,
            max_generations,
            timeout,
        )
    });
    let explored_fingerprints: Vec<SolutionFingerprint<D>> =
        explored.solutions.values().cloned().collect();

    let python_final_solutions: Vec<_> = archive_solutions
        .iter()
        .map(|s| {
            let sol: crate::solution::Solution = (s, &pls_problem).into();
            sol
        })
        .collect();

    build_solving_result(
        python_final_solutions,
        explored_fingerprints,
        objectives,
        objective_bounds,
        timeout,
        format!("PseudoSeeded-MOEAD-Baseline-{D}D"),
        trace,
        include_dominated,
    )
}

// ---------------------------------------------------------------------------
// Pseudosolver-seeded SIMS-tailored EAs (improved-tier hybrids).
//
// These mirror the pseudo-seeded *baseline* bindings above, but route the seed
// population into the SIMS-tailored `nsga2`/`nsga3`/`moead` algorithms (composite
// mutation, stagnation injection, delta/nr mating, PBI, etc.) instead of the
// literal-paper baselines. They reuse the same `run_*_tailored_dim` helpers as
// the standalone tailored bindings, just with a non-empty initial population.
// ---------------------------------------------------------------------------

/// Pseudosolver-seeded SIMS-tailored NSGA-II. Full timeout to the EA.
#[expect(
    clippy::too_many_arguments,
    reason = "It's okay for Python API to have many parameters"
)]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string(), "min_max_incidence_angle".to_string(), "min_resolution".to_string()],
    timeout=Duration::from_secs(120),
    initial_population=None,
    population_size=100usize,
    max_generations=500000usize,
    seed=42u64,
    trace=true,
    objective_bounds=None,
    include_dominated=false,
    crossover_rate=0.9,
    swap_mutation_rate=0.4,
    add_prune_mutation_rate=0.3,
    bitflip_mutation_rate=0.0,
    multi_swap_max_removals=3usize,
    multi_swap_rate=0.2,
    shift_mutation_rate=0.25,
    coverage_biased_crossover_fraction=0.5,
    ensure_mutation=true,
    stagnation_limit=50usize,
    use_contribution_distance=true
))]
pub fn solve_with_pseudo_seeded_nsga2(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: Vec<String>,
    timeout: Duration,
    initial_population: Option<Vec<crate::solution::Solution>>,
    population_size: usize,
    max_generations: usize,
    seed: u64,
    trace: bool,
    objective_bounds: Option<Vec<Vec<u64>>>,
    include_dominated: bool,
    crossover_rate: f64,
    swap_mutation_rate: f64,
    add_prune_mutation_rate: f64,
    bitflip_mutation_rate: f64,
    multi_swap_max_removals: usize,
    multi_swap_rate: f64,
    shift_mutation_rate: f64,
    coverage_biased_crossover_fraction: f64,
    ensure_mutation: bool,
    stagnation_limit: usize,
    use_contribution_distance: bool,
) -> PyResult<SolvingResult> {
    let config = pls::evolutionary::nsga2::Nsga2Config {
        population_size,
        crossover_rate,
        swap_mutation_rate,
        add_prune_mutation_rate,
        bitflip_mutation_rate,
        multi_swap_max_removals,
        multi_swap_rate,
        shift_mutation_rate,
        coverage_biased_crossover_fraction,
        ensure_mutation,
        stagnation_limit,
        use_contribution_distance,
    };

    match objectives.len() {
        2 => run_nsga2_tailored_dim::<2>(
            py,
            sims_instance,
            &objectives,
            timeout,
            initial_population,
            config,
            max_generations,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        3 => run_nsga2_tailored_dim::<3>(
            py,
            sims_instance,
            &objectives,
            timeout,
            initial_population,
            config,
            max_generations,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        4 => run_nsga2_tailored_dim::<4>(
            py,
            sims_instance,
            &objectives,
            timeout,
            initial_population,
            config,
            max_generations,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        n => Err(PyValueError::new_err(format!(
            "solve_with_pseudo_seeded_nsga2 only supports 2, 3, or 4 objectives, got {n}"
        ))),
    }
}

/// Pseudosolver-seeded SIMS-tailored NSGA-III. Full timeout to the EA.
#[expect(
    clippy::too_many_arguments,
    reason = "It's okay for Python API to have many parameters"
)]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string(), "min_max_incidence_angle".to_string(), "min_resolution".to_string()],
    timeout=Duration::from_secs(120),
    initial_population=None,
    target_pop_size=200usize,
    num_divisions=12usize,
    auto_divisions=true,
    max_generations=500000usize,
    seed=42u64,
    trace=true,
    objective_bounds=None,
    include_dominated=false,
    crossover_rate=0.9,
    swap_mutation_rate=0.4,
    add_prune_mutation_rate=0.3,
    bitflip_mutation_rate=0.0,
    multi_swap_max_removals=3usize,
    multi_swap_rate=0.2,
    shift_mutation_rate=0.25,
    coverage_biased_crossover_fraction=0.5,
    ensure_mutation=true,
    stagnation_limit=50usize
))]
pub fn solve_with_pseudo_seeded_nsga3(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: Vec<String>,
    timeout: Duration,
    initial_population: Option<Vec<crate::solution::Solution>>,
    target_pop_size: usize,
    num_divisions: usize,
    auto_divisions: bool,
    max_generations: usize,
    seed: u64,
    trace: bool,
    objective_bounds: Option<Vec<Vec<u64>>>,
    include_dominated: bool,
    crossover_rate: f64,
    swap_mutation_rate: f64,
    add_prune_mutation_rate: f64,
    bitflip_mutation_rate: f64,
    multi_swap_max_removals: usize,
    multi_swap_rate: f64,
    shift_mutation_rate: f64,
    coverage_biased_crossover_fraction: f64,
    ensure_mutation: bool,
    stagnation_limit: usize,
) -> PyResult<SolvingResult> {
    let config = pls::evolutionary::nsga3::Nsga3Config {
        num_divisions,
        target_pop_size,
        auto_divisions,
        crossover_rate,
        swap_mutation_rate,
        add_prune_mutation_rate,
        bitflip_mutation_rate,
        multi_swap_max_removals,
        multi_swap_rate,
        shift_mutation_rate,
        coverage_biased_crossover_fraction,
        ensure_mutation,
        stagnation_limit,
    };

    match objectives.len() {
        2 => run_nsga3_tailored_dim::<2>(
            py,
            sims_instance,
            &objectives,
            timeout,
            initial_population,
            config,
            max_generations,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        3 => run_nsga3_tailored_dim::<3>(
            py,
            sims_instance,
            &objectives,
            timeout,
            initial_population,
            config,
            max_generations,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        4 => run_nsga3_tailored_dim::<4>(
            py,
            sims_instance,
            &objectives,
            timeout,
            initial_population,
            config,
            max_generations,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        n => Err(PyValueError::new_err(format!(
            "solve_with_pseudo_seeded_nsga3 only supports 2, 3, or 4 objectives, got {n}"
        ))),
    }
}

/// Pseudosolver-seeded SIMS-tailored MOEA/D. Full timeout to the EA.
#[expect(
    clippy::too_many_arguments,
    reason = "It's okay for Python API to have many parameters"
)]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string(), "min_max_incidence_angle".to_string(), "min_resolution".to_string()],
    timeout=Duration::from_secs(120),
    initial_population=None,
    population_size=200usize,
    max_generations=500000usize,
    seed=42u64,
    trace=true,
    objective_bounds=None,
    include_dominated=false,
    num_divisions=99usize,
    neighbourhood_size=20usize,
    delta=0.9,
    max_replacements=3usize,
    crossover_rate=1.0,
    swap_mutation_rate=0.3,
    add_prune_mutation_rate=0.2,
    multi_swap_max_removals=3usize,
    multi_swap_rate=0.15,
    shift_mutation_rate=0.15,
    coverage_biased_crossover_fraction=0.5,
    ensure_mutation=false,
    auto_divisions=true,
    use_pbi=false,
    pbi_theta=5.0,
    stagnation_limit=80usize
))]
pub fn solve_with_pseudo_seeded_moead(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: Vec<String>,
    timeout: Duration,
    initial_population: Option<Vec<crate::solution::Solution>>,
    population_size: usize,
    max_generations: usize,
    seed: u64,
    trace: bool,
    objective_bounds: Option<Vec<Vec<u64>>>,
    include_dominated: bool,
    num_divisions: usize,
    neighbourhood_size: usize,
    delta: f64,
    max_replacements: usize,
    crossover_rate: f64,
    swap_mutation_rate: f64,
    add_prune_mutation_rate: f64,
    multi_swap_max_removals: usize,
    multi_swap_rate: f64,
    shift_mutation_rate: f64,
    coverage_biased_crossover_fraction: f64,
    ensure_mutation: bool,
    auto_divisions: bool,
    use_pbi: bool,
    pbi_theta: f64,
    stagnation_limit: usize,
) -> PyResult<SolvingResult> {
    let config = pls::evolutionary::moead::MoeadConfig {
        population_size,
        num_divisions,
        target_pop_size: population_size,
        auto_divisions,
        neighbourhood_size,
        delta,
        max_replacements,
        crossover_rate,
        swap_mutation_rate,
        add_prune_mutation_rate,
        multi_swap_max_removals,
        multi_swap_rate,
        shift_mutation_rate,
        coverage_biased_crossover_fraction,
        ensure_mutation,
        use_pbi,
        pbi_theta,
        stagnation_limit,
    };

    match objectives.len() {
        2 => run_moead_tailored_dim::<2>(
            py,
            sims_instance,
            &objectives,
            timeout,
            initial_population,
            config,
            max_generations,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        3 => run_moead_tailored_dim::<3>(
            py,
            sims_instance,
            &objectives,
            timeout,
            initial_population,
            config,
            max_generations,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        4 => run_moead_tailored_dim::<4>(
            py,
            sims_instance,
            &objectives,
            timeout,
            initial_population,
            config,
            max_generations,
            seed,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        n => Err(PyValueError::new_err(format!(
            "solve_with_pseudo_seeded_moead only supports 2, 3, or 4 objectives, got {n}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Third-party crate integrations (moors / optirustic), feature-gated
// ---------------------------------------------------------------------------
//
// These wrap independently-implemented NSGA-II (moors, optirustic) and
// NSGA-III (optirustic) algorithms, used as an external sanity check against
// our own literal-paper baselines and SIMS-tailored implementations. Neither
// adapter tracks intermediate solution-discovery times (they're black boxes
// from our side), so we register the final archive at the actual elapsed
// wall-time rather than fabricating intermediate timestamps -- the resulting
// HV-over-time curve is an honest step function (flat at 0, then jumps to
// the final value when the run completes), not a smooth approximation.

#[cfg(feature = "external_solvers")]
fn register_final_archive_at_elapsed<P, const D: usize>(
    archive: &[pls::solution_impl::bitset_encoded_solution::BitsetEncodedSolution<P, D>],
    elapsed: Duration,
    timeout: Duration,
) -> Vec<SolutionFingerprint<D>>
where
    P: pls::problem::SetCoverProblem<D> + Clone + Send + Sync,
{
    use pls::explored_solutions_data::ExploredSolutionsData;

    // `build_solving_result` sets the trace's total_duration to the nominal
    // requested `timeout`, not the actual measured `elapsed` time. moors and
    // optirustic only check their deadline between generations (not
    // preemptively), so `elapsed` can run slightly past `timeout`. If we
    // registered the raw `elapsed` timestamp, it would fall outside the
    // trace's [0, total_duration] sampling window and never appear in any
    // HV-curve bucket -- producing a flat-zero curve despite a non-empty
    // archive. Clamp to `timeout` so the final archive is always visible at
    // the last sample point.
    let clamped = elapsed.min(timeout);
    let mut explored = ExploredSolutionsData::<D>::new([u64::MAX; D]);
    for sol in archive {
        explored.register_without_selected_images(0, sol, clamped);
    }
    explored.solutions.values().cloned().collect()
}

/// Run NSGA-II from the `moors` crate (uniform binary crossover + bit-flip
/// mutation, native binary operators) as an external sanity check.
#[cfg(feature = "external_solvers")]
#[expect(
    clippy::too_many_arguments,
    reason = "It's okay for Python API to have many parameters"
)]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string(), "min_max_incidence_angle".to_string(), "min_resolution".to_string()],
    timeout=Duration::from_secs(120),
    population_size=100usize,
    num_offsprings=50usize,
    num_iterations=500_000usize,
    seed=42u64,
    trace=true,
    objective_bounds=None,
    include_dominated=false,
    crossover_rate=0.9,
    mutation_rate=0.1,
    bitflip_probability=0.05
))]
pub fn solve_with_nsga2_moors(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: Vec<String>,
    timeout: Duration,
    population_size: usize,
    num_offsprings: usize,
    num_iterations: usize,
    seed: u64,
    trace: bool,
    objective_bounds: Option<Vec<Vec<u64>>>,
    include_dominated: bool,
    crossover_rate: f64,
    mutation_rate: f64,
    bitflip_probability: f64,
) -> PyResult<SolvingResult> {
    let config = pls::evolutionary::moors_adapter::MoorsConfig {
        population_size,
        num_offsprings,
        num_iterations,
        crossover_rate,
        mutation_rate,
        bitflip_probability,
        crossover_type: pls::evolutionary::moors_adapter::MoorsCrossoverType::Uniform,
        seed,
    };

    match objectives.len() {
        2 => run_nsga2_moors_dim::<2>(
            py,
            sims_instance,
            &objectives,
            timeout,
            config,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        3 => run_nsga2_moors_dim::<3>(
            py,
            sims_instance,
            &objectives,
            timeout,
            config,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        4 => run_nsga2_moors_dim::<4>(
            py,
            sims_instance,
            &objectives,
            timeout,
            config,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        n => Err(PyValueError::new_err(format!(
            "solve_with_nsga2_moors only supports 2, 3, or 4 objectives, got {n}"
        ))),
    }
}

#[cfg(feature = "external_solvers")]
#[allow(clippy::too_many_arguments)]
fn run_nsga2_moors_dim<const D: usize>(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: &[String],
    timeout: Duration,
    config: pls::evolutionary::moors_adapter::MoorsConfig,
    trace: bool,
    objective_bounds: &Option<Vec<Vec<u64>>>,
    include_dominated: bool,
) -> PyResult<SolvingResult> {
    use pls::evolutionary::moors_adapter::run_moors_nsga2;
    use pls::problem_bitset::ProblemBitset;

    let raw_instance = build_raw_instance(sims_instance);
    let objective_definitions = parse_objective_definitions::<D>(objectives)?;
    let mut pls_problem =
        ProblemBitset::<D>::from_raw_with_objectives(&raw_instance, objective_definitions);
    apply_objective_bounds(&mut pls_problem, objective_bounds)?;

    let start = std::time::Instant::now();
    let (archive_solutions, _explored) =
        py.detach(|| run_moors_nsga2::<ProblemBitset<D>, D>(&pls_problem, config, timeout));
    let elapsed = start.elapsed();

    info!(
        "moors NSGA-II completed: {} archive solutions in {:?}",
        archive_solutions.len(),
        elapsed
    );

    let python_final_solutions: Vec<_> = archive_solutions
        .iter()
        .map(|s| {
            let sol: crate::solution::Solution = (s, &pls_problem).into();
            sol
        })
        .collect();

    let explored_fingerprints = register_final_archive_at_elapsed::<ProblemBitset<D>, D>(
        &archive_solutions,
        elapsed,
        timeout,
    );

    build_solving_result(
        python_final_solutions,
        explored_fingerprints,
        objectives,
        objective_bounds,
        timeout,
        format!("NSGA2-moors-{D}D"),
        trace,
        include_dominated,
    )
}

/// Run NSGA-II from the `optirustic` crate (continuous relaxation + SBX
/// crossover + polynomial mutation) as an external sanity check.
#[cfg(feature = "external_solvers")]
#[expect(
    clippy::too_many_arguments,
    reason = "It's okay for Python API to have many parameters"
)]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string(), "min_max_incidence_angle".to_string(), "min_resolution".to_string()],
    timeout=Duration::from_secs(120),
    population_size=100usize,
    max_generations=500_000usize,
    seed=42u64,
    trace=true,
    objective_bounds=None,
    include_dominated=false,
    crossover_probability=0.9,
    crossover_distribution_index=20.0,
    mutation_distribution_index=20.0
))]
pub fn solve_with_nsga2_optirustic(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: Vec<String>,
    timeout: Duration,
    population_size: usize,
    max_generations: usize,
    seed: u64,
    trace: bool,
    objective_bounds: Option<Vec<Vec<u64>>>,
    include_dominated: bool,
    crossover_probability: f64,
    crossover_distribution_index: f64,
    mutation_distribution_index: f64,
) -> PyResult<SolvingResult> {
    let config = pls::evolutionary::optirustic_adapter::OptirusticConfig {
        population_size,
        max_generations,
        crossover_distribution_index,
        crossover_probability,
        mutation_distribution_index,
        mutation_probability: None,
        seed: Some(seed),
        parallel: false,
    };

    match objectives.len() {
        2 => run_nsga2_optirustic_dim::<2>(
            py,
            sims_instance,
            &objectives,
            timeout,
            config,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        3 => run_nsga2_optirustic_dim::<3>(
            py,
            sims_instance,
            &objectives,
            timeout,
            config,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        4 => run_nsga2_optirustic_dim::<4>(
            py,
            sims_instance,
            &objectives,
            timeout,
            config,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        n => Err(PyValueError::new_err(format!(
            "solve_with_nsga2_optirustic only supports 2, 3, or 4 objectives, got {n}"
        ))),
    }
}

#[cfg(feature = "external_solvers")]
#[allow(clippy::too_many_arguments)]
fn run_nsga2_optirustic_dim<const D: usize>(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: &[String],
    timeout: Duration,
    config: pls::evolutionary::optirustic_adapter::OptirusticConfig,
    trace: bool,
    objective_bounds: &Option<Vec<Vec<u64>>>,
    include_dominated: bool,
) -> PyResult<SolvingResult> {
    use pls::evolutionary::optirustic_adapter::run_optirustic_nsga2;
    use pls::problem_bitset::ProblemBitset;

    let raw_instance = build_raw_instance(sims_instance);
    let objective_definitions = parse_objective_definitions::<D>(objectives)?;
    let mut pls_problem =
        ProblemBitset::<D>::from_raw_with_objectives(&raw_instance, objective_definitions);
    apply_objective_bounds(&mut pls_problem, objective_bounds)?;

    let start = std::time::Instant::now();
    let (archive_solutions, _explored) =
        py.detach(|| run_optirustic_nsga2::<ProblemBitset<D>, D>(&pls_problem, config, timeout));
    let elapsed = start.elapsed();

    info!(
        "optirustic NSGA-II completed: {} archive solutions in {:?}",
        archive_solutions.len(),
        elapsed
    );

    let python_final_solutions: Vec<_> = archive_solutions
        .iter()
        .map(|s| {
            let sol: crate::solution::Solution = (s, &pls_problem).into();
            sol
        })
        .collect();

    let explored_fingerprints = register_final_archive_at_elapsed::<ProblemBitset<D>, D>(
        &archive_solutions,
        elapsed,
        timeout,
    );

    build_solving_result(
        python_final_solutions,
        explored_fingerprints,
        objectives,
        objective_bounds,
        timeout,
        format!("NSGA2-optirustic-{D}D"),
        trace,
        include_dominated,
    )
}

/// Run NSGA-III from the `optirustic` crate (continuous relaxation + SBX
/// crossover + polynomial mutation + reference-point niching) as an
/// external sanity check.
#[cfg(feature = "external_solvers")]
#[expect(
    clippy::too_many_arguments,
    reason = "It's okay for Python API to have many parameters"
)]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string(), "min_max_incidence_angle".to_string(), "min_resolution".to_string()],
    timeout=Duration::from_secs(120),
    population_size=100usize,
    max_generations=500_000usize,
    seed=42u64,
    trace=true,
    objective_bounds=None,
    include_dominated=false,
    crossover_probability=0.9,
    crossover_distribution_index=20.0,
    mutation_distribution_index=20.0
))]
pub fn solve_with_nsga3_optirustic(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: Vec<String>,
    timeout: Duration,
    population_size: usize,
    max_generations: usize,
    seed: u64,
    trace: bool,
    objective_bounds: Option<Vec<Vec<u64>>>,
    include_dominated: bool,
    crossover_probability: f64,
    crossover_distribution_index: f64,
    mutation_distribution_index: f64,
) -> PyResult<SolvingResult> {
    let config = pls::evolutionary::optirustic_adapter::OptirusticConfig {
        population_size,
        max_generations,
        crossover_distribution_index,
        crossover_probability,
        mutation_distribution_index,
        mutation_probability: None,
        seed: Some(seed),
        parallel: false,
    };

    match objectives.len() {
        2 => run_nsga3_optirustic_dim::<2>(
            py,
            sims_instance,
            &objectives,
            timeout,
            config,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        3 => run_nsga3_optirustic_dim::<3>(
            py,
            sims_instance,
            &objectives,
            timeout,
            config,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        4 => run_nsga3_optirustic_dim::<4>(
            py,
            sims_instance,
            &objectives,
            timeout,
            config,
            trace,
            &objective_bounds,
            include_dominated,
        ),
        n => Err(PyValueError::new_err(format!(
            "solve_with_nsga3_optirustic only supports 2, 3, or 4 objectives, got {n}"
        ))),
    }
}

#[cfg(feature = "external_solvers")]
#[allow(clippy::too_many_arguments)]
fn run_nsga3_optirustic_dim<const D: usize>(
    py: Python<'_>,
    sims_instance: &SimsDiscreteProblem,
    objectives: &[String],
    timeout: Duration,
    config: pls::evolutionary::optirustic_adapter::OptirusticConfig,
    trace: bool,
    objective_bounds: &Option<Vec<Vec<u64>>>,
    include_dominated: bool,
) -> PyResult<SolvingResult> {
    use pls::evolutionary::optirustic_adapter::run_optirustic_nsga3;
    use pls::problem_bitset::ProblemBitset;

    let raw_instance = build_raw_instance(sims_instance);
    let objective_definitions = parse_objective_definitions::<D>(objectives)?;
    let mut pls_problem =
        ProblemBitset::<D>::from_raw_with_objectives(&raw_instance, objective_definitions);
    apply_objective_bounds(&mut pls_problem, objective_bounds)?;

    let start = std::time::Instant::now();
    let (archive_solutions, _explored) =
        py.detach(|| run_optirustic_nsga3::<ProblemBitset<D>, D>(&pls_problem, config, timeout));
    let elapsed = start.elapsed();

    info!(
        "optirustic NSGA-III completed: {} archive solutions in {:?}",
        archive_solutions.len(),
        elapsed
    );

    let python_final_solutions: Vec<_> = archive_solutions
        .iter()
        .map(|s| {
            let sol: crate::solution::Solution = (s, &pls_problem).into();
            sol
        })
        .collect();

    let explored_fingerprints = register_final_archive_at_elapsed::<ProblemBitset<D>, D>(
        &archive_solutions,
        elapsed,
        timeout,
    );

    build_solving_result(
        python_final_solutions,
        explored_fingerprints,
        objectives,
        objective_bounds,
        timeout,
        format!("NSGA3-optirustic-{D}D"),
        trace,
        include_dominated,
    )
}

fn build_raw_instance(sims_instance: &SimsDiscreteProblem) -> pls::problem::SIMSProblemInstanceRaw {
    pls::problem::SIMSProblemInstanceRaw {
        name: "python_instance".to_string(),
        num_images: sims_instance.num_images,
        universe_size: sims_instance.universe,
        images: sims_instance.images.clone(),
        costs: sims_instance.costs.iter().map(|&c| c as u64).collect(),
        clouds: sims_instance.clouds.clone(),
        areas: sims_instance.areas.iter().map(|&a| a as u64).collect(),
        max_cloud_area: sims_instance.max_cloud_area as u64,
        resolution: sims_instance.resolution.iter().map(|&r| r as u64).collect(),
        incidence_angle: sims_instance
            .incidence_angle
            .iter()
            .map(|&i| i as u64)
            .collect(),
    }
}

fn parse_objective_definitions<const D: usize>(
    objectives: &[String],
) -> PyResult<[pls::objectives::ObjectiveType; D]> {
    let mut defs = [pls::objectives::ObjectiveType::TotalCost; D];
    for (i, obj_name) in objectives.iter().enumerate() {
        defs[i] = match obj_name.as_str() {
            "min_cost" => pls::objectives::ObjectiveType::TotalCost,
            "cloud_coverage" => pls::objectives::ObjectiveType::CloudyArea,
            "min_resolution" => pls::objectives::ObjectiveType::MinResolution,
            "min_max_incidence_angle" => pls::objectives::ObjectiveType::MaxIncidenceAngle,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown objective: {obj_name}. Valid: min_cost, cloud_coverage, min_resolution, min_max_incidence_angle"
                )))
            }
        };
    }
    Ok(defs)
}

fn apply_objective_bounds<const D: usize>(
    pls_problem: &mut pls::problem_bitset::ProblemBitset<D>,
    objective_bounds: &Option<Vec<Vec<u64>>>,
) -> PyResult<()> {
    if let Some(bounds) = objective_bounds {
        let bounds_vec: Vec<[u64; 2]> = bounds
            .iter()
            .map(|b| {
                if b.len() != 2 {
                    return Err(PyValueError::new_err(format!(
                        "Each objective bound must have exactly 2 elements [min, max], got {}",
                        b.len()
                    )));
                }
                Ok([b[0], b[1]])
            })
            .collect::<PyResult<_>>()?;
        let bounds_array: [[u64; 2]; D] = bounds_vec.try_into().map_err(|_| {
            PyValueError::new_err(format!(
                "Expected exactly {D} objective bounds for {D}D problem"
            ))
        })?;
        pls_problem.set_objective_bounds(bounds_array);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_solving_result<const D: usize>(
    python_final_solutions: Vec<crate::solution::Solution>,
    explored_fingerprints: Vec<pls::explored_solutions_data::SolutionFingerprint<D>>,
    objectives: &[String],
    objective_bounds: &Option<Vec<Vec<u64>>>,
    timeout: Duration,
    algorithm_name: String,
    trace: bool,
    include_dominated: bool,
) -> PyResult<SolvingResult> {
    if !trace {
        return Ok(crate::solution::SolvingResult::new(python_final_solutions));
    }

    let dominance_info = if include_dominated {
        crate::trace::compute_dominance_info(explored_fingerprints, false)
    } else {
        crate::trace::compute_dominance_info(explored_fingerprints, true)
    };

    let trace_solutions = dominance_info.solutions;
    let domination_indices = Some(dominance_info.domination_indices);

    let (trace_objective_bounds, reference_point) = if let Some(provided_bounds) = objective_bounds
    {
        if provided_bounds.len() != objectives.len() {
            return Err(PyValueError::new_err(format!(
                "objective_bounds length ({}) does not match objectives length ({})",
                provided_bounds.len(),
                objectives.len()
            )));
        }
        let mut bounds_vec = Vec::new();
        let mut ref_point = Vec::new();
        for bound in provided_bounds {
            if bound.len() != 2 {
                return Err(PyValueError::new_err(format!(
                    "Each objective bound must have exactly 2 elements [min, max], got {}",
                    bound.len()
                )));
            }
            bounds_vec.push([bound[0], bound[1]]);
            ref_point.push(bound[1] + 1);
        }
        (bounds_vec, ref_point)
    } else {
        crate::trace::calculate_objective_bounds_from_solutions(&trace_solutions).map_err(|e| {
            PyValueError::new_err(format!("Failed to calculate objective bounds: {e}"))
        })?
    };

    let trace_archive = crate::trace::create_optimization_trace_archive(
        trace_solutions,
        objectives.to_vec(),
        timeout.as_micros() as u64,
        algorithm_name,
        trace_objective_bounds,
        reference_point,
        domination_indices,
    )
    .map_err(|e| PyValueError::new_err(format!("Failed to create trace archive: {e}")))?;

    Ok(crate::solution::SolvingResult::with_trace(
        python_final_solutions,
        trace_archive,
    ))
}

fn read_profiling_trace_data(
    guard: Option<tracing_chrome::FlushGuard>,
    buffer: Option<std::sync::Arc<std::sync::Mutex<Vec<u8>>>>,
) -> Option<Vec<u8>> {
    if let (Some(guard), Some(buf)) = (guard, buffer) {
        // Explicitly drop the guard to flush the trace data
        drop(guard);

        // Small delay to ensure all data is flushed
        std::thread::sleep(Duration::from_millis(50));

        // Extract data from the shared buffer
        match buf.lock() {
            Ok(data) => {
                let trace_data = data.clone();
                info!(
                    "Successfully captured {} bytes of profiling trace data",
                    trace_data.len()
                );
                Some(trace_data)
            }
            Err(e) => {
                error!("Failed to lock profiling buffer: {}", e);
                None
            }
        }
    } else {
        None
    }
}

/// Run the Pascoletti-Serafini box search and adapt its front to the shared type.
///
/// Unlike the other two first-phase methods this one reports only points whose
/// solve proved optimality, and reports how many solver calls it used and
/// whether the front is complete.
#[cfg(feature = "gurobi")]
fn run_psbox(args: RunPsbox<'_>) -> Result<augmecon::solution::ParetoFront, PyErr> {
    // Converting the problem and building the model happen inside the method's
    // budget, not before it. The caller works out how much time is left when it
    // dispatches; everything after that point is spent from it, and on these
    // instances the conversion plus a 27905-row model build is seconds, not
    // milliseconds. Left unaccounted it put a 200-second run at 204.4s.
    let entered = std::time::Instant::now();
    let RunPsbox { problem, num_objectives, timeout, per_solve_cap, lex_extremes } = args;
    let ps_problem = to_psbox_problem(problem)?;
    let seeds = to_psbox_seeds(problem, lex_extremes);
    let ps_config = psbox::Config {
        deadline: Some(timeout.saturating_sub(entered.elapsed())),
        per_solve: Some(per_solve_cap),
        delta: None,
        bounds: to_psbox_bounds(&seeds),
        seeds,
        // These runs are always cut short, so the order points arrive in is the
        // whole result; see `psbox::Split`.
        split: psbox::Split::Bisect,
    };
    let outcome = psbox::solve(&ps_problem, &ps_config);
    info!(
        "psbox: {} nondominated points in {} solver calls ({})",
        outcome.front.len(),
        outcome.solves,
        if outcome.exhaustive { "complete front" } else { "stopped on deadline" }
    );
    let mut front = augmecon::solution::ParetoFront::new(vec![
        augmecon::model::ObjectiveDirection::Minimize;
        num_objectives
    ]);
    for s in outcome.front.solutions() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "objective magnitudes are far below 2^53; see psbox::model"
        )]
        let objectives: Vec<f64> = s.objectives.iter().map(|&v| v as f64).collect();
        let mut sol = augmecon::solution::Solution::new(objectives, s.variables.clone());
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "elapsed seconds in a bounded run; microseconds fit in u64"
        )]
        let found_us = (s.found_at * 1e6) as u64;
        sol.metadata.insert("timestamp_us".to_string(), found_us.to_string());
        front.add_solution_with_precision(sol, 0);
    }
    Ok(front)
}



/// The budget a first-phase method may actually spend.
///
/// What remains of the caller's limit after the preparation already done.
///
/// The limit governs *solving*: the payoff/lexicographic preparation and the
/// method's own solves are spent from it, and nothing is held back. Filtering
/// the front, converting it for Python and writing the trace happen after the
/// last solve returns and are deliberately outside it -- they report the answer
/// rather than compute it, and charging them to the budget would take search
/// time away to pay for bookkeeping.
///
/// A share used to be withheld here. It was sized for a per-solve model rebuild
/// that happened outside Gurobi's own time limit; `psbox::solve::Session`
/// removed the rebuild, and what was left was 0.3-0.4s of reporting tail --
/// fixed rather than proportional, and not the solver's to pay for.
///
/// Every first-phase method must take its deadline from here. Two of them --
/// Aneja & Nair and GPBA-A -- were instead handed the caller's whole `timeout`
/// after the preparation had already spent up to half of it, so they ran for
/// preparation plus the full budget. That is both an overrun in its own right
/// and an unfair comparison: the methods that did subtract their preparation
/// were measured against two that did not.
fn method_budget(timeout: Duration, elapsed: Duration) -> Duration {
    timeout.saturating_sub(elapsed)
}

/// Share of the wall-clock budget the pre-method preparation may consume.
///
/// Half. The preparation (ideal bounds and the lexicographic extremes) is
/// useful but optional refinement; the first-phase method is the thing being
/// measured, and it has to be left a working budget on instances where the
/// preparation is slow.
const PREWORK_SHARE: f64 = 0.5;

/// Weighted-sum solves the quadtree search issues before searching.
///
/// The paper uses twenty; twenty-four here, raised twice from an initial eight.
/// These solves produced nearly everything the earlier runs had to show for
/// themselves, while the feasibility checks that followed cost about four times
/// as much apiece and mostly subdivided space without closing any of it. Now
/// that the sweep is dichotomic rather than uniform no solve is wasted -- each
/// one either finds a new point or retires a segment -- so raising the count
/// costs nothing when the frontier runs out, and the shared deadline stops it
/// eating the search's budget when it does not.
#[cfg(feature = "gurobi")]
const QUADTREE_WARM_START: usize = 24;

/// Share of the budget the warm start may spend before the region search runs.
///
/// Two thirds. Every dichotomic solve is productive, so left alone the warm
/// start spends everything -- measured on `lagos_nigeria_150` as twenty-two
/// solves, twenty-two checks, and not one region examined, so none of the
/// nineteen bounds it proved was ever applied. Two thirds keeps the solves that
/// actually produce points while leaving the search enough to use them.
#[cfg(feature = "gurobi")]
const QUADTREE_WARM_START_SHARE: f64 = 2.0 / 3.0;

/// Time limit for a single quadtree feasibility check.
///
/// The paper suggests `beta * log(size)` with `beta` around five seconds. Here
/// the whole budget is small relative to how long one solve takes, and an
/// undecided region costs only a subdivision, so a short limit is the right
/// trade: it converts hard regions into smaller questions instead of waiting on
/// them. A tenth of the budget, bounded to a sensible window.
#[cfg(feature = "gurobi")]
fn quadtree_node_timeout(timeout: Duration) -> Duration {
    timeout
        .checked_div(10)
        .unwrap_or(timeout)
        .clamp(Duration::from_secs(5), Duration::from_secs(30))
}

/// Run the quadtree criterion-space search.
///
/// Reports the points carrying a proof and no others.
///
/// A point qualifies two ways. Either it came from a strictly positively
/// weighted sum solved to optimality -- which cannot return a dominated point,
/// since anything dominating it would score strictly lower -- or no unexplored
/// region can hold anything that dominates it. The first route is much the
/// commoner at these budgets, and unlike the second it does not need the search
/// to have closed off the criterion space.
///
/// Both stricter and looser policies were measured and are worse. Reporting
/// only points certified by *coverage* meant reporting nothing: a 200-second
/// run leaves over 99% of the criterion space unexplored, so the search found
/// eight points on `lagos_nigeria_150` and returned none. Reporting the whole
/// front instead admitted unproven incumbents from timed-out solves, and four
/// of seven points on `tokyo_bay_225` turned out to be dominated -- worse than
/// the method this crate exists to improve on.
#[cfg(feature = "gurobi")]
fn run_quadtree(args: RunPsbox<'_>) -> Result<augmecon::solution::ParetoFront, PyErr> {
    // Converting the problem and building the model happen inside the method's
    // budget, not before it. The caller works out how much time is left when it
    // dispatches; everything after that point is spent from it, and on these
    // instances the conversion plus a 27905-row model build is seconds, not
    // milliseconds. Left unaccounted it put a 200-second run at 204.4s.
    let entered = std::time::Instant::now();
    let RunPsbox { problem, num_objectives, timeout, per_solve_cap, lex_extremes } = args;
    let qt_problem = to_psbox_problem(problem)?;
    let seeds = to_psbox_seeds(problem, lex_extremes);

    let bounds = match to_psbox_bounds(&seeds) {
        Some(bounds) => bounds,
        None => {
            // No extremes from the caller, so pay for them here. Anti-ideal
            // bounds are looser than the extremes would give, but valid.
            let budget = psbox::solve::Budget::new_with(Some(timeout), Some(per_solve_cap));
            let mut solves = 0;
            let mut session = psbox::solve::Session::new(&qt_problem);
            psbox::compute_bounds(&mut session, &budget, &mut solves).ok_or_else(|| {
                PyValueError::new_err(
                    "quadtree could not establish criterion-space bounds within the budget",
                )
            })?
        }
    };

    let config = psbox::quadtree::Config {
        deadline: Some(timeout.saturating_sub(entered.elapsed())),
        node_timeout: Some(quadtree_node_timeout(timeout)),
        bounds,
        seeds,
        warm_start: QUADTREE_WARM_START,
        warm_start_timeout: Some(quadtree_node_timeout(timeout)),
        warm_start_share: QUADTREE_WARM_START_SHARE,
    };
    let outcome = psbox::quadtree::solve(&qt_problem, &config);
    let certified = outcome.certified.iter().filter(|c| **c).count();
    info!(
        "quadtree: {} points ({certified} proven) in {} feasibility checks, {} regions retired by \
         bound, {}, {:.2}% of the criterion space unexplored",
        outcome.front.len(),
        outcome.checks,
        outcome.eliminated,
        if outcome.exhaustive { "complete front" } else { "stopped on deadline" },
        outcome.gap
    );

    let mut front = augmecon::solution::ParetoFront::new(vec![
        augmecon::model::ObjectiveDirection::Minimize;
        num_objectives
    ]);
    for (solution, proven) in outcome.front.solutions().iter().zip(&outcome.certified) {
        if *proven {
            front.add_solution_with_precision(to_augmecon_solution(solution), 0);
        }
    }
    Ok(front)
}

/// Dispatch to [`run_quadtree`], or explain why it is unavailable.
#[cfg(not(feature = "gurobi"))]
fn run_quadtree_dispatch(_args: RunPsbox<'_>) -> Result<augmecon::solution::ParetoFront, PyErr> {
    Err(PyValueError::new_err(
        "method 'quadtree' needs the gurobi feature: it relies on the solver reporting \
         whether a region is infeasible, undecided, or holds a point",
    ))
}

/// Dispatch to [`run_quadtree`].
#[cfg(feature = "gurobi")]
fn run_quadtree_dispatch(args: RunPsbox<'_>) -> Result<augmecon::solution::ParetoFront, PyErr> {
    run_quadtree(args)
}

/// Adapt a `psbox` solution for the shared Pareto front.
#[cfg(feature = "gurobi")]
fn to_augmecon_solution(solution: &psbox::Solution) -> augmecon::solution::Solution {
    #[expect(
        clippy::cast_precision_loss,
        reason = "objective magnitudes are far below 2^53; see psbox::model"
    )]
    let objectives: Vec<f64> = solution.objectives.iter().map(|&v| v as f64).collect();
    let mut adapted = augmecon::solution::Solution::new(objectives, solution.variables.clone());
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "elapsed seconds in a bounded run; microseconds fit in u64"
    )]
    let found_us = (solution.found_at * 1e6) as u64;
    adapted.metadata.insert("timestamp_us".to_string(), found_us.to_string());
    adapted
}

/// Arguments for [`run_psbox`].
///
/// A struct rather than five positional parameters, two of which are
/// `Duration`s that would otherwise be trivial to transpose at the call site.
struct RunPsbox<'a> {
    problem: &'a augmecon::model::MultiObjectiveProblem,
    num_objectives: usize,
    /// Wall-clock left for the box search, not the caller's original budget.
    timeout: Duration,
    per_solve_cap: Duration,
    /// Lexicographic extremes already computed by the caller, if any.
    lex_extremes: &'a [augmecon::solution::Solution],
}

/// Adapt the caller's lexicographic extremes into `psbox` solutions.
///
/// These are already proven nondominated, so they are handed over as seeds
/// rather than re-derived, saving the solves that would cost.
#[cfg(feature = "gurobi")]
fn to_psbox_seeds(
    problem: &augmecon::model::MultiObjectiveProblem,
    lex_extremes: &[augmecon::solution::Solution],
) -> Vec<psbox::Solution> {
    use augmecon::model::ObjectiveDirection;
    lex_extremes
        .iter()
        .map(|sol| {
            let objectives = sol
                .objectives()
                .iter()
                .zip(&problem.objectives)
                .map(|(&raw, (_, direction))| {
                    // `psbox` minimises every objective, so a maximised one is
                    // stored negated there -- the convention `to_psbox_problem`
                    // applies to the expressions themselves.
                    let signed = match direction {
                        ObjectiveDirection::Minimize => raw,
                        ObjectiveDirection::Maximize => -raw,
                    };
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "objective values are integral and far below 2^53"
                    )]
                    let value = signed.round() as i64;
                    value
                })
                .collect();
            psbox::Solution {
                objectives,
                variables: sol.decision_variables.clone(),
                found_at: 0.0,
            }
        })
        .collect()
}

/// Derive `psbox`'s bounds from the lexicographic extremes.
///
/// For two objectives the extremes give both bounds exactly: the ideal value of
/// each objective is its minimum over the extremes, and its largest *nondominated*
/// value is its maximum over them, since no nondominated point lies outside the
/// range the extremes span. That is tighter than the anti-ideal `psbox` would
/// otherwise compute, and it saves the `2p` solves computing it would cost.
///
/// Returns `None` for anything but a complete pair, in which case `psbox`
/// establishes the bounds itself: a partial set would understate the range and
/// silently truncate the search.
#[cfg(feature = "gurobi")]
fn to_psbox_bounds(seeds: &[psbox::Solution]) -> Option<psbox::Bounds> {
    let [first, second] = seeds else { return None };
    let pairs = || first.objectives.iter().zip(&second.objectives);
    Some(psbox::Bounds {
        ideal: pairs().map(|(a, b)| *a.min(b)).collect(),
        anti_ideal: pairs().map(|(a, b)| *a.max(b)).collect(),
    })
}

/// Translate the shared MILP model into `psbox`'s form.
///
/// `psbox` minimises every objective by construction -- the rectangle geometry
/// is stated that way -- so a maximised objective is negated here rather than
/// carried as a direction flag. Both crates build on the same `good_lp` types,
/// so variables and constraints move across unchanged.
#[cfg(feature = "gurobi")]
fn to_psbox_problem(
    problem: &augmecon::model::MultiObjectiveProblem,
) -> Result<psbox::Problem, PyErr> {
    use augmecon::model::ObjectiveDirection;
    if problem.objectives.len() < 2 {
        return Err(PyValueError::new_err(format!(
            "psbox needs at least two objectives; got {}",
            problem.objectives.len()
        )));
    }
    let convert = |(expr, dir): &(augmecon::Expression, ObjectiveDirection)| match dir {
        ObjectiveDirection::Minimize => expr.clone(),
        ObjectiveDirection::Maximize => -expr.clone(),
    };
    Ok(psbox::Problem {
        variables: problem.variables.clone(),
        constraints: problem.constraints.clone(),
        objectives: problem.objectives.iter().map(convert).collect(),
        var_names: problem.var_map.clone(),
    })
}

/// Dispatch to [`run_psbox`], or explain why it is unavailable.
///
/// The box search distinguishes a proven optimum from an incumbent by reading
/// the solver's stopping status, and its guarantee rests on that, so it is
/// offered only where that is available rather than degraded silently.
#[cfg(not(feature = "gurobi"))]
fn run_psbox_dispatch(_args: RunPsbox<'_>) -> Result<augmecon::solution::ParetoFront, PyErr> {
    Err(PyValueError::new_err(
        "method 'psbox' needs the gurobi feature: it relies on the solver reporting \
         whether each solve proved optimality",
    ))
}

/// Dispatch to [`run_psbox`].
#[cfg(feature = "gurobi")]
fn run_psbox_dispatch(args: RunPsbox<'_>) -> Result<augmecon::solution::ParetoFront, PyErr> {
    run_psbox(args)
}
