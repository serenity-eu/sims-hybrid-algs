#[cfg(feature = "milp")]
use augmecon::{sims_problem::{SimsInstance, SimsObjective}, HasObjectives, Options, GpbaA, GpbaConfig};
use log::{debug, error, info};
use std::str::FromStr;
use pareto::ParetoFront;
use pls::explored_solutions_data::SolutionFingerprint;
use pls::pareto_local_search::ParetoLocalSearch;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::{iter::IntoIterator, ops::RangeInclusive, thread, time::Duration};
#[cfg(feature = "milp")]
use std::collections::HashSet;

use crate::problem::SimsDiscreteProblem;
use crate::solution::SolvingResult;
#[cfg(feature = "milp")]
use crate::solution::Solution;
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
        solver_name="coin_cbc".to_string()
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
    profiling_trace=false
))]
#[allow(unused_variables, reason = "plot_output_path is used only when plotting feature is enabled")]
pub fn solve_with_pls(
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
                log::warn!("Global tracing subscriber already set, profiling may not work correctly");
            }
        }
        (Some(guard), Some(buffer))
    } else {
        (None, None)
    };
    
    // Convert string to enum
    let solution_set_type = pareto_archive.parse::<SolutionSetType>()
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
            info!("PLS starting with {} solutions from initial population", pop.len());
            format!("provided {} solutions", pop.len())
        }
        None => {
            info!("PLS starting with random initial population of size {}", initial_population_size);
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
            use pls::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;
            use pls::solution_set_impl::{LinkedListSolutionSet, VecSolutionSet, BTreeSolutionSet};
            use pls::problem_bitset::ProblemBitset;

            debug!("Using 2D optimization with objectives: {objectives:?}, solution_set_type: {:?}", solution_set_type);

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

            let mut pls_problem = ProblemBitset::from_raw_with_objectives(
                &raw_instance,
                objective_definitions,
            );

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
                let bounds_array: [[u64; 2]; 2] = bounds_vec.try_into()
                    .map_err(|_| PyValueError::new_err("Expected exactly 2 objective bounds for 2D problem"))?;
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
                    let mut pareto_local_search = pls::pareto_local_search::ParetoLocalSearch::new(
                        &pls_problem,
                        &initial_solution_set,
                        neighborhood_size_range,
                        is_deterministic,
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
            
            // Select solution set type based on enum and run PLS
            let (final_solutions, explored_solutions) = match solution_set_type {
                SolutionSetType::LinkedList => {
                    info!("Using LinkedListSolutionSet for 2D PLS");
                    run_pls_2d_with_archive!(LinkedListSolutionSet<BitsetEncodedSolution<ProblemBitset<2>, 2>, 2>, "linked-list")
                },
                SolutionSetType::Vector => {
                    info!("Using VecSolutionSet for 2D PLS");
                    run_pls_2d_with_archive!(VecSolutionSet<BitsetEncodedSolution<ProblemBitset<2>, 2>, 2>, "vector")
                },
                SolutionSetType::NdTree => {
                    // For 2D, nd-tree uses BTreeSolutionSet
                    info!("Using BTreeSolutionSet for 2D PLS (nd-tree)");
                    run_pls_2d_with_archive!(BTreeSolutionSet<BitsetEncodedSolution<ProblemBitset<2>, 2>, 2>, "nd-tree")
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
                let (trace_objective_bounds, reference_point) = if let Some(provided_bounds) = &objective_bounds {
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
                    trace::calculate_objective_bounds_from_solutions(&trace_solutions)
                        .map_err(|e| PyValueError::new_err(format!("Failed to calculate objective bounds: {}", e)))?
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
            use pls::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;
            use pls::solution_set_impl::{LinkedListSolutionSet, VecSolutionSet, NdTreeSolutionSet};
            use pls::problem_bitset::ProblemBitset;

            debug!("Using 3D optimization with objectives: {objectives:?}, solution_set_type: {:?}", solution_set_type);

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

            let mut pls_problem = ProblemBitset::from_raw_with_objectives(
                &raw_instance,
                objective_definitions,
            );

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
                let bounds_array: [[u64; 2]; 3] = bounds_vec.try_into()
                    .map_err(|_| PyValueError::new_err("Expected exactly 3 objective bounds for 3D problem"))?;
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
                    let mut pareto_local_search = ParetoLocalSearch::new(
                        &pls_problem,
                        &initial_solution_set,
                        neighborhood_size_range,
                        is_deterministic,
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
            
            // Select solution set type based on enum and run PLS
            let (final_solutions, explored_solutions) = match solution_set_type {
                SolutionSetType::LinkedList => {
                    info!("Using LinkedListSolutionSet for 3D PLS");
                    run_pls_3d_with_archive!(LinkedListSolutionSet<BitsetEncodedSolution<ProblemBitset<3>, 3>, 3>, "linked-list")
                },
                SolutionSetType::Vector => {
                    info!("Using VecSolutionSet for 3D PLS");
                    run_pls_3d_with_archive!(VecSolutionSet<BitsetEncodedSolution<ProblemBitset<3>, 3>, 3>, "vector")
                },
                SolutionSetType::NdTree => {
                    info!("Using NdTreeSolutionSet for 3D PLS (nd-tree)");
                    run_pls_3d_with_archive!(NdTreeSolutionSet<BitsetEncodedSolution<ProblemBitset<3>, 3>, 3>, "nd-tree")
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
                let (trace_objective_bounds, reference_point) = if let Some(provided_bounds) = &objective_bounds {
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
                    trace::calculate_objective_bounds_from_solutions(&trace_solutions)
                        .map_err(|e| PyValueError::new_err(format!("Failed to calculate objective bounds: {}", e)))?
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
            use pls::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;
            use pls::solution_set_impl::{LinkedListSolutionSet, VecSolutionSet, NdTreeSolutionSet};
            use pls::problem_bitset::ProblemBitset;

            debug!("Using 4D optimization with objectives: {objectives:?}, solution_set_type: {:?}", solution_set_type);

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

            let mut pls_problem = ProblemBitset::from_raw_with_objectives(
                &raw_instance,
                objective_definitions,
            );

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
                let bounds_array: [[u64; 2]; 4] = bounds_vec.try_into()
                    .map_err(|_| PyValueError::new_err("Expected exactly 4 objective bounds for 4D problem"))?;
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
                    let mut pareto_local_search = pls::pareto_local_search::ParetoLocalSearch::new(
                        &pls_problem,
                        &initial_solution_set,
                        neighborhood_size_range,
                        is_deterministic,
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
            
            // Select solution set type based on enum and run PLS
            let (final_solutions, explored_solutions) = match solution_set_type {
                SolutionSetType::LinkedList => {
                    info!("Using LinkedListSolutionSet for 4D PLS");
                    run_pls_4d_with_archive!(LinkedListSolutionSet<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4>, "linked-list")
                },
                SolutionSetType::Vector => {
                    info!("Using VecSolutionSet for 4D PLS");
                    run_pls_4d_with_archive!(VecSolutionSet<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4>, "vector")
                },
                SolutionSetType::NdTree => {
                    info!("Using NdTreeSolutionSet for 4D PLS (nd-tree)");
                    run_pls_4d_with_archive!(NdTreeSolutionSet<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4>, "nd-tree")
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
                info!("Generating 4D optimization trace archive ({} explored solutions)", explored_solutions.len());
                
                // Compute dominance info (filtering + domination indices in one pass)
                info!("Computing dominance info (filter_dominated={})...", !include_dominated);
                let dominance_start = std::time::Instant::now();
                let dominance_info = if include_dominated {
                    // Don't filter, but still compute domination indices
                    trace::compute_dominance_info(explored_solutions, false)
                } else {
                    info!("Filtering dominated solutions from trace");
                    // Filter and compute domination indices
                    trace::compute_dominance_info(explored_solutions, true)
                };
                info!("Dominance computation done in {:.3}s ({} -> {} solutions)",
                    dominance_start.elapsed().as_secs_f64(),
                    dominance_info.domination_indices.len(),
                    dominance_info.solutions.len(),
                );
                
                let trace_solutions = dominance_info.solutions;
                let domination_indices = Some(dominance_info.domination_indices);
                
                // Use provided objective bounds or calculate from solutions
                info!("Computing objective bounds for {} trace solutions...", trace_solutions.len());
                let bounds_start = std::time::Instant::now();
                let (trace_objective_bounds, reference_point) = if let Some(provided_bounds) = &objective_bounds {
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
                    trace::calculate_objective_bounds_from_solutions(&trace_solutions)
                        .map_err(|e| PyValueError::new_err(format!("Failed to calculate objective bounds: {}", e)))?
                };
                info!("Objective bounds computed in {:.3}s", bounds_start.elapsed().as_secs_f64());
                
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
                info!("Trace archive created in {:.3}s ({} bytes)",
                    archive_start.elapsed().as_secs_f64(), trace_archive.len());
                
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
fn compute_min_resolutions_sum(selected_images: &[usize], problem_data: &SimsDiscreteProblem) -> i64 {
    // For each universe element, find minimum resolution among images that cover it
    (0..problem_data.universe)
        .map(|u| {
            selected_images.iter()
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
#[allow(unused_variables, reason = "grid_points parameter reserved for future use")]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string()],
    grid_points=50,
    timeout=Duration::from_secs(300),
    bypass_coefficient=true,
    early_exit=true,
    flag_array=true,
    solver_name="coin_cbc".to_string(),
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
) -> PyResult<SolvingResult> {
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

    // BUILD CLOUD COVERAGE RELATIONSHIPS FIRST (matching test_gpba_phases.rs logic)
    // sims_instance.clouds contains which elements are cloudy in each image
    // We need to build which clouds each image can COVER
    // A cloud from image i can be covered by image j if:
    // - j contains that element AND
    // - j does not have clouds on that element
    
    use std::collections::HashMap;
    
    // Build cloud_id_to_area mapping
    let mut cloud_id_to_area: HashMap<usize, f64> = HashMap::new();
    for cloudy_elements in &sims_instance.clouds {
        for &cloud_id in cloudy_elements {
            if cloud_id < sims_instance.areas.len() {
                cloud_id_to_area
                    .entry(cloud_id)
                    .or_insert(sims_instance.areas[cloud_id] as f64);
            }
        }
    }
    
    let actual_num_clouds = cloud_id_to_area.len();
    info!("Found {} unique clouds across all images", actual_num_clouds);
    
    // Convert SimsDiscreteProblem to SimsInstance for augmecon
    // CRITICAL: Use actual number of unique clouds, not num_images!
    let mut sims_augmecon_instance = SimsInstance::new(
        sims_instance.num_images,
        sims_instance.universe,
        actual_num_clouds,  // Use actual cloud count (431 for lagos_nigeria_30)
        sims_instance.max_cloud_area as i32,
    );

    // Convert images (sets of universe points covered)
    for (i, image_set) in sims_instance.images.iter().enumerate() {
        let coverage_set: std::collections::HashSet<usize> = image_set.iter().cloned().collect();
        debug!("Image {}: covers {} universe points", i, coverage_set.len());
        sims_augmecon_instance.set_image_coverage(i, coverage_set);
        sims_augmecon_instance.set_cost(i, sims_instance.costs[i] as f64);
        sims_augmecon_instance.set_resolution(i, sims_instance.resolution[i] as f64);
        sims_augmecon_instance.set_incidence_angle(i, sims_instance.incidence_angle[i] as f64);
    }

    // Set universe point areas
    for (k, &area) in sims_instance.areas.iter().enumerate() {
        sims_augmecon_instance.set_area(k, area as f64);
    }
    
    // Build image_clouds: which clouds each image can cover
    // For each cloud in image i, check if OTHER images j (i != j) can cover it
    let mut image_clouds: Vec<std::collections::HashSet<usize>> = 
        vec![std::collections::HashSet::new(); sims_instance.num_images];
    
    for i in 0..sims_instance.clouds.len() {
        let image_cloud_set: std::collections::HashSet<usize> = 
            sims_instance.clouds[i].iter().copied().collect();
        
        for &cloud_id in &image_cloud_set {
            // Check which OTHER images (j != i) can cover this cloud
            for (j, image) in image_clouds.iter_mut().enumerate().take(sims_instance.num_images) {
                if i != j && sims_instance.images[j].contains(&cloud_id) {
                    // Image j contains this element, check if it's cloud-free
                    let j_has_cloud = j < sims_instance.clouds.len() 
                        && sims_instance.clouds[j].contains(&cloud_id);
                    
                    if !j_has_cloud {
                        // Image j can cover cloud_id (has element, no clouds on it)
                        image.insert(cloud_id);
                    }
                }
            }
        }
    }
    
    // Set cloud coverage (which clouds each image can COVER)
    for (i, clouds) in image_clouds.iter().enumerate() {
        sims_augmecon_instance.set_cloud_coverage(i, clouds.clone());
    }
    
    // Set cloud_ids vector (list of all unique cloud IDs)
    let mut cloud_ids_vec: Vec<usize> = cloud_id_to_area.keys().copied().collect();
    cloud_ids_vec.sort_unstable();
    sims_augmecon_instance.cloud_ids = cloud_ids_vec;
    
    // Set cloud areas for all clouds
    for (&cloud_id, &area) in &cloud_id_to_area {
        sims_augmecon_instance.set_cloud_area(cloud_id, area);
    }

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
        _ => {
            error!("Unknown solver_name '{}'", solver_name);
            return Err(PyValueError::new_err(format!(
                "Unknown solver_name '{}'. Valid options are: default, coin_cbc, highs, scip",
                solver_name
            )));
        }
    };
    
    info!("Using solver: {} (from solver_name='{}')", solver, solver_name);
    
    // Configure options for GPBA-A with Python parameters
    let options = Options::default()
        .with_solver(solver)
        .with_bypass_coefficient(bypass_coefficient)
        .with_early_exit(early_exit)
        .with_flag_array(flag_array);
    
    info!("GPBA-A Options: solver={}, bypass_coefficient={}, early_exit={}, flag_array={}", 
          solver, bypass_coefficient, early_exit, flag_array);
    
    // Calculate heuristic nadir bounds (matching Python non-inlined version)
    // This avoids unbounded maximization issues with auxiliary variables
    let objectives_enum: Vec<SimsObjective> = objectives.iter().map(|obj_name| {
        match obj_name.as_str() {
            "min_cost" => SimsObjective::MinCost,
            "cloud_coverage" => SimsObjective::CloudCoverage,
            "min_resolution" => SimsObjective::MinResolution,
            "min_max_incidence_angle" => SimsObjective::MaxIncidenceAngle,
            _ => panic!("Invalid objective: {}", obj_name),  // Already validated earlier
        }
    }).collect();
    let nadir_heuristic = sims_augmecon_instance.calculate_nadir_heuristic(&objectives_enum);
    info!("Using heuristic nadir bounds: {:?}", nadir_heuristic);
    
    // Compute ideal bounds by minimizing each objective
    info!("Computing ideal bounds by minimizing each objective");
    let mut ideal_bounds = Vec::with_capacity(objectives.len());
    let start_time = std::time::Instant::now();
    for (i, _objective) in objectives.iter().enumerate() {
        let elapsed = start_time.elapsed();
        if elapsed >= timeout {
            return Err(PyValueError::new_err("Timeout exceeded while computing ideal bounds"));
        }
        let timeout_remaining = timeout.checked_sub(elapsed).unwrap_or(Duration::from_secs(0));
        let solution = augmecon::single_objective::SingleObjectiveSolver::new(&problem, &options)
            .solve_objective(i, Some(timeout_remaining))
            .map_err(|e| PyValueError::new_err(format!("Failed to compute ideal for objective {}: {}", i, e)))?;
        
        if !solution.feasible {
            return Err(PyValueError::new_err(format!("Problem infeasible when minimizing objective {}", i)));
        }
        
        ideal_bounds.push(solution.objective_values[i]);
        info!("Ideal value for objective {} ({}): {}", i, objectives[i], solution.objective_values[i]);
    }
    
    // Configure GPBA-A for Python-compatible dynamic interval exploration (gamma=1)
    let config = GpbaConfig {
        primary_objective: 0,  // First objective is primary
        manual_bounds: Some((ideal_bounds, nadir_heuristic)),  // Use computed ideal + heuristic nadir
    };

    info!("Using GPBA-A algorithm with Python-compatible dynamic interval exploration (gamma=1)");
    
    // Solve with GPBA-A (Coverage-focused representation)
    let mut gpba_a = GpbaA::new(config);
    
    // Set timeout if provided
    if timeout.as_secs() > 0 {
        gpba_a = gpba_a.with_timeout(timeout);
    }
    
    let pareto_front = gpba_a
        .generate_representation(&problem, &options)
        .map_err(|e| PyValueError::new_err(format!("GPBA-A solving failed: {e}")))?;
    
    let pareto_solutions = &pareto_front.solutions;

    info!("GPBA-A solving completed with {} solutions, converting", pareto_solutions.len());

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
                    let computed_cost: i64 = selected_images.iter()
                        .map(|&img_idx| sims_instance.costs[img_idx])
                        .sum();
                    cost = Some(computed_cost as u64);
                }
                "cloud_coverage" => {
                    let cloudy = compute_cloudy_area(&selected_images, sims_instance);
                    cloudy_area = Some(cloudy as u64);
                }
                "max_incidence_angle" => {
                    if let Some(max_angle) = selected_images.iter()
                        .map(|&img_idx| sims_instance.incidence_angle[img_idx])
                        .max() {
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

        // Create Python solution with computed objective values
        let py_solution = Solution::create(
            selected_images.clone(),
            cost,
            cloudy_area,
            i as u64 * 1000, // Simple timestamp based on index
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

    Ok(crate::solution::SolvingResult::new(
        python_solutions,
    ))
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
            false, // No trace for internal hybrid calls
            None,  // No objective bounds
            false, // Don't include dominated solutions
            "nd-tree".to_string(), // Default pareto archive
            false, // No profiling trace for hybrid
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
    )?;

    info!(
        "MILP phase completed with {} solutions",
        milp_solutions.final_solutions.len()
    );

    if milp_solutions.final_solutions.is_empty() {
        info!("MILP found no solutions, falling back to PLS only");
        let solving_result = solve_with_pls(
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
            false, // No trace for internal hybrid calls
            None,  // No objective bounds
            false, // Don't include dominated solutions
            "nd-tree".to_string(), // Default pareto archive
            false, // No profiling trace for hybrid
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
    use pls::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;
    use pls::solution_set_impl::BTreeSolutionSet;
    use pls::problem_bitset::ProblemBitset;

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

    let pls_problem =
        ProblemBitset::from_raw_with_objectives(&raw_instance, objective_definitions);

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
    let final_solutions: Vec<BitsetEncodedSolution<ProblemBitset<2>, 2>> = final_solution_set.into_iter().collect();
    let mut python_solutions = Vec::new();

    for solution in final_solutions.iter() {
        let py_solution: Solution = (solution, &pls_problem).into();
        python_solutions.push(py_solution);
    }

    info!(
        "Successfully converted {} hybrid 2D solutions",
        python_solutions.len()
    );

    Ok(crate::solution::SolvingResult::new(
        python_solutions,
    ))
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
    use pls::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;
    use pls::solution_set_impl::NdTreeSolutionSet;
    use pls::problem_bitset::ProblemBitset;

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

    let pls_problem =
        ProblemBitset::from_raw_with_objectives(&raw_instance, objective_definitions);

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
    let final_solutions: Vec<BitsetEncodedSolution<ProblemBitset<3>, 3>> = final_solution_set.into_iter().collect();
    let mut python_solutions = Vec::new();

    for solution in final_solutions.iter() {
        let py_solution: Solution = (solution, &pls_problem).into();
        python_solutions.push(py_solution);
    }

    info!(
        "Successfully converted {} hybrid 3D solutions",
        python_solutions.len()
    );
    Ok(crate::solution::SolvingResult::new(
        python_solutions,
    ))
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
    use pls::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;
    use pls::solution_set_impl::NdTreeSolutionSet;
    use pls::problem_bitset::ProblemBitset;

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

    let pls_problem =
        ProblemBitset::from_raw_with_objectives(&raw_instance, objective_definitions);

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
    let final_solutions: Vec<BitsetEncodedSolution<ProblemBitset<4>, 4>> = final_solution_set.into_iter().collect();
    let mut python_solutions = Vec::new();

    for solution in final_solutions.iter() {
        let py_solution: Solution = (solution, &pls_problem).into();
        python_solutions.push(py_solution);
    }

    info!(
        "Successfully converted {} hybrid 4D solutions",
        python_solutions.len()
    );
    Ok(crate::solution::SolvingResult::new(
        python_solutions,
    ))
}

/// Helper function to extract Chrome tracing data from shared buffer
fn read_profiling_trace_data(
    guard: Option<tracing_chrome::FlushGuard>,
    buffer: Option<std::sync::Arc<std::sync::Mutex<Vec<u8>>>>
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
                info!("Successfully captured {} bytes of profiling trace data", trace_data.len());
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
