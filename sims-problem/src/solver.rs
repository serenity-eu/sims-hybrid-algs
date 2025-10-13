#[cfg(feature = "milp")]
use augmecon::{sims_problem::SimsInstance, Augmecon, HasObjectives, ObjectiveDirection, Options};
use log::{debug, info};
use pareto::ParetoFront;
use pls::explored_solutions_data::SolutionFingerprint;
use pls::pareto_local_search::ParetoLocalSearch;
#[cfg(feature = "milp")]
use pls::pareto_local_search::ParetoLocalSearch;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::{iter::IntoIterator, ops::RangeInclusive, time::Duration};

use crate::problem::SimsDiscreteProblem;
use crate::solution::SolvingResult;
#[cfg(feature = "milp")]
use crate::solution::Solution;
use crate::trace;

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
        solver_name="cbc".to_string()
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
    objective_bounds=None
))]
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
) -> PyResult<SolvingResult> {
    debug!(
        "solve_with_pls_monolith called with {} objectives",
        objectives.len()
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
        Some(pop) => format!("provided {} solutions", pop.len()),
        None => format!("random generation size {}", initial_population_size),
    };
    let bounds_info = match &objective_bounds {
        Some(bounds) => format!("provided bounds: {:?}", bounds),
        None => "no bounds provided".to_string(),
    };
    info!(
        "Starting PLS algorithm with {} objectives: {objectives:?}, plots: {plots}, timeout: {timeout_seconds}s, max_iterations: {max_iterations}, deterministic: {is_deterministic}, initial_population: {initial_pop_info}, neighborhood: {neighborhood_size_min}..{neighborhood_size_max}, objective_bounds: {bounds_info}",
        objectives.len()
    );

    let neighborhood_size_range: RangeInclusive<u32> =
        neighborhood_size_min..=neighborhood_size_max;

    // Convert to PLS problem format - common for all dimensions
    let raw_instance = pls::problem::SIMSProblemInstanceRaw {
        name: "python_instance".to_string(),
        num_images: sims_instance.num_images,
        universe_size: sims_instance.universe,
        images: sims_instance
            .images
            .iter()
            .map(|img| img.iter().map(|&x| x + 1).collect())
            .collect(),
        costs: sims_instance.costs.iter().map(|&c| c as u64).collect(),
        clouds: sims_instance
            .clouds
            .iter()
            .map(|cloud| cloud.iter().map(|&x| x + 1).collect())
            .collect(),
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
            use pls::solution_set_impl::BTreeSolutionSet;

            debug!("Using 2D optimization with objectives: {objectives:?}");

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

            let mut pls_problem = pls::problem::Problem::from_raw_with_objectives(
                raw_instance,
                objective_definitions,
            )
            .map_err(|e| PyValueError::new_err(format!("Failed to create 2D problem: {e}")))?;

            // Set objective bounds if provided
            if let Some(ref bounds) = objective_bounds {
                let bounds_array: Vec<[u64; 2]> = bounds
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
                pls_problem.set_objective_bounds(bounds_array);
            }

            // Create initial population manually for 2D
            let mut initial_solution_set = BTreeSolutionSet::new("initial_2d_solutions");
            
            if let Some(provided_population) = &initial_population {
                if !provided_population.is_empty() {
                    // Use provided initial population
                    info!("Using provided initial population of {} solutions for 2D PLS", provided_population.len());
                    for solution in provided_population {
                        let selected_images: Vec<usize> = solution.selected_images.iter().cloned().collect();
                        let pls_solution = BitsetEncodedSolution::from_selected_images(&selected_images, &pls_problem);
                        initial_solution_set.try_insert(&pls_solution);
                    }
                } else {
                    // Provided population is empty, generate random population
                    info!("Provided initial population is empty, generating random initial population of {} solutions for 2D PLS", initial_population_size);
                    for _ in 0..initial_population_size {
                        let random_solution = if is_deterministic {
                            BitsetEncodedSolution::random_with_seed(&pls_problem, 1_234_567_890)
                        } else {
                            BitsetEncodedSolution::random(&pls_problem)
                        };
                        initial_solution_set.try_insert(&random_solution);
                    }
                }
            } else {
                // Generate random initial population
                info!("Generating random initial population of {} solutions for 2D PLS", initial_population_size);
                for _ in 0..initial_population_size {
                    let random_solution = if is_deterministic {
                        BitsetEncodedSolution::random_with_seed(&pls_problem, 1_234_567_890)
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

            info!(
                "2D PLS completed, processing {} solutions",
                final_solution_set.len()
            );

            // Generate 2D plot if requested
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

            // Convert 2D solutions back to Python format
            let final_solutions: Vec<BitsetEncodedSolution<2>> =
                final_solution_set.into_iter().collect();
            let mut python_final_solutions = Vec::new();
            for solution in final_solutions.iter() {
                let py_solution: crate::solution::Solution = (solution, &pls_problem).into();
                python_final_solutions.push(py_solution);
            }

            let explored_solutions: Vec<SolutionFingerprint<2>> = pareto_local_search.explored_solutions_fingerprints();

            info!(
                "Successfully converted {} 2D final solutions to Python format",
                python_final_solutions.len(),
            );

            // Generate trace if requested
            if trace {
                info!("Generating 2D optimization trace archive");
                
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
                    // Calculate from explored solutions
                    trace::calculate_objective_bounds_from_solutions(&explored_solutions)
                        .map_err(|e| PyValueError::new_err(format!("Failed to calculate objective bounds: {}", e)))?
                };
                
                let trace_archive = trace::create_optimization_trace_archive(
                    explored_solutions,
                    objectives,
                    timeout.as_micros() as u64,
                    "PLS-2D".to_string(),
                    trace_objective_bounds,
                    reference_point,
                )
                .map_err(|e| {
                    PyValueError::new_err(format!("Failed to create trace archive: {}", e))
                })?;
                Ok(crate::solution::SolvingResult::with_trace(
                    python_final_solutions,
                    trace_archive,
                ))
            } else {
                Ok(crate::solution::SolvingResult::new(
                    python_final_solutions,
                ))
            }
        }
        3 => {
            use pls::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;
            use pls::solution_set_impl::NdTreeSolutionSet;

            debug!("Using 3D optimization with objectives: {objectives:?}");

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

            let mut pls_problem = pls::problem::Problem::from_raw_with_objectives(
                raw_instance,
                objective_definitions,
            )
            .map_err(|e| PyValueError::new_err(format!("Failed to create 3D problem: {e}")))?;

            // Set objective bounds if provided
            if let Some(ref bounds) = objective_bounds {
                let bounds_array: Vec<[u64; 2]> = bounds
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
                pls_problem.set_objective_bounds(bounds_array);
            }

            // Create initial population manually for 3D using ND-Tree
            let mut initial_solution_set: NdTreeSolutionSet<BitsetEncodedSolution<3>, 3> =
                NdTreeSolutionSet::new("initial_3d_solutions");
            
            if let Some(provided_population) = &initial_population {
                if !provided_population.is_empty() {
                    // Use provided initial population
                    info!("Using provided initial population of {} solutions for 3D PLS", provided_population.len());
                    for solution in provided_population {
                        let selected_images: Vec<usize> = solution.selected_images.iter().cloned().collect();
                        let pls_solution = BitsetEncodedSolution::from_selected_images(&selected_images, &pls_problem);
                        initial_solution_set.try_insert(&pls_solution);
                    }
                } else {
                    // Provided population is empty, generate random population
                    info!("Provided initial population is empty, generating random initial population of {} solutions for 3D PLS", initial_population_size);
                    for _ in 0..initial_population_size {
                        let random_solution = if is_deterministic {
                            BitsetEncodedSolution::random_with_seed(&pls_problem, 1_234_567_890)
                        } else {
                            BitsetEncodedSolution::random(&pls_problem)
                        };
                        initial_solution_set.try_insert(&random_solution);
                    }
                }
            } else {
                // Generate random initial population
                info!("Generating random initial population of {} solutions for 3D PLS", initial_population_size);
                for _ in 0..initial_population_size {
                    let random_solution = if is_deterministic {
                        BitsetEncodedSolution::random_with_seed(&pls_problem, 1_234_567_890)
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

            info!(
                "3D PLS completed, processing {} solutions",
                final_solution_set.len()
            );

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

            // Convert 3D solutions back to Python format
            let final_solutions: Vec<BitsetEncodedSolution<3>> =
                final_solution_set.into_iter().collect();
            let mut python_final_solutions = Vec::new();
            for solution in final_solutions.iter() {
                let py_solution: crate::solution::Solution = (solution, &pls_problem).into();
                python_final_solutions.push(py_solution);
            }

            let explored_solutions: Vec<SolutionFingerprint<3>> = pareto_local_search.explored_solutions_fingerprints();

            info!(
                "Successfully converted {} 3D final solutions to Python format",
                python_final_solutions.len(),
            );

            // Generate trace if requested
            if trace {
                info!("Generating 3D optimization trace archive");
                
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
                    // Calculate from explored solutions
                    trace::calculate_objective_bounds_from_solutions(&explored_solutions)
                        .map_err(|e| PyValueError::new_err(format!("Failed to calculate objective bounds: {}", e)))?
                };
                
                let trace_archive = trace::create_optimization_trace_archive(
                    explored_solutions,
                    objectives,
                    timeout.as_micros() as u64,
                    "PLS-3D".to_string(),
                    trace_objective_bounds,
                    reference_point,
                )
                .map_err(|e| {
                    PyValueError::new_err(format!("Failed to create trace archive: {}", e))
                })?;
                Ok(crate::solution::SolvingResult::with_trace(
                    python_final_solutions,
                    trace_archive,
                ))
            } else {
                Ok(crate::solution::SolvingResult::new(
                    python_final_solutions,
                ))
            }
        }
        4 => {
            use pls::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;
            use pls::solution_set_impl::NdTreeSolutionSet;

            debug!("Using 4D optimization with objectives: {objectives:?}");

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

            let mut pls_problem = pls::problem::Problem::from_raw_with_objectives(
                raw_instance,
                objective_definitions,
            )
            .map_err(|e| PyValueError::new_err(format!("Failed to create 4D problem: {e}")))?;

            // Set objective bounds if provided
            if let Some(ref bounds) = objective_bounds {
                let bounds_array: Vec<[u64; 2]> = bounds
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
                pls_problem.set_objective_bounds(bounds_array);
            }

            // Create initial population manually for 4D using ND-Tree
            let mut initial_solution_set: NdTreeSolutionSet<BitsetEncodedSolution<4>, 4> =
                NdTreeSolutionSet::new("initial_4d_solutions");
            
            if let Some(provided_population) = &initial_population {
                if !provided_population.is_empty() {
                    // Use provided initial population
                    info!("Using provided initial population of {} solutions for 4D PLS", provided_population.len());
                    for solution in provided_population {
                        let selected_images: Vec<usize> = solution.selected_images.iter().cloned().collect();
                        let pls_solution = BitsetEncodedSolution::from_selected_images(&selected_images, &pls_problem);
                        initial_solution_set.try_insert(&pls_solution);
                    }
                } else {
                    // Provided population is empty, generate random population
                    info!("Provided initial population is empty, generating random initial population of {} solutions for 4D PLS", initial_population_size);
                    for _ in 0..initial_population_size {
                        let random_solution = if is_deterministic {
                            BitsetEncodedSolution::random_with_seed(&pls_problem, 1_234_567_890)
                        } else {
                            BitsetEncodedSolution::random(&pls_problem)
                        };
                        initial_solution_set.try_insert(&random_solution);
                    }
                }
            } else {
                // Generate random initial population
                info!("Generating random initial population of {} solutions for 4D PLS", initial_population_size);
                for _ in 0..initial_population_size {
                    let random_solution = if is_deterministic {
                        BitsetEncodedSolution::random_with_seed(&pls_problem, 1_234_567_890)
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

            info!(
                "4D PLS completed, processing {} solutions",
                final_solution_set.len()
            );

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

            // Convert 4D solutions back to Python format
            let final_solutions: Vec<BitsetEncodedSolution<4>> =
                final_solution_set.into_iter().collect();
            let mut python_final_solutions = Vec::new();
            for solution in final_solutions.iter() {
                let py_solution: crate::solution::Solution = (solution, &pls_problem).into();
                python_final_solutions.push(py_solution);
            }

            let explored_solutions: Vec<SolutionFingerprint<4>> = pareto_local_search.explored_solutions_fingerprints();

            info!(
                "Successfully converted {} 4D final solutions to Python format",
                python_final_solutions.len(),
            );

            // Generate trace if requested
            if trace {
                info!("Generating 4D optimization trace archive");
                
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
                    // Calculate from explored solutions
                    trace::calculate_objective_bounds_from_solutions(&explored_solutions)
                        .map_err(|e| PyValueError::new_err(format!("Failed to calculate objective bounds: {}", e)))?
                };
                
                let trace_archive = trace::create_optimization_trace_archive(
                    explored_solutions,
                    objectives,
                    timeout.as_micros() as u64,
                    "PLS-4D".to_string(),
                    trace_objective_bounds,
                    reference_point,
                )
                .map_err(|e| {
                    PyValueError::new_err(format!("Failed to create trace archive: {}", e))
                })?;
                Ok(crate::solution::SolvingResult::with_trace(
                    python_final_solutions,
                    trace_archive,
                ))
            } else {
                Ok(crate::solution::SolvingResult::new(
                    python_final_solutions,
                ))
            }
        }
        n => Err(PyValueError::new_err(format!(
            "Unsupported number of objectives: {n}. Supported: 2, 3, or 4 objectives."
        ))),
    }
}

/// Solves the SIMS problem using MILP with AUGMECON for exact Pareto solutions
#[cfg(feature = "milp")]
#[expect(
    clippy::too_many_arguments,
    reason = "It's okay for Python API to have many parameters"
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
    solver_name="cbc".to_string(),
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
        "max_incidence_angle",
    ];
    for obj in &objectives {
        if !valid_objectives.contains(&obj.as_str()) {
            return Err(PyValueError::new_err(format!(
                "Invalid objective '{obj}'. Valid objectives are: {valid_objectives:?}"
            )));
        }
    }

    // Convert objectives to indices for augmecon
    let mut objective_indices = Vec::new();
    let mut objective_directions = Vec::new();

    for obj in &objectives {
        match obj.as_str() {
            "min_cost" => {
                objective_indices.push(0);
                objective_directions.push(ObjectiveDirection::Minimize);
            }
            "cloud_coverage" => {
                objective_indices.push(1);
                objective_directions.push(ObjectiveDirection::Minimize);
            }
            "min_resolution" => {
                objective_indices.push(2);
                objective_directions.push(ObjectiveDirection::Minimize);
            }
            "max_incidence_angle" => {
                objective_indices.push(3);
                objective_directions.push(ObjectiveDirection::Minimize);
            }
            _ => unreachable!(), // Already validated above
        }
    }

    let timeout_seconds = timeout.as_secs_f64();
    info!(
        "Starting MILP algorithm with objectives: {objectives:?}, grid_points: {grid_points}, timeout: {timeout_seconds}s, solver: {solver_name}"
    );

    // Note: The timeout parameter is passed to AUGMECON but may not be fully enforced
    // in the current implementation. The solver will attempt to respect the timeout
    // but this is dependent on the underlying AUGMECON solver implementation.

    // Convert SimsDiscreteProblem to SimsInstance for augmecon
    let mut sims_augmecon_instance = SimsInstance::new(
        sims_instance.num_images,
        sims_instance.universe,
        sims_instance.num_images, // Use one cloud entity per image
        sims_instance.max_cloud_area,
    );

    // Convert images (sets of universe points covered)
    for (i, image_set) in sims_instance.images.iter().enumerate() {
        let coverage_set: std::collections::HashSet<usize> = image_set.iter().cloned().collect();
        sims_augmecon_instance.set_image_coverage(i, coverage_set);
        sims_augmecon_instance.set_cost(i, sims_instance.costs[i] as f64);
        sims_augmecon_instance.set_resolution(i, sims_instance.resolution[i] as f64);
        sims_augmecon_instance.set_incidence_angle(i, sims_instance.incidence_angle[i] as f64);
    }

    // Set universe point areas
    for (k, &area) in sims_instance.areas.iter().enumerate() {
        sims_augmecon_instance.set_area(k, area as f64);
    }

    // Convert cloud coverage data - each image has its own cloud entity
    for (i, cloud_set) in sims_instance.clouds.iter().enumerate() {
        let cloud_coverage_set: std::collections::HashSet<usize> =
            cloud_set.iter().cloned().collect();
        sims_augmecon_instance.set_cloud_coverage(i, cloud_coverage_set);

        // Calculate cloud area as sum of covered universe areas
        let cloud_area: f64 = cloud_set
            .iter()
            .map(|&point| sims_instance.areas[point] as f64)
            .sum();
        sims_augmecon_instance.set_cloud_area(i, cloud_area);
    }

    // Create MultiObjectiveProblem using the free function
    let problem = augmecon::sims_problem::create_sims_problem(&sims_augmecon_instance);

    // Configure options
    let options = Options::new()
        .with_name("sims_problem")
        .with_grid_points(grid_points)
        .with_bypass_coefficient(bypass_coefficient)
        .with_early_exit(early_exit)
        .with_flag_array(flag_array)
        .with_timeout(timeout);

    // Solve with AUGMECON
    let mut solver = Augmecon::try_new(problem, options)
        .map_err(|e| PyValueError::new_err(format!("Failed to create AUGMECON solver: {e}")))?;

    let pareto_solutions = solver
        .solve()
        .map_err(|e| PyValueError::new_err(format!("MILP solving failed: {e}")))?;

    info!("MILP solving completed, converting solutions");

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

        // Extract objective values
        let milp_objectives = milp_solution.objectives();
        let cost = milp_objectives.first().copied().unwrap_or(0.0) as u64;
        let cloudy_area = milp_objectives.get(1).copied().unwrap_or(0.0) as u64;

        // Handle optional objectives
        let min_resolution_sum = if milp_objectives.len() > 2 {
            Some(milp_objectives[2] as u64)
        } else {
            None
        };

        let max_incidence_angle = if milp_objectives.len() > 3 {
            Some(milp_objectives[3] as u64)
        } else {
            None
        };

        // Create Python solution with timestamp (use solution index * 1ms as timestamp)
        let py_solution = Solution::create(
            selected_images,
            cost,
            cloudy_area,
            i as u64 * 1000, // Simple timestamp based on index
            max_incidence_angle,
            min_resolution_sum,
        );

        debug!(
            "Converted MILP solution {}: cost={}, cloudy_area={}, selected_images={:?}",
            i,
            py_solution.cost,
            py_solution.cloudy_area,
            py_solution.get_selected_images_list()
        );

        python_solutions.push(py_solution);
    }

    info!(
        "Successfully converted {} MILP solutions to Python format",
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

    debug!("Using 2D hybrid optimization");

    // Convert to PLS problem format and create 2D problem with specified objectives
    let raw_instance = pls::problem::SIMSProblemInstanceRaw {
        name: "python_hybrid_instance".to_string(),
        num_images: sims_instance.num_images,
        universe_size: sims_instance.universe,
        images: sims_instance
            .images
            .iter()
            .map(|img| img.iter().map(|&x| x + 1).collect())
            .collect(),
        costs: sims_instance.costs.iter().map(|&c| c as u64).collect(),
        clouds: sims_instance
            .clouds
            .iter()
            .map(|cloud| cloud.iter().map(|&x| x + 1).collect())
            .collect(),
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
        pls::problem::Problem::from_raw_with_objectives(raw_instance, objective_definitions)
            .map_err(|e| PyValueError::new_err(format!("Failed to create 2D problem: {e}")))?;

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
        let mut random_solutions: BTreeSolutionSet<BitsetEncodedSolution<2>, 2> =
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
    let final_solutions: Vec<BitsetEncodedSolution<2>> = final_solution_set.into_iter().collect();
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

    debug!("Using 3D hybrid optimization");

    // Convert to PLS problem format
    let raw_instance = pls::problem::SIMSProblemInstanceRaw {
        name: "python_hybrid_instance".to_string(),
        num_images: sims_instance.num_images,
        universe_size: sims_instance.universe,
        images: sims_instance
            .images
            .iter()
            .map(|img| img.iter().map(|&x| x + 1).collect())
            .collect(),
        costs: sims_instance.costs.iter().map(|&c| c as u64).collect(),
        clouds: sims_instance
            .clouds
            .iter()
            .map(|cloud| cloud.iter().map(|&x| x + 1).collect())
            .collect(),
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
        pls::problem::Problem::from_raw_with_objectives(raw_instance, objective_definitions)
            .map_err(|e| PyValueError::new_err(format!("Failed to create 3D problem: {e}")))?;

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
        let mut random_solutions: NdTreeSolutionSet<BitsetEncodedSolution<3>, 3> =
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
    let final_solutions: Vec<BitsetEncodedSolution<3>> = final_solution_set.into_iter().collect();
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

    debug!("Using 4D hybrid optimization");

    // Convert to PLS problem format
    let raw_instance = pls::problem::SIMSProblemInstanceRaw {
        name: "python_hybrid_instance".to_string(),
        num_images: sims_instance.num_images,
        universe_size: sims_instance.universe,
        images: sims_instance
            .images
            .iter()
            .map(|img| img.iter().map(|&x| x + 1).collect())
            .collect(),
        costs: sims_instance.costs.iter().map(|&c| c as u64).collect(),
        clouds: sims_instance
            .clouds
            .iter()
            .map(|cloud| cloud.iter().map(|&x| x + 1).collect())
            .collect(),
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
        pls::problem::Problem::from_raw_with_objectives(raw_instance, objective_definitions)
            .map_err(|e| PyValueError::new_err(format!("Failed to create 4D problem: {e}")))?;

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
        let mut random_solutions: NdTreeSolutionSet<BitsetEncodedSolution<4>, 4> =
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
    let final_solutions: Vec<BitsetEncodedSolution<4>> = final_solution_set.into_iter().collect();
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
