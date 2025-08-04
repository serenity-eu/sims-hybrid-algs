use augmecon::{sims_problem::SimsInstance, Augmecon, HasObjectives, ObjectiveDirection, Options};
use log::{debug, info};
use pls::{pareto_local_search::ParetoLocalSearch, solution_set::SolutionSet};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::{ops::RangeInclusive, time::Duration};

use crate::conversion::PlsSolutionWithTimestamp;
use crate::problem::SimsDiscreteProblem;
use crate::solution::Solution;

/// Solves the SIMS problem using Pareto Local Search with flexible objective configuration
#[expect(clippy::too_many_arguments, reason = "It's okay for Python API to have many parameters")]
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
    neighborhood_size_min=1,
    neighborhood_size_max=6
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
    neighborhood_size_min: u32,
    neighborhood_size_max: u32,
) -> PyResult<Vec<Solution>> {
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

    // Determine if we need 2D or 4D optimization based on objectives
    let use_multiobjective = objectives.len() > 2
        || objectives.contains(&"min_resolution".to_string())
        || objectives.contains(&"max_incidence_angle".to_string());

    let timeout_seconds = timeout.as_secs_f64();
    info!(
        "Starting PLS algorithm with objectives: {objectives:?}, plots: {plots}, timeout: {timeout_seconds}s, max_iterations: {max_iterations}, deterministic: {is_deterministic}, population_size: {initial_population_size}, neighborhood: {neighborhood_size_min}..{neighborhood_size_max}"
    );

    let neighborhood_size_range: RangeInclusive<u32> =
        neighborhood_size_min..=neighborhood_size_max;

    if use_multiobjective {
        // 4D optimization (cost + cloud coverage + resolution + incidence angle)
        use pls::objectives::{
            CloudyAreaObjective, MaxIncidenceAngleObjective, MinResolutionObjective,
            TotalCostObjective,
        };
        use pls::solution_set_impl::NdTreeSolutionSet;

        debug!("Using 4D optimization (cost + cloud coverage + resolution + incidence angle)");

        // Convert to PLS problem format and create 4D problem with all objectives
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

        // Create 4D problem with all objectives
        let objective_definitions: Vec<Box<dyn pls::objectives::ObjectiveDefinition<4>>> = vec![
            Box::new(TotalCostObjective { index: 0 }),
            Box::new(CloudyAreaObjective { index: 1 }),
            Box::new(MinResolutionObjective { index: 2 }),
            Box::new(MaxIncidenceAngleObjective { index: 3 }),
        ];

        let pls_problem = pls::problem::Problem::from_raw_with_objective_definitions(
            raw_instance,
            objective_definitions,
        )
        .map_err(|e| PyValueError::new_err(format!("Failed to create 4D problem: {e}")))?;

        debug!(
            "Created 4D PLS problem: {} images, universe size {}",
            sims_instance.num_images, sims_instance.universe
        );

        // Create initial population using ND-Tree for 4D optimization
        let initial_solution_set: NdTreeSolutionSet<pls::solution::BitsetEncodedSolution<4>, 4> =
            if is_deterministic {
                NdTreeSolutionSet::random_with_seed(
                    initial_population_size,
                    &pls_problem,
                    1_234_567_890,
                )
            } else {
                NdTreeSolutionSet::random(initial_population_size, &pls_problem)
            };

        // Create and run PLS
        let mut pareto_local_search = ParetoLocalSearch::new(
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

        // Generate 4D plot grid if requested
        if plots {
            #[cfg(feature = "plotting")]
            {
                let objective_names = pls_problem.objective_names();
                pls::plotting::draw_solutions_plot(
                    &pareto_local_search.explored_solutions,
                    &objective_names,
                );

                // Handle custom plot output path
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

        // Convert solutions back to Python format
        let final_solutions: Vec<pls::solution::BitsetEncodedSolution<4>> =
            final_solution_set.into_iter().collect();

        debug!(
            "Converting {} 4D PLS solutions to Python format",
            final_solutions.len()
        );

        let mut python_solutions = Vec::new();
        for (i, solution) in final_solutions.iter().enumerate() {
            debug!(
                "Processing 4D solution {}: objectives = {:?}",
                i, solution.objectives
            );

            // Get timestamp from explored solutions if available
            let timestamp_us = pareto_local_search
                .explored_solutions
                .get_solution_fingerprint(solution)
                .map(|fp| fp.time.as_micros() as u64)
                .unwrap_or(i as u64 * 1000); // Fallback: use index * 1ms

            let py_solution: Solution =
                PlsSolutionWithTimestamp::new(solution, timestamp_us).into();
            debug!(
                "Converted 4D solution {}: cost={}, cloudy_area={}, selected_images={:?}",
                i,
                py_solution.cost,
                py_solution.cloudy_area,
                py_solution.get_selected_images_list()
            );
            python_solutions.push(py_solution);
        }

        info!(
            "Successfully converted {} 4D solutions to Python format",
            python_solutions.len()
        );

        Ok(python_solutions)
    } else {
        // 2D optimization (cost + cloud coverage)
        use pls::solution_set_impl::BTreeSolutionSet;

        debug!("Using 2D optimization (cost + cloud coverage)");

        // Convert to PLS problem format
        let pls_problem = sims_instance.to_pls_problem();
        debug!(
            "Converted SIMS problem to PLS format: {} images, universe size {}",
            sims_instance.num_images, sims_instance.universe
        );

        // Create initial population
        let initial_solution_set = if is_deterministic {
            BTreeSolutionSet::random_with_seed(initial_population_size, &pls_problem, 1_234_567_890)
        } else {
            BTreeSolutionSet::random(initial_population_size, &pls_problem)
        };

        // Create and run PLS
        let mut pareto_local_search = ParetoLocalSearch::new(
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

                // Handle custom plot output path
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

        // Convert solutions back to Python format
        let final_solutions: Vec<pls::solution::EncodedSolution<2>> =
            final_solution_set.into_iter().collect();

        debug!(
            "Converting {} PLS solutions to Python format",
            final_solutions.len()
        );

        let mut python_solutions = Vec::new();
        for (i, solution) in final_solutions.iter().enumerate() {
            debug!(
                "Processing solution {}: objectives = {:?}",
                i, solution.objectives
            );

            // Get timestamp from explored solutions if available
            let timestamp_us = pareto_local_search
                .explored_solutions
                .get_solution_fingerprint(solution)
                .map(|fp| fp.time.as_micros() as u64)
                .unwrap_or(i as u64 * 1000); // Fallback: use index * 1ms

            let py_solution: Solution =
                PlsSolutionWithTimestamp::new(solution, timestamp_us).into();
            debug!(
                "Converted solution {}: cost={}, cloudy_area={}, selected_images={:?}",
                i,
                py_solution.cost,
                py_solution.cloudy_area,
                py_solution.get_selected_images_list()
            );
            python_solutions.push(py_solution);
        }

        info!(
            "Successfully converted {} solutions to Python format",
            python_solutions.len()
        );

        Ok(python_solutions)
    }
}

/// Solves the SIMS problem using MILP with AUGMECON for exact Pareto solutions
#[expect(clippy::too_many_arguments, reason = "It's okay for Python API to have many parameters")]
#[pyfunction]
#[pyo3(signature = (
    sims_instance,
    objectives=vec!["min_cost".to_string(), "cloud_coverage".to_string()],
    grid_points=50,
    timeout=Duration::from_secs(300),
    bypass_coefficient=true,
    early_exit=true,
    flag_array=true,
    solver_name="cbc".to_string()
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
) -> PyResult<Vec<Solution>> {
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
        let cost = milp_objectives.first().copied().unwrap_or(0.0) as i32;
        let cloudy_area = milp_objectives.get(1).copied().unwrap_or(0.0) as i32;

        // Handle optional objectives
        let min_resolution_sum = if milp_objectives.len() > 2 {
            Some(milp_objectives[2] as i32)
        } else {
            None
        };

        let max_incidence_angle = if milp_objectives.len() > 3 {
            Some(milp_objectives[3] as i32)
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

    Ok(python_solutions)
}
