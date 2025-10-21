use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use pareto::{HasObjectives, MoSolution};
use pls::explored_solutions_data::SolutionFingerprint;
use pyo3::prelude::*;
use serde_json;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::time::Duration;

use crate::solution::Solution;
use crate::hypervolume::{compute_hypervolume, hypervolume_4d_min_generic, hypervolume_2d_min_generic, hypervolume_3d_min_generic, HVNumeric};

/// Represents metadata about an optimization trace
#[derive(serde::Serialize, serde::Deserialize)]
struct TraceMetadata {
    objectives: Vec<String>,
    total_duration: u64, // in microseconds
    algorithm: String,
    solution_count: usize,
    ratio: Option<Vec<u8>>,
    objective_bounds: Vec<[u64; 2]>,
    reference_point: Vec<u64>,
    second_phase_start_index: Option<usize>,
}

/// Result of dominance computation containing filtered solutions and domination mapping
pub struct DominanceInfo<const D: usize> {
    /// Filtered solutions (non-dominated at discovery if filtering was requested, all if not)
    pub solutions: Vec<SolutionFingerprint<D>>,
    /// Domination indices for the solutions in the `solutions` vec (u32::MAX = not dominated)
    /// When filter_dominated=true: shows eventual domination by later solutions
    /// When filter_dominated=false: shows domination by any solution in the set
    pub domination_indices: Vec<u32>,
}

/// Computes dominance information for a set of solutions.
/// 
/// This function performs dominance checks efficiently in a single pass and returns both:
/// - Filtered solutions (if filter_dominated=true, only solutions non-dominated at discovery time)
/// - Domination indices showing eventual domination relationships
/// 
/// When filtering (filter_dominated=true):
/// - Keeps solutions that were non-dominated at discovery time (temporal order)
/// - Computes eventual domination: shows if kept solutions are later dominated
/// - Example: S0 kept (non-dominated at t=10), but dominated.bin shows dominated by S2 (at t=30)
/// 
/// # Arguments
/// * `solutions` - Vector of solutions (will be sorted by timestamp before processing)
/// * `filter_dominated` - If true, filters out solutions dominated at discovery; if false, keeps all
/// 
/// # Returns
/// DominanceInfo containing solutions and their domination indices
pub fn compute_dominance_info<const D: usize>(
    mut solutions: Vec<SolutionFingerprint<D>>,
    filter_dominated: bool,
) -> DominanceInfo<D> {
    // Sort by timestamp first to ensure temporal order (discovery order)
    solutions.sort_by_key(|s| s.timestamp);
    
    if filter_dominated {
        // Single-pass filtering with eventual domination tracking
        let mut filtered_solutions: Vec<SolutionFingerprint<D>> = Vec::new();
        let mut domination_indices: Vec<u32> = Vec::new();
        
        'next_solution: for (sol_idx, new_solution) in solutions.into_iter().enumerate() {
            let current_kept_idx = filtered_solutions.len() as u32;
            
            // Single pass: check against all kept solutions
            for (kept_idx, kept_solution) in filtered_solutions.iter().enumerate() {
                // Is new solution dominated by this kept solution?
                if new_solution.is_dominated_by(kept_solution.objectives()) {
                    // Dominated at discovery → skip it
                    continue 'next_solution;
                }
                
                // Does the new solution dominate this kept one?
                if new_solution.dominates(kept_solution.objectives()) {
                    // Update domination index only if not already dominated
                    if domination_indices[kept_idx] == u32::MAX {
                        domination_indices[kept_idx] = current_kept_idx;
                    }
                }
            }
            
            // Not dominated at discovery → keep it
            filtered_solutions.push(new_solution);
            domination_indices.push(u32::MAX); // Not dominated yet
        }
        
        DominanceInfo {
            solutions: filtered_solutions,
            domination_indices,
        }
    } else {
        // No filtering: keep all solutions, compute domination indices
        let mut domination_indices = Vec::with_capacity(solutions.len());
        
        for (i, solution) in solutions.iter().enumerate() {
            let dominating_index = 'find_dominator: {
                for (j, other_solution) in solutions.iter().enumerate() {
                    if i != j && solution.is_dominated_by(other_solution.objectives()) {
                        break 'find_dominator j as u32;
                    }
                }
                u32::MAX
            };
            
            domination_indices.push(dominating_index);
        }
        
        DominanceInfo {
            solutions,
            domination_indices,
        }
    }
}


/// Filters dominated solutions from a list of solutions.
/// 
/// Returns only non-dominated solutions, maintaining temporal order.
/// A solution is kept if it was non-dominated at the time of its discovery
/// (i.e., not dominated by any previously discovered solution).
/// 
/// # Arguments
/// * `solutions` - Vector of solutions sorted by timestamp
/// 
/// # Returns
/// Filtered vector containing only non-dominated solutions
pub fn filter_dominated_solutions<const D: usize>(
    solutions: Vec<SolutionFingerprint<D>>
) -> Vec<SolutionFingerprint<D>> {
    compute_dominance_info(solutions, true).solutions
}

/// Creates a gzipped tar archive containing binary optimization trace data
///
/// This function creates a compressed archive containing detailed optimization traces
/// suitable for analysis and visualization. The archive contains:
///
/// - **objectives.bin**: u64 LE objectives for each solution (values correspond to objectives parameter)
/// - **dominated.bin**: u32 LE domination indices (u32::MAX = not dominated)
/// - **timestamp.bin**: u32 LE timestamps in microseconds since start
/// - **metadata.json**: JSON metadata about the trace including objective names
///
/// # Arguments
/// * `explored_solutions` - List of solutions sorted by timestamp
/// * `objectives` - Names of the objectives (e.g., ["min_cost", "cloud_coverage", "max_incidence_angle"])
/// * `total_duration_us` - Total optimization duration in microseconds
/// * `algorithm` - Name of the optimization algorithm used
/// * `objective_bounds` - Bounds for each objective
/// * `reference_point` - Reference point for hypervolume computation
/// * `precomputed_domination` - Optional pre-computed domination indices to avoid recomputation
///
/// # Returns
/// Compressed tar archive as bytes
///
/// # Example
/// ```python
/// import sims_problem
///
/// # Create some solutions
/// solutions = [
///     sims_problem.Solution.create([0, 2, 5], 1500, 250, 100000, 45, 800),
///     sims_problem.Solution.create([1, 3, 7], 1200, 300, 500000, 40, 900),
/// ]
///
/// # Create trace archive
/// archive_bytes = sims_problem.create_optimization_trace_archive(
///     solutions, ["min_cost", "cloud_coverage", "max_incidence_angle"], 2000000, "pls"
/// )
///
/// # Save to file
/// with open("trace.tar.gz", "wb") as f:
///     f.write(archive_bytes)
/// ```
pub fn create_optimization_trace_archive<const D: usize>(
    explored_solutions: Vec<SolutionFingerprint<D>>,
    objectives: Vec<String>,
    total_duration_us: u64,
    algorithm: String,
    objective_bounds: Vec<[u64; 2]>,
    reference_point: Vec<u64>,
    precomputed_domination: Option<Vec<u32>>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Keep solutions in original order (PLS discovery order)
    // Timestamps are still stored for each solution in timestamp.bin
    let solutions = explored_solutions;

    if solutions.is_empty() {
        return Err("Cannot create trace archive from empty solution list".into());
    }

    // Create binary data
    let objectives_data = create_objectives_binary(&solutions)?;
    let dominated_data = match precomputed_domination {
        Some(ref indices) => {
            // Use precomputed indices for both dominated.bin and hypervolume computation
            let dominated_bin = create_dominated_binary_from_indices(indices)?;
            let hypervolume_data = create_hypervolume_binary_with_indices(&solutions, &objective_bounds, &reference_point, indices)?;
            (dominated_bin, hypervolume_data)
        },
        None => {
            // Compute domination and use for both
            let dominated_bin = create_dominated_binary(&solutions)?;
            let hypervolume_data = create_hypervolume_binary(&solutions, &objective_bounds, &reference_point)?;
            (dominated_bin, hypervolume_data)
        }
    };
    let timestamp_data = create_timestamp_binary(&solutions)?;
    
    let metadata_data = create_metadata_json(&solutions, objectives, total_duration_us, algorithm, None, objective_bounds, reference_point)?;

    // Create tar archive in memory
    let archive_data = create_tar_archive(
        objectives_data,
        dominated_data.0,
        timestamp_data,
        dominated_data.1,
        metadata_data,
    )?;

    // Compress with gzip
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(&archive_data)
        .map_err(|e| format!("Failed to compress archive: {}", e))?;
    let compressed_data = encoder
        .finish()
        .map_err(|e| format!("Failed to finish compression: {}", e))?;

    Ok(compressed_data)
}

/// Creates objectives.bin - binary file with objectives in u64 LE format
fn create_objectives_binary<const N: usize>(
    solutions: &[SolutionFingerprint<N>],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut data = Vec::new();

    for solution in solutions {
        // Extract objectives from BitsetEncodedSolution
        let objectives = solution.objectives();
        for &objective_value in objectives.iter() {
            data.extend_from_slice(&objective_value.to_le_bytes());
        }
    }

    Ok(data)
}

/// Creates dominated.bin - binary file with domination indices in u32 LE format
fn create_dominated_binary<const N: usize>(
    solutions: &[SolutionFingerprint<N>],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut data = Vec::new();
    let mut domination_map: HashMap<usize, usize> = HashMap::new();

    // For each solution, find the first solution that dominates it
    for (i, solution_i) in solutions.iter().enumerate() {
        let mut dominating_index = u32::MAX; // Use MAX to indicate no domination

        for (j, solution_j) in solutions.iter().enumerate() {
            if i != j && solution_j.dominates(solution_i.objectives()) {
                dominating_index = j as u32;
                break; // Found the first dominating solution
            }
        }

        domination_map.insert(i, dominating_index as usize);
        data.extend_from_slice(&dominating_index.to_le_bytes());
    }

    Ok(data)
}

/// Creates dominated.bin from pre-computed domination indices
fn create_dominated_binary_from_indices(
    indices: &[u32],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut data = Vec::with_capacity(indices.len() * 4);
    
    for &index in indices {
        data.extend_from_slice(&index.to_le_bytes());
    }
    
    Ok(data)
}

/// Creates timestamp.bin - binary file with timestamps in u32 LE format
fn create_timestamp_binary<const D: usize>(
    solutions: &[SolutionFingerprint<D>],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut data = Vec::new();

    for solution in solutions {
        // Use the solution's timestamp directly (already represents time since optimization start)
        let timestamp_us = solution.timestamp.as_micros() as u32; // Truncate to u32
        data.extend_from_slice(&timestamp_us.to_le_bytes());
    }

    Ok(data)
}

/// Creates hypervolume.bin with cumulative hypervolume progression
/// 
/// This function computes hypervolume at each point in time, considering only solutions
/// that contributed to hypervolume changes (i.e., were non-dominated at discovery time).
/// For indices between computed values, hypervolume is assumed constant.
fn create_hypervolume_binary<const N: usize>(
    solutions: &[SolutionFingerprint<N>],
    objective_bounds: &[[u64; 2]],
    reference_point: &[u64],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let num_solutions = solutions.len();
    
    if num_solutions == 0 {
        return Ok(Vec::new());
    }
    
    // Create dominated binary to get domination information
    let dominated_data = create_dominated_binary(solutions)?;
    let mut dominated_indices = Vec::new();
    
    // Parse dominated binary data (u32 LE format)
    for i in 0..num_solutions {
        let byte_offset = i * 4;
        let dominated_bytes = &dominated_data[byte_offset..byte_offset + 4];
        let dominated_idx = u32::from_le_bytes(dominated_bytes.try_into().unwrap());
        dominated_indices.push(dominated_idx);
    }
    
    create_hypervolume_binary_with_indices(solutions, objective_bounds, reference_point, &dominated_indices)
}

/// Creates hypervolume.bin using pre-computed domination indices
fn create_hypervolume_binary_with_indices<const N: usize>(
    solutions: &[SolutionFingerprint<N>],
    objective_bounds: &[[u64; 2]],
    reference_point: &[u64],
    dominated_indices: &[u32],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let num_solutions = solutions.len();
    
    if num_solutions == 0 {
        return Ok(Vec::new());
    }
    
    // Find indices where hypervolume actually changes
    // These are solutions that were non-dominated when discovered (dominated[i] > i or u32::MAX)
    let mut hypervolume_change_indices = Vec::new();
    
    for i in 0..num_solutions {
        let dominated_by = dominated_indices[i];
        
        // Solution was non-dominated at discovery if:
        // - dominated[i] > i (dominated by future solution)  
        // - dominated[i] == u32::MAX (never dominated)
        if dominated_by > i as u32 || dominated_by == u32::MAX {
            hypervolume_change_indices.push(i);
        }
    }
    
    // Ensure we have at least one change point
    if hypervolume_change_indices.is_empty() && num_solutions > 0 {
        hypervolume_change_indices.push(0);
    }
    
    // Compute hypervolume for each change point
    let mut computed_hypervolumes = HashMap::new();
    
    for &change_idx in &hypervolume_change_indices {
        // Build pareto front up to this index (inclusive) using dominated info
        let mut pareto_front = Vec::new();
        
        for i in 0..=change_idx {
            let dominated_by = dominated_indices[i];
            
            // Include solution in pareto front if:
            // - Never dominated (u32::MAX)
            // - Dominated by solution discovered after change_idx
            if dominated_by == u32::MAX || dominated_by > change_idx as u32 {
                // Convert to points format for hypervolume computation
                let mut point = Vec::new();
                for obj_idx in 0..N {
                    point.push(solutions[i].objectives().iter().nth(obj_idx).copied().unwrap_or(0));
                }
                pareto_front.push(point);
            }
        }
        
        // Compute scaled hypervolume
        let scaled_hv = if pareto_front.is_empty() {
            0.0
        } else {
            compute_scaled_hypervolume(&pareto_front, objective_bounds, reference_point)?
        };
        
        computed_hypervolumes.insert(change_idx, scaled_hv);
    }
    
    // Fill in hypervolume values for all indices
    // Pre-allocate vector with correct size (8 bytes per f64)
    let mut data = vec![0u8; num_solutions * 8];
    let mut current_hv: f64 = 0.0;
    let mut change_iter = hypervolume_change_indices.iter().peekable();
    
    for i in 0..num_solutions {
        // Check if this is a change point
        if let Some(&&change_idx) = change_iter.peek() {
            if i == change_idx {
                current_hv = computed_hypervolumes[&change_idx];
                change_iter.next();
            }
        }
        
        // Write hypervolume value to the correct position using slice access
        let byte_offset = i * 8;
        data[byte_offset..byte_offset + 8].copy_from_slice(&current_hv.to_le_bytes());
    }
    
    Ok(data)
}

/// Computes scaled hypervolume for a pareto front using optimized functions from hypervolume.rs
fn compute_scaled_hypervolume(
    pareto_front: &[Vec<u64>],
    objective_bounds: &[[u64; 2]],
    reference_point: &[u64],
) -> Result<f64, Box<dyn std::error::Error>> {
    if pareto_front.is_empty() {
        return Ok(0.0);
    }
    
    let dimension = pareto_front[0].len();
    
    // Scale the pareto front to [0, 1] range using reference point normalization
    let mut scaled_front = Vec::new();
    for point in pareto_front {
        let mut scaled_point = Vec::new();
        for (i, &obj_value) in point.iter().enumerate() {
            let min_bound = objective_bounds[i][0] as f64;
            let ref_value = reference_point[i] as f64;
            
            let scaled_value = if ref_value > min_bound {
                (obj_value as f64 - min_bound) / (ref_value - min_bound)
            } else {
                0.0
            };
            scaled_point.push(scaled_value);
        }
        scaled_front.push(scaled_point);
    }
    
    // Scale reference point (should always be 1.0 with this normalization)
    let mut scaled_reference = Vec::new();
    for (i, &ref_value) in reference_point.iter().enumerate() {
        let min_bound = objective_bounds[i][0] as f64;
        
        let scaled_ref = if ref_value as f64 > min_bound {
            (ref_value as f64 - min_bound) / (ref_value as f64 - min_bound)
        } else {
            1.0 // Use 1.0 as reference if bounds are equal
        };
        scaled_reference.push(scaled_ref);
    }
    
    // Use optimized hypervolume computation functions from hypervolume.rs
    let result = match dimension {
        2 => {
            let mut mutable_front = scaled_front;
            hypervolume_2d_min_generic(&mut mutable_front, &scaled_reference)
        },
        3 => {
            let mut mutable_front = scaled_front;
            hypervolume_3d_min_generic(&mut mutable_front, &scaled_reference)
        },
        4 => {
            let mut mutable_front = scaled_front;
            hypervolume_4d_min_generic(&mut mutable_front, &scaled_reference).to_f64()
        },
        _ => return Err(format!("Hypervolume computation not supported for {} dimensions", dimension).into()),
    };
    
    // Assert that hypervolume is within expected bounds for scaled computation
    assert!(result >= 0.0, "Hypervolume must be non-negative, got: {}", result);
    assert!(result <= 1.0, "Scaled hypervolume must be <= 1.0, got: {}", result);
    
    Ok(result)
}

/// Creates metadata.json with trace information
fn create_metadata_json<const N: usize>(
    solutions: &[SolutionFingerprint<N>],
    objectives: Vec<String>,
    total_duration_us: u64,
    algorithm: String,
    ratio: Option<Vec<u8>>,
    objective_bounds: Vec<[u64; 2]>,
    reference_point: Vec<u64>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let metadata = TraceMetadata {
        objectives,
        total_duration: total_duration_us,
        algorithm,
        solution_count: solutions.len(),
        ratio,
        objective_bounds,
        reference_point,
        second_phase_start_index: None,
    };

    let json_string = serde_json::to_string_pretty(&metadata)
        .map_err(|e| format!("Failed to serialize metadata: {}", e))?;

    Ok(json_string.into_bytes())
}

/// Creates a tar archive containing all the binary files
fn create_tar_archive(
    objectives_data: Vec<u8>,
    dominated_data: Vec<u8>,
    timestamp_data: Vec<u8>,
    hypervolume_data: Vec<u8>,
    metadata_data: Vec<u8>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut archive_data = Vec::new();
    {
        let mut builder = tar::Builder::new(&mut archive_data);

        // Add objectives.bin
        let mut header = tar::Header::new_gnu();
        header
            .set_path("objectives.bin")
            .map_err(|e| format!("Invalid path for objectives.bin: {}", e))?;
        header.set_size(objectives_data.len() as u64);
        header.set_cksum();
        builder
            .append(&header, objectives_data.as_slice())
            .map_err(|e| format!("Failed to add objectives.bin: {}", e))?;

        // Add dominated.bin
        let mut header = tar::Header::new_gnu();
        header
            .set_path("dominated.bin")
            .map_err(|e| format!("Invalid path for dominated.bin: {}", e))?;
        header.set_size(dominated_data.len() as u64);
        header.set_cksum();
        builder
            .append(&header, dominated_data.as_slice())
            .map_err(|e| format!("Failed to add dominated.bin: {}", e))?;

        // Add timestamp.bin
        let mut header = tar::Header::new_gnu();
        header
            .set_path("timestamp.bin")
            .map_err(|e| format!("Invalid path for timestamp.bin: {}", e))?;
        header.set_size(timestamp_data.len() as u64);
        header.set_cksum();
        builder
            .append(&header, timestamp_data.as_slice())
            .map_err(|e| format!("Failed to add timestamp.bin: {}", e))?;

        // Add hypervolume.bin
        let mut header = tar::Header::new_gnu();
        header
            .set_path("hypervolume.bin")
            .map_err(|e| format!("Invalid path for hypervolume.bin: {}", e))?;
        header.set_size(hypervolume_data.len() as u64);
        header.set_cksum();
        builder
            .append(&header, hypervolume_data.as_slice())
            .map_err(|e| format!("Failed to add hypervolume.bin: {}", e))?;

        // Add metadata.json
        let mut header = tar::Header::new_gnu();
        header
            .set_path("metadata.json")
            .map_err(|e| format!("Invalid path for metadata.json: {}", e))?;
        header.set_size(metadata_data.len() as u64);
        header.set_cksum();
        builder
            .append(&header, metadata_data.as_slice())
            .map_err(|e| format!("Failed to add metadata.json: {}", e))?;

        builder
            .finish()
            .map_err(|e| format!("Failed to finish archive: {}", e))?;
    }

    Ok(archive_data)
}

/// Python binding for trace generation
/// 
/// Generates a compressed trace.tar.gz archive from a list of Solution objects.
/// The archive contains binary files with optimization timeline data.
///
/// # Arguments
/// * `solutions` - List of Solution objects (must be sorted by timestamp)
/// * `objectives` - Names of the objectives (e.g., ["min_cost", "cloud_coverage", "max_incidence_angle"])
/// * `algorithm` - Name of the algorithm that generated the solutions
/// * `num_objectives` - Number of objectives (2, 3, or 4)
///
/// # Returns
/// * Bytes of the compressed tar.gz archive
#[pyfunction]
pub fn generate_trace(
    solutions: Vec<PyRef<Solution>>,
    objectives: Vec<String>,
    algorithm: String,
    num_objectives: usize,
    objective_bounds: Vec<[u64; 2]>,
    reference_point: Vec<u64>,
    include_dominated: Option<bool>,
) -> PyResult<Vec<u8>> {
    // Convert solutions to the format expected by trace generation
    let include_dominated = include_dominated.unwrap_or(false);
    match num_objectives {
        2 => generate_trace_impl::<2>(solutions, objectives, algorithm, objective_bounds, reference_point, include_dominated),
        3 => generate_trace_impl::<3>(solutions, objectives, algorithm, objective_bounds, reference_point, include_dominated),
        4 => generate_trace_impl::<4>(solutions, objectives, algorithm, objective_bounds, reference_point, include_dominated),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Unsupported number of objectives. Only 2, 3, or 4 objectives are supported.",
        )),
    }
}

/// Implementation of trace generation with compile-time objective count
fn generate_trace_impl<const D: usize>(
    solutions: Vec<PyRef<Solution>>,
    objectives: Vec<String>,
    algorithm: String,
    objective_bounds: Vec<[u64; 2]>,
    reference_point: Vec<u64>,
    include_dominated: bool,
) -> PyResult<Vec<u8>> {
    if objectives.len() != D {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Expected {} objective names, got {}", D, objectives.len()),
        ));
    }

    // Convert Python Solution objects to SolutionFingerprint format
    let mut explored_solutions: Vec<SolutionFingerprint<D>> = Vec::new();
    let mut total_duration = Duration::ZERO;

    for (index, solution) in solutions.iter().enumerate() {
        
        let obj_values: pareto::Objectives<D> = std::array::from_fn(|i| {
            let objective_name = &objectives[i];
            match objective_name.as_str() {
                "min_cost" => solution.cost,
                "cloud_coverage" => solution.cloudy_area,
                "min_max_incidence_angle" => solution.max_incidence_angle.expect("min_max_incidence_angle should be set"),
                "min_resolution" => solution.min_resolutions_sum.expect("min_resolutions_sum should be set"),
                _ => unreachable!()
            }
        });

        let fingerprint = SolutionFingerprint {
            explored_neighborhood_size: 0,
            objectives: obj_values,
            iteration: index as u16, // Sequential iteration numbers
            timestamp: solution.timestamp,
        };
        explored_solutions.push(fingerprint);

        // Track total duration
        if solution.timestamp > total_duration {
            total_duration = solution.timestamp;
        }
    }

    // Sort solutions by timestamp (should already be sorted, but ensure it)
    explored_solutions.sort_by_key(|s| s.timestamp);

    // Filter dominated solutions if requested using the extracted helper function
    if !include_dominated {
        explored_solutions = filter_dominated_solutions(explored_solutions);
    }

    // Call the existing trace generation function (no pre-computed domination for this path)
    let archive_bytes = create_optimization_trace_archive(
        explored_solutions,
        objectives,
        total_duration.as_micros() as u64,
        algorithm,
        objective_bounds,
        reference_point,
        None,  // Will compute domination inside create_optimization_trace_archive
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Trace generation failed: {}", e)))?;

    Ok(archive_bytes)
}

/// Merges two trace archives into a single archive with adjusted timestamps
///
/// This function takes two compressed trace archives and combines them into one.
/// The second trace's timestamps are offset by the execution time of the first phase.
///
/// # Arguments
/// * `first_trace` - Bytes of the first trace archive (tar.gz)
/// * `second_trace` - Bytes of the second trace archive (tar.gz)  
/// * `first_phase_duration_us` - Duration of the first phase in microseconds
/// * `combined_algorithm` - Algorithm name for the merged trace
///
/// # Returns
/// * Bytes of the merged trace archive
#[pyfunction]
pub fn merge_traces(
    first_trace: Vec<u8>,
    second_trace: Vec<u8>,
    combined_algorithm: String,
    objective_bounds: Vec<[u64; 2]>,
    reference_point: Vec<u64>,
) -> PyResult<Vec<u8>> {
    // Extract data from first trace
    let first_data = extract_trace_data(&first_trace)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to extract first trace: {}", e)
        ))?;
        
    // Extract data from second trace
    let second_data = extract_trace_data(&second_trace)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to extract second trace: {}", e)
        ))?;

    // Verify objectives are compatible
    if first_data.metadata.objectives != second_data.metadata.objectives {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cannot merge traces with different objectives"
        ));
    }

    // Merge binary data
    let merged_objectives = merge_binary_data(&first_data.objectives, &second_data.objectives);
    let merged_dominated = merge_dominated_indices(
        &first_data.dominated, 
        &second_data.dominated, 
        first_data.metadata.solution_count
    );
    let merged_timestamps = merge_timestamps(
        &first_data.timestamps, 
        &second_data.timestamps, 
        first_data.metadata.total_duration
    );

    // Calculate total solution count
    let total_solutions = first_data.metadata.solution_count + second_data.metadata.solution_count;
    
    // Calculate actual total duration from last solution timestamp
    let total_duration = if total_solutions > 0 {
        let last_timestamp_bytes = &merged_timestamps[(total_solutions - 1) * 4..total_solutions * 4];
        u32::from_le_bytes(last_timestamp_bytes.try_into().unwrap()) as u64
    } else {
        0
    };

    // Extract ratio from algorithm name (e.g., "two-phase-20-80" -> [20, 80])
    let ratio = extract_ratio_from_algorithm(&combined_algorithm);

    // Recalculate hypervolume for merged solution set to ensure monotonicity
    let merged_hypervolume = create_hypervolume_binary_from_raw_data(
        &merged_objectives,
        &merged_dominated,
        &first_data.metadata.objectives.len(),
        total_solutions,
        &objective_bounds,
        &reference_point,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
        format!("Failed to recalculate hypervolume: {}", e)
    ))?;

    // Create merged metadata - change algorithm to "hybrid"
    let merged_metadata = TraceMetadata {
        objectives: first_data.metadata.objectives.clone(),
        total_duration,
        algorithm: "hybrid".to_string(),
        solution_count: total_solutions,
        ratio: Some(ratio),
        objective_bounds,
        reference_point,
        second_phase_start_index: Some(first_data.metadata.solution_count),
    };

    let metadata_bytes = serde_json::to_string_pretty(&merged_metadata)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to serialize merged metadata: {}", e)
        ))?
        .into_bytes();

    // Create merged archive
    let archive_data = create_tar_archive(
        merged_objectives,
        merged_dominated,
        merged_timestamps,
        merged_hypervolume,
        metadata_bytes,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
        format!("Failed to create merged archive: {}", e)
    ))?;

    // Compress the merged archive
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(&archive_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to compress merged archive: {}", e)
        ))?;
    let compressed_data = encoder
        .finish()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to finish compression: {}", e)
        ))?;

    Ok(compressed_data)
}

/// Structure to hold extracted trace data
struct TraceData {
    objectives: Vec<u8>,
    dominated: Vec<u8>,
    timestamps: Vec<u8>,
    hypervolume: Vec<u8>,
    metadata: TraceMetadata,
}

/// Extracts data from a compressed trace archive
fn extract_trace_data(archive_bytes: &[u8]) -> Result<TraceData, Box<dyn std::error::Error>> {
    // Decompress the archive
    let mut decoder = GzDecoder::new(archive_bytes);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;

    // Read tar archive
    let mut archive = tar::Archive::new(decompressed.as_slice());
    let entries = archive.entries()?;

    let mut objectives = Vec::new();
    let mut dominated = Vec::new();
    let mut timestamps = Vec::new();
    let mut hypervolume = Vec::new();
    let mut metadata = None;

    for entry in entries {
        let mut entry = entry?;
        let path = entry.path()?.to_string_lossy().to_string();
        let mut content = Vec::new();
        entry.read_to_end(&mut content)?;

        match path.as_str() {
            "objectives.bin" => objectives = content,
            "dominated.bin" => dominated = content,
            "timestamp.bin" => timestamps = content,
            "hypervolume.bin" => hypervolume = content,
            "metadata.json" => {
                let json_str = String::from_utf8(content)?;
                metadata = Some(serde_json::from_str::<TraceMetadata>(&json_str)?);
            }
            _ => {} // Ignore unknown files
        }
    }

    let metadata = metadata.ok_or("Missing metadata.json in trace archive")?;

    Ok(TraceData {
        objectives,
        dominated,
        timestamps,
        hypervolume,
        metadata,
    })
}

/// Merges two binary data vectors
fn merge_binary_data(first: &[u8], second: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(first.len() + second.len());
    result.extend_from_slice(first);
    result.extend_from_slice(second);
    result
}

/// Merges dominated indices with proper offset for the second trace
fn merge_dominated_indices(
    first: &[u8], 
    second: &[u8], 
    first_trace_solution_count: usize
) -> Vec<u8> {
    let mut result = Vec::with_capacity(first.len() + second.len());
    
    // Add first trace dominated indices as-is
    result.extend_from_slice(first);
    
    // Add second trace dominated indices with offset
    // Each dominated index is u32 LE (4 bytes)
    let dominated_count = second.len() / 4;
    for i in 0..dominated_count {
        let start_idx = i * 4;
        let dominated_bytes = &second[start_idx..start_idx + 4];
        let original_dominated = u32::from_le_bytes(dominated_bytes.try_into().unwrap());
        
        // Adjust dominated index to account for first trace solutions
        let adjusted_dominated = if original_dominated == u32::MAX {
            // Never dominated - keep as u32::MAX
            u32::MAX
        } else {
            // Add offset to point to correct solution in merged trace
            original_dominated + first_trace_solution_count as u32
        };
        
        result.extend_from_slice(&adjusted_dominated.to_le_bytes());
    }
    
    result
}

/// Merges timestamp data with offset for the second trace
fn merge_timestamps(
    first: &[u8], 
    second: &[u8], 
    offset_us: u64
) -> Vec<u8> {
    let mut result = Vec::with_capacity(first.len() + second.len());
    
    // Add first trace timestamps as-is
    result.extend_from_slice(first);
    
    // Add second trace timestamps with offset
    // Each timestamp is u32 LE (4 bytes)
    let timestamp_count = second.len() / 4;
    for i in 0..timestamp_count {
        let start_idx = i * 4;
        let timestamp_bytes = &second[start_idx..start_idx + 4];
        let original_timestamp = u32::from_le_bytes(timestamp_bytes.try_into().unwrap()) as u64;
        let adjusted_timestamp = (original_timestamp + offset_us) as u32;
        result.extend_from_slice(&adjusted_timestamp.to_le_bytes());
    }
    
    result
}

/// Unified hypervolume binary creation from raw data (used for merging traces)
/// This reuses the same optimized logic as create_hypervolume_binary but works with raw binary data
fn create_hypervolume_binary_from_raw_data(
    objectives_data: &[u8],
    dominated_data: &[u8],
    num_objectives: &usize,
    num_solutions: usize,
    objective_bounds: &[[u64; 2]],
    reference_point: &[u64],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let num_obj = *num_objectives;
    
    if num_solutions == 0 {
        return Ok(Vec::new());
    }
    
    // Parse objectives from binary data
    let mut all_solutions = Vec::new();
    for i in 0..num_solutions {
        let mut obj_values = Vec::new();
        for j in 0..num_obj {
            let offset = (i * num_obj + j) * 8;
            let obj_bytes = &objectives_data[offset..offset + 8];
            let obj_value = u64::from_le_bytes(obj_bytes.try_into().unwrap());
            obj_values.push(obj_value);
        }
        all_solutions.push(obj_values);
    }
    
    // Parse dominated indices from binary data (u32 LE format)
    let mut dominated_indices = Vec::new();
    for i in 0..num_solutions {
        let byte_offset = i * 4;
        let dominated_bytes = &dominated_data[byte_offset..byte_offset + 4];
        let dominated_idx = u32::from_le_bytes(dominated_bytes.try_into().unwrap());
        dominated_indices.push(dominated_idx);
    }
    
    // Find indices where hypervolume actually changes
    // These are solutions that were non-dominated when discovered (dominated[i] > i or u32::MAX)
    let mut hypervolume_change_indices = Vec::new();
    
    for i in 0..num_solutions {
        let dominated_by = dominated_indices[i];
        
        // Solution was non-dominated at discovery if:
        // - dominated[i] > i (dominated by future solution)  
        // - dominated[i] == u32::MAX (never dominated)
        if dominated_by > i as u32 || dominated_by == u32::MAX {
            hypervolume_change_indices.push(i);
        }
    }
    
    // Ensure we have at least one change point
    if hypervolume_change_indices.is_empty() && num_solutions > 0 {
        hypervolume_change_indices.push(0);
    }
    
    // Compute hypervolume for each change point
    let mut computed_hypervolumes = HashMap::new();
    
    for &change_idx in &hypervolume_change_indices {
        // Build pareto front up to this index (inclusive) using dominated info
        let mut pareto_front = Vec::new();
        
        for i in 0..=change_idx {
            let dominated_by = dominated_indices[i];
            
            // Include solution in pareto front if:
            // - Never dominated (u32::MAX)
            // - Dominated by solution discovered after change_idx
            if dominated_by == u32::MAX || dominated_by > change_idx as u32 {
                pareto_front.push(all_solutions[i].clone());
            }
        }
        
        // Compute scaled hypervolume
        let scaled_hv = if pareto_front.is_empty() {
            0.0
        } else {
            match num_obj {
                2 => compute_scaled_hypervolume_2d(&pareto_front, objective_bounds, reference_point),
                3 => compute_scaled_hypervolume_3d(&pareto_front, objective_bounds, reference_point),
                4 => compute_scaled_hypervolume_4d(&pareto_front, objective_bounds, reference_point),
                _ => return Err(format!("Unsupported number of objectives: {}", num_obj).into()),
            }?
        };
        
        computed_hypervolumes.insert(change_idx, scaled_hv);
    }
    
    // Fill in hypervolume values for all indices
    // Pre-allocate vector with correct size (8 bytes per f64)
    let mut data = vec![0u8; num_solutions * 8];
    let mut current_hv: f64 = 0.0;
    let mut change_iter = hypervolume_change_indices.iter().peekable();
    
    for i in 0..num_solutions {
        // Check if this is a change point
        if let Some(&&change_idx) = change_iter.peek() {
            if i == change_idx {
                current_hv = computed_hypervolumes[&change_idx];
                change_iter.next();
            }
        }
        
        // Ensure hypervolume is within bounds
        let bounded_hv = current_hv.min(1.0).max(0.0);
        assert!(bounded_hv >= 0.0 && bounded_hv <= 1.0, 
            "Hypervolume value {} is out of bounds [0.0, 1.0]", bounded_hv);
        
        // Write hypervolume value to the correct position using slice access
        let byte_offset = i * 8;
        data[byte_offset..byte_offset + 8].copy_from_slice(&bounded_hv.to_le_bytes());
    }
    
    Ok(data)
}

/// Helper functions for hypervolume calculation by dimension
fn compute_scaled_hypervolume_2d(
    solutions: &[Vec<u64>], 
    objective_bounds: &[[u64; 2]], 
    reference_point: &[u64]
) -> Result<f64, Box<dyn std::error::Error>> {
    // Create normalization bounds using reference point as upper bound
    let norm_bounds_0 = [objective_bounds[0][0], reference_point[0]];
    let norm_bounds_1 = [objective_bounds[1][0], reference_point[1]];
    
    let mut normalized_solutions: Vec<Vec<f64>> = solutions.iter()
        .map(|sol| vec![
            normalize_objective(sol[0], norm_bounds_0),
            normalize_objective(sol[1], norm_bounds_1),
        ])
        .collect();
    
    let normalized_ref = vec![
        normalize_objective(reference_point[0], norm_bounds_0),  // This will be 1.0
        normalize_objective(reference_point[1], norm_bounds_1),  // This will be 1.0
    ];
    
    Ok(hypervolume_2d_min_generic(&mut normalized_solutions, &normalized_ref))
}

fn compute_scaled_hypervolume_3d(
    solutions: &[Vec<u64>], 
    objective_bounds: &[[u64; 2]], 
    reference_point: &[u64]
) -> Result<f64, Box<dyn std::error::Error>> {
    // Create normalization bounds using reference point as upper bound
    let norm_bounds_0 = [objective_bounds[0][0], reference_point[0]];
    let norm_bounds_1 = [objective_bounds[1][0], reference_point[1]];
    let norm_bounds_2 = [objective_bounds[2][0], reference_point[2]];
    
    let mut normalized_solutions: Vec<Vec<f64>> = solutions.iter()
        .map(|sol| vec![
            normalize_objective(sol[0], norm_bounds_0),
            normalize_objective(sol[1], norm_bounds_1),
            normalize_objective(sol[2], norm_bounds_2),
        ])
        .collect();
    
    let normalized_ref = vec![
        normalize_objective(reference_point[0], norm_bounds_0),  // This will be 1.0
        normalize_objective(reference_point[1], norm_bounds_1),  // This will be 1.0
        normalize_objective(reference_point[2], norm_bounds_2),  // This will be 1.0
    ];
    
    Ok(hypervolume_3d_min_generic(&mut normalized_solutions, &normalized_ref))
}

fn compute_scaled_hypervolume_4d(
    solutions: &[Vec<u64>], 
    objective_bounds: &[[u64; 2]], 
    reference_point: &[u64]
) -> Result<f64, Box<dyn std::error::Error>> {
    // Create normalization bounds using reference point as upper bound
    let norm_bounds_0 = [objective_bounds[0][0], reference_point[0]];
    let norm_bounds_1 = [objective_bounds[1][0], reference_point[1]];
    let norm_bounds_2 = [objective_bounds[2][0], reference_point[2]];
    let norm_bounds_3 = [objective_bounds[3][0], reference_point[3]];
    
    let mut normalized_solutions: Vec<Vec<f64>> = solutions.iter()
        .map(|sol| vec![
            normalize_objective(sol[0], norm_bounds_0),
            normalize_objective(sol[1], norm_bounds_1),
            normalize_objective(sol[2], norm_bounds_2),
            normalize_objective(sol[3], norm_bounds_3),
        ])
        .collect();
    
    let normalized_ref = vec![
        normalize_objective(reference_point[0], norm_bounds_0),  // This will be 1.0
        normalize_objective(reference_point[1], norm_bounds_1),  // This will be 1.0
        normalize_objective(reference_point[2], norm_bounds_2),  // This will be 1.0
        normalize_objective(reference_point[3], norm_bounds_3),  // This will be 1.0
    ];
    
    Ok(hypervolume_4d_min_generic(&mut normalized_solutions, &normalized_ref))
}

fn normalize_objective(value: u64, bounds: [u64; 2]) -> f64 {
    let [min_val, max_val] = bounds;
    if max_val == min_val {
        0.0
    } else {
        // Normalize relative to the range, where max_val (reference point) becomes 1.0
        // and min_val (best possible) becomes 0.0
        (value as f64 - min_val as f64) / (max_val as f64 - min_val as f64)
    }
}

/// Extracts ratio from algorithm name (e.g., "two-phase-20-80" -> [20, 80])
fn extract_ratio_from_algorithm(algorithm: &str) -> Vec<u8> {
    if let Some(phase_part) = algorithm.strip_prefix("two-phase-") {
        let parts: Vec<&str> = phase_part.split('-').collect();
        if parts.len() == 2 {
            if let (Ok(first), Ok(second)) = (parts[0].parse::<u8>(), parts[1].parse::<u8>()) {
                return vec![first, second];
            }
        }
    }
    // Default ratio if parsing fails
    vec![50, 50]
}

/// Calculates objective bounds and reference point from merged objective data
fn calculate_objective_bounds_and_ref_point(
    merged_objectives: &[u8], 
    num_objectives: usize
) -> PyResult<(Vec<[u64; 2]>, Vec<u64>)> {
    if merged_objectives.len() % (num_objectives * 8) != 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Invalid objective data length"
        ));
    }

    let solution_count = merged_objectives.len() / (num_objectives * 8);
    let mut min_values = vec![u64::MAX; num_objectives];
    let mut max_values = vec![u64::MIN; num_objectives];

    // Read each solution's objectives
    for sol_idx in 0..solution_count {
        for obj_idx in 0..num_objectives {
            let byte_offset = (sol_idx * num_objectives + obj_idx) * 8;
            let obj_bytes = &merged_objectives[byte_offset..byte_offset + 8];
            let obj_value = u64::from_le_bytes(obj_bytes.try_into().unwrap());
            
            min_values[obj_idx] = min_values[obj_idx].min(obj_value);
            max_values[obj_idx] = max_values[obj_idx].max(obj_value);
        }
    }

    // Create objective bounds as [min, max] pairs
    let objective_bounds: Vec<[u64; 2]> = (0..num_objectives)
        .map(|i| [min_values[i], max_values[i]])
        .collect();

    // Create reference point as max + 1 for each objective
    let reference_point: Vec<u64> = max_values.iter().map(|&max_val| max_val + 1).collect();

    Ok((objective_bounds, reference_point))
}

/// Calculates objective bounds and reference point from solution fingerprints
pub fn calculate_objective_bounds_from_solutions<const N: usize>(
    solutions: &[SolutionFingerprint<N>]
) -> Result<(Vec<[u64; 2]>, Vec<u64>), Box<dyn std::error::Error>> {
    if solutions.is_empty() {
        return Err("Cannot calculate objective bounds from empty solution list".into());
    }

    let num_objectives = N;
    let mut min_values = vec![u64::MAX; num_objectives];
    let mut max_values = vec![u64::MIN; num_objectives];

    // Find min and max for each objective
    for solution in solutions {
        for obj_idx in 0..num_objectives {
            let obj_value = solution.objectives().iter().nth(obj_idx).copied().unwrap_or(0);
            min_values[obj_idx] = min_values[obj_idx].min(obj_value);
            max_values[obj_idx] = max_values[obj_idx].max(obj_value);
        }
    }

    // Create objective bounds as [min, max] pairs
    let objective_bounds: Vec<[u64; 2]> = (0..num_objectives)
        .map(|i| [min_values[i], max_values[i]])
        .collect();

    // Create reference point as max + 1 for each objective
    let reference_point: Vec<u64> = max_values.iter().map(|&max_val| max_val + 1).collect();

    Ok((objective_bounds, reference_point))
}
