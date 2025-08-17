use flate2::write::GzEncoder;
use flate2::Compression;
use pareto::{HasObjectives, MoSolution};
use pls::solution::{bitset_encoded_solution::BitsetEncodedSolution, EncodedSolution};
use serde_json;
use std::collections::HashMap;
use std::io::Write;

/// Represents metadata about an optimization trace
#[derive(serde::Serialize)]
struct TraceMetadata {
    objectives: Vec<String>,
    total_duration: u64, // in microseconds
    algorithm: String,
    solution_count: usize,
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
pub fn create_optimization_trace_archive<const N: usize>(
    explored_solutions: Vec<BitsetEncodedSolution<N>>,
    objectives: Vec<String>,
    total_duration_us: u64,
    algorithm: String,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Sort solutions by timestamp (should already be sorted, but ensure it)
    let mut solutions = explored_solutions;
    solutions.sort_by_key(|a| a.timestamp());

    if solutions.is_empty() {
        return Err("Cannot create trace archive from empty solution list".into());
    }

    // Create binary data
    let objectives_data = create_objectives_binary(&solutions)?;
    let dominated_data = create_dominated_binary(&solutions)?;
    let timestamp_data = create_timestamp_binary(&solutions)?;
    let metadata_data = create_metadata_json(&solutions, objectives, total_duration_us, algorithm)?;

    // Create tar archive in memory
    let archive_data = create_tar_archive(
        objectives_data,
        dominated_data,
        timestamp_data,
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
    solutions: &[BitsetEncodedSolution<N>],
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
    solutions: &[BitsetEncodedSolution<N>],
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

/// Creates timestamp.bin - binary file with timestamps in u32 LE format
fn create_timestamp_binary<const N: usize>(
    solutions: &[BitsetEncodedSolution<N>],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut data = Vec::new();

    for solution in solutions {
        // Use the solution's timestamp directly (already represents time since optimization start)
        let timestamp_us = solution.timestamp().as_micros() as u32; // Truncate to u32
        data.extend_from_slice(&timestamp_us.to_le_bytes());
    }

    Ok(data)
}

/// Creates metadata.json with trace information
fn create_metadata_json<const N: usize>(
    solutions: &[BitsetEncodedSolution<N>],
    objectives: Vec<String>,
    total_duration_us: u64,
    algorithm: String,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let metadata = TraceMetadata {
        objectives,
        total_duration: total_duration_us,
        algorithm,
        solution_count: solutions.len(),
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
