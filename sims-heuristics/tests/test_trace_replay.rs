//! Test that verifies different Pareto archive implementations produce identical results
//! when processing the same sequence of solutions (trace replay).
//!
//! This test loads PLS exploration traces from actual runs and replays them into different
//! archive implementations. If the implementations are correct, they should all converge
//! to the same final Pareto front, regardless of their internal data structures.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::doc_markdown)]

use pareto::{HasObjectives, ParetoFront};
use pls::objectives::ObjectiveType;
use pls::problem::SIMSProblemInstanceRaw;
use pls::problem_bitset::ProblemBitset;
use pls::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;
use pls::solution_set_impl::{LinkedListSolutionSet, NdTreeSolutionSet, VecSolutionSet};
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

/// Create a minimal dummy problem for initializing trackers in tests
fn create_dummy_problem<const D: usize>(
    num_images: usize,
    universe_size: usize,
) -> ProblemBitset<D> {
    let objectives = std::array::from_fn(|_| ObjectiveType::TotalCost);

    let raw_data = SIMSProblemInstanceRaw {
        name: "dummy".to_string(),
        num_images,
        universe_size,
        images: vec![vec![]; num_images],
        costs: vec![0; num_images],
        clouds: vec![vec![]; num_images],
        areas: vec![0; universe_size],
        max_cloud_area: 0,
        resolution: vec![0; num_images],
        incidence_angle: vec![0; num_images],
    };

    ProblemBitset::from_raw_with_objectives(&raw_data, objectives)
}

/// Load trace data from JSON file and create BitsetEncodedSolutions
fn load_trace_as_solutions<const D: usize>(
    filename: &str,
    num_images: usize,
) -> Vec<BitsetEncodedSolution<ProblemBitset<D>, D>> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push(filename);

    let json_data = fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Failed to read trace file: {}", path.display()));

    let parsed: serde_json::Value =
        serde_json::from_str(&json_data).expect("Failed to parse trace JSON");

    let solutions = parsed["solutions"]
        .as_array()
        .expect("Expected 'solutions' array in trace JSON");

    // Create a dummy problem for initializing trackers
    let dummy_problem = create_dummy_problem::<D>(num_images, 1);

    solutions
        .iter()
        .filter_map(|sol| {
            let objectives_array = sol["objectives"].as_array()?;
            let objectives_vec: Vec<u64> = objectives_array
                .iter()
                .map(|v| v.as_u64().expect("Expected u64 objectives"))
                .collect();

            if objectives_vec.len() != D {
                return None;
            }

            let mut objectives = [0u64; D];
            objectives.copy_from_slice(&objectives_vec);

            let selected_images_array = sol["selected_images"].as_array()?;
            let selected_images_vec: Vec<usize> = selected_images_array
                .iter()
                .map(|v| v.as_u64().expect("Expected u64 image indices") as usize)
                .collect();

            Some(BitsetEncodedSolution::from_selected_images(
                &selected_images_vec,
                &dummy_problem,
            ))
        })
        .collect()
}

/// Extract final Pareto front as a set of objective vectors
fn extract_pareto_front<S, const D: usize>(archive: &S) -> HashSet<[u64; D]>
where
    S: for<'a> ParetoFront<'a, BitsetEncodedSolution<ProblemBitset<D>, D>>,
{
    archive.iter().map(|sol| *sol.objectives()).collect()
}

#[test]
fn test_ndtree_trace_replay_4d() {
    // Load the nd-tree trace
    let solutions = load_trace_as_solutions::<4>("tests/data/pls_trace_4d_ndtree.json", 50);

    println!("Loaded nd-tree trace with {} solutions", solutions.len());

    // Create all three archives
    let mut ndtree_archive =
        NdTreeSolutionSet::<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4>::new("ndtree");
    let mut vector_archive =
        VecSolutionSet::<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4>::new("vector");
    let mut linkedlist_archive =
        LinkedListSolutionSet::<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4>::new("linkedlist");

    // Replay the trace into all archives
    for solution in solutions {
        ndtree_archive.try_insert(&solution);
        vector_archive.try_insert(&solution);
        linkedlist_archive.try_insert(&solution);
    }

    // Extract final Pareto fronts
    let ndtree_front = extract_pareto_front(&ndtree_archive);
    let vector_front = extract_pareto_front(&vector_archive);
    let linkedlist_front = extract_pareto_front(&linkedlist_archive);

    println!("Final Pareto front sizes:");
    println!("  nd-tree: {}", ndtree_front.len());
    println!("  vector: {}", vector_front.len());
    println!("  linked list: {}", linkedlist_front.len());

    // All should produce identical Pareto fronts
    assert_eq!(
        ndtree_front, vector_front,
        "nd-tree and vector produced different Pareto fronts from same trace"
    );
    assert_eq!(
        ndtree_front, linkedlist_front,
        "nd-tree and linked list produced different Pareto fronts from same trace"
    );
}

#[test]
fn test_vector_trace_replay_4d() {
    // Load the vector trace
    let solutions = load_trace_as_solutions::<4>("tests/data/pls_trace_4d_vector.json", 50);

    println!("Loaded vector trace with {} solutions", solutions.len());

    // Create all three archives
    let mut ndtree_archive =
        NdTreeSolutionSet::<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4>::new("ndtree");
    let mut vector_archive =
        VecSolutionSet::<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4>::new("vector");
    let mut linkedlist_archive =
        LinkedListSolutionSet::<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4>::new("linkedlist");

    // Replay the trace into all archives
    for solution in solutions {
        ndtree_archive.try_insert(&solution);
        vector_archive.try_insert(&solution);
        linkedlist_archive.try_insert(&solution);
    }

    // Extract final Pareto fronts
    let ndtree_front = extract_pareto_front(&ndtree_archive);
    let vector_front = extract_pareto_front(&vector_archive);
    let linkedlist_front = extract_pareto_front(&linkedlist_archive);

    println!("Final Pareto front sizes:");
    println!("  nd-tree: {}", ndtree_front.len());
    println!("  vector: {}", vector_front.len());
    println!("  linked list: {}", linkedlist_front.len());

    // All should produce identical Pareto fronts
    assert_eq!(
        ndtree_front, vector_front,
        "nd-tree and vector produced different Pareto fronts from same trace"
    );
    assert_eq!(
        ndtree_front, linkedlist_front,
        "nd-tree and linked list produced different Pareto fronts from same trace"
    );
}

#[test]
fn test_combined_trace_replay_4d() {
    println!("Testing combined trace replay:");

    // Load both traces
    let ndtree_solutions = load_trace_as_solutions::<4>("tests/data/pls_trace_4d_ndtree.json", 50);
    let vector_solutions = load_trace_as_solutions::<4>("tests/data/pls_trace_4d_vector.json", 50);

    println!("  nd-tree trace: {} solutions", ndtree_solutions.len());
    println!("  vector trace: {} solutions", vector_solutions.len());

    // Create all three archives
    let mut ndtree_archive =
        NdTreeSolutionSet::<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4>::new("ndtree");
    let mut vector_archive =
        VecSolutionSet::<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4>::new("vector");
    let mut linkedlist_archive =
        LinkedListSolutionSet::<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4>::new("linkedlist");

    // Replay both traces into all archives
    for solution in ndtree_solutions.iter().chain(vector_solutions.iter()) {
        ndtree_archive.try_insert(solution);
        vector_archive.try_insert(solution);
        linkedlist_archive.try_insert(solution);
    }

    // Extract final Pareto fronts
    let ndtree_front = extract_pareto_front(&ndtree_archive);
    let vector_front = extract_pareto_front(&vector_archive);
    let linkedlist_front = extract_pareto_front(&linkedlist_archive);

    println!("Combined replay results:");
    println!("  nd-tree archive: {} solutions", ndtree_front.len());
    println!("  vector archive: {} solutions", vector_front.len());
    println!(
        "  linked list archive: {} solutions",
        linkedlist_front.len()
    );

    // All should converge to the same Pareto front
    assert_eq!(
        ndtree_front, vector_front,
        "nd-tree and vector archives produced different results from combined trace"
    );
    assert_eq!(
        ndtree_front, linkedlist_front,
        "nd-tree and linked list archives produced different results from combined trace"
    );
}

#[test]
fn test_cross_trace_comparison_4d() {
    println!("Loaded traces:");

    // Load both traces
    let ndtree_solutions = load_trace_as_solutions::<4>("tests/data/pls_trace_4d_ndtree.json", 50);
    let vector_solutions = load_trace_as_solutions::<4>("tests/data/pls_trace_4d_vector.json", 50);

    println!("  nd-tree: {} solutions", ndtree_solutions.len());
    println!("  vector: {} solutions", vector_solutions.len());

    // Process nd-tree trace in all three archives
    let mut ndtree_archive_with_ndtree_trace =
        NdTreeSolutionSet::<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4>::new("ndtree_ndtree");
    let mut vector_archive_with_ndtree_trace =
        VecSolutionSet::<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4>::new("vector_ndtree");
    let mut linkedlist_archive_with_ndtree_trace = LinkedListSolutionSet::<
        BitsetEncodedSolution<ProblemBitset<4>, 4>,
        4,
    >::new("linkedlist_ndtree");
    for solution in &ndtree_solutions {
        ndtree_archive_with_ndtree_trace.try_insert(solution);
        vector_archive_with_ndtree_trace.try_insert(solution);
        linkedlist_archive_with_ndtree_trace.try_insert(solution);
    }

    // Process vector trace in all three archives
    let mut ndtree_archive_with_vector_trace =
        NdTreeSolutionSet::<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4>::new("ndtree_vector");
    let mut vector_archive_with_vector_trace =
        VecSolutionSet::<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4>::new("vector_vector");
    let mut linkedlist_archive_with_vector_trace = LinkedListSolutionSet::<
        BitsetEncodedSolution<ProblemBitset<4>, 4>,
        4,
    >::new("linkedlist_vector");
    for solution in &vector_solutions {
        ndtree_archive_with_vector_trace.try_insert(solution);
        vector_archive_with_vector_trace.try_insert(solution);
        linkedlist_archive_with_vector_trace.try_insert(solution);
    }

    // Extract fronts
    let ndtree_ndtree = extract_pareto_front(&ndtree_archive_with_ndtree_trace);
    let vector_ndtree = extract_pareto_front(&vector_archive_with_ndtree_trace);
    let linkedlist_ndtree = extract_pareto_front(&linkedlist_archive_with_ndtree_trace);
    let ndtree_vector = extract_pareto_front(&ndtree_archive_with_vector_trace);
    let vector_vector = extract_pareto_front(&vector_archive_with_vector_trace);
    let linkedlist_vector = extract_pareto_front(&linkedlist_archive_with_vector_trace);

    println!("\nReplay results:");
    println!(
        "  nd-tree trace → nd-tree archive: {} solutions",
        ndtree_ndtree.len()
    );
    println!(
        "  nd-tree trace → vector archive: {} solutions",
        vector_ndtree.len()
    );
    println!(
        "  nd-tree trace → linked list archive: {} solutions",
        linkedlist_ndtree.len()
    );
    println!(
        "  vector trace → nd-tree archive: {} solutions",
        ndtree_vector.len()
    );
    println!(
        "  vector trace → vector archive: {} solutions",
        vector_vector.len()
    );
    println!(
        "  vector trace → linked list archive: {} solutions",
        linkedlist_vector.len()
    );

    // Same trace should produce same result regardless of archive type
    assert_eq!(
        ndtree_ndtree, vector_ndtree,
        "nd-tree trace produced different results: nd-tree vs vector archives"
    );
    assert_eq!(
        ndtree_ndtree, linkedlist_ndtree,
        "nd-tree trace produced different results: nd-tree vs linked list archives"
    );
    assert_eq!(
        ndtree_vector, vector_vector,
        "vector trace produced different results: nd-tree vs vector archives"
    );
    assert_eq!(
        ndtree_vector, linkedlist_vector,
        "vector trace produced different results: nd-tree vs linked list archives"
    );
}
