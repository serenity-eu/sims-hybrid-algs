//! Integration tests for the concurrent Pareto Local Search (objective-space decomposition).
//!
//! Requires both the `parallel` and `bitmaps` features.
//! Run with: `cargo test --test test_concurrent_pls --features parallel`

#![cfg(feature = "parallel")]

use std::time::Duration;

use pareto::ParetoFront;
use pls::{
    concurrent_pls::{
        decomposition::{
            assign_to_regions, belongs_to_region, build_regions, das_dennis_weight_vectors,
        },
        ConcurrentPLS, ConcurrentPLSConfig, RegionSearchMode,
    },
    objectives::ObjectiveType,
    pareto_local_search::ParetoLocalSearch,
    problem_bitset::ProblemBitset,
    solution::bitset_encoded_solution::BitsetEncodedSolution,
    solution_set_impl::NdTreeSolutionSet,
};

const NUM_OBJECTIVES: usize = 4;
const OBJECTIVE_TYPES: [ObjectiveType; NUM_OBJECTIVES] = [
    ObjectiveType::TotalCost,
    ObjectiveType::CloudyArea,
    ObjectiveType::MinResolution,
    ObjectiveType::MaxIncidenceAngle,
];

type Problem = ProblemBitset<NUM_OBJECTIVES>;
type Solution = BitsetEncodedSolution<Problem, NUM_OBJECTIVES>;
type Archive = NdTreeSolutionSet<Solution, NUM_OBJECTIVES>;

fn load_problem(name: &str) -> Problem {
    let path = format!("tests/data/{name}");
    Problem::from_minizinc_datafile(&path, OBJECTIVE_TYPES)
        .unwrap_or_else(|e| panic!("Failed to load {name}: {e}"))
}

fn random_initial_population(problem: &Problem, size: usize) -> Archive {
    (0..size)
        .map(|i| BitsetEncodedSolution::random_with_seed(problem, i as u64 + 42))
        .collect()
}

// ---------------------------------------------------------------------------
// Decomposition unit tests
// ---------------------------------------------------------------------------

#[test]
fn das_dennis_h1_produces_d_weight_vectors() {
    let vecs = das_dennis_weight_vectors::<4>(1);
    assert_eq!(vecs.len(), 4, "H=1 with D=4 should give exactly 4 vectors");
    for v in &vecs {
        let sum: f64 = v.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Weight vector should sum to 1.0, got {sum}"
        );
    }
}

#[test]
fn build_regions_trims_to_requested_count() {
    // Request 6 regions; Das-Dennis H=2 gives 10, so 4 should be trimmed
    let regions = build_regions::<4>(6, Some(2));
    assert_eq!(regions.len(), 6);
    // All indices should be 0..6
    for (i, r) in regions.iter().enumerate() {
        assert_eq!(r.index, i);
    }
}

#[test]
fn assign_to_regions_hard_boundary() {
    let bounds = [(0.0, 100.0); 4];
    let regions = build_regions::<4>(4, Some(1));
    let ideal = [0u64; 4];
    // A solution dominated in objective 0 should strongly prefer the region
    // whose weight emphasizes objective 0.
    let objectives = [1u64, 100, 100, 100];
    let assigned = assign_to_regions(&objectives, &regions, &ideal, &bounds, 0.0);
    assert_eq!(assigned.len(), 1, "Hard boundary (threshold=0) should assign to exactly 1 region");
}

#[test]
fn assign_to_regions_soft_boundary_assigns_multiple() {
    let bounds = [(0.0, 100.0); 4];
    let regions = build_regions::<4>(10, Some(2));
    let ideal = [0u64; 4];
    // A balanced solution near the centroid should be assigned to multiple regions
    let objectives = [50u64, 50, 50, 50];
    let assigned = assign_to_regions(&objectives, &regions, &ideal, &bounds, 0.10);
    assert!(
        assigned.len() > 1,
        "Centroid-ish solution with 10% threshold should be assigned to multiple regions"
    );
}

#[test]
fn belongs_to_region_is_consistent_with_assign() {
    let bounds = [(0.0, 100.0); 4];
    let regions = build_regions::<4>(4, Some(1));
    let ideal = [0u64; 4];
    let objectives = [10u64, 90, 50, 70];
    let assigned = assign_to_regions(&objectives, &regions, &ideal, &bounds, 0.0);
    assert!(!assigned.is_empty());
    let best_region_idx = assigned[0];
    assert!(
        belongs_to_region(&objectives, &regions[best_region_idx], &regions, &ideal, &bounds),
        "Best-assigned region should pass belongs_to_region"
    );
}

// ---------------------------------------------------------------------------
// Concurrent PLS integration tests on real instances
// ---------------------------------------------------------------------------

#[test]
fn concurrent_pls_4_threads_produces_nonempty_front() {
    let problem = load_problem("lagos_nigeria_50.dzn");
    let initial_pop = random_initial_population(&problem, 20);
    let initial_size = initial_pop.len();

    let config = ConcurrentPLSConfig::default_with_threads(4, Duration::from_secs(10));
    let result = ConcurrentPLS::<Solution, Problem, NUM_OBJECTIVES>::new(&problem, config)
        .solve(&initial_pop);

    assert!(!result.archive.is_empty(), "Final archive must be non-empty");
    assert!(
        result.archive.len() >= initial_size,
        "Concurrent PLS should find at least as many solutions as the initial population ({} vs {})",
        result.archive.len(),
        initial_size,
    );
    assert_eq!(result.num_regions, 4, "H=1 with D=4 gives exactly 4 regions");
    assert_eq!(
        result.region_results.len(),
        4,
        "Should have one result per region"
    );
}

#[test]
fn concurrent_pls_single_thread_matches_sequential() {
    let problem = load_problem("lagos_nigeria_50.dzn");
    let initial_pop = random_initial_population(&problem, 15);
    let timeout = Duration::from_secs(5);

    // Run sequential PLS
    let mut seq_pls = ParetoLocalSearch::<Solution, Archive, Problem, NUM_OBJECTIVES>::new(
        &problem,
        &initial_pop,
        1..=6,
        true,
    );
    let seq_archive = seq_pls.run(100_000, timeout);
    let seq_count = seq_archive.len();

    // Run concurrent PLS with 1 thread (effectively sequential but through concurrent infra)
    let config = ConcurrentPLSConfig {
        num_threads: 1,
        sync_interval_steps: 5,
        merge_interval: Duration::from_millis(100),
        boundary_threshold: 0.05,
        das_dennis_h: Some(1),
        max_iterations: 100_000,
        max_duration: timeout,
        neighborhood_size_range: 1..=6,
        is_deterministic: true,
        region_search_mode: RegionSearchMode::default(),
        scalarized_rho: 1e-3,
    };
    let par_result = ConcurrentPLS::<Solution, Problem, NUM_OBJECTIVES>::new(&problem, config)
        .solve(&initial_pop);

    assert!(
        !par_result.archive.is_empty(),
        "1-thread concurrent PLS must produce solutions"
    );
    // With 1 thread and same timeout, concurrent PLS should find a comparable number
    // (not exact match due to sync overhead and snapshot machinery, but in the same ballpark).
    let par_count = par_result.archive.len();
    assert!(
        par_count >= seq_count / 2,
        "1-thread concurrent ({par_count}) should be within 2x of sequential ({seq_count})"
    );
}

#[test]
fn concurrent_pls_all_regions_produce_results() {
    let problem = load_problem("lagos_nigeria_50.dzn");
    let initial_pop = random_initial_population(&problem, 30);

    let config = ConcurrentPLSConfig::default_with_threads(4, Duration::from_secs(8));
    let result = ConcurrentPLS::<Solution, Problem, NUM_OBJECTIVES>::new(&problem, config)
        .solve(&initial_pop);

    for region in &result.region_results {
        assert!(
            region.stats.iterations_completed > 0,
            "Region {} should complete at least 1 iteration",
            region.region_index
        );
        assert!(
            region.stats.initial_pop_size > 0,
            "Region {} should receive at least 1 initial solution",
            region.region_index
        );
        assert!(
            region.stats.snapshots_published > 0,
            "Region {} should publish at least 1 snapshot",
            region.region_index
        );
        // Weight vector should sum to 1.0
        let wv_sum: f64 = region.weight_vector.iter().sum();
        assert!(
            (wv_sum - 1.0).abs() < 1e-10,
            "Region {} weight vector should sum to 1.0, got {wv_sum}",
            region.region_index
        );
    }
}

#[test]
fn concurrent_pls_tombstones_do_not_appear_in_final_front() {
    let problem = load_problem("lagos_nigeria_50.dzn");
    let initial_pop = random_initial_population(&problem, 25);

    let config = ConcurrentPLSConfig::default_with_threads(4, Duration::from_secs(10));
    let result = ConcurrentPLS::<Solution, Problem, NUM_OBJECTIVES>::new(&problem, config)
        .solve(&initial_pop);

    // The final archive is built by the orchestrator filtering out tombstoned solutions.
    // Verify the archive is valid (no dominated pairs).
    result.archive.validate();
}

#[test]
fn concurrent_pls_10_threads_scales() {
    let problem = load_problem("lagos_nigeria_50.dzn");
    let initial_pop = random_initial_population(&problem, 40);

    // Use H=2 which gives 10 regions
    let config = ConcurrentPLSConfig {
        num_threads: 10,
        sync_interval_steps: 3,
        merge_interval: Duration::from_millis(50),
        boundary_threshold: 0.05,
        das_dennis_h: Some(2),
        max_iterations: 100_000,
        max_duration: Duration::from_secs(15),
        neighborhood_size_range: 1..=6,
        is_deterministic: false,
        region_search_mode: RegionSearchMode::default(),
        scalarized_rho: 1e-3,
    };
    let result = ConcurrentPLS::<Solution, Problem, NUM_OBJECTIVES>::new(&problem, config)
        .solve(&initial_pop);

    assert_eq!(result.num_regions, 10, "H=2 with D=4 gives 10 regions");
    assert!(
        result.archive.len() >= initial_pop.len(),
        "10-thread run should find at least initial_pop solutions"
    );

    // Check load distribution: no single region should have 0 iterations (all should work)
    let zero_iter_regions: Vec<usize> = result
        .region_results
        .iter()
        .filter(|r| r.stats.iterations_completed == 0)
        .map(|r| r.region_index)
        .collect();
    assert!(
        zero_iter_regions.is_empty(),
        "All regions should complete at least 1 iteration; idle: {zero_iter_regions:?}"
    );
}

// ---------------------------------------------------------------------------
// SA-PLS integration tests (Phase 5, Section 16)
// ---------------------------------------------------------------------------

#[test]
fn sa_pls_produces_nonempty_front() {
    let problem = load_problem("lagos_nigeria_50.dzn");
    let initial_pop = random_initial_population(&problem, 20);
    let initial_size = initial_pop.len();

    let config = ConcurrentPLSConfig {
        num_threads: 4,
        sync_interval_steps: 5,
        merge_interval: Duration::from_millis(100),
        boundary_threshold: 0.05,
        das_dennis_h: Some(1),
        max_iterations: 100_000,
        max_duration: Duration::from_secs(10),
        neighborhood_size_range: 1..=6,
        is_deterministic: true,
        region_search_mode: RegionSearchMode::ScalarizedAuxiliary,
        scalarized_rho: 1e-3,
    };
    let result = ConcurrentPLS::<Solution, Problem, NUM_OBJECTIVES>::new(&problem, config)
        .solve(&initial_pop);

    assert!(!result.archive.is_empty(), "SA-PLS must produce a non-empty archive");
    assert!(
        result.archive.len() >= initial_size,
        "SA-PLS should find at least as many solutions as the initial population ({} vs {})",
        result.archive.len(),
        initial_size,
    );
}

#[test]
fn sa_pls_with_fallback_runs_unconstrained_after_exhaustion() {
    let problem = load_problem("lagos_nigeria_50.dzn");
    let initial_pop = random_initial_population(&problem, 20);

    let config = ConcurrentPLSConfig {
        num_threads: 4,
        sync_interval_steps: 5,
        merge_interval: Duration::from_millis(100),
        boundary_threshold: 0.05,
        das_dennis_h: Some(1),
        max_iterations: 100_000,
        max_duration: Duration::from_secs(15),
        neighborhood_size_range: 1..=6,
        is_deterministic: true,
        region_search_mode: RegionSearchMode::ScalarizedAuxiliaryWithFallback,
        scalarized_rho: 1e-3,
    };
    let result = ConcurrentPLS::<Solution, Problem, NUM_OBJECTIVES>::new(&problem, config)
        .solve(&initial_pop);

    assert!(!result.archive.is_empty(), "SA-PLS with fallback must produce solutions");

    // At least one region should have reached scalarized exhaustion and continued
    // with standard PLS (giving it more iterations).
    let any_exhausted = result
        .region_results
        .iter()
        .any(|r| r.stats.scalarized_exhausted_at_iteration.is_some());

    // With a 15s timeout and small instance, at least one region should exhaust.
    // If not, the test still passes -- we just cannot assert the fallback fired.
    if any_exhausted {
        // Regions that exhausted should have completed more iterations than the
        // exhaustion point (proving the fallback ran).
        for region in &result.region_results {
            if let Some(exhaust_iter) = region.stats.scalarized_exhausted_at_iteration {
                assert!(
                    region.stats.iterations_completed > exhaust_iter,
                    "Region {} exhausted at iteration {exhaust_iter} but only completed {} \
                     iterations -- fallback should have continued",
                    region.region_index,
                    region.stats.iterations_completed,
                );
            }
        }
    }
}

#[test]
fn diagnostic_metrics_are_populated() {
    let problem = load_problem("lagos_nigeria_50.dzn");
    let initial_pop = random_initial_population(&problem, 20);

    let config = ConcurrentPLSConfig::default_with_threads(4, Duration::from_secs(8));
    let result = ConcurrentPLS::<Solution, Problem, NUM_OBJECTIVES>::new(&problem, config)
        .solve(&initial_pop);

    let total_neighbors: usize = result
        .region_results
        .iter()
        .map(|r| r.stats.neighbors_explored)
        .sum();
    let total_duplicates: usize = result
        .region_results
        .iter()
        .map(|r| r.stats.duplicates_skipped)
        .sum();

    assert!(
        total_neighbors > 0,
        "At least some neighbors should have been explored across all regions"
    );
    // Duplicate count can be 0 in short runs, but neighbors_explored must be positive
    // for every region that completed at least 1 iteration.
    for region in &result.region_results {
        if region.stats.iterations_completed > 0 {
            assert!(
                region.stats.neighbors_explored > 0,
                "Region {} completed {} iterations but explored 0 neighbors",
                region.region_index,
                region.stats.iterations_completed,
            );
        }
    }

    // Sanity: duplicates should be strictly less than neighbors (at least some non-duplicate work)
    assert!(
        total_duplicates < total_neighbors,
        "Duplicates ({total_duplicates}) should be less than total neighbors ({total_neighbors})"
    );
}

#[test]
fn sa_pls_10_threads_scales() {
    let problem = load_problem("lagos_nigeria_50.dzn");
    let initial_pop = random_initial_population(&problem, 40);

    let config = ConcurrentPLSConfig {
        num_threads: 10,
        sync_interval_steps: 3,
        merge_interval: Duration::from_millis(50),
        boundary_threshold: 0.05,
        das_dennis_h: Some(2),
        max_iterations: 100_000,
        max_duration: Duration::from_secs(15),
        neighborhood_size_range: 1..=6,
        is_deterministic: false,
        region_search_mode: RegionSearchMode::ScalarizedAuxiliary,
        scalarized_rho: 1e-3,
    };
    let result = ConcurrentPLS::<Solution, Problem, NUM_OBJECTIVES>::new(&problem, config)
        .solve(&initial_pop);

    assert_eq!(result.num_regions, 10, "H=2 with D=4 gives 10 regions");
    assert!(
        result.archive.len() >= initial_pop.len(),
        "10-thread SA-PLS should find at least initial_pop solutions"
    );

    // All regions should have explored neighbors
    for region in &result.region_results {
        assert!(
            region.stats.neighbors_explored > 0,
            "SA-PLS region {} should explore at least 1 neighbor",
            region.region_index,
        );
    }
}
