//! Benchmarks for comparing different `ObjectiveTracker` implementations
//!
//! This benchmark suite tests the performance of different tracking strategies
//! for incremental objective evaluation during local search operations.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fixedbitset::FixedBitSet;
use pls::objectives::ObjectiveType;
use pls::problem::{Problem, SIMSProblemInstanceRaw};
use pls::solution::{SIMSCore, SIMSModifiable};
use pls::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;
use pls::trackers::{ObjectiveTracker, StandardTracker, StandardTrackerArray, TrackerCollection};
use rand::SeedableRng;
use rand::Rng;
use std::hint::black_box;
use std::time::Duration;

/// Create a test problem with specified dimensions
fn create_test_problem<const D: usize>(
    num_images: usize,
    universe_size: usize,
    avg_coverage: usize,
) -> Problem<BitsetEncodedSolution<D>, D> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    
    // Generate random image coverage
    let images: Vec<Vec<usize>> = (0..num_images)
        .map(|_| {
            let coverage_size = (avg_coverage / 2) + rng.random_range(0..avg_coverage);
            #[allow(clippy::cast_precision_loss)]
            let mut coverage: Vec<usize> = (0..universe_size)
                .filter(|_| rng.random_bool((coverage_size as f64) / (universe_size as f64)))
                .collect();
            coverage.sort_unstable();
            coverage.into_iter().map(|x| x + 1).collect() // 1-indexed
        })
        .collect();
    
    // Generate random cloud coverage (subset of image coverage)
    let clouds: Vec<Vec<usize>> = images
        .iter()
        .map(|img| {
            img.iter()
                .filter(|_| rng.random_bool(0.3)) // 30% cloud coverage
                .copied()
                .collect()
        })
        .collect();
    
    let objectives = std::array::from_fn(|i| match i % 4 {
        0 => ObjectiveType::TotalCost,
        1 => ObjectiveType::CloudyArea,
        2 => ObjectiveType::MinResolution,
        _ => ObjectiveType::MaxIncidenceAngle,
    });
    
    Problem::from_raw_with_objectives(
        SIMSProblemInstanceRaw {
            name: "benchmark".to_string(),
            num_images,
            universe_size,
            images,
            costs: (1..=num_images as u64).collect(),
            clouds,
            areas: vec![100; universe_size],
            max_cloud_area: (universe_size * 100) as u64,
            resolution: (0..num_images).map(|i| (i as u64 % 10) + 1).collect(),
            incidence_angle: (0..num_images).map(|i| (i as u64 % 45) + 1).collect(),
        },
        objectives,
    )
    .expect("Failed to create test problem")
}

/// Create a solution with random selection
fn create_test_solution<const D: usize>(
    problem: &Problem<BitsetEncodedSolution<D>, D>,
    selection_rate: f64,
) -> BitsetEncodedSolution<D> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    let mut selected_images = FixedBitSet::with_capacity(problem.images.len());
    
    for i in 0..problem.images.len() {
        if rng.random_bool(selection_rate) {
            selected_images.insert(i);
        }
    }
    
    let mut solution = BitsetEncodedSolution {
        selected_images,
        objectives: [0; D],
        timestamp: Duration::from_secs(0),
        trackers: StandardTrackerArray::new(problem),
    };
    
    solution.recalculate_objectives(problem);
    solution
}

/// Benchmark: `peek_delta` operation (read-only delta calculation)
fn bench_peek_delta(c: &mut Criterion) {
    let mut group = c.benchmark_group("peek_delta");
    
    for &size in &[50, 100, 500] {
        let problem = create_test_problem::<4>(size, size * 2, size / 5);
        let solution = create_test_solution(&problem, 0.3);
        
        group.bench_with_input(
            BenchmarkId::new("CloudyArea", size),
            &size,
            |b, _| {
                let tracker = solution.trackers().get(1); // CloudyArea
                b.iter(|| {
                    let delta = tracker.peek_delta(
                        black_box(10),
                        black_box(false),
                        black_box(&problem),
                        black_box(&solution),
                    );
                    black_box(delta);
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("TotalCost", size),
            &size,
            |b, _| {
                let tracker = solution.trackers().get(0); // TotalCost
                b.iter(|| {
                    let delta = tracker.peek_delta(
                        black_box(10),
                        black_box(false),
                        black_box(&problem),
                        black_box(&solution),
                    );
                    black_box(delta);
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("MinResolution", size),
            &size,
            |b, _| {
                let tracker = solution.trackers().get(2); // MinResolution
                b.iter(|| {
                    let delta = tracker.peek_delta(
                        black_box(10),
                        black_box(false),
                        black_box(&problem),
                        black_box(&solution),
                    );
                    black_box(delta);
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("MaxIncidenceAngle", size),
            &size,
            |b, _| {
                let tracker = solution.trackers().get(3); // MaxIncidenceAngle
                b.iter(|| {
                    let delta = tracker.peek_delta(
                        black_box(10),
                        black_box(false),
                        black_box(&problem),
                        black_box(&solution),
                    );
                    black_box(delta);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark: apply operation (state update)
fn bench_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply");
    
    for &size in &[50, 100, 500] {
        let problem = create_test_problem::<4>(size, size * 2, size / 5);
        
        group.bench_with_input(
            BenchmarkId::new("CloudyArea", size),
            &size,
            |b, _| {
                b.iter_batched(
                    || {
                        let solution = create_test_solution(&problem, 0.3);
                        (solution.trackers().get(1).clone(), false)
                    },
                    |(mut tracker, is_removing)| {
                        tracker.apply(
                            black_box(10),
                            black_box(is_removing),
                            black_box(&problem),
                        );
                        black_box(tracker);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("TotalCost", size),
            &size,
            |b, _| {
                b.iter_batched(
                    || {
                        let solution = create_test_solution(&problem, 0.3);
                        (solution.trackers().get(0).clone(), false)
                    },
                    |(mut tracker, is_removing)| {
                        tracker.apply(
                            black_box(10),
                            black_box(is_removing),
                            black_box(&problem),
                        );
                        black_box(tracker);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark: Full local search step (peek + apply for all objectives)
fn bench_local_search_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("local_search_step");
    
    for &size in &[50, 100, 500] {
        let problem = create_test_problem::<4>(size, size * 2, size / 5);
        
        group.bench_with_input(
            BenchmarkId::new("add_image", size),
            &size,
            |b, _| {
                b.iter_batched(
                    || create_test_solution(&problem, 0.3),
                    |mut solution| {
                        // Simulate evaluating a move
                        let image_idx = 10;
                        
                        // Calculate deltas for all objectives
                        let mut deltas = [0i64; 4];
                        for (i, delta) in deltas.iter_mut().enumerate() {
                            *delta = solution.trackers().get(i).peek_delta(
                                image_idx,
                                false,
                                &problem,
                                &solution,
                            );
                        }
                        
                        // Apply the move
                        for i in 0..4 {
                            solution.trackers_mut().get_mut(i).apply(
                                image_idx,
                                false,
                                &problem,
                            );
                        }
                        
                        black_box((solution, deltas));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("remove_image", size),
            &size,
            |b, _| {
                b.iter_batched(
                    || create_test_solution(&problem, 0.3),
                    |mut solution| {
                        let image_idx = 10;
                        
                        // Calculate deltas for all objectives
                        let mut deltas = [0i64; 4];
                        for (i, delta) in deltas.iter_mut().enumerate() {
                            *delta = solution.trackers().get(i).peek_delta(
                                image_idx,
                                true,
                                &problem,
                                &solution,
                            );
                        }
                        
                        // Apply the move
                        for i in 0..4 {
                            solution.trackers_mut().get_mut(i).apply(
                                image_idx,
                                true,
                                &problem,
                            );
                        }
                        
                        black_box((solution, deltas));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark: Neighborhood evaluation (many `peek_delta` calls)
fn bench_neighborhood_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighborhood_evaluation");
    
    for &size in &[50, 100, 500] {
        let problem = create_test_problem::<4>(size, size * 2, size / 5);
        
        group.bench_with_input(
            BenchmarkId::new("evaluate_all_adds", size),
            &size,
            |b, _| {
                let solution = create_test_solution(&problem, 0.3);
                let unselected: Vec<usize> = solution.unselected_images().collect();
                
                b.iter(|| {
                    let mut best_delta = i64::MAX;
                    
                    for &img_idx in &unselected {
                        let mut total_delta = 0i64;
                        for i in 0..4 {
                            total_delta += solution.trackers().get(i).peek_delta(
                                img_idx,
                                false,
                                &problem,
                                &solution,
                            );
                        }
                        if total_delta < best_delta {
                            best_delta = total_delta;
                        }
                    }
                    
                    black_box(best_delta);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark: Tracker initialization
fn bench_tracker_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("tracker_initialization");
    
    for &size in &[50, 100, 500, 1000] {
        let problem = create_test_problem::<4>(size, size * 2, size / 5);
        
        group.bench_with_input(
            BenchmarkId::new("create_trackers", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let trackers = StandardTrackerArray::new(black_box(&problem));
                    black_box(trackers);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark: Memory access patterns
fn bench_memory_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_access");
    
    let problem = create_test_problem::<4>(100, 200, 20);
    let solution = create_test_solution(&problem, 0.3);
    
    group.bench_function("sequential_tracker_access", |b| {
        b.iter(|| {
            for i in 0..4 {
                let tracker: &StandardTracker = solution.trackers().get(i);
                black_box(<StandardTracker as ObjectiveTracker<4>>::value(tracker));
            }
        });
    });
    
    group.bench_function("tracker_clone", |b| {
        let tracker = solution.trackers().get(1);
        b.iter(|| {
            let cloned = black_box(tracker).clone();
            black_box(cloned);
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_peek_delta,
    bench_apply,
    bench_local_search_step,
    bench_neighborhood_evaluation,
    bench_tracker_initialization,
    bench_memory_access,
);

criterion_main!(benches);
