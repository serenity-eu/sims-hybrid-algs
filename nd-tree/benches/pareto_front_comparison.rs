#![feature(adt_const_params)]
#![feature(linked_list_cursors)]
#![feature(linked_list_retain)]
#![expect(
    clippy::cast_precision_loss,
    reason = "Legacy code style, extensive refactor needed"
)]

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration, Throughput,
};
use nd_tree::nd_tree::Solution; // Import the native Solution type
use pareto::{HasObjectives, MoSolution, ParetoFront};
use rand::prelude::*;

mod fronts;
use fronts::linkedlist_pareto_front::LinkedListParetoFront;
use fronts::nd_tree_pareto_front::NdTreeParetoFront;
use fronts::vec_pareto_front::VecParetoFront;

/// Test solution for benchmarking Vec and `LinkedList` implementations
#[derive(Debug, Clone, PartialEq)]
struct BenchSolution<const D: usize> {
    objectives: [u64; D],
    id: u64,
}

impl<const D: usize> HasObjectives<D> for BenchSolution<D> {
    fn objectives(&self) -> &[u64; D] {
        &self.objectives
    }
}

impl<const D: usize> MoSolution<D> for BenchSolution<D> {}

/// Benchmark configuration parameters
const VMAX: u64 = 10_000;
const SMALL_N: usize = 1_000;
const MEDIUM_N: usize = 5_000; // Reduced for higher dimensions
const LARGE_N: usize = 20_000; // Reduced for higher dimensions

/// Configuration for different dimensional benchmarks
const DIM_2D_SIZES: [usize; 10] = [
    1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 8_000, 9_000, 10_000,
];
const DIM_3D_SIZES: [usize; 10] = [
    1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 8_000, 9_000, 10_000,
];
const DIM_4D_SIZES: [usize; 10] = [
    1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 8_000, 9_000, 10_000,
];
const DIM_5D_SIZES: [usize; 10] = [
    1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 8_000, 9_000, 10_000,
];
// const DIM_3D_SIZES: [usize; 2] = [1_000, 3_000];
// const DIM_4D_SIZES: [usize; 2] = [500, 1_000];
// const DIM_5D_SIZES: [usize; 5] = [250, 500, 750, 1000, 1250];

/// Generate test points using the same approach as the existing benchmark
fn generate_pareto_solutions<const D: usize>(
    n: usize,
    v_max: u64,
    eps: f64,
) -> Vec<BenchSolution<D>> {
    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducible results
    let v_max_f = v_max as f64;
    let v2 = v_max_f * v_max_f;
    let min_sum = (1.0 - eps) * v2;
    let max_sum = v2;
    let mut solutions = Vec::with_capacity(n);

    while solutions.len() < n {
        let mut objectives = [0u64; D];
        for obj in &mut objectives {
            *obj = rng.gen_range(0..=v_max);
        }

        let sum: f64 = objectives
            .iter()
            .map(|&obj| (v_max_f - obj as f64).powi(2))
            .sum();

        if sum >= min_sum && sum <= max_sum {
            solutions.push(BenchSolution {
                objectives,
                id: solutions.len() as u64,
            });
        }
    }

    solutions
}

/// Generate native Solution types for ND-Tree benchmarking
fn generate_nd_tree_solutions<const D: usize>(n: usize, v_max: u64, eps: f64) -> Vec<Solution<D>> {
    let mut rng = StdRng::seed_from_u64(42); // Same seed for consistency
    let v_max_f = v_max as f64;
    let v2 = v_max_f * v_max_f;
    let min_sum = (1.0 - eps) * v2;
    let max_sum = v2;
    let mut solutions = Vec::with_capacity(n);

    while solutions.len() < n {
        let mut objectives = [0u64; D];
        for obj in &mut objectives {
            *obj = rng.gen_range(0..=v_max);
        }

        let sum: f64 = objectives
            .iter()
            .map(|&obj| (v_max_f - obj as f64).powi(2))
            .sum();

        if sum >= min_sum && sum <= max_sum {
            solutions.push(Solution { objectives });
        }
    }

    solutions
}

/// Generate random solutions with uniform distribution (for stress testing)
fn generate_random_solutions<const D: usize>(n: usize, v_max: u64) -> Vec<BenchSolution<D>> {
    let mut rng = StdRng::seed_from_u64(123); // Different seed for different pattern
    let mut solutions = Vec::with_capacity(n);

    for i in 0..n {
        let mut objectives = [0u64; D];
        for obj in &mut objectives {
            *obj = rng.gen_range(0..=v_max);
        }
        solutions.push(BenchSolution {
            objectives,
            id: i as u64,
        });
    }

    solutions
}

/// Generate random nd-tree solutions with uniform distribution
fn generate_random_nd_tree_solutions<const D: usize>(n: usize, v_max: u64) -> Vec<Solution<D>> {
    let mut rng = StdRng::seed_from_u64(123); // Same seed for consistency
    let mut solutions = Vec::with_capacity(n);

    for _i in 0..n {
        let mut objectives = [0u64; D];
        for obj in &mut objectives {
            *obj = rng.gen_range(0..=v_max);
        }
        solutions.push(Solution { objectives });
    }

    solutions
}

/// Configure plotting for consistent scales across benchmarks
fn configure_group_for_comparison(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) {
    // Set consistent plotting configuration - only what's not in .criterion.toml
    let plot_config =
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Linear);
    group.plot_config(plot_config);
}

/// Benchmark insertion performance for 2D problems
fn bench_insertion_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("insertion_2d");
    configure_group_for_comparison(&mut group);

    for size in DIM_2D_SIZES {
        // Adjust measurement time based on specific size requirements from benchmark analysis
        let measurement_time = match size {
            1000..=7000 => std::time::Duration::from_secs(8), // Default time works fine
            8000 => std::time::Duration::from_secs(9), // ndtree needs 5.7s, linkedlist needs 5.2s
            9000 => std::time::Duration::from_secs(12), // ndtree needs 7.7s, linkedlist needs 5.9s
            10000 => std::time::Duration::from_secs(15), // ndtree needs 7.2s, vec needs 5.5s, linkedlist needs 6.7s
            _ => std::time::Duration::from_secs(5),
        };
        group.measurement_time(measurement_time);
        let solutions = generate_pareto_solutions::<2>(size, VMAX, 0.25);
        let nd_tree_solutions = generate_nd_tree_solutions::<2>(size, VMAX, 0.25);

        group.throughput(Throughput::Elements(size as u64));

        // Benchmark ND-Tree
        group.bench_with_input(
            BenchmarkId::new("ndtree", size),
            &nd_tree_solutions,
            |b, solutions| {
                b.iter(|| {
                    let mut pf = NdTreeParetoFront::<Solution<2>, 32, 2, 4>::new("bench");
                    for solution in solutions {
                        pf.try_insert(solution);
                    }
                    pf.len() // Return something to prevent optimization
                });
            },
        );

        // Benchmark Vec-based
        group.bench_with_input(BenchmarkId::new("vec", size), &solutions, |b, solutions| {
            b.iter(|| {
                let mut pf = VecParetoFront::<BenchSolution<2>, 2>::new("bench");
                for solution in solutions {
                    pf.try_insert(solution);
                }
                pf.len()
            });
        });

        // Benchmark LinkedList-based
        group.bench_with_input(
            BenchmarkId::new("linkedlist", size),
            &solutions,
            |b, solutions| {
                b.iter(|| {
                    let mut pf = LinkedListParetoFront::<BenchSolution<2>, 2>::new("bench");
                    for solution in solutions {
                        pf.try_insert(solution);
                    }
                    pf.len()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark insertion performance for 3D problems
fn bench_insertion_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("insertion_3d");
    configure_group_for_comparison(&mut group);

    for size in DIM_3D_SIZES {
        // Adjust measurement time based on specific size requirements from benchmark analysis
        let measurement_time = match size {
            3000 => std::time::Duration::from_secs(8),
            4000 => std::time::Duration::from_secs(9),
            5000 | 6000 => std::time::Duration::from_secs(10),
            7000 | 8000 => std::time::Duration::from_secs(14),
            10000 => std::time::Duration::from_secs(24),
            _ => std::time::Duration::from_secs(5),
        };
        group.measurement_time(measurement_time);
        let solutions = generate_pareto_solutions::<3>(size, VMAX, 0.1);
        let nd_tree_solutions = generate_nd_tree_solutions::<3>(size, VMAX, 0.1);

        group.throughput(Throughput::Elements(size as u64));

        // Benchmark ND-Tree
        group.bench_with_input(
            BenchmarkId::new("ndtree", size),
            &nd_tree_solutions,
            |b, solutions| {
                b.iter(|| {
                    let mut pf = NdTreeParetoFront::<Solution<3>, 32, 3, 4>::new("bench");
                    for solution in solutions {
                        pf.try_insert(solution);
                    }
                    pf.len()
                });
            },
        );

        // Benchmark Vec-based
        group.bench_with_input(BenchmarkId::new("vec", size), &solutions, |b, solutions| {
            b.iter(|| {
                let mut pf = VecParetoFront::<BenchSolution<3>, 3>::new("bench");
                for solution in solutions {
                    pf.try_insert(solution);
                }
                pf.len()
            });
        });

        // Benchmark LinkedList-based
        group.bench_with_input(
            BenchmarkId::new("linkedlist", size),
            &solutions,
            |b, solutions| {
                b.iter(|| {
                    let mut pf = LinkedListParetoFront::<BenchSolution<3>, 3>::new("bench");
                    for solution in solutions {
                        pf.try_insert(solution);
                    }
                    pf.len()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark insertion performance for 4D problems
fn bench_insertion_4d(c: &mut Criterion) {
    let mut group = c.benchmark_group("insertion_4d");
    configure_group_for_comparison(&mut group);
    group.sample_size(50); // Override: Reduce sample size for higher dimensions

    for size in DIM_4D_SIZES {
        // Adjust measurement time based on specific size requirements from benchmark analysis (50 samples)
        let measurement_time = match size {
            2000 | 3000 => std::time::Duration::from_secs(18),
            5000 => std::time::Duration::from_secs(20),
            6000 | 7000 => std::time::Duration::from_secs(25),
            8000 => std::time::Duration::from_secs(29),
            9000 => std::time::Duration::from_secs(36),
            10000 => std::time::Duration::from_secs(70),
            _ => std::time::Duration::from_secs(5),
        };
        group.measurement_time(measurement_time);
        let solutions = generate_pareto_solutions::<4>(size, VMAX, 0.05);
        let nd_tree_solutions = generate_nd_tree_solutions::<4>(size, VMAX, 0.05);

        group.throughput(Throughput::Elements(size as u64));

        // Benchmark ND-Tree
        group.bench_with_input(
            BenchmarkId::new("ndtree", size),
            &nd_tree_solutions,
            |b, solutions| {
                b.iter(|| {
                    let mut pf = NdTreeParetoFront::<Solution<4>, 32, 4, 4>::new("bench");
                    for solution in solutions {
                        pf.try_insert(solution);
                    }
                    pf.len()
                });
            },
        );

        // Benchmark Vec-based
        group.bench_with_input(BenchmarkId::new("vec", size), &solutions, |b, solutions| {
            b.iter(|| {
                let mut pf = VecParetoFront::<BenchSolution<4>, 4>::new("bench");
                for solution in solutions {
                    pf.try_insert(solution);
                }
                pf.len()
            });
        });

        // Benchmark LinkedList-based
        group.bench_with_input(
            BenchmarkId::new("linkedlist", size),
            &solutions,
            |b, solutions| {
                b.iter(|| {
                    let mut pf = LinkedListParetoFront::<BenchSolution<4>, 4>::new("bench");
                    for solution in solutions {
                        pf.try_insert(solution);
                    }
                    pf.len()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark insertion performance for 5D problems
fn bench_insertion_5d(c: &mut Criterion) {
    let mut group = c.benchmark_group("insertion_5d");
    configure_group_for_comparison(&mut group);
    group.sample_size(30); // Override: Reduce sample size further for 5D

    for size in DIM_5D_SIZES {
        // Adjust measurement time based on specific size requirements from benchmark analysis (30 samples)
        let measurement_time = match size {
            1000 | 2000 => std::time::Duration::from_secs(6),
            3000 => std::time::Duration::from_secs(7),
            4000 => std::time::Duration::from_secs(8),
            5000 => std::time::Duration::from_secs(10),
            6000 => std::time::Duration::from_secs(15),
            7000 => std::time::Duration::from_secs(24),
            8000 => std::time::Duration::from_secs(28),
            9000 => std::time::Duration::from_secs(33),
            10000 => std::time::Duration::from_secs(43),
            _ => std::time::Duration::from_secs(5),
        };
        group.measurement_time(measurement_time);
        let solutions = generate_pareto_solutions::<5>(size, VMAX, 0.02);
        let nd_tree_solutions = generate_nd_tree_solutions::<5>(size, VMAX, 0.02);

        group.throughput(Throughput::Elements(size as u64));

        // Benchmark ND-Tree
        group.bench_with_input(
            BenchmarkId::new("ndtree", size),
            &nd_tree_solutions,
            |b, solutions| {
                b.iter(|| {
                    let mut pf = NdTreeParetoFront::<Solution<5>, 32, 5, 4>::new("bench");
                    for solution in solutions {
                        pf.try_insert(solution);
                    }
                    pf.len()
                });
            },
        );

        // Benchmark Vec-based
        group.bench_with_input(BenchmarkId::new("vec", size), &solutions, |b, solutions| {
            b.iter(|| {
                let mut pf = VecParetoFront::<BenchSolution<5>, 5>::new("bench");
                for solution in solutions {
                    pf.try_insert(solution);
                }
                pf.len()
            });
        });

        // Benchmark LinkedList-based
        group.bench_with_input(
            BenchmarkId::new("linkedlist", size),
            &solutions,
            |b, solutions| {
                b.iter(|| {
                    let mut pf = LinkedListParetoFront::<BenchSolution<5>, 5>::new("bench");
                    for solution in solutions {
                        pf.try_insert(solution);
                    }
                    pf.len()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark performance with different data patterns (2D only for simplicity)
fn bench_data_patterns_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_patterns_2d");
    configure_group_for_comparison(&mut group);
    let size = MEDIUM_N;

    // Test with quality-based data (existing approach)
    let quality_solutions = generate_pareto_solutions::<2>(size, VMAX, 0.25);
    let quality_nd_tree_solutions = generate_nd_tree_solutions::<2>(size, VMAX, 0.25);

    // Test with random data
    let random_solutions = generate_random_solutions::<2>(size, VMAX);
    let random_nd_tree_solutions = generate_random_nd_tree_solutions::<2>(size, VMAX);

    let test_cases = [
        ("quality", quality_solutions, quality_nd_tree_solutions),
        ("random", random_solutions, random_nd_tree_solutions),
    ];

    for (dataset_name, solutions, nd_tree_solutions) in test_cases {
        group.throughput(Throughput::Elements(size as u64));

        // Benchmark ND-Tree
        group.bench_with_input(
            BenchmarkId::new(format!("ndtree_{dataset_name}"), size),
            &nd_tree_solutions,
            |b, solutions| {
                b.iter(|| {
                    let mut pf = NdTreeParetoFront::<Solution<2>, 32, 2, 4>::new("bench");
                    for solution in solutions {
                        pf.try_insert(solution);
                    }
                    pf.len()
                });
            },
        );

        // Benchmark Vec-based
        group.bench_with_input(
            BenchmarkId::new(format!("vec_{dataset_name}"), size),
            &solutions,
            |b, solutions| {
                b.iter(|| {
                    let mut pf = VecParetoFront::<BenchSolution<2>, 2>::new("bench");
                    for solution in solutions {
                        pf.try_insert(solution);
                    }
                    pf.len()
                });
            },
        );

        // Benchmark LinkedList-based
        group.bench_with_input(
            BenchmarkId::new(format!("linkedlist_{dataset_name}"), size),
            &solutions,
            |b, solutions| {
                b.iter(|| {
                    let mut pf = LinkedListParetoFront::<BenchSolution<2>, 2>::new("bench");
                    for solution in solutions {
                        pf.try_insert(solution);
                    }
                    pf.len()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark iteration performance
fn bench_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("iteration");
    configure_group_for_comparison(&mut group);
    let size = MEDIUM_N;
    let solutions = generate_pareto_solutions::<2>(size, VMAX, 0.25);
    let nd_tree_solutions = generate_nd_tree_solutions::<2>(size, VMAX, 0.25);

    // Pre-build the Pareto fronts
    let mut ndtree_pf = NdTreeParetoFront::<Solution<2>, 32, 2, 4>::new("bench");
    let mut vec_pf = VecParetoFront::<BenchSolution<2>, 2>::new("bench");
    let mut linkedlist_pf = LinkedListParetoFront::<BenchSolution<2>, 2>::new("bench");

    for solution in &nd_tree_solutions {
        ndtree_pf.try_insert(solution);
    }

    for solution in &solutions {
        vec_pf.try_insert(solution);
        linkedlist_pf.try_insert(solution);
    }

    let final_size = vec_pf.len(); // Should be approximately same for all implementations
    group.throughput(Throughput::Elements(final_size as u64));

    // Benchmark iteration
    group.bench_function("ndtree", |b| {
        b.iter(|| {
            let mut count = 0;
            for _solution in ndtree_pf.iter() {
                count += 1;
            }
            count
        });
    });

    group.bench_function("vec", |b| {
        b.iter(|| {
            let mut count = 0;
            for _solution in vec_pf.iter() {
                count += 1;
            }
            count
        });
    });

    group.bench_function("linkedlist", |b| {
        b.iter(|| {
            let mut count = 0;
            for _solution in linkedlist_pf.iter() {
                count += 1;
            }
            count
        });
    });

    group.finish();
}

/// Benchmark vec vs linkedlist comparison (excluding ndtree for simplicity)
fn bench_vec_vs_linkedlist(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec_vs_linkedlist");
    configure_group_for_comparison(&mut group);

    for size in [SMALL_N, MEDIUM_N, LARGE_N] {
        let solutions = generate_pareto_solutions::<2>(size, VMAX, 0.25);

        group.throughput(Throughput::Elements(size as u64));

        // Benchmark Vec-based
        group.bench_with_input(BenchmarkId::new("vec", size), &solutions, |b, solutions| {
            b.iter(|| {
                let mut pf = VecParetoFront::<BenchSolution<2>, 2>::new("bench");
                for solution in solutions {
                    pf.try_insert(solution);
                }
                pf.len()
            });
        });

        // Benchmark LinkedList-based
        group.bench_with_input(
            BenchmarkId::new("linkedlist", size),
            &solutions,
            |b, solutions| {
                b.iter(|| {
                    let mut pf = LinkedListParetoFront::<BenchSolution<2>, 2>::new("bench");
                    for solution in solutions {
                        pf.try_insert(solution);
                    }
                    pf.len()
                });
            },
        );
    }

    group.finish();
}

/// Direct comparison benchmark - all methods with same data, optimized for statistical comparison
fn bench_direct_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("direct_comparison");
    configure_group_for_comparison(&mut group);
    // Uses global defaults from .criterion.toml

    let size = 2000; // Medium size for good statistics
    let solutions = generate_pareto_solutions::<2>(size, VMAX, 0.25);
    let nd_tree_solutions = generate_nd_tree_solutions::<2>(size, VMAX, 0.25);

    group.throughput(Throughput::Elements(size as u64));

    // All three methods with same input size for direct comparison
    group.bench_function("ndtree_2d", |b| {
        b.iter(|| {
            let mut pf = NdTreeParetoFront::<Solution<2>, 32, 2, 4>::new("bench");
            for solution in &nd_tree_solutions {
                pf.try_insert(solution);
            }
            pf.len()
        });
    });

    group.bench_function("vec_2d", |b| {
        b.iter(|| {
            let mut pf = VecParetoFront::<BenchSolution<2>, 2>::new("bench");
            for solution in &solutions {
                pf.try_insert(solution);
            }
            pf.len()
        });
    });

    group.bench_function("linkedlist_2d", |b| {
        b.iter(|| {
            let mut pf = LinkedListParetoFront::<BenchSolution<2>, 2>::new("bench");
            for solution in &solutions {
                pf.try_insert(solution);
            }
            pf.len()
        });
    });

    group.finish();
}

/// Comprehensive comparison benchmark designed for consistent plotting
fn bench_comprehensive_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("comprehensive_comparison");
    configure_group_for_comparison(&mut group);

    // Use fixed dataset sizes for consistent comparison
    let test_sizes = [500, 1000, 2000, 3000];

    for size in test_sizes {
        let solutions = generate_pareto_solutions::<2>(size, VMAX, 0.25);
        let nd_tree_solutions = generate_nd_tree_solutions::<2>(size, VMAX, 0.25);

        group.throughput(Throughput::Elements(size as u64));

        // All three implementations with same input size
        group.bench_with_input(
            BenchmarkId::new("ndtree", size),
            &nd_tree_solutions,
            |b, solutions| {
                b.iter(|| {
                    let mut pf = NdTreeParetoFront::<Solution<2>, 32, 2, 4>::new("bench");
                    for solution in solutions {
                        pf.try_insert(solution);
                    }
                    pf.len()
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("vec", size), &solutions, |b, solutions| {
            b.iter(|| {
                let mut pf = VecParetoFront::<BenchSolution<2>, 2>::new("bench");
                for solution in solutions {
                    pf.try_insert(solution);
                }
                pf.len()
            });
        });

        group.bench_with_input(
            BenchmarkId::new("linkedlist", size),
            &solutions,
            |b, solutions| {
                b.iter(|| {
                    let mut pf = LinkedListParetoFront::<BenchSolution<2>, 2>::new("bench");
                    for solution in solutions {
                        pf.try_insert(solution);
                    }
                    pf.len()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark for ND-Tree Pareto Front in 2D
fn bench_nd_tree_pareto_front_2d(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default();
    let mut group = c.benchmark_group("ParetoFront_2D_NDTree");
    group.plot_config(plot_config);

    for &size in &DIM_2D_SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("NDTree", size), &size, |b, &s| {
            let solutions = generate_nd_tree_solutions::<2>(s, VMAX, 0.1);
            b.iter(|| {
                let mut pf = NdTreeParetoFront::<Solution<2>, 32, 2, 4>::new("bench");
                for sol in &solutions {
                    pf.try_insert(sol);
                }
            });
        });
    }
    group.finish();
}

/// Benchmark for ND-Tree Pareto Front in 3D
fn bench_nd_tree_pareto_front_3d(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default();
    let mut group = c.benchmark_group("ParetoFront_3D_NDTree");
    group.plot_config(plot_config);

    for &size in &DIM_3D_SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("NDTree", size), &size, |b, &s| {
            let solutions = generate_nd_tree_solutions::<3>(s, VMAX, 0.1);
            b.iter(|| {
                let mut pf = NdTreeParetoFront::<Solution<3>, 32, 3, 4>::new("bench");
                for sol in &solutions {
                    pf.try_insert(sol);
                }
            });
        });
    }
    group.finish();
}

/// Benchmark for ND-Tree Pareto Front in 4D
fn bench_nd_tree_pareto_front_4d(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default();
    let mut group = c.benchmark_group("ParetoFront_4D_NDTree");
    group.plot_config(plot_config);

    for &size in &DIM_4D_SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("NDTree", size), &size, |b, &s| {
            let solutions = generate_nd_tree_solutions::<4>(s, VMAX, 0.1);
            b.iter(|| {
                let mut pf = NdTreeParetoFront::<Solution<4>, 32, 4, 4>::new("bench");
                for sol in &solutions {
                    pf.try_insert(sol);
                }
            });
        });
    }
    group.finish();
}

/// Benchmark for ND-Tree Pareto Front in 5D
fn bench_nd_tree_pareto_front_5d(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default();
    let mut group = c.benchmark_group("ParetoFront_5D_NDTree");
    group.plot_config(plot_config);

    for &size in &DIM_5D_SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("NDTree", size), &size, |b, &s| {
            let solutions = generate_nd_tree_solutions::<5>(s, VMAX, 0.1);
            b.iter(|| {
                let mut pf = NdTreeParetoFront::<Solution<5>, 32, 5, 4>::new("bench");
                for sol in &solutions {
                    pf.try_insert(sol);
                }
            });
        });
    }
    group.finish();
}

/// Benchmark for Vec Pareto Front in 2D
fn bench_vec_pareto_front_2d(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default();
    let mut group = c.benchmark_group("ParetoFront_2D_Vec");
    group.plot_config(plot_config);

    for &size in &DIM_2D_SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("Vec", size), &size, |b, &s| {
            let solutions = generate_pareto_solutions::<2>(s, VMAX, 0.1);
            b.iter(|| {
                let mut pf = VecParetoFront::<BenchSolution<2>, 2>::new("bench");
                for sol in &solutions {
                    pf.try_insert(sol);
                }
            });
        });
    }
    group.finish();
}

/// Compare implementations: Vec, `LinkedList`, and ND-Tree
fn compare_implementations(c: &mut Criterion) {
    let mut group = c.benchmark_group("ImplementationComparison");
    let solutions = generate_pareto_solutions::<2>(SMALL_N, VMAX, 0.1);
    let nd_solutions = generate_nd_tree_solutions::<2>(SMALL_N, VMAX, 0.1);

    group.bench_function("VecParetoFront", |b| {
        b.iter(|| {
            let mut vec_pf = VecParetoFront::<BenchSolution<2>, 2>::new("bench");
            for sol in &solutions {
                vec_pf.try_insert(sol);
            }
        });
    });

    group.bench_function("LinkedListParetoFront", |b| {
        b.iter(|| {
            let mut list_pf = LinkedListParetoFront::<BenchSolution<2>, 2>::new("bench");
            for sol in &solutions {
                list_pf.try_insert(sol);
            }
        });
    });

    group.bench_function("NdTreeParetoFront", |b| {
        b.iter(|| {
            let mut ndtree_pf = NdTreeParetoFront::<Solution<2>, 32, 2, 4>::new("bench");
            for sol in &nd_solutions {
                ndtree_pf.try_insert(sol);
            }
        });
    });

    group.finish();
}

/// Benchmark insertion strategies: `TryInsert` vs `InsertUnchecked`
fn bench_insertion_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("InsertionStrategies");
    let solutions = generate_pareto_solutions::<2>(MEDIUM_N, VMAX, 0.1);
    let nd_solutions = generate_nd_tree_solutions::<2>(MEDIUM_N, VMAX, 0.1);

    group.bench_function("Vec_TryInsert", |b| {
        b.iter(|| {
            let mut pf = VecParetoFront::<BenchSolution<2>, 2>::new("bench");
            for sol in &solutions {
                pf.try_insert(sol);
            }
        });
    });

    group.bench_function("Vec_InsertUnchecked", |b| {
        b.iter(|| {
            let mut pf = VecParetoFront::<BenchSolution<2>, 2>::new("bench");
            for sol in &solutions {
                pf.insert_unchecked(sol);
            }
        });
    });

    group.bench_function("NDTree_TryInsert", |b| {
        b.iter(|| {
            let mut pf = NdTreeParetoFront::<Solution<2>, 32, 2, 4>::new("bench");
            for sol in &nd_solutions {
                pf.try_insert(sol);
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_insertion_2d,
    bench_insertion_3d,
    bench_insertion_4d,
    bench_insertion_5d,
    bench_data_patterns_2d,
    bench_iteration,
    bench_vec_vs_linkedlist,
    bench_direct_comparison,
    bench_comprehensive_comparison,
    bench_nd_tree_pareto_front_2d,
    bench_nd_tree_pareto_front_3d,
    bench_nd_tree_pareto_front_4d,
    bench_nd_tree_pareto_front_5d,
    bench_vec_pareto_front_2d,
    compare_implementations,
    bench_insertion_strategies
);
criterion_main!(benches);
