use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nd_tree::nd_tree::{NDTree, Solution};
use rand::prelude::*;

const VMAX: u64 = 10_000;
const N_POINTS: usize = 100_000;
const QUALITIES: [(usize, f64); 4] = [(1, 0.5), (2, 0.25), (3, 0.1), (4, 0.05)];

fn generate_points<const D: usize>(n: usize, v_max: u64, eps: f64) -> Vec<Solution<D>> {
    let mut rng = rand::rng();
    let v_max_f = v_max as f64;
    let v2 = v_max_f * v_max_f;
    let min_sum = (1.0 - eps) * v2;
    let max_sum = v2;
    let mut points = Vec::with_capacity(n);
    while points.len() < n {
        let mut y = [0u64; D];
        for yk in y.iter_mut() {
            *yk = rng.random_range(0..=v_max);
        }
        let sum: f64 = y.iter().map(|&yk| (v_max_f - yk as f64).powi(2)).sum();
        if sum >= min_sum && sum <= max_sum {
            points.push(Solution { objectives: y });
        }
    }
    points
}

fn bench_nd_tree(c: &mut Criterion) {
    for p in 2..=6 {
        for &(quality, eps) in &QUALITIES {
            let bench_id = BenchmarkId::new(format!("nd_tree_p{}_q{}", p, quality), N_POINTS);

            // Measurement time based on dimensionality for large dataset (100k points)
            // These are more conservative since we're dealing with 100k points vs the smaller datasets
            let measurement_time = match p {
                2 => std::time::Duration::from_secs(11), // 2D with 100k points
                3 => std::time::Duration::from_secs(18), // 3D with 100k points
                4 => std::time::Duration::from_secs(33), // 4D with 100k points
                5 => std::time::Duration::from_secs(63), // 5D with 100k points
                6 => std::time::Duration::from_secs(123), // 6D with 100k points
                _ => std::time::Duration::from_secs(8),
            };

            c.benchmark_group(format!("nd_tree_p{}_q{}", p, quality))
                .measurement_time(measurement_time)
                .bench_with_input(bench_id, &(), |b, _| {
                    b.iter(|| match p {
                        2 => {
                            let points = generate_points::<2>(N_POINTS, VMAX, eps);
                            let mut tree = NDTree::<32, 2, 4>::new();
                            for pt in points {
                                tree.update(pt);
                            }
                        }
                        3 => {
                            let points = generate_points::<3>(N_POINTS, VMAX, eps);
                            let mut tree = NDTree::<32, 3, 4>::new();
                            for pt in points {
                                tree.update(pt);
                            }
                        }
                        4 => {
                            let points = generate_points::<4>(N_POINTS, VMAX, eps);
                            let mut tree = NDTree::<32, 4, 4>::new();
                            for pt in points {
                                tree.update(pt);
                            }
                        }
                        5 => {
                            let points = generate_points::<5>(N_POINTS, VMAX, eps);
                            let mut tree = NDTree::<32, 5, 4>::new();
                            for pt in points {
                                tree.update(pt);
                            }
                        }
                        6 => {
                            let points = generate_points::<6>(N_POINTS, VMAX, eps);
                            let mut tree = NDTree::<32, 6, 4>::new();
                            for pt in points {
                                tree.update(pt);
                            }
                        }
                        _ => unreachable!(),
                    });
                });
        }
    }
}

criterion_group!(benches, bench_nd_tree);
criterion_main!(benches);
