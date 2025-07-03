use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nd_tree::nd_tree::{NDTree, Solution};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn generate_solutions<const D: usize>(n: usize, v_max: u64) -> Vec<Solution<D>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut solutions = Vec::with_capacity(n);
    for _ in 0..n {
        let mut objectives = [0u64; D];
        for obj in &mut objectives {
            *obj = rng.random_range(0..v_max);
        }
        solutions.push(Solution { objectives });
    }
    solutions
}

macro_rules! bench_dim {
    ($name:ident, $dim:expr) => {
        fn $name(c: &mut Criterion) {
            let mut group = c.benchmark_group(format!("ND-Tree Benchmark {}D", $dim));

            let sizes = [100, 1_000, 10_000];

            for size in sizes.iter() {
                let solutions = generate_solutions::<$dim>(*size, 10_000);
                group.bench_with_input(
                    BenchmarkId::new(format!("{}D", $dim), size),
                    &solutions,
                    |b, s| {
                        b.iter(|| {
                            let mut tree = NDTree::<Solution<$dim>, 32, $dim, 4>::new();
                            for sol in s {
                                tree.update(sol.clone());
                            }
                        });
                    },
                );
            }

            group.finish();
        }
    };
}

bench_dim!(bench_nd_tree_2d, 2);
bench_dim!(bench_nd_tree_3d, 3);
bench_dim!(bench_nd_tree_4d, 4);
bench_dim!(bench_nd_tree_5d, 5);
bench_dim!(bench_nd_tree_6d, 6);

criterion_group!(
    benches,
    bench_nd_tree_2d,
    bench_nd_tree_3d,
    bench_nd_tree_4d,
    bench_nd_tree_5d,
    bench_nd_tree_6d
);
criterion_main!(benches);
