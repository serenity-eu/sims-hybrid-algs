use criterion::{criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use std::collections::HashSet;
use std::hint::black_box;

fn unique_count_sort_dedup(vec: &mut Vec<u32>) -> usize {
    vec.sort_unstable();
    vec.dedup();
    vec.len()
}

fn unique_count_hashset(vec: &[u32]) -> usize {
    let set: HashSet<_> = vec.iter().collect();
    set.len()
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand::rng();
    let sizes = [10, 100, 1_000, 10_000, 100_000];

    for &size in &sizes {
        let vec: Vec<u32> = (0..size).map(|_| rng.random()).collect();

        c.bench_function(&format!("sort_dedup_{size}"), |b| {
            b.iter(|| unique_count_sort_dedup(black_box(&mut vec.clone())));
        });

        c.bench_function(&format!("hashset_{size}"), |b| {
            b.iter(|| unique_count_hashset(black_box(&vec)));
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
