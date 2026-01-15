use criterion::{Criterion, criterion_group, criterion_main};
use fixedbitset::FixedBitSet;
use pls::problem::SIMSProblemInstanceRaw;
use pls::problem_bitset::ProblemBitset;
use rand::prelude::*;

fn load_raw_instance(path: &str) -> SIMSProblemInstanceRaw {
    SIMSProblemInstanceRaw::from_minizinc_datafile(path)
}

fn generate_covering_solutions<const D: usize>(
    pb: &ProblemBitset<D>,
    batch: usize,
    rng: &mut impl Rng,
) -> Vec<Vec<usize>> {
    let num_images = pb.num_images();
    let universe_size = pb.universe_size;
    let mut solutions = Vec::with_capacity(batch);

    for _ in 0..batch {
        // Greedy random covering
        let mut covered = FixedBitSet::with_capacity(universe_size);
        let mut indices: Vec<usize> = (0..num_images).collect();
        indices.shuffle(rng);
        let mut sol = Vec::new();
        for &img_idx in &indices {
            let before = covered.count_ones(..);
            covered.union_with(&pb.images[img_idx]);
            if covered.count_ones(..) > before {
                sol.push(img_idx);
            }
            if covered.count_ones(..) == universe_size {
                break;
            }
        }
        // If not covered, fallback to all images
        if covered.count_ones(..) < universe_size {
            sol = (0..num_images).collect();
        }
        solutions.push(sol);
    }
    solutions
}

fn bench_cover_check(c: &mut Criterion) {
    for path in [
        "tests/data/lagos_nigeria_50.dzn",
        "tests/data/lagos_nigeria_145.dzn",
    ] {
        let raw = load_raw_instance(path);
        let universe_size = raw.universe_size;
        let batch = 10_000;
        let mut rng = SmallRng::seed_from_u64(42);

        // Bitset implementation
        let pb = ProblemBitset::from_raw_with_objectives(
            &raw,
            [
                pls::objectives::ObjectiveType::TotalCost,
                pls::objectives::ObjectiveType::CloudyArea,
            ],
        );
        let random_solutions = generate_covering_solutions(&pb, batch, &mut rng);
        c.bench_function(&format!("bitset_cover_check_{path}"), |b| {
            b.iter(|| {
                for sol in &random_solutions {
                    let mut covered = FixedBitSet::with_capacity(universe_size);
                    for &img_idx in sol {
                        covered.union_with(&pb.images[img_idx]);
                    }
                    assert!(covered.contains_all_in_range(..universe_size));
                }
            });
        });

        // Default implementation
        let problem = pls::problem_bitset::ProblemBitset::<2>::from_raw_with_objectives(
            &raw,
            [
                pls::objectives::ObjectiveType::TotalCost,
                pls::objectives::ObjectiveType::CloudyArea,
            ],
        );

        c.bench_function(&format!("default_cover_check_{path}"), |b| {
            b.iter(|| {
                for sol in &random_solutions {
                    let mut covered = fixedbitset::FixedBitSet::with_capacity(universe_size);
                    for &img_idx in sol {
                        covered.union_with(&problem.images[img_idx]);
                    }
                    assert!(covered.is_full());
                }
            });
        });
    }
}

criterion_group!(benches, bench_cover_check);
criterion_main!(benches);
