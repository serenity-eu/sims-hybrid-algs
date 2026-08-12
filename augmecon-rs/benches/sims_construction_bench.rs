//! Benchmarks turning a parsed `.dzn` instance into a ready-to-solve
//! biobjective (min-cost, cloud-coverage) MILP model — the two-stage
//! pipeline `SimsInstance::from_raw_refs` (build the augmecon-facing
//! instance, including the `image_clouds` relation) followed by
//! `create_sims_problem_with_objectives` (variables + constraints +
//! objectives) — against the real `lagos_nigeria` random-clouds fixtures
//! that motivated the original performance investigation.
//!
//! `.dzn.gz` fixtures are committed to the repo; parsing (via the `sims-dzn`
//! dev-dependency) happens once per instance outside the timed section so
//! this bench isolates construction cost from parse cost — see
//! `sims-dzn/benches/parse_bench.rs` for the parser's own benchmark.

use std::collections::HashSet;
use std::io::Read as _;
use std::path::PathBuf;

use augmecon::sims_problem::{create_sims_problem_with_objectives, SimsInstance, SimsObjective};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use sims_dzn::RawSimsData;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("publication-data")
        .join("satellite-data")
        .join("instances_random_clouds")
}

fn load_fixture(name: &str) -> RawSimsData {
    let gz_path = fixture_dir().join(format!("{name}.dzn.gz"));
    let file = std::fs::File::open(&gz_path)
        .unwrap_or_else(|e| panic!("missing benchmark fixture {}: {e}", gz_path.display()));
    let mut decoder = flate2::read::GzDecoder::new(file);
    let mut bytes = Vec::new();
    decoder
        .read_to_end(&mut bytes)
        .unwrap_or_else(|e| panic!("failed to decompress {}: {e}", gz_path.display()));
    sims_dzn::parse_dzn_bytes(&bytes)
        .unwrap_or_else(|e| panic!("failed to parse fixture {name}: {e}"))
}

/// The objective set used by the production pseudo-solution generation path
/// (`generate_pseudo.py`): min-cost + cloud-coverage only, which skips the
/// resolution/incidence-angle variables entirely.
fn cost_and_cloud_objectives() -> HashSet<SimsObjective> {
    [SimsObjective::MinCost, SimsObjective::CloudCoverage]
        .into_iter()
        .collect()
}

fn bench_from_raw_refs(c: &mut Criterion) {
    let mut group = c.benchmark_group("SimsInstance::from_raw_refs");
    group.sample_size(10);

    for name in [
        "lagos_nigeria_100",
        "lagos_nigeria_300",
        "lagos_nigeria_500",
    ] {
        let data = load_fixture(name);
        group.bench_with_input(BenchmarkId::from_parameter(name), &data, |b, data| {
            b.iter(|| {
                std::hint::black_box(SimsInstance::from_raw_refs(
                    &data.images,
                    &data.clouds,
                    &data.costs,
                    &data.areas,
                    &data.resolution,
                    &data.incidence_angle,
                    data.universe,
                    data.max_cloud_area,
                ))
            });
        });
    }

    group.finish();
}

fn bench_full_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_construction_pipeline");
    group.sample_size(10);
    let objectives = cost_and_cloud_objectives();

    for name in [
        "lagos_nigeria_100",
        "lagos_nigeria_300",
        "lagos_nigeria_500",
    ] {
        let data = load_fixture(name);
        group.bench_with_input(BenchmarkId::from_parameter(name), &data, |b, data| {
            b.iter(|| {
                let instance = SimsInstance::from_raw_refs(
                    &data.images,
                    &data.clouds,
                    &data.costs,
                    &data.areas,
                    &data.resolution,
                    &data.incidence_angle,
                    data.universe,
                    data.max_cloud_area,
                );
                std::hint::black_box(create_sims_problem_with_objectives(
                    &instance,
                    Some(&objectives),
                ))
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_from_raw_refs, bench_full_construction);
criterion_main!(benches);
