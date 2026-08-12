//! Benchmarks `.dzn` parsing against the real random-clouds SIMS instances
//! that motivated the original performance investigation (`lagos_nigeria` at
//! sizes 100/300/500 — up to ~107 MB of raw text, ~101k universe fragments).
//!
//! Fixture `.dzn.gz` files are committed to the repo (the raw `.dzn` is
//! gitignored — see `.gitignore`), so they're decompressed once per instance
//! outside the timed section and the in-memory bytes are reused across
//! iterations. This isolates parser throughput from disk I/O / decompression.

use std::io::Read as _;
use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("publication-data")
        .join("satellite-data")
        .join("instances_random_clouds")
}

/// Decompresses `{name}.dzn.gz` into memory. Panics (failing the benchmark
/// loudly) if the fixture isn't present, rather than silently skipping —
/// these files are committed, so their absence means something is wrong
/// with the checkout.
fn load_fixture_bytes(name: &str) -> Vec<u8> {
    let gz_path = fixture_dir().join(format!("{name}.dzn.gz"));
    let file = std::fs::File::open(&gz_path)
        .unwrap_or_else(|e| panic!("missing benchmark fixture {}: {e}", gz_path.display()));
    let mut decoder = flate2::read::GzDecoder::new(file);
    let mut bytes = Vec::new();
    decoder
        .read_to_end(&mut bytes)
        .unwrap_or_else(|e| panic!("failed to decompress {}: {e}", gz_path.display()));
    bytes
}

fn bench_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse_dzn_bytes");
    // These instances are large enough that even a handful of iterations
    // takes real wall-clock time; keep sample counts modest so `cargo bench`
    // finishes in a reasonable window.
    group.sample_size(10);

    for name in [
        "lagos_nigeria_100",
        "lagos_nigeria_300",
        "lagos_nigeria_500",
    ] {
        let bytes = load_fixture_bytes(name);
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(name), &bytes, |b, bytes| {
            b.iter(|| sims_dzn::parse_dzn_bytes(std::hint::black_box(bytes)).unwrap());
        });
    }

    group.finish();
}

criterion_group!(benches, bench_parse);
criterion_main!(benches);
