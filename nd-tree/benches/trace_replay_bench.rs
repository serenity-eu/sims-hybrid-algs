//! Benchmark Pareto archive implementations using real PLS trace data.
//!
//! Loads objective vectors from binary `.bin` files (u64 LE, D values per solution)
//! and replays them into Vec, LinkedList, and ND-Tree archives.
//!
//! Trace files are extracted `objectives.bin` from optimization trace archives,
//! renamed to `<instance_name>.bin` and placed in `benches/data/`.
//!
//! To prepare a trace file from a `.tar.gz` archive:
//! ```bash
//! tar xzf trace.tar.gz objectives.bin
//! mv objectives.bin benches/data/<instance_name>.bin
//! ```

#![feature(adt_const_params)]
#![feature(linked_list_cursors)]
#![feature(linked_list_retain)]
#![expect(
    clippy::cast_precision_loss,
    reason = "Benchmark code, precision loss acceptable"
)]

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration, Throughput,
};
use nd_tree::nd_tree::Solution;
use pareto::{HasObjectives, MoSolution, ParetoFront};
use std::fs;
use std::path::PathBuf;

mod fronts;
use fronts::linkedlist_pareto_front::LinkedListParetoFront;
use fronts::nd_tree_pareto_front::NdTreeParetoFront;
use fronts::vec_pareto_front::VecParetoFront;

/// Test solution for benchmarking Vec and LinkedList implementations
#[derive(Debug, Clone, PartialEq)]
struct BenchSolution<const D: usize> {
    objectives: [u64; D],
}

impl<const D: usize> HasObjectives<D> for BenchSolution<D> {
    fn objectives(&self) -> &[u64; D] {
        &self.objectives
    }
}

impl<const D: usize> MoSolution<D> for BenchSolution<D> {}

/// Load objectives from a binary file (u64 LE, D values per solution).
///
/// Returns a pair of solution vectors: one for Vec/LinkedList benchmarks,
/// one for ND-Tree benchmarks (different concrete types required).
fn load_trace_objectives<const D: usize>(
    filename: &str,
) -> (Vec<BenchSolution<D>>, Vec<Solution<D>>) {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("benches/data");
    path.push(filename);

    let data = fs::read(&path)
        .unwrap_or_else(|e| panic!("Failed to read trace file {}: {}", path.display(), e));

    let bytes_per_solution = D * std::mem::size_of::<u64>();
    assert!(
        data.len() % bytes_per_solution == 0,
        "File size {} is not a multiple of {} bytes ({} objectives x 8 bytes)",
        data.len(),
        bytes_per_solution,
        D
    );

    let num_solutions = data.len() / bytes_per_solution;
    let mut bench_solutions = Vec::with_capacity(num_solutions);
    let mut nd_tree_solutions = Vec::with_capacity(num_solutions);

    for i in 0..num_solutions {
        let mut objectives = [0u64; D];
        for (j, obj) in objectives.iter_mut().enumerate() {
            let offset = (i * D + j) * 8;
            *obj = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        }

        bench_solutions.push(BenchSolution { objectives });
        nd_tree_solutions.push(Solution { objectives });
    }

    (bench_solutions, nd_tree_solutions)
}

/// Discover all `.bin` trace files in benches/data/
fn discover_trace_files() -> Vec<String> {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.push("benches/data");

    let mut files: Vec<String> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Failed to read benches/data directory: {}", e))
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().into_string().ok()?;
            if name.ends_with(".bin") {
                Some(name)
            } else {
                None
            }
        })
        .collect();

    files.sort();
    files
}

/// Returns the instance name from a filename (strips .bin extension)
fn instance_name(filename: &str) -> &str {
    filename.strip_suffix(".bin").unwrap_or(filename)
}

/// Benchmark all three archive implementations on each trace file,
/// replaying the full trace (all solutions).
fn bench_trace_replay_full(c: &mut Criterion) {
    let files = discover_trace_files();
    if files.is_empty() {
        eprintln!("No .bin trace files found in benches/data/. Skipping trace replay benchmarks.");
        return;
    }

    let mut group = c.benchmark_group("trace_replay_4d");
    let plot_config = PlotConfiguration::default().summary_scale(criterion::AxisScale::Linear);
    group.plot_config(plot_config);
    group.sample_size(20);

    for filename in &files {
        let (bench_solutions, nd_tree_solutions) = load_trace_objectives::<4>(filename);
        let name = instance_name(filename);
        let n = bench_solutions.len();

        eprintln!(
            "Trace '{}': {} solutions loaded ({} bytes/solution)",
            name,
            n,
            4 * 8
        );

        group.throughput(Throughput::Elements(n as u64));

        // ND-Tree
        group.bench_with_input(
            BenchmarkId::new("ndtree", name),
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

        // Vec
        group.bench_with_input(
            BenchmarkId::new("vec", name),
            &bench_solutions,
            |b, solutions| {
                b.iter(|| {
                    let mut pf = VecParetoFront::<BenchSolution<4>, 4>::new("bench");
                    for solution in solutions {
                        pf.try_insert(solution);
                    }
                    pf.len()
                });
            },
        );

        // LinkedList
        group.bench_with_input(
            BenchmarkId::new("linkedlist", name),
            &bench_solutions,
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

/// Benchmark with increasing prefix sizes from a single trace,
/// showing how each archive scales as the number of insertions grows.
fn bench_trace_replay_scaling(c: &mut Criterion) {
    let files = discover_trace_files();
    if files.is_empty() {
        return;
    }

    // Use the first (or largest) trace file for scaling analysis
    let filename = &files[0];
    let (bench_solutions, nd_tree_solutions) = load_trace_objectives::<4>(filename);
    let name = instance_name(filename);
    let total = bench_solutions.len();

    // Pick prefix sizes: 10%, 25%, 50%, 75%, 100% of trace
    let fractions = [10, 25, 50, 75, 100];
    let sizes: Vec<usize> = fractions
        .iter()
        .map(|&pct| (total * pct / 100).max(1))
        .collect();

    let mut group = c.benchmark_group(format!("trace_scaling_{name}"));
    let plot_config = PlotConfiguration::default().summary_scale(criterion::AxisScale::Linear);
    group.plot_config(plot_config);
    group.sample_size(20);

    for &size in &sizes {
        let bench_prefix: Vec<_> = bench_solutions[..size].to_vec();
        let nd_tree_prefix: Vec<_> = nd_tree_solutions[..size].to_vec();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("ndtree", size),
            &nd_tree_prefix,
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

        group.bench_with_input(
            BenchmarkId::new("vec", size),
            &bench_prefix,
            |b, solutions| {
                b.iter(|| {
                    let mut pf = VecParetoFront::<BenchSolution<4>, 4>::new("bench");
                    for solution in solutions {
                        pf.try_insert(solution);
                    }
                    pf.len()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("linkedlist", size),
            &bench_prefix,
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

criterion_group!(benches, bench_trace_replay_full, bench_trace_replay_scaling);
criterion_main!(benches);
