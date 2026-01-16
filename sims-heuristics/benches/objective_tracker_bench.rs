use std::hint::black_box;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;

use criterion::{
    BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
use fixedbitset::FixedBitSet;
use pls::objective_tracker::{
    AltTrackerArray, SimdTrackerArray, StandardTrackerArray, TrackerCollection, 
    ExplicitSimdTrackerArray, SaturatingTrackerArray, SafeTrackerArray, SimpleTrackerArray
};
use pls::objectives::ObjectiveType;
use pls::problem_bitset::ProblemBitset;
use pls::solution::ImageSet;

const OBJECTIVE_TYPES: [ObjectiveType; 4] = [
    ObjectiveType::TotalCost,
    ObjectiveType::CloudyArea,
    ObjectiveType::MinResolution,
    ObjectiveType::MaxIncidenceAngle,
];

const TRACE_PATH: &str = "benches/data/debug_lagos_30_tracker_calls.u16";

// The full debug-lagos-30 trace is intentionally huge (millions of events). Criterion defaults
// (100 samples in ~5s) become impractical if we replay the entire trace each iteration.
//
// We therefore benchmark a bounded prefix by default. Override with:
//   SIMS_TRACKER_BENCH_TRACE_LIMIT=0   (replay full trace)
//   SIMS_TRACKER_BENCH_TRACE_LIMIT=500000
const DEFAULT_TRACE_LIMIT_RECORDS: usize = 250_000;

#[derive(Clone, Debug)]
struct ReplaySolution {
    selected: FixedBitSet,
    num_images: usize,
}

impl ReplaySolution {
    fn new(num_images: usize) -> Self {
        Self {
            selected: FixedBitSet::with_capacity(num_images),
            num_images,
        }
    }
}

impl<const D: usize> ImageSet<D> for ReplaySolution {
    fn selected_images(&self) -> impl Iterator<Item = usize> {
        self.selected.ones()
    }

    fn unselected_images(&self) -> impl Iterator<Item = usize> {
        (0..self.num_images).filter(move |&i| !self.selected.contains(i))
    }

    fn is_image_selected(&self, image_index: usize) -> bool {
        self.selected.contains(image_index)
    }

    fn num_selected_images(&self) -> usize {
        self.selected.count_ones(..)
    }

    fn set_image(&mut self, image_index: usize, selected: bool) {
        self.selected.set(image_index, selected);
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Debug)]
enum Op {
    TrackAdd = 0,
    TrackRem = 1,
    PeekAdd = 2,
    PeekRem = 3,
    Reset = 4,
}

#[derive(Copy, Clone, Debug)]
struct TraceEvent {
    op: Op,
    image_index: u16,
}

fn decode_record(record: u16) -> TraceEvent {
    let op = match (record >> 12) as u8 {
        0 => Op::TrackAdd,
        1 => Op::TrackRem,
        2 => Op::PeekAdd,
        3 => Op::PeekRem,
        4 => Op::Reset,
        other => panic!("unknown op: {other}"),
    };

    let image_index = record & 0x0FFF;
    TraceEvent { op, image_index }
}

fn load_problem() -> ProblemBitset<4> {
    let instance_path = Path::new("data").join("lagos_nigeria_30.dzn");
    ProblemBitset::<4>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES)
        .expect("failed to load data/lagos_nigeria_30.dzn")
}

fn trace_events() -> &'static [TraceEvent] {
    static TRACE: OnceLock<Vec<TraceEvent>> = OnceLock::new();

    TRACE.get_or_init(|| {
        let bytes = std::fs::read(TRACE_PATH)
            .unwrap_or_else(|e| panic!("failed to read {TRACE_PATH}: {e}"));
        assert!(
            bytes.len().is_multiple_of(2),
            "trace file must have even byte length"
        );

        bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .map(decode_record)
            .collect()
    })
}

fn trace_limit_records() -> usize {
    match std::env::var("SIMS_TRACKER_BENCH_TRACE_LIMIT") {
        Ok(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                DEFAULT_TRACE_LIMIT_RECORDS
            } else {
                trimmed.parse::<usize>().unwrap_or_else(|e| {
                    panic!("invalid SIMS_TRACKER_BENCH_TRACE_LIMIT='{trimmed}': {e}")
                })
            }
        }
        Err(std::env::VarError::NotPresent) => DEFAULT_TRACE_LIMIT_RECORDS,
        Err(e) => panic!("failed reading SIMS_TRACKER_BENCH_TRACE_LIMIT: {e}"),
    }
}


fn replay_trace<const D: usize, T: TrackerCollection<D>>(
    mut trackers: T,
    problem: &ProblemBitset<D>,
    events: &[TraceEvent],
) -> (T, i64) {
    let mut solution = ReplaySolution::new(problem.num_images());
    let mut sink: i64 = 0;
    let mut resets: u64 = 0;

    for &event in events {
        let op = event.op;
        let image_index = event.image_index as usize;

        match op {
            Op::Reset => {
                trackers = T::new(problem);
                solution = ReplaySolution::new(problem.num_images());
                resets += 1;
            }
            Op::TrackAdd => {
                if solution.selected.contains(image_index) {
                    trackers = T::new(problem);
                    solution = ReplaySolution::new(problem.num_images());
                    resets += 1;
                }
                let deltas = trackers.track_image_addition(image_index, problem);
                solution.selected.set(image_index, true);
                sink ^= deltas[0];
            }
            Op::TrackRem => {
                if !solution.selected.contains(image_index) {
                    trackers = T::new(problem);
                    solution = ReplaySolution::new(problem.num_images());
                    resets += 1;
                }
                let deltas = trackers.track_image_removal(image_index, problem);
                solution.selected.set(image_index, false);
                sink ^= deltas[0];
            }
            Op::PeekAdd => {
                let deltas = trackers.peek_addition_delta(image_index, problem, &solution);
                sink ^= deltas[0];
            }
            Op::PeekRem => {
                let deltas = trackers.peek_removal_delta(image_index, problem, &solution);
                sink ^= deltas[0];
            }
        }
    }

    // Ensure resets are not optimized away (also provides a tiny perturbation).
    let resets_i64 = i64::try_from(resets).unwrap_or(i64::MAX);
    sink ^= resets_i64;

    (trackers, sink)
}

fn bench_tracker_replay_lagos_30(c: &mut Criterion) {
    let problem = load_problem();
    let all_events = trace_events();
    let limit = trace_limit_records();
    let events = if limit == 0 {
        all_events
    } else {
        &all_events[..std::cmp::min(limit, all_events.len())]
    };

    // Basic sanity checks up-front so failures are obvious.
    assert!(!events.is_empty(), "trace file is empty");

    // Ensure all indices are in range (cost paid once).
    let num_images = problem.num_images();
    for &event in events.iter().take(10_000) {
        let idx = event.image_index as usize;
        assert!(
            idx < num_images,
            "trace index {idx} out of range {num_images}"
        );
    }

    let mut group = c.benchmark_group("objective_tracker_bench");
    // Keep this benchmark usable with `cargo bench` defaults.
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(10));
    group.throughput(Throughput::Elements(events.len() as u64));

    group.bench_with_input(
        BenchmarkId::new("StandardTrackerArray", TRACE_PATH),
        &(),
        |b, &()| {
            b.iter_batched(
                || StandardTrackerArray::<4>::new(&problem),
                |trackers| {
                    let (_trackers, sink) = replay_trace::<4, _>(trackers, &problem, events);
                    black_box(sink);
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("AltTrackerArray", TRACE_PATH),
        &(),
        |b, &()| {
            b.iter_batched(
                || AltTrackerArray::<4>::new(&problem),
                |trackers| {
                    let (_trackers, sink) = replay_trace::<4, _>(trackers, &problem, events);
                    black_box(sink);
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("SimdTrackerArray", TRACE_PATH),
        &(),
        |b, &()| {
            b.iter_batched(
                || SimdTrackerArray::<4>::new(&problem),
                |trackers| {
                    let (_trackers, sink) = replay_trace::<4, _>(trackers, &problem, events);
                    black_box(sink);
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("ExplicitSimdTrackerArray", TRACE_PATH),
        &(),
        |b, &()| {
            b.iter_batched(
                || ExplicitSimdTrackerArray::<4>::new(&problem),
                |trackers| {
                    let (_trackers, sink) = replay_trace::<4, _>(trackers, &problem, events);
                    black_box(sink);
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("SaturatingTrackerArray", TRACE_PATH),
        &(),
        |b, &()| {
            b.iter_batched(
                || SaturatingTrackerArray::<4>::new(&problem),
                |trackers| {
                    let (_trackers, sink) = replay_trace::<4, _>(trackers, &problem, events);
                    black_box(sink);
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("SafeTrackerArray", TRACE_PATH),
        &(),
        |b, &()| {
            b.iter_batched(
                || SafeTrackerArray::<4>::new(&problem),
                |trackers| {
                    let (_trackers, sink) = replay_trace::<4, _>(trackers, &problem, events);
                    black_box(sink);
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("SimpleTrackerArray", TRACE_PATH),
        &(),
        |b, &()| {
            b.iter_batched(
                || SimpleTrackerArray::<4>::new(&problem),
                |trackers| {
                    let (_trackers, sink) = replay_trace::<4, _>(trackers, &problem, events);
                    black_box(sink);
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.finish();
}

criterion_group!(benches, bench_tracker_replay_lagos_30);
criterion_main!(benches);
