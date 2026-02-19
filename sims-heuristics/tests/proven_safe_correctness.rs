//! Correctness test for ProvenSafeTrackerArray against StandardTrackerArray.
//!
//! Replays the same trace data used by the objective_tracker_bench benchmark through
//! both implementations simultaneously, asserting that deltas and objective values
//! match after every single operation.

use fixedbitset::FixedBitSet;
use pls::objective_tracker::{ProvenSafeTrackerArray, StandardTrackerArray, TrackerCollection};
use pls::objectives::ObjectiveType;
use pls::problem_bitset::ProblemBitset;
use pls::solution::ImageSet;
use std::path::Path;

const OBJECTIVE_TYPES: [ObjectiveType; 4] = [
    ObjectiveType::TotalCost,
    ObjectiveType::CloudyArea,
    ObjectiveType::MinResolution,
    ObjectiveType::MaxIncidenceAngle,
];

const TRACE_PATH: &str = "benches/data/debug_lagos_30_tracker_calls.u16";

// ---------------------------------------------------------------------------
// Minimal solution type to track selected images (mirrors bench ReplaySolution)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct TestSolution {
    selected: FixedBitSet,
    num_images: usize,
}

impl TestSolution {
    fn new(num_images: usize) -> Self {
        Self {
            selected: FixedBitSet::with_capacity(num_images),
            num_images,
        }
    }
}

impl<const D: usize> ImageSet<D> for TestSolution {
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

// ---------------------------------------------------------------------------
// Trace decoding (identical to benchmark)
// ---------------------------------------------------------------------------

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

fn load_trace_events() -> Vec<TraceEvent> {
    let bytes = std::fs::read(TRACE_PATH)
        .unwrap_or_else(|e| panic!("failed to read {TRACE_PATH}: {e}"));
    assert!(
        bytes.len() % 2 == 0,
        "trace file must have even byte length"
    );
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .map(decode_record)
        .collect()
}

fn load_problem() -> ProblemBitset<4> {
    let instance_path = Path::new("data").join("lagos_nigeria_30.dzn");
    ProblemBitset::<4>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES)
        .expect("failed to load data/lagos_nigeria_30.dzn")
}

// ---------------------------------------------------------------------------
// Main correctness test
// ---------------------------------------------------------------------------

/// Replays every trace event through both StandardTrackerArray and
/// ProvenSafeTrackerArray, asserting identical deltas and objective values
/// after every single operation.
#[test]
fn proven_safe_matches_standard_on_full_trace() {
    let problem = load_problem();
    let events = load_trace_events();

    let mut std_tracker = StandardTrackerArray::<4>::new(&problem);
    let mut proven_tracker = ProvenSafeTrackerArray::<4>::new(&problem);

    // Both should start with the same initial objectives.
    assert_eq!(
        std_tracker.initial_objectives(),
        proven_tracker.initial_objectives(),
        "initial_objectives mismatch"
    );
    assert_eq!(
        std_tracker.values(),
        proven_tracker.values(),
        "initial values mismatch"
    );

    let mut solution = TestSolution::new(problem.num_images());
    let mut step: u64 = 0;
    let mut resets: u64 = 0;

    for &event in &events {
        let image_index = event.image_index as usize;

        match event.op {
            Op::Reset => {
                std_tracker = StandardTrackerArray::<4>::new(&problem);
                proven_tracker = ProvenSafeTrackerArray::<4>::new(&problem);
                solution = TestSolution::new(problem.num_images());
                resets += 1;

                assert_eq!(
                    std_tracker.values(),
                    proven_tracker.values(),
                    "values diverged after Reset at step {step} (reset #{resets})"
                );
            }
            Op::TrackAdd => {
                // Guard: if already selected, reset (same logic as benchmark).
                if solution.selected.contains(image_index) {
                    std_tracker = StandardTrackerArray::<4>::new(&problem);
                    proven_tracker = ProvenSafeTrackerArray::<4>::new(&problem);
                    solution = TestSolution::new(problem.num_images());
                    resets += 1;
                }

                let std_deltas = std_tracker.track_image_addition(image_index, &problem);
                let proven_deltas = proven_tracker.track_image_addition(image_index, &problem);
                solution.selected.set(image_index, true);

                assert_eq!(
                    std_deltas, proven_deltas,
                    "TrackAdd delta mismatch at step {step}, image {image_index}\n  \
                     standard: {std_deltas:?}\n  proven:   {proven_deltas:?}"
                );
                assert_eq!(
                    std_tracker.values(),
                    proven_tracker.values(),
                    "values diverged after TrackAdd at step {step}, image {image_index}"
                );
            }
            Op::TrackRem => {
                // Guard: if not selected, reset (same logic as benchmark).
                if !solution.selected.contains(image_index) {
                    std_tracker = StandardTrackerArray::<4>::new(&problem);
                    proven_tracker = ProvenSafeTrackerArray::<4>::new(&problem);
                    solution = TestSolution::new(problem.num_images());
                    resets += 1;
                }

                let std_deltas = std_tracker.track_image_removal(image_index, &problem);
                let proven_deltas = proven_tracker.track_image_removal(image_index, &problem);
                solution.selected.set(image_index, false);

                assert_eq!(
                    std_deltas, proven_deltas,
                    "TrackRem delta mismatch at step {step}, image {image_index}\n  \
                     standard: {std_deltas:?}\n  proven:   {proven_deltas:?}"
                );
                assert_eq!(
                    std_tracker.values(),
                    proven_tracker.values(),
                    "values diverged after TrackRem at step {step}, image {image_index}"
                );
            }
            Op::PeekAdd => {
                let std_deltas =
                    std_tracker.peek_addition_delta(image_index, &problem, &solution);
                let proven_deltas =
                    proven_tracker.peek_addition_delta(image_index, &problem, &solution);

                assert_eq!(
                    std_deltas, proven_deltas,
                    "PeekAdd delta mismatch at step {step}, image {image_index}\n  \
                     standard: {std_deltas:?}\n  proven:   {proven_deltas:?}"
                );
                // Peek operations do not change state, but verify anyway.
                assert_eq!(
                    std_tracker.values(),
                    proven_tracker.values(),
                    "values diverged after PeekAdd at step {step}, image {image_index}"
                );
            }
            Op::PeekRem => {
                let std_deltas =
                    std_tracker.peek_removal_delta(image_index, &problem, &solution);
                let proven_deltas =
                    proven_tracker.peek_removal_delta(image_index, &problem, &solution);

                assert_eq!(
                    std_deltas, proven_deltas,
                    "PeekRem delta mismatch at step {step}, image {image_index}\n  \
                     standard: {std_deltas:?}\n  proven:   {proven_deltas:?}"
                );
                assert_eq!(
                    std_tracker.values(),
                    proven_tracker.values(),
                    "values diverged after PeekRem at step {step}, image {image_index}"
                );
            }
        }

        step += 1;
    }

    eprintln!(
        "Correctness check passed: {step} trace events replayed, {resets} resets, \
         all deltas and values matched between Standard and ProvenSafe."
    );
}
