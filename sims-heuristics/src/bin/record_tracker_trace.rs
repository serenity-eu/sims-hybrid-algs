//! Record tracker operations to a binary trace file.
//!
//! This binary runs a short PLS simulation and records all tracker operations
//! to a u16 binary file for use in benchmarks and validation tests.
//!
//! Format: Each record is a u16 with `(op << 12) | image_index`
//! Op codes: 0=TrackAdd, 1=TrackRem, 2=PeekAdd, 3=PeekRem, 4=Reset

use std::cell::RefCell;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use pareto::ParetoFront;
use pls::objective_tracker::{SimdTrackerArray, TrackerCollection};
use pls::objectives::ObjectiveType;
use pls::problem_bitset::ProblemBitset;
use pls::solution::ImageSet;
use pls::solution::bitset_encoded_solution::BitsetEncodedSolution;
use pls::solution_set_impl::NdTreeSolutionSet;

const OBJECTIVE_TYPES: [ObjectiveType; 4] = [
    ObjectiveType::TotalCost,
    ObjectiveType::CloudyArea,
    ObjectiveType::MinResolution,
    ObjectiveType::MaxIncidenceAngle,
];

#[derive(Parser, Debug)]
#[command(name = "record-tracker-trace")]
#[command(about = "Record tracker operations to a binary trace file")]
struct Args {
    /// Path to the problem instance (.dzn file)
    #[arg(short, long)]
    instance: PathBuf,

    /// Output trace file path
    #[arg(short, long)]
    output: PathBuf,

    /// Timeout in seconds
    #[arg(short, long, default_value = "300")]
    timeout: u64,

    /// Initial population size
    #[arg(short, long, default_value = "100")]
    population: usize,
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

// Thread-local storage for the trace writer
thread_local! {
    static TRACE_WRITER: RefCell<Option<BufWriter<File>>> = const { RefCell::new(None) };
}

fn record_event(op: Op, image_index: usize) {
    TRACE_WRITER.with_borrow_mut(|writer| {
        if let Some(w) = writer.as_mut() {
            let masked = u16::try_from(image_index & 0x0FFF).expect("masked to 12 bits");
            let record = (u16::from(op as u8) << 12) | masked;
            let _ = w.write_all(&record.to_le_bytes());
        }
    });
}

fn init_trace_writer(path: &std::path::Path) -> std::io::Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::with_capacity(1024 * 1024, file); // 1MB buffer
    TRACE_WRITER.with_borrow_mut(|w| *w = Some(writer));
    Ok(())
}

fn flush_trace_writer() {
    TRACE_WRITER.with_borrow_mut(|writer| {
        if let Some(w) = writer.as_mut() {
            let _ = w.flush();
        }
    });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Loading instance: {}", args.instance.display());
    let problem = ProblemBitset::<4>::from_minizinc_datafile(&args.instance, OBJECTIVE_TYPES)?;

    println!(
        "Problem loaded: {} images, {} universe elements",
        problem.num_images(),
        problem.universe_size
    );

    // Initialize trace writer
    init_trace_writer(&args.output)?;
    println!("Recording trace to: {}", args.output.display());

    // Create initial population
    println!(
        "Generating random initial population of {} solutions...",
        args.population
    );
    let mut initial_population: NdTreeSolutionSet<BitsetEncodedSolution<ProblemBitset<4>, 4>, 4> =
        NdTreeSolutionSet::new("recording_population");

    for i in 0..args.population {
        let solution = BitsetEncodedSolution::random_with_seed(&problem, i as u64);
        initial_population.try_insert(&solution);
    }

    println!(
        "Initial population: {} non-dominated solutions",
        initial_population.len()
    );

    // Injecting a recording tracker into the real PLS would require a custom
    // solution type, so instead we drive `SimdTrackerArray` directly through a
    // PLS-like operation sequence and record each op via the thread-local writer.
    println!("Running PLS for {} seconds...", args.timeout);
    let timeout = Duration::from_secs(args.timeout);

    let mut trackers = SimdTrackerArray::<4>::new(&problem);

    // Simulate PLS-like operations by iterating through solutions and their neighborhoods
    let start = std::time::Instant::now();
    let mut total_ops: u64 = 0;

    for iteration in 0u64.. {
        if start.elapsed() >= timeout {
            println!("Timeout reached after {iteration} iterations");
            break;
        }

        // Create a random solution
        let solution = BitsetEncodedSolution::random_with_seed(&problem, iteration);

        // Reset event
        record_event(Op::Reset, 0);
        trackers.initialize_from(&solution, &problem);
        total_ops += 1;

        // Simulate neighborhood exploration
        for image_idx in 0..problem.num_images() {
            // Peek operations (read-only)
            if solution.is_image_selected(image_idx) {
                record_event(Op::PeekRem, image_idx);
                let _ = trackers.peek_removal_delta(image_idx, &problem, &solution);
            } else {
                record_event(Op::PeekAdd, image_idx);
                let _ = trackers.peek_addition_delta(image_idx, &problem, &solution);
            }
            total_ops += 1;
        }

        // Simulate some track operations (state changes)
        let selected: Vec<_> = solution.selected_images().collect();
        let unselected: Vec<_> = solution.unselected_images().collect();

        // Remove a few selected images
        for &idx in selected.iter().take(3) {
            record_event(Op::TrackRem, idx);
            trackers.track_image_removal(idx, &problem);
            total_ops += 1;
        }

        // Add a few unselected images
        for &idx in unselected.iter().take(3) {
            record_event(Op::TrackAdd, idx);
            trackers.track_image_addition(idx, &problem);
            total_ops += 1;
        }

        if iteration % 1000 == 0 {
            println!(
                "Iteration {}: {} total ops, {:.1}s elapsed",
                iteration,
                total_ops,
                start.elapsed().as_secs_f64()
            );
        }
    }

    flush_trace_writer();

    // Get file size
    let metadata = std::fs::metadata(&args.output)?;
    let file_size = metadata.len();
    let num_records = file_size / 2;

    println!("\nTrace recording complete:");
    println!("  Total operations: {total_ops}");
    println!("  File size: {file_size} bytes ({num_records} records)");
    println!("  Output: {}", args.output.display());

    Ok(())
}
