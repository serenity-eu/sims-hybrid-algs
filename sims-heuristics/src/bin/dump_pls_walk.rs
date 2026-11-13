//! Dump every solution a PLS run explores, dominated ones included.
//!
//! The Pareto front alone cannot say why a decomposed, parallel PLS has its
//! workers rediscovering each other's solutions: the front is what survives,
//! and the question is about where the search *went*. This records the whole
//! walk -- objectives and decision vector per explored solution -- so the two
//! spaces can be compared directly.
//!
//! Usage:
//!   cargo run --release --bin dump-pls-walk -- \
//!     --instance `tests/data/lagos_nigeria_50.dzn` --timeout 20s --out walk.jsonl
use std::{
    fs::File,
    io::{BufWriter, Write},
    ops::RangeInclusive,
    path::PathBuf,
    time::Duration,
};

use clap::Parser;
use nd_tree::{TrackedNdTree, TrackedNdTreeConfig};
use pareto::{HasObjectives, MoSolution, Objectives, ParetoFront};

/// Minimal carrier so the walk can be replayed through the tracked tree.
#[derive(Clone, PartialEq)]
struct TracePoint {
    objectives: Objectives<D>,
}
impl HasObjectives<D> for TracePoint {
    fn objectives(&self) -> &Objectives<D> {
        &self.objectives
    }
}
impl MoSolution<D> for TracePoint {}
use pls::{
    PlsOptimizations, objectives::ObjectiveType, pareto_local_search::ParetoLocalSearch,
    problem_bitset::ProblemBitset, solution_impl::bitset_encoded_solution::BitsetEncodedSolution,
    solution_set_impl::NdTreeSolutionSet,
};

const D: usize = 4;
const NEIGHBORHOOD: RangeInclusive<u32> = 1..=6;
const OBJECTIVE_TYPES: [ObjectiveType; D] = [
    ObjectiveType::TotalCost,
    ObjectiveType::CloudyArea,
    ObjectiveType::MinResolution,
    ObjectiveType::MaxIncidenceAngle,
];

type Problem = ProblemBitset<D>;
type Solution = BitsetEncodedSolution<Problem, D>;
type Archive = NdTreeSolutionSet<Solution, D>;

#[derive(Parser)]
struct Cli {
    #[arg(short, long)]
    instance: PathBuf,
    #[arg(short, long, value_parser = humantime::parse_duration, default_value = "20s")]
    timeout: Duration,
    #[arg(short = 'p', long, default_value = "50")]
    initial_pop: usize,
    #[arg(short, long, default_value = "walk.jsonl")]
    out: PathBuf,
    /// Cap on neighbours yielded per solution. Unset = systematic (full
    /// neighbourhood); small values approximate the random-sampling regime.
    #[arg(short = 'b', long)]
    budget: Option<usize>,
}

fn main() {
    let args = Cli::parse();
    let problem =
        Problem::from_minizinc_datafile(&args.instance, OBJECTIVE_TYPES).expect("load instance");
    let name = args
        .instance
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("?");
    eprintln!(
        "{name}: {} images, universe {}",
        problem.num_images(),
        problem.num_elements()
    );

    let initial: Archive = (0..args.initial_pop)
        .map(|i| Solution::random_with_seed(&problem, i as u64 + 42))
        .collect();
    let opts = PlsOptimizations {
        neighborhood_budget: args.budget,
        ..PlsOptimizations::default()
    };
    let mut pls = ParetoLocalSearch::new(&problem, &initial, NEIGHBORHOOD, false, opts);
    let front: Archive = pls.run(usize::MAX, args.timeout);

    // The archive holds only survivors; the registry holds everything visited.
    let front_keys: std::collections::HashSet<[u64; D]> =
        front.iter().map(|s: &Solution| *s.objectives()).collect();

    // Replay the walk in discovery order through a TrackedNdTree with filtering
    // OFF: every solution is indexed, and each insertion reports which earlier
    // solutions it dominated. That is the domination *structure* over time,
    // which the surviving front alone cannot show.
    let mut walk: Vec<&pls::explored_solutions_data::SolutionFingerprint<D>> =
        pls.explored_solutions.solutions.values().collect();
    walk.sort_by_key(|f| f.timestamp);
    let mut tree: TrackedNdTree<TracePoint, 8, D, 4> = TrackedNdTree::new(TrackedNdTreeConfig {
        filter_dominated: false,
    });
    let mut dominated_by_insert: Vec<Vec<u32>> = Vec::with_capacity(walk.len());
    for fp in &walk {
        let r = tree.insert(TracePoint {
            objectives: fp.objectives,
        });
        dominated_by_insert.push(r.dominated_indices.clone());
    }
    eprintln!(
        "TrackedNdTree(filter_dominated=false): indexed {} of {} explored",
        dominated_by_insert.len(),
        walk.len()
    );

    let mut w = BufWriter::new(File::create(&args.out).expect("create output"));
    let mut n = 0usize;
    for (idx, fp) in walk.iter().enumerate() {
        let objs: Vec<u64> = fp.objectives.to_vec();
        let imgs: Vec<usize> = fp.selected_images.clone();
        writeln!(
            w,
            r#"{{"i":{idx},"obj":{objs:?},"images":{imgs:?},"iter":{},"t_us":{},"nbhd":{},"on_front":{},"dominated":{:?}}}"#,
            fp.iteration,
            fp.timestamp.as_micros(),
            fp.explored_neighborhood_size,
            front_keys.contains(&fp.objectives),
            dominated_by_insert[idx],
        )
        .expect("write");
        n += 1;
    }
    eprintln!(
        "explored {n} solutions, front {} -> {}",
        front.len(),
        args.out.display()
    );
}
