//! Measure what a redundancy-removal pass adds to a finished PLS front.
//!
//! An image is redundant when every element it covers is also covered by
//! another selected image. Dropping one cannot break feasibility, and since
//! `TotalCost` is exactly additive over the selection -- verified to the last
//! unit across a million explored solutions -- it strictly reduces cost. It may
//! raise `CloudyArea` or `MinResolution`, because a cell that had two candidate
//! images now has one, so the polished solution is a new trade-off rather than
//! a guaranteed replacement; both are offered to the archive and dominance
//! decides.
//!
//! The EA operators already do this after every genetic operation
//! (`evolutionary::operators::remove_redundant_images`); Pareto local search
//! never has. Measured on `lagos_nigeria_50`, 98.6% of front solutions carry at
//! least one redundant image, averaging 6.2 each -- so the question is not
//! whether they exist but whether removing them is worth anything.
use std::{ops::RangeInclusive, path::PathBuf, time::Duration};

use clap::Parser;
use pareto::{HasObjectives, ParetoFront};
use pls::{
    PlsOptimizations, evolutionary::operators::remove_redundant_images, objectives::ObjectiveType,
    pareto_local_search::ParetoLocalSearch, problem_bitset::ProblemBitset,
    solution_impl::bitset_encoded_solution::BitsetEncodedSolution,
    solution_set_impl::NdTreeSolutionSet,
};

const D: usize = 4;

const NAMES: [&str; D] = ["TotalCost", "CloudyArea", "MinResolution", "MaxIncAngle"];
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
}

fn dominates(a: &[u64; D], b: &[u64; D]) -> bool {
    a.iter().zip(b).all(|(x, y)| x <= y) && a.iter().zip(b).any(|(x, y)| x < y)
}

fn main() {
    let args = Cli::parse();
    let problem =
        Problem::from_minizinc_datafile(&args.instance, OBJECTIVE_TYPES).expect("load instance");
    eprintln!(
        "{}: {} images, universe {}",
        args.instance
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?"),
        problem.num_images(),
        problem.num_elements()
    );

    let initial: Archive = (0..args.initial_pop)
        .map(|i| Solution::random_with_seed(&problem, i as u64 + 42))
        .collect();
    let mut pls = ParetoLocalSearch::new(
        &problem,
        &initial,
        NEIGHBORHOOD,
        false,
        PlsOptimizations::default(),
    );
    let front: Archive = pls.run(usize::MAX, args.timeout);

    let original: Vec<[u64; D]> = front.iter().map(|s: &Solution| *s.objectives()).collect();

    // Polish every front member, then let the archive arbitrate.
    let mut polished: Archive = front.iter().cloned().collect();
    let (mut touched, mut improved) = (0usize, 0usize);
    let (mut harmed, mut helped, mut traded) = (0usize, 0usize, 0usize);
    let mut worsened = [0usize; D];
    for sol in front.iter() {
        let mut selected = sol.selected_images().collect::<Vec<_>>();
        let mut bits = fixedbitset::FixedBitSet::with_capacity(problem.num_images());
        for i in &selected {
            bits.insert(*i);
        }
        let before = bits.count_ones(..);
        let mut trackers = Solution::synced_trackers(&bits, &problem);
        remove_redundant_images::<Problem, D>(&mut bits, &problem, &mut trackers);
        if bits.count_ones(..) == before {
            continue;
        }
        touched += 1;
        selected = bits.ones().collect();
        let candidate = Solution::from_selected_images(&selected, &problem);
        // Does stripping actively harm THIS solution, or merely trade?
        let (a, b) = (sol.objectives(), candidate.objectives());
        if dominates(a, b) {
            harmed += 1;
        } else if dominates(b, a) {
            helped += 1;
        } else {
            traded += 1;
        }
        for j in 0..D {
            if b[j] > a[j] {
                worsened[j] += 1;
            }
        }
        if polished.try_insert(&candidate) {
            improved += 1;
        }
    }

    let after: Vec<[u64; D]> = polished
        .iter()
        .map(|s: &Solution| *s.objectives())
        .collect();
    let cov = |a: &[[u64; D]], b: &[[u64; D]]| {
        if b.is_empty() {
            0.0
        } else {
            // usize -> f64 loses precision above 2^53. These are front sizes,
            // so u32 is ample and the conversion is exact.
            let hit = u32::try_from(
                b.iter()
                    .filter(|y| a.iter().any(|x| dominates(x, y)))
                    .count(),
            )
            .expect("front size fits in u32");
            let total = u32::try_from(b.len()).expect("front size fits in u32");
            100.0 * f64::from(hit) / f64::from(total)
        }
    };

    println!("\nfront before polish : {}", original.len());
    println!("front after polish  : {}", after.len());
    println!("solutions with redundancy removed : {touched}");
    println!("polished entries accepted by archive: {improved}");
    println!(
        "\npolished front dominates {:.1}% of the original;\noriginal dominates {:.1}% of the polished",
        cov(&after, &original),
        cov(&original, &after)
    );
    println!("\nper-solution effect of stripping redundant images:");
    println!("  strictly WORSE (original dominates it): {harmed}");
    println!("  strictly better                       : {helped}");
    println!("  incomparable trade-off                : {traded}");
    println!("\n  objective degraded, per objective:");
    for j in 0..D {
        println!(
            "    {:<14} worsened in {} of {touched}",
            NAMES[j], worsened[j]
        );
    }
}
