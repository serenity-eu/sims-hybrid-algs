//! A/B the cross-worker claim check in ConcurrentPLS.
//!
//! Reports front quality and duplicate counts with the shared filter on and
//! off, on the same instance, seed and budget. The question it answers is
//! whether stopping workers repeating each other buys front quality, or merely
//! moves the work somewhere else.
use std::{ops::RangeInclusive, path::PathBuf, time::Duration};

use clap::Parser;
use pareto::{HasObjectives, ParetoFront};
use pls::{
    concurrent_pls::{
        config::{ConcurrentPLSConfig, RegionSearchMode},
        orchestrator::ConcurrentPLS,
    },
    objectives::ObjectiveType,
    problem_bitset::ProblemBitset,
    solution_impl::bitset_encoded_solution::BitsetEncodedSolution,
    solution_set_impl::NdTreeSolutionSet,
};

const D: usize = 4;
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
    #[arg(short = 'n', long, default_value = "8")]
    threads: usize,
    #[arg(short = 'p', long, default_value = "50")]
    initial_pop: usize,
}

struct RunReport {
    front: Vec<[u64; D]>,
    dupes: usize,
    explored: usize,
    /// Per-region neighbours explored, to expose load imbalance.
    per_region: Vec<usize>,
    /// Per-region archive entries another region owns.
    out_of_region: usize,
}

fn run(problem: &Problem, args: &Cli, dedup: bool, mode: RegionSearchMode) -> RunReport {
    let initial: Archive = (0..args.initial_pop)
        .map(|i| Solution::random_with_seed(problem, i as u64 + 42))
        .collect();
    let mut config = ConcurrentPLSConfig::default_with_threads(args.threads, args.timeout);
    config.is_deterministic = true;
    config.neighborhood_size_range = RangeInclusive::new(1, 6);
    config.shared_visited_capacity = if dedup { Some(8_000_000) } else { None };
    config.region_search_mode = mode;
    let result = ConcurrentPLS::<Solution, Problem, D>::new(problem, config).solve(&initial);
    RunReport {
        front: result.archive.iter().map(|s| *s.objectives()).collect(),
        dupes: result
            .region_results
            .iter()
            .map(|r| r.stats.duplicates_skipped)
            .sum(),
        explored: result
            .region_results
            .iter()
            .map(|r| r.stats.neighbors_explored)
            .sum(),
        per_region: result
            .region_results
            .iter()
            .map(|r| r.stats.neighbors_explored)
            .collect(),
        out_of_region: result
            .region_results
            .iter()
            .map(|r| r.stats.out_of_region_count)
            .sum(),
    }
}

fn dominates(a: &[u64; D], b: &[u64; D]) -> bool {
    a.iter().zip(b).all(|(x, y)| x <= y) && a.iter().zip(b).any(|(x, y)| x < y)
}

fn coverage(a: &[[u64; D]], b: &[[u64; D]]) -> f64 {
    if b.is_empty() {
        return 0.0;
    }
    100.0
        * b.iter()
            .filter(|y| a.iter().any(|x| dominates(x, y)))
            .count() as f64
        / b.len() as f64
}

fn main() {
    let args = Cli::parse();
    let problem =
        Problem::from_minizinc_datafile(&args.instance, OBJECTIVE_TYPES).expect("load instance");
    eprintln!(
        "{}: {} images, universe {}, {} threads, {:?}",
        args.instance
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?"),
        problem.num_images(),
        problem.num_elements(),
        args.threads,
        args.timeout,
    );

    let modes = [
        ("Unconstrained", RegionSearchMode::Unconstrained),
        ("SA-PLS", RegionSearchMode::ScalarizedAuxiliary),
        (
            "SA-PLS+fallback",
            RegionSearchMode::ScalarizedAuxiliaryWithFallback,
        ),
    ];
    let mut runs = Vec::new();
    for (name, mode) in modes {
        for dedup in [false, true] {
            let r = run(&problem, &args, dedup, mode);
            println!(
                "{name:<18} dedup {:<3}  front {:>6}  explored {:>9}  out-of-region {:>7}",
                if dedup { "ON" } else { "off" },
                r.front.len(),
                r.explored,
                r.out_of_region
            );
            runs.push((format!("{name}/{}", if dedup { "on" } else { "off" }), r));
        }
    }
    println!("\npairwise coverage (row dominates column, %):");
    print!("{:<20}", "");
    for (n, _) in &runs {
        print!("{:>18}", n);
    }
    println!();
    for (na, a) in &runs {
        print!("{na:<20}");
        for (_, b) in &runs {
            print!("{:>18.1}", coverage(&a.front, &b.front));
        }
        println!();
    }
    let off = run(&problem, &args, false, RegionSearchMode::Unconstrained);
    let on = run(&problem, &args, true, RegionSearchMode::Unconstrained);

    let imbalance = |v: &[usize]| {
        let (mn, mx) = (
            v.iter().min().copied().unwrap_or(0),
            v.iter().max().copied().unwrap_or(0),
        );
        if mn == 0 {
            f64::INFINITY
        } else {
            mx as f64 / mn as f64
        }
    };

    println!("\n{:<24}{:>12}{:>12}", "", "dedup OFF", "dedup ON");
    println!(
        "{:<24}{:>12}{:>12}",
        "front size",
        off.front.len(),
        on.front.len()
    );
    println!(
        "{:<24}{:>12}{:>12}",
        "neighbours explored", off.explored, on.explored
    );
    println!(
        "{:<24}{:>12}{:>12}",
        "duplicates skipped", off.dupes, on.dupes
    );
    println!(
        "{:<24}{:>12}{:>12}",
        "archive out-of-region", off.out_of_region, on.out_of_region
    );
    println!(
        "{:<24}{:>12.1}{:>12.1}",
        "region load max/min",
        imbalance(&off.per_region),
        imbalance(&on.per_region)
    );
    println!(
        "\ncoverage: ON dominates {:.1}% of OFF's front; OFF dominates {:.1}% of ON's",
        coverage(&on.front, &off.front),
        coverage(&off.front, &on.front),
    );
    println!("\nper-region neighbours explored:");
    println!("  OFF: {:?}", off.per_region);
    println!("  ON : {:?}", on.per_region);
}
