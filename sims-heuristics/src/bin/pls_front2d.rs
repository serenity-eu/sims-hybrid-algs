//! Dump a 2-objective PLS front, for comparison against the exact methods'.
use std::{ops::RangeInclusive, path::PathBuf, time::Duration};
use clap::Parser;
use pareto::{HasObjectives, ParetoFront};
use pls::{
    PlsOptimizations, objectives::ObjectiveType, pareto_local_search::ParetoLocalSearch,
    problem_bitset::ProblemBitset,
    solution_impl::bitset_encoded_solution::BitsetEncodedSolution,
    solution_set_impl::NdTreeSolutionSet,
};
const D: usize = 2;
const OBJECTIVE_TYPES: [ObjectiveType; D] = [ObjectiveType::TotalCost, ObjectiveType::CloudyArea];
type Problem = ProblemBitset<D>;
type Solution = BitsetEncodedSolution<Problem, D>;
type Archive = NdTreeSolutionSet<Solution, D>;

#[derive(Parser)]
struct Cli {
    #[arg(short, long)] instance: PathBuf,
    #[arg(short, long, value_parser = humantime::parse_duration, default_value = "60s")]
    timeout: Duration,
    #[arg(short = 'p', long, default_value = "50")] initial_pop: usize,
    /// Disable ranked-candidate truncation (explore the full neighbourhood).
    #[arg(long)] no_ranking: bool,
    /// Cap on ranked removal candidates at k=1.
    #[arg(long)] k1: Option<usize>,
    /// Smallest neighbourhood structure to explore.
    #[arg(long, default_value = "1")] kmin: u32,
    /// Largest neighbourhood structure to explore.
    #[arg(long, default_value = "6")] kmax: u32,
}

fn main() {
    let a = Cli::parse();
    let problem = Problem::from_minizinc_datafile(&a.instance, OBJECTIVE_TYPES).expect("load");
    let initial: Archive = (0..a.initial_pop)
        .map(|i| Solution::random_with_seed(&problem, i as u64 + 42)).collect();
    let mut opts = PlsOptimizations::default();
    if a.no_ranking {
        opts.flags.remove(pls::pls_config::PlsFlags::USE_RANKED_CANDIDATES);
    }
    if let Some(k) = a.k1 { opts.max_k1_candidates = k; }
    let mut pls = ParetoLocalSearch::new(
        &problem, &initial, RangeInclusive::new(a.kmin, a.kmax), false, opts);
    let front: Archive = pls.run(usize::MAX, a.timeout);
    for s in front.iter() {
        let o: &[u64; D] = s.objectives();
        let sel: Vec<String> = s.selected_images().map(|i| i.to_string()).collect();
        println!("{},{},{},{}", o[0], o[1], sel.len(), sel.join(" "));
    }
}
