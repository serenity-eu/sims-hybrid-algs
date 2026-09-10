//! Run MILP-based large-neighbourhood search over a PLS front.
//!
//! Reads a front produced by `pls_front2d` (`cost,cloud,|S|,"i j k"`), improves
//! each solution with [`pls::milp_lns`], and writes the improved front in the
//! same format so the two are directly comparable.
use std::{path::PathBuf, time::{Duration, Instant}};

use clap::Parser;
use fixedbitset::FixedBitSet;
use pls::{
    milp_lns::{LnsConfig, MilpLns, operator_names},
    objectives::ObjectiveType,
    problem_bitset::ProblemBitset,
};

const D: usize = 2;
const OBJECTIVE_TYPES: [ObjectiveType; D] = [ObjectiveType::TotalCost, ObjectiveType::CloudyArea];

#[derive(Parser)]
struct Cli {
    #[arg(short, long)] instance: PathBuf,
    /// Front CSV from `pls_front2d`.
    #[arg(short, long)] front: PathBuf,
    #[arg(long, default_value = "10")] destroy: usize,
    #[arg(long, default_value = "40")] candidates: usize,
    #[arg(short, long, value_parser = humantime::parse_duration, default_value = "60s")]
    timeout: Duration,
    #[arg(long, value_parser = humantime::parse_duration, default_value = "3s")]
    mip_time: Duration,
    #[arg(long, default_value = "42")] seed: u64,
    /// Run a heuristic epsilon-constraint sweep with this many levels instead
    /// of improving each input solution under its own clear-area floor.
    #[arg(long)] sweep: Option<usize>,
    /// Adaptive epsilon-constraint enumeration: walk the whole front by raising
    /// the clear-area floor past each point found.
    #[arg(long)] enumerate: bool,
    /// Balanced Box Method: recursive criterion-space split with lexicographic
    /// solves. Spreads its solves over the whole front, so a partial budget
    /// still covers both ends.
    #[arg(long)] balanced_box: bool,
}

fn main() {
    let a = Cli::parse();
    let problem: ProblemBitset<D> =
        ProblemBitset::from_minizinc_datafile(&a.instance, OBJECTIVE_TYPES).expect("load instance");

    let mut starts: Vec<FixedBitSet> = Vec::new();
    for line in std::fs::read_to_string(&a.front).expect("read front").lines() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 4 {
            continue;
        }
        let mut bs = FixedBitSet::with_capacity(problem.images.len());
        for tok in fields[3].split_whitespace() {
            if let Ok(i) = tok.parse::<usize>() {
                bs.insert(i);
            }
        }
        starts.push(bs);
    }

    let config = LnsConfig {
        destroy_size: a.destroy,
        candidate_pool: a.candidates,
        mip_time: a.mip_time,
        total_time: a.timeout,
        seed: a.seed,
        ..LnsConfig::default()
    };
    let mut lns = MilpLns::new(&problem, config);
    eprintln!("{} starting solutions, budget {:?}", starts.len(), a.timeout);

    let deadline = Instant::now() + a.timeout;
    if a.balanced_box {
        use pls::milp_lns::{balanced_box::BalancedBox, model::{clear_coverage, PersistentModel}};
        let (clear, areas) = clear_coverage(&problem);
        let total: u64 = areas.iter().sum();
        let mut model = PersistentModel::new(&problem, &clear, &areas);
        let warm = starts
            .iter()
            .min_by_key(|s| s.ones().map(|i| problem.image_cost(i)).sum::<u64>())
            .cloned()
            .unwrap_or_else(|| FixedBitSet::with_capacity(problem.images.len()));
        let mut bb = BalancedBox::new(&mut model, total, a.mip_time);
        let (pts, stats) = bb.run(&warm, Instant::now() + a.timeout);
        for pt in &pts {
            let sel: Vec<String> = pt.images.ones().map(|i| i.to_string()).collect();
            println!("{},{},{},{}", pt.cost, pt.cloud, sel.len(), sel.join(" "));
        }
        eprintln!(
            "balanced box: {} points | solves {} (optimal {}, timed out {}, infeasible {}) | boxes closed {} left {} | exact {}",
            pts.len(), stats.solves, stats.optimal, stats.timed_out, stats.infeasible,
            stats.boxes_closed, stats.boxes_left, stats.exact
        );
        return;
    }

    let results: Vec<FixedBitSet> = if a.enumerate {
        lns.enumerate_front(&starts, deadline)
    } else if let Some(levels) = a.sweep {
        lns.sweep(&starts, levels, deadline)
    } else {
        // Share the budget across starts so a slow early solution cannot consume
        // the whole pass and leave the rest of the front untouched.
        let mut v = Vec::with_capacity(starts.len());
        for (idx, start) in starts.iter().enumerate() {
            if Instant::now() >= deadline {
                break;
            }
            let remaining = starts.len() - idx;
            let slice = (deadline - Instant::now()) / u32::try_from(remaining).unwrap_or(1);
            v.push(lns.improve(start, Instant::now() + slice));
        }
        v
    };
    for out in &results {
        let cost = lns.cost_of(out);
        let cloud = lns.cloudy_area_of(out);
        let sel: Vec<String> = out.ones().map(|i| i.to_string()).collect();
        println!("{cost},{cloud},{},{}", sel.len(), sel.join(" "));
    }

    let s = lns.stats();
    let w = lns.operator_weights();
    eprintln!(
        "moves {} improving {} | solves: optimal {} timed-out {} failed {}",
        s.moves, s.improving, s.optimal_solves, s.timed_out_solves, s.failed_solves
    );
    let names = operator_names();
    eprintln!(
        "operator weights: {}",
        (0..3).map(|i| format!("{}={:.2}", names[i], w[i])).collect::<Vec<_>>().join(" ")
    );
}
