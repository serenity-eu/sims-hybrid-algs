use std::fs::OpenOptions;
/// Hybrid two-phase benchmark: pseudosolver seeds + PLS variants.
///
/// Compares three PLS algorithms (exhaustive, concurrent, probabilistic probing)
/// across four time-allocation ratios (100:0, 50:50, 25:75, 0:100) using
/// pre-recorded exact-phase solutions as initial population seeds.
///
/// Usage:
///   cargo run --release --bin hybrid-algorithm-benchmark \
///     --features "parallel,probabilistic_probing" -- \
///     --instances-dir tests/data \
///     --solutions-dir ../sims-core/tests/data/pseudo_solver_solutions \
///     --timeout 30s --repeats 3 --json-output results/hybrid_benchmark.json
use std::{
    collections::HashMap,
    io::Write,
    path::PathBuf,
    time::{Duration, Instant},
};

use clap::Parser;
use pareto::{HasObjectives, ParetoFront};
use pls::{
    PlsOptimizations,
    concurrent_pls::{ConcurrentPLS, ConcurrentPLSConfig, RegionSearchMode},
    objectives::ObjectiveType,
    pareto_local_search::ParetoLocalSearch,
    problem_bitset::ProblemBitset,
    residual_problem::{PROBING_BUDGET_EXHAUSTIVE, set_runtime_probing_budget},
    solution_impl::bitset_encoded_solution::BitsetEncodedSolution,
    solution_set_impl::NdTreeSolutionSet,
};

// ═══════════════════════════════════════════════════════════════════════════
//  Constants
// ═══════════════════════════════════════════════════════════════════════════

const NUM_OBJECTIVES: usize = 4;
const OBJECTIVE_TYPES: [ObjectiveType; NUM_OBJECTIVES] = [
    ObjectiveType::TotalCost,
    ObjectiveType::CloudyArea,
    ObjectiveType::MinResolution,
    ObjectiveType::MaxIncidenceAngle,
];

type Problem = ProblemBitset<NUM_OBJECTIVES>;
type Solution = BitsetEncodedSolution<Problem, NUM_OBJECTIVES>;
type Archive = NdTreeSolutionSet<Solution, NUM_OBJECTIVES>;

const DEFAULT_SEED: u64 = 42;
const DEFAULT_POP_SIZE: usize = 100;
const NEIGHBORHOOD_RANGE: std::ops::RangeInclusive<u32> = 1..=6;

/// The default probabilistic probing budget (matches DEFAULT_PROBING_BUDGET in residual_problem).
const PROBING_BUDGET_DEFAULT: usize = 0; // 0 = use compile-time default (1000)

/// Time-allocation ratios: (exact_pct, pls_pct).
const RATIOS: [(u32, u32); 4] = [(100, 0), (50, 50), (25, 75), (0, 100)];

/// Algorithm identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum AlgoKind {
    ExhaustivePls,
    ConcurrentPls,
    ProbabilisticPls,
}

impl AlgoKind {
    const fn label(self) -> &'static str {
        match self {
            Self::ExhaustivePls => "Exhaustive PLS",
            Self::ConcurrentPls => "Concurrent PLS",
            Self::ProbabilisticPls => "Probabilistic PLS",
        }
    }

    const fn short(self) -> &'static str {
        match self {
            Self::ExhaustivePls => "exhaustive",
            Self::ConcurrentPls => "concurrent",
            Self::ProbabilisticPls => "probabilistic",
        }
    }
}

const ALL_ALGOS: [AlgoKind; 3] = [
    AlgoKind::ExhaustivePls,
    AlgoKind::ConcurrentPls,
    AlgoKind::ProbabilisticPls,
];

// ═══════════════════════════════════════════════════════════════════════════
//  CLI
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Parser)]
#[command(
    name = "hybrid-algorithm-benchmark",
    about = "Benchmark exhaustive / concurrent / probabilistic PLS in hybrid two-phase setup"
)]
struct Cli {
    /// Directory containing .dzn problem instance files.
    #[arg(short = 'i', long = "instances-dir", default_value = "tests/data")]
    instances_dir: PathBuf,

    /// Directory containing pseudosolver JSON solution files.
    #[arg(
        short = 's',
        long = "solutions-dir",
        default_value = "../sims-core/tests/data/pseudo_solver_solutions"
    )]
    solutions_dir: PathBuf,

    /// Per-instance total time budget (e.g. 30s, 1m, 2m).
    #[arg(
        short = 't',
        long = "timeout",
        value_parser = humantime::parse_duration,
        default_value = "30s"
    )]
    timeout: Duration,

    /// Filter instances by filename substring (e.g. "_30" for small).
    #[arg(short = 'f', long = "filter")]
    filter: Option<String>,

    /// Number of repetitions per (algorithm, ratio, instance) triple.
    #[arg(long = "repeats", default_value_t = 3)]
    repeats: usize,

    /// Random seed.
    #[arg(long = "seed", default_value_t = DEFAULT_SEED)]
    seed: u64,

    /// Population size for random initial populations (ratio 0:100).
    #[arg(long = "pop-size", default_value_t = DEFAULT_POP_SIZE)]
    pop_size: usize,

    /// Number of threads for concurrent PLS (0 = physical CPU count).
    #[arg(long = "threads", default_value_t = 0)]
    threads: usize,

    /// Write machine-readable JSON results to this path.
    #[arg(long = "json-output")]
    json_output: Option<PathBuf>,

    /// Skip specific algorithms (comma-separated: exhaustive,concurrent,probabilistic).
    #[arg(long = "skip-algo")]
    skip_algo: Option<String>,

    /// Probing budget for probabilistic PLS (0 = compile-time default, typically 1000).
    #[arg(long = "probing-budget", default_value_t = PROBING_BUDGET_DEFAULT)]
    probing_budget: usize,

    /// Use per-instance adaptive timeout (overrides --timeout based on instance size).
    #[arg(long = "adaptive-timeout")]
    adaptive_timeout: bool,

    /// Maximum total wall-clock time for the entire benchmark run (e.g. 6h, 90m).
    /// The benchmark will stop gracefully after finishing the current instance
    /// once this limit is exceeded.  Results for completed instances are preserved
    /// in the JSONL checkpoint file.
    #[arg(
        long = "max-wall-time",
        value_parser = humantime::parse_duration,
    )]
    max_wall_time: Option<Duration>,
}

// ═══════════════════════════════════════════════════════════════════════════
//  Pseudosolver JSON parsing
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, serde::Deserialize)]
struct PseudoSolverData {
    #[allow(dead_code)]
    instance_name: String,
    #[allow(dead_code)]
    num_solutions: usize,
    solutions: Vec<PseudoSolution>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct PseudoSolution {
    selected_images: Vec<usize>,
    #[allow(dead_code)]
    cost: u64,
    #[allow(dead_code)]
    cloudy_area: u64,
    #[allow(dead_code)]
    max_incidence_angle: Option<u64>,
    #[allow(dead_code)]
    min_resolutions_sum: Option<u64>,
    timestamp_s: f64,
    phase: String,
    #[allow(dead_code)]
    index: usize,
}

/// Load pseudosolver solutions for a given instance, filtered by time budget.
fn load_pseudo_solutions(
    solutions_dir: &PathBuf,
    instance_name: &str,
    max_timestamp_s: f64,
) -> Vec<Vec<usize>> {
    let json_path = solutions_dir.join(format!("{instance_name}.json"));
    if !json_path.exists() {
        eprintln!(
            "  [warn] No pseudosolver data for {instance_name} at {}",
            json_path.display()
        );
        return Vec::new();
    }

    let content = std::fs::read_to_string(&json_path)
        .unwrap_or_else(|e| panic!("Cannot read pseudosolver file {}: {e}", json_path.display()));

    let data: PseudoSolverData = serde_json::from_str(&content).unwrap_or_else(|e| {
        panic!(
            "Cannot parse pseudosolver JSON {}: {e}",
            json_path.display()
        )
    });

    data.solutions
        .into_iter()
        .filter(|s| s.phase == "exact" && s.timestamp_s <= max_timestamp_s)
        .map(|s| s.selected_images)
        .collect()
}

/// Create an initial population from pseudosolver image-set lists.
fn build_seed_population(problem: &Problem, image_sets: &[Vec<usize>]) -> Archive {
    image_sets
        .iter()
        .map(|imgs| Solution::from_selected_images(imgs, problem))
        .collect()
}

/// Create a random initial population (for ratio 0:100).
fn build_random_population(problem: &Problem, pop_size: usize, seed: u64) -> Archive {
    (0..pop_size)
        .map(|i| Solution::random_with_seed(problem, seed + i as u64))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Algorithm result
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
struct BenchmarkResult {
    algo: String,
    algo_kind: String,
    ratio_exact: u32,
    ratio_pls: u32,
    instance: String,
    num_images: usize,
    num_elements: usize,
    seed_count: usize,
    archive_size: usize,
    wall_time: Duration,
    hypervolume: f64,
    objectives: Vec<[u64; NUM_OBJECTIVES]>,
    repetition: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
//  PLS runners
// ═══════════════════════════════════════════════════════════════════════════

fn run_exhaustive_pls(
    problem: &Problem,
    initial_pop: &Archive,
    timeout: Duration,
) -> (Archive, Duration) {
    // Force exhaustive residual enumeration
    set_runtime_probing_budget(PROBING_BUDGET_EXHAUSTIVE);

    let start = Instant::now();
    let mut pls = ParetoLocalSearch::new(problem, initial_pop, NEIGHBORHOOD_RANGE, false, PlsOptimizations::default());
    let archive = pls.run(usize::MAX, timeout);
    let wall = start.elapsed();
    (archive, wall)
}

fn run_probabilistic_pls(
    problem: &Problem,
    initial_pop: &Archive,
    timeout: Duration,
    budget: usize,
) -> (Archive, Duration) {
    // Use the specified (or default) probabilistic probing budget
    set_runtime_probing_budget(budget);

    let start = Instant::now();
    let mut pls = ParetoLocalSearch::new(problem, initial_pop, NEIGHBORHOOD_RANGE, false, PlsOptimizations::default());
    let archive = pls.run(usize::MAX, timeout);
    let wall = start.elapsed();
    (archive, wall)
}

fn run_concurrent_pls(
    problem: &Problem,
    initial_pop: &Archive,
    timeout: Duration,
    num_threads: usize,
) -> (Archive, Duration) {
    // Concurrent PLS uses exhaustive residual enumeration per worker
    set_runtime_probing_budget(PROBING_BUDGET_EXHAUSTIVE);

    let config = ConcurrentPLSConfig {
        num_threads,
        sync_interval_steps: 5,
        merge_interval: Duration::from_millis(100),
        boundary_threshold: 0.05,
        das_dennis_h: None,
        max_iterations: usize::MAX,
        max_duration: timeout,
        neighborhood_size_range: NEIGHBORHOOD_RANGE,
        is_deterministic: false,
        region_search_mode: RegionSearchMode::Unconstrained,
        scalarized_rho: 1e-3,
    };

    let start = Instant::now();
    let solver: ConcurrentPLS<'_, Solution, Problem, NUM_OBJECTIVES> =
        ConcurrentPLS::new(problem, config);
    let result = solver.solve(initial_pop);
    let wall = start.elapsed();
    (result.archive, wall)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Hypervolume (4-D normalised, same as algorithm_benchmark)
// ═══════════════════════════════════════════════════════════════════════════

fn bounds_from_objectives(objs: &[[u64; NUM_OBJECTIVES]]) -> [[u64; 2]; NUM_OBJECTIVES] {
    let mut bounds = [[u64::MAX, 0u64]; NUM_OBJECTIVES];
    for o in objs {
        for i in 0..NUM_OBJECTIVES {
            bounds[i][0] = bounds[i][0].min(o[i]);
            bounds[i][1] = bounds[i][1].max(o[i]);
        }
    }
    for b in &mut bounds {
        if b[0] == u64::MAX {
            *b = [0, 1];
        }
        if b[1] <= b[0] {
            b[1] = b[0] + 1;
        }
    }
    bounds
}

fn normalized_hv_4d(
    solutions: impl Iterator<Item = [u64; NUM_OBJECTIVES]>,
    bounds: &[[u64; 2]; NUM_OBJECTIVES],
) -> f64 {
    let ref_pt = [1.1f64; NUM_OBJECTIVES];
    let ref_volume: f64 = ref_pt.iter().product();
    let mut pts: Vec<Vec<f64>> = solutions
        .map(|o| {
            o.iter()
                .enumerate()
                .map(|(i, &v)| {
                    let lo = bounds[i][0] as f64;
                    let hi = bounds[i][1] as f64;
                    let range = (hi - lo).max(1.0);
                    ((v as f64 - lo) / range).clamp(0.0, 1.0)
                })
                .collect()
        })
        .collect();
    let raw = hv_4d(&mut pts, &ref_pt);
    raw / ref_volume
}

fn hv_4d(pts: &mut [Vec<f64>], r: &[f64; 4]) -> f64 {
    if pts.is_empty() {
        return 0.0;
    }
    pts.sort_by(|a, b| a[3].partial_cmp(&b[3]).unwrap());
    let mut total = 0.0;
    let mut prev = r[3];
    let mut i = pts.len();
    while i > 0 {
        i -= 1;
        let bound = pts[i][3];
        if bound < prev {
            let mut slice: Vec<Vec<f64>> =
                pts[..=i].iter().map(|p| vec![p[0], p[1], p[2]]).collect();
            total += (prev - bound) * hv_3d(&mut slice, &[r[0], r[1], r[2]]);
            prev = bound;
        }
        while i > 0 && (pts[i - 1][3] - bound).abs() < f64::EPSILON {
            i -= 1;
        }
    }
    total
}

fn hv_3d(pts: &mut [Vec<f64>], r: &[f64; 3]) -> f64 {
    if pts.is_empty() {
        return 0.0;
    }
    pts.sort_by(|a, b| a[2].partial_cmp(&b[2]).unwrap());
    let mut total = 0.0;
    let mut prev = r[2];
    let mut i = pts.len();
    while i > 0 {
        i -= 1;
        let bound = pts[i][2];
        if bound < prev {
            let mut slice: Vec<Vec<f64>> = pts[..=i].iter().map(|p| vec![p[0], p[1]]).collect();
            total += (prev - bound) * hv_2d(&mut slice, &[r[0], r[1]]);
            prev = bound;
        }
        while i > 0 && (pts[i - 1][2] - bound).abs() < f64::EPSILON {
            i -= 1;
        }
    }
    total
}

fn hv_2d(pts: &mut [Vec<f64>], r: &[f64; 2]) -> f64 {
    if pts.is_empty() {
        return 0.0;
    }
    pts.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
    let mut total = 0.0;
    let mut prev_y = r[1];
    for p in pts {
        if p[0] < r[0] {
            let curr_y = p[1];
            if curr_y < prev_y {
                let min_x = p[0];
                total += (prev_y - curr_y) * (r[0] - min_x);
                prev_y = curr_y;
            }
        }
    }
    total
}

// ═══════════════════════════════════════════════════════════════════════════
//  Adaptive timeout
// ═══════════════════════════════════════════════════════════════════════════

/// Per-instance-size total timeout calibrated so that the 50:50 ratio yields
/// ~10 exact-phase seeds on average across instances in each size tier,
/// leaving PLS a meaningful fraction of the Pareto front to discover.
///
/// Calibration data (from pseudosolver replay files, timestamp of 10th exact
/// solution per instance):
///
/// | Size | Median t10 |  Max t10 | Chosen total | Exact@50:50 | Avg seeds (% of front) |
/// |------|-----------|----------|-------------|-------------|------------------------|
/// |   30 |    0.13s  |   0.24s  |       10s   |      5.00s  | ~49 (70%)              |
/// |   50 |    0.61s  |   0.88s  |       20s   |     10.00s  | ~42 (32%)              |
/// |  100 |   24.06s  | 435.15s  |       50s   |        25s  | ~10 (4/5 inst)         |
/// |  150 |  127.86s  | 157.87s  |      260s   |       130s  | ~10-14                 |
/// |  200 |  162.22s* | 489.21s* |      600s   |       300s  | 1-4 (cap)              |
///
/// *Size-200 instances have only 1-4 total solutions; 10 seeds is unreachable.
///
/// Small instances (30, 50): the exact solver is extremely fast so even short
/// exact budgets pass more than 10 seeds, but the short total timeout ensures PLS
/// still has a meaningful fraction of the front left to discover.  PLS iterations
/// are cheap on small instances, so 5-15s of PLS time allows many iterations.
fn adaptive_timeout_for_size(num_images: usize) -> Duration {
    match num_images {
        0..=35 => Duration::from_secs(10),
        36..=60 => Duration::from_secs(20),
        61..=110 => Duration::from_secs(50),
        111..=160 => Duration::from_secs(260),
        _ => Duration::from_secs(600),
    }
}

fn size_tier(num_images: usize) -> &'static str {
    match num_images {
        0..=35 => "small",
        36..=60 => "medium",
        61..=110 => "large",
        111..=160 => "x-large",
        _ => "huge",
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Printing helpers
// ═══════════════════════════════════════════════════════════════════════════

fn ratio_label(exact: u32, pls: u32) -> String {
    format!("{exact}:{pls}")
}

fn print_instance_header(name: &str, num_images: usize, num_elements: usize, timeout: &Duration) {
    println!();
    println!(
        "━━━ {} (images={}, elements={}, timeout={}) ━━━",
        name,
        num_images,
        num_elements,
        humantime::format_duration(*timeout)
    );
}

fn print_ratio_table(results: &[BenchmarkResult]) {
    if results.is_empty() {
        return;
    }

    let name_width = results
        .iter()
        .map(|r| r.algo.len())
        .max()
        .unwrap_or(20)
        .max(25);

    println!(
        "  {:<nw$}  {:>7}  {:>6}  {:>10}  {:>10}  {:>8}",
        "Algorithm (ratio)",
        "Seeds",
        "Front",
        "HV",
        "Wall (ms)",
        "Rank",
        nw = name_width
    );
    println!(
        "  {:-<nw$}  {:-<7}  {:-<6}  {:-<10}  {:-<10}  {:-<8}",
        "",
        "",
        "",
        "",
        "",
        "",
        nw = name_width
    );

    // Rank by HV descending
    let mut indices: Vec<usize> = (0..results.len()).collect();
    indices.sort_by(|&a, &b| {
        results[b]
            .hypervolume
            .partial_cmp(&results[a].hypervolume)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut ranks = vec![0usize; results.len()];
    for (rank, &idx) in indices.iter().enumerate() {
        ranks[idx] = rank + 1;
    }

    for (i, r) in results.iter().enumerate() {
        let rank_str = if r.archive_size == 0 {
            "-".to_string()
        } else {
            format!("#{}", ranks[i])
        };
        println!(
            "  {:<nw$}  {:>7}  {:>6}  {:>10.6}  {:>10}  {:>8}",
            r.algo,
            r.seed_count,
            r.archive_size,
            r.hypervolume,
            r.wall_time.as_millis(),
            rank_str,
            nw = name_width
        );
    }
}

fn print_grand_summary(all_results: &[BenchmarkResult]) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                        GRAND SUMMARY (avg across instances)                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Group by (algo_kind, ratio)
    let mut groups: HashMap<(String, String), Vec<&BenchmarkResult>> = HashMap::new();
    for r in all_results {
        let key = (r.algo_kind.clone(), ratio_label(r.ratio_exact, r.ratio_pls));
        groups.entry(key).or_default().push(r);
    }

    // Sorted display
    let mut keys: Vec<_> = groups.keys().cloned().collect();
    keys.sort();

    let nw = 30;
    println!(
        "  {:<nw$}  {:>7}  {:>10}  {:>10}  {:>10}  {:>6}",
        "Algorithm (ratio)",
        "Avg Frt",
        "Avg HV",
        "Avg ms",
        "Avg Seeds",
        "N",
        nw = nw
    );
    println!(
        "  {:-<nw$}  {:-<7}  {:-<10}  {:-<10}  {:-<10}  {:-<6}",
        "",
        "",
        "",
        "",
        "",
        "",
        nw = nw
    );

    for (algo_kind, ratio) in &keys {
        let entries = &groups[&(algo_kind.clone(), ratio.clone())];
        let n = entries.len() as f64;
        let avg_hv: f64 = entries.iter().map(|r| r.hypervolume).sum::<f64>() / n;
        let avg_front: f64 = entries.iter().map(|r| r.archive_size as f64).sum::<f64>() / n;
        let avg_ms: f64 = entries
            .iter()
            .map(|r| r.wall_time.as_millis() as f64)
            .sum::<f64>()
            / n;
        let avg_seeds: f64 = entries.iter().map(|r| r.seed_count as f64).sum::<f64>() / n;

        let label = format!("{} ({})", algo_kind, ratio);
        println!(
            "  {:<nw$}  {:>7.1}  {:>10.6}  {:>10.0}  {:>10.1}  {:>6}",
            label,
            avg_front,
            avg_hv,
            avg_ms,
            avg_seeds,
            entries.len(),
            nw = nw
        );
    }

    // Also print per-algorithm averages (across all ratios)
    println!();
    println!("  Per-algorithm averages (all ratios):");
    println!(
        "  {:<nw$}  {:>7}  {:>10}  {:>10}",
        "Algorithm",
        "Avg Frt",
        "Avg HV",
        "Avg ms",
        nw = 25
    );
    println!(
        "  {:-<nw$}  {:-<7}  {:-<10}  {:-<10}",
        "",
        "",
        "",
        "",
        nw = 25
    );

    let mut algo_groups: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
    for r in all_results {
        algo_groups.entry(r.algo_kind.clone()).or_default().push(r);
    }
    let mut algo_keys: Vec<_> = algo_groups.keys().cloned().collect();
    algo_keys.sort();
    for algo in &algo_keys {
        let entries = &algo_groups[algo];
        let n = entries.len() as f64;
        let avg_hv: f64 = entries.iter().map(|r| r.hypervolume).sum::<f64>() / n;
        let avg_front: f64 = entries.iter().map(|r| r.archive_size as f64).sum::<f64>() / n;
        let avg_ms: f64 = entries
            .iter()
            .map(|r| r.wall_time.as_millis() as f64)
            .sum::<f64>()
            / n;
        println!(
            "  {:<nw$}  {:>7.1}  {:>10.6}  {:>10.0}",
            algo,
            avg_front,
            avg_hv,
            avg_ms,
            nw = 25
        );
    }

    // Per-ratio averages (across all algorithms)
    println!();
    println!("  Per-ratio averages (all algorithms):");
    println!(
        "  {:<nw$}  {:>7}  {:>10}  {:>10}",
        "Ratio",
        "Avg Frt",
        "Avg HV",
        "Avg ms",
        nw = 10
    );
    println!(
        "  {:-<nw$}  {:-<7}  {:-<10}  {:-<10}",
        "",
        "",
        "",
        "",
        nw = 10
    );

    let mut ratio_groups: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
    for r in all_results {
        ratio_groups
            .entry(ratio_label(r.ratio_exact, r.ratio_pls))
            .or_default()
            .push(r);
    }
    let ratio_order = ["100:0", "50:50", "25:75", "0:100"];
    for ratio in &ratio_order {
        if let Some(entries) = ratio_groups.get(*ratio) {
            let n = entries.len() as f64;
            let avg_hv: f64 = entries.iter().map(|r| r.hypervolume).sum::<f64>() / n;
            let avg_front: f64 = entries.iter().map(|r| r.archive_size as f64).sum::<f64>() / n;
            let avg_ms: f64 = entries
                .iter()
                .map(|r| r.wall_time.as_millis() as f64)
                .sum::<f64>()
                / n;
            println!(
                "  {:<nw$}  {:>7.1}  {:>10.6}  {:>10.0}",
                ratio,
                avg_front,
                avg_hv,
                avg_ms,
                nw = 10
            );
        }
    }
}

fn print_size_tier_summary(all_results: &[BenchmarkResult]) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                        SUMMARY BY INSTANCE SIZE TIER                        ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Group by (size_tier, algo_kind, ratio)
    let mut groups: HashMap<(String, String, String), Vec<&BenchmarkResult>> = HashMap::new();
    for r in all_results {
        let tier = size_tier(r.num_images).to_string();
        let key = (
            tier,
            r.algo_kind.clone(),
            ratio_label(r.ratio_exact, r.ratio_pls),
        );
        groups.entry(key).or_default().push(r);
    }

    let tiers = ["small", "medium", "large", "x-large", "huge"];
    let ratio_order = ["100:0", "50:50", "25:75", "0:100"];

    for tier in &tiers {
        // Check if any results exist for this tier
        let has_data = groups.keys().any(|(t, _, _)| t == *tier);
        if !has_data {
            continue;
        }

        println!("  ── {} instances ──", tier);
        let nw = 30;
        println!(
            "    {:<nw$}  {:>10}  {:>7}  {:>10}",
            "Algorithm (ratio)",
            "Avg HV",
            "Avg Frt",
            "Avg ms",
            nw = nw
        );
        println!(
            "    {:-<nw$}  {:-<10}  {:-<7}  {:-<10}",
            "",
            "",
            "",
            "",
            nw = nw
        );

        for algo in &ALL_ALGOS {
            for ratio in &ratio_order {
                let key = (
                    tier.to_string(),
                    algo.short().to_string(),
                    ratio.to_string(),
                );
                if let Some(entries) = groups.get(&key) {
                    let n = entries.len() as f64;
                    let avg_hv: f64 = entries.iter().map(|r| r.hypervolume).sum::<f64>() / n;
                    let avg_front: f64 =
                        entries.iter().map(|r| r.archive_size as f64).sum::<f64>() / n;
                    let avg_ms: f64 = entries
                        .iter()
                        .map(|r| r.wall_time.as_millis() as f64)
                        .sum::<f64>()
                        / n;
                    let label = format!("{} ({})", algo.label(), ratio);
                    println!(
                        "    {:<nw$}  {:>10.6}  {:>7.1}  {:>10.0}",
                        label,
                        avg_hv,
                        avg_front,
                        avg_ms,
                        nw = nw
                    );
                }
            }
        }
        println!();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  JSON output
// ═══════════════════════════════════════════════════════════════════════════

/// Append one result as a JSON line to the checkpoint JSONL file.
/// Each line is a self-contained JSON object — safe to read even after a crash.
fn append_jsonl_result(path: &PathBuf, r: &BenchmarkResult) {
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|e| panic!("Cannot open JSONL checkpoint file {}: {e}", path.display()));

    // Build compact single-line JSON for this result
    let objs_str: String = r
        .objectives
        .iter()
        .map(|o| format!("[{},{},{},{}]", o[0], o[1], o[2], o[3]))
        .collect::<Vec<_>>()
        .join(",");

    writeln!(
        f,
        "{{\"instance\":\"{}\",\"num_images\":{},\"num_elements\":{},\"size_tier\":\"{}\",\
         \"algorithm\":\"{}\",\"algorithm_kind\":\"{}\",\
         \"ratio_exact\":{},\"ratio_pls\":{},\"ratio_label\":\"{}\",\
         \"seed_count\":{},\"archive_size\":{},\"hypervolume\":{},\
         \"wall_time_ms\":{},\"repetition\":{},\"objectives\":[{}]}}",
        r.instance,
        r.num_images,
        r.num_elements,
        size_tier(r.num_images),
        r.algo,
        r.algo_kind,
        r.ratio_exact,
        r.ratio_pls,
        ratio_label(r.ratio_exact, r.ratio_pls),
        r.seed_count,
        r.archive_size,
        r.hypervolume,
        r.wall_time.as_millis(),
        r.repetition,
        objs_str,
    )
    .unwrap();
}

/// Flush a batch of results (one instance's worth) to the JSONL checkpoint.
fn flush_instance_results(jsonl_path: &PathBuf, results: &[BenchmarkResult]) {
    for r in results {
        append_jsonl_result(jsonl_path, r);
    }
    println!(
        "  [checkpoint] {} result(s) appended to {}",
        results.len(),
        jsonl_path.display()
    );
}

fn write_json_results(path: &PathBuf, results: &[BenchmarkResult], args: &Cli) {
    let mut f = std::fs::File::create(path).expect("Cannot create JSON output file");

    writeln!(f, "{{").unwrap();
    writeln!(f, "  \"benchmark\": \"hybrid_two_phase\",").unwrap();
    writeln!(f, "  \"timeout_ms\": {},", args.timeout.as_millis()).unwrap();
    writeln!(f, "  \"adaptive_timeout\": {},", args.adaptive_timeout).unwrap();
    writeln!(f, "  \"population_size\": {},", args.pop_size).unwrap();
    writeln!(f, "  \"seed\": {},", args.seed).unwrap();
    writeln!(f, "  \"repeats\": {},", args.repeats).unwrap();
    writeln!(f, "  \"threads\": {},", args.threads).unwrap();
    writeln!(f, "  \"probing_budget\": {},", args.probing_budget).unwrap();
    writeln!(
        f,
        "  \"ratios\": [{}],",
        RATIOS
            .iter()
            .map(|(e, p)| format!("[{e},{p}]"))
            .collect::<Vec<_>>()
            .join(", ")
    )
    .unwrap();
    writeln!(
        f,
        "  \"algorithms\": [\"{}\", \"{}\", \"{}\"],",
        AlgoKind::ExhaustivePls.short(),
        AlgoKind::ConcurrentPls.short(),
        AlgoKind::ProbabilisticPls.short(),
    )
    .unwrap();
    writeln!(f, "  \"results\": [").unwrap();

    for (i, r) in results.iter().enumerate() {
        writeln!(f, "    {{").unwrap();
        writeln!(f, "      \"instance\": \"{}\",", r.instance).unwrap();
        writeln!(f, "      \"num_images\": {},", r.num_images).unwrap();
        writeln!(f, "      \"num_elements\": {},", r.num_elements).unwrap();
        writeln!(f, "      \"size_tier\": \"{}\",", size_tier(r.num_images)).unwrap();
        writeln!(f, "      \"algorithm\": \"{}\",", r.algo).unwrap();
        writeln!(f, "      \"algorithm_kind\": \"{}\",", r.algo_kind).unwrap();
        writeln!(f, "      \"ratio_exact\": {},", r.ratio_exact).unwrap();
        writeln!(f, "      \"ratio_pls\": {},", r.ratio_pls).unwrap();
        writeln!(
            f,
            "      \"ratio_label\": \"{}\",",
            ratio_label(r.ratio_exact, r.ratio_pls)
        )
        .unwrap();
        writeln!(f, "      \"seed_count\": {},", r.seed_count).unwrap();
        writeln!(f, "      \"archive_size\": {},", r.archive_size).unwrap();
        writeln!(f, "      \"hypervolume\": {},", r.hypervolume).unwrap();
        writeln!(f, "      \"wall_time_ms\": {},", r.wall_time.as_millis()).unwrap();
        writeln!(f, "      \"repetition\": {},", r.repetition).unwrap();

        // Objective vectors
        write!(f, "      \"objectives\": [").unwrap();
        for (oi, obj) in r.objectives.iter().enumerate() {
            if oi > 0 {
                write!(f, ", ").unwrap();
            }
            write!(f, "[{}, {}, {}, {}]", obj[0], obj[1], obj[2], obj[3]).unwrap();
        }
        writeln!(f, "]").unwrap();

        let comma = if i + 1 < results.len() { "," } else { "" };
        writeln!(f, "    }}{comma}").unwrap();
    }

    writeln!(f, "  ]").unwrap();
    writeln!(f, "}}").unwrap();

    println!("\n  JSON results written to {}", path.display());
}

// ═══════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_lines)]
fn main() {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .compact()
        .with_max_level(tracing::Level::WARN)
        .init();

    let args = Cli::parse();

    let num_threads = if args.threads == 0 {
        num_cpus::get_physical()
    } else {
        args.threads
    };

    // Parse skip list
    let skip_set: Vec<String> = args
        .skip_algo
        .as_deref()
        .unwrap_or("")
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();

    let active_algos: Vec<AlgoKind> = ALL_ALGOS
        .iter()
        .filter(|a| !skip_set.contains(&a.short().to_string()))
        .copied()
        .collect();

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║          SIMS Hybrid Two-Phase Algorithm Benchmark                          ║");
    println!("║  Exhaustive PLS vs Concurrent PLS vs Probabilistic Probing PLS              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  instances dir  : {}", args.instances_dir.display());
    println!("  solutions dir  : {}", args.solutions_dir.display());
    println!(
        "  base timeout   : {}{}",
        humantime::format_duration(args.timeout),
        if args.adaptive_timeout {
            " (adaptive override per instance)"
        } else {
            ""
        }
    );
    println!("  population     : {}", args.pop_size);
    println!("  seed           : {}", args.seed);
    println!("  repeats        : {}", args.repeats);
    println!(
        "  threads        : {} (physical CPUs: {})",
        num_threads,
        num_cpus::get_physical()
    );
    println!(
        "  probing budget : {}",
        if args.probing_budget == 0 {
            "default (1000)".to_string()
        } else {
            args.probing_budget.to_string()
        }
    );
    println!(
        "  ratios         : {}",
        RATIOS
            .iter()
            .map(|(e, p)| format!("{e}:{p}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "  algorithms     : {}",
        active_algos
            .iter()
            .map(|a| a.label())
            .collect::<Vec<_>>()
            .join(", ")
    );
    if !skip_set.is_empty() {
        println!("  skipped        : {}", skip_set.join(", "));
    }
    if let Some(ref f) = args.filter {
        println!("  filter         : {f}");
    }
    if let Some(ref mwt) = args.max_wall_time {
        println!("  max wall time  : {}", humantime::format_duration(*mwt));
    }
    println!();

    // Collect .dzn files
    let mut instance_paths: Vec<PathBuf> = std::fs::read_dir(&args.instances_dir)
        .expect("Cannot read instances directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "dzn"))
        .collect();

    instance_paths.sort();

    if let Some(ref filter) = args.filter {
        instance_paths.retain(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map_or(false, |n| n.contains(filter.as_str()))
        });
    }

    if instance_paths.is_empty() {
        eprintln!("No .dzn files found in {}", args.instances_dir.display());
        std::process::exit(1);
    }

    println!(
        "Found {} instance(s). Running benchmark...",
        instance_paths.len()
    );

    // Derive the JSONL checkpoint path from the JSON output path (or use a default)
    let jsonl_path: PathBuf = args
        .json_output
        .as_ref()
        .map(|p| p.with_extension("jsonl"))
        .unwrap_or_else(|| PathBuf::from("results/hybrid_benchmark_checkpoint.jsonl"));

    // Create parent directories for checkpoint
    if let Some(parent) = jsonl_path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).ok();
        }
    }

    // Truncate the JSONL file at start (fresh run)
    std::fs::File::create(&jsonl_path).unwrap_or_else(|e| {
        panic!(
            "Cannot create JSONL checkpoint {}: {e}",
            jsonl_path.display()
        )
    });
    println!(
        "  JSONL checkpoint : {} (results saved after each instance)",
        jsonl_path.display()
    );
    println!();

    let benchmark_start = Instant::now();

    let mut all_results: Vec<BenchmarkResult> = Vec::new();
    let total_configs = instance_paths.len() * RATIOS.len() * active_algos.len() * args.repeats;
    let mut configs_done = 0usize;
    let mut instances_completed = 0usize;

    for path in &instance_paths {
        // Check wall-time limit before starting next instance
        if let Some(max_wall) = args.max_wall_time {
            let elapsed = benchmark_start.elapsed();
            if elapsed >= max_wall {
                println!();
                println!(
                    "⏱  Max wall time reached ({} elapsed, limit {}). Stopping gracefully.",
                    humantime::format_duration(elapsed),
                    humantime::format_duration(max_wall),
                );
                println!(
                    "   Completed {}/{} instances. Results saved to {}",
                    instances_completed,
                    instance_paths.len(),
                    jsonl_path.display(),
                );
                break;
            }
            let remaining = max_wall.saturating_sub(elapsed);
            println!(
                "  [wall time: {} elapsed, {} remaining]",
                humantime::format_duration(elapsed),
                humantime::format_duration(remaining),
            );
        }

        let instance_name = path
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("?")
            .to_string();

        let problem = match Problem::from_minizinc_datafile(path, OBJECTIVE_TYPES) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("  [!] Failed to load {}: {e}", path.display());
                continue;
            }
        };

        let num_images = problem.num_images();
        let num_elements = problem.num_elements();

        let instance_timeout = if args.adaptive_timeout {
            adaptive_timeout_for_size(num_images)
        } else {
            args.timeout
        };

        print_instance_header(&instance_name, num_images, num_elements, &instance_timeout);

        // Pre-load ALL pseudosolver solutions for this instance (max timestamp = infinity)
        // We'll filter by ratio later
        let all_pseudo_solutions =
            load_pseudo_solutions(&args.solutions_dir, &instance_name, f64::MAX);
        println!(
            "  Pseudosolver solutions available: {}",
            all_pseudo_solutions.len()
        );

        // Collect results for this instance (across all ratios, algos, repeats)
        let mut instance_results: Vec<BenchmarkResult> = Vec::new();

        for &(exact_pct, pls_pct) in &RATIOS {
            let exact_timeout_s = instance_timeout.as_secs_f64() * (exact_pct as f64 / 100.0);
            let pls_timeout =
                Duration::from_secs_f64(instance_timeout.as_secs_f64() * (pls_pct as f64 / 100.0));

            // Load pseudosolver solutions filtered by exact-phase time budget
            let seed_solutions = if exact_pct == 0 {
                Vec::new()
            } else {
                load_pseudo_solutions(&args.solutions_dir, &instance_name, exact_timeout_s)
            };

            let seed_pop = build_seed_population(&problem, &seed_solutions);
            let seed_count = seed_pop.len();

            println!(
                "\n  Ratio {}:{} → exact budget={:.1}s, PLS budget={:.1}s, seeds={}",
                exact_pct,
                pls_pct,
                exact_timeout_s,
                pls_timeout.as_secs_f64(),
                seed_count
            );

            for algo in &active_algos {
                for rep in 0..args.repeats {
                    let rep_seed = args.seed + rep as u64 * 1000;
                    configs_done += 1;

                    // Build initial population for this run
                    let initial_pop: Archive = if exact_pct == 0 || seed_count == 0 {
                        // No exact-phase seeds → random initial population
                        build_random_population(&problem, args.pop_size, rep_seed)
                    } else if pls_pct == 0 {
                        // 100% exact, 0% PLS → just the seed population, no PLS
                        seed_pop.clone()
                    } else {
                        // Hybrid: merge seeds with some random solutions if seed count is small
                        if seed_count < 5 {
                            // Add random solutions to beef up the initial population
                            let mut pop = seed_pop.clone();
                            for i in 0..args.pop_size.saturating_sub(seed_count) {
                                let random_sol = Solution::random_with_seed(
                                    &problem,
                                    rep_seed + 10000 + i as u64,
                                );
                                pop.try_insert(&random_sol);
                            }
                            pop
                        } else {
                            seed_pop.clone()
                        }
                    };

                    let (archive, wall_time) = if pls_pct == 0 {
                        // No PLS phase — just return the initial population as-is
                        let start = Instant::now();
                        let wall = start.elapsed();
                        (initial_pop.clone(), wall)
                    } else {
                        match algo {
                            AlgoKind::ExhaustivePls => {
                                run_exhaustive_pls(&problem, &initial_pop, pls_timeout)
                            }
                            AlgoKind::ConcurrentPls => {
                                run_concurrent_pls(&problem, &initial_pop, pls_timeout, num_threads)
                            }
                            AlgoKind::ProbabilisticPls => run_probabilistic_pls(
                                &problem,
                                &initial_pop,
                                pls_timeout,
                                args.probing_budget,
                            ),
                        }
                    };

                    let objectives: Vec<[u64; NUM_OBJECTIVES]> =
                        archive.iter().map(|s| *s.objectives()).collect();

                    let algo_label = if pls_pct == 0 {
                        format!("Seeds only ({}:{})", exact_pct, pls_pct)
                    } else {
                        format!("{} ({}:{})", algo.label(), exact_pct, pls_pct)
                    };

                    let result = BenchmarkResult {
                        algo: algo_label,
                        algo_kind: if pls_pct == 0 {
                            "seeds_only".to_string()
                        } else {
                            algo.short().to_string()
                        },
                        ratio_exact: exact_pct,
                        ratio_pls: pls_pct,
                        instance: instance_name.clone(),
                        num_images,
                        num_elements,
                        seed_count,
                        archive_size: archive.len(),
                        wall_time,
                        hypervolume: 0.0, // computed below with shared bounds
                        objectives,
                        repetition: rep,
                    };

                    instance_results.push(result);

                    print!(
                        "\r  [{configs_done}/{total_configs}] {} rep={} → front={}, {:.0}ms",
                        if pls_pct == 0 {
                            format!("Seeds only ({}:{})", exact_pct, pls_pct)
                        } else {
                            format!("{} ({}:{})", algo.label(), exact_pct, pls_pct)
                        },
                        rep,
                        archive.len(),
                        wall_time.as_secs_f64() * 1000.0,
                    );
                    std::io::stdout().flush().ok();

                    // For ratio (100:0), all algorithms produce the same result (seeds only),
                    // so we only need to run once
                    if pls_pct == 0 {
                        // Duplicate for remaining repeats
                        for extra_rep in (rep + 1)..args.repeats {
                            configs_done += 1;
                            let mut dup = instance_results.last().unwrap().clone();
                            dup.repetition = extra_rep;
                            instance_results.push(dup);
                        }
                        break; // break repeats loop
                    }
                }

                // For ratio (100:0), also break the algo loop (only one "seeds only" run needed)
                if pls_pct == 0 {
                    break;
                }
            }
        }

        println!();

        // Compute HV with shared bounds across all results for this instance
        let all_objs: Vec<[u64; NUM_OBJECTIVES]> = instance_results
            .iter()
            .flat_map(|r| r.objectives.iter().copied())
            .collect();

        if !all_objs.is_empty() {
            let hv_bounds = bounds_from_objectives(&all_objs);
            for r in &mut instance_results {
                r.hypervolume = normalized_hv_4d(r.objectives.iter().copied(), &hv_bounds);
            }
        }

        // Aggregate repeats: average per (algo, ratio)
        let aggregated = aggregate_repeats(&instance_results, args.repeats);

        println!();
        print_ratio_table(&aggregated);

        // ── Flush to JSONL checkpoint (crash-safe) ──────────────────
        flush_instance_results(&jsonl_path, &aggregated);

        all_results.extend(aggregated);
        instances_completed += 1;
    }

    // ── Grand summary ───────────────────────────────────────────────────
    print_grand_summary(&all_results);
    print_size_tier_summary(&all_results);

    // ── JSON export ─────────────────────────────────────────────────────
    if let Some(ref json_path) = args.json_output {
        // Create parent directories if needed
        if let Some(parent) = json_path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent).ok();
            }
        }
        write_json_results(json_path, &all_results, &args);
    }

    println!();
    println!("Benchmark complete.");
    println!();
}

/// Aggregate repeated runs: average per (algo_kind, ratio_exact, ratio_pls, instance).
fn aggregate_repeats(results: &[BenchmarkResult], repeats: usize) -> Vec<BenchmarkResult> {
    if repeats <= 1 {
        return results.to_vec();
    }

    // Group by key
    let mut groups: HashMap<(String, u32, u32, String), Vec<&BenchmarkResult>> = HashMap::new();
    for r in results {
        let key = (
            r.algo_kind.clone(),
            r.ratio_exact,
            r.ratio_pls,
            r.instance.clone(),
        );
        groups.entry(key).or_default().push(r);
    }

    let mut aggregated: Vec<BenchmarkResult> = Vec::new();

    for (_, entries) in &groups {
        if entries.is_empty() {
            continue;
        }
        let n = entries.len() as f64;
        let first = entries[0];

        let avg_hv = entries.iter().map(|r| r.hypervolume).sum::<f64>() / n;
        let avg_front = (entries.iter().map(|r| r.archive_size).sum::<usize>() as f64 / n) as usize;
        let avg_ms = entries
            .iter()
            .map(|r| r.wall_time.as_millis())
            .sum::<u128>()
            / entries.len() as u128;
        let avg_seeds = (entries.iter().map(|r| r.seed_count).sum::<usize>() as f64 / n) as usize;

        // Collect all objective vectors from all repeats
        let all_objs: Vec<[u64; NUM_OBJECTIVES]> = entries
            .iter()
            .flat_map(|r| r.objectives.iter().copied())
            .collect();

        let label = if first.ratio_pls == 0 {
            format!(
                "Seeds only ({}:{}) [avg of {}]",
                first.ratio_exact,
                first.ratio_pls,
                entries.len()
            )
        } else {
            format!(
                "{} ({}:{}) [avg of {}]",
                first.algo_kind,
                first.ratio_exact,
                first.ratio_pls,
                entries.len()
            )
        };

        aggregated.push(BenchmarkResult {
            algo: label,
            algo_kind: first.algo_kind.clone(),
            ratio_exact: first.ratio_exact,
            ratio_pls: first.ratio_pls,
            instance: first.instance.clone(),
            num_images: first.num_images,
            num_elements: first.num_elements,
            seed_count: avg_seeds,
            archive_size: avg_front,
            wall_time: Duration::from_millis(avg_ms as u64),
            hypervolume: avg_hv,
            objectives: all_objs,
            repetition: 0,
        });
    }

    // Sort for consistent display: by ratio, then algo kind
    aggregated.sort_by(|a, b| {
        a.ratio_exact
            .cmp(&b.ratio_exact)
            .reverse()
            .then(a.algo_kind.cmp(&b.algo_kind))
    });

    aggregated
}
