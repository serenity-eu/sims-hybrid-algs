/// Comparative benchmark: all multi-objective SIMS solvers under the same wall-clock budget.
///
/// Runs PLS, custom NSGA-II, custom MOEA/D, moors NSGA-II, moors SPEA-2,
/// optirustic NSGA-II, and optirustic NSGA-III on real `.dzn` instances,
/// then compares hypervolume indicator and archive size.
///
/// Usage:
///   cargo run --release --bin algorithm-benchmark --features external_solvers -- \
///     --instances-dir tests/data --timeout 10s
///
/// The `--filter` flag selects instances by filename substring (e.g. `_30` for small).
use std::{
    io::Write,
    path::PathBuf,
    time::{Duration, Instant},
};

use clap::{Parser, ValueEnum};
use pareto::{HasObjectives, ParetoFront};
use pls::{
    PlsOptimizations,
    evolutionary::{
        memetic::{EaBackend, MemeticAlgorithm, MemeticConfig},
        moead::{Moead, MoeadConfig},
        moors_adapter::{MoorsConfig, MoorsCrossoverType, run_moors_nsga2, run_moors_spea2},
        nsga2::{Nsga2, Nsga2Config},
        optirustic_adapter::{OptirusticConfig, run_optirustic_nsga2, run_optirustic_nsga3},
    },
    objectives::ObjectiveType,
    pareto_local_search::ParetoLocalSearch,
    pls_config::SolutionSelectionMode,
    problem_bitset::ProblemBitset,
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

const SEED: u64 = 42;
const POP_SIZE: usize = 100;
const NEIGHBORHOOD_RANGE: std::ops::RangeInclusive<u32> = 1..=6;

// ═══════════════════════════════════════════════════════════════════════════
//  CLI
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliPlsAblationMode {
    Default,
    Baseline,
    DiverseProbe,
    #[cfg(feature = "scalarized_selection")]
    ScalarizedChebycheff,
    #[cfg(feature = "scalarized_selection")]
    DiverseThenScalarizedChebycheff,
}

impl CliPlsAblationMode {
    const fn to_runtime(self) -> Option<SolutionSelectionMode> {
        match self {
            Self::Default | Self::Baseline => None,
            Self::DiverseProbe => Some(SolutionSelectionMode::DiverseProbe),
            #[cfg(feature = "scalarized_selection")]
            Self::ScalarizedChebycheff => Some(SolutionSelectionMode::ScalarizedChebycheff),
            #[cfg(feature = "scalarized_selection")]
            Self::DiverseThenScalarizedChebycheff => {
                Some(SolutionSelectionMode::DiverseThenScalarizedChebycheff)
            }
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Default => "PLS (default)",
            Self::Baseline => "PLS (baseline)",
            Self::DiverseProbe => "PLS (diverse probe)",
            #[cfg(feature = "scalarized_selection")]
            Self::ScalarizedChebycheff => "PLS (scalarized Chebycheff)",
            #[cfg(feature = "scalarized_selection")]
            Self::DiverseThenScalarizedChebycheff => "PLS (diverse + scalarized Chebycheff)",
        }
    }
}

#[derive(Parser)]
#[command(
    name = "algorithm-benchmark",
    about = "Compare all SIMS multi-objective solvers under the same time budget"
)]
struct Cli {
    /// Directory containing .dzn problem instance files
    #[arg(short = 'i', long = "instances-dir", default_value = "tests/data")]
    instances_dir: PathBuf,

    /// Per-instance time budget (e.g. 10s, 30s, 1m)
    #[arg(
        short = 't',
        long = "timeout",
        value_parser = humantime::parse_duration,
        default_value = "10s"
    )]
    timeout: Duration,

    /// Filter instances by filename substring (e.g. "_30" for small)
    #[arg(short = 'f', long = "filter")]
    filter: Option<String>,

    /// Population size for all population-based algorithms
    #[arg(long = "pop-size", default_value_t = POP_SIZE)]
    pop_size: usize,

    /// Number of repetitions per algorithm/instance pair (results are averaged)
    #[arg(long = "repeats", default_value_t = 1)]
    repeats: usize,

    /// Random seed
    #[arg(long = "seed", default_value_t = SEED)]
    seed: u64,

    /// Skip PLS (useful when only comparing evolutionary algorithms)
    #[arg(long = "skip-pls")]
    skip_pls: bool,

    /// PLS ablation modes to include in the benchmark.
    /// Repeat the flag to compare multiple PLS variants in the same HV experiment.
    #[arg(
        long = "pls-ablation",
        value_enum,
        num_args = 1..,
        action = clap::ArgAction::Append
    )]
    pls_ablation: Vec<CliPlsAblationMode>,

    /// Number of parent solutions to probe when using diverse probing
    #[arg(long = "diverse-probe-budget")]
    diverse_probe_budget: Option<usize>,

    #[cfg(feature = "scalarized_selection")]
    /// Maximum number of parent solutions selected per step by scalarized selection
    #[arg(long = "scalarized-parent-budget")]
    scalarized_parent_budget: Option<usize>,

    #[cfg(feature = "scalarized_selection")]
    /// Number of random weight vectors sampled per step for scalarized selection
    #[arg(long = "scalarized-weight-samples", default_value_t = 1)]
    scalarized_weight_samples: usize,

    #[cfg(feature = "scalarized_selection")]
    /// Augmentation coefficient for weighted Chebycheff scalarization
    #[arg(long = "scalarized-rho", default_value_t = 1e-3)]
    scalarized_rho: f64,

    /// Skip external solvers (moors, optirustic) — run only custom algorithms
    #[arg(long = "skip-external")]
    skip_external: bool,

    /// Skip memetic hybrids
    #[arg(long = "skip-memetic")]
    skip_memetic: bool,

    /// PLS time fraction for memetic hybrids (0.0–1.0)
    #[arg(long = "pls-fraction", default_value_t = 0.3)]
    pls_fraction: f64,

    /// Write machine-readable JSON results to this file (for plotting scripts)
    #[arg(long = "json-output")]
    json_output: Option<PathBuf>,
}

// ═══════════════════════════════════════════════════════════════════════════
//  Algorithm result
// ═══════════════════════════════════════════════════════════════════════════

struct AlgorithmResult {
    name: String,
    archive_size: usize,
    wall_time: Duration,
    hypervolume: f64,
    objectives: Vec<[u64; NUM_OBJECTIVES]>,
}

// ═══════════════════════════════════════════════════════════════════════════
//  Runner functions
// ═══════════════════════════════════════════════════════════════════════════

fn run_pls(
    problem: &Problem,
    timeout: Duration,
    pop_size: usize,
    seed: u64,
    mode: CliPlsAblationMode,
    diverse_probe_budget: Option<usize>,
    #[cfg(feature = "scalarized_selection")] scalarized_parent_budget: Option<usize>,
    #[cfg(feature = "scalarized_selection")] scalarized_weight_samples: usize,
    #[cfg(feature = "scalarized_selection")] scalarized_rho: f64,
) -> AlgorithmResult {
    let initial_pop: Archive = (0..pop_size)
        .map(|i| Solution::random_with_seed(problem, seed + i as u64))
        .collect();

    let mut optimizations = match mode {
        CliPlsAblationMode::Baseline => PlsOptimizations::baseline(),
        _ => PlsOptimizations::default(),
    };

    if let Some(selection_mode) = mode.to_runtime() {
        optimizations.solution_selection_mode = selection_mode;
    }

    optimizations.use_diverse_probing = matches!(mode, CliPlsAblationMode::DiverseProbe);
    optimizations.diverse_probe_budget = diverse_probe_budget;

    #[cfg(feature = "scalarized_selection")]
    {
        optimizations.scalarized_parent_budget = scalarized_parent_budget;
        optimizations.scalarized_weight_samples = scalarized_weight_samples;
        optimizations.scalarized_rho = scalarized_rho;
    }

    let start = Instant::now();
    let mut pls = ParetoLocalSearch::new(
        problem,
        &initial_pop,
        NEIGHBORHOOD_RANGE,
        false,
        optimizations,
    );
    let archive = pls.run(usize::MAX, timeout);
    let wall = start.elapsed();

    let objectives: Vec<[u64; NUM_OBJECTIVES]> = archive.iter().map(|s| *s.objectives()).collect();
    AlgorithmResult {
        name: mode.label().to_string(),
        archive_size: archive.len(),
        wall_time: wall,
        hypervolume: 0.0,
        objectives,
    }
}

fn run_custom_nsga2(
    problem: &Problem,
    timeout: Duration,
    pop_size: usize,
    seed: u64,
) -> AlgorithmResult {
    let config = Nsga2Config {
        population_size: pop_size,
        ..Default::default()
    };

    let start = Instant::now();
    let mut nsga2 = Nsga2::<Problem, NUM_OBJECTIVES>::new(problem, config, None, seed);
    let archive = nsga2.run(usize::MAX, timeout);
    let wall = start.elapsed();

    let objectives: Vec<[u64; NUM_OBJECTIVES]> = archive.iter().map(|s| *s.objectives()).collect();
    AlgorithmResult {
        name: "NSGA-II (custom)".to_string(),
        archive_size: archive.len(),
        wall_time: wall,
        hypervolume: 0.0,
        objectives,
    }
}

fn run_custom_moead(
    problem: &Problem,
    timeout: Duration,
    _pop_size: usize,
    seed: u64,
) -> AlgorithmResult {
    let config = MoeadConfig {
        num_divisions: 12,
        auto_divisions: false,
        ..Default::default()
    };

    let start = Instant::now();
    let mut moead = Moead::<Problem, NUM_OBJECTIVES>::new(problem, config, None, seed);
    let archive = moead.run(usize::MAX, timeout);
    let wall = start.elapsed();

    let objectives: Vec<[u64; NUM_OBJECTIVES]> = archive.iter().map(|s| *s.objectives()).collect();
    AlgorithmResult {
        name: "MOEA/D (custom)".to_string(),
        archive_size: archive.len(),
        wall_time: wall,
        hypervolume: 0.0,
        objectives,
    }
}

// ── Memetic hybrids ─────────────────────────────────────────────────────

fn run_memetic_nsga2(
    problem: &Problem,
    timeout: Duration,
    pop_size: usize,
    seed: u64,
    pls_fraction: f64,
) -> AlgorithmResult {
    let config = MemeticConfig {
        pls_time_fraction: pls_fraction,
        ea_backend: EaBackend::Nsga2,
        nsga2_config: Nsga2Config {
            population_size: pop_size,
            ..Nsga2Config::default()
        },
        pls_initial_pop_size: pop_size,
        seed,
        ..MemeticConfig::default()
    };

    let start = Instant::now();
    let result = MemeticAlgorithm::run::<Problem, NUM_OBJECTIVES>(problem, config, timeout);
    let wall = start.elapsed();

    let objectives: Vec<[u64; NUM_OBJECTIVES]> =
        result.archive.iter().map(|s| *s.objectives()).collect();
    AlgorithmResult {
        name: format!("Memetic PLS+NSGA-II ({:.0}%)", pls_fraction * 100.0),
        archive_size: result.archive.len(),
        wall_time: wall,
        hypervolume: 0.0,
        objectives,
    }
}

fn run_memetic_moead(
    problem: &Problem,
    timeout: Duration,
    seed: u64,
    pls_fraction: f64,
) -> AlgorithmResult {
    let config = MemeticConfig {
        pls_time_fraction: pls_fraction,
        ea_backend: EaBackend::Moead,
        moead_config: MoeadConfig {
            num_divisions: 12,
            auto_divisions: false,
            ..MoeadConfig::default()
        },
        pls_initial_pop_size: 50,
        seed,
        ..MemeticConfig::default()
    };

    let start = Instant::now();
    let result = MemeticAlgorithm::run::<Problem, NUM_OBJECTIVES>(problem, config, timeout);
    let wall = start.elapsed();

    let objectives: Vec<[u64; NUM_OBJECTIVES]> =
        result.archive.iter().map(|s| *s.objectives()).collect();
    AlgorithmResult {
        name: format!("Memetic PLS+MOEA/D ({:.0}%)", pls_fraction * 100.0),
        archive_size: result.archive.len(),
        wall_time: wall,
        hypervolume: 0.0,
        objectives,
    }
}

/// Calibrate moors/optirustic: run a batch of warmup iterations, measure wall time,
/// then extrapolate how many iterations fit in the budget.
///
/// We use 30+ warmup iterations so the one-time startup cost (population init,
/// initial fitness evaluation, cold caches) is amortised.  The fill factor is
/// 1.0 because the warmup naturally *overestimates* per-iteration cost — steady-
/// state iterations are faster once caches are warm and allocations are reused.
fn calibrate_moors_iterations(
    problem: &Problem,
    timeout: Duration,
    pop_size: usize,
    seed: u64,
    warmup_iters: usize,
) -> usize {
    let config = MoorsConfig {
        population_size: pop_size,
        num_offsprings: pop_size / 2,
        num_iterations: warmup_iters,
        crossover_rate: 0.9,
        mutation_rate: 0.1,
        bitflip_probability: 0.05,
        crossover_type: MoorsCrossoverType::Uniform,
        seed,
    };

    let start = Instant::now();
    let _ = run_moors_nsga2::<Problem, NUM_OBJECTIVES>(problem, config, timeout);
    let elapsed = start.elapsed();

    if elapsed.is_zero() {
        return 10_000; // fallback
    }

    let iters_per_sec = warmup_iters as f64 / elapsed.as_secs_f64();
    // Fill factor 1.0: warmup overestimates cost, so this is effectively conservative
    let target = (iters_per_sec * timeout.as_secs_f64()).max(10.0) as usize;
    target
}

/// Calibrate SPEA-2 separately — its survivor selection is heavier than NSGA-II's.
fn calibrate_spea2_iterations(
    problem: &Problem,
    timeout: Duration,
    pop_size: usize,
    seed: u64,
    warmup_iters: usize,
) -> usize {
    let config = MoorsConfig {
        population_size: pop_size,
        num_offsprings: pop_size / 2,
        num_iterations: warmup_iters,
        crossover_rate: 0.9,
        mutation_rate: 0.1,
        bitflip_probability: 0.05,
        crossover_type: MoorsCrossoverType::Uniform,
        seed,
    };

    let start = Instant::now();
    let _ = run_moors_spea2::<Problem, NUM_OBJECTIVES>(problem, config, timeout);
    let elapsed = start.elapsed();

    if elapsed.is_zero() {
        return 5_000;
    }

    let iters_per_sec = warmup_iters as f64 / elapsed.as_secs_f64();
    // SPEA-2 survivor selection is O(n²) — use 0.95 factor for slight safety margin
    let target = (iters_per_sec * timeout.as_secs_f64() * 0.95).max(10.0) as usize;
    target
}

/// Calibrate optirustic iterations via a small run.
fn calibrate_optirustic_iterations(
    problem: &Problem,
    timeout: Duration,
    pop_size: usize,
    seed: u64,
    warmup_gens: usize,
) -> usize {
    let config = OptirusticConfig {
        population_size: pop_size,
        max_generations: warmup_gens,
        parallel: false,
        seed: Some(seed),
        ..Default::default()
    };

    let start = Instant::now();
    let _ = run_optirustic_nsga2::<Problem, NUM_OBJECTIVES>(problem, config, timeout);
    let elapsed = start.elapsed();

    if elapsed.is_zero() {
        return 5_000;
    }

    let gens_per_sec = warmup_gens as f64 / elapsed.as_secs_f64();
    // Fill factor 1.0: warmup overestimates cost
    let target = (gens_per_sec * timeout.as_secs_f64()).max(10.0) as usize;
    target
}

fn run_moors_nsga2_bench(
    problem: &Problem,
    timeout: Duration,
    pop_size: usize,
    seed: u64,
    calibrated_iters: usize,
) -> AlgorithmResult {
    let config = MoorsConfig {
        population_size: pop_size,
        num_offsprings: pop_size / 2,
        num_iterations: calibrated_iters,
        crossover_rate: 0.9,
        mutation_rate: 0.1,
        bitflip_probability: 0.05,
        crossover_type: MoorsCrossoverType::Uniform,
        seed,
    };

    let start = Instant::now();
    let (archive, _) = run_moors_nsga2::<Problem, NUM_OBJECTIVES>(problem, config, timeout);
    let wall = start.elapsed();

    let objectives: Vec<[u64; NUM_OBJECTIVES]> = archive.iter().map(|s| *s.objectives()).collect();
    AlgorithmResult {
        name: format!("moors NSGA-II ({calibrated_iters} iter)"),
        archive_size: archive.len(),
        wall_time: wall,
        hypervolume: 0.0,
        objectives,
    }
}

fn run_moors_spea2_bench(
    problem: &Problem,
    timeout: Duration,
    pop_size: usize,
    seed: u64,
    calibrated_iters: usize,
) -> AlgorithmResult {
    let config = MoorsConfig {
        population_size: pop_size,
        num_offsprings: pop_size / 2,
        num_iterations: calibrated_iters,
        crossover_rate: 0.9,
        mutation_rate: 0.1,
        bitflip_probability: 0.05,
        crossover_type: MoorsCrossoverType::Uniform,
        seed,
    };

    let start = Instant::now();
    let (archive, _) = run_moors_spea2::<Problem, NUM_OBJECTIVES>(problem, config, timeout);
    let wall = start.elapsed();

    let objectives: Vec<[u64; NUM_OBJECTIVES]> = archive.iter().map(|s| *s.objectives()).collect();
    AlgorithmResult {
        name: format!("moors SPEA-2 ({calibrated_iters} iter)"),
        archive_size: archive.len(),
        wall_time: wall,
        hypervolume: 0.0,
        objectives,
    }
}

fn run_optirustic_nsga2_bench(
    problem: &Problem,
    timeout: Duration,
    pop_size: usize,
    seed: u64,
    calibrated_gens: usize,
) -> AlgorithmResult {
    let config = OptirusticConfig {
        population_size: pop_size,
        max_generations: calibrated_gens,
        parallel: false,
        seed: Some(seed),
        ..Default::default()
    };

    let start = Instant::now();
    let (archive, _) = run_optirustic_nsga2::<Problem, NUM_OBJECTIVES>(problem, config, timeout);
    let wall = start.elapsed();

    let objectives: Vec<[u64; NUM_OBJECTIVES]> = archive.iter().map(|s| *s.objectives()).collect();
    AlgorithmResult {
        name: format!("optirustic NSGA-II ({calibrated_gens} gen)"),
        archive_size: archive.len(),
        wall_time: wall,
        hypervolume: 0.0,
        objectives,
    }
}

fn run_optirustic_nsga3_bench(
    problem: &Problem,
    timeout: Duration,
    pop_size: usize,
    seed: u64,
    calibrated_gens: usize,
) -> AlgorithmResult {
    let config = OptirusticConfig {
        population_size: pop_size,
        max_generations: calibrated_gens,
        parallel: false,
        seed: Some(seed),
        ..Default::default()
    };

    let start = Instant::now();
    let (archive, _) = run_optirustic_nsga3::<Problem, NUM_OBJECTIVES>(problem, config, timeout);
    let wall = start.elapsed();

    let objectives: Vec<[u64; NUM_OBJECTIVES]> = archive.iter().map(|s| *s.objectives()).collect();
    AlgorithmResult {
        name: format!("optirustic NSGA-III ({calibrated_gens} gen)"),
        archive_size: archive.len(),
        wall_time: wall,
        hypervolume: 0.0,
        objectives,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Hypervolume computation (4-D minimisation)
// ═══════════════════════════════════════════════════════════════════════════

/// Per-dimension [min, max] across a set of objective vectors.
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

/// Normalised 4-D minimisation hypervolume ratio in `[0, 1]`.
///
/// Reference point is `[1.1; 4]` — 10% beyond the normalised nadir so that
/// even the worst solutions in each dimension contribute *some* volume.
/// Using `[1.0; 4]` would assign HV=0 to any front whose solutions touch
/// the per-dimension maximum, which makes comparison uninformative.
///
/// The raw dominated volume is divided by the reference-point volume
/// (`1.1^4`) so the returned value is always in `[0, 1]`.
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
    hv_4d(&mut pts, &ref_pt) / ref_volume
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
    pts.sort_by(|a, b| a[1].partial_cmp(&b[1]).unwrap());
    let mut total = 0.0;
    let mut prev_y = r[1];
    let mut i = pts.len();
    while i > 0 {
        i -= 1;
        let curr_y = pts[i][1];
        if curr_y < prev_y {
            let min_x = pts[..=i].iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
            if min_x < r[0] {
                total += (prev_y - curr_y) * (r[0] - min_x);
            }
            prev_y = curr_y;
        }
    }
    total
}

// ═══════════════════════════════════════════════════════════════════════════
//  Printing
// ═══════════════════════════════════════════════════════════════════════════

fn print_instance_table(results: &[AlgorithmResult]) {
    let name_width = results
        .iter()
        .map(|r| r.name.len())
        .max()
        .unwrap_or(10)
        .max(20);

    println!(
        "  {:<nw$}  {:>8}  {:>10}  {:>10}  {:>8}",
        "Algorithm",
        "Archive",
        "HV",
        "Wall (ms)",
        "Rank",
        nw = name_width
    );
    println!(
        "  {:-<nw$}  {:-<8}  {:-<10}  {:-<10}  {:-<8}",
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
            "  {:<nw$}  {:>8}  {:>10.6}  {:>10}  {:>8}",
            r.name,
            r.archive_size,
            r.hypervolume,
            r.wall_time.as_millis(),
            rank_str,
            nw = name_width
        );
    }
}

fn print_summary_table(all_results: &[(String, Vec<AlgorithmResult>)]) {
    if all_results.is_empty() {
        return;
    }

    // Collect all algorithm names (in order of first appearance)
    let algo_names: Vec<String> = {
        let mut names = Vec::new();
        for (_, results) in all_results {
            for r in results {
                // Normalise external-solver names by stripping parenthesised iteration count
                let base_name = base_algo_name(&r.name);
                if !names.contains(&base_name) {
                    names.push(base_name);
                }
            }
        }
        names
    };

    let name_width = algo_names
        .iter()
        .map(|n| n.len())
        .max()
        .unwrap_or(20)
        .max(20);

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    AGGREGATE SUMMARY (avg HV)                   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    println!(
        "  {:<nw$}  {:>10}  {:>10}  {:>10}  {:>8}",
        "Algorithm",
        "Avg HV",
        "Avg Front",
        "Avg ms",
        "Avg Rank",
        nw = name_width
    );
    println!(
        "  {:-<nw$}  {:-<10}  {:-<10}  {:-<10}  {:-<8}",
        "",
        "",
        "",
        "",
        "",
        nw = name_width
    );

    for algo in &algo_names {
        let mut hv_sum = 0.0;
        let mut front_sum = 0.0;
        let mut ms_sum = 0.0;
        let mut rank_sum = 0.0;
        let mut count = 0.0;

        for (_, results) in all_results {
            // Compute ranks for this instance
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
                let base = base_algo_name(&r.name);
                if &base == algo {
                    hv_sum += r.hypervolume;
                    front_sum += r.archive_size as f64;
                    ms_sum += r.wall_time.as_millis() as f64;
                    rank_sum += ranks[i] as f64;
                    count += 1.0;
                }
            }
        }

        if count > 0.0 {
            println!(
                "  {:<nw$}  {:>10.6}  {:>10.1}  {:>10.0}  {:>8.1}",
                algo,
                hv_sum / count,
                front_sum / count,
                ms_sum / count,
                rank_sum / count,
                nw = name_width
            );
        }
    }
}

/// Strip parenthesised iteration/generation counts from algorithm names for aggregation.
fn base_algo_name(name: &str) -> String {
    if let Some(idx) = name.find('(') {
        name[..idx].trim().to_string()
    } else {
        name.to_string()
    }
}

/// Write all results as a JSON file for downstream plotting scripts.
fn write_json_results(
    path: &PathBuf,
    all_results: &[(String, usize, usize, Vec<AlgorithmResult>)],
    timeout: &Duration,
    pop_size: usize,
    seed: u64,
) {
    let mut f = std::fs::File::create(path).expect("Cannot create JSON output file");

    writeln!(f, "{{").unwrap();
    writeln!(f, "  \"timeout_ms\": {},", timeout.as_millis()).unwrap();
    writeln!(f, "  \"population_size\": {pop_size},").unwrap();
    writeln!(f, "  \"seed\": {seed},").unwrap();
    writeln!(f, "  \"instances\": [").unwrap();

    for (inst_idx, (name, num_images, num_elements, results)) in all_results.iter().enumerate() {
        writeln!(f, "    {{").unwrap();
        writeln!(f, "      \"name\": \"{name}\",").unwrap();
        writeln!(f, "      \"num_images\": {num_images},").unwrap();
        writeln!(f, "      \"num_elements\": {num_elements},").unwrap();
        writeln!(f, "      \"algorithms\": [").unwrap();

        for (algo_idx, r) in results.iter().enumerate() {
            let base = base_algo_name(&r.name);
            writeln!(f, "        {{").unwrap();
            writeln!(f, "          \"name\": \"{}\",", r.name).unwrap();
            writeln!(f, "          \"base_name\": \"{base}\",").unwrap();
            writeln!(f, "          \"archive_size\": {},", r.archive_size).unwrap();
            writeln!(f, "          \"hypervolume\": {},", r.hypervolume).unwrap();
            writeln!(
                f,
                "          \"wall_time_ms\": {},",
                r.wall_time.as_millis()
            )
            .unwrap();

            // Write objective vectors
            write!(f, "          \"objectives\": [").unwrap();
            for (oi, obj) in r.objectives.iter().enumerate() {
                if oi > 0 {
                    write!(f, ", ").unwrap();
                }
                write!(f, "[{}, {}, {}, {}]", obj[0], obj[1], obj[2], obj[3]).unwrap();
            }
            writeln!(f, "]").unwrap();

            let comma = if algo_idx + 1 < results.len() {
                ","
            } else {
                ""
            };
            writeln!(f, "        }}{comma}").unwrap();
        }

        writeln!(f, "      ]").unwrap();
        let comma = if inst_idx + 1 < all_results.len() {
            ","
        } else {
            ""
        };
        writeln!(f, "    }}{comma}").unwrap();
    }

    writeln!(f, "  ]").unwrap();
    writeln!(f, "}}").unwrap();

    println!("  JSON results written to {}", path.display());
}

// ═══════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_lines)]
fn main() {
    // Suppress tracing noise — only warnings and errors
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .compact()
        .with_max_level(tracing::Level::WARN)
        .init();

    let args = Cli::parse();

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          SIMS Multi-Objective Algorithm Benchmark               ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  instances dir : {}", args.instances_dir.display());
    println!(
        "  time budget   : {}",
        humantime::format_duration(args.timeout)
    );
    println!("  population    : {}", args.pop_size);
    println!("  seed          : {}", args.seed);
    println!("  repeats       : {}", args.repeats);
    println!("  pls_fraction  : {:.0}%", args.pls_fraction * 100.0);
    if args.skip_pls {
        println!("  PLS           : SKIPPED");
    } else if args.pls_ablation.is_empty() {
        println!("  PLS modes     : default");
    } else {
        let labels: Vec<&str> = args.pls_ablation.iter().map(|m| m.label()).collect();
        println!("  PLS modes     : {}", labels.join(", "));
    }
    if args.skip_external {
        println!("  external      : SKIPPED");
    }
    if args.skip_memetic {
        println!("  memetic       : SKIPPED");
    }
    if let Some(ref f) = args.filter {
        println!("  filter        : {f}");
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
        "Found {} instance(s). Running benchmark...\n",
        instance_paths.len()
    );

    let mut all_results: Vec<(String, usize, usize, Vec<AlgorithmResult>)> = Vec::new();

    for path in &instance_paths {
        let name = path
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

        println!(
            "━━━ {} (images={}, elements={}) ━━━",
            name, num_images, num_elements
        );

        // ── Calibrate external solvers ──────────────────────────────────
        // Use a small warmup run to estimate iterations/second, then scale
        // to fill the time budget.  We use max(5, ...) warmup iters so
        // calibration itself is very fast.
        let warmup_iters = 30;
        let warmup_gens = 20;

        let (moors_iters, spea2_iters, optirustic_gens) = if args.skip_external {
            println!("  External solvers: SKIPPED");
            (0, 0, 0)
        } else {
            print!("  Calibrating moors NSGA-II iterations... ");
            let mi = calibrate_moors_iterations(
                &problem,
                args.timeout,
                args.pop_size,
                args.seed,
                warmup_iters,
            );
            println!("{mi} iterations");

            print!("  Calibrating moors SPEA-2 iterations... ");
            let si = calibrate_spea2_iterations(
                &problem,
                args.timeout,
                args.pop_size,
                args.seed,
                warmup_iters,
            );
            println!("{si} iterations");

            print!("  Calibrating optirustic generations... ");
            let og = calibrate_optirustic_iterations(
                &problem,
                args.timeout,
                args.pop_size,
                args.seed,
                warmup_gens,
            );
            println!("{og} generations");
            (mi, si, og)
        };
        println!();

        // ── Run all algorithms (possibly repeated) ──────────────────────
        let mut instance_results: Vec<AlgorithmResult> = Vec::new();
        let pls_modes: Vec<CliPlsAblationMode> = if args.pls_ablation.is_empty() {
            vec![CliPlsAblationMode::Default]
        } else {
            args.pls_ablation.clone()
        };

        for rep in 0..args.repeats {
            let rep_seed = args.seed + rep as u64 * 1000;

            // 1. PLS ablations
            if !args.skip_pls {
                for mode in &pls_modes {
                    let r = run_pls(
                        &problem,
                        args.timeout,
                        args.pop_size,
                        rep_seed,
                        *mode,
                        args.diverse_probe_budget,
                        #[cfg(feature = "scalarized_selection")]
                        args.scalarized_parent_budget,
                        #[cfg(feature = "scalarized_selection")]
                        args.scalarized_weight_samples,
                        #[cfg(feature = "scalarized_selection")]
                        args.scalarized_rho,
                    );
                    instance_results.push(r);
                }
            }

            // 2. Custom NSGA-II
            let r = run_custom_nsga2(&problem, args.timeout, args.pop_size, rep_seed);
            instance_results.push(r);

            // 3. Custom MOEA/D
            let r = run_custom_moead(&problem, args.timeout, args.pop_size, rep_seed);
            instance_results.push(r);

            // 4. Memetic PLS+NSGA-II
            if !args.skip_memetic {
                let r = run_memetic_nsga2(
                    &problem,
                    args.timeout,
                    args.pop_size,
                    rep_seed,
                    args.pls_fraction,
                );
                instance_results.push(r);
            }

            // 5. Memetic PLS+MOEA/D
            if !args.skip_memetic {
                let r = run_memetic_moead(&problem, args.timeout, rep_seed, args.pls_fraction);
                instance_results.push(r);
            }

            // 6. moors NSGA-II
            if !args.skip_external {
                let r = run_moors_nsga2_bench(
                    &problem,
                    args.timeout,
                    args.pop_size,
                    rep_seed,
                    moors_iters,
                );
                instance_results.push(r);
            }

            // 7. moors SPEA-2 (separately calibrated)
            if !args.skip_external {
                let r = run_moors_spea2_bench(
                    &problem,
                    args.timeout,
                    args.pop_size,
                    rep_seed,
                    spea2_iters,
                );
                instance_results.push(r);
            }

            // 8. optirustic NSGA-II
            if !args.skip_external {
                let r = run_optirustic_nsga2_bench(
                    &problem,
                    args.timeout,
                    args.pop_size,
                    rep_seed,
                    optirustic_gens,
                );
                instance_results.push(r);
            }

            // 9. optirustic NSGA-III
            if !args.skip_external {
                let r = run_optirustic_nsga3_bench(
                    &problem,
                    args.timeout,
                    args.pop_size,
                    rep_seed,
                    optirustic_gens,
                );
                instance_results.push(r);
            }
        }

        // ── Compute HV with shared bounds ───────────────────────────────
        let all_objs: Vec<[u64; NUM_OBJECTIVES]> = instance_results
            .iter()
            .flat_map(|r| r.objectives.iter().copied())
            .collect();
        let hv_bounds = bounds_from_objectives(&all_objs);

        for r in &mut instance_results {
            r.hypervolume = normalized_hv_4d(r.objectives.iter().copied(), &hv_bounds);
        }

        // If repeated, aggregate by algorithm
        let aggregated = if args.repeats > 1 {
            aggregate_repeats(&instance_results, args.repeats)
        } else {
            instance_results.clone()
        };

        print_instance_table(&aggregated);
        println!();

        all_results.push((name, num_images, num_elements, aggregated));
    }

    // ── Summary across all instances ────────────────────────────────────
    let summary_view: Vec<(String, Vec<AlgorithmResult>)> = all_results
        .iter()
        .map(|(n, _, _, r)| (n.clone(), r.clone()))
        .collect();
    print_summary_table(&summary_view);

    // ── JSON export ─────────────────────────────────────────────────────
    if let Some(ref json_path) = args.json_output {
        write_json_results(
            json_path,
            &all_results,
            &args.timeout,
            args.pop_size,
            args.seed,
        );
    }

    println!();
}

/// When multiple repeats are used, average results per algorithm.
fn aggregate_repeats(results: &[AlgorithmResult], repeats: usize) -> Vec<AlgorithmResult> {
    if repeats <= 1 {
        return results.to_vec();
    }

    let algos_per_rep = results.len() / repeats;
    let mut aggregated = Vec::with_capacity(algos_per_rep);

    for algo_idx in 0..algos_per_rep {
        let mut hv_sum = 0.0;
        let mut front_sum = 0usize;
        let mut ms_sum = 0u128;
        let mut all_objs = Vec::new();

        for rep in 0..repeats {
            let r = &results[rep * algos_per_rep + algo_idx];
            hv_sum += r.hypervolume;
            front_sum += r.archive_size;
            ms_sum += r.wall_time.as_millis();
            all_objs.extend_from_slice(&r.objectives);
        }

        let first = &results[algo_idx];
        aggregated.push(AlgorithmResult {
            name: format!("{} (avg of {})", base_algo_name(&first.name), repeats),
            archive_size: front_sum / repeats,
            wall_time: Duration::from_millis((ms_sum / repeats as u128) as u64),
            hypervolume: hv_sum / repeats as f64,
            objectives: all_objs,
        });
    }

    aggregated
}

// Required for aggregate_repeats to compile
impl Clone for AlgorithmResult {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            archive_size: self.archive_size,
            wall_time: self.wall_time,
            hypervolume: self.hypervolume,
            objectives: self.objectives.clone(),
        }
    }
}
