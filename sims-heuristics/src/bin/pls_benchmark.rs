/// Benchmark: sequential PLS vs concurrent PLS on real SIMS instances.
///
/// Runs both algorithms on the same problem instances for the same wall-clock duration,
/// then prints a comparison table showing Pareto front size improvement and
/// per-region diagnostics exposing how well each thread searched its own space.
///
/// Usage:
///   cargo run --release --bin pls-benchmark --features parallel -- \
///     --instances-dir tests/data --timeout 120s
use std::{
    ops::RangeInclusive,
    path::PathBuf,
    time::{Duration, Instant},
};

use clap::{Parser, ValueEnum};
use pareto::{HasObjectives, ParetoFront};
use pls::{
    PlsOptimizations,
    concurrent_pls::{
        ConcurrentPLS, ConcurrentPLSConfig, ConcurrentPLSResult, RegionResult, RegionSearchMode,
    },
    objectives::ObjectiveType,
    pareto_local_search::ParetoLocalSearch,
    pls_config::SolutionSelectionMode,
    problem_bitset::ProblemBitset,
    solution_impl::bitset_encoded_solution::BitsetEncodedSolution,
    solution_set_impl::NdTreeSolutionSet,
};

const NUM_OBJECTIVES: usize = 4;
const INITIAL_POPULATION_SIZE: usize = 50;
const NEIGHBORHOOD_SIZE_RANGE: RangeInclusive<u32> = 1..=6;

const OBJECTIVE_TYPES: [ObjectiveType; NUM_OBJECTIVES] = [
    ObjectiveType::TotalCost,
    ObjectiveType::CloudyArea,
    ObjectiveType::MinResolution,
    ObjectiveType::MaxIncidenceAngle,
];

/// Short labels for the 4 SIMS objectives (for weight-vector column compact display).
const OBJ_LABELS: [&str; NUM_OBJECTIVES] = ["Cost", "Cloud", "Res", "Angle"];

type Problem = ProblemBitset<NUM_OBJECTIVES>;
type Solution = BitsetEncodedSolution<Problem, NUM_OBJECTIVES>;
type Archive = NdTreeSolutionSet<Solution, NUM_OBJECTIVES>;

#[derive(Parser)]
#[command(
    name = "pls-benchmark",
    about = "Benchmark sequential vs concurrent PLS on real SIMS instances with per-region diagnostics"
)]
struct Cli {
    /// Directory containing .dzn problem instance files
    #[arg(
        short = 'i',
        long = "instances-dir",
        help = "Path to directory containing .dzn problem files",
        default_value = "tests/data"
    )]
    instances_dir: PathBuf,

    /// Per-instance timeout (e.g. 30s, 1m, 2m30s)
    #[arg(
        short = 't',
        long = "timeout",
        help = "Per-instance timeout for both sequential and concurrent runs",
        value_parser = humantime::parse_duration,
        default_value = "120s"
    )]
    timeout: Duration,

    /// Number of parallel threads for concurrent PLS (0 = physical CPU count)
    #[arg(
        short = 'n',
        long = "threads",
        help = "Number of threads for concurrent PLS (0 = physical CPU count)",
        default_value = "0"
    )]
    threads: usize,

    /// Filter instances by name substring (e.g. "_100" for all size-100 instances)
    #[arg(
        short = 'f',
        long = "filter",
        help = "Only benchmark instances whose filename contains this substring"
    )]
    filter: Option<String>,

    /// Initial population size
    #[arg(
        long = "pop-size",
        help = "Size of the initial random population",
        default_value_t = INITIAL_POPULATION_SIZE
    )]
    pop_size: usize,

    /// Skip sequential PLS; only run concurrent (faster when you just need parallel diagnostics)
    #[arg(
        long = "parallel-only",
        help = "Skip sequential PLS run (faster for diagnostics-only mode)"
    )]
    parallel_only: bool,

    /// Region search mode for concurrent PLS
    #[arg(
        long = "region-search-mode",
        value_enum,
        help = "How region workers steer their auxiliary population",
        default_value = "unconstrained"
    )]
    region_search_mode: CliRegionSearchMode,

    /// Augmentation coefficient (rho) for Tchebycheff scalarization in SA-PLS
    #[arg(
        long = "scalarized-rho",
        help = "Augmentation coefficient for SA-PLS Tchebycheff scalarization",
        default_value_t = 1e-3
    )]
    scalarized_rho: f64,

    /// Parent-solution selection policy for sequential PLS
    #[arg(
        long = "solution-selection",
        value_enum,
        default_value = "random-shuffle",
        help = "Parent-solution selection policy for sequential PLS"
    )]
    solution_selection: CliSolutionSelectionMode,

    /// Number of parent solutions to probe when using diverse probing
    #[arg(
        long = "diverse-probe-budget",
        help = "Number of parent solutions to probe when using diverse probing"
    )]
    diverse_probe_budget: Option<usize>,

    #[cfg(feature = "scalarized_selection")]
    /// Maximum number of parent solutions selected per step by scalarized selection
    #[arg(
        long = "sequential-scalarized-parent-budget",
        help = "Maximum number of parent solutions selected per step by sequential scalarized selection"
    )]
    sequential_scalarized_parent_budget: Option<usize>,

    #[cfg(feature = "scalarized_selection")]
    /// Number of random weight vectors sampled per step for scalarized selection
    #[arg(
        long = "sequential-scalarized-weight-samples",
        default_value_t = 1,
        help = "Number of random weight vectors sampled per step for sequential scalarized selection"
    )]
    sequential_scalarized_weight_samples: usize,

    #[cfg(feature = "scalarized_selection")]
    /// Augmentation coefficient for weighted Chebycheff scalarization in sequential PLS
    #[arg(
        long = "sequential-scalarized-rho",
        default_value_t = 1e-3,
        help = "Augmentation coefficient for weighted Chebycheff scalarization in sequential PLS"
    )]
    sequential_scalarized_rho: f64,
}

/// CLI-facing enum mirroring `RegionSearchMode` with `clap::ValueEnum` derive.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliRegionSearchMode {
    /// Phase 3 behaviour: unconstrained Pareto search, parallel restarts.
    Unconstrained,
    /// SA-PLS: auxiliary criterion = scalarized improvement over parent.
    Scalarized,
    /// SA-PLS with fallback: standard PLS after scalarized exhaustion.
    ScalarizedWithFallback,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliSolutionSelectionMode {
    RandomShuffle,
    DiverseProbe,
    #[cfg(feature = "scalarized_selection")]
    ScalarizedChebycheff,
    #[cfg(feature = "scalarized_selection")]
    DiverseThenScalarizedChebycheff,
}

impl CliSolutionSelectionMode {
    const fn to_runtime(self) -> SolutionSelectionMode {
        match self {
            Self::RandomShuffle => SolutionSelectionMode::RandomShuffle,
            Self::DiverseProbe => SolutionSelectionMode::DiverseProbe,
            #[cfg(feature = "scalarized_selection")]
            Self::ScalarizedChebycheff => SolutionSelectionMode::ScalarizedChebycheff,
            #[cfg(feature = "scalarized_selection")]
            Self::DiverseThenScalarizedChebycheff => {
                SolutionSelectionMode::DiverseThenScalarizedChebycheff
            }
        }
    }
}

impl CliRegionSearchMode {
    const fn to_config(self) -> RegionSearchMode {
        match self {
            Self::Unconstrained => RegionSearchMode::Unconstrained,
            Self::Scalarized => RegionSearchMode::ScalarizedAuxiliary,
            Self::ScalarizedWithFallback => RegionSearchMode::ScalarizedAuxiliaryWithFallback,
        }
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .with_thread_ids(true)
        .compact()
        .with_max_level(tracing::Level::WARN)
        .init();

    let args = Cli::parse();

    let physical_cpus = num_cpus::get_physical();
    let logical_cpus = num_cpus::get();
    let num_threads = if args.threads == 0 {
        physical_cpus
    } else {
        args.threads
    };

    println!();
    println!("=== PLS Benchmark: Sequential vs Concurrent ===");
    println!("  instances dir  : {}", args.instances_dir.display());
    println!(
        "  timeout        : {}",
        humantime::format_duration(args.timeout)
    );
    println!(
        "  threads        : {} (physical={}, logical={})",
        num_threads, physical_cpus, logical_cpus
    );
    println!("  population     : {}", args.pop_size);
    println!("  search mode    : {:?}", args.region_search_mode);
    println!("  seq selection  : {:?}", args.solution_selection);
    if let Some(budget) = args.diverse_probe_budget {
        println!("  diverse budget : {budget}");
    }
    if matches!(
        args.region_search_mode,
        CliRegionSearchMode::Scalarized | CliRegionSearchMode::ScalarizedWithFallback
    ) {
        println!("  scalarized rho : {:.0e}", args.scalarized_rho);
    }
    #[cfg(feature = "scalarized_selection")]
    if matches!(
        args.solution_selection,
        CliSolutionSelectionMode::ScalarizedChebycheff
            | CliSolutionSelectionMode::DiverseThenScalarizedChebycheff
    ) {
        if let Some(parent_budget) = args.sequential_scalarized_parent_budget {
            println!("  seq scal. par. : {parent_budget}");
        }
        println!(
            "  seq scal. samp.: {}",
            args.sequential_scalarized_weight_samples
        );
        println!("  seq scal. rho  : {:.0e}", args.sequential_scalarized_rho);
    }
    if let Some(ref f) = args.filter {
        println!("  filter         : {f}");
    }
    if args.parallel_only {
        println!("  mode           : parallel-only (skipping sequential)");
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

    // -----------------------------------------------------------------------
    // Per-instance summary table header
    // -----------------------------------------------------------------------
    println!(
        " {:<30} {:>6} {:>10} {:>10} {:>11} {:>11} {:>10} {:>10} {:>8}",
        "Instance",
        "Size",
        "Seq.front",
        "Par.front",
        "Seq.HV",
        "Par.HV",
        "Seq.iter",
        "Par.iter",
        "Speedup"
    );
    println!(" {}", "-".repeat(108));

    let mut total_seq_front = 0usize;
    let mut total_par_front = 0usize;
    let mut total_seq_hv = 0.0f64;
    let mut total_par_hv = 0.0f64;
    let mut all_par_results: Vec<(String, ConcurrentPLSResult<Solution, NUM_OBJECTIVES>)> =
        Vec::new();
    let mut seq_iters_total = 0usize;
    let mut par_iters_total = 0usize;
    let mut num_instances = 0usize;

    for path in &instance_paths {
        let name = path.file_stem().and_then(|n| n.to_str()).unwrap_or("?");

        let problem = match Problem::from_minizinc_datafile(path, OBJECTIVE_TYPES) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("  [!] Failed to load {}: {e}", path.display());
                continue;
            }
        };

        let size = problem.num_images();

        let initial_pop: Archive = (0..args.pop_size)
            .map(|i| Solution::random_with_seed(&problem, i as u64 + 42))
            .collect();

        // Sequential run
        let seq_result = if args.parallel_only {
            None
        } else {
            Some(run_sequential(
                &problem,
                initial_pop.clone(),
                args.timeout,
                args.solution_selection,
                args.diverse_probe_budget,
                #[cfg(feature = "scalarized_selection")]
                args.sequential_scalarized_parent_budget,
                #[cfg(feature = "scalarized_selection")]
                args.sequential_scalarized_weight_samples,
                #[cfg(feature = "scalarized_selection")]
                args.sequential_scalarized_rho,
            ))
        };

        // Concurrent run
        let par_result = run_concurrent(
            &problem,
            initial_pop.clone(),
            args.timeout,
            num_threads,
            args.region_search_mode.to_config(),
            args.scalarized_rho,
        );

        // Compute normalised HV.
        // Derive per-objective bounds from the union of both fronts so
        // normalisation is identical for seq and par (fair comparison).
        // Reference = nadir * 1.1, which strictly dominates every solution.
        let seq_front_size = seq_result.as_ref().map_or(0, |r| r.archive.len());
        let seq_iters = seq_result.as_ref().map_or(0, |r| r.iterations);

        let union_objs: Vec<[u64; NUM_OBJECTIVES]> = seq_result
            .as_ref()
            .into_iter()
            .flat_map(|r| r.archive.iter().map(|s| *s.objectives()))
            .chain(par_result.archive.iter().map(|s| *s.objectives()))
            .collect();

        let hv_bounds = bounds_from_objectives(&union_objs);

        let seq_hv = seq_result.as_ref().map_or(0.0, |r| {
            normalized_hv_4d(r.archive.iter().map(|s| *s.objectives()), &hv_bounds)
        });
        let par_hv = normalized_hv_4d(
            par_result.archive.iter().map(|s| *s.objectives()),
            &hv_bounds,
        );

        let speedup = if seq_front_size > 0 {
            par_result.archive.len() as f64 / seq_front_size as f64
        } else {
            f64::NAN
        };

        println!(
            " {:<30} {:>6} {:>10} {:>10} {:>11.6} {:>11.6} {:>10} {:>10} {:>7.2}x",
            name,
            size,
            if args.parallel_only {
                "-".to_string()
            } else {
                seq_front_size.to_string()
            },
            par_result.archive.len(),
            if args.parallel_only { f64::NAN } else { seq_hv },
            par_hv,
            if args.parallel_only {
                "-".to_string()
            } else {
                seq_iters.to_string()
            },
            par_result.total_iterations,
            if speedup.is_nan() { 0.0 } else { speedup },
        );

        total_seq_front += seq_front_size;
        total_par_front += par_result.archive.len();
        total_seq_hv += seq_hv;
        total_par_hv += par_hv;
        seq_iters_total += seq_iters;
        par_iters_total += par_result.total_iterations;
        num_instances += 1;

        all_par_results.push((name.to_string(), par_result));
    }

    println!(" {}", "-".repeat(108));
    let avg_speedup = if total_seq_front > 0 {
        total_par_front as f64 / total_seq_front as f64
    } else {
        0.0
    };
    let avg_seq_hv = if num_instances > 0 {
        total_seq_hv / num_instances as f64
    } else {
        0.0
    };
    let avg_par_hv = if num_instances > 0 {
        total_par_hv / num_instances as f64
    } else {
        0.0
    };
    println!(
        " {:<30} {:>6} {:>10} {:>10} {:>11.6} {:>11.6} {:>10} {:>10} {:>7.2}x",
        format!("TOTAL ({num_instances})"),
        "",
        if args.parallel_only {
            "-".to_string()
        } else {
            total_seq_front.to_string()
        },
        total_par_front,
        if args.parallel_only {
            f64::NAN
        } else {
            avg_seq_hv
        },
        avg_par_hv,
        if args.parallel_only {
            "-".to_string()
        } else {
            seq_iters_total.to_string()
        },
        par_iters_total,
        avg_speedup,
    );
    if !args.parallel_only && avg_seq_hv > 0.0 {
        let hv_ratio = avg_par_hv / avg_seq_hv;
        println!(
            "  => Average HV ratio (par/seq): {:.4}  ({:+.1}%)",
            hv_ratio,
            (hv_ratio - 1.0) * 100.0,
        );
    }

    if !args.parallel_only {
        println!();
        let improvement_pct =
            (total_par_front as f64 - total_seq_front as f64) / total_seq_front as f64 * 100.0;
        if total_par_front >= total_seq_front {
            println!(
                "  => Concurrent PLS found {:.1}% MORE non-dominated solutions ({} vs {}).",
                improvement_pct, total_par_front, total_seq_front
            );
        } else {
            println!(
                "  => Concurrent PLS found {:.1}% FEWER non-dominated solutions ({} vs {}).",
                -improvement_pct, total_par_front, total_seq_front
            );
        }
    }

    // -----------------------------------------------------------------------
    // Per-instance per-region diagnostics
    // -----------------------------------------------------------------------
    for (name, par_result) in &all_par_results {
        println!();
        println!(
            "  --- Per-region diagnostics: {} ({} regions) ---",
            name, par_result.num_regions
        );
        print_region_table(&par_result.region_results);
    }

    // -----------------------------------------------------------------------
    // Aggregate per-region stats across all instances
    // -----------------------------------------------------------------------
    if all_par_results.len() > 1 {
        println!();
        println!("  === Aggregate per-region averages across all instances ===");
        print_aggregate_region_stats(&all_par_results);
    }
}

// ---------------------------------------------------------------------------
// Region diagnostics table
// ---------------------------------------------------------------------------

fn print_region_table(regions: &[RegionResult<Solution, NUM_OBJECTIVES>]) {
    // Detect if any region used SA-PLS (has scalarized_exhausted_at_iteration)
    let any_sa_pls = regions
        .iter()
        .any(|r| r.stats.scalarized_exhausted_at_iteration.is_some());

    println!(
        "    {:>3}  {:<26}  {:>8}  {:>7}  {:>9}  {:>7}  {:>8}  {:>9}  {:>7}  {:>7}  {:>8}{}",
        "R#",
        "Weight vector (dom. obj)",
        "Init pop",
        "Archive",
        "In-region",
        "Iters",
        "Iter/s",
        "Neighbors",
        "Adopted",
        "Pruned",
        "Dup skip%",
        if any_sa_pls { "  SA-exhaust" } else { "" },
    );
    let line_width = if any_sa_pls { 137 } else { 125 };
    println!("    {}", "-".repeat(line_width));

    let mut total_init = 0usize;
    let mut total_archive = 0usize;
    let mut total_out = 0usize;
    let mut total_iters = 0usize;
    let mut total_adopted = 0usize;
    let mut total_pruned = 0usize;
    let mut total_neighbors = 0usize;
    let mut total_dupes = 0usize;
    let mut total_secs = 0.0f64;
    let n = regions.len();

    for r in regions {
        let s = &r.stats;
        let wv = format_weight_vector(&r.weight_vector);
        let archive = s.final_archive_size;
        let in_region = archive.saturating_sub(s.out_of_region_count);
        let in_pct = if archive > 0 {
            in_region as f64 / archive as f64 * 100.0
        } else {
            0.0
        };
        let secs = s.wall_time.as_secs_f64();
        let iter_per_s = if secs > 0.0 {
            s.iterations_completed as f64 / secs
        } else {
            0.0
        };
        let dup_pct = if s.neighbors_explored + s.duplicates_skipped > 0 {
            s.duplicates_skipped as f64 / (s.neighbors_explored + s.duplicates_skipped) as f64
                * 100.0
        } else {
            0.0
        };
        let sa_exhaust = if any_sa_pls {
            match s.scalarized_exhausted_at_iteration {
                Some(it) => format!("  {:>10}", format!("@{it}")),
                None => "           -".to_string(),
            }
        } else {
            String::new()
        };

        println!(
            "    {:>3}  {:<26}  {:>8}  {:>7}  {:>8.1}%  {:>7}  {:>8.0}  {:>9}  {:>7}  {:>7}  {:>7.1}%{}",
            r.region_index,
            wv,
            s.initial_pop_size,
            archive,
            in_pct,
            s.iterations_completed,
            iter_per_s,
            s.neighbors_explored,
            s.solutions_adopted,
            s.solutions_pruned,
            dup_pct,
            sa_exhaust,
        );

        total_init += s.initial_pop_size;
        total_archive += archive;
        total_out += s.out_of_region_count;
        total_iters += s.iterations_completed;
        total_adopted += s.solutions_adopted;
        total_pruned += s.solutions_pruned;
        total_neighbors += s.neighbors_explored;
        total_dupes += s.duplicates_skipped;
        total_secs += secs;
    }

    let avg_in_pct = if total_archive > 0 {
        total_archive.saturating_sub(total_out) as f64 / total_archive as f64 * 100.0
    } else {
        0.0
    };
    let avg_iter_per_s = if total_secs > 0.0 {
        total_iters as f64 / total_secs
    } else {
        0.0
    };
    let avg_dup_pct = if total_neighbors + total_dupes > 0 {
        total_dupes as f64 / (total_neighbors + total_dupes) as f64 * 100.0
    } else {
        0.0
    };

    println!("    {}", "-".repeat(line_width));
    println!(
        "    {:>3}  {:<26}  {:>8}  {:>7}  {:>8.1}%  {:>7}  {:>8.0}  {:>9}  {:>7}  {:>7}  {:>7.1}%",
        format!("/{n}"),
        "TOTALS / AVG",
        total_init,
        total_archive,
        avg_in_pct,
        total_iters,
        avg_iter_per_s,
        total_neighbors,
        total_adopted,
        total_pruned,
        avg_dup_pct,
    );

    // Load-balance insight
    let min_iters = regions
        .iter()
        .map(|r| r.stats.iterations_completed)
        .min()
        .unwrap_or(0);
    let max_iters = regions
        .iter()
        .map(|r| r.stats.iterations_completed)
        .max()
        .unwrap_or(0);
    let min_arch = regions
        .iter()
        .map(|r| r.stats.final_archive_size)
        .min()
        .unwrap_or(0);
    let max_arch = regions
        .iter()
        .map(|r| r.stats.final_archive_size)
        .max()
        .unwrap_or(0);
    let min_adopted = regions
        .iter()
        .map(|r| r.stats.solutions_adopted)
        .min()
        .unwrap_or(0);
    let max_adopted = regions
        .iter()
        .map(|r| r.stats.solutions_adopted)
        .max()
        .unwrap_or(0);
    println!(
        "    Load balance -- iters: [{min_iters}..{max_iters}] ({:.1}x range)  \
         archive: [{min_arch}..{max_arch}] ({:.1}x range)  \
         adopted: [{min_adopted}..{max_adopted}]",
        if min_iters > 0 {
            max_iters as f64 / min_iters as f64
        } else {
            f64::NAN
        },
        if min_arch > 0 {
            max_arch as f64 / min_arch as f64
        } else {
            f64::NAN
        },
    );
}

// ---------------------------------------------------------------------------
// Aggregate per-region stats across all instances
// ---------------------------------------------------------------------------

fn print_aggregate_region_stats(
    results: &[(String, ConcurrentPLSResult<Solution, NUM_OBJECTIVES>)],
) {
    let num_regions = results
        .iter()
        .map(|(_, r)| r.num_regions)
        .max()
        .unwrap_or(0);
    if num_regions == 0 {
        return;
    }

    println!(
        "    {:>3}  {:<26}  {:>10}  {:>10}  {:>10}  {:>10}  {:>11}  {:>10}  {:>9}",
        "R#",
        "Weight vector",
        "Avg archive",
        "Avg in-reg%",
        "Avg iters",
        "Avg iter/s",
        "Avg neighbs",
        "Avg adopted",
        "Avg dup%",
    );
    println!("    {}", "-".repeat(112));

    for ri in 0..num_regions {
        let region_rows: Vec<_> = results
            .iter()
            .filter_map(|(_, r)| r.region_results.iter().find(|rr| rr.region_index == ri))
            .collect();
        if region_rows.is_empty() {
            continue;
        }

        let n = region_rows.len() as f64;
        let wv = format_weight_vector(&region_rows[0].weight_vector);

        let avg_archive = region_rows
            .iter()
            .map(|r| r.stats.final_archive_size)
            .sum::<usize>() as f64
            / n;
        let avg_out_pct = region_rows
            .iter()
            .map(|r| {
                if r.stats.final_archive_size > 0 {
                    r.stats.out_of_region_count as f64 / r.stats.final_archive_size as f64 * 100.0
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / n;
        let avg_in_pct = 100.0 - avg_out_pct;
        let avg_iters = region_rows
            .iter()
            .map(|r| r.stats.iterations_completed)
            .sum::<usize>() as f64
            / n;
        let avg_iter_s = region_rows
            .iter()
            .map(|r| {
                let secs = r.stats.wall_time.as_secs_f64();
                if secs > 0.0 {
                    r.stats.iterations_completed as f64 / secs
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / n;
        let avg_neighbors = region_rows
            .iter()
            .map(|r| r.stats.neighbors_explored)
            .sum::<usize>() as f64
            / n;
        let avg_adopted = region_rows
            .iter()
            .map(|r| r.stats.solutions_adopted)
            .sum::<usize>() as f64
            / n;
        let avg_dup_pct = region_rows
            .iter()
            .map(|r| {
                let total = r.stats.neighbors_explored + r.stats.duplicates_skipped;
                if total > 0 {
                    r.stats.duplicates_skipped as f64 / total as f64 * 100.0
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / n;

        println!(
            "    {:>3}  {:<26}  {:>10.1}  {:>9.1}%  {:>10.0}  {:>10.0}  {:>11.0}  {:>10.1}  {:>8.1}%",
            ri,
            wv,
            avg_archive,
            avg_in_pct,
            avg_iters,
            avg_iter_s,
            avg_neighbors,
            avg_adopted,
            avg_dup_pct,
        );
    }
    println!("    {}", "-".repeat(112));
    println!("  Notes:");
    println!(
        "    - in-reg% = fraction of final archive belonging to this region's weight direction."
    );
    println!("    - dup% = fraction of neighbor evaluations skipped (already explored).");
    println!(
        "    - Large iter-range means load imbalance -- some regions exhaust their space early."
    );
    println!(
        "    - High adoption + low in-reg% = cross-pollination is active but region search is weak."
    );
}

// ---------------------------------------------------------------------------
// Format weight vector compactly
// ---------------------------------------------------------------------------

fn format_weight_vector(wv: &[f64; NUM_OBJECTIVES]) -> String {
    let dominant = wv
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let parts: String = wv
        .iter()
        .enumerate()
        .map(|(i, &w)| {
            if i == dominant && w > 0.25 {
                format!("[{:.2}]", w)
            } else {
                format!("{:.2}", w)
            }
        })
        .collect::<Vec<_>>()
        .join("/");

    format!("{} {}", parts, OBJ_LABELS[dominant])
}

// ---------------------------------------------------------------------------
// Sequential PLS runner
// ---------------------------------------------------------------------------

struct SeqResult {
    archive: Archive,
    iterations: usize,
    #[allow(dead_code)]
    wall_time: Duration,
}

fn run_sequential(
    problem: &Problem,
    initial_pop: Archive,
    timeout: Duration,
    solution_selection: CliSolutionSelectionMode,
    diverse_probe_budget: Option<usize>,
    #[cfg(feature = "scalarized_selection")] sequential_scalarized_parent_budget: Option<usize>,
    #[cfg(feature = "scalarized_selection")] sequential_scalarized_weight_samples: usize,
    #[cfg(feature = "scalarized_selection")] sequential_scalarized_rho: f64,
) -> SeqResult {
    let start = Instant::now();
    let mut optimizations = PlsOptimizations::default();
    optimizations.solution_selection_mode = solution_selection.to_runtime();
    optimizations.use_diverse_probing =
        matches!(solution_selection, CliSolutionSelectionMode::DiverseProbe);
    optimizations.diverse_probe_budget = diverse_probe_budget;
    #[cfg(feature = "scalarized_selection")]
    {
        optimizations.scalarized_parent_budget = sequential_scalarized_parent_budget;
        optimizations.scalarized_weight_samples = sequential_scalarized_weight_samples;
        optimizations.scalarized_rho = sequential_scalarized_rho;
    }

    let mut pls = ParetoLocalSearch::new(
        problem,
        &initial_pop,
        NEIGHBORHOOD_SIZE_RANGE,
        false,
        optimizations,
    );
    let archive = pls.run(usize::MAX, timeout);
    SeqResult {
        archive,
        iterations: pls.explored_solutions.num_iterations,
        wall_time: start.elapsed(),
    }
}

// ---------------------------------------------------------------------------
// Normalised 4-D minimisation hypervolume
// ---------------------------------------------------------------------------
// Mirrors the algorithm in sims-problem/src/hypervolume.rs but uses f64.
// Bounds come from the union of both fronts so seq and par share the same
// scale. Reference = nadir * 1.1, strictly dominating every solution.

/// Compute per-dimension [min, max] across a set of objective vectors.
fn bounds_from_objectives(objs: &[[u64; NUM_OBJECTIVES]]) -> [[u64; 2]; NUM_OBJECTIVES] {
    let mut bounds = [[u64::MAX, 0u64]; NUM_OBJECTIVES];
    for o in objs {
        for i in 0..NUM_OBJECTIVES {
            bounds[i][0] = bounds[i][0].min(o[i]);
            bounds[i][1] = bounds[i][1].max(o[i]);
        }
    }
    // Guard: if all objectives identical (or empty), ensure range >= 1.
    for b in &mut bounds {
        if b[0] == u64::MAX {
            *b = [0, 1];
        } // empty
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
    // Normalize to [0, 1] using the supplied bounds.
    // Reference = [1.0; 4] — the nadir corner; HV is in [0, 1).
    let ref_pt = [1.0f64; NUM_OBJECTIVES];
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
    hv_4d(&mut pts, &ref_pt)
}

/// 4-D minimisation HV via sweep over the 4th objective.
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

/// 3-D minimisation HV via sweep over the 3rd objective.
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

/// 2-D minimisation HV.
/// Sweeps by y (dim 1), maintaining the running minimum x seen so far
/// (same logic as `hypervolume_2d_min_generic` in sims-problem).
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
            // minimum x among pts[0..=i]
            let min_x = pts[..=i].iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
            if min_x < r[0] {
                total += (prev_y - curr_y) * (r[0] - min_x);
            }
            prev_y = curr_y;
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Concurrent PLS runner
// ---------------------------------------------------------------------------

fn run_concurrent(
    problem: &Problem,
    initial_pop: Archive,
    timeout: Duration,
    num_threads: usize,
    region_search_mode: RegionSearchMode,
    scalarized_rho: f64,
) -> ConcurrentPLSResult<Solution, NUM_OBJECTIVES> {
    let config = ConcurrentPLSConfig {
        num_threads,
        sync_interval_steps: 5,
        merge_interval: Duration::from_millis(100),
        boundary_threshold: 0.05,
        das_dennis_h: None,
        max_iterations: usize::MAX,
        max_duration: timeout,
        neighborhood_size_range: NEIGHBORHOOD_SIZE_RANGE,
        is_deterministic: false,
        region_search_mode,
        scalarized_rho,
    };

    let solver: ConcurrentPLS<'_, Solution, Problem, NUM_OBJECTIVES> =
        ConcurrentPLS::new(problem, config);
    solver.solve(&initial_pop)
}
