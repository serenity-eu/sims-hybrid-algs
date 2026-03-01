/// Analyze objective-space geometry for data-driven decomposition.
///
/// Loads a SIMS instance, runs a quick PLS to build a Pareto front, and then
/// computes detailed statistics to understand why in-region% is low and how
/// to improve decomposition.
///
/// Usage:
///   cargo run --release --bin analyze-decomposition --features parallel -- \
///     --instances-dir tests/data --timeout 30s

use std::{
    ops::RangeInclusive,
    path::PathBuf,
    time::Duration,
};

use clap::Parser;
use pareto::{HasObjectives, ParetoFront};
use pls::{
    SetCoverProblem,
    concurrent_pls::decomposition::{
        belongs_to_region, build_regions, das_dennis_weight_vectors,
        auto_select_h,
    },
    objectives::ObjectiveType,
    pareto_local_search::ParetoLocalSearch,
    problem_bitset::ProblemBitset,
    solution_impl::bitset_encoded_solution::BitsetEncodedSolution,
    solution_set_impl::NdTreeSolutionSet,
};

const D: usize = 4;
const INITIAL_POP: usize = 50;
const NEIGHBORHOOD: RangeInclusive<u32> = 1..=6;

const OBJECTIVE_TYPES: [ObjectiveType; D] = [
    ObjectiveType::TotalCost,
    ObjectiveType::CloudyArea,
    ObjectiveType::MinResolution,
    ObjectiveType::MaxIncidenceAngle,
];

const OBJ_NAMES: [&str; D] = ["TotalCost", "CloudyArea", "MinResolution", "MaxIncAngle"];

type Problem = ProblemBitset<D>;
type Solution = BitsetEncodedSolution<Problem, D>;
type Archive = NdTreeSolutionSet<Solution, D>;

#[derive(Parser)]
#[command(name = "analyze-decomposition")]
struct Cli {
    #[arg(short = 'i', long = "instances-dir", default_value = "tests/data")]
    instances_dir: PathBuf,

    #[arg(
        short = 't', long = "timeout",
        value_parser = humantime::parse_duration,
        default_value = "30s"
    )]
    timeout: Duration,

    #[arg(short = 'f', long = "filter")]
    filter: Option<String>,

    #[arg(short = 'n', long = "threads", default_value = "10")]
    threads: usize,
}

fn main() {
    let args = Cli::parse();

    let mut paths: Vec<PathBuf> = std::fs::read_dir(&args.instances_dir)
        .expect("Cannot read instances directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "dzn"))
        .collect();
    paths.sort();

    if let Some(ref f) = args.filter {
        paths.retain(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map_or(false, |n| n.contains(f.as_str()))
        });
    }

    for path in &paths {
        let name = path.file_stem().and_then(|n| n.to_str()).unwrap_or("?");
        let problem = match Problem::from_minizinc_datafile(path, OBJECTIVE_TYPES) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to load {name}: {e}");
                continue;
            }
        };

        println!();
        println!("{}", "=".repeat(80));
        println!("  Instance: {name}");
        println!("  Images: {}, Universe: {}", problem.num_images(), problem.num_elements());
        println!("{}", "=".repeat(80));

        // --- Section 1: Theoretical bounds (what we currently use) ---
        let max_obj = problem.max_objectives();
        println!();
        println!("  1) THEORETICAL BOUNDS (current normalization: 0..max_objectives)");
        println!("  {:>15} {:>15} {:>15}", "Objective", "min", "max");
        println!("  {}", "-".repeat(50));
        for j in 0..D {
            println!("  {:>15} {:>15} {:>15}", OBJ_NAMES[j], 0, max_obj[j]);
        }

        // --- Section 2: Per-image objective contributions ---
        println!();
        println!("  2) PER-IMAGE RAW DATA RANGES");
        analyze_per_image_data(&problem);

        // --- Section 3: Run PLS to get a Pareto front ---
        println!();
        println!("  3) PARETO FRONT ANALYSIS (PLS {}s)", args.timeout.as_secs());

        let initial_pop: Archive = (0..INITIAL_POP)
            .map(|i| Solution::random_with_seed(&problem, i as u64 + 42))
            .collect();

        let mut pls = ParetoLocalSearch::new(
            &problem,
            &initial_pop,
            NEIGHBORHOOD,
            false,
        );
        let result_archive: Archive = pls.run(usize::MAX, args.timeout);
        let front_size = result_archive.len();

        // Collect objective values from the Pareto front
        let mut obj_values: Vec<[u64; D]> = Vec::with_capacity(front_size);
        for sol in result_archive.iter() {
            obj_values.push(*sol.objectives());
        }

        if obj_values.is_empty() {
            println!("    No solutions found!");
            continue;
        }

        // Compute actual ranges on the Pareto front
        let mut actual_min = [u64::MAX; D];
        let mut actual_max = [0u64; D];
        let mut sums = [0.0f64; D];
        for objs in &obj_values {
            for j in 0..D {
                actual_min[j] = actual_min[j].min(objs[j]);
                actual_max[j] = actual_max[j].max(objs[j]);
                sums[j] += objs[j] as f64;
            }
        }
        let means: [f64; D] = std::array::from_fn(|j| sums[j] / front_size as f64);

        // Standard deviation
        let mut var_sums = [0.0f64; D];
        for objs in &obj_values {
            for j in 0..D {
                let diff = objs[j] as f64 - means[j];
                var_sums[j] += diff * diff;
            }
        }
        let stds: [f64; D] = std::array::from_fn(|j| (var_sums[j] / front_size as f64).sqrt());

        println!("    Front size: {front_size}");
        println!();
        println!("    {:>15} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
                 "Objective", "PF min", "PF max", "PF range", "mean", "std", "theo. max");
        println!("    {}", "-".repeat(87));
        for j in 0..D {
            let pf_range = actual_max[j] - actual_min[j];
            println!("    {:>15} {:>12} {:>12} {:>12} {:>12.1} {:>12.1} {:>12}",
                     OBJ_NAMES[j], actual_min[j], actual_max[j], pf_range,
                     means[j], stds[j], max_obj[j]);
        }

        // --- Section 4: Range ratio analysis ---
        println!();
        println!("  4) NORMALIZATION MISMATCH (PF range vs theoretical range)");
        println!("    {:>15} {:>15} {:>15} {:>10}",
                 "Objective", "PF range", "Theo range", "Ratio");
        println!("    {}", "-".repeat(60));
        for j in 0..D {
            let pf_range = (actual_max[j] - actual_min[j]) as f64;
            let theo_range = max_obj[j] as f64;
            let ratio = if theo_range > 0.0 { pf_range / theo_range } else { f64::NAN };
            println!("    {:>15} {:>15.0} {:>15.0} {:>9.4}x",
                     OBJ_NAMES[j], pf_range, theo_range, ratio);
        }

        // --- Section 5: Decomposition analysis with current bounds ---
        println!();
        println!("  5) DECOMPOSITION ANALYSIS: CURRENT BOUNDS (0, max_obj)");
        let current_bounds: [(f64, f64); D] = std::array::from_fn(|j| (0.0, max_obj[j] as f64));
        let ideal_current: [u64; D] = std::array::from_fn(|j| actual_min[j]);
        analyze_decomposition(
            &obj_values, &current_bounds, &ideal_current, args.threads, "CURRENT",
        );

        // --- Section 6: Decomposition with Pareto-front-derived bounds ---
        println!();
        println!("  6) DECOMPOSITION ANALYSIS: PF-DERIVED BOUNDS (actual_min, actual_max)");
        let pf_bounds: [(f64, f64); D] = std::array::from_fn(|j| {
            (actual_min[j] as f64, actual_max[j] as f64)
        });
        analyze_decomposition(
            &obj_values, &pf_bounds, &ideal_current, args.threads, "PF-DERIVED",
        );

        // --- Section 7: Decomposition with padded PF bounds (10% margin) ---
        println!();
        println!("  7) DECOMPOSITION ANALYSIS: PADDED PF BOUNDS (10% margin)");
        let padded_bounds: [(f64, f64); D] = std::array::from_fn(|j| {
            let pf_range = (actual_max[j] - actual_min[j]) as f64;
            let margin = pf_range * 0.1;
            ((actual_min[j] as f64 - margin).max(0.0), actual_max[j] as f64 + margin)
        });
        analyze_decomposition(
            &obj_values, &padded_bounds, &ideal_current, args.threads, "PADDED-PF",
        );

        // --- Section 8: Correlation matrix ---
        println!();
        println!("  8) OBJECTIVE CORRELATION MATRIX (Pearson)");
        print_correlation_matrix(&obj_values, front_size);

        // --- Section 9: Weight vector analysis ---
        println!();
        println!("  9) WEIGHT VECTOR DETAILS (H=auto for {} threads)", args.threads);
        let h = auto_select_h::<D>(args.threads);
        let weight_vectors = das_dennis_weight_vectors::<D>(h);
        println!("    H={h}, {} weight vectors generated", weight_vectors.len());
        for (i, wv) in weight_vectors.iter().enumerate() {
            let dominant = wv.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| OBJ_NAMES[idx])
                .unwrap_or("?");
            println!("    W{i:>2}: [{:.2}, {:.2}, {:.2}, {:.2}]  dom={}",
                     wv[0], wv[1], wv[2], wv[3], dominant);
        }

        // --- Section 10: Per-weight-vector score distribution ---
        println!();
        println!("  10) SCORE DISTRIBUTION PER REGION (current vs PF bounds)");
        analyze_score_distributions(&obj_values, &current_bounds, &pf_bounds, &ideal_current, args.threads);
    }
}

fn analyze_per_image_data(problem: &Problem) {
    let obj_states = SetCoverProblem::<D>::objectives(problem);
    println!("    {:>15} {:>12} {:>12} {:>12}",
             "", "min", "max", "sum/count");
    println!("    {}", "-".repeat(55));

    // Cost: per-image costs
    if let pls::objectives::ObjectiveState::TotalCost { costs, max_value, .. } = &obj_states[0] {
        let min_c = costs.iter().min().copied().unwrap_or(0);
        let max_c = costs.iter().max().copied().unwrap_or(0);
        println!("    {:>15} {:>12} {:>12} {:>12} (sum=max_value)",
                 "img cost", min_c, max_c, max_value);
    }

    // Resolution: per-image resolutions
    if let pls::objectives::ObjectiveState::MinResolution { resolutions, max_value, .. } = &obj_states[2] {
        let min_r = resolutions.iter().min().copied().unwrap_or(0);
        let max_r = resolutions.iter().max().copied().unwrap_or(0);
        let mean_r: f64 = resolutions.iter().map(|&r| r as f64).sum::<f64>() / resolutions.len() as f64;
        println!("    {:>15} {:>12} {:>12} {:>12.0} (mean), max_value={}",
                 "img resolution", min_r, max_r, mean_r, max_value);
    }

    // Incidence angle: per-image angles
    if let pls::objectives::ObjectiveState::MaxIncidenceAngle { incidence_angles, max_value, .. } = &obj_states[3] {
        let min_a = incidence_angles.iter().min().copied().unwrap_or(0);
        let max_a = incidence_angles.iter().max().copied().unwrap_or(0);
        let mean_a: f64 = incidence_angles.iter().map(|&a| a as f64).sum::<f64>() / incidence_angles.len() as f64;
        println!("    {:>15} {:>12} {:>12} {:>12.0} (mean), max_value={}",
                 "img inc. angle", min_a, max_a, mean_a, max_value);
    }

    // Universe & areas
    if let pls::objectives::ObjectiveState::CloudyArea { areas, max_value, .. } = &obj_states[1] {
        let min_a = areas.iter().min().copied().unwrap_or(0);
        let max_a = areas.iter().max().copied().unwrap_or(0);
        let mean_a: f64 = areas.iter().map(|&a| a as f64).sum::<f64>() / areas.len() as f64;
        println!("    {:>15} {:>12} {:>12} {:>12.0} (mean), max_value={} (sum)",
                 "elem area", min_a, max_a, mean_a, max_value);
    }

    println!("    universe_size = {}", problem.num_elements());
}

fn analyze_decomposition(
    obj_values: &[[u64; D]],
    bounds: &[(f64, f64); D],
    ideal: &[u64; D],
    num_threads: usize,
    label: &str,
) {
    let regions = build_regions::<D>(num_threads, None);
    let n = obj_values.len();

    // Count assignments
    let mut region_counts = vec![0usize; regions.len()];
    let mut region_belongs = vec![0usize; regions.len()];

    for objs in obj_values {
        // Primary assignment (best region)
        let scores: Vec<f64> = regions.iter().map(|r| r.score(objs, ideal, bounds)).collect();
        let best_idx = scores
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        region_counts[best_idx] += 1;

        // belongs_to_region check
        for (ri, region) in regions.iter().enumerate() {
            if belongs_to_region(objs, region, &regions, ideal, bounds) {
                region_belongs[ri] += 1;
            }
        }
    }

    // Print assignment table
    println!("    [{label}] {} regions, {} solutions", regions.len(), n);
    println!("    {:>3}  {:<26}  {:>8}  {:>8}  {:>7}  {:>8}",
             "R#", "Weight vector", "Primary", "%", "Belongs", "Bel. %");
    println!("    {}", "-".repeat(72));

    let mut non_empty = 0;
    let mut min_pct = 100.0f64;
    let mut max_pct = 0.0f64;

    for (i, region) in regions.iter().enumerate() {
        let wv = &region.weight_vector;
        let pct = 100.0 * region_counts[i] as f64 / n as f64;
        let bel_pct = 100.0 * region_belongs[i] as f64 / n as f64;
        if region_counts[i] > 0 {
            non_empty += 1;
            min_pct = min_pct.min(pct);
            max_pct = max_pct.max(pct);
        }
        let dominant = wv.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| OBJ_NAMES[idx])
            .unwrap_or("?");
        println!("    {:>3}  [{:.2},{:.2},{:.2},{:.2}] {:>5}  {:>8}  {:>7.1}%  {:>8}  {:>7.1}%",
                 i,
                 wv[0], wv[1], wv[2], wv[3],
                 dominant,
                 region_counts[i], pct,
                 region_belongs[i], bel_pct);
    }

    let expected_pct = 100.0 / regions.len() as f64;
    let balance = if max_pct > 0.0 { min_pct / max_pct } else { 0.0 };
    println!();
    println!("    Non-empty regions: {}/{}", non_empty, regions.len());
    println!("    Expected uniform: {:.1}%", expected_pct);
    println!("    Actual range: {:.1}% - {:.1}%", min_pct, max_pct);
    println!("    Balance ratio (min/max): {:.3}", balance);

    // Show normalized distance statistics per dimension
    println!();
    println!("    Normalized distance stats per dimension:");
    println!("    {:>15} {:>12} {:>12} {:>12} {:>12} {:>12}",
             "Objective", "norm range", "mean d/R", "std d/R", "min d/R", "max d/R");
    println!("    {}", "-".repeat(77));

    for j in 0..D {
        let (min_j, max_j) = bounds[j];
        let range = (max_j - min_j).max(1.0);
        let norm_vals: Vec<f64> = obj_values.iter()
            .map(|objs| (objs[j] as f64 - ideal[j] as f64).max(0.0) / range)
            .collect();
        let mean_norm = norm_vals.iter().sum::<f64>() / norm_vals.len() as f64;
        let std_norm = (norm_vals.iter().map(|v| (v - mean_norm).powi(2)).sum::<f64>() / norm_vals.len() as f64).sqrt();
        let min_norm = norm_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_norm = norm_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("    {:>15} {:>12.1} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                 OBJ_NAMES[j], range, mean_norm, std_norm, min_norm, max_norm);
    }
}

fn print_correlation_matrix(obj_values: &[[u64; D]], n: usize) {
    // Compute means
    let mut means = [0.0f64; D];
    for objs in obj_values {
        for j in 0..D {
            means[j] += objs[j] as f64;
        }
    }
    for j in 0..D {
        means[j] /= n as f64;
    }

    // Compute covariance matrix
    let mut cov = [[0.0f64; D]; D];
    for objs in obj_values {
        for j in 0..D {
            for k in 0..D {
                cov[j][k] += (objs[j] as f64 - means[j]) * (objs[k] as f64 - means[k]);
            }
        }
    }
    for j in 0..D {
        for k in 0..D {
            cov[j][k] /= n as f64;
        }
    }

    // Compute correlation
    println!("    {:>15} {:>12} {:>12} {:>12} {:>12}",
             "", OBJ_NAMES[0], OBJ_NAMES[1], OBJ_NAMES[2], OBJ_NAMES[3]);
    println!("    {}", "-".repeat(65));
    for j in 0..D {
        print!("    {:>15}", OBJ_NAMES[j]);
        for k in 0..D {
            let denom = (cov[j][j] * cov[k][k]).sqrt();
            let corr = if denom > 0.0 { cov[j][k] / denom } else { 0.0 };
            print!(" {:>12.4}", corr);
        }
        println!();
    }
}

fn analyze_score_distributions(
    obj_values: &[[u64; D]],
    current_bounds: &[(f64, f64); D],
    pf_bounds: &[(f64, f64); D],
    ideal: &[u64; D],
    num_threads: usize,
) {
    let current_regions = build_regions::<D>(num_threads, None);
    let pf_regions = build_regions::<D>(num_threads, None);

    println!("    {:>3}  {:<14}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
             "R#", "Weight", "Cur.mean", "Cur.std", "Cur.range", "PF.mean", "PF.std", "PF.range");
    println!("    {}", "-".repeat(90));

    for i in 0..current_regions.len().min(pf_regions.len()) {
        let wv = &current_regions[i].weight_vector;
        let wv_str = format!("[{:.1},{:.1},{:.1},{:.1}]", wv[0], wv[1], wv[2], wv[3]);

        // Current bounds scores
        let cur_scores: Vec<f64> = obj_values.iter()
            .map(|objs| current_regions[i].score(objs, ideal, current_bounds))
            .collect();
        let cur_mean = cur_scores.iter().sum::<f64>() / cur_scores.len() as f64;
        let cur_std = (cur_scores.iter().map(|s| (s - cur_mean).powi(2)).sum::<f64>() / cur_scores.len() as f64).sqrt();
        let cur_min = cur_scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let cur_max = cur_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // PF bounds scores
        let pf_scores: Vec<f64> = obj_values.iter()
            .map(|objs| pf_regions[i].score(objs, ideal, pf_bounds))
            .collect();
        let pf_mean = pf_scores.iter().sum::<f64>() / pf_scores.len() as f64;
        let pf_std = (pf_scores.iter().map(|s| (s - pf_mean).powi(2)).sum::<f64>() / pf_scores.len() as f64).sqrt();
        let pf_min = pf_scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let pf_max = pf_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("    {:>3}  {:<14}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}",
                 i, wv_str,
                 cur_mean, cur_std, cur_max - cur_min,
                 pf_mean, pf_std, pf_max - pf_min);
    }
}
