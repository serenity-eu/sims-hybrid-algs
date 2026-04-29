//! Memetic Hybrid Algorithms for SIMS
//!
//! Combines Pareto Local Search (PLS) with evolutionary algorithms (NSGA-II or
//! MOEA/D) in a two-phase approach:
//!
//! 1. **Phase 1 — PLS warm-start**: Run PLS for a configurable fraction of the
//!    total time budget to generate a high-quality initial set of non-dominated
//!    solutions. PLS excels at quickly finding good individual solutions through
//!    domain-aware neighborhood exploration.
//!
//! 2. **Phase 2 — EA refinement**: Use the PLS archive to seed the population of
//!    an evolutionary algorithm (NSGA-II or MOEA/D). The EA then explores the
//!    search space using population-based operators (crossover, mutation) for the
//!    remaining time budget, potentially discovering solutions in unexplored
//!    regions of objective space.
//!
//! This hybrid approach leverages the complementary strengths of both paradigms:
//! - PLS is highly effective at finding non-dominated solutions quickly through
//!   targeted local search, especially on small/medium instances.
//! - EAs provide population-level diversity and can escape local optima through
//!   recombination, especially beneficial on larger instances.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use pls::evolutionary::memetic::{MemeticConfig, MemeticAlgorithm, EaBackend};
//!
//! let config = MemeticConfig {
//!     pls_time_fraction: 0.3,  // 30% of budget for PLS
//!     ea_backend: EaBackend::Nsga2,
//!     ..Default::default()
//! };
//!
//! let result = MemeticAlgorithm::run(&problem, config, timeout, seed);
//! ```

use std::time::{Duration, Instant};

use pareto::{HasObjectives, MoSolution};
use crate::PlsOptimizations;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use tracing::info;

use crate::explored_solutions_data::ExploredSolutionsData;
use crate::pareto_local_search::ParetoLocalSearch;
use crate::problem::SetCoverProblem;

use crate::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;
use crate::solution_set_impl::NdTreeSolutionSet;

use super::moead::{Moead, MoeadConfig};
use super::nsga2::{Nsga2, Nsga2Config};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Which evolutionary algorithm to use in Phase 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EaBackend {
    /// NSGA-II with coverage-aware operators and contribution-distance selection.
    Nsga2,
    /// MOEA/D with Tchebycheff scalarization and adaptive weight vectors.
    Moead,
}

impl std::fmt::Display for EaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nsga2 => write!(f, "NSGA-II"),
            Self::Moead => write!(f, "MOEA/D"),
        }
    }
}

/// Configuration for the memetic hybrid algorithm.
#[derive(Debug, Clone)]
pub struct MemeticConfig {
    /// Fraction of the total time budget allocated to PLS (Phase 1).
    /// Must be in (0.0, 1.0). Default: 0.3 (30%).
    ///
    /// Lower values give more time to the EA; higher values let PLS build a
    /// stronger initial archive but leave less room for EA exploration.
    ///
    /// Recommended ranges:
    /// - Small instances (30 images): 0.4–0.6 (PLS is very effective)
    /// - Medium instances (50–100): 0.2–0.4 (balanced)
    /// - Large instances (150+): 0.1–0.2 (EA benefits from more time)
    pub pls_time_fraction: f64,

    /// Which EA to use in Phase 2.
    pub ea_backend: EaBackend,

    /// NSGA-II configuration (used when `ea_backend == EaBackend::Nsga2`).
    pub nsga2_config: Nsga2Config,

    /// MOEA/D configuration (used when `ea_backend == EaBackend::Moead`).
    pub moead_config: MoeadConfig,

    /// Initial population size for PLS Phase 1.
    /// More initial solutions give PLS better starting diversity.
    pub pls_initial_pop_size: usize,

    /// Neighborhood size range for PLS.
    pub pls_neighborhood_range: std::ops::RangeInclusive<u32>,

    /// Maximum number of PLS archive solutions to seed into the EA population.
    /// If the PLS archive is larger than this, solutions are sampled to preserve
    /// diversity (using farthest-point selection in objective space).
    /// If 0, all PLS archive solutions are used (no cap).
    pub max_pls_seed_size: usize,

    /// Whether to merge the PLS archive into the final EA archive at the end.
    /// This ensures no good PLS solutions are lost even if the EA didn't
    /// rediscover them. Default: true.
    pub merge_pls_archive: bool,

    /// Random seed.
    pub seed: u64,
}

impl Default for MemeticConfig {
    fn default() -> Self {
        Self {
            pls_time_fraction: 0.3,
            ea_backend: EaBackend::Nsga2,
            nsga2_config: Nsga2Config::default(),
            moead_config: MoeadConfig {
                num_divisions: 12,
                auto_divisions: false,
                ..MoeadConfig::default()
            },
            pls_initial_pop_size: 50,
            pls_neighborhood_range: 1..=6,
            max_pls_seed_size: 0, // no cap
            merge_pls_archive: true,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Result of a memetic hybrid run, including phase-level diagnostics.
pub struct MemeticResult<P, const D: usize>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    /// Final combined archive of non-dominated solutions.
    pub archive: Vec<BitsetEncodedSolution<P, D>>,

    /// Number of non-dominated solutions found by PLS (Phase 1).
    pub pls_archive_size: usize,

    /// Number of non-dominated solutions in the EA archive at end of Phase 2
    /// (before merging with PLS archive).
    pub ea_archive_size: usize,

    /// Wall-clock time spent in Phase 1 (PLS).
    pub pls_duration: Duration,

    /// Wall-clock time spent in Phase 2 (EA).
    pub ea_duration: Duration,

    /// Total wall-clock time.
    pub total_duration: Duration,

    /// Explored solutions data (combined from both phases).
    pub explored_solutions: ExploredSolutionsData<D>,
}

// ---------------------------------------------------------------------------
// Algorithm
// ---------------------------------------------------------------------------

/// Memetic hybrid algorithm combining PLS warm-start with EA refinement.
pub struct MemeticAlgorithm;

impl MemeticAlgorithm {
    /// Run the memetic hybrid on a 4-objective SIMS problem instance.
    ///
    /// This is the primary entry point. It orchestrates:
    /// 1. PLS for `pls_time_fraction * max_duration`
    /// 2. EA (NSGA-II or MOEA/D) for the remaining time
    /// 3. Archive merging (if enabled)
    pub fn run<P, const D: usize>(
        problem: &P,
        config: MemeticConfig,
        max_duration: Duration,
    ) -> MemeticResult<P, D>
    where
        P: SetCoverProblem<D> + Clone + Send + Sync,
    {
        let overall_start = Instant::now();

        // Compute time budgets
        let pls_fraction = config.pls_time_fraction.clamp(0.01, 0.99);
        let pls_budget = Duration::from_secs_f64(max_duration.as_secs_f64() * pls_fraction);
        // We'll compute the EA budget dynamically after PLS finishes to account
        // for any PLS overhead.

        info!(
            "Memetic hybrid starting: backend={}, pls_fraction={:.0}%, pls_budget={:?}, total_budget={:?}, seed={}",
            config.ea_backend,
            pls_fraction * 100.0,
            pls_budget,
            max_duration,
            config.seed,
        );

        // ─── Phase 1: PLS ───────────────────────────────────────────────
        let pls_start = Instant::now();
        let (pls_archive_solutions, pls_explored) =
            Self::run_pls_phase(problem, &config, pls_budget);
        let pls_duration = pls_start.elapsed();
        let pls_archive_size = pls_archive_solutions.len();

        info!(
            "Phase 1 (PLS) complete: archive_size={}, duration={:?}",
            pls_archive_size, pls_duration,
        );

        // ─── Prepare EA seed population ─────────────────────────────────
        let ea_seed = Self::select_seed_population(
            &pls_archive_solutions,
            config.max_pls_seed_size,
            config.seed,
        );

        info!(
            "EA seed population: {} solutions (from {} PLS archive)",
            ea_seed.len(),
            pls_archive_size,
        );

        // ─── Phase 2: EA ────────────────────────────────────────────────
        let elapsed_so_far = overall_start.elapsed();
        let ea_budget = if elapsed_so_far < max_duration {
            max_duration - elapsed_so_far
        } else {
            Duration::from_millis(100) // minimum EA time
        };

        let ea_start = Instant::now();
        let (ea_archive_solutions, ea_explored) =
            Self::run_ea_phase(problem, &config, ea_seed, ea_budget);
        let ea_duration = ea_start.elapsed();
        let ea_archive_size = ea_archive_solutions.len();

        info!(
            "Phase 2 ({}) complete: archive_size={}, duration={:?}",
            config.ea_backend, ea_archive_size, ea_duration,
        );

        // ─── Merge archives ─────────────────────────────────────────────
        let final_archive = if config.merge_pls_archive {
            Self::merge_archives(pls_archive_solutions.clone(), ea_archive_solutions)
        } else {
            ea_archive_solutions
        };

        info!(
            "Final merged archive: {} solutions (PLS={}, EA={})",
            final_archive.len(),
            pls_archive_size,
            ea_archive_size,
        );

        // ─── Combine explored solutions ─────────────────────────────────
        let mut combined_explored = pls_explored;
        // Merge EA explored solutions into the combined tracker.
        for (hash, fingerprint) in &ea_explored.solutions {
            if !combined_explored.solutions.contains_key(hash) {
                combined_explored
                    .solutions
                    .insert(*hash, fingerprint.clone());
            }
        }

        let total_duration = overall_start.elapsed();

        MemeticResult {
            archive: final_archive,
            pls_archive_size,
            ea_archive_size,
            pls_duration,
            ea_duration,
            total_duration,
            explored_solutions: combined_explored,
        }
    }

    // -----------------------------------------------------------------
    // Phase 1: PLS
    // -----------------------------------------------------------------

    fn run_pls_phase<P, const D: usize>(
        problem: &P,
        config: &MemeticConfig,
        budget: Duration,
    ) -> (Vec<BitsetEncodedSolution<P, D>>, ExploredSolutionsData<D>)
    where
        P: SetCoverProblem<D> + Clone + Send + Sync,
    {
        type Archive<Sol, const N: usize> = NdTreeSolutionSet<Sol, N>;

        let initial_pop: Archive<BitsetEncodedSolution<P, D>, D> = (0..config.pls_initial_pop_size)
            .map(|i| BitsetEncodedSolution::random_with_seed(problem, config.seed + i as u64))
            .collect();

        let mut pls = ParetoLocalSearch::new(
            problem,
            &initial_pop,
            config.pls_neighborhood_range.clone(),
            false,
            PlsOptimizations::default(),
        );

        let archive = pls.run(usize::MAX, budget);
        let explored = std::mem::replace(
            &mut pls.explored_solutions,
            ExploredSolutionsData::new(problem.max_objectives()),
        );

        let solutions: Vec<BitsetEncodedSolution<P, D>> = archive.into_iter().collect();
        (solutions, explored)
    }

    // -----------------------------------------------------------------
    // Phase 2: EA
    // -----------------------------------------------------------------

    fn run_ea_phase<P, const D: usize>(
        problem: &P,
        config: &MemeticConfig,
        seed_population: Vec<BitsetEncodedSolution<P, D>>,
        budget: Duration,
    ) -> (Vec<BitsetEncodedSolution<P, D>>, ExploredSolutionsData<D>)
    where
        P: SetCoverProblem<D> + Clone + Send + Sync,
    {
        match config.ea_backend {
            EaBackend::Nsga2 => Self::run_nsga2_phase(
                problem,
                &config.nsga2_config,
                seed_population,
                budget,
                config.seed,
            ),
            EaBackend::Moead => Self::run_moead_phase(
                problem,
                &config.moead_config,
                seed_population,
                budget,
                config.seed,
            ),
        }
    }

    fn run_nsga2_phase<P, const D: usize>(
        problem: &P,
        nsga2_config: &Nsga2Config,
        seed_population: Vec<BitsetEncodedSolution<P, D>>,
        budget: Duration,
        seed: u64,
    ) -> (Vec<BitsetEncodedSolution<P, D>>, ExploredSolutionsData<D>)
    where
        P: SetCoverProblem<D> + Clone + Send + Sync,
    {
        let config = nsga2_config.clone();

        let initial_pop = if seed_population.is_empty() {
            None
        } else {
            Some(seed_population)
        };

        let mut nsga2 = Nsga2::<P, D>::new(
            problem,
            config,
            initial_pop,
            seed.wrapping_add(10_000), // offset seed to differ from PLS
        );

        let archive = nsga2.run(usize::MAX, budget);
        let explored = std::mem::replace(
            &mut nsga2.explored_solutions,
            ExploredSolutionsData::new(problem.max_objectives()),
        );

        (archive, explored)
    }

    fn run_moead_phase<P, const D: usize>(
        problem: &P,
        moead_config: &MoeadConfig,
        seed_population: Vec<BitsetEncodedSolution<P, D>>,
        budget: Duration,
        seed: u64,
    ) -> (Vec<BitsetEncodedSolution<P, D>>, ExploredSolutionsData<D>)
    where
        P: SetCoverProblem<D> + Clone + Send + Sync,
    {
        let config = moead_config.clone();

        let initial_pop = if seed_population.is_empty() {
            None
        } else {
            Some(seed_population)
        };

        let mut moead = Moead::<P, D>::new(problem, config, initial_pop, seed.wrapping_add(10_000));

        let archive = moead.run(usize::MAX, budget);
        let explored = std::mem::replace(
            &mut moead.explored_solutions,
            ExploredSolutionsData::new(problem.max_objectives()),
        );

        (archive, explored)
    }

    // -----------------------------------------------------------------
    // Seed population selection
    // -----------------------------------------------------------------

    /// Select solutions from the PLS archive to seed the EA population.
    ///
    /// If `max_size` is 0 or the archive is smaller than `max_size`, all
    /// solutions are returned. Otherwise, farthest-point sampling is used
    /// to select a diverse subset in normalised objective space.
    fn select_seed_population<P, const D: usize>(
        archive: &[BitsetEncodedSolution<P, D>],
        max_size: usize,
        seed: u64,
    ) -> Vec<BitsetEncodedSolution<P, D>>
    where
        P: SetCoverProblem<D> + Clone + Send + Sync,
    {
        if archive.is_empty() {
            return Vec::new();
        }

        if max_size == 0 || archive.len() <= max_size {
            return archive.to_vec();
        }

        // Farthest-point sampling in normalised objective space
        Self::farthest_point_sample(archive, max_size, seed)
    }

    /// Greedy farthest-point sampling: iteratively picks the solution with
    /// maximum minimum-distance to already-selected solutions.
    ///
    /// This produces a well-spread subset that preserves diversity in the
    /// objective space, much better than random sampling for seeding an EA.
    fn farthest_point_sample<P, const D: usize>(
        archive: &[BitsetEncodedSolution<P, D>],
        target_size: usize,
        seed: u64,
    ) -> Vec<BitsetEncodedSolution<P, D>>
    where
        P: SetCoverProblem<D> + Clone + Send + Sync,
    {
        let n = archive.len();

        // 1. Normalise objectives to [0,1]
        let mut obj_min = [f64::INFINITY; D];
        let mut obj_max = [f64::NEG_INFINITY; D];
        for sol in archive {
            for d in 0..D {
                let v = sol.objectives()[d] as f64;
                if v < obj_min[d] {
                    obj_min[d] = v;
                }
                if v > obj_max[d] {
                    obj_max[d] = v;
                }
            }
        }
        let obj_range: Vec<f64> = (0..D)
            .map(|d| {
                let r = obj_max[d] - obj_min[d];
                if r < f64::EPSILON { 1.0 } else { r }
            })
            .collect();

        let normalised: Vec<[f64; D]> = archive
            .iter()
            .map(|sol| {
                let mut normed = [0.0f64; D];
                for d in 0..D {
                    normed[d] = (sol.objectives()[d] as f64 - obj_min[d]) / obj_range[d];
                }
                normed
            })
            .collect();

        // 2. Greedy farthest-point selection
        let mut selected_indices: Vec<usize> = Vec::with_capacity(target_size);
        let mut available = vec![true; n];
        let mut min_dist = vec![f64::INFINITY; n];

        // Seed with a deterministic first pick based on the seed
        let mut rng = SmallRng::seed_from_u64(seed);
        let first = rand::Rng::random_range(&mut rng, 0..n);
        selected_indices.push(first);
        available[first] = false;

        // Update min_dist from first pick
        for i in 0..n {
            if available[i] {
                let d = euclidean_distance(&normalised[i], &normalised[first]);
                if d < min_dist[i] {
                    min_dist[i] = d;
                }
            }
        }

        while selected_indices.len() < target_size {
            // Pick the available candidate with largest min_dist
            let mut best_idx = usize::MAX;
            let mut best_dist = f64::NEG_INFINITY;
            for i in 0..n {
                if available[i] && min_dist[i] > best_dist {
                    best_dist = min_dist[i];
                    best_idx = i;
                }
            }
            if best_idx == usize::MAX {
                break;
            }

            selected_indices.push(best_idx);
            available[best_idx] = false;

            // Update min_dist for remaining candidates
            let picked = &normalised[best_idx];
            for i in 0..n {
                if available[i] {
                    let d = euclidean_distance(&normalised[i], picked);
                    if d < min_dist[i] {
                        min_dist[i] = d;
                    }
                }
            }
        }

        selected_indices
            .into_iter()
            .map(|i| archive[i].clone())
            .collect()
    }

    // -----------------------------------------------------------------
    // Archive merging
    // -----------------------------------------------------------------

    /// Merge two archives, removing any dominated solutions.
    /// Returns the combined non-dominated set.
    fn merge_archives<P, const D: usize>(
        archive_a: Vec<BitsetEncodedSolution<P, D>>,
        archive_b: Vec<BitsetEncodedSolution<P, D>>,
    ) -> Vec<BitsetEncodedSolution<P, D>>
    where
        P: SetCoverProblem<D> + Clone + Send + Sync,
    {
        let mut merged: Vec<BitsetEncodedSolution<P, D>> = Vec::new();

        // Insert all solutions from both archives into a non-dominated set
        let all_solutions = archive_a.into_iter().chain(archive_b);

        for candidate in all_solutions {
            let dominated_by_existing = merged.iter().any(|existing| {
                existing.dominates(candidate.objectives())
                    || existing.objectives() == candidate.objectives()
            });

            if !dominated_by_existing {
                // Remove any existing solutions dominated by the candidate
                merged.retain(|existing| !candidate.dominates(existing.objectives()));
                merged.push(candidate);
            }
        }

        merged
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Euclidean distance between two D-dimensional points.
#[inline]
fn euclidean_distance<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    let mut sum = 0.0f64;
    for d in 0..D {
        let diff = a[d] - b[d];
        sum += diff * diff;
    }
    sum.sqrt()
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Run the memetic hybrid with NSGA-II backend using default configuration.
///
/// This is the simplest entry point for the memetic approach.
pub fn run_memetic_nsga2<P, const D: usize>(
    problem: &P,
    timeout: Duration,
    pop_size: usize,
    pls_time_fraction: f64,
    seed: u64,
) -> MemeticResult<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    let config = MemeticConfig {
        pls_time_fraction,
        ea_backend: EaBackend::Nsga2,
        nsga2_config: Nsga2Config {
            population_size: pop_size,
            ..Nsga2Config::default()
        },
        seed,
        ..MemeticConfig::default()
    };

    MemeticAlgorithm::run(problem, config, timeout)
}

/// Run the memetic hybrid with MOEA/D backend using default configuration.
pub fn run_memetic_moead<P, const D: usize>(
    problem: &P,
    timeout: Duration,
    pls_time_fraction: f64,
    seed: u64,
) -> MemeticResult<P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    let config = MemeticConfig {
        pls_time_fraction,
        ea_backend: EaBackend::Moead,
        seed,
        ..MemeticConfig::default()
    };

    MemeticAlgorithm::run(problem, config, timeout)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objectives::ObjectiveType;
    use crate::problem_bitset::ProblemBitset;
    use crate::solution::SIMSModifiable;
    use std::path::PathBuf;

    const NUM_OBJECTIVES: usize = 4;
    const OBJECTIVE_TYPES: [ObjectiveType; NUM_OBJECTIVES] = [
        ObjectiveType::TotalCost,
        ObjectiveType::CloudyArea,
        ObjectiveType::MinResolution,
        ObjectiveType::MaxIncidenceAngle,
    ];

    fn find_test_instance() -> Option<PathBuf> {
        // Try multiple known locations for test data
        let candidates = [
            "tests/data/lagos_nigeria_30.dzn",
            "../augmecon-rs/tests/input/sims/lagos_nigeria_30.dzn",
            "../sims-core/tests/data/lagos_nigeria_30.dzn",
        ];
        for path in &candidates {
            let p = PathBuf::from(path);
            if p.exists() {
                return Some(p);
            }
        }
        None
    }

    fn make_test_problem() -> Option<ProblemBitset<NUM_OBJECTIVES>> {
        let path = find_test_instance()?;
        ProblemBitset::from_minizinc_datafile(&path, OBJECTIVE_TYPES).ok()
    }

    #[test]
    fn test_memetic_nsga2_runs_and_produces_solutions() {
        let problem = match make_test_problem() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no test instance found");
                return;
            }
        };

        let result = run_memetic_nsga2(
            &problem,
            Duration::from_secs(3),
            50,  // pop_size
            0.3, // 30% PLS
            42,
        );

        assert!(
            !result.archive.is_empty(),
            "Memetic NSGA-II should produce at least one solution"
        );
        assert!(result.pls_archive_size > 0, "PLS should find solutions");
        assert!(result.ea_archive_size > 0, "EA should find solutions");
        assert!(
            result.pls_duration > Duration::ZERO,
            "PLS phase should take some time"
        );
        assert!(
            result.ea_duration > Duration::ZERO,
            "EA phase should take some time"
        );

        // Verify all solutions are feasible (cover all elements)
        for sol in &result.archive {
            assert!(
                sol.is_valid(&problem),
                "All archive solutions must be feasible"
            );
        }

        // Verify no dominated solutions in the archive
        for (i, a) in result.archive.iter().enumerate() {
            for (j, b) in result.archive.iter().enumerate() {
                if i != j {
                    assert!(
                        !a.dominates(b.objectives()),
                        "Archive should contain no dominated solutions: {} dominates {}",
                        i,
                        j,
                    );
                }
            }
        }

        println!(
            "Memetic NSGA-II: PLS={} sols in {:?}, EA={} sols in {:?}, merged={} sols",
            result.pls_archive_size,
            result.pls_duration,
            result.ea_archive_size,
            result.ea_duration,
            result.archive.len(),
        );
    }

    #[test]
    fn test_memetic_moead_runs_and_produces_solutions() {
        let problem = match make_test_problem() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no test instance found");
                return;
            }
        };

        let config = MemeticConfig {
            pls_time_fraction: 0.3,
            ea_backend: EaBackend::Moead,
            moead_config: MoeadConfig {
                num_divisions: 5, // small for fast test
                auto_divisions: false,
                ..MoeadConfig::default()
            },
            pls_initial_pop_size: 20,
            seed: 123,
            ..MemeticConfig::default()
        };

        let result = MemeticAlgorithm::run::<ProblemBitset<NUM_OBJECTIVES>, NUM_OBJECTIVES>(
            &problem,
            config,
            Duration::from_secs(3),
        );

        assert!(
            !result.archive.is_empty(),
            "Memetic MOEA/D should produce at least one solution"
        );

        // Verify feasibility
        for sol in &result.archive {
            assert!(
                sol.is_valid(&problem),
                "All archive solutions must be feasible"
            );
        }

        println!(
            "Memetic MOEA/D: PLS={} sols in {:?}, EA={} sols in {:?}, merged={} sols",
            result.pls_archive_size,
            result.pls_duration,
            result.ea_archive_size,
            result.ea_duration,
            result.archive.len(),
        );
    }

    #[test]
    fn test_memetic_with_high_pls_fraction() {
        let problem = match make_test_problem() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no test instance found");
                return;
            }
        };

        // Give 80% to PLS, only 20% to EA
        let result = run_memetic_nsga2(&problem, Duration::from_secs(3), 30, 0.8, 99);

        assert!(
            !result.archive.is_empty(),
            "Should produce solutions even with high PLS fraction"
        );

        // PLS should have had enough time to find decent solutions
        assert!(
            result.pls_archive_size >= 1,
            "PLS with 80% budget should find multiple solutions"
        );
    }

    #[test]
    fn test_farthest_point_sampling() {
        let problem = match make_test_problem() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no test instance found");
                return;
            }
        };

        // Generate a bunch of random solutions
        let solutions: Vec<BitsetEncodedSolution<ProblemBitset<NUM_OBJECTIVES>, NUM_OBJECTIVES>> =
            (0..50)
                .map(|i| BitsetEncodedSolution::random_with_seed(&problem, i))
                .collect();

        // Sample a subset
        let subset = MemeticAlgorithm::farthest_point_sample::<
            ProblemBitset<NUM_OBJECTIVES>,
            NUM_OBJECTIVES,
        >(&solutions, 10, 42);

        assert_eq!(subset.len(), 10, "Should select exactly 10 solutions");

        // Verify all selected solutions exist in the original set
        for sel in &subset {
            assert!(
                solutions.iter().any(|s| s.objectives() == sel.objectives()),
                "Selected solution should be from the original set"
            );
        }
    }

    #[test]
    fn test_merge_archives_removes_dominated() {
        let problem = match make_test_problem() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no test instance found");
                return;
            }
        };

        let archive_a: Vec<BitsetEncodedSolution<ProblemBitset<NUM_OBJECTIVES>, NUM_OBJECTIVES>> =
            (0..20)
                .map(|i| BitsetEncodedSolution::random_with_seed(&problem, i))
                .collect();
        let archive_b: Vec<BitsetEncodedSolution<ProblemBitset<NUM_OBJECTIVES>, NUM_OBJECTIVES>> =
            (100..120)
                .map(|i| BitsetEncodedSolution::random_with_seed(&problem, i))
                .collect();

        let merged = MemeticAlgorithm::merge_archives(archive_a, archive_b);

        // Verify no dominated solutions in merged archive
        for (i, a) in merged.iter().enumerate() {
            for (j, b) in merged.iter().enumerate() {
                if i != j {
                    assert!(
                        !a.dominates(b.objectives()),
                        "Merged archive should have no dominated solutions"
                    );
                }
            }
        }
    }

    #[test]
    fn test_memetic_no_merge_option() {
        let problem = match make_test_problem() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no test instance found");
                return;
            }
        };

        let config = MemeticConfig {
            pls_time_fraction: 0.3,
            ea_backend: EaBackend::Nsga2,
            nsga2_config: Nsga2Config {
                population_size: 30,
                ..Nsga2Config::default()
            },
            merge_pls_archive: false, // Don't merge PLS archive
            pls_initial_pop_size: 20,
            seed: 42,
            ..MemeticConfig::default()
        };

        let result = MemeticAlgorithm::run::<ProblemBitset<NUM_OBJECTIVES>, NUM_OBJECTIVES>(
            &problem,
            config,
            Duration::from_secs(2),
        );

        assert!(
            !result.archive.is_empty(),
            "Should produce solutions even without merging"
        );

        // Without merge, the final archive size should equal the EA archive size
        assert_eq!(
            result.archive.len(),
            result.ea_archive_size,
            "Without merge, final archive should be EA-only"
        );
    }

    #[test]
    fn test_select_seed_population_no_cap() {
        let problem = match make_test_problem() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no test instance found");
                return;
            }
        };

        let archive: Vec<BitsetEncodedSolution<ProblemBitset<NUM_OBJECTIVES>, NUM_OBJECTIVES>> = (0
            ..10)
            .map(|i| BitsetEncodedSolution::random_with_seed(&problem, i))
            .collect();

        // max_size = 0 means no cap
        let seed = MemeticAlgorithm::select_seed_population(&archive, 0, 42);
        assert_eq!(seed.len(), 10, "No cap should return all solutions");

        // max_size > archive size should also return all
        let seed = MemeticAlgorithm::select_seed_population(&archive, 100, 42);
        assert_eq!(seed.len(), 10, "Cap larger than archive should return all");
    }

    #[test]
    fn test_select_seed_population_with_cap() {
        let problem = match make_test_problem() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no test instance found");
                return;
            }
        };

        let archive: Vec<BitsetEncodedSolution<ProblemBitset<NUM_OBJECTIVES>, NUM_OBJECTIVES>> = (0
            ..30)
            .map(|i| BitsetEncodedSolution::random_with_seed(&problem, i))
            .collect();

        let seed = MemeticAlgorithm::select_seed_population(&archive, 10, 42);
        assert_eq!(seed.len(), 10, "Should respect the cap");
    }

    #[test]
    fn test_empty_archive_handling() {
        let empty: Vec<BitsetEncodedSolution<ProblemBitset<NUM_OBJECTIVES>, NUM_OBJECTIVES>> =
            Vec::new();
        let seed = MemeticAlgorithm::select_seed_population(&empty, 10, 42);
        assert!(seed.is_empty(), "Empty archive should produce empty seed");
    }

    #[test]
    fn test_ea_backend_display() {
        assert_eq!(format!("{}", EaBackend::Nsga2), "NSGA-II");
        assert_eq!(format!("{}", EaBackend::Moead), "MOEA/D");
    }
}
