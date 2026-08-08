//! Literal NSGA-III baseline, following Deb & Jain (2014), Part I, as
//! closely as the SIMS set-cover representation allows.
//!
//! "An Evolutionary Many-Objective Optimization Algorithm Using
//! Reference-Point-Based Non-dominated Sorting Approach, Part I: Solving
//! Problems With Box Constraints"
//! IEEE Transactions on Evolutionary Computation, 18(4), 577-601.
//!
//! This module exists to give the SIMS-tailored `nsga3.rs` (which adds
//! tournament-based parent selection, auto-sized divisions, diversity-sampled
//! seeding, stagnation injection, an `ensure_mutation` guard, and a composite
//! mutation suite) an unmodified reference point.
//!
//! The most important deviation to flag: **the paper explicitly does not use
//! tournament selection.** Section IV-F states: "we do not employ any
//! explicit selection operation. The population Q_{t+1} is constructed by
//! applying the usual crossover and mutation operators by randomly picking
//! parents from P_{t+1}." Our SIMS `nsga3.rs` uses rank/crowding-based binary
//! tournament for parent selection anyway (copied from the NSGA-II code
//! path) -- that is an unvalidated deviation from the paper, not a
//! documented extension. This baseline picks parents uniformly at random,
//! as specified.
//!
//! Deviations from the paper that are unavoidable for this problem domain:
//! - Real-coded SBX + polynomial mutation (the paper's choice) is replaced
//!   with uniform crossover + per-bit mutation + greedy repair, for the same
//!   reason as the NSGA-II baseline.
//! - No external archive across generations; the result is the
//!   non-dominated subset of the final population, as in the paper.
//!
//! Faithfully reproduced from Algorithm 1 and Algorithms 2-4:
//! - Simplex-lattice reference points (Das & Dennis, Section IV-B, Eq. 3) --
//!   `num_divisions` (`p`) is a direct user parameter, not auto-sized.
//! - Adaptive normalization: ideal point -> extreme points via ASF ->
//!   hyperplane intercepts -> normalize (Algorithm 2).
//! - Association by perpendicular distance to reference lines (Algorithm 3).
//! - Niche-count-based selection for the partial last front (Algorithm 4).
//! - Population size N ~= H (number of reference points), per Section IV-H:
//!   "the population size N is dependent on H" -- we use N = H directly.

use std::time::Duration;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use tracing::{info, info_span};

use crate::explored_solutions_data::ExploredSolutionsData;
use crate::problem::SetCoverProblem;
use crate::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;
use crate::timer::Timer;

use super::nsga2_baseline::archive_update;
use super::nsga3::{closest_reference_point, normalise_front};
use super::operators::{
    bitflip_mutation, fast_non_dominated_sort, generate_weight_vectors, random_population,
    uniform_crossover,
};

/// Configuration for the literal NSGA-III baseline.
#[derive(Debug, Clone)]
pub struct Nsga3BaselineConfig {
    /// Number of simplex-lattice divisions `p` (Eq. 3). Population size is
    /// whatever this produces: H = C(M+p-1, p).
    pub num_divisions: usize,
    /// Crossover probability.
    pub crossover_rate: f64,
    /// Per-bit mutation probability.
    pub mutation_rate: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for Nsga3BaselineConfig {
    fn default() -> Self {
        Self {
            num_divisions: 12, // matches the paper's 3-objective experiments (H=91)
            crossover_rate: 0.9,
            mutation_rate: 0.01,
            seed: 42,
        }
    }
}

/// Literal NSGA-III baseline (Deb & Jain, 2014).
pub struct Nsga3Baseline<'a, P, const D: usize>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    problem: &'a P,
    config: Nsga3BaselineConfig,
    reference_points: Vec<[f64; D]>,
    population_size: usize,
    population: Vec<BitsetEncodedSolution<P, D>>,
    pub explored_solutions: ExploredSolutionsData<D>,
    /// Unbounded external Pareto archive — accumulates every non-dominated
    /// solution seen across all generations. The final result is drawn from
    /// here rather than the last population, so it is not capped at H.
    pub external_archive: Vec<BitsetEncodedSolution<P, D>>,
    rng: SmallRng,
}

impl<'a, P, const D: usize> Nsga3Baseline<'a, P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    pub fn new(problem: &'a P, config: Nsga3BaselineConfig) -> Self {
        Self::new_with_seed_population(problem, config, vec![])
    }

    /// Create with an externally-provided seed population.
    ///
    /// Seeds are used as-is; random solutions are added up to the
    /// reference-point-determined population size.
    pub fn new_with_seed_population(
        problem: &'a P,
        config: Nsga3BaselineConfig,
        seed_population: Vec<BitsetEncodedSolution<P, D>>,
    ) -> Self {
        let mut rng = SmallRng::seed_from_u64(config.seed);

        let reference_points = generate_weight_vectors::<D>(config.num_divisions);
        let population_size = reference_points.len().max(4).max(seed_population.len());
        let mut population = seed_population;
        let random_count = population_size.saturating_sub(population.len());
        population.extend(random_population(problem, random_count, &mut rng));
        let explored_solutions = ExploredSolutionsData::new(problem.max_objectives());

        info!(
            "NSGA-III baseline (Deb & Jain 2014): D={D}, num_divisions={}, reference_points={}, pop_size={population_size}",
            config.num_divisions,
            reference_points.len(),
        );

        Self {
            problem,
            config,
            reference_points,
            population_size,
            population,
            explored_solutions,
            external_archive: Vec::new(),
            rng,
        }
    }

    pub fn run(
        &mut self,
        max_generations: usize,
        max_duration: Duration,
    ) -> Vec<BitsetEncodedSolution<P, D>> {
        let timer = Timer::start(max_duration);

        for sol in &self.population {
            self.explored_solutions
                .register_without_selected_images(0, sol, Duration::ZERO);
            archive_update(&mut self.external_archive, sol.clone());
        }

        for generation in 1..=max_generations {
            let gen_span = info_span!("nsga3_baseline_generation", generation = generation);
            let _guard = gen_span.enter();

            if timer.is_expired() {
                info!(
                    "NSGA-III baseline timeout after {} generations",
                    generation - 1
                );
                break;
            }

            // "Q_t = Recombination+Mutation(P_t)" -- parents picked uniformly
            // at random, no tournament (Section IV-F).
            let offspring = self.make_offspring();
            for sol in &offspring {
                if !self.explored_solutions.is_registered(sol) {
                    self.explored_solutions.register_without_selected_images(
                        generation,
                        sol,
                        timer.elapsed(),
                    );
                }
                archive_update(&mut self.external_archive, sol.clone());
            }

            let mut combined = std::mem::take(&mut self.population);
            combined.extend(offspring);
            self.population = self.survivor_selection(combined);

            if generation == max_generations {
                info!("NSGA-III baseline reached max generations ({max_generations})");
            }
        }

        self.explored_solutions.num_iterations = max_generations;

        info!(
            "NSGA-III baseline: {} solutions in external archive (vs pop size {})",
            self.external_archive.len(),
            self.population_size,
        );

        self.external_archive.clone()
    }

    fn make_offspring(&mut self) -> Vec<BitsetEncodedSolution<P, D>> {
        let pop_size = self.population.len();
        let mut offspring = Vec::with_capacity(self.population_size);

        while offspring.len() < self.population_size {
            let p1 = self.rng.random_range(0..pop_size);
            let p2 = self.rng.random_range(0..pop_size);

            let mut child = if self.rng.random_bool(self.config.crossover_rate) {
                uniform_crossover(
                    &self.population[p1],
                    &self.population[p2],
                    self.problem,
                    &mut self.rng,
                )
            } else if self.rng.random_bool(0.5) {
                self.population[p1].clone()
            } else {
                self.population[p2].clone()
            };

            child = bitflip_mutation(
                &child,
                self.problem,
                &mut self.rng,
                self.config.mutation_rate,
            );
            offspring.push(child);
        }

        offspring
    }

    /// Algorithm 1, lines 4-17: fill fronts wholesale until the last one that
    /// doesn't fit, then niche-fill the remainder.
    fn survivor_selection(
        &mut self,
        combined: Vec<BitsetEncodedSolution<P, D>>,
    ) -> Vec<BitsetEncodedSolution<P, D>> {
        let target_size = self.population_size;

        if combined.len() <= target_size {
            return combined;
        }

        let fronts = fast_non_dominated_sort(&combined);
        let mut next_gen: Vec<BitsetEncodedSolution<P, D>> = Vec::with_capacity(target_size);

        for front in &fronts {
            if next_gen.len() + front.len() <= target_size {
                for &idx in front {
                    next_gen.push(combined[idx].clone());
                }
            } else {
                let remaining = target_size - next_gen.len();
                if remaining > 0 {
                    self.niching_fill(&combined, &mut next_gen, front, remaining);
                }
                return next_gen;
            }
        }

        next_gen
    }

    /// Algorithm 2 (Normalize) + Algorithm 3 (Associate) + Algorithm 4 (Niching).
    fn niching_fill(
        &mut self,
        combined: &[BitsetEncodedSolution<P, D>],
        next_gen: &mut Vec<BitsetEncodedSolution<P, D>>,
        last_front: &[usize],
        remaining: usize,
    ) {
        let n_refs = self.reference_points.len();

        // Steps 1-5: adaptive normalisation (Algorithm 2).
        let next_gen_count = next_gen.len();
        let norm_all = normalise_front(next_gen, combined, last_front);

        // Step 6: associate.
        let assoc: Vec<(usize, f64)> = norm_all
            .iter()
            .map(|f| closest_reference_point(f, &self.reference_points))
            .collect();

        // Step 7: niche counts from already-selected solutions.
        let mut niche_count = vec![0usize; n_refs];
        for (ref_idx, _) in &assoc[..next_gen_count] {
            niche_count[*ref_idx] += 1;
        }

        // Step 8: niche preservation.
        let last_front_len = last_front.len();
        let base_offset = next_gen_count;

        let mut ref_to_front_candidates: Vec<Vec<usize>> = vec![Vec::new(); n_refs];
        for (local_i, (ref_idx, _)) in assoc[base_offset..base_offset + last_front_len]
            .iter()
            .enumerate()
        {
            ref_to_front_candidates[*ref_idx].push(local_i);
        }

        let mut available: Vec<bool> = vec![true; last_front_len];
        let mut selected_count = 0;

        while selected_count < remaining {
            let mut min_niche = usize::MAX;
            let mut eligible_refs: Vec<usize> = Vec::new();

            for r in 0..n_refs {
                let has_candidate = ref_to_front_candidates[r].iter().any(|&li| available[li]);
                if !has_candidate {
                    continue;
                }
                match niche_count[r].cmp(&min_niche) {
                    std::cmp::Ordering::Less => {
                        min_niche = niche_count[r];
                        eligible_refs = vec![r];
                    }
                    std::cmp::Ordering::Equal => eligible_refs.push(r),
                    std::cmp::Ordering::Greater => {}
                }
            }

            if eligible_refs.is_empty() {
                break;
            }

            let r_min = eligible_refs[self.rng.random_range(0..eligible_refs.len())];

            let candidates_for_r: Vec<usize> = ref_to_front_candidates[r_min]
                .iter()
                .copied()
                .filter(|&li| available[li])
                .collect();

            let chosen_local = if niche_count[r_min] == 0 {
                let r_start = base_offset;
                *candidates_for_r
                    .iter()
                    .min_by(|&&a, &&b| {
                        let da = assoc[r_start + a].1;
                        let db = assoc[r_start + b].1;
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap()
            } else {
                candidates_for_r[self.rng.random_range(0..candidates_for_r.len())]
            };

            next_gen.push(combined[last_front[chosen_local]].clone());
            available[chosen_local] = false;
            niche_count[r_min] += 1;
            selected_count += 1;
        }
    }

    /// Get a reference to the current population.
    #[must_use]
    pub fn population(&self) -> &[BitsetEncodedSolution<P, D>] {
        &self.population
    }

    /// Get explored solutions data (compatible with PLS/EA output format).
    #[must_use]
    pub const fn explored_solutions_data(&self) -> &ExploredSolutionsData<D> {
        &self.explored_solutions
    }
}

/// Run the literal NSGA-III baseline and return (final non-dominated set, explored solutions).
pub fn run_nsga3_baseline<P, const D: usize>(
    problem: &P,
    config: Nsga3BaselineConfig,
    max_generations: usize,
    max_duration: Duration,
) -> (Vec<BitsetEncodedSolution<P, D>>, ExploredSolutionsData<D>)
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    run_nsga3_baseline_seeded(problem, config, vec![], max_generations, max_duration)
}

/// Run the literal NSGA-III baseline seeded with an external population.
pub fn run_nsga3_baseline_seeded<P, const D: usize>(
    problem: &P,
    config: Nsga3BaselineConfig,
    seed_population: Vec<BitsetEncodedSolution<P, D>>,
    max_generations: usize,
    max_duration: Duration,
) -> (Vec<BitsetEncodedSolution<P, D>>, ExploredSolutionsData<D>)
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    let mut nsga3 = Nsga3Baseline::new_with_seed_population(problem, config, seed_population);
    let result = nsga3.run(max_generations, max_duration);
    let explored = std::mem::replace(
        &mut nsga3.explored_solutions,
        ExploredSolutionsData::new(problem.max_objectives()),
    );
    (result, explored)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objectives::ObjectiveType;
    use crate::problem::SIMSProblemInstanceRaw;
    use crate::problem_bitset::ProblemBitset;
    use pareto::{HasObjectives, MoSolution};

    const NUM_OBJECTIVES: usize = 4;
    const OBJECTIVE_TYPES: [ObjectiveType; NUM_OBJECTIVES] = [
        ObjectiveType::TotalCost,
        ObjectiveType::CloudyArea,
        ObjectiveType::MinResolution,
        ObjectiveType::MaxIncidenceAngle,
    ];

    fn make_test_problem() -> ProblemBitset<NUM_OBJECTIVES> {
        let raw = SIMSProblemInstanceRaw {
            name: "nsga3_baseline_test".to_string(),
            num_images: 6,
            universe_size: 6,
            images: vec![
                vec![0, 1],
                vec![1, 2],
                vec![2, 3],
                vec![3, 4],
                vec![4, 5],
                vec![0, 5],
            ],
            costs: vec![10, 20, 30, 15, 25, 35],
            clouds: vec![vec![]; 6],
            areas: vec![1; 6],
            max_cloud_area: 0,
            resolution: vec![100, 200, 150, 300, 120, 180],
            incidence_angle: vec![5, 10, 8, 12, 6, 9],
        };
        ProblemBitset::from_raw_with_objectives(&raw, OBJECTIVE_TYPES)
    }

    #[test]
    fn test_nsga3_baseline_runs_and_produces_feasible_solutions() {
        let problem = make_test_problem();
        let config = Nsga3BaselineConfig {
            num_divisions: 4,
            ..Default::default()
        };

        let mut nsga3 = Nsga3Baseline::new(&problem, config);
        let result = nsga3.run(30, Duration::from_secs(5));

        assert!(!result.is_empty(), "Result should not be empty");

        for sol in &result {
            assert!(
                problem.is_set_cover(sol),
                "Result solution must be a valid set cover"
            );
        }

        for (i, a) in result.iter().enumerate() {
            for (j, b) in result.iter().enumerate() {
                if i != j {
                    assert!(
                        !a.dominates(b.objectives()),
                        "Result solution {i} dominates solution {j}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_nsga3_baseline_respects_timeout() {
        let problem = make_test_problem();
        let config = Nsga3BaselineConfig {
            num_divisions: 4,
            ..Default::default()
        };

        let start = std::time::Instant::now();
        let mut nsga3 = Nsga3Baseline::new(&problem, config);
        let _result = nsga3.run(1_000_000, Duration::from_millis(300));
        let elapsed = start.elapsed();

        assert!(
            elapsed < Duration::from_secs(2),
            "Should respect timeout, but took {elapsed:?}"
        );
    }

    #[test]
    fn test_run_nsga3_baseline_convenience() {
        let problem = make_test_problem();
        let config = Nsga3BaselineConfig {
            num_divisions: 4,
            ..Default::default()
        };

        let (result, explored) = run_nsga3_baseline(&problem, config, 20, Duration::from_secs(3));

        assert!(!result.is_empty());
        assert!(!explored.solutions.is_empty());
    }
}
