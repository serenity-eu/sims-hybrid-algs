//! Literal NSGA-II baseline, following Deb, Pratap, Agarwal & Meyarivan (2002)
//! as closely as the SIMS set-cover representation allows.
//!
//! "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"
//! IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
//!
//! This module exists to give the SIMS-tailored `nsga2.rs` (which adds
//! contribution-distance selection for D>=3, stagnation injection, an
//! `ensure_mutation` guard, and a composite mutation suite) an unmodified
//! reference point. None of those four additions are in the 2002 paper, so
//! they should not be assumed to help without an ablation against this
//! baseline.
//!
//! Deviations from the paper that are unavoidable for this problem domain
//! (not algorithmic choices):
//! - The paper is generic over crossover/mutation operators (real-coded SBX
//!   + polynomial mutation in their experiments). For a discrete set-cover
//!   bitset we use uniform crossover + per-bit mutation, each followed by
//!   greedy repair to restore feasibility -- the established pattern for
//!   binary-encoded set-covering GAs (Beasley & Chu, 1996).
//! - There is no external archive in the paper -- the result is the
//!   non-dominated subset of the final generation's population. We keep an
//!   `ExploredSolutionsData` log purely for trace/HV-curve instrumentation;
//!   it does not feed back into the algorithm.
//!
//! Faithfully reproduced, directly from the paper's boxed procedures:
//! - `fast-non-dominated-sort` (Section III-A)
//! - `crowding-distance-assignment` (Section III-B.1)
//! - the crowded-comparison operator `<_n` (Section III-B.2): lower rank
//!   wins; ties broken by higher crowding distance
//! - the main loop (Section III-C): `R_t = P_t ∪ Q_t`, sort into fronts,
//!   fill `P_{t+1}` front-by-front, and for the partial last front, sort by
//!   crowding distance descending and take the top `N - |P_{t+1}|` -- with
//!   *no* fallback to a different selection rule for D >= 3; the paper uses
//!   crowding distance regardless of objective count.

use std::time::Duration;

use pareto::{HasObjectives, MoSolution};
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use tracing::{info, info_span};

use crate::explored_solutions_data::ExploredSolutionsData;
use crate::problem::SetCoverProblem;
use crate::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;
use crate::timer::Timer;

use super::operators::{
    binary_tournament, bitflip_mutation, crowding_distance, crowding_from_fronts,
    fast_non_dominated_sort, random_population, ranks_from_fronts, uniform_crossover,
};

/// Incrementally maintain a non-dominated external archive.
///
/// Skips `solution` if it is dominated by any current archive member, or if the
/// archive already holds a solution at the exact same objective point. The
/// equal-objective skip is what keeps this unbounded archive from bloating: with
/// strict dominance alone, every generation's re-discovery of the same (cost,
/// cloud) trade-off (different image-set, identical objectives) was pushed as a
/// new entry — inflating `final_solutions` into the 100k+ range without adding
/// any hypervolume. Deduping by objective value collapses that to the true
/// distinct-front size while leaving HV (and every dominance relation) identical.
pub(crate) fn archive_update<P, const D: usize>(
    archive: &mut Vec<BitsetEncodedSolution<P, D>>,
    solution: BitsetEncodedSolution<P, D>,
) where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    if archive
        .iter()
        .any(|a| a.dominates(solution.objectives()) || a.objectives() == solution.objectives())
    {
        return;
    }
    archive.retain(|a| !solution.dominates(a.objectives()));
    archive.push(solution);
}

/// Configuration for the literal NSGA-II baseline.
#[derive(Debug, Clone)]
pub struct Nsga2BaselineConfig {
    /// Population size N.
    pub population_size: usize,
    /// Crossover probability.
    pub crossover_rate: f64,
    /// Per-bit mutation probability.
    pub mutation_rate: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for Nsga2BaselineConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            crossover_rate: 0.9,
            mutation_rate: 0.01,
            seed: 42,
        }
    }
}

/// Literal NSGA-II baseline (Deb et al., 2002).
pub struct Nsga2Baseline<'a, P, const D: usize>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    problem: &'a P,
    config: Nsga2BaselineConfig,
    population: Vec<BitsetEncodedSolution<P, D>>,
    pub explored_solutions: ExploredSolutionsData<D>,
    /// Unbounded external Pareto archive — accumulates every non-dominated
    /// solution seen across all generations. The final result is drawn from
    /// here rather than the last population, so it is not capped at N.
    pub external_archive: Vec<BitsetEncodedSolution<P, D>>,
    rng: SmallRng,
}

impl<'a, P, const D: usize> Nsga2Baseline<'a, P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    /// "Initially a random parent population P0 is created."
    pub fn new(problem: &'a P, config: Nsga2BaselineConfig) -> Self {
        Self::new_with_seed_population(problem, config, vec![])
    }

    /// Create with an externally-provided seed population.
    ///
    /// Seeds are used as-is; random solutions are added up to `population_size`.
    pub fn new_with_seed_population(
        problem: &'a P,
        config: Nsga2BaselineConfig,
        seed_population: Vec<BitsetEncodedSolution<P, D>>,
    ) -> Self {
        let mut rng = SmallRng::seed_from_u64(config.seed);
        let mut population = seed_population;
        let random_count = config.population_size.saturating_sub(population.len());
        population.extend(random_population(problem, random_count, &mut rng));
        let explored_solutions = ExploredSolutionsData::new(problem.max_objectives());

        Self {
            problem,
            config,
            population,
            explored_solutions,
            external_archive: Vec::new(),
            rng,
        }
    }

    /// Run for `max_generations` or `max_duration`, whichever comes first.
    /// Returns the non-dominated subset of the final population.
    pub fn run(
        &mut self,
        max_generations: usize,
        max_duration: Duration,
    ) -> Vec<BitsetEncodedSolution<P, D>> {
        let timer = Timer::start(max_duration);
        let target_size = self.config.population_size;

        for sol in &self.population {
            self.explored_solutions
                .register_without_selected_images(0, sol, Duration::ZERO);
            archive_update(&mut self.external_archive, sol.clone());
        }

        for generation in 1..=max_generations {
            let gen_span = info_span!("nsga2_baseline_generation", generation = generation);
            let _guard = gen_span.enter();

            if timer.is_expired() {
                info!(
                    "NSGA-II baseline timeout after {} generations",
                    generation - 1
                );
                break;
            }

            // Rank + crowding distance of the current population, used by the
            // crowded-comparison tournament that builds Q_t.
            let fronts = fast_non_dominated_sort(&self.population);
            let ranks = ranks_from_fronts(&fronts, self.population.len());
            let crowding = crowding_from_fronts(&self.population, &fronts);

            // make-new-pop(P_t): binary tournament + crossover + mutation, size N.
            let offspring = self.make_new_pop(&ranks, &crowding);
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

            // R_t = P_t ∪ Q_t (size 2N), then elitist replacement.
            let mut combined = std::mem::take(&mut self.population);
            combined.extend(offspring);
            self.population = self.survivor_selection(combined, target_size);

            if generation == max_generations {
                info!("NSGA-II baseline reached max generations ({max_generations})");
            }
        }

        self.explored_solutions.num_iterations = max_generations;

        info!(
            "NSGA-II baseline: {} solutions in external archive (vs pop size {})",
            self.external_archive.len(),
            target_size,
        );

        self.external_archive.clone()
    }

    fn make_new_pop(
        &mut self,
        ranks: &[usize],
        crowding: &[f64],
    ) -> Vec<BitsetEncodedSolution<P, D>> {
        let pop_size = self.population.len();
        let target_size = self.config.population_size;
        let mut offspring = Vec::with_capacity(target_size);

        while offspring.len() < target_size {
            let p1 = binary_tournament(ranks, crowding, &mut self.rng, pop_size);
            let p2 = binary_tournament(ranks, crowding, &mut self.rng, pop_size);

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

    /// `R_t = P_t ∪ Q_t`, sort into fronts, fill `P_{t+1}` front-by-front;
    /// for the partial last front, sort by crowding distance descending.
    fn survivor_selection(
        &self,
        combined: Vec<BitsetEncodedSolution<P, D>>,
        target_size: usize,
    ) -> Vec<BitsetEncodedSolution<P, D>> {
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
                let cd = crowding_distance(&combined, front);
                let mut ranked: Vec<(usize, f64)> = front
                    .iter()
                    .enumerate()
                    .map(|(local, &global)| (global, cd[local]))
                    .collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (global_idx, _cd) in ranked.into_iter().take(remaining) {
                    next_gen.push(combined[global_idx].clone());
                }
                break;
            }
        }

        next_gen
    }

    /// Get a reference to the current population.
    pub fn population(&self) -> &[BitsetEncodedSolution<P, D>] {
        &self.population
    }

    /// Get explored solutions data (compatible with PLS/EA output format).
    pub fn explored_solutions_data(&self) -> &ExploredSolutionsData<D> {
        &self.explored_solutions
    }
}

/// Run the literal NSGA-II baseline and return (final non-dominated set, explored solutions).
pub fn run_nsga2_baseline<P, const D: usize>(
    problem: &P,
    config: Nsga2BaselineConfig,
    max_generations: usize,
    max_duration: Duration,
) -> (Vec<BitsetEncodedSolution<P, D>>, ExploredSolutionsData<D>)
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    run_nsga2_baseline_seeded(problem, config, vec![], max_generations, max_duration)
}

/// Run the literal NSGA-II baseline seeded with an external population.
pub fn run_nsga2_baseline_seeded<P, const D: usize>(
    problem: &P,
    config: Nsga2BaselineConfig,
    seed_population: Vec<BitsetEncodedSolution<P, D>>,
    max_generations: usize,
    max_duration: Duration,
) -> (Vec<BitsetEncodedSolution<P, D>>, ExploredSolutionsData<D>)
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    let mut nsga2 = Nsga2Baseline::new_with_seed_population(problem, config, seed_population);
    let result = nsga2.run(max_generations, max_duration);
    let explored = std::mem::replace(
        &mut nsga2.explored_solutions,
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

    fn make_test_problem() -> ProblemBitset<2> {
        let raw = SIMSProblemInstanceRaw {
            name: "nsga2_baseline_test".to_string(),
            num_images: 4,
            universe_size: 5,
            images: vec![
                vec![0, 1, 2],
                vec![2, 3, 4],
                vec![0, 1, 3, 4],
                vec![0, 1, 2, 3, 4],
            ],
            costs: vec![10, 20, 30, 50],
            clouds: vec![vec![], vec![], vec![], vec![]],
            areas: vec![1, 1, 1, 1, 1],
            max_cloud_area: 0,
            resolution: vec![100, 200, 150, 300],
            incidence_angle: vec![5, 10, 8, 12],
        };
        let objective_types = [ObjectiveType::TotalCost, ObjectiveType::CloudyArea];
        ProblemBitset::from_raw_with_objectives(&raw, objective_types)
    }

    #[test]
    fn test_nsga2_baseline_runs_and_produces_feasible_solutions() {
        let problem = make_test_problem();
        let config = Nsga2BaselineConfig {
            population_size: 20,
            ..Default::default()
        };

        let mut nsga2 = Nsga2Baseline::new(&problem, config);
        let result = nsga2.run(50, Duration::from_secs(5));

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
    fn test_nsga2_baseline_respects_timeout() {
        let problem = make_test_problem();
        let config = Nsga2BaselineConfig {
            population_size: 20,
            ..Default::default()
        };

        let start = std::time::Instant::now();
        let mut nsga2 = Nsga2Baseline::new(&problem, config);
        let _result = nsga2.run(1_000_000, Duration::from_millis(200));
        let elapsed = start.elapsed();

        assert!(
            elapsed < Duration::from_secs(2),
            "Should respect timeout, but took {:?}",
            elapsed
        );
    }

    #[test]
    fn test_run_nsga2_baseline_convenience() {
        let problem = make_test_problem();
        let config = Nsga2BaselineConfig {
            population_size: 15,
            ..Default::default()
        };

        let (result, explored) = run_nsga2_baseline(&problem, config, 20, Duration::from_secs(3));

        assert!(!result.is_empty());
        assert!(!explored.solutions.is_empty());
    }
}
