//! NSGA-II (Non-dominated Sorting Genetic Algorithm II) for SIMS
//!
//! Implements the classic NSGA-II algorithm (Deb et al., 2002) adapted for the
//! Satellite Image Mosaic Selection problem. Key adaptations for the set-cover
//! constraint:
//!
//! - **Coverage-aware crossover**: Uniform crossover on image-selection bitsets
//!   followed by greedy repair to ensure all elements are covered.
//! - **Feasibility-preserving mutation**: Swap, add-then-prune, shift, and bit-flip
//!   mutations that maintain or restore feasibility via repair operators.
//! - **Redundancy removal**: After every genetic operation, redundant images
//!   (whose elements are fully covered by other selected images) are pruned
//!   to keep solutions lean.
//!
//! ## Algorithm Outline
//!
//! 1. Generate initial population of `N` random feasible solutions.
//! 2. **Selection**: Binary tournament using non-dominated rank and crowding distance.
//! 3. **Crossover**: Coverage-biased or uniform crossover with probability `p_c`.
//! 4. **Mutation**: Composite mutation (swap + shift + add-then-prune + bit-flip).
//! 5. **Survivor selection**: Merge parent + offspring (2N), non-dominated sort,
//!    fill next generation by front order, breaking the last front by a composite
//!    diversity score that combines crowding distance with contribution distance
//!    (minimum normalised Euclidean distance to other selected survivors) for
//!    improved diversity in many-objective (D ≥ 3) settings.
//! 6. Repeat until timeout or max generations.
//!
//! The algorithm returns the final approximation of the Pareto front as a
//! `BTreeSolutionSet` (for D=2) or collects non-dominated solutions into
//! a generic Pareto archive.

use std::time::Duration;

use pareto::{HasObjectives, MoSolution};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tracing::{info, info_span};

use crate::explored_solutions_data::ExploredSolutionsData;
use crate::problem::SetCoverProblem;
use crate::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;
use crate::timer::Timer;

use super::operators::{
    add_then_prune_mutation, binary_tournament, bitflip_mutation, coverage_biased_crossover,
    crowding_from_fronts, ensure_mutated, fast_non_dominated_sort, multi_swap_mutation,
    random_population, ranks_from_fronts, shift_mutation, swap_mutation, uniform_crossover,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for NSGA-II.
#[derive(Debug, Clone)]
pub struct Nsga2Config {
    /// Population size (number of individuals per generation).
    pub population_size: usize,
    /// Crossover probability.
    pub crossover_rate: f64,
    /// Per-individual mutation probability for swap mutation.
    pub swap_mutation_rate: f64,
    /// Per-individual mutation probability for add-then-prune mutation.
    pub add_prune_mutation_rate: f64,
    /// Per-bit mutation probability for bit-flip mutation.
    pub bitflip_mutation_rate: f64,
    /// Maximum number of images to remove in multi-swap mutation.
    pub multi_swap_max_removals: usize,
    /// Per-individual probability of applying multi-swap instead of single swap.
    pub multi_swap_rate: f64,
    /// Per-individual probability of applying shift (coverage-guided) mutation.
    pub shift_mutation_rate: f64,
    /// Fraction of crossovers that use coverage-biased crossover vs uniform.
    pub coverage_biased_crossover_fraction: f64,
    /// If true, guarantee at least one mutation operator fires on every offspring.
    /// This prevents wasted evaluations on unmodified crossover products.
    pub ensure_mutation: bool,
    /// Number of consecutive generations with no archive improvement before
    /// injecting random individuals to restore diversity.
    pub stagnation_limit: usize,
}

impl Default for Nsga2Config {
    fn default() -> Self {
        Self {
            population_size: 100,
            crossover_rate: 0.9,
            swap_mutation_rate: 0.4,
            add_prune_mutation_rate: 0.3,
            bitflip_mutation_rate: 0.0, // disabled by default; per-bit rate is very aggressive
            multi_swap_max_removals: 3,
            multi_swap_rate: 0.2,
            shift_mutation_rate: 0.25,
            coverage_biased_crossover_fraction: 0.5,
            ensure_mutation: true,
            stagnation_limit: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-generation statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct GenerationStats {
    generation: usize,
    num_fronts: usize,
    pareto_front_size: usize,
    best_objectives: Vec<u64>,
    elapsed_ms: u128,
    archive_size: usize,
    offspring_generated: usize,
    offspring_novel_genotype: usize,
    offspring_novel_objectives: usize,
    offspring_archive_inserted: usize,
}

// ---------------------------------------------------------------------------
// NSGA-II algorithm
// ---------------------------------------------------------------------------

/// NSGA-II solver for the SIMS multi-objective set-cover problem.
pub struct Nsga2<'a, P, const D: usize>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    /// Reference to the problem instance.
    problem: &'a P,
    /// Algorithm configuration.
    config: Nsga2Config,
    /// Current population.
    population: Vec<BitsetEncodedSolution<P, D>>,
    /// External archive of non-dominated solutions found across all generations.
    archive: Vec<BitsetEncodedSolution<P, D>>,
    /// Explored solutions tracker (compatible with PLS output).
    pub explored_solutions: ExploredSolutionsData<D>,
    /// RNG for reproducibility.
    rng: SmallRng,
    /// Counter for stagnation detection (generations without archive improvement).
    stagnation_counter: usize,
    /// Previous archive size (for stagnation detection).
    prev_archive_size: usize,
}

/// Per-generation diagnostics for evolutionary search dynamics.
#[derive(Debug, Clone, Default)]
struct EvolutionDiagnostics {
    offspring_generated: usize,
    offspring_novel_genotype: usize,
    offspring_novel_objectives: usize,
    offspring_archive_inserted: usize,
}

impl<'a, P, const D: usize> Nsga2<'a, P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    /// Create a new NSGA-II instance.
    ///
    /// If `initial_population` is `Some`, those solutions seed the population
    /// (padded with random solutions if smaller than `config.population_size`).
    /// Otherwise a fully random population is generated.
    pub fn new(
        problem: &'a P,
        config: Nsga2Config,
        initial_population: Option<Vec<BitsetEncodedSolution<P, D>>>,
        seed: u64,
    ) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let pop_size = config.population_size;

        let population = match initial_population {
            Some(init) => {
                let mut pop = init;
                // Pad with random solutions if needed
                while pop.len() < pop_size {
                    let s: u64 = rng.random();
                    pop.push(BitsetEncodedSolution::random_with_seed(problem, s));
                }
                // Truncate if too many
                pop.truncate(pop_size);
                pop
            }
            None => random_population(problem, pop_size, &mut rng),
        };

        let explored_solutions = ExploredSolutionsData::<D>::new(problem.max_objectives());

        let archive = Vec::new();

        Self {
            problem,
            config,
            population,
            archive,
            explored_solutions,
            rng,
            stagnation_counter: 0,
            prev_archive_size: 0,
        }
    }

    // -----------------------------------------------------------------
    // Main loop
    // -----------------------------------------------------------------

    /// Run NSGA-II until `max_generations` or `max_duration`, whichever comes first.
    ///
    /// Returns the final Pareto-optimal set as a vector of solutions.
    pub fn run(
        &mut self,
        max_generations: usize,
        max_duration: Duration,
    ) -> Vec<BitsetEncodedSolution<P, D>> {
        let timer = Timer::start(max_duration);

        info!(
            "NSGA-II starting: pop_size={}, crossover_rate={}, swap_mut={}, add_prune_mut={}, shift_mut={}, multi_swap_rate={}, ensure_mutation={}, timeout={:?}",
            self.config.population_size,
            self.config.crossover_rate,
            self.config.swap_mutation_rate,
            self.config.add_prune_mutation_rate,
            self.config.shift_mutation_rate,
            self.config.multi_swap_rate,
            self.config.ensure_mutation,
            max_duration,
        );

        // Register initial population
        for sol in &self.population {
            self.explored_solutions
                .register_without_selected_images(0, sol, Duration::ZERO);
        }

        // Seed archive with initial population (incremental)
        let init_pop: Vec<_> = self.population.clone();
        for sol in &init_pop {
            self.try_insert_into_archive(sol);
        }

        for generation in 1..=max_generations {
            let gen_span = info_span!("nsga2_generation", generation = generation);
            let _guard = gen_span.enter();

            if timer.is_expired() {
                info!(
                    "NSGA-II timeout after {} generations, elapsed {:?}",
                    generation - 1,
                    timer.elapsed()
                );
                break;
            }

            // Stagnation detection: inject random individuals if stuck
            if self.archive.len() == self.prev_archive_size {
                self.stagnation_counter += 1;
            } else {
                self.stagnation_counter = 0;
                self.prev_archive_size = self.archive.len();
            }
            if self.config.stagnation_limit > 0
                && self.stagnation_counter >= self.config.stagnation_limit
            {
                let inject_count = self.config.population_size / 4;
                info!(
                    "NSGA-II stagnation detected ({} gens), injecting {} random individuals",
                    self.stagnation_counter, inject_count
                );
                for _ in 0..inject_count {
                    let seed: u64 = self.rng.random();
                    let random_sol = BitsetEncodedSolution::random_with_seed(self.problem, seed);
                    // Replace worst-ranked individuals
                    let idx = self.rng.random_range(0..self.population.len());
                    self.population[idx] = random_sol;
                }
                self.stagnation_counter = 0;
            }

            // 1. Compute ranks and crowding for current population
            let fronts = fast_non_dominated_sort(&self.population);
            let ranks = ranks_from_fronts(&fronts, self.population.len());
            let crowding = crowding_from_fronts(&self.population, &fronts);

            // 2. Generate offspring
            let offspring = self.generate_offspring(&ranks, &crowding);

            // Register offspring and incrementally update archive
            let mut diagnostics = EvolutionDiagnostics {
                offspring_generated: offspring.len(),
                ..EvolutionDiagnostics::default()
            };
            let mut seen_objectives_this_generation = std::collections::HashSet::new();
            for sol in &offspring {
                if !self.explored_solutions.is_registered(sol) {
                    diagnostics.offspring_novel_genotype += 1;
                    self.explored_solutions.register_without_selected_images(
                        generation,
                        sol,
                        timer.elapsed(),
                    );
                }
                if seen_objectives_this_generation.insert(*sol.objectives()) {
                    diagnostics.offspring_novel_objectives += 1;
                }
                if self.try_insert_into_archive(sol) {
                    diagnostics.offspring_archive_inserted += 1;
                }
            }

            // 3. Merge parent + offspring
            let mut combined = std::mem::take(&mut self.population);
            combined.extend(offspring);

            // 4. Survivor selection (with deduplication): fill next generation
            self.population = self.survivor_selection(combined);

            // 5. Log stats
            if generation % 10 == 0 || generation <= 5 {
                let stats = self.generation_stats(generation, &timer, &diagnostics);
                self.log_stats(&stats);
            }

            if generation == max_generations {
                info!(
                    "NSGA-II reached max generations ({}), elapsed {:?}",
                    max_generations,
                    timer.elapsed()
                );
            }
        }

        self.explored_solutions.num_iterations = max_generations;

        info!(
            "NSGA-II completed: archive_size={}, total_explored={}",
            self.archive.len(),
            self.explored_solutions.solutions.len(),
        );

        self.archive.clone()
    }

    // -----------------------------------------------------------------
    // Offspring generation
    // -----------------------------------------------------------------

    fn generate_offspring(
        &mut self,
        ranks: &[usize],
        crowding: &[f64],
    ) -> Vec<BitsetEncodedSolution<P, D>> {
        let target_size = self.config.population_size;
        // Use actual population length for tournament selection (may be smaller
        // than config.population_size after deduplication in survivor_selection).
        let actual_pop_size = self.population.len();
        let mut offspring = Vec::with_capacity(target_size);

        while offspring.len() < target_size {
            // Select parents via binary tournament
            let p1_idx = binary_tournament(ranks, crowding, &mut self.rng, actual_pop_size);
            let p2_idx = binary_tournament(ranks, crowding, &mut self.rng, actual_pop_size);

            // Crossover
            let mut child = if self.rng.random_bool(self.config.crossover_rate) {
                if self
                    .rng
                    .random_bool(self.config.coverage_biased_crossover_fraction)
                {
                    coverage_biased_crossover(
                        &self.population[p1_idx],
                        &self.population[p2_idx],
                        self.problem,
                        &mut self.rng,
                    )
                } else {
                    uniform_crossover(
                        &self.population[p1_idx],
                        &self.population[p2_idx],
                        self.problem,
                        &mut self.rng,
                    )
                }
            } else {
                // No crossover: copy a parent
                if self.rng.random_bool(0.5) {
                    self.population[p1_idx].clone()
                } else {
                    self.population[p2_idx].clone()
                }
            };

            // Mutation: apply a composite of mutations
            child = self.mutate(child);

            offspring.push(child);
        }

        offspring
    }

    /// Apply composite mutation: tries multi-swap, shift, swap, add-then-prune in
    /// sequence. Each is applied independently with its own probability.
    /// When `ensure_mutation` is enabled, guarantees at least one operator fires
    /// so no offspring is an unmodified copy of a crossover product.
    fn mutate(&mut self, solution: BitsetEncodedSolution<P, D>) -> BitsetEncodedSolution<P, D> {
        let original = solution.clone();
        let mut result = solution;

        // Multi-swap mutation (most disruptive, lower rate)
        if self.rng.random_bool(self.config.multi_swap_rate) {
            result = multi_swap_mutation(
                &result,
                self.problem,
                &mut self.rng,
                1.0, // rate=1.0 because we already gated on multi_swap_rate
                self.config.multi_swap_max_removals,
            );
        } else if self.rng.random_bool(self.config.shift_mutation_rate) {
            // Shift mutation (coverage-guided replacement)
            result = shift_mutation(
                &result,
                self.problem,
                &mut self.rng,
                1.0, // rate=1.0 because we already gated above
            );
        } else {
            // Single swap mutation
            result = swap_mutation(
                &result,
                self.problem,
                &mut self.rng,
                self.config.swap_mutation_rate,
            );
        }

        // Add-then-prune mutation
        result = add_then_prune_mutation(
            &result,
            self.problem,
            &mut self.rng,
            self.config.add_prune_mutation_rate,
        );

        // Bit-flip mutation (if enabled)
        if self.config.bitflip_mutation_rate > 0.0 {
            let per_bit = self.config.bitflip_mutation_rate / self.problem.num_images() as f64;
            result = bitflip_mutation(&result, self.problem, &mut self.rng, per_bit);
        }

        // Ensure at least one mutation fired (avoid wasted evaluations)
        if self.config.ensure_mutation {
            result = ensure_mutated(&original, result, self.problem, &mut self.rng);
        }

        result
    }

    // -----------------------------------------------------------------
    // Survivor selection (NSGA-II elitist replacement)
    // -----------------------------------------------------------------

    /// From a combined population of size 2N, select N survivors using
    /// non-dominated sorting + crowding distance. Deduplicates solutions with
    /// identical objective vectors to preserve population diversity.
    ///
    /// For the critical "last front" (partially accepted), we use a composite
    /// diversity score that combines crowding distance with a contribution
    /// distance metric. This is particularly important for D ≥ 3 where
    /// crowding distance degrades: many solutions get `f64::INFINITY`
    /// (boundary solutions) or near-identical scores, leading to essentially
    /// random selection. The contribution distance greedily picks solutions
    /// that are farthest from already-selected survivors, ensuring good spread.
    fn survivor_selection(
        &self,
        mut combined: Vec<BitsetEncodedSolution<P, D>>,
    ) -> Vec<BitsetEncodedSolution<P, D>> {
        let target_size = self.config.population_size;

        // Deduplicate by objective vector — keep first occurrence only.
        // This prevents the population from being flooded by clones of a
        // single good solution, which kills diversity and stalls search.
        {
            let mut seen = std::collections::HashSet::new();
            combined.retain(|sol| seen.insert(*sol.objectives()));
        }

        if combined.len() <= target_size {
            return combined;
        }

        let fronts = fast_non_dominated_sort(&combined);
        let mut next_gen: Vec<BitsetEncodedSolution<P, D>> = Vec::with_capacity(target_size);

        for front in &fronts {
            if next_gen.len() + front.len() <= target_size {
                // Whole front fits
                for &idx in front {
                    next_gen.push(combined[idx].clone());
                }
            } else {
                // Partial front: select `remaining` solutions with best diversity.
                let remaining = target_size - next_gen.len();

                if D >= 3 && front.len() > 2 {
                    // Many-objective: use greedy contribution-distance selection.
                    // This picks solutions one at a time, always choosing the one
                    // farthest (in normalised objective space) from the already-
                    // selected set. This avoids the crowding-distance degeneracy
                    // where boundary solutions all get infinity and interior
                    // solutions get near-identical scores.
                    self.contribution_distance_selection(
                        &combined,
                        front,
                        remaining,
                        &mut next_gen,
                    );
                } else {
                    // Low-dimensional: standard crowding distance works well.
                    let cd = super::operators::crowding_distance(&combined, front);
                    let mut ranked: Vec<(usize, f64)> = front
                        .iter()
                        .enumerate()
                        .map(|(local, &global)| (global, cd[local]))
                        .collect();
                    ranked
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    for (global_idx, _cd) in ranked.into_iter().take(remaining) {
                        next_gen.push(combined[global_idx].clone());
                    }
                }
                break;
            }
        }

        next_gen
    }

    /// Greedy contribution-distance selection for the last (partial) front.
    ///
    /// Normalises objectives to [0,1] within the front, then iteratively picks
    /// the candidate with maximum minimum-distance to already-selected survivors.
    /// The first pick is the candidate with the highest crowding distance (to
    /// seed with a boundary solution). This produces much better spread than
    /// pure crowding distance in ≥3 objectives.
    fn contribution_distance_selection(
        &self,
        combined: &[BitsetEncodedSolution<P, D>],
        front: &[usize],
        remaining: usize,
        next_gen: &mut Vec<BitsetEncodedSolution<P, D>>,
    ) {
        let n = front.len();

        // 1. Compute per-objective min/max within this front for normalisation
        let mut obj_min = [f64::INFINITY; D];
        let mut obj_max = [f64::NEG_INFINITY; D];
        for &idx in front {
            for d in 0..D {
                let v = combined[idx].objectives()[d] as f64;
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

        // 2. Build normalised objective vectors for front members
        let norm_objs: Vec<[f64; D]> = front
            .iter()
            .map(|&idx| {
                let mut normed = [0.0f64; D];
                for d in 0..D {
                    normed[d] = (combined[idx].objectives()[d] as f64 - obj_min[d]) / obj_range[d];
                }
                normed
            })
            .collect();

        // Also normalise the already-selected survivors (from earlier fronts)
        // so the first pick avoids clustering near them.
        let survivor_norms: Vec<[f64; D]> = next_gen
            .iter()
            .map(|sol| {
                let mut normed = [0.0f64; D];
                for d in 0..D {
                    normed[d] = (sol.objectives()[d] as f64 - obj_min[d]) / obj_range[d];
                }
                normed
            })
            .collect();

        // 3. Greedy farthest-point selection
        let mut selected_local: Vec<usize> = Vec::with_capacity(remaining);
        let mut available: Vec<bool> = vec![true; n];

        // min_dist_to_selected[i] = minimum distance from front[i] to any
        // already-picked solution (initialised from existing survivors).
        let mut min_dist: Vec<f64> = vec![f64::INFINITY; n];

        // Initialise min_dist from already-selected survivors
        if !survivor_norms.is_empty() {
            for i in 0..n {
                for sv in &survivor_norms {
                    let d = Self::normalised_distance(&norm_objs[i], sv);
                    if d < min_dist[i] {
                        min_dist[i] = d;
                    }
                }
            }
        }

        for _ in 0..remaining {
            // Pick the available candidate with largest min_dist
            let mut best_local = usize::MAX;
            let mut best_dist = f64::NEG_INFINITY;
            for i in 0..n {
                if available[i] && min_dist[i] > best_dist {
                    best_dist = min_dist[i];
                    best_local = i;
                }
            }
            if best_local == usize::MAX {
                break; // no more candidates
            }

            selected_local.push(best_local);
            available[best_local] = false;

            // Update min_dist for remaining candidates
            let picked = &norm_objs[best_local];
            for i in 0..n {
                if available[i] {
                    let d = Self::normalised_distance(&norm_objs[i], picked);
                    if d < min_dist[i] {
                        min_dist[i] = d;
                    }
                }
            }
        }

        for local_idx in selected_local {
            next_gen.push(combined[front[local_idx]].clone());
        }
    }

    /// Euclidean distance between two normalised objective vectors.
    #[inline]
    fn normalised_distance(a: &[f64; D], b: &[f64; D]) -> f64 {
        let mut sum = 0.0f64;
        for d in 0..D {
            let diff = a[d] - b[d];
            sum += diff * diff;
        }
        sum.sqrt()
    }

    // -----------------------------------------------------------------
    // External archive maintenance
    // -----------------------------------------------------------------

    /// Try to insert a single solution into the external archive.
    /// Only inserts if the solution is non-dominated by existing members.
    /// Removes any existing members that the new solution dominates.
    /// Returns `true` if the solution was inserted.
    fn try_insert_into_archive(&mut self, solution: &BitsetEncodedSolution<P, D>) -> bool {
        // Check if dominated by or duplicate of any existing archive member
        for existing in &self.archive {
            if existing.dominates(solution.objectives())
                || existing.objectives() == solution.objectives()
            {
                return false;
            }
        }
        // Remove any archive members dominated by the new solution
        self.archive
            .retain(|existing| !solution.dominates(existing.objectives()));
        self.archive.push(solution.clone());
        true
    }

    // -----------------------------------------------------------------
    // Statistics and logging
    // -----------------------------------------------------------------

    fn generation_stats(
        &self,
        generation: usize,
        timer: &Timer,
        diagnostics: &EvolutionDiagnostics,
    ) -> GenerationStats {
        let fronts = fast_non_dominated_sort(&self.population);
        let pareto_front_size = fronts.first().map_or(0, Vec::len);

        // Best objective values from the archive
        let mut best_objectives = vec![u64::MAX; D];
        for sol in &self.archive {
            for i in 0..D {
                if sol.objectives()[i] < best_objectives[i] {
                    best_objectives[i] = sol.objectives()[i];
                }
            }
        }

        GenerationStats {
            generation,
            num_fronts: fronts.len(),
            pareto_front_size,
            best_objectives,
            elapsed_ms: timer.elapsed().as_millis(),
            archive_size: self.archive.len(),
            offspring_generated: diagnostics.offspring_generated,
            offspring_novel_genotype: diagnostics.offspring_novel_genotype,
            offspring_novel_objectives: diagnostics.offspring_novel_objectives,
            offspring_archive_inserted: diagnostics.offspring_archive_inserted,
        }
    }

    fn log_stats(&self, stats: &GenerationStats) {
        let novelty_pct = if stats.offspring_generated == 0 {
            0.0
        } else {
            100.0 * stats.offspring_novel_genotype as f64 / stats.offspring_generated as f64
        };
        let objective_novelty_pct = if stats.offspring_generated == 0 {
            0.0
        } else {
            100.0 * stats.offspring_novel_objectives as f64 / stats.offspring_generated as f64
        };
        let archive_insert_pct = if stats.offspring_generated == 0 {
            0.0
        } else {
            100.0 * stats.offspring_archive_inserted as f64 / stats.offspring_generated as f64
        };

        info!(
            "NSGA-II gen={}: fronts={}, pareto_front={}, archive={}, best={:?}, offspring={}/{}, novel_genotype={:.1}%, novel_objectives={:.1}%, archive_inserted={:.1}%, stagnation={}, elapsed={}ms",
            stats.generation,
            stats.num_fronts,
            stats.pareto_front_size,
            stats.archive_size,
            stats.best_objectives,
            stats.offspring_generated,
            self.config.population_size,
            novelty_pct,
            objective_novelty_pct,
            archive_insert_pct,
            self.stagnation_counter,
            stats.elapsed_ms,
        );
    }

    // -----------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------

    /// Get a reference to the current external archive.
    pub fn archive(&self) -> &[BitsetEncodedSolution<P, D>] {
        &self.archive
    }

    /// Get a reference to the current population.
    pub fn population(&self) -> &[BitsetEncodedSolution<P, D>] {
        &self.population
    }

    /// Get explored solutions data (compatible with PLS output format).
    pub fn explored_solutions_data(&self) -> &ExploredSolutionsData<D> {
        &self.explored_solutions
    }
}

// ---------------------------------------------------------------------------
// Convenience: run NSGA-II with a BTreeSolutionSet output (for D=2)
// ---------------------------------------------------------------------------

/// Run NSGA-II and return results in a `BTreeSolutionSet` (for 2-objective problems).
///
/// This is a convenience wrapper that mirrors the PLS interface.
pub fn run_nsga2<P>(
    problem: &P,
    config: Nsga2Config,
    initial_population: Option<Vec<BitsetEncodedSolution<P, 2>>>,
    max_generations: usize,
    max_duration: Duration,
    seed: u64,
) -> (Vec<BitsetEncodedSolution<P, 2>>, ExploredSolutionsData<2>)
where
    P: SetCoverProblem<2> + Clone + Send + Sync,
{
    let mut nsga2 = Nsga2::new(problem, config, initial_population, seed);
    let archive = nsga2.run(max_generations, max_duration);
    let explored = std::mem::replace(
        &mut nsga2.explored_solutions,
        ExploredSolutionsData::new(problem.max_objectives()),
    );
    (archive, explored)
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

    /// Build a tiny test problem: 5 elements, 4 images.
    fn make_test_problem() -> ProblemBitset<2> {
        let raw = SIMSProblemInstanceRaw {
            name: "nsga2_test".to_string(),
            num_images: 4,
            universe_size: 5,
            images: vec![
                vec![0, 1, 2],       // image 0
                vec![2, 3, 4],       // image 1
                vec![0, 1, 3, 4],    // image 2
                vec![0, 1, 2, 3, 4], // image 3
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
    fn test_nsga2_runs_and_produces_feasible_solutions() {
        let problem = make_test_problem();
        let config = Nsga2Config {
            population_size: 20,
            ..Default::default()
        };

        let mut nsga2 = Nsga2::new(&problem, config, None, 42);
        let archive = nsga2.run(50, Duration::from_secs(5));

        assert!(!archive.is_empty(), "Archive should not be empty");

        // All archive solutions must be feasible
        for sol in &archive {
            assert!(
                problem.is_set_cover(sol),
                "Archive solution must be a valid set cover"
            );
        }

        // No solution in archive should dominate another
        for (i, a) in archive.iter().enumerate() {
            for (j, b) in archive.iter().enumerate() {
                if i != j {
                    assert!(
                        !a.dominates(b.objectives()),
                        "Archive solution {i} dominates solution {j}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_nsga2_with_initial_population() {
        let problem = make_test_problem();

        // Create a manual initial population
        let init = vec![
            BitsetEncodedSolution::from_selected_images(&[0, 1], &problem),
            BitsetEncodedSolution::from_selected_images(&[3], &problem),
            BitsetEncodedSolution::from_selected_images(&[0, 1, 2], &problem),
        ];

        let config = Nsga2Config {
            population_size: 10,
            ..Default::default()
        };

        let mut nsga2 = Nsga2::new(&problem, config, Some(init), 123);
        let archive = nsga2.run(30, Duration::from_secs(3));

        assert!(!archive.is_empty());
        for sol in &archive {
            assert!(problem.is_set_cover(sol));
        }
    }

    #[test]
    fn test_nsga2_respects_timeout() {
        let problem = make_test_problem();
        let config = Nsga2Config {
            population_size: 20,
            ..Default::default()
        };

        let start = std::time::Instant::now();
        let mut nsga2 = Nsga2::new(&problem, config, None, 99);
        let _archive = nsga2.run(1_000_000, Duration::from_millis(200));
        let elapsed = start.elapsed();

        // Should finish within roughly the timeout (allow some overhead)
        assert!(
            elapsed < Duration::from_secs(2),
            "Should respect timeout, but took {:?}",
            elapsed
        );
    }

    #[test]
    fn test_run_nsga2_convenience() {
        let problem = make_test_problem();
        let config = Nsga2Config {
            population_size: 15,
            ..Default::default()
        };

        let (archive, explored) = run_nsga2(&problem, config, None, 20, Duration::from_secs(3), 7);

        assert!(!archive.is_empty());
        assert!(!explored.solutions.is_empty());
    }
}
