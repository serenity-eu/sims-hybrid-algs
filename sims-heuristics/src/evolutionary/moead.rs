//! MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) for SIMS
//!
//! Implements the MOEA/D algorithm (Zhang & Li, 2007) adapted for the Satellite
//! Image Mosaic Selection problem. Key adaptations for the set-cover constraint:
//!
//! - **Coverage-aware crossover**: Uniform and coverage-biased crossover on
//!   image-selection bitsets followed by greedy repair to ensure feasibility.
//! - **Feasibility-preserving mutation**: Swap, add-then-prune, shift, and multi-swap
//!   mutations that maintain or restore feasibility via repair operators.
//! - **Redundancy removal**: After every genetic operation, redundant images
//!   are pruned to keep solutions lean.
//!
//! ## Algorithm Outline
//!
//! 1. Generate `N` uniformly distributed weight vectors on the (D-1)-simplex.
//! 2. Compute weight-vector neighbourhoods (closest `T` weight vectors).
//! 3. Initialize population: one random feasible solution per weight vector.
//! 4. Compute the ideal point `z*` (component-wise minimum across population).
//! 5. For each subproblem `i`:
//!    a. Select two parents from the neighbourhood of `i`.
//!    b. Apply crossover + mutation to produce an offspring `y`.
//!    c. Update the ideal point with `y`'s objectives.
//!    d. For each neighbour `j` of `i`: if `y` improves the Tchebycheff value
//!       of subproblem `j`, replace the solution of `j` with `y`.
//! 6. Maintain an external Pareto archive of all non-dominated solutions found.
//! 7. Repeat until timeout or max generations.
//!
//! ## SIMS-specific enhancements
//!
//! - **Automatic population sizing**: For D>2, `num_divisions` is automatically
//!   computed from `target_pop_size` to avoid the combinatorial explosion of
//!   C(n+D-1, D-1) weight vectors. For D=4 and target 100, this uses ~7
//!   divisions (120 weight vectors) instead of the 455 from 12 divisions.
//! - **Adaptive neighbourhood probability** (`delta`): with probability `delta`,
//!   mating is restricted to the neighbourhood; otherwise the entire population
//!   is used. This balances exploitation and exploration.
//! - **Maximum replacement limit** (`nr`): each offspring can replace at most
//!   `nr` neighbours, preventing a single good solution from flooding the
//!   population and reducing diversity.
//! - **Composite mutation** tailored for set-cover: multi-swap for disruption,
//!   shift mutation for coverage-guided replacement, add-then-prune for fine-tuning.
//! - **Guaranteed mutation**: optionally ensures every offspring is modified.
//! - **Diversity injection**: injects random individuals when the population stagnates.

use std::time::Duration;

use pareto::{HasObjectives, MoSolution};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use tracing::{info, info_span};

use crate::explored_solutions_data::ExploredSolutionsData;
use crate::problem::SetCoverProblem;

use crate::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;
use crate::timer::Timer;

use super::operators::{
    add_then_prune_mutation, compute_ideal_point, compute_neighbourhoods,
    coverage_biased_crossover, ensure_mutated, generate_weight_vectors, multi_swap_mutation,
    random_population, shift_mutation, swap_mutation, tchebycheff_value, uniform_crossover,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for MOEA/D.
#[derive(Debug, Clone)]
pub struct MoeadConfig {
    /// Number of subproblems (weight vectors / population size).
    /// For 2 objectives this equals `num_divisions + 1`.
    pub population_size: usize,

    /// Number of weight-vector divisions for simplex-lattice design.
    /// The actual population size will be C(num_divisions + D - 1, D - 1).
    /// For 2-objective problems: population_size = num_divisions + 1.
    ///
    /// **For D >= 3**: If `auto_divisions` is true (default), this field is
    /// ignored and `num_divisions` is computed automatically from
    /// `target_pop_size` to produce approximately that many weight vectors.
    pub num_divisions: usize,

    /// Target population size for automatic `num_divisions` computation.
    /// Only used when `auto_divisions` is true and D >= 3.
    /// The actual population size will be the nearest C(n+D-1, D-1) value.
    pub target_pop_size: usize,

    /// Whether to automatically compute `num_divisions` from `target_pop_size`.
    /// Defaults to true. For D=2, `num_divisions` is always used directly.
    pub auto_divisions: bool,

    /// Neighbourhood size `T`: number of closest weight vectors considered
    /// neighbours for each subproblem.
    pub neighbourhood_size: usize,

    /// Probability of restricting mating to the neighbourhood.
    /// With probability `1 - delta`, mating uses the whole population.
    pub delta: f64,

    /// Maximum number of neighbours that one offspring can replace (`nr`).
    pub max_replacements: usize,

    /// Crossover probability.
    pub crossover_rate: f64,

    /// Per-individual mutation probability for swap mutation.
    pub swap_mutation_rate: f64,

    /// Per-individual mutation probability for add-then-prune mutation.
    pub add_prune_mutation_rate: f64,

    /// Maximum number of images to remove in multi-swap mutation.
    pub multi_swap_max_removals: usize,

    /// Per-individual probability of applying multi-swap instead of single swap.
    pub multi_swap_rate: f64,

    /// Per-individual probability of applying shift (coverage-guided) mutation.
    pub shift_mutation_rate: f64,

    /// Fraction of crossovers that use coverage-biased crossover vs uniform.
    pub coverage_biased_crossover_fraction: f64,

    /// If true, guarantee at least one mutation operator fires on every offspring.
    pub ensure_mutation: bool,

    /// Whether to use the penalty-based boundary intersection (PBI) approach
    /// instead of Tchebycheff. Defaults to false (Tchebycheff).
    pub use_pbi: bool,

    /// PBI penalty parameter theta (only used when `use_pbi` is true).
    pub pbi_theta: f64,

    /// Number of consecutive generations with no archive improvement before
    /// injecting random individuals to restore diversity.
    pub stagnation_limit: usize,
}

impl Default for MoeadConfig {
    fn default() -> Self {
        Self {
            population_size: 100, // will be overridden by num_divisions
            num_divisions: 99,    // => 100 weight vectors for 2 objectives
            target_pop_size: 200, // for D=4: ~165 weight vectors (num_divisions=7)
            auto_divisions: true,
            neighbourhood_size: 20,
            delta: 0.9,
            max_replacements: 3,
            crossover_rate: 1.0,
            swap_mutation_rate: 0.3,
            add_prune_mutation_rate: 0.2,
            multi_swap_max_removals: 3,
            multi_swap_rate: 0.15,
            shift_mutation_rate: 0.15,
            coverage_biased_crossover_fraction: 0.5,
            // Decomposition-based algorithms rely on scalarization to guide search;
            // forcing mutation on every offspring adds too much noise and prevents
            // offspring from improving their target subproblem.
            ensure_mutation: false,
            use_pbi: false,
            pbi_theta: 5.0,
            stagnation_limit: 80,
        }
    }
}

/// Compute the number of simplex-lattice divisions `n` such that
/// C(n + D - 1, D - 1) is as close to `target` as possible (from below).
fn compute_divisions_for_target<const D: usize>(target: usize) -> usize {
    if D <= 1 {
        return target;
    }
    // For D=2: C(n+1,1) = n+1, so n = target - 1
    if D == 2 {
        return target.saturating_sub(1).max(1);
    }
    // For D >= 3: binary search for n such that C(n+D-1, D-1) <= target
    let mut best_n = 1usize;
    for n in 1..200 {
        let count = simplex_lattice_count(n, D);
        if count <= target {
            best_n = n;
        } else {
            break;
        }
    }
    best_n.max(1)
}

/// Compute C(n + D - 1, D - 1) — the number of weight vectors for `n` divisions
/// and `D` objectives using the simplex-lattice design.
fn simplex_lattice_count(n: usize, d: usize) -> usize {
    // C(n + d - 1, d - 1)
    let k = d - 1;
    let mut result = 1u128;
    for i in 0..k {
        result = result * (n + d - 1 - i) as u128 / (i + 1) as u128;
    }
    result.min(usize::MAX as u128) as usize
}

// ---------------------------------------------------------------------------
// Per-generation statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct GenerationStats {
    generation: usize,
    archive_size: usize,
    ideal_point: Vec<f64>,
    replacements_this_gen: usize,
    elapsed_ms: u128,
    offspring_generated: usize,
    offspring_novel_genotype: usize,
    offspring_novel_objectives: usize,
    offspring_archive_inserted: usize,
}

// ---------------------------------------------------------------------------
// MOEA/D algorithm
// ---------------------------------------------------------------------------

/// MOEA/D solver for the SIMS multi-objective set-cover problem.
pub struct Moead<'a, P, const D: usize>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    /// Reference to the problem instance.
    problem: &'a P,

    /// Algorithm configuration.
    config: MoeadConfig,

    /// Weight vectors (one per subproblem).
    weights: Vec<[f64; D]>,

    /// Neighbourhood indices for each weight vector.
    neighbourhoods: Vec<Vec<usize>>,

    /// Current population: `population[i]` is the solution assigned to subproblem `i`.
    population: Vec<BitsetEncodedSolution<P, D>>,

    /// Ideal point `z*`: component-wise minimum of all evaluated solutions.
    ideal_point: [f64; D],

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
    replacements_this_gen: usize,
    offspring_generated: usize,
    offspring_novel_genotype: usize,
    offspring_novel_objectives: usize,
    offspring_archive_inserted: usize,
}

impl<'a, P, const D: usize> Moead<'a, P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    /// Create a new MOEA/D instance.
    ///
    /// Weight vectors are generated via the simplex-lattice design using
    /// `config.num_divisions`. The actual population size is determined by
    /// the number of generated weight vectors.
    ///
    /// If `initial_population` is `Some`, those solutions seed the population
    /// (cycled or truncated to match the number of weight vectors).
    pub fn new(
        problem: &'a P,
        config: MoeadConfig,
        initial_population: Option<Vec<BitsetEncodedSolution<P, D>>>,
        seed: u64,
    ) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);

        // 1. Determine number of divisions (auto-compute for D >= 3 if enabled)
        let effective_divisions = if config.auto_divisions && D >= 3 {
            let computed = compute_divisions_for_target::<D>(config.target_pop_size);
            let count = simplex_lattice_count(computed, D);
            info!(
                "MOEA/D: auto-computed num_divisions={} for target_pop_size={} (D={}), actual pop_size={}",
                computed, config.target_pop_size, D, count
            );
            computed
        } else {
            config.num_divisions
        };

        // Generate weight vectors
        let weights = generate_weight_vectors::<D>(effective_divisions);
        let actual_pop_size = weights.len();

        info!(
            "MOEA/D: generated {} weight vectors from {} divisions for {} objectives",
            actual_pop_size, effective_divisions, D
        );

        // 2. Compute neighbourhoods
        let neighbourhoods = compute_neighbourhoods(&weights, config.neighbourhood_size);

        // 3. Initialize population
        let population = match initial_population {
            Some(init) if !init.is_empty() => {
                let mut pop: Vec<BitsetEncodedSolution<P, D>> = Vec::with_capacity(actual_pop_size);
                // Cycle through provided solutions to fill population
                for i in 0..actual_pop_size {
                    if i < init.len() {
                        pop.push(init[i].clone());
                    } else {
                        let s: u64 = rng.random();
                        pop.push(BitsetEncodedSolution::random_with_seed(problem, s));
                    }
                }
                pop
            }
            _ => random_population(problem, actual_pop_size, &mut rng),
        };

        // 4. Compute ideal point
        let ideal_point = compute_ideal_point(&population);

        let explored_solutions = ExploredSolutionsData::<D>::new(problem.max_objectives());
        let archive = Vec::new();

        let mut moead = Self {
            problem,
            config,
            weights,
            neighbourhoods,
            population,
            ideal_point,
            archive,
            explored_solutions,
            rng,
            stagnation_counter: 0,
            prev_archive_size: 0,
        };

        // Update config's population_size to actual
        moead.config.population_size = actual_pop_size;
        moead.config.num_divisions = effective_divisions;

        // Scale neighbourhood size to actual population size (at least 5, at most pop/4)
        let scaled_neighbourhood = (actual_pop_size / 5)
            .max(5)
            .min(actual_pop_size / 4)
            .min(moead.config.neighbourhood_size);
        if scaled_neighbourhood != moead.config.neighbourhood_size {
            info!(
                "MOEA/D: scaled neighbourhood_size from {} to {} for pop_size={}",
                moead.config.neighbourhood_size, scaled_neighbourhood, actual_pop_size
            );
            moead.config.neighbourhood_size = scaled_neighbourhood;
        }

        moead
    }

    // -----------------------------------------------------------------
    // Main loop
    // -----------------------------------------------------------------

    /// Run MOEA/D until `max_generations` or `max_duration`, whichever comes first.
    ///
    /// Returns the final Pareto-optimal set as a vector of solutions.
    pub fn run(
        &mut self,
        max_generations: usize,
        max_duration: Duration,
    ) -> Vec<BitsetEncodedSolution<P, D>> {
        let timer = Timer::start(max_duration);

        info!(
            "MOEA/D starting: pop_size={}, divisions={}, neighbourhood={}, delta={}, nr={}, crossover={}, swap_mut={}, shift_mut={}, ensure_mutation={}, timeout={:?}",
            self.config.population_size,
            self.config.num_divisions,
            self.config.neighbourhood_size,
            self.config.delta,
            self.config.max_replacements,
            self.config.crossover_rate,
            self.config.swap_mutation_rate,
            self.config.shift_mutation_rate,
            self.config.ensure_mutation,
            max_duration,
        );

        // Register initial population
        for sol in &self.population {
            if !self.explored_solutions.is_registered(sol) {
                self.explored_solutions
                    .register_without_selected_images(0, sol, Duration::ZERO);
            }
        }

        // Update archive with initial population
        self.update_archive();

        for generation in 1..=max_generations {
            let gen_span = info_span!("moead_generation", generation = generation);
            let _guard = gen_span.enter();

            if timer.is_expired() {
                info!(
                    "MOEA/D timeout after {} generations, elapsed {:?}",
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
                let inject_count = (self.config.population_size / 5).max(1);
                info!(
                    "MOEA/D stagnation detected ({} gens), injecting {} random individuals",
                    self.stagnation_counter, inject_count
                );
                let pop_size = self.config.population_size;
                for _ in 0..inject_count {
                    let idx = self.rng.random_range(0..pop_size);
                    let seed_val: u64 = self.rng.random();
                    let random_sol =
                        BitsetEncodedSolution::random_with_seed(self.problem, seed_val);
                    self.update_ideal_point(&random_sol);
                    self.population[idx] = random_sol;
                }
                self.stagnation_counter = 0;
            }

            let diagnostics = self.generation_step(generation, &timer);

            // Log stats periodically
            if generation % 10 == 0 || generation <= 5 {
                let stats = GenerationStats {
                    generation,
                    archive_size: self.archive.len(),
                    ideal_point: self.ideal_point.to_vec(),
                    replacements_this_gen: diagnostics.replacements_this_gen,
                    elapsed_ms: timer.elapsed().as_millis(),
                    offspring_generated: diagnostics.offspring_generated,
                    offspring_novel_genotype: diagnostics.offspring_novel_genotype,
                    offspring_novel_objectives: diagnostics.offspring_novel_objectives,
                    offspring_archive_inserted: diagnostics.offspring_archive_inserted,
                };
                self.log_stats(&stats);
            }

            if generation == max_generations {
                info!(
                    "MOEA/D reached max generations ({}), elapsed {:?}",
                    max_generations,
                    timer.elapsed()
                );
            }
        }

        self.explored_solutions.num_iterations = max_generations;

        info!(
            "MOEA/D completed: archive_size={}, total_explored={}",
            self.archive.len(),
            self.explored_solutions.solutions.len(),
        );

        self.archive.clone()
    }

    // -----------------------------------------------------------------
    // Single generation step
    // -----------------------------------------------------------------

    /// Execute one generation: iterate over all subproblems, create offspring,
    /// and update neighbours. Returns per-generation diagnostics.
    fn generation_step(&mut self, generation: usize, timer: &Timer) -> EvolutionDiagnostics {
        let pop_size = self.config.population_size;
        let mut diagnostics = EvolutionDiagnostics::default();
        let mut seen_objectives_this_generation = std::collections::HashSet::new();

        // Process subproblems in random order for fairness
        let mut order: Vec<usize> = (0..pop_size).collect();
        order.shuffle(&mut self.rng);

        for &i in &order {
            if timer.is_expired() {
                break;
            }

            // Step 1: Determine mating pool
            let use_neighbourhood = self.rng.random_bool(self.config.delta);

            // Step 2: Select two parents
            // Clone the neighbourhood slice to avoid borrow conflict with &mut self
            let (p1_idx, p2_idx) = if use_neighbourhood {
                let pool: Vec<usize> = self.neighbourhoods[i].clone();
                self.select_parents_from_pool(&pool)
            } else {
                self.select_parents_from_population(pop_size)
            };

            // Step 3: Crossover
            let child = if self.rng.random_bool(self.config.crossover_rate) {
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
                self.population[i].clone()
            };

            // Step 4: Mutation
            let child = self.mutate(child);

            // Step 5: Update ideal point
            self.update_ideal_point(&child);

            diagnostics.offspring_generated += 1;

            // Step 6: Register explored solution
            if !self.explored_solutions.is_registered(&child) {
                diagnostics.offspring_novel_genotype += 1;
                self.explored_solutions.register_without_selected_images(
                    generation,
                    &child,
                    timer.elapsed(),
                );
            }

            if seen_objectives_this_generation.insert(*child.objectives()) {
                diagnostics.offspring_novel_objectives += 1;
            }

            // Step 7: Update neighbours
            let update_pool: Vec<usize> = if use_neighbourhood {
                self.neighbourhoods[i].clone()
            } else {
                (0..pop_size).collect()
            };

            let replacements = self.update_neighbours(&child, &update_pool);
            diagnostics.replacements_this_gen += replacements;

            // Step 8: Update archive
            if self.try_insert_into_archive(&child) {
                diagnostics.offspring_archive_inserted += 1;
            }
        }

        diagnostics
    }

    // -----------------------------------------------------------------
    // Parent selection
    // -----------------------------------------------------------------

    fn select_parents_from_pool(&mut self, pool: &[usize]) -> (usize, usize) {
        if pool.len() < 2 {
            let idx = pool.first().copied().unwrap_or(0);
            return (idx, idx);
        }
        let a = pool[self.rng.random_range(0..pool.len())];
        let mut b = pool[self.rng.random_range(0..pool.len())];
        // Ensure different parents (try a few times)
        for _ in 0..5 {
            if b != a {
                break;
            }
            b = pool[self.rng.random_range(0..pool.len())];
        }
        (a, b)
    }

    fn select_parents_from_population(&mut self, pop_size: usize) -> (usize, usize) {
        let a = self.rng.random_range(0..pop_size);
        let mut b = self.rng.random_range(0..pop_size);
        for _ in 0..5 {
            if b != a {
                break;
            }
            b = self.rng.random_range(0..pop_size);
        }
        (a, b)
    }

    // -----------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------

    /// Apply composite mutation tailored for set-cover problems.
    /// When `ensure_mutation` is enabled, guarantees at least one operator fires.
    fn mutate(&mut self, solution: BitsetEncodedSolution<P, D>) -> BitsetEncodedSolution<P, D> {
        let original = solution.clone();
        let mut result = solution;

        // Multi-swap mutation (most disruptive, lower rate)
        if self.rng.random_bool(self.config.multi_swap_rate) {
            result = multi_swap_mutation(
                &result,
                self.problem,
                &mut self.rng,
                1.0, // rate=1.0 because gated on multi_swap_rate
                self.config.multi_swap_max_removals,
            );
        } else if self.rng.random_bool(self.config.shift_mutation_rate) {
            // Shift mutation (coverage-guided replacement)
            result = shift_mutation(
                &result,
                self.problem,
                &mut self.rng,
                1.0, // rate=1.0 because gated above
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

        // Ensure at least one mutation fired (avoid wasted evaluations)
        if self.config.ensure_mutation {
            result = ensure_mutated(&original, result, self.problem, &mut self.rng);
        }

        result
    }

    // -----------------------------------------------------------------
    // Ideal point update
    // -----------------------------------------------------------------

    fn update_ideal_point(&mut self, solution: &BitsetEncodedSolution<P, D>) {
        for i in 0..D {
            let val = solution.objectives()[i] as f64;
            if val < self.ideal_point[i] {
                self.ideal_point[i] = val;
            }
        }
    }

    // -----------------------------------------------------------------
    // Neighbour replacement
    // -----------------------------------------------------------------

    /// Try to replace neighbours with `child` using Tchebycheff (or PBI) scalarization.
    /// Returns the number of replacements made (capped at `max_replacements`).
    fn update_neighbours(&mut self, child: &BitsetEncodedSolution<P, D>, pool: &[usize]) -> usize {
        let nr = self.config.max_replacements;
        let mut replacements = 0usize;

        // Shuffle pool to give each neighbour fair chance
        let mut shuffled_pool: Vec<usize> = pool.to_vec();
        shuffled_pool.shuffle(&mut self.rng);

        for &j in &shuffled_pool {
            if replacements >= nr {
                break;
            }

            let child_value = self.scalarized_value(child.objectives(), &self.weights[j]);
            let current_value =
                self.scalarized_value(self.population[j].objectives(), &self.weights[j]);

            if child_value < current_value {
                self.population[j] = child.clone();
                replacements += 1;
            }
        }

        replacements
    }

    /// Compute the scalarized value for a solution under a given weight vector.
    fn scalarized_value(&self, objectives: &[u64; D], weight: &[f64; D]) -> f64 {
        if self.config.use_pbi {
            self.pbi_value(objectives, weight)
        } else {
            tchebycheff_value(objectives, weight, &self.ideal_point)
        }
    }

    /// Penalty-Based Boundary Intersection (PBI) scalarization.
    ///
    /// `PBI(x, w, z*) = d1 + theta * d2`
    ///
    /// where `d1` is the distance from `z*` along the weight direction, and `d2` is the
    /// perpendicular distance from the objective vector to the weight line through `z*`.
    fn pbi_value(&self, objectives: &[u64; D], weight: &[f64; D]) -> f64 {
        // Normalized weight direction
        let norm: f64 = weight.iter().map(|&w| w * w).sum::<f64>().sqrt();
        if norm < 1e-12 {
            return f64::INFINITY;
        }

        // f - z* vector
        let mut diff = [0.0f64; D];
        for i in 0..D {
            diff[i] = (objectives[i] as f64) - self.ideal_point[i];
        }

        // d1 = (f - z*) . w_hat  (projection onto weight direction)
        let mut d1 = 0.0f64;
        for i in 0..D {
            d1 += diff[i] * (weight[i] / norm);
        }

        // d2 = || (f - z*) - d1 * w_hat ||  (perpendicular distance)
        let mut d2_sq = 0.0f64;
        for i in 0..D {
            let component = diff[i] - d1 * (weight[i] / norm);
            d2_sq += component * component;
        }
        let d2 = d2_sq.sqrt();

        d1 + self.config.pbi_theta * d2
    }

    // -----------------------------------------------------------------
    // External archive maintenance
    // -----------------------------------------------------------------

    /// Try to insert a solution into the external archive. Removes dominated members.
    /// Returns `true` if the solution was inserted.
    fn try_insert_into_archive(&mut self, solution: &BitsetEncodedSolution<P, D>) -> bool {
        // Check if dominated by any archive member
        for existing in &self.archive {
            if existing.dominates(solution.objectives())
                || existing.objectives() == solution.objectives()
            {
                return false; // dominated or duplicate
            }
        }

        // Remove any archive members dominated by the new solution
        self.archive
            .retain(|existing| !solution.dominates(existing.objectives()));

        self.archive.push(solution.clone());
        true
    }

    /// Full archive rebuild from the current population (used during initialization).
    fn update_archive(&mut self) {
        for sol in self.population.clone() {
            let _ = self.try_insert_into_archive(&sol);
        }
    }

    // -----------------------------------------------------------------
    // Statistics and logging
    // -----------------------------------------------------------------

    fn log_stats(&self, stats: &GenerationStats) {
        let ideal_str: Vec<String> = stats
            .ideal_point
            .iter()
            .map(|v| format!("{v:.0}"))
            .collect();
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
            "MOEA/D gen={}: archive={}, ideal=[{}], replacements={}, offspring={}, novel_genotype={:.1}%, novel_objectives={:.1}%, archive_inserted={:.1}%, stagnation={}, elapsed={}ms",
            stats.generation,
            stats.archive_size,
            ideal_str.join(", "),
            stats.replacements_this_gen,
            stats.offspring_generated,
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

    /// Get a reference to the current external archive (non-dominated solutions).
    pub fn archive(&self) -> &[BitsetEncodedSolution<P, D>] {
        &self.archive
    }

    /// Get a reference to the current population.
    pub fn population(&self) -> &[BitsetEncodedSolution<P, D>] {
        &self.population
    }

    /// Get a reference to the weight vectors.
    pub fn weights(&self) -> &[[f64; D]] {
        &self.weights
    }

    /// Get the current ideal point.
    pub fn ideal_point(&self) -> &[f64; D] {
        &self.ideal_point
    }

    /// Get explored solutions data (compatible with PLS output format).
    pub fn explored_solutions_data(&self) -> &ExploredSolutionsData<D> {
        &self.explored_solutions
    }
}

// ---------------------------------------------------------------------------
// Convenience: run MOEA/D and return results (for 2-objective problems)
// ---------------------------------------------------------------------------

/// Run MOEA/D and return results as a vector of non-dominated solutions.
///
/// This is a convenience wrapper that mirrors the PLS / NSGA-II interface.
pub fn run_moead<P>(
    problem: &P,
    config: MoeadConfig,
    initial_population: Option<Vec<BitsetEncodedSolution<P, 2>>>,
    max_generations: usize,
    max_duration: Duration,
    seed: u64,
) -> (Vec<BitsetEncodedSolution<P, 2>>, ExploredSolutionsData<2>)
where
    P: SetCoverProblem<2> + Clone + Send + Sync,
{
    let mut moead = Moead::new(problem, config, initial_population, seed);
    let archive = moead.run(max_generations, max_duration);
    let explored = std::mem::replace(
        &mut moead.explored_solutions,
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
            name: "moead_test".to_string(),
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
    fn test_simplex_lattice_count() {
        // D=2: C(n+1, 1) = n+1
        assert_eq!(simplex_lattice_count(10, 2), 11);
        assert_eq!(simplex_lattice_count(99, 2), 100);

        // D=3: C(n+2, 2)
        assert_eq!(simplex_lattice_count(5, 3), 21); // C(7,2) = 21
        assert_eq!(simplex_lattice_count(10, 3), 66); // C(12,2) = 66

        // D=4: C(n+3, 3)
        assert_eq!(simplex_lattice_count(7, 4), 120); // C(10,3) = 120
        assert_eq!(simplex_lattice_count(8, 4), 165); // C(11,3) = 165
        assert_eq!(simplex_lattice_count(12, 4), 455); // C(15,3) = 455
    }

    #[test]
    fn test_compute_divisions_for_target_d2() {
        // D=2: n = target - 1
        assert_eq!(compute_divisions_for_target::<2>(100), 99);
        assert_eq!(compute_divisions_for_target::<2>(10), 9);
    }

    #[test]
    fn test_compute_divisions_for_target_d4() {
        // D=4: find largest n such that C(n+3,3) <= target
        let n = compute_divisions_for_target::<4>(100);
        let count = simplex_lattice_count(n, 4);
        assert!(count <= 100, "count {count} should be <= 100");
        // Next division should exceed target
        let next_count = simplex_lattice_count(n + 1, 4);
        assert!(next_count > 100, "next count {next_count} should be > 100");

        // For target=200: should give n=8 (C(11,3)=165)
        let n200 = compute_divisions_for_target::<4>(200);
        assert_eq!(n200, 8);
        assert_eq!(simplex_lattice_count(n200, 4), 165);

        // For target=500: should give n=12 (C(15,3)=455)
        let n500 = compute_divisions_for_target::<4>(500);
        assert_eq!(n500, 12);
        assert_eq!(simplex_lattice_count(n500, 4), 455);
    }

    #[test]
    fn test_moead_runs_and_produces_feasible_solutions() {
        let problem = make_test_problem();
        let config = MoeadConfig {
            num_divisions: 9, // 10 weight vectors
            neighbourhood_size: 5,
            max_replacements: 2,
            ..Default::default()
        };

        let mut moead = Moead::new(&problem, config, None, 42);
        let archive = moead.run(50, Duration::from_secs(5));

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
                        "Archive solution {i} dominates solution {j}: {:?} vs {:?}",
                        a.objectives(),
                        b.objectives()
                    );
                }
            }
        }
    }

    #[test]
    fn test_moead_with_initial_population() {
        let problem = make_test_problem();

        // Create manual initial population
        let init = vec![
            BitsetEncodedSolution::from_selected_images(&[0, 1], &problem),
            BitsetEncodedSolution::from_selected_images(&[3], &problem),
            BitsetEncodedSolution::from_selected_images(&[0, 1, 2], &problem),
        ];

        let config = MoeadConfig {
            num_divisions: 9,
            neighbourhood_size: 4,
            ..Default::default()
        };

        let mut moead = Moead::new(&problem, config, Some(init), 123);
        let archive = moead.run(30, Duration::from_secs(3));

        assert!(!archive.is_empty());
        for sol in &archive {
            assert!(problem.is_set_cover(sol));
        }
    }

    #[test]
    fn test_moead_respects_timeout() {
        let problem = make_test_problem();
        let config = MoeadConfig {
            num_divisions: 9,
            neighbourhood_size: 5,
            ..Default::default()
        };

        let start = std::time::Instant::now();
        let mut moead = Moead::new(&problem, config, None, 99);
        let _archive = moead.run(1_000_000, Duration::from_millis(200));
        let elapsed = start.elapsed();

        assert!(
            elapsed < Duration::from_secs(2),
            "Should respect timeout, but took {:?}",
            elapsed
        );
    }

    #[test]
    fn test_moead_pbi_mode() {
        let problem = make_test_problem();
        let config = MoeadConfig {
            num_divisions: 9,
            neighbourhood_size: 5,
            use_pbi: true,
            pbi_theta: 5.0,
            ..Default::default()
        };

        let mut moead = Moead::new(&problem, config, None, 42);
        let archive = moead.run(30, Duration::from_secs(3));

        assert!(!archive.is_empty());
        for sol in &archive {
            assert!(problem.is_set_cover(sol));
        }
    }

    #[test]
    fn test_ideal_point_updated() {
        let problem = make_test_problem();
        let config = MoeadConfig {
            num_divisions: 9,
            neighbourhood_size: 5,
            ..Default::default()
        };

        let mut moead = Moead::new(&problem, config, None, 42);
        let initial_ideal = moead.ideal_point;

        let _archive = moead.run(20, Duration::from_secs(3));

        // Ideal point should be <= initial for all objectives (minimization)
        for i in 0..2 {
            assert!(
                moead.ideal_point[i] <= initial_ideal[i],
                "Ideal point should not worsen: obj {} was {} now {}",
                i,
                initial_ideal[i],
                moead.ideal_point[i]
            );
        }
    }

    #[test]
    fn test_run_moead_convenience() {
        let problem = make_test_problem();
        let config = MoeadConfig {
            num_divisions: 9,
            neighbourhood_size: 5,
            ..Default::default()
        };

        let (archive, explored) = run_moead(&problem, config, None, 20, Duration::from_secs(3), 7);

        assert!(!archive.is_empty());
        assert!(!explored.solutions.is_empty());
    }

    #[test]
    fn test_scalarization_consistency() {
        let problem = make_test_problem();
        let config = MoeadConfig {
            num_divisions: 4,
            neighbourhood_size: 3,
            ..Default::default()
        };

        let moead = Moead::new(&problem, config, None, 42);

        // All weight vectors should sum to 1
        for w in &moead.weights {
            let sum: f64 = w.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "Weight vector must sum to 1.0, got {sum}"
            );
        }

        // Tchebycheff of ideal point should be 0
        for w in &moead.weights {
            let val = tchebycheff_value(&[0, 0], w, &[0.0, 0.0]);
            assert!(
                val.abs() < 1e-6,
                "Tchebycheff of ideal point should be ~0, got {val}"
            );
        }
    }
}
