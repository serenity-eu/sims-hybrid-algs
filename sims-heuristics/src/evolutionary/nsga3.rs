//! NSGA-III (Non-dominated Sorting Genetic Algorithm III) for SIMS
//!
//! Implements NSGA-III (Deb & Jain, 2014) adapted for the Satellite Image Mosaic
//! Selection problem. NSGA-III replaces NSGA-II's crowding-distance diversity
//! mechanism with **reference-point niching**, which is significantly better for
//! many-objective (D ≥ 3) problems where crowding distance degrades.
//!
//! ## Key Difference from NSGA-II
//!
//! Survivor selection for the partial last Pareto front uses structured reference
//! points on the (D-1)-simplex (same lattice as MOEA/D weight vectors). Solutions
//! are associated with the nearest reference point by perpendicular distance in
//! normalised objective space. The niche-preservation operator iteratively selects
//! solutions from under-populated reference directions, ensuring even coverage of
//! the Pareto front approximation.
//!
//! ## Algorithm Outline
//!
//! 1. Generate reference points (simplex-lattice design, auto-sized for D ≥ 3).
//! 2. Initialise population of size N = |reference points|.
//! 3. For each generation:
//!    a. Binary tournament selection + coverage-aware crossover + mutation.
//!    b. Fast non-dominated sort of parent + offspring (size 2N).
//!    c. Fill next generation front-by-front.
//!    d. For the partial last front: NSGA-III niching (normalisation →
//!       association → niche-count-based selection).
//! 4. Maintain an external Pareto archive updated incrementally each generation.

use std::time::Duration;

use pareto::{HasObjectives, MoSolution};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand::{seq::SliceRandom, Rng};
use tracing::{info, info_span};

use crate::explored_solutions_data::ExploredSolutionsData;
use crate::problem::SetCoverProblem;
use crate::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;
use crate::timer::Timer;

use super::operators::{
    add_then_prune_mutation, binary_tournament, bitflip_mutation, coverage_biased_crossover,
    crowding_from_fronts, ensure_mutated, fast_non_dominated_sort, generate_weight_vectors,
    multi_swap_mutation, random_population, ranks_from_fronts, shift_mutation, swap_mutation,
    uniform_crossover,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for NSGA-III.
#[derive(Debug, Clone)]
pub struct Nsga3Config {
    /// Number of weight-vector divisions for simplex-lattice reference points.
    /// For D=2: population_size = num_divisions + 1.
    /// For D ≥ 3 with `auto_divisions=true`, this is ignored and computed from
    /// `target_pop_size`.
    pub num_divisions: usize,

    /// Target population size for automatic `num_divisions` computation.
    /// Only used when `auto_divisions=true` and D ≥ 3.
    pub target_pop_size: usize,

    /// Whether to automatically compute `num_divisions` from `target_pop_size`
    /// for D ≥ 3. Defaults to true. For D=2, `num_divisions` is always used.
    pub auto_divisions: bool,

    /// Crossover probability.
    pub crossover_rate: f64,
    /// Per-individual mutation probability for swap mutation.
    pub swap_mutation_rate: f64,
    /// Per-individual mutation probability for add-then-prune mutation.
    pub add_prune_mutation_rate: f64,
    /// Per-bit mutation probability for bit-flip mutation (0.0 = disabled).
    pub bitflip_mutation_rate: f64,
    /// Maximum number of images to remove in multi-swap mutation.
    pub multi_swap_max_removals: usize,
    /// Per-individual probability of applying multi-swap instead of single swap.
    pub multi_swap_rate: f64,
    /// Per-individual probability of applying shift (coverage-guided) mutation.
    pub shift_mutation_rate: f64,
    /// Fraction of crossovers using coverage-biased crossover vs uniform.
    pub coverage_biased_crossover_fraction: f64,
    /// If true, guarantee at least one mutation operator fires on every offspring.
    pub ensure_mutation: bool,
    /// Number of consecutive generations with no archive improvement before
    /// injecting random individuals to restore diversity.
    pub stagnation_limit: usize,
}

impl Default for Nsga3Config {
    fn default() -> Self {
        Self {
            num_divisions: 12,
            target_pop_size: 200,
            auto_divisions: true,
            crossover_rate: 0.9,
            swap_mutation_rate: 0.4,
            add_prune_mutation_rate: 0.3,
            bitflip_mutation_rate: 0.0,
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
    archive_size: usize,
    offspring_generated: usize,
    offspring_novel_objectives: usize,
    offspring_archive_inserted: usize,
    elapsed_ms: u128,
}

#[derive(Debug, Clone, Default)]
struct EvolutionDiagnostics {
    offspring_generated: usize,
    offspring_novel_genotype: usize,
    offspring_novel_objectives: usize,
    offspring_archive_inserted: usize,
}

// ---------------------------------------------------------------------------
// NSGA-III algorithm
// ---------------------------------------------------------------------------

/// NSGA-III solver for the SIMS multi-objective set-cover problem.
pub struct Nsga3<'a, P, const D: usize>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    problem: &'a P,
    config: Nsga3Config,
    /// Reference points on the (D-1)-simplex (weight vectors from simplex-lattice design).
    reference_points: Vec<[f64; D]>,
    /// Actual population size (= number of reference points).
    population_size: usize,
    population: Vec<BitsetEncodedSolution<P, D>>,
    /// External archive of non-dominated solutions.
    archive: Vec<BitsetEncodedSolution<P, D>>,
    pub explored_solutions: ExploredSolutionsData<D>,
    rng: SmallRng,
    stagnation_counter: usize,
    prev_archive_size: usize,
}

impl<'a, P, const D: usize> Nsga3<'a, P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    /// Create a new NSGA-III instance.
    ///
    /// If `initial_population` is provided, those solutions seed the population
    /// (padded with random solutions to reach `population_size`, or truncated
    /// by contribution-distance selection if too large).
    pub fn new(
        problem: &'a P,
        config: Nsga3Config,
        initial_population: Option<Vec<BitsetEncodedSolution<P, D>>>,
        seed: u64,
    ) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);

        // Compute number of divisions and reference points.
        let num_divisions = if D >= 3 && config.auto_divisions {
            auto_divisions_for_target::<D>(config.target_pop_size)
        } else {
            config.num_divisions
        };
        let reference_points = generate_weight_vectors::<D>(num_divisions);
        let population_size = reference_points.len().max(4);

        info!(
            "NSGA-III: D={D}, num_divisions={num_divisions}, \
             reference_points={}, effective_pop_size={population_size}",
            reference_points.len()
        );

        // Build initial population.
        let population = match initial_population {
            Some(mut init) if !init.is_empty() => {
                if init.len() > population_size {
                    // Too many seeds: diversity-sample down to population_size.
                    init = contribution_distance_sample(&init, population_size, seed);
                }
                // Pad up to population_size with random solutions.
                while init.len() < population_size {
                    let s: u64 = rng.random();
                    init.push(BitsetEncodedSolution::random_with_seed(problem, s));
                }
                init
            }
            _ => random_population(problem, population_size, &mut rng),
        };

        let archive = Vec::new();
        let explored_solutions = ExploredSolutionsData::new(problem.max_objectives());

        Self {
            problem,
            config,
            reference_points,
            population_size,
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

    pub fn run(
        &mut self,
        max_generations: usize,
        max_duration: Duration,
    ) -> Vec<BitsetEncodedSolution<P, D>> {
        let timer = Timer::start(max_duration);

        info!(
            "NSGA-III starting: pop_size={}, ref_points={}, crossover_rate={}, \
             swap_mut={}, shift_mut={}, ensure_mutation={}, timeout={:?}",
            self.population_size,
            self.reference_points.len(),
            self.config.crossover_rate,
            self.config.swap_mutation_rate,
            self.config.shift_mutation_rate,
            self.config.ensure_mutation,
            max_duration,
        );

        // Register and archive the initial population.
        for sol in &self.population {
            self.explored_solutions
                .register_without_selected_images(0, sol, Duration::ZERO);
        }
        let init_pop: Vec<_> = self.population.clone();
        for sol in &init_pop {
            self.try_insert_into_archive(sol);
        }

        for generation in 1..=max_generations {
            let gen_span = info_span!("nsga3_generation", generation = generation);
            let _guard = gen_span.enter();

            if timer.is_expired() {
                info!(
                    "NSGA-III timeout after {} generations, elapsed {:?}",
                    generation - 1,
                    timer.elapsed()
                );
                break;
            }

            // Stagnation detection: inject random individuals when stuck.
            if self.archive.len() == self.prev_archive_size {
                self.stagnation_counter += 1;
            } else {
                self.stagnation_counter = 0;
                self.prev_archive_size = self.archive.len();
            }
            if self.config.stagnation_limit > 0
                && self.stagnation_counter >= self.config.stagnation_limit
            {
                let inject_count = self.population_size / 4;
                info!(
                    "NSGA-III stagnation ({} gens), injecting {} random individuals",
                    self.stagnation_counter, inject_count
                );
                for _ in 0..inject_count {
                    let s: u64 = self.rng.random();
                    let idx = self.rng.random_range(0..self.population.len());
                    self.population[idx] = BitsetEncodedSolution::random_with_seed(self.problem, s);
                }
                self.stagnation_counter = 0;
            }

            // 1. Compute ranks and crowding for tournament selection.
            let fronts = fast_non_dominated_sort(&self.population);
            let ranks = ranks_from_fronts(&fronts, self.population.len());
            let crowding = crowding_from_fronts(&self.population, &fronts);

            // 2. Generate offspring.
            let offspring = self.generate_offspring(&ranks, &crowding);

            // 3. Register offspring and update archive.
            let mut diagnostics = EvolutionDiagnostics {
                offspring_generated: offspring.len(),
                ..EvolutionDiagnostics::default()
            };
            let mut seen_objectives = std::collections::HashSet::new();
            for sol in &offspring {
                if !self.explored_solutions.is_registered(sol) {
                    diagnostics.offspring_novel_genotype += 1;
                    self.explored_solutions.register_without_selected_images(
                        generation,
                        sol,
                        timer.elapsed(),
                    );
                }
                if seen_objectives.insert(*sol.objectives()) {
                    diagnostics.offspring_novel_objectives += 1;
                }
                if self.try_insert_into_archive(sol) {
                    diagnostics.offspring_archive_inserted += 1;
                }
            }

            // 4. Merge parent + offspring, apply NSGA-III survivor selection.
            let mut combined = std::mem::take(&mut self.population);
            combined.extend(offspring);
            self.population = self.survivor_selection(combined);

            // 5. Log stats.
            if generation % 10 == 0 || generation <= 5 {
                let stats = GenerationStats {
                    generation,
                    archive_size: self.archive.len(),
                    offspring_generated: diagnostics.offspring_generated,
                    offspring_novel_objectives: diagnostics.offspring_novel_objectives,
                    offspring_archive_inserted: diagnostics.offspring_archive_inserted,
                    elapsed_ms: timer.elapsed().as_millis(),
                };
                info!(
                    "gen={} archive={} novel_obj={}/{} inserted={} elapsed={}ms",
                    stats.generation,
                    stats.archive_size,
                    stats.offspring_novel_objectives,
                    stats.offspring_generated,
                    stats.offspring_archive_inserted,
                    stats.elapsed_ms,
                );
            }

            if generation == max_generations {
                info!(
                    "NSGA-III reached max generations ({}), elapsed {:?}",
                    max_generations,
                    timer.elapsed()
                );
            }
        }

        self.explored_solutions.num_iterations = max_generations;
        info!(
            "NSGA-III completed: archive_size={}, total_explored={}",
            self.archive.len(),
            self.explored_solutions.solutions.len(),
        );

        self.archive.clone()
    }

    // -----------------------------------------------------------------
    // Offspring generation (identical to NSGA-II)
    // -----------------------------------------------------------------

    fn generate_offspring(
        &mut self,
        ranks: &[usize],
        crowding: &[f64],
    ) -> Vec<BitsetEncodedSolution<P, D>> {
        let actual_pop_size = self.population.len();
        let mut offspring = Vec::with_capacity(self.population_size);

        while offspring.len() < self.population_size {
            let p1_idx = binary_tournament(ranks, crowding, &mut self.rng, actual_pop_size);
            let p2_idx = binary_tournament(ranks, crowding, &mut self.rng, actual_pop_size);

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
                if self.rng.random_bool(0.5) {
                    self.population[p1_idx].clone()
                } else {
                    self.population[p2_idx].clone()
                }
            };

            child = self.mutate(child);
            offspring.push(child);
        }
        offspring
    }

    fn mutate(&mut self, solution: BitsetEncodedSolution<P, D>) -> BitsetEncodedSolution<P, D> {
        let original = solution.clone();
        let mut result = solution;

        if self.rng.random_bool(self.config.multi_swap_rate) {
            result = multi_swap_mutation(
                &result,
                self.problem,
                &mut self.rng,
                1.0,
                self.config.multi_swap_max_removals,
            );
        } else if self.rng.random_bool(self.config.shift_mutation_rate) {
            result = shift_mutation(&result, self.problem, &mut self.rng, 1.0);
        } else {
            result = swap_mutation(
                &result,
                self.problem,
                &mut self.rng,
                self.config.swap_mutation_rate,
            );
        }

        result = add_then_prune_mutation(
            &result,
            self.problem,
            &mut self.rng,
            self.config.add_prune_mutation_rate,
        );

        if self.config.bitflip_mutation_rate > 0.0 {
            let per_bit = self.config.bitflip_mutation_rate / self.problem.num_images() as f64;
            result = bitflip_mutation(&result, self.problem, &mut self.rng, per_bit);
        }

        if self.config.ensure_mutation {
            result = ensure_mutated(&original, result, self.problem, &mut self.rng);
        }

        result
    }

    // -----------------------------------------------------------------
    // NSGA-III survivor selection
    // -----------------------------------------------------------------

    /// Survivor selection using NSGA-III reference-point niching.
    ///
    /// Fronts 0..l-1 are accepted wholesale. For the partial last front F_l,
    /// reference-point niching (normalisation + association + niche preservation)
    /// selects the k remaining individuals to fill the population.
    fn survivor_selection(
        &mut self,
        mut combined: Vec<BitsetEncodedSolution<P, D>>,
    ) -> Vec<BitsetEncodedSolution<P, D>> {
        let target_size = self.population_size;

        // Deduplicate by objective vector to prevent population collapse.
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
                // Accept entire front.
                for &idx in front {
                    next_gen.push(combined[idx].clone());
                }
            } else {
                // Partial last front: use NSGA-III niching.
                let remaining = target_size - next_gen.len();
                if remaining > 0 {
                    self.nsga3_niching_fill(&combined, &mut next_gen, front, remaining);
                }
                return next_gen;
            }
        }

        next_gen
    }

    /// Fill `next_gen` with `remaining` solutions from `last_front` using NSGA-III niching.
    fn nsga3_niching_fill(
        &mut self,
        combined: &[BitsetEncodedSolution<P, D>],
        next_gen: &mut Vec<BitsetEncodedSolution<P, D>>,
        last_front: &[usize],
        remaining: usize,
    ) {
        let n_refs = self.reference_points.len();

        // --- Step 1: Ideal point over next_gen ∪ last_front ---
        let mut ideal = [f64::INFINITY; D];
        for sol in next_gen.iter() {
            for d in 0..D {
                let v = sol.objectives()[d] as f64;
                if v < ideal[d] {
                    ideal[d] = v;
                }
            }
        }
        for &idx in last_front {
            for d in 0..D {
                let v = combined[idx].objectives()[d] as f64;
                if v < ideal[d] {
                    ideal[d] = v;
                }
            }
        }

        // --- Step 2: Translate objectives ---
        let translated: Vec<[f64; D]> = next_gen
            .iter()
            .chain(last_front.iter().map(|&i| &combined[i]))
            .map(|sol| {
                let mut t = [0.0f64; D];
                for d in 0..D {
                    t[d] = sol.objectives()[d] as f64 - ideal[d];
                }
                t
            })
            .collect();

        let next_gen_count = next_gen.len();

        // --- Step 3: Extreme points via ASF (Achievement Scalarising Function) ---
        // For axis j: w_j = 1, w_{i≠j} = 1e-6.
        let mut extreme = [[0.0f64; D]; D];
        for j in 0..D {
            let mut best_asf = f64::INFINITY;
            let mut best_idx = 0;
            for (i, t) in translated.iter().enumerate() {
                let asf = (0..D)
                    .map(|k| {
                        let w = if k == j { 1.0 } else { 1e-6 };
                        t[k] / w
                    })
                    .fold(f64::NEG_INFINITY, f64::max);
                if asf < best_asf {
                    best_asf = asf;
                    best_idx = i;
                }
            }
            extreme[j] = translated[best_idx];
        }

        // --- Step 4: Hyperplane intercepts via Gaussian elimination ---
        // Solve E * (1/a) = 1 where E[j][k] = extreme[j][k].
        // If degenerate, fall back to per-axis range normalisation.
        let intercepts = compute_intercepts(&extreme);

        // --- Step 5: Normalise objectives ---
        let normalise = |t: &[f64; D]| -> [f64; D] {
            let mut n = [0.0f64; D];
            for d in 0..D {
                let denom = intercepts[d];
                n[d] = if denom.abs() > 1e-10 {
                    t[d] / denom
                } else {
                    t[d]
                };
            }
            n
        };

        let norm_all: Vec<[f64; D]> = translated.iter().map(|t| normalise(t)).collect();

        // --- Step 6: Associate each solution with nearest reference point ---
        // Perpendicular distance from normalised objective to reference line through origin.
        let assoc: Vec<(usize, f64)> = norm_all
            .iter()
            .map(|f| closest_reference_point(f, &self.reference_points))
            .collect();

        // --- Step 7: Niche counts from already-selected solutions ---
        let mut niche_count = vec![0usize; n_refs];
        for (ref_idx, _) in &assoc[..next_gen_count] {
            niche_count[*ref_idx] += 1;
        }

        // --- Step 8: Niche preservation — select `remaining` from last_front ---
        // Build per-reference-point lists of last_front solution indices (local indexing).
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
            // Find reference point with minimum niche count among those with
            // at least one available candidate in the last front.
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
                    std::cmp::Ordering::Equal => {
                        eligible_refs.push(r);
                    }
                    std::cmp::Ordering::Greater => {}
                }
            }

            if eligible_refs.is_empty() {
                break; // No more candidates (shouldn't happen on well-formed inputs).
            }

            // Random tie-break among equally-loaded reference points.
            let r_min = eligible_refs[self.rng.random_range(0..eligible_refs.len())];

            // Pick the candidate associated with r_min.
            let candidates_for_r: Vec<usize> = ref_to_front_candidates[r_min]
                .iter()
                .copied()
                .filter(|&li| available[li])
                .collect();

            let chosen_local = if niche_count[r_min] == 0 {
                // Niche empty: pick solution with minimum perpendicular distance to r_min.
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
                // Niche occupied: pick randomly.
                candidates_for_r[self.rng.random_range(0..candidates_for_r.len())]
            };

            next_gen.push(combined[last_front[chosen_local]].clone());
            available[chosen_local] = false;
            niche_count[r_min] += 1;
            selected_count += 1;
        }
    }

    // -----------------------------------------------------------------
    // External archive maintenance
    // -----------------------------------------------------------------

    fn try_insert_into_archive(&mut self, solution: &BitsetEncodedSolution<P, D>) -> bool {
        for existing in &self.archive {
            if existing.dominates(solution.objectives())
                || existing.objectives() == solution.objectives()
            {
                return false;
            }
        }
        self.archive
            .retain(|existing| !solution.dominates(existing.objectives()));
        self.archive.push(solution.clone());
        true
    }
}

// ---------------------------------------------------------------------------
// NSGA-III helper functions
// ---------------------------------------------------------------------------

/// Compute hyperplane intercepts from D extreme points using Gaussian elimination.
///
/// Solves the system: `sum_j(extreme[i][j] / a[j]) = 1` for each extreme point i.
/// i.e., the matrix equation `E * x = ones` where `x[j] = 1/a[j]`.
/// Returns intercepts `a[j]`. Falls back to per-axis max if degenerate.
fn compute_intercepts<const D: usize>(extreme: &[[f64; D]; D]) -> [f64; D] {
    // Build augmented matrix [E | 1]
    let mut mat = [[0.0f64; D]; D];
    for i in 0..D {
        mat[i] = extreme[i];
    }
    let mut rhs = [1.0f64; D];

    // Gaussian elimination with partial pivoting
    for col in 0..D {
        // Find pivot
        let mut max_val = mat[col][col].abs();
        let mut pivot_row = col;
        for row in (col + 1)..D {
            if mat[row][col].abs() > max_val {
                max_val = mat[row][col].abs();
                pivot_row = row;
            }
        }
        if max_val < 1e-10 {
            // Degenerate: fall back to per-axis max over all extreme points.
            let mut fallback = [1.0f64; D];
            for d in 0..D {
                let max_d = extreme
                    .iter()
                    .map(|e| e[d])
                    .fold(f64::NEG_INFINITY, f64::max);
                fallback[d] = if max_d > 1e-10 { max_d } else { 1.0 };
            }
            return fallback;
        }
        mat.swap(col, pivot_row);
        rhs.swap(col, pivot_row);

        let pivot = mat[col][col];
        for k in col..D {
            mat[col][k] /= pivot;
        }
        rhs[col] /= pivot;

        for row in 0..D {
            if row == col {
                continue;
            }
            let factor = mat[row][col];
            for k in col..D {
                mat[row][k] -= factor * mat[col][k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    // rhs[j] = 1/a[j], so a[j] = 1/rhs[j]
    let mut intercepts = [1.0f64; D];
    for d in 0..D {
        let inv = rhs[d];
        intercepts[d] = if inv.abs() > 1e-10 { 1.0 / inv } else { 1.0 };
        if intercepts[d] <= 0.0 {
            intercepts[d] = 1.0; // guard against negative intercepts
        }
    }
    intercepts
}

/// Associate a normalised objective vector with the nearest reference point.
///
/// Returns `(ref_idx, perpendicular_distance)`.
/// Distance = ||f'' - (f'' · r̂) r̂|| where r̂ = r / ||r||.
fn closest_reference_point<const D: usize>(
    f_norm: &[f64; D],
    reference_points: &[[f64; D]],
) -> (usize, f64) {
    let mut best_ref = 0;
    let mut best_dist = f64::INFINITY;

    for (i, r) in reference_points.iter().enumerate() {
        // Compute unit vector of reference direction.
        let r_norm_sq: f64 = r.iter().map(|&v| v * v).sum();
        let r_len = r_norm_sq.sqrt();
        if r_len < 1e-12 {
            continue;
        }

        // Projection of f_norm onto r̂: d1 = (f · r̂)
        let dot: f64 = f_norm
            .iter()
            .zip(r.iter())
            .map(|(&fv, &rv)| fv * rv / r_len)
            .sum();

        // Perpendicular distance: d_perp = ||f - d1 * r̂||
        let mut perp_sq = 0.0f64;
        for d in 0..D {
            let proj = dot * r[d] / r_len;
            let diff = f_norm[d] - proj;
            perp_sq += diff * diff;
        }
        let dist = perp_sq.sqrt();

        if dist < best_dist {
            best_dist = dist;
            best_ref = i;
        }
    }

    (best_ref, best_dist)
}

/// Compute the number of simplex-lattice divisions such that C(n+D-1, D-1) ≈ target.
fn auto_divisions_for_target<const D: usize>(target: usize) -> usize {
    fn n_weight_vectors(n: usize, d: usize) -> usize {
        // C(n + d - 1, d - 1)
        let mut result = 1usize;
        for i in 0..(d - 1) {
            result = result.saturating_mul(n + d - 1 - i);
            result /= i + 1;
        }
        result
    }

    // Find the smallest n such that n_weight_vectors(n, D) >= target,
    // or the n that produces the closest count.
    let mut best_n = 1usize;
    let mut best_diff = usize::MAX;
    for n in 1..=50 {
        let count = n_weight_vectors(n, D);
        let diff = if count >= target {
            count - target
        } else {
            target - count
        };
        if diff < best_diff {
            best_diff = diff;
            best_n = n;
        }
        if count >= target {
            break;
        }
    }
    best_n
}

/// Greedy contribution-distance sampling: select `target` diverse solutions from `archive`.
fn contribution_distance_sample<P, const D: usize>(
    archive: &[BitsetEncodedSolution<P, D>],
    target: usize,
    seed: u64,
) -> Vec<BitsetEncodedSolution<P, D>>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    let n = archive.len();
    if n <= target {
        return archive.to_vec();
    }

    // Normalise objectives.
    let mut obj_min = [f64::INFINITY; D];
    let mut obj_max = [f64::NEG_INFINITY; D];
    for sol in archive {
        for d in 0..D {
            let v = sol.objectives()[d] as f64;
            obj_min[d] = obj_min[d].min(v);
            obj_max[d] = obj_max[d].max(v);
        }
    }
    let norm: Vec<[f64; D]> = archive
        .iter()
        .map(|sol| {
            let mut n = [0.0f64; D];
            for d in 0..D {
                let r = obj_max[d] - obj_min[d];
                n[d] = if r > 1e-12 {
                    (sol.objectives()[d] as f64 - obj_min[d]) / r
                } else {
                    0.0
                };
            }
            n
        })
        .collect();

    let mut rng = SmallRng::seed_from_u64(seed);
    let first = rng.random_range(0..n);
    let mut selected = vec![first];
    let mut available = vec![true; n];
    available[first] = false;
    let mut min_dist: Vec<f64> = (0..n)
        .map(|i| euclidean_sq(&norm[i], &norm[first]).sqrt())
        .collect();
    min_dist[first] = 0.0;

    while selected.len() < target {
        let best = (0..n).filter(|&i| available[i]).max_by(|&a, &b| {
            min_dist[a]
                .partial_cmp(&min_dist[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let Some(pick) = best else { break };
        selected.push(pick);
        available[pick] = false;
        for i in 0..n {
            if available[i] {
                let d = euclidean_sq(&norm[i], &norm[pick]).sqrt();
                if d < min_dist[i] {
                    min_dist[i] = d;
                }
            }
        }
    }

    selected.into_iter().map(|i| archive[i].clone()).collect()
}

#[inline]
fn euclidean_sq<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    let mut s = 0.0f64;
    for d in 0..D {
        let diff = a[d] - b[d];
        s += diff * diff;
    }
    s
}
