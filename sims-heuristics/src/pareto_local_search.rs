use std::{
    mem,
    ops::RangeInclusive,
    time::{Duration, Instant},
};

#[cfg(feature = "scalarized_selection")]
use std::collections::HashSet;

use log::{debug, info};
use pareto::{ParetoFront, ScalarizedArchiveQuery};
use rand::SeedableRng;
use rand::prelude::*;
use tracing::{debug_span, info_span, instrument};

use crate::{
    diverse_probe_iter::diverse_probe_iter,
    explored_solutions_data::{ExploredSolutionsData, SolutionFingerprint},
    objective_tracker::TrackerCollection,
    pls_config::{PlsOptimizations, SolutionSelectionMode},
    problem::SetCoverProblem,
    solution::{EncodedSolution, ImageSet},
    timer::Timer,
};

#[cfg(feature = "scalarized_selection")]
use crate::{
    pls_config::ScalarizedSelectionSource,
    scalarization::{
        WeightedChebycheffCoeffs, bounds_from_ideal_nadir, compute_ideal_from_objectives,
        compute_nadir_from_objectives, sample_weight_vectors,
    },
};

/// Statistics tracking during a single step of the algorithm.
/// Exposed as `pub(crate)` so concurrent workers can aggregate per-step metrics.
pub(crate) struct StepStats {
    pub(crate) explored_neighbor_count: usize,
    pub(crate) duplicated_neighbor_count: usize,
    pub(crate) auxiliary_added_count: usize,
    pub(crate) auxiliary_len: usize,
    pub(crate) pareto_added_count: usize,
    pub(crate) pareto_initial_count: usize,
}

impl StepStats {
    pub(crate) const fn new(pareto_initial_count: usize) -> Self {
        Self {
            explored_neighbor_count: 0,
            duplicated_neighbor_count: 0,
            auxiliary_added_count: 0,
            auxiliary_len: 0,
            pareto_added_count: 0,
            pareto_initial_count,
        }
    }
}

/// Result bundle from a single PLS step: status + accumulated statistics.
pub(crate) struct StepResult {
    pub(crate) status: StepStatus,
    pub(crate) stats: StepStats,
}

/// Metrics calculated during an iteration for logging
struct IterationMetrics {
    auxiliary_removed_count: usize,
    pareto_removed_count: usize,
    neighborhood_size: u32,
    duration_us: u128,
    duplicated_percent: f32,
    per_solution_search_time: f32,
}

pub struct ParetoLocalSearch<'a, T, S, P, const D: usize>
where
    T: ImageSet<D> + EncodedSolution<P, D>,
    S: ParetoFront<'a, T> + Clone,
    P: SetCoverProblem<D>,
{
    /// Reference to problem instance
    problem: &'a P,
    /// Current population
    population: S,
    /// Approximation of Pareto set
    approximated_pareto_set: S,
    /// Whether algorithm is deterministic
    is_deterministic: bool,
    /// Current neighborhood size
    pub neigborhood_structure: u32,
    /// Range of possible neighborhood sizes
    pub neighborhood_size_range: RangeInclusive<u32>,
    /// Explored solutions objectives
    pub explored_solutions: ExploredSolutionsData<D>,
    /// Spare tracker for neighborhood exploration (reused to avoid allocations)
    spare_tracker: T::Trackers,
    pub optimizations: PlsOptimizations,
    _phantom: std::marker::PhantomData<T>,
}

#[derive(Eq, PartialEq)]
pub enum StepStatus {
    /// New population was found
    NewPopulation,
    /// Neighborhood structure was increased
    IncreasedNeighborhoodStructure,
    /// All neighborhood structures were explored
    AllNeighborhoodStructuresExplored,
}

/// SA-PLS step outcome. Maps to the design doc's section 16.7 termination semantics.
#[derive(Eq, PartialEq, Debug, Clone, Copy)]
#[cfg(feature = "parallel")]
pub enum SAPlsStatus {
    /// Scalarized auxiliary was non-empty; new population generated.
    NewPopulation,
    /// Scalarized auxiliary was empty; neighborhood structure increased.
    IncreasedNeighborhoodStructure,
    /// All neighborhood structures exhausted under scalarized criterion.
    /// Worker can switch to fallback (standard PLS) or stop.
    ScalarizedExhausted,
}

impl<'a, T, S, P, const D: usize> ParetoLocalSearch<'a, T, S, P, D>
where
    T: ImageSet<D> + EncodedSolution<P, D>,
    S: ParetoFront<'a, T>
        + Clone
        + FromIterator<T>
        + IntoIterator<Item = T>
        + ScalarizedArchiveQuery<T, D>,
    P: SetCoverProblem<D>,
{
    #[expect(
        clippy::too_many_arguments,
        reason = "Plumbs explicit mutable state to allow streaming neighbor evaluation"
    )]
    fn process_neighbor_streaming(
        problem: &P,
        explored_solutions: &mut ExploredSolutionsData<D>,
        approximated_pareto_set: &mut S,
        neighbor: &T,
        neighbor_index: usize,
        current_solution: &T,
        iteration: usize,
        timer: &Timer,
        step_stats: &mut StepStats,
        auxiliary_population: &mut S,
    ) -> bool {
        step_stats.explored_neighbor_count += 1;
        debug!("######## NEIGHBOR {neighbor_index} {neighbor:?} ########");

        #[cfg(debug_assertions)]
        {
            let is_valid = neighbor.is_valid(problem);
            if !is_valid {
                eprintln!("Generated neighbor {neighbor_index} is invalid: {neighbor:?}");
                // Check coverage manually to debug
                let selected_images: Vec<usize> = neighbor.selected_images().collect();
                eprintln!("  Selected images: {selected_images:?}");
                let mut covered_elements = std::collections::HashSet::new();
                for &img_idx in &selected_images {
                    for elem in problem.image_elements(img_idx) {
                        covered_elements.insert(elem);
                    }
                }
                eprintln!(
                    "  Total elements covered: {}/{}",
                    covered_elements.len(),
                    problem.num_elements()
                );
                if covered_elements.len() < problem.num_elements() {
                    let uncovered: Vec<usize> = (0..problem.num_elements())
                        .filter(|e| !covered_elements.contains(e))
                        .take(20)
                        .collect();
                    eprintln!("  Uncovered elements (first 20): {uncovered:?}");
                }
            }
            debug_assert!(
                is_valid,
                "Generated neighbor {neighbor_index} is invalid: {neighbor:?}"
            );
        }

        if explored_solutions.is_registered(neighbor) {
            step_stats.duplicated_neighbor_count += 1;
            tracing::trace!("Neighbor nr {neighbor_index} already explored, skipping");
            return false;
        }

        explored_solutions.register_without_selected_images(iteration, neighbor, timer.elapsed());

        if neighbor.is_covered_by(current_solution.objectives()) {
            tracing::trace!(
                "Neighbor nr {neighbor_index} is dominated by current solution, discarding"
            );
            return false;
        }

        if approximated_pareto_set.try_insert(neighbor) {
            step_stats.pareto_added_count += 1;

            if auxiliary_population.try_insert(neighbor) {
                step_stats.auxiliary_added_count += 1;
            } else {
                debug!(
                    "Neighbor nr {neighbor_index} is dominated so it wasn't added to auxiliary population"
                );
            }
        } else {
            tracing::trace!("Neighbor nr {neighbor_index} rejected from Pareto set");
        }

        if timer.is_expired() {
            info!("Timer expired. Stop exploring neighbors.");
            tracing::warn!("Timer expired during neighbor processing");
            return true;
        }

        false
    }

    pub fn new(
        problem: &'a P,
        initial_population: &S,
        neighborhood_size_range: RangeInclusive<u32>,
        is_deterministic: bool,
        optimizations: PlsOptimizations,
    ) -> Self {
        let mut population = S::new("population");
        // Initialize ExploredSolutionsData with the problem's max objectives array
        let mut explored_solutions = ExploredSolutionsData::<D>::new(problem.max_objectives());
        let mut initial_timestamp_us: u64 = 1;
        initial_population.iter().for_each(|solution| {
            if population.try_insert(solution) {
                explored_solutions.register(
                    0,
                    solution,
                    Duration::from_micros(initial_timestamp_us),
                    solution.selected_images().collect(),
                );
                initial_timestamp_us += 1;
            }
        });

        // Add greedy-constructed solutions targeting each objective individually.
        if optimizations.use_greedy_initial_population {
            let greedy_solutions = T::greedy_initial_solutions(problem);
            info!(
                "Generated {} greedy initial solutions",
                greedy_solutions.len()
            );
            for solution in &greedy_solutions {
                if population.try_insert(solution) {
                    explored_solutions.register(
                        0,
                        solution,
                        Duration::from_micros(initial_timestamp_us),
                        solution.selected_images().collect(),
                    );
                    initial_timestamp_us += 1;
                    debug!("Greedy solution added to population: {solution:?}");
                }
            }
        }

        for (i, solution) in population.iter().enumerate() {
            debug!("Initial solution {i}: {solution:?}");
        }

        let approximated_pareto_set = population.clone().with_name("approximated Pareto set");
        let spare_tracker = T::Trackers::new(problem);
        ParetoLocalSearch {
            problem,
            population,
            approximated_pareto_set,
            neigborhood_structure: *neighborhood_size_range.start(),
            neighborhood_size_range,
            explored_solutions,
            is_deterministic,
            spare_tracker,
            optimizations,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Single PLS step, exposed for concurrent worker use.
    /// Returns `StepResult` bundling both the step status and per-step statistics.
    #[instrument(level = "debug", skip(self, timer), fields(
        iteration,
        population_size = self.population.len(),
        neighborhood_structure = self.neigborhood_structure
    ))]
    pub(crate) fn step(&mut self, iteration: usize, timer: &Timer) -> StepResult {
        let step_time = Instant::now();
        let mut step_stats = StepStats::new(self.approximated_pareto_set.len());
        let mut auxiliary_population = S::new("auxiliary");

        tracing::debug!("Starting PLS step");

        let population_validation_span = debug_span!("validate_population");
        let _validation_guard = population_validation_span.enter();
        self.population.validate();

        let population = mem::replace(&mut self.population, S::new("population"));

        self.explore_population_neighborhoods(
            population,
            iteration,
            timer,
            &mut step_stats,
            &mut auxiliary_population,
        );

        step_stats.auxiliary_len = auxiliary_population.len();

        tracing::debug!(
            explored_neighbors = step_stats.explored_neighbor_count,
            duplicated_neighbors = step_stats.duplicated_neighbor_count,
            auxiliary_size = step_stats.auxiliary_len,
            "Step neighborhood exploration completed"
        );

        self.log_iteration_stats(iteration, step_time, &step_stats);
        Self::log_auxiliary_population(&auxiliary_population);
        self.log_pareto_front();

        let status = self.determine_next_step(auxiliary_population);
        StepResult {
            status,
            stats: step_stats,
        }
    }

    #[instrument(level = "debug", skip(self, population, timer, step_stats, auxiliary_population), fields(
        iteration,
        population_size = %population.len()
    ))]
    fn explore_population_neighborhoods(
        &mut self,
        population: S,
        iteration: usize,
        timer: &Timer,
        step_stats: &mut StepStats,
        auxiliary_population: &mut S,
    ) {
        // Stream neighbors without collecting them into a Vec.
        // We explicitly borrow disjoint mutable fields, so we can hold `&mut spare_tracker`
        // while updating the explored-solution registry and Pareto archive.
        let spare_tracker = &mut self.spare_tracker;
        let explored_solutions = &mut self.explored_solutions;
        let approximated_pareto_set = &mut self.approximated_pareto_set;

        let problem = self.problem;
        let neighborhood_structure = self.neigborhood_structure;
        let is_deterministic = self.is_deterministic;
        let optimizations = &self.optimizations;

        // Configure probabilistic probing if requested
        #[cfg(feature = "probabilistic_probing")]
        {
            use crate::residual_problem::set_runtime_probing_budget;
            if let Some(budget) = self.optimizations.probing_budget {
                set_runtime_probing_budget(budget);
            } else {
                // Exhaustive mode
                set_runtime_probing_budget(crate::residual_problem::PROBING_BUDGET_EXHAUSTIVE);
            }
        }

        let population_vec: Vec<_> = population.into_iter().collect();
        let selected_parents = Self::select_parents_for_exploration(
            &population_vec,
            approximated_pareto_set,
            explored_solutions,
            neighborhood_structure,
            optimizations,
            iteration as u64,
            is_deterministic,
        );

        'population: for (index, solution) in selected_parents.into_iter().enumerate() {
            let solution_span = debug_span!(
                "explore_solution",
                solution_index = index,
                neighborhood_structure = neighborhood_structure
            );
            let _solution_guard = solution_span.enter();

            let neighborhood_generation_span = debug_span!(
                "generate_neighborhood",
                neighborhood_structure = neighborhood_structure
            );
            let _gen_guard = neighborhood_generation_span.enter();

            let neighbor_evaluation_span = debug_span!("evaluate_neighbors");
            let _evaluation_guard = neighbor_evaluation_span.enter();

            let mut neighborhood_iter = solution.neighborhood_iter(
                spare_tracker,
                neighborhood_structure,
                problem,
                timer,
                is_deterministic,
                optimizations,
            );

            for (neighbor_index, neighbor) in neighborhood_iter.by_ref().enumerate() {
                if Self::process_neighbor_streaming(
                    problem,
                    explored_solutions,
                    approximated_pareto_set,
                    &neighbor,
                    neighbor_index,
                    &solution,
                    iteration,
                    timer,
                    step_stats,
                    auxiliary_population,
                ) {
                    tracing::debug!("Timer expired, breaking population exploration");
                    break 'population;
                }
            }

            explored_solutions.update_explored_neighborhood_size(&solution, neighborhood_structure);
        }

        tracing::debug!(
            total_neighbors_explored = step_stats.explored_neighbor_count,
            duplicates_found = step_stats.duplicated_neighbor_count,
            "Population neighborhood exploration completed"
        );
    }

    fn select_parents_for_exploration(
        population_vec: &[T],
        approximated_pareto_set: &S,
        explored_solutions: &mut ExploredSolutionsData<D>,
        neighborhood_structure: u32,
        optimizations: &PlsOptimizations,
        iteration_seed: u64,
        is_deterministic: bool,
    ) -> Vec<T> {
        let effective_mode = if optimizations.use_diverse_probing {
            SolutionSelectionMode::DiverseProbe
        } else {
            optimizations.solution_selection_mode
        };

        match effective_mode {
            SolutionSelectionMode::RandomShuffle => {
                let mut v = population_vec.to_vec();
                if is_deterministic {
                    let mut rng = SmallRng::seed_from_u64(iteration_seed);
                    v.shuffle(&mut rng);
                } else {
                    v.shuffle(&mut rand::rng());
                }
                v
            }
            SolutionSelectionMode::DiverseProbe => {
                if population_vec.len() > 1 {
                    tracing::info!(
                        population_size = population_vec.len(),
                        budget = ?optimizations.diverse_probe_budget,
                        "Using diverse probing for population exploration"
                    );
                    diverse_probe_iter::<T, D>(
                        population_vec.to_vec(),
                        optimizations.diverse_probe_budget,
                        iteration_seed,
                    )
                    .collect()
                } else {
                    population_vec.to_vec()
                }
            }
            #[cfg(feature = "scalarized_selection")]
            SolutionSelectionMode::ScalarizedChebycheff => Self::select_scalarized_parents(
                population_vec,
                approximated_pareto_set,
                explored_solutions,
                neighborhood_structure,
                optimizations,
                iteration_seed,
            ),
            #[cfg(feature = "scalarized_selection")]
            SolutionSelectionMode::DiverseThenScalarizedChebycheff => {
                let prefiltered: Vec<T> = if population_vec.len() > 1 {
                    diverse_probe_iter::<T, D>(
                        population_vec.to_vec(),
                        optimizations.diverse_probe_budget,
                        iteration_seed,
                    )
                    .collect()
                } else {
                    population_vec.to_vec()
                };

                Self::select_scalarized_parents(
                    &prefiltered,
                    approximated_pareto_set,
                    explored_solutions,
                    neighborhood_structure,
                    optimizations,
                    iteration_seed,
                )
            }
        }
    }

    #[cfg(feature = "scalarized_selection")]
    fn select_scalarized_parents(
        population_vec: &[T],
        approximated_pareto_set: &S,
        explored_solutions: &mut ExploredSolutionsData<D>,
        neighborhood_structure: u32,
        optimizations: &PlsOptimizations,
        iteration_seed: u64,
    ) -> Vec<T> {
        let weight_samples = optimizations.scalarized_weight_samples.max(1);
        let weights = sample_weight_vectors::<D>(weight_samples, iteration_seed);

        match optimizations.scalarized_selection_source {
            ScalarizedSelectionSource::Population => {
                let candidate_pool: Vec<T> = population_vec
                    .iter()
                    .filter(|solution| {
                        explored_solutions.explored_neighborhood_size(solution)
                            < neighborhood_structure
                    })
                    .cloned()
                    .collect();

                if candidate_pool.is_empty() {
                    return Vec::new();
                }

                let ideal =
                    compute_ideal_from_objectives(candidate_pool.iter().map(|s| *s.objectives()));
                let nadir =
                    compute_nadir_from_objectives(candidate_pool.iter().map(|s| *s.objectives()));
                let bounds = bounds_from_ideal_nadir(&ideal, &nadir);

                let mut selected_indices = Vec::new();
                let mut seen = HashSet::new();

                for weight in &weights {
                    let coeffs = WeightedChebycheffCoeffs::new(
                        weight,
                        &bounds,
                        optimizations.scalarized_rho,
                    );

                    let best = candidate_pool
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| {
                            let score_a = coeffs.score(a.objectives(), &ideal);
                            let score_b = coeffs.score(b.objectives(), &ideal);
                            score_a
                                .partial_cmp(&score_b)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx);

                    if let Some(idx) = best
                        && seen.insert(idx)
                    {
                        selected_indices.push(idx);
                    }
                }

                if selected_indices.is_empty() {
                    selected_indices.push(0);
                }

                let mut selected: Vec<T> = selected_indices
                    .into_iter()
                    .map(|idx| candidate_pool[idx].clone())
                    .collect();

                if let Some(parent_budget) = optimizations.scalarized_parent_budget {
                    selected.truncate(parent_budget);
                }

                tracing::info!(
                    candidate_pool_size = candidate_pool.len(),
                    selected_parent_count = selected.len(),
                    weight_samples = weight_samples,
                    source = ?optimizations.scalarized_selection_source,
                    accelerated_archive_query = false,
                    "Using scalarized parent selection for population exploration"
                );

                selected
            }
            ScalarizedSelectionSource::Archive => {
                let eligible_archive: Vec<&T> = approximated_pareto_set
                    .iter()
                    .filter(|solution| {
                        explored_solutions.explored_neighborhood_size(solution)
                            < neighborhood_structure
                    })
                    .collect();

                if eligible_archive.is_empty() {
                    return Vec::new();
                }

                let ideal =
                    compute_ideal_from_objectives(eligible_archive.iter().map(|s| *s.objectives()));
                let nadir =
                    compute_nadir_from_objectives(eligible_archive.iter().map(|s| *s.objectives()));
                let bounds = bounds_from_ideal_nadir(&ideal, &nadir);

                let mut selected: Vec<T> = Vec::new();
                let mut seen_objectives: HashSet<[u64; D]> = HashSet::new();

                let accelerated = optimizations.use_nd_tree_scalarized_query;

                if accelerated {
                    for weight in &weights {
                        let coeffs = WeightedChebycheffCoeffs::new(
                            weight,
                            &bounds,
                            optimizations.scalarized_rho,
                        );

                        if let Some((best, _score)) = approximated_pareto_set
                            .find_best_with_pruning(
                                |solution: &T| {
                                    explored_solutions.explored_neighborhood_size(solution)
                                        < neighborhood_structure
                                        && !seen_objectives.contains(solution.objectives())
                                },
                                |node_ideal| coeffs.score(node_ideal, &ideal),
                                |solution: &T| coeffs.score(solution.objectives(), &ideal),
                            )
                        {
                            if seen_objectives.insert(*best.objectives()) {
                                selected.push(best.clone());
                            }
                        }

                        if let Some(parent_budget) = optimizations.scalarized_parent_budget
                            && selected.len() >= parent_budget
                        {
                            break;
                        }
                    }
                }

                if selected.is_empty() {
                    for weight in &weights {
                        let coeffs = WeightedChebycheffCoeffs::new(
                            weight,
                            &bounds,
                            optimizations.scalarized_rho,
                        );

                        let best = eligible_archive
                            .iter()
                            .copied()
                            .filter(|solution| !seen_objectives.contains(solution.objectives()))
                            .min_by(|a, b| {
                                let score_a = coeffs.score(a.objectives(), &ideal);
                                let score_b = coeffs.score(b.objectives(), &ideal);
                                score_a
                                    .partial_cmp(&score_b)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            });

                        if let Some(best) = best
                            && seen_objectives.insert(*best.objectives())
                        {
                            selected.push(best.clone());
                        }

                        if let Some(parent_budget) = optimizations.scalarized_parent_budget
                            && selected.len() >= parent_budget
                        {
                            break;
                        }
                    }
                }

                if selected.is_empty()
                    && let Some(first) = eligible_archive.first()
                {
                    selected.push((**first).clone());
                }

                if let Some(parent_budget) = optimizations.scalarized_parent_budget {
                    selected.truncate(parent_budget);
                }

                tracing::info!(
                    candidate_pool_size = eligible_archive.len(),
                    selected_parent_count = selected.len(),
                    weight_samples = weight_samples,
                    source = ?optimizations.scalarized_selection_source,
                    accelerated_archive_query = accelerated,
                    "Using scalarized parent selection for archive exploration"
                );

                selected
            }
        }
    }

    #[instrument(level = "debug", skip(self, step_time, step_stats), fields(
        iteration,
        pareto_set_size = self.approximated_pareto_set.len()
    ))]
    fn log_iteration_stats(&self, iteration: usize, step_time: Instant, step_stats: &StepStats) {
        let iteration_metrics = self.calculate_iteration_metrics(step_time, step_stats);

        // Log detailed metrics via tracing
        tracing::info!(
            iteration = iteration,
            duration_us = iteration_metrics.duration_us,
            per_solution_time_us = iteration_metrics.per_solution_search_time,
            neighborhood_size = iteration_metrics.neighborhood_size,
            neighbors_explored = step_stats.explored_neighbor_count,
            neighbors_duplicated = step_stats.duplicated_neighbor_count,
            duplicate_percentage = iteration_metrics.duplicated_percent,
            auxiliary_added = step_stats.auxiliary_added_count,
            auxiliary_removed = iteration_metrics.auxiliary_removed_count,
            pareto_added = step_stats.pareto_added_count,
            pareto_removed = iteration_metrics.pareto_removed_count,
            pareto_set_final_size = self.approximated_pareto_set.len(),
            "Iteration completed"
        );

        Self::log_metrics(
            iteration,
            &iteration_metrics,
            step_stats,
            self.approximated_pareto_set.len(),
        );
    }

    fn calculate_iteration_metrics(
        &self,
        step_time: Instant,
        step_stats: &StepStats,
    ) -> IterationMetrics {
        let auxiliary_removed_count = step_stats.auxiliary_added_count - step_stats.auxiliary_len;
        let pareto_removed_count = step_stats.pareto_initial_count + step_stats.pareto_added_count
            - self.approximated_pareto_set.len();
        let neighborhood_size = self.neigborhood_structure;
        let duration_us = step_time.elapsed().as_micros();
        let duplicated_percent = step_stats.duplicated_neighbor_count as f32
            / step_stats.explored_neighbor_count as f32
            * 100.0;
        let per_solution_search_time =
            duration_us as f32 / step_stats.explored_neighbor_count as f32;

        IterationMetrics {
            auxiliary_removed_count,
            pareto_removed_count,
            neighborhood_size,
            duration_us,
            duplicated_percent,
            per_solution_search_time,
        }
    }

    fn log_metrics(
        iteration: usize,
        metrics: &IterationMetrics,
        step_stats: &StepStats,
        pareto_final_size: usize,
    ) {
        info!(
            "Iteration {iteration} [{:.3} s, {} us/sol], neighbors: size: {}, explored: {}, duplicated: {} ({} %), auxiliary: +{}-{}={}, pareto: +{}-{}={}",
            metrics.duration_us as f64 / 1_000_000.0,
            metrics.per_solution_search_time,
            metrics.neighborhood_size,
            step_stats.explored_neighbor_count,
            step_stats.duplicated_neighbor_count,
            metrics.duplicated_percent,
            step_stats.auxiliary_added_count,
            metrics.auxiliary_removed_count,
            step_stats.auxiliary_len,
            step_stats.pareto_added_count,
            metrics.pareto_removed_count,
            pareto_final_size,
        );
    }

    fn log_auxiliary_population(auxiliary_population: &S) {
        debug!("===== Auxiliary population solutions: =====");
        for solution in auxiliary_population.iter() {
            debug!("{solution:?}");
        }
    }

    fn log_pareto_front(&self) {
        debug!("===== Pareto Front solutions: =====");
        for solution in self.approximated_pareto_set.iter() {
            debug!("{solution:?}");
        }
    }

    #[instrument(level = "debug", skip(self, auxiliary_population), fields(
        auxiliary_size = auxiliary_population.len(),
        current_neighborhood_structure = self.neigborhood_structure,
        max_neighborhood_structure = self.neighborhood_size_range.end()
    ))]
    fn determine_next_step(&mut self, auxiliary_population: S) -> StepStatus {
        if !auxiliary_population.is_empty() {
            tracing::debug!("New population found, replacing current population");
            self.replace_population_with_auxiliary(auxiliary_population);
            return StepStatus::NewPopulation;
        }

        // Perturbation restart: inject perturbed archive solutions before
        // resorting to expensive higher-k neighborhoods.
        if self.optimizations.use_perturbation_restart {
            let injected = self.inject_perturbed_archive_solutions(2);
            if injected > 0 {
                info!(
                    "Perturbation restart: injected {injected} perturbed solutions, restarting with k={}",
                    self.neighborhood_size_range.start()
                );
                tracing::info!(
                    injected_count = injected,
                    "Perturbation restart: restarting with smallest neighborhood structure"
                );
                self.neigborhood_structure = *self.neighborhood_size_range.start();
                return StepStatus::NewPopulation;
            }
        }

        if self.can_increase_neighborhood_structure() {
            info!("Increasing neighborhood structure.");
            tracing::info!(
                old_structure = self.neigborhood_structure,
                new_structure = self.neigborhood_structure + 1,
                "Increasing neighborhood structure"
            );
            self.neigborhood_structure += 1;
            self.add_eligible_pareto_solutions();
            StepStatus::IncreasedNeighborhoodStructure
        } else {
            info!("Reached maximum neighborhood structure.");
            tracing::debug!("Maximum neighborhood structure reached, algorithm will terminate");
            StepStatus::AllNeighborhoodStructuresExplored
        }
    }

    /// Inject perturbed copies of archive solutions into the current population.
    ///
    /// For each archive solution, creates `num_perturbations` random perturbations
    /// by removing one randomly selected image and greedily adding unselected
    /// images to restore full element coverage. Only solutions that form a valid
    /// set cover and are non-dominated by the archive are injected.
    ///
    /// Returns the number of solutions successfully injected.
    fn inject_perturbed_archive_solutions(&mut self, num_perturbations: usize) -> usize {
        let mut rng = if self.is_deterministic {
            SmallRng::seed_from_u64(
                (self.explored_solutions.num_iterations as u64).wrapping_mul(31337),
            )
        } else {
            SmallRng::from_rng(&mut rand::rng())
        };

        let num_images = self.problem.num_images();
        let num_elements = self.problem.num_elements();

        // Snapshot archive solutions so we can mutate self freely.
        let archive_solutions: Vec<T> = self.approximated_pareto_set.iter().cloned().collect();
        let mut injected_count: usize = 0;

        for solution in &archive_solutions {
            let selected: Vec<usize> = solution.selected_images().collect();
            if selected.is_empty() {
                continue;
            }

            for _ in 0..num_perturbations {
                // 1. Pick a random selected image to remove.
                let remove_idx = selected[rng.random_range(0..selected.len())];

                // 2. Clone and remove the chosen image.
                let mut perturbed = solution.clone();
                perturbed.set_image(remove_idx, false);

                // 3. Determine which elements are now uncovered.
                let mut uncovered = vec![false; num_elements];
                let mut num_uncovered: usize = 0;
                for elem in self.problem.uncovered_elements(perturbed.selected_images()) {
                    uncovered[elem] = true;
                    num_uncovered += 1;
                }

                // 4. Greedily add unselected images until coverage is restored.
                //    Each step picks the image covering the most uncovered elements.
                while num_uncovered > 0 {
                    let mut best_image: Option<usize> = None;
                    let mut best_coverage: usize = 0;

                    for img in 0..num_images {
                        if perturbed.is_image_selected(img) {
                            continue;
                        }
                        let coverage = self
                            .problem
                            .image_elements(img)
                            .filter(|&elem| uncovered[elem])
                            .count();
                        if coverage > best_coverage {
                            best_coverage = coverage;
                            best_image = Some(img);
                        }
                    }

                    if let Some(img) = best_image {
                        perturbed.set_image(img, true);
                        for elem in self.problem.image_elements(img) {
                            if uncovered[elem] {
                                uncovered[elem] = false;
                                num_uncovered -= 1;
                            }
                        }
                    } else {
                        // No image can cover remaining elements (should not happen
                        // with valid problem instances).
                        break;
                    }
                }

                // 5. Recompute objective values from scratch.
                perturbed.recalculate_objectives(self.problem);

                // 6. Skip invalid solutions (coverage gap).
                if !self.problem.is_set_cover(&perturbed) {
                    continue;
                }

                // 7. Skip solutions that were already explored.
                if self.explored_solutions.is_registered(&perturbed) {
                    continue;
                }

                // 8. Register in explored solutions so it won't be re-explored.
                self.explored_solutions.register(
                    self.explored_solutions.num_iterations,
                    &perturbed,
                    Duration::from_secs(0),
                    perturbed.selected_images().collect(),
                );

                // 9. Only inject if non-dominated by the current archive.
                if self.approximated_pareto_set.try_insert(&perturbed) {
                    self.population.insert_unchecked(&perturbed);
                    injected_count += 1;
                }
            }
        }

        tracing::debug!(
            archive_size = archive_solutions.len(),
            num_perturbations,
            injected_count,
            "Perturbation restart completed"
        );
        injected_count
    }

    #[instrument(level = "debug", skip(self, auxiliary_population), fields(
        old_population_size = self.population.len(),
        auxiliary_size = auxiliary_population.len(),
        old_neighborhood_structure = self.neigborhood_structure
    ))]
    fn replace_population_with_auxiliary(&mut self, auxiliary_population: S) {
        info!("Replacing current population with auxiliary population.");
        self.population = auxiliary_population.with_name("population");
        info!("Start again with smallest neighborhood structure.");
        self.neigborhood_structure = *self.neighborhood_size_range.start();
    }

    fn can_increase_neighborhood_structure(&self) -> bool {
        let next_structure = self.neigborhood_structure + 1;
        self.neighborhood_size_range.contains(&next_structure)
    }

    #[instrument(level = "debug", skip(self), fields(
        current_pareto_size = self.approximated_pareto_set.len(),
        current_population_size = self.population.len(),
        neighborhood_structure = self.neigborhood_structure
    ))]
    fn add_eligible_pareto_solutions(&mut self) {
        info!(
            "Use solutions from approximated pareto set which are not already Pareto local optimum"
        );
        let eligible_solutions = self.approximated_pareto_set.iter().filter(|&solution| {
            self.explored_solutions.explored_neighborhood_size(solution)
                < self.neigborhood_structure
        });

        for solution in eligible_solutions {
            self.population.insert_unchecked(solution);
        }
    }

    /// Returns a reference to the approximated Pareto set (concurrent worker use).
    #[cfg(feature = "parallel")]
    pub(crate) fn archive(&self) -> &S {
        &self.approximated_pareto_set
    }

    /// Returns a mutable reference to the archive (concurrent worker use).
    #[cfg(feature = "parallel")]
    pub(crate) fn archive_mut(&mut self) -> &mut S {
        &mut self.approximated_pareto_set
    }

    /// Returns a reference to the explored solutions data (concurrent worker use).
    #[cfg(feature = "parallel")]
    pub(crate) fn explored_solutions_data(&self) -> &ExploredSolutionsData<D> {
        &self.explored_solutions
    }

    /// Try to adopt a globally non-dominated solution into archive and population.
    /// Also registers the solution in explored_solutions (with placeholder iteration/time) so
    /// the PLS internals never see an unregistered solution.
    /// Returns `true` if the solution was inserted into the archive.
    #[cfg(feature = "parallel")]
    pub(crate) fn try_adopt_solution(&mut self, solution: &T) -> bool {
        if self.approximated_pareto_set.try_insert(solution) {
            // Register in explored_solutions so add_eligible_pareto_solutions won't panic.
            // Use register_without_selected_images with iteration=0 and time=0 as placeholders.
            if !self.explored_solutions.is_registered(solution) {
                self.explored_solutions.register_without_selected_images(
                    0,
                    solution,
                    std::time::Duration::from_secs(0),
                );
            }
            self.population.insert_unchecked(solution);
            true
        } else {
            false
        }
    }

    /// SA-PLS step: sorts population by augmented Tchebycheff score, then explores neighborhoods
    /// with decoupled archive/auxiliary gates.
    ///
    /// - **Gate 1 (archive)**: `archive.try_insert(n)` -- global Pareto non-dominance (unchanged).
    /// - **Gate 2 (auxiliary)**: `g_atch(n, w, z*) < g_atch(parent, w, z*)` -- scalarized
    ///   improvement over the parent, independent of archive insertion.
    /// - **Gate 3 (ideal)**: updates `local_ideal` component-wise for every explored neighbor.
    ///
    /// Returns `SAPlsStatus` indicating whether a new population was generated, k was increased,
    /// or scalarized search is exhausted.
    #[cfg(feature = "parallel")]
    pub(crate) fn step_scalarized(
        &mut self,
        iteration: usize,
        timer: &Timer,
        weight: &[f64; D],
        local_ideal: &mut pareto::Objectives<D>,
        objective_bounds: &[(f64, f64); D],
        rho: f64,
    ) -> (SAPlsStatus, StepStats) {
        use crate::concurrent_pls::decomposition::TchebycheffCoeffs;

        let step_time = Instant::now();
        let mut step_stats = StepStats::new(self.approximated_pareto_set.len());

        tracing::debug!("Starting SA-PLS step");

        self.population.validate();

        let population = mem::replace(&mut self.population, S::new("population"));

        // Precompute Tchebycheff coefficients (weight/range ratios) once per step
        // to avoid repeated divisions in the inner loop. Recomputed each step
        // because local_ideal changes across steps.
        let coeffs = TchebycheffCoeffs::new(weight, objective_bounds, rho);

        // Sort population by g_atch ascending (best scalarized score first) -- D15
        let mut population_vec: Vec<T> = population.into_iter().collect();
        population_vec.sort_by(|a, b| {
            let score_a = coeffs.score(a.objectives(), local_ideal);
            let score_b = coeffs.score(b.objectives(), local_ideal);
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Scalarized auxiliary candidates (independent from archive -- section 16.6)
        let mut aux_candidates: Vec<T> = Vec::new();

        let spare_tracker = &mut self.spare_tracker;
        let explored_solutions = &mut self.explored_solutions;
        let approximated_pareto_set = &mut self.approximated_pareto_set;
        let problem = self.problem;
        let neighborhood_structure = self.neigborhood_structure;
        let is_deterministic = self.is_deterministic;
        let optimizations = &self.optimizations;

        'population: for (_index, solution) in population_vec.iter().enumerate() {
            let parent_score = coeffs.score(solution.objectives(), local_ideal);

            let mut neighborhood_iter = solution.neighborhood_iter(
                spare_tracker,
                neighborhood_structure,
                problem,
                timer,
                is_deterministic,
                optimizations,
            );

            for (_neighbor_index, neighbor) in neighborhood_iter.by_ref().enumerate() {
                step_stats.explored_neighbor_count += 1;

                // Dedup check (unchanged)
                if explored_solutions.is_registered(&neighbor) {
                    step_stats.duplicated_neighbor_count += 1;
                    continue;
                }

                explored_solutions.register_without_selected_images(
                    iteration,
                    &neighbor,
                    timer.elapsed(),
                );

                // Gate 3: update local ideal for every explored neighbor (section 16.4)
                for j in 0..D {
                    let fj = neighbor.objectives()[j];
                    if fj < local_ideal[j] {
                        local_ideal[j] = fj;
                    }
                }

                // Skip if dominated by parent (unchanged from standard PLS)
                if neighbor.is_covered_by(solution.objectives()) {
                    continue;
                }

                // Gate 1: global archive insertion (unchanged, always run)
                if approximated_pareto_set.try_insert(&neighbor) {
                    step_stats.pareto_added_count += 1;
                }

                // Gate 2: scalarized auxiliary (independent of Gate 1 -- section 16.3)
                let neighbor_score = coeffs.score(neighbor.objectives(), local_ideal);
                if neighbor_score < parent_score {
                    aux_candidates.push(neighbor);
                    step_stats.auxiliary_added_count += 1;
                }

                if timer.is_expired() {
                    tracing::debug!("Timer expired during SA-PLS neighbor processing");
                    break 'population;
                }
            }

            explored_solutions.update_explored_neighborhood_size(solution, neighborhood_structure);
        }

        step_stats.auxiliary_len = aux_candidates.len();

        tracing::debug!(
            explored_neighbors = step_stats.explored_neighbor_count,
            duplicated_neighbors = step_stats.duplicated_neighbor_count,
            aux_candidates = step_stats.auxiliary_len,
            "SA-PLS step completed"
        );

        self.log_iteration_stats(iteration, step_time, &step_stats);

        // Determine next step using scalarized semantics (section 16.7)
        let status = if !aux_candidates.is_empty() {
            // Dominance-filter to bound auxiliary size (R9 mitigation).
            // NdTreeSolutionSet::from_iter inserts all candidates via try_insert,
            // retaining only non-dominated solutions. No pre-sort needed: the tree
            // does not preserve insertion order, and the population will be re-sorted
            // by g_atch at the start of the next step (D15).
            self.population = S::from_iter(aux_candidates).with_name("population");
            self.neigborhood_structure = *self.neighborhood_size_range.start();
            SAPlsStatus::NewPopulation
        } else if self.can_increase_neighborhood_structure() {
            tracing::info!(
                old_structure = self.neigborhood_structure,
                new_structure = self.neigborhood_structure + 1,
                "SA-PLS: increasing neighborhood structure"
            );
            self.neigborhood_structure += 1;
            self.add_eligible_pareto_solutions();
            SAPlsStatus::IncreasedNeighborhoodStructure
        } else {
            tracing::info!("SA-PLS: scalarized search exhausted");
            SAPlsStatus::ScalarizedExhausted
        };

        (status, step_stats)
    }

    /// Re-seed the population from the archive for fallback mode after scalarized exhaustion.
    /// Uses the same k-increase mechanism as standard PLS.
    #[cfg(feature = "parallel")]
    pub(crate) fn reseed_population_from_archive(&mut self) {
        self.neigborhood_structure = *self.neighborhood_size_range.start();
        self.population = self.approximated_pareto_set.clone().with_name("population");
    }

    #[instrument(level = "info", skip(self), fields(
        max_iterations,
        max_duration_ms = max_duration.as_millis(),
        population_size = self.population.len(),
        pareto_set_size = self.approximated_pareto_set.len()
    ))]
    pub fn run(&mut self, max_iterations: usize, max_duration: Duration) -> S {
        let pls_timer = Timer::start(max_duration);

        info!(
            "PLS using tracker type: {}",
            std::any::type_name::<T::Trackers>()
        );
        info!(
            "Initial population after dominance filtering: {} solutions",
            self.population.len()
        );

        for i in 1..=max_iterations {
            let iteration_span = info_span!(
                "pls_iteration",
                iteration = i,
                population_size = self.population.len(),
                neighborhood_structure = self.neigborhood_structure
            );
            let _iteration_guard = iteration_span.enter();

            debug!("******************************************************");
            debug!(
                "******** ITERATION {} POPULATION SIZE {} ********",
                i,
                self.population.len()
            );
            debug!("******************************************************");

            let step_result = self.step(i, &pls_timer);
            self.explored_solutions.num_iterations = i;

            if step_result.status == StepStatus::AllNeighborhoodStructuresExplored {
                info!(
                    "All neighborhood structures were explored. Number of iterations: {i}. Elapsed time: [{:?} ms]",
                    pls_timer.elapsed()
                );
                break;
            }
            if pls_timer.elapsed() > max_duration {
                info!(
                    "Maximum duration reached. Number of iterations: {i}. Elapsed time [{:?}]",
                    pls_timer.elapsed()
                );
                break;
            }
            if i == max_iterations {
                info!(
                    "Maximum iterations reached. Elaped time: [{:?}]",
                    pls_timer.elapsed()
                );
            }
        }

        tracing::info!(
            pareto_set_final_size = self.approximated_pareto_set.len(),
            explored_solutions_total = self.explored_solutions.solutions.len(),
            "PLS algorithm completed"
        );

        // Validate final approximated Pareto set for dominated solutions
        self.approximated_pareto_set.validate();

        // Print all solutions' objectives in lexicographically sorted order
        self.print_sorted_objectives();

        self.approximated_pareto_set.clone()
    }

    fn print_sorted_objectives(&self) {
        let mut objectives: Vec<_> = self
            .approximated_pareto_set
            .iter()
            .map(T::objectives)
            .collect();

        // Sort objectives lexicographically
        objectives.sort();

        debug!(
            "===== Final Pareto Set Objectives - {} (lexicographically sorted) =====",
            self.problem.instance_name()
        );
        for (i, obj) in objectives.iter().enumerate() {
            debug!("Solution {}: {:?}", i + 1, obj);
        }
        debug!("===== Total solutions: {} =====", objectives.len());
    }

    pub fn explored_solutions_fingerprints(&self) -> Vec<SolutionFingerprint<D>> {
        self.explored_solutions
            .solutions
            .values()
            .cloned()
            .collect()
    }
}
