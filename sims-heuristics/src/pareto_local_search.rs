use std::{
    mem,
    ops::RangeInclusive,
    time::{Duration, Instant},
};

use log::{debug, error, info};
use rand::prelude::*;
use rand::SeedableRng;
use pareto::ParetoFront;
use tracing::{debug_span, info_span, instrument};

use crate::{
    explored_solutions_data::{ExploredSolutionsData, SolutionFingerprint},
    objective_tracker::TrackerCollection,
    problem::SetCoverProblem,
    solution::{EncodedSolution, ImageSet},
    timer::Timer,
};

/// Statistics tracking during a single step of the algorithm
struct StepStats {
    explored_neighbor_count: usize,
    duplicated_neighbor_count: usize,
    auxiliary_added_count: usize,
    auxiliary_len: usize,
    pareto_added_count: usize,
    pareto_initial_count: usize,
}

impl StepStats {
    const fn new(pareto_initial_count: usize) -> Self {
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

impl<'a, T, S, P, const D: usize> ParetoLocalSearch<'a, T, S, P, D>
where
    T: ImageSet<D> + EncodedSolution<P, D>,
    S: ParetoFront<'a, T> + Clone + FromIterator<T> + IntoIterator<Item = T>,
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
    ) -> Self {
        let mut population = S::new("population");
        // Initialize ExploredSolutionsData with the problem's max objectives array
        let mut explored_solutions = ExploredSolutionsData::<D>::new(problem.max_objectives());
        initial_population.iter().for_each(|solution| {
            if population.try_insert(solution) {
                explored_solutions.register(
                    0,
                    solution,
                    Duration::from_secs(0),
                    solution.selected_images().collect(),
                );
            }
        });
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
            _phantom: std::marker::PhantomData,
        }
    }

    #[instrument(level = "debug", skip(self, timer), fields(
        iteration,
        population_size = self.population.len(),
        neighborhood_structure = self.neigborhood_structure
    ))]
    fn step(&mut self, iteration: usize, timer: &Timer) -> StepStatus {
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

        self.determine_next_step(auxiliary_population)
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

        // Shuffle population to vary exploration order each iteration
        let mut population_vec: Vec<_> = population.into_iter().collect();
        if is_deterministic {
            // Seeded RNG for reproducible but varied order per iteration
            let mut rng = SmallRng::seed_from_u64(iteration as u64);
            population_vec.shuffle(&mut rng);
        } else {
            population_vec.shuffle(&mut rand::rng());
        }

        'population: for (index, solution) in population_vec.into_iter().enumerate() {
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

        Self::log_metrics(iteration, &iteration_metrics, step_stats);
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

    fn log_metrics(iteration: usize, metrics: &IterationMetrics, step_stats: &StepStats) {
        error!(
            "Iteration {iteration} [{:.3} s, {} us/sol], neighbors: size: {}, explored: {}, duplicated: {} ({} %), auxiliary: +{}-{}, pareto: +{}-{}",
            metrics.duration_us as f64 / 1_000_000.0,
            metrics.per_solution_search_time,
            metrics.neighborhood_size,
            step_stats.explored_neighbor_count,
            step_stats.duplicated_neighbor_count,
            metrics.duplicated_percent,
            step_stats.auxiliary_added_count,
            metrics.auxiliary_removed_count,
            step_stats.pareto_added_count,
            metrics.pareto_removed_count,
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

    #[instrument(level = "info", skip(self), fields(
        max_iterations,
        max_duration_ms = max_duration.as_millis(),
        population_size = self.population.len(),
        pareto_set_size = self.approximated_pareto_set.len()
    ))]
    pub fn run(&mut self, max_iterations: usize, max_duration: Duration) -> S {
        let pls_timer = Timer::start(max_duration);

        error!(
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

            let step_status = self.step(i, &pls_timer);
            self.explored_solutions.num_iterations = i;

            if step_status == StepStatus::AllNeighborhoodStructuresExplored {
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

        error!(
            "===== Final Pareto Set Objectives - {} (lexicographically sorted) =====",
            self.problem.instance_name()
        );
        for (i, obj) in objectives.iter().enumerate() {
            error!("Solution {}: {:?}", i + 1, obj);
        }
        error!("===== Total solutions: {} =====", objectives.len());
    }

    pub fn explored_solutions_fingerprints(&self) -> Vec<SolutionFingerprint<D>> {
        self.explored_solutions
            .solutions
            .values()
            .cloned()
            .collect()
    }
}
