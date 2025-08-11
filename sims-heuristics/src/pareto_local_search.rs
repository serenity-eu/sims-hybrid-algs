use std::{
    mem,
    ops::RangeInclusive,
    time::{Duration, Instant},
};

use log::{debug, error, info};
use pareto::ParetoFront;
use tracing::{debug_span, info_span, instrument};

use crate::{
    explored_solutions_data::ExploredSolutionsData,
    problem::Problem,
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

pub struct ParetoLocalSearch<'a, T, S, const D: usize>
where
    T: ImageSet<D> + EncodedSolution<D>,
    S: ParetoFront<'a, T> + Clone,
{
    /// Reference to problem instance
    problem: &'a Problem<T, D>,
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

impl<'a, T, S, const D: usize> ParetoLocalSearch<'a, T, S, D>
where
    T: ImageSet<D> + EncodedSolution<D>,
    S: ParetoFront<'a, T> + Clone + FromIterator<T> + IntoIterator<Item = T>,
{
    pub fn new(
        problem: &'a Problem<T, D>,
        initial_population: &S,
        neighborhood_size_range: RangeInclusive<u32>,
        is_deterministic: bool,
    ) -> Self {
        let mut population = S::new("population");
        // Initialize ExploredSolutionsData with the problem's max objectives array
        let mut explored_solutions = ExploredSolutionsData::<D>::new(problem.max_objectives);
        initial_population.iter().for_each(|solution| {
            if population.try_insert(solution) {
                explored_solutions.register(0, solution, Duration::from_secs(0));
            }
        });
        for (i, solution) in population.iter().enumerate() {
            debug!("Initial solution {i}: {solution:?}");
        }

        let approximated_pareto_set = population.clone().with_name("approximated Pareto set");
        ParetoLocalSearch {
            problem,
            population,
            approximated_pareto_set,
            neigborhood_structure: *neighborhood_size_range.start(),
            neighborhood_size_range,
            explored_solutions,
            is_deterministic,
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

        let neighborhood_exploration_span = info_span!(
            "explore_neighborhoods",
            initial_population_size = population.len()
        );
        let exploration_guard = neighborhood_exploration_span.enter();

        self.explore_population_neighborhoods(
            population,
            iteration,
            timer,
            &mut step_stats,
            &mut auxiliary_population,
        );
        drop(exploration_guard);

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
        'population: for (index, solution) in population.into_iter().enumerate() {
            let solution_span = debug_span!(
                "explore_solution",
                solution_index = index,
                neighborhood_structure = self.neigborhood_structure
            );
            let _solution_guard = solution_span.enter();

            let neighbors = solution.neighborhood(
                self.neigborhood_structure,
                self.problem,
                timer,
                self.is_deterministic,
            );
            let neighbor_count = neighbors.len();

            tracing::debug!(
                neighbors_generated = neighbor_count,
                "Generated neighborhood for solution"
            );

            let neighbor_evaluation_span =
                debug_span!("evaluate_neighbors", neighbor_count = neighbor_count);
            let _evaluation_guard = neighbor_evaluation_span.enter();

            for (neighbor_index, neighbor) in neighbors.into_iter().enumerate() {
                if self.process_neighbor(
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

            self.explored_solutions
                .update_explored_neighborhood_size(&solution, self.neigborhood_structure);
        }

        tracing::debug!(
            total_neighbors_explored = step_stats.explored_neighbor_count,
            duplicates_found = step_stats.duplicated_neighbor_count,
            "Population neighborhood exploration completed"
        );
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Necessary for logical separation of concerns"
    )]
    #[instrument(
        level = "trace",
        skip(
            self,
            neighbor,
            current_solution,
            timer,
            step_stats,
            auxiliary_population
        ),
        fields(neighbor_index, iteration)
    )]
    fn process_neighbor(
        &mut self,
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
        debug_assert!(neighbor.is_valid(self.problem));

        if self.explored_solutions.is_registered(neighbor) {
            debug!("Neighbor nr {neighbor_index} was already explored.");
            step_stats.duplicated_neighbor_count += 1;
            tracing::trace!("Neighbor already explored, skipping");
            return false;
        }

        tracing::trace!("Evaluating new neighbor");

        let evaluation_span = debug_span!("evaluate_neighbor");
        let _eval_guard = evaluation_span.enter();

        self.evaluate_neighbor(
            neighbor,
            neighbor_index,
            current_solution,
            iteration,
            timer,
            step_stats,
            auxiliary_population,
        );

        if timer.is_expired() {
            info!("Timer expired. Stop exploring neighbors.");
            tracing::warn!("Timer expired during neighbor processing");
            return true;
        }

        false
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Necessary for logical separation of concerns"
    )]
    #[instrument(
        level = "trace",
        skip(
            self,
            neighbor,
            current_solution,
            timer,
            step_stats,
            auxiliary_population
        ),
        fields(neighbor_index, iteration)
    )]
    fn evaluate_neighbor(
        &mut self,
        neighbor: &T,
        neighbor_index: usize,
        current_solution: &T,
        iteration: usize,
        timer: &Timer,
        step_stats: &mut StepStats,
        auxiliary_population: &mut S,
    ) {
        if neighbor.is_covered_by(current_solution.objectives()) {
            debug!("Neighbor nr {neighbor_index} is weakly dominated by current solution.");
            tracing::trace!("Neighbor dominated by current solution, discarding");
            return;
        }

        let pareto_evaluation_span = debug_span!("pareto_evaluation");
        let pareto_guard = pareto_evaluation_span.enter();

        if self.try_add_to_pareto_set(neighbor, neighbor_index, iteration, timer, step_stats) {
            drop(pareto_guard);

            let auxiliary_evaluation_span = debug_span!("auxiliary_evaluation");
            let _aux_guard = auxiliary_evaluation_span.enter();

            Self::try_add_to_auxiliary_population(
                neighbor,
                neighbor_index,
                step_stats,
                auxiliary_population,
            );

            tracing::trace!("Neighbor added to Pareto set and auxiliary population evaluated");
        } else {
            tracing::trace!("Neighbor not added to Pareto set");
        }
    }

    #[instrument(level = "trace", skip(self, neighbor, timer, step_stats), fields(
        neighbor_index,
        iteration,
        pareto_set_size = self.approximated_pareto_set.len()
    ))]
    fn try_add_to_pareto_set(
        &mut self,
        neighbor: &T,
        neighbor_index: usize,
        iteration: usize,
        timer: &Timer,
        step_stats: &mut StepStats,
    ) -> bool {
        if self.approximated_pareto_set.try_insert(neighbor) {
            self.log_pareto_set_addition(neighbor_index);
            self.explored_solutions
                .register(iteration, neighbor, timer.elapsed());
            step_stats.pareto_added_count += 1;

            tracing::trace!(
                new_pareto_set_size = self.approximated_pareto_set.len(),
                "Neighbor successfully added to Pareto set"
            );
            true
        } else {
            debug!("Neighbor nr {neighbor_index} wasn't added to approximated pareto set.");
            tracing::trace!("Neighbor rejected from Pareto set");
            false
        }
    }

    fn log_pareto_set_addition(&self, neighbor_index: usize) {
        debug!(
            "Neighbor nr {neighbor_index} was added to approximated pareto set. Approximated pareto set size: {}",
            self.approximated_pareto_set.len()
        );
    }

    fn try_add_to_auxiliary_population(
        neighbor: &T,
        neighbor_index: usize,
        step_stats: &mut StepStats,
        auxiliary_population: &mut S,
    ) {
        if auxiliary_population.try_insert(neighbor) {
            step_stats.auxiliary_added_count += 1;
        } else {
            debug!(
                "Neighbor nr {neighbor_index} is dominated so it wasn't added to auxiliary population"
            );
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
            "Iteration {iteration} [{} us, {} us/sol], neighbors: size: {}, explored: {}, duplicated: {} ({} %), auxiliary: +{}-{}, pareto: +{}-{}",
            metrics.duration_us,
            metrics.per_solution_search_time,
            metrics.neighborhood_size,
            step_stats.explored_neighbor_count,
            step_stats.duplicated_neighbor_count,
            metrics.duplicated_percent,
            step_stats.auxiliary_added_count,
            metrics.auxiliary_removed_count,
            step_stats.pareto_added_count,
            metrics.pareto_removed_count
        );
    }

    fn log_auxiliary_population(auxiliary_population: &S) {
        debug!("===== Auxiliary population solutions: =====");
        for solution in auxiliary_population.iter() {
            info!("{solution:?}");
        }
    }

    fn log_pareto_front(&self) {
        info!("===== Pareto Front solutions: =====");
        for solution in self.approximated_pareto_set.iter() {
            info!("{solution:?}");
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
            tracing::debug!(
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
        let start_time = Instant::now();

        info!("Replacing current population with auxiliary population.");
        self.population = auxiliary_population.with_name("population");
        info!("Start again with smallest neighborhood structure.");
        self.neigborhood_structure = *self.neighborhood_size_range.start();

        let elapsed = start_time.elapsed();
        tracing::debug!(
            new_population_size = self.population.len(),
            new_neighborhood_structure = self.neigborhood_structure,
            duration_us = elapsed.as_micros(),
            "Population replaced with auxiliary population"
        );
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
        let start_time = Instant::now();

        info!(
            "Use solutions from approximated pareto set which are not already Pareto local optimum"
        );
        let eligible_solutions: Vec<_> = self
            .approximated_pareto_set
            .iter()
            .filter(|&solution| {
                self.explored_solutions.explored_neighborhood_size(solution)
                    < self.neigborhood_structure
            })
            .collect();

        let eligible_count = eligible_solutions.len();
        tracing::debug!(
            eligible_solutions_count = eligible_count,
            "Found eligible Pareto solutions"
        );

        for solution in eligible_solutions {
            self.population.insert_unchecked(solution);
        }

        let elapsed = start_time.elapsed();
        tracing::debug!(
            new_population_size = self.population.len(),
            eligible_added = eligible_count,
            duration_us = elapsed.as_micros(),
            "Eligible Pareto solutions added to population"
        );
    }

    #[instrument(level = "info", skip(self), fields(
        max_iterations,
        max_duration_ms = max_duration.as_millis(),
        population_size = self.population.len(),
        pareto_set_size = self.approximated_pareto_set.len()
    ))]
    pub fn run(&mut self, max_iterations: usize, max_duration: Duration) -> S {
        let pls_timer = Timer::start(max_duration);

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
            self.problem.instance_name
        );
        for (i, obj) in objectives.iter().enumerate() {
            error!("Solution {}: {:?}", i + 1, obj);
        }
        error!("===== Total solutions: {} =====", objectives.len());
    }
}
