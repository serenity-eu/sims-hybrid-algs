use std::{
    mem,
    ops::RangeInclusive,
    time::{Duration, Instant},
};

use log::{debug, error, info};

use crate::{
    explored_solutions_data::ExploredSolutionsData,
    problem::Problem,
    solution::{VecEncodedSolution, SIMSSolutionTrait},
    solution_set::SolutionSet,
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

pub struct ParetoLocalSearch<'a, S, const D: usize>
where
    S: SolutionSet<'a, VecEncodedSolution<D>, D> + Clone,
{
    /// Reference to problem instance
    problem: &'a Problem<D>,
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

impl<'a, S, const D: usize> ParetoLocalSearch<'a, S, D>
where
    S: SolutionSet<'a, VecEncodedSolution<D>, D> + Clone,
{
    pub fn new(
        problem: &'a Problem<D>,
        initial_population: &S,
        neighborhood_size_range: RangeInclusive<u32>,
        is_deterministic: bool,
    ) -> Self {
        let mut population = S::new("population".to_string());
        // TODO: Hardcoded for 2 objectives, should be generalized
        let mut explored_solutions =
            ExploredSolutionsData::<D>::new(problem.max_objectives[0], problem.max_objectives[1]);
        initial_population.iter().for_each(|solution| {
            if population.try_add(solution) {
                explored_solutions.register(0, solution, Duration::from_secs(0));
            }
        });
        for (i, solution) in population.iter().enumerate() {
            debug!("Initial solution {i}: {solution:?}");
        }

        let approximated_pareto_set = population
            .clone()
            .with_name("approximated Pareto set".to_string());
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

    fn step(&mut self, iteration: usize, timer: &Timer) -> StepStatus {
        let step_time = Instant::now();
        let mut step_stats = StepStats::new(self.approximated_pareto_set.len());
        let mut auxiliary_population = S::new("auxiliary".to_string());

        self.validate_population();
        let population = mem::replace(&mut self.population, S::new("population".to_string()));

        self.explore_population_neighborhoods(
            population,
            iteration,
            timer,
            &mut step_stats,
            &mut auxiliary_population,
        );

        step_stats.auxiliary_len = auxiliary_population.len();

        self.log_iteration_stats(iteration, step_time, &step_stats);
        Self::log_auxiliary_population(&auxiliary_population);
        self.log_pareto_front();

        self.determine_next_step(auxiliary_population)
    }

    fn validate_population(&self) {
        for solution in self.population.iter() {
            debug_assert!(solution.is_valid(self.problem));
        }
    }

    fn explore_population_neighborhoods(
        &mut self,
        population: S,
        iteration: usize,
        timer: &Timer,
        step_stats: &mut StepStats,
        auxiliary_population: &mut S,
    ) {
        'population: for (index, solution) in population.into_iter().enumerate() {
            Self::log_solution_debug(index, &solution);

            let neighbors = solution.neighborhood(
                self.neigborhood_structure,
                self.problem,
                timer,
                self.is_deterministic,
            );

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
                    break 'population;
                }
            }

            self.explored_solutions
                .update_explored_neighborhood_size(&solution, self.neigborhood_structure);
        }
    }

    fn log_solution_debug(index: usize, solution: &VecEncodedSolution<D>) {
        debug!("######################################################");
        debug!("######## SOLUTION {} ########", index + 1);
        debug!("######## {solution:?} ########");
        debug!("######################################################");
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Necessary for logical separation of concerns"
    )]
    fn process_neighbor(
        &mut self,
        neighbor: &VecEncodedSolution<D>,
        neighbor_index: usize,
        current_solution: &VecEncodedSolution<D>,
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
            return false;
        }

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
            return true;
        }

        false
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Necessary for logical separation of concerns"
    )]
    fn evaluate_neighbor(
        &mut self,
        neighbor: &VecEncodedSolution<D>,
        neighbor_index: usize,
        current_solution: &VecEncodedSolution<D>,
        iteration: usize,
        timer: &Timer,
        step_stats: &mut StepStats,
        auxiliary_population: &mut S,
    ) {
        if neighbor.is_weakly_dominated(current_solution) {
            debug!("Neighbor nr {neighbor_index} is weakly dominated by current solution.");
            return;
        }

        if self.try_add_to_pareto_set(neighbor, neighbor_index, iteration, timer, step_stats) {
            Self::try_add_to_auxiliary_population(
                neighbor,
                neighbor_index,
                step_stats,
                auxiliary_population,
            );
        }
    }

    fn try_add_to_pareto_set(
        &mut self,
        neighbor: &VecEncodedSolution<D>,
        neighbor_index: usize,
        iteration: usize,
        timer: &Timer,
        step_stats: &mut StepStats,
    ) -> bool {
        if self.approximated_pareto_set.try_add(neighbor) {
            self.log_pareto_set_addition(neighbor_index);
            self.explored_solutions
                .register(iteration, neighbor, timer.elapsed());
            step_stats.pareto_added_count += 1;
            true
        } else {
            debug!("Neighbor nr {neighbor_index} wasn't added to approximated pareto set.");
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
        neighbor: &VecEncodedSolution<D>,
        neighbor_index: usize,
        step_stats: &mut StepStats,
        auxiliary_population: &mut S,
    ) {
        if auxiliary_population.try_add(neighbor) {
            step_stats.auxiliary_added_count += 1;
        } else {
            debug!(
                "Neighbor nr {neighbor_index} is dominated so it wasn't added to auxiliary population"
            );
        }
    }

    fn log_iteration_stats(&self, iteration: usize, step_time: Instant, step_stats: &StepStats) {
        let iteration_metrics = self.calculate_iteration_metrics(step_time, step_stats);
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

    fn determine_next_step(&mut self, auxiliary_population: S) -> StepStatus {
        if !auxiliary_population.is_empty() {
            self.replace_population_with_auxiliary(auxiliary_population);
            return StepStatus::NewPopulation;
        }

        if self.can_increase_neighborhood_structure() {
            info!("Increasing neighborhood structure.");
            self.neigborhood_structure += 1;
            self.add_eligible_pareto_solutions();
            StepStatus::IncreasedNeighborhoodStructure
        } else {
            info!("Reached maximum neighborhood structure.");
            StepStatus::AllNeighborhoodStructuresExplored
        }
    }

    fn replace_population_with_auxiliary(&mut self, auxiliary_population: S) {
        info!("Replacing current population with auxiliary population.");
        self.population = auxiliary_population.with_name("population".to_string());
        info!("Start again with smallest neighborhood structure.");
        self.neigborhood_structure = *self.neighborhood_size_range.start();
    }

    fn can_increase_neighborhood_structure(&self) -> bool {
        let next_structure = self.neigborhood_structure + 1;
        self.neighborhood_size_range.contains(&next_structure)
    }

    fn add_eligible_pareto_solutions(&mut self) {
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

        for solution in eligible_solutions {
            self.population.force_add(solution);
        }
    }

    pub fn run(&mut self, max_iterations: usize, max_duration: Duration) -> S {
        let pls_timer = Timer::start(max_duration);
        for i in 1..=max_iterations {
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
        self.approximated_pareto_set.clone()
    }
}
