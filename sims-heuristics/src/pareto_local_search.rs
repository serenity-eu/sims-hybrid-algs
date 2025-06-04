use std::{
    mem::replace,
    ops::RangeInclusive,
    time::{Duration, Instant},
};

use log::{debug, error, info};

use crate::{
    explored_solutions_data::ExploredSolutionsData,
    problem::Problem,
    solution::{EncodedSolution, MOSolution},
    solution_set::SolutionSet,
    timer::Timer,
};

pub struct ParetoLocalSearch<'a, S>
where
    S: SolutionSet<'a, EncodedSolution> + Clone,
{
    /// Reference to problem instance
    problem: &'a Problem,
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
    pub explored_solutions: ExploredSolutionsData,
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

impl<'a, S> ParetoLocalSearch<'a, S>
where
    S: SolutionSet<'a, EncodedSolution> + Clone,
{
    pub fn new(
        problem: &'a Problem,
        initial_population: S,
        neighborhood_size_range: RangeInclusive<u32>,
        is_deterministic: bool,
    ) -> Self {
        let mut population = S::new("population".to_string());
        let mut explored_solutions =
            ExploredSolutionsData::new(problem.max_objectives.0, problem.max_objectives.1);
        initial_population.iter().for_each(|solution| {
            if population.try_add(solution) {
                explored_solutions.register(0, solution, Duration::from_secs(0));
            }
        });
        for (i, solution) in population.iter().enumerate() {
            debug!("Initial solution {}: {:?}", i, solution);
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
        let mut explored_neighbor_count: usize = 0;
        let mut duplicated_neighbor_count: usize = 0;
        let mut auxiliary_added_count: usize = 0;
        let mut pareto_added_count: usize = 0;
        let pareto_initial_count = self.approximated_pareto_set.len();
        // let pareto_snapshot_interval = timer.duration() / 10;
        // let mut pareto_snapshot_time = Instant::now();

        let mut auxiliary_population = S::new("auxiliary".to_string());
        // for solution in self.population.iter() {
        //     debug_assert!(solution.is_valid(self.problem));
        // }

        // We peform replace to be able to consume self.population which is behind mutable &self
        let population = replace(&mut self.population, S::new("population".to_string()));

        'population: for (index, solution) in population.into_iter().enumerate() {
            debug!("######################################################");
            debug!("######## SOLUTION {} ########", index + 1);
            debug!("######## {:?} ########", solution);
            debug!("######################################################");

            let neighbors: Vec<EncodedSolution> = solution.neighborhood(
                self.neigborhood_structure,
                self.problem,
                timer,
                self.is_deterministic,
            );

            for (neighbor_index, neighbor) in neighbors.into_iter().enumerate() {
                explored_neighbor_count += 1;
                debug!(
                    "######## NEIGHBOR {} {:?} ########",
                    neighbor_index, neighbor
                );
                debug_assert!(neighbor.is_valid(self.problem));
                if self.explored_solutions.is_registered(&neighbor) {
                    debug!("Neighbor nr {} was already explored.", neighbor_index);
                    duplicated_neighbor_count += 1;
                    continue;
                }

                if !neighbor.is_weakly_dominated(&solution) {
                    if self.approximated_pareto_set.try_add(&neighbor) {
                        debug!("Neighbor nr {neighbor_index} was added to approximated pareto set. Approximated pareto set size: {}", self.approximated_pareto_set.len());
                        self.explored_solutions
                            .register(iteration, &neighbor, timer.elapsed());
                        pareto_added_count += 1;
                        if auxiliary_population.try_add(&neighbor) {
                            auxiliary_added_count += 1;
                        } else {
                            debug!("Neighbor nr {neighbor_index} is dominated so it wasn't added to auxiliary population");
                        }
                    } else {
                        debug!(
                            "Neighbor nr {neighbor_index} wasn't added to approximated pareto set."
                        );
                    }
                } else {
                    debug!("Neighbor nr {neighbor_index} is weakly dominated by current solution.");
                }
                if timer.is_expired() {
                    info!("Timer expired. Stop exploring neighbors.");
                    break 'population;
                }

                // if pareto_snapshot_time.elapsed() > pareto_snapshot_interval {
                //     pareto_snapshot_time = Instant::now();
                //     self.explored_solutions.register_pareto_snapshot(
                //         iteration,
                //         timer.elapsed(),
                //         self.approximated_pareto_set.iter(),
                //     );
                // }
            }

            self.explored_solutions
                .update_explored_neighborhood_size(&solution, self.neigborhood_structure)
        }
        let auxiliary_removed_count = auxiliary_added_count - auxiliary_population.len();
        let pareto_removed_count =
            pareto_initial_count + pareto_added_count - self.approximated_pareto_set.len();
        let neighborhood_size = self.neigborhood_structure;
        let duration_us = step_time.elapsed().as_micros();
        let duplicated_percent =
            duplicated_neighbor_count as f32 / explored_neighbor_count as f32 * 100.0;
        let per_solution_search_time = duration_us as f32 / explored_neighbor_count as f32;
        error!("Iteration {iteration} [{duration_us} us, {per_solution_search_time} us/sol], neighbors: size: {neighborhood_size}, explored: {explored_neighbor_count}, duplicated: {duplicated_neighbor_count} ({duplicated_percent} %), auxiliary: +{auxiliary_added_count}-{auxiliary_removed_count}, pareto: +{pareto_added_count}-{pareto_removed_count}");
        debug!("===== Auxiliary population solutions: =====");
        for solution in auxiliary_population.iter() {
            info!("{:?}", solution);
        }
        info!("===== Pareto Front solutions: =====");
        for solution in self.approximated_pareto_set.iter() {
            info!("{:?}", solution);
        }

        if !auxiliary_population.is_empty() {
            info!("Replacing current population with auxiliary population.");
            self.population = auxiliary_population.with_name("population".to_string());
            info!("Start again with smallest neighborhood structure.");
            self.neigborhood_structure = *self.neighborhood_size_range.start();
            return StepStatus::NewPopulation;
        } else {
            info!("Increasing neighborhood structure.");
            self.neigborhood_structure += 1;
            if self
                .neighborhood_size_range
                .contains(&self.neigborhood_structure)
            {
                info!("Use solutions from approximated pareto set which are not already Pareto local optimum");
                for solution in self.approximated_pareto_set.iter().filter(|&solution| {
                    self.explored_solutions.explored_neighborhood_size(solution)
                        < self.neigborhood_structure
                }) {
                    self.population.force_add(solution);
                }
            } else {
                info!("Reached maximum neighborhood structure.");
                return StepStatus::AllNeighborhoodStructuresExplored;
            }
            return StepStatus::IncreasedNeighborhoodStructure;
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
                info!("All neighborhood structures were explored. Number of iterations: {i}. Elapsed time: [{:?} ms]", pls_timer.elapsed());
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
