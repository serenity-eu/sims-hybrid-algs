//! Literal MOEA/D baseline, following Zhang & Li (2007) as closely as the
//! SIMS set-cover representation allows.
//!
//! "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition"
//! IEEE Transactions on Evolutionary Computation, 11(6), 712-731.
//!
//! This module exists to give the SIMS-tailored `moead.rs` (which adds a
//! neighbourhood/global mating switch `delta`, a replacement cap `nr`, PBI,
//! stagnation injection, and a composite mutation suite) an unmodified
//! reference point. Those four additions are from Li & Zhang (2009),
//! "Multiobjective Optimization Problems with Complicated Pareto Sets,
//! MOEA/D and NSGA-II" (IEEE TEC 13(2)) -- not the original 2007 paper -- so
//! they should not be assumed to help without an ablation against this
//! baseline.
//!
//! Deviations from the 2007 paper that are unavoidable for this problem
//! domain (not algorithmic choices):
//! - The paper is generic over "genetic operators" and a "problem-specific
//!   repair/improvement heuristic" (Step 2.1/2.2). We use uniform crossover
//!   + per-bit mutation on the image-selection bitset, each followed by
//!   greedy repair to restore set-cover feasibility -- the same operator
//!   family the paper's own MOKP example uses (one-point crossover + 0.01
//!   per-bit mutation + a greedy repair heuristic).
//!
//! Faithfully reproduced from Section III-A "General Framework", Step 2:
//! - Step 2.1 (Reproduction): parents `k, l` are drawn **only** from the
//!   neighbourhood `B(i)` -- no probability of mating from the whole
//!   population (that `delta` parameter is a 2009 addition).
//! - Step 2.4 (Update of Neighbouring Solutions): **every** neighbour
//!   `j in B(i)` whose Tchebycheff value improves is replaced -- no
//!   replacement cap (`nr` is also a 2009 addition).
//! - Tchebycheff decomposition only (the paper studies Tchebycheff as its
//!   primary approach; PBI is presented as an alternative in a later
//!   section, not the Step-2 framework).
//! - External population (archive) `EP`, maintained exactly as in Step 2.5.

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
    bitflip_mutation, compute_ideal_point, compute_neighbourhoods, generate_weight_vectors,
    random_population, tchebycheff_value, uniform_crossover,
};

/// Configuration for the literal MOEA/D baseline.
#[derive(Debug, Clone)]
pub struct MoeadBaselineConfig {
    /// Number of simplex-lattice divisions used to generate weight vectors.
    /// Population size N is whatever this produces (C(n+D-1, D-1)).
    pub num_divisions: usize,
    /// Neighbourhood size T.
    pub neighbourhood_size: usize,
    /// Crossover probability.
    pub crossover_rate: f64,
    /// Per-bit mutation probability (paper's MOKP example uses 0.01).
    pub mutation_rate: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for MoeadBaselineConfig {
    fn default() -> Self {
        Self {
            num_divisions: 99,      // 100 weight vectors for D=2, per the paper's MOKP setup
            neighbourhood_size: 10, // T=10, as used throughout the paper's experiments
            crossover_rate: 1.0,
            mutation_rate: 0.01,
            seed: 42,
        }
    }
}

/// Literal MOEA/D baseline (Zhang & Li, 2007).
pub struct MoeadBaseline<'a, P, const D: usize>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    problem: &'a P,
    config: MoeadBaselineConfig,
    weights: Vec<[f64; D]>,
    neighbourhoods: Vec<Vec<usize>>,
    population: Vec<BitsetEncodedSolution<P, D>>,
    ideal_point: [f64; D],
    archive: Vec<BitsetEncodedSolution<P, D>>,
    pub explored_solutions: ExploredSolutionsData<D>,
    rng: SmallRng,
}

impl<'a, P, const D: usize> MoeadBaseline<'a, P, D>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    /// Step 1: Initialization.
    pub fn new(problem: &'a P, config: MoeadBaselineConfig) -> Self {
        Self::new_with_seed_population(problem, config, vec![])
    }

    /// Create with an externally-provided seed population.
    ///
    /// Seeds fill subproblem slots first; remaining slots are filled randomly.
    pub fn new_with_seed_population(
        problem: &'a P,
        config: MoeadBaselineConfig,
        seed_population: Vec<BitsetEncodedSolution<P, D>>,
    ) -> Self {
        let mut rng = SmallRng::seed_from_u64(config.seed);

        // Step 1.2: weight vectors + T-closest neighbourhoods.
        let weights = generate_weight_vectors::<D>(config.num_divisions);
        let neighbourhoods = compute_neighbourhoods(&weights, config.neighbourhood_size);

        // Step 1.3: initial population — seed slots first, fill remainder randomly.
        let n = weights.len();
        let mut population = seed_population;
        population.truncate(n); // MOEA/D needs exactly N = |weights| solutions
        let random_count = n.saturating_sub(population.len());
        population.extend(random_population(problem, random_count, &mut rng));

        // Step 1.4: ideal point z.
        let ideal_point = compute_ideal_point(&population);

        info!(
            "MOEA/D baseline (Zhang & Li 2007): N={}, T={}, divisions={}",
            weights.len(),
            config.neighbourhood_size,
            config.num_divisions,
        );

        // Step 1.1 / EP init: archive = non-dominated subset of the initial population.
        let mut moead = Self {
            problem,
            config,
            weights,
            neighbourhoods,
            population,
            ideal_point,
            archive: Vec::new(),
            explored_solutions: ExploredSolutionsData::new(problem.max_objectives()),
            rng,
        };
        let init_pop = moead.population.clone();
        for sol in &init_pop {
            moead.try_insert_into_archive(sol);
        }
        moead
    }

    /// Step 2 / Step 3: run until `max_generations` passes over all subproblems,
    /// or `max_duration` elapses, whichever comes first. Returns EP.
    pub fn run(
        &mut self,
        max_generations: usize,
        max_duration: Duration,
    ) -> Vec<BitsetEncodedSolution<P, D>> {
        let timer = Timer::start(max_duration);
        let pop_size = self.population.len();

        for sol in &self.population {
            self.explored_solutions
                .register_without_selected_images(0, sol, Duration::ZERO);
        }

        for generation in 1..=max_generations {
            let gen_span = info_span!("moead_baseline_generation", generation = generation);
            let _guard = gen_span.enter();

            if timer.is_expired() {
                info!(
                    "MOEA/D baseline timeout after {} generations",
                    generation - 1
                );
                break;
            }

            // Step 2: "For i = 1, ..., N do" -- fixed order, exactly as written.
            for i in 0..pop_size {
                if timer.is_expired() {
                    break;
                }

                // Step 2.1: Reproduction -- parents drawn only from B(i).
                let (k, l) = self.select_two_from_neighbourhood(i);
                let mut y = if self.rng.random_bool(self.config.crossover_rate) {
                    uniform_crossover(
                        &self.population[k],
                        &self.population[l],
                        self.problem,
                        &mut self.rng,
                    )
                } else {
                    self.population[k].clone()
                };
                y = bitflip_mutation(&y, self.problem, &mut self.rng, self.config.mutation_rate);
                // (Repair is performed inside uniform_crossover/bitflip_mutation,
                // matching Step 2.2's "problem-specific repair/improvement heuristic".)

                if !self.explored_solutions.is_registered(&y) {
                    self.explored_solutions.register_without_selected_images(
                        generation,
                        &y,
                        timer.elapsed(),
                    );
                }

                // Step 2.3: Update of z.
                for d in 0..D {
                    let val = y.objectives()[d] as f64;
                    if val < self.ideal_point[d] {
                        self.ideal_point[d] = val;
                    }
                }

                // Step 2.4: Update of Neighbouring Solutions -- replace *every*
                // improving neighbour in B(i), no cap.
                let neighbours = self.neighbourhoods[i].clone();
                for j in neighbours {
                    let y_value =
                        tchebycheff_value(y.objectives(), &self.weights[j], &self.ideal_point);
                    let current_value = tchebycheff_value(
                        self.population[j].objectives(),
                        &self.weights[j],
                        &self.ideal_point,
                    );
                    if y_value <= current_value {
                        self.population[j] = y.clone();
                    }
                }

                // Step 2.5: Update of EP.
                self.try_insert_into_archive(&y);
            }
        }

        self.explored_solutions.num_iterations = max_generations;
        info!(
            "MOEA/D baseline completed: EP size={}, total_explored={}",
            self.archive.len(),
            self.explored_solutions.solutions.len(),
        );

        self.archive.clone()
    }

    /// Draw two distinct indices from B(i) (falls back to the same index twice
    /// if the neighbourhood has fewer than 2 members, e.g. tiny T).
    fn select_two_from_neighbourhood(&mut self, i: usize) -> (usize, usize) {
        let pool = &self.neighbourhoods[i];
        if pool.len() < 2 {
            let idx = pool.first().copied().unwrap_or(i);
            return (idx, idx);
        }
        let a = pool[self.rng.random_range(0..pool.len())];
        let mut b = pool[self.rng.random_range(0..pool.len())];
        for _ in 0..5 {
            if b != a {
                break;
            }
            b = pool[self.rng.random_range(0..pool.len())];
        }
        (a, b)
    }

    /// Step 2.5: remove EP members dominated by `y'`; add `y'` if not dominated
    /// by any current EP member.
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

    /// Get a reference to the current external population (archive).
    pub fn archive(&self) -> &[BitsetEncodedSolution<P, D>] {
        &self.archive
    }

    /// Get a reference to the current internal population.
    pub fn population(&self) -> &[BitsetEncodedSolution<P, D>] {
        &self.population
    }

    /// Get explored solutions data (compatible with PLS/EA output format).
    pub fn explored_solutions_data(&self) -> &ExploredSolutionsData<D> {
        &self.explored_solutions
    }
}

/// Run the literal MOEA/D baseline and return (EP, explored solutions).
pub fn run_moead_baseline<P, const D: usize>(
    problem: &P,
    config: MoeadBaselineConfig,
    max_generations: usize,
    max_duration: Duration,
) -> (Vec<BitsetEncodedSolution<P, D>>, ExploredSolutionsData<D>)
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    run_moead_baseline_seeded(problem, config, vec![], max_generations, max_duration)
}

/// Run the literal MOEA/D baseline seeded with an external population.
pub fn run_moead_baseline_seeded<P, const D: usize>(
    problem: &P,
    config: MoeadBaselineConfig,
    seed_population: Vec<BitsetEncodedSolution<P, D>>,
    max_generations: usize,
    max_duration: Duration,
) -> (Vec<BitsetEncodedSolution<P, D>>, ExploredSolutionsData<D>)
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    let mut moead = MoeadBaseline::new_with_seed_population(problem, config, seed_population);
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

    fn make_test_problem() -> ProblemBitset<2> {
        let raw = SIMSProblemInstanceRaw {
            name: "moead_baseline_test".to_string(),
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
    fn test_moead_baseline_runs_and_produces_feasible_solutions() {
        let problem = make_test_problem();
        let config = MoeadBaselineConfig {
            num_divisions: 9, // 10 weight vectors
            neighbourhood_size: 5,
            ..Default::default()
        };

        let mut moead = MoeadBaseline::new(&problem, config);
        let archive = moead.run(50, Duration::from_secs(5));

        assert!(!archive.is_empty(), "EP should not be empty");

        for sol in &archive {
            assert!(
                problem.is_set_cover(sol),
                "EP solution must be a valid set cover"
            );
        }

        for (i, a) in archive.iter().enumerate() {
            for (j, b) in archive.iter().enumerate() {
                if i != j {
                    assert!(
                        !a.dominates(b.objectives()),
                        "EP solution {i} dominates solution {j}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_moead_baseline_respects_timeout() {
        let problem = make_test_problem();
        let config = MoeadBaselineConfig {
            num_divisions: 9,
            neighbourhood_size: 5,
            ..Default::default()
        };

        let start = std::time::Instant::now();
        let mut moead = MoeadBaseline::new(&problem, config);
        let _archive = moead.run(1_000_000, Duration::from_millis(200));
        let elapsed = start.elapsed();

        assert!(
            elapsed < Duration::from_secs(2),
            "Should respect timeout, but took {:?}",
            elapsed
        );
    }

    #[test]
    fn test_moead_baseline_no_replacement_cap() {
        // Sanity check: with neighbourhood_size >= population size, a single
        // improving offspring can in principle replace every neighbour --
        // unlike the SIMS moead.rs, there is no `max_replacements` to cap this.
        let problem = make_test_problem();
        let config = MoeadBaselineConfig {
            num_divisions: 4,        // small population
            neighbourhood_size: 100, // larger than population: clamped internally
            ..Default::default()
        };
        let mut moead = MoeadBaseline::new(&problem, config);
        let archive = moead.run(20, Duration::from_secs(5));
        assert!(!archive.is_empty());
    }

    #[test]
    fn test_run_moead_baseline_convenience() {
        let problem = make_test_problem();
        let config = MoeadBaselineConfig {
            num_divisions: 9,
            neighbourhood_size: 5,
            ..Default::default()
        };
        let (archive, explored) =
            run_moead_baseline(&problem, config, 20, Duration::from_secs(3));
        assert!(!archive.is_empty());
        assert!(!explored.solutions.is_empty());
    }
}
