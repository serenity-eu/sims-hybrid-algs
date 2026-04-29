//! Adapter for the `optirustic` crate's multi-objective evolutionary algorithms.
//!
//! This module wraps external NSGA-II and NSGA-III implementations from the
//! [`optirustic`](https://crates.io/crates/optirustic) crate, adapting them to solve
//! SIMS (Satellite Image Mosaic Selection) problems.
//!
//! ## Approach
//!
//! SIMS is a constrained set-cover problem with binary decision variables.
//! The `optirustic` crate uses continuous variable operators (Simulated Binary
//! Crossover + Polynomial Mutation), so we model SIMS as a **continuous relaxation**:
//!
//! - Each image gets a real-valued variable in `[0, 1]`.
//! - In the evaluator, variables are thresholded at 0.5 to produce a binary selection.
//! - **Greedy repair** restores feasibility (set-cover constraint).
//! - **Redundancy removal** prunes unnecessary images for leaner solutions.
//! - Objectives are computed on the repaired binary solution.
//!
//! This allows optirustic's SBX crossover to explore the continuous landscape,
//! where proximity to 0 or 1 encodes selection confidence, while the evaluator
//! always produces valid SIMS solutions.
//!
//! ## Available Algorithms
//!
//! - **NSGA-II** via [`run_optirustic_nsga2`]
//! - **NSGA-III** via [`run_optirustic_nsga3`]
//!
//! Both return results as `Vec<BitsetEncodedSolution<P, D>>` compatible with
//! the rest of the framework.

use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;
use std::time::Duration;

use fixedbitset::FixedBitSet;
use tracing::{info, info_span};

use crate::explored_solutions_data::ExploredSolutionsData;
use crate::objectives::{ObjectiveState, ObjectiveType};
use crate::problem::SetCoverProblem;
use crate::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for optirustic-based algorithms.
#[derive(Debug, Clone)]
pub struct OptirusticConfig {
    /// Number of individuals in the population.
    pub population_size: usize,
    /// Maximum number of generations.
    pub max_generations: usize,
    /// Distribution index for SBX crossover (higher = more exploitation).
    pub crossover_distribution_index: f64,
    /// Crossover probability.
    pub crossover_probability: f64,
    /// Distribution index for polynomial mutation (higher = more exploitation).
    pub mutation_distribution_index: f64,
    /// Per-variable mutation probability. If `None`, defaults to `1/num_vars`.
    pub mutation_probability: Option<f64>,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
    /// Whether to run evaluation in parallel.
    pub parallel: bool,
}

impl Default for OptirusticConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            max_generations: 200,
            crossover_distribution_index: 20.0,
            crossover_probability: 0.9,
            mutation_distribution_index: 20.0,
            mutation_probability: None,
            parallel: false,
            seed: Some(42),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: shared problem data for the evaluator
// ---------------------------------------------------------------------------

/// Precomputed SIMS problem data shared by the evaluator.
/// Must be `Send + Sync` for optirustic's parallel evaluation.
#[derive(Debug)]
struct SimsProblemData<const D: usize> {
    num_images: usize,
    num_elements: usize,
    image_elements: Vec<FixedBitSet>,
    element_to_images: Vec<Vec<usize>>,
    objective_modes: Vec<ObjectiveMode>,
    /// Per-objective per-image cost/value arrays.
    objective_costs: Vec<Vec<u64>>,
}

/// How to compute each SIMS objective from selected images.
#[derive(Debug, Clone)]
enum ObjectiveMode {
    /// Sum of per-image costs (e.g., TotalCost).
    SumCosts,
    /// Cloudy area: sum of areas for elements not "clear" in any selected image.
    CloudyArea {
        clear_elements: Vec<FixedBitSet>,
        areas: Vec<u64>,
    },
    /// Maximum value among selected images (e.g., MaxIncidenceAngle).
    MaxValue,
    /// Maximum value among selected (for MinResolution, which stores worst-is-max).
    MaxOfSelected,
}

impl<const D: usize> SimsProblemData<D> {
    fn from_problem<P: SetCoverProblem<D>>(problem: &P) -> Self {
        let num_images = problem.num_images();
        let num_elements = problem.num_elements();

        let image_elements: Vec<FixedBitSet> = (0..num_images)
            .map(|i| {
                let mut bs = FixedBitSet::with_capacity(num_elements);
                for e in problem.image_elements(i) {
                    bs.insert(e);
                }
                bs
            })
            .collect();

        let mut element_to_images = vec![Vec::new(); num_elements];
        for (img_idx, bs) in image_elements.iter().enumerate() {
            for elem in bs.ones() {
                element_to_images[elem].push(img_idx);
            }
        }

        let obj_types = problem.objective_types();
        let objectives = problem.objectives();

        let mut objective_costs = Vec::with_capacity(D);
        let mut objective_modes = Vec::with_capacity(D);

        for i in 0..D {
            match obj_types[i] {
                ObjectiveType::TotalCost => {
                    if let ObjectiveState::TotalCost { ref costs, .. } = objectives[i] {
                        objective_costs.push(costs.clone());
                    } else {
                        objective_costs.push(vec![0; num_images]);
                    }
                    objective_modes.push(ObjectiveMode::SumCosts);
                }
                ObjectiveType::CloudyArea => {
                    if let ObjectiveState::CloudyArea {
                        ref clear_images,
                        ref areas,
                        ..
                    } = objectives[i]
                    {
                        objective_costs.push(vec![0; num_images]);
                        objective_modes.push(ObjectiveMode::CloudyArea {
                            clear_elements: clear_images.clone(),
                            areas: areas.clone(),
                        });
                    } else {
                        objective_costs.push(vec![0; num_images]);
                        objective_modes.push(ObjectiveMode::SumCosts);
                    }
                }
                ObjectiveType::MinResolution => {
                    if let ObjectiveState::MinResolution {
                        ref resolutions, ..
                    } = objectives[i]
                    {
                        objective_costs.push(resolutions.clone());
                    } else {
                        objective_costs.push(vec![0; num_images]);
                    }
                    objective_modes.push(ObjectiveMode::MaxOfSelected);
                }
                ObjectiveType::MaxIncidenceAngle => {
                    if let ObjectiveState::MaxIncidenceAngle {
                        ref incidence_angles,
                        ..
                    } = objectives[i]
                    {
                        objective_costs.push(incidence_angles.clone());
                    } else {
                        objective_costs.push(vec![0; num_images]);
                    }
                    objective_modes.push(ObjectiveMode::MaxValue);
                }
            }
        }

        Self {
            num_images,
            num_elements,
            image_elements,
            element_to_images,
            objective_costs,
            objective_modes,
        }
    }

    /// Threshold real variables at 0.5, greedy-repair, and remove redundant images.
    fn decode_and_repair(&self, real_vars: &[f64]) -> FixedBitSet {
        let mut selected = FixedBitSet::with_capacity(self.num_images);
        for (i, &val) in real_vars.iter().enumerate() {
            if i < self.num_images && val >= 0.5 {
                selected.insert(i);
            }
        }

        // Greedy repair
        let mut covered = FixedBitSet::with_capacity(self.num_elements);
        for img in selected.ones() {
            covered.union_with(&self.image_elements[img]);
        }

        if covered.count_ones(..) < self.num_elements {
            for elem in 0..self.num_elements {
                if covered.contains(elem) {
                    continue;
                }
                let mut best_image = None;
                let mut best_gain: usize = 0;
                for &candidate in &self.element_to_images[elem] {
                    if selected.contains(candidate) {
                        continue;
                    }
                    let gain = self.image_elements[candidate].difference(&covered).count();
                    if gain > best_gain {
                        best_gain = gain;
                        best_image = Some(candidate);
                    }
                }
                if let Some(img) = best_image {
                    selected.insert(img);
                    covered.union_with(&self.image_elements[img]);
                } else if let Some(&img) = self.element_to_images[elem]
                    .iter()
                    .find(|&&c| !selected.contains(c))
                {
                    selected.insert(img);
                    covered.union_with(&self.image_elements[img]);
                }
            }
        }

        // Redundancy removal
        let mut coverage_count = vec![0u32; self.num_elements];
        for img in selected.ones() {
            for elem in self.image_elements[img].ones() {
                coverage_count[elem] += 1;
            }
        }
        let mut candidates: Vec<usize> = selected.ones().collect();
        candidates.sort_by_key(|&img| self.image_elements[img].count_ones(..));
        for img in candidates {
            if !selected.contains(img) {
                continue;
            }
            let is_redundant = self.image_elements[img]
                .ones()
                .all(|elem| coverage_count[elem] >= 2);
            if is_redundant {
                selected.set(img, false);
                for elem in self.image_elements[img].ones() {
                    coverage_count[elem] -= 1;
                }
            }
        }

        selected
    }

    /// Compute objective values for a repaired binary solution.
    fn compute_objectives(&self, selected: &FixedBitSet) -> [f64; D] {
        let mut result = [0.0f64; D];
        for obj_idx in 0..D {
            let value: u64 = match &self.objective_modes[obj_idx] {
                ObjectiveMode::SumCosts => {
                    let costs = &self.objective_costs[obj_idx];
                    selected.ones().map(|img| costs[img]).sum()
                }
                ObjectiveMode::CloudyArea {
                    clear_elements,
                    areas,
                } => {
                    let mut clear = FixedBitSet::with_capacity(self.num_elements);
                    for img in selected.ones() {
                        clear.union_with(&clear_elements[img]);
                    }
                    (0..self.num_elements)
                        .filter(|&e| !clear.contains(e))
                        .map(|e| areas[e])
                        .sum()
                }
                ObjectiveMode::MaxValue | ObjectiveMode::MaxOfSelected => {
                    let costs = &self.objective_costs[obj_idx];
                    selected.ones().map(|img| costs[img]).max().unwrap_or(0)
                }
            };
            result[obj_idx] = value as f64;
        }
        result
    }
}

// ---------------------------------------------------------------------------
// optirustic Evaluator implementation
// ---------------------------------------------------------------------------

/// Evaluator that decodes continuous [0,1] variables into binary SIMS solutions.
#[derive(Debug)]
struct SimsEvaluator<const D: usize> {
    data: Arc<SimsProblemData<D>>,
    /// Objective names in order, for inserting into EvaluationResult.
    objective_names: Vec<String>,
}

impl<const D: usize> optirustic::core::Evaluator for SimsEvaluator<D> {
    fn evaluate(
        &self,
        individual: &optirustic::core::Individual,
    ) -> Result<optirustic::core::EvaluationResult, Box<dyn Error>> {
        // Extract real variables (one per image)
        let mut real_vars = vec![0.0f64; self.data.num_images];
        for i in 0..self.data.num_images {
            let var_name = format!("img_{i}");
            let val = individual.get_variable_value(&var_name)?.as_real()?;
            real_vars[i] = val;
        }

        // Decode, repair, evaluate
        let selected = self.data.decode_and_repair(&real_vars);
        let objs = self.data.compute_objectives(&selected);

        let mut objectives = HashMap::new();
        for (i, name) in self.objective_names.iter().enumerate() {
            objectives.insert(name.clone(), objs[i]);
        }

        Ok(optirustic::core::EvaluationResult {
            constraints: None,
            objectives,
        })
    }
}

// ---------------------------------------------------------------------------
// Problem construction helpers
// ---------------------------------------------------------------------------

/// Build an optirustic `Problem` from SIMS problem data.
fn build_optirustic_problem<P, const D: usize>(
    problem: &P,
    data: Arc<SimsProblemData<D>>,
) -> Result<optirustic::core::Problem, Box<dyn Error>>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    use optirustic::core::{BoundedNumber, Objective, ObjectiveDirection, Problem, VariableType};

    let num_images = problem.num_images();

    // Variables: one real variable per image in [0, 1]
    let variables: Vec<VariableType> = (0..num_images)
        .map(|i| {
            VariableType::Real(
                BoundedNumber::new(&format!("img_{i}"), 0.0, 1.0).expect("Invalid variable bounds"),
            )
        })
        .collect();

    // Objectives: all minimization (SIMS convention)
    let obj_types = problem.objective_types();
    let objective_names: Vec<String> = obj_types.iter().map(|ot| ot.id().to_string()).collect();

    let objectives: Vec<Objective> = objective_names
        .iter()
        .map(|name| Objective::new(name, ObjectiveDirection::Minimise))
        .collect();

    // Evaluator
    let evaluator = SimsEvaluator {
        data: Arc::clone(&data),
        objective_names,
    };

    let opti_problem = Problem::new(objectives, variables, None, Box::new(evaluator))?;
    Ok(opti_problem)
}

// ---------------------------------------------------------------------------
// Result conversion
// ---------------------------------------------------------------------------

/// Extract final population from an optirustic algorithm and convert to
/// `BitsetEncodedSolution`, applying repair and Pareto filtering.
fn extract_solutions<P, const D: usize>(
    population: &optirustic::core::Population,
    problem: &P,
    data: &SimsProblemData<D>,
) -> Vec<BitsetEncodedSolution<P, D>>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    use pareto::{HasObjectives, MoSolution};

    let individuals = population.individuals();
    let mut solutions = Vec::with_capacity(individuals.len());

    for individual in individuals {
        // Extract real variables
        let mut real_vars = vec![0.0f64; data.num_images];
        for i in 0..data.num_images {
            let var_name = format!("img_{i}");
            if let Ok(val) = individual.get_variable_value(&var_name) {
                if let Ok(r) = val.as_real() {
                    real_vars[i] = r;
                }
            }
        }

        let selected = data.decode_and_repair(&real_vars);
        let selected_vec: Vec<usize> = selected.ones().collect();
        if selected_vec.is_empty() {
            continue;
        }

        let sol = BitsetEncodedSolution::from_selected_images(&selected_vec, problem);
        if !problem.is_set_cover(&sol) {
            continue;
        }

        solutions.push(sol);
    }

    // Filter to non-dominated
    let mut archive: Vec<BitsetEncodedSolution<P, D>> = Vec::new();
    for sol in solutions {
        let dominated = archive
            .iter()
            .any(|existing| existing.dominates(sol.objectives()));
        if dominated {
            continue;
        }
        archive.retain(|existing| !sol.dominates(existing.objectives()));
        archive.push(sol);
    }

    archive
}

// ---------------------------------------------------------------------------
// NSGA-II via optirustic
// ---------------------------------------------------------------------------

/// Run NSGA-II from the `optirustic` crate on a SIMS problem instance.
///
/// Uses continuous relaxation: each image has a real variable in `[0, 1]`,
/// thresholded and repaired in the evaluator. SBX crossover + polynomial
/// mutation operate on the continuous representation.
///
/// # Returns
/// A tuple of (non-dominated solutions, explored solutions data).
pub fn run_optirustic_nsga2<P, const D: usize>(
    problem: &P,
    config: OptirusticConfig,
    _max_duration: Duration,
) -> (Vec<BitsetEncodedSolution<P, D>>, ExploredSolutionsData<D>)
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    let _span = info_span!("optirustic_nsga2").entered();
    let start = std::time::Instant::now();

    let data = Arc::new(SimsProblemData::<D>::from_problem(problem));

    info!(
        num_images = data.num_images,
        num_elements = data.num_elements,
        population_size = config.population_size,
        max_generations = config.max_generations,
        "Starting optirustic NSGA-II"
    );

    let opti_problem = match build_optirustic_problem(problem, Arc::clone(&data)) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!("Failed to build optirustic problem: {e}");
            let explored = ExploredSolutionsData::<D>::new(problem.max_objectives());
            return (Vec::new(), explored);
        }
    };

    use optirustic::algorithms::{Algorithm, NSGA2, NSGA2Arg, StoppingCondition};

    let args = NSGA2Arg {
        number_of_individuals: config.population_size,
        crossover_operator_options: None, // use defaults
        mutation_operator_options: None,  // use defaults
        resume_from_file: None,
        seed: config.seed,
        stopping_condition: StoppingCondition::MaxGeneration(
            config.max_generations.try_into().unwrap_or(u32::MAX),
        ),
        parallel: Some(config.parallel),
        export_history: None,
    };

    let mut algo = match NSGA2::new(opti_problem, args) {
        Ok(a) => a,
        Err(e) => {
            tracing::error!("Failed to create optirustic NSGA-II: {e}");
            let explored = ExploredSolutionsData::<D>::new(problem.max_objectives());
            return (Vec::new(), explored);
        }
    };

    if let Err(e) = algo.run() {
        tracing::error!("optirustic NSGA-II run failed: {e}");
        let explored = ExploredSolutionsData::<D>::new(problem.max_objectives());
        return (Vec::new(), explored);
    }

    let population = algo.population();
    let archive = extract_solutions(population, problem, &data);

    let elapsed = start.elapsed();
    info!(
        archive_size = archive.len(),
        elapsed_ms = elapsed.as_millis(),
        "optirustic NSGA-II completed"
    );

    let explored = ExploredSolutionsData::<D>::new(problem.max_objectives());
    (archive, explored)
}

// ---------------------------------------------------------------------------
// NSGA-III via optirustic
// ---------------------------------------------------------------------------

/// Run NSGA-III from the `optirustic` crate on a SIMS problem instance.
///
/// NSGA-III extends NSGA-II with reference-point-based selection, which is
/// particularly effective for many-objective (>=3) problems. For 2-objective
/// problems it behaves similarly to NSGA-II.
///
/// # Returns
/// A tuple of (non-dominated solutions, explored solutions data).
pub fn run_optirustic_nsga3<P, const D: usize>(
    problem: &P,
    config: OptirusticConfig,
    _max_duration: Duration,
) -> (Vec<BitsetEncodedSolution<P, D>>, ExploredSolutionsData<D>)
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    let _span = info_span!("optirustic_nsga3").entered();
    let start = std::time::Instant::now();

    let data = Arc::new(SimsProblemData::<D>::from_problem(problem));

    info!(
        num_images = data.num_images,
        num_elements = data.num_elements,
        population_size = config.population_size,
        max_generations = config.max_generations,
        "Starting optirustic NSGA-III"
    );

    let opti_problem = match build_optirustic_problem(problem, Arc::clone(&data)) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!("Failed to build optirustic problem: {e}");
            let explored = ExploredSolutionsData::<D>::new(problem.max_objectives());
            return (Vec::new(), explored);
        }
    };

    use optirustic::algorithms::{
        Algorithm, NSGA3, NSGA3Arg, Nsga3NumberOfIndividuals, StoppingCondition,
    };

    let partitions = choose_partitions_for_d(D);
    let args = NSGA3Arg {
        // Use EqualToReferencePointCount so optirustic picks a population
        // size that is compatible with the number of reference points.
        number_of_individuals: Nsga3NumberOfIndividuals::EqualToReferencePointCount,
        crossover_operator_options: None,
        mutation_operator_options: None,
        resume_from_file: None,
        seed: config.seed,
        stopping_condition: StoppingCondition::MaxGeneration(
            config.max_generations.try_into().unwrap_or(u32::MAX),
        ),
        parallel: Some(config.parallel),
        export_history: None,
        number_of_partitions: optirustic::utils::NumberOfPartitions::OneLayer(partitions),
    };

    // NSGA3::new takes (problem, args, adaptive)
    let mut algo = match NSGA3::new(opti_problem, args, false) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("[optirustic NSGA-III] Failed to create: {e}");
            tracing::error!("Failed to create optirustic NSGA-III: {e}");
            let explored = ExploredSolutionsData::<D>::new(problem.max_objectives());
            return (Vec::new(), explored);
        }
    };

    if let Err(e) = algo.run() {
        eprintln!("[optirustic NSGA-III] run failed: {e}");
        tracing::error!("optirustic NSGA-III run failed: {e}");
        let explored = ExploredSolutionsData::<D>::new(problem.max_objectives());
        return (Vec::new(), explored);
    }

    let population = algo.population();
    let archive = extract_solutions(population, problem, &data);

    let elapsed = start.elapsed();
    info!(
        archive_size = archive.len(),
        elapsed_ms = elapsed.as_millis(),
        "optirustic NSGA-III completed"
    );

    let explored = ExploredSolutionsData::<D>::new(problem.max_objectives());
    (archive, explored)
}

/// Choose a reasonable number of reference-point partitions for NSGA-III
/// based on the number of objectives.
fn choose_partitions_for_d(d: usize) -> usize {
    match d {
        1 => 12,
        2 => 12,
        3 => 8,
        4 => 6,
        5 => 5,
        _ => 4,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objectives::ObjectiveType;
    use crate::problem_bitset::ProblemBitset;

    const INSTANCES_PATH: &str = "tests/data";

    fn load_problem_2d(instance: &str) -> ProblemBitset<2> {
        let path = format!("{INSTANCES_PATH}/{instance}");
        let obj_types = [ObjectiveType::TotalCost, ObjectiveType::CloudyArea];
        ProblemBitset::from_minizinc_datafile(&path, obj_types)
            .unwrap_or_else(|e| panic!("Failed to load {instance}: {e}"))
    }

    fn load_problem_4d(instance: &str) -> ProblemBitset<4> {
        let path = format!("{INSTANCES_PATH}/{instance}");
        let obj_types = [
            ObjectiveType::TotalCost,
            ObjectiveType::CloudyArea,
            ObjectiveType::MinResolution,
            ObjectiveType::MaxIncidenceAngle,
        ];
        ProblemBitset::from_minizinc_datafile(&path, obj_types)
            .unwrap_or_else(|e| panic!("Failed to load {instance}: {e}"))
    }

    fn validate_archive<P, const D: usize>(
        archive: &[BitsetEncodedSolution<P, D>],
        problem: &P,
        label: &str,
    ) where
        P: SetCoverProblem<D> + Clone + Send + Sync,
    {
        use pareto::{HasObjectives, MoSolution};

        assert!(!archive.is_empty(), "[{label}] archive must not be empty");

        for (idx, sol) in archive.iter().enumerate() {
            assert!(
                problem.is_set_cover(sol),
                "[{label}] solution {idx} is not a valid set cover"
            );

            for obj_idx in 0..D {
                let expected = problem.objective(obj_idx).calculate_value(sol, problem);
                let actual = sol.objectives()[obj_idx];
                assert_eq!(
                    actual, expected,
                    "[{label}] solution {idx} objective {obj_idx} mismatch: stored={actual} expected={expected}"
                );
            }
        }

        // Non-dominance within the archive
        for i in 0..archive.len() {
            for j in 0..archive.len() {
                if i != j {
                    assert!(
                        !archive[i].dominates(archive[j].objectives()),
                        "[{label}] solution {i} dominates solution {j}: {:?} vs {:?}",
                        archive[i].objectives(),
                        archive[j].objectives(),
                    );
                }
            }
        }
    }

    #[test]
    fn test_optirustic_nsga2_2d_small() {
        let problem = load_problem_2d("lagos_nigeria_30.dzn");
        let config = OptirusticConfig {
            population_size: 50,
            max_generations: 30,
            parallel: false,
            ..Default::default()
        };
        let (archive, _explored) = run_optirustic_nsga2(&problem, config, Duration::from_secs(60));
        validate_archive(&archive, &problem, "optirustic_nsga2_2d_lagos_30");
    }

    #[test]
    fn test_optirustic_nsga2_4d_small() {
        let problem = load_problem_4d("lagos_nigeria_30.dzn");
        let config = OptirusticConfig {
            population_size: 50,
            max_generations: 30,
            parallel: false,
            ..Default::default()
        };
        let (archive, _explored) = run_optirustic_nsga2(&problem, config, Duration::from_secs(60));
        validate_archive(&archive, &problem, "optirustic_nsga2_4d_lagos_30");
    }

    #[test]
    fn test_optirustic_nsga3_2d_small() {
        let problem = load_problem_2d("lagos_nigeria_30.dzn");
        let config = OptirusticConfig {
            population_size: 50,
            max_generations: 30,
            parallel: false,
            ..Default::default()
        };
        let (archive, _explored) = run_optirustic_nsga3(&problem, config, Duration::from_secs(60));
        validate_archive(&archive, &problem, "optirustic_nsga3_2d_lagos_30");
    }

    #[test]
    fn test_optirustic_nsga3_4d_small() {
        let problem = load_problem_4d("lagos_nigeria_30.dzn");
        let config = OptirusticConfig {
            population_size: 50,
            max_generations: 30,
            parallel: false,
            ..Default::default()
        };
        let (archive, _explored) = run_optirustic_nsga3(&problem, config, Duration::from_secs(60));
        validate_archive(&archive, &problem, "optirustic_nsga3_4d_lagos_30");
    }

    #[test]
    fn test_optirustic_nsga2_determinism() {
        let problem = load_problem_2d("lagos_nigeria_30.dzn");
        let config = OptirusticConfig {
            population_size: 40,
            max_generations: 20,
            parallel: false,
            seed: Some(12345),
            ..Default::default()
        };

        let (archive1, _) = run_optirustic_nsga2(&problem, config.clone(), Duration::from_secs(60));
        let (archive2, _) = run_optirustic_nsga2(&problem, config, Duration::from_secs(60));

        use pareto::HasObjectives;
        assert_eq!(
            archive1.len(),
            archive2.len(),
            "Determinism: archive sizes differ"
        );

        let mut objs1: Vec<_> = archive1.iter().map(|s| s.objectives().to_vec()).collect();
        let mut objs2: Vec<_> = archive2.iter().map(|s| s.objectives().to_vec()).collect();
        objs1.sort();
        objs2.sort();
        assert_eq!(objs1, objs2, "Determinism: archive objectives differ");
    }
}
