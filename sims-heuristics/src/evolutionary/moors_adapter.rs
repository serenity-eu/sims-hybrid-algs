//! Adapter for the `moors` crate's multi-objective evolutionary algorithms.
//!
//! This module wraps external NSGA-II and SPEA-2 implementations from the
//! [`moors`](https://crates.io/crates/moors) crate, adapting them to solve
//! SIMS (Satellite Image Mosaic Selection) problems.
//!
//! ## Approach
//!
//! SIMS is a constrained set-cover problem with binary decision variables
//! (select image or not). The `moors` crate natively supports binary
//! representations with operators like `UniformBinaryCrossover`,
//! `SinglePointBinaryCrossover`, and `BitFlipMutation`.
//!
//! Since `moors` operators may produce infeasible offspring (not covering all
//! elements), the fitness function internally applies **greedy repair** +
//! **redundancy removal** before evaluating objectives. This ensures that
//! fitness values always correspond to feasible solutions, guiding evolution
//! toward good coverage structures.
//!
//! ## Available Algorithms
//!
//! - **NSGA-II** via [`run_moors_nsga2`]
//! - **SPEA-2** via [`run_moors_spea2`]
//! - **AGE-MOEA** via [`run_moors_age_moea`]
//!
//! Both return results as `Vec<BitsetEncodedSolution<P, D>>` compatible with
//! the rest of the framework.

use std::sync::Arc;
use std::time::Duration;

use fixedbitset::FixedBitSet;
use ndarray::Array2;
use tracing::{info, info_span};

use crate::explored_solutions_data::ExploredSolutionsData;
use crate::problem::SetCoverProblem;
use crate::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for moors-based algorithms.
#[derive(Debug, Clone)]
pub struct MoorsConfig {
    /// Population size.
    pub population_size: usize,
    /// Number of offspring per generation.
    pub num_offsprings: usize,
    /// Number of iterations (generations).
    pub num_iterations: usize,
    /// Crossover rate (probability of crossover per pair).
    pub crossover_rate: f64,
    /// Mutation rate (probability of mutation per individual).
    pub mutation_rate: f64,
    /// Bit-flip probability for `BitFlipMutation` (per-bit probability).
    pub bitflip_probability: f64,
    /// Which crossover operator to use.
    pub crossover_type: MoorsCrossoverType,
    /// Random seed for reproducibility.
    pub seed: u64,
}

/// Crossover operator selection for moors adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoorsCrossoverType {
    /// Uniform binary crossover (each gene inherited from either parent with p=0.5).
    Uniform,
    /// Single-point binary crossover.
    SinglePoint,
    /// Two-point binary crossover.
    TwoPoint,
}

impl Default for MoorsConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            num_offsprings: 50,
            num_iterations: 200,
            crossover_rate: 0.9,
            mutation_rate: 0.1,
            bitflip_probability: 0.05,
            crossover_type: MoorsCrossoverType::Uniform,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: problem data wrapper for fitness/constraint closures
// ---------------------------------------------------------------------------

/// Holds precomputed problem data needed by the fitness function.
/// This is `Send + Sync` so it can be shared with moors' internal threads.
struct SimsProblemData<const D: usize> {
    /// Number of images (= number of binary decision variables).
    num_images: usize,
    /// Number of elements in the universe.
    num_elements: usize,
    /// Each image's covered elements as a `FixedBitSet`.
    image_elements: Vec<FixedBitSet>,
    /// Inverted index: which images cover each element.
    element_to_images: Vec<Vec<usize>>,
    /// Precomputed objective data per image, stored per objective.
    /// For each objective index, stores a closure-friendly representation.
    objective_costs: Vec<Vec<u64>>,
    /// Objective computation mode for each of the D objectives.
    objective_modes: Vec<ObjectiveMode>,
}

/// How to compute each SIMS objective from selected images.
#[derive(Debug, Clone)]
enum ObjectiveMode {
    /// Sum of per-image costs (e.g., TotalCost).
    SumCosts,
    /// Cloudy area: sum of areas for elements not "clear" in any selected image.
    CloudyArea {
        /// For each image, the set of elements it covers clearly (no cloud).
        clear_elements: Vec<FixedBitSet>,
        /// Area of each element.
        areas: Vec<u64>,
    },
    /// Worst-case (max) per-image value (e.g., MaxIncidenceAngle).
    MaxValue,
    /// Worst-case (min→max via negation trick) per-image value (e.g., MinResolution).
    /// We store the values and compute `max(values[i])` for selected images,
    /// which for MinResolution means worst (highest) resolution value.
    MaxOfSelected,
}

impl<const D: usize> SimsProblemData<D> {
    /// Build from a `SetCoverProblem`.
    fn from_problem<P: SetCoverProblem<D>>(problem: &P) -> Self {
        let num_images = problem.num_images();
        let num_elements = problem.num_elements();

        // Build image element bitsets
        let image_elements: Vec<FixedBitSet> = (0..num_images)
            .map(|i| {
                let mut bs = FixedBitSet::with_capacity(num_elements);
                for e in problem.image_elements(i) {
                    bs.insert(e);
                }
                bs
            })
            .collect();

        // Build inverted index
        let mut element_to_images = vec![Vec::new(); num_elements];
        for (img_idx, bs) in image_elements.iter().enumerate() {
            for elem in bs.ones() {
                element_to_images[elem].push(img_idx);
            }
        }

        // Extract objective data
        let obj_types = problem.objective_types();
        let objectives = problem.objectives();

        let mut objective_costs = Vec::with_capacity(D);
        let mut objective_modes = Vec::with_capacity(D);

        for i in 0..D {
            use crate::objectives::{ObjectiveState, ObjectiveType};
            match obj_types[i] {
                ObjectiveType::TotalCost => {
                    if let ObjectiveState::TotalCost { ref costs, .. } = objectives[i] {
                        objective_costs.push(costs.clone());
                        objective_modes.push(ObjectiveMode::SumCosts);
                    } else {
                        objective_costs.push(vec![0; num_images]);
                        objective_modes.push(ObjectiveMode::SumCosts);
                    }
                }
                ObjectiveType::CloudyArea => {
                    if let ObjectiveState::CloudyArea {
                        ref clear_images,
                        ref areas,
                        ..
                    } = objectives[i]
                    {
                        objective_costs.push(vec![0; num_images]); // placeholder
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
                        objective_modes.push(ObjectiveMode::MaxOfSelected);
                    } else {
                        objective_costs.push(vec![0; num_images]);
                        objective_modes.push(ObjectiveMode::MaxOfSelected);
                    }
                }
                ObjectiveType::MaxIncidenceAngle => {
                    if let ObjectiveState::MaxIncidenceAngle {
                        ref incidence_angles,
                        ..
                    } = objectives[i]
                    {
                        objective_costs.push(incidence_angles.clone());
                        objective_modes.push(ObjectiveMode::MaxValue);
                    } else {
                        objective_costs.push(vec![0; num_images]);
                        objective_modes.push(ObjectiveMode::MaxValue);
                    }
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

    /// Repair a gene row (binary 0/1 selection) in-place and return the repaired
    /// `FixedBitSet` of selected images.
    fn repair_genes(&self, genes_row: &[f64]) -> FixedBitSet {
        let mut selected = FixedBitSet::with_capacity(self.num_images);
        for (i, &val) in genes_row.iter().enumerate() {
            if i < self.num_images && val > 0.5 {
                selected.insert(i);
            }
        }

        // Greedy repair: add images until all elements are covered
        let mut covered = FixedBitSet::with_capacity(self.num_elements);
        for img in selected.ones() {
            covered.union_with(&self.image_elements[img]);
        }

        if covered.count_ones(..) < self.num_elements {
            // Need to add images to cover remaining elements
            for elem in 0..self.num_elements {
                if covered.contains(elem) {
                    continue;
                }
                // Find the best image covering this element
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

        // Redundancy removal: remove images whose elements are all covered by others
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

    /// Compute objective values for a repaired solution.
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
                    // An element is "clear" if at least one selected image clears it
                    let mut clear = FixedBitSet::with_capacity(self.num_elements);
                    for img in selected.ones() {
                        clear.union_with(&clear_elements[img]);
                    }
                    // Cloudy area = sum of areas for non-clear elements
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
// Result conversion
// ---------------------------------------------------------------------------

/// Convert a moors population gene matrix back to `BitsetEncodedSolution`s,
/// applying repair and deduplication via Pareto dominance.
fn convert_population_to_solutions<P, const D: usize>(
    genes: &Array2<f64>,
    problem: &P,
    data: &SimsProblemData<D>,
) -> Vec<BitsetEncodedSolution<P, D>>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    use pareto::{HasObjectives, MoSolution};

    let mut solutions = Vec::with_capacity(genes.nrows());

    for row_idx in 0..genes.nrows() {
        let row = genes.row(row_idx);
        let row_slice = row.as_slice().unwrap_or(&[]);
        let selected = data.repair_genes(row_slice);
        let selected_vec: Vec<usize> = selected.ones().collect();

        if selected_vec.is_empty() {
            continue;
        }

        let sol = BitsetEncodedSolution::from_selected_images(&selected_vec, problem);

        // Check feasibility
        if !problem.is_set_cover(&sol) {
            continue;
        }

        solutions.push(sol);
    }

    // Filter to non-dominated solutions
    let mut archive: Vec<BitsetEncodedSolution<P, D>> = Vec::new();
    for sol in solutions {
        let dominated_by_existing = archive
            .iter()
            .any(|existing| existing.dominates(sol.objectives()));
        if dominated_by_existing {
            continue;
        }
        // Remove any existing solutions dominated by this new one
        archive.retain(|existing| !sol.dominates(existing.objectives()));
        archive.push(sol);
    }

    archive
}

// ---------------------------------------------------------------------------
// NSGA-II via moors
// ---------------------------------------------------------------------------

/// Run NSGA-II from the `moors` crate on a SIMS problem instance.
///
/// Uses binary operators (uniform/single-point/two-point crossover + bit-flip
/// mutation) with greedy repair in the fitness function to ensure feasibility.
///
/// # Type Parameters
/// - `P`: The problem type implementing `SetCoverProblem<D>`
/// - `D`: Number of objectives (compile-time constant)
///
/// # Returns
/// A tuple of (non-dominated solutions, explored solutions data).
pub fn run_moors_nsga2<P, const D: usize>(
    problem: &P,
    config: MoorsConfig,
    _max_duration: Duration,
) -> (Vec<BitsetEncodedSolution<P, D>>, ExploredSolutionsData<D>)
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    let _span = info_span!("moors_nsga2").entered();
    let start = std::time::Instant::now();

    let data = Arc::new(SimsProblemData::<D>::from_problem(problem));
    let num_images = data.num_images;

    info!(
        num_images = num_images,
        num_elements = data.num_elements,
        population_size = config.population_size,
        num_iterations = config.num_iterations,
        crossover_type = ?config.crossover_type,
        "Starting moors NSGA-II"
    );

    let num_iterations = config.num_iterations;

    // Build fitness function: repair genes, compute objectives (all minimization)
    let data_fitness = Arc::clone(&data);
    let fitness_fn = move |pop_genes: &Array2<f64>| -> Array2<f64> {
        let nrows = pop_genes.nrows();
        let mut result = Array2::<f64>::zeros((nrows, D));
        for i in 0..nrows {
            let row = pop_genes.row(i);
            let row_slice = row.as_slice().unwrap_or(&[]);
            let selected = data_fitness.repair_genes(row_slice);
            let objs = data_fitness.compute_objectives(&selected);
            for j in 0..D {
                result[[i, j]] = objs[j];
            }
        }
        result
    };

    // Build and run NSGA-II
    let result = match config.crossover_type {
        MoorsCrossoverType::Uniform => {
            run_nsga2_with_crossover::<P, D, moors::operators::UniformBinaryCrossover>(
                problem,
                &config,
                num_iterations,
                num_images,
                fitness_fn,
                moors::operators::UniformBinaryCrossover::new(),
                &data,
            )
        }
        MoorsCrossoverType::SinglePoint => {
            run_nsga2_with_crossover::<P, D, moors::operators::SinglePointBinaryCrossover>(
                problem,
                &config,
                num_iterations,
                num_images,
                fitness_fn,
                moors::operators::SinglePointBinaryCrossover::new(),
                &data,
            )
        }
        MoorsCrossoverType::TwoPoint => {
            run_nsga2_with_crossover::<P, D, moors::operators::TwoPointBinaryCrossover>(
                problem,
                &config,
                num_iterations,
                num_images,
                fitness_fn,
                moors::operators::TwoPointBinaryCrossover,
                &data,
            )
        }
    };

    let elapsed = start.elapsed();
    info!(
        archive_size = result.len(),
        elapsed_ms = elapsed.as_millis(),
        "moors NSGA-II completed"
    );

    let explored = ExploredSolutionsData::<D>::new(problem.max_objectives());
    (result, explored)
}

/// Internal helper: run moors NSGA-II with a specific crossover operator type.
fn run_nsga2_with_crossover<P, const D: usize, Cross>(
    problem: &P,
    config: &MoorsConfig,
    num_iterations: usize,
    num_images: usize,
    fitness_fn: impl Fn(&Array2<f64>) -> Array2<f64> + Send + Sync + 'static,
    crossover: Cross,
    data: &SimsProblemData<D>,
) -> Vec<BitsetEncodedSolution<P, D>>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
    Cross: moors::operators::CrossoverOperator + 'static,
{
    use moors::{
        ExactDuplicatesCleaner, NoConstraints, Nsga2Builder, RandomSamplingBinary,
        operators::BitFlipMutation,
    };

    let build_result = Nsga2Builder::default()
        .fitness_fn(fitness_fn)
        .constraints_fn(NoConstraints)
        .sampler(RandomSamplingBinary::new())
        .crossover(crossover)
        .mutation(BitFlipMutation::new(config.bitflip_probability))
        .duplicates_cleaner(ExactDuplicatesCleaner::new())
        .num_vars(num_images)
        .population_size(config.population_size)
        .crossover_rate(config.crossover_rate)
        .mutation_rate(config.mutation_rate)
        .num_offsprings(config.num_offsprings)
        .num_iterations(num_iterations)
        .seed(config.seed)
        .build();

    let mut algo = match build_result {
        Ok(a) => a,
        Err(e) => {
            tracing::error!("Failed to build moors NSGA-II: {e:?}");
            return Vec::new();
        }
    };

    if let Err(e) = algo.run() {
        tracing::error!("moors NSGA-II run failed: {e:?}");
        return Vec::new();
    }

    // Extract population genes from the final population
    let population = match &algo.population {
        Some(pop) => pop,
        None => {
            tracing::warn!("moors NSGA-II returned no population");
            return Vec::new();
        }
    };

    convert_population_to_solutions(&population.genes, problem, data)
}

// ---------------------------------------------------------------------------
// SPEA-2 via moors
// ---------------------------------------------------------------------------

/// Run SPEA-2 from the `moors` crate on a SIMS problem instance.
///
/// SPEA-2 (Strength Pareto Evolutionary Algorithm 2) uses a fine-grained fitness
/// assignment based on dominance strength and density estimation via k-th nearest
/// neighbor distance. It maintains an external archive of non-dominated solutions.
///
/// # Returns
/// A tuple of (non-dominated solutions, explored solutions data).
pub fn run_moors_spea2<P, const D: usize>(
    problem: &P,
    config: MoorsConfig,
    _max_duration: Duration,
) -> (Vec<BitsetEncodedSolution<P, D>>, ExploredSolutionsData<D>)
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    let _span = info_span!("moors_spea2").entered();
    let start = std::time::Instant::now();

    let data = Arc::new(SimsProblemData::<D>::from_problem(problem));
    let num_images = data.num_images;

    info!(
        num_images = num_images,
        num_elements = data.num_elements,
        population_size = config.population_size,
        num_iterations = config.num_iterations,
        "Starting moors SPEA-2"
    );

    let num_iterations = config.num_iterations;

    // Build fitness function
    let data_fitness = Arc::clone(&data);
    let fitness_fn = move |pop_genes: &Array2<f64>| -> Array2<f64> {
        let nrows = pop_genes.nrows();
        let mut result = Array2::<f64>::zeros((nrows, D));
        for i in 0..nrows {
            let row = pop_genes.row(i);
            let row_slice = row.as_slice().unwrap_or(&[]);
            let selected = data_fitness.repair_genes(row_slice);
            let objs = data_fitness.compute_objectives(&selected);
            for j in 0..D {
                result[[i, j]] = objs[j];
            }
        }
        result
    };

    let result = match config.crossover_type {
        MoorsCrossoverType::Uniform => {
            run_spea2_with_crossover::<P, D, moors::operators::UniformBinaryCrossover>(
                problem,
                &config,
                num_iterations,
                num_images,
                fitness_fn,
                moors::operators::UniformBinaryCrossover::new(),
                &data,
            )
        }
        MoorsCrossoverType::SinglePoint => {
            run_spea2_with_crossover::<P, D, moors::operators::SinglePointBinaryCrossover>(
                problem,
                &config,
                num_iterations,
                num_images,
                fitness_fn,
                moors::operators::SinglePointBinaryCrossover::new(),
                &data,
            )
        }
        MoorsCrossoverType::TwoPoint => {
            run_spea2_with_crossover::<P, D, moors::operators::TwoPointBinaryCrossover>(
                problem,
                &config,
                num_iterations,
                num_images,
                fitness_fn,
                moors::operators::TwoPointBinaryCrossover,
                &data,
            )
        }
    };

    let elapsed = start.elapsed();
    info!(
        archive_size = result.len(),
        elapsed_ms = elapsed.as_millis(),
        "moors SPEA-2 completed"
    );

    let explored = ExploredSolutionsData::<D>::new(problem.max_objectives());
    (result, explored)
}

/// Internal helper: run moors SPEA-2 with a specific crossover operator type.
fn run_spea2_with_crossover<P, const D: usize, Cross>(
    problem: &P,
    config: &MoorsConfig,
    num_iterations: usize,
    num_images: usize,
    fitness_fn: impl Fn(&Array2<f64>) -> Array2<f64> + Send + Sync + 'static,
    crossover: Cross,
    data: &SimsProblemData<D>,
) -> Vec<BitsetEncodedSolution<P, D>>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
    Cross: moors::operators::CrossoverOperator + 'static,
{
    use moors::{
        ExactDuplicatesCleaner, NoConstraints, RandomSamplingBinary, Spea2Builder,
        operators::BitFlipMutation,
    };

    let build_result = Spea2Builder::default()
        .fitness_fn(fitness_fn)
        .constraints_fn(NoConstraints)
        .sampler(RandomSamplingBinary::new())
        .crossover(crossover)
        .mutation(BitFlipMutation::new(config.bitflip_probability))
        .duplicates_cleaner(ExactDuplicatesCleaner::new())
        .num_vars(num_images)
        .population_size(config.population_size)
        .crossover_rate(config.crossover_rate)
        .mutation_rate(config.mutation_rate)
        .num_offsprings(config.num_offsprings)
        .num_iterations(num_iterations)
        .seed(config.seed)
        .build();

    let mut algo = match build_result {
        Ok(a) => a,
        Err(e) => {
            tracing::error!("Failed to build moors SPEA-2: {e:?}");
            return Vec::new();
        }
    };

    if let Err(e) = algo.run() {
        tracing::error!("moors SPEA-2 run failed: {e:?}");
        return Vec::new();
    }

    let population = match &algo.population {
        Some(pop) => pop,
        None => {
            tracing::warn!("moors SPEA-2 returned no population");
            return Vec::new();
        }
    };

    convert_population_to_solutions(&population.genes, problem, data)
}

// ---------------------------------------------------------------------------
// AGE-MOEA via moors (bonus algorithm)
// ---------------------------------------------------------------------------

/// Run AGE-MOEA from the `moors` crate on a SIMS problem instance.
///
/// AGE-MOEA (Adaptive Geometry Estimation based MOEA) is a recent algorithm
/// that adapts its geometry estimation to the shape of the Pareto front.
///
/// # Returns
/// A tuple of (non-dominated solutions, explored solutions data).
pub fn run_moors_age_moea<P, const D: usize>(
    problem: &P,
    config: MoorsConfig,
    _max_duration: Duration,
) -> (Vec<BitsetEncodedSolution<P, D>>, ExploredSolutionsData<D>)
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
{
    let _span = info_span!("moors_age_moea").entered();
    let start = std::time::Instant::now();

    let data = Arc::new(SimsProblemData::<D>::from_problem(problem));
    let num_images = data.num_images;

    info!(
        num_images = num_images,
        num_elements = data.num_elements,
        population_size = config.population_size,
        num_iterations = config.num_iterations,
        "Starting moors AGE-MOEA"
    );

    let num_iterations = config.num_iterations;

    let data_fitness = Arc::clone(&data);
    let fitness_fn = move |pop_genes: &Array2<f64>| -> Array2<f64> {
        let nrows = pop_genes.nrows();
        let mut result = Array2::<f64>::zeros((nrows, D));
        for i in 0..nrows {
            let row = pop_genes.row(i);
            let row_slice = row.as_slice().unwrap_or(&[]);
            let selected = data_fitness.repair_genes(row_slice);
            let objs = data_fitness.compute_objectives(&selected);
            for j in 0..D {
                result[[i, j]] = objs[j];
            }
        }
        result
    };

    let result = run_age_moea_with_crossover::<P, D, moors::operators::UniformBinaryCrossover>(
        problem,
        &config,
        num_iterations,
        num_images,
        fitness_fn,
        moors::operators::UniformBinaryCrossover::new(),
        &data,
    );

    let elapsed = start.elapsed();
    info!(
        archive_size = result.len(),
        elapsed_ms = elapsed.as_millis(),
        "moors AGE-MOEA completed"
    );

    let explored = ExploredSolutionsData::<D>::new(problem.max_objectives());
    (result, explored)
}

/// Internal helper: run moors AGE-MOEA with a specific crossover operator type.
fn run_age_moea_with_crossover<P, const D: usize, Cross>(
    problem: &P,
    config: &MoorsConfig,
    num_iterations: usize,
    num_images: usize,
    fitness_fn: impl Fn(&Array2<f64>) -> Array2<f64> + Send + Sync + 'static,
    crossover: Cross,
    data: &SimsProblemData<D>,
) -> Vec<BitsetEncodedSolution<P, D>>
where
    P: SetCoverProblem<D> + Clone + Send + Sync,
    Cross: moors::operators::CrossoverOperator + 'static,
{
    use moors::{
        AgeMoeaBuilder, ExactDuplicatesCleaner, NoConstraints, RandomSamplingBinary,
        operators::BitFlipMutation,
    };

    let build_result = AgeMoeaBuilder::default()
        .fitness_fn(fitness_fn)
        .constraints_fn(NoConstraints)
        .sampler(RandomSamplingBinary::new())
        .crossover(crossover)
        .mutation(BitFlipMutation::new(config.bitflip_probability))
        .duplicates_cleaner(ExactDuplicatesCleaner::new())
        .num_vars(num_images)
        .population_size(config.population_size)
        .crossover_rate(config.crossover_rate)
        .mutation_rate(config.mutation_rate)
        .num_offsprings(config.num_offsprings)
        .num_iterations(num_iterations)
        .seed(config.seed)
        .build();

    let mut algo = match build_result {
        Ok(a) => a,
        Err(e) => {
            tracing::error!("Failed to build moors AGE-MOEA: {e:?}");
            return Vec::new();
        }
    };

    // AGE-MOEA's survival operator in `moors` can panic on certain problem
    // structures (e.g. when all repaired solutions collapse to the same
    // front geometry).  Wrap in `catch_unwind` so callers get an empty
    // archive instead of an unrecoverable abort.
    let run_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| algo.run()));
    match run_result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            tracing::error!("moors AGE-MOEA run failed: {e:?}");
            return Vec::new();
        }
        Err(_panic) => {
            tracing::warn!(
                "moors AGE-MOEA panicked internally (known edge-case in the crate); returning empty archive"
            );
            return Vec::new();
        }
    }

    let population = match &algo.population {
        Some(pop) => pop,
        None => {
            tracing::warn!("moors AGE-MOEA returned no population");
            return Vec::new();
        }
    };

    convert_population_to_solutions(&population.genes, problem, data)
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

            // Verify objectives match recalculated values
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
    fn test_moors_nsga2_2d_small() {
        let problem = load_problem_2d("lagos_nigeria_30.dzn");
        let config = MoorsConfig {
            population_size: 50,
            num_offsprings: 25,
            num_iterations: 50,
            ..Default::default()
        };
        let (archive, _explored) = run_moors_nsga2(&problem, config, Duration::from_secs(30));
        validate_archive(&archive, &problem, "moors_nsga2_2d_lagos_30");
    }

    #[test]
    fn test_moors_nsga2_4d_small() {
        let problem = load_problem_4d("lagos_nigeria_30.dzn");
        let config = MoorsConfig {
            population_size: 50,
            num_offsprings: 25,
            num_iterations: 50,
            ..Default::default()
        };
        let (archive, _explored) = run_moors_nsga2(&problem, config, Duration::from_secs(30));
        validate_archive(&archive, &problem, "moors_nsga2_4d_lagos_30");
    }

    #[test]
    fn test_moors_spea2_2d_small() {
        let problem = load_problem_2d("lagos_nigeria_30.dzn");
        let config = MoorsConfig {
            population_size: 50,
            num_offsprings: 25,
            num_iterations: 50,
            ..Default::default()
        };
        let (archive, _explored) = run_moors_spea2(&problem, config, Duration::from_secs(30));
        validate_archive(&archive, &problem, "moors_spea2_2d_lagos_30");
    }

    #[test]
    fn test_moors_age_moea_2d_small() {
        let problem = load_problem_2d("lagos_nigeria_30.dzn");
        let config = MoorsConfig {
            population_size: 50,
            num_offsprings: 25,
            num_iterations: 50,
            ..Default::default()
        };
        let (archive, _explored) = run_moors_age_moea(&problem, config, Duration::from_secs(30));
        validate_archive(&archive, &problem, "moors_age_moea_2d_lagos_30");
    }

    #[test]
    fn test_moors_nsga2_crossover_variants() {
        let problem = load_problem_2d("lagos_nigeria_30.dzn");

        for crossover_type in [
            MoorsCrossoverType::Uniform,
            MoorsCrossoverType::SinglePoint,
            MoorsCrossoverType::TwoPoint,
        ] {
            let config = MoorsConfig {
                population_size: 40,
                num_offsprings: 20,
                num_iterations: 30,
                crossover_type,
                ..Default::default()
            };
            let (archive, _) = run_moors_nsga2(&problem, config, Duration::from_secs(30));
            validate_archive(
                &archive,
                &problem,
                &format!("moors_nsga2_{crossover_type:?}"),
            );
        }
    }

    #[test]
    fn test_moors_determinism() {
        let problem = load_problem_2d("lagos_nigeria_30.dzn");
        let config = MoorsConfig {
            population_size: 40,
            num_offsprings: 20,
            num_iterations: 30,
            seed: 12345,
            ..Default::default()
        };

        let (archive1, _) = run_moors_nsga2(&problem, config.clone(), Duration::from_secs(30));
        let (archive2, _) = run_moors_nsga2(&problem, config, Duration::from_secs(30));

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
