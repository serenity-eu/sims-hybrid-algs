//! # Satellite Image Mosaic Selection Problem (SIMS)
//!
//! This module implements the satellite image mosaic selection problem. The problem involves selecting satellite images to cover a universe
//! of points while optimizing multiple objectives.
//!
//! ## Problem Description
//!
//! Given:
//! - A set of satellite images, each covering certain universe points
//! - Each image has clouds covering some of the universe points
//! - Each image has a cost, resolution, and incidence angle
//! - Each universe point has an area
//!
//! Objectives:
//! 1. Minimize total cost of selected images (equation 9)
//! 2. Minimize cloudy area using partial set cover model (equations 14-15)
//! 3. Minimize sum of minimum resolution for each part (equations 10-12)
//! 4. Minimize maximum incidence angle (equation 13)
//!
//! Constraints:
//! - Set covering: every universe point must be covered by at least one selected image (equation 8)
//! - Cloud coverage modeled as partial set cover problem
//! - Complex resolution constraints with auxiliary binary variables for min-min optimization
//!
//! ## MILP Model Implementation
//!
//! This implementation follows the mathematical model described in the MILP documentation:
//! - Uses auxiliary variables z_{kj} for resolution optimization
//! - Models clouds as separate entities with `y_c` variables
//! - Implements Big-M constraints for linearization

use crate::{
    model::{MultiObjectiveProblem, VariableType},
    ObjectiveDirection,
};
use fixedbitset::FixedBitSet;
use good_lp::{constraint, Expression, Variable};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// Objectives available for the SIMS problem
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimsObjective {
    /// Minimize total cost (objective 0)
    MinCost,
    /// Minimize cloudy/uncovered area (objective 1)
    CloudCoverage,
    /// Minimize sum of minimum resolutions (objective 2)
    MinResolution,
    /// Minimize maximum incidence angle (objective 3)
    MaxIncidenceAngle,
}

/// Configuration for the Satellite Image Selection Problem
#[derive(Debug, Clone)]
pub struct SimsInstance {
    /// Number of images available
    pub num_images: usize,
    /// Number of universe points to cover
    pub universe_size: usize,
    /// Number of distinct cloud entities
    pub num_clouds: usize,
    /// Maximum cloud area threshold
    pub max_cloud_area: i32,
    /// For each image: set of universe points it covers
    pub images: Vec<HashSet<usize>>,
    /// For each image: set of cloud entities it can cover
    pub image_clouds: Vec<HashSet<usize>>,
    /// List of actual cloud IDs (universe element indices that are clouds)
    pub cloud_ids: Vec<usize>,
    /// Cost of each image
    pub costs: Vec<f64>,
    /// Area of each universe point
    pub areas: Vec<f64>,
    /// Area of each cloud entity (indexed by universe element ID)
    pub cloud_areas: Vec<f64>,
    /// Resolution of each image (higher is better)
    pub resolution: Vec<f64>,
    /// Incidence angle of each image
    pub incidence_angle: Vec<f64>,
}

impl SimsInstance {
    /// Create a new SIMS configuration
    #[must_use]
    pub fn new(
        num_images: usize,
        universe_size: usize,
        num_clouds: usize,
        max_cloud_area: i32,
    ) -> Self {
        Self {
            num_images,
            universe_size,
            num_clouds,
            max_cloud_area,
            images: vec![HashSet::new(); num_images],
            image_clouds: vec![HashSet::new(); num_images],
            cloud_ids: vec![],
            costs: vec![0.0; num_images],
            areas: vec![1.0; universe_size],
            cloud_areas: vec![1.0; universe_size],
            resolution: vec![1.0; num_images],
            incidence_angle: vec![0.0; num_images],
        }
    }

    /// Build a `SimsInstance` directly from raw parsed `.dzn`-shaped arrays
    /// (e.g. from `sims_dzn::RawSimsData`).
    ///
    /// `clouds[i]` follows the raw `.dzn` convention: the fragments that are
    /// cloudy *in image `i` itself* (a subset of `images[i]`) — not the
    /// "which fragments can image `i` substitute a cloud with" relation that
    /// this computes into [`SimsInstance::image_clouds`].
    ///
    /// Fragment membership tests use [`FixedBitSet`] rather than
    /// `HashSet<usize>` — with universe sizes up to ~100k fragments, a
    /// per-image bitset is only a few KB (comfortably cache-resident) and a
    /// membership test is a single indexed load, versus a `HashSet` lookup's
    /// hash computation and bucket probe. The per-image `image_clouds` entry
    /// depends only on that image's own coverage/cloud bitsets plus the
    /// shared "cloudy somewhere" bitset, so the whole computation is
    /// embarrassingly parallel across images (via `rayon`).
    #[must_use]
    pub fn from_raw_refs(
        images: &[Vec<usize>],
        clouds: &[Vec<usize>],
        costs: &[i64],
        areas: &[i64],
        resolution: &[i64],
        incidence_angle: &[i64],
        universe_size: usize,
        max_cloud_area: i64,
    ) -> Self {
        let num_images = images.len();

        let to_bitset = |members: &Vec<usize>| {
            let mut bs = FixedBitSet::with_capacity(universe_size);
            for &f in members {
                bs.insert(f);
            }
            bs
        };
        let image_bitsets: Vec<FixedBitSet> = images.par_iter().map(to_bitset).collect();
        let cloud_bitsets: Vec<FixedBitSet> = clouds.par_iter().map(to_bitset).collect();

        // Fragments cloudy in *some* image: the union of every image's own
        // cloud bitset. Shared read-only across the parallel step below.
        let mut all_cloudy = FixedBitSet::with_capacity(universe_size);
        for bs in &cloud_bitsets {
            all_cloudy.union_with(bs);
        }

        // image_clouds[j]: fragments cloudy somewhere, covered by j, and
        // clear in j — independent per image, hence parallelizable.
        let image_clouds: Vec<HashSet<usize>> = images
            .par_iter()
            .zip(cloud_bitsets.par_iter())
            .map(|(image, j_clouds)| {
                image
                    .iter()
                    .copied()
                    .filter(|&f| all_cloudy.contains(f) && !j_clouds.contains(f))
                    .collect()
            })
            .collect();

        let cloud_ids: Vec<usize> = all_cloudy.ones().collect();
        let mut cloud_areas = vec![0.0_f64; universe_size];
        #[allow(
            clippy::cast_precision_loss,
            reason = "Fragment areas fit comfortably in f64's 52-bit mantissa for this problem's scale"
        )]
        for &c in &cloud_ids {
            cloud_areas[c] = areas.get(c).copied().unwrap_or(0) as f64;
        }

        let images_hashsets: Vec<HashSet<usize>> = image_bitsets
            .into_par_iter()
            .map(|bs| bs.ones().collect())
            .collect();

        #[allow(
            clippy::cast_precision_loss,
            reason = "Cost/resolution/incidence-angle values fit comfortably in f64's 52-bit mantissa"
        )]
        let (costs_f64, resolution_f64, incidence_angle_f64): (
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
        ) = (
            costs.iter().map(|&c| c as f64).collect(),
            resolution.iter().map(|&r| r as f64).collect(),
            incidence_angle.iter().map(|&a| a as f64).collect(),
        );
        #[allow(
            clippy::cast_precision_loss,
            reason = "Fragment areas fit comfortably in f64's 52-bit mantissa for this problem's scale"
        )]
        let areas_f64: Vec<f64> = areas.iter().map(|&a| a as f64).collect();
        #[allow(
            clippy::cast_possible_truncation,
            reason = "max_cloud_area is a threshold well within i32 range for this problem's scale"
        )]
        let max_cloud_area_i32 = max_cloud_area as i32;

        Self {
            num_images,
            universe_size,
            num_clouds: cloud_ids.len(),
            max_cloud_area: max_cloud_area_i32,
            images: images_hashsets,
            image_clouds,
            cloud_ids,
            costs: costs_f64,
            areas: areas_f64,
            cloud_areas,
            resolution: resolution_f64,
            incidence_angle: incidence_angle_f64,
        }
    }

    /// Set which universe points an image covers
    pub fn set_image_coverage(&mut self, image_idx: usize, coverage: HashSet<usize>) {
        self.images[image_idx] = coverage;
    }

    /// Set which cloud entities an image can cover
    pub fn set_cloud_coverage(&mut self, image_idx: usize, clouds: HashSet<usize>) {
        self.image_clouds[image_idx] = clouds;
    }

    /// Set the cost of an image
    pub fn set_cost(&mut self, image_idx: usize, cost: f64) {
        self.costs[image_idx] = cost;
    }

    /// Set the area of a universe point
    pub fn set_area(&mut self, universe_idx: usize, area: f64) {
        self.areas[universe_idx] = area;
    }

    /// Set the area of a cloud entity
    pub fn set_cloud_area(&mut self, cloud_idx: usize, area: f64) {
        self.cloud_areas[cloud_idx] = area;
    }

    /// Set the resolution of an image
    pub fn set_resolution(&mut self, image_idx: usize, resolution: f64) {
        self.resolution[image_idx] = resolution;
    }

    /// Set the incidence angle of an image
    pub fn set_incidence_angle(&mut self, image_idx: usize, angle: f64) {
        self.incidence_angle[image_idx] = angle;
    }

    /// Calculate heuristic nadir bounds for specified objectives
    /// This matches Python's `get_nadir_bound_estimation()` method
    #[must_use]
    pub fn calculate_nadir_heuristic(&self, objectives: &[SimsObjective]) -> Vec<f64> {
        objectives
            .iter()
            .map(|obj| {
                match obj {
                    SimsObjective::MinCost => {
                        // Worst case: all images selected
                        self.costs.iter().sum()
                    }
                    SimsObjective::CloudCoverage => {
                        // Worst case: all cloud areas covered
                        self.areas.iter().sum()
                    }
                    SimsObjective::MinResolution => {
                        // Worst case: sum of max resolution per universe point
                        let mut resolution_parts_max = vec![0.0; self.universe_size];
                        for (idx, image) in self.images.iter().enumerate() {
                            for &u in image {
                                if resolution_parts_max[u] < self.resolution[idx] {
                                    resolution_parts_max[u] = self.resolution[idx];
                                }
                            }
                        }
                        resolution_parts_max.iter().sum()
                    }
                    SimsObjective::MaxIncidenceAngle => {
                        // Worst case: maximum incidence angle
                        self.incidence_angle
                            .iter()
                            .fold(0.0_f64, |acc, &x| acc.max(x))
                    }
                }
            })
            .collect()
    }

    /// Get the set of images that contain each universe point (`L_k` in the MILP model)
    fn get_universe_point_images(&self) -> Vec<HashSet<usize>> {
        let mut point_images = vec![HashSet::new(); self.universe_size];

        for (u, image_set) in point_images.iter_mut().enumerate() {
            for i in 0..self.num_images {
                if self.images[i].contains(&u) {
                    image_set.insert(i);
                }
            }
        }

        point_images
    }

    /// Create decision variables for the MILP model.
    ///
    /// When `objectives` is `Some`, only the auxiliary variables required by those
    /// objectives are created:
    /// - `r_k` and `z_{k,j}` are created only when [`SimsObjective::MinResolution`] is present
    /// - `maxf` is created only when [`SimsObjective::MaxIncidenceAngle`] is present
    /// - `x_i` (image selection) and `y_c` (cloud coverage) are always created
    ///
    /// When `objectives` is `None` all variables are created (backwards-compatible).
    ///
    /// Returns the created `Variable` handles directly (alongside populating
    /// `problem.var_map` for name-based lookup/introspection), so downstream
    /// constraint-building never needs to reconstruct a variable's name with
    /// `format!` and re-look it up in a string-keyed map — on a model with
    /// hundreds of thousands of variables, that `format!` + `HashMap<String, _>`
    /// round trip inside a hot loop dominates construction time far more than
    /// the algorithmic complexity of the loops themselves.
    fn create_variables(
        &self,
        problem: &mut MultiObjectiveProblem,
        objectives: Option<&HashSet<SimsObjective>>,
    ) -> CreatedVariables {
        let needs_resolution =
            objectives.is_none_or(|set| set.contains(&SimsObjective::MinResolution));
        let needs_incidence_angle =
            objectives.is_none_or(|set| set.contains(&SimsObjective::MaxIncidenceAngle));

        // x_i: binary variables for each image (equation 8)
        let x_vars: Vec<Variable> = (0..self.num_images)
            .map(|i| problem.add_variable(format!("x_{i}"), VariableType::Binary))
            .collect();

        // y_c: binary variables for cloud coverage (equations 14-15)
        // Use actual cloud IDs (universe element indices) as variable indices
        let y_vars: HashMap<usize, Variable> = self
            .cloud_ids
            .iter()
            .map(|&cloud_id| {
                let var = problem.add_variable(format!("y_{cloud_id}"), VariableType::Binary);
                (cloud_id, var)
            })
            .collect();

        // r_k and z_{kj}: only needed for MinResolution objective. Only pay
        // for `get_universe_point_images` (O(universe_size * num_images))
        // when it's actually going to be used.
        let (point_images, r_vars, z_vars) = if needs_resolution {
            let point_images = self.get_universe_point_images();

            // r_k: auxiliary variables for minimum resolution of each universe point (equations 11-12)
            let max_resolution = self.resolution.iter().fold(0.0_f64, |acc, &x| acc.max(x));
            let r_vars: Vec<Variable> = (0..self.universe_size)
                .map(|k| {
                    problem.add_variable(
                        format!("r_{k}"),
                        VariableType::Continuous {
                            min: Some(0.0),
                            max: Some(max_resolution),
                        },
                    )
                })
                .collect();

            // z_{kj}: auxiliary binary variables for resolution constraints (equation 10)
            let z_vars: Vec<HashMap<usize, Variable>> = point_images
                .iter()
                .enumerate()
                .map(|(k, image_set)| {
                    image_set
                        .iter()
                        .map(|&j| {
                            let var =
                                problem.add_variable(format!("z_{k}_{j}"), VariableType::Binary);
                            (j, var)
                        })
                        .collect()
                })
                .collect();

            log::debug!(
                "Created resolution variables: {} r_k + {} z_{{k,j}}",
                self.universe_size,
                point_images.iter().map(HashSet::len).sum::<usize>()
            );
            (point_images, r_vars, z_vars)
        } else {
            log::info!(
                "Skipping resolution variables (r_k, z_{{k,j}}) and the O(universe*images) \
                 point-images precompute: MinResolution not in objectives"
            );
            (Vec::new(), Vec::new(), Vec::new())
        };

        // maxf: only needed for MaxIncidenceAngle objective
        let maxf_var = if needs_incidence_angle {
            let max_incidence = self
                .incidence_angle
                .iter()
                .fold(0.0_f64, |acc, &x| acc.max(x));
            Some(problem.add_variable(
                "maxf".to_string(),
                VariableType::Continuous {
                    min: Some(0.0),
                    max: Some(max_incidence),
                },
            ))
        } else {
            log::info!("Skipping maxf variable: MaxIncidenceAngle not in objectives");
            None
        };

        log::info!(
            "Model variables: {} total ({} x_i, {} y_c{}{})",
            problem.var_map.len(),
            self.num_images,
            self.cloud_ids.len(),
            if needs_resolution {
                format!(
                    ", {} r_k, {} z_{{k,j}}",
                    r_vars.len(),
                    z_vars.iter().map(HashMap::len).sum::<usize>()
                )
            } else {
                String::new()
            },
            if maxf_var.is_some() { ", 1 maxf" } else { "" },
        );

        CreatedVariables {
            point_images,
            x_vars,
            y_vars,
            r_vars,
            z_vars,
            maxf_var,
        }
    }

    /// Add set covering and cloud coverage constraints.
    ///
    /// Builds one running [`Expression`] per universe fragment / cloud id by
    /// iterating each image's own coverage list once — O(total coverage
    /// entries) — instead of testing `self.images[i].contains(&k)` for every
    /// `(k, i)` combination, which is O(`universe_size` * `num_images`) and
    /// was the dominant cost on large instances (tens of millions of
    /// membership tests, the vast majority of which never hit).
    fn add_coverage_constraints(
        &self,
        problem: &mut MultiObjectiveProblem,
        x_vars: &[Variable],
        y_vars: &HashMap<usize, Variable>,
    ) {
        // Constraint 1: Set covering (equation 8)
        // Sum_{i: k in P_i} x_i >= 1, for all k in U
        //
        // `Expression` is backed by a `HashMap<Variable, f64>` (good_lp), so
        // growing each of the `universe_size` expressions incrementally from
        // empty triggers repeated hash-table reallocation across ~100k+
        // small maps. A cheap first pass over the (already in-hand) coverage
        // lists computes each point's degree so every expression's map can
        // be allocated once at its final size.
        let mut degree = vec![0usize; self.universe_size];
        for image in &self.images {
            for &k in image {
                degree[k] += 1;
            }
        }
        let mut coverage_exprs: Vec<Expression> =
            degree.into_iter().map(Expression::with_capacity).collect();
        for (i, image) in self.images.iter().enumerate() {
            let x_i = x_vars[i];
            for &k in image {
                coverage_exprs[k] += x_i;
            }
        }
        for expr in coverage_exprs {
            problem.add_constraint(constraint!(expr >= 1.0));
        }

        // Constraint 2: Cloud coverage constraints (equation 15)
        // Lower bound: Sum_{i: c in P_{ic}} x_i >= y_c, for all c in C
        // Upper bound: Sum_{i: c in P_{ic}} x_i <= y_c * num_images, for all c in C
        // The upper bound ensures that if any image covering cloud c is selected,
        // then y_c must be 1 (cloud is covered). This prevents the optimizer from
        // setting y_c = 0 when images are selected during maximization.
        // Special case: if cloud is uncoverable (no images can cover it), set y_c = 0
        let mut cloud_degree = vec![0usize; self.universe_size];
        for clouds in &self.image_clouds {
            for &c in clouds {
                cloud_degree[c] += 1;
            }
        }
        let mut cloud_coverage_exprs: HashMap<usize, Expression> =
            HashMap::with_capacity(self.cloud_ids.len());
        for (i, clouds) in self.image_clouds.iter().enumerate() {
            let x_i = x_vars[i];
            for &c in clouds {
                *cloud_coverage_exprs
                    .entry(c)
                    .or_insert_with(|| Expression::with_capacity(cloud_degree[c])) += x_i;
            }
        }

        #[allow(
            clippy::cast_precision_loss,
            reason = "Number of images is always much less than 2^53, so f64 conversion is safe"
        )]
        let num_images_f64 = self.num_images as f64;

        for &c in &self.cloud_ids {
            let Some(&y_var) = y_vars.get(&c) else {
                continue;
            };
            match cloud_coverage_exprs.remove(&c) {
                Some(expr) => {
                    // Lower bound: Sum of covering images >= y_c
                    problem.add_constraint(constraint!(expr.clone() >= y_var));
                    // Upper bound: Sum of covering images <= y_c * num_images
                    // This forces y_c = 1 if any covering image is selected
                    let upper_bound_expr = y_var * num_images_f64;
                    problem.add_constraint(constraint!(expr <= upper_bound_expr));
                }
                None => {
                    // Uncoverable cloud: must set y_c = 0 to avoid infeasibility
                    problem.add_constraint(constraint!(y_var == 0.0));
                }
            }
        }
    }

    /// Add resolution-related constraints (equations 10-12).
    ///
    /// Only call this when `MinResolution` is among the active objectives, since
    /// the `r_k` and `z_{k,j}` variables must already exist in `problem`.
    fn add_resolution_constraints(
        &self,
        problem: &mut MultiObjectiveProblem,
        point_images: &[HashSet<usize>],
        x_vars: &[Variable],
        r_vars: &[Variable],
        z_vars: &[HashMap<usize, Variable>],
    ) {
        let mut num_constraints: usize = 0;

        // Constraint 3: Resolution auxiliary variables constraints (equation 10)
        // Sum_{j in L_k} z_{kj} = |L_k| - 1, for each k
        for (k, image_set) in point_images.iter().enumerate() {
            if image_set.len() > 1 {
                let mut z_sum = Expression::from(0.0);
                for &j in image_set {
                    if let Some(&z_var) = z_vars[k].get(&j) {
                        z_sum += z_var;
                    }
                }
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "Image set length is always much less than 2^53, so casting is safe"
                )]
                let target = (image_set.len() - 1) as f64;
                problem.add_constraint(constraint!(z_sum == target));
                num_constraints += 1;
            }
        }

        // Constraint 4: Resolution minimum constraints (equation 11)
        // r_k >= (x_j * R_j + B(1 - x_j)) - 2B * z_{kj}, for all j in L_k
        let big_b = self.resolution.iter().fold(0.0_f64, |acc, &x| acc.max(x)) * 10.0; // B > max resolution

        for (k, image_set) in point_images.iter().enumerate() {
            let r_var = r_vars[k];
            for &j in image_set {
                if let Some(&z_var) = z_vars[k].get(&j) {
                    let x_var = x_vars[j];
                    // r_k >= x_j * R_j + B * (1 - x_j) - 2B * z_{kj}
                    // Simplified: r_k >= (R_j - B) * x_j + B - 2B * z_{kj}
                    let coeff = self.resolution[j] - big_b;
                    problem.add_constraint(constraint!(
                        r_var >= coeff * x_var + big_b - 2.0 * big_b * z_var
                    ));
                    num_constraints += 1;
                }
            }
        }

        log::debug!("Added {num_constraints} resolution constraints");
    }

    /// Add incidence-angle constraints (equation 13).
    ///
    /// Only call this when `MaxIncidenceAngle` is among the active objectives,
    /// since the `maxf` variable must already exist in `problem`.
    fn add_incidence_angle_constraints(
        &self,
        problem: &mut MultiObjectiveProblem,
        x_vars: &[Variable],
        maxf_var: Variable,
    ) {
        // Constraint 5: Maximum incidence angle constraints (equation 13)
        // maxf >= x_i * F_i, for all i
        for (i, &x_var) in x_vars.iter().enumerate() {
            problem.add_constraint(constraint!(maxf_var >= self.incidence_angle[i] * x_var));
        }
        log::debug!("Added {} incidence angle constraints", x_vars.len());
    }

    /// Add objective functions
    /// If objectives is None, adds all objectives. Otherwise, only adds the specified objectives.
    fn add_objectives(
        &self,
        problem: &mut MultiObjectiveProblem,
        objectives: Option<&HashSet<SimsObjective>>,
        vars: &CreatedVariables,
    ) {
        // Helper to check if objective should be added
        let should_add = |obj: SimsObjective| objectives.is_none_or(|set| set.contains(&obj));

        // Objective 1: Minimize total cost (equation 9)
        if should_add(SimsObjective::MinCost) {
            let mut cost_expr = Expression::from(0.0);
            for (i, &x_var) in vars.x_vars.iter().enumerate() {
                cost_expr += self.costs[i] * x_var;
            }
            problem.add_objective(cost_expr, ObjectiveDirection::Minimize);
        }

        // Objective 2: Minimize cloudy area (equation 14)
        if should_add(SimsObjective::CloudCoverage) {
            let total_cloud_area: f64 = self
                .cloud_ids
                .iter()
                .map(|&cloud_id| self.cloud_areas[cloud_id])
                .sum();

            let mut cloud_area_expr = Expression::from(total_cloud_area);
            for &cloud_id in &self.cloud_ids {
                if let Some(&y_var) = vars.y_vars.get(&cloud_id) {
                    cloud_area_expr -= self.cloud_areas[cloud_id] * y_var;
                }
            }
            problem.add_objective(cloud_area_expr, ObjectiveDirection::Minimize);
        }

        // Objective 3: Minimize sum of minimum resolutions (equation 12)
        if should_add(SimsObjective::MinResolution) {
            let mut resolution_expr = Expression::from(0.0);
            for &r_var in &vars.r_vars {
                resolution_expr += r_var;
            }
            problem.add_objective(resolution_expr, ObjectiveDirection::Minimize);
        }

        // Objective 4: Minimize maximum incidence angle (equation 13)
        if should_add(SimsObjective::MaxIncidenceAngle) {
            if let Some(maxf_var) = vars.maxf_var {
                problem.add_objective(Expression::from(maxf_var), ObjectiveDirection::Minimize);
            }
        }
    }
}

/// `Variable` handles created by [`SimsInstance::create_variables`], returned
/// directly so constraint-building code can index into them instead of
/// reconstructing a variable's name and looking it up in a string-keyed map.
struct CreatedVariables {
    point_images: Vec<HashSet<usize>>,
    x_vars: Vec<Variable>,
    y_vars: HashMap<usize, Variable>,
    r_vars: Vec<Variable>,
    /// `z_vars[k]` maps image index `j` to the `z_{k,j}` variable, for each
    /// universe fragment `k`. Empty when resolution variables weren't created.
    z_vars: Vec<HashMap<usize, Variable>>,
    maxf_var: Option<Variable>,
}

/// Create a SIMS multi-objective optimization problem following the MILP model
///
/// # Arguments
/// * `config` - The SIMS instance configuration
/// * `objectives` - Optional set of objectives to include. If None, all objectives are added.
#[must_use]
pub fn create_sims_problem(config: &SimsInstance) -> MultiObjectiveProblem {
    create_sims_problem_with_objectives(config, None)
}

/// Create a SIMS problem with specific objectives
///
/// When `objectives` is `Some`, only the variables, constraints, and objective
/// functions required by those objectives are created.  This can dramatically
/// reduce model size — for example, omitting [`SimsObjective::MinResolution`]
/// avoids creating `O(universe_size * num_images)` auxiliary binary variables
/// (`z_{k,j}`) and their Big-M constraints.
#[must_use]
#[allow(
    clippy::implicit_hasher,
    reason = "Public API uses standard HashSet for simplicity - users can convert if needed"
)]
pub fn create_sims_problem_with_objectives(
    config: &SimsInstance,
    objectives: Option<&HashSet<SimsObjective>>,
) -> MultiObjectiveProblem {
    let needs_resolution = objectives.is_none_or(|set| set.contains(&SimsObjective::MinResolution));
    let needs_incidence_angle =
        objectives.is_none_or(|set| set.contains(&SimsObjective::MaxIncidenceAngle));

    let mut problem = MultiObjectiveProblem::new();

    // Create variables (conditionally based on objectives)
    let vars = config.create_variables(&mut problem, objectives);

    // Always add set-covering and cloud-coverage constraints
    config.add_coverage_constraints(&mut problem, &vars.x_vars, &vars.y_vars);

    // Add resolution constraints only when MinResolution is active
    if needs_resolution {
        config.add_resolution_constraints(
            &mut problem,
            &vars.point_images,
            &vars.x_vars,
            &vars.r_vars,
            &vars.z_vars,
        );
    } else {
        log::info!("Skipping resolution constraints: MinResolution not in objectives");
    }

    // Add incidence angle constraints only when MaxIncidenceAngle is active
    if needs_incidence_angle {
        if let Some(maxf_var) = vars.maxf_var {
            config.add_incidence_angle_constraints(&mut problem, &vars.x_vars, maxf_var);
        }
    } else {
        log::info!("Skipping incidence angle constraints: MaxIncidenceAngle not in objectives");
    }

    log::info!(
        "SIMS MILP model: {} variables, {} constraints, {} objectives",
        problem.var_map.len(),
        problem.constraints.len(),
        objectives.map_or(4, HashSet::len),
    );

    // Add objectives (all or subset)
    config.add_objectives(&mut problem, objectives, &vars);

    problem
}

/// Create a sample SIMS problem for testing
#[must_use]
pub fn create_sample_sims_problem() -> MultiObjectiveProblem {
    let mut config = SimsInstance::new(5, 4, 3, 100); // 5 images, 4 universe points, 3 clouds

    // Set up image coverage (which universe points each image covers)
    config.set_image_coverage(0, [0, 1].iter().copied().collect());
    config.set_image_coverage(1, [1, 2].iter().copied().collect());
    config.set_image_coverage(2, [2, 3].iter().copied().collect());
    config.set_image_coverage(3, [0, 3].iter().copied().collect());
    config.set_image_coverage(4, [0, 1, 2, 3].iter().copied().collect());

    // Set up cloud coverage (which cloud entities each image can cover)
    config.set_cloud_coverage(0, std::iter::once(0).collect()); // Image 0 can cover cloud 0
    config.set_cloud_coverage(1, std::iter::once(1).collect()); // Image 1 can cover cloud 1
    config.set_cloud_coverage(2, [1, 2].iter().copied().collect()); // Image 2 can cover clouds 1, 2
    config.set_cloud_coverage(3, [0, 2].iter().copied().collect()); // Image 3 can cover clouds 0, 2
    config.set_cloud_coverage(4, [0, 1, 2].iter().copied().collect()); // Image 4 can cover all clouds

    // Set costs
    config.set_cost(0, 10.0);
    config.set_cost(1, 15.0);
    config.set_cost(2, 20.0);
    config.set_cost(3, 12.0);
    config.set_cost(4, 25.0);

    // Set universe point areas
    config.set_area(0, 5.0);
    config.set_area(1, 3.0);
    config.set_area(2, 4.0);
    config.set_area(3, 6.0);

    // Set cloud areas
    config.set_cloud_area(0, 2.0);
    config.set_cloud_area(1, 1.5);
    config.set_cloud_area(2, 3.0);

    // Set resolutions
    config.set_resolution(0, 1.0);
    config.set_resolution(1, 2.0);
    config.set_resolution(2, 3.0);
    config.set_resolution(3, 1.5);
    config.set_resolution(4, 2.5);

    // Set incidence angles
    config.set_incidence_angle(0, 10.0);
    config.set_incidence_angle(1, 20.0);
    config.set_incidence_angle(2, 15.0);
    config.set_incidence_angle(3, 25.0);
    config.set_incidence_angle(4, 30.0);

    create_sims_problem(&config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sims_config_creation() {
        let config = SimsInstance::new(3, 2, 2, 50); // 3 images, 2 universe points, 2 clouds
        assert_eq!(config.num_images, 3);
        assert_eq!(config.universe_size, 2);
        assert_eq!(config.num_clouds, 2);
        assert_eq!(config.max_cloud_area, 50);
    }

    #[test]
    fn test_universe_point_images() {
        let mut config = SimsInstance::new(2, 2, 1, 100); // 2 images, 2 universe points, 1 cloud

        // Image 0 covers universe points 0, 1
        config.set_image_coverage(0, [0, 1].iter().copied().collect());
        // Image 1 covers universe point 1 only
        config.set_image_coverage(1, std::iter::once(&1).copied().collect());

        let point_images = config.get_universe_point_images();

        // Universe point 0 is covered by image 0 only
        assert!(point_images[0].contains(&0));
        assert!(!point_images[0].contains(&1));

        // Universe point 1 is covered by both images 0 and 1
        assert!(point_images[1].contains(&0));
        assert!(point_images[1].contains(&1));
    }

    #[test]
    fn test_sample_problem_creation() {
        let problem = create_sample_sims_problem();

        // Should have variables for images (x_i), clouds (y_c), resolutions (r_k), z variables, and maxf
        assert!(!problem.var_map.is_empty());

        // Should have 4 objectives as per MILP model
        assert_eq!(problem.objectives.len(), 4);

        // Should have constraints
        assert!(!problem.constraints.is_empty());
    }

    /// Helper to build the sample config used by several tests below.
    fn sample_config() -> SimsInstance {
        let mut config = SimsInstance::new(5, 4, 3, 100);

        config.set_image_coverage(0, [0, 1].iter().copied().collect());
        config.set_image_coverage(1, [1, 2].iter().copied().collect());
        config.set_image_coverage(2, [2, 3].iter().copied().collect());
        config.set_image_coverage(3, [0, 3].iter().copied().collect());
        config.set_image_coverage(4, [0, 1, 2, 3].iter().copied().collect());

        config.set_cloud_coverage(0, std::iter::once(0).collect());
        config.set_cloud_coverage(1, std::iter::once(1).collect());
        config.set_cloud_coverage(2, [1, 2].iter().copied().collect());
        config.set_cloud_coverage(3, [0, 2].iter().copied().collect());
        config.set_cloud_coverage(4, [0, 1, 2].iter().copied().collect());

        config.cloud_ids = vec![0, 1, 2];

        for (i, cost) in [10.0, 15.0, 20.0, 12.0, 25.0].iter().enumerate() {
            config.set_cost(i, *cost);
        }
        for (k, area) in [5.0, 3.0, 4.0, 6.0].iter().enumerate() {
            config.set_area(k, *area);
        }
        for (c, area) in [(0, 2.0), (1, 1.5), (2, 3.0)] {
            config.set_cloud_area(c, area);
        }
        for (i, res) in [1.0, 2.0, 3.0, 1.5, 2.5].iter().enumerate() {
            config.set_resolution(i, *res);
        }
        for (i, angle) in [10.0, 20.0, 15.0, 25.0, 30.0].iter().enumerate() {
            config.set_incidence_angle(i, *angle);
        }

        config
    }

    #[test]
    fn test_conditional_variables_all_objectives() {
        // With all objectives: model should contain r_k, z_{k,j}, and maxf
        let config = sample_config();
        let problem = create_sims_problem(&config);

        // r_k variables (4 universe points)
        for k in 0..config.universe_size {
            assert!(
                problem.var_map.contains_key(&format!("r_{k}")),
                "r_{k} should exist when all objectives are active"
            );
        }
        // maxf variable
        assert!(
            problem.var_map.contains_key("maxf"),
            "maxf should exist when all objectives are active"
        );
        // z_{k,j} at least some should exist
        assert!(
            problem.var_map.contains_key("z_0_0") || problem.var_map.contains_key("z_0_4"),
            "z_{{k,j}} variables should exist when all objectives are active"
        );
        assert_eq!(problem.objectives.len(), 4);
    }

    #[test]
    fn test_conditional_variables_cost_and_cloud_only() {
        // With only cost + cloud_coverage: no r_k, no z_{k,j}, no maxf
        let config = sample_config();
        let objectives: HashSet<SimsObjective> =
            [SimsObjective::MinCost, SimsObjective::CloudCoverage]
                .into_iter()
                .collect();
        let problem = create_sims_problem_with_objectives(&config, Some(&objectives));

        // r_k must NOT exist
        for k in 0..config.universe_size {
            assert!(
                !problem.var_map.contains_key(&format!("r_{k}")),
                "r_{k} should NOT exist when MinResolution is not requested"
            );
        }
        // maxf must NOT exist
        assert!(
            !problem.var_map.contains_key("maxf"),
            "maxf should NOT exist when MaxIncidenceAngle is not requested"
        );
        // x_i must still exist
        for i in 0..config.num_images {
            assert!(
                problem.var_map.contains_key(&format!("x_{i}")),
                "x_{i} must always exist"
            );
        }
        // y_c must still exist
        for &c in &config.cloud_ids {
            assert!(
                problem.var_map.contains_key(&format!("y_{c}")),
                "y_{c} must always exist"
            );
        }
        assert_eq!(problem.objectives.len(), 2);
    }

    #[test]
    fn test_conditional_variables_with_resolution() {
        // With cost + resolution: r_k and z_{k,j} should exist, maxf should NOT
        let config = sample_config();
        let objectives: HashSet<SimsObjective> =
            [SimsObjective::MinCost, SimsObjective::MinResolution]
                .into_iter()
                .collect();
        let problem = create_sims_problem_with_objectives(&config, Some(&objectives));

        for k in 0..config.universe_size {
            assert!(
                problem.var_map.contains_key(&format!("r_{k}")),
                "r_{k} should exist when MinResolution is requested"
            );
        }
        assert!(
            !problem.var_map.contains_key("maxf"),
            "maxf should NOT exist when MaxIncidenceAngle is not requested"
        );
        assert_eq!(problem.objectives.len(), 2);
    }

    #[test]
    fn test_conditional_variables_with_incidence_angle() {
        // With cost + incidence angle: maxf should exist, r_k/z_{k,j} should NOT
        let config = sample_config();
        let objectives: HashSet<SimsObjective> =
            [SimsObjective::MinCost, SimsObjective::MaxIncidenceAngle]
                .into_iter()
                .collect();
        let problem = create_sims_problem_with_objectives(&config, Some(&objectives));

        assert!(
            problem.var_map.contains_key("maxf"),
            "maxf should exist when MaxIncidenceAngle is requested"
        );
        for k in 0..config.universe_size {
            assert!(
                !problem.var_map.contains_key(&format!("r_{k}")),
                "r_{k} should NOT exist when MinResolution is not requested"
            );
        }
        assert_eq!(problem.objectives.len(), 2);
    }

    #[test]
    fn test_model_size_reduction() {
        // Verify that omitting resolution dramatically reduces variable count
        let config = sample_config();

        let full_problem = create_sims_problem(&config);
        let full_var_count = full_problem.var_map.len();
        let full_constraint_count = full_problem.constraints.len();

        let objectives: HashSet<SimsObjective> =
            [SimsObjective::MinCost, SimsObjective::CloudCoverage]
                .into_iter()
                .collect();
        let reduced_problem = create_sims_problem_with_objectives(&config, Some(&objectives));
        let reduced_var_count = reduced_problem.var_map.len();
        let reduced_constraint_count = reduced_problem.constraints.len();

        println!("Full model:    {full_var_count} vars, {full_constraint_count} constraints");
        println!("Reduced model: {reduced_var_count} vars, {reduced_constraint_count} constraints");

        // Reduced model should have strictly fewer variables (no r_k, z_{kj}, maxf)
        assert!(
            reduced_var_count < full_var_count,
            "Reduced model ({reduced_var_count} vars) should have fewer variables \
             than full model ({full_var_count} vars)"
        );
        // Reduced model should have strictly fewer constraints
        assert!(
            reduced_constraint_count < full_constraint_count,
            "Reduced model ({reduced_constraint_count} constraints) should have fewer \
             constraints than full model ({full_constraint_count} constraints)"
        );
    }
}
