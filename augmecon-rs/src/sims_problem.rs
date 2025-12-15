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
use good_lp::{constraint, Expression};
use std::collections::HashSet;

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
        objectives.iter().map(|obj| {
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
                    self.incidence_angle.iter().fold(0.0_f64, |acc, &x| acc.max(x))
                }
            }
        }).collect()
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

    fn create_variables(&self, problem: &mut MultiObjectiveProblem) -> Vec<HashSet<usize>> {
        // x_i: binary variables for each image (equation 8)
        for i in 0..self.num_images {
            let var_name = format!("x_{i}");
            problem.add_variable(var_name, VariableType::Binary);
        }

        // y_c: binary variables for cloud coverage (equations 14-15)
        // Use actual cloud IDs (universe element indices) as variable indices
        for &cloud_id in &self.cloud_ids {
            let var_name = format!("y_{cloud_id}");
            problem.add_variable(var_name, VariableType::Binary);
        }

        // r_k: auxiliary variables for minimum resolution of each universe point (equations 11-12)
        // Upper bound is set to max resolution to prevent unbounded maximization during nadir calculation
        let max_resolution = self.resolution.iter().fold(0.0_f64, |acc, &x| acc.max(x));
        for k in 0..self.universe_size {
            let var_name = format!("r_{k}");
            problem.add_variable(
                var_name,
                VariableType::Continuous {
                    min: Some(0.0),
                    max: Some(max_resolution),
                },
            );
        }

        // z_{kj}: auxiliary binary variables for resolution constraints (equation 10)
        let point_images = self.get_universe_point_images();
        for (k, image_set) in point_images.iter().enumerate() {
            for &j in image_set {
                let var_name = format!("z_{k}_{j}");
                problem.add_variable(var_name, VariableType::Binary);
            }
        }

        // maxf: auxiliary variable for maximum incidence angle (equation 13)
        // Upper bound is set to max incidence angle to prevent unbounded maximization during nadir calculation
        let max_incidence = self.incidence_angle.iter().fold(0.0_f64, |acc, &x| acc.max(x));
        problem.add_variable(
            "maxf".to_string(),
            VariableType::Continuous {
                min: Some(0.0),
                max: Some(max_incidence),
            },
        );

        point_images
    }

    /// Add set covering and cloud coverage constraints
    fn add_coverage_constraints(&self, problem: &mut MultiObjectiveProblem) {
        // Constraint 1: Set covering (equation 8)
        // Sum_{i: k in P_i} x_i >= 1, for all k in U
        for k in 0..self.universe_size {
            let mut coverage_expr = Expression::from(0.0);
            for i in 0..self.num_images {
                if self.images[i].contains(&k) {
                    if let Some(&var) = problem.var_map.get(&format!("x_{i}")) {
                        coverage_expr += var;
                    }
                }
            }
            problem.add_constraint(constraint!(coverage_expr >= 1.0));
        }

        // Constraint 2: Cloud coverage constraints (equation 15)
        // Lower bound: Sum_{i: c in P_{ic}} x_i >= y_c, for all c in C
        // Upper bound: Sum_{i: c in P_{ic}} x_i <= y_c * num_images, for all c in C
        // The upper bound ensures that if any image covering cloud c is selected,
        // then y_c must be 1 (cloud is covered). This prevents the optimizer from
        // setting y_c = 0 when images are selected during maximization.
        // Special case: if cloud is uncoverable (no images can cover it), set y_c = 0
        for &c in &self.cloud_ids {
            let mut cloud_coverage_expr = Expression::from(0.0);
            let mut has_covering_images = false;

            for i in 0..self.num_images {
                if self.image_clouds[i].contains(&c) {
                    if let Some(&var) = problem.var_map.get(&format!("x_{i}")) {
                        cloud_coverage_expr += var;
                        has_covering_images = true;
                    }
                }
            }

            if let Some(&y_var) = problem.var_map.get(&format!("y_{c}")) {
                if has_covering_images {
                    // Lower bound: Sum of covering images >= y_c
                    problem.add_constraint(constraint!(cloud_coverage_expr.clone() >= y_var));
                    // Upper bound: Sum of covering images <= y_c * num_images
                    // This forces y_c = 1 if any covering image is selected
                    #[allow(clippy::cast_precision_loss, reason = "Number of images is always much less than 2^53, so f64 conversion is safe")]
                    let upper_bound_expr = y_var * (self.num_images as f64);
                    problem.add_constraint(constraint!(cloud_coverage_expr <= upper_bound_expr));
                } else {
                    // Uncoverable cloud: must set y_c = 0 to avoid infeasibility
                    problem.add_constraint(constraint!(y_var == 0.0));
                }
            }
        }
    }

    /// Add resolution-related constraints
    fn add_resolution_constraints(
        &self,
        problem: &mut MultiObjectiveProblem,
        point_images: &[HashSet<usize>],
    ) {
        // Constraint 3: Resolution auxiliary variables constraints (equation 10)
        // Sum_{j in L_k} z_{kj} = |L_k| - 1, for each k
        for (k, image_set) in point_images.iter().enumerate() {
            if image_set.len() > 1 {
                let mut z_sum = Expression::from(0.0);
                for &j in image_set {
                    if let Some(&z_var) = problem.var_map.get(&format!("z_{k}_{j}")) {
                        z_sum += z_var;
                    }
                }
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "Image set length is always much less than 2^53, so casting is safe"
                )]
                let target = (image_set.len() - 1) as f64;
                problem.add_constraint(constraint!(z_sum == target));
            }
        }

        // Constraint 4: Resolution minimum constraints (equation 11)
        // r_k >= (x_j * R_j + B(1 - x_j)) - 2B * z_{kj}, for all j in L_k
        let big_b = self.resolution.iter().fold(0.0_f64, |acc, &x| acc.max(x)) * 10.0; // B > max resolution

        for (k, image_set) in point_images.iter().enumerate() {
            if let Some(&r_var) = problem.var_map.get(&format!("r_{k}")) {
                for &j in image_set {
                    if let (Some(&x_var), Some(&z_var)) = (
                        problem.var_map.get(&format!("x_{j}")),
                        problem.var_map.get(&format!("z_{k}_{j}")),
                    ) {
                        // r_k >= x_j * R_j + B * (1 - x_j) - 2B * z_{kj}
                        // Simplified: r_k >= (R_j - B) * x_j + B - 2B * z_{kj}
                        let coeff = self.resolution[j] - big_b;
                        problem.add_constraint(constraint!(
                            r_var >= coeff * x_var + big_b - 2.0 * big_b * z_var
                        ));
                    }
                }
            }
        }

        // Constraint 5: Maximum incidence angle constraints (equation 13)
        // maxf >= x_i * F_i, for all i
        if let Some(&maxf_var) = problem.var_map.get("maxf") {
            for i in 0..self.num_images {
                if let Some(&x_var) = problem.var_map.get(&format!("x_{i}")) {
                    problem
                        .add_constraint(constraint!(maxf_var >= self.incidence_angle[i] * x_var));
                }
            }
        }
    }

    /// Add objective functions
    /// If objectives is None, adds all objectives. Otherwise, only adds the specified objectives.
    fn add_objectives(&self, problem: &mut MultiObjectiveProblem, objectives: Option<&HashSet<SimsObjective>>) {
        // Helper to check if objective should be added
        let should_add = |obj: SimsObjective| objectives.is_none_or(|set| set.contains(&obj));

        // Objective 1: Minimize total cost (equation 9)
        if should_add(SimsObjective::MinCost) {
            let mut cost_expr = Expression::from(0.0);
            for i in 0..self.num_images {
                if let Some(&x_var) = problem.var_map.get(&format!("x_{i}")) {
                    cost_expr += self.costs[i] * x_var;
                }
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
                if let Some(&y_var) = problem.var_map.get(&format!("y_{cloud_id}")) {
                    cloud_area_expr -= self.cloud_areas[cloud_id] * y_var;
                }
            }
            problem.add_objective(cloud_area_expr, ObjectiveDirection::Minimize);
        }

        // Objective 3: Minimize sum of minimum resolutions (equation 12)
        if should_add(SimsObjective::MinResolution) {
            let mut resolution_expr = Expression::from(0.0);
            for k in 0..self.universe_size {
                if let Some(&r_var) = problem.var_map.get(&format!("r_{k}")) {
                    resolution_expr += r_var;
                }
            }
            problem.add_objective(resolution_expr, ObjectiveDirection::Minimize);
        }

        // Objective 4: Minimize maximum incidence angle (equation 13)
        if should_add(SimsObjective::MaxIncidenceAngle) {
            if let Some(&maxf_var) = problem.var_map.get("maxf") {
                problem.add_objective(Expression::from(maxf_var), ObjectiveDirection::Minimize);
            }
        }
    }
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
#[must_use]
#[allow(clippy::implicit_hasher, reason = "Public API uses standard HashSet for simplicity - users can convert if needed")]
pub fn create_sims_problem_with_objectives(
    config: &SimsInstance,
    objectives: Option<&HashSet<SimsObjective>>,
) -> MultiObjectiveProblem {
    let mut problem = MultiObjectiveProblem::new();

    // Create all variables and get point-image mapping
    let point_images = config.create_variables(&mut problem);

    // Add all constraints
    config.add_coverage_constraints(&mut problem);
    config.add_resolution_constraints(&mut problem, &point_images);

    // Add objectives (all or subset)
    config.add_objectives(&mut problem, objectives);

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
}
