use rand::Rng;
use rand::distr::Open01;

// Forward declaration for Problem to avoid circular imports
use crate::problem::Problem;

/// Concrete objective types for the SIMS problem (legacy enum for specific use cases)
#[derive(Clone)]
pub enum ObjectiveType {
    TotalCost,
    CloudyArea,
    MinResolution,
    MaxIncidenceAngle,
}

impl ObjectiveType {
    /// Unique identifier for this objective
    #[must_use]
    pub const fn id(&self) -> &'static str {
        match self {
            Self::TotalCost => "total_cost",
            Self::CloudyArea => "cloudy_area",
            Self::MinResolution => "min_resolution",
            Self::MaxIncidenceAngle => "max_incidence_angle",
        }
    }

    /// Human-readable name for this objective
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::TotalCost => "Total Cost",
            Self::CloudyArea => "Cloudy Area",
            Self::MinResolution => "Minimum Resolution Sum",
            Self::MaxIncidenceAngle => "Maximum Incidence Angle",
        }
    }

    /// Whether this is a minimization objective (true) or maximization (false)
    #[must_use]
    pub const fn is_minimization(&self) -> bool {
        true // Both cost and cloudy area are minimization objectives
    }
}

/// Generic trait for defining objectives in the SIMS problem
pub trait ObjectiveDefinition<const D: usize>: Send + Sync + std::fmt::Debug {
    /// Unique identifier for this objective
    fn id(&self) -> &'static str;

    /// Human-readable name
    fn name(&self) -> &'static str;

    /// Whether this is minimization (true) or maximization (false)
    fn is_minimization(&self) -> bool;

    /// Calculate the objective value for a solution
    fn calculate_value(&self, solution: &dyn SolutionEvaluator<D>, problem: &Problem<D>) -> u64;

    /// Calculate the delta when adding/removing an image
    fn calculate_delta(
        &self,
        image_index: usize,
        is_selected: bool,
        solution: &dyn SolutionEvaluator<D>,
        problem: &Problem<D>,
    ) -> i64;

    /// Get maximum possible value for this objective
    fn max_value(&self, problem: &Problem<D>) -> u64;

    /// Get the index of this objective in the objectives array
    fn objective_index(&self) -> usize;
}

/// Trait for solution evaluation - provides access to solution state for objective calculation
pub trait SolutionEvaluator<const D: usize>: crate::solution::ImageSet {
    /// Get clear parts counts for each element
    fn clear_parts_counts(&self) -> &[usize];

    /// Get element coverage counts
    fn element_coverage(&self) -> &[usize];
}

/// Total cost objective implementation
#[derive(Debug, Clone)]
pub struct TotalCostObjective {
    pub index: usize,
}

impl<const D: usize> ObjectiveDefinition<D> for TotalCostObjective {
    fn id(&self) -> &'static str {
        "total_cost"
    }

    fn name(&self) -> &'static str {
        "Total Cost"
    }

    fn is_minimization(&self) -> bool {
        true
    }

    fn objective_index(&self) -> usize {
        self.index
    }

    fn calculate_value(&self, solution: &dyn SolutionEvaluator<D>, problem: &Problem<D>) -> u64 {
        solution
            .selected_images()
            .iter()
            .map(|&i| problem.images[i].cost())
            .sum()
    }

    fn calculate_delta(
        &self,
        image_index: usize,
        is_selected: bool,
        _solution: &dyn SolutionEvaluator<D>,
        problem: &Problem<D>,
    ) -> i64 {
        if is_selected {
            -(problem.images[image_index].cost() as i64)
        } else {
            problem.images[image_index].cost() as i64
        }
    }

    fn max_value(&self, problem: &Problem<D>) -> u64 {
        problem.images.iter().map(super::problem::Image::cost).sum()
    }
}

/// Cloudy area objective implementation
#[derive(Debug, Clone)]
pub struct CloudyAreaObjective {
    pub index: usize,
}

impl<const D: usize> ObjectiveDefinition<D> for CloudyAreaObjective {
    fn id(&self) -> &'static str {
        "cloudy_area"
    }

    fn name(&self) -> &'static str {
        "Cloudy Area"
    }

    fn is_minimization(&self) -> bool {
        true
    }

    fn objective_index(&self) -> usize {
        self.index
    }

    fn calculate_value(&self, solution: &dyn SolutionEvaluator<D>, problem: &Problem<D>) -> u64 {
        let mut clear_elements = vec![false; problem.universe.len()];

        // Mark elements that are clear in selected images
        for &image_index in &solution.selected_images() {
            for &clear_part in &problem.images[image_index].clear_parts {
                clear_elements[clear_part] = true;
            }
        }

        // Calculate cloudy area (elements not covered by clear parts)
        clear_elements
            .iter()
            .enumerate()
            .filter_map(|(element_index, &is_clear)| {
                if is_clear {
                    None
                } else {
                    Some(problem.universe[element_index].area)
                }
            })
            .sum()
    }

    fn calculate_delta(
        &self,
        image_index: usize,
        is_selected: bool,
        solution: &dyn SolutionEvaluator<D>,
        problem: &Problem<D>,
    ) -> i64 {
        let mut cloudy_area_delta: i64 = 0;

        if is_selected {
            // Removing image - check if any clear parts become uncovered
            for &clear_part in &problem.images[image_index].clear_parts {
                // If this is the last image with clear part covering the element, add element area to delta
                if solution.clear_parts_counts()[clear_part] == 1 {
                    cloudy_area_delta += problem.universe[clear_part].area as i64;
                }
            }
        } else {
            // Adding image - check if any clear parts become newly covered
            for &clear_part in &problem.images[image_index].clear_parts {
                // If this is the first image with clear part covering the element, subtract element area from delta
                if solution.clear_parts_counts()[clear_part] == 0 {
                    cloudy_area_delta -= problem.universe[clear_part].area as i64;
                }
            }
        }

        cloudy_area_delta
    }

    fn max_value(&self, problem: &Problem<D>) -> u64 {
        problem.total_area()
    }
}

/// Minimum resolution sum objective implementation.
///
/// For each universe element, finds the best (minimum) resolution among selected images that cover it,
/// then sums these minimum resolutions across all universe elements.
#[derive(Debug, Clone)]
pub struct MinResolutionObjective {
    pub index: usize,
}

impl<const D: usize> ObjectiveDefinition<D> for MinResolutionObjective {
    fn id(&self) -> &'static str {
        "min_resolution"
    }

    fn name(&self) -> &'static str {
        "Minimum Resolution Sum"
    }

    fn is_minimization(&self) -> bool {
        true
    }

    fn objective_index(&self) -> usize {
        self.index
    }

    fn calculate_value(&self, solution: &dyn SolutionEvaluator<D>, problem: &Problem<D>) -> u64 {
        let mut min_resolution_sum = 0u64;

        for element_index in 0..problem.universe.len() {
            // Find minimum resolution among selected images that cover this element
            let min_resolution = solution
                .selected_images()
                .iter()
                .filter(|&&image_index| problem.images[image_index].parts.contains(&element_index))
                .map(|&image_index| problem.raw_instance.resolution[image_index])
                .min()
                .unwrap_or(0);

            min_resolution_sum += min_resolution;
        }

        min_resolution_sum
    }

    fn calculate_delta(
        &self,
        image_index: usize,
        is_selected: bool,
        solution: &dyn SolutionEvaluator<D>,
        problem: &Problem<D>,
    ) -> i64 {
        let mut delta = 0i64;
        let image_resolution = problem.raw_instance.resolution[image_index];

        for &element_index in &problem.images[image_index].parts {
            // Current minimum resolution for this element
            let current_min = solution
                .selected_images()
                .iter()
                .filter(|&&idx| {
                    idx != image_index && problem.images[idx].parts.contains(&element_index)
                })
                .map(|&idx| problem.raw_instance.resolution[idx])
                .min();

            if is_selected {
                // Removing image - check if this was providing the minimum resolution
                if let Some(new_min) = current_min {
                    if image_resolution < new_min {
                        // This image was providing the minimum, delta increases
                        delta += (new_min - image_resolution) as i64;
                    }
                } else {
                    // This was the only image covering this element
                    delta -= image_resolution as i64;
                }
            } else {
                // Adding image - check if this provides a better minimum
                if let Some(current_min_val) = current_min {
                    if image_resolution < current_min_val {
                        // This image provides better resolution, delta decreases
                        delta -= (current_min_val - image_resolution) as i64;
                    }
                } else {
                    // This is the first image to cover this element
                    delta += image_resolution as i64;
                }
            }
        }

        delta
    }

    fn max_value(&self, problem: &Problem<D>) -> u64 {
        // Maximum possible value: worst resolution for each element
        let max_resolution = problem
            .raw_instance
            .resolution
            .iter()
            .max()
            .copied()
            .unwrap_or(0);
        max_resolution * (problem.universe.len() as u64)
    }
}

/// Maximum incidence angle objective implementation
/// Finds the worst (maximum) incidence angle among all selected images.
#[derive(Debug, Clone)]
pub struct MaxIncidenceAngleObjective {
    pub index: usize,
}

impl<const D: usize> ObjectiveDefinition<D> for MaxIncidenceAngleObjective {
    fn id(&self) -> &'static str {
        "max_incidence_angle"
    }

    fn name(&self) -> &'static str {
        "Maximum Incidence Angle"
    }

    fn is_minimization(&self) -> bool {
        true
    }

    fn objective_index(&self) -> usize {
        self.index
    }

    fn calculate_value(&self, solution: &dyn SolutionEvaluator<D>, problem: &Problem<D>) -> u64 {
        solution
            .selected_images()
            .iter()
            .map(|&image_index| problem.raw_instance.incidence_angle[image_index])
            .max()
            .unwrap_or(0)
    }

    fn calculate_delta(
        &self,
        image_index: usize,
        is_selected: bool,
        solution: &dyn SolutionEvaluator<D>,
        problem: &Problem<D>,
    ) -> i64 {
        let image_angle = problem.raw_instance.incidence_angle[image_index];

        if is_selected {
            // Removing image
            let current_max = solution
                .selected_images()
                .iter()
                .map(|&idx| problem.raw_instance.incidence_angle[idx])
                .max()
                .unwrap_or(0);

            if image_angle == current_max {
                // This image had the maximum angle, find new maximum
                let new_max = solution
                    .selected_images()
                    .iter()
                    .filter(|&&idx| idx != image_index)
                    .map(|&idx| problem.raw_instance.incidence_angle[idx])
                    .max()
                    .unwrap_or(0);
                return (new_max as i64) - (current_max as i64);
            }
            // Image wasn't the maximum, no change
            0
        } else {
            // Adding image
            let current_max = solution
                .selected_images()
                .iter()
                .map(|&idx| problem.raw_instance.incidence_angle[idx])
                .max()
                .unwrap_or(0);

            if image_angle > current_max {
                // New image has higher angle, becomes new maximum
                (image_angle as i64) - (current_max as i64)
            } else {
                // New image doesn't affect maximum
                0
            }
        }
    }

    fn max_value(&self, problem: &Problem<D>) -> u64 {
        problem
            .raw_instance
            .incidence_angle
            .iter()
            .max()
            .copied()
            .unwrap_or(0)
    }
}

#[must_use]
pub fn generate_weights<const D: usize>() -> [f32; D] {
    let mut weights = [0.0f32; D];
    let mut remaining = 1.0_f32;

    for weight in weights.iter_mut().take(D - 1) {
        let random_weight: f32 = rand::rng().sample(Open01);
        *weight = random_weight * remaining;
        remaining -= random_weight * remaining;
    }
    weights[D - 1] = remaining; // Last weight gets the remainder
    weights
}

#[must_use]
pub fn weighted_sum<const D: usize>(objectives: &pareto::Objectives<D>, weights: &[f32; D]) -> f32 {
    objectives
        .iter()
        .zip(weights.iter())
        .map(|(&obj, &weight)| obj as f32 * weight)
        .sum()
}

#[must_use]
pub fn weighted_sum_f32<const D: usize>(values: &[f32; D], weights: &[f32; D]) -> f32 {
    values
        .iter()
        .zip(weights.iter())
        .map(|(&val, &weight)| val * weight)
        .sum()
}

pub fn apply_delta<const D: usize>(objectives: &mut pareto::Objectives<D>, deltas: &[i64; D]) {
    for (i, &delta) in deltas.iter().enumerate() {
        if delta < 0 {
            objectives[i] -= delta.unsigned_abs();
        } else {
            objectives[i] += delta as u64;
        }
    }
}
