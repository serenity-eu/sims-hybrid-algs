use core::panic;
use rand::Rng;
use rand::distr::Open01;
use std::fmt::Debug;

// Forward declaration for Problem to avoid circular imports
use crate::problem::Problem;
use crate::solution::ImageSet;

/// Concrete objective types for the SIMS problem
#[derive(Clone, Debug)]
pub enum ObjectiveType<T: ImageSet<D>, const D: usize> {
    TotalCost,
    CloudyArea,
    MinResolution,
    MaxIncidenceAngle,
    _PhantomData(std::marker::PhantomData<T>),
}

/// Trait for solution evaluation - provides access to solution state for objective calculation
pub trait SolutionEvaluator<const D: usize>: crate::solution::ImageSet<D> {
    /// Get clear parts counts for each element
    fn clear_parts_counts(&self) -> &[usize];

    /// Get element coverage counts
    fn element_coverage(&self) -> &[usize];
}

/// Implementation of `ObjectiveType`
impl<T: ImageSet<D>, const D: usize> ObjectiveType<T, D> {
    /// Returns the string identifier for this objective type.
    ///
    /// # Panics
    ///
    /// Panics if the objective type is `_PhantomData`, which should not be used in normal operations.
    #[must_use]
    pub fn id(&self) -> &'static str {
        match self {
            Self::TotalCost => "total_cost",
            Self::CloudyArea => "cloudy_area",
            Self::MinResolution => "min_resolution",
            Self::MaxIncidenceAngle => "max_incidence_angle",
            Self::_PhantomData(_) => panic!("Unknown objective type"),
        }
    }

    /// Returns the human-readable name for this objective type.
    ///
    /// # Panics
    ///
    /// Panics if the objective type is `_PhantomData`, which should not be used in normal operations.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::TotalCost => "Total Cost",
            Self::CloudyArea => "Cloudy Area",
            Self::MinResolution => "Minimum Resolution Sum",
            Self::MaxIncidenceAngle => "Maximum Incidence Angle",
            Self::_PhantomData(_) => panic!("Unknown objective type"),
        }
    }

    #[must_use]
    pub const fn is_minimization(&self) -> bool {
        true
    }

    /// Calculates the objective value for a given solution.
    ///
    /// # Panics
    ///
    /// Panics if the objective type is `_PhantomData`, which should not be used in normal operations.
    pub fn calculate_value<RT: ImageSet<D>>(&self, solution: &RT, problem: &Problem<T, D>) -> u64 {
        match self {
            Self::TotalCost { .. } => solution
                .selected_images()
                .iter()
                .map(|&i| problem.images[i].cost())
                .sum(),
            Self::CloudyArea { .. } => {
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
            Self::MinResolution { .. } => {
                let mut min_resolution_sum = 0u64;

                for element_index in 0..problem.universe.len() {
                    // Find minimum resolution among selected images that cover this element
                    let min_resolution = solution
                        .selected_images()
                        .iter()
                        .filter(|&&image_index| {
                            problem.images[image_index].parts.contains(&element_index)
                        })
                        .map(|&image_index| problem.raw_instance.resolution[image_index])
                        .min()
                        .unwrap_or(0);

                    min_resolution_sum += min_resolution;
                }

                min_resolution_sum
            }
            Self::MaxIncidenceAngle { .. } => solution
                .selected_images()
                .iter()
                .map(|&image_index| problem.raw_instance.incidence_angle[image_index])
                .max()
                .unwrap_or(0),
            Self::_PhantomData(_) => panic!("Unknown objective type"),
        }
    }

    /// Calculates the delta in objective value when adding or removing an image.
    ///
    /// # Panics
    ///
    /// Panics if the objective type is `_PhantomData`, which should not be used in normal operations.
    pub fn calculate_delta(
        &self,
        image_index: usize,
        is_selected: bool,
        solution: &T,
        problem: &Problem<T, D>,
    ) -> i64 {
        match self {
            Self::TotalCost { .. } => {
                let cost = problem.images[image_index].cost() as i64;
                if is_selected {
                    -cost // Removing image decreases cost
                } else {
                    cost // Adding image increases cost
                }
            }
            Self::CloudyArea { .. } => {
                let mut cloudy_area_delta: i64 = 0;

                if is_selected {
                    // Removing image - check if any clear parts become uncovered
                    for &clear_part in &problem.images[image_index].clear_parts {
                        // If this is the last image with clear part covering the element, add element area to delta
                        if solution.clear_parts_counts()[clear_part] == 1 {
                            let area = problem.universe[clear_part].area as i64;
                            cloudy_area_delta += area;
                            log::debug!(
                                "CloudyArea: REMOVING image {}, clear_part {} becomes uncovered (count was 1), adding area {} to delta. New delta: {}",
                                image_index, clear_part, area, cloudy_area_delta
                            );
                        } else {
                            log::debug!(
                                "CloudyArea: REMOVING image {}, clear_part {} still covered (count is {}), no delta change",
                                image_index, clear_part, solution.clear_parts_counts()[clear_part]
                            );
                        }
                    }
                } else {
                    // Adding image - check if any clear parts become covered for the first time
                    for &clear_part in &problem.images[image_index].clear_parts {
                        // If this is the first image with clear part covering the element, subtract element area from delta
                        if solution.clear_parts_counts()[clear_part] == 0 {
                            let area = problem.universe[clear_part].area as i64;
                            cloudy_area_delta -= area;
                            log::debug!(
                                "CloudyArea: ADDING image {}, clear_part {} becomes covered for first time (count was 0), subtracting area {} from delta. New delta: {}",
                                image_index, clear_part, area, cloudy_area_delta
                            );
                        } else {
                            log::debug!(
                                "CloudyArea: ADDING image {}, clear_part {} already covered (count is {}), no delta change",
                                image_index, clear_part, solution.clear_parts_counts()[clear_part]
                            );
                        }
                    }
                }

                log::debug!(
                    "CloudyArea delta calculation complete for image {}: final delta = {} ({})",
                    image_index, cloudy_area_delta, if is_selected { "REMOVING" } else { "ADDING" }
                );
                cloudy_area_delta
            }
            Self::MinResolution { .. } => {
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
            Self::MaxIncidenceAngle { .. } => {
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
            Self::_PhantomData(_) => panic!("Unknown objective type"),
        }
    }

    /// Returns the maximum possible value for this objective.
    ///
    /// # Panics
    ///
    /// Panics if the objective type is `_PhantomData`, which should not be used in normal operations.
    #[must_use]
    pub fn max_value(&self, problem: &Problem<T, D>) -> u64 {
        match self {
            Self::TotalCost => problem.images.iter().map(super::problem::Image::cost).sum(),
            Self::CloudyArea => problem.universe.iter().map(|elem| elem.area).sum(),
            Self::MinResolution => problem
                .raw_instance
                .resolution
                .iter()
                .max()
                .copied()
                .unwrap_or(0),
            Self::MaxIncidenceAngle => problem
                .raw_instance
                .incidence_angle
                .iter()
                .max()
                .copied()
                .unwrap_or(0),
            Self::_PhantomData(_) => panic!("Unknown objective type"),
        }
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
            let subtraction_amount = delta.unsigned_abs();
            if objectives[i] < subtraction_amount {
                panic!(
                    "Objective underflow detected! Attempted to subtract {} from objective[{}] = {}, which would cause underflow to u64::MAX",
                    subtraction_amount, i, objectives[i]
                );
            }
            objectives[i] -= subtraction_amount;
        } else {
            objectives[i] += delta as u64;
        }
    }
}
