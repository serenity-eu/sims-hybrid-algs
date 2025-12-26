use core::panic;
use rand::Rng;
use rand::distr::Open01;
use std::fmt::Debug;
use fixedbitset::FixedBitSet;
use std::hash::Hash;

use crate::problem::{SIMSProblemInstanceRaw, Problem};
use crate::solution::ImageSet;

/// Lightweight identifier for objective types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ObjectiveType {
    TotalCost,
    CloudyArea,
    MinResolution,
    MaxIncidenceAngle,
}

/// Implementation of `ObjectiveType`
impl ObjectiveType {
    /// Returns the string identifier for this objective type.
    #[must_use]
    pub const fn id(&self) -> &'static str {
        match self {
            Self::TotalCost => "total_cost",
            Self::CloudyArea => "cloudy_area",
            Self::MinResolution => "min_resolution",
            Self::MaxIncidenceAngle => "max_incidence_angle",
        }
    }

    /// Returns the human-readable name for this objective type.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::TotalCost => "Total Cost",
            Self::CloudyArea => "Cloudy Area",
            Self::MinResolution => "Minimum Resolution Sum",
            Self::MaxIncidenceAngle => "Maximum Incidence Angle",
        }
    }

    #[must_use]
    pub const fn is_minimization(&self) -> bool {
        true
    }
}

/// State and logic for objectives.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ObjectiveState<const D: usize> {
    TotalCost {
        costs: Vec<u64>,
        max_value: u64,
        bounds: Option<[u64; 2]>,
    },
    CloudyArea {
        clear_images: Vec<FixedBitSet>,
        areas: Vec<u64>,
        max_value: u64,
        bounds: Option<[u64; 2]>,
    },
    MinResolution {
        resolutions: Vec<u64>,
        max_value: u64,
        bounds: Option<[u64; 2]>,
    },
    MaxIncidenceAngle {
        incidence_angles: Vec<u64>,
        max_value: u64,
        bounds: Option<[u64; 2]>,
    },
}

impl<const D: usize> ObjectiveState<D> {
    /// Construct array of states from array of types and raw problem data.
    #[must_use]
    pub fn from_types_and_raw(
        objective_types: &[ObjectiveType; D],
        raw: &SIMSProblemInstanceRaw,
    ) -> [Self; D] {
        std::array::from_fn(|i| Self::from_type(objective_types[i], raw))
    }

    /// Construct state from type and raw problem data.
    /// Assumes `raw` has been normalized (0-based indices) if it comes from `Problem`.
    #[must_use]
    pub fn from_type(obj_type: ObjectiveType, raw: &SIMSProblemInstanceRaw) -> Self {
        match obj_type {
            ObjectiveType::TotalCost => {
                let costs = raw.costs.clone();
                let max_value = costs.iter().sum();
                Self::TotalCost {
                    costs,
                    max_value,
                    bounds: None,
                }
            }
            ObjectiveType::CloudyArea => {
                let universe_size = raw.universe_size;
                let clear_images = raw
                    .images
                    .iter()
                    .zip(raw.clouds.iter())
                    .map(|(image, cloud)| {
                        let mut bs = FixedBitSet::with_capacity(universe_size);
                        let cloud_set: std::collections::HashSet<_> = cloud.iter().copied().collect();
                        
                        for &elem in image {
                            if !cloud_set.contains(&elem) && elem < universe_size {
                                bs.insert(elem);
                            }
                        }
                        bs
                    })
                    .collect();

                let areas = raw.areas.clone();
                let max_value = areas.iter().sum();
                Self::CloudyArea {
                    clear_images,
                    areas,
                    max_value,
                    bounds: None,
                }
            }
            ObjectiveType::MinResolution => {
                let resolutions = raw.resolution.clone();
                let max_value = resolutions.iter().max().copied().unwrap_or(0);
                Self::MinResolution {
                    resolutions,
                    max_value,
                    bounds: None,
                }
            }
            ObjectiveType::MaxIncidenceAngle => {
                let incidence_angles = raw.incidence_angle.clone();
                let max_value = incidence_angles.iter().max().copied().unwrap_or(0);
                Self::MaxIncidenceAngle {
                    incidence_angles,
                    max_value,
                    bounds: None,
                }
            }
        }
    }

    /// Calculates the objective value for a given solution.
    pub fn calculate_value<RT: ImageSet<D>>(&self, solution: &RT, problem: &Problem<RT, D>) -> u64 {
        match self {
            Self::TotalCost { costs, .. } => solution
                .selected_images()
                .iter()
                .map(|&i| costs[i])
                .sum(),
            Self::CloudyArea {
                clear_images,
                areas,
                ..
            } => {
                let mut covered_clear = FixedBitSet::with_capacity(areas.len());
                for image_index in solution.selected_images().iter().copied() {
                    if let Some(img_clear) = clear_images.get(image_index) {
                        covered_clear.union_with(img_clear);
                    }
                }
                areas
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| !covered_clear.contains(*idx))
                    .map(|(_, &area)| area)
                    .sum()
            }
            Self::MinResolution { resolutions, .. } => {
                let mut min_resolution_sum = 0u64;

                for element_index in 0..problem.universe.len() {
                    let min_resolution = solution
                        .selected_images()
                        .iter()
                        .filter(|&&image_index| {
                            problem.images[image_index].parts.contains(&element_index)
                        })
                        .map(|&image_index| resolutions[image_index])
                        .min()
                        .unwrap_or(0);

                    min_resolution_sum += min_resolution;
                }

                min_resolution_sum
            }
            Self::MaxIncidenceAngle { incidence_angles, .. } => solution
                .selected_images()
                .iter()
                .map(|&image_index| incidence_angles[image_index])
                .max()
                .unwrap_or(0),
        }
    }

    /// Calculates the delta in objective value when adding or removing an image.
    ///
    /// # Panics
    ///
    /// Panics if called on `CloudyArea` variant - use trackers instead for efficient delta calculation.
    pub fn calculate_delta<S: ImageSet<D>>(
        &self,
        image_index: usize,
        is_selected: bool,
        solution: &S,
        problem: &Problem<S, D>,
    ) -> i64 {
        match self {
            Self::TotalCost { costs, .. } => {
                let cost = costs[image_index] as i64;
                if is_selected {
                    -cost
                } else {
                    cost
                }
            }
            Self::CloudyArea { .. } => {
                panic!("CloudyArea::calculate_delta should not be called - use trackers instead");
            }
            Self::MinResolution { resolutions, .. } => {
                let mut delta = 0i64;
                let image_resolution = resolutions[image_index];

                for &element_index in &problem.images[image_index].parts {
                    let current_min = solution
                        .selected_images()
                        .iter()
                        .filter(|&&idx| {
                            idx != image_index && problem.images[idx].parts.contains(&element_index)
                        })
                        .map(|&idx| resolutions[idx])
                        .min();

                    if is_selected {
                        // Removing the image
                        let next_min = solution
                            .selected_images()
                            .iter()
                            .filter(|&&idx| {
                                idx != image_index
                                    && problem.images[idx].parts.contains(&element_index)
                            })
                            .map(|&idx| resolutions[idx])
                            .min();

                        if let Some(new_min) = next_min {
                            delta += (image_resolution - new_min) as i64;
                        } else {
                            delta -= image_resolution as i64;
                        }
                    } else if let Some(current_min_val) = current_min {
                        if image_resolution < current_min_val {
                            delta -= (current_min_val - image_resolution) as i64;
                        }
                    } else {
                        delta += image_resolution as i64;
                    }
                }
                delta
            }
            Self::MaxIncidenceAngle { incidence_angles, .. } => {
                let image_angle = incidence_angles[image_index];

                if is_selected {
                    let current_max = solution
                        .selected_images()
                        .iter()
                        .map(|&idx| incidence_angles[idx])
                        .max()
                        .unwrap_or(0);

                    if image_angle == current_max {
                        let new_max = solution
                            .selected_images()
                            .iter()
                            .filter(|&&idx| idx != image_index)
                            .map(|&idx| incidence_angles[idx])
                            .max()
                            .unwrap_or(0);
                        (new_max as i64) - (current_max as i64)
                    } else {
                        0
                    }
                } else {
                    let current_max = solution
                        .selected_images()
                        .iter()
                        .map(|&idx| incidence_angles[idx])
                        .max()
                        .unwrap_or(0);
                    
                    if image_angle > current_max {
                        (image_angle as i64) - (current_max as i64)
                    } else {
                        0
                    }
                }
            }
        }
    }

    /// Returns the maximum possible value for this objective.
    #[must_use]
    pub const fn max_value(&self) -> u64 {
        match self {
            Self::TotalCost { max_value, .. }
            | Self::CloudyArea { max_value, .. }
            | Self::MinResolution { max_value, .. }
            | Self::MaxIncidenceAngle { max_value, .. } => *max_value,
        }
    }

    /// Returns the bounds for this objective, if set.
    #[must_use]
    pub const fn bounds(&self) -> Option<[u64; 2]> {
        match self {
            Self::TotalCost { bounds, .. }
            | Self::CloudyArea { bounds, .. }
            | Self::MinResolution { bounds, .. }
            | Self::MaxIncidenceAngle { bounds, .. } => *bounds,
        }
    }

    /// Sets the bounds for this objective.
    pub const fn set_bounds(&mut self, new_bounds: [u64; 2]) {
        match self {
            Self::TotalCost { bounds, .. }
            | Self::CloudyArea { bounds, .. }
            | Self::MinResolution { bounds, .. }
            | Self::MaxIncidenceAngle { bounds, .. } => *bounds = Some(new_bounds),
        }
    }
}

#[must_use]
pub fn generate_weights<const D: usize>() -> [f32; D] {
    let mut remaining = 1.0_f32;
    std::array::from_fn(|i| {
        if i < D - 1 {
            let random_weight: f32 = rand::rng().sample(Open01);
            let weight = random_weight * remaining;
            remaining -= weight;
            weight
        } else {
            remaining // Last weight gets the remainder
        }
    })
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
