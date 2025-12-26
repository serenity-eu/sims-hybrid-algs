//! Objective Trackers for Incremental Evaluation
//!
//! This module provides tracker implementations for maintaining objective-specific state
//! to enable efficient delta calculations during local search operations.

use fixedbitset::FixedBitSet;
use std::fmt::Debug;

use crate::problem::Problem;
use crate::solution::ImageSet;

/// Trait for a single objective tracker that maintains incremental state.
pub trait ObjectiveTracker<const D: usize>: Clone + Debug + Send + Sync {
    /// Calculate the change in objective value if we toggle an image.
    /// Does NOT modify internal state.
    fn peek_delta(
        &self,
        image_index: usize,
        is_removing: bool,
        problem: &Problem<impl ImageSet<D>, D>,
        solution: &impl ImageSet<D>,
    ) -> i64;

    /// Update internal state after a move is committed.
    fn apply(
        &mut self,
        image_index: usize,
        is_removing: bool,
        problem: &Problem<impl ImageSet<D>, D>,
    );

    /// Get the current absolute value of the objective (if tracked).
    fn value(&self) -> Option<u64>;
}

/// A collection of D trackers corresponding to D objectives.
pub trait TrackerCollection<const D: usize>: Clone + Debug + Send + Sync {
    /// Get a reference to the tracker for a specific objective index.
    fn get(&self, index: usize) -> &StandardTracker;

    /// Get a mutable reference to the tracker.
    fn get_mut(&mut self, index: usize) -> &mut StandardTracker;

    /// Initialize the collection based on the problem definition.
    fn new<T: ImageSet<D>>(problem: &Problem<T, D>) -> Self;
}

/// State for tracking Total Cost incrementally.
#[derive(Clone, Debug)]
pub struct TotalCostState {
    pub current_cost: u64,
}

/// State for tracking Cloudy Area incrementally.
#[derive(Clone, Debug)]
pub struct CloudyAreaState {
    /// Count of images covering element i with clear parts. Max ~200, so u16 is sufficient.
    pub counts: Vec<u16>,
    /// Bitset: 1 = Cloudy (count 0), 0 = Clear (count > 0)
    pub cloudy_elements: FixedBitSet,
    pub current_area: u64,
}

/// State for tracking Minimum Resolution Sum incrementally.
#[derive(Clone, Debug)]
pub struct MinResolutionState {
    /// Sorted list of resolutions covering each element.
    /// Using Vec because coverage is small (~10-50), making it faster than `BTreeMap`.
    pub element_resolutions: Vec<Vec<u64>>,
    pub current_sum: u64,
}

/// State for tracking Maximum Incidence Angle incrementally.
#[derive(Clone, Debug)]
pub struct MaxIncidenceAngleState {
    /// Sorted list of incidence angles of ALL selected images.
    /// With N=200, this is tiny and fits in L1 cache.
    pub sorted_angles: Vec<u64>,
}

/// The standard tracker enum wrapping specific state implementations.
#[derive(Clone, Debug)]
pub enum StandardTracker {
    TotalCost(TotalCostState),
    CloudyArea(CloudyAreaState),
    MinResolution(MinResolutionState),
    MaxIncidenceAngle(MaxIncidenceAngleState),
    Stateless,
}

impl<const D: usize> ObjectiveTracker<D> for StandardTracker {
    fn peek_delta(
        &self,
        image_index: usize,
        is_removing: bool,
        problem: &Problem<impl ImageSet<D>, D>,
        _solution: &impl ImageSet<D>,
    ) -> i64 {
        match self {
            Self::TotalCost(_) => {
                let cost = problem.images[image_index].cost() as i64;
                if is_removing {
                    -cost
                } else {
                    cost
                }
            }
            Self::CloudyArea(state) => {
                let mut delta: i64 = 0;
                let image = &problem.images[image_index];

                for &element_idx in &image.clear_parts {
                    let count = state.counts[element_idx];
                    let element_area = problem.universe[element_idx].area as i64;

                    if is_removing {
                        // If removing the LAST image covering this part, it becomes cloudy
                        if count == 1 {
                            delta += element_area;
                        }
                    } else {
                        // If adding the FIRST image covering this part, it becomes clear
                        if count == 0 {
                            delta -= element_area;
                        }
                    }
                }
                delta
            }
            Self::MinResolution(state) => {
                let mut delta: i64 = 0;
                let img_res = problem.raw_instance.resolution[image_index];
                let image = &problem.images[image_index];

                for &element_idx in &image.parts {
                    let resolutions = &state.element_resolutions[element_idx];
                    let current_min = resolutions.first().copied().unwrap_or(0);

                    if is_removing {
                        // Only affects score if removing the image that provides current minimum
                        if img_res == current_min {
                            // Check second element in sorted list (next minimum)
                            let next_min = if resolutions.len() > 1 { resolutions[1] } else { 0 };
                            delta += (next_min as i64) - (current_min as i64);
                        }
                    } else {
                        // Only affects score if new image is better than current minimum
                        if resolutions.is_empty() {
                            delta += img_res as i64;
                        } else if img_res < current_min {
                            delta -= (current_min - img_res) as i64;
                        }
                    }
                }
                delta
            }
            Self::MaxIncidenceAngle(state) => {
                let angle = problem.raw_instance.incidence_angle[image_index];
                let current_max = state.sorted_angles.last().copied().unwrap_or(0);

                if is_removing {
                    // If not removing the max, or if duplicates exist, delta is 0
                    if angle < current_max {
                        return 0;
                    }
                    // Check if it's a unique max
                    let len = state.sorted_angles.len();
                    if len > 1 && state.sorted_angles[len - 2] == current_max {
                        return 0; // Duplicate max exists
                    }
                    // Unique max removed. New max is second-to-last (or 0)
                    let next_max = if len > 1 { state.sorted_angles[len - 2] } else { 0 };
                    (next_max as i64) - (current_max as i64)
                } else {
                    // Adding: delta > 0 only if new angle exceeds current max
                    if angle > current_max {
                        (angle as i64) - (current_max as i64)
                    } else {
                        0
                    }
                }
            }
            Self::Stateless => 0,
        }
    }

    fn apply(
        &mut self,
        image_index: usize,
        is_removing: bool,
        problem: &Problem<impl ImageSet<D>, D>,
    ) {
        match self {
            Self::TotalCost(state) => {
                let cost = problem.images[image_index].cost();
                if is_removing {
                    state.current_cost -= cost;
                } else {
                    state.current_cost += cost;
                }
            }
            Self::CloudyArea(state) => {
                let image = &problem.images[image_index];
                
                for &element_idx in &image.clear_parts {
                    let element_area = problem.universe[element_idx].area;
                    
                    if is_removing {
                        state.counts[element_idx] -= 1;
                        if state.counts[element_idx] == 0 {
                            state.current_area += element_area;
                            state.cloudy_elements.insert(element_idx);
                        }
                    } else {
                        if state.counts[element_idx] == 0 {
                            state.current_area -= element_area;
                            state.cloudy_elements.set(element_idx, false);
                        }
                        state.counts[element_idx] += 1;
                    }
                }
            }
            Self::MinResolution(state) => {
                let img_res = problem.raw_instance.resolution[image_index];
                let image = &problem.images[image_index];

                for &element_idx in &image.parts {
                    let resolutions = &mut state.element_resolutions[element_idx];
                    let old_min = resolutions.first().copied().unwrap_or(0);

                    if is_removing {
                        // Binary search to find and remove one instance
                        if let Ok(pos) = resolutions.binary_search(&img_res) {
                            resolutions.remove(pos);
                        }
                    } else {
                        // Binary search to insert keeping sorted order
                        let pos = resolutions.binary_search(&img_res).unwrap_or_else(|e| e);
                        resolutions.insert(pos, img_res);
                    }

                    let new_min = resolutions.first().copied().unwrap_or(0);
                    
                    // Update global sum incrementally
                    if new_min > old_min {
                        state.current_sum += new_min - old_min;
                    } else if new_min < old_min {
                        state.current_sum -= old_min - new_min;
                    }
                }
            }
            Self::MaxIncidenceAngle(state) => {
                let angle = problem.raw_instance.incidence_angle[image_index];

                if is_removing {
                    // Binary search is fast on ~200 elements
                    if let Ok(pos) = state.sorted_angles.binary_search(&angle) {
                        state.sorted_angles.remove(pos);
                    }
                } else {
                    let pos = state.sorted_angles.binary_search(&angle).unwrap_or_else(|e| e);
                    state.sorted_angles.insert(pos, angle);
                }
            }
            Self::Stateless => {}
        }
    }

    fn value(&self) -> Option<u64> {
        match self {
            Self::TotalCost(s) => Some(s.current_cost),
            Self::CloudyArea(s) => Some(s.current_area),
            Self::MinResolution(s) => Some(s.current_sum),
            Self::MaxIncidenceAngle(s) => Some(s.sorted_angles.last().copied().unwrap_or(0)),
            Self::Stateless => None,
        }
    }
}

/// Standard array-based tracker collection.
#[derive(Clone, Debug)]
pub struct StandardTrackerArray<const D: usize> {
    trackers: [StandardTracker; D],
}

impl<const D: usize> TrackerCollection<D> for StandardTrackerArray<D> {
    fn get(&self, index: usize) -> &StandardTracker {
        &self.trackers[index]
    }

    fn get_mut(&mut self, index: usize) -> &mut StandardTracker {
        &mut self.trackers[index]
    }

    fn new<T: ImageSet<D>>(problem: &Problem<T, D>) -> Self {
        let trackers = std::array::from_fn(|i| {
            match &problem.objectives[i] {
                crate::objectives::ObjectiveState::TotalCost { .. } => StandardTracker::TotalCost(TotalCostState {
                    current_cost: 0,
                }),
                crate::objectives::ObjectiveState::CloudyArea { .. } => {
                    let total_area: u64 = problem.universe.iter().map(|e| e.area).sum();
                    let mut cloudy = FixedBitSet::with_capacity(problem.universe.len());
                    cloudy.set_range(.., true); // All starts as cloudy
                    
                    StandardTracker::CloudyArea(CloudyAreaState {
                        counts: vec![0; problem.universe.len()],
                        cloudy_elements: cloudy,
                        current_area: total_area,
                    })
                }
                crate::objectives::ObjectiveState::MinResolution { .. } => {
                    StandardTracker::MinResolution(MinResolutionState {
                        element_resolutions: vec![Vec::new(); problem.universe.len()],
                        current_sum: 0,
                    })
                }
                crate::objectives::ObjectiveState::MaxIncidenceAngle { .. } => {
                    StandardTracker::MaxIncidenceAngle(MaxIncidenceAngleState {
                        sorted_angles: Vec::with_capacity(200), // Pre-allocate for typical image count
                    })
                }
            }
        });

        Self { trackers }
    }
}
