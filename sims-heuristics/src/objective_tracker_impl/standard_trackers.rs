use std::sync::Arc;

use arrayvec::ArrayVec;
use fixedbitset::FixedBitSet;

use crate::{
    SetCoverProblem,
    objective_tracker::{ObjectiveTracker, TrackerCollection},
    solution::ImageSet,
};

/// State for tracking Total Cost incrementally.
#[derive(Clone, Debug)]
pub struct TotalCostState {
    pub current_cost: u64,
    pub image_costs: Arc<Vec<u64>>,
}

/// State for tracking Cloudy Area incrementally.
#[derive(Clone, Debug)]
pub struct CloudyAreaState {
    /// Count of images covering element i with clear parts. Max ~200, so u16 is sufficient.
    pub counts: Vec<u16>,
    /// Bitset: 1 = Cloudy (count 0), 0 = Clear (count > 0)
    pub cloudy_elements: FixedBitSet,
    pub current_area: u64,
    pub element_areas: Arc<Vec<u64>>,
    pub clear_images: Arc<Vec<FixedBitSet>>,
}

/// State for tracking Minimum Resolution Sum.
/// Maintains per-element resolution multisets with cached minima.
#[derive(Clone, Debug)]
pub struct MinResolutionState {
    pub image_resolutions: Arc<Vec<u64>>,
    /// Unsorted resolutions of selected images covering each element.
    ///
    /// We keep this unsorted to make add/remove `O(1)` (`push`/`swap_remove`) and
    /// rely on cached minima for objective deltas.
    pub element_resolutions: Vec<ArrayVec<u64, 64>>,
    /// Cached minimum resolution per element (0 if uncovered)
    pub element_min: Vec<u64>,
    /// Cached count of the minimum resolution per element
    pub element_min_count: Vec<u16>,
    pub current_sum: u64,
}

/// State for tracking Maximum Incidence Angle incrementally.
#[derive(Clone, Debug)]
pub struct MaxIncidenceAngleState {
    /// Sorted list of incidence angles of ALL selected images.
    /// With N=200, this is tiny and fits in L1 cache.
    pub sorted_angles: Vec<u64>,
    pub image_incidence_angles: Arc<Vec<u64>>,
    /// Cached current maximum angle
    pub current_max: u64,
}

/// The standard tracker enum wrapping specific state implementations.
#[derive(Clone, Debug)]
pub enum StandardTracker {
    TotalCost(TotalCostState),
    CloudyArea(CloudyAreaState),
    MinResolution(MinResolutionState),
    MaxIncidenceAngle(MaxIncidenceAngleState),
}

impl StandardTracker {
    #[must_use]
    pub const fn value(&self) -> u64 {
        match self {
            Self::TotalCost(s) => s.current_cost,
            Self::CloudyArea(s) => s.current_area,
            Self::MinResolution(s) => s.current_sum,
            Self::MaxIncidenceAngle(s) => s.current_max,
        }
    }
}

impl<const D: usize> ObjectiveTracker<D> for TotalCostState {
    fn peek_removal_delta(
        &self,
        image_index: usize,
        _problem: &impl SetCoverProblem<D>,
        _solution: &impl ImageSet<D>,
    ) -> i64 {
        -(self.image_costs[image_index] as i64)
    }

    fn peek_addition_delta(
        &self,
        image_index: usize,
        _problem: &impl SetCoverProblem<D>,
        _solution: &impl ImageSet<D>,
    ) -> i64 {
        self.image_costs[image_index] as i64
    }

    fn track_image_removal(
        &mut self,
        image_index: usize,
        _problem: &impl SetCoverProblem<D>,
    ) -> i64 {
        let cost = self.image_costs[image_index];
        self.current_cost -= cost;
        -(cost as i64)
    }

    fn track_image_addition(
        &mut self,
        image_index: usize,
        _problem: &impl SetCoverProblem<D>,
    ) -> i64 {
        let cost = self.image_costs[image_index];
        self.current_cost += cost;
        cost as i64
    }

    fn value(&self) -> u64 {
        self.current_cost
    }
}

impl<const D: usize> ObjectiveTracker<D> for CloudyAreaState {
    fn peek_removal_delta(
        &self,
        image_index: usize,
        _problem: &impl SetCoverProblem<D>,
        _solution: &impl ImageSet<D>,
    ) -> i64 {
        let mut delta: i64 = 0;
        if let Some(clear_bitset) = self.clear_images.get(image_index) {
            for element_idx in clear_bitset.ones() {
                let count = self.counts[element_idx];
                if count == 1 {
                    delta += self.element_areas[element_idx] as i64;
                }
            }
        }
        delta
    }

    fn peek_addition_delta(
        &self,
        image_index: usize,
        _problem: &impl SetCoverProblem<D>,
        _solution: &impl ImageSet<D>,
    ) -> i64 {
        let mut delta: i64 = 0;
        if let Some(clear_bitset) = self.clear_images.get(image_index) {
            for element_idx in clear_bitset.ones() {
                let count = self.counts[element_idx];
                if count == 0 {
                    delta -= self.element_areas[element_idx] as i64;
                }
            }
        }
        delta
    }

    fn track_image_removal(
        &mut self,
        image_index: usize,
        _problem: &impl SetCoverProblem<D>,
    ) -> i64 {
        let mut delta = 0i64;
        log::debug!("    CloudyArea tracker: removing image {image_index}");
        log::debug!("      Current cloudy area: {}", self.current_area);

        if let Some(clear_bitset) = self.clear_images.get(image_index) {
            log::debug!(
                "      Image {image_index} clears {} elements",
                clear_bitset.count_ones(..)
            );
            for element_idx in clear_bitset.ones() {
                let element_area = self.element_areas[element_idx];
                let count_before = self.counts[element_idx];

                self.counts[element_idx] -= 1;
                let count_after = self.counts[element_idx];

                if count_after == 0 {
                    self.current_area += element_area;
                    self.cloudy_elements.insert(element_idx);
                    delta += element_area as i64;
                    log::debug!(
                        "        Element {element_idx} count: {count_before} -> {count_after} (becomes cloudy, area={element_area})"
                    );
                }
            }
        } else {
            log::debug!(
                "      Image {image_index} has no clear_bitset (doesn't clear any elements)"
            );
        }
        log::debug!("    Total delta for CloudyArea: {delta}");
        delta
    }

    fn track_image_addition(
        &mut self,
        image_index: usize,
        _problem: &impl SetCoverProblem<D>,
    ) -> i64 {
        let mut delta = 0i64;
        if let Some(clear_bitset) = self.clear_images.get(image_index) {
            for element_idx in clear_bitset.ones() {
                let element_area = self.element_areas[element_idx];

                if self.counts[element_idx] == 0 {
                    self.current_area -= element_area;
                    self.cloudy_elements.set(element_idx, false);
                    delta -= element_area as i64;
                }
                self.counts[element_idx] += 1;
            }
        }
        delta
    }

    fn value(&self) -> u64 {
        self.current_area
    }
}

impl<const D: usize> ObjectiveTracker<D> for MinResolutionState {
    fn peek_removal_delta(
        &self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
        _solution: &impl ImageSet<D>,
    ) -> i64 {
        let mut delta: i64 = 0;
        let img_res = self.image_resolutions[image_index];

        for element_idx in problem.image_elements(image_index) {
            let current_min = self.element_min[element_idx];
            if img_res != current_min {
                continue;
            }

            let min_count = self.element_min_count[element_idx];
            if min_count > 1 {
                continue;
            }

            // Unique minimum would be removed; next min is the minimum of the remaining values.
            let resolutions = &self.element_resolutions[element_idx];
            let mut skipped = false;
            let mut next_min: u64 = 0;
            for &val in resolutions {
                if !skipped && val == img_res {
                    skipped = true;
                    continue;
                }
                if next_min == 0 || val < next_min {
                    next_min = val;
                }
            }

            delta += (next_min as i64) - (current_min as i64);
        }
        delta
    }

    fn peek_addition_delta(
        &self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
        _solution: &impl ImageSet<D>,
    ) -> i64 {
        let mut delta: i64 = 0;
        let img_res = self.image_resolutions[image_index];

        for element_idx in problem.image_elements(image_index) {
            let current_min = self.element_min[element_idx];
            if current_min == 0 {
                delta += img_res as i64;
            } else if img_res < current_min {
                delta -= (current_min - img_res) as i64;
            }
        }
        delta
    }

    fn track_image_removal(
        &mut self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
    ) -> i64 {
        let img_res = self.image_resolutions[image_index];
        let mut delta = 0i64;

        for element_idx in problem.image_elements(image_index) {
            let resolutions = &mut self.element_resolutions[element_idx];

            let current_min = self.element_min[element_idx];
            let resolution_position = resolutions
                .iter()
                .position(|&r| r == img_res)
                .expect("image resolution should be present for given element");
            resolutions.swap_remove(resolution_position);

            if img_res != current_min {
                continue;
            }

            let min_count = self.element_min_count[element_idx];
            if min_count > 1 {
                self.element_min_count[element_idx] = min_count - 1;
                continue;
            }

            // Unique minimum removed: recompute min and count from remaining resolutions.
            let mut next_min: u64 = 0;
            let mut next_count: u16 = 0;
            for &val in resolutions.as_slice() {
                if next_min == 0 || val < next_min {
                    next_min = val;
                    next_count = 1;
                } else if val == next_min {
                    next_count += 1;
                }
            }

            self.element_min[element_idx] = next_min;
            self.element_min_count[element_idx] = next_count;
            self.current_sum = self.current_sum - current_min + next_min;
            delta += (next_min as i64) - (current_min as i64);
        }
        delta
    }

    fn track_image_addition(
        &mut self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
    ) -> i64 {
        let img_res = self.image_resolutions[image_index];
        let mut delta = 0i64;

        for element_idx in problem.image_elements(image_index) {
            let resolutions = &mut self.element_resolutions[element_idx];

            let current_min = self.element_min[element_idx];
            resolutions.push(img_res);

            if current_min == 0 {
                self.element_min[element_idx] = img_res;
                self.element_min_count[element_idx] = 1;
                self.current_sum += img_res;
                delta += img_res as i64;
            } else if img_res < current_min {
                self.element_min[element_idx] = img_res;
                self.element_min_count[element_idx] = 1;
                self.current_sum = self.current_sum - current_min + img_res;
                delta += (img_res as i64) - (current_min as i64);
            } else if img_res == current_min {
                self.element_min_count[element_idx] =
                    self.element_min_count[element_idx].saturating_add(1);
            }
        }
        delta
    }

    fn value(&self) -> u64 {
        self.current_sum
    }
}

impl<const D: usize> ObjectiveTracker<D> for MaxIncidenceAngleState {
    fn peek_removal_delta(
        &self,
        image_index: usize,
        _problem: &impl SetCoverProblem<D>,
        _solution: &impl ImageSet<D>,
    ) -> i64 {
        let angle = self.image_incidence_angles[image_index];
        let current_max = self.sorted_angles.last().copied().unwrap_or(0);

        if angle < current_max {
            return 0;
        }
        let len = self.sorted_angles.len();
        if len > 1 && self.sorted_angles[len - 2] == current_max {
            return 0;
        }
        let next_max = if len > 1 {
            self.sorted_angles[len - 2]
        } else {
            0
        };
        (next_max as i64) - (current_max as i64)
    }

    fn peek_addition_delta(
        &self,
        image_index: usize,
        _problem: &impl SetCoverProblem<D>,
        _solution: &impl ImageSet<D>,
    ) -> i64 {
        let angle = self.image_incidence_angles[image_index];
        let current_max = self.sorted_angles.last().copied().unwrap_or(0);
        if angle > current_max {
            (angle as i64) - (current_max as i64)
        } else {
            0
        }
    }

    fn track_image_removal(
        &mut self,
        image_index: usize,
        _problem: &impl SetCoverProblem<D>,
    ) -> i64 {
        let angle = self.image_incidence_angles[image_index];
        let old_max = self.current_max;

        if let Ok(pos) = self.sorted_angles.binary_search(&angle) {
            self.sorted_angles.remove(pos);
        }
        self.current_max = self.sorted_angles.last().copied().unwrap_or(0);
        (self.current_max as i64) - (old_max as i64)
    }

    fn track_image_addition(
        &mut self,
        image_index: usize,
        _problem: &impl SetCoverProblem<D>,
    ) -> i64 {
        let angle = self.image_incidence_angles[image_index];
        let old_max = self.current_max;

        let pos = self
            .sorted_angles
            .binary_search(&angle)
            .unwrap_or_else(|e| e);
        self.sorted_angles.insert(pos, angle);
        if angle > self.current_max {
            self.current_max = angle;
        }
        (self.current_max as i64) - (old_max as i64)
    }

    fn value(&self) -> u64 {
        self.current_max
    }
}

impl<const D: usize> ObjectiveTracker<D> for StandardTracker {
    fn peek_removal_delta(
        &self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
        solution: &impl ImageSet<D>,
    ) -> i64 {
        match self {
            Self::TotalCost(s) => s.peek_removal_delta(image_index, problem, solution),
            Self::CloudyArea(s) => s.peek_removal_delta(image_index, problem, solution),
            Self::MinResolution(s) => s.peek_removal_delta(image_index, problem, solution),
            Self::MaxIncidenceAngle(s) => s.peek_removal_delta(image_index, problem, solution),
        }
    }

    fn peek_addition_delta(
        &self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
        solution: &impl ImageSet<D>,
    ) -> i64 {
        match self {
            Self::TotalCost(s) => s.peek_addition_delta(image_index, problem, solution),
            Self::CloudyArea(s) => s.peek_addition_delta(image_index, problem, solution),
            Self::MinResolution(s) => s.peek_addition_delta(image_index, problem, solution),
            Self::MaxIncidenceAngle(s) => s.peek_addition_delta(image_index, problem, solution),
        }
    }

    fn track_image_removal(
        &mut self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
    ) -> i64 {
        match self {
            Self::TotalCost(s) => s.track_image_removal(image_index, problem),
            Self::CloudyArea(s) => s.track_image_removal(image_index, problem),
            Self::MinResolution(s) => s.track_image_removal(image_index, problem),
            Self::MaxIncidenceAngle(s) => s.track_image_removal(image_index, problem),
        }
    }

    fn track_image_addition(
        &mut self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
    ) -> i64 {
        match self {
            Self::TotalCost(s) => s.track_image_addition(image_index, problem),
            Self::CloudyArea(s) => s.track_image_addition(image_index, problem),
            Self::MinResolution(s) => s.track_image_addition(image_index, problem),
            Self::MaxIncidenceAngle(s) => s.track_image_addition(image_index, problem),
        }
    }

    fn value(&self) -> u64 {
        Self::value(self)
    }
}

/// Standard array-based tracker collection.
#[derive(Clone, Debug)]
pub struct StandardTrackerArray<const D: usize> {
    trackers: [StandardTracker; D],
}

impl<const D: usize> TrackerCollection<D> for StandardTrackerArray<D> {
    type Tracker = StandardTracker;

    fn get(&self, index: usize) -> &StandardTracker {
        &self.trackers[index]
    }

    fn get_mut(&mut self, index: usize) -> &mut StandardTracker {
        &mut self.trackers[index]
    }

    fn new(problem: &impl SetCoverProblem<D>) -> Self {
        let trackers = std::array::from_fn(|i| {
            match problem.objective(i) {
                crate::objectives::ObjectiveState::TotalCost { costs, .. } => {
                    StandardTracker::TotalCost(TotalCostState {
                        current_cost: 0,
                        image_costs: Arc::new(costs.clone()),
                    })
                }
                crate::objectives::ObjectiveState::CloudyArea {
                    clear_images,
                    areas,
                    ..
                } => {
                    let element_areas = areas.clone();
                    let total_area: u64 = element_areas.iter().sum();
                    let mut cloudy = FixedBitSet::with_capacity(problem.num_elements());
                    cloudy.set_range(.., true); // All starts as cloudy

                    StandardTracker::CloudyArea(CloudyAreaState {
                        counts: vec![0; problem.num_elements()],
                        cloudy_elements: cloudy,
                        current_area: total_area,
                        element_areas: Arc::new(element_areas),
                        clear_images: Arc::new(clear_images.clone()),
                    })
                }
                crate::objectives::ObjectiveState::MinResolution { resolutions, .. } => {
                    StandardTracker::MinResolution(MinResolutionState {
                        image_resolutions: Arc::new(resolutions.clone()),
                        element_resolutions: vec![ArrayVec::new(); problem.num_elements()],
                        element_min: vec![0; problem.num_elements()],
                        element_min_count: vec![0; problem.num_elements()],
                        current_sum: 0,
                    })
                }
                crate::objectives::ObjectiveState::MaxIncidenceAngle {
                    incidence_angles, ..
                } => {
                    StandardTracker::MaxIncidenceAngle(MaxIncidenceAngleState {
                        sorted_angles: Vec::with_capacity(200), // Pre-allocate for typical image count
                        image_incidence_angles: Arc::new(incidence_angles.clone()),
                        current_max: 0,
                    })
                }
            }
        });

        Self { trackers }
    }

    /// Get the initial objective values from all trackers.
    /// Returns an array of D objective values corresponding to the empty solution state.
    ///
    /// # Panics
    /// Panics if any tracker fails to provide an initial value.
    fn initial_objectives(&self) -> [u64; D] {
        std::array::from_fn(|i| {
            match &self.trackers[i] {
                StandardTracker::MinResolution(_) => 0, // Empty solution has 0 MinResolution
                _ => self.trackers[i].value(),
            }
        })
    }

    /// Calculate the delta for removing an image across all objectives.
    /// Does NOT modify internal state.
    fn peek_removal_delta(
        &self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
        solution: &impl ImageSet<D>,
    ) -> [i64; D] {
        std::array::from_fn(|i| self.trackers[i].peek_removal_delta(image_index, problem, solution))
    }

    /// Calculate the delta for adding an image across all objectives.
    /// Does NOT modify internal state.
    fn peek_addition_delta(
        &self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
        solution: &impl ImageSet<D>,
    ) -> [i64; D] {
        std::array::from_fn(|i| {
            self.trackers[i].peek_addition_delta(image_index, problem, solution)
        })
    }

    /// Track the removal of an image across all objectives.
    /// Returns the delta array and updates internal state.
    fn track_image_removal(
        &mut self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
    ) -> [i64; D] {
        std::array::from_fn(|i| self.trackers[i].track_image_removal(image_index, problem))
    }

    /// Track the addition of an image across all objectives.
    /// Returns the delta array and updates internal state.
    fn track_image_addition(
        &mut self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
    ) -> [i64; D] {
        std::array::from_fn(|i| self.trackers[i].track_image_addition(image_index, problem))
    }

    /// Get the current values from all trackers.
    /// Returns an array of D optional values.
    fn values(&self) -> [u64; D] {
        std::array::from_fn(|i| self.trackers[i].value())
    }

    /// Reinitialize tracker state from a solution's selected images.
    /// Reuses existing allocations for efficiency.
    fn initialize_from(&mut self, solution: &impl ImageSet<D>, problem: &impl SetCoverProblem<D>) {
        // Reset to empty state (reuses allocations in Vec/ArrayVec)
        *self = Self::new(problem);

        // Apply all selected images
        for img in solution.selected_images() {
            self.track_image_addition(img, problem);
        }
        log::debug!(
            "Tracker initialization complete. Tracker values: {:?}",
            self.values()
        );
    }
}
