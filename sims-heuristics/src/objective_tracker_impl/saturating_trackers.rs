//! Saturating u8 counter-based objective trackers.
//!
//! Key insight: For MinResolution and CloudyArea, we only care about
//! three states: 0, 1, and 2+. Using u8 with saturation at 254 allows:
//! - 4x more elements per cache line (vs u32)
//! - Better SIMD vectorization (32 u8s per AVX2 register)
//! - Same correctness since count > 1 is treated same as count == 2
//!
//! This module implements the "Saturating Counters" approach from
//! advanced_optimizations.md Section 7.

#![allow(unused)]

use std::sync::Arc;

use crate::objective_tracker::{ObjectiveTracker, TrackerCollection};
use crate::problem::SetCoverProblem;
use crate::solution::ImageSet;

use super::simd_trackers::{simd_shared_data, Interval, SimdTrackerSharedData, 
    SimdTotalCostState, SimdMaxIncidenceAngleState};

// =============================================================================
// Saturating Counter MinResolution Tracker
// =============================================================================

/// MinResolution tracker using saturating u8 counters.
/// c0 and c1 are separate u8 arrays, allowing 2x cache efficiency vs packed u32.
#[derive(Clone, Debug)]
pub struct SaturatingMinResState {
    /// Coverage counts for level 0 (low-res), saturates at 254
    pub c0_counts: Vec<u8>,
    /// Coverage counts for level 1 (high-res), saturates at 254  
    pub c1_counts: Vec<u8>,
    pub low_val: u64,
    pub high_val: u64,
    pub diff: i64,
    pub current_sum: u64,
    pub image_intervals: Arc<Vec<Interval>>,
    pub image_intervals_offsets: Arc<Vec<usize>>,
    pub image_resolution_level: Arc<Vec<u8>>,
}

impl SaturatingMinResState {
    /// Track removal for level 0 using saturating u8 counters.
    #[inline]
    fn track_removal_level0(&mut self, int_start: usize, int_end: usize) -> i64 {
        let low_val = self.low_val as i64;
        let diff = self.diff;
        let mut delta = 0i64;
        
        let c0_ptr = self.c0_counts.as_mut_ptr();
        let c1_ptr = self.c1_counts.as_ptr();
        
        for int_idx in int_start..int_end {
            let interval = unsafe { *self.image_intervals.get_unchecked(int_idx) };
            let start = interval.start as usize;
            let end = start + interval.len as usize;
            
            for idx in start..end {
                unsafe {
                    let c0_slot = c0_ptr.add(idx);
                    let c0 = *c0_slot;
                    let c1 = *c1_ptr.add(idx);
                    
                    // Saturating decrement
                    *c0_slot = c0.saturating_sub(1);
                    
                    // Delta computation
                    if c0 == 1 {
                        if c1 > 0 {
                            delta += diff; // Switch to high-res
                        } else {
                            delta -= low_val; // Become uncovered
                        }
                    }
                }
            }
        }
        
        self.current_sum = (self.current_sum as i64 + delta) as u64;
        delta
    }
    
    /// Track removal for level 1 using saturating u8 counters.
    #[inline]
    fn track_removal_level1(&mut self, int_start: usize, int_end: usize) -> i64 {
        let high_val = self.high_val as i64;
        let mut delta = 0i64;
        
        let c0_ptr = self.c0_counts.as_ptr();
        let c1_ptr = self.c1_counts.as_mut_ptr();
        
        for int_idx in int_start..int_end {
            let interval = unsafe { *self.image_intervals.get_unchecked(int_idx) };
            let start = interval.start as usize;
            let end = start + interval.len as usize;
            
            for idx in start..end {
                unsafe {
                    let c1_slot = c1_ptr.add(idx);
                    let c0 = *c0_ptr.add(idx);
                    let c1 = *c1_slot;
                    
                    // Saturating decrement
                    *c1_slot = c1.saturating_sub(1);
                    
                    // Delta: if c1 was 1 and c0 is 0, element becomes uncovered
                    if c1 == 1 && c0 == 0 {
                        delta -= high_val;
                    }
                }
            }
        }
        
        self.current_sum = (self.current_sum as i64 + delta) as u64;
        delta
    }
    
    /// Track addition for level 0.
    #[inline]
    fn track_addition_level0(&mut self, int_start: usize, int_end: usize) -> i64 {
        let low_val = self.low_val as i64;
        let diff = self.diff;
        let mut delta = 0i64;
        
        let c0_ptr = self.c0_counts.as_mut_ptr();
        let c1_ptr = self.c1_counts.as_ptr();
        
        for int_idx in int_start..int_end {
            let interval = unsafe { *self.image_intervals.get_unchecked(int_idx) };
            let start = interval.start as usize;
            let end = start + interval.len as usize;
            
            for idx in start..end {
                unsafe {
                    let c0_slot = c0_ptr.add(idx);
                    let c0 = *c0_slot;
                    let c1 = *c1_ptr.add(idx);
                    
                    // Delta computation before update
                    if c0 == 0 {
                        if c1 > 0 {
                            delta -= diff; // Switch from high to low
                        } else {
                            delta += low_val; // First coverage
                        }
                    }
                    
                    // Saturating increment (cap at 254 to avoid overflow issues)
                    *c0_slot = c0.saturating_add(1).min(254);
                }
            }
        }
        
        self.current_sum = (self.current_sum as i64 + delta) as u64;
        delta
    }
    
    /// Track addition for level 1.
    #[inline]
    fn track_addition_level1(&mut self, int_start: usize, int_end: usize) -> i64 {
        let high_val = self.high_val as i64;
        let mut delta = 0i64;
        
        let c0_ptr = self.c0_counts.as_ptr();
        let c1_ptr = self.c1_counts.as_mut_ptr();
        
        for int_idx in int_start..int_end {
            let interval = unsafe { *self.image_intervals.get_unchecked(int_idx) };
            let start = interval.start as usize;
            let end = start + interval.len as usize;
            
            for idx in start..end {
                unsafe {
                    let c1_slot = c1_ptr.add(idx);
                    let c0 = *c0_ptr.add(idx);
                    let c1 = *c1_slot;
                    
                    // Delta: if both c0 and c1 are 0, element gets first coverage
                    if c0 == 0 && c1 == 0 {
                        delta += high_val;
                    }
                    
                    // Saturating increment
                    *c1_slot = c1.saturating_add(1).min(254);
                }
            }
        }
        
        self.current_sum = (self.current_sum as i64 + delta) as u64;
        delta
    }
    
    /// Peek removal for level 0 (read-only).
    #[inline]
    fn peek_removal_level0(&self, int_start: usize, int_end: usize) -> i64 {
        let low_val = self.low_val as i64;
        let diff = self.diff;
        let mut delta = 0i64;
        
        for int_idx in int_start..int_end {
            let interval = unsafe { *self.image_intervals.get_unchecked(int_idx) };
            let start = interval.start as usize;
            let end = start + interval.len as usize;
            
            for idx in start..end {
                unsafe {
                    let c0 = *self.c0_counts.get_unchecked(idx);
                    let c1 = *self.c1_counts.get_unchecked(idx);
                    
                    if c0 == 1 {
                        if c1 > 0 {
                            delta += diff;
                        } else {
                            delta -= low_val;
                        }
                    }
                }
            }
        }
        
        delta
    }
    
    /// Peek removal for level 1 (read-only).
    #[inline]
    fn peek_removal_level1(&self, int_start: usize, int_end: usize) -> i64 {
        let high_val = self.high_val as i64;
        let mut delta = 0i64;
        
        for int_idx in int_start..int_end {
            let interval = unsafe { *self.image_intervals.get_unchecked(int_idx) };
            let start = interval.start as usize;
            let end = start + interval.len as usize;
            
            for idx in start..end {
                unsafe {
                    let c0 = *self.c0_counts.get_unchecked(idx);
                    let c1 = *self.c1_counts.get_unchecked(idx);
                    
                    if c1 == 1 && c0 == 0 {
                        delta -= high_val;
                    }
                }
            }
        }
        
        delta
    }
    
    /// Peek addition for level 0 (read-only).
    #[inline]
    fn peek_addition_level0(&self, int_start: usize, int_end: usize) -> i64 {
        let low_val = self.low_val as i64;
        let diff = self.diff;
        let mut delta = 0i64;
        
        for int_idx in int_start..int_end {
            let interval = unsafe { *self.image_intervals.get_unchecked(int_idx) };
            let start = interval.start as usize;
            let end = start + interval.len as usize;
            
            for idx in start..end {
                unsafe {
                    let c0 = *self.c0_counts.get_unchecked(idx);
                    let c1 = *self.c1_counts.get_unchecked(idx);
                    
                    if c0 == 0 {
                        if c1 > 0 {
                            delta -= diff;
                        } else {
                            delta += low_val;
                        }
                    }
                }
            }
        }
        
        delta
    }
    
    /// Peek addition for level 1 (read-only).
    #[inline]
    fn peek_addition_level1(&self, int_start: usize, int_end: usize) -> i64 {
        let high_val = self.high_val as i64;
        let mut delta = 0i64;
        
        for int_idx in int_start..int_end {
            let interval = unsafe { *self.image_intervals.get_unchecked(int_idx) };
            let start = interval.start as usize;
            let end = start + interval.len as usize;
            
            for idx in start..end {
                unsafe {
                    let c0 = *self.c0_counts.get_unchecked(idx);
                    let c1 = *self.c1_counts.get_unchecked(idx);
                    
                    if c0 == 0 && c1 == 0 {
                        delta += high_val;
                    }
                }
            }
        }
        
        delta
    }
}

impl<const D: usize> ObjectiveTracker<D> for SaturatingMinResState {
    fn peek_removal_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;
        let int_start = unsafe { *self.image_intervals_offsets.get_unchecked(image_index) };
        let int_end = unsafe { *self.image_intervals_offsets.get_unchecked(image_index + 1) };
        
        if img_level == 0 {
            self.peek_removal_level0(int_start, int_end)
        } else {
            self.peek_removal_level1(int_start, int_end)
        }
    }

    fn peek_addition_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;
        let int_start = unsafe { *self.image_intervals_offsets.get_unchecked(image_index) };
        let int_end = unsafe { *self.image_intervals_offsets.get_unchecked(image_index + 1) };
        
        if img_level == 0 {
            self.peek_addition_level0(int_start, int_end)
        } else {
            self.peek_addition_level1(int_start, int_end)
        }
    }

    fn track_image_removal(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;
        let int_start = unsafe { *self.image_intervals_offsets.get_unchecked(image_index) };
        let int_end = unsafe { *self.image_intervals_offsets.get_unchecked(image_index + 1) };
        
        if img_level == 0 {
            self.track_removal_level0(int_start, int_end)
        } else {
            self.track_removal_level1(int_start, int_end)
        }
    }

    fn track_image_addition(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;
        let int_start = unsafe { *self.image_intervals_offsets.get_unchecked(image_index) };
        let int_end = unsafe { *self.image_intervals_offsets.get_unchecked(image_index + 1) };
        
        if img_level == 0 {
            self.track_addition_level0(int_start, int_end)
        } else {
            self.track_addition_level1(int_start, int_end)
        }
    }

    fn value(&self) -> u64 {
        self.current_sum
    }
}

// =============================================================================
// Saturating Counter CloudyArea Tracker
// =============================================================================

/// CloudyArea tracker using saturating u8 counters.
#[derive(Clone, Debug)]
pub struct SaturatingCloudyAreaState {
    /// Coverage counts, saturates at 254
    pub counts: Vec<u8>,
    pub element_areas: Arc<Vec<u64>>,
    pub current_area: u64,
    pub clear_intervals: Arc<Vec<Interval>>,
    pub clear_intervals_offsets: Arc<Vec<usize>>,
}

impl<const D: usize> ObjectiveTracker<D> for SaturatingCloudyAreaState {
    fn peek_removal_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let int_start = unsafe { *self.clear_intervals_offsets.get_unchecked(image_index) };
        let int_end = unsafe { *self.clear_intervals_offsets.get_unchecked(image_index + 1) };
        
        let mut delta_area = 0u64;
        
        for int_idx in int_start..int_end {
            let interval = unsafe { *self.clear_intervals.get_unchecked(int_idx) };
            let start = interval.start as usize;
            let end = start + interval.len as usize;
            
            for idx in start..end {
                unsafe {
                    if *self.counts.get_unchecked(idx) == 1 {
                        delta_area += *self.element_areas.get_unchecked(idx);
                    }
                }
            }
        }
        
        delta_area as i64
    }

    fn peek_addition_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let int_start = unsafe { *self.clear_intervals_offsets.get_unchecked(image_index) };
        let int_end = unsafe { *self.clear_intervals_offsets.get_unchecked(image_index + 1) };
        
        let mut delta_area = 0u64;
        
        for int_idx in int_start..int_end {
            let interval = unsafe { *self.clear_intervals.get_unchecked(int_idx) };
            let start = interval.start as usize;
            let end = start + interval.len as usize;
            
            for idx in start..end {
                unsafe {
                    if *self.counts.get_unchecked(idx) == 0 {
                        delta_area += *self.element_areas.get_unchecked(idx);
                    }
                }
            }
        }
        
        -(delta_area as i64)
    }

    fn track_image_removal(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let int_start = unsafe { *self.clear_intervals_offsets.get_unchecked(image_index) };
        let int_end = unsafe { *self.clear_intervals_offsets.get_unchecked(image_index + 1) };
        
        let mut delta_area = 0u64;
        let counts_ptr = self.counts.as_mut_ptr();
        let areas_ptr = self.element_areas.as_ptr();
        
        for int_idx in int_start..int_end {
            let interval = unsafe { *self.clear_intervals.get_unchecked(int_idx) };
            let start = interval.start as usize;
            let end = start + interval.len as usize;
            
            for idx in start..end {
                unsafe {
                    let count_slot = counts_ptr.add(idx);
                    let count = *count_slot;
                    *count_slot = count.saturating_sub(1);
                    if count == 1 {
                        delta_area += *areas_ptr.add(idx);
                    }
                }
            }
        }
        
        self.current_area += delta_area;
        delta_area as i64
    }

    fn track_image_addition(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let int_start = unsafe { *self.clear_intervals_offsets.get_unchecked(image_index) };
        let int_end = unsafe { *self.clear_intervals_offsets.get_unchecked(image_index + 1) };
        
        let mut delta_area = 0u64;
        let counts_ptr = self.counts.as_mut_ptr();
        let areas_ptr = self.element_areas.as_ptr();
        
        for int_idx in int_start..int_end {
            let interval = unsafe { *self.clear_intervals.get_unchecked(int_idx) };
            let start = interval.start as usize;
            let end = start + interval.len as usize;
            
            for idx in start..end {
                unsafe {
                    let count_slot = counts_ptr.add(idx);
                    let count = *count_slot;
                    if count == 0 {
                        delta_area += *areas_ptr.add(idx);
                    }
                    *count_slot = count.saturating_add(1).min(254);
                }
            }
        }
        
        self.current_area -= delta_area;
        -(delta_area as i64)
    }

    fn value(&self) -> u64 {
        self.current_area
    }
}

// =============================================================================
// Saturating Counter Tracker Array
// =============================================================================

/// Tracker enum for saturating counter implementations
#[derive(Clone, Debug)]
pub enum SaturatingTracker {
    TotalCost(SimdTotalCostState),
    CloudyArea(SaturatingCloudyAreaState),
    MinResolution(SaturatingMinResState),
    MaxIncidenceAngle(SimdMaxIncidenceAngleState),
}

impl SaturatingTracker {
    #[must_use]
    pub fn value(&self) -> u64 {
        match self {
            Self::TotalCost(s) => s.current_cost,
            Self::CloudyArea(s) => s.current_area,
            Self::MinResolution(s) => s.current_sum,
            Self::MaxIncidenceAngle(s) => s.current_max,
        }
    }
}

impl<const D: usize> ObjectiveTracker<D> for SaturatingTracker {
    fn peek_removal_delta(&self, image_index: usize, problem: &impl SetCoverProblem<D>, solution: &impl ImageSet<D>) -> i64 {
        match self {
            Self::TotalCost(s) => s.peek_removal_delta(image_index, problem, solution),
            Self::CloudyArea(s) => s.peek_removal_delta(image_index, problem, solution),
            Self::MinResolution(s) => s.peek_removal_delta(image_index, problem, solution),
            Self::MaxIncidenceAngle(s) => s.peek_removal_delta(image_index, problem, solution),
        }
    }

    fn peek_addition_delta(&self, image_index: usize, problem: &impl SetCoverProblem<D>, solution: &impl ImageSet<D>) -> i64 {
        match self {
            Self::TotalCost(s) => s.peek_addition_delta(image_index, problem, solution),
            Self::CloudyArea(s) => s.peek_addition_delta(image_index, problem, solution),
            Self::MinResolution(s) => s.peek_addition_delta(image_index, problem, solution),
            Self::MaxIncidenceAngle(s) => s.peek_addition_delta(image_index, problem, solution),
        }
    }

    fn track_image_removal(&mut self, image_index: usize, problem: &impl SetCoverProblem<D>) -> i64 {
        match self {
            Self::TotalCost(s) => s.track_image_removal(image_index, problem),
            Self::CloudyArea(s) => s.track_image_removal(image_index, problem),
            Self::MinResolution(s) => s.track_image_removal(image_index, problem),
            Self::MaxIncidenceAngle(s) => s.track_image_removal(image_index, problem),
        }
    }

    fn track_image_addition(&mut self, image_index: usize, problem: &impl SetCoverProblem<D>) -> i64 {
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

/// Tracker array using saturating u8 counter implementations
#[derive(Clone, Debug)]
pub struct SaturatingTrackerArray<const D: usize> {
    trackers: [SaturatingTracker; D],
}

impl<const D: usize> TrackerCollection<D> for SaturatingTrackerArray<D> {
    type Tracker = SaturatingTracker;

    fn get(&self, index: usize) -> &SaturatingTracker {
        &self.trackers[index]
    }

    fn get_mut(&mut self, index: usize) -> &mut SaturatingTracker {
        &mut self.trackers[index]
    }

    fn new(problem: &impl SetCoverProblem<D>) -> Self {
        let shared = simd_shared_data(problem);

        let trackers = std::array::from_fn(|i| match problem.objective(i) {
            crate::objectives::ObjectiveState::TotalCost { .. } => {
                SaturatingTracker::TotalCost(SimdTotalCostState {
                    current_cost: 0,
                    image_costs: Arc::clone(&shared.image_costs),
                })
            }
            crate::objectives::ObjectiveState::CloudyArea { .. } => {
                let total_area: u64 = shared.element_areas.iter().sum();

                SaturatingTracker::CloudyArea(SaturatingCloudyAreaState {
                    counts: vec![0u8; problem.num_elements()],
                    current_area: total_area,
                    element_areas: Arc::clone(&shared.element_areas),
                    clear_intervals: Arc::clone(&shared.clear_intervals),
                    clear_intervals_offsets: Arc::clone(&shared.clear_intervals_offsets),
                })
            }
            crate::objectives::ObjectiveState::MinResolution { .. } => {
                let num_levels = shared.resolution_levels.len();
                let low_val = shared.resolution_levels[0];
                let high_val = shared.resolution_levels[num_levels - 1];

                SaturatingTracker::MinResolution(SaturatingMinResState {
                    c0_counts: vec![0u8; problem.num_elements()],
                    c1_counts: vec![0u8; problem.num_elements()],
                    low_val,
                    high_val,
                    diff: (high_val - low_val) as i64,
                    current_sum: 0,
                    image_intervals: Arc::clone(&shared.image_intervals),
                    image_intervals_offsets: Arc::clone(&shared.image_intervals_offsets),
                    image_resolution_level: Arc::clone(&shared.image_resolution_level),
                })
            }
            crate::objectives::ObjectiveState::MaxIncidenceAngle { .. } => {
                SaturatingTracker::MaxIncidenceAngle(SimdMaxIncidenceAngleState {
                    incidence_levels: Arc::clone(&shared.incidence_levels),
                    image_incidence_level: Arc::clone(&shared.image_incidence_level),
                    level_counts: vec![0; shared.incidence_levels.len()],
                    current_max_level: u8::MAX,
                    current_max: 0,
                })
            }
        });

        Self { trackers }
    }

    fn initial_objectives(&self) -> [u64; D] {
        std::array::from_fn(|i| self.trackers[i].value())
    }

    fn peek_removal_delta(&self, image_index: usize, problem: &impl SetCoverProblem<D>, solution: &impl ImageSet<D>) -> [i64; D] {
        std::array::from_fn(|i| self.trackers[i].peek_removal_delta(image_index, problem, solution))
    }

    fn peek_addition_delta(&self, image_index: usize, problem: &impl SetCoverProblem<D>, solution: &impl ImageSet<D>) -> [i64; D] {
        std::array::from_fn(|i| self.trackers[i].peek_addition_delta(image_index, problem, solution))
    }

    fn track_image_removal(&mut self, image_index: usize, problem: &impl SetCoverProblem<D>) -> [i64; D] {
        std::array::from_fn(|i| self.trackers[i].track_image_removal(image_index, problem))
    }

    fn track_image_addition(&mut self, image_index: usize, problem: &impl SetCoverProblem<D>) -> [i64; D] {
        std::array::from_fn(|i| self.trackers[i].track_image_addition(image_index, problem))
    }

    fn values(&self) -> [u64; D] {
        std::array::from_fn(|i| self.trackers[i].value())
    }

    fn initialize_from(&mut self, solution: &impl ImageSet<D>, problem: &impl SetCoverProblem<D>) {
        *self = Self::new(problem);
        for img in solution.selected_images() {
            self.track_image_addition(img, problem);
        }
    }
}
