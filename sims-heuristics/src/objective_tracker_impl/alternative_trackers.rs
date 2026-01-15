//! Alternative objective tracker implementation(s) for benchmarking.
//!
//! This module exists to let Criterion benchmarks compare tracker strategies side-by-side
//! without perturbing the production tracker code.

use fixedbitset::FixedBitSet;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use crate::objective_tracker::{ObjectiveTracker, TrackerCollection};
use crate::problem::SetCoverProblem;
use crate::solution::ImageSet;

/// State for tracking Total Cost incrementally.
#[derive(Clone, Debug)]
pub struct AltTotalCostState {
    pub current_cost: u64,
    pub image_costs: Arc<Vec<u64>>,
}

/// State for tracking Cloudy Area incrementally.
#[derive(Clone, Debug)]
pub struct AltCloudyAreaState {
    pub counts: Vec<u16>,
    pub cloudy_elements: FixedBitSet,
    pub current_area: u64,
    pub element_areas: Arc<Vec<u64>>,
    /// Clear elements in CSR format.
    pub clear_elements: Arc<Vec<u32>>,
    pub clear_elements_offsets: Arc<Vec<usize>>,
}

/// State for tracking Minimum Resolution Sum.
#[derive(Clone, Debug)]
pub struct AltMinResolutionState {
    /// Distinct resolution values present in the instance, sorted ascending.
    pub resolution_levels: Arc<Vec<u64>>,
    pub two_level: bool,
    pub two_level_low: u64,
    pub two_level_high: u64,
    /// Per-image mapping to a resolution level index.
    pub image_resolution_level: Arc<Vec<u8>>,
    /// Image elements in CSR format.
    pub image_elements: Arc<Vec<u32>>,
    pub image_elements_offsets: Arc<Vec<usize>>,
    /// Packed per-element counts for 2-level instances.
    ///
    /// Low count is stored in the lower 16 bits; high count is stored in the upper 16 bits.
    pub element_packed_counts: Vec<u32>,
    /// Packed per-element counts for instances with <= 8 levels.
    /// Each level uses 8 bits (up to 255 count).
    pub element_packed_small: Vec<u64>,
    /// Flat array storing counts per `(element, resolution_level)`.
    /// Index: `element_idx * num_levels + level_idx`
    pub element_level_counts: Vec<u16>,
    /// Bitmask blocks of non-zero resolution levels per element.
    ///
    /// Layout: `element_level_masks[element_idx * mask_words + word_idx]` is a 64-bit word where
    /// bit `b` is set iff `element_level_counts[element_idx * num_levels + (word_idx*64 + b)] > 0`.
    pub element_level_masks: Vec<u64>,
    /// Number of 64-bit words per element in `element_level_masks`.
    pub mask_words: u8,
    /// Current minimum resolution level for each element.
    /// Sentinel value 255 means uncovered.
    pub element_min_level: Vec<u8>,
    pub current_sum: u64,
}

#[derive(Debug)]
struct AltTrackerSharedData {
    image_costs: Arc<Vec<u64>>,
    element_areas: Arc<Vec<u64>>,
    // CSR format for clear elements
    clear_elements: Arc<Vec<u32>>,
    clear_elements_offsets: Arc<Vec<usize>>, // Size M+1
    resolution_levels: Arc<Vec<u64>>,
    image_resolution_level: Arc<Vec<u8>>,
    // CSR format for image elements
    image_elements: Arc<Vec<u32>>,
    image_elements_offsets: Arc<Vec<usize>>, // Size M+1
    incidence_levels: Arc<Vec<u64>>,
    image_incidence_level: Arc<Vec<u8>>,
}

#[allow(clippy::too_many_lines)]
fn alt_shared_data<const D: usize>(problem: &impl SetCoverProblem<D>) -> Arc<AltTrackerSharedData> {
    // Key by address of the concrete problem instance.
    #[allow(clippy::ref_as_ptr)]
    let key = problem as *const _ as usize;

    // Benchmarks are effectively single-threaded; use a thread-local cache to avoid lock overhead.
    thread_local! {
        static CACHE: RefCell<HashMap<usize, Arc<AltTrackerSharedData>>> = RefCell::new(HashMap::new());
    }

    if let Some(hit) = CACHE.with(|cache| cache.borrow().get(&key).cloned()) {
        return hit;
    }

    // Build immutable shared data once.
    let num_images = problem.num_images();
    
    // Build Image Elements CSR
    let mut image_elements = Vec::with_capacity(num_images * 20); // Estimate
    let mut image_elements_offsets = Vec::with_capacity(num_images + 1);
    image_elements_offsets.push(0);
    
    for img in 0..num_images {
        for e in problem.image_elements(img) {
            image_elements.push(e as u32);
        }
        image_elements_offsets.push(image_elements.len());
    }
    
    let image_elements = Arc::new(image_elements);
    let image_elements_offsets = Arc::new(image_elements_offsets);

    let mut image_costs: Option<Arc<Vec<u64>>> = None;
    let mut element_areas: Option<Arc<Vec<u64>>> = None;
    let mut clear_elements: Option<Arc<Vec<u32>>> = None;
    let mut clear_elements_offsets: Option<Arc<Vec<usize>>> = None;
    let mut resolution_levels: Option<Arc<Vec<u64>>> = None;
    let mut image_resolution_level: Option<Arc<Vec<u8>>> = None;
    let mut incidence_levels: Option<Arc<Vec<u64>>> = None;
    let mut image_incidence_level: Option<Arc<Vec<u8>>> = None;

    for i in 0..D {
        match problem.objective(i) {
            crate::objectives::ObjectiveState::TotalCost { costs, .. } => {
                image_costs = Some(Arc::new(costs.clone()));
            }
            crate::objectives::ObjectiveState::CloudyArea {
                clear_images,
                areas,
                ..
            } => {
                element_areas = Some(Arc::new(areas.clone()));
                
                // Build Clear Elements CSR
                let mut ce = Vec::with_capacity(num_images * 20);
                let mut ce_offsets = Vec::with_capacity(num_images + 1);
                ce_offsets.push(0);
                
                for bits in clear_images.iter() {
                    for e in bits.ones() {
                        ce.push(e as u32);
                    }
                    ce_offsets.push(ce.len());
                }
                
                clear_elements = Some(Arc::new(ce));
                clear_elements_offsets = Some(Arc::new(ce_offsets));
            }
            crate::objectives::ObjectiveState::MinResolution { resolutions, .. } => {
                let mut levels = resolutions.clone();
                levels.sort_unstable();
                levels.dedup();
                let num_levels = levels.len();
                #[allow(clippy::checked_conversions)]
                let max_levels = usize::from(u8::MAX);
                assert!(
                    num_levels <= max_levels,
                    "too many distinct resolution values ({num_levels}) for AltMinResolutionState"
                );

                let image_levels: Vec<u8> = resolutions
                    .iter()
                    .map(|&r| {
                        let idx = levels
                            .binary_search(&r)
                            .expect("resolution value should exist in levels");
                        idx as u8
                    })
                    .collect();

                resolution_levels = Some(Arc::new(levels));
                image_resolution_level = Some(Arc::new(image_levels));
            }
            crate::objectives::ObjectiveState::MaxIncidenceAngle {
                incidence_angles,
                ..
            } => {
                let mut levels = incidence_angles.clone();
                levels.sort_unstable();
                levels.dedup();
                let num_levels = levels.len();
                #[allow(clippy::checked_conversions)]
                let max_levels = usize::from(u8::MAX);
                assert!(
                    num_levels <= max_levels,
                    "too many distinct incidence angle values ({num_levels}) for AltMaxIncidenceAngleState"
                );

                let image_levels: Vec<u8> = incidence_angles
                    .iter()
                    .map(|&a| {
                        let idx = levels
                            .binary_search(&a)
                            .expect("incidence angle value should exist in levels");
                        idx as u8
                    })
                    .collect();

                incidence_levels = Some(Arc::new(levels));
                image_incidence_level = Some(Arc::new(image_levels));
            }
        }
    }

    let shared = Arc::new(AltTrackerSharedData {
        image_costs: image_costs.expect("AltTrackerArray requires TotalCost objective"),
        element_areas: element_areas.expect("AltTrackerArray requires CloudyArea objective"),
        clear_elements: clear_elements.expect("AltTrackerArray requires CloudyArea objective"),
        clear_elements_offsets: clear_elements_offsets.expect("AltTrackerArray requires CloudyArea objective"),
        resolution_levels: resolution_levels.expect("AltTrackerArray requires MinResolution objective"),
        image_resolution_level: image_resolution_level
            .expect("AltTrackerArray requires MinResolution objective"),
        image_elements,
        image_elements_offsets,
        incidence_levels: incidence_levels
            .expect("AltTrackerArray requires MaxIncidenceAngle objective"),
        image_incidence_level: image_incidence_level
            .expect("AltTrackerArray requires MaxIncidenceAngle objective"),
    });

    CACHE.with(|cache| {
        cache.borrow_mut().insert(key, Arc::clone(&shared));
    });
    shared
}

/// State for tracking Maximum Incidence Angle incrementally.
#[derive(Clone, Debug)]
pub struct AltMaxIncidenceAngleState {
    pub incidence_levels: Arc<Vec<u64>>,
    pub image_incidence_level: Arc<Vec<u8>>,
    pub level_counts: Vec<u16>,
    /// Sentinel value 255 means empty.
    pub current_max_level: u8,
    pub current_max: u64,
}

/// Alternative tracker enum (one variant per objective).
#[derive(Clone, Debug)]
pub enum AltTracker {
    TotalCost(AltTotalCostState),
    CloudyArea(AltCloudyAreaState),
    MinResolution(AltMinResolutionState),
    MaxIncidenceAngle(AltMaxIncidenceAngleState),
}

impl AltTracker {
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

impl<const D: usize> ObjectiveTracker<D> for AltTotalCostState {
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

impl<const D: usize> ObjectiveTracker<D> for AltCloudyAreaState {
    fn peek_removal_delta(
        &self,
        image_index: usize,
        _problem: &impl SetCoverProblem<D>,
        _solution: &impl ImageSet<D>,
    ) -> i64 {
        let mut delta: i64 = 0;
        let start = unsafe { *self.clear_elements_offsets.get_unchecked(image_index) };
        let end = unsafe { *self.clear_elements_offsets.get_unchecked(image_index + 1) };
        let clear_elements = unsafe { self.clear_elements.get_unchecked(start..end) };
        
        for &element_u32 in clear_elements {
            let element_idx = element_u32 as usize;
            // Safety: element indices come from the problem's own data.
            unsafe {
                if *self.counts.get_unchecked(element_idx) == 1 {
                    delta += *self.element_areas.get_unchecked(element_idx) as i64;
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
        let start = unsafe { *self.clear_elements_offsets.get_unchecked(image_index) };
        let end = unsafe { *self.clear_elements_offsets.get_unchecked(image_index + 1) };
        let clear_elements = unsafe { self.clear_elements.get_unchecked(start..end) };
        
        for &element_u32 in clear_elements {
            let element_idx = element_u32 as usize;
            unsafe {
                if *self.counts.get_unchecked(element_idx) == 0 {
                    delta -= *self.element_areas.get_unchecked(element_idx) as i64;
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
        let start = unsafe { *self.clear_elements_offsets.get_unchecked(image_index) };
        let end = unsafe { *self.clear_elements_offsets.get_unchecked(image_index + 1) };
        let clear_elements = unsafe { self.clear_elements.get_unchecked(start..end) };
        
        for &element_u32 in clear_elements {
            let element_idx = element_u32 as usize;
            unsafe {
                let element_area = *self.element_areas.get_unchecked(element_idx);
                let count = self.counts.get_unchecked_mut(element_idx);
                *count -= 1;
                if *count == 0 {
                    self.current_area += element_area;
                    delta += element_area as i64;
                }
            }
        }
        delta
    }

    fn track_image_addition(
        &mut self,
        image_index: usize,
        _problem: &impl SetCoverProblem<D>,
    ) -> i64 {
        let mut delta = 0i64;
        let start = unsafe { *self.clear_elements_offsets.get_unchecked(image_index) };
        let end = unsafe { *self.clear_elements_offsets.get_unchecked(image_index + 1) };
        let clear_elements = unsafe { self.clear_elements.get_unchecked(start..end) };
        
        for &element_u32 in clear_elements {
            let element_idx = element_u32 as usize;
            unsafe {
                let element_area = *self.element_areas.get_unchecked(element_idx);
                let count = self.counts.get_unchecked_mut(element_idx);
                if *count == 0 {
                    self.current_area -= element_area;
                    delta -= element_area as i64;
                }
                *count += 1;
            }
        }
        delta
    }

    fn value(&self) -> u64 {
        self.current_area
    }
}

impl<const D: usize> ObjectiveTracker<D> for AltMinResolutionState {
    fn peek_removal_delta(
        &self,
        image_index: usize,
        _problem: &impl SetCoverProblem<D>,
        _solution: &impl ImageSet<D>,
    ) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;
        let resolution_levels = &self.resolution_levels;
        
        let start = unsafe { *self.image_elements_offsets.get_unchecked(image_index) };
        let end = unsafe { *self.image_elements_offsets.get_unchecked(image_index + 1) };
        let image_elements = unsafe { self.image_elements.get_unchecked(start..end) };

        // Fast path: instances with exactly two distinct resolution values.
        if self.two_level {
            let low_val = self.two_level_low;
            let high_val = self.two_level_high;
            let diff = (high_val - low_val) as i64;
            let mut delta: i64 = 0;

            for &element_u32 in image_elements {
                let element_idx = element_u32 as usize;
                unsafe {
                    let packed = *self.element_packed_counts.get_unchecked(element_idx);
                    let c0 = (packed & 0xFFFF) as u16;
                    let c1 = (packed >> 16) as u16;

                    if img_level == 0 {
                        // Removing a low-res image only matters if it is the last low-res cover.
                        if c0 == 1 {
                            if c1 > 0 {
                                delta += diff;
                            } else {
                                delta -= low_val as i64;
                            }
                        }
                    } else {
                        // Removing a high-res image only matters if it is the last cover.
                        if c0 == 0 && c1 == 1 {
                            delta -= high_val as i64;
                        }
                    }
                }
            }

            return delta;
        }

        if !self.element_packed_small.is_empty() {
            let mut delta: i64 = 0;
            let shift = img_level * 8;
            let mask_lower = (1u64 << shift) - 1;
            // Mask to clear the current level byte and all lower bytes.
            let mask_higher = !((1u64 << (shift + 8)) - 1);

            for &element_u32 in image_elements {
                let element_idx = element_u32 as usize;
                unsafe {
                    let packed = *self.element_packed_small.get_unchecked(element_idx);
                    let count = (packed >> shift) & 0xFF;
                    
                    if count == 1 {
                        // Removing the last image at this level.
                        // Check if this level was the minimum (all lower counts are 0).
                        if (packed & mask_lower) == 0 {
                            let current_val = resolution_levels[img_level];
                            let remaining = packed & mask_higher;
                            
                            if remaining == 0 {
                                delta -= current_val as i64;
                            } else {
                                let next_level = remaining.trailing_zeros() / 8;
                                let next_val = resolution_levels[next_level as usize];
                                delta += (next_val as i64) - (current_val as i64);
                            }
                        }
                    }
                }
            }
            return delta;
        }

        // General case (many levels).
        let num_levels = resolution_levels.len();
        let mut delta: i64 = 0;
        let mask_words = self.mask_words as usize;

        for &element_u32 in image_elements {
            let element_idx = element_u32 as usize;
            let current_min_level = unsafe { *self.element_min_level.get_unchecked(element_idx) };
            if current_min_level == u8::MAX {
                continue;
            }

            let current_min_level_usize = current_min_level as usize;
            if img_level != current_min_level_usize {
                continue;
            }

            let base = element_idx * num_levels;
            let count_at_level = unsafe { *self.element_level_counts.get_unchecked(base + img_level) };
            if count_at_level > 1 {
                continue;
            }
            
            // If count is 1, and it's the current minimum, finding next min is needed.
            let next_level = if mask_words == 1 {
                let mask = unsafe { *self.element_level_masks.get_unchecked(element_idx) };
                let new_mask = mask & !(1u64 << img_level);
                if new_mask == 0 {
                    u8::MAX
                } else {
                    new_mask.trailing_zeros() as u8
                }
            } else {
                let element_masks_base = element_idx * mask_words;
                let word_idx = img_level / 64;
                let bit_idx = img_level % 64;

                let first_word_mask = !(1u64 << bit_idx);
                let first_word = unsafe { 
                    *self.element_level_masks.get_unchecked(element_masks_base + word_idx) 
                } & first_word_mask;

                if first_word != 0 {
                    (word_idx * 64 + first_word.trailing_zeros() as usize) as u8
                } else {
                    let mut found = u8::MAX;
                    let mut w_i = word_idx + 1;
                    while w_i < mask_words {
                        let w = unsafe {
                            *self
                                .element_level_masks
                                .get_unchecked(element_masks_base + w_i)
                        };
                        if w != 0 {
                            found = (w_i * 64 + w.trailing_zeros() as usize) as u8;
                            break;
                        }
                        w_i += 1;
                    }
                    found
                }
            };

            let current_val = resolution_levels[current_min_level_usize];
            let next_val = if next_level == u8::MAX {
                0
            } else {
                resolution_levels[next_level as usize]
            };
            delta += (next_val as i64) - (current_val as i64);
        }

        delta
    }

    fn peek_addition_delta(
        &self,
        image_index: usize,
        _problem: &impl SetCoverProblem<D>,
        _solution: &impl ImageSet<D>,
    ) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;
        let resolution_levels = &self.resolution_levels;

        let start = unsafe { *self.image_elements_offsets.get_unchecked(image_index) };
        let end = unsafe { *self.image_elements_offsets.get_unchecked(image_index + 1) };
        let image_elements = unsafe { self.image_elements.get_unchecked(start..end) };

        if self.two_level {
            let low_val = self.two_level_low;
            let high_val = self.two_level_high;
            let diff = (high_val - low_val) as i64;
            let mut delta: i64 = 0;

            for &element_u32 in image_elements {
                let element_idx = element_u32 as usize;
                unsafe {
                    let packed = *self.element_packed_counts.get_unchecked(element_idx);
                    let c0 = (packed & 0xFFFF) as u16;
                    let c1 = (packed >> 16) as u16;

                    if img_level == 0 {
                        if c0 == 0 {
                            if c1 > 0 {
                                delta -= diff;
                            } else {
                                delta += low_val as i64;
                            }
                        }
                    } else if c0 == 0 && c1 == 0 {
                        delta += high_val as i64;
                    }
                }
            }

            return delta;
        }

        if !self.element_packed_small.is_empty() {
             let mut delta: i64 = 0;
             let shift = img_level * 8;
             let mask_lower = (1u64 << shift) - 1;
             
             for &element_u32 in image_elements {
                 let element_idx = element_u32 as usize;
                 unsafe {
                     let packed = *self.element_packed_small.get_unchecked(element_idx);
                     // Only changes if no better (lower index) levels count > 0
                     if (packed & mask_lower) == 0 {
                         let count = (packed >> shift) & 0xFF;
                         if count == 0 {
                             let current_val = resolution_levels[img_level];
                             if packed == 0 {
                                 delta += current_val as i64;
                             } else {
                                 // Was covered by something worse (higher index)
                                 let old_min_level = packed.trailing_zeros() / 8;
                                 let old_val = resolution_levels[old_min_level as usize];
                                 delta += (current_val as i64) - (old_val as i64);
                             }
                         }
                     }
                 }
             }
             return delta;
        }

        let mut delta: i64 = 0;
        let img_val = resolution_levels[img_level];

        for &element_u32 in image_elements {
            let element_idx = element_u32 as usize;
            let current_min_level = unsafe { *self.element_min_level.get_unchecked(element_idx) };
            if current_min_level == u8::MAX {
                delta += img_val as i64;
                continue;
            }

            let current_level = current_min_level as usize;
            if img_level < current_level {
                let current_val = resolution_levels[current_level];
                delta -= (current_val - img_val) as i64;
            }
        }

        delta
    }

    #[allow(clippy::too_many_lines)]
    fn track_image_removal(
        &mut self,
        image_index: usize,
        _problem: &impl SetCoverProblem<D>,
    ) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;
        let resolution_levels = &self.resolution_levels;

        let start = unsafe { *self.image_elements_offsets.get_unchecked(image_index) };
        let end = unsafe { *self.image_elements_offsets.get_unchecked(image_index + 1) };
        let image_elements = unsafe { self.image_elements.get_unchecked(start..end) };

        if self.two_level {
            let low_val = self.two_level_low;
            let high_val = self.two_level_high;
            let diff = (high_val - low_val) as i64;
            let mut delta = 0i64;

            for &element_u32 in image_elements {
                let element_idx = element_u32 as usize;
                unsafe {
                    let slot = self.element_packed_counts.get_unchecked_mut(element_idx);
                    let packed = *slot;
                    let c0 = (packed & 0xFFFF) as u16;
                    let c1 = (packed >> 16) as u16;

                    if img_level == 0 {
                        if c0 == 0 {
                            continue;
                        }
                        if c0 == 1 {
                            // Low count goes to zero.
                            if c1 > 0 {
                                self.current_sum += high_val - low_val;
                                delta += diff;
                            } else {
                                self.current_sum -= low_val;
                                delta -= low_val as i64;
                            }
                        }
                        *slot = packed - 1;
                    } else {
                        if c1 == 0 {
                            continue;
                        }
                        if c1 == 1 && c0 == 0 {
                            self.current_sum -= high_val;
                            delta -= high_val as i64;
                        }
                        *slot = packed - (1u32 << 16);
                    }
                }
            }

            return delta;
        }

        if !self.element_packed_small.is_empty() {
            let mut delta: i64 = 0;
            let shift = img_level * 8;
            let mask_lower = (1u64 << shift) - 1; 
            
            for &element_u32 in image_elements {
                let element_idx = element_u32 as usize;
                
                unsafe {
                    let slot = self.element_packed_small.get_unchecked_mut(element_idx);
                    let packed = *slot;
                    let count = (packed >> shift) & 0xFF;
                    
                    *slot = packed - (1u64 << shift);
                    
                    if count == 1 {
                        // Count dropped to 0. Min might change.
                        if (packed & mask_lower) == 0 {
                            // Was min level.
                            let remaining = *slot;
                            let current_val = resolution_levels[img_level];
                            
                            if remaining == 0 {
                                self.current_sum -= current_val;
                                delta -= current_val as i64;
                            } else {
                                // Find next min
                                let next_level = remaining.trailing_zeros() / 8;
                                let next_val = resolution_levels[next_level as usize];
                                self.current_sum = self.current_sum - current_val + next_val;
                                delta += (next_val as i64) - (current_val as i64);
                            }
                        }
                    }
                }
            }
            return delta;
        }

        let num_levels = resolution_levels.len();
        let mut delta = 0i64;
        let mask_words = self.mask_words as usize;

        for &element_u32 in image_elements {
            let element_idx = element_u32 as usize;
            let base = element_idx * num_levels;
            let count_slot = unsafe { self.element_level_counts.get_unchecked_mut(base + img_level) };
            
            // Assume count > 0 for valid removal, but check only if needed or keep check for safety.
            // Keeping safety for now:
            if *count_slot == 0 { continue; }
            *count_slot -= 1;

            if *count_slot > 0 {
                // If count remains positive, min level definitely cannot change (unless we weren't min, which we handle later).
                // Actually, if we weren't min, we don't care.
                // If we were min, and count > 0, we are still min.
                // So no change to min level.
                // But we must NOT update masks if count > 0.
                continue;
            }
            
            // Claim: Count dropped to 0. Update mask.
            let element_masks_base = element_idx * mask_words;
            let word_idx = img_level / 64;
            let bit_idx = img_level % 64;
            unsafe {
                let m = self
                    .element_level_masks
                    .get_unchecked_mut(element_masks_base + word_idx);
                *m &= !(1u64 << bit_idx);
            }
            
            // Now check if this affects the minimum.
            let current_min_level = unsafe { *self.element_min_level.get_unchecked(element_idx) };
            
            // Optimization: if img_level > current_min_level, no change to min.
            if (img_level as u8) > current_min_level {
                // If it was equal, we need to check. If it was less (impossible if min is maintained), ...
                continue;
            }
            
            // Here img_level == current_min_level (or <, which implies logic error previously, but equality is the constraint).
            // Since count dropped to 0, we must find new min.

            // Removed the last image at the current minimum level: find next minimum level.
            let next_level = if mask_words == 1 {
                let mask = unsafe { *self.element_level_masks.get_unchecked(element_idx) };
                if mask == 0 {
                    u8::MAX
                } else {
                    mask.trailing_zeros() as u8
                }
            } else {
                let mut found = u8::MAX;
                // Start search from word_idx, since clear bit might leave higher bits in same word.
                let mut w_i = word_idx;
                while w_i < mask_words {
                    let w = unsafe {
                        *self
                            .element_level_masks
                            .get_unchecked(element_masks_base + w_i)
                    };
                    // Mask out lower bits if using first word?
                    // No, because we are guaranteed that img_level was the MINIMUM.
                    // So there are no set bits < img_level.
                    // So simply trailing_zeros on the word is sufficient.
                    if w != 0 {
                        found = (w_i * 64 + w.trailing_zeros() as usize) as u8;
                        break;
                    }
                    w_i += 1;
                }
                found
            };

            let current_val = resolution_levels[current_min_level as usize];
            let next_val = if next_level == u8::MAX {
                0
            } else {
                resolution_levels[next_level as usize]
            };

            unsafe {
                *self.element_min_level.get_unchecked_mut(element_idx) = next_level;
            }
            self.current_sum = self.current_sum - current_val + next_val;
            delta += (next_val as i64) - (current_val as i64);
        }

        delta
    }

    fn track_image_addition(
        &mut self,
        image_index: usize,
        _problem: &impl SetCoverProblem<D>,
    ) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;
        let resolution_levels = &self.resolution_levels;

        let start = unsafe { *self.image_elements_offsets.get_unchecked(image_index) };
        let end = unsafe { *self.image_elements_offsets.get_unchecked(image_index + 1) };
        let image_elements = unsafe { self.image_elements.get_unchecked(start..end) };

        if self.two_level {
            let low_val = self.two_level_low;
            let high_val = self.two_level_high;
            let diff = (high_val - low_val) as i64;
            let mut delta = 0i64;

            for &element_u32 in image_elements {
                let element_idx = element_u32 as usize;
                unsafe {
                    let slot = self.element_packed_counts.get_unchecked_mut(element_idx);
                    let packed = *slot;
                    let c0 = (packed & 0xFFFF) as u16;
                    let c1 = (packed >> 16) as u16;

                    if img_level == 0 {
                        if c0 == 0 {
                            if c1 > 0 {
                                self.current_sum -= high_val - low_val;
                                delta -= diff;
                            } else {
                                self.current_sum += low_val;
                                delta += low_val as i64;
                            }
                        }
                        *slot = packed + 1;
                    } else {
                        if c0 == 0 && c1 == 0 {
                            self.current_sum += high_val;
                            delta += high_val as i64;
                        }
                        *slot = packed + (1u32 << 16);
                    }
                }
            }

            return delta;
        }

        if !self.element_packed_small.is_empty() {
             let mut delta: i64 = 0;
             let shift = img_level * 8;
             let mask_lower = (1u64 << shift) - 1;
             
             for &element_u32 in image_elements {
                 let element_idx = element_u32 as usize;
                 unsafe {
                     let slot = self.element_packed_small.get_unchecked_mut(element_idx);
                     let packed = *slot;
                     
                     if (packed & mask_lower) == 0 {
                         let count = (packed >> shift) & 0xFF;
                         if count == 0 {
                             let current_val = resolution_levels[img_level];
                             if packed == 0 {
                                 self.current_sum += current_val;
                                 delta += current_val as i64;
                             } else {
                                 let old_min_level = packed.trailing_zeros() / 8;
                                 let old_val = resolution_levels[old_min_level as usize];
                                 self.current_sum = self.current_sum - old_val + current_val;
                                 delta += (current_val as i64) - (old_val as i64);
                             }
                         }
                     }
                     *slot = packed + (1u64 << shift);
                 }
             }
             return delta;
        }

        let num_levels = resolution_levels.len();
        let mut delta = 0i64;
        let img_val = resolution_levels[img_level];
        let mask_words = self.mask_words as usize;

        for &element_u32 in image_elements {
            let element_idx = element_u32 as usize;
            let base = element_idx * num_levels;
            unsafe {
                let slot = self.element_level_counts.get_unchecked_mut(base + img_level);
                let was_zero = *slot == 0;
                *slot += 1;
                if was_zero {
                    let element_masks_base = element_idx * mask_words;
                    let word_idx = img_level / 64;
                    let bit_idx = img_level % 64;
                    let m = self
                        .element_level_masks
                        .get_unchecked_mut(element_masks_base + word_idx);
                    *m |= 1u64 << bit_idx;
                }
            }

            let current_min_level = unsafe { *self.element_min_level.get_unchecked(element_idx) };
            if current_min_level == u8::MAX {
                unsafe {
                    *self.element_min_level.get_unchecked_mut(element_idx) = img_level as u8;
                }
                self.current_sum += img_val;
                delta += img_val as i64;
                continue;
            }

            let current_level = current_min_level as usize;
            if img_level < current_level {
                let current_val = resolution_levels[current_level];
                unsafe {
                    *self.element_min_level.get_unchecked_mut(element_idx) = img_level as u8;
                }
                self.current_sum = self.current_sum - current_val + img_val;
                delta += (img_val as i64) - (current_val as i64);
            }
        }

        delta
    }

    fn value(&self) -> u64 {
        self.current_sum
    }
}

impl<const D: usize> ObjectiveTracker<D> for AltMaxIncidenceAngleState {
    fn peek_removal_delta(
        &self,
        image_index: usize,
        _problem: &impl SetCoverProblem<D>,
        _solution: &impl ImageSet<D>,
    ) -> i64 {
        let img_level = self.image_incidence_level[image_index];
        let current_max_level = self.current_max_level;
        let current_max = self.current_max;

        if current_max_level == u8::MAX {
            return 0;
        }

        if img_level < current_max_level {
            return 0;
        }

        let level_idx = current_max_level as usize;
        let count_at_max = unsafe { *self.level_counts.get_unchecked(level_idx) };
        if count_at_max > 1 {
            return 0;
        }

        // Find the next maximum level.
        let mut next_level: i32 = i32::from(current_max_level) - 1;
        while next_level >= 0 {
            let c = unsafe { *self.level_counts.get_unchecked(next_level as usize) };
            if c != 0 {
                let next_val = unsafe { *self.incidence_levels.get_unchecked(next_level as usize) };
                return (next_val as i64) - (current_max as i64);
            }
            next_level -= 1;
        }

        -(current_max as i64)
    }

    fn peek_addition_delta(
        &self,
        image_index: usize,
        _problem: &impl SetCoverProblem<D>,
        _solution: &impl ImageSet<D>,
    ) -> i64 {
        let img_level = self.image_incidence_level[image_index];
        let current_max_level = self.current_max_level;
        let current_max = self.current_max;

        if current_max_level == u8::MAX || img_level > current_max_level {
            let next_val = unsafe { *self.incidence_levels.get_unchecked(img_level as usize) };
            (next_val as i64) - (current_max as i64)
        } else {
            0
        }
    }

    fn track_image_removal(
        &mut self,
        image_index: usize,
        _problem: &impl SetCoverProblem<D>,
    ) -> i64 {
        let img_level = self.image_incidence_level[image_index] as usize;
        let old_max = self.current_max;

        unsafe {
            let slot = self.level_counts.get_unchecked_mut(img_level);
            if *slot != 0 {
                *slot -= 1;
            }
        }

        if self.current_max_level == u8::MAX {
            return (self.current_max as i64) - (old_max as i64);
        }

        if img_level as u8 == self.current_max_level {
            let still_any = unsafe { *self.level_counts.get_unchecked(img_level) } != 0;
            if !still_any {
                let mut next_level: i32 = (img_level as i32) - 1;
                while next_level >= 0 {
                    let c = unsafe { *self.level_counts.get_unchecked(next_level as usize) };
                    if c != 0 {
                        self.current_max_level = next_level as u8;
                        self.current_max =
                            unsafe { *self.incidence_levels.get_unchecked(next_level as usize) };
                        return (self.current_max as i64) - (old_max as i64);
                    }
                    next_level -= 1;
                }
                self.current_max_level = u8::MAX;
                self.current_max = 0;
            }
        }

        (self.current_max as i64) - (old_max as i64)
    }

    fn track_image_addition(
        &mut self,
        image_index: usize,
        _problem: &impl SetCoverProblem<D>,
    ) -> i64 {
        let img_level = self.image_incidence_level[image_index];
        let old_max = self.current_max;

        unsafe {
            *self.level_counts.get_unchecked_mut(img_level as usize) += 1;
        }

        if self.current_max_level == u8::MAX || img_level > self.current_max_level {
            self.current_max_level = img_level;
            self.current_max = unsafe { *self.incidence_levels.get_unchecked(img_level as usize) };
        }

        (self.current_max as i64) - (old_max as i64)
    }

    fn value(&self) -> u64 {
        self.current_max
    }
}

impl<const D: usize> ObjectiveTracker<D> for AltTracker {
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

    fn value(&self) -> u64 {
        Self::value(self)
    }
}

/// Alternative array-based tracker collection.
#[derive(Clone, Debug)]
pub struct AltTrackerArray<const D: usize> {
    trackers: [AltTracker; D],
}

impl<const D: usize> TrackerCollection<D> for AltTrackerArray<D> {
    type Tracker = AltTracker;

    fn get(&self, index: usize) -> &AltTracker {
        &self.trackers[index]
    }

    fn get_mut(&mut self, index: usize) -> &mut AltTracker {
        &mut self.trackers[index]
    }

    fn new(problem: &impl SetCoverProblem<D>) -> Self {
        let shared = alt_shared_data(problem);

        let trackers = std::array::from_fn(|i| match problem.objective(i) {
            crate::objectives::ObjectiveState::TotalCost { costs, .. } => {
                let _ = costs;
                AltTracker::TotalCost(AltTotalCostState {
                    current_cost: 0,
                    image_costs: Arc::clone(&shared.image_costs),
                })
            }
            crate::objectives::ObjectiveState::CloudyArea {
                clear_images,
                areas,
                ..
            } => {
                let _ = (clear_images, areas);
                let total_area: u64 = shared.element_areas.iter().sum();
                let mut cloudy = FixedBitSet::with_capacity(problem.num_elements());
                cloudy.set_range(.., true);

                AltTracker::CloudyArea(AltCloudyAreaState {
                    counts: vec![0; problem.num_elements()],
                    cloudy_elements: cloudy,
                    current_area: total_area,
                    element_areas: Arc::clone(&shared.element_areas),
                    clear_elements: Arc::clone(&shared.clear_elements),
                    clear_elements_offsets: Arc::clone(&shared.clear_elements_offsets),
                })
            }
            crate::objectives::ObjectiveState::MinResolution { resolutions, .. } => {
                let _ = resolutions;
                let num_levels = shared.resolution_levels.len();
                let two_level = num_levels == 2;
                let small_level = num_levels > 2 && num_levels <= 8;
                let two_level_low = shared.resolution_levels[0];
                let two_level_high = shared.resolution_levels[num_levels - 1];
                let mask_words: usize = num_levels.div_ceil(64);

                AltTracker::MinResolution(AltMinResolutionState {
                    resolution_levels: Arc::clone(&shared.resolution_levels),
                    two_level,
                    two_level_low,
                    two_level_high,
                    image_resolution_level: Arc::clone(&shared.image_resolution_level),
                    image_elements: Arc::clone(&shared.image_elements),
                    image_elements_offsets: Arc::clone(&shared.image_elements_offsets),
                    element_packed_counts: if two_level {
                        vec![0; problem.num_elements()]
                    } else {
                        Vec::new()
                    },
                    element_packed_small: if small_level {
                         vec![0; problem.num_elements()]
                    } else {
                         Vec::new()
                    },
                    element_level_counts: if two_level || small_level {
                        Vec::new()
                    } else {
                        vec![0; problem.num_elements() * num_levels]
                    },
                    element_level_masks: if two_level || small_level {
                        Vec::new()
                    } else {
                        vec![0; problem.num_elements() * mask_words]
                    },
                    mask_words: if two_level || small_level { 0 } else { mask_words as u8 },
                    element_min_level: if two_level || small_level {
                        Vec::new()
                    } else {
                        vec![u8::MAX; problem.num_elements()]
                    },
                    current_sum: 0,
                })
            }
            crate::objectives::ObjectiveState::MaxIncidenceAngle {
                incidence_angles, ..
            } => {
                let _ = incidence_angles;
                AltTracker::MaxIncidenceAngle(AltMaxIncidenceAngleState {
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
        let mut out = [0u64; D];
        for (i, out_i) in out.iter_mut().enumerate() {
            *out_i = self.trackers[i].value();
        }
        out
    }

    fn peek_removal_delta(
        &self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
        solution: &impl ImageSet<D>,
    ) -> [i64; D] {
        let mut out = [0i64; D];
        for (i, out_i) in out.iter_mut().enumerate() {
            *out_i = self.trackers[i].peek_removal_delta(image_index, problem, solution);
        }
        out
    }

    fn peek_addition_delta(
        &self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
        solution: &impl ImageSet<D>,
    ) -> [i64; D] {
        let mut out = [0i64; D];
        for (i, out_i) in out.iter_mut().enumerate() {
            *out_i = self.trackers[i].peek_addition_delta(image_index, problem, solution);
        }
        out
    }

    fn track_image_removal(
        &mut self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
    ) -> [i64; D] {
        let mut out = [0i64; D];
        for (i, out_i) in out.iter_mut().enumerate() {
            *out_i = self.trackers[i].track_image_removal(image_index, problem);
        }
        out
    }

    fn track_image_addition(
        &mut self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
    ) -> [i64; D] {
        let mut out = [0i64; D];
        for (i, out_i) in out.iter_mut().enumerate() {
            *out_i = self.trackers[i].track_image_addition(image_index, problem);
        }
        out
    }

    fn values(&self) -> [u64; D] {
        let mut out = [0u64; D];
        for (i, out_i) in out.iter_mut().enumerate() {
            *out_i = self.trackers[i].value();
        }
        out
    }

    fn initialize_from(&mut self, solution: &impl ImageSet<D>, problem: &impl SetCoverProblem<D>) {
        *self = Self::new(problem);
        for img in solution.selected_images() {
            self.track_image_addition(img, problem);
        }
    }
}
