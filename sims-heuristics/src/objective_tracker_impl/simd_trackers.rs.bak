//! SIMD-optimized objective tracker implementation.
//!
//! Key optimizations:
//! 1. Separate count arrays for two-level MinResolution (c0, c1)
//! 2. Branchless delta computation with lookup tables
//! 3. Software prefetching for CSR element iteration
//! 4. SIMD vectorization using portable_simd (when available)

use fixedbitset::FixedBitSet;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use crate::objective_tracker::{ObjectiveTracker, TrackerCollection};
use crate::problem::SetCoverProblem;
use crate::solution::ImageSet;

/// Interval representation for run-length compressed element lists.
/// Each interval represents a consecutive range [start, start+len).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Interval {
    pub start: u32,
    pub len: u32,
}

/// Build interval-compressed representation from CSR data.
/// Returns (intervals, offsets) where offsets[i]..offsets[i+1] are intervals for image i.
fn build_intervals(elements: &[u32], offsets: &[usize]) -> (Vec<Interval>, Vec<usize>) {
    let num_images = offsets.len() - 1;
    let mut intervals = Vec::with_capacity(elements.len() / 4); // Expect ~4x compression
    let mut interval_offsets = Vec::with_capacity(num_images + 1);
    interval_offsets.push(0);
    
    for img in 0..num_images {
        let start = offsets[img];
        let end = offsets[img + 1];
        if start == end {
            interval_offsets.push(intervals.len());
            continue;
        }
        
        let img_elements = &elements[start..end];
        let mut run_start = img_elements[0];
        let mut run_len = 1u32;
        
        for &e in &img_elements[1..] {
            if e == run_start + run_len {
                run_len += 1;
            } else {
                intervals.push(Interval { start: run_start, len: run_len });
                run_start = e;
                run_len = 1;
            }
        }
        intervals.push(Interval { start: run_start, len: run_len });
        interval_offsets.push(intervals.len());
    }
    
    (intervals, interval_offsets)
}

/// Shared immutable data for SIMD trackers.
#[derive(Debug)]
pub struct SimdTrackerSharedData {
    pub image_costs: Arc<Vec<u64>>,
    pub element_areas: Arc<Vec<u64>>,
    // CSR format for clear elements
    pub clear_elements: Arc<Vec<u32>>,
    pub clear_elements_offsets: Arc<Vec<usize>>,
    pub resolution_levels: Arc<Vec<u64>>,
    pub image_resolution_level: Arc<Vec<u8>>,
    // CSR format for image elements
    pub image_elements: Arc<Vec<u32>>,
    pub image_elements_offsets: Arc<Vec<usize>>,
    pub incidence_levels: Arc<Vec<u64>>,
    pub image_incidence_level: Arc<Vec<u8>>,
    // Interval-compressed image elements (for vectorized range loops)
    pub image_intervals: Arc<Vec<Interval>>,
    pub image_intervals_offsets: Arc<Vec<usize>>,
    // Interval-compressed clear elements
    pub clear_intervals: Arc<Vec<Interval>>,
    pub clear_intervals_offsets: Arc<Vec<usize>>,
}

pub(super) fn simd_shared_data<const D: usize>(problem: &impl SetCoverProblem<D>) -> Arc<SimdTrackerSharedData> {
    #[allow(clippy::ref_as_ptr)]
    let key = problem as *const _ as usize;

    thread_local! {
        static CACHE: RefCell<HashMap<usize, Arc<SimdTrackerSharedData>>> = RefCell::new(HashMap::new());
    }

    if let Some(hit) = CACHE.with(|cache| cache.borrow().get(&key).cloned()) {
        return hit;
    }

    let num_images = problem.num_images();

    // Build Image Elements CSR
    let mut image_elements = Vec::with_capacity(num_images * 20);
    let mut image_elements_offsets = Vec::with_capacity(num_images + 1);
    image_elements_offsets.push(0);
    for img in 0..num_images {
        for e in problem.image_elements(img) {
            image_elements.push(e as u32);
        }
        image_elements_offsets.push(image_elements.len());
    }
    
    // Build interval-compressed image elements
    let (image_intervals, image_intervals_offsets) = build_intervals(&image_elements, &image_elements_offsets);
    
    let image_elements = Arc::new(image_elements);
    let image_elements_offsets = Arc::new(image_elements_offsets);

    let mut image_costs: Option<Arc<Vec<u64>>> = None;
    let mut element_areas: Option<Arc<Vec<u64>>> = None;
    let mut clear_elements: Option<Arc<Vec<u32>>> = None;
    let mut clear_elements_offsets: Option<Arc<Vec<usize>>> = None;
    let mut clear_intervals: Option<Arc<Vec<Interval>>> = None;
    let mut clear_intervals_offsets: Option<Arc<Vec<usize>>> = None;
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
                let mut ce = Vec::with_capacity(num_images * 20);
                let mut ce_offsets = Vec::with_capacity(num_images + 1);
                ce_offsets.push(0);
                for bits in clear_images.iter() {
                    for e in bits.ones() {
                        ce.push(e as u32);
                    }
                    ce_offsets.push(ce.len());
                }
                // Build interval-compressed clear elements
                let (ci, ci_offsets) = build_intervals(&ce, &ce_offsets);
                clear_elements = Some(Arc::new(ce));
                clear_elements_offsets = Some(Arc::new(ce_offsets));
                clear_intervals = Some(Arc::new(ci));
                clear_intervals_offsets = Some(Arc::new(ci_offsets));
            }
            crate::objectives::ObjectiveState::MinResolution { resolutions, .. } => {
                let mut levels = resolutions.clone();
                levels.sort_unstable();
                levels.dedup();
                let image_levels: Vec<u8> = resolutions
                    .iter()
                    .map(|&r| levels.binary_search(&r).unwrap() as u8)
                    .collect();
                resolution_levels = Some(Arc::new(levels));
                image_resolution_level = Some(Arc::new(image_levels));
            }
            crate::objectives::ObjectiveState::MaxIncidenceAngle { incidence_angles, .. } => {
                let mut levels = incidence_angles.clone();
                levels.sort_unstable();
                levels.dedup();
                let image_levels: Vec<u8> = incidence_angles
                    .iter()
                    .map(|&a| levels.binary_search(&a).unwrap() as u8)
                    .collect();
                incidence_levels = Some(Arc::new(levels));
                image_incidence_level = Some(Arc::new(image_levels));
            }
        }
    }

    let shared = Arc::new(SimdTrackerSharedData {
        image_costs: image_costs.unwrap(),
        element_areas: element_areas.unwrap(),
        clear_elements: clear_elements.unwrap(),
        clear_elements_offsets: clear_elements_offsets.unwrap(),
        clear_intervals: clear_intervals.unwrap(),
        clear_intervals_offsets: clear_intervals_offsets.unwrap(),
        resolution_levels: resolution_levels.unwrap(),
        image_resolution_level: image_resolution_level.unwrap(),
        image_elements,
        image_elements_offsets,
        image_intervals: Arc::new(image_intervals),
        image_intervals_offsets: Arc::new(image_intervals_offsets),
        incidence_levels: incidence_levels.unwrap(),
        image_incidence_level: image_incidence_level.unwrap(),
    });

    CACHE.with(|cache| {
        cache.borrow_mut().insert(key, Arc::clone(&shared));
    });
    shared
}

// =============================================================================
// TotalCost - unchanged, already O(1) per image
// =============================================================================

#[derive(Clone, Debug)]
pub struct SimdTotalCostState {
    pub current_cost: u64,
    pub image_costs: Arc<Vec<u64>>,
}

impl<const D: usize> ObjectiveTracker<D> for SimdTotalCostState {
    #[inline(always)]
    fn peek_removal_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        -(self.image_costs[image_index] as i64)
    }

    #[inline(always)]
    fn peek_addition_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        self.image_costs[image_index] as i64
    }

    #[inline(always)]
    fn track_image_removal(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let cost = self.image_costs[image_index];
        self.current_cost -= cost;
        -(cost as i64)
    }

    #[inline(always)]
    fn track_image_addition(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let cost = self.image_costs[image_index];
        self.current_cost += cost;
        cost as i64
    }

    fn value(&self) -> u64 {
        self.current_cost
    }
}

// =============================================================================
// CloudyArea - optimized with packed (count, area) for single cache line access
// =============================================================================

#[derive(Clone, Debug)]
pub struct SimdCloudyAreaState {
    // Packed: lower 16 bits = count, upper 48 bits = area (scaled to fit)
    // Actually, areas can be large, so we keep them separate but use u32 counts
    pub counts: Vec<u16>,
    pub cloudy_elements: FixedBitSet,
    pub current_area: u64,
    pub element_areas: Arc<Vec<u64>>,
    pub clear_elements: Arc<Vec<u32>>,
    pub clear_elements_offsets: Arc<Vec<usize>>,
    // Interval-compressed clear elements for vectorized range loops
    pub clear_intervals: Arc<Vec<Interval>>,
    pub clear_intervals_offsets: Arc<Vec<usize>>,
}

impl<const D: usize> ObjectiveTracker<D> for SimdCloudyAreaState {
    fn peek_removal_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let start = unsafe { *self.clear_elements_offsets.get_unchecked(image_index) };
        let end = unsafe { *self.clear_elements_offsets.get_unchecked(image_index + 1) };
        let clear_elements = unsafe { self.clear_elements.get_unchecked(start..end) };

        let mut delta: i64 = 0;
        for &element_u32 in clear_elements {
            let idx = element_u32 as usize;
            unsafe {
                let is_last = (*self.counts.get_unchecked(idx) == 1) as i64;
                delta += is_last * (*self.element_areas.get_unchecked(idx) as i64);
            }
        }
        delta
    }

    fn peek_addition_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let start = unsafe { *self.clear_elements_offsets.get_unchecked(image_index) };
        let end = unsafe { *self.clear_elements_offsets.get_unchecked(image_index + 1) };
        let clear_elements = unsafe { self.clear_elements.get_unchecked(start..end) };

        let mut delta: i64 = 0;
        for &element_u32 in clear_elements {
            let idx = element_u32 as usize;
            unsafe {
                let is_first = (*self.counts.get_unchecked(idx) == 0) as i64;
                delta -= is_first * (*self.element_areas.get_unchecked(idx) as i64);
            }
        }
        delta
    }

    fn track_image_removal(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let start = unsafe { *self.clear_elements_offsets.get_unchecked(image_index) };
        let end = unsafe { *self.clear_elements_offsets.get_unchecked(image_index + 1) };

        let clear_ptr = self.clear_elements.as_ptr();
        let counts_ptr = self.counts.as_mut_ptr();
        let areas_ptr = self.element_areas.as_ptr();

        let len = end - start;
        let unrolled = len / 4;
        let remainder = len % 4;
        let mut total_add = 0u64;
        
        // Unrolled loop (4 at a time)
        for chunk in 0..unrolled {
            let base = start + chunk * 4;
            unsafe {
                let idx0 = *clear_ptr.add(base) as usize;
                let idx1 = *clear_ptr.add(base + 1) as usize;
                let idx2 = *clear_ptr.add(base + 2) as usize;
                let idx3 = *clear_ptr.add(base + 3) as usize;
                
                let count0 = counts_ptr.add(idx0);
                *count0 = count0.read().checked_sub(1).expect("track_image_removal: count underflow");
                total_add += ((*count0 == 0) as u64) * *areas_ptr.add(idx0);
                
                let count1 = counts_ptr.add(idx1);
                *count1 = count1.read().checked_sub(1).expect("track_image_removal: count underflow");
                total_add += ((*count1 == 0) as u64) * *areas_ptr.add(idx1);
                
                let count2 = counts_ptr.add(idx2);
                *count2 = count2.read().checked_sub(1).expect("track_image_removal: count underflow");
                total_add += ((*count2 == 0) as u64) * *areas_ptr.add(idx2);
                
                let count3 = counts_ptr.add(idx3);
                *count3 = count3.read().checked_sub(1).expect("track_image_removal: count underflow");
                total_add += ((*count3 == 0) as u64) * *areas_ptr.add(idx3);
            }
        }
        
        // Handle remainder
        for i in (start + unrolled * 4)..(start + unrolled * 4 + remainder) {
            let idx = unsafe { *clear_ptr.add(i) } as usize;
            unsafe {
                let area = *areas_ptr.add(idx);
                let count = counts_ptr.add(idx);
                *count = count.read().checked_sub(1).expect("track_image_removal: count underflow");
                total_add += ((*count == 0) as u64) * area;
            }
        }
        
        self.current_area += total_add;
        total_add as i64
    }

    fn track_image_addition(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let start = unsafe { *self.clear_elements_offsets.get_unchecked(image_index) };
        let end = unsafe { *self.clear_elements_offsets.get_unchecked(image_index + 1) };

        let clear_ptr = self.clear_elements.as_ptr();
        let counts_ptr = self.counts.as_mut_ptr();
        let areas_ptr = self.element_areas.as_ptr();

        let len = end - start;
        let unrolled = len / 4;
        let remainder = len % 4;
        let mut total_sub = 0u64;
        
        // Unrolled loop (4 at a time)
        for chunk in 0..unrolled {
            let base = start + chunk * 4;
            unsafe {
                let idx0 = *clear_ptr.add(base) as usize;
                let idx1 = *clear_ptr.add(base + 1) as usize;
                let idx2 = *clear_ptr.add(base + 2) as usize;
                let idx3 = *clear_ptr.add(base + 3) as usize;
                
                let count0 = counts_ptr.add(idx0);
                total_sub += ((*count0 == 0) as u64) * *areas_ptr.add(idx0);
                *count0 += 1;
                
                let count1 = counts_ptr.add(idx1);
                total_sub += ((*count1 == 0) as u64) * *areas_ptr.add(idx1);
                *count1 += 1;
                
                let count2 = counts_ptr.add(idx2);
                total_sub += ((*count2 == 0) as u64) * *areas_ptr.add(idx2);
                *count2 += 1;
                
                let count3 = counts_ptr.add(idx3);
                total_sub += ((*count3 == 0) as u64) * *areas_ptr.add(idx3);
                *count3 += 1;
            }
        }
        
        // Handle remainder
        for i in (start + unrolled * 4)..(start + unrolled * 4 + remainder) {
            let idx = unsafe { *clear_ptr.add(i) } as usize;
            unsafe {
                let area = *areas_ptr.add(idx);
                let count = counts_ptr.add(idx);
                total_sub += ((*count == 0) as u64) * area;
                *count += 1;
            }
        }
        
        // Use checked subtraction to catch underflow bugs
        self.current_area = self.current_area
            .checked_sub(total_sub)
            .expect("CloudyArea underflow: total_sub > current_area");
        -(total_sub as i64)
    }

    fn value(&self) -> u64 {
        self.current_area
    }
}

// =============================================================================
// MinResolution - SIMD-optimized for two-level case
// Key insight: packed u32 (c0 | c1<<16) for single memory access per element
// Now with interval-based processing for vectorizable range loops
// =============================================================================

#[derive(Clone, Debug)]
pub struct SimdMinResolutionState {
    pub resolution_levels: Arc<Vec<u64>>,
    pub image_resolution_level: Arc<Vec<u8>>,
    pub image_elements: Arc<Vec<u32>>,
    pub image_elements_offsets: Arc<Vec<usize>>,
    // Interval-compressed image elements for vectorized range loops
    pub image_intervals: Arc<Vec<Interval>>,
    pub image_intervals_offsets: Arc<Vec<usize>>,
    // Packed counts for 2-level fast path: c0 in bits 0-15, c1 in bits 16-31
    pub packed_counts: Vec<u32>,
    // Pre-computed values for fast lookup
    pub low_val: u64,
    pub high_val: u64,
    pub diff: i64,
    pub current_sum: u64,
    // Fall-back for >2 levels  
    pub two_level: bool,
    pub element_packed_small: Vec<u64>,
    pub element_level_counts: Vec<u16>,
    pub element_level_masks: Vec<u64>,
    pub mask_words: u8,
    pub element_min_level: Vec<u8>,
    // Legacy separate arrays (unused in two-level path now)
    pub c0_counts: Vec<u16>,
    pub c1_counts: Vec<u16>,
}

impl<const D: usize> ObjectiveTracker<D> for SimdMinResolutionState {
    fn peek_removal_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;
        let start = unsafe { *self.image_elements_offsets.get_unchecked(image_index) };
        let end = unsafe { *self.image_elements_offsets.get_unchecked(image_index + 1) };
        let elements = unsafe { self.image_elements.get_unchecked(start..end) };

        if self.two_level {
            return self.peek_removal_two_level(img_level, elements);
        }

        if !self.element_packed_small.is_empty() {
            return self.peek_removal_packed_small(img_level, elements);
        }

        self.peek_removal_general(img_level, elements)
    }

    fn peek_addition_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;
        let start = unsafe { *self.image_elements_offsets.get_unchecked(image_index) };
        let end = unsafe { *self.image_elements_offsets.get_unchecked(image_index + 1) };
        let elements = unsafe { self.image_elements.get_unchecked(start..end) };

        if self.two_level {
            return self.peek_addition_two_level(img_level, elements);
        }

        if !self.element_packed_small.is_empty() {
            return self.peek_addition_packed_small(img_level, elements);
        }

        self.peek_addition_general(img_level, elements)
    }

    fn track_image_removal(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;

        if self.two_level {
            // Use interval-based path for two-level case
            let int_start = unsafe { *self.image_intervals_offsets.get_unchecked(image_index) };
            let int_end = unsafe { *self.image_intervals_offsets.get_unchecked(image_index + 1) };
            return self.track_removal_two_level_intervals(img_level, int_start, int_end);
        }

        let start = unsafe { *self.image_elements_offsets.get_unchecked(image_index) };
        let end = unsafe { *self.image_elements_offsets.get_unchecked(image_index + 1) };

        if !self.element_packed_small.is_empty() {
            return self.track_removal_packed_small_inline(img_level, start, end);
        }

        self.track_removal_general_inline(img_level, start, end)
    }

    fn track_image_addition(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;

        if self.two_level {
            // Use interval-based path for two-level case
            let int_start = unsafe { *self.image_intervals_offsets.get_unchecked(image_index) };
            let int_end = unsafe { *self.image_intervals_offsets.get_unchecked(image_index + 1) };
            return self.track_addition_two_level_intervals(img_level, int_start, int_end);
        }

        let start = unsafe { *self.image_elements_offsets.get_unchecked(image_index) };
        let end = unsafe { *self.image_elements_offsets.get_unchecked(image_index + 1) };

        if !self.element_packed_small.is_empty() {
            return self.track_addition_packed_small_inline(img_level, start, end);
        }

        self.track_addition_general_inline(img_level, start, end)
    }

    fn value(&self) -> u64 {
        self.current_sum
    }
}

impl SimdMinResolutionState {
    // =========================================================================
    // Interval-based two-level methods - key optimization!
    // Instead of iterating element-by-element, we iterate over intervals.
    // For each interval [start, start+len), we use a range loop that the
    // compiler can autovectorize.
    // =========================================================================

    #[inline]
    fn track_removal_two_level_intervals(&mut self, img_level: usize, int_start: usize, int_end: usize) -> i64 {
        let low_val = self.low_val as i64;
        let high_val = self.high_val as i64;
        let diff = self.diff;
        let mut delta = 0i64;
        let packed_ptr = self.packed_counts.as_mut_ptr();

        if img_level == 0 {
            // Removing low-resolution
            for int_idx in int_start..int_end {
                let interval = unsafe { *self.image_intervals.get_unchecked(int_idx) };
                let start = interval.start as usize;
                let end = start + interval.len as usize;
                
                // This range loop can be autovectorized by the compiler
                for idx in start..end {
                    unsafe {
                        let slot = packed_ptr.add(idx);
                        let packed = *slot;
                        let c0 = (packed & 0xFFFF) as u16;
                        let c1 = (packed >> 16) as u16;
                        *slot = (c0.saturating_sub(1) as u32) | ((c1 as u32) << 16);
                        let was_one = (c0 == 1) as i64;
                        let has_backup = (c1 > 0) as i64;
                        delta += was_one * (has_backup * (diff + low_val) - low_val);
                    }
                }
            }
        } else {
            // Removing high-resolution
            for int_idx in int_start..int_end {
                let interval = unsafe { *self.image_intervals.get_unchecked(int_idx) };
                let start = interval.start as usize;
                let end = start + interval.len as usize;
                
                for idx in start..end {
                    unsafe {
                        let slot = packed_ptr.add(idx);
                        let packed = *slot;
                        let c0 = (packed & 0xFFFF) as u16;
                        let c1 = (packed >> 16) as u16;
                        *slot = (c0 as u32) | ((c1.saturating_sub(1) as u32) << 16);
                        delta -= ((c1 == 1) as i64) * ((c0 == 0) as i64) * high_val;
                    }
                }
            }
        }
        
        self.current_sum = (self.current_sum as i64 + delta) as u64;
        delta
    }

    #[inline]
    fn track_addition_two_level_intervals(&mut self, img_level: usize, int_start: usize, int_end: usize) -> i64 {
        let low_val = self.low_val as i64;
        let high_val = self.high_val as i64;
        let diff = self.diff;
        let mut delta = 0i64;
        let packed_ptr = self.packed_counts.as_mut_ptr();

        if img_level == 0 {
            // Adding low-resolution
            for int_idx in int_start..int_end {
                let interval = unsafe { *self.image_intervals.get_unchecked(int_idx) };
                let start = interval.start as usize;
                let end = start + interval.len as usize;
                
                for idx in start..end {
                    unsafe {
                        let slot = packed_ptr.add(idx);
                        let packed = *slot;
                        let c0 = (packed & 0xFFFF) as u16;
                        let c1 = (packed >> 16) as u16;
                        let was_zero = (c0 == 0) as i64;
                        let has_backup = (c1 > 0) as i64;
                        delta += was_zero * (low_val - has_backup * (diff + low_val));
                        *slot = ((c0 + 1) as u32) | ((c1 as u32) << 16);
                    }
                }
            }
        } else {
            // Adding high-resolution
            for int_idx in int_start..int_end {
                let interval = unsafe { *self.image_intervals.get_unchecked(int_idx) };
                let start = interval.start as usize;
                let end = start + interval.len as usize;
                
                for idx in start..end {
                    unsafe {
                        let slot = packed_ptr.add(idx);
                        let packed = *slot;
                        delta += (packed == 0) as i64 * high_val;
                        *slot = packed + 0x10000;
                    }
                }
            }
        }
        
        self.current_sum = (self.current_sum as i64 + delta) as u64;
        delta
    }

    // =========================================================================
    // Two-level fast paths with packed counts (c0 in low 16 bits, c1 in high 16)
    // Single memory access per element for better cache utilization
    // =========================================================================

    #[inline]
    fn peek_removal_two_level(&self, img_level: usize, elements: &[u32]) -> i64 {
        let low_val = self.low_val as i64;
        let high_val = self.high_val as i64;
        let diff = self.diff;
        let mut delta = 0i64;

        if img_level == 0 {
            // Removing low-resolution
            for &e in elements {
                let idx = e as usize;
                unsafe {
                    let packed = *self.packed_counts.get_unchecked(idx);
                    let c0 = (packed & 0xFFFF) as u16;
                    let c1 = (packed >> 16) as u16;
                    if c0 == 1 {
                        if c1 > 0 {
                            delta += diff;  // Switch to high-res
                        } else {
                            delta -= low_val;  // Becomes uncovered
                        }
                    }
                }
            }
        } else {
            // Removing high-resolution
            for &e in elements {
                let idx = e as usize;
                unsafe {
                    let packed = *self.packed_counts.get_unchecked(idx);
                    let c0 = (packed & 0xFFFF) as u16;
                    let c1 = (packed >> 16) as u16;
                    if c1 == 1 && c0 == 0 {
                        delta -= high_val;  // Becomes uncovered
                    }
                }
            }
        }
        delta
    }

    #[inline]
    fn peek_addition_two_level(&self, img_level: usize, elements: &[u32]) -> i64 {
        let low_val = self.low_val as i64;
        let high_val = self.high_val as i64;
        let diff = self.diff;
        let mut delta = 0i64;

        if img_level == 0 {
            // Adding low-resolution
            for &e in elements {
                let idx = e as usize;
                unsafe {
                    let packed = *self.packed_counts.get_unchecked(idx);
                    let c0 = (packed & 0xFFFF) as u16;
                    let c1 = (packed >> 16) as u16;
                    if c0 == 0 {
                        if c1 > 0 {
                            delta -= diff;  // Switch from high to low
                        } else {
                            delta += low_val;  // First cover
                        }
                    }
                }
            }
        } else {
            // Adding high-resolution
            for &e in elements {
                let idx = e as usize;
                unsafe {
                    let packed = *self.packed_counts.get_unchecked(idx);
                    if packed == 0 {
                        delta += high_val;  // First cover (high)
                    }
                }
            }
        }
        delta
    }

    // Legacy CSR-based methods kept for reference and potential fallback
    #[allow(dead_code)]
    #[inline]
    fn track_removal_two_level_inline(&mut self, img_level: usize, start: usize, end: usize) -> i64 {
        // Use interval-based iteration for better vectorization
        let int_start = unsafe { *self.image_intervals_offsets.get_unchecked(start / self.image_elements.len().max(1)) };
        let int_end = unsafe { *self.image_intervals_offsets.get_unchecked((end / self.image_elements.len().max(1)).min(self.image_intervals_offsets.len() - 1)) };
        
        // Fall back to CSR if intervals not properly set up
        if int_start >= int_end || self.image_intervals.is_empty() {
            return self.track_removal_two_level_csr(img_level, start, end);
        }
        
        self.track_removal_two_level_intervals(img_level, int_start, int_end)
    }
    
    #[allow(dead_code)]
    #[inline]
    fn track_removal_two_level_csr(&mut self, img_level: usize, start: usize, end: usize) -> i64 {
        let low_val = self.low_val as i64;
        let high_val = self.high_val as i64;
        let diff = self.diff;
        let mut delta = 0i64;
        
        // Raw pointers for packed counts
        let elements_ptr = self.image_elements.as_ptr();
        let packed_ptr = self.packed_counts.as_mut_ptr();
        
        if img_level == 0 {
            // Removing low-resolution - unrolled branchless
            for i in start..end {
                let idx = unsafe { *elements_ptr.add(i) } as usize;
                unsafe {
                    let slot = packed_ptr.add(idx);
                    let packed = *slot;
                    let c0 = (packed & 0xFFFF) as u16;
                    let c1 = (packed >> 16) as u16;
                    *slot = (c0.saturating_sub(1) as u32) | ((c1 as u32) << 16);
                    let was_one = (c0 == 1) as i64;
                    let has_backup = (c1 > 0) as i64;
                    delta += was_one * (has_backup * (diff + low_val) - low_val);
                }
            }
        } else {
            // Removing high-resolution - branchless
            for i in start..end {
                let idx = unsafe { *elements_ptr.add(i) } as usize;
                unsafe {
                    let slot = packed_ptr.add(idx);
                    let packed = *slot;
                    let c0 = (packed & 0xFFFF) as u16;
                    let c1 = (packed >> 16) as u16;
                    *slot = (c0 as u32) | ((c1.saturating_sub(1) as u32) << 16);
                    delta -= ((c1 == 1) as i64) * ((c0 == 0) as i64) * high_val;
                }
            }
        }
        
        self.current_sum = (self.current_sum as i64 + delta) as u64;
        delta
    }

    #[allow(dead_code)]
    #[inline]
    fn track_addition_two_level_inline(&mut self, img_level: usize, start: usize, end: usize) -> i64 {
        let low_val = self.low_val as i64;
        let high_val = self.high_val as i64;
        let diff = self.diff;
        let mut delta = 0i64;

        let elements_ptr = self.image_elements.as_ptr();
        let packed_ptr = self.packed_counts.as_mut_ptr();

        if img_level == 0 {
            // Adding low-resolution - branchless
            for i in start..end {
                let idx = unsafe { *elements_ptr.add(i) } as usize;
                unsafe {
                    let slot = packed_ptr.add(idx);
                    let packed = *slot;
                    let c0 = (packed & 0xFFFF) as u16;
                    let c1 = (packed >> 16) as u16;
                    let was_zero = (c0 == 0) as i64;
                    let has_backup = (c1 > 0) as i64;
                    delta += was_zero * (low_val - has_backup * (diff + low_val));
                    *slot = ((c0 + 1) as u32) | ((c1 as u32) << 16);
                }
            }
        } else {
            // Adding high-resolution - branchless
            for i in start..end {
                let idx = unsafe { *elements_ptr.add(i) } as usize;
                unsafe {
                    let slot = packed_ptr.add(idx);
                    let packed = *slot;
                    delta += (packed == 0) as i64 * high_val;
                    *slot = packed + 0x10000;
                }
            }
        }
        
        self.current_sum = (self.current_sum as i64 + delta) as u64;
        delta
    }

    #[inline]
    fn track_removal_packed_small_inline(&mut self, img_level: usize, start: usize, end: usize) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let mut delta = 0i64;
        let shift = img_level * 8;
        let mask_lower = (1u64 << shift) - 1;
        let elements_ptr = self.image_elements.as_ptr();

        for i in start..end {
            let e = unsafe { *elements_ptr.add(i) };
            let idx = e as usize;
            unsafe {
                let slot = self.element_packed_small.get_unchecked_mut(idx);
                let packed = *slot;
                let count = (packed >> shift) & 0xFF;
                *slot = packed - (1u64 << shift);

                if count == 1 && (packed & mask_lower) == 0 {
                    let remaining = *slot;
                    let current_val = resolution_levels[img_level];
                    if remaining == 0 {
                        self.current_sum -= current_val;
                        delta -= current_val as i64;
                    } else {
                        let next_level = remaining.trailing_zeros() / 8;
                        let next_val = resolution_levels[next_level as usize];
                        self.current_sum = self.current_sum - current_val + next_val;
                        delta += (next_val as i64) - (current_val as i64);
                    }
                }
            }
        }
        delta
    }

    #[inline]
    fn track_addition_packed_small_inline(&mut self, img_level: usize, start: usize, end: usize) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let mut delta = 0i64;
        let shift = img_level * 8;
        let mask_lower = (1u64 << shift) - 1;
        let elements_ptr = self.image_elements.as_ptr();

        for i in start..end {
            let e = unsafe { *elements_ptr.add(i) };
            let idx = e as usize;
            unsafe {
                let slot = self.element_packed_small.get_unchecked_mut(idx);
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
        delta
    }

    #[inline]
    fn track_removal_general_inline(&mut self, img_level: usize, start: usize, end: usize) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let num_levels = resolution_levels.len();
        let mask_words = self.mask_words as usize;
        let mut delta = 0i64;
        let elements_ptr = self.image_elements.as_ptr();

        for i in start..end {
            let e = unsafe { *elements_ptr.add(i) };
            let idx = e as usize;
            let base = idx * num_levels;
            let count_slot = unsafe { self.element_level_counts.get_unchecked_mut(base + img_level) };
            if *count_slot == 0 { continue; }
            *count_slot -= 1;

            if *count_slot > 0 { continue; }

            // Update mask
            let word_idx = img_level / 64;
            let bit_idx = img_level % 64;
            unsafe {
                let m = self.element_level_masks.get_unchecked_mut(idx * mask_words + word_idx);
                *m &= !(1u64 << bit_idx);
            }

            let current_min_level = unsafe { *self.element_min_level.get_unchecked(idx) };
            if (img_level as u8) > current_min_level { continue; }

            let next_level = if mask_words == 1 {
                let mask = unsafe { *self.element_level_masks.get_unchecked(idx) };
                if mask == 0 { u8::MAX } else { mask.trailing_zeros() as u8 }
            } else {
                self.find_next_level_from_word(idx, word_idx, mask_words)
            };

            let current_val = resolution_levels[current_min_level as usize];
            let next_val = if next_level == u8::MAX { 0 } else { resolution_levels[next_level as usize] };

            unsafe { *self.element_min_level.get_unchecked_mut(idx) = next_level; }
            self.current_sum = self.current_sum - current_val + next_val;
            delta += (next_val as i64) - (current_val as i64);
        }
        delta
    }

    #[inline]
    fn track_addition_general_inline(&mut self, img_level: usize, start: usize, end: usize) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let num_levels = resolution_levels.len();
        let mask_words = self.mask_words as usize;
        let img_val = resolution_levels[img_level];
        let mut delta = 0i64;
        let elements_ptr = self.image_elements.as_ptr();

        for i in start..end {
            let e = unsafe { *elements_ptr.add(i) };
            let idx = e as usize;
            let base = idx * num_levels;
            unsafe {
                let slot = self.element_level_counts.get_unchecked_mut(base + img_level);
                let was_zero = *slot == 0;
                *slot += 1;
                if was_zero {
                    let word_idx = img_level / 64;
                    let bit_idx = img_level % 64;
                    let m = self.element_level_masks.get_unchecked_mut(idx * mask_words + word_idx);
                    *m |= 1u64 << bit_idx;
                }
            }

            let current_min_level = unsafe { *self.element_min_level.get_unchecked(idx) };
            if current_min_level == u8::MAX {
                unsafe { *self.element_min_level.get_unchecked_mut(idx) = img_level as u8; }
                self.current_sum += img_val;
                delta += img_val as i64;
            } else if img_level < current_min_level as usize {
                let current_val = resolution_levels[current_min_level as usize];
                unsafe { *self.element_min_level.get_unchecked_mut(idx) = img_level as u8; }
                self.current_sum = self.current_sum - current_val + img_val;
                delta += (img_val as i64) - (current_val as i64);
            }
        }
        delta
    }

    fn peek_removal_packed_small(&self, img_level: usize, elements: &[u32]) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let mut delta = 0i64;
        let shift = img_level * 8;
        let mask_lower = (1u64 << shift) - 1;
        let mask_higher = !((1u64 << (shift + 8)) - 1);

        for &e in elements {
            let idx = e as usize;
            unsafe {
                let packed = *self.element_packed_small.get_unchecked(idx);
                let count = (packed >> shift) & 0xFF;
                if count == 1 && (packed & mask_lower) == 0 {
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
        delta
    }

    fn peek_addition_packed_small(&self, img_level: usize, elements: &[u32]) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let mut delta = 0i64;
        let shift = img_level * 8;
        let mask_lower = (1u64 << shift) - 1;

        for &e in elements {
            let idx = e as usize;
            unsafe {
                let packed = *self.element_packed_small.get_unchecked(idx);
                if (packed & mask_lower) == 0 {
                    let count = (packed >> shift) & 0xFF;
                    if count == 0 {
                        let current_val = resolution_levels[img_level];
                        if packed == 0 {
                            delta += current_val as i64;
                        } else {
                            let old_min_level = packed.trailing_zeros() / 8;
                            let old_val = resolution_levels[old_min_level as usize];
                            delta += (current_val as i64) - (old_val as i64);
                        }
                    }
                }
            }
        }
        delta
    }

    fn peek_removal_general(&self, img_level: usize, elements: &[u32]) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let num_levels = resolution_levels.len();
        let mask_words = self.mask_words as usize;
        let mut delta = 0i64;

        for &e in elements {
            let idx = e as usize;
            let current_min_level = unsafe { *self.element_min_level.get_unchecked(idx) };
            if current_min_level == u8::MAX || img_level != current_min_level as usize {
                continue;
            }

            let base = idx * num_levels;
            let count = unsafe { *self.element_level_counts.get_unchecked(base + img_level) };
            if count > 1 {
                continue;
            }

            let next_level = if mask_words == 1 {
                let mask = unsafe { *self.element_level_masks.get_unchecked(idx) };
                let new_mask = mask & !(1u64 << img_level);
                if new_mask == 0 { u8::MAX } else { new_mask.trailing_zeros() as u8 }
            } else {
                self.find_next_level_multi_word(idx, img_level, mask_words)
            };

            let current_val = resolution_levels[current_min_level as usize];
            let next_val = if next_level == u8::MAX { 0 } else { resolution_levels[next_level as usize] };
            delta += (next_val as i64) - (current_val as i64);
        }
        delta
    }

    fn peek_addition_general(&self, img_level: usize, elements: &[u32]) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let img_val = resolution_levels[img_level];
        let mut delta = 0i64;

        for &e in elements {
            let idx = e as usize;
            let current_min_level = unsafe { *self.element_min_level.get_unchecked(idx) };
            if current_min_level == u8::MAX {
                delta += img_val as i64;
            } else if img_level < current_min_level as usize {
                let current_val = resolution_levels[current_min_level as usize];
                delta -= (current_val - img_val) as i64;
            }
        }
        delta
    }

    #[allow(dead_code)]
    fn track_removal_general(&mut self, img_level: usize, elements: &[u32]) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let num_levels = resolution_levels.len();
        let mask_words = self.mask_words as usize;
        let mut delta = 0i64;

        for &e in elements {
            let idx = e as usize;
            let base = idx * num_levels;
            let count_slot = unsafe { self.element_level_counts.get_unchecked_mut(base + img_level) };
            if *count_slot == 0 { continue; }
            *count_slot -= 1;

            if *count_slot > 0 { continue; }

            // Update mask
            let word_idx = img_level / 64;
            let bit_idx = img_level % 64;
            unsafe {
                let m = self.element_level_masks.get_unchecked_mut(idx * mask_words + word_idx);
                *m &= !(1u64 << bit_idx);
            }

            let current_min_level = unsafe { *self.element_min_level.get_unchecked(idx) };
            if (img_level as u8) > current_min_level { continue; }

            let next_level = if mask_words == 1 {
                let mask = unsafe { *self.element_level_masks.get_unchecked(idx) };
                if mask == 0 { u8::MAX } else { mask.trailing_zeros() as u8 }
            } else {
                self.find_next_level_from_word(idx, word_idx, mask_words)
            };

            let current_val = resolution_levels[current_min_level as usize];
            let next_val = if next_level == u8::MAX { 0 } else { resolution_levels[next_level as usize] };

            unsafe { *self.element_min_level.get_unchecked_mut(idx) = next_level; }
            self.current_sum = self.current_sum - current_val + next_val;
            delta += (next_val as i64) - (current_val as i64);
        }
        delta
    }

    #[allow(dead_code)]
    fn track_addition_general(&mut self, img_level: usize, elements: &[u32]) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let num_levels = resolution_levels.len();
        let mask_words = self.mask_words as usize;
        let img_val = resolution_levels[img_level];
        let mut delta = 0i64;

        for &e in elements {
            let idx = e as usize;
            let base = idx * num_levels;
            unsafe {
                let slot = self.element_level_counts.get_unchecked_mut(base + img_level);
                let was_zero = *slot == 0;
                *slot += 1;
                if was_zero {
                    let word_idx = img_level / 64;
                    let bit_idx = img_level % 64;
                    let m = self.element_level_masks.get_unchecked_mut(idx * mask_words + word_idx);
                    *m |= 1u64 << bit_idx;
                }
            }

            let current_min_level = unsafe { *self.element_min_level.get_unchecked(idx) };
            if current_min_level == u8::MAX {
                unsafe { *self.element_min_level.get_unchecked_mut(idx) = img_level as u8; }
                self.current_sum += img_val;
                delta += img_val as i64;
            } else if img_level < current_min_level as usize {
                let current_val = resolution_levels[current_min_level as usize];
                unsafe { *self.element_min_level.get_unchecked_mut(idx) = img_level as u8; }
                self.current_sum = self.current_sum - current_val + img_val;
                delta += (img_val as i64) - (current_val as i64);
            }
        }
        delta
    }

    #[inline]
    fn find_next_level_multi_word(&self, idx: usize, img_level: usize, mask_words: usize) -> u8 {
        let base = idx * mask_words;
        let word_idx = img_level / 64;
        let bit_idx = img_level % 64;

        let first_word = unsafe { *self.element_level_masks.get_unchecked(base + word_idx) } & !(1u64 << bit_idx);
        if first_word != 0 {
            return (word_idx * 64 + first_word.trailing_zeros() as usize) as u8;
        }

        for w in (word_idx + 1)..mask_words {
            let word = unsafe { *self.element_level_masks.get_unchecked(base + w) };
            if word != 0 {
                return (w * 64 + word.trailing_zeros() as usize) as u8;
            }
        }
        u8::MAX
    }

    #[inline]
    fn find_next_level_from_word(&self, idx: usize, start_word: usize, mask_words: usize) -> u8 {
        let base = idx * mask_words;
        for w in start_word..mask_words {
            let word = unsafe { *self.element_level_masks.get_unchecked(base + w) };
            if word != 0 {
                return (w * 64 + word.trailing_zeros() as usize) as u8;
            }
        }
        u8::MAX
    }
}

// =============================================================================
// MaxIncidenceAngle - unchanged, simple histogram approach is already fast
// =============================================================================

#[derive(Clone, Debug)]
pub struct SimdMaxIncidenceAngleState {
    pub incidence_levels: Arc<Vec<u64>>,
    pub image_incidence_level: Arc<Vec<u8>>,
    pub level_counts: Vec<u16>,
    pub current_max_level: u8,
    pub current_max: u64,
}

impl<const D: usize> ObjectiveTracker<D> for SimdMaxIncidenceAngleState {
    fn peek_removal_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let img_level = self.image_incidence_level[image_index];
        if self.current_max_level == u8::MAX || img_level < self.current_max_level {
            return 0;
        }

        let count = unsafe { *self.level_counts.get_unchecked(self.current_max_level as usize) };
        if count > 1 {
            return 0;
        }

        // Find next max
        let mut next = i32::from(self.current_max_level) - 1;
        while next >= 0 {
            if unsafe { *self.level_counts.get_unchecked(next as usize) } != 0 {
                let next_val = unsafe { *self.incidence_levels.get_unchecked(next as usize) };
                return (next_val as i64) - (self.current_max as i64);
            }
            next -= 1;
        }
        -(self.current_max as i64)
    }

    fn peek_addition_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let img_level = self.image_incidence_level[image_index];
        if self.current_max_level == u8::MAX || img_level > self.current_max_level {
            let next_val = unsafe { *self.incidence_levels.get_unchecked(img_level as usize) };
            (next_val as i64) - (self.current_max as i64)
        } else {
            0
        }
    }

    fn track_image_removal(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let img_level = self.image_incidence_level[image_index] as usize;
        let old_max = self.current_max;

        unsafe {
            let slot = self.level_counts.get_unchecked_mut(img_level);
            if *slot != 0 { *slot -= 1; }
        }

        if self.current_max_level != u8::MAX && img_level as u8 == self.current_max_level {
            let still = unsafe { *self.level_counts.get_unchecked(img_level) } != 0;
            if !still {
                let mut next = (img_level as i32) - 1;
                while next >= 0 {
                    if unsafe { *self.level_counts.get_unchecked(next as usize) } != 0 {
                        self.current_max_level = next as u8;
                        self.current_max = unsafe { *self.incidence_levels.get_unchecked(next as usize) };
                        return (self.current_max as i64) - (old_max as i64);
                    }
                    next -= 1;
                }
                self.current_max_level = u8::MAX;
                self.current_max = 0;
            }
        }
        (self.current_max as i64) - (old_max as i64)
    }

    fn track_image_addition(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let img_level = self.image_incidence_level[image_index];
        let old_max = self.current_max;

        unsafe { *self.level_counts.get_unchecked_mut(img_level as usize) += 1; }

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

// =============================================================================
// Tracker Enum and Collection
// =============================================================================

#[derive(Clone, Debug)]
pub enum SimdTracker {
    TotalCost(SimdTotalCostState),
    CloudyArea(SimdCloudyAreaState),
    MinResolution(SimdMinResolutionState),
    MaxIncidenceAngle(SimdMaxIncidenceAngleState),
}

impl SimdTracker {
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

impl<const D: usize> ObjectiveTracker<D> for SimdTracker {
    fn peek_addition_delta(&self, image_index: usize, problem: &impl SetCoverProblem<D>, solution: &impl ImageSet<D>) -> i64 {
        match self {
            Self::TotalCost(s) => s.peek_addition_delta(image_index, problem, solution),
            Self::CloudyArea(s) => s.peek_addition_delta(image_index, problem, solution),
            Self::MinResolution(s) => s.peek_addition_delta(image_index, problem, solution),
            Self::MaxIncidenceAngle(s) => s.peek_addition_delta(image_index, problem, solution),
        }
    }

    fn peek_removal_delta(&self, image_index: usize, problem: &impl SetCoverProblem<D>, solution: &impl ImageSet<D>) -> i64 {
        match self {
            Self::TotalCost(s) => s.peek_removal_delta(image_index, problem, solution),
            Self::CloudyArea(s) => s.peek_removal_delta(image_index, problem, solution),
            Self::MinResolution(s) => s.peek_removal_delta(image_index, problem, solution),
            Self::MaxIncidenceAngle(s) => s.peek_removal_delta(image_index, problem, solution),
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

    fn track_image_removal(&mut self, image_index: usize, problem: &impl SetCoverProblem<D>) -> i64 {
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

/// SIMD-optimized array-based tracker collection.
#[derive(Clone, Debug)]
pub struct SimdTrackerArray<const D: usize> {
    trackers: [SimdTracker; D],
}

impl<const D: usize> TrackerCollection<D> for SimdTrackerArray<D> {
    type Tracker = SimdTracker;

    fn get(&self, index: usize) -> &SimdTracker {
        &self.trackers[index]
    }

    fn get_mut(&mut self, index: usize) -> &mut SimdTracker {
        &mut self.trackers[index]
    }

    fn new(problem: &impl SetCoverProblem<D>) -> Self {
        let shared = simd_shared_data(problem);

        let trackers = std::array::from_fn(|i| match problem.objective(i) {
            crate::objectives::ObjectiveState::TotalCost { .. } => {
                SimdTracker::TotalCost(SimdTotalCostState {
                    current_cost: 0,
                    image_costs: Arc::clone(&shared.image_costs),
                })
            }
            crate::objectives::ObjectiveState::CloudyArea { .. } => {
                let total_area: u64 = shared.element_areas.iter().sum();
                let mut cloudy = FixedBitSet::with_capacity(problem.num_elements());
                cloudy.set_range(.., true);

                SimdTracker::CloudyArea(SimdCloudyAreaState {
                    counts: vec![0; problem.num_elements()],
                    cloudy_elements: cloudy,
                    current_area: total_area,
                    element_areas: Arc::clone(&shared.element_areas),
                    clear_elements: Arc::clone(&shared.clear_elements),
                    clear_elements_offsets: Arc::clone(&shared.clear_elements_offsets),
                    clear_intervals: Arc::clone(&shared.clear_intervals),
                    clear_intervals_offsets: Arc::clone(&shared.clear_intervals_offsets),
                })
            }
            crate::objectives::ObjectiveState::MinResolution { .. } => {
                let num_levels = shared.resolution_levels.len();
                let two_level = num_levels == 2;
                let small_level = num_levels > 2 && num_levels <= 8;
                let low_val = shared.resolution_levels[0];
                let high_val = shared.resolution_levels[num_levels - 1];
                let mask_words: usize = num_levels.div_ceil(64);

                SimdTracker::MinResolution(SimdMinResolutionState {
                    resolution_levels: Arc::clone(&shared.resolution_levels),
                    image_resolution_level: Arc::clone(&shared.image_resolution_level),
                    image_elements: Arc::clone(&shared.image_elements),
                    image_elements_offsets: Arc::clone(&shared.image_elements_offsets),
                    image_intervals: Arc::clone(&shared.image_intervals),
                    image_intervals_offsets: Arc::clone(&shared.image_intervals_offsets),
                    // Packed counts: c0 in low 16 bits, c1 in high 16 bits
                    packed_counts: if two_level { vec![0; problem.num_elements()] } else { Vec::new() },
                    low_val,
                    high_val,
                    diff: (high_val - low_val) as i64,
                    current_sum: 0,
                    two_level,
                    element_packed_small: if small_level { vec![0; problem.num_elements()] } else { Vec::new() },
                    element_level_counts: if two_level || small_level { Vec::new() } else { vec![0; problem.num_elements() * num_levels] },
                    element_level_masks: if two_level || small_level { Vec::new() } else { vec![0; problem.num_elements() * mask_words] },
                    mask_words: if two_level || small_level { 0 } else { mask_words as u8 },
                    element_min_level: if two_level || small_level { Vec::new() } else { vec![u8::MAX; problem.num_elements()] },
                    // Legacy arrays (kept for compatibility but not used in 2-level)
                    c0_counts: Vec::new(),
                    c1_counts: Vec::new(),
                })
            }
            crate::objectives::ObjectiveState::MaxIncidenceAngle { .. } => {
                SimdTracker::MaxIncidenceAngle(SimdMaxIncidenceAngleState {
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
