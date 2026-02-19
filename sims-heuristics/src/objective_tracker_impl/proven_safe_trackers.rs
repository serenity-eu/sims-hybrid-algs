//! Fully safe objective tracker implementation -- zero `unsafe` code.
//!
//! This module is annotated with `#![deny(unsafe_code)]` and achieves performance
//! competitive with the fastest tracker (`SafeTrackerArray`) by:
//!
//! - **Pre-slicing**: Slicing arrays to exactly `max_idx + 1` elements at construction
//!   so the compiler can prove all subsequent element-derived indices are in-bounds.
//! - **Branchless arithmetic**: Using `u64::from(condition) * value` patterns instead of
//!   `if condition { value } else { 0 }` to avoid branch mispredictions.
//! - **Interval-based traversal**: Using run-length-compressed element lists for
//!   contiguous memory access patterns, enabling auto-vectorization.
//! - **Packed counts in u64**: 8 resolution level counts packed per u64 word,
//!   using `trailing_zeros` for O(1) min-finding.
//!
//! # Design
//!
//! Modeled after `safe_simd_trackers.rs` (the current fastest) but replaces all
//! `get_unchecked` with proven-safe indexing. The key trick is that CSR element
//! arrays are bounded by `max_element_idx`, so we store a pre-sliced reference to
//! exactly `[0..max_element_idx+1]` of mutable state. Since all element indices
//! are `<= max_element_idx` (validated at construction), standard `[]` indexing
//! within these pre-sliced arrays lets LLVM elide the bounds check.

#![deny(unsafe_code)]

use std::sync::Arc;

use crate::objective_tracker::{ObjectiveTracker, TrackerCollection};
use crate::problem::SetCoverProblem;
use crate::solution::ImageSet;

// =============================================================================
// Self-contained shared data types (no dependency on simd_trackers)
// =============================================================================

/// Run-length compressed element range `[start, start+len)`.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(crate) struct Interval {
    pub start: u32,
    pub len: u32,
}

/// Shared immutable precomputed data extracted from the problem.
#[derive(Debug)]
struct SharedData {
    image_costs: Arc<Vec<u64>>,
    element_areas: Arc<Vec<u64>>,
    // CSR: clear elements per image (CloudyArea)
    clear_elements: Arc<Vec<u32>>,
    clear_elements_offsets: Arc<Vec<usize>>,
    // Resolution / Incidence level mappings
    resolution_levels: Arc<Vec<u64>>,
    image_resolution_level: Arc<Vec<u8>>,
    // CSR: image elements per image (MinResolution)
    image_elements: Arc<Vec<u32>>,
    image_elements_offsets: Arc<Vec<usize>>,
    incidence_levels: Arc<Vec<u64>>,
    image_incidence_level: Arc<Vec<u8>>,
    // Interval-compressed image elements
    image_intervals: Arc<Vec<Interval>>,
    image_intervals_offsets: Arc<Vec<usize>>,
}

/// Build interval-compressed representation from CSR data.
/// Returns `(intervals, offsets)` where `offsets[i]..offsets[i+1]` covers image `i`.
fn build_intervals(elements: &[u32], offsets: &[usize]) -> (Vec<Interval>, Vec<usize>) {
    let num_images = offsets.len() - 1;
    let mut intervals = Vec::with_capacity(elements.len() / 4);
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

/// Build all shared data from the problem trait -- fully self-contained.
fn build_shared_data<const D: usize>(problem: &impl SetCoverProblem<D>) -> SharedData {
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
    let (image_intervals, image_intervals_offsets) =
        build_intervals(&image_elements, &image_elements_offsets);

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
                let image_levels: Vec<u8> = resolutions
                    .iter()
                    .map(|&r| levels.binary_search(&r).unwrap() as u8)
                    .collect();
                resolution_levels = Some(Arc::new(levels));
                image_resolution_level = Some(Arc::new(image_levels));
            }
            crate::objectives::ObjectiveState::MaxIncidenceAngle {
                incidence_angles, ..
            } => {
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

    SharedData {
        image_costs: image_costs.unwrap(),
        element_areas: element_areas.unwrap(),
        clear_elements: clear_elements.unwrap(),
        clear_elements_offsets: clear_elements_offsets.unwrap(),
        resolution_levels: resolution_levels.unwrap(),
        image_resolution_level: image_resolution_level.unwrap(),
        image_elements,
        image_elements_offsets,
        image_intervals: Arc::new(image_intervals),
        image_intervals_offsets: Arc::new(image_intervals_offsets),
        incidence_levels: incidence_levels.unwrap(),
        image_incidence_level: image_incidence_level.unwrap(),
    }
}

// =============================================================================
// TotalCost -- trivially safe, O(1) per image
// =============================================================================

#[derive(Clone, Debug)]
pub struct ProvenSafeTotalCostState {
    current_cost: u64,
    image_costs: Arc<Vec<u64>>,
}

impl<const D: usize> ObjectiveTracker<D> for ProvenSafeTotalCostState {
    #[inline(always)]
    fn peek_removal_delta(
        &self,
        image_index: usize,
        _p: &impl SetCoverProblem<D>,
        _s: &impl ImageSet<D>,
    ) -> i64 {
        -(self.image_costs[image_index] as i64)
    }

    #[inline(always)]
    fn peek_addition_delta(
        &self,
        image_index: usize,
        _p: &impl SetCoverProblem<D>,
        _s: &impl ImageSet<D>,
    ) -> i64 {
        self.image_costs[image_index] as i64
    }

    #[inline(always)]
    fn track_image_removal(
        &mut self,
        image_index: usize,
        _p: &impl SetCoverProblem<D>,
    ) -> i64 {
        let cost = self.image_costs[image_index];
        self.current_cost -= cost;
        -(cost as i64)
    }

    #[inline(always)]
    fn track_image_addition(
        &mut self,
        image_index: usize,
        _p: &impl SetCoverProblem<D>,
    ) -> i64 {
        let cost = self.image_costs[image_index];
        self.current_cost += cost;
        cost as i64
    }

    #[inline(always)]
    fn value(&self) -> u64 {
        self.current_cost
    }
}

// =============================================================================
// CloudyArea -- safe version using pre-bounded slicing
// =============================================================================

#[derive(Clone, Debug)]
pub struct ProvenSafeCloudyAreaState {
    /// Per-element coverage count. Length == num_elements.
    counts: Vec<u16>,
    /// Current total cloudy area.
    current_area: u64,
    /// Area contribution per element (shared, length >= num_elements).
    element_areas: Arc<Vec<u64>>,
    /// Flattened clear elements for each image (shared). All values < counts.len().
    clear_elements: Arc<Vec<u32>>,
    /// CSR offsets into clear_elements per image (shared).
    clear_elements_offsets: Arc<Vec<usize>>,
    /// Pre-validated bound: all element indices in clear_elements are < this value.
    /// This equals min(counts.len(), element_areas.len()).
    bound: usize,
}

impl ProvenSafeCloudyAreaState {
    /// Get elements for an image from the CSR structure.
    #[inline(always)]
    fn image_clear_elements(&self, image_index: usize) -> &[u32] {
        let start = self.clear_elements_offsets[image_index];
        let end = self.clear_elements_offsets[image_index + 1];
        &self.clear_elements[start..end]
    }
}

impl<const D: usize> ObjectiveTracker<D> for ProvenSafeCloudyAreaState {
    #[inline(always)]
    fn peek_removal_delta(
        &self,
        image_index: usize,
        _p: &impl SetCoverProblem<D>,
        _s: &impl ImageSet<D>,
    ) -> i64 {
        let clear_elements = self.image_clear_elements(image_index);
        let counts = &self.counts[..self.bound];
        let areas = &self.element_areas[..self.bound];
        let mut delta: i64 = 0;
        for &element_u32 in clear_elements {
            let idx = element_u32 as usize;
            // Branchless: add area if this was the last image covering this element
            let is_last = i64::from(counts[idx] == 1);
            delta += is_last * (areas[idx] as i64);
        }
        delta
    }

    #[inline(always)]
    fn peek_addition_delta(
        &self,
        image_index: usize,
        _p: &impl SetCoverProblem<D>,
        _s: &impl ImageSet<D>,
    ) -> i64 {
        let clear_elements = self.image_clear_elements(image_index);
        let counts = &self.counts[..self.bound];
        let areas = &self.element_areas[..self.bound];
        let mut delta: i64 = 0;
        for &element_u32 in clear_elements {
            let idx = element_u32 as usize;
            // Branchless: subtract area if this is the first image covering element
            let is_first = i64::from(counts[idx] == 0);
            delta -= is_first * (areas[idx] as i64);
        }
        delta
    }

    #[inline(always)]
    fn track_image_removal(
        &mut self,
        image_index: usize,
        _p: &impl SetCoverProblem<D>,
    ) -> i64 {
        let start = self.clear_elements_offsets[image_index];
        let end = self.clear_elements_offsets[image_index + 1];
        let clear_elements = &self.clear_elements[start..end];
        let bound = self.bound;
        let counts = &mut self.counts[..bound];
        let areas = &self.element_areas[..bound];

        let mut total_add = 0u64;
        for &element_u32 in clear_elements {
            let idx = element_u32 as usize;
            let count = counts[idx]
                .checked_sub(1)
                .expect("track_image_removal: count underflow");
            counts[idx] = count;
            total_add += u64::from(count == 0) * areas[idx];
        }

        self.current_area += total_add;
        total_add as i64
    }

    #[inline(always)]
    fn track_image_addition(
        &mut self,
        image_index: usize,
        _p: &impl SetCoverProblem<D>,
    ) -> i64 {
        let start = self.clear_elements_offsets[image_index];
        let end = self.clear_elements_offsets[image_index + 1];
        let clear_elements = &self.clear_elements[start..end];
        let bound = self.bound;
        let counts = &mut self.counts[..bound];
        let areas = &self.element_areas[..bound];

        let mut total_sub = 0u64;
        for &element_u32 in clear_elements {
            let idx = element_u32 as usize;
            let count = counts[idx];
            total_sub += u64::from(count == 0) * areas[idx];
            counts[idx] = count + 1;
        }

        self.current_area = self
            .current_area
            .checked_sub(total_sub)
            .expect("CloudyArea underflow");
        -(total_sub as i64)
    }

    fn value(&self) -> u64 {
        self.current_area
    }
}

// =============================================================================
// MinResolution -- safe version with packed counts
// =============================================================================

#[derive(Clone, Debug)]
pub struct ProvenSafeMinResolutionState {
    /// Resolution values per level (shared, sorted ascending).
    resolution_levels: Arc<Vec<u64>>,
    /// Resolution level index per image (shared).
    image_resolution_level: Arc<Vec<u8>>,
    /// Elements covered by each image (shared, flattened CSR).
    image_elements: Arc<Vec<u32>>,
    /// CSR offsets into image_elements per image (shared).
    image_elements_offsets: Arc<Vec<usize>>,
    /// Interval-compressed image elements for mutation paths.
    image_intervals: Arc<Vec<Interval>>,
    /// CSR offsets into image_intervals per image (shared).
    image_intervals_offsets: Arc<Vec<usize>>,
    // -- Two-level path (exactly 2 resolution levels) --
    /// Packed counts for two-level: c0 in low 16 bits, c1 in high 16 bits.
    packed_counts_2l: Vec<u32>,
    low_val: u64,
    high_val: u64,
    diff: i64,
    two_level: bool,
    // -- Packed-small path (3..=8 resolution levels) --
    /// Each u64 stores up to 8 level counts (1 byte each).
    packed_small: Vec<u64>,
    small_level: bool,
    // -- General path (>8 resolution levels) --
    /// Per-element per-level counts. Indexed as [element * num_levels + level].
    level_counts_general: Vec<u16>,
    /// Bitmask of active levels per element. Indexed as [element * mask_words + word].
    level_masks_general: Vec<u64>,
    mask_words: u8,
    /// Cached min-level per element (u8::MAX = uncovered).
    element_min_level: Vec<u8>,
    // -- Current sum (all paths) --
    current_sum: u64,
    /// Validated upper bound: all element indices in image_elements satisfy idx < bound.
    bound: usize,
}

impl ProvenSafeMinResolutionState {
    /// Get elements for an image from the CSR structure.
    #[inline(always)]
    fn image_elements(&self, image_index: usize) -> &[u32] {
        let start = self.image_elements_offsets[image_index];
        let end = self.image_elements_offsets[image_index + 1];
        &self.image_elements[start..end]
    }

    // =========================================================================
    // Two-level specialization
    // =========================================================================

    #[inline(always)]
    fn peek_removal_two_level(&self, img_level: usize, elements: &[u32]) -> i64 {
        let packed = &self.packed_counts_2l[..self.bound];
        let low_val = self.low_val as i64;
        let diff = self.diff;
        let high_val = self.high_val as i64;
        let mut delta = 0i64;

        if img_level == 0 {
            for &e in elements {
                let idx = e as usize;
                let p = packed[idx];
                let c0 = (p & 0xFFFF) as u16;
                let c1 = (p >> 16) as u16;
                if c0 == 1 {
                    if c1 > 0 {
                        delta += diff;
                    } else {
                        delta -= low_val;
                    }
                }
            }
        } else {
            for &e in elements {
                let idx = e as usize;
                let p = packed[idx];
                let c0 = (p & 0xFFFF) as u16;
                let c1 = (p >> 16) as u16;
                if c1 == 1 && c0 == 0 {
                    delta -= high_val;
                }
            }
        }
        delta
    }

    #[inline(always)]
    fn peek_addition_two_level(&self, img_level: usize, elements: &[u32]) -> i64 {
        let packed = &self.packed_counts_2l[..self.bound];
        let low_val = self.low_val as i64;
        let high_val = self.high_val as i64;
        let diff = self.diff;
        let mut delta = 0i64;

        if img_level == 0 {
            for &e in elements {
                let idx = e as usize;
                let p = packed[idx];
                let c0 = (p & 0xFFFF) as u16;
                let c1 = (p >> 16) as u16;
                if c0 == 0 {
                    if c1 > 0 {
                        delta -= diff;
                    } else {
                        delta += low_val;
                    }
                }
            }
        } else {
            for &e in elements {
                let idx = e as usize;
                let p = packed[idx];
                if p == 0 {
                    delta += high_val;
                }
            }
        }
        delta
    }

    #[inline(always)]
    fn track_removal_two_level_intervals(
        &mut self,
        img_level: usize,
        int_start: usize,
        int_end: usize,
    ) -> i64 {
        let low_val = self.low_val as i64;
        let high_val = self.high_val as i64;
        let diff = self.diff;
        let bound = self.bound;
        let packed = &mut self.packed_counts_2l[..bound];
        let mut delta = 0i64;

        if img_level == 0 {
            for int_idx in int_start..int_end {
                let interval = self.image_intervals[int_idx];
                let start = interval.start as usize;
                let end = start + interval.len as usize;
                for idx in start..end {
                    let p = packed[idx];
                    let c0 = (p & 0xFFFF) as u16;
                    let c1 = (p >> 16) as u16;
                    packed[idx] = u32::from(c0.saturating_sub(1)) | (u32::from(c1) << 16);
                    let was_one = i64::from(c0 == 1);
                    let has_backup = i64::from(c1 > 0);
                    delta += was_one * (has_backup * (diff + low_val) - low_val);
                }
            }
        } else {
            for int_idx in int_start..int_end {
                let interval = self.image_intervals[int_idx];
                let start = interval.start as usize;
                let end = start + interval.len as usize;
                for idx in start..end {
                    let p = packed[idx];
                    let c0 = (p & 0xFFFF) as u16;
                    let c1 = (p >> 16) as u16;
                    packed[idx] = u32::from(c0) | (u32::from(c1.saturating_sub(1)) << 16);
                    delta -= i64::from(c1 == 1) * i64::from(c0 == 0) * high_val;
                }
            }
        }

        self.current_sum = (self.current_sum as i64 + delta) as u64;
        delta
    }

    #[inline(always)]
    fn track_addition_two_level_intervals(
        &mut self,
        img_level: usize,
        int_start: usize,
        int_end: usize,
    ) -> i64 {
        let low_val = self.low_val as i64;
        let high_val = self.high_val as i64;
        let diff = self.diff;
        let bound = self.bound;
        let packed = &mut self.packed_counts_2l[..bound];
        let mut delta = 0i64;

        if img_level == 0 {
            for int_idx in int_start..int_end {
                let interval = self.image_intervals[int_idx];
                let start = interval.start as usize;
                let end = start + interval.len as usize;
                for idx in start..end {
                    let p = packed[idx];
                    let c0 = (p & 0xFFFF) as u16;
                    let c1 = (p >> 16) as u16;
                    let was_zero = i64::from(c0 == 0);
                    let has_backup = i64::from(c1 > 0);
                    delta += was_zero * (low_val - has_backup * (diff + low_val));
                    packed[idx] = u32::from(c0 + 1) | (u32::from(c1) << 16);
                }
            }
        } else {
            for int_idx in int_start..int_end {
                let interval = self.image_intervals[int_idx];
                let start = interval.start as usize;
                let end = start + interval.len as usize;
                for idx in start..end {
                    let p = packed[idx];
                    delta += i64::from(p == 0) * high_val;
                    packed[idx] = p + 0x10000;
                }
            }
        }

        self.current_sum = (self.current_sum as i64 + delta) as u64;
        delta
    }

    // =========================================================================
    // Packed-small specialization (3..=8 levels, 1 u64 per element)
    // =========================================================================

    #[inline]
    fn peek_removal_packed_small(&self, img_level: usize, elements: &[u32]) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let packed = &self.packed_small[..self.bound];
        let shift = img_level * 8;
        let mask_lower = (1u64 << shift) - 1;
        let mask_higher = !((1u64 << (shift + 8)) - 1);
        let mut delta = 0i64;

        for &e in elements {
            let idx = e as usize;
            let p = packed[idx];
            let count = (p >> shift) & 0xFF;
            if count == 1 && (p & mask_lower) == 0 {
                let current_val = resolution_levels[img_level];
                let remaining = p & mask_higher;
                if remaining == 0 {
                    delta -= current_val as i64;
                } else {
                    let next_level = remaining.trailing_zeros() / 8;
                    let next_val = resolution_levels[next_level as usize];
                    delta += (next_val as i64) - (current_val as i64);
                }
            }
        }
        delta
    }

    #[inline]
    fn peek_addition_packed_small(&self, img_level: usize, elements: &[u32]) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let packed = &self.packed_small[..self.bound];
        let shift = img_level * 8;
        let mask_lower = (1u64 << shift) - 1;
        let mut delta = 0i64;

        for &e in elements {
            let idx = e as usize;
            let p = packed[idx];
            if (p & mask_lower) == 0 {
                let count = (p >> shift) & 0xFF;
                if count == 0 {
                    let current_val = resolution_levels[img_level];
                    if p == 0 {
                        delta += current_val as i64;
                    } else {
                        let old_min_level = p.trailing_zeros() / 8;
                        let old_val = resolution_levels[old_min_level as usize];
                        delta += (current_val as i64) - (old_val as i64);
                    }
                }
            }
        }
        delta
    }

    #[inline]
    fn track_removal_packed_small(
        &mut self,
        img_level: usize,
        start: usize,
        end: usize,
    ) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let bound = self.bound;
        let packed = &mut self.packed_small[..bound];
        let image_elements = &self.image_elements[start..end];
        let shift = img_level * 8;
        let mask_lower = (1u64 << shift) - 1;
        let mut delta = 0i64;

        for &e in image_elements {
            let idx = e as usize;
            let p = packed[idx];
            let count = (p >> shift) & 0xFF;
            let new_p = p - (1u64 << shift);
            packed[idx] = new_p;

            if count == 1 && (p & mask_lower) == 0 {
                let current_val = resolution_levels[img_level];
                if new_p == 0 {
                    self.current_sum -= current_val;
                    delta -= current_val as i64;
                } else {
                    let next_level = new_p.trailing_zeros() / 8;
                    let next_val = resolution_levels[next_level as usize];
                    self.current_sum = self.current_sum - current_val + next_val;
                    delta += (next_val as i64) - (current_val as i64);
                }
            }
        }
        delta
    }

    #[inline]
    fn track_addition_packed_small(
        &mut self,
        img_level: usize,
        start: usize,
        end: usize,
    ) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let bound = self.bound;
        let packed = &mut self.packed_small[..bound];
        let image_elements = &self.image_elements[start..end];
        let shift = img_level * 8;
        let mask_lower = (1u64 << shift) - 1;
        let mut delta = 0i64;

        for &e in image_elements {
            let idx = e as usize;
            let p = packed[idx];

            if (p & mask_lower) == 0 {
                let count = (p >> shift) & 0xFF;
                if count == 0 {
                    let current_val = resolution_levels[img_level];
                    if p == 0 {
                        self.current_sum += current_val;
                        delta += current_val as i64;
                    } else {
                        let old_min_level = p.trailing_zeros() / 8;
                        let old_val = resolution_levels[old_min_level as usize];
                        self.current_sum = self.current_sum - old_val + current_val;
                        delta += (current_val as i64) - (old_val as i64);
                    }
                }
            }
            packed[idx] = p + (1u64 << shift);
        }
        delta
    }

    // =========================================================================
    // General path (>8 levels)
    // =========================================================================

    #[inline]
    fn find_next_level(&self, element_idx: usize, start_word: usize) -> u8 {
        let mask_words = usize::from(self.mask_words);
        let base = element_idx * mask_words;
        for w in start_word..mask_words {
            let word = self.level_masks_general[base + w];
            if word != 0 {
                return (w * 64 + word.trailing_zeros() as usize) as u8;
            }
        }
        u8::MAX
    }

    #[inline]
    fn find_next_level_excluding(
        &self,
        element_idx: usize,
        excluded_level: usize,
    ) -> u8 {
        let mask_words = usize::from(self.mask_words);
        let base = element_idx * mask_words;
        let word_idx = excluded_level / 64;
        let bit_idx = excluded_level % 64;

        let first_word = self.level_masks_general[base + word_idx] & !(1u64 << bit_idx);
        if first_word != 0 {
            return (word_idx * 64 + first_word.trailing_zeros() as usize) as u8;
        }

        for w in (word_idx + 1)..mask_words {
            let word = self.level_masks_general[base + w];
            if word != 0 {
                return (w * 64 + word.trailing_zeros() as usize) as u8;
            }
        }
        u8::MAX
    }

    #[inline]
    fn peek_removal_general(&self, img_level: usize, elements: &[u32]) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let num_levels = resolution_levels.len();
        let min_levels = &self.element_min_level[..self.bound];
        let mut delta = 0i64;

        for &e in elements {
            let idx = e as usize;
            let current_min_level = min_levels[idx];
            if current_min_level == u8::MAX || img_level != usize::from(current_min_level) {
                continue;
            }

            let base = idx * num_levels;
            let count = self.level_counts_general[base + img_level];
            if count > 1 {
                continue;
            }

            let next_level = self.find_next_level_excluding(idx, img_level);

            let current_val = resolution_levels[usize::from(current_min_level)];
            let next_val = if next_level == u8::MAX {
                0
            } else {
                resolution_levels[usize::from(next_level)]
            };
            delta += (next_val as i64) - (current_val as i64);
        }
        delta
    }

    #[inline]
    fn peek_addition_general(&self, img_level: usize, elements: &[u32]) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let min_levels = &self.element_min_level[..self.bound];
        let img_val = resolution_levels[img_level];
        let mut delta = 0i64;

        for &e in elements {
            let idx = e as usize;
            let current_min_level = min_levels[idx];
            if current_min_level == u8::MAX {
                delta += img_val as i64;
            } else if img_level < usize::from(current_min_level) {
                let current_val = resolution_levels[usize::from(current_min_level)];
                delta -= (current_val - img_val) as i64;
            }
        }
        delta
    }

    #[inline]
    fn track_removal_general(
        &mut self,
        img_level: usize,
        start: usize,
        end: usize,
    ) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let num_levels = resolution_levels.len();
        let mask_words = usize::from(self.mask_words);
        let image_elements = &self.image_elements[start..end];
        let mut delta = 0i64;

        for &e in image_elements {
            let idx = e as usize;
            let base = idx * num_levels;
            let count = self.level_counts_general[base + img_level];
            if count == 0 {
                continue;
            }
            self.level_counts_general[base + img_level] = count - 1;

            if count > 1 {
                continue;
            }

            // Update mask -- clear this level bit
            let word_idx = img_level / 64;
            let bit_idx = img_level % 64;
            self.level_masks_general[idx * mask_words + word_idx] &= !(1u64 << bit_idx);

            let current_min_level = self.element_min_level[idx];
            if (img_level as u8) > current_min_level {
                continue;
            }

            let next_level = self.find_next_level(idx, word_idx);

            let current_val = resolution_levels[usize::from(current_min_level)];
            let next_val = if next_level == u8::MAX {
                0
            } else {
                resolution_levels[usize::from(next_level)]
            };

            self.element_min_level[idx] = next_level;
            self.current_sum = self.current_sum - current_val + next_val;
            delta += (next_val as i64) - (current_val as i64);
        }
        delta
    }

    #[inline]
    fn track_addition_general(
        &mut self,
        img_level: usize,
        start: usize,
        end: usize,
    ) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let num_levels = resolution_levels.len();
        let mask_words = usize::from(self.mask_words);
        let img_val = resolution_levels[img_level];
        let image_elements = &self.image_elements[start..end];
        let mut delta = 0i64;

        for &e in image_elements {
            let idx = e as usize;
            let base = idx * num_levels;
            let count = self.level_counts_general[base + img_level];
            self.level_counts_general[base + img_level] = count + 1;

            if count == 0 {
                let word_idx = img_level / 64;
                let bit_idx = img_level % 64;
                self.level_masks_general[idx * mask_words + word_idx] |= 1u64 << bit_idx;
            }

            let current_min_level = self.element_min_level[idx];
            if current_min_level == u8::MAX {
                self.element_min_level[idx] = img_level as u8;
                self.current_sum += img_val;
                delta += img_val as i64;
            } else if img_level < usize::from(current_min_level) {
                let current_val = resolution_levels[usize::from(current_min_level)];
                self.element_min_level[idx] = img_level as u8;
                self.current_sum = self.current_sum - current_val + img_val;
                delta += (img_val as i64) - (current_val as i64);
            }
        }
        delta
    }
}

impl<const D: usize> ObjectiveTracker<D> for ProvenSafeMinResolutionState {
    #[inline(always)]
    fn peek_removal_delta(
        &self,
        image_index: usize,
        _p: &impl SetCoverProblem<D>,
        _s: &impl ImageSet<D>,
    ) -> i64 {
        let img_level = usize::from(self.image_resolution_level[image_index]);
        let elements = self.image_elements(image_index);

        if self.two_level {
            return self.peek_removal_two_level(img_level, elements);
        }
        if self.small_level {
            return self.peek_removal_packed_small(img_level, elements);
        }
        self.peek_removal_general(img_level, elements)
    }

    #[inline(always)]
    fn peek_addition_delta(
        &self,
        image_index: usize,
        _p: &impl SetCoverProblem<D>,
        _s: &impl ImageSet<D>,
    ) -> i64 {
        let img_level = usize::from(self.image_resolution_level[image_index]);
        let elements = self.image_elements(image_index);

        if self.two_level {
            return self.peek_addition_two_level(img_level, elements);
        }
        if self.small_level {
            return self.peek_addition_packed_small(img_level, elements);
        }
        self.peek_addition_general(img_level, elements)
    }

    #[inline(always)]
    fn track_image_removal(
        &mut self,
        image_index: usize,
        _p: &impl SetCoverProblem<D>,
    ) -> i64 {
        let img_level = usize::from(self.image_resolution_level[image_index]);

        if self.two_level {
            let int_start = self.image_intervals_offsets[image_index];
            let int_end = self.image_intervals_offsets[image_index + 1];
            return self.track_removal_two_level_intervals(img_level, int_start, int_end);
        }

        let start = self.image_elements_offsets[image_index];
        let end = self.image_elements_offsets[image_index + 1];

        if self.small_level {
            return self.track_removal_packed_small(img_level, start, end);
        }
        self.track_removal_general(img_level, start, end)
    }

    #[inline(always)]
    fn track_image_addition(
        &mut self,
        image_index: usize,
        _p: &impl SetCoverProblem<D>,
    ) -> i64 {
        let img_level = usize::from(self.image_resolution_level[image_index]);

        if self.two_level {
            let int_start = self.image_intervals_offsets[image_index];
            let int_end = self.image_intervals_offsets[image_index + 1];
            return self.track_addition_two_level_intervals(img_level, int_start, int_end);
        }

        let start = self.image_elements_offsets[image_index];
        let end = self.image_elements_offsets[image_index + 1];

        if self.small_level {
            return self.track_addition_packed_small(img_level, start, end);
        }
        self.track_addition_general(img_level, start, end)
    }

    fn value(&self) -> u64 {
        self.current_sum
    }
}

// =============================================================================
// MaxIncidenceAngle -- safe version
// =============================================================================

#[derive(Clone, Debug)]
pub struct ProvenSafeMaxIncidenceAngleState {
    incidence_levels: Arc<Vec<u64>>,
    image_incidence_level: Arc<Vec<u8>>,
    level_counts: Vec<u16>,
    current_max_level: u8,
    current_max: u64,
}

impl<const D: usize> ObjectiveTracker<D> for ProvenSafeMaxIncidenceAngleState {
    fn peek_removal_delta(
        &self,
        image_index: usize,
        _p: &impl SetCoverProblem<D>,
        _s: &impl ImageSet<D>,
    ) -> i64 {
        let img_level = self.image_incidence_level[image_index];
        if self.current_max_level == u8::MAX || img_level < self.current_max_level {
            return 0;
        }

        let count = self.level_counts[usize::from(self.current_max_level)];
        if count > 1 {
            return 0;
        }

        let mut next = i32::from(self.current_max_level) - 1;
        while next >= 0 {
            if self.level_counts[next as usize] != 0 {
                let next_val = self.incidence_levels[next as usize];
                return (next_val as i64) - (self.current_max as i64);
            }
            next -= 1;
        }
        -(self.current_max as i64)
    }

    fn peek_addition_delta(
        &self,
        image_index: usize,
        _p: &impl SetCoverProblem<D>,
        _s: &impl ImageSet<D>,
    ) -> i64 {
        let img_level = self.image_incidence_level[image_index];
        if self.current_max_level == u8::MAX || img_level > self.current_max_level {
            let next_val = self.incidence_levels[usize::from(img_level)];
            (next_val as i64) - (self.current_max as i64)
        } else {
            0
        }
    }

    fn track_image_removal(
        &mut self,
        image_index: usize,
        _p: &impl SetCoverProblem<D>,
    ) -> i64 {
        let img_level = usize::from(self.image_incidence_level[image_index]);
        let old_max = self.current_max;

        let slot = &mut self.level_counts[img_level];
        if *slot != 0 {
            *slot -= 1;
        }

        if self.current_max_level != u8::MAX && img_level == usize::from(self.current_max_level) {
            if self.level_counts[img_level] == 0 {
                let mut next = (img_level as i32) - 1;
                while next >= 0 {
                    if self.level_counts[next as usize] != 0 {
                        self.current_max_level = next as u8;
                        self.current_max = self.incidence_levels[next as usize];
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

    fn track_image_addition(
        &mut self,
        image_index: usize,
        _p: &impl SetCoverProblem<D>,
    ) -> i64 {
        let img_level = self.image_incidence_level[image_index];
        let old_max = self.current_max;

        self.level_counts[usize::from(img_level)] += 1;

        if self.current_max_level == u8::MAX || img_level > self.current_max_level {
            self.current_max_level = img_level;
            self.current_max = self.incidence_levels[usize::from(img_level)];
        }
        (self.current_max as i64) - (old_max as i64)
    }

    fn value(&self) -> u64 {
        self.current_max
    }
}

// =============================================================================
// ProvenSafeTracker enum + ProvenSafeTrackerArray
// =============================================================================

#[derive(Clone, Debug)]
pub enum ProvenSafeTracker {
    TotalCost(ProvenSafeTotalCostState),
    CloudyArea(ProvenSafeCloudyAreaState),
    MinResolution(ProvenSafeMinResolutionState),
    MaxIncidenceAngle(ProvenSafeMaxIncidenceAngleState),
}

impl<const D: usize> ObjectiveTracker<D> for ProvenSafeTracker {
    #[inline(always)]
    fn peek_removal_delta(
        &self,
        image_index: usize,
        p: &impl SetCoverProblem<D>,
        s: &impl ImageSet<D>,
    ) -> i64 {
        match self {
            Self::TotalCost(t) => t.peek_removal_delta(image_index, p, s),
            Self::CloudyArea(t) => t.peek_removal_delta(image_index, p, s),
            Self::MinResolution(t) => t.peek_removal_delta(image_index, p, s),
            Self::MaxIncidenceAngle(t) => t.peek_removal_delta(image_index, p, s),
        }
    }

    #[inline(always)]
    fn peek_addition_delta(
        &self,
        image_index: usize,
        p: &impl SetCoverProblem<D>,
        s: &impl ImageSet<D>,
    ) -> i64 {
        match self {
            Self::TotalCost(t) => t.peek_addition_delta(image_index, p, s),
            Self::CloudyArea(t) => t.peek_addition_delta(image_index, p, s),
            Self::MinResolution(t) => t.peek_addition_delta(image_index, p, s),
            Self::MaxIncidenceAngle(t) => t.peek_addition_delta(image_index, p, s),
        }
    }

    #[inline(always)]
    fn track_image_removal(
        &mut self,
        image_index: usize,
        p: &impl SetCoverProblem<D>,
    ) -> i64 {
        match self {
            Self::TotalCost(t) => t.track_image_removal(image_index, p),
            Self::CloudyArea(t) => t.track_image_removal(image_index, p),
            Self::MinResolution(t) => t.track_image_removal(image_index, p),
            Self::MaxIncidenceAngle(t) => t.track_image_removal(image_index, p),
        }
    }

    #[inline(always)]
    fn track_image_addition(
        &mut self,
        image_index: usize,
        p: &impl SetCoverProblem<D>,
    ) -> i64 {
        match self {
            Self::TotalCost(t) => t.track_image_addition(image_index, p),
            Self::CloudyArea(t) => t.track_image_addition(image_index, p),
            Self::MinResolution(t) => t.track_image_addition(image_index, p),
            Self::MaxIncidenceAngle(t) => t.track_image_addition(image_index, p),
        }
    }

    #[inline(always)]
    fn value(&self) -> u64 {
        match self {
            Self::TotalCost(t) => ObjectiveTracker::<D>::value(t),
            Self::CloudyArea(t) => ObjectiveTracker::<D>::value(t),
            Self::MinResolution(t) => ObjectiveTracker::<D>::value(t),
            Self::MaxIncidenceAngle(t) => ObjectiveTracker::<D>::value(t),
        }
    }
}

/// Fully safe tracker array -- zero `unsafe` code.
#[derive(Clone, Debug)]
pub struct ProvenSafeTrackerArray<const D: usize> {
    trackers: [ProvenSafeTracker; D],
}

impl<const D: usize> TrackerCollection<D> for ProvenSafeTrackerArray<D> {
    type Tracker = ProvenSafeTracker;

    fn get(&self, index: usize) -> &ProvenSafeTracker {
        &self.trackers[index]
    }

    fn get_mut(&mut self, index: usize) -> &mut ProvenSafeTracker {
        &mut self.trackers[index]
    }

    fn new(problem: &impl SetCoverProblem<D>) -> Self {
        let shared = build_shared_data(problem);
        let num_elements = problem.num_elements();

        // Compute validated bounds for element indices.
        // +1 so that all element indices (which are <=max) are < bound.
        let clear_bound = shared
            .clear_elements
            .iter()
            .map(|&e| e as usize)
            .max()
            .map_or(num_elements, |m| m + 1)
            .min(num_elements);

        let image_bound = shared
            .image_elements
            .iter()
            .map(|&e| e as usize)
            .max()
            .map_or(num_elements, |m| m + 1)
            .min(num_elements);

        let trackers = std::array::from_fn(|i| match problem.objective(i) {
            crate::objectives::ObjectiveState::TotalCost { .. } => {
                ProvenSafeTracker::TotalCost(ProvenSafeTotalCostState {
                    current_cost: 0,
                    image_costs: Arc::clone(&shared.image_costs),
                })
            }
            crate::objectives::ObjectiveState::CloudyArea { .. } => {
                let total_area: u64 = shared.element_areas.iter().sum();

                ProvenSafeTracker::CloudyArea(ProvenSafeCloudyAreaState {
                    counts: vec![0; num_elements],
                    current_area: total_area,
                    element_areas: Arc::clone(&shared.element_areas),
                    clear_elements: Arc::clone(&shared.clear_elements),
                    clear_elements_offsets: Arc::clone(&shared.clear_elements_offsets),
                    bound: clear_bound,
                })
            }
            crate::objectives::ObjectiveState::MinResolution { .. } => {
                let num_levels = shared.resolution_levels.len();
                let two_level = num_levels == 2;
                let small_level = num_levels > 2 && num_levels <= 8;
                let mask_words: usize = num_levels.div_ceil(64);

                ProvenSafeTracker::MinResolution(ProvenSafeMinResolutionState {
                    resolution_levels: Arc::clone(&shared.resolution_levels),
                    image_resolution_level: Arc::clone(&shared.image_resolution_level),
                    image_elements: Arc::clone(&shared.image_elements),
                    image_elements_offsets: Arc::clone(&shared.image_elements_offsets),
                    image_intervals: Arc::clone(&shared.image_intervals),
                    image_intervals_offsets: Arc::clone(&shared.image_intervals_offsets),
                    packed_counts_2l: if two_level {
                        vec![0; num_elements]
                    } else {
                        Vec::new()
                    },
                    low_val: shared.resolution_levels[0],
                    high_val: shared.resolution_levels[num_levels - 1],
                    diff: (shared.resolution_levels[num_levels - 1] - shared.resolution_levels[0])
                        as i64,
                    two_level,
                    packed_small: if small_level {
                        vec![0; num_elements]
                    } else {
                        Vec::new()
                    },
                    small_level,
                    level_counts_general: if two_level || small_level {
                        Vec::new()
                    } else {
                        vec![0; num_elements * num_levels]
                    },
                    level_masks_general: if two_level || small_level {
                        Vec::new()
                    } else {
                        vec![0; num_elements * mask_words]
                    },
                    mask_words: if two_level || small_level {
                        0
                    } else {
                        mask_words as u8
                    },
                    element_min_level: if two_level || small_level {
                        Vec::new()
                    } else {
                        vec![u8::MAX; num_elements]
                    },
                    current_sum: 0,
                    bound: image_bound,
                })
            }
            crate::objectives::ObjectiveState::MaxIncidenceAngle { .. } => {
                ProvenSafeTracker::MaxIncidenceAngle(ProvenSafeMaxIncidenceAngleState {
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
        std::array::from_fn(|i| ObjectiveTracker::<D>::value(&self.trackers[i]))
    }

    fn peek_removal_delta(
        &self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
        solution: &impl ImageSet<D>,
    ) -> [i64; D] {
        std::array::from_fn(|i| self.trackers[i].peek_removal_delta(image_index, problem, solution))
    }

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

    fn track_image_removal(
        &mut self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
    ) -> [i64; D] {
        std::array::from_fn(|i| self.trackers[i].track_image_removal(image_index, problem))
    }

    fn track_image_addition(
        &mut self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
    ) -> [i64; D] {
        std::array::from_fn(|i| self.trackers[i].track_image_addition(image_index, problem))
    }

    fn values(&self) -> [u64; D] {
        std::array::from_fn(|i| ObjectiveTracker::<D>::value(&self.trackers[i]))
    }

    fn initialize_from(&mut self, solution: &impl ImageSet<D>, problem: &impl SetCoverProblem<D>) {
        *self = Self::new(problem);
        for img in solution.selected_images() {
            self.track_image_addition(img, problem);
        }
    }
}
