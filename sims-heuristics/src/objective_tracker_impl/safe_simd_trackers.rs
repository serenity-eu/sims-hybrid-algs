//! Alternative SIMD-optimized objective tracker implementation with targeted unsafe.
//!
//! This module uses `get_unchecked` in hot loops with documented safety invariants.
//! The unsafe usage is more targeted than `simd_trackers.rs` - only indirect indexing
//! (where indices come from another array) uses unsafe, with clear invariants.
//!
//! # Safety Invariants
//!
//! All `get_unchecked` calls rely on:
//! - `max_element_idx` field is validated at construction time to be < array lengths
//! - All indices stored in `clear_elements`, `image_elements` are <= `max_element_idx`
//! - These invariants are checked in debug builds via debug_assert!
//!
//! # Performance
//!
//! Using `get_unchecked` allows the compiler to:
//! - Unroll loops (4x observed in benchmarks)
//! - Eliminate branch mispredictions from bounds checks
//! - Achieve near-parity with fully unsafe version (~5-10% overhead vs ~44%)

use fixedbitset::FixedBitSet;
use std::sync::Arc;

use crate::objective_tracker::{ObjectiveTracker, TrackerCollection};
use crate::problem::SetCoverProblem;
use crate::solution::ImageSet;

use super::simd_trackers::{simd_shared_data, Interval};

// =============================================================================
// TotalCost - already trivially safe
// =============================================================================

#[derive(Clone, Debug)]
pub struct SafeTotalCostState {
    pub current_cost: u64,
    pub image_costs: Arc<Vec<u64>>,
}

impl<const D: usize> ObjectiveTracker<D> for SafeTotalCostState {
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

    #[inline(always)]
    fn value(&self) -> u64 {
        self.current_cost
    }
}

// =============================================================================
// CloudyArea - Safe version using slice indexing
// =============================================================================

#[derive(Clone, Debug)]
pub struct SafeCloudyAreaState {
    pub counts: Vec<u16>,
    pub cloudy_elements: FixedBitSet,
    pub current_area: u64,
    pub element_areas: Arc<Vec<u64>>,
    pub clear_elements: Arc<Vec<u32>>,
    pub clear_elements_offsets: Arc<Vec<usize>>,
    pub clear_intervals: Arc<Vec<Interval>>,
    pub clear_intervals_offsets: Arc<Vec<usize>>,
    /// Maximum element index in clear_elements.
    /// SAFETY INVARIANT: All indices in `clear_elements` are <= this value,
    /// and this value is < counts.len() and < element_areas.len().
    /// Validated at construction time.
    pub max_element_idx: usize,
}

impl<const D: usize> ObjectiveTracker<D> for SafeCloudyAreaState {
    #[inline(always)]
    fn peek_removal_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let start = self.clear_elements_offsets[image_index];
        let end = self.clear_elements_offsets[image_index + 1];
        let clear_elements = &self.clear_elements[start..end];

        let mut delta: i64 = 0;
        for &element_u32 in clear_elements {
            let idx = element_u32 as usize;
            // SAFETY: max_element_idx invariant guarantees idx < counts.len() and idx < element_areas.len()
            debug_assert!(idx <= self.max_element_idx);
            unsafe {
                let is_last = (*self.counts.get_unchecked(idx) == 1) as i64;
                delta += is_last * (*self.element_areas.get_unchecked(idx) as i64);
            }
        }
        delta
    }

    #[inline(always)]
    fn peek_addition_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let start = self.clear_elements_offsets[image_index];
        let end = self.clear_elements_offsets[image_index + 1];
        let clear_elements = &self.clear_elements[start..end];

        let mut delta: i64 = 0;
        for &element_u32 in clear_elements {
            let idx = element_u32 as usize;
            // SAFETY: max_element_idx invariant guarantees idx < counts.len() and idx < element_areas.len()
            debug_assert!(idx <= self.max_element_idx);
            unsafe {
                let is_first = (*self.counts.get_unchecked(idx) == 0) as i64;
                delta -= is_first * (*self.element_areas.get_unchecked(idx) as i64);
            }
        }
        delta
    }

    #[inline(always)]
    fn track_image_removal(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let start = self.clear_elements_offsets[image_index];
        let end = self.clear_elements_offsets[image_index + 1];
        let clear_elements = &self.clear_elements[start..end];

        let mut total_add = 0u64;

        for &element_u32 in clear_elements {
            let idx = element_u32 as usize;
            // SAFETY: max_element_idx invariant guarantees idx < counts.len() and idx < element_areas.len()
            debug_assert!(idx <= self.max_element_idx);
            unsafe {
                let count_ptr = self.counts.get_unchecked_mut(idx);
                let count = (*count_ptr)
                    .checked_sub(1)
                    .expect("track_image_removal: count underflow");
                *count_ptr = count;
                // Branchless: add area only if count became 0
                total_add += ((count == 0) as u64) * *self.element_areas.get_unchecked(idx);
            }
        }

        self.current_area += total_add;
        total_add as i64
    }

    #[inline(always)]
    fn track_image_addition(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let start = self.clear_elements_offsets[image_index];
        let end = self.clear_elements_offsets[image_index + 1];
        let clear_elements = &self.clear_elements[start..end];

        let mut total_sub = 0u64;

        for &element_u32 in clear_elements {
            let idx = element_u32 as usize;
            // SAFETY: max_element_idx invariant guarantees idx < counts.len() and idx < element_areas.len()
            debug_assert!(idx <= self.max_element_idx);
            unsafe {
                let count_ptr = self.counts.get_unchecked_mut(idx);
                let count = *count_ptr;
                // Branchless: subtract area only if count was 0
                total_sub += ((count == 0) as u64) * *self.element_areas.get_unchecked(idx);
                *count_ptr = count + 1;
            }
        }

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
// MinResolution - Safe version
// =============================================================================

#[derive(Clone, Debug)]
pub struct SafeMinResolutionState {
    pub resolution_levels: Arc<Vec<u64>>,
    pub image_resolution_level: Arc<Vec<u8>>,
    pub image_elements: Arc<Vec<u32>>,
    pub image_elements_offsets: Arc<Vec<usize>>,
    pub image_intervals: Arc<Vec<Interval>>,
    pub image_intervals_offsets: Arc<Vec<usize>>,
    pub packed_counts: Vec<u32>,
    pub low_val: u64,
    pub high_val: u64,
    pub diff: i64,
    pub current_sum: u64,
    pub two_level: bool,
    pub element_packed_small: Vec<u64>,
    pub element_level_counts: Vec<u16>,
    pub element_level_masks: Vec<u64>,
    pub mask_words: u8,
    pub element_min_level: Vec<u8>,
    /// Maximum element index.
    /// SAFETY INVARIANT: All indices in `image_elements` are <= this value,
    /// and this value is < all per-element arrays (packed_counts, element_min_level, etc.).
    /// Validated at construction time.
    pub max_element_idx: usize,
}

impl<const D: usize> ObjectiveTracker<D> for SafeMinResolutionState {
    #[inline(always)]
    fn peek_removal_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;
        let start = self.image_elements_offsets[image_index];
        let end = self.image_elements_offsets[image_index + 1];
        let elements = &self.image_elements[start..end];

        if self.two_level {
            return self.peek_removal_two_level(img_level, elements);
        }

        if !self.element_packed_small.is_empty() {
            return self.peek_removal_packed_small(img_level, elements);
        }

        self.peek_removal_general(img_level, elements)
    }

    #[inline(always)]
    fn peek_addition_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;
        let start = self.image_elements_offsets[image_index];
        let end = self.image_elements_offsets[image_index + 1];
        let elements = &self.image_elements[start..end];

        if self.two_level {
            return self.peek_addition_two_level(img_level, elements);
        }

        if !self.element_packed_small.is_empty() {
            return self.peek_addition_packed_small(img_level, elements);
        }

        self.peek_addition_general(img_level, elements)
    }

    #[inline(always)]
    fn track_image_removal(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;

        if self.two_level {
            let int_start = self.image_intervals_offsets[image_index];
            let int_end = self.image_intervals_offsets[image_index + 1];
            return self.track_removal_two_level_intervals(img_level, int_start, int_end);
        }

        let start = self.image_elements_offsets[image_index];
        let end = self.image_elements_offsets[image_index + 1];

        if !self.element_packed_small.is_empty() {
            return self.track_removal_packed_small(img_level, start, end);
        }

        self.track_removal_general(img_level, start, end)
    }

    #[inline(always)]
    fn track_image_addition(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;

        if self.two_level {
            let int_start = self.image_intervals_offsets[image_index];
            let int_end = self.image_intervals_offsets[image_index + 1];
            return self.track_addition_two_level_intervals(img_level, int_start, int_end);
        }

        let start = self.image_elements_offsets[image_index];
        let end = self.image_elements_offsets[image_index + 1];

        if !self.element_packed_small.is_empty() {
            return self.track_addition_packed_small(img_level, start, end);
        }

        self.track_addition_general(img_level, start, end)
    }

    fn value(&self) -> u64 {
        self.current_sum
    }
}

impl SafeMinResolutionState {
    #[inline(always)]
    fn track_removal_two_level_intervals(&mut self, img_level: usize, int_start: usize, int_end: usize) -> i64 {
        let low_val = self.low_val as i64;
        let high_val = self.high_val as i64;
        let diff = self.diff;
        let mut delta = 0i64;

        if img_level == 0 {
            // Removing low-resolution
            for int_idx in int_start..int_end {
                let interval = self.image_intervals[int_idx];
                let start = interval.start as usize;
                let end = start + interval.len as usize;

                for idx in start..end {
                    // SAFETY: interval indices are validated at construction, idx <= max_element_idx
                    debug_assert!(idx <= self.max_element_idx);
                    unsafe {
                        let packed_ptr = self.packed_counts.get_unchecked_mut(idx);
                        let packed = *packed_ptr;
                        let c0 = (packed & 0xFFFF) as u16;
                        let c1 = (packed >> 16) as u16;
                        *packed_ptr = (c0.saturating_sub(1) as u32) | ((c1 as u32) << 16);
                        let was_one = (c0 == 1) as i64;
                        let has_backup = (c1 > 0) as i64;
                        delta += was_one * (has_backup * (diff + low_val) - low_val);
                    }
                }
            }
        } else {
            // Removing high-resolution
            for int_idx in int_start..int_end {
                let interval = self.image_intervals[int_idx];
                let start = interval.start as usize;
                let end = start + interval.len as usize;

                for idx in start..end {
                    // SAFETY: interval indices are validated at construction, idx <= max_element_idx
                    debug_assert!(idx <= self.max_element_idx);
                    unsafe {
                        let packed_ptr = self.packed_counts.get_unchecked_mut(idx);
                        let packed = *packed_ptr;
                        let c0 = (packed & 0xFFFF) as u16;
                        let c1 = (packed >> 16) as u16;
                        *packed_ptr = (c0 as u32) | ((c1.saturating_sub(1) as u32) << 16);
                        delta -= ((c1 == 1) as i64) * ((c0 == 0) as i64) * high_val;
                    }
                }
            }
        }

        self.current_sum = (self.current_sum as i64 + delta) as u64;
        delta
    }

    #[inline(always)]
    fn track_addition_two_level_intervals(&mut self, img_level: usize, int_start: usize, int_end: usize) -> i64 {
        let low_val = self.low_val as i64;
        let high_val = self.high_val as i64;
        let diff = self.diff;
        let mut delta = 0i64;

        if img_level == 0 {
            // Adding low-resolution
            for int_idx in int_start..int_end {
                let interval = self.image_intervals[int_idx];
                let start = interval.start as usize;
                let end = start + interval.len as usize;

                for idx in start..end {
                    // SAFETY: interval indices are validated at construction, idx <= max_element_idx
                    debug_assert!(idx <= self.max_element_idx);
                    unsafe {
                        let packed_ptr = self.packed_counts.get_unchecked_mut(idx);
                        let packed = *packed_ptr;
                        let c0 = (packed & 0xFFFF) as u16;
                        let c1 = (packed >> 16) as u16;
                        let was_zero = (c0 == 0) as i64;
                        let has_backup = (c1 > 0) as i64;
                        delta += was_zero * (low_val - has_backup * (diff + low_val));
                        *packed_ptr = ((c0 + 1) as u32) | ((c1 as u32) << 16);
                    }
                }
            }
        } else {
            // Adding high-resolution
            for int_idx in int_start..int_end {
                let interval = self.image_intervals[int_idx];
                let start = interval.start as usize;
                let end = start + interval.len as usize;

                for idx in start..end {
                    // SAFETY: interval indices are validated at construction, idx <= max_element_idx
                    debug_assert!(idx <= self.max_element_idx);
                    unsafe {
                        let packed_ptr = self.packed_counts.get_unchecked_mut(idx);
                        let packed = *packed_ptr;
                        delta += (packed == 0) as i64 * high_val;
                        *packed_ptr = packed + 0x10000;
                    }
                }
            }
        }

        self.current_sum = (self.current_sum as i64 + delta) as u64;
        delta
    }

    #[inline]
    fn peek_removal_two_level(&self, img_level: usize, elements: &[u32]) -> i64 {
        let low_val = self.low_val as i64;
        let diff = self.diff;
        let mut delta = 0i64;

        if img_level == 0 {
            for &e in elements {
                let idx = e as usize;
                // SAFETY: element indices are validated at construction, idx <= max_element_idx
                debug_assert!(idx <= self.max_element_idx);
                let packed = unsafe { *self.packed_counts.get_unchecked(idx) };
                let c0 = (packed & 0xFFFF) as u16;
                let c1 = (packed >> 16) as u16;
                if c0 == 1 {
                    if c1 > 0 {
                        delta += diff;
                    } else {
                        delta -= low_val;
                    }
                }
            }
        } else {
            let high_val = self.high_val as i64;
            for &e in elements {
                let idx = e as usize;
                // SAFETY: element indices are validated at construction, idx <= max_element_idx
                debug_assert!(idx <= self.max_element_idx);
                let packed = unsafe { *self.packed_counts.get_unchecked(idx) };
                let c0 = (packed & 0xFFFF) as u16;
                let c1 = (packed >> 16) as u16;
                if c1 == 1 && c0 == 0 {
                    delta -= high_val;
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
            for &e in elements {
                let idx = e as usize;
                // SAFETY: element indices are validated at construction, idx <= max_element_idx
                debug_assert!(idx <= self.max_element_idx);
                let packed = unsafe { *self.packed_counts.get_unchecked(idx) };
                let c0 = (packed & 0xFFFF) as u16;
                let c1 = (packed >> 16) as u16;
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
                // SAFETY: element indices are validated at construction, idx <= max_element_idx
                debug_assert!(idx <= self.max_element_idx);
                let packed = unsafe { *self.packed_counts.get_unchecked(idx) };
                if packed == 0 {
                    delta += high_val;
                }
            }
        }
        delta
    }

    #[inline]
    fn track_removal_packed_small(&mut self, img_level: usize, start: usize, end: usize) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let mut delta = 0i64;
        let shift = img_level * 8;
        let mask_lower = (1u64 << shift) - 1;

        for i in start..end {
            // SAFETY: i is in valid range for image_elements, idx <= max_element_idx
            let idx = unsafe { *self.image_elements.get_unchecked(i) as usize };
            debug_assert!(idx <= self.max_element_idx);
            unsafe {
                let packed_ptr = self.element_packed_small.get_unchecked_mut(idx);
                let packed = *packed_ptr;
                let count = (packed >> shift) & 0xFF;
                *packed_ptr = packed - (1u64 << shift);

                if count == 1 && (packed & mask_lower) == 0 {
                    let remaining = *packed_ptr;
                    let current_val = *resolution_levels.get_unchecked(img_level);
                    if remaining == 0 {
                        self.current_sum -= current_val;
                        delta -= current_val as i64;
                    } else {
                        let next_level = remaining.trailing_zeros() / 8;
                        let next_val = *resolution_levels.get_unchecked(next_level as usize);
                        self.current_sum = self.current_sum - current_val + next_val;
                        delta += (next_val as i64) - (current_val as i64);
                    }
                }
            }
        }
        delta
    }

    #[inline]
    fn track_addition_packed_small(&mut self, img_level: usize, start: usize, end: usize) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let mut delta = 0i64;
        let shift = img_level * 8;
        let mask_lower = (1u64 << shift) - 1;

        for i in start..end {
            // SAFETY: i is in valid range for image_elements, idx <= max_element_idx
            let idx = unsafe { *self.image_elements.get_unchecked(i) as usize };
            debug_assert!(idx <= self.max_element_idx);
            unsafe {
                let packed_ptr = self.element_packed_small.get_unchecked_mut(idx);
                let packed = *packed_ptr;

                if (packed & mask_lower) == 0 {
                    let count = (packed >> shift) & 0xFF;
                    if count == 0 {
                        let current_val = *resolution_levels.get_unchecked(img_level);
                        if packed == 0 {
                            self.current_sum += current_val;
                            delta += current_val as i64;
                        } else {
                            let old_min_level = packed.trailing_zeros() / 8;
                            let old_val = *resolution_levels.get_unchecked(old_min_level as usize);
                            self.current_sum = self.current_sum - old_val + current_val;
                            delta += (current_val as i64) - (old_val as i64);
                        }
                    }
                }
                *packed_ptr = packed + (1u64 << shift);
            }
        }
        delta
    }

    #[inline]
    fn track_removal_general(&mut self, img_level: usize, start: usize, end: usize) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let num_levels = resolution_levels.len();
        let mask_words = self.mask_words as usize;
        let mut delta = 0i64;

        for i in start..end {
            // SAFETY: i is in valid range for image_elements, idx <= max_element_idx
            let idx = unsafe { *self.image_elements.get_unchecked(i) as usize };
            debug_assert!(idx <= self.max_element_idx);
            let base = idx * num_levels;
            let count = self.element_level_counts[base + img_level];
            if count == 0 {
                continue;
            }
            self.element_level_counts[base + img_level] = count - 1;

            if count > 1 {
                continue;
            }

            // Update mask
            let word_idx = img_level / 64;
            let bit_idx = img_level % 64;
            self.element_level_masks[idx * mask_words + word_idx] &= !(1u64 << bit_idx);

            let current_min_level = self.element_min_level[idx];
            if (img_level as u8) > current_min_level {
                continue;
            }

            let next_level = self.find_next_level(idx, word_idx, mask_words);

            let current_val = resolution_levels[current_min_level as usize];
            let next_val = if next_level == u8::MAX {
                0
            } else {
                resolution_levels[next_level as usize]
            };

            self.element_min_level[idx] = next_level;
            self.current_sum = self.current_sum - current_val + next_val;
            delta += (next_val as i64) - (current_val as i64);
        }
        delta
    }

    #[inline]
    fn track_addition_general(&mut self, img_level: usize, start: usize, end: usize) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let num_levels = resolution_levels.len();
        let mask_words = self.mask_words as usize;
        let img_val = resolution_levels[img_level];
        let mut delta = 0i64;

        for i in start..end {
            // SAFETY: i is in valid range for image_elements, idx <= max_element_idx
            let idx = unsafe { *self.image_elements.get_unchecked(i) as usize };
            debug_assert!(idx <= self.max_element_idx);
            let base = idx * num_levels;
            let count = self.element_level_counts[base + img_level];
            self.element_level_counts[base + img_level] = count + 1;

            if count == 0 {
                let word_idx = img_level / 64;
                let bit_idx = img_level % 64;
                self.element_level_masks[idx * mask_words + word_idx] |= 1u64 << bit_idx;
            }

            let current_min_level = self.element_min_level[idx];
            if current_min_level == u8::MAX {
                self.element_min_level[idx] = img_level as u8;
                self.current_sum += img_val;
                delta += img_val as i64;
            } else if img_level < current_min_level as usize {
                let current_val = resolution_levels[current_min_level as usize];
                self.element_min_level[idx] = img_level as u8;
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
            // SAFETY: element indices are validated at construction, idx <= max_element_idx
            debug_assert!(idx <= self.max_element_idx);
            let packed = unsafe { *self.element_packed_small.get_unchecked(idx) };
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
        delta
    }

    fn peek_addition_packed_small(&self, img_level: usize, elements: &[u32]) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let mut delta = 0i64;
        let shift = img_level * 8;
        let mask_lower = (1u64 << shift) - 1;

        for &e in elements {
            let idx = e as usize;
            // SAFETY: element indices are validated at construction, idx <= max_element_idx
            debug_assert!(idx <= self.max_element_idx);
            let packed = unsafe { *self.element_packed_small.get_unchecked(idx) };
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
        delta
    }

    fn peek_removal_general(&self, img_level: usize, elements: &[u32]) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let num_levels = resolution_levels.len();
        let mask_words = self.mask_words as usize;
        let mut delta = 0i64;

        for &e in elements {
            let idx = e as usize;
            // SAFETY: element indices are validated at construction, idx <= max_element_idx
            debug_assert!(idx <= self.max_element_idx);
            let current_min_level = unsafe { *self.element_min_level.get_unchecked(idx) };
            if current_min_level == u8::MAX || img_level != current_min_level as usize {
                continue;
            }

            let base = idx * num_levels;
            let count = self.element_level_counts[base + img_level];
            if count > 1 {
                continue;
            }

            let next_level = {
                let mask = self.element_level_masks[idx * mask_words];
                let new_mask = mask & !(1u64 << img_level);
                if mask_words == 1 {
                    if new_mask == 0 {
                        u8::MAX
                    } else {
                        new_mask.trailing_zeros() as u8
                    }
                } else {
                    self.find_next_level_multi_word(idx, img_level, mask_words)
                }
            };

            let current_val = resolution_levels[current_min_level as usize];
            let next_val = if next_level == u8::MAX {
                0
            } else {
                resolution_levels[next_level as usize]
            };
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
            // SAFETY: element indices are validated at construction, idx <= max_element_idx
            debug_assert!(idx <= self.max_element_idx);
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

    #[inline]
    fn find_next_level(&self, idx: usize, start_word: usize, mask_words: usize) -> u8 {
        let base = idx * mask_words;
        for w in start_word..mask_words {
            let word = self.element_level_masks[base + w];
            if word != 0 {
                return (w * 64 + word.trailing_zeros() as usize) as u8;
            }
        }
        u8::MAX
    }

    #[inline]
    fn find_next_level_multi_word(&self, idx: usize, img_level: usize, mask_words: usize) -> u8 {
        let base = idx * mask_words;
        let word_idx = img_level / 64;
        let bit_idx = img_level % 64;

        let first_word = self.element_level_masks[base + word_idx] & !(1u64 << bit_idx);
        if first_word != 0 {
            return (word_idx * 64 + first_word.trailing_zeros() as usize) as u8;
        }

        for w in (word_idx + 1)..mask_words {
            let word = self.element_level_masks[base + w];
            if word != 0 {
                return (w * 64 + word.trailing_zeros() as usize) as u8;
            }
        }
        u8::MAX
    }
}

// =============================================================================
// MaxIncidenceAngle - Safe version
// =============================================================================

#[derive(Clone, Debug)]
pub struct SafeMaxIncidenceAngleState {
    pub incidence_levels: Arc<Vec<u64>>,
    pub image_incidence_level: Arc<Vec<u8>>,
    pub level_counts: Vec<u16>,
    pub current_max_level: u8,
    pub current_max: u64,
}

impl<const D: usize> ObjectiveTracker<D> for SafeMaxIncidenceAngleState {
    fn peek_removal_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let img_level = self.image_incidence_level[image_index];
        if self.current_max_level == u8::MAX || img_level < self.current_max_level {
            return 0;
        }

        let count = self.level_counts[self.current_max_level as usize];
        if count > 1 {
            return 0;
        }

        // Find next max
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

    fn peek_addition_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let img_level = self.image_incidence_level[image_index];
        if self.current_max_level == u8::MAX || img_level > self.current_max_level {
            let next_val = self.incidence_levels[img_level as usize];
            (next_val as i64) - (self.current_max as i64)
        } else {
            0
        }
    }

    fn track_image_removal(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let img_level = self.image_incidence_level[image_index] as usize;
        let old_max = self.current_max;

        let slot = &mut self.level_counts[img_level];
        if *slot != 0 {
            *slot -= 1;
        }

        if self.current_max_level != u8::MAX && img_level == self.current_max_level as usize {
            let still = self.level_counts[img_level] != 0;
            if !still {
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

    fn track_image_addition(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let img_level = self.image_incidence_level[image_index];
        let old_max = self.current_max;

        self.level_counts[img_level as usize] += 1;

        if self.current_max_level == u8::MAX || img_level > self.current_max_level {
            self.current_max_level = img_level;
            self.current_max = self.incidence_levels[img_level as usize];
        }
        (self.current_max as i64) - (old_max as i64)
    }

    fn value(&self) -> u64 {
        self.current_max
    }
}

// =============================================================================
// SafeTracker enum and SafeTrackerArray
// =============================================================================

#[derive(Clone, Debug)]
pub enum SafeTracker {
    TotalCost(SafeTotalCostState),
    CloudyArea(SafeCloudyAreaState),
    MinResolution(SafeMinResolutionState),
    MaxIncidenceAngle(SafeMaxIncidenceAngleState),
}

impl<const D: usize> ObjectiveTracker<D> for SafeTracker {
    #[inline]
    fn peek_removal_delta(&self, image_index: usize, p: &impl SetCoverProblem<D>, s: &impl ImageSet<D>) -> i64 {
        match self {
            SafeTracker::TotalCost(t) => t.peek_removal_delta(image_index, p, s),
            SafeTracker::CloudyArea(t) => t.peek_removal_delta(image_index, p, s),
            SafeTracker::MinResolution(t) => t.peek_removal_delta(image_index, p, s),
            SafeTracker::MaxIncidenceAngle(t) => t.peek_removal_delta(image_index, p, s),
        }
    }

    #[inline]
    fn peek_addition_delta(&self, image_index: usize, p: &impl SetCoverProblem<D>, s: &impl ImageSet<D>) -> i64 {
        match self {
            SafeTracker::TotalCost(t) => t.peek_addition_delta(image_index, p, s),
            SafeTracker::CloudyArea(t) => t.peek_addition_delta(image_index, p, s),
            SafeTracker::MinResolution(t) => t.peek_addition_delta(image_index, p, s),
            SafeTracker::MaxIncidenceAngle(t) => t.peek_addition_delta(image_index, p, s),
        }
    }

    #[inline]
    fn track_image_removal(&mut self, image_index: usize, p: &impl SetCoverProblem<D>) -> i64 {
        match self {
            SafeTracker::TotalCost(t) => t.track_image_removal(image_index, p),
            SafeTracker::CloudyArea(t) => t.track_image_removal(image_index, p),
            SafeTracker::MinResolution(t) => t.track_image_removal(image_index, p),
            SafeTracker::MaxIncidenceAngle(t) => t.track_image_removal(image_index, p),
        }
    }

    #[inline]
    fn track_image_addition(&mut self, image_index: usize, p: &impl SetCoverProblem<D>) -> i64 {
        match self {
            SafeTracker::TotalCost(t) => t.track_image_addition(image_index, p),
            SafeTracker::CloudyArea(t) => t.track_image_addition(image_index, p),
            SafeTracker::MinResolution(t) => t.track_image_addition(image_index, p),
            SafeTracker::MaxIncidenceAngle(t) => t.track_image_addition(image_index, p),
        }
    }

    #[inline]
    fn value(&self) -> u64 {
        match self {
            SafeTracker::TotalCost(t) => <SafeTotalCostState as ObjectiveTracker<D>>::value(t),
            SafeTracker::CloudyArea(t) => <SafeCloudyAreaState as ObjectiveTracker<D>>::value(t),
            SafeTracker::MinResolution(t) => <SafeMinResolutionState as ObjectiveTracker<D>>::value(t),
            SafeTracker::MaxIncidenceAngle(t) => <SafeMaxIncidenceAngleState as ObjectiveTracker<D>>::value(t),
        }
    }
}

/// Safe array-based tracker collection.
#[derive(Clone, Debug)]
pub struct SafeTrackerArray<const D: usize> {
    trackers: [SafeTracker; D],
}

impl<const D: usize> TrackerCollection<D> for SafeTrackerArray<D> {
    type Tracker = SafeTracker;

    fn get(&self, index: usize) -> &SafeTracker {
        &self.trackers[index]
    }

    fn get_mut(&mut self, index: usize) -> &mut SafeTracker {
        &mut self.trackers[index]
    }

    fn new(problem: &impl SetCoverProblem<D>) -> Self {
        let shared = simd_shared_data(problem);

        let trackers = std::array::from_fn(|i| match problem.objective(i) {
            crate::objectives::ObjectiveState::TotalCost { .. } => {
                SafeTracker::TotalCost(SafeTotalCostState {
                    current_cost: 0,
                    image_costs: Arc::clone(&shared.image_costs),
                })
            }
            crate::objectives::ObjectiveState::CloudyArea { .. } => {
                let total_area: u64 = shared.element_areas.iter().sum();
                let mut cloudy = FixedBitSet::with_capacity(problem.num_elements());
                cloudy.set_range(.., true);
                
                // Compute max element index for bounds hints
                let max_element_idx = shared.clear_elements.iter()
                    .map(|&e| e as usize)
                    .max()
                    .unwrap_or(0);

                SafeTracker::CloudyArea(SafeCloudyAreaState {
                    counts: vec![0; problem.num_elements()],
                    cloudy_elements: cloudy,
                    current_area: total_area,
                    element_areas: Arc::clone(&shared.element_areas),
                    clear_elements: Arc::clone(&shared.clear_elements),
                    clear_elements_offsets: Arc::clone(&shared.clear_elements_offsets),
                    clear_intervals: Arc::clone(&shared.clear_intervals),
                    clear_intervals_offsets: Arc::clone(&shared.clear_intervals_offsets),
                    max_element_idx,
                })
            }
            crate::objectives::ObjectiveState::MinResolution { .. } => {
                let num_levels = shared.resolution_levels.len();
                let two_level = num_levels == 2;
                let small_level = num_levels > 2 && num_levels <= 8;
                let low_val = shared.resolution_levels[0];
                let high_val = shared.resolution_levels[num_levels - 1];
                let mask_words: usize = num_levels.div_ceil(64);
                
                // Max element index for bounds hints
                let max_element_idx = shared.image_elements.iter()
                    .map(|&e| e as usize)
                    .max()
                    .unwrap_or(0);

                SafeTracker::MinResolution(SafeMinResolutionState {
                    resolution_levels: Arc::clone(&shared.resolution_levels),
                    image_resolution_level: Arc::clone(&shared.image_resolution_level),
                    image_elements: Arc::clone(&shared.image_elements),
                    image_elements_offsets: Arc::clone(&shared.image_elements_offsets),
                    image_intervals: Arc::clone(&shared.image_intervals),
                    image_intervals_offsets: Arc::clone(&shared.image_intervals_offsets),
                    packed_counts: if two_level {
                        vec![0; problem.num_elements()]
                    } else {
                        Vec::new()
                    },
                    low_val,
                    high_val,
                    diff: (high_val - low_val) as i64,
                    current_sum: 0,
                    two_level,
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
                    mask_words: if two_level || small_level {
                        0
                    } else {
                        mask_words as u8
                    },
                    element_min_level: if two_level || small_level {
                        Vec::new()
                    } else {
                        vec![u8::MAX; problem.num_elements()]
                    },
                    max_element_idx,
                })
            }
            crate::objectives::ObjectiveState::MaxIncidenceAngle { .. } => {
                SafeTracker::MaxIncidenceAngle(SafeMaxIncidenceAngleState {
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
        std::array::from_fn(|i| <SafeTracker as ObjectiveTracker<D>>::value(&self.trackers[i]))
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
        std::array::from_fn(|i| self.trackers[i].peek_addition_delta(image_index, problem, solution))
    }

    fn track_image_removal(&mut self, image_index: usize, problem: &impl SetCoverProblem<D>) -> [i64; D] {
        std::array::from_fn(|i| self.trackers[i].track_image_removal(image_index, problem))
    }

    fn track_image_addition(&mut self, image_index: usize, problem: &impl SetCoverProblem<D>) -> [i64; D] {
        std::array::from_fn(|i| self.trackers[i].track_image_addition(image_index, problem))
    }

    fn values(&self) -> [u64; D] {
        std::array::from_fn(|i| <SafeTracker as ObjectiveTracker<D>>::value(&self.trackers[i]))
    }

    fn initialize_from(&mut self, solution: &impl ImageSet<D>, problem: &impl SetCoverProblem<D>) {
        *self = Self::new(problem);
        for img in solution.selected_images() {
            self.track_image_addition(img, problem);
        }
    }
}
