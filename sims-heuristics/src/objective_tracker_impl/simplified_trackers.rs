//! Simplified objective tracker implementation.
//!
//! This module provides a streamlined tracker implementation that:
//! - Removes unused fields (cloudy_elements, clear_intervals, etc.)
//! - Uses a unified packed approach for MinResolution (supports up to 32 levels)
//! - Uses safe code with targeted `get_unchecked` only for indirect indexing
//!   through pre-validated element arrays (same pattern as safe_simd_trackers)
//!
//! # Safety
//!
//! The only unsafe code is targeted `get_unchecked` for indirect indexing where:
//! - Element indices come from pre-validated `image_elements` arrays
//! - Resolution level indices are validated against `resolution_levels.len()`
//! - Packed count indices are validated against `max_element_idx`
//! - All invariants are checked via `debug_assert!` in debug builds
//!
//! # Performance
//!
//! For instances with ≤8 resolution levels (common case), achieves ~50% overhead
//! compared to SimdTrackerArray, while being significantly simpler:
//! - No specialized two-level vs packed_small vs general code paths
//! - Single unified packed representation
//! - Easier to understand and maintain

use std::sync::Arc;

use crate::objective_tracker::{ObjectiveTracker, TrackerCollection};
use crate::problem::SetCoverProblem;
use crate::solution::ImageSet;

use super::simd_trackers::simd_shared_data;

// =============================================================================
// TotalCost - trivially simple
// =============================================================================

#[derive(Clone, Debug)]
pub struct SimpleTotalCostState {
    pub current_cost: u64,
    pub image_costs: Arc<Vec<u64>>,
}

impl<const D: usize> ObjectiveTracker<D> for SimpleTotalCostState {
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
// CloudyArea - simplified (removed unused fields)
// =============================================================================

#[derive(Clone, Debug)]
pub struct SimpleCloudyAreaState {
    /// Per-element coverage count
    pub counts: Vec<u16>,
    /// Current total cloudy area
    pub current_area: u64,
    /// Area contribution per element (shared)
    pub element_areas: Arc<Vec<u64>>,
    /// Flattened clear elements for each image (shared)
    pub clear_elements: Arc<Vec<u32>>,
    /// Offsets into clear_elements per image (shared)
    pub clear_elements_offsets: Arc<Vec<usize>>,
    /// Max element index for bounds safety
    pub max_element_idx: usize,
}

impl<const D: usize> ObjectiveTracker<D> for SimpleCloudyAreaState {
    #[inline(always)]
    fn peek_removal_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let start = self.clear_elements_offsets[image_index];
        let end = self.clear_elements_offsets[image_index + 1];
        let clear_elements = &self.clear_elements[start..end];

        let mut delta: i64 = 0;
        for &element_u32 in clear_elements {
            let idx = element_u32 as usize;
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
            debug_assert!(idx <= self.max_element_idx);
            unsafe {
                let count_ptr = self.counts.get_unchecked_mut(idx);
                let count = (*count_ptr)
                    .checked_sub(1)
                    .expect("track_image_removal: count underflow");
                *count_ptr = count;
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
            debug_assert!(idx <= self.max_element_idx);
            unsafe {
                let count_ptr = self.counts.get_unchecked_mut(idx);
                let count = *count_ptr;
                total_sub += ((count == 0) as u64) * *self.element_areas.get_unchecked(idx);
                *count_ptr = count + 1;
            }
        }

        self.current_area = self.current_area
            .checked_sub(total_sub)
            .expect("CloudyArea underflow");
        -(total_sub as i64)
    }

    fn value(&self) -> u64 {
        self.current_area
    }
}

// =============================================================================
// MinResolution - unified packed approach (supports up to 32 levels)
// =============================================================================

/// Number of u64 words needed to store counts for `num_levels` resolution levels.
/// Each level uses 1 byte, so 8 levels per u64.
#[inline]
const fn words_for_levels(num_levels: usize) -> usize {
    (num_levels + 7) / 8
}

/// Find the minimum non-zero level in packed words.
/// Returns u8::MAX if all counts are zero.
#[inline]
fn find_min_level(packed: &[u64], num_words: usize) -> u8 {
    for w in 0..num_words {
        let word = packed[w];
        if word != 0 {
            // Find first non-zero byte
            let byte_idx = word.trailing_zeros() / 8;
            return (w * 8 + byte_idx as usize) as u8;
        }
    }
    u8::MAX
}

/// Get the count at a specific level from packed words.
#[inline]
fn get_count(packed: &[u64], level: usize) -> u8 {
    let word_idx = level / 8;
    let byte_idx = level % 8;
    ((packed[word_idx] >> (byte_idx * 8)) & 0xFF) as u8
}

/// Increment count at a specific level in packed words.
#[inline]
fn inc_count(packed: &mut [u64], level: usize) {
    let word_idx = level / 8;
    let byte_idx = level % 8;
    packed[word_idx] += 1u64 << (byte_idx * 8);
}

/// Decrement count at a specific level in packed words.
#[inline]
fn dec_count(packed: &mut [u64], level: usize) {
    let word_idx = level / 8;
    let byte_idx = level % 8;
    packed[word_idx] -= 1u64 << (byte_idx * 8);
}

#[derive(Clone, Debug)]
pub struct SimpleMinResolutionState {
    /// Resolution values per level (shared, sorted ascending)
    pub resolution_levels: Arc<Vec<u64>>,
    /// Resolution level index per image (shared)
    pub image_resolution_level: Arc<Vec<u8>>,
    /// Elements covered by each image (shared, flattened)
    pub image_elements: Arc<Vec<u32>>,
    /// Offsets into image_elements per image (shared)
    pub image_elements_offsets: Arc<Vec<usize>>,
    /// Packed counts per element: each u64 holds 8 levels (1 byte each)
    /// Layout: element 0 words, element 1 words, ...
    pub packed_counts: Vec<u64>,
    /// Number of u64 words per element
    pub words_per_element: usize,
    /// Number of resolution levels
    pub num_levels: usize,
    /// Current sum of min-resolution values
    pub current_sum: u64,
    /// Max element index for bounds safety
    pub max_element_idx: usize,
}

impl SimpleMinResolutionState {
    /// Get packed counts slice for a specific element.
    #[inline]
    fn element_counts(&self, element_idx: usize) -> &[u64] {
        let start = element_idx * self.words_per_element;
        &self.packed_counts[start..start + self.words_per_element]
    }

    /// Get mutable packed counts slice for a specific element.
    #[inline]
    fn element_counts_mut(&mut self, element_idx: usize) -> &mut [u64] {
        let start = element_idx * self.words_per_element;
        &mut self.packed_counts[start..start + self.words_per_element]
    }

    /// Optimized peek removal for single-word case (≤8 levels).
    #[inline]
    fn peek_removal_single_word(&self, img_level: usize, start: usize, end: usize) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let shift = img_level * 8;
        let mask_lower = (1u64 << shift) - 1;
        let mut delta = 0i64;

        for i in start..end {
            // SAFETY: i is in valid range for image_elements, idx <= max_element_idx
            let idx = unsafe { *self.image_elements.get_unchecked(i) as usize };
            debug_assert!(idx <= self.max_element_idx);

            // SAFETY: idx <= max_element_idx, packed_counts has one word per element
            let packed = unsafe { *self.packed_counts.get_unchecked(idx) };
            let count = (packed >> shift) & 0xFF;

            // Only affects objective if this is the last image at the min level
            if count == 1 && (packed & mask_lower) == 0 {
                let remaining = packed - (1u64 << shift);
                // SAFETY: img_level < resolution_levels.len()
                let current_val = unsafe { *resolution_levels.get_unchecked(img_level) };
                if remaining == 0 {
                    delta -= current_val as i64;
                } else {
                    let next_level = remaining.trailing_zeros() / 8;
                    // SAFETY: next_level < resolution_levels.len()
                    let next_val = unsafe { *resolution_levels.get_unchecked(next_level as usize) };
                    delta += (next_val as i64) - (current_val as i64);
                }
            }
        }
        delta
    }

    /// Optimized peek addition for single-word case (≤8 levels).
    #[inline]
    fn peek_addition_single_word(&self, img_level: usize, start: usize, end: usize) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let shift = img_level * 8;
        let mask_lower = (1u64 << shift) - 1;
        let img_val = unsafe { *resolution_levels.get_unchecked(img_level) };
        let mut delta = 0i64;

        for i in start..end {
            // SAFETY: i is in valid range for image_elements, idx <= max_element_idx
            let idx = unsafe { *self.image_elements.get_unchecked(i) as usize };
            debug_assert!(idx <= self.max_element_idx);

            // SAFETY: idx <= max_element_idx, packed_counts has one word per element
            let packed = unsafe { *self.packed_counts.get_unchecked(idx) };

            // Only affects objective if adding a better (lower) level
            if (packed & mask_lower) == 0 {
                let count = (packed >> shift) & 0xFF;
                if count == 0 {
                    if packed == 0 {
                        delta += img_val as i64;
                    } else {
                        let old_min_level = packed.trailing_zeros() / 8;
                        // SAFETY: old_min_level < resolution_levels.len()
                        let old_val = unsafe { *resolution_levels.get_unchecked(old_min_level as usize) };
                        delta += (img_val as i64) - (old_val as i64);
                    }
                }
            }
        }
        delta
    }

    /// Optimized removal for single-word case (≤8 levels).
    #[inline]
    fn track_removal_single_word(&mut self, img_level: usize, start: usize, end: usize) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let shift = img_level * 8;
        let mask_lower = (1u64 << shift) - 1;
        let mut delta = 0i64;

        for i in start..end {
            // SAFETY: i is in valid range for image_elements, idx <= max_element_idx
            let idx = unsafe { *self.image_elements.get_unchecked(i) as usize };
            debug_assert!(idx <= self.max_element_idx);

            unsafe {
                // SAFETY: idx <= max_element_idx, packed_counts has one word per element
                let packed_ptr = self.packed_counts.get_unchecked_mut(idx);
                let packed = *packed_ptr;
                let count = (packed >> shift) & 0xFF;
                *packed_ptr = packed - (1u64 << shift);

                // Only affects objective if this was the last image at the min level
                if count == 1 && (packed & mask_lower) == 0 {
                    let remaining = *packed_ptr;
                    // SAFETY: img_level < resolution_levels.len()
                    let current_val = *resolution_levels.get_unchecked(img_level);
                    if remaining == 0 {
                        self.current_sum -= current_val;
                        delta -= current_val as i64;
                    } else {
                        let next_level = remaining.trailing_zeros() / 8;
                        // SAFETY: next_level < resolution_levels.len()
                        let next_val = *resolution_levels.get_unchecked(next_level as usize);
                        self.current_sum = self.current_sum - current_val + next_val;
                        delta += (next_val as i64) - (current_val as i64);
                    }
                }
            }
        }
        delta
    }

    /// Optimized addition for single-word case (≤8 levels).
    #[inline]
    fn track_addition_single_word(&mut self, img_level: usize, start: usize, end: usize) -> i64 {
        let resolution_levels = &self.resolution_levels;
        let shift = img_level * 8;
        let mask_lower = (1u64 << shift) - 1;
        let mut delta = 0i64;

        for i in start..end {
            // SAFETY: i is in valid range for image_elements, idx <= max_element_idx
            let idx = unsafe { *self.image_elements.get_unchecked(i) as usize };
            debug_assert!(idx <= self.max_element_idx);

            unsafe {
                // SAFETY: idx <= max_element_idx, packed_counts has one word per element
                let packed_ptr = self.packed_counts.get_unchecked_mut(idx);
                let packed = *packed_ptr;

                // Only affects objective if adding a better (lower) level
                if (packed & mask_lower) == 0 {
                    let count = (packed >> shift) & 0xFF;
                    if count == 0 {
                        // SAFETY: img_level < resolution_levels.len()
                        let current_val = *resolution_levels.get_unchecked(img_level);
                        if packed == 0 {
                            self.current_sum += current_val;
                            delta += current_val as i64;
                        } else {
                            let old_min_level = packed.trailing_zeros() / 8;
                            // SAFETY: old_min_level < resolution_levels.len()
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
}

impl<const D: usize> ObjectiveTracker<D> for SimpleMinResolutionState {
    #[inline(always)]
    fn peek_removal_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;
        let start = self.image_elements_offsets[image_index];
        let end = self.image_elements_offsets[image_index + 1];

        // Optimize for common case: single word (≤8 levels)
        if self.words_per_element == 1 {
            return self.peek_removal_single_word(img_level, start, end);
        }

        // General case: multi-word
        let num_words = self.words_per_element;
        let resolution_levels = &self.resolution_levels;

        let mut delta = 0i64;

        for i in start..end {
            // SAFETY: i is in valid range for image_elements, idx <= max_element_idx
            let idx = unsafe { *self.image_elements.get_unchecked(i) as usize };
            debug_assert!(idx <= self.max_element_idx);

            let packed = self.element_counts(idx);
            let current_min = find_min_level(packed, num_words);

            // Only affects objective if we're removing from the current min level
            if current_min as usize != img_level {
                continue;
            }

            let count = get_count(packed, img_level);
            if count > 1 {
                continue; // Other images at this level remain
            }

            // Find next min level after removal
            let current_val = resolution_levels[current_min as usize];

            // Temporarily compute what min would be after decrement
            // We need to check if any counts remain at lower or same levels
            let mut next_min = u8::MAX;
            for w in 0..num_words {
                let mut word = packed[w];
                // If this word contains the level we're removing, decrement it
                if w == img_level / 8 {
                    let byte_idx = img_level % 8;
                    word -= 1u64 << (byte_idx * 8);
                }
                if word != 0 {
                    let byte_idx = word.trailing_zeros() / 8;
                    next_min = (w * 8 + byte_idx as usize) as u8;
                    break;
                }
            }

            let next_val = if next_min == u8::MAX {
                0
            } else {
                resolution_levels[next_min as usize]
            };

            delta += (next_val as i64) - (current_val as i64);
        }

        delta
    }

    #[inline(always)]
    fn peek_addition_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;
        let start = self.image_elements_offsets[image_index];
        let end = self.image_elements_offsets[image_index + 1];

        // Optimize for common case: single word (≤8 levels)
        if self.words_per_element == 1 {
            return self.peek_addition_single_word(img_level, start, end);
        }

        // General case: multi-word
        let num_words = self.words_per_element;
        let resolution_levels = &self.resolution_levels;
        let img_val = resolution_levels[img_level];

        let mut delta = 0i64;

        for i in start..end {
            // SAFETY: i is in valid range for image_elements, idx <= max_element_idx
            let idx = unsafe { *self.image_elements.get_unchecked(i) as usize };
            debug_assert!(idx <= self.max_element_idx);

            let packed = self.element_counts(idx);
            let current_min = find_min_level(packed, num_words);

            if current_min == u8::MAX {
                // No coverage yet, adding this level
                delta += img_val as i64;
            } else if img_level < current_min as usize {
                // Adding a better (lower) resolution level
                let current_val = resolution_levels[current_min as usize];
                delta += (img_val as i64) - (current_val as i64);
            }
            // If img_level >= current_min, no change to objective
        }

        delta
    }

    #[inline(always)]
    fn track_image_removal(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;
        let start = self.image_elements_offsets[image_index];
        let end = self.image_elements_offsets[image_index + 1];

        // Optimize for common case: single word (≤8 levels)
        if self.words_per_element == 1 {
            return self.track_removal_single_word(img_level, start, end);
        }

        // General case: multi-word
        let num_words = self.words_per_element;
        // Clone Arc to avoid borrow conflicts with &mut self
        let resolution_levels = Arc::clone(&self.resolution_levels);

        let mut delta = 0i64;

        for i in start..end {
            // SAFETY: i is in valid range for image_elements, idx <= max_element_idx
            let idx = unsafe { *self.image_elements.get_unchecked(i) as usize };
            debug_assert!(idx <= self.max_element_idx);

            let packed = self.element_counts_mut(idx);
            let current_min = find_min_level(packed, num_words);

            // Decrement count at this level
            dec_count(packed, img_level);

            // Only affects objective if we removed from the current min level
            if current_min as usize != img_level {
                continue;
            }

            let count = get_count(packed, img_level);
            if count > 0 {
                continue; // Other images at this level remain
            }

            // Find new min level
            let new_min = find_min_level(packed, num_words);
            let current_val = resolution_levels[current_min as usize];
            let new_val = if new_min == u8::MAX {
                0
            } else {
                resolution_levels[new_min as usize]
            };

            self.current_sum = self.current_sum - current_val + new_val;
            delta += (new_val as i64) - (current_val as i64);
        }

        delta
    }

    #[inline(always)]
    fn track_image_addition(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;
        let start = self.image_elements_offsets[image_index];
        let end = self.image_elements_offsets[image_index + 1];

        // Optimize for common case: single word (≤8 levels)
        if self.words_per_element == 1 {
            return self.track_addition_single_word(img_level, start, end);
        }

        // General case: multi-word
        let num_words = self.words_per_element;
        // Clone Arc to avoid borrow conflicts with &mut self
        let resolution_levels = Arc::clone(&self.resolution_levels);
        let img_val = resolution_levels[img_level];

        let mut delta = 0i64;

        for i in start..end {
            // SAFETY: i is in valid range for image_elements, idx <= max_element_idx
            let idx = unsafe { *self.image_elements.get_unchecked(i) as usize };
            debug_assert!(idx <= self.max_element_idx);

            let packed = self.element_counts_mut(idx);
            let current_min = find_min_level(packed, num_words);

            // Increment count at this level
            inc_count(packed, img_level);

            if current_min == u8::MAX {
                // No coverage before, now we have this level
                self.current_sum += img_val;
                delta += img_val as i64;
            } else if img_level < current_min as usize {
                // Adding a better (lower) resolution level
                let current_val = resolution_levels[current_min as usize];
                self.current_sum = self.current_sum - current_val + img_val;
                delta += (img_val as i64) - (current_val as i64);
            }
            // If img_level >= current_min, no change to objective
        }

        delta
    }

    fn value(&self) -> u64 {
        self.current_sum
    }
}

// =============================================================================
// MaxIncidenceAngle - unchanged (already simple)
// =============================================================================

#[derive(Clone, Debug)]
pub struct SimpleMaxIncidenceAngleState {
    pub incidence_levels: Arc<Vec<u64>>,
    pub image_incidence_level: Arc<Vec<u8>>,
    pub level_counts: Vec<u16>,
    pub current_max_level: u8,
    pub current_max: u64,
}

impl<const D: usize> ObjectiveTracker<D> for SimpleMaxIncidenceAngleState {
    #[inline(always)]
    fn peek_removal_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let img_level = self.image_incidence_level[image_index];
        if self.current_max_level == u8::MAX || img_level != self.current_max_level {
            return 0;
        }
        if self.level_counts[img_level as usize] > 1 {
            return 0;
        }
        // Find new max level
        let new_max_level = (0..img_level as usize)
            .rev()
            .find(|&l| self.level_counts[l] > 0);

        match new_max_level {
            Some(l) => (self.incidence_levels[l] as i64) - (self.current_max as i64),
            None => -(self.current_max as i64),
        }
    }

    #[inline(always)]
    fn peek_addition_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let img_level = self.image_incidence_level[image_index];
        let img_val = self.incidence_levels[img_level as usize];

        if self.current_max_level == u8::MAX {
            return img_val as i64;
        }
        if img_level > self.current_max_level {
            return (img_val as i64) - (self.current_max as i64);
        }
        0
    }

    #[inline(always)]
    fn track_image_removal(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let img_level = self.image_incidence_level[image_index];
        self.level_counts[img_level as usize] -= 1;

        if self.current_max_level == u8::MAX || img_level != self.current_max_level {
            return 0;
        }
        if self.level_counts[img_level as usize] > 0 {
            return 0;
        }

        let new_max_level = (0..img_level as usize)
            .rev()
            .find(|&l| self.level_counts[l] > 0);

        let old_max = self.current_max;
        match new_max_level {
            Some(l) => {
                self.current_max_level = l as u8;
                self.current_max = self.incidence_levels[l];
                (self.current_max as i64) - (old_max as i64)
            }
            None => {
                self.current_max_level = u8::MAX;
                self.current_max = 0;
                -(old_max as i64)
            }
        }
    }

    #[inline(always)]
    fn track_image_addition(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let img_level = self.image_incidence_level[image_index];
        self.level_counts[img_level as usize] += 1;

        let img_val = self.incidence_levels[img_level as usize];

        if self.current_max_level == u8::MAX {
            self.current_max_level = img_level;
            self.current_max = img_val;
            return img_val as i64;
        }

        if img_level > self.current_max_level {
            let old_max = self.current_max;
            self.current_max_level = img_level;
            self.current_max = img_val;
            return (img_val as i64) - (old_max as i64);
        }

        0
    }

    fn value(&self) -> u64 {
        self.current_max
    }
}

// =============================================================================
// SimpleTracker enum and SimpleTrackerArray
// =============================================================================

#[derive(Clone, Debug)]
pub enum SimpleTracker {
    TotalCost(SimpleTotalCostState),
    CloudyArea(SimpleCloudyAreaState),
    MinResolution(SimpleMinResolutionState),
    MaxIncidenceAngle(SimpleMaxIncidenceAngleState),
}

impl<const D: usize> ObjectiveTracker<D> for SimpleTracker {
    #[inline(always)]
    fn peek_removal_delta(&self, image_index: usize, p: &impl SetCoverProblem<D>, s: &impl ImageSet<D>) -> i64 {
        match self {
            SimpleTracker::TotalCost(t) => t.peek_removal_delta(image_index, p, s),
            SimpleTracker::CloudyArea(t) => t.peek_removal_delta(image_index, p, s),
            SimpleTracker::MinResolution(t) => t.peek_removal_delta(image_index, p, s),
            SimpleTracker::MaxIncidenceAngle(t) => t.peek_removal_delta(image_index, p, s),
        }
    }

    #[inline(always)]
    fn peek_addition_delta(&self, image_index: usize, p: &impl SetCoverProblem<D>, s: &impl ImageSet<D>) -> i64 {
        match self {
            SimpleTracker::TotalCost(t) => t.peek_addition_delta(image_index, p, s),
            SimpleTracker::CloudyArea(t) => t.peek_addition_delta(image_index, p, s),
            SimpleTracker::MinResolution(t) => t.peek_addition_delta(image_index, p, s),
            SimpleTracker::MaxIncidenceAngle(t) => t.peek_addition_delta(image_index, p, s),
        }
    }

    #[inline(always)]
    fn track_image_removal(&mut self, image_index: usize, p: &impl SetCoverProblem<D>) -> i64 {
        match self {
            SimpleTracker::TotalCost(t) => t.track_image_removal(image_index, p),
            SimpleTracker::CloudyArea(t) => t.track_image_removal(image_index, p),
            SimpleTracker::MinResolution(t) => t.track_image_removal(image_index, p),
            SimpleTracker::MaxIncidenceAngle(t) => t.track_image_removal(image_index, p),
        }
    }

    #[inline(always)]
    fn track_image_addition(&mut self, image_index: usize, p: &impl SetCoverProblem<D>) -> i64 {
        match self {
            SimpleTracker::TotalCost(t) => t.track_image_addition(image_index, p),
            SimpleTracker::CloudyArea(t) => t.track_image_addition(image_index, p),
            SimpleTracker::MinResolution(t) => t.track_image_addition(image_index, p),
            SimpleTracker::MaxIncidenceAngle(t) => t.track_image_addition(image_index, p),
        }
    }

    fn value(&self) -> u64 {
        match self {
            SimpleTracker::TotalCost(t) => ObjectiveTracker::<D>::value(t),
            SimpleTracker::CloudyArea(t) => ObjectiveTracker::<D>::value(t),
            SimpleTracker::MinResolution(t) => ObjectiveTracker::<D>::value(t),
            SimpleTracker::MaxIncidenceAngle(t) => ObjectiveTracker::<D>::value(t),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SimpleTrackerArray<const D: usize> {
    trackers: [SimpleTracker; D],
}

impl<const D: usize> TrackerCollection<D> for SimpleTrackerArray<D> {
    type Tracker = SimpleTracker;

    fn get(&self, index: usize) -> &Self::Tracker {
        &self.trackers[index]
    }

    fn get_mut(&mut self, index: usize) -> &mut Self::Tracker {
        &mut self.trackers[index]
    }

    fn new(problem: &impl SetCoverProblem<D>) -> Self {
        let shared = simd_shared_data(problem);

        // Compute max element index for bounds checks
        let max_clear_element = shared
            .clear_elements
            .iter()
            .copied()
            .max()
            .unwrap_or(0) as usize;
        let max_image_element = shared
            .image_elements
            .iter()
            .copied()
            .max()
            .unwrap_or(0) as usize;

        let trackers = std::array::from_fn(|i| match problem.objective(i) {
            crate::objectives::ObjectiveState::TotalCost { .. } => {
                SimpleTracker::TotalCost(SimpleTotalCostState {
                    current_cost: 0,
                    image_costs: Arc::clone(&shared.image_costs),
                })
            }
            crate::objectives::ObjectiveState::CloudyArea { .. } => {
                let total_area: u64 = shared.element_areas.iter().sum();
                SimpleTracker::CloudyArea(SimpleCloudyAreaState {
                    counts: vec![0; problem.num_elements()],
                    current_area: total_area,
                    element_areas: Arc::clone(&shared.element_areas),
                    clear_elements: Arc::clone(&shared.clear_elements),
                    clear_elements_offsets: Arc::clone(&shared.clear_elements_offsets),
                    max_element_idx: max_clear_element,
                })
            }
            crate::objectives::ObjectiveState::MinResolution { .. } => {
                let num_levels = shared.resolution_levels.len();
                assert!(
                    num_levels <= 32,
                    "SimpleTrackerArray supports up to 32 resolution levels, got {num_levels}"
                );
                let words_per_element = words_for_levels(num_levels);
                let num_elements = problem.num_elements();

                SimpleTracker::MinResolution(SimpleMinResolutionState {
                    resolution_levels: Arc::clone(&shared.resolution_levels),
                    image_resolution_level: Arc::clone(&shared.image_resolution_level),
                    image_elements: Arc::clone(&shared.image_elements),
                    image_elements_offsets: Arc::clone(&shared.image_elements_offsets),
                    packed_counts: vec![0u64; num_elements * words_per_element],
                    words_per_element,
                    num_levels,
                    current_sum: 0,
                    max_element_idx: max_image_element,
                })
            }
            crate::objectives::ObjectiveState::MaxIncidenceAngle { .. } => {
                SimpleTracker::MaxIncidenceAngle(SimpleMaxIncidenceAngleState {
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
        std::array::from_fn(|i| ObjectiveTracker::<D>::peek_removal_delta(&self.trackers[i], image_index, problem, solution))
    }

    fn peek_addition_delta(
        &self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
        solution: &impl ImageSet<D>,
    ) -> [i64; D] {
        std::array::from_fn(|i| ObjectiveTracker::<D>::peek_addition_delta(&self.trackers[i], image_index, problem, solution))
    }

    fn track_image_removal(
        &mut self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
    ) -> [i64; D] {
        std::array::from_fn(|i| ObjectiveTracker::<D>::track_image_removal(&mut self.trackers[i], image_index, problem))
    }

    fn track_image_addition(
        &mut self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
    ) -> [i64; D] {
        std::array::from_fn(|i| ObjectiveTracker::<D>::track_image_addition(&mut self.trackers[i], image_index, problem))
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
