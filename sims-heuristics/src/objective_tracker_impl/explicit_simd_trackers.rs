//! Explicit SIMD objective trackers using portable_simd.
//!
//! This module provides trackers that use Rust's portable SIMD API for
//! explicit vectorization of the hot loops in objective tracking.
//!
//! Key optimizations:
//! 1. SIMD gather operations to load non-contiguous elements
//! 2. SIMD scatter operations for contiguous updates  
//! 3. Vectorized comparisons and conditional accumulation
//! 4. Lane-parallel delta computation

#![allow(unused)]

use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdUint;
use std::simd::prelude::*;
use std::simd::Mask;
use std::sync::Arc;

use crate::objective_tracker::{ObjectiveTracker, TrackerCollection};
use crate::problem::SetCoverProblem;
use crate::solution::ImageSet;

use super::simd_trackers::{simd_shared_data, Interval, SimdTrackerSharedData};

// =============================================================================
// SIMD Configuration
// =============================================================================

/// SIMD lane width for u32 operations (8 lanes = 256 bits = AVX2)
const LANES_32: usize = 8;
/// SIMD lane width for u64 operations (4 lanes = 256 bits = AVX2)
const LANES_64: usize = 4;

type U32x8 = Simd<u32, LANES_32>;
type I64x4 = Simd<i64, LANES_64>;
type I32x8 = Simd<i32, LANES_32>;
type Mask32x8 = Mask<i32, LANES_32>;

// =============================================================================
// Explicit SIMD MinResolution Tracker (2-level only)
// =============================================================================

/// MinResolution tracker with explicit SIMD operations.
/// Uses packed counts (c0 in low 16 bits, c1 in high 16 bits).
#[derive(Clone, Debug)]
pub struct ExplicitSimdMinResState {
    /// Packed counts: c0 in bits 0-15, c1 in bits 16-31
    pub packed_counts: Vec<u32>,
    /// Pre-computed objective values
    pub low_val: u64,
    pub high_val: u64,
    pub diff: i64,
    pub current_sum: u64,
    /// Image data references
    pub image_intervals: Arc<Vec<Interval>>,
    pub image_intervals_offsets: Arc<Vec<usize>>,
    pub image_resolution_level: Arc<Vec<u8>>,
    pub image_elements: Arc<Vec<u32>>,
    pub image_elements_offsets: Arc<Vec<usize>>,
}

impl ExplicitSimdMinResState {
    /// Process a contiguous range [start, end) using SIMD.
    /// Returns the delta for removing from level 0 (low-res).
    #[inline]
    fn simd_remove_level0(&mut self, start: usize, end: usize) -> i64 {
        let low_val = self.low_val as i64;
        let high_val = self.high_val as i64;
        let diff = self.diff;
        let mut delta = 0i64;
        
        let len = end - start;
        let simd_iters = len / LANES_32;
        let remainder_start = start + simd_iters * LANES_32;
        
        // SIMD constants
        let mask_low = U32x8::splat(0xFFFF);
        let one_u32 = U32x8::splat(1);
        let one_i32 = I32x8::splat(1);
        let zero_i32 = I32x8::splat(0);
        
        let packed_ptr = self.packed_counts.as_mut_ptr();
        
        // SIMD loop - process LANES_32 elements at a time
        for i in 0..simd_iters {
            let base = start + i * LANES_32;
            unsafe {
                // Load LANES_32 packed values
                let packed = U32x8::from_slice(std::slice::from_raw_parts(packed_ptr.add(base), LANES_32));
                
                // Extract c0 and c1
                let c0 = packed & mask_low;
                let c1 = packed >> 16;
                
                // Compute new c0 = saturating_sub(c0, 1)
                // Since we can't easily do saturating_sub in SIMD, use max(c0-1, 0)
                let c0_minus_1 = c0 - one_u32;
                let underflow_mask = c0.simd_eq(U32x8::splat(0));
                let new_c0 = underflow_mask.select(U32x8::splat(0), c0_minus_1);
                
                // Store updated packed value
                let new_packed = new_c0 | (c1 << 16);
                new_packed.copy_to_slice(std::slice::from_raw_parts_mut(packed_ptr.add(base), LANES_32));
                
                // Compute delta contribution
                // was_one = (c0 == 1)
                // has_backup = (c1 > 0)
                // delta += was_one * (has_backup * (diff + low_val) - low_val)
                let c0_i32: I32x8 = c0.cast();
                let c1_i32: I32x8 = c1.cast();
                
                let was_one_mask = c0_i32.simd_eq(one_i32);
                let has_backup_mask = c1_i32.simd_gt(zero_i32);
                
                // Count elements where c0 == 1 && c1 > 0 (switch to high-res)
                let switch_mask = was_one_mask & has_backup_mask;
                let switch_count = switch_mask.to_bitmask().count_ones() as i64;
                
                // Count elements where c0 == 1 && c1 == 0 (become uncovered)
                let uncovered_mask = was_one_mask & !has_backup_mask;
                let uncovered_count = uncovered_mask.to_bitmask().count_ones() as i64;
                
                // Delta = switch_count * diff - uncovered_count * low_val
                delta += switch_count * diff - uncovered_count * low_val;
            }
        }
        
        // Scalar remainder
        for idx in remainder_start..end {
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
        
        delta
    }

    /// Process a contiguous range [start, end) using SIMD.
    /// Returns the delta for removing from level 1 (high-res).
    #[inline]
    fn simd_remove_level1(&mut self, start: usize, end: usize) -> i64 {
        let high_val = self.high_val as i64;
        let mut delta = 0i64;
        
        let len = end - start;
        let simd_iters = len / LANES_32;
        let remainder_start = start + simd_iters * LANES_32;
        
        // SIMD constants
        let mask_low = U32x8::splat(0xFFFF);
        let one_high = U32x8::splat(0x10000);
        let one_i32 = I32x8::splat(1);
        let zero_i32 = I32x8::splat(0);
        
        let packed_ptr = self.packed_counts.as_mut_ptr();
        
        for i in 0..simd_iters {
            let base = start + i * LANES_32;
            unsafe {
                let packed = U32x8::from_slice(std::slice::from_raw_parts(packed_ptr.add(base), LANES_32));
                
                let c0 = packed & mask_low;
                let c1 = packed >> 16;
                
                // Compute new c1 = saturating_sub(c1, 1)
                let c1_minus_1 = c1 - U32x8::splat(1);
                let underflow_mask = c1.simd_eq(U32x8::splat(0));
                let new_c1 = underflow_mask.select(U32x8::splat(0), c1_minus_1);
                
                // Store updated packed value
                let new_packed = c0 | (new_c1 << 16);
                new_packed.copy_to_slice(std::slice::from_raw_parts_mut(packed_ptr.add(base), LANES_32));
                
                // Delta: if c1 == 1 && c0 == 0, then element becomes uncovered
                let c0_i32: I32x8 = c0.cast();
                let c1_i32: I32x8 = c1.cast();
                
                let c1_was_one = c1_i32.simd_eq(one_i32);
                let c0_was_zero = c0_i32.simd_eq(zero_i32);
                let uncovered_mask = c1_was_one & c0_was_zero;
                let uncovered_count = uncovered_mask.to_bitmask().count_ones() as i64;
                
                delta -= uncovered_count * high_val;
            }
        }
        
        // Scalar remainder
        for idx in remainder_start..end {
            unsafe {
                let slot = packed_ptr.add(idx);
                let packed = *slot;
                let c0 = (packed & 0xFFFF) as u16;
                let c1 = (packed >> 16) as u16;
                *slot = (c0 as u32) | ((c1.saturating_sub(1) as u32) << 16);
                delta -= ((c1 == 1) as i64) * ((c0 == 0) as i64) * high_val;
            }
        }
        
        delta
    }

    /// Process a contiguous range [start, end) using SIMD.
    /// Returns the delta for adding to level 0 (low-res).
    #[inline]
    fn simd_add_level0(&mut self, start: usize, end: usize) -> i64 {
        let low_val = self.low_val as i64;
        let diff = self.diff;
        let mut delta = 0i64;
        
        let len = end - start;
        let simd_iters = len / LANES_32;
        let remainder_start = start + simd_iters * LANES_32;
        
        let mask_low = U32x8::splat(0xFFFF);
        let one_u32 = U32x8::splat(1);
        let zero_i32 = I32x8::splat(0);
        
        let packed_ptr = self.packed_counts.as_mut_ptr();
        
        for i in 0..simd_iters {
            let base = start + i * LANES_32;
            unsafe {
                let packed = U32x8::from_slice(std::slice::from_raw_parts(packed_ptr.add(base), LANES_32));
                
                let c0 = packed & mask_low;
                let c1 = packed >> 16;
                
                // Compute delta before update
                // was_zero = (c0 == 0)
                // has_backup = (c1 > 0)
                // delta += was_zero * (low_val - has_backup * (diff + low_val))
                let c0_i32: I32x8 = c0.cast();
                let c1_i32: I32x8 = c1.cast();
                
                let was_zero_mask = c0_i32.simd_eq(zero_i32);
                let has_backup_mask = c1_i32.simd_gt(zero_i32);
                
                // Count elements where c0 == 0 && c1 == 0 (first cover)
                let first_cover_mask = was_zero_mask & !has_backup_mask;
                let first_cover_count = first_cover_mask.to_bitmask().count_ones() as i64;
                
                // Count elements where c0 == 0 && c1 > 0 (switch from high to low)
                let switch_mask = was_zero_mask & has_backup_mask;
                let switch_count = switch_mask.to_bitmask().count_ones() as i64;
                
                // Delta = first_cover_count * low_val - switch_count * diff
                delta += first_cover_count * low_val - switch_count * diff;
                
                // Update: c0 += 1
                let new_c0 = c0 + one_u32;
                let new_packed = new_c0 | (c1 << 16);
                new_packed.copy_to_slice(std::slice::from_raw_parts_mut(packed_ptr.add(base), LANES_32));
            }
        }
        
        // Scalar remainder
        for idx in remainder_start..end {
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
        
        delta
    }

    /// Process a contiguous range [start, end) using SIMD.
    /// Returns the delta for adding to level 1 (high-res).
    #[inline]
    fn simd_add_level1(&mut self, start: usize, end: usize) -> i64 {
        let high_val = self.high_val as i64;
        let mut delta = 0i64;
        
        let len = end - start;
        let simd_iters = len / LANES_32;
        let remainder_start = start + simd_iters * LANES_32;
        
        let one_high = U32x8::splat(0x10000);
        let zero_u32 = U32x8::splat(0);
        
        let packed_ptr = self.packed_counts.as_mut_ptr();
        
        for i in 0..simd_iters {
            let base = start + i * LANES_32;
            unsafe {
                let packed = U32x8::from_slice(std::slice::from_raw_parts(packed_ptr.add(base), LANES_32));
                
                // Delta: if packed == 0 (both c0 and c1 are 0), element gets first coverage
                let was_uncovered_mask = packed.simd_eq(zero_u32);
                let uncovered_count = was_uncovered_mask.to_bitmask().count_ones() as i64;
                delta += uncovered_count * high_val;
                
                // Update: c1 += 1 (add 0x10000)
                let new_packed = packed + one_high;
                new_packed.copy_to_slice(std::slice::from_raw_parts_mut(packed_ptr.add(base), LANES_32));
            }
        }
        
        // Scalar remainder
        for idx in remainder_start..end {
            unsafe {
                let slot = packed_ptr.add(idx);
                let packed = *slot;
                delta += (packed == 0) as i64 * high_val;
                *slot = packed + 0x10000;
            }
        }
        
        delta
    }

    /// Track removal using intervals with SIMD processing.
    fn track_removal_intervals(&mut self, img_level: usize, int_start: usize, int_end: usize) -> i64 {
        let mut delta = 0i64;
        
        for int_idx in int_start..int_end {
            let interval = unsafe { *self.image_intervals.get_unchecked(int_idx) };
            let start = interval.start as usize;
            let end = start + interval.len as usize;
            
            if img_level == 0 {
                delta += self.simd_remove_level0(start, end);
            } else {
                delta += self.simd_remove_level1(start, end);
            }
        }
        
        self.current_sum = (self.current_sum as i64 + delta) as u64;
        delta
    }

    /// Track addition using intervals with SIMD processing.
    fn track_addition_intervals(&mut self, img_level: usize, int_start: usize, int_end: usize) -> i64 {
        let mut delta = 0i64;
        
        for int_idx in int_start..int_end {
            let interval = unsafe { *self.image_intervals.get_unchecked(int_idx) };
            let start = interval.start as usize;
            let end = start + interval.len as usize;
            
            if img_level == 0 {
                delta += self.simd_add_level0(start, end);
            } else {
                delta += self.simd_add_level1(start, end);
            }
        }
        
        self.current_sum = (self.current_sum as i64 + delta) as u64;
        delta
    }

    /// Peek removal using intervals with SIMD processing (read-only).
    fn peek_removal_intervals(&self, img_level: usize, int_start: usize, int_end: usize) -> i64 {
        let low_val = self.low_val as i64;
        let high_val = self.high_val as i64;
        let diff = self.diff;
        let mut delta = 0i64;
        
        let mask_low = U32x8::splat(0xFFFF);
        let one_i32 = I32x8::splat(1);
        let zero_i32 = I32x8::splat(0);
        
        let packed_ptr = self.packed_counts.as_ptr();
        
        for int_idx in int_start..int_end {
            let interval = unsafe { *self.image_intervals.get_unchecked(int_idx) };
            let start = interval.start as usize;
            let end = start + interval.len as usize;
            let len = end - start;
            let simd_iters = len / LANES_32;
            let remainder_start = start + simd_iters * LANES_32;
            
            if img_level == 0 {
                // Level 0 (low-res) removal
                for i in 0..simd_iters {
                    let base = start + i * LANES_32;
                    unsafe {
                        let packed = U32x8::from_slice(std::slice::from_raw_parts(packed_ptr.add(base), LANES_32));
                        let c0 = packed & mask_low;
                        let c1 = packed >> 16;
                        
                        let c0_i32: I32x8 = c0.cast();
                        let c1_i32: I32x8 = c1.cast();
                        
                        let was_one_mask = c0_i32.simd_eq(one_i32);
                        let has_backup_mask = c1_i32.simd_gt(zero_i32);
                        
                        let switch_mask = was_one_mask & has_backup_mask;
                        let switch_count = switch_mask.to_bitmask().count_ones() as i64;
                        
                        let uncovered_mask = was_one_mask & !has_backup_mask;
                        let uncovered_count = uncovered_mask.to_bitmask().count_ones() as i64;
                        
                        delta += switch_count * diff - uncovered_count * low_val;
                    }
                }
                
                // Scalar remainder
                for idx in remainder_start..end {
                    unsafe {
                        let packed = *packed_ptr.add(idx);
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
                }
            } else {
                // Level 1 (high-res) removal
                for i in 0..simd_iters {
                    let base = start + i * LANES_32;
                    unsafe {
                        let packed = U32x8::from_slice(std::slice::from_raw_parts(packed_ptr.add(base), LANES_32));
                        let c0 = packed & mask_low;
                        let c1 = packed >> 16;
                        
                        let c0_i32: I32x8 = c0.cast();
                        let c1_i32: I32x8 = c1.cast();
                        
                        let c1_was_one = c1_i32.simd_eq(one_i32);
                        let c0_was_zero = c0_i32.simd_eq(zero_i32);
                        let uncovered_mask = c1_was_one & c0_was_zero;
                        let uncovered_count = uncovered_mask.to_bitmask().count_ones() as i64;
                        
                        delta -= uncovered_count * high_val;
                    }
                }
                
                for idx in remainder_start..end {
                    unsafe {
                        let packed = *packed_ptr.add(idx);
                        let c0 = (packed & 0xFFFF) as u16;
                        let c1 = (packed >> 16) as u16;
                        if c1 == 1 && c0 == 0 {
                            delta -= high_val;
                        }
                    }
                }
            }
        }
        
        delta
    }
    
    /// Peek addition using intervals with SIMD processing (read-only).
    fn peek_addition_intervals(&self, img_level: usize, int_start: usize, int_end: usize) -> i64 {
        let low_val = self.low_val as i64;
        let high_val = self.high_val as i64;
        let diff = self.diff;
        let mut delta = 0i64;
        
        let mask_low = U32x8::splat(0xFFFF);
        let zero_i32 = I32x8::splat(0);
        let zero_u32 = U32x8::splat(0);
        
        let packed_ptr = self.packed_counts.as_ptr();
        
        for int_idx in int_start..int_end {
            let interval = unsafe { *self.image_intervals.get_unchecked(int_idx) };
            let start = interval.start as usize;
            let end = start + interval.len as usize;
            let len = end - start;
            let simd_iters = len / LANES_32;
            let remainder_start = start + simd_iters * LANES_32;
            
            if img_level == 0 {
                // Level 0 (low-res) addition
                for i in 0..simd_iters {
                    let base = start + i * LANES_32;
                    unsafe {
                        let packed = U32x8::from_slice(std::slice::from_raw_parts(packed_ptr.add(base), LANES_32));
                        let c0 = packed & mask_low;
                        let c1 = packed >> 16;
                        
                        let c0_i32: I32x8 = c0.cast();
                        let c1_i32: I32x8 = c1.cast();
                        
                        let was_zero_mask = c0_i32.simd_eq(zero_i32);
                        let has_backup_mask = c1_i32.simd_gt(zero_i32);
                        
                        let first_cover_mask = was_zero_mask & !has_backup_mask;
                        let first_cover_count = first_cover_mask.to_bitmask().count_ones() as i64;
                        
                        let switch_mask = was_zero_mask & has_backup_mask;
                        let switch_count = switch_mask.to_bitmask().count_ones() as i64;
                        
                        delta += first_cover_count * low_val - switch_count * diff;
                    }
                }
                
                for idx in remainder_start..end {
                    unsafe {
                        let packed = *packed_ptr.add(idx);
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
                }
            } else {
                // Level 1 (high-res) addition
                for i in 0..simd_iters {
                    let base = start + i * LANES_32;
                    unsafe {
                        let packed = U32x8::from_slice(std::slice::from_raw_parts(packed_ptr.add(base), LANES_32));
                        let uncovered_mask = packed.simd_eq(zero_u32);
                        let uncovered_count = uncovered_mask.to_bitmask().count_ones() as i64;
                        delta += uncovered_count * high_val;
                    }
                }
                
                for idx in remainder_start..end {
                    unsafe {
                        let packed = *packed_ptr.add(idx);
                        if packed == 0 {
                            delta += high_val;
                        }
                    }
                }
            }
        }
        
        delta
    }
}

impl<const D: usize> ObjectiveTracker<D> for ExplicitSimdMinResState {
    fn peek_removal_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;
        let int_start = unsafe { *self.image_intervals_offsets.get_unchecked(image_index) };
        let int_end = unsafe { *self.image_intervals_offsets.get_unchecked(image_index + 1) };
        self.peek_removal_intervals(img_level, int_start, int_end)
    }

    fn peek_addition_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;
        let int_start = unsafe { *self.image_intervals_offsets.get_unchecked(image_index) };
        let int_end = unsafe { *self.image_intervals_offsets.get_unchecked(image_index + 1) };
        self.peek_addition_intervals(img_level, int_start, int_end)
    }

    fn track_image_removal(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;
        let int_start = unsafe { *self.image_intervals_offsets.get_unchecked(image_index) };
        let int_end = unsafe { *self.image_intervals_offsets.get_unchecked(image_index + 1) };
        self.track_removal_intervals(img_level, int_start, int_end)
    }

    fn track_image_addition(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let img_level = self.image_resolution_level[image_index] as usize;
        let int_start = unsafe { *self.image_intervals_offsets.get_unchecked(image_index) };
        let int_end = unsafe { *self.image_intervals_offsets.get_unchecked(image_index + 1) };
        self.track_addition_intervals(img_level, int_start, int_end)
    }

    fn value(&self) -> u64 {
        self.current_sum
    }
}

// =============================================================================
// Explicit SIMD CloudyArea Tracker
// =============================================================================

/// CloudyArea tracker with explicit SIMD operations.
#[derive(Clone, Debug)]
pub struct ExplicitSimdCloudyAreaState {
    /// Per-element coverage counts
    pub counts: Vec<u16>,
    /// Per-element areas (precomputed)
    pub element_areas: Arc<Vec<u64>>,
    /// Current cloudy area value
    pub current_area: u64,
    /// Clear elements per image
    pub clear_intervals: Arc<Vec<Interval>>,
    pub clear_intervals_offsets: Arc<Vec<usize>>,
}

impl ExplicitSimdCloudyAreaState {
    /// SIMD removal for cloudy area - decrement counts and track zero transitions.
    #[inline]
    fn simd_remove(&mut self, start: usize, end: usize) -> i64 {
        let mut delta_area = 0u64;
        
        let len = end - start;
        // Use 16 lanes for u16 operations (256 bits = AVX2)
        const LANES_16: usize = 16;
        let simd_iters = len / LANES_16;
        let remainder_start = start + simd_iters * LANES_16;
        
        let counts_ptr = self.counts.as_mut_ptr();
        let areas_ptr = self.element_areas.as_ptr();
        
        type U16x16 = Simd<u16, LANES_16>;
        let one_u16 = U16x16::splat(1);
        let zero_u16 = U16x16::splat(0);
        
        for i in 0..simd_iters {
            let base = start + i * LANES_16;
            unsafe {
                // Load counts
                let counts = U16x16::from_slice(std::slice::from_raw_parts(counts_ptr.add(base), LANES_16));
                
                // Decrement (saturating)
                let underflow_mask = counts.simd_eq(zero_u16);
                let new_counts = underflow_mask.select(zero_u16, counts - one_u16);
                
                // Store new counts
                new_counts.copy_to_slice(std::slice::from_raw_parts_mut(counts_ptr.add(base), LANES_16));
                
                // Find elements that became zero (count was 1, now 0)
                let became_zero_mask = counts.simd_eq(one_u16);
                
                // Sum areas for elements that became zero
                // Since areas are u64 and we need to accumulate selectively, we use scalar fallback
                // for the area summation
                let mask_bits = became_zero_mask.to_bitmask();
                if mask_bits != 0 {
                    for j in 0..LANES_16 {
                        if (mask_bits >> j) & 1 != 0 {
                            delta_area += *areas_ptr.add(base + j);
                        }
                    }
                }
            }
        }
        
        // Scalar remainder
        for idx in remainder_start..end {
            unsafe {
                let count = counts_ptr.add(idx);
                let old_count = *count;
                *count = old_count.saturating_sub(1);
                if old_count == 1 {
                    delta_area += *areas_ptr.add(idx);
                }
            }
        }
        
        self.current_area += delta_area;
        delta_area as i64
    }

    /// SIMD addition for cloudy area - increment counts and track zero exits.
    #[inline]
    fn simd_add(&mut self, start: usize, end: usize) -> i64 {
        let mut delta_area = 0u64;
        
        let len = end - start;
        const LANES_16: usize = 16;
        let simd_iters = len / LANES_16;
        let remainder_start = start + simd_iters * LANES_16;
        
        let counts_ptr = self.counts.as_mut_ptr();
        let areas_ptr = self.element_areas.as_ptr();
        
        type U16x16 = Simd<u16, LANES_16>;
        let one_u16 = U16x16::splat(1);
        let zero_u16 = U16x16::splat(0);
        
        for i in 0..simd_iters {
            let base = start + i * LANES_16;
            unsafe {
                let counts = U16x16::from_slice(std::slice::from_raw_parts(counts_ptr.add(base), LANES_16));
                
                // Find elements that were zero (will become non-zero)
                let was_zero_mask = counts.simd_eq(zero_u16);
                
                // Sum areas for elements that were zero
                let mask_bits = was_zero_mask.to_bitmask();
                if mask_bits != 0 {
                    for j in 0..LANES_16 {
                        if (mask_bits >> j) & 1 != 0 {
                            delta_area += *areas_ptr.add(base + j);
                        }
                    }
                }
                
                // Increment counts
                let new_counts = counts + one_u16;
                new_counts.copy_to_slice(std::slice::from_raw_parts_mut(counts_ptr.add(base), LANES_16));
            }
        }
        
        // Scalar remainder
        for idx in remainder_start..end {
            unsafe {
                let count = counts_ptr.add(idx);
                let old_count = *count;
                if old_count == 0 {
                    delta_area += *areas_ptr.add(idx);
                }
                *count = old_count + 1;
            }
        }
        
        self.current_area -= delta_area;
        -(delta_area as i64)
    }
}

impl<const D: usize> ObjectiveTracker<D> for ExplicitSimdCloudyAreaState {
    fn peek_removal_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let int_start = unsafe { *self.clear_intervals_offsets.get_unchecked(image_index) };
        let int_end = unsafe { *self.clear_intervals_offsets.get_unchecked(image_index + 1) };
        
        let mut delta_area = 0u64;
        let counts_ptr = self.counts.as_ptr();
        let areas_ptr = self.element_areas.as_ptr();
        
        const LANES_16: usize = 16;
        type U16x16 = Simd<u16, LANES_16>;
        let one_u16 = U16x16::splat(1);
        
        for int_idx in int_start..int_end {
            let interval = unsafe { *self.clear_intervals.get_unchecked(int_idx) };
            let start = interval.start as usize;
            let end = start + interval.len as usize;
            let len = end - start;
            let simd_iters = len / LANES_16;
            let remainder_start = start + simd_iters * LANES_16;
            
            for i in 0..simd_iters {
                let base = start + i * LANES_16;
                unsafe {
                    let counts = U16x16::from_slice(std::slice::from_raw_parts(counts_ptr.add(base), LANES_16));
                    let would_become_zero = counts.simd_eq(one_u16);
                    let mask_bits = would_become_zero.to_bitmask();
                    if mask_bits != 0 {
                        for j in 0..LANES_16 {
                            if (mask_bits >> j) & 1 != 0 {
                                delta_area += *areas_ptr.add(base + j);
                            }
                        }
                    }
                }
            }
            
            for idx in remainder_start..end {
                unsafe {
                    if *counts_ptr.add(idx) == 1 {
                        delta_area += *areas_ptr.add(idx);
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
        let counts_ptr = self.counts.as_ptr();
        let areas_ptr = self.element_areas.as_ptr();
        
        const LANES_16: usize = 16;
        type U16x16 = Simd<u16, LANES_16>;
        let zero_u16 = U16x16::splat(0);
        
        for int_idx in int_start..int_end {
            let interval = unsafe { *self.clear_intervals.get_unchecked(int_idx) };
            let start = interval.start as usize;
            let end = start + interval.len as usize;
            let len = end - start;
            let simd_iters = len / LANES_16;
            let remainder_start = start + simd_iters * LANES_16;
            
            for i in 0..simd_iters {
                let base = start + i * LANES_16;
                unsafe {
                    let counts = U16x16::from_slice(std::slice::from_raw_parts(counts_ptr.add(base), LANES_16));
                    let is_zero = counts.simd_eq(zero_u16);
                    let mask_bits = is_zero.to_bitmask();
                    if mask_bits != 0 {
                        for j in 0..LANES_16 {
                            if (mask_bits >> j) & 1 != 0 {
                                delta_area += *areas_ptr.add(base + j);
                            }
                        }
                    }
                }
            }
            
            for idx in remainder_start..end {
                unsafe {
                    if *counts_ptr.add(idx) == 0 {
                        delta_area += *areas_ptr.add(idx);
                    }
                }
            }
        }
        
        -(delta_area as i64)
    }

    fn track_image_removal(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let int_start = unsafe { *self.clear_intervals_offsets.get_unchecked(image_index) };
        let int_end = unsafe { *self.clear_intervals_offsets.get_unchecked(image_index + 1) };
        
        let mut total_delta = 0i64;
        for int_idx in int_start..int_end {
            let interval = unsafe { *self.clear_intervals.get_unchecked(int_idx) };
            let start = interval.start as usize;
            let end = start + interval.len as usize;
            total_delta += self.simd_remove(start, end);
        }
        total_delta
    }

    fn track_image_addition(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let int_start = unsafe { *self.clear_intervals_offsets.get_unchecked(image_index) };
        let int_end = unsafe { *self.clear_intervals_offsets.get_unchecked(image_index + 1) };
        
        let mut total_delta = 0i64;
        for int_idx in int_start..int_end {
            let interval = unsafe { *self.clear_intervals.get_unchecked(int_idx) };
            let start = interval.start as usize;
            let end = start + interval.len as usize;
            total_delta += self.simd_add(start, end);
        }
        total_delta
    }

    fn value(&self) -> u64 {
        self.current_area
    }
}

// =============================================================================
// Explicit SIMD Tracker Array
// =============================================================================

use super::simd_trackers::{SimdTotalCostState, SimdMaxIncidenceAngleState};

/// Tracker enum for explicit SIMD implementations
#[derive(Clone, Debug)]
pub enum ExplicitSimdTracker {
    TotalCost(SimdTotalCostState),
    CloudyArea(ExplicitSimdCloudyAreaState),
    MinResolution(ExplicitSimdMinResState),
    MaxIncidenceAngle(SimdMaxIncidenceAngleState),
}

impl ExplicitSimdTracker {
    /// Get the current value directly (inherent method to avoid trait dispatch issues)
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

impl<const D: usize> ObjectiveTracker<D> for ExplicitSimdTracker {
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

/// Tracker array using explicit SIMD implementations
#[derive(Clone, Debug)]
pub struct ExplicitSimdTrackerArray<const D: usize> {
    trackers: [ExplicitSimdTracker; D],
}

impl<const D: usize> TrackerCollection<D> for ExplicitSimdTrackerArray<D> {
    type Tracker = ExplicitSimdTracker;

    fn get(&self, index: usize) -> &ExplicitSimdTracker {
        &self.trackers[index]
    }

    fn get_mut(&mut self, index: usize) -> &mut ExplicitSimdTracker {
        &mut self.trackers[index]
    }

    fn new(problem: &impl SetCoverProblem<D>) -> Self {
        let shared = simd_shared_data(problem);

        let trackers = std::array::from_fn(|i| match problem.objective(i) {
            crate::objectives::ObjectiveState::TotalCost { .. } => {
                ExplicitSimdTracker::TotalCost(SimdTotalCostState {
                    current_cost: 0,
                    image_costs: Arc::clone(&shared.image_costs),
                })
            }
            crate::objectives::ObjectiveState::CloudyArea { .. } => {
                let total_area: u64 = shared.element_areas.iter().sum();

                ExplicitSimdTracker::CloudyArea(ExplicitSimdCloudyAreaState {
                    counts: vec![0; problem.num_elements()],
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

                ExplicitSimdTracker::MinResolution(ExplicitSimdMinResState {
                    packed_counts: vec![0; problem.num_elements()],
                    low_val,
                    high_val,
                    diff: (high_val - low_val) as i64,
                    current_sum: 0,
                    image_intervals: Arc::clone(&shared.image_intervals),
                    image_intervals_offsets: Arc::clone(&shared.image_intervals_offsets),
                    image_resolution_level: Arc::clone(&shared.image_resolution_level),
                    image_elements: Arc::clone(&shared.image_elements),
                    image_elements_offsets: Arc::clone(&shared.image_elements_offsets),
                })
            }
            crate::objectives::ObjectiveState::MaxIncidenceAngle { .. } => {
                ExplicitSimdTracker::MaxIncidenceAngle(SimdMaxIncidenceAngleState {
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
