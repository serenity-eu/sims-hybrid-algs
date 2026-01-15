//! Segment Tree-based objective trackers for O(intervals * log U) updates.
//!
//! Key insight: When element lists are interval-compressible (7.23x on Lagos-30),
//! we can use range-update segment trees instead of per-element iteration.
//!
//! For MinResolution with 2-level resolution:
//! - Each node stores a 3x3 histogram: count of elements in each (c0_state, c1_state) bucket
//! - c0_state, c1_state ∈ {0, 1, 2+}
//! - Range updates shift entire rows/columns of the histogram
//! - Query is O(1): read root node's histogram and compute weighted sum

use std::sync::Arc;

use crate::objective_tracker::ObjectiveTracker;
use crate::problem::SetCoverProblem;
use crate::solution::ImageSet;

use super::simd_trackers::Interval;

// =============================================================================
// Segment Tree for MinResolution (2-level case)
// =============================================================================

/// 3x3 histogram for (c0_state, c1_state) where state ∈ {0, 1, 2+}
/// Plus lazy tags for pending +1/-1 operations on each counter dimension.
#[derive(Clone, Copy, Debug, Default)]
struct MinResNode {
    /// hist[c0_state][c1_state] = count of elements in this bucket
    /// Flattened as hist[c0_state * 3 + c1_state]
    hist: [u32; 9],
    /// Pending delta for c0 counter (-1, 0, or +1)
    lazy_c0: i8,
    /// Pending delta for c1 counter (-1, 0, or +1)
    lazy_c1: i8,
}

impl MinResNode {
    #[inline]
    fn total(&self) -> u32 {
        self.hist.iter().sum()
    }

    /// Merge two child nodes into parent (no lazy propagation)
    #[inline]
    fn merge(left: &Self, right: &Self) -> Self {
        let mut hist = [0u32; 9];
        for i in 0..9 {
            hist[i] = left.hist[i] + right.hist[i];
        }
        Self { hist, lazy_c0: 0, lazy_c1: 0 }
    }

    /// Compute objective contribution: low_val * (elements with c0>0) + 
    /// (high_val - low_val) * (elements with c0==0 && c1>0)
    #[inline]
    fn objective_value(&self, low_val: u64, high_val: u64) -> u64 {
        // Elements with c0 > 0: columns 1,2 for all c1 rows
        // hist indices where c0_state ∈ {1, 2} are: 1,2, 4,5, 7,8
        let c0_positive = self.hist[1] + self.hist[2] + self.hist[4] + self.hist[5] + self.hist[7] + self.hist[8];
        
        // Elements with c0 == 0 && c1 > 0: hist[0][1] + hist[0][2] = indices 3, 6
        let c0_zero_c1_positive = self.hist[3] + self.hist[6];
        
        (c0_positive as u64) * low_val + (c0_zero_c1_positive as u64) * high_val
    }
}

/// Segment tree for MinResolution with lazy propagation.
/// Supports O(log U) range updates for +1/-1 on c0 or c1 dimension.
#[derive(Clone, Debug)]
pub struct MinResSegmentTree {
    nodes: Vec<MinResNode>,
    n: usize, // Universe size (rounded to power of 2)
    low_val: u64,
    high_val: u64,
}

impl MinResSegmentTree {
    /// Create a new segment tree for `universe_size` elements.
    /// Initially all elements have c0=0, c1=0 (hist[0][0] = universe_size).
    pub fn new(universe_size: usize, low_val: u64, high_val: u64) -> Self {
        // Round up to power of 2
        let n = universe_size.next_power_of_two();
        let mut nodes = vec![MinResNode::default(); 2 * n];
        
        // Initialize leaves: each element starts in state (0, 0)
        for i in 0..universe_size {
            nodes[n + i].hist[0] = 1; // hist[0][0] = 1
        }
        
        // Build tree bottom-up
        for i in (1..n).rev() {
            nodes[i] = MinResNode::merge(&nodes[2 * i], &nodes[2 * i + 1]);
        }
        
        Self { nodes, n, low_val, high_val }
    }

    /// Push lazy tags from parent to children
    #[inline]
    fn push_down(&mut self, node: usize) {
        if node >= self.n {
            return; // Leaf node
        }
        
        let lazy_c0 = self.nodes[node].lazy_c0;
        let lazy_c1 = self.nodes[node].lazy_c1;
        
        if lazy_c0 != 0 || lazy_c1 != 0 {
            // Apply to children
            self.apply_lazy(2 * node, lazy_c0, lazy_c1);
            self.apply_lazy(2 * node + 1, lazy_c0, lazy_c1);
            self.nodes[node].lazy_c0 = 0;
            self.nodes[node].lazy_c1 = 0;
        }
    }

    /// Apply lazy delta to a node's histogram (shift buckets)
    #[inline]
    fn apply_lazy(&mut self, node: usize, delta_c0: i8, delta_c1: i8) {
        let n = &mut self.nodes[node];
        
        // Shift histogram based on deltas
        if delta_c0 == 1 {
            // c0 += 1: shift columns right (0->1, 1->2, 2->2)
            // new_hist[r][c] = old_hist[r][c-1] for c>0, with c=2 absorbing overflow
            let mut new_hist = [0u32; 9];
            for r in 0..3 {
                new_hist[r * 3 + 1] = n.hist[r * 3 + 0]; // 0 -> 1
                new_hist[r * 3 + 2] = n.hist[r * 3 + 1] + n.hist[r * 3 + 2]; // 1,2 -> 2
            }
            n.hist = new_hist;
        } else if delta_c0 == -1 {
            // c0 -= 1: shift columns left (0->0, 1->0, 2->1)
            let mut new_hist = [0u32; 9];
            for r in 0..3 {
                new_hist[r * 3 + 0] = n.hist[r * 3 + 0] + n.hist[r * 3 + 1]; // 0,1 -> 0
                new_hist[r * 3 + 1] = n.hist[r * 3 + 2]; // 2 -> 1
            }
            n.hist = new_hist;
        }
        
        if delta_c1 == 1 {
            // c1 += 1: shift rows down (0->1, 1->2, 2->2)
            let mut new_hist = [0u32; 9];
            for c in 0..3 {
                new_hist[1 * 3 + c] = n.hist[0 * 3 + c]; // row 0 -> row 1
                new_hist[2 * 3 + c] = n.hist[1 * 3 + c] + n.hist[2 * 3 + c]; // rows 1,2 -> row 2
            }
            n.hist = new_hist;
        } else if delta_c1 == -1 {
            // c1 -= 1: shift rows up (0->0, 1->0, 2->1)
            let mut new_hist = [0u32; 9];
            for c in 0..3 {
                new_hist[0 * 3 + c] = n.hist[0 * 3 + c] + n.hist[1 * 3 + c]; // rows 0,1 -> row 0
                new_hist[1 * 3 + c] = n.hist[2 * 3 + c]; // row 2 -> row 1
            }
            n.hist = new_hist;
        }
        
        // Accumulate lazy tags
        n.lazy_c0 = (n.lazy_c0 + delta_c0).clamp(-1, 1);
        n.lazy_c1 = (n.lazy_c1 + delta_c1).clamp(-1, 1);
    }

    /// Range update: add delta_c0 to c0 counter and delta_c1 to c1 counter
    /// for all elements in [l, r)
    fn range_update(&mut self, l: usize, r: usize, delta_c0: i8, delta_c1: i8) {
        self.range_update_impl(1, 0, self.n, l, r, delta_c0, delta_c1);
    }

    fn range_update_impl(
        &mut self,
        node: usize,
        node_l: usize,
        node_r: usize,
        l: usize,
        r: usize,
        delta_c0: i8,
        delta_c1: i8,
    ) {
        if r <= node_l || node_r <= l {
            return; // No overlap
        }
        
        if l <= node_l && node_r <= r {
            // Fully covered - apply lazy update
            self.apply_lazy(node, delta_c0, delta_c1);
            return;
        }
        
        // Partial overlap - push down and recurse
        self.push_down(node);
        let mid = (node_l + node_r) / 2;
        self.range_update_impl(2 * node, node_l, mid, l, r, delta_c0, delta_c1);
        self.range_update_impl(2 * node + 1, mid, node_r, l, r, delta_c0, delta_c1);
        
        // Merge children back
        self.nodes[node] = MinResNode::merge(&self.nodes[2 * node], &self.nodes[2 * node + 1]);
    }

    /// Get current objective value from root
    #[inline]
    pub fn value(&self) -> u64 {
        self.nodes[1].objective_value(self.low_val, self.high_val)
    }

    /// Update for intervals: apply delta to c0 or c1 based on resolution level
    pub fn update_intervals(&mut self, intervals: &[Interval], level: usize, delta: i8) {
        let (delta_c0, delta_c1) = if level == 0 {
            (delta, 0)
        } else {
            (0, delta)
        };
        
        for interval in intervals {
            let start = interval.start as usize;
            let end = start + interval.len as usize;
            self.range_update(start, end, delta_c0, delta_c1);
        }
    }
}

// =============================================================================
// Segment Tree for CloudyArea (weighted count of zeros)
// =============================================================================

/// Node for CloudyArea segment tree.
/// Tracks count of elements in each state (0, 1, 2+) and their weighted sum (area).
#[derive(Clone, Copy, Debug, Default)]
struct CloudyNode {
    /// count[state] = number of elements with count in that state
    count: [u32; 3],
    /// area_sum[state] = sum of areas of elements in that state
    area_sum: [u64; 3],
    /// Pending delta (-1, 0, or +1)
    lazy: i8,
}

impl CloudyNode {
    #[inline]
    fn merge(left: &Self, right: &Self) -> Self {
        Self {
            count: [
                left.count[0] + right.count[0],
                left.count[1] + right.count[1],
                left.count[2] + right.count[2],
            ],
            area_sum: [
                left.area_sum[0] + right.area_sum[0],
                left.area_sum[1] + right.area_sum[1],
                left.area_sum[2] + right.area_sum[2],
            ],
            lazy: 0,
        }
    }

    /// CloudyArea = sum of areas where count == 0
    #[inline]
    fn cloudy_area(&self) -> u64 {
        self.area_sum[0]
    }
}

/// Segment tree for CloudyArea with lazy propagation.
#[derive(Clone, Debug)]
pub struct CloudySegmentTree {
    nodes: Vec<CloudyNode>,
    n: usize,
    element_areas: Vec<u64>, // Store areas for initialization
}

impl CloudySegmentTree {
    pub fn new(element_areas: &[u64]) -> Self {
        let universe_size = element_areas.len();
        let n = universe_size.next_power_of_two();
        let mut nodes = vec![CloudyNode::default(); 2 * n];
        
        // Initialize leaves: each element starts with count=0
        for i in 0..universe_size {
            nodes[n + i].count[0] = 1;
            nodes[n + i].area_sum[0] = element_areas[i];
        }
        
        // Build tree bottom-up
        for i in (1..n).rev() {
            nodes[i] = CloudyNode::merge(&nodes[2 * i], &nodes[2 * i + 1]);
        }
        
        Self { nodes, n, element_areas: element_areas.to_vec() }
    }

    #[inline]
    fn push_down(&mut self, node: usize) {
        if node >= self.n {
            return;
        }
        
        let lazy = self.nodes[node].lazy;
        if lazy != 0 {
            self.apply_lazy(2 * node, lazy);
            self.apply_lazy(2 * node + 1, lazy);
            self.nodes[node].lazy = 0;
        }
    }

    #[inline]
    fn apply_lazy(&mut self, node: usize, delta: i8) {
        let n = &mut self.nodes[node];
        
        if delta == 1 {
            // count += 1: shift states right (0->1, 1->2, 2->2)
            let new_count = [0, n.count[0], n.count[1] + n.count[2]];
            let new_area = [0, n.area_sum[0], n.area_sum[1] + n.area_sum[2]];
            n.count = new_count;
            n.area_sum = new_area;
        } else if delta == -1 {
            // count -= 1: shift states left (0->0, 1->0, 2->1)
            let new_count = [n.count[0] + n.count[1], n.count[2], 0];
            let new_area = [n.area_sum[0] + n.area_sum[1], n.area_sum[2], 0];
            n.count = new_count;
            n.area_sum = new_area;
        }
        
        n.lazy = (n.lazy + delta).clamp(-1, 1);
    }

    fn range_update(&mut self, l: usize, r: usize, delta: i8) {
        self.range_update_impl(1, 0, self.n, l, r, delta);
    }

    fn range_update_impl(
        &mut self,
        node: usize,
        node_l: usize,
        node_r: usize,
        l: usize,
        r: usize,
        delta: i8,
    ) {
        if r <= node_l || node_r <= l {
            return;
        }
        
        if l <= node_l && node_r <= r {
            self.apply_lazy(node, delta);
            return;
        }
        
        self.push_down(node);
        let mid = (node_l + node_r) / 2;
        self.range_update_impl(2 * node, node_l, mid, l, r, delta);
        self.range_update_impl(2 * node + 1, mid, node_r, l, r, delta);
        self.nodes[node] = CloudyNode::merge(&self.nodes[2 * node], &self.nodes[2 * node + 1]);
    }

    #[inline]
    pub fn cloudy_area(&self) -> u64 {
        self.nodes[1].cloudy_area()
    }

    pub fn update_intervals(&mut self, intervals: &[Interval], delta: i8) {
        for interval in intervals {
            let start = interval.start as usize;
            let end = start + interval.len as usize;
            self.range_update(start, end, delta);
        }
    }
}

// =============================================================================
// Segment Tree Tracker Implementation
// =============================================================================

/// Segment tree-based tracker state for MinResolution
#[derive(Clone, Debug)]
pub struct SegTreeMinResolutionState {
    tree: MinResSegmentTree,
    image_intervals: Arc<Vec<Interval>>,
    image_intervals_offsets: Arc<Vec<usize>>,
    image_resolution_level: Arc<Vec<u8>>,
}

impl<const D: usize> ObjectiveTracker<D> for SegTreeMinResolutionState {
    fn peek_removal_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        // For peek, we need to compute the delta without modifying
        // This is expensive with lazy segment trees - we'd need to clone
        // For now, use a simpler approach: compute from scratch
        let old_val = self.tree.value();
        
        // Clone tree and apply update
        let mut tree_copy = MinResSegmentTree {
            nodes: self.tree.nodes.clone(),
            n: self.tree.n,
            low_val: self.tree.low_val,
            high_val: self.tree.high_val,
        };
        
        let int_start = self.image_intervals_offsets[image_index];
        let int_end = self.image_intervals_offsets[image_index + 1];
        let intervals = &self.image_intervals[int_start..int_end];
        let level = self.image_resolution_level[image_index] as usize;
        tree_copy.update_intervals(intervals, level, -1);
        
        let new_val = tree_copy.value();
        new_val as i64 - old_val as i64
    }

    fn peek_addition_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let old_val = self.tree.value();
        
        let mut tree_copy = MinResSegmentTree {
            nodes: self.tree.nodes.clone(),
            n: self.tree.n,
            low_val: self.tree.low_val,
            high_val: self.tree.high_val,
        };
        
        let int_start = self.image_intervals_offsets[image_index];
        let int_end = self.image_intervals_offsets[image_index + 1];
        let intervals = &self.image_intervals[int_start..int_end];
        let level = self.image_resolution_level[image_index] as usize;
        tree_copy.update_intervals(intervals, level, 1);
        
        let new_val = tree_copy.value();
        new_val as i64 - old_val as i64
    }

    fn track_image_removal(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let old_val = self.tree.value();
        
        let int_start = self.image_intervals_offsets[image_index];
        let int_end = self.image_intervals_offsets[image_index + 1];
        let intervals = &self.image_intervals[int_start..int_end];
        let level = self.image_resolution_level[image_index] as usize;
        self.tree.update_intervals(intervals, level, -1);
        
        let new_val = self.tree.value();
        new_val as i64 - old_val as i64
    }

    fn track_image_addition(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let old_val = self.tree.value();
        
        let int_start = self.image_intervals_offsets[image_index];
        let int_end = self.image_intervals_offsets[image_index + 1];
        let intervals = &self.image_intervals[int_start..int_end];
        let level = self.image_resolution_level[image_index] as usize;
        self.tree.update_intervals(intervals, level, 1);
        
        let new_val = self.tree.value();
        new_val as i64 - old_val as i64
    }

    fn value(&self) -> u64 {
        self.tree.value()
    }
}

/// Segment tree-based tracker state for CloudyArea
#[derive(Clone, Debug)]
pub struct SegTreeCloudyAreaState {
    tree: CloudySegmentTree,
    clear_intervals: Arc<Vec<Interval>>,
    clear_intervals_offsets: Arc<Vec<usize>>,
}

impl<const D: usize> ObjectiveTracker<D> for SegTreeCloudyAreaState {
    fn peek_removal_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let old_val = self.tree.cloudy_area();
        
        let mut tree_copy = CloudySegmentTree {
            nodes: self.tree.nodes.clone(),
            n: self.tree.n,
            element_areas: self.tree.element_areas.clone(),
        };
        
        let int_start = self.clear_intervals_offsets[image_index];
        let int_end = self.clear_intervals_offsets[image_index + 1];
        let intervals = &self.clear_intervals[int_start..int_end];
        tree_copy.update_intervals(intervals, -1);
        
        let new_val = tree_copy.cloudy_area();
        new_val as i64 - old_val as i64
    }

    fn peek_addition_delta(&self, image_index: usize, _p: &impl SetCoverProblem<D>, _s: &impl ImageSet<D>) -> i64 {
        let old_val = self.tree.cloudy_area();
        
        let mut tree_copy = CloudySegmentTree {
            nodes: self.tree.nodes.clone(),
            n: self.tree.n,
            element_areas: self.tree.element_areas.clone(),
        };
        
        let int_start = self.clear_intervals_offsets[image_index];
        let int_end = self.clear_intervals_offsets[image_index + 1];
        let intervals = &self.clear_intervals[int_start..int_end];
        tree_copy.update_intervals(intervals, 1);
        
        let new_val = tree_copy.cloudy_area();
        new_val as i64 - old_val as i64
    }

    fn track_image_removal(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let old_val = self.tree.cloudy_area();
        
        let int_start = self.clear_intervals_offsets[image_index];
        let int_end = self.clear_intervals_offsets[image_index + 1];
        let intervals = &self.clear_intervals[int_start..int_end];
        self.tree.update_intervals(intervals, -1);
        
        let new_val = self.tree.cloudy_area();
        new_val as i64 - old_val as i64
    }

    fn track_image_addition(&mut self, image_index: usize, _p: &impl SetCoverProblem<D>) -> i64 {
        let old_val = self.tree.cloudy_area();
        
        let int_start = self.clear_intervals_offsets[image_index];
        let int_end = self.clear_intervals_offsets[image_index + 1];
        let intervals = &self.clear_intervals[int_start..int_end];
        self.tree.update_intervals(intervals, 1);
        
        let new_val = self.tree.cloudy_area();
        new_val as i64 - old_val as i64
    }

    fn value(&self) -> u64 {
        self.tree.cloudy_area()
    }
}
