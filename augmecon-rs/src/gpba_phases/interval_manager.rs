use std::collections::BTreeSet;

/// Manages intervals for adaptive Pareto front exploration in GPBA-A
///
/// This replaces the uniform grid approach with adaptive interval-based exploration
/// that focuses on the largest gaps in the Pareto front.
///
/// Uses `BTreeSet` instead of `HashSet` for deterministic iteration order,
/// which eliminates run-to-run variability when multiple intervals tie in length.
#[derive(Debug, Clone)]
pub struct IntervalManager {
    /// Set of non-overlapping intervals (start, end) where both are inclusive.
    /// Stored in a `BTreeSet` for deterministic ordering by (start, end).
    pub intervals: BTreeSet<(i64, i64)>,
    /// Minimum value in the managed range
    pub min_value: i64,
    /// Maximum value in the managed range
    pub max_value: i64,
}

impl IntervalManager {
    /// Create a new `IntervalManager` with initial full range
    #[must_use]
    pub fn new(min_value: i64, max_value: i64) -> Self {
        let mut intervals = BTreeSet::new();
        intervals.insert((min_value, max_value));
        log::debug!("IntervalManager initialized: [{min_value}, {max_value}]");
        Self {
            intervals,
            min_value,
            max_value,
        }
    }

    /// Add interval, merging with overlapping intervals
    pub fn add_interval(&mut self, start: i64, end: i64) {
        let mut new_intervals = BTreeSet::new();
        let mut to_add = (start, end);

        for &interval in &self.intervals {
            if interval.1 < start || interval.0 > end {
                // No overlap
                new_intervals.insert(interval);
            } else {
                // Merge overlapping intervals
                to_add = (to_add.0.min(interval.0), to_add.1.max(interval.1));
            }
        }

        new_intervals.insert(to_add);
        self.intervals = new_intervals;
        log::debug!(
            "Added interval [{}, {}], total intervals: {}",
            start,
            end,
            self.intervals.len()
        );
    }

    /// Remove a single point, splitting intervals if necessary
    pub fn remove_one_point(&mut self, point: i64) {
        let mut new_intervals = BTreeSet::new();

        for &interval in &self.intervals {
            if interval.0 <= point && point <= interval.1 {
                // Point within interval - split
                if interval.0 < point {
                    new_intervals.insert((interval.0, point - 1));
                }
                if interval.1 > point {
                    new_intervals.insert((point + 1, interval.1));
                }
                log::debug!(
                    "Removed point {} from interval [{}, {}]",
                    point,
                    interval.0,
                    interval.1
                );
            } else {
                // No overlap
                new_intervals.insert(interval);
            }
        }

        self.intervals = new_intervals;
    }

    /// Remove interval, adjusting or splitting existing intervals
    pub fn remove_interval(&mut self, start: i64, end: i64) {
        let mut new_intervals = BTreeSet::new();

        for &interval in &self.intervals {
            if interval.1 < start || interval.0 > end {
                // No overlap
                new_intervals.insert(interval);
            } else {
                // Adjust or split interval
                if interval.0 < start {
                    new_intervals.insert((interval.0, start - 1));
                }
                if interval.1 > end {
                    new_intervals.insert((end + 1, interval.1));
                }
                log::debug!(
                    "Removed interval [{}, {}] from [{}, {}]",
                    start,
                    end,
                    interval.0,
                    interval.1
                );
            }
        }

        self.intervals = new_intervals;
    }

    /// Find and return the largest interval by length.
    ///
    /// Tie-breaking is deterministic: when two intervals have equal length,
    /// the interval with the **smaller start value** (leftmost) is preferred.
    /// This eliminates run-to-run variability that previously occurred with
    /// `HashSet` iteration order.
    #[must_use]
    pub fn find_largest_interval(&self) -> Option<(i64, i64)> {
        if self.intervals.is_empty() {
            return None;
        }

        // Deterministic tie-breaking: prefer largest length, then smallest start
        let largest = self
            .intervals
            .iter()
            .max_by(|&&(s1, e1), &&(s2, e2)| {
                let len1 = e1 - s1;
                let len2 = e2 - s2;
                len1.cmp(&len2).then_with(|| s2.cmp(&s1))
            })
            .copied();

        if let Some((start, end)) = largest {
            log::debug!(
                "Largest interval: [{}, {}] (length: {})",
                start,
                end,
                end - start + 1
            );
        }

        largest
    }

    /// Returns the total number of integer points covered by all intervals.
    #[must_use]
    pub fn total_coverage(&self) -> i64 {
        self.intervals
            .iter()
            .map(|&(start, end)| end - start + 1)
            .sum()
    }

    /// Returns true if there are no intervals remaining.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    /// Returns the number of disjoint intervals.
    #[must_use]
    pub fn num_intervals(&self) -> usize {
        self.intervals.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_interval() {
        let manager = IntervalManager::new(0, 100);
        assert_eq!(manager.intervals.len(), 1);
        assert!(manager.intervals.contains(&(0, 100)));
    }

    #[test]
    fn test_remove_one_point() {
        let mut manager = IntervalManager::new(0, 100);
        manager.remove_one_point(50);

        assert_eq!(manager.intervals.len(), 2);
        assert!(manager.intervals.contains(&(0, 49)));
        assert!(manager.intervals.contains(&(51, 100)));
    }

    #[test]
    fn test_remove_one_point_at_start() {
        let mut manager = IntervalManager::new(0, 100);
        manager.remove_one_point(0);

        assert_eq!(manager.intervals.len(), 1);
        assert!(manager.intervals.contains(&(1, 100)));
    }

    #[test]
    fn test_remove_one_point_at_end() {
        let mut manager = IntervalManager::new(0, 100);
        manager.remove_one_point(100);

        assert_eq!(manager.intervals.len(), 1);
        assert!(manager.intervals.contains(&(0, 99)));
    }

    #[test]
    fn test_remove_interval() {
        let mut manager = IntervalManager::new(0, 100);
        manager.remove_interval(30, 70);

        assert_eq!(manager.intervals.len(), 2);
        assert!(manager.intervals.contains(&(0, 29)));
        assert!(manager.intervals.contains(&(71, 100)));
    }

    #[test]
    fn test_remove_interval_partial_overlap_left() {
        let mut manager = IntervalManager::new(50, 100);
        manager.remove_interval(30, 70);

        assert_eq!(manager.intervals.len(), 1);
        assert!(manager.intervals.contains(&(71, 100)));
    }

    #[test]
    fn test_remove_interval_partial_overlap_right() {
        let mut manager = IntervalManager::new(0, 50);
        manager.remove_interval(30, 70);

        assert_eq!(manager.intervals.len(), 1);
        assert!(manager.intervals.contains(&(0, 29)));
    }

    #[test]
    fn test_remove_interval_complete_overlap() {
        let mut manager = IntervalManager::new(40, 60);
        manager.remove_interval(30, 70);

        assert_eq!(manager.intervals.len(), 0);
    }

    #[test]
    fn test_find_largest_interval() {
        let mut manager = IntervalManager::new(0, 100);
        manager.remove_interval(30, 40);

        let largest = manager.find_largest_interval().unwrap();
        assert_eq!(largest, (41, 100)); // Length 60 vs length 30
    }

    #[test]
    fn test_find_largest_interval_empty() {
        let mut manager = IntervalManager::new(0, 10);
        manager.remove_interval(0, 10);

        assert!(manager.find_largest_interval().is_none());
    }

    #[test]
    fn test_add_interval_merging() {
        let mut manager = IntervalManager::new(0, 10);
        manager.remove_interval(3, 7);
        // Now have: (0,2) and (8,10)

        manager.add_interval(2, 9);
        // Should merge to: (0,10)

        assert_eq!(manager.intervals.len(), 1);
        assert!(manager.intervals.contains(&(0, 10)));
    }

    #[test]
    fn test_add_interval_no_overlap() {
        let mut manager = IntervalManager::new(0, 10);
        manager.remove_interval(3, 7);
        // Now have: (0,2) and (8,10)

        manager.add_interval(20, 30);

        assert_eq!(manager.intervals.len(), 3);
        assert!(manager.intervals.contains(&(0, 2)));
        assert!(manager.intervals.contains(&(8, 10)));
        assert!(manager.intervals.contains(&(20, 30)));
    }

    #[test]
    fn test_multiple_operations() {
        let mut manager = IntervalManager::new(0, 100);

        // Remove middle section
        manager.remove_interval(40, 60);
        assert_eq!(manager.intervals.len(), 2);

        // Remove a point from the left interval
        manager.remove_one_point(20);
        assert_eq!(manager.intervals.len(), 3);
        assert!(manager.intervals.contains(&(0, 19)));
        assert!(manager.intervals.contains(&(21, 39)));
        assert!(manager.intervals.contains(&(61, 100)));

        // Find largest interval
        let largest = manager.find_largest_interval().unwrap();
        assert_eq!(largest, (61, 100)); // Length 40
    }

    #[test]
    fn test_remove_single_point_interval() {
        let mut manager = IntervalManager::new(5, 5);
        manager.remove_one_point(5);

        assert_eq!(manager.intervals.len(), 0);
    }

    #[test]
    fn test_adjacent_intervals_after_removal() {
        let mut manager = IntervalManager::new(0, 10);
        manager.remove_one_point(5);

        assert_eq!(manager.intervals.len(), 2);
        assert!(manager.intervals.contains(&(0, 4)));
        assert!(manager.intervals.contains(&(6, 10)));

        // The intervals should not merge as they are not overlapping
        manager.remove_one_point(6);
        assert_eq!(manager.intervals.len(), 2);
        assert!(manager.intervals.contains(&(0, 4)));
        assert!(manager.intervals.contains(&(7, 10)));
    }

    // ────────────────────────────────────────────────────────────────
    //  Determinism tests (these would fail with HashSet)
    // ────────────────────────────────────────────────────────────────

    #[test]
    fn test_find_largest_interval_deterministic_tie_breaking() {
        // Create two intervals of equal length and verify the leftmost is chosen
        let mut manager = IntervalManager::new(0, 100);

        // Remove the middle so we get two intervals of equal length:
        // (0, 49) length=50 and (51, 100) length=50
        manager.remove_one_point(50);

        let largest = manager.find_largest_interval().unwrap();
        // With deterministic tie-breaking (prefer smaller start), should always be (0, 49)
        assert_eq!(
            largest,
            (0, 49),
            "Deterministic tie-breaking should prefer the leftmost interval"
        );
    }

    #[test]
    fn test_find_largest_interval_deterministic_many_equal() {
        // Create many intervals of equal length.
        // Range (0, 98) with points 9,19,...,89 removed yields 10 intervals
        // each of length 9: (0,8), (10,18), (20,28), ..., (90,98).
        let mut manager = IntervalManager::new(0, 98);

        for p in (9..98).step_by(10) {
            manager.remove_one_point(p);
        }

        assert_eq!(manager.intervals.len(), 10);

        // Run many times — should always return (0, 8)
        for _ in 0..100 {
            let largest = manager.find_largest_interval().unwrap();
            assert_eq!(
                largest,
                (0, 8),
                "Should always pick the leftmost interval on tie"
            );
        }
    }

    #[test]
    fn test_find_largest_interval_prefers_longer_over_leftmost() {
        // Verify that a longer interval is still preferred over a leftmost shorter one
        let mut manager = IntervalManager::new(0, 100);
        manager.remove_interval(20, 30);
        // Now: (0, 19) length=20 and (31, 100) length=70
        // Should pick (31, 100) because it's longer, regardless of position

        let largest = manager.find_largest_interval().unwrap();
        assert_eq!(largest, (31, 100));
    }

    #[test]
    fn test_total_coverage() {
        let mut manager = IntervalManager::new(0, 100);
        assert_eq!(manager.total_coverage(), 101);

        manager.remove_one_point(50);
        assert_eq!(manager.total_coverage(), 100);

        manager.remove_interval(20, 30);
        assert_eq!(manager.total_coverage(), 89); // 100 - 11
    }

    #[test]
    fn test_is_empty() {
        let mut manager = IntervalManager::new(5, 5);
        assert!(!manager.is_empty());

        manager.remove_one_point(5);
        assert!(manager.is_empty());
    }

    #[test]
    fn test_num_intervals() {
        let mut manager = IntervalManager::new(0, 100);
        assert_eq!(manager.num_intervals(), 1);

        manager.remove_one_point(50);
        assert_eq!(manager.num_intervals(), 2);

        manager.remove_one_point(25);
        assert_eq!(manager.num_intervals(), 3);
    }
}
