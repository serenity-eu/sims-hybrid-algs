use std::collections::HashSet;

/// Manages intervals for adaptive Pareto front exploration in GPBA-A
///
/// This replaces the uniform grid approach with adaptive interval-based exploration
/// that focuses on the largest gaps in the Pareto front.
#[derive(Debug, Clone)]
pub struct IntervalManager {
    /// Set of non-overlapping intervals (start, end) where both are inclusive
    pub intervals: HashSet<(i64, i64)>,
    /// Minimum value in the managed range
    pub min_value: i64,
    /// Maximum value in the managed range
    pub max_value: i64,
}

impl IntervalManager {
    /// Create a new `IntervalManager` with initial full range
    #[must_use]
    pub fn new(min_value: i64, max_value: i64) -> Self {
        let mut intervals = HashSet::new();
        intervals.insert((min_value, max_value));
        log::debug!(
            "IntervalManager initialized: [{min_value}, {max_value}]"
        );
        Self {
            intervals,
            min_value,
            max_value,
        }
    }

    /// Add interval, merging with overlapping intervals
    pub fn add_interval(&mut self, start: i64, end: i64) {
        let mut new_intervals = HashSet::new();
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
        let mut new_intervals = HashSet::new();

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
        let mut new_intervals = HashSet::new();

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

    /// Find and return the largest interval by length
    #[must_use]
    pub fn find_largest_interval(&self) -> Option<(i64, i64)> {
        if self.intervals.is_empty() {
            return None;
        }

        let largest = self
            .intervals
            .iter()
            .max_by_key(|&&(start, end)| end - start)
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
        assert_eq!(largest, (41, 100)); // Length 60 vs length 31
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
}
