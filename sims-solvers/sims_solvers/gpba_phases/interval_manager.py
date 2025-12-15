"""IntervalManager for adaptive Pareto front exploration.

Manages intervals for efficient coverage in GPBA-A algorithm.
Extracted from solve.py lines 46-105.
"""


class IntervalManager:
    """Manages intervals for efficient Pareto front coverage in GPBA-A algorithm.
    
    This class tracks unexplored regions of the objective space and identifies
    the largest gaps where new Pareto solutions should be searched.
    """
    
    def __init__(self, min_value, max_value):
        """Initialize with a single interval covering [min_value, max_value]."""
        self.intervals = set()
        self.min_value = min_value
        self.max_value = max_value
        self.add_interval(min_value, max_value)
    
    def add_interval(self, start, end):
        """Add interval, merging with existing overlapping intervals."""
        new_intervals = set()
        to_add = (start, end)
        
        for interval in self.intervals:
            if interval[1] < start or interval[0] > end:  # No overlap
                new_intervals.add(interval)
            else:  # Merge overlapping intervals
                to_add = (min(to_add[0], interval[0]), max(to_add[1], interval[1]))
        
        new_intervals.add(to_add)
        self.intervals = new_intervals
    
    def remove_one_point(self, point):
        """Remove a single point, splitting intervals if necessary."""
        new_intervals = set()
        
        for interval in self.intervals:
            if interval[0] <= point <= interval[1]:  # Point within interval
                if interval[0] < point:
                    new_intervals.add((interval[0], point - 1))
                if interval[1] > point:
                    new_intervals.add((point + 1, interval[1]))
            else:  # No overlap
                new_intervals.add(interval)
        
        self.intervals = new_intervals
    
    def remove_interval(self, start, end):
        """Remove interval, adjusting or splitting existing intervals."""
        new_intervals = set()
        
        for interval in self.intervals:
            if interval[1] < start or interval[0] > end:  # No overlap
                new_intervals.add(interval)
            else:
                # Adjust or split interval
                if interval[0] < start:
                    new_intervals.add((interval[0], start - 1))
                if interval[1] > end:
                    new_intervals.add((end + 1, interval[1]))
        
        self.intervals = new_intervals
    
    def find_largest_interval(self):
        """Find and return the largest interval by length.
        
        Returns:
            Tuple (start, end) of largest interval, or None if no intervals exist.
        """
        if not self.intervals:
            return None
        return max(self.intervals, key=lambda x: x[1] - x[0])
    
    def __repr__(self):
        """String representation for debugging."""
        return f"IntervalManager(intervals={sorted(self.intervals)}, min={self.min_value}, max={self.max_value})"
