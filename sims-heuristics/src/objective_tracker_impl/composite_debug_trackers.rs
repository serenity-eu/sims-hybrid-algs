//! Composite debug trackers that run two implementations in parallel and assert equality.
//!
//! This module provides a wrapper that holds two tracker implementations (e.g., Standard and Simd),
//! forwards all operations to both, and asserts that results match. Useful for validation and debugging.

use crate::objective_tracker::TrackerCollection;
use crate::objective_tracker_impl::simd_trackers::SimdTrackerArray;
use crate::objective_tracker_impl::simplified_trackers::SimpleTrackerArray;
use crate::objective_tracker_impl::standard_trackers::{StandardTracker, StandardTrackerArray};
use crate::problem::SetCoverProblem;
use crate::solution::ImageSet;

/// A composite tracker that runs StandardTrackerArray and SimdTrackerArray in parallel,
/// asserting that all results match.
#[derive(Clone, Debug)]
pub struct CompositeDebugTrackerArray<const D: usize> {
    standard: StandardTrackerArray<D>,
    simd: SimdTrackerArray<D>,
}

impl<const D: usize> TrackerCollection<D> for CompositeDebugTrackerArray<D> {
    type Tracker = StandardTracker;

    fn get(&self, index: usize) -> &Self::Tracker {
        self.standard.get(index)
    }

    fn get_mut(&mut self, index: usize) -> &mut Self::Tracker {
        self.standard.get_mut(index)
    }

    fn new(problem: &impl SetCoverProblem<D>) -> Self {
        Self {
            standard: StandardTrackerArray::new(problem),
            simd: SimdTrackerArray::new(problem),
        }
    }

    fn initial_objectives(&self) -> [u64; D] {
        let standard_result = self.standard.initial_objectives();
        let simd_result = self.simd.initial_objectives();
        assert_eq!(
            standard_result, simd_result,
            "initial_objectives mismatch: standard={standard_result:?}, simd={simd_result:?}"
        );
        standard_result
    }

    fn peek_removal_delta(
        &self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
        solution: &impl ImageSet<D>,
    ) -> [i64; D] {
        let standard_result = self.standard.peek_removal_delta(image_index, problem, solution);
        let simd_result = self.simd.peek_removal_delta(image_index, problem, solution);
        assert_eq!(
            standard_result, simd_result,
            "peek_removal_delta mismatch for image {image_index}: standard={standard_result:?}, simd={simd_result:?}"
        );
        standard_result
    }

    fn peek_addition_delta(
        &self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
        solution: &impl ImageSet<D>,
    ) -> [i64; D] {
        let standard_result = self.standard.peek_addition_delta(image_index, problem, solution);
        let simd_result = self.simd.peek_addition_delta(image_index, problem, solution);
        assert_eq!(
            standard_result, simd_result,
            "peek_addition_delta mismatch for image {image_index}: standard={standard_result:?}, simd={simd_result:?}"
        );
        standard_result
    }

    fn track_image_removal(
        &mut self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
    ) -> [i64; D] {
        let standard_result = self.standard.track_image_removal(image_index, problem);
        let simd_result = self.simd.track_image_removal(image_index, problem);
        assert_eq!(
            standard_result, simd_result,
            "track_image_removal mismatch for image {image_index}: standard={standard_result:?}, simd={simd_result:?}"
        );
        standard_result
    }

    fn track_image_addition(
        &mut self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
    ) -> [i64; D] {
        let standard_result = self.standard.track_image_addition(image_index, problem);
        let simd_result = self.simd.track_image_addition(image_index, problem);
        assert_eq!(
            standard_result, simd_result,
            "track_image_addition mismatch for image {image_index}: standard={standard_result:?}, simd={simd_result:?}"
        );
        standard_result
    }

    fn values(&self) -> [u64; D] {
        let standard_result = self.standard.values();
        let simd_result = self.simd.values();
        assert_eq!(
            standard_result, simd_result,
            "values mismatch: standard={standard_result:?}, simd={simd_result:?}"
        );
        standard_result
    }

    fn initialize_from(&mut self, solution: &impl ImageSet<D>, problem: &impl SetCoverProblem<D>) {
        self.standard.initialize_from(solution, problem);
        self.simd.initialize_from(solution, problem);

        // Verify state after initialization
        let standard_values = self.standard.values();
        let simd_values = self.simd.values();
        assert_eq!(
            standard_values, simd_values,
            "initialize_from values mismatch: standard={standard_values:?}, simd={simd_values:?}"
        );
    }
}

/// A composite tracker that runs StandardTrackerArray and SimpleTrackerArray in parallel,
/// asserting that all results match.
#[derive(Clone, Debug)]
pub struct StandardSimpleDebugTrackerArray<const D: usize> {
    standard: StandardTrackerArray<D>,
    simple: SimpleTrackerArray<D>,
}

impl<const D: usize> TrackerCollection<D> for StandardSimpleDebugTrackerArray<D> {
    type Tracker = StandardTracker;

    fn get(&self, index: usize) -> &Self::Tracker {
        self.standard.get(index)
    }

    fn get_mut(&mut self, index: usize) -> &mut Self::Tracker {
        self.standard.get_mut(index)
    }

    fn new(problem: &impl SetCoverProblem<D>) -> Self {
        Self {
            standard: StandardTrackerArray::new(problem),
            simple: SimpleTrackerArray::new(problem),
        }
    }

    fn initial_objectives(&self) -> [u64; D] {
        let standard_result = self.standard.initial_objectives();
        let simple_result = self.simple.initial_objectives();
        assert_eq!(
            standard_result, simple_result,
            "initial_objectives mismatch: standard={standard_result:?}, simple={simple_result:?}"
        );
        standard_result
    }

    fn peek_removal_delta(
        &self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
        solution: &impl ImageSet<D>,
    ) -> [i64; D] {
        let standard_result = self.standard.peek_removal_delta(image_index, problem, solution);
        let simple_result = self.simple.peek_removal_delta(image_index, problem, solution);
        assert_eq!(
            standard_result, simple_result,
            "peek_removal_delta mismatch for image {image_index}: standard={standard_result:?}, simple={simple_result:?}"
        );
        standard_result
    }

    fn peek_addition_delta(
        &self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
        solution: &impl ImageSet<D>,
    ) -> [i64; D] {
        let standard_result = self.standard.peek_addition_delta(image_index, problem, solution);
        let simple_result = self.simple.peek_addition_delta(image_index, problem, solution);
        assert_eq!(
            standard_result, simple_result,
            "peek_addition_delta mismatch for image {image_index}: standard={standard_result:?}, simple={simple_result:?}"
        );
        standard_result
    }

    fn track_image_removal(
        &mut self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
    ) -> [i64; D] {
        let standard_result = self.standard.track_image_removal(image_index, problem);
        let simple_result = self.simple.track_image_removal(image_index, problem);
        assert_eq!(
            standard_result, simple_result,
            "track_image_removal mismatch for image {image_index}: standard={standard_result:?}, simple={simple_result:?}"
        );
        standard_result
    }

    fn track_image_addition(
        &mut self,
        image_index: usize,
        problem: &impl SetCoverProblem<D>,
    ) -> [i64; D] {
        let standard_result = self.standard.track_image_addition(image_index, problem);
        let simple_result = self.simple.track_image_addition(image_index, problem);
        assert_eq!(
            standard_result, simple_result,
            "track_image_addition mismatch for image {image_index}: standard={standard_result:?}, simple={simple_result:?}"
        );
        standard_result
    }

    fn values(&self) -> [u64; D] {
        let standard_result = self.standard.values();
        let simple_result = self.simple.values();
        assert_eq!(
            standard_result, simple_result,
            "values mismatch: standard={standard_result:?}, simple={simple_result:?}"
        );
        standard_result
    }

    fn initialize_from(&mut self, solution: &impl ImageSet<D>, problem: &impl SetCoverProblem<D>) {
        self.standard.initialize_from(solution, problem);
        self.simple.initialize_from(solution, problem);

        // Verify state after initialization
        let standard_values = self.standard.values();
        let simple_values = self.simple.values();
        assert_eq!(
            standard_values, simple_values,
            "initialize_from values mismatch: standard={standard_values:?}, simple={simple_values:?}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objectives::ObjectiveType;
    use crate::problem_bitset::ProblemBitset;
    use std::path::Path;

    const OBJECTIVE_TYPES: [ObjectiveType; 4] = [
        ObjectiveType::TotalCost,
        ObjectiveType::CloudyArea,
        ObjectiveType::MinResolution,
        ObjectiveType::MaxIncidenceAngle,
    ];

    #[test]
    fn test_composite_standard_simd_consistency() {
        let instance_path = Path::new("data").join("lagos_nigeria_30.dzn");
        let problem = ProblemBitset::<4>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES)
            .expect("Failed to load problem");

        let mut trackers = CompositeDebugTrackerArray::<4>::new(&problem);

        // Test some operations
        for i in 0..10 {
            trackers.track_image_addition(i, &problem);
        }

        for i in 0..5 {
            trackers.track_image_removal(i, &problem);
        }

        // If we get here without panic, the implementations match
    }

    #[test]
    fn test_composite_standard_simple_consistency() {
        let instance_path = Path::new("data").join("lagos_nigeria_30.dzn");
        let problem = ProblemBitset::<4>::from_minizinc_datafile(&instance_path, OBJECTIVE_TYPES)
            .expect("Failed to load problem");

        let mut trackers = StandardSimpleDebugTrackerArray::<4>::new(&problem);

        // Test some operations
        for i in 0..10 {
            trackers.track_image_addition(i, &problem);
        }

        for i in 0..5 {
            trackers.track_image_removal(i, &problem);
        }

        // If we get here without panic, the implementations match
    }
}
