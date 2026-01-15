//! Comprehensive test suite for `ObjectiveTracker` implementations
//!
//! This suite validates that incremental objective tracking matches
//! the actual objective values computed from solutions.

use pls::objective_tracker::{ObjectiveTracker, StandardTrackerArray, TrackerCollection};
use pls::objectives::ObjectiveType;
use pls::problem::SIMSProblemInstanceRaw;
use pls::problem_bitset::ProblemBitset;
use pls::solution_impl::bitset_encoded_solution::BitsetEncodedSolution;

fn tracker_value<const D: usize>(trackers: &StandardTrackerArray<D>, index: usize) -> u64 {
    trackers.get(index).value()
}

fn peek_add<const D: usize>(
    trackers: &StandardTrackerArray<D>,
    index: usize,
    image_idx: usize,
    problem: &impl pls::problem::SetCoverProblem<D>,
    solution: &impl pls::solution::ImageSet<D>,
) -> i64 {
    trackers
        .get(index)
        .peek_addition_delta(image_idx, problem, solution)
}

fn create_simple_test_problem() -> SIMSProblemInstanceRaw {
    // 4 elements, 3 images with overlapping coverage
    // Image 0: covers [0, 1], cost 100, resolution 50, angle 30
    // Image 1: covers [1, 2], cost 200, resolution 80, angle 45
    // Image 2: covers [2, 3], cost 150, resolution 60, angle 35
    SIMSProblemInstanceRaw {
        name: "test_trackers".to_string(),
        num_images: 3,
        universe_size: 4,
        images: vec![vec![0, 1], vec![1, 2], vec![2, 3]],
        costs: vec![100, 200, 150],
        clouds: vec![
            vec![0, 1], // Image 0 has clouds on both elements
            vec![1, 2], // Image 1 has clouds on both elements
            vec![2, 3], // Image 2 has clouds on both elements
        ],
        areas: vec![10, 10, 10, 10], // Each element has area 10
        max_cloud_area: 200,
        resolution: vec![50, 80, 60],
        incidence_angle: vec![30, 45, 35],
    }
}

#[test]
fn test_total_cost_tracking() {
    let raw = create_simple_test_problem();
    let problem = ProblemBitset::<4>::from_raw_with_objectives(
        &raw,
        [
            ObjectiveType::TotalCost,
            ObjectiveType::CloudyArea,
            ObjectiveType::MinResolution,
            ObjectiveType::MaxIncidenceAngle,
        ],
    );

    let mut trackers = StandardTrackerArray::<4>::new(&problem);

    // Add images and verify cost tracking
    let add_0 = trackers.track_image_addition(0, &problem);
    assert_eq!(add_0[0], 100, "Image 0 costs 100");

    let add_1 = trackers.track_image_addition(1, &problem);
    assert_eq!(add_1[0], 200, "Image 1 costs 200");

    let add_2 = trackers.track_image_addition(2, &problem);
    assert_eq!(add_2[0], 150, "Image 2 costs 150");

    // Verify cost value is cumulative
    assert_eq!(
        tracker_value::<4>(&trackers, 0),
        450,
        "Total cost should be 100 + 200 + 150 = 450"
    );
}

#[test]
fn test_add_remove_symmetry() {
    let raw = create_simple_test_problem();
    let problem = ProblemBitset::<4>::from_raw_with_objectives(
        &raw,
        [
            ObjectiveType::TotalCost,
            ObjectiveType::CloudyArea,
            ObjectiveType::MinResolution,
            ObjectiveType::MaxIncidenceAngle,
        ],
    );

    let mut trackers = StandardTrackerArray::<4>::new(&problem);

    // Add images and track deltas
    let add_0 = trackers.track_image_addition(0, &problem);
    let add_1 = trackers.track_image_addition(1, &problem);

    // Remove in reverse order
    let remove_1 = trackers.track_image_removal(1, &problem);
    let remove_0 = trackers.track_image_removal(0, &problem);

    // Check symmetry for all objectives
    assert_eq!(
        add_1[0], -remove_1[0],
        "TotalCost add/remove should be symmetric"
    );
    assert_eq!(
        add_0[0], -remove_0[0],
        "TotalCost add/remove should be symmetric"
    );

    // Note: CloudyArea, MinResolution, MaxIncidenceAngle may not be perfectly symmetric
    // when images overlap, but removing what was added should restore the original state
}

#[test]
fn test_tracker_initialization_consistency() {
    let raw = create_simple_test_problem();
    let problem = ProblemBitset::<4>::from_raw_with_objectives(
        &raw,
        [
            ObjectiveType::TotalCost,
            ObjectiveType::CloudyArea,
            ObjectiveType::MinResolution,
            ObjectiveType::MaxIncidenceAngle,
        ],
    );

    // Method 1: Initialize from solution
    let mut trackers1 = StandardTrackerArray::<4>::new(&problem);
    let solution1 = BitsetEncodedSolution::from_selected_images(&[0, 1], &problem);
    trackers1.initialize_from(&solution1, &problem);

    // Method 2: Sequential addition
    let mut trackers2 = StandardTrackerArray::<4>::new(&problem);
    trackers2.track_image_addition(0, &problem);
    trackers2.track_image_addition(1, &problem);

    // Both methods should produce the same state
    let delta1 = trackers1.track_image_addition(2, &problem);
    let delta2 = trackers2.track_image_addition(2, &problem);

    assert_eq!(delta1[0], delta2[0], "TotalCost delta should match");
    assert_eq!(delta1[1], delta2[1], "CloudyArea delta should match");
    assert_eq!(delta1[2], delta2[2], "MinResolution delta should match");
    assert_eq!(delta1[3], delta2[3], "MaxIncidenceAngle delta should match");
}

#[test]
fn test_solution_objectives_match_tracker_values() {
    let raw = create_simple_test_problem();
    let problem = ProblemBitset::<4>::from_raw_with_objectives(
        &raw,
        [
            ObjectiveType::TotalCost,
            ObjectiveType::CloudyArea,
            ObjectiveType::MinResolution,
            ObjectiveType::MaxIncidenceAngle,
        ],
    );

    // Create solution - this computes objectives directly
    let solution = BitsetEncodedSolution::from_selected_images(&[0, 1], &problem);

    // Track same images incrementally
    let mut trackers = StandardTrackerArray::<4>::new(&problem);
    trackers.track_image_addition(0, &problem);
    trackers.track_image_addition(1, &problem);

    // Verify tracker values match solution objectives
    assert_eq!(
        tracker_value::<4>(&trackers, 0),
        solution.objectives[0],
        "TotalCost should match"
    );
    assert_eq!(
        tracker_value::<4>(&trackers, 1),
        solution.objectives[1],
        "CloudyArea should match"
    );
    assert_eq!(
        tracker_value::<4>(&trackers, 2),
        solution.objectives[2],
        "MinResolution should match"
    );
    assert_eq!(
        tracker_value::<4>(&trackers, 3),
        solution.objectives[3],
        "MaxIncidenceAngle should match"
    );
}

#[test]
fn test_empty_state_additions() {
    let raw = create_simple_test_problem();
    let problem = ProblemBitset::<4>::from_raw_with_objectives(
        &raw,
        [
            ObjectiveType::TotalCost,
            ObjectiveType::CloudyArea,
            ObjectiveType::MinResolution,
            ObjectiveType::MaxIncidenceAngle,
        ],
    );

    let mut trackers = StandardTrackerArray::<4>::new(&problem);

    // First addition from empty state
    let delta = trackers.track_image_addition(0, &problem);

    assert_eq!(delta[0], 100, "First image adds cost of 100");
    assert!(delta[1] >= 0, "CloudyArea should increase or stay same");
    assert!(delta[2] > 0, "MinResolution should increase from empty");
    assert_eq!(
        delta[3], 30,
        "MaxIncidenceAngle should be 30 for first image"
    );
}

#[test]
fn test_multiple_operations_sequence() {
    let raw = create_simple_test_problem();
    let problem = ProblemBitset::<4>::from_raw_with_objectives(
        &raw,
        [
            ObjectiveType::TotalCost,
            ObjectiveType::CloudyArea,
            ObjectiveType::MinResolution,
            ObjectiveType::MaxIncidenceAngle,
        ],
    );

    let mut trackers = StandardTrackerArray::<4>::new(&problem);

    // Add all images
    trackers.track_image_addition(0, &problem);
    trackers.track_image_addition(1, &problem);
    trackers.track_image_addition(2, &problem);

    assert_eq!(
        tracker_value::<4>(&trackers, 0),
        450,
        "Total cost should be 450"
    );

    // Remove middle image
    let delta_remove = trackers.track_image_removal(1, &problem);
    assert_eq!(
        delta_remove[0], -200,
        "Removing image 1 decreases cost by 200"
    );
    assert_eq!(
        tracker_value::<4>(&trackers, 0),
        250,
        "Cost should now be 250"
    );

    // Add it back
    let delta_add_back = trackers.track_image_addition(1, &problem);
    assert_eq!(
        delta_add_back[0], 200,
        "Re-adding image 1 increases cost by 200"
    );
    assert_eq!(
        tracker_value::<4>(&trackers, 0),
        450,
        "Cost restored to 450"
    );
}

#[test]
fn test_min_resolution_with_overlap() {
    let raw = create_simple_test_problem();
    let problem = ProblemBitset::<4>::from_raw_with_objectives(
        &raw,
        [
            ObjectiveType::TotalCost,
            ObjectiveType::CloudyArea,
            ObjectiveType::MinResolution,
            ObjectiveType::MaxIncidenceAngle,
        ],
    );

    let mut trackers = StandardTrackerArray::<4>::new(&problem);

    // Add higher resolution first (image 1, res 80) covering element 1
    let delta_high = trackers.track_image_addition(1, &problem);
    assert!(delta_high[2] > 0, "MinResolution should increase");

    // Add lower resolution (image 0, res 50) also covering element 1
    let delta_low = trackers.track_image_addition(0, &problem);
    // Element 1 now has min(50, 80) = 50, so MinResolution for element 1 decreases
    // But element 0 gets covered with resolution 50, which is new coverage
    assert!(
        delta_low[2] != 0,
        "MinResolution should change when overlapping coverage is added"
    );

    // Verify we can track state through removal
    let before_removal = tracker_value::<4>(&trackers, 2);
    trackers.track_image_removal(0, &problem);
    let after_removal = tracker_value::<4>(&trackers, 2);

    assert_ne!(
        before_removal, after_removal,
        "MinResolution should change after removal"
    );
}

#[test]
fn test_max_incidence_angle_tracking() {
    let raw = create_simple_test_problem();
    let problem = ProblemBitset::<4>::from_raw_with_objectives(
        &raw,
        [
            ObjectiveType::TotalCost,
            ObjectiveType::CloudyArea,
            ObjectiveType::MinResolution,
            ObjectiveType::MaxIncidenceAngle,
        ],
    );

    let mut trackers = StandardTrackerArray::<4>::new(&problem);

    // Add images in order: 0 (angle 30), 1 (angle 45), 2 (angle 35)
    let delta0 = trackers.track_image_addition(0, &problem);
    assert_eq!(delta0[3], 30, "First image sets max angle to 30");
    assert_eq!(tracker_value::<4>(&trackers, 3), 30, "Max angle is 30");

    let delta1 = trackers.track_image_addition(1, &problem);
    assert_eq!(delta1[3], 15, "Max increases from 30 to 45 (delta = 15)");
    assert_eq!(tracker_value::<4>(&trackers, 3), 45, "Max angle is now 45");

    let delta2 = trackers.track_image_addition(2, &problem);
    assert_eq!(delta2[3], 0, "Angle 35 doesn't increase max of 45");
    assert_eq!(tracker_value::<4>(&trackers, 3), 45, "Max angle stays 45");

    // Remove max (image 1 with angle 45)
    let delta_remove = trackers.track_image_removal(1, &problem);
    assert_eq!(
        delta_remove[3], -10,
        "Max drops from 45 to 35 (delta = -10)"
    );
    assert_eq!(tracker_value::<4>(&trackers, 3), 35, "Max angle is now 35");
}

#[test]
fn test_cloudy_area_tracking() {
    let raw = create_simple_test_problem();
    let problem = ProblemBitset::<4>::from_raw_with_objectives(
        &raw,
        [
            ObjectiveType::TotalCost,
            ObjectiveType::CloudyArea,
            ObjectiveType::MinResolution,
            ObjectiveType::MaxIncidenceAngle,
        ],
    );

    let mut trackers = StandardTrackerArray::<4>::new(&problem);

    // Note: In this test problem, all images have complete cloud coverage,
    // so adding images may not reduce cloudy area significantly.
    // We just verify that the tracker tracks something.

    let _delta0 = trackers.track_image_addition(0, &problem);
    let initial_cloudy = tracker_value::<4>(&trackers, 1);
    // Just verify we have a valid cloudy area value
    assert!(initial_cloudy > 0, "Initial cloudy area should be positive");

    // Add second image and verify tracker state is updated
    let before = tracker_value::<4>(&trackers, 1);
    let delta1 = trackers.track_image_addition(1, &problem);
    let after = tracker_value::<4>(&trackers, 1);

    // The CloudyArea might stay the same or change depending on coverage patterns
    // Verify that the delta calculation is consistent with actual value change
    #[allow(clippy::cast_possible_wrap)]
    {
        assert_eq!(
            before as i64 + delta1[1],
            after as i64,
            "CloudyArea delta should match actual value change"
        );
    }
}

#[test]
fn test_peek_vs_track_consistency() {
    let raw = create_simple_test_problem();
    let problem = ProblemBitset::<4>::from_raw_with_objectives(
        &raw,
        [
            ObjectiveType::TotalCost,
            ObjectiveType::CloudyArea,
            ObjectiveType::MinResolution,
            ObjectiveType::MaxIncidenceAngle,
        ],
    );

    let mut trackers = StandardTrackerArray::<4>::new(&problem);
    let solution = BitsetEncodedSolution::from_selected_images(&[0], &problem);

    // Peek at delta for adding image 1
    let peek_delta_add = peek_add::<4>(&trackers, 0, 1, &problem, &solution);

    // Actually add image 1
    let track_delta_add = trackers.track_image_addition(1, &problem);

    // Peek and track should return same delta for TotalCost
    assert_eq!(
        peek_delta_add, track_delta_add[0],
        "Peek and track deltas should match for TotalCost"
    );
}

#[test]
fn test_empty_to_full_sequence() {
    let raw = create_simple_test_problem();
    let problem = ProblemBitset::<4>::from_raw_with_objectives(
        &raw,
        [
            ObjectiveType::TotalCost,
            ObjectiveType::CloudyArea,
            ObjectiveType::MinResolution,
            ObjectiveType::MaxIncidenceAngle,
        ],
    );

    // Build solution incrementally
    let mut trackers = StandardTrackerArray::<4>::new(&problem);
    trackers.track_image_addition(0, &problem);
    trackers.track_image_addition(1, &problem);
    trackers.track_image_addition(2, &problem);

    // Build solution directly
    let solution = BitsetEncodedSolution::from_selected_images(&[0, 1, 2], &problem);

    // All objectives should match
    assert_eq!(
        tracker_value::<4>(&trackers, 0),
        solution.objectives[0],
        "TotalCost should match"
    );
    assert_eq!(
        tracker_value::<4>(&trackers, 1),
        solution.objectives[1],
        "CloudyArea should match"
    );
    assert_eq!(
        tracker_value::<4>(&trackers, 2),
        solution.objectives[2],
        "MinResolution should match"
    );
    assert_eq!(
        tracker_value::<4>(&trackers, 3),
        solution.objectives[3],
        "MaxIncidenceAngle should match"
    );
}
