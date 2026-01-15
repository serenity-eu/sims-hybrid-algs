use fixedbitset::FixedBitSet;
use pls::ProblemBitset;
use pls::objectives::{ObjectiveState, ObjectiveType};

#[test]
fn test_sims_problem_bitset_basic() {
    // 3 images, 5 elements
    let images = [vec![0, 2, 4], vec![1, 3], vec![0, 1, 2, 3, 4]];
    let universe_size = 5;
    // Convert images to FixedBitSet
    let bitsets: Vec<FixedBitSet> = images
        .iter()
        .map(|idxs| {
            let mut fv = FixedBitSet::with_capacity(universe_size);
            for &idx in idxs {
                fv.insert(idx);
            }
            fv
        })
        .collect();

    // Build element_to_images mapping
    let mut element_to_images = vec![Vec::new(); universe_size];
    for (img_idx, indices) in images.iter().enumerate() {
        for &elem_idx in indices {
            element_to_images[elem_idx].push(img_idx);
        }
    }

    // Create dummy objectives for testing
    let objective_types = [ObjectiveType::TotalCost, ObjectiveType::CloudyArea];
    let objectives = [
        ObjectiveState::TotalCost {
            costs: vec![10, 20, 30],
            max_value: 60,
            bounds: None,
        },
        ObjectiveState::CloudyArea {
            clear_images: vec![
                FixedBitSet::with_capacity(universe_size),
                FixedBitSet::with_capacity(universe_size),
                FixedBitSet::with_capacity(universe_size),
            ],
            areas: vec![1, 1, 1, 1, 1],
            max_value: 5,
            bounds: None,
        },
    ];

    let pb = ProblemBitset::new(
        "test".to_string(),
        bitsets,
        universe_size,
        element_to_images,
        objectives,
        objective_types,
    );

    assert_eq!(pb.num_images(), 3);
    assert_eq!(pb.num_elements(), 5);
    assert_eq!(pb.instance_name, "test");

    // Check bitsets
    for (i, img_indices) in images.iter().enumerate() {
        for idx in 0..universe_size {
            let should_be_set = img_indices.contains(&idx);
            assert_eq!(
                pb.image_contains(i, idx),
                should_be_set,
                "Image {i} element {idx}"
            );
        }
    }
}
