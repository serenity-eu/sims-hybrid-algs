use pls::ProblemBitset;
use fixedbitset::FixedBitSet;

#[test]
fn test_sims_problem_bitset_basic() {
    // 3 images, 5 elements
    let images = [
        vec![0, 2, 4],
        vec![1, 3],
        vec![0, 1, 2, 3, 4],
    ];
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
    let pb = ProblemBitset::new(bitsets, universe_size);

    assert_eq!(pb.num_images(), 3);
    assert_eq!(pb.num_elements(), 5);

    // Check bitsets
    for (i, img_indices) in images.iter().enumerate() {
        let bits = pb.image(i);
        for idx in 0..universe_size {
            let should_be_set = img_indices.contains(&idx);
            assert_eq!(bits.contains(idx), should_be_set, "Image {i} element {idx}");
        }
    }
}