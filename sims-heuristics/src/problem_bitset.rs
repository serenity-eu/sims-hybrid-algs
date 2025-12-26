use fixedbitset::FixedBitSet;
use crate::problem::SIMSProblemInstanceRaw;
use std::hash::{Hash, Hasher};
use std::fmt;

/// `FixedVec` is a bit vector representing a set of elements.
pub type FixedVec = FixedBitSet;

    /// `SimsProblemBitset` stores all images as bitsets of elements.
#[derive(Clone)]
pub struct ProblemBitset {
    /// Each image is represented as a `FixedVec` (bitset) of elements it contains.
    pub images: Vec<FixedBitSet>,
    /// Number of elements in the universe.
    pub universe_size: usize,
}

impl ProblemBitset {
    /// Create a new `ProblemBitset` from a list of images, each as a `FixedBitSet`.
    #[must_use]
    pub const fn new(images: Vec<FixedBitSet>, universe_size: usize) -> Self {
        Self { images, universe_size }
    }

    /// Construct from `SIMSProblemInstanceRaw`, using its images and `universe_size`.
    #[must_use]
    pub fn from_raw(raw: &SIMSProblemInstanceRaw) -> Self {
        let images = raw.images
            .iter()
            .map(|indices| {
                let zero_based: Vec<usize> = indices.iter()
                    .filter(|&&i| i > 0 && i <= raw.universe_size)
                    .map(|&i| {
                        if i > raw.universe_size {
                            println!("Warning: index {} out of bounds for universe_size {}", i, raw.universe_size);
                        }
                        i - 1
                    })
                    .collect();
                let mut fv = FixedBitSet::with_capacity(raw.universe_size);
                for idx in zero_based {
                    fv.insert(idx);
                }
                fv
            })
            .collect();
        Self::new(images, raw.universe_size)
    }

    /// Get the image at the given index.
    #[must_use]
    pub fn image(&self, idx: usize) -> &FixedBitSet {
        &self.images[idx]
    }

    /// Number of images.
    #[must_use]
    pub const fn num_images(&self) -> usize {
        self.images.len()
    }

    /// Number of elements in the universe.
    #[must_use]
    pub const fn num_elements(&self) -> usize {
        self.universe_size
    }

    /// Iterator over all images.
    pub fn iter_images(&self) -> impl Iterator<Item = &FixedBitSet> {
        self.images.iter()
    }

    /// Iterator over all image indices.
    pub fn image_indices(&self) -> impl Iterator<Item = usize> {
        0..self.images.len()
    }
}

// PartialEq, Eq, Hash
impl PartialEq for ProblemBitset {
    fn eq(&self, other: &Self) -> bool {
        self.images == other.images && self.universe_size == other.universe_size
    }
}
impl Eq for ProblemBitset {}
impl Hash for ProblemBitset {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for img in &self.images {
            img.hash(state);
        }
        self.universe_size.hash(state);
    }
}

// Debug
impl fmt::Debug for ProblemBitset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProblemBitset")
            .field("num_images", &self.images.len())
            .field("universe_size", &self.universe_size)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_problem_bitset_basic() {
        // 3 images, 5 elements
        let image_indices = [
            vec![0, 2, 4],
            vec![1, 3],
            vec![0, 1, 2, 3, 4],
        ];
        let universe_size = 5;

        // Prepare bitsets directly
        let images: Vec<FixedBitSet> = image_indices
            .iter()
            .map(|idxs| {
                let mut fv = FixedBitSet::with_capacity(universe_size);
                for &idx in idxs {
                    fv.insert(idx);
                }
                fv
            })
            .collect();

        let pb = ProblemBitset::new(images.clone(), universe_size);

        assert_eq!(pb.num_images(), 3);
        assert_eq!(pb.num_elements(), 5);

        // Check bitsets
        for (i, img) in images.iter().enumerate() {
            let bits = pb.image(i);
            for idx in 0..universe_size {
                let should_be_set = img.contains(idx);
                assert_eq!(bits.contains(idx), should_be_set, "Image {i} element {idx}");
            }
        }
    }
}