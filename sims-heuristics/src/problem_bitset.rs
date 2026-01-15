use crate::objectives::{ObjectiveState, ObjectiveType};
use crate::problem::{SIMSProblemInstanceRaw, SetCoverProblem};
use crate::solution::ImageSet;
use fixedbitset::FixedBitSet;
use std::path::Path;

/// `ProblemBitset` stores problem data using bitsets for efficient set operations.
///
/// This is a non-generic alternative to `Problem<T, D>` that uses bitsets throughout.
/// Data that's objective-specific (costs, areas, `clear_images`) is stored in `ObjectiveState`.
#[derive(Clone, Debug)]
pub struct ProblemBitset<const D: usize> {
    /// Name of the problem instance
    pub instance_name: String,
    /// Each image is represented as a `FixedBitSet` of elements it contains.
    pub images: Vec<FixedBitSet>,
    /// Number of elements in the universe.
    pub universe_size: usize,
    /// Which images cover each element (inverted index for efficient lookups)
    pub element_to_images: Vec<Vec<usize>>,
    /// Objective states containing data and logic (includes costs, areas, `clear_images`, etc.)
    pub objectives: [ObjectiveState<D>; D],
    /// Objective types for metadata
    pub objective_types: [ObjectiveType; D],
}

impl<const D: usize> ProblemBitset<D> {
    /// Create a new `ProblemBitset` from a list of images, each as a `FixedBitSet`.
    #[must_use]
    pub const fn new(
        instance_name: String,
        images: Vec<FixedBitSet>,
        universe_size: usize,
        element_to_images: Vec<Vec<usize>>,
        objectives: [ObjectiveState<D>; D],
        objective_types: [ObjectiveType; D],
    ) -> Self {
        Self {
            instance_name,
            images,
            universe_size,
            element_to_images,
            objectives,
            objective_types,
        }
    }

    /// Construct from `SIMSProblemInstanceRaw` with specified objectives.
    ///
    /// # Note
    /// This method expects `raw` to already have normalized (0-based) indices.
    /// If loading from a `MiniZinc` file, use `SIMSProblemInstanceRaw::from_minizinc_datafile`
    /// which handles normalization during reading.
    #[must_use]
    pub fn from_raw_with_objectives(
        raw: &SIMSProblemInstanceRaw,
        objective_types: [ObjectiveType; D],
    ) -> Self {
        // Convert image indices to bitsets (already 0-based)
        let images: Vec<FixedBitSet> = raw
            .images
            .iter()
            .map(|indices| indices.iter().copied().collect())
            .collect();

        // Build inverted index: which images cover each element
        let mut element_to_images = vec![Vec::new(); raw.universe_size];
        for (image_idx, image) in raw.images.iter().enumerate() {
            for &element_idx in image {
                element_to_images[element_idx].push(image_idx);
            }
        }

        // Create objective states from raw data (already normalized)
        let objectives = ObjectiveState::from_types_and_raw(&objective_types, raw);

        Self::new(
            raw.name.clone(),
            images,
            raw.universe_size,
            element_to_images,
            objectives,
            objective_types,
        )
    }

    /// Load problem instance from minizinc data file.
    ///
    /// # Errors
    ///
    /// Returns an error message as String if the file cannot be read or parsed.
    pub fn from_minizinc_datafile<P: AsRef<Path>>(
        model_path: P,
        objective_types: [ObjectiveType; D],
    ) -> Result<Self, String> {
        let raw = SIMSProblemInstanceRaw::from_minizinc_datafile(model_path);
        Ok(Self::from_raw_with_objectives(&raw, objective_types))
    }

    /// Check if an element is contained in an image.
    #[must_use]
    pub fn image_contains(&self, image_idx: usize, element_idx: usize) -> bool {
        self.images[image_idx].contains(element_idx)
    }

    /// Get an iterator over all element indices contained in an image.
    pub fn iter_image_elements(&self, image_idx: usize) -> impl Iterator<Item = usize> + '_ {
        self.images[image_idx].ones()
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

    /// Iterator over all image indices.
    pub fn image_indices(&self) -> impl Iterator<Item = usize> {
        0..self.images.len()
    }

    /// Get max objectives array from objective states
    #[must_use]
    pub fn max_objectives(&self) -> [u64; D] {
        std::array::from_fn(|i| self.objectives[i].max_value())
    }

    /// Get objective bounds if set
    ///
    /// # Panics
    /// Panics if bounds are partially set (some objectives have bounds while others don't).
    #[must_use]
    pub fn objective_bounds(&self) -> Option<[[u64; 2]; D]> {
        if self.objectives.iter().all(|obj| obj.bounds().is_some()) {
            Some(std::array::from_fn(|i| {
                self.objectives[i].bounds().unwrap()
            }))
        } else {
            None
        }
    }

    /// Set objective bounds for normalization in scalarization
    pub fn set_objective_bounds(&mut self, bounds: [[u64; 2]; D]) {
        self.objectives
            .iter_mut()
            .zip(bounds)
            .for_each(|(obj, bound)| obj.set_bounds(bound));
    }

    /// Get objective by index
    #[must_use]
    pub const fn objective(&self, index: usize) -> &ObjectiveState<D> {
        &self.objectives[index]
    }

    #[must_use]
    pub const fn objective_type(&self, index: usize) -> ObjectiveType {
        self.objective_types[index]
    }

    /// Compute the overlap between two images (number of shared elements)
    #[must_use]
    pub fn overlap(&self, image_i: usize, image_j: usize) -> usize {
        self.images[image_i].intersection_count(&self.images[image_j])
    }

    /// Get number of objectives
    #[must_use]
    pub const fn num_objectives(&self) -> usize {
        D
    }

    /// Get all objective names
    #[must_use]
    pub fn objective_names(&self) -> Vec<&str> {
        self.objective_types
            .iter()
            .map(ObjectiveType::name)
            .collect()
    }

    /// Get objective by ID
    #[must_use]
    pub fn objective_type_by_id(&self, id: &str) -> Option<ObjectiveType> {
        self.objective_types
            .iter()
            .copied()
            .find(|obj| obj.id() == id)
    }

    /// Get iterator over elements in an image (replaces problem.images[i].parts)
    pub fn image_elements(&self, image_idx: usize) -> impl Iterator<Item = usize> + '_ {
        self.images[image_idx].ones()
    }

    /// Get cost of an image. Data comes from `ObjectiveState::TotalCost`
    #[must_use]
    pub fn image_cost(&self, image_idx: usize) -> u64 {
        self.objectives
            .iter()
            .find_map(|obj| {
                if let ObjectiveState::TotalCost { costs, .. } = obj {
                    Some(costs[image_idx])
                } else {
                    None
                }
            })
            .unwrap_or(0)
    }

    /// Get which images cover a given element (replaces problem.universe[`element_idx`].images)
    #[must_use]
    pub fn element_covering_images(&self, element_idx: usize) -> &[usize] {
        &self.element_to_images[element_idx]
    }

    /// Get area of an element. Data comes from `ObjectiveState::CloudyArea`
    #[must_use]
    pub fn element_area(&self, element_idx: usize) -> u64 {
        self.objectives
            .iter()
            .find_map(|obj| {
                if let ObjectiveState::CloudyArea { areas, .. } = obj {
                    Some(areas[element_idx])
                } else {
                    None
                }
            })
            .unwrap_or(0)
    }
}

// Implement SetCoverProblem trait for ProblemBitset
impl<const D: usize> SetCoverProblem<D> for ProblemBitset<D> {
    fn is_set_cover(&self, solution: &impl ImageSet<D>) -> bool {
        let mut covered_elements = FixedBitSet::with_capacity(self.universe_size);

        // Mark all elements covered by selected images
        for image_index in solution.selected_images() {
            covered_elements.union_with(&self.images[image_index]);
        }

        // Check if all elements are covered (all bits set)
        covered_elements.is_full()
    }

    fn objective(&self, index: usize) -> &ObjectiveState<D> {
        &self.objectives[index]
    }

    fn objectives(&self) -> &[ObjectiveState<D>; D] {
        &self.objectives
    }

    fn max_objectives(&self) -> [u64; D] {
        std::array::from_fn(|i| self.objectives[i].max_value())
    }

    fn objective_bounds(&self) -> Option<[[u64; 2]; D]> {
        if self.objectives.iter().all(|obj| obj.bounds().is_some()) {
            Some(std::array::from_fn(|i| {
                self.objectives[i].bounds().unwrap()
            }))
        } else {
            None
        }
    }

    fn num_images(&self) -> usize {
        self.images.len()
    }

    fn num_elements(&self) -> usize {
        self.universe_size
    }

    fn image_contains_element(&self, image_index: usize, element_index: usize) -> bool {
        self.images[image_index].contains(element_index)
    }

    fn image_elements(&self, image_index: usize) -> impl Iterator<Item = usize> + '_ {
        self.images[image_index].ones()
    }

    // fn image_cost(&self, image_index: usize) -> u64 {
    //     // Delegate to existing method
    //     self.image_cost(image_index)
    // }

    // fn element_area(&self, element_index: usize) -> u64 {
    //     // Delegate to existing method
    //     self.element_area(element_index)
    // }

    fn overlap(&self, image_i: usize, image_j: usize) -> usize {
        // Delegate to existing method
        self.overlap(image_i, image_j)
    }

    fn objective_types(&self) -> &[ObjectiveType; D] {
        &self.objective_types
    }

    fn instance_name(&self) -> &str {
        &self.instance_name
    }

    fn element_images(&self, element_index: usize) -> impl Iterator<Item = usize> + '_ {
        self.element_to_images[element_index].iter().copied()
    }

    // fn image_clear_parts(&self, image_index: usize) -> impl Iterator<Item = usize> + '_ {
    //     // Return elements that are in the image (bitset ones)
    //     self.images[image_index].ones()
    // }

    // fn image_resolution(&self, _image_index: usize) -> u64 {
    //     // ProblemBitset doesn't store resolution data, return default
    //     0
    // }

    // fn image_incidence_angle(&self, _image_index: usize) -> u64 {
    //     // ProblemBitset doesn't store incidence angle data, return default
    //     0
    // }

    /// Optimized implementation using bitset operations
    fn uncovered_elements<'a, I>(&'a self, selected_images: I) -> impl Iterator<Item = usize> + 'a
    where
        I: Iterator<Item = usize> + 'a,
    {
        // Build covered elements bitset using union operations
        let mut covered = FixedBitSet::with_capacity(self.universe_size);
        for image_index in selected_images {
            covered.union_with(&self.images[image_index]);
        }

        // Toggle to get uncovered, then return consuming ones iterator
        covered.toggle_range(..);
        covered.into_ones()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objectives::ObjectiveType;

    #[test]
    fn test_problem_bitset_basic() {
        // 3 images, 5 elements
        let image_indices = [vec![0, 2, 4], vec![1, 3], vec![0, 1, 2, 3, 4]];
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

        // Build element_to_images mapping
        let mut element_to_images = vec![Vec::new(); universe_size];
        for (img_idx, indices) in image_indices.iter().enumerate() {
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
            images.clone(),
            universe_size,
            element_to_images,
            objectives,
            objective_types,
        );

        assert_eq!(pb.num_images(), 3);
        assert_eq!(pb.num_elements(), 5);
        assert_eq!(pb.instance_name, "test");

        // Check bitsets
        for (i, img) in images.iter().enumerate() {
            for idx in 0..universe_size {
                let should_be_set = img.contains(idx);
                assert_eq!(
                    pb.image_contains(i, idx),
                    should_be_set,
                    "Image {i} element {idx}"
                );
            }
        }
    }

    #[test]
    fn test_from_raw_with_objectives() {
        // Create a simple raw problem instance with 0-based indices (normalized)
        let raw = SIMSProblemInstanceRaw {
            name: "test".to_string(),
            num_images: 2,
            universe_size: 3,
            images: vec![vec![0, 1], vec![1, 2]],
            costs: vec![10, 20],
            clouds: vec![vec![], vec![2]],
            areas: vec![1, 1, 1],
            max_cloud_area: 0,
            resolution: vec![100, 200],
            incidence_angle: vec![5, 10],
        };

        let objective_types = [ObjectiveType::TotalCost, ObjectiveType::CloudyArea];
        let pb = ProblemBitset::from_raw_with_objectives(&raw, objective_types);

        // Verify basic structure
        assert_eq!(pb.num_images(), 2);
        assert_eq!(pb.num_elements(), 3);

        // Verify images (already 0-based)
        assert!(pb.image_contains(0, 0));
        assert!(pb.image_contains(0, 1));
        assert!(!pb.image_contains(0, 2));

        assert!(pb.image_contains(1, 1));
        assert!(pb.image_contains(1, 2));
        assert!(!pb.image_contains(1, 0));

        // Verify objectives were created
        assert_eq!(pb.objective_types.len(), 2);
        assert_eq!(pb.objective_types[0], ObjectiveType::TotalCost);
        assert_eq!(pb.objective_types[1], ObjectiveType::CloudyArea);
    }
}
