use std::{array, path::Path};

use log::error;
use regex::Regex;

use crate::{
    objectives::{ObjectiveState, ObjectiveType},
    solution::ImageSet,
    util::{DifferenceIterator, IntersectionIterator},
};

/// Trait for Set Cover Problem operations
pub trait SetCoverProblem {
    /// Check if a solution forms a valid set cover (covers all elements)
    fn is_set_cover<const D: usize>(&self, solution: &impl ImageSet<D>) -> bool;
}

#[derive(Debug, Default, Clone)]
pub struct SIMSProblemInstanceRaw {
    pub name: String,
    pub num_images: usize,
    pub universe_size: usize,
    pub images: Vec<Vec<usize>>,
    pub costs: Vec<u64>,
    pub clouds: Vec<Vec<usize>>,
    pub areas: Vec<u64>,
    pub max_cloud_area: u64,
    pub resolution: Vec<u64>,
    pub incidence_angle: Vec<u64>,
}

fn parse_vec_of_sets(input_str: &str) -> Vec<Vec<usize>> {
    // Initialize empty vector of vectors
    let mut result: Vec<Vec<usize>> = Vec::new();
    // Remove trailing '[' and ']'
    let sets = input_str.trim_start_matches('[').trim_end_matches(']');
    // Split into sets on those commas that are not inside a set (it will remove the trailing curly braces sadly)
    let re = Regex::new(r"},\s*").unwrap();
    for set in re.split(sets) {
        // Remove leading and trailing curly braces
        let set = set.trim_start_matches('{').trim_end_matches('}');

        // Handle empty sets
        if set.is_empty() {
            result.push(Vec::new());
            continue;
        }

        // Parse set
        let set = set
            .split(',')
            .map(|value| value.trim().parse::<usize>().unwrap())
            .collect::<Vec<_>>();

        // Add set to result
        result.push(set);
    }
    result
}

fn parse_vec(input_str: &str) -> Vec<u64> {
    // Trim leading and trailing '[' and ']'
    let list_of_numbers = input_str.trim_start_matches('[').trim_end_matches(']');

    list_of_numbers
        .split(", ")
        .map(|value| value.parse::<u64>().unwrap())
        .collect::<Vec<_>>()
}

/// Parses a set of vectors from a string representation.
///
/// # Panics
///
/// This function will panic if the input string is malformed or if any value cannot be parsed as a `usize`.
#[must_use]
pub fn parse_set_of_vecs(input_str: &str) -> Vec<Vec<usize>> {
    // Initialize empty vector of vectors
    let mut result: Vec<Vec<usize>> = Vec::new();
    // Remove trailing '{' and '}'
    let vecs = input_str.trim_start_matches('{').trim_end_matches('}');
    // Split into vecs on those commas that are not inside a vec (it will remove the trailing square braces sadly)
    let re = Regex::new(r"],\s*").unwrap();
    for vec in re.split(vecs) {
        // Remove leading and trailing curly braces
        let vec = vec.trim_start_matches('[').trim_end_matches(']');

        // Handle empty vecs
        if vec.is_empty() {
            result.push(Vec::new());
            continue;
        }

        // Parse vec
        let vec = vec
            .split(',')
            .map(|value| value.trim().parse::<usize>().unwrap())
            .collect::<Vec<_>>();

        // Add set to result
        result.push(vec);
    }
    result
}

impl SIMSProblemInstanceRaw {
    /// Constructs a `SIMSProblemInstanceRaw` from a `MiniZinc` data file.
    ///
    /// # Panics
    ///
    /// This function will panic if the file cannot be read, or if the file name, file stem,
    /// or any expected data is missing or malformed.
    pub fn from_minizinc_datafile<P: AsRef<Path>>(model_path: P) -> Self {
        let name = model_path
            .as_ref()
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let mut sims_problem = Self {
            name,
            ..Default::default()
        };
        let model_str = std::fs::read_to_string(model_path).expect("Failed to read model file");

        for line in model_str.lines() {
            // Split line into name and value
            let mut split = line.trim_end_matches(';').split(" = ");

            // Get name
            let key = split.next().unwrap().trim();

            // Parse value
            let value = split.next().unwrap().trim();
            match key {
                "num_images" => {
                    sims_problem.num_images = value.parse::<usize>().unwrap();
                }
                "universe" => {
                    sims_problem.universe_size = value.parse::<usize>().unwrap();
                }
                "images" => {
                    sims_problem.images = parse_vec_of_sets(value);
                }
                "costs" => {
                    sims_problem.costs = parse_vec(value);
                }
                "clouds" => {
                    sims_problem.clouds = parse_vec_of_sets(value);
                }
                "areas" => {
                    sims_problem.areas = parse_vec(value);
                }
                "max_cloud_area" => {
                    sims_problem.max_cloud_area = value.parse::<u64>().unwrap();
                }
                "resolution" => {
                    sims_problem.resolution = parse_vec(value);
                }
                "incidence_angle" => {
                    sims_problem.incidence_angle = parse_vec(value);
                }
                _ => {
                    error!("Unknown variable: {key}");
                }
            }
        }

        sims_problem
    }
}
#[derive(Default, Clone)]
pub struct Image {
    pub index: usize,
    pub parts: Vec<usize>,
    pub clear_parts: Vec<usize>,
    pub cost: u64,
}

impl Image {
    #[must_use]
    pub const fn new(index: usize, cost: u64, parts: Vec<usize>, clear_parts: Vec<usize>) -> Self {
        Self {
            index,
            parts,
            clear_parts,
            cost,
        }
    }

    #[must_use]
    pub const fn cost(&self) -> u64 {
        self.cost
    }
}

#[derive(Default, Clone)]
pub struct Element {
    pub area: u64,
    pub images: Vec<usize>,
}

#[derive(Clone)]
pub struct ImageObjectiveDeltas<const D: usize> {
    pub image_index: usize,
    pub deltas: [i64; D],
}

#[derive(Clone)]
pub struct ScaledObjectiveDeltas<const D: usize> {
    pub image_index: usize,
    pub raw_deltas: [i64; D],
    pub scaled_deltas: [f32; D],
}

#[derive(Clone, Eq, Debug)]
pub struct ComparableImage {
    pub index: usize,
    pub key: usize,
}

impl PartialEq for ComparableImage {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

#[expect(clippy::non_canonical_partial_ord_impl, reason = "Compare only by key")]
impl PartialOrd for ComparableImage {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.key.partial_cmp(&other.key)
    }
}

impl Ord for ComparableImage {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

pub struct Problem<T: ImageSet<D>, const D: usize> {
    /// Name of the problem instance
    pub instance_name: String,
    /// Vector of sets of indices representing the universe, each set represents which images contain the corresponding element
    pub universe: Vec<Element>,
    /// Vector of sets of indices representing the images, each set represents which elements are contained in the corresponding image
    pub images: Vec<Image>,
    /// Matrix of size of overlaps between images
    pub overlap_matrix: Vec<Vec<usize>>,
    /// Objective states containing data and logic
    pub objectives: [ObjectiveState<D>; D],
    /// Objective types for metadata
    pub objective_types: [ObjectiveType; D],
    /// Raw instance data for accessing resolution and incidence angle
    pub raw_instance: SIMSProblemInstanceRaw,
    /// Phantom data to maintain type parameter T
    _phantom: std::marker::PhantomData<T>,
}

// Implement SetCoverProblem trait for Problem
impl<T: ImageSet<D>, const D: usize> SetCoverProblem for Problem<T, D> {
    fn is_set_cover<const D2: usize>(&self, solution: &impl ImageSet<D2>) -> bool {
        let mut covered_elements = vec![false; self.universe.len()];
        
        // Mark all elements covered by selected images
        for image_index in solution.selected_images() {
            for &element_index in &self.images[image_index].parts {
                covered_elements[element_index] = true;
            }
        }
        
        // Check if all elements are covered
        covered_elements.iter().all(|&covered| covered)
    }
}

impl<T: ImageSet<D>, const D: usize> Problem<T, D> {
    /// Constructs a `Problem` from a raw SIMS problem instance with custom objective definitions.
    ///
    /// Create a problem instance from raw data with specified objectives.
    ///
    /// # Errors
    ///
    /// Returns an `Err` if the number of objectives does not match `D` or if any objective has an incorrect index.
    ///
    /// # Panics
    ///
    /// Panics if an unknown objective type is encountered during max objective calculation.
    pub fn from_raw_with_objectives(
        mut raw: SIMSProblemInstanceRaw,
        objective_types: [ObjectiveType; D],
    ) -> Result<Self, String> {
        // Normalize all indices to be zero-based (same as from_raw)
        raw.images.iter_mut().for_each(|image| {
            for index in image.iter_mut() {
                *index -= 1;
            }
            image.sort_unstable();
        });
        raw.clouds.iter_mut().for_each(|image| {
            for index in image.iter_mut() {
                *index -= 1;
            }
            image.sort_unstable();
        });

        // Create universe (same as from_raw)
        let mut universe = vec![Element::default(); raw.universe_size];
        raw.images
            .iter()
            .enumerate()
            .for_each(|(image_index, image)| {
                for &element_index in image {
                    universe[element_index].images.push(image_index);
                }
            });
        raw.areas
            .iter()
            .enumerate()
            .for_each(|(element_index, &area)| {
                universe[element_index].area = area;
            });

        // Create images (same as from_raw)
        let images: Vec<Image> = (0..raw.num_images)
            .map(|image_index| {
                let cost = raw.costs[image_index];
                let clear_parts = raw.images[image_index]
                    .iter()
                    .difference(raw.clouds[image_index].iter())
                    .copied()
                    .collect();
                let parts = raw.images[image_index].clone();
                Image::new(image_index, cost, parts, clear_parts)
            })
            .collect();

        // Create overlap matrix (same as from_raw)
        let mut overlap_matrix: Vec<Vec<usize>> = vec![vec![0; raw.num_images]; raw.num_images];
        #[allow(clippy::needless_range_loop, reason = "Iterator pattern would require multiple mutable borrows of overlap_matrix, which violates borrow checker rules")]
        for i in 0..raw.num_images {
            for j in 0..=i {
                let common_elements = raw.images[i]
                    .iter()
                    .intersection(raw.images[j].iter())
                    .count();
                overlap_matrix[i][j] = common_elements;
                if i != j {
                    overlap_matrix[j][i] = common_elements;
                }
            }
        }

        // Create ObjectiveState from types and raw data
        let objectives = ObjectiveState::from_types_and_raw(&objective_types, &raw);

        Ok(Self {
            instance_name: raw.name.clone(),
            universe,
            images,
            overlap_matrix,
            objectives,
            objective_types,
            raw_instance: raw,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Load problem instance from minizinc data file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed, or if problem creation fails.
    pub fn from_minizinc_datafile<P: AsRef<Path>>(
        model_path: &P,
        objective_types: [ObjectiveType; D],
    ) -> Result<Self, String> {
        let raw = SIMSProblemInstanceRaw::from_minizinc_datafile(model_path);
        Self::from_raw_with_objectives(raw, objective_types)
    }

    /// Get total cost of all images
    #[must_use]
    pub fn total_cost(&self) -> u64 {
        self.images.iter().map(|image| image.cost).sum()
    }

    /// Get total area of all elements
    #[must_use]
    pub fn total_area(&self) -> u64 {
        self.universe.iter().map(|element| element.area).sum()
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

    /// Get objective by ID
    #[must_use]
    pub fn objective_type_by_id(&self, id: &str) -> Option<ObjectiveType> {
        self.objective_types.iter().copied().find(|obj| obj.id() == id)
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

    /// Get max objectives array from objective states
    #[must_use]
    pub fn max_objectives(&self) -> [u64; D] {
        std::array::from_fn(|i| self.objectives[i].max_value())
    }

    /// Get objective bounds if set
    ///
    /// # Panics
    ///
    /// Panics if unwrapping bounds fails (should not happen in normal operation)
    #[must_use]
    pub fn objective_bounds(&self) -> Option<[[u64; 2]; D]> {
        if self.objectives.iter().all(|obj| obj.bounds().is_some()) {
            Some(std::array::from_fn(|i| self.objectives[i].bounds().unwrap()))
        } else {
            None
        }
    }

    /// Set objective bounds for normalization in scalarization
    /// Bounds should be [[min, max], [min, max], ...] for each objective
    pub fn set_objective_bounds(&mut self, bounds: [[u64; 2]; D]) {
        self.objectives
            .iter_mut()
            .zip(bounds)
            .for_each(|(obj, bound)| obj.set_bounds(bound));
    }

    /// Create a builder for this problem
    #[must_use]
    pub const fn builder(raw_data: SIMSProblemInstanceRaw) -> ProblemBuilder<T, D> {
        ProblemBuilder::new(raw_data)
    }
}

impl<T: ImageSet<D>, const D: usize> Default for Problem<T, D> {
    fn default() -> Self {
        // Create default objective types array based on the dimension D
        let objective_types: [ObjectiveType; D] = array::from_fn(|i| match i {
            0 => ObjectiveType::TotalCost,
            1 => ObjectiveType::CloudyArea,
            2 => ObjectiveType::MinResolution,
            3 => ObjectiveType::MaxIncidenceAngle,
            _ => panic!("Unknown objective"),
        });

        let objectives: [ObjectiveState<D>; D] = array::from_fn(|i| match i {
            0 => ObjectiveState::TotalCost { costs: vec![], max_value: 0, bounds: None },
            1 => ObjectiveState::CloudyArea { clear_images: vec![], areas: vec![], max_value: 0, bounds: None },
            2 => ObjectiveState::MinResolution { resolutions: vec![], max_value: 0, bounds: None },
            3 => ObjectiveState::MaxIncidenceAngle { incidence_angles: vec![], max_value: 0, bounds: None },
            _ => panic!("Unknown objective"),
        });

        Self {
            instance_name: String::new(),
            universe: Vec::new(),
            images: Vec::new(),
            overlap_matrix: Vec::new(),
            objectives,
            objective_types,
            raw_instance: SIMSProblemInstanceRaw::default(),
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Builder for constructing Problem instances with custom objectives
pub struct ProblemBuilder<T: ImageSet<D>, const D: usize> {
    raw_data: SIMSProblemInstanceRaw,
    objectives: Option<Vec<ObjectiveType>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: ImageSet<D>, const D: usize> ProblemBuilder<T, D> {
    /// Create a new builder with raw problem data
    #[must_use]
    pub const fn new(raw_data: SIMSProblemInstanceRaw) -> Self {
        Self {
            raw_data,
            objectives: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set custom objectives for the problem
    #[must_use]
    pub fn with_objectives(mut self, objectives: Vec<ObjectiveType>) -> Self {
        self.objectives = Some(objectives);
        self
    }

    /// Build the final Problem instance
    ///
    /// # Errors
    ///
    /// Returns an `Err` if the number of objectives does not match `D` or if any objective has an incorrect index.
    pub fn build(self) -> Result<Problem<T, D>, String> {
        if let Some(objectives) = self.objectives {
            if objectives.len() != D {
                return Err(format!(
                    "Expected {} objectives, got {}",
                    D,
                    objectives.len()
                ));
            }

            // Convert Vec to array
            let objectives_array: [ObjectiveType; D] = objectives
                .try_into()
                .map_err(|_| "Failed to convert vector to array")?;

            // Build problem with custom objectives
            Problem::from_raw_with_objectives(self.raw_data, objectives_array)
        } else {
            Err("No objectives set. Use with_objectives() to set custom objectives.".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solution_impl::vec_encoded_solution::VecEncodedSolution;

    #[test]
    fn test_set_cover_problem_trait() {
        // Create a simple problem instance
        let raw = SIMSProblemInstanceRaw {
            name: "test".to_string(),
            num_images: 2,
            universe_size: 3,
            images: vec![vec![1, 2], vec![2, 3]],
            costs: vec![10, 20],
            clouds: vec![vec![], vec![]],
            areas: vec![1, 1, 1],
            max_cloud_area: 0,
            resolution: vec![100, 100],
            incidence_angle: vec![0, 0],
        };

        let objectives = [
            crate::objectives::ObjectiveType::TotalCost,
            crate::objectives::ObjectiveType::CloudyArea,
        ];
        
        let problem: Problem<VecEncodedSolution<2>, 2> = 
            Problem::from_raw_with_objectives(raw, objectives).unwrap();
        
        // Test Problem struct fields directly
        assert_eq!(problem.images.len(), 2);
        assert_eq!(problem.universe.len(), 3);
        assert_eq!(problem.instance_name, "test");
        
        // Test specific accessors
        assert_eq!(problem.images[0].cost, 10);
        assert_eq!(problem.images[1].cost, 20);
    }

    #[test]
    fn test_set_cover_problem_trait_generic() {
        // Demonstrate that the trait works with generic functions
        fn check_if_valid_cover<const D: usize, P: SetCoverProblem>(
            problem: &P,
            solution: &impl ImageSet<D>,
        ) -> bool {
            problem.is_set_cover(solution)
        }

        let raw = SIMSProblemInstanceRaw {
            name: "generic_test".to_string(),
            num_images: 3,
            universe_size: 3,
            images: vec![vec![1, 2], vec![2, 3], vec![1, 3]],
            costs: vec![10, 20, 30],
            clouds: vec![vec![], vec![], vec![]],
            areas: vec![1, 1, 1],
            max_cloud_area: 0,
            resolution: vec![100, 100, 100],
            incidence_angle: vec![0, 0, 0],
        };

        let objectives = [
            crate::objectives::ObjectiveType::TotalCost,
            crate::objectives::ObjectiveType::CloudyArea,
        ];
        
        let problem: Problem<VecEncodedSolution<2>, 2> = 
            Problem::from_raw_with_objectives(raw, objectives).unwrap();
        
        // Test with a valid cover
        let solution = VecEncodedSolution::<2>::from_selected_images(&[0, 1], &problem);
        assert!(check_if_valid_cover(&problem, &solution));
    }

    #[test]
    fn test_is_set_cover() {
        // Create a problem where elements 1,2,3 need to be covered
        let raw = SIMSProblemInstanceRaw {
            name: "cover_test".to_string(),
            num_images: 3,
            universe_size: 3,
            images: vec![
                vec![1, 2],    // Image 0 covers elements 0, 1
                vec![2, 3],    // Image 1 covers elements 1, 2
                vec![1, 3],    // Image 2 covers elements 0, 2
            ],
            costs: vec![10, 20, 15],
            clouds: vec![vec![], vec![], vec![]],
            areas: vec![1, 1, 1],
            max_cloud_area: 0,
            resolution: vec![100, 100, 100],
            incidence_angle: vec![0, 0, 0],
        };

        let objectives = [
            crate::objectives::ObjectiveType::TotalCost,
            crate::objectives::ObjectiveType::CloudyArea,
        ];
        
        let problem: Problem<VecEncodedSolution<2>, 2> = 
            Problem::from_raw_with_objectives(raw, objectives).unwrap();
        
        // Test valid set cover: images 0 and 1 should cover all elements
        let solution1 = VecEncodedSolution::<2>::from_selected_images(&[0, 1], &problem);
        assert!(problem.is_set_cover(&solution1), "Images 0 and 1 should form a valid set cover");
        
        // Test another valid set cover: images 0 and 2
        let solution2 = VecEncodedSolution::<2>::from_selected_images(&[0, 2], &problem);
        assert!(problem.is_set_cover(&solution2), "Images 0 and 2 should form a valid set cover");
        
        // Test invalid set cover: only image 0 doesn't cover all elements
        let solution3 = VecEncodedSolution::<2>::from_selected_images(&[0], &problem);
        assert!(!problem.is_set_cover(&solution3), "Only image 0 should not form a valid set cover");
        
        // Test all images selected (definitely a valid cover)
        let solution4 = VecEncodedSolution::<2>::from_selected_images(&[0, 1, 2], &problem);
        assert!(problem.is_set_cover(&solution4), "All images should form a valid set cover");
    }
}
