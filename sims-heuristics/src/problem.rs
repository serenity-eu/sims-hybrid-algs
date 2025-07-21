use std::path::Path;

use log::error;
use regex::Regex;

use crate::{
    objectives::{ObjectiveDefinition, ObjectiveType},
    util::{DifferenceIterator, IntersectionIterator},
};

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

pub struct Problem<const D: usize> {
    /// Name of the problem instance
    pub instance_name: String,
    /// Vector of sets of indices representing the universe, each set represents which images contain the corresponding element
    pub universe: Vec<Element>,
    /// Vector of sets of indices representing the images, each set represents which elements are contained in the corresponding image
    pub images: Vec<Image>,
    /// Matrix of size of overlaps between images
    pub overlap_matrix: Vec<Vec<usize>>,
    /// Legacy objective definitions (for backward compatibility)
    pub objectives: Vec<ObjectiveType>,
    /// Generic objective definitions (new system)
    pub objective_definitions: Option<Vec<Box<dyn ObjectiveDefinition<D>>>>,
    /// Max values of objectives
    pub max_objectives: pareto::Objectives<D>,
    /// Raw instance data for accessing resolution and incidence angle
    pub raw_instance: SIMSProblemInstanceRaw,
}

impl<const D: usize> Problem<D> {
    /// Constructs a `Problem` from a raw SIMS problem instance.
    ///
    /// # Panics
    ///
    /// Panics if `D` is not 2, as only 2D objectives are currently supported.
    #[must_use]
    pub fn from_raw(mut raw: SIMSProblemInstanceRaw) -> Self {
        // Normalize all indices to be zero-based
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

        // Create universe
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

        // Create images
        let images = (0..raw.num_images)
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

        let mut overlap_matrix: Vec<Vec<usize>> = vec![vec![0; raw.num_images]; raw.num_images];

        for i in 0..raw.num_images {
            for j in 0..=i {
                // Find common element count between image i and j
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

        let max_cost: u64 = raw.costs.iter().sum();
        let max_cloudy_area = raw.max_cloud_area;

        assert_eq!(
            D, 2,
            "from_raw only supports 2D objectives - use ProblemBuilder for higher dimensions"
        );
        let mut max_objectives = [0u64; D];
        max_objectives[0] = max_cost;
        max_objectives[1] = max_cloudy_area;

        // Create default 2D objectives
        let objectives = vec![
            crate::objectives::ObjectiveType::TotalCost,
            crate::objectives::ObjectiveType::CloudyArea,
        ];

        Self {
            instance_name: raw.name.clone(),
            universe,
            images,
            overlap_matrix,
            objectives,
            objective_definitions: None, // Legacy constructor uses None
            max_objectives,
            raw_instance: raw,
        }
    }

    /// Constructs a `Problem` from a raw SIMS problem instance with custom objective definitions.
    ///
    /// # Errors
    ///
    /// Returns an `Err` if the number of objectives does not match `D` or if any objective has an incorrect index.
    pub fn from_raw_with_objective_definitions(
        mut raw: SIMSProblemInstanceRaw,
        objective_definitions: Vec<Box<dyn ObjectiveDefinition<D>>>,
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

        // Calculate max objectives using the provided objective definitions
        let mut max_objectives = [0u64; D];
        for (i, obj_def) in objective_definitions.iter().enumerate() {
            // Create a dummy problem to calculate max values
            let dummy_problem = Self {
                instance_name: raw.name.clone(),
                universe: universe.clone(),
                images: images.clone(),
                overlap_matrix: overlap_matrix.clone(),
                objectives: vec![], // Empty for dummy
                objective_definitions: None,
                max_objectives: [0u64; D],
                raw_instance: raw.clone(),
            };
            max_objectives[i] = obj_def.max_value(&dummy_problem);
        }

        // Create legacy objectives for backward compatibility
        let legacy_objectives = if D == 2 {
            vec![
                crate::objectives::ObjectiveType::TotalCost,
                crate::objectives::ObjectiveType::CloudyArea,
            ]
        } else {
            vec![] // For non-2D problems, we don't have legacy objectives
        };

        Ok(Self {
            instance_name: raw.name.clone(),
            universe,
            images,
            overlap_matrix,
            objectives: legacy_objectives,
            objective_definitions: Some(objective_definitions),
            max_objectives,
            raw_instance: raw,
        })
    }

    /// Load problem instance from minizinc data file
    pub fn from_minizinc_datafile<P: AsRef<Path>>(model_path: &P) -> Self {
        let raw = SIMSProblemInstanceRaw::from_minizinc_datafile(model_path);
        Self::from_raw(raw)
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
    pub fn objective(&self, index: usize) -> &ObjectiveType {
        &self.objectives[index]
    }

    /// Get objective by ID
    #[must_use]
    pub fn objective_by_id(&self, id: &str) -> Option<&ObjectiveType> {
        self.objectives.iter().find(|obj| obj.id() == id)
    }

    /// Get objective definition by index (new system)
    #[must_use]
    pub fn get_objective_definition(&self, index: usize) -> Option<&dyn ObjectiveDefinition<D>> {
        self.objective_definitions
            .as_ref()
            .and_then(|defs| defs.get(index))
            .map(std::convert::AsRef::as_ref)
    }

    /// Get objective definition by ID (new system)
    #[must_use]
    pub fn get_objective_definition_by_id(&self, id: &str) -> Option<&dyn ObjectiveDefinition<D>> {
        self.objective_definitions.as_ref().and_then(|defs| {
            defs.iter()
                .find(|obj| obj.id() == id)
                .map(std::convert::AsRef::as_ref)
        })
    }

    /// Get number of objectives
    #[must_use]
    pub const fn num_objectives(&self) -> usize {
        D
    }

    /// Get all objective names (from new system if available, otherwise legacy)
    #[must_use]
    pub fn objective_names(&self) -> Vec<&str> {
        self.objective_definitions.as_ref().map_or_else(
            || {
                self.objectives
                    .iter()
                    .map(super::objectives::ObjectiveType::name)
                    .collect()
            },
            |objective_definitions| objective_definitions.iter().map(|obj| obj.name()).collect(),
        )
    }

    /// Check if using new objective system
    #[must_use]
    pub fn has_objective_definitions(&self) -> bool {
        self.objective_definitions.is_some()
    }

    /// Create a builder for this problem
    #[must_use]
    pub fn builder(raw_data: SIMSProblemInstanceRaw) -> ProblemBuilder<D> {
        ProblemBuilder::new(raw_data)
    }
}

impl Default for Problem<2> {
    fn default() -> Self {
        Self {
            instance_name: String::new(),
            universe: Vec::new(),
            images: Vec::new(),
            overlap_matrix: Vec::new(),
            objectives: vec![
                crate::objectives::ObjectiveType::TotalCost,
                crate::objectives::ObjectiveType::CloudyArea,
            ],
            objective_definitions: None, // Default constructor uses None
            max_objectives: [0, 0],
            raw_instance: SIMSProblemInstanceRaw::default(),
        }
    }
}

/// Builder for constructing Problem instances with custom objectives
pub struct ProblemBuilder<const D: usize> {
    raw_data: SIMSProblemInstanceRaw,
    objective_definitions: Option<Vec<Box<dyn ObjectiveDefinition<D>>>>,
}

impl<const D: usize> ProblemBuilder<D> {
    /// Create a new builder from raw problem data
    #[must_use]
    pub fn new(raw_data: SIMSProblemInstanceRaw) -> Self {
        Self {
            raw_data,
            objective_definitions: None,
        }
    }

    /// Set the objective definitions for this problem
    #[must_use]
    pub fn with_objective_definitions(
        mut self,
        objectives: Vec<Box<dyn ObjectiveDefinition<D>>>,
    ) -> Self {
        self.objective_definitions = Some(objectives);
        self
    }

    /// Build the final Problem instance
    ///
    /// # Errors
    ///
    /// Returns an `Err` if the number of objectives does not match `D` or if any objective has an incorrect index.
    pub fn build(self) -> Result<Problem<D>, String> {
        if let Some(objective_definitions) = self.objective_definitions {
            if objective_definitions.len() != D {
                return Err(format!(
                    "Expected {} objectives, got {}",
                    D,
                    objective_definitions.len()
                ));
            }

            // Validate objective indices
            for (i, obj) in objective_definitions.iter().enumerate() {
                if obj.objective_index() != i {
                    return Err(format!(
                        "Objective at position {} has incorrect index {}",
                        i,
                        obj.objective_index()
                    ));
                }
            }

            // Build problem with custom objectives
            Problem::from_raw_with_objective_definitions(self.raw_data, objective_definitions)
        } else {
            // Use default objectives for backward compatibility
            Ok(Problem::from_raw(self.raw_data))
        }
    }
}
