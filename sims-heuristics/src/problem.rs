use std::path::Path;

use log::error;
use regex::Regex;

use crate::util::{DifferenceIterator, IntersectionIterator};

#[derive(Debug, Default)]
pub struct SIMSProblemInstanceRaw {
    name: String,
    num_images: usize,
    universe_size: usize,
    images: Vec<Vec<usize>>,
    costs: Vec<u64>,
    clouds: Vec<Vec<usize>>,
    areas: Vec<u64>,
    max_cloud_area: u64,
    resolution: Vec<u64>,
    incidence_angle: Vec<u64>,
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
    pub fn from_minizinc_datafile<P: AsRef<Path>>(model_path: P) -> Self {
        let mut sims_problem = SIMSProblemInstanceRaw::default();
        sims_problem.name = model_path
            .as_ref()
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
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
                    error!("Unknown variable: {}", key);
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
    pub fn new(index: usize, cost: u64, parts: Vec<usize>, clear_parts: Vec<usize>) -> Self {
        Image {
            index,
            cost,
            clear_parts,
            parts,
        }
    }

    pub fn cost(&self) -> u64 {
        self.cost
    }
}

#[derive(Default, Clone)]
pub struct Element {
    pub area: u64,
    pub images: Vec<usize>,
}

#[derive(Clone)]
pub struct ImageObjectiveDeltas {
    pub image_index: usize,
    pub deltas: (i64, i64),
}

#[derive(Clone)]
pub struct ScaledObjectiveDeltas {
    pub image_index: usize,
    pub scaled_objectives: (f32, f32),
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
    /// Max values of objectives
    pub max_objectives: pareto::Objectives<D>,
}

impl<const D: usize> Problem<D> {
    pub fn from_raw(mut raw: SIMSProblemInstanceRaw) -> Self {
        // Normalize all indices to be zero-based
        raw.images.iter_mut().for_each(|image| {
            image.iter_mut().for_each(|index| {
                *index -= 1;
            });
            image.sort();
        });
        raw.clouds.iter_mut().for_each(|image| {
            image.iter_mut().for_each(|index| {
                *index -= 1;
            });
            image.sort();
        });

        // Create universe
        let mut universe = vec![Element::default(); raw.universe_size];
        raw.images
            .iter()
            .enumerate()
            .for_each(|(image_index, image)| {
                image.iter().for_each(|&element_index| {
                    universe[element_index].images.push(image_index);
                })
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

        assert_eq!(D, 2, "Problem currently only supports 2D objectives");
        let mut max_objectives = [0u64; D];
        max_objectives[0] = max_cost;
        max_objectives[1] = max_cloudy_area;

        Problem {
            instance_name: raw.name,
            universe,
            images,
            overlap_matrix,
            max_objectives,
        }
    }

    /// Load problem instance from minizinc data file
    pub fn from_minizinc_datafile<P: AsRef<Path>>(model_path: &P) -> Self {
        let raw = SIMSProblemInstanceRaw::from_minizinc_datafile(model_path);
        Problem::from_raw(raw)
    }

    /// Get total cost of all images
    pub fn total_cost(&self) -> u64 {
        self.images.iter().map(|image| image.cost).sum()
    }

    /// Get total area of all elements
    pub fn total_area(&self) -> u64 {
        self.universe.iter().map(|element| element.area).sum()
    }
}

impl Default for Problem<2> {
    fn default() -> Self {
        Problem {
            instance_name: String::new(),
            universe: Vec::new(),
            images: Vec::new(),
            overlap_matrix: Vec::new(),
            max_objectives: [0, 0],
        }
    }
}
