use log::{debug, info};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::{
    prelude::*,
    types::{PyDict, PyType},
};
use std::collections::HashSet;
use std::time::Duration;

/// Represents a SIMS discrete problem instance
#[pyclass]
#[derive(Clone, Debug)]
pub struct SimsDiscreteProblem {
    #[pyo3(get, set)]
    pub num_images: usize,
    #[pyo3(get, set)]
    pub universe: usize,
    #[pyo3(get, set)]
    pub images: Vec<Vec<usize>>,
    #[pyo3(get, set)]
    pub costs: Vec<i32>,
    #[pyo3(get, set)]
    pub clouds: Vec<Vec<usize>>,
    #[pyo3(get, set)]
    pub areas: Vec<i32>,
    #[pyo3(get, set)]
    pub resolution: Vec<i32>,
    #[pyo3(get, set)]
    pub incidence_angle: Vec<i32>,
    #[pyo3(get, set)]
    pub max_cloud_area: i32,
}

#[pymethods]
impl SimsDiscreteProblem {
    #[new]
    pub fn new(
        num_images: usize,
        universe: usize,
        images: Vec<Vec<usize>>,
        costs: Vec<i32>,
        clouds: Vec<Vec<usize>>,
        areas: Vec<i32>,
        resolution: Vec<i32>,
        incidence_angle: Vec<i32>,
        max_cloud_area: i32,
    ) -> Self {
        Self {
            num_images,
            universe,
            images,
            costs,
            clouds,
            areas,
            resolution,
            incidence_angle,
            max_cloud_area,
        }
    }

    /// Create SimsDiscreteProblem from a dictionary
    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyDict>) -> PyResult<Self> {
        let num_images: usize = data.get_item("num_images")?.unwrap().extract()?;
        let universe: usize = data.get_item("universe")?.unwrap().extract()?;
        let images: Vec<Vec<usize>> = data.get_item("images")?.unwrap().extract()?;
        let costs: Vec<i32> = data.get_item("costs")?.unwrap().extract()?;
        let clouds: Vec<Vec<usize>> = data.get_item("clouds")?.unwrap().extract()?;
        let areas: Vec<i32> = data.get_item("areas")?.unwrap().extract()?;
        let resolution: Vec<i32> = data.get_item("resolution")?.unwrap().extract()?;
        let incidence_angle: Vec<i32> = data.get_item("incidence_angle")?.unwrap().extract()?;
        let max_cloud_area: i32 = data.get_item("max_cloud_area")?.unwrap().extract()?;

        Ok(Self::new(
            num_images,
            universe,
            images,
            costs,
            clouds,
            areas,
            resolution,
            incidence_angle,
            max_cloud_area,
        ))
    }

    /// Create SimsDiscreteProblem from a MiniZinc data file (.dzn)
    #[classmethod]
    fn from_dzn(_cls: &Bound<'_, PyType>, file_path: &str) -> PyResult<Self> {
        use std::collections::HashMap;
        use std::fs;

        // Read file content
        let content = fs::read_to_string(file_path)
            .map_err(|e| PyValueError::new_err(format!("Failed to read file {file_path}: {e}")))?;

        let mut data: HashMap<String, String> = HashMap::new();

        // Parse simple integer values
        for field in ["num_images", "universe", "max_cloud_area"] {
            let pattern = format!(r"{field}\s*=\s*(\d+);");
            if let Some(captures) = regex::Regex::new(&pattern).unwrap().captures(&content) {
                data.insert(field.to_string(), captures[1].to_string());
            }
        }

        // Parse array of integers
        for field in ["costs", "areas", "resolution", "incidence_angle"] {
            let pattern = format!(r"{field}\s*=\s*\[(.*?)\];");
            if let Some(captures) = regex::Regex::new(&pattern).unwrap().captures(&content) {
                data.insert(field.to_string(), captures[1].to_string());
            }
        }

        // Parse array of sets (for images and clouds)
        for field in ["images", "clouds"] {
            let pattern = format!(r"{field}\s*=\s*\[(.*?)\];");
            if let Some(captures) = regex::Regex::new(&pattern).unwrap().captures(&content) {
                data.insert(field.to_string(), captures[1].to_string());
            }
        }

        // Extract and parse the data
        let num_images: usize = data
            .get("num_images")
            .ok_or_else(|| PyValueError::new_err("Missing num_images"))?
            .parse()
            .map_err(|e| PyValueError::new_err(format!("Invalid num_images: {e}")))?;

        let universe: usize = data
            .get("universe")
            .ok_or_else(|| PyValueError::new_err("Missing universe"))?
            .parse()
            .map_err(|e| PyValueError::new_err(format!("Invalid universe: {e}")))?;

        let max_cloud_area: i32 = data
            .get("max_cloud_area")
            .ok_or_else(|| PyValueError::new_err("Missing max_cloud_area"))?
            .parse()
            .map_err(|e| PyValueError::new_err(format!("Invalid max_cloud_area: {e}")))?;

        // Parse integer arrays
        let costs: Vec<i32> = Self::parse_int_array(data.get("costs").map_or("", |v| v))?;
        let areas: Vec<i32> = Self::parse_int_array(data.get("areas").map_or("", |v| v))?;
        let resolution: Vec<i32> = Self::parse_int_array(data.get("resolution").map_or("", |v| v))?;
        let incidence_angle: Vec<i32> =
            Self::parse_int_array(data.get("incidence_angle").map_or("", |v| v))?;

        // Parse set arrays (convert from 1-based to 0-based indexing)
        let images: Vec<Vec<usize>> = Self::parse_set_array(data.get("images").map_or("", |v| v))?;
        let clouds: Vec<Vec<usize>> = Self::parse_set_array(data.get("clouds").map_or("", |v| v))?;

        Ok(Self::new(
            num_images,
            universe,
            images,
            costs,
            clouds,
            areas,
            resolution,
            incidence_angle,
            max_cloud_area,
        ))
    }

    /// Convert to dictionary representation
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("num_images", self.num_images)?;
        dict.set_item("universe", self.universe)?;
        dict.set_item("images", &self.images)?;
        dict.set_item("costs", &self.costs)?;
        dict.set_item("clouds", &self.clouds)?;
        dict.set_item("areas", &self.areas)?;
        dict.set_item("resolution", &self.resolution)?;
        dict.set_item("incidence_angle", &self.incidence_angle)?;
        dict.set_item("max_cloud_area", self.max_cloud_area)?;
        Ok(dict.into())
    }

    /// Get maximum values for cost and area objectives
    fn get_max_values(&self) -> (i32, i32) {
        let max_cost: i32 = self.costs.iter().sum();
        let max_area: i32 = self.areas.iter().sum();
        (max_cost, max_area)
    }

    /// Get reference point for Pareto optimization
    fn get_ref_point(&self) -> (i32, i32) {
        let (max_cost, max_area) = self.get_max_values();
        (max_cost + 1, max_area + 1)
    }

    /// Validate the problem instance
    fn validate(&self) -> PyResult<()> {
        if self.num_images == 0 {
            return Err(PyValueError::new_err(
                "num_images must be a positive integer",
            ));
        }
        if self.universe == 0 {
            return Err(PyValueError::new_err("universe must be a positive integer"));
        }
        if self.images.len() != self.num_images {
            return Err(PyValueError::new_err(
                "Number of images does not match num_images",
            ));
        }
        if self.costs.len() != self.num_images {
            return Err(PyValueError::new_err(
                "Number of costs does not match num_images",
            ));
        }
        if self.clouds.len() != self.num_images {
            return Err(PyValueError::new_err(
                "Number of clouds does not match num_images",
            ));
        }
        if self.areas.len() != self.universe {
            return Err(PyValueError::new_err(
                "Number of areas does not match universe",
            ));
        }
        if self.resolution.len() != self.num_images {
            return Err(PyValueError::new_err(
                "Number of resolutions does not match num_images",
            ));
        }
        if self.incidence_angle.len() != self.num_images {
            return Err(PyValueError::new_err(
                "Number of incidence angles does not match num_images",
            ));
        }

        // Check if the union of all image fragments covers all indices from 0 to universe-1
        let mut all_indices = HashSet::new();
        for image in &self.images {
            for &fragment in image {
                all_indices.insert(fragment);
            }
        }

        let expected_indices: HashSet<usize> = (0..self.universe).collect();
        if all_indices != expected_indices {
            let missing: Vec<usize> = expected_indices.difference(&all_indices).cloned().collect();
            return Err(PyValueError::new_err(format!(
                "Images do not cover all indices from 0 to universe-1. Missing indices: {missing:?}"
            )));
        }

        Ok(())
    }

    /// Get the image fragments for a specific image index
    fn get_image_fragments(&self, image_idx: usize) -> PyResult<Vec<usize>> {
        if image_idx >= self.num_images {
            return Err(PyIndexError::new_err("Image index out of bounds"));
        }
        Ok(self.images[image_idx].clone())
    }

    /// Get the cloud fragments for a specific image index
    fn get_cloud_fragments(&self, image_idx: usize) -> PyResult<Vec<usize>> {
        if image_idx >= self.num_images {
            return Err(PyIndexError::new_err("Image index out of bounds"));
        }
        Ok(self.clouds[image_idx].clone())
    }

    /// Check if a fragment is covered by clouds in a specific image
    fn is_fragment_cloudy(&self, image_idx: usize, fragment_idx: usize) -> PyResult<bool> {
        if image_idx >= self.num_images {
            return Err(PyIndexError::new_err("Image index out of bounds"));
        }
        if fragment_idx >= self.universe {
            return Err(PyIndexError::new_err("Fragment index out of bounds"));
        }
        Ok(self.clouds[image_idx].contains(&fragment_idx))
    }

    /// Get total cost for a set of selected images
    fn calculate_total_cost(&self, selected_images: Vec<usize>) -> PyResult<i32> {
        let mut total_cost = 0;
        for &image_idx in &selected_images {
            if image_idx >= self.num_images {
                return Err(PyIndexError::new_err("Image index out of bounds"));
            }
            total_cost += self.costs[image_idx];
        }
        Ok(total_cost)
    }

    /// Get total cloud area for a set of selected images
    fn calculate_total_cloud_area(&self, selected_images: Vec<usize>) -> PyResult<i32> {
        let mut total_cloud_area = 0;
        let mut covered_cloud_fragments = HashSet::new();

        for &image_idx in &selected_images {
            if image_idx >= self.num_images {
                return Err(PyIndexError::new_err("Image index out of bounds"));
            }
            for &fragment_idx in &self.clouds[image_idx] {
                covered_cloud_fragments.insert(fragment_idx);
            }
        }

        for &fragment_idx in &covered_cloud_fragments {
            total_cloud_area += self.areas[fragment_idx];
        }

        Ok(total_cloud_area)
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "SimsDiscreteProblem(num_images={}, universe={}, max_cloud_area={})",
            self.num_images, self.universe, self.max_cloud_area
        )
    }

    /// String representation
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl SimsDiscreteProblem {
    /// Helper method to parse integer arrays from dzn format
    fn parse_int_array(values_str: &str) -> PyResult<Vec<i32>> {
        let mut result = Vec::new();
        for value in values_str.split(',') {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                result.push(
                    trimmed
                        .parse()
                        .map_err(|e| PyValueError::new_err(format!("Invalid integer: {e}")))?,
                );
            }
        }
        Ok(result)
    }

    /// Helper method to parse set arrays from dzn format (converts 1-based to 0-based indexing)
    fn parse_set_array(sets_str: &str) -> PyResult<Vec<Vec<usize>>> {
        let mut result = Vec::new();
        let set_regex = regex::Regex::new(r"\{([^}]*)\}").unwrap();

        for capture in set_regex.captures_iter(sets_str) {
            let set_content = &capture[1];
            let mut set_elements = Vec::new();

            if !set_content.trim().is_empty() {
                for element in set_content.split(',') {
                    let trimmed = element.trim();
                    if !trimmed.is_empty() {
                        let value: usize = trimmed.parse().map_err(|e| {
                            PyValueError::new_err(format!("Invalid set element: {e}"))
                        })?;
                        // Convert from 1-based to 0-based indexing
                        if value > 0 {
                            set_elements.push(value - 1);
                        } else {
                            return Err(PyValueError::new_err(format!(
                                "Invalid index {value} (must be > 0)"
                            )));
                        }
                    }
                }
            }
            result.push(set_elements);
        }
        Ok(result)
    }

    /// Convert to sims-heuristics Problem format directly in memory
    fn to_pls_problem(&self) -> pls::problem::Problem<2> {
        debug!(
            "Converting SIMS problem to PLS format: {} images, universe size {}",
            self.num_images, self.universe
        );
        debug!(
            "Image sets: {} total, first few: {:?}",
            self.images.len(),
            self.images.iter().take(3).collect::<Vec<_>>()
        );

        // Convert from Python 0-based indexing to MiniZinc 1-based indexing for the raw data
        let converted_images: Vec<Vec<usize>> = self
            .images
            .iter()
            .map(|img| img.iter().map(|&x| x + 1).collect())
            .collect();
        debug!("Converted to 1-based indexing for PLS");

        let raw_instance = pls::problem::SIMSProblemInstanceRaw {
            name: "python_instance".to_string(),
            num_images: self.num_images,
            universe_size: self.universe,
            images: converted_images,
            costs: self.costs.iter().map(|&c| c as u64).collect(),
            clouds: self
                .clouds
                .iter()
                .map(|cloud| cloud.iter().map(|&x| x + 1).collect())
                .collect(),
            areas: self.areas.iter().map(|&a| a as u64).collect(),
            max_cloud_area: self.max_cloud_area as u64,
            resolution: self.resolution.iter().map(|&r| r as u64).collect(),
            incidence_angle: self.incidence_angle.iter().map(|&i| i as u64).collect(),
        };

        debug!("Creating PLS problem from raw instance");
        let pls_problem = pls::problem::Problem::from_raw(raw_instance);
        info!("Successfully created PLS problem");
        pls_problem
    }
}

/// Represents a solution to the SIMS problem
#[pyclass]
#[derive(Clone, Debug)]
pub struct Solution {
    #[pyo3(get, set)]
    pub selected_images: HashSet<usize>,
    #[pyo3(get, set)]
    pub cost: i32,
    #[pyo3(get, set)]
    pub cloudy_area: i32,
    #[pyo3(get, set)]
    pub timestamp_us: u64, // Using u64 for microseconds
    #[pyo3(get, set)]
    pub max_incidence_angle: Option<i32>,
    #[pyo3(get, set)]
    pub min_resolutions_sum: Option<i32>,
}

impl PartialEq for Solution {
    fn eq(&self, other: &Self) -> bool {
        self.selected_images == other.selected_images
    }
}

impl Eq for Solution {}

impl std::hash::Hash for Solution {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash based on selected_images only, like in the Python version
        for &image in &self.selected_images {
            image.hash(state);
        }
    }
}

#[pymethods]
impl Solution {
    #[new]
    fn __new__() -> PyResult<Self> {
        Err(PyValueError::new_err(
            "Use Solution.create() instead of Solution()",
        ))
    }

    #[staticmethod]
    fn create(
        selected_images: Vec<usize>,
        cost: i32,
        cloudy_area: i32,
        timestamp_us: u64,
        max_incidence_angle: Option<i32>,
        min_resolutions_sum: Option<i32>,
    ) -> Self {
        Self {
            selected_images: selected_images.into_iter().collect(),
            cost,
            cloudy_area,
            timestamp_us,
            max_incidence_angle,
            min_resolutions_sum,
        }
    }

    /// Convert solution to JSON-compatible dictionary
    fn to_json(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);

            // Convert HashSet to Vec for JSON serialization
            let selected_images_list: Vec<usize> = self.selected_images.iter().cloned().collect();
            dict.set_item("selected_images", selected_images_list)?;
            dict.set_item("cost", self.cost)?;
            dict.set_item("cloudy_area", self.cloudy_area)?;
            dict.set_item("timestamp_us", self.timestamp_us)?;
            dict.set_item("max_incidence_angle", self.max_incidence_angle)?;
            dict.set_item("min_resolutions_sum", self.min_resolutions_sum)?;

            Ok(dict.into())
        })
    }

    /// Get selected images as a list
    fn get_selected_images_list(&self) -> Vec<usize> {
        self.selected_images.iter().cloned().collect()
    }

    /// Add an image to the selection
    fn add_image(&mut self, image_idx: usize) {
        self.selected_images.insert(image_idx);
    }

    /// Remove an image from the selection
    fn remove_image(&mut self, image_idx: usize) -> bool {
        self.selected_images.remove(&image_idx)
    }

    /// Check if an image is selected
    fn contains_image(&self, image_idx: usize) -> bool {
        self.selected_images.contains(&image_idx)
    }

    /// Get the number of selected images
    fn num_selected_images(&self) -> usize {
        self.selected_images.len()
    }

    /// Validate the solution against a problem instance
    fn validate(&self, problem: &SimsDiscreteProblem) -> PyResult<bool> {
        // Check if all universe elements are covered
        let mut coverage = HashSet::new();
        for &image_idx in &self.selected_images {
            if image_idx >= problem.num_images {
                return Err(PyIndexError::new_err(format!(
                    "Image index {} out of bounds (max: {})",
                    image_idx,
                    problem.num_images - 1
                )));
            }
            for &fragment in &problem.images[image_idx] {
                coverage.insert(fragment);
            }
        }

        let is_complete_coverage = coverage.len() == problem.universe;

        if !is_complete_coverage {
            let uncovered: Vec<usize> = (0..problem.universe)
                .filter(|i| !coverage.contains(i))
                .collect();
            println!(
                "Error: the selected images do not cover the whole universe, uncovered elements: {uncovered:?}"
            );
        }

        // Validate objectives if they are computed
        let objectives_valid = self.validate_objectives(problem)?;

        Ok(is_complete_coverage && objectives_valid)
    }

    /// Compute the objectives for this solution
    fn compute_objectives(&self, problem: &SimsDiscreteProblem) -> PyResult<(i32, i32, i32, i32)> {
        // Total cost
        let total_cost: i32 = self
            .selected_images
            .iter()
            .map(|&i| {
                if i >= problem.num_images {
                    0 // Handle out of bounds gracefully
                } else {
                    problem.costs[i]
                }
            })
            .sum();

        // Calculate clear parts (areas covered but not cloudy)
        let mut clear_parts = HashSet::new();
        for &image_idx in &self.selected_images {
            if image_idx < problem.num_images {
                // Add fragments that are in the image but not in clouds
                for &fragment in &problem.images[image_idx] {
                    if !problem.clouds[image_idx].contains(&fragment) {
                        clear_parts.insert(fragment);
                    }
                }
            }
        }

        // Calculate cloudy area (areas in universe that are not clear)
        let cloudy_area: i32 = (0..problem.universe)
            .filter(|&u| !clear_parts.contains(&u))
            .map(|u| problem.areas[u])
            .sum();

        // Max incidence angle
        let max_incidence_angle = self
            .selected_images
            .iter()
            .filter(|&&i| i < problem.num_images)
            .map(|&i| problem.incidence_angle[i])
            .max()
            .unwrap_or(0);

        // Min resolutions sum
        let min_resolutions_sum: i32 = (0..problem.universe)
            .map(|u| {
                self.selected_images
                    .iter()
                    .filter(|&&i| i < problem.num_images && problem.images[i].contains(&u))
                    .map(|&i| problem.resolution[i])
                    .min()
                    .unwrap_or(0)
            })
            .sum();

        Ok((
            total_cost,
            cloudy_area,
            max_incidence_angle,
            min_resolutions_sum,
        ))
    }

    /// Validate the computed objectives against stored values
    fn validate_objectives(&self, problem: &SimsDiscreteProblem) -> PyResult<bool> {
        let (computed_cost, computed_cloudy_area, computed_max_angle, computed_min_res) =
            self.compute_objectives(problem)?;

        let cost_valid = self.cost == computed_cost;
        let cloudy_area_valid = self.cloudy_area == computed_cloudy_area;

        let max_angle_valid = self
            .max_incidence_angle
            .map(|stored| stored == computed_max_angle || stored == -1)
            .unwrap_or(true);

        let min_res_valid = self
            .min_resolutions_sum
            .map(|stored| stored == computed_min_res || stored == -1)
            .unwrap_or(true);

        Ok(cost_valid && cloudy_area_valid && max_angle_valid && min_res_valid)
    }

    /// Fix/recompute the objectives based on the problem instance
    fn fix_objectives(&mut self, problem: &SimsDiscreteProblem) -> PyResult<()> {
        if self.cost < 0 || self.cloudy_area < 0 {
            let (computed_cost, computed_cloudy_area, computed_max_angle, computed_min_res) =
                self.compute_objectives(problem)?;

            if self.cost != computed_cost || self.cost < 0 {
                println!("Fixing cost from {} to {}", self.cost, computed_cost);
                self.cost = computed_cost;
            }

            if self.cloudy_area != computed_cloudy_area || self.cloudy_area < 0 {
                println!(
                    "Fixing cloudy area from {} to {}",
                    self.cloudy_area, computed_cloudy_area
                );
                self.cloudy_area = computed_cloudy_area;
            }

            // Update optional objectives if they were invalid (-1)
            if self.max_incidence_angle.is_none_or(|x| x == -1) {
                self.max_incidence_angle = Some(computed_max_angle);
            }

            if self.min_resolutions_sum.is_none_or(|x| x == -1) {
                self.min_resolutions_sum = Some(computed_min_res);
            }
        }
        Ok(())
    }
}

/// Convert PLS EncodedSolution to Python Solution
fn pls_solution_to_python_solution(
    pls_solution: &pls::solution::EncodedSolution<2>,
    timestamp_us: u64,
) -> Solution {
    // Debug logging: Show raw PLS solution data
    let raw_selected_images: Vec<usize> = pls_solution.selected_images().collect();
    debug!(
        "Converting PLS solution: {} selected images, objectives: {:?}",
        raw_selected_images.len(),
        pls_solution.objectives
    );

    // PLS already uses 0-based indexing internally, no conversion needed
    let selected_images: Vec<usize> = raw_selected_images;

    debug!("Selected images (0-based): {selected_images:?}");

    let cost = pls_solution.objectives[0] as i32;
    let cloudy_area = pls_solution.objectives[1] as i32;

    debug!("Created Python solution: cost={cost}, cloudy_area={cloudy_area}");

    Solution::create(
        selected_images,
        cost,
        cloudy_area,
        timestamp_us,
        None, // max_incidence_angle will be computed later if needed
        None, // min_resolutions_sum will be computed later if needed
    )
}

/// Solves the SIMS problem using Pareto Local Search with advanced parameters
#[pyfunction]
#[pyo3(signature = (sims_instance, timeout_seconds=240.0, max_iterations=50000, is_deterministic=false, initial_population_size=100, neighborhood_size_min=1, neighborhood_size_max=6))]
fn solve_with_pls_advanced(
    sims_instance: &SimsDiscreteProblem,
    timeout_seconds: f64,
    max_iterations: usize,
    is_deterministic: bool,
    initial_population_size: usize,
    neighborhood_size_min: u32,
    neighborhood_size_max: u32,
) -> PyResult<Vec<Solution>> {
    use pls::pareto_local_search::ParetoLocalSearch;
    use pls::solution_set::SolutionSet;
    use pls::solution_set_impl::BTreeSolutionSet;
    use std::ops::RangeInclusive;

    info!(
        "Starting PLS algorithm with parameters: timeout={timeout_seconds}s, max_iterations={max_iterations}, deterministic={is_deterministic}, population_size={initial_population_size}, neighborhood={neighborhood_size_min}..{neighborhood_size_max}"
    );

    // Convert to PLS problem format
    let pls_problem = sims_instance.to_pls_problem();
    debug!(
        "Converted SIMS problem to PLS format: {} images, universe size {}",
        sims_instance.num_images, sims_instance.universe
    );

    debug!("Creating initial population of size {initial_population_size}");

    // Create initial population
    let neighborhood_size_range: RangeInclusive<u32> =
        neighborhood_size_min..=neighborhood_size_max;
    let initial_solution_set = if is_deterministic {
        BTreeSolutionSet::random_with_seed(initial_population_size, &pls_problem, 1_234_567_890)
    } else {
        BTreeSolutionSet::random(initial_population_size, &pls_problem)
    };

    println!("DEBUG: Running PLS with max_iterations={max_iterations}, timeout={timeout_seconds}s");

    // Create and run PLS
    let mut pareto_local_search = ParetoLocalSearch::new(
        &pls_problem,
        &initial_solution_set,
        neighborhood_size_range,
        is_deterministic,
    );

    let timeout = Duration::from_secs_f64(timeout_seconds);
    info!("Starting PLS execution with {max_iterations} iterations timeout");
    let final_solution_set = pareto_local_search.run(max_iterations, timeout);

    info!(
        "PLS completed, processing {} solutions",
        final_solution_set.len()
    );

    // Convert solutions back to Python format
    let final_solutions: Vec<pls::solution::EncodedSolution<2>> =
        final_solution_set.into_iter().collect();

    debug!(
        "Converting {} PLS solutions to Python format",
        final_solutions.len()
    );

    let mut python_solutions = Vec::new();
    for (i, solution) in final_solutions.iter().enumerate() {
        debug!(
            "Processing solution {}: objectives = {:?}",
            i, solution.objectives
        );

        // Get timestamp from explored solutions if available
        let timestamp_us = pareto_local_search
            .explored_solutions
            .get_solution_fingerprint(solution)
            .map(|fp| fp.time.as_micros() as u64)
            .unwrap_or(i as u64 * 1000); // Fallback: use index * 1ms

        let py_solution = pls_solution_to_python_solution(solution, timestamp_us);
        debug!(
            "Converted solution {}: cost={}, cloudy_area={}, selected_images={:?}",
            i,
            py_solution.cost,
            py_solution.cloudy_area,
            py_solution.get_selected_images_list()
        );
        python_solutions.push(py_solution);
    }

    info!(
        "Successfully converted {} solutions to Python format",
        python_solutions.len()
    );

    Ok(python_solutions)
}

/// Solves the SIMS problem using Pareto Local Search with 4 objectives (cost, cloudy area, resolution, incidence angle)
#[pyfunction]
#[pyo3(signature = (sims_instance, timeout_seconds=240.0, max_iterations=50000, is_deterministic=false, initial_population_size=100, neighborhood_size_min=1, neighborhood_size_max=6))]
fn solve_with_pls_multiobjective(
    sims_instance: &SimsDiscreteProblem,
    timeout_seconds: f64,
    max_iterations: usize,
    is_deterministic: bool,
    initial_population_size: usize,
    neighborhood_size_min: u32,
    neighborhood_size_max: u32,
) -> PyResult<Vec<Solution>> {
    use pls::objectives::{
        CloudyAreaObjective, MaxIncidenceAngleObjective, MinResolutionObjective, TotalCostObjective,
    };
    use pls::pareto_local_search::ParetoLocalSearch;
    use pls::solution_set::SolutionSet;
    use pls::solution_set_impl::NdTreeSolutionSet;
    use std::ops::RangeInclusive;

    info!(
        "Starting multi-objective PLS (4D) with parameters: timeout={timeout_seconds}s, max_iterations={max_iterations}, deterministic={is_deterministic}, population_size={initial_population_size}, neighborhood={neighborhood_size_min}..{neighborhood_size_max}"
    );

    // Convert to PLS problem format and create 4D problem with all objectives
    let raw_instance = pls::problem::SIMSProblemInstanceRaw {
        name: "python_instance".to_string(),
        num_images: sims_instance.num_images,
        universe_size: sims_instance.universe,
        images: sims_instance
            .images
            .iter()
            .map(|img| img.iter().map(|&x| x + 1).collect())
            .collect(),
        costs: sims_instance.costs.iter().map(|&c| c as u64).collect(),
        clouds: sims_instance
            .clouds
            .iter()
            .map(|cloud| cloud.iter().map(|&x| x + 1).collect())
            .collect(),
        areas: sims_instance.areas.iter().map(|&a| a as u64).collect(),
        max_cloud_area: sims_instance.max_cloud_area as u64,
        resolution: sims_instance.resolution.iter().map(|&r| r as u64).collect(),
        incidence_angle: sims_instance
            .incidence_angle
            .iter()
            .map(|&i| i as u64)
            .collect(),
    };

    // Create 4D problem with all objectives
    let objective_definitions: Vec<Box<dyn pls::objectives::ObjectiveDefinition<4>>> = vec![
        Box::new(TotalCostObjective { index: 0 }),
        Box::new(CloudyAreaObjective { index: 1 }),
        Box::new(MinResolutionObjective { index: 2 }),
        Box::new(MaxIncidenceAngleObjective { index: 3 }),
    ];

    let pls_problem = pls::problem::Problem::from_raw_with_objective_definitions(
        raw_instance,
        objective_definitions,
    )
    .map_err(|e| PyValueError::new_err(format!("Failed to create 4D problem: {e}")))?;

    debug!(
        "Created 4D PLS problem: {} images, universe size {}",
        sims_instance.num_images, sims_instance.universe
    );

    debug!("Creating initial population of size {initial_population_size}");

    // Create initial population using ND-Tree for 4D optimization
    let neighborhood_size_range: RangeInclusive<u32> =
        neighborhood_size_min..=neighborhood_size_max;
    let initial_solution_set: NdTreeSolutionSet<pls::solution::BitsetEncodedSolution<4>, 4> =
        if is_deterministic {
            NdTreeSolutionSet::random_with_seed(
                initial_population_size,
                &pls_problem,
                1_234_567_890,
            )
        } else {
            NdTreeSolutionSet::random(initial_population_size, &pls_problem)
        };

    println!("DEBUG: Running multi-objective PLS with max_iterations={max_iterations}, timeout={timeout_seconds}s");

    // Create and run PLS
    let mut pareto_local_search = ParetoLocalSearch::new(
        &pls_problem,
        &initial_solution_set,
        neighborhood_size_range,
        is_deterministic,
    );

    let timeout = Duration::from_secs_f64(timeout_seconds);
    info!("Starting multi-objective PLS execution with {max_iterations} iterations timeout");
    let final_solution_set = pareto_local_search.run(max_iterations, timeout);

    info!(
        "Multi-objective PLS completed, processing {} solutions",
        final_solution_set.len()
    );

    // Convert solutions back to Python format
    let final_solutions: Vec<pls::solution::BitsetEncodedSolution<4>> =
        final_solution_set.into_iter().collect();

    debug!(
        "Converting {} 4D PLS solutions to Python format",
        final_solutions.len()
    );

    let mut python_solutions = Vec::new();
    for (i, solution) in final_solutions.iter().enumerate() {
        debug!(
            "Processing 4D solution {}: objectives = {:?}",
            i, solution.objectives
        );

        // Get timestamp from explored solutions if available
        let timestamp_us = pareto_local_search
            .explored_solutions
            .get_solution_fingerprint(solution)
            .map(|fp| fp.time.as_micros() as u64)
            .unwrap_or(i as u64 * 1000); // Fallback: use index * 1ms

        let py_solution = pls_4d_solution_to_python_solution(solution, timestamp_us);
        debug!(
            "Converted 4D solution {}: cost={}, cloudy_area={}, selected_images={:?}",
            i,
            py_solution.cost,
            py_solution.cloudy_area,
            py_solution.get_selected_images_list()
        );
        python_solutions.push(py_solution);
    }

    info!(
        "Successfully converted {} 4D solutions to Python format",
        python_solutions.len()
    );

    Ok(python_solutions)
}

/// Convert 4D PLS solution to Python Solution (includes resolution and incidence angle objectives)
fn pls_4d_solution_to_python_solution(
    pls_solution: &pls::solution::BitsetEncodedSolution<4>,
    timestamp_us: u64,
) -> Solution {
    // Debug logging: Show raw PLS solution data
    let raw_selected_images: Vec<usize> = pls_solution.selected_images().collect();
    debug!(
        "Converting 4D PLS solution: {} selected images, objectives: {:?}",
        raw_selected_images.len(),
        pls_solution.objectives
    );

    let selected_images: Vec<usize> = raw_selected_images;
    debug!("Selected images (0-based): {selected_images:?}");

    let cost = pls_solution.objectives[0] as i32;
    let cloudy_area = pls_solution.objectives[1] as i32;
    let min_resolution_sum = pls_solution.objectives[2] as i32;
    let max_incidence_angle = pls_solution.objectives[3] as i32;

    debug!("Created 4D Python solution: cost={cost}, cloudy_area={cloudy_area}, min_resolution_sum={min_resolution_sum}, max_incidence_angle={max_incidence_angle}");

    Solution::create(
        selected_images,
        cost,
        cloudy_area,
        timestamp_us,
        Some(max_incidence_angle),
        Some(min_resolution_sum),
    )
}

/// Solves the SIMS problem using Pareto Local Search with 4 objectives and advanced parameters
#[pyfunction]
#[pyo3(signature = (sims_instance, timeout_seconds=240.0, max_iterations=50000, is_deterministic=false, initial_population_size=100, neighborhood_size_min=1, neighborhood_size_max=6))]
fn solve_with_pls_multiobjective_advanced(
    sims_instance: &SimsDiscreteProblem,
    timeout_seconds: f64,
    max_iterations: usize,
    is_deterministic: bool,
    initial_population_size: usize,
    neighborhood_size_min: u32,
    neighborhood_size_max: u32,
) -> PyResult<Vec<Solution>> {
    // Call the base multiobjective function with the advanced parameters
    solve_with_pls_multiobjective(
        sims_instance,
        timeout_seconds,
        max_iterations,
        is_deterministic,
        initial_population_size,
        neighborhood_size_min,
        neighborhood_size_max,
    )
}

/// Solves the SIMS problem using Pareto Local Search with default parameters
#[pyfunction]
fn solve_with_pls(sims_instance: &SimsDiscreteProblem) -> PyResult<Vec<Solution>> {
    // Call the advanced function with default parameters
    solve_with_pls_advanced(
        sims_instance,
        240.0, // timeout_seconds
        50000, // max_iterations
        false, // is_deterministic
        100,   // initial_population_size
        1,     // neighborhood_size_min
        6,     // neighborhood_size_max
    )
}

/// Solves the SIMS problem using PLS and generates 2D plot artifacts
#[pyfunction]
#[pyo3(signature = (sims_instance, plot_output_path, timeout_seconds=240.0, max_iterations=50000, is_deterministic=false, initial_population_size=100, neighborhood_size_min=1, neighborhood_size_max=6))]
fn solve_with_pls_and_plot_2d(
    sims_instance: &SimsDiscreteProblem,
    plot_output_path: &str,
    timeout_seconds: f64,
    max_iterations: usize,
    is_deterministic: bool,
    initial_population_size: usize,
    neighborhood_size_min: u32,
    neighborhood_size_max: u32,
) -> PyResult<Vec<Solution>> {
    use pls::pareto_local_search::ParetoLocalSearch;
    use pls::solution_set::SolutionSet;
    use pls::solution_set_impl::BTreeSolutionSet;

    // Convert to PLS problem format
    let pls_problem = sims_instance.to_pls_problem();
    
    // Create initial population using the same pattern as existing functions
    let neighborhood_size_range = neighborhood_size_min..=neighborhood_size_max;
    let initial_solution_set = if is_deterministic {
        BTreeSolutionSet::random_with_seed(initial_population_size, &pls_problem, 1_234_567_890)
    } else {
        BTreeSolutionSet::random(initial_population_size, &pls_problem)
    };

    // Create and run PLS
    let mut pareto_local_search = ParetoLocalSearch::new(
        &pls_problem,
        &initial_solution_set,
        neighborhood_size_range,
        is_deterministic,
    );

    let timeout = Duration::from_secs_f64(timeout_seconds);
    let final_solution_set = pareto_local_search.run(max_iterations, timeout);

    // Generate 2D plot
    #[cfg(feature = "plotting")]
    {
        let objective_names = pls_problem.objective_names();
        pls::plotting::draw_solutions_plot(&pareto_local_search.explored_solutions, &objective_names);
        
        // If a custom path is provided, try to move the generated file
        if plot_output_path != "pareto_solutions_2d.svg" {
            if let Err(e) = std::fs::rename("pareto_solutions_2d.svg", plot_output_path) {
                log::warn!("Failed to move plot to {plot_output_path}: {e}");
            }
        }
    }

    // Convert solutions to Python format (same as existing function)
    let final_solutions: Vec<pls::solution::EncodedSolution<2>> = final_solution_set.into_iter().collect();
    let mut python_solutions = Vec::new();
    
    for (i, solution) in final_solutions.iter().enumerate() {
        let timestamp_us = pareto_local_search
            .explored_solutions
            .get_solution_fingerprint(solution)
            .map(|fp| fp.time.as_micros() as u64)
            .unwrap_or(i as u64 * 1000);
        
        let py_solution = pls_solution_to_python_solution(solution, timestamp_us);
        python_solutions.push(py_solution);
    }

    Ok(python_solutions)
}

/// Solves the SIMS problem using multiobjective PLS and generates 4D plot grid artifacts
#[pyfunction]
#[pyo3(signature = (sims_instance, plot_output_path, timeout_seconds=240.0, max_iterations=50000, is_deterministic=false, initial_population_size=100, neighborhood_size_min=1, neighborhood_size_max=6))]
fn solve_with_pls_multiobjective_and_plot_4d(
    sims_instance: &SimsDiscreteProblem,
    plot_output_path: &str,
    timeout_seconds: f64,
    max_iterations: usize,
    is_deterministic: bool,
    initial_population_size: usize,
    neighborhood_size_min: u32,
    neighborhood_size_max: u32,
) -> PyResult<Vec<Solution>> {
    use pls::objectives::{
        CloudyAreaObjective, MaxIncidenceAngleObjective, MinResolutionObjective, TotalCostObjective,
    };
    use pls::pareto_local_search::ParetoLocalSearch;
    use pls::solution_set::SolutionSet;
    use pls::solution_set_impl::NdTreeSolutionSet;

    // Convert to 4D PLS problem format (same pattern as existing multiobjective function)
    let raw_instance = pls::problem::SIMSProblemInstanceRaw {
        name: "python_4d_instance".to_string(),
        num_images: sims_instance.num_images,
        universe_size: sims_instance.universe,
        images: sims_instance
            .images
            .iter()
            .map(|img| img.iter().map(|&x| x + 1).collect())
            .collect(),
        costs: sims_instance.costs.iter().map(|&c| c as u64).collect(),
        clouds: sims_instance
            .clouds
            .iter()
            .map(|cloud| cloud.iter().map(|&x| x + 1).collect())
            .collect(),
        areas: sims_instance.areas.iter().map(|&a| a as u64).collect(),
        max_cloud_area: sims_instance.max_cloud_area as u64,
        resolution: sims_instance.resolution.iter().map(|&r| r as u64).collect(),
        incidence_angle: sims_instance
            .incidence_angle
            .iter()
            .map(|&i| i as u64)
            .collect(),
    };

    // Create 4D problem with all objectives
    let objective_definitions: Vec<Box<dyn pls::objectives::ObjectiveDefinition<4>>> = vec![
        Box::new(TotalCostObjective { index: 0 }),
        Box::new(CloudyAreaObjective { index: 1 }),
        Box::new(MinResolutionObjective { index: 2 }),
        Box::new(MaxIncidenceAngleObjective { index: 3 }),
    ];

    let pls_problem = pls::problem::Problem::from_raw_with_objective_definitions(
        raw_instance,
        objective_definitions,
    )
    .map_err(|e| PyValueError::new_err(format!("Failed to create 4D problem: {e}")))?;

    // Create initial population using ND-Tree for 4D optimization
    let neighborhood_size_range = neighborhood_size_min..=neighborhood_size_max;
    let initial_solution_set: NdTreeSolutionSet<pls::solution::BitsetEncodedSolution<4>, 4> =
        if is_deterministic {
            NdTreeSolutionSet::random_with_seed(initial_population_size, &pls_problem, 1_234_567_890)
        } else {
            NdTreeSolutionSet::random(initial_population_size, &pls_problem)
        };

    // Create and run PLS
    let mut pareto_local_search = ParetoLocalSearch::new(
        &pls_problem,
        &initial_solution_set,
        neighborhood_size_range,
        is_deterministic,
    );

    let timeout = Duration::from_secs_f64(timeout_seconds);
    let final_solution_set = pareto_local_search.run(max_iterations, timeout);

    // Generate 4D plot grid
    #[cfg(feature = "plotting")]
    {
        let objective_names = pls_problem.objective_names();
        pls::plotting::draw_solutions_plot(&pareto_local_search.explored_solutions, &objective_names);
        
        // If a custom path is provided, try to move the generated file
        if plot_output_path != "pareto_solutions_grid.svg" {
            if let Err(e) = std::fs::rename("pareto_solutions_grid.svg", plot_output_path) {
                log::warn!("Failed to move plot to {plot_output_path}: {e}");
            }
        }
    }

    // Convert solutions to Python format (same pattern as existing multiobjective function)
    let final_solutions: Vec<pls::solution::BitsetEncodedSolution<4>> = final_solution_set.into_iter().collect();
    let mut python_solutions = Vec::new();
    
    for (i, solution) in final_solutions.iter().enumerate() {
        let timestamp_us = pareto_local_search
            .explored_solutions
            .get_solution_fingerprint(solution)
            .map(|fp| fp.time.as_micros() as u64)
            .unwrap_or(i as u64 * 1000);
        
        let py_solution = pls_4d_solution_to_python_solution(solution, timestamp_us);
        python_solutions.push(py_solution);
    }

    Ok(python_solutions)
}

/// A Python module implemented in Rust.
#[pymodule]
fn sims_problem(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize logging bridge from Rust to Python
    pyo3_log::init();

    m.add_function(wrap_pyfunction!(solve_with_pls, m)?)?;
    m.add_function(wrap_pyfunction!(solve_with_pls_advanced, m)?)?;
    m.add_function(wrap_pyfunction!(solve_with_pls_multiobjective, m)?)?;
    m.add_function(wrap_pyfunction!(solve_with_pls_multiobjective_advanced, m)?)?;
    m.add_function(wrap_pyfunction!(solve_with_pls_and_plot_2d, m)?)?;
    m.add_function(wrap_pyfunction!(solve_with_pls_multiobjective_and_plot_4d, m)?)?;
    m.add_class::<SimsDiscreteProblem>()?;
    m.add_class::<Solution>()?;

    Ok(())
}
