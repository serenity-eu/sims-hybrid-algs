use log::debug;
use pls::problem::{Problem, SIMSProblemInstanceRaw};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::{
    prelude::*,
    types::{PyDict, PyType},
};
use std::collections::{HashMap, HashSet};
use std::fs;

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
    pub fn to_pls_problem(&self) -> Problem<2> {
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

        let raw_instance = SIMSProblemInstanceRaw {
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
        let pls_problem = Problem::from_raw(raw_instance);
        log::info!("Successfully created PLS problem");
        pls_problem
    }
}
