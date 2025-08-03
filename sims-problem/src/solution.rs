use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::{prelude::*, types::PyDict};
use std::{
    collections::HashSet,
    hash::{Hash, Hasher},
    time::Duration,
};

use crate::problem::SimsDiscreteProblem;

/// Represents a solution to the SIMS problem
///
/// A solution contains:
/// - The set of selected satellite images
/// - Associated objective values (cost, cloudy area coverage)
/// - Optional additional objectives (resolution, incidence angle)
/// - Timestamp indicating when the solution was found
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
    pub timestamp: Duration, // Using Duration for timedelta compatibility
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

impl Hash for Solution {
    fn hash<H: Hasher>(&self, state: &mut H) {
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
    pub fn create(
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
            timestamp: Duration::from_micros(timestamp_us),
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
            dict.set_item("timestamp", self.timestamp)?; // PyO3 automatically converts Duration to timedelta
            dict.set_item("max_incidence_angle", self.max_incidence_angle)?;
            dict.set_item("min_resolutions_sum", self.min_resolutions_sum)?;

            Ok(dict.into())
        })
    }

    /// Get selected images as a list
    pub fn get_selected_images_list(&self) -> Vec<usize> {
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
