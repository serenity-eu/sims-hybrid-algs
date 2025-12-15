use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::{prelude::*, types::PyDict};
use std::{
    collections::HashSet,
    hash::{Hash, Hasher},
    time::Duration,
};
use pareto::Objectives;

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
    pub cost: Option<u64>,
    #[pyo3(get, set)]
    pub cloudy_area: Option<u64>,
    #[pyo3(get, set)]
    pub timestamp: Duration, // Using Duration for timedelta compatibility
    #[pyo3(get, set)]
    pub max_incidence_angle: Option<u64>,
    #[pyo3(get, set)]
    pub min_resolutions_sum: Option<u64>,
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
        // Sort the images first to ensure consistent hashing regardless of HashSet iteration order
        let mut sorted_images: Vec<usize> = self.selected_images.iter().copied().collect();
        sorted_images.sort_unstable();
        for image in sorted_images {
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
        cost: Option<u64>,
        cloudy_area: Option<u64>,
        timestamp_us: u64,
        max_incidence_angle: Option<u64>,
        min_resolutions_sum: Option<u64>,
    ) -> PyResult<Self> {
        // Count how many objectives are set
        let mut objectives_set = 0;
        if cost.is_some() {
            objectives_set += 1;
        }
        if cloudy_area.is_some() {
            objectives_set += 1;
        }
        if max_incidence_angle.is_some() {
            objectives_set += 1;
        }
        if min_resolutions_sum.is_some() {
            objectives_set += 1;
        }
        
        // Require at least 2 objectives to be set
        if objectives_set < 2 {
            return Err(PyValueError::new_err(
                format!(
                    "At least 2 objectives must be set (got {}). Provide non-None values for at least 2 of: cost, cloudy_area, max_incidence_angle, min_resolutions_sum",
                    objectives_set
                )
            ));
        }
        
        Ok(Self {
            selected_images: selected_images.into_iter().collect(),
            cost,
            cloudy_area,
            timestamp: Duration::from_micros(timestamp_us),
            max_incidence_angle,
            min_resolutions_sum,
        })
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
        let mut images: Vec<usize> = self.selected_images.iter().cloned().collect();
        images.sort_unstable();
        images
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
    fn compute_objectives(&self, problem: &SimsDiscreteProblem) -> PyResult<(u64, u64, u64, u64)> {
        // Total cost
        let total_cost: u64 = self
            .selected_images
            .iter()
            .map(|&i| {
                if i >= problem.num_images {
                    0 // Handle out of bounds gracefully
                } else {
                    problem.costs[i] as u64
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
        let cloudy_area: u64 = (0..problem.universe)
            .filter(|&u| !clear_parts.contains(&u))
            .map(|u| problem.areas[u] as u64)
            .sum();

        // Max incidence angle
        let max_incidence_angle = self
            .selected_images
            .iter()
            .filter(|&&i| i < problem.num_images)
            .map(|&i| problem.incidence_angle[i] as u64)
            .max()
            .unwrap_or(0);

        // Min resolutions sum
        let min_resolutions_sum: u64 = (0..problem.universe)
            .map(|u| {
                self.selected_images
                    .iter()
                    .filter(|&&i| i < problem.num_images && problem.images[i].contains(&u))
                    .map(|&i| problem.resolution[i] as u64)
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

        let cost_valid = self.cost
            .map(|stored| stored == computed_cost)
            .unwrap_or(true); // If not set, consider valid

        let cloudy_area_valid = self.cloudy_area
            .map(|stored| stored == computed_cloudy_area)
            .unwrap_or(true); // If not set, consider valid

        let max_angle_valid = self
            .max_incidence_angle
            .map(|stored| stored == computed_max_angle || stored == u64::MAX)
            .unwrap_or(true);

        let min_res_valid = self
            .min_resolutions_sum
            .map(|stored| stored == computed_min_res || stored == u64::MAX)
            .unwrap_or(true);

        Ok(cost_valid && cloudy_area_valid && max_angle_valid && min_res_valid)
    }

    /// Fix/recompute the objectives based on the problem instance
    fn fix_objectives(&mut self, problem: &SimsDiscreteProblem) -> PyResult<()> {
        let (computed_cost, computed_cloudy_area, computed_max_angle, computed_min_res) =
            self.compute_objectives(problem)?;

        // Only fix if the objective is set and incorrect
        if let Some(cost) = self.cost {
            if cost != computed_cost {
                println!("Fixing cost from {} to {}", cost, computed_cost);
                self.cost = Some(computed_cost);
            }
        }

        if let Some(cloudy_area) = self.cloudy_area {
            if cloudy_area != computed_cloudy_area {
                println!(
                    "Fixing cloudy area from {} to {}",
                    cloudy_area, computed_cloudy_area
                );
                self.cloudy_area = Some(computed_cloudy_area);
            }
        }

        // Update optional objectives if they were invalid (u64::MAX)
        if self.max_incidence_angle.is_none_or(|x| x == u64::MAX) {
            self.max_incidence_angle = Some(computed_max_angle);
        }

        if self.min_resolutions_sum.is_none_or(|x| x == u64::MAX) {
            self.min_resolutions_sum = Some(computed_min_res);
        }

        Ok(())
    }
    
    /// Get objectives as a 2D array (cost, cloudy_area)
    pub fn objectives_2d(&self) -> Objectives<2> {
        [
            self.cost.unwrap_or(u64::MAX),
            self.cloudy_area.unwrap_or(u64::MAX)
        ]
    }
    
    /// Get objectives as a 3D array (cost, cloudy_area, max_incidence_angle)
    pub fn objectives_3d(&self) -> Objectives<3> {
        let max_angle = self.max_incidence_angle.unwrap_or(u64::MAX);
        [
            self.cost.unwrap_or(u64::MAX),
            self.cloudy_area.unwrap_or(u64::MAX),
            max_angle
        ]
    }
    
    /// Get objectives as a 4D array (cost, cloudy_area, max_incidence_angle, min_resolutions_sum)
    pub fn objectives_4d(&self) -> Objectives<4> {
        let max_angle = self.max_incidence_angle.unwrap_or(u64::MAX);
        let min_res = self.min_resolutions_sum.unwrap_or(u64::MAX);
        [
            self.cost.unwrap_or(u64::MAX),
            self.cloudy_area.unwrap_or(u64::MAX),
            max_angle,
            min_res
        ]
    }
}

/// Result structure containing both final Pareto solutions and all explored solutions
/// for comprehensive visualization and analysis
#[pyclass]
#[derive(Clone, Debug)]
pub struct SolvingResult {
    /// The final Pareto-optimal solutions
    #[pyo3(get, set)]
    pub final_solutions: Vec<Solution>,
    /// Binary trace archive of the optimization process (None if trace=False)
    #[pyo3(get, set)]
    pub trace: Option<Vec<u8>>,
    /// Chrome tracing JSON profiling data (None if profiling_trace=False)
    #[pyo3(get, set)]
    pub profiling_trace_data: Option<Vec<u8>>,
}

#[pymethods]
impl SolvingResult {
    #[new]
    pub fn new(final_solutions: Vec<Solution>) -> Self {
        Self {
            final_solutions,
            trace: None,
            profiling_trace_data: None,
        }
    }

    #[staticmethod]
    pub fn with_trace(
        final_solutions: Vec<Solution>,
        trace: Vec<u8>,
    ) -> Self {
        Self {
            final_solutions,
            trace: Some(trace),
            profiling_trace_data: None,
        }
    }

    #[staticmethod]
    pub fn with_trace_and_profiling(
        final_solutions: Vec<Solution>,
        trace: Vec<u8>,
        profiling_trace_data: Vec<u8>,
    ) -> Self {
        Self {
            final_solutions,
            trace: Some(trace),
            profiling_trace_data: Some(profiling_trace_data),
        }
    }
}
