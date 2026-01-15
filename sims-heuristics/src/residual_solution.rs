use crate::{
    problem::{Problem, SetCoverProblem},
    solution::{ImageSet, SIMSCore, SIMSSolution},
    timer::Timer,
};
use pareto::{HasObjectives, MoSolution};
use std::{fmt::Debug, time::Duration};

#[derive(Clone, Eq, Hash)]
#[allow(clippy::derived_hash_with_manual_eq)]
pub struct ResidualSolution<const D: usize> {
    pub selected_images: Vec<usize>,
    pub objectives: pareto::Objectives<D>,
    pub timestamp: Duration,
}

impl<const D: usize> ResidualSolution<D> {
    #[must_use]
    pub fn from_selected_images<P>(selected_images: &[usize], problem: &P, timer: &Timer) -> Self
    where
        P: SetCoverProblem<D>,
    {
        let mut solution = Self {
            selected_images: selected_images.to_vec(),
            objectives: [0; D],
            timestamp: timer.elapsed(),
        };
        // Calculate objectives directly for residual solution
        for i in 0..D {
            solution.objectives[i] = problem.objective(i).calculate_value(&solution, problem);
        }
        solution
    }

    /// Create `ResidualSolution` from condensed indices (used by `ResidualProblem`)
    /// This stores condensed indices (0..N) internally but calculates objectives using original indices
    #[must_use]
    pub fn from_selected_images_condensed<P>(
        condensed_indices: &[usize],
        image_index_map: &[usize],
        problem: &P,
        timer: &Timer,
    ) -> Self
    where
        P: SetCoverProblem<D>,
    {
        // Map condensed indices to original indices for objective calculation
        let original_indices: Vec<usize> = condensed_indices
            .iter()
            .map(|&condensed_idx| image_index_map[condensed_idx])
            .collect();

        // Create a temporary solution with original indices for objective calculation
        let temp_solution = Self {
            selected_images: original_indices,
            objectives: [0; D],
            timestamp: timer.elapsed(),
        };

        let mut objectives = [0; D];
        for (i, obj) in objectives.iter_mut().enumerate().take(D) {
            *obj = problem
                .objective(i)
                .calculate_value(&temp_solution, problem);
        }

        // Return solution with condensed indices stored
        Self {
            selected_images: condensed_indices.to_vec(),
            objectives,
            timestamp: timer.elapsed(),
        }
    }
}

impl<const D: usize> PartialEq for ResidualSolution<D> {
    fn eq(&self, other: &Self) -> bool {
        self.selected_images == other.selected_images
    }
}

impl<const D: usize> Debug for ResidualSolution<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SIMSResidualSolution")
            .field("selected_images", &self.selected_images)
            .field("objectives", &self.objectives)
            .field("timestamp", &self.timestamp)
            .finish()
    }
}

impl<const D: usize> HasObjectives<D> for ResidualSolution<D> {
    fn objectives(&self) -> &pareto::Objectives<D> {
        &self.objectives
    }
}

impl<const D: usize> MoSolution<D> for ResidualSolution<D> {}

// Implement ImageSet<D> trait for ResidualSolution
impl<const D: usize> ImageSet<D> for ResidualSolution<D> {
    fn selected_images(&self) -> impl Iterator<Item = usize> {
        self.selected_images.iter().copied()
    }

    fn unselected_images(&self) -> impl Iterator<Item = usize> {
        let selected = self.selected_images.clone();
        (0..selected.len()).filter(move |&i| !selected.contains(&i))
    }

    fn is_image_selected(&self, image_index: usize) -> bool {
        self.selected_images.contains(&image_index)
    }

    fn num_selected_images(&self) -> usize {
        self.selected_images.len()
    }

    fn set_image(&mut self, image_index: usize, selected: bool) {
        if selected && !self.selected_images.contains(&image_index) {
            self.selected_images.push(image_index);
        } else if !selected {
            self.selected_images.retain(|&x| x != image_index);
        }
    }
}

// Implement SIMSCore trait for ResidualSolution
impl<const D: usize> SIMSCore<Problem<Self, D>, D> for ResidualSolution<D> {
    fn to_debug_solution(&self) -> SIMSSolution {
        SIMSSolution {
            selected_images: self.selected_images.clone(),
        }
    }

    fn objectives_mut(&mut self) -> &mut pareto::Objectives<D> {
        &mut self.objectives
    }
}
