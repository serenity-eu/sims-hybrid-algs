use crate::{
    problem::Problem,
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
    pub fn from_selected_images<T: ImageSet<D>>(
        selected_images: &[usize],
        problem: &Problem<T, D>,
        timer: &Timer,
    ) -> Self {
        let mut solution = Self {
            selected_images: selected_images.to_vec(),
            objectives: [0; D],
            timestamp: timer.elapsed(),
        };
        // Calculate objectives directly for residual solution
        for i in 0..D {
            // Cast solution to ResidualSolution for calculate_value
            let residual_sol = Self {
                selected_images: solution.selected_images.clone(),
                objectives: [0; D],
                timestamp: solution.timestamp,
            };
            solution.objectives[i] = problem.objective(i).calculate_value(&residual_sol, unsafe {
                // SAFETY: Problem<T, D> and Problem<Self, D> have identical memory layout
                // and Self implements ImageSet<D> just like T
                &*std::ptr::from_ref::<Problem<T, D>>(problem).cast::<Problem<Self, D>>()
            });
        }
        solution
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
    fn selected_images(&self) -> Vec<usize> {
        self.selected_images.clone()
    }

    fn unselected_images(&self) -> Vec<usize> {
        (0..self.selected_images.len())
            .filter(|&i| !self.selected_images.contains(&i))
            .collect()
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
impl<const D: usize> SIMSCore<D> for ResidualSolution<D> {
    fn to_debug_solution(&self) -> SIMSSolution {
        SIMSSolution {
            selected_images: self.selected_images.clone(),
        }
    }

    fn objectives_mut(&mut self) -> &mut pareto::Objectives<D> {
        &mut self.objectives
    }
}
