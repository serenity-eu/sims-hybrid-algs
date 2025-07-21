use crate::{
    objectives::SolutionEvaluator,
    residual_problem::ResidualProblem,
    solution::{ImageSet, ResidualSolutionCapable, SIMSCore, SIMSSolution},
};
use pareto::{HasObjectives, MoSolution};
use std::fmt::Debug;

#[derive(Clone, Eq, Hash)]
#[allow(clippy::derived_hash_with_manual_eq)]
pub struct ResidualSolution<const D: usize> {
    pub selected_images: Vec<usize>,
    pub objectives: pareto::Objectives<D>,
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
            .finish()
    }
}

#[expect(
    clippy::non_canonical_partial_ord_impl,
    reason = "Compare only first objective"
)]
impl<const D: usize> PartialOrd for ResidualSolution<D> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.objectives[0].partial_cmp(&other.objectives[0])
    }
}

impl<const D: usize> Ord for ResidualSolution<D> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.objectives[0].cmp(&other.objectives[0])
    }
}

impl<const D: usize> HasObjectives<D> for ResidualSolution<D> {
    fn objectives(&self) -> &pareto::Objectives<D> {
        &self.objectives
    }
}

impl<const D: usize> MoSolution<D> for ResidualSolution<D> {}

// Implement ImageSet trait for ResidualSolution
impl<const D: usize> ImageSet for ResidualSolution<D> {
    fn selected_images(&self) -> Vec<usize> {
        self.selected_images.clone()
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

// Implement SolutionEvaluator trait for ResidualSolution
impl<const D: usize> SolutionEvaluator<D> for ResidualSolution<D> {
    fn clear_parts_counts(&self) -> &[usize] {
        // ResidualSolution doesn't maintain clear_parts_counts
        // Return empty slice as it's not used in residual solution context
        &[]
    }

    fn element_coverage(&self) -> &[usize] {
        // ResidualSolution doesn't maintain element_coverage
        // Return empty slice as it's not used in residual solution context
        &[]
    }
}

impl<const D: usize> ResidualSolution<D> {
    #[must_use]
    pub fn from_selected_images<S: ResidualSolutionCapable<D> + Clone>(
        selected_images: Vec<usize>,
        residual_problem: &ResidualProblem<S, D>,
    ) -> Self {
        let mut solution = Self {
            selected_images,
            objectives: [0; D],
        };
        solution.compute_objectives(residual_problem);
        solution
    }

    fn compute_objectives<S: ResidualSolutionCapable<D> + Clone>(
        &mut self,
        residual_problem: &ResidualProblem<S, D>,
    ) {
        // For multiobjective support, we need to use the problem's objective definitions
        if let Some(objective_definitions) = &residual_problem.problem.objective_definitions {
            // Use generic objective calculation with definitions
            for (i, objective_def) in objective_definitions.iter().enumerate() {
                self.objectives[i] = objective_def.calculate_value(self, residual_problem.problem);
            }
        } else {
            // Legacy fallback for 2D case
            assert!(D == 2, "ResidualSolution without objective definitions only supports D = 2");
            
            // Compute cost as sum of costs of selected images
            let cost: u64 = self
                .selected_images
                .iter()
                .map(|&image_index| residual_problem.all_images[image_index].cost)
                .sum();

            // To compute cloudy area, we first use information from unmodified solution to determine which parts are clear
            let mut clear_parts = residual_problem
                .original_clear_parts_counts
                .iter()
                .map(|&count| count > 0)
                .collect::<Vec<_>>();

            // Then we add information from selected images
            self.selected_images.iter().for_each(|&image_index| {
                residual_problem.all_images[image_index]
                    .clear_parts
                    .iter()
                    .for_each(|&clear_part| {
                        clear_parts[clear_part] = true;
                    });
            });

            // We compute cloudy area as sum of areas of all elements that are not clear
            let cloudy_area: u64 = residual_problem
                .uncovered_elements
                .iter()
                .zip(clear_parts.iter())
                .filter_map(
                    |(element, &is_clear)| {
                        if is_clear { None } else { Some(element.area) }
                    },
                )
                .sum();

            self.objectives[0] = cost;
            self.objectives[1] = cloudy_area;
        }
    }
}
