use crate::{
    problem::{Problem, ScaledObjectiveDeltas},
    residual_problem::ResidualProblem,
    solution::{ResidualSolutionCapable, SIMSSolution, SIMSSolutionTrait},
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

impl<const D: usize> SIMSSolutionTrait<D> for ResidualSolution<D> {
    // Never be used, so leaving stub. TODO: Remove
    fn random(_problem: &Problem<D>) -> Self {
        unimplemented!()
    }

    fn random_with_seed(_problem: &Problem<D>, _seed: u64) -> Self {
        unimplemented!()
    }

    fn from_selected_images(_selected_images_vec: &[usize], _problem: &Problem<D>) -> Self {
        unimplemented!(
            "ResidualSolution should be created using from_selected_images with ResidualProblem"
        )
    }

    fn is_dominated(&self, other: &Self) -> bool {
        // Solution is dominated by other solution iff it is greater or equal in all objectives, with at least one objective being strictly greater
        let dominance_relation = self.objectives.partial_cmp(&other.objectives);
        return dominance_relation == Some(std::cmp::Ordering::Greater);
    }

    fn is_weakly_dominated(&self, other: &Self) -> bool {
        // Solution is weakly dominated by other solution iff greater or equal in all objectives
        let dominance_relation = self.objectives.partial_cmp(&other.objectives);
        return (dominance_relation == Some(std::cmp::Ordering::Greater))
            || (dominance_relation == Some(std::cmp::Ordering::Equal));
    }

    fn selected_images(&self) -> Vec<usize> {
        self.selected_images.clone()
    }

    fn unselected_images(&self) -> Vec<usize> {
        // For ResidualSolution, this doesn't make much sense but we need to implement it
        // Return empty vector since ResidualSolution only tracks selected images
        Vec::new()
    }

    fn is_image_selected(&self, image_index: usize) -> bool {
        self.selected_images.contains(&image_index)
    }

    fn num_selected_images(&self) -> usize {
        self.selected_images.len()
    }

    fn clear_parts_counts(&self) -> &[usize] {
        // ResidualSolution doesn't track this information
        // Return empty slice as this operation doesn't make sense for ResidualSolution
        &[]
    }

    fn element_coverage(&self) -> &[usize] {
        // ResidualSolution doesn't track this information
        // Return empty slice as this operation doesn't make sense for ResidualSolution
        &[]
    }

    fn add_image(&mut self, _image_index: usize, _problem: &Problem<D>) {
        unimplemented!("ResidualSolution doesn't support dynamic modification")
    }

    fn remove_image(&mut self, _image_index: usize, _problem: &Problem<D>) {
        unimplemented!("ResidualSolution doesn't support dynamic modification")
    }

    fn scaled_image_objective_deltas(
        &self,
        _images: &[usize],
        _problem: &Problem<D>,
    ) -> Vec<ScaledObjectiveDeltas> {
        unimplemented!("ResidualSolution doesn't support scaled objective deltas computation")
    }

    fn find_best_image_to_add(&self, _problem: &Problem<D>) -> Option<usize> {
        unimplemented!("ResidualSolution doesn't support dynamic modification")
    }

    fn find_best_image_to_remove(&self, _problem: &Problem<D>) -> Option<usize> {
        unimplemented!("ResidualSolution doesn't support dynamic modification")
    }

    fn get_neighborhood(&self, _problem: &Problem<D>) -> Vec<Self> {
        unimplemented!("ResidualSolution doesn't support neighborhood generation")
    }

    fn to_debug_solution(&self) -> SIMSSolution {
        SIMSSolution {
            selected_images: self.selected_images.clone(),
        }
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

        assert!(D == 2, "ResidualSolution only supports D = 2 for now");
        self.objectives[0] = cost;
        self.objectives[1] = cloudy_area;
    }
}
