use std::collections::HashMap;

use itertools::Itertools;
use log::{debug, trace};
use pareto::{HasObjectives, ParetoFront};

use crate::{
    problem::{Element, Image, Problem},
    residual_solution::ResidualSolution,
    solution::{ImageSet, MergeableWithResidual},
};

pub struct ResidualProblem<'a, R: MergeableWithResidual<D> + Clone, const D: usize> {
    /// Original solutions with unmodified image only
    pub unmodified_solution: R,
    /// Images - candidates to be removed
    pub removal_candidates_indices: Vec<usize>,
    /// All images participating in residual problem
    pub all_images: Vec<Image>,
    /// Uncovered elements
    pub uncovered_elements: Vec<Element>,
    /// Uncovered elements clear parts counts
    pub original_clear_parts_counts: Vec<usize>,
    /// Original problem instance
    pub problem: &'a Problem<R, D>,
}

impl<'a, R: MergeableWithResidual<D> + Clone, const D: usize> ResidualProblem<'a, R, D> {
    #[must_use]
    pub fn new(
        unmodified_solution: R,
        removal_candidates_original_indices: Vec<usize>,
        addition_candidates: Vec<usize>,
        uncovered_elements_indices: Vec<usize>,
        original_clear_parts_counts: Vec<usize>,
        problem: &'a Problem<R, D>,
    ) -> Self {
        debug!("######################################################");
        debug!(
            "######## RESIDUAL PROBLEM removed: {removal_candidates_original_indices:?} added: {addition_candidates:?} ######"
        );
        debug!("######## base: {unmodified_solution:?} ########");
        debug!("######################################################");

        let uncovered_elements_map: HashMap<usize, usize> = uncovered_elements_indices
            .iter()
            .enumerate()
            .map(|(i, &x)| (x, i))
            .collect::<HashMap<_, _>>();

        let all_images: Vec<Image> = removal_candidates_original_indices
            .clone()
            .into_iter()
            .merge(addition_candidates)
            .map(|image_index| {
                let image = &problem.images[image_index];
                let parts = image
                    .parts
                    .iter()
                    .filter_map(|&part| uncovered_elements_map.get(&part))
                    .copied()
                    .collect::<Vec<_>>();
                let clear_parts = image
                    .clear_parts
                    .iter()
                    .filter_map(|&part| uncovered_elements_map.get(&part))
                    .copied()
                    .collect::<Vec<_>>();
                Image::new(image_index, image.cost, parts, clear_parts)
            })
            .collect();

        let all_images_map: HashMap<usize, usize> = all_images
            .iter()
            .enumerate()
            .map(|(i, image)| (image.index, i))
            .collect::<HashMap<_, _>>();

        let removal_candidates_indices: Vec<usize> = removal_candidates_original_indices
            .into_iter()
            .filter_map(|image_index| all_images_map.get(&image_index))
            .copied()
            .collect();

        let mut uncovered_elements: Vec<Element> = uncovered_elements_indices
            .into_iter()
            .map(|element_index| {
                let element = &problem.universe[element_index];
                Element {
                    area: element.area,
                    images: Vec::new(),
                }
            })
            .collect();

        all_images
            .iter()
            .enumerate()
            .for_each(|(image_index, image)| {
                image.parts.iter().for_each(|&part| {
                    uncovered_elements[part].images.push(image_index);
                });
            });

        ResidualProblem {
            unmodified_solution,
            removal_candidates_indices,
            all_images,
            uncovered_elements,
            original_clear_parts_counts,
            problem,
        }
    }

    pub fn solve_with_backtracing<S: ParetoFront<'a, ResidualSolution<D>> + Default>(
        &mut self,
    ) -> MergedSolutionIter<'_, R, D> {
        let mut non_dominated_residual_set: S = S::default();

        let element_images = self
            .uncovered_elements
            .iter()
            .map(|element| element.images.iter());

        for cover in element_images.multi_cartesian_product() {
            let mut unique_cover: Vec<usize> = cover.into_iter().copied().collect();
            unique_cover.sort_unstable();
            unique_cover.dedup();

            let residual_solution =
                ResidualSolution::from_selected_images(&unique_cover, self.problem);

            let was_added = non_dominated_residual_set.try_insert(&residual_solution);

            trace!("#####################################################");
            trace!(
                "######### RESIDUAL: OBJECTIVES {:?} | IMAGES {:?} {} #########################",
                residual_solution.objectives(),
                residual_solution.selected_images(),
                if was_added { "ADDED" } else { "NOT ADDED" }
            );
        }

        let solutions_iter: Vec<ResidualSolution<D>> =
            non_dominated_residual_set.into_iter().collect();

        trace!("*****************************************************");
        for solution in &solutions_iter {
            trace!(
                "****** NONDOMINANT RESIDUAL: OBJECTIVES {:?} | IMAGES {:?} ******",
                solution.objectives(),
                solution.selected_images()
            );
        }
        trace!("*****************************************************");

        MergedSolutionIter {
            unmodified_solution: &self.unmodified_solution,
            solutions_iter,
            residual_problem: self,
            problem: self.problem,
        }
    }

    // // Check whether selected images cover all elements
    // fn do_selected_images_cover(
    //     &self,
    //     selected_images: &[usize],
    //     coverage_bitmaps: &[ElementSubset],
    //     all_elements_mask: ElementSubset,
    // ) -> bool {
    //     selected_images.iter().fold(ElementSubset::default(), |acc, &image_index| {
    //         acc | coverage_bitmaps[image_index]
    //     }) == all_elements_mask
    // }

    /*
    pub fn solve_with_bitmaps(&self) -> MergedSolutionIter {
        const K: usize = 3;
        const MAX_IMAGES: usize = 20;
        const MAX_ELEMENTS: usize = 128;
        type ImageSubset = BitArr!(for MAX_IMAGES);
        type ElementSubset = BitArr!(for MAX_ELEMENTS);

        let images_count = self.all_images.len();
        let elements_count = self.uncovered_elements.len();

        let all_elements_mask: ElementSubset = ElementSubset::default();
        all_elements_mask[0..elements_count].fill(true);
        let all_images_mask: ImageSubset = ImageSubset::default();
        all_images_mask[0..images_count].fill(true);

        let coverage_bitmaps: Vec<ElementSubset> = self.uncovered_elements.iter().map(|element| {
            let mut bitmap = ElementSubset::default();
            element.images.iter().for_each(|&image_index| {
                bitmap.set(image_index, true);
            });
            bitmap
        }).collect();

        let clear_part_bitmaps: Vec<ElementSubset> = vec![ElementSubset::default(); self.uncovered_elements.len()];
        self.all_images.iter().enumerate().for_each(|(image_index, image)| {
            image.clear_parts.iter().for_each(|&clear_part| {
                clear_part_bitmaps[clear_part].set(image_index, true);
            })
        });

        let mut non_dominated_residual_set: BTreeSet<ResidualSolution<D>> = BTreeSet::new();

        let mut subset_storage = [bitarr![0; MAX_IMAGES]; K];
        let mut current_indices = [0; K];
        let mut current_recursion_level: usize = 0;
        let mut recursive_subsets: [&mut BitSlice; K] = subset_storage.iter_mut().map(|subset| subset.get_mut(0..images_count).unwrap()).collect();

        loop {
            if current_indices[current_recursion_level] == 0 {
                // Select first element empty subset
                recursive_subsets[current_recursion_level].set(current_indices[current_recursion_level], true);
            } else {
                // Select next element in subset
                recursive_subsets[current_recursion_level].set(current_indices[current_recursion_level]-1, false);
                recursive_subsets[current_recursion_level].set(current_indices[current_recursion_level], true);
            }
            current_indices[current_recursion_level] += 1;

            let selected_images: Vec<usize> = recursive_subsets[current_recursion_level].iter().enumerate().filter_map(|(i, &is_selected)| {
                if is_selected {
                    Some(i)
                } else {
                    None
                }
            }).collect();

            if self.do_selected_images_cover(&selected_images, &coverage_bitmaps, &all_elements_mask) {
                let residual_solution = ResidualSolution::<D>::from_selected_images(selected_images.clone(), self);

                if !non_dominated_residual_set.contains(&residual_solution) {
                // Simple dominance check - only add if not dominated by existing solutions
                let is_dominated = non_dominated_residual_set
                    .iter()
                    .any(|existing| residual_solution.is_dominated_by(existing.objectives()));

                let was_added = if !is_dominated {
                    // Remove solutions dominated by the new one
                    non_dominated_residual_set.retain(|existing| !existing.is_dominated_by(residual_solution.objectives()));
                    non_dominated_residual_set.insert(residual_solution.clone());
                    true
                } else {
                    false
                };
                    trace!("#####################################################");
                    trace!(
                        "######### RESIDUAL: OBJECTIVES {:?} | IMAGES {:?} {} #########################",
                        residual_solution.objectives,
                        residual_solution.selected_images,
                        if was_added { "ADDED" } else { "NOT ADDED" }
                    );
                }
            }
            break;
        }

        MergedSolutionIter {
            unmodified_solution: &self.unmodified_solution,
            solutions_iter: Vec::new(),
            residual_problem: self,
            problem: self.problem,
        }
    }
    */

    pub fn solve<S: ParetoFront<'a, ResidualSolution<D>> + Default>(
        &mut self,
    ) -> MergedSolutionIter<'_, R, D> {
        let mut non_dominated_residual_set: S = S::default();

        let images_indices = (0..self.all_images.len()).collect::<Vec<_>>();

        let combs_0_to_5 = (0..=5).flat_map(|i| images_indices.iter().combinations(i));
        // Horrible brute force but it should work
        // for image_combination in images_indices.iter().powerset() {
        for image_combination in combs_0_to_5 {
            if image_combination
                .iter()
                .copied()
                .eq(self.removal_candidates_indices.iter())
            {
                trace!("Skipping image combination as it is equal to original one");
                continue;
            }

            let mut covered_elements = vec![false; self.uncovered_elements.len()];
            for &image_index in &image_combination {
                self.all_images[*image_index]
                    .parts
                    .iter()
                    .for_each(|&part| {
                        covered_elements[part] = true;
                    });
            }

            if !covered_elements.iter().all(|&is_covered| is_covered) {
                continue;
            }

            let selected_images: Vec<usize> = image_combination.iter().copied().copied().collect();
            let residual_solution =
                ResidualSolution::from_selected_images(&selected_images, self.problem);

            let was_added = non_dominated_residual_set.try_insert(&residual_solution);

            trace!("#####################################################");
            trace!(
                "######### RESIDUAL: OBJECTIVES {:?} | IMAGES {:?} {} #########################",
                residual_solution.objectives(),
                residual_solution.selected_images(),
                if was_added { "ADDED" } else { "NOT ADDED" }
            );
        }

        let solutions_iter: Vec<ResidualSolution<D>> =
            non_dominated_residual_set.into_iter().collect();

        trace!("*****************************************************");
        for solution in &solutions_iter {
            trace!(
                "****** NONDOMINANT RESIDUAL: OBJECTIVES {:?} | IMAGES {:?} ******",
                solution.objectives(),
                solution.selected_images()
            );
        }
        trace!("*****************************************************");

        MergedSolutionIter {
            unmodified_solution: &self.unmodified_solution,
            solutions_iter,
            residual_problem: self,
            problem: self.problem,
        }
    }
}

pub struct MergedSolutionIter<'a, R: MergeableWithResidual<D> + Clone, const D: usize> {
    unmodified_solution: &'a R,
    solutions_iter: Vec<ResidualSolution<D>>,
    residual_problem: &'a ResidualProblem<'a, R, D>,
    problem: &'a Problem<R, D>,
}

impl<R: MergeableWithResidual<D> + Clone, const D: usize> Iterator
    for MergedSolutionIter<'_, R, D>
{
    type Item = R;

    fn next(&mut self) -> Option<Self::Item> {
        let residual_solution = self.solutions_iter.pop()?;
        let mut new_solution = self.unmodified_solution.clone();
        new_solution.merge_residual_solution(
            &residual_solution,
            self.residual_problem,
            self.problem,
        );
        return Some(new_solution);
    }
}
