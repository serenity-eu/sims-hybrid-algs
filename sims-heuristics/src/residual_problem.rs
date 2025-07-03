use std::collections::HashMap;

use itertools::Itertools;
use log::{debug, trace};

use crate::{
    problem::{Element, Image, Problem},
    residual_solution::ResidualSolution,
    solution::EncodedSolution,
    solution_set::SolutionSet,
    solution_set_impl::BTreeSolutionSet,
};

pub struct ResidualProblem<'a, const D: usize> {
    /// Original solutions with unmodified image only
    pub unmodified_solution: EncodedSolution<D>,
    /// Images - candidates to be removed
    pub removal_candidates_indices: Vec<usize>,
    /// All images participating in residual problem
    pub all_images: Vec<Image>,
    /// Uncovered elements
    pub uncovered_elements: Vec<Element>,
    /// Uncovered elements clear parts counts
    pub original_clear_parts_counts: Vec<usize>,
    /// Original problem instance
    pub problem: &'a Problem<D>,
}

impl<'a, const D: usize> ResidualProblem<'a, D> {
    #[must_use]
    pub fn new(
        unmodified_solution: EncodedSolution<D>,
        removal_candidates_original_indices: Vec<usize>,
        addition_candidates: Vec<usize>,
        uncovered_elements_indices: Vec<usize>,
        original_clear_parts_counts: Vec<usize>,
        problem: &'a Problem<D>,
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

    pub fn solve_with_backtracing(&mut self) -> MergedSolutionIter<'_, D> {
        let mut non_dominated_residual_set: BTreeSolutionSet<ResidualSolution<D>, D> =
            BTreeSolutionSet::new("residual".to_string());

        let element_images = self
            .uncovered_elements
            .iter()
            .map(|element| element.images.iter());

        for cover in element_images.multi_cartesian_product() {
            let mut unique_cover: Vec<usize> = cover.into_iter().copied().collect();
            unique_cover.sort_unstable();
            unique_cover.dedup();

            let residual_solution = ResidualSolution::<D>::from_selected_images(unique_cover, self);

            if !non_dominated_residual_set.contains(&residual_solution) {
                let was_added = non_dominated_residual_set.try_add(&residual_solution);
                trace!("#####################################################");
                trace!(
                    "######### RESIDUAL: OBJECTIVES {:?} | IMAGES {:?} {} #########################",
                    residual_solution.objectives,
                    residual_solution.selected_images,
                    if was_added { "ADDED" } else { "NOT ADDED" }
                );
            }
        }

        let solutions_iter: Vec<ResidualSolution<D>> =
            non_dominated_residual_set.into_iter().collect();

        trace!("*****************************************************");
        for solution in &solutions_iter {
            trace!(
                "****** NONDOMINANT RESIDUAL: OBJECTIVES {:?} | IMAGES {:?} ******",
                solution.objectives, solution.selected_images
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

        let mut non_dominated_residual_set: BTreeSolutionSet<ResidualSolution> =
            BTreeSolutionSet::new("residual".to_string());

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
                    let was_added = non_dominated_residual_set.try_add(&residual_solution);
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

    pub fn solve(&mut self) -> MergedSolutionIter<'_, D> {
        let mut non_dominated_residual_set: BTreeSolutionSet<ResidualSolution<D>, D> =
            BTreeSolutionSet::new("residual".to_string());

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
                ResidualSolution::<D>::from_selected_images(selected_images, self);
            let was_added = non_dominated_residual_set.try_add(&residual_solution);
            trace!("#####################################################");
            trace!(
                "######### RESIDUAL: OBJECTIVES {:?} | IMAGES {:?} {} #########################",
                residual_solution.objectives,
                residual_solution.selected_images,
                if was_added { "ADDED" } else { "NOT ADDED" }
            );
        }

        let solutions_iter: Vec<ResidualSolution<D>> =
            non_dominated_residual_set.into_iter().collect();

        trace!("*****************************************************");
        for solution in &solutions_iter {
            trace!(
                "****** NONDOMINANT RESIDUAL: OBJECTIVES {:?} | IMAGES {:?} ******",
                solution.objectives, solution.selected_images
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

pub struct MergedSolutionIter<'a, const D: usize> {
    unmodified_solution: &'a EncodedSolution<D>,
    solutions_iter: Vec<ResidualSolution<D>>,
    residual_problem: &'a ResidualProblem<'a, D>,
    problem: &'a Problem<D>,
}

impl<const D: usize> Iterator for MergedSolutionIter<'_, D> {
    type Item = EncodedSolution<D>;

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
