use std::collections::HashMap;

use fixedbitset::FixedBitSet;
use itertools::Itertools;
use log::{debug, trace};
use pareto::{HasObjectives, ParetoFront};

use crate::{
    problem::Problem,
    residual_solution::ResidualSolution,
    solution::{ImageSet, MergeableWithResidual},
};

pub struct ResidualProblem<'a, R: MergeableWithResidual<D> + Clone, const D: usize> {
    /// Original solutions with unmodified image only
    pub unmodified_solution: R,
    /// Trackers reflecting the state after removing images (before adding candidates)
    pub partial_trackers: R::Trackers,
    /// Condensed selected images (bitset) - maps to `removal_candidates_original_indices`
    pub condensed_selected_images: FixedBitSet,
    /// Map from condensed image index to original image index
    pub image_index_map: Vec<usize>,
    /// Map from condensed element index to original element index  
    pub element_index_map: Vec<usize>,
    /// Condensed images as bitsets (each bitset represents which condensed elements the image covers)
    pub condensed_images: Vec<FixedBitSet>,
    /// Uncovered elements clear parts counts
    pub original_clear_parts_counts: Vec<usize>,
    /// Original problem instance
    pub problem: &'a Problem<R, D>,
}

impl<'a, R: MergeableWithResidual<D> + Clone, const D: usize> ResidualProblem<'a, R, D> {
    #[must_use]
    pub fn new(
        unmodified_solution: R,
        partial_trackers: R::Trackers,
        removal_candidates_original_indices: &[usize],
        addition_candidates: &[usize],
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

        // Build element index map (condensed -> original)
        let element_index_map = uncovered_elements_indices;
        let num_elements = element_index_map.len();
        
        // Create reverse map (original -> condensed) for fast lookup
        let element_map_reverse: HashMap<usize, usize> = element_index_map
            .iter()
            .enumerate()
            .map(|(condensed_idx, &original_idx)| (original_idx, condensed_idx))
            .collect();

        // Build image index map: [removed_images..., addition_candidates...]
        let image_index_map: Vec<usize> = removal_candidates_original_indices
            .iter()
            .copied()
            .chain(addition_candidates.iter().copied())
            .collect();
        
        let num_images = image_index_map.len();
        
        // Build condensed selected images bitset (only removed images are initially selected)
        let mut condensed_selected_images = FixedBitSet::with_capacity(num_images);
        for i in 0..removal_candidates_original_indices.len() {
            condensed_selected_images.insert(i);
        }

        // Build condensed images (bitsets representing element coverage)
        let mut condensed_images = Vec::with_capacity(num_images);
        for &original_img_idx in &image_index_map {
            let mut element_coverage = FixedBitSet::with_capacity(num_elements);
            
            // Map original element indices to condensed indices
            for &original_elem_idx in &problem.images[original_img_idx].parts {
                if let Some(&condensed_elem_idx) = element_map_reverse.get(&original_elem_idx) {
                    element_coverage.insert(condensed_elem_idx);
                }
            }
            
            condensed_images.push(element_coverage);
        }

        ResidualProblem {
            unmodified_solution,
            partial_trackers,
            condensed_selected_images,
            image_index_map,
            element_index_map,
            condensed_images,
            original_clear_parts_counts,
            problem,
        }
    }

    /// Check if the given selection of condensed images forms a set cover
    /// Uses efficient bitset operations
    #[must_use]
    pub fn is_set_cover(&self, selected_images: &FixedBitSet) -> bool {
        let num_elements = self.element_index_map.len();
        let mut covered = FixedBitSet::with_capacity(num_elements);
        
        // Union all coverage bitsets for selected images
        for img_idx in selected_images.ones() {
            covered.union_with(&self.condensed_images[img_idx]);
        }
        
        // Check if all elements are covered
        covered.count_ones(..) == num_elements
    }

    /// Get images that cover a specific element
    #[must_use]
    pub fn images_covering_element(&self, element_idx: usize) -> Vec<usize> {
        self.condensed_images
            .iter()
            .enumerate()
            .filter(|(_, coverage)| coverage[element_idx])
            .map(|(img_idx, _)| img_idx)
            .collect()
    }

    pub fn solve_with_backtracing<S: ParetoFront<'a, ResidualSolution<D>> + Default>(
        &mut self,
        timer: &crate::timer::Timer,
    ) -> MergedSolutionIter<'_, R, D> {
        let mut non_dominated_residual_set: S = S::default();

        // Build list of images covering each element for cartesian product
        let element_images: Vec<Vec<usize>> = (0..self.element_index_map.len())
            .map(|elem_idx| self.images_covering_element(elem_idx))
            .collect();

        let element_images_refs: Vec<_> = element_images.iter().map(|v| v.iter()).collect();

        for cover in element_images_refs.into_iter().multi_cartesian_product() {
            let mut unique_cover: Vec<usize> = cover.into_iter().copied().collect();
            unique_cover.sort_unstable();
            unique_cover.dedup();

            // Map condensed indices back to original indices
            let original_indices: Vec<usize> = unique_cover
                .iter()
                .map(|&condensed_idx| self.image_index_map[condensed_idx])
                .collect();

            let residual_solution =
                ResidualSolution::from_selected_images(&original_indices, self.problem, timer);

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
                let residual_solution = ResidualSolution::<D>::from_selected_images(selected_images.clone(), self, timer);

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
        timer: &crate::timer::Timer,
    ) -> MergedSolutionIter<'_, R, D> {
        use tracing::debug_span;
        
        let init_span = debug_span!("initialize_residual_solve");
        let init_guard = init_span.enter();
        let mut non_dominated_residual_set: S = S::default();

        let images_indices = (0..self.image_index_map.len()).collect::<Vec<_>>();
        drop(init_guard);

        let combinations_span = debug_span!("generate_all_combinations", num_images = self.image_index_map.len());
        let combs_0_to_5 = {
            let _comb_guard = combinations_span.enter();
            (0..=5).flat_map(|i| images_indices.iter().combinations(i))
        };

        let enumerate_span = debug_span!("enumerate_combinations");
        let enum_guard = enumerate_span.enter();
        
        // Horrible brute force but it should work
        // for image_combination in images_indices.iter().powerset() {
        for image_combination in combs_0_to_5 {
            let check_skip_span = debug_span!("check_skip_combination");
            let should_skip = {
                let _skip_guard = check_skip_span.enter();
                // Check if this combination matches the initially selected images (removed candidates)
                let mut test_bitset = FixedBitSet::with_capacity(self.image_index_map.len());
                for &&idx in &image_combination {
                    test_bitset.insert(idx);
                }
                test_bitset == self.condensed_selected_images
            };
            
            if should_skip {
                trace!("Skipping image combination as it is equal to original one");
                continue;
            }

            let coverage_check_span = debug_span!("check_coverage");
            let is_valid_combination = {
                let _coverage_guard = coverage_check_span.enter();
                let mut selected = FixedBitSet::with_capacity(self.image_index_map.len());
                for &&img_idx in &image_combination {
                    selected.insert(img_idx);
                }
                self.is_set_cover(&selected)
            };

            if !is_valid_combination {
                continue;
            }

            let create_solution_span = debug_span!("create_residual_solution");
            let residual_solution = {
                let _create_guard = create_solution_span.enter();
                // Map condensed indices back to original indices
                let selected_images: Vec<usize> = image_combination
                    .iter()
                    .map(|&&condensed_idx| self.image_index_map[condensed_idx])
                    .collect();
                ResidualSolution::from_selected_images(&selected_images, self.problem, timer)
            };

            let insert_span = debug_span!("try_insert_to_pareto");
            let was_added = {
                let _insert_guard = insert_span.enter();
                non_dominated_residual_set.try_insert(&residual_solution)
            };

            trace!("#####################################################");
            trace!(
                "######### RESIDUAL: OBJECTIVES {:?} | IMAGES {:?} {} #########################",
                residual_solution.objectives(),
                residual_solution.selected_images(),
                if was_added { "ADDED" } else { "NOT ADDED" }
            );
        }
        drop(enum_guard);

        let collect_span = debug_span!("collect_solutions");
        let solutions_iter: Vec<ResidualSolution<D>> = {
            let _collect_guard = collect_span.enter();
            non_dominated_residual_set.into_iter().collect()
        };

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
