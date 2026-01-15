use std::collections::HashMap;

use fixedbitset::FixedBitSet;
use itertools::Itertools;
use log::{debug, trace};
use pareto::{HasObjectives, ParetoFront};

use crate::{
    problem::SetCoverProblem,
    residual_solution::ResidualSolution,
    solution::{ImageSet, MergeableWithResidual},
    util::UnionIterator,
};

pub struct ResidualProblem<R, P, const D: usize>
where
    R: MergeableWithResidual<P, D> + Clone,
    P: SetCoverProblem<D>,
{
    /// Owned copy of original solution with unmodified images only
    pub unmodified_solution: R,
    /// Condensed indices corresponding to `removed_images` in `image_index_map`.
    /// Used to skip the "no-op" residual combination that would reconstruct the original solution.
    pub condensed_original_removed_images: FixedBitSet,
    /// Map from condensed image index to original image index
    pub image_map_condensed_to_original: Vec<usize>,
    /// Map from condensed element index to original element index
    pub element_map_condensed_to_original: Vec<usize>,
    /// Condensed images as bitsets (each bitset represents which condensed elements the image covers)
    pub condensed_images: Vec<FixedBitSet>,
    /// Iterator state for generating combinations
    combination_iter: Box<dyn Iterator<Item = Vec<usize>>>,
    /// Phantom data to use P type parameter
    _phantom: std::marker::PhantomData<P>,
}

impl<R, P, const D: usize> ResidualProblem<R, P, D>
where
    R: MergeableWithResidual<P, D> + Clone,
    P: SetCoverProblem<D>,
{
    /// Creates a new residual problem from a solution with removed images.
    ///
    /// # Panics
    ///
    /// Panics if a removed image index is not found in the constructed image index map.
    /// This should never happen if the inputs are valid.
    #[must_use]
    pub fn new(
        unmodified_solution: R,
        removed_images: &[usize],
        addition_candidates: &[usize],
        uncovered_elements_indices: Vec<usize>,
        problem: &P,
    ) -> Self {
        debug!("######################################################");
        debug!("######## RESIDUAL PROBLEM removed images: {removed_images:?} ######");
        debug!("######## RESIDUAL PROBLEM addition candidates: {addition_candidates:?} ######");
        debug!("######## base: {unmodified_solution:?} ########");
        debug!("######################################################");

        // Build element index map (condensed -> original)
        let element_map_condensed_to_original = uncovered_elements_indices;

        // Create reverse map (original -> condensed) for fast lookup
        let element_map_original_to_condensed: HashMap<usize, usize> =
            element_map_condensed_to_original
                .iter()
                .enumerate()
                .map(|(condensed_idx, &original_idx)| (original_idx, condensed_idx))
                .collect();

        // Build image index map as a UNION of both ordered lists.
        // NOTE: `util::UnionIterator::union()` is a merge-based union, so it assumes:
        // - both inputs are sorted ascending
        // - both inputs are unique (set semantics)
        // Keep ordering stable for deterministic behavior.
        #[cfg(debug_assertions)]
        {
            debug_assert!(
                removed_images.windows(2).all(|w| w[0] < w[1]),
                "removed_images must be a sorted unique set"
            );
            debug_assert!(
                addition_candidates.windows(2).all(|w| w[0] < w[1]),
                "addition_candidates must be a sorted unique set"
            );
        }
        let image_map_condensed_to_original: Vec<usize> = removed_images
            .iter()
            .copied()
            .union(addition_candidates.iter().copied())
            .collect();

        // Build reverse map (original image index -> condensed index) for removed-images bitset.
        let image_map_original_to_condensed: HashMap<usize, usize> =
            image_map_condensed_to_original
                .iter()
                .enumerate()
                .map(|(condensed_idx, &original_idx)| (original_idx, condensed_idx))
                .collect();

        // Bitset of condensed indices corresponding to the original `removed_images`.
        let condensed_original_removed_images = removed_images
            .iter()
            .map(|&img| {
                image_map_original_to_condensed
                    .get(&img)
                    .copied()
                    .expect("removed image must be in image_map_original_to_condensed")
            })
            .collect::<FixedBitSet>();

        // Build condensed images (bitsets representing element coverage)
        let condensed_images: Vec<FixedBitSet> = image_map_condensed_to_original
            .iter()
            .map(|&original_img_idx| {
                // Map original element indices to condensed indices
                problem
                    .image_elements(original_img_idx)
                    .filter_map(|original_elem_idx| {
                        element_map_original_to_condensed
                            .get(&original_elem_idx)
                            .copied()
                    })
                    .collect()
            })
            .collect();

        let m = image_map_condensed_to_original.len();
        let combination_iter = Box::new((0..=5).flat_map(move |i| (0..m).combinations(i)));

        Self {
            unmodified_solution,
            condensed_original_removed_images,
            image_map_condensed_to_original,
            element_map_condensed_to_original,
            condensed_images,
            combination_iter,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Check if the given selection of condensed images forms a set cover
    /// Uses efficient bitset operations
    #[must_use]
    pub fn is_set_cover(&self, selected_images: &FixedBitSet) -> bool {
        let num_elements = self.element_map_condensed_to_original.len();
        let mut covered = FixedBitSet::with_capacity(num_elements);

        // Union all coverage bitsets for selected images
        for img_idx in selected_images.ones() {
            covered.union_with(&self.condensed_images[img_idx]);
        }

        // Check if all elements are covered
        covered.is_full()
    }

    /// Get images that cover a specific element
    #[must_use]
    pub fn images_covering_element(&self, element_idx: usize) -> Vec<usize> {
        self.condensed_images
            .iter()
            .enumerate()
            .filter_map(|(img_idx, coverage)| coverage.contains(element_idx).then_some(img_idx))
            .collect()
    }

    pub fn solve_with_backtracing<'a, S: ParetoFront<'a, ResidualSolution<D>> + Default>(
        &mut self,
        problem: &P,
        timer: &crate::timer::Timer,
    ) -> Vec<ResidualSolution<D>> {
        let mut non_dominated_residual_set: S = S::default();

        // Build list of images covering each element for cartesian product
        let element_images: Vec<Vec<usize>> = (0..self.element_map_condensed_to_original.len())
            .map(|elem_idx| self.images_covering_element(elem_idx))
            .collect();

        let element_images_refs: Vec<_> = element_images.iter().map(|v| v.iter()).collect();

        for cover in element_images_refs.into_iter().multi_cartesian_product() {
            let mut unique_cover: Vec<usize> = cover.into_iter().copied().collect();
            unique_cover.sort_unstable();
            unique_cover.dedup();

            // Keep condensed indices for ResidualSolution
            let residual_solution = ResidualSolution::from_selected_images_condensed(
                &unique_cover,
                &self.image_map_condensed_to_original,
                problem,
                timer,
            );

            let was_added = non_dominated_residual_set.try_insert(&residual_solution);

            trace!("#####################################################");
            trace!(
                "######### RESIDUAL: OBJECTIVES {:?} | IMAGES {:?} {} #########################",
                residual_solution.objectives(),
                residual_solution.selected_images().collect::<Vec<_>>(),
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
                solution.selected_images().collect::<Vec<_>>()
            );
        }
        trace!("*****************************************************");

        solutions_iter
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

    /// Get the next valid residual solution from the combination iterator
    /// Returns None when all combinations have been exhausted
    pub fn solve_next(
        &mut self,
        problem: &P,
        timer: &crate::timer::Timer,
    ) -> Option<ResidualSolution<D>> {
        // Iterate through combinations to find the next valid one
        while let Some(combination) = self.combination_iter.next() {
            // Check if this combination matches the initially selected images (removed candidates)
            let selected: FixedBitSet = combination.iter().copied().collect();

            // Skip the original combination (would reconstruct the original solution)
            if selected == self.condensed_original_removed_images {
                continue;
            }

            if !self.is_set_cover(&selected) {
                continue;
            }

            // Create and return the residual solution
            let residual_solution = ResidualSolution::from_selected_images_condensed(
                &combination,
                &self.image_map_condensed_to_original,
                problem,
                timer,
            );

            trace!(
                "RESIDUAL: OBJ {:?} | IMG {:?}",
                residual_solution.objectives(),
                residual_solution.selected_images().collect::<Vec<_>>(),
            );

            return Some(residual_solution);
        }

        // All combinations exhausted
        None
    }

    #[tracing::instrument(skip(self, problem, timer), level = "debug")]
    pub fn solve<'a, S: ParetoFront<'a, ResidualSolution<D>> + Default>(
        &'a mut self,
        problem: &'a P,
        timer: &crate::timer::Timer,
        partial_trackers: R::Trackers,
    ) -> MergedSolutionIter<'a, R, P, D> {
        use tracing::debug_span;

        let init_span = debug_span!("initialize_residual_solve");
        let _init_guard = init_span.enter();

        let m = self.image_map_condensed_to_original.len();
        // Use 0..m range for combinations to avoid vector allocation
        let combs_0_to_5 = (0..=5).flat_map(|i| (0..m).combinations(i));

        let mut non_dominated_residual_set: S = S::default();

        let enumerate_span = debug_span!("enumerate_combinations");
        let _enum_guard = enumerate_span.enter();

        for image_combination in combs_0_to_5 {
            // Check if this combination matches the initially selected images (removed candidates)
            let test_bitset: FixedBitSet = image_combination.iter().copied().collect();
            if test_bitset == self.condensed_original_removed_images {
                trace!("Skipping image combination as it is equal to original one");
                continue;
            }

            // Check coverage
            let is_valid_combination = {
                let selected: FixedBitSet = image_combination.iter().copied().collect();
                self.is_set_cover(&selected)
            };

            if !is_valid_combination {
                continue;
            }

            // Create solution - image_combination is Vec<usize> (condensed indices)
            let residual_solution = ResidualSolution::from_selected_images_condensed(
                &image_combination,
                &self.image_map_condensed_to_original,
                problem,
                timer,
            );

            // Add to Pareto front
            let was_added = non_dominated_residual_set.try_insert(&residual_solution);

            trace!(
                "RESIDUAL: OBJ {:?} | IMG {:?} {}",
                residual_solution.objectives(),
                residual_solution.selected_images().collect::<Vec<_>>(),
                if was_added { "ADDED" } else { "SKIP" }
            );
        }

        let solutions_iter: Vec<ResidualSolution<D>> =
            non_dominated_residual_set.into_iter().collect();

        MergedSolutionIter {
            unmodified_solution: &self.unmodified_solution,
            solutions_iter,
            residual_problem: self,
            problem,
            partial_trackers,
        }
    }
}

pub struct MergedSolutionIter<'a, R, P, const D: usize>
where
    R: MergeableWithResidual<P, D> + Clone,
    P: SetCoverProblem<D>,
{
    pub unmodified_solution: &'a R,
    pub solutions_iter: Vec<ResidualSolution<D>>,
    pub residual_problem: &'a ResidualProblem<R, P, D>,
    pub problem: &'a P,
    pub partial_trackers: R::Trackers,
}
/*
pub struct ResidualSolutionIter<'b, 'a, R, P, const D: usize, S>
where
    R: MergeableWithResidual<P, D> + Clone,
    P: SetCoverProblem<D>,
    S: ParetoFront<'a, ResidualSolution<D>> + Default,
{
    residual_problem: &'b ResidualProblem<'a, R, P, D>,
    timer: &'b crate::timer::Timer,
    partial_trackers: R::Trackers,
    images_indices: Vec<usize>,
    combination_iter: Option<Box<dyn Iterator<Item = Vec<&'b usize>> + 'b>>,
    non_dominated_set: S,
    non_dominated_iter: Option<Box<dyn Iterator<Item = ResidualSolution<D>> + 'a>>,
    unmodified_solution: &'a R,
    problem: &'a P,
}

impl<'b, 'a, R, P, const D: usize, S> Iterator for ResidualSolutionIter<'b, 'a, R, P, D, S>
where
    R: MergeableWithResidual<P, D> + Clone,
    P: SetCoverProblem<D>,
    S: ParetoFront<'a, ResidualSolution<D>> + Default,
{
    type Item = R;

    fn next(&mut self) -> Option<Self::Item> {
        // If we already have non-dominated solutions collected, yield from them
        if let Some(iter) = &mut self.non_dominated_iter {
            if let Some(residual_solution) = iter.next() {
                let mut new_solution = self.unmodified_solution.clone();
                new_solution.merge_residual_solution(
                    &residual_solution,
                    self.residual_problem,
                    self.problem,
                    self.partial_trackers.clone(),
                );
                return Some(new_solution);
            }
        }

        // Initialize combination iterator on first call
        if self.combination_iter.is_none() {
            let combs_0_to_5 = (0..=5).flat_map(|i| self.images_indices.iter().combinations(i));
            self.combination_iter = Some(Box::new(combs_0_to_5));
        }

        // Enumerate combinations and build non-dominated set
        if let Some(iter) = &mut self.combination_iter {
            for image_combination in iter.by_ref() {
                // Check if this combination matches the initially selected images (removed candidates)
                let test_bitset: FixedBitSet = image_combination.iter().copied().copied().collect();
                if test_bitset == self.residual_problem.condensed_original_removed_images {
                    trace!("Skipping image combination as it is equal to original one");
                    continue;
                }

                let selected: FixedBitSet = image_combination.iter().copied().copied().collect();
                if !self.residual_problem.is_set_cover(&selected) {
                    continue;
                }

                // Keep condensed indices for ResidualSolution
                let condensed_images: Vec<usize> = image_combination
                    .iter()
                    .map(|&&condensed_idx| condensed_idx)
                    .collect();
                let residual_solution = ResidualSolution::from_selected_images_condensed(
                    &condensed_images,
                    &self.residual_problem.image_index_map,
                    self.problem,
                    self.timer,
                );

                let was_added = self.non_dominated_set.try_insert(&residual_solution);

                trace!("#####################################################");
                trace!(
                    "######### RESIDUAL: OBJECTIVES {:?} | IMAGES {:?} {} #########################",
                    residual_solution.objectives(),
                    residual_solution.selected_images().collect::<Vec<_>>(),
                    if was_added { "ADDED" } else { "NOT ADDED" }
                );
            }
        }

        // All combinations processed, now yield from non-dominated set
        trace!("*****************************************************");
        trace!("****** Processing non-dominated solutions ******");

        let solutions: Vec<ResidualSolution<D>> = std::mem::take(&mut self.non_dominated_set).into_iter().collect();
        for solution in &solutions {
            trace!(
                "****** NONDOMINANT RESIDUAL: OBJECTIVES {:?} | IMAGES {:?} ******",
                solution.objectives(),
                solution.selected_images().collect::<Vec<_>>()
            );
        }
        trace!("*****************************************************");

        self.non_dominated_iter = Some(Box::new(solutions.into_iter()));

        // Yield first solution from the iterator
        if let Some(iter) = &mut self.non_dominated_iter {
            if let Some(residual_solution) = iter.next() {
                let mut new_solution = self.unmodified_solution.clone();
                new_solution.merge_residual_solution(
                    &residual_solution,
                    self.residual_problem,
                    self.problem,
                    self.partial_trackers.clone(),
                );
                return Some(new_solution);
            }
        }

        None
    }
}
*/

impl<R, P, const D: usize> Iterator for MergedSolutionIter<'_, R, P, D>
where
    R: MergeableWithResidual<P, D> + Clone,
    P: SetCoverProblem<D>,
{
    type Item = R;

    fn next(&mut self) -> Option<Self::Item> {
        let residual_solution = self.solutions_iter.pop()?;
        let mut new_solution = self.unmodified_solution.clone();
        new_solution.merge_residual_solution(
            &residual_solution,
            self.residual_problem,
            self.problem,
            &mut self.partial_trackers,
        );
        return Some(new_solution);
    }
}
