use log::trace;
use pareto::{MoSolution, ParetoFront, Random, RandomCollection};

#[derive(Clone)]
pub struct VecSolutionSet<T, const D: usize>
where
    T: MoSolution<D> + PartialEq + Sized,
{
    name: &'static str,
    last_added_position: usize,
    vec_set: Vec<T>,
}

impl<T, const D: usize> VecSolutionSet<T, D>
where
    T: MoSolution<D> + PartialEq + Sized + Clone,
{
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            last_added_position: 0,
            vec_set: Vec::new(),
        }
    }

    fn binary_search_objectives(&self, solution: &T) -> Result<usize, usize> {
        self.vec_set
            .binary_search_by(|s| s.objectives().cmp(solution.objectives()))
    }

    /// Check if an exact duplicate exists (same objectives AND same `selected_images`).
    /// Uses binary search to find objective range, then checks only solutions with matching objectives.
    /// Returns true if duplicate found, false otherwise.
    fn contains_exact_duplicate(&self, solution: &T) -> bool {
        match self.binary_search_objectives(solution) {
            Ok(pos) => {
                // Found at least one solution with same objectives
                // Check backward from pos for all solutions with same objectives
                let mut check_pos = pos;
                loop {
                    if self.vec_set[check_pos] == *solution {
                        return true;
                    }
                    if check_pos == 0 {
                        break;
                    }
                    check_pos -= 1;
                    if self.vec_set[check_pos].objectives() != solution.objectives() {
                        break;
                    }
                }
                // Check forward from pos+1 for all solutions with same objectives
                check_pos = pos + 1;
                while check_pos < self.vec_set.len()
                    && self.vec_set[check_pos].objectives() == solution.objectives()
                {
                    if self.vec_set[check_pos] == *solution {
                        return true;
                    }
                    check_pos += 1;
                }
                false
            }
            Err(_) => {
                // No solution with same objectives exists
                false
            }
        }
    }
}

impl<T, const D: usize> ParetoFront<'_, T> for VecSolutionSet<T, D>
where
    T: MoSolution<D> + PartialEq + Sized + Clone,
{
    type Iter<'b>
        = std::slice::Iter<'b, T>
    where
        T: 'b;

    fn new(name: &'static str) -> Self {
        Self {
            name,
            last_added_position: 0,
            vec_set: Vec::new(),
        }
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.vec_set.iter()
    }

    fn contains(&self, solution: &T) -> bool {
        self.contains_exact_duplicate(solution)
    }

    fn try_insert(&mut self, solution: &T) -> bool {
        // Insert if set is empty
        if self.vec_set.is_empty() {
            self.vec_set.push(solution.clone());
            return true;
        }

        let was_inserted;

        match self.binary_search_objectives(solution) {
            Ok(position) | Err(position) => {
                // Check for exact duplicate among solutions with same objectives
                if position < self.vec_set.len()
                    && self.vec_set[position].objectives() == solution.objectives()
                    && self.contains_exact_duplicate(solution)
                {
                    return false;
                }
                // Check if new solution is dominated by ANY existing solution
                if self
                    .vec_set
                    .iter()
                    .any(|s| solution.is_covered_by(s.objectives()))
                {
                    was_inserted = false;
                } else {
                    self.vec_set.insert(position, solution.clone());
                    self.last_added_position = position;
                    was_inserted = true;
                }
            }
        }
        if was_inserted {
            let size_before = self.vec_set.len();
            let mut to_remove = Vec::new();

            // Check ALL solutions for dominance by the newly inserted solution
            for (index, existing) in self.vec_set.iter().enumerate() {
                if index != self.last_added_position
                    && existing.is_dominated_by(solution.objectives())
                {
                    to_remove.push(index);
                }
            }

            // Remove dominated solutions in reverse order to maintain indices
            for &index in to_remove.iter().rev() {
                self.vec_set.remove(index);
                // Adjust last_added_position if we removed something before it
                if index < self.last_added_position {
                    self.last_added_position -= 1;
                }
            }

            let size_after = self.vec_set.len();
            if size_before != size_after {
                trace!(
                    "Removed {} dominated solutions from the {} set. Size after: {}",
                    size_before - size_after,
                    self.name,
                    size_after
                );
            }
        }
        was_inserted
    }

    fn with_name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }

    fn insert_unchecked(&mut self, solution: &T) {
        // Even in unchecked mode, prevent exact duplicates to maintain archive invariants
        match self.binary_search_objectives(solution) {
            Ok(pos) | Err(pos) => {
                // Only check for duplicates if there are solutions with same objectives
                if pos < self.vec_set.len()
                    && self.vec_set[pos].objectives() == solution.objectives()
                    && self.contains_exact_duplicate(solution)
                {
                    return;
                }
                self.vec_set.insert(pos, solution.clone());
            }
        }
    }
}

impl<T, const D: usize> FromIterator<T> for VecSolutionSet<T, D>
where
    T: MoSolution<D> + PartialEq + Sized + Clone,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let vec_set: Vec<T> = iter.into_iter().collect();
        Self {
            name: "unnamed",
            vec_set,
            last_added_position: 0,
        }
    }
}

impl<T, const D: usize> IntoIterator for VecSolutionSet<T, D>
where
    T: MoSolution<D> + PartialEq + Sized + Clone,
{
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.vec_set.into_iter()
    }
}

impl<T, const D: usize> RandomCollection<T> for VecSolutionSet<T, D> where
    T: MoSolution<D> + PartialEq + Sized + Clone + Random
{
}
