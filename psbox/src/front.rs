//! The nondominated set.

use std::collections::HashMap;

/// One solution: objective values and the decision variables that produced them.
#[derive(Clone, Debug, PartialEq)]
pub struct Solution {
    /// Objective values, all minimised. Integer: see `Problem::objective_values`.
    pub objectives: Vec<i64>,
    /// Decision variable values by name.
    pub variables: HashMap<String, f64>,
    /// Seconds from the start of the run, for anytime reporting.
    pub found_at: f64,
}

impl Solution {
    /// Does `self` dominate `other` -- no worse everywhere, better somewhere?
    #[must_use]
    pub fn dominates(&self, other: &Self) -> bool {
        let pairs = || self.objectives.iter().zip(&other.objectives);
        pairs().all(|(a, b)| a <= b) && pairs().any(|(a, b)| a < b)
    }
}

/// A set of mutually nondominated solutions, kept sorted by the first objective.
///
/// The set stays small -- tens of points, not thousands -- so insertion scans
/// it rather than indexing it. Sorting is for stable, readable output, not for
/// the search.
#[derive(Clone, Debug, Default)]
pub struct Front {
    solutions: Vec<Solution>,
}

impl Front {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            solutions: Vec::new(),
        }
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.solutions.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    #[must_use]
    pub fn solutions(&self) -> &[Solution] {
        &self.solutions
    }

    /// Insert if nondominated, removing anything it dominates.
    ///
    /// Returns true when the point was added. A point equal to one already held
    /// is rejected, so a weakly dominated duplicate cannot enter the set.
    pub fn insert(&mut self, candidate: Solution) -> bool {
        if self
            .solutions
            .iter()
            .any(|s| s.objectives == candidate.objectives || s.dominates(&candidate))
        {
            return false;
        }
        self.solutions.retain(|s| !candidate.dominates(s));
        let pos = self
            .solutions
            .partition_point(|s| s.objectives[0] < candidate.objectives[0]);
        self.solutions.insert(pos, candidate);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::{Front, Solution};
    use std::collections::HashMap;

    fn sol(objectives: &[i64]) -> Solution {
        Solution {
            objectives: objectives.to_vec(),
            variables: HashMap::new(),
            found_at: 0.0,
        }
    }

    #[test]
    fn keeps_mutually_nondominated_points() {
        let mut f = Front::new();
        assert!(f.insert(sol(&[1, 9])));
        assert!(f.insert(sol(&[5, 5])));
        assert!(f.insert(sol(&[9, 1])));
        assert_eq!(f.len(), 3);
        let first: Vec<i64> = f.solutions().iter().map(|s| s.objectives[0]).collect();
        assert_eq!(first, vec![1, 5, 9]);
    }

    #[test]
    fn rejects_dominated_and_duplicate() {
        let mut f = Front::new();
        f.insert(sol(&[5, 5]));
        assert!(!f.insert(sol(&[6, 6])), "strictly dominated");
        assert!(!f.insert(sol(&[5, 5])), "duplicate");
        assert_eq!(f.len(), 1);
    }

    /// The case this crate exists for: equal on one objective, worse on the
    /// other. A front that accepts these is reporting weakly nondominated
    /// points as nondominated.
    #[test]
    fn rejects_weakly_dominated() {
        let mut f = Front::new();
        f.insert(sol(&[5, 5]));
        assert!(!f.insert(sol(&[5, 7])), "same first, worse second");
        assert!(!f.insert(sol(&[8, 5])), "same second, worse first");
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn new_point_evicts_those_it_dominates() {
        let mut f = Front::new();
        f.insert(sol(&[4, 8]));
        f.insert(sol(&[6, 6]));
        f.insert(sol(&[8, 4]));
        assert!(f.insert(sol(&[3, 3])), "dominates all three");
        assert_eq!(f.len(), 1);
        assert_eq!(f.solutions()[0].objectives, vec![3, 3]);
    }

    #[test]
    fn dominance_generalises_beyond_two_objectives() {
        let mut f = Front::new();
        assert!(f.insert(sol(&[1, 5, 5])));
        assert!(!f.insert(sol(&[1, 5, 6])), "weakly dominated in 3d");
        assert!(f.insert(sol(&[1, 4, 6])), "trades the second for the third");
        assert_eq!(f.len(), 2);
    }

    /// Insertion order must not change the resulting set.
    #[test]
    fn order_independent() {
        let pts: [&[i64]; 5] = [&[1, 9], &[5, 5], &[9, 1], &[6, 6], &[2, 8]];
        let mut forward = Front::new();
        for p in pts {
            forward.insert(sol(p));
        }
        let mut backward = Front::new();
        for p in pts.iter().rev() {
            backward.insert(sol(p));
        }
        let f: Vec<_> = forward
            .solutions()
            .iter()
            .map(|s| s.objectives.clone())
            .collect();
        let b: Vec<_> = backward
            .solutions()
            .iter()
            .map(|s| s.objectives.clone())
            .collect();
        assert_eq!(f, b);
    }
}
