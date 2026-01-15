use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::marker::ConstParamTy;

pub type Objectives<const D: usize> = [u64; D];

pub trait ParetoFront<'a, T>: IntoIterator<Item = T> {
    type Iter<'b>: Iterator<Item = &'b T>
    where
        Self: 'b,
        T: 'b;

    /// Create an empty solution set
    fn new(name: &'static str) -> Self;

    /// Set name
    #[must_use]
    fn with_name(self, name: &'static str) -> Self;

    /// Iterate over the solutions in the set
    fn iter(&self) -> Self::Iter<'_>;

    /// Check if solution is in the set
    fn contains(&self, solution: &T) -> bool;

    /// Try add new solution to the set, return true if it there was no solution in the set that dominated it and it was added
    fn try_insert(&mut self, solution: &T) -> bool;

    /// Forcefuly add solution to the set, for use only with collect from another `SolutionSet`
    fn insert_unchecked(&mut self, solution: &T);

    /// Return length of the solution set
    fn len(&self) -> usize {
        self.iter().count()
    }

    /// Return true if the solution set is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn validate<const D: usize>(&self)
    where
        T: MoSolution<D> + HasObjectives<D>,
    {
        let solutions: Vec<_> = self.iter().collect();
        for (i, solution_a) in solutions.iter().enumerate() {
            for (j, solution_b) in solutions.iter().enumerate() {
                if i != j {
                    // Check if solution_a dominates solution_b
                    let dominates = solution_a.dominates(solution_b.objectives());

                    assert!(
                        !dominates,
                        "Solution set contains dominated solution! Solution at index {} with objectives {:?} dominates solution at index {} with objectives {:?}",
                        i,
                        solution_a.objectives(),
                        j,
                        solution_b.objectives()
                    );
                }
            }
        }
    }
}

pub trait Random {
    fn random() -> Self;
    fn random_with_seed(seed: u64) -> Self;
}

pub trait RandomCollection<T: Random>: FromIterator<T> {
    #[must_use]
    fn random(size: usize) -> Self {
        (0..size).map(|_| T::random()).collect()
    }

    #[must_use]
    fn random_with_seed(size: usize, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..size)
            .map(|_| T::random_with_seed(rng.random()))
            .collect()
    }
}

#[derive(ConstParamTy, PartialEq, Eq)]
pub enum Sense {
    Minimization,
    Maximization,
}

// TODO: Fon now, this is hardcoded for Minimization
pub trait MoSolution<const D: usize>: HasObjectives<D> {
    fn is_dominated_by(&self, point: &Objectives<D>) -> bool {
        non_dominance_relation::<D, { Sense::Minimization }>(self.objectives(), point)
            == Dominance::IsDominated
    }
    fn is_covered_by(&self, point: &Objectives<D>) -> bool {
        let relation =
            non_dominance_relation::<D, { Sense::Minimization }>(self.objectives(), point);
        relation == Dominance::IsDominated || relation == Dominance::Equals
    }
    fn dominates(&self, point: &Objectives<D>) -> bool {
        non_dominance_relation::<D, { Sense::Minimization }>(self.objectives(), point)
            == Dominance::Dominates
    }
    fn covers(&self, point: &Objectives<D>) -> bool {
        let relation =
            non_dominance_relation::<D, { Sense::Minimization }>(self.objectives(), point);
        relation == Dominance::Dominates || relation == Dominance::Equals
    }
}

pub trait HasObjectives<const D: usize> {
    fn objectives(&self) -> &Objectives<D>;

    fn squared_distance_to(&self, point: &Objectives<D>) -> u64 {
        let diff = self
            .objectives()
            .iter()
            .zip(point.iter())
            .map(|(a, b)| a.abs_diff(*b));
        diff.map(|d| d * d).sum()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dominance {
    Dominates,
    IsDominated,
    Equals,
    NonDominated,
}

fn non_dominance_relation<const D: usize, const S: Sense>(
    a: &Objectives<D>,
    b: &Objectives<D>,
) -> Dominance {
    match S {
        Sense::Minimization => {
            if a.eq(b) {
                return Dominance::Equals;
            }
            if a.iter().zip(b.iter()).all(|(a, b)| a <= b) {
                return Dominance::Dominates;
            } else if a.iter().zip(b.iter()).all(|(a, b)| a >= b) {
                return Dominance::IsDominated;
            }
            Dominance::NonDominated
        }
        Sense::Maximization => {
            if a.eq(b) {
                return Dominance::Equals;
            }
            if a.iter().zip(b.iter()).all(|(a, b)| a >= b) {
                return Dominance::Dominates;
            } else if a.iter().zip(b.iter()).all(|(a, b)| a <= b) {
                return Dominance::IsDominated;
            }
            Dominance::NonDominated
        }
    }
}
