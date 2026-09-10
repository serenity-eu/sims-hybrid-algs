//! Destroy operators and the adaptive weights that choose between them.
//!
//! The repair half of the ruin-and-recreate is exact (a MILP solve), so the
//! only thing left to steer is *what* to tear out. Following Ropke & Pisinger's
//! adaptive scheme, each operator carries a weight that is reinforced when it
//! produces an improvement and decayed otherwise, so the mix adapts to the
//! instance instead of being fixed by a flag.
use fixedbitset::FixedBitSet;
use rand::{Rng, seq::{IndexedRandom, SliceRandom}};

use crate::problem_bitset::ProblemBitset;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DestroyOperator {
    /// Uniformly sample selected images.
    Random,
    /// Remove the most expensive selected images — biases toward cost descent.
    WorstCost,
    /// Shaw-style relatedness: seed on one image and take the selected images
    /// whose coverage overlaps it most, so the freed region is contiguous and
    /// the repair has real substitutes available.
    Related,
}

impl DestroyOperator {
    pub const ALL: [Self; 3] = [Self::Random, Self::WorstCost, Self::Related];

    /// Choose `count` incumbent images to release.
    pub fn select<const D: usize, R: Rng>(
        self,
        problem: &ProblemBitset<D>,
        incumbent: &FixedBitSet,
        count: usize,
        rng: &mut R,
        scratch: &mut Vec<usize>,
    ) -> FixedBitSet {
        scratch.clear();
        scratch.extend(incumbent.ones());
        match self {
            Self::Random => scratch.shuffle(rng),
            Self::WorstCost => {
                scratch.sort_unstable_by_key(|&i| std::cmp::Reverse(problem.image_cost(i)));
            }
            Self::Related => {
                if let Some(&seed) = scratch.choose(rng) {
                    scratch.sort_unstable_by_key(|&i| std::cmp::Reverse(problem.overlap(seed, i)));
                }
            }
        }
        let mut out = FixedBitSet::with_capacity(problem.images.len());
        for &i in scratch.iter().take(count) {
            out.insert(i);
        }
        out
    }
}

/// Adaptive operator selection with exponential-decay credit.
pub struct AdaptiveWeights {
    weights: [f64; 3],
    decay: f64,
    reward: f64,
}

impl AdaptiveWeights {
    #[must_use]
    pub const fn new(decay: f64, reward: f64) -> Self {
        Self { weights: [1.0; 3], decay, reward }
    }

    pub fn pick<R: Rng>(&self, rng: &mut R) -> (usize, DestroyOperator) {
        let total: f64 = self.weights.iter().sum();
        let mut t = rng.random_range(0.0..total);
        for (idx, &w) in self.weights.iter().enumerate() {
            t -= w;
            if t <= 0.0 {
                return (idx, DestroyOperator::ALL[idx]);
            }
        }
        (0, DestroyOperator::ALL[0])
    }

    /// Reinforce on success, decay otherwise. Weights are floored so an
    /// operator that stalls early can still be re-tried later in the run.
    pub fn update(&mut self, idx: usize, improved: bool) {
        let w = &mut self.weights[idx];
        *w = if improved {
            self.decay.mul_add(*w, self.reward)
        } else {
            (self.decay * *w).max(0.05)
        };
    }

    #[must_use]
    pub const fn weights(&self) -> &[f64; 3] {
        &self.weights
    }
}
