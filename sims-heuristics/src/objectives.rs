use std::{
    fmt::Debug,
    iter::Sum,
    ops::{Add, AddAssign, Sub, SubAssign},
};

use rand::{distributions::Open01, Rng};

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub struct Objectives(pub u64, pub u64);

impl Objectives {
    pub fn generate_weights() -> (f32, f32) {
        let weight1 = rand::thread_rng().sample(Open01);
        let weight2 = 1.0 - weight1;
        (weight1, weight2)
    }

    pub fn weighted_sum(&self, weights: (f32, f32), _max_values: Objectives) -> f32 {
        self.0 as f32 * weights.0 + self.1 as f32 * weights.1
    }

    pub fn apply_delta(&mut self, delta: (i64, i64)) {
        if delta.0 < 0 {
            self.0 -= delta.0.unsigned_abs();
        } else {
            self.0 += delta.0 as u64;
        }

        if delta.1 < 0 {
            self.1 -= delta.1.unsigned_abs();
        } else {
            self.1 += delta.1 as u64;
        }
    }
}

impl Debug for Objectives {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("")
            .field("cost", &self.0)
            .field("cloudy_area", &self.1)
            .finish()
    }
}

impl PartialOrd for Objectives {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let Objectives(a1, a2) = self;
        let Objectives(b1, b2) = other;

        if a1 == b1 && a2 == b2 {
            Some(std::cmp::Ordering::Equal)
        } else if a1 <= b1 && a2 <= b2 {
            Some(std::cmp::Ordering::Less)
        } else if a1 >= b1 && a2 >= b2 {
            Some(std::cmp::Ordering::Greater)
        } else {
            None
        }
    }
}

impl Add for Objectives {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let Objectives(a1, a2) = self;
        let Objectives(b1, b2) = other;
        Objectives(a1 + b1, a2 + b2)
    }
}

impl Sub for Objectives {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        let Objectives(a1, a2) = self;
        let Objectives(b1, b2) = other;
        Objectives(a1 - b1, a2 - b2)
    }
}

impl AddAssign for Objectives {
    fn add_assign(&mut self, other: Self) {
        let Objectives(a1, a2) = self;
        let Objectives(b1, b2) = other;
        *a1 += b1;
        *a2 += b2;
    }
}

impl SubAssign for Objectives {
    fn sub_assign(&mut self, other: Self) {
        let Objectives(a1, a2) = self;
        let Objectives(b1, b2) = other;
        *a1 -= b1;
        *a2 -= b2;
    }
}

impl Sum for Objectives {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |a, b| a + b)
    }
}

impl From<Objectives> for (i32, i32) {
    fn from(objectives: Objectives) -> Self {
        let Objectives(cost, cloudy_area) = objectives;
        (cost as i32, cloudy_area as i32)
    }
}
