use crate::clouds::generate_clouds;

// ── Continuous (float) problem ────────────────────────────────────────────────

/// SIMS problem with continuous (float) objective data, before discretisation.
///
/// This mirrors Python's `SimsProblem` dataclass.  Areas are in m²
/// (pre-projected to CEA by the caller).
#[derive(Debug, Clone)]
pub struct SimsProblem {
    pub num_images: usize,
    /// Number of atomic fragments that partition the AOI.
    pub universe: usize,
    /// For each image: sorted list of 0-based fragment indices it covers.
    pub images: Vec<Vec<usize>>,
    /// Purchase cost per image (currency units, typically integer-valued floats).
    pub costs: Vec<f64>,
    /// Cloud coverage percentage (0–100) per image.
    pub cloud_coverages: Vec<f64>,
    /// Area in m² for each fragment (indexed by fragment index).
    pub areas: Vec<f64>,
    /// Ground sampling distance (m/px) per image.
    pub resolution: Vec<f64>,
    /// Off-nadir incidence angle (degrees) per image.
    pub incidence_angle: Vec<f64>,
    /// Total area (m²) of all fragments; upper bound for the cloud objective.
    pub max_cloud_area: f64,
}

impl SimsProblem {
    /// Discretise the problem for use by the MILP / PLS solvers.
    ///
    /// Scaling mirrors the Python implementation:
    /// * `costs`            → truncate to i64 (typically already integer-valued)
    /// * `areas`            → truncate to i64
    /// * `resolution`       → multiply by 100, truncate to i64
    /// * `incidence_angle`  → multiply by 10, round to nearest i64
    ///
    /// Cloud masks are generated randomly; pass `seed` for reproducibility.
    pub fn discretize(&self, seed: u64) -> SimsDiscreteProblem {
        let costs: Vec<i64> = self.costs.iter().map(|&c| c as i64).collect();
        let areas: Vec<i64> = self.areas.iter().map(|&a| a as i64).collect();
        let resolution: Vec<i64> = self
            .resolution
            .iter()
            .map(|&r| (r * 100.0) as i64)
            .collect();
        let incidence_angle: Vec<i64> = self
            .incidence_angle
            .iter()
            .map(|&a| (a * 10.0).round() as i64)
            .collect();
        let max_cloud_area: i64 = areas.iter().sum();

        let clouds = generate_clouds(&self.images, &self.areas, &self.cloud_coverages, seed);

        SimsDiscreteProblem {
            num_images: self.num_images,
            universe: self.universe,
            images: self.images.clone(),
            costs,
            clouds,
            areas,
            resolution,
            incidence_angle,
            max_cloud_area,
        }
    }
}

// ── Discrete (integer) problem ────────────────────────────────────────────────

/// SIMS problem with integer-valued objectives, ready for the solver.
///
/// This is the Rust counterpart of both Python's `SimsDiscreteProblem` and the
/// existing `SimsDiscreteProblem` PyO3 class in `sims-problem`.
#[derive(Debug, Clone)]
pub struct SimsDiscreteProblem {
    pub num_images: usize,
    pub universe: usize,
    /// For each image: sorted list of 0-based fragment indices it covers.
    pub images: Vec<Vec<usize>>,
    pub costs: Vec<i64>,
    /// For each image: sorted list of 0-based fragment indices that are cloudy.
    pub clouds: Vec<Vec<usize>>,
    /// Area (integer m²) of each fragment.
    pub areas: Vec<i64>,
    /// Resolution (GSD × 100) per image.
    pub resolution: Vec<i64>,
    /// Incidence angle (degrees × 10) per image.
    pub incidence_angle: Vec<i64>,
    /// Sum of all fragment areas; upper bound for the cloud objective.
    pub max_cloud_area: i64,
}

impl SimsDiscreteProblem {
    /// Maximum cost and maximum area over the whole instance.
    pub fn max_values(&self) -> (i64, i64) {
        (self.costs.iter().sum(), self.areas.iter().sum())
    }

    /// Reference point for hypervolume: `(max_cost + 1, max_area + 1)`.
    pub fn ref_point(&self) -> (i64, i64) {
        let (c, a) = self.max_values();
        (c + 1, a + 1)
    }

    /// Sanity-check the problem data.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_images == 0 {
            return Err("num_images must be > 0".into());
        }
        if self.universe == 0 {
            return Err("universe must be > 0".into());
        }
        macro_rules! check_len {
            ($field:ident, $expected:expr) => {
                if self.$field.len() != $expected {
                    return Err(format!(
                        "length of {} ({}) != expected ({})",
                        stringify!($field),
                        self.$field.len(),
                        $expected
                    ));
                }
            };
        }
        check_len!(images, self.num_images);
        check_len!(costs, self.num_images);
        check_len!(clouds, self.num_images);
        check_len!(areas, self.universe);
        check_len!(resolution, self.num_images);
        check_len!(incidence_angle, self.num_images);

        let all_fragments: std::collections::BTreeSet<usize> =
            self.images.iter().flatten().copied().collect();
        let expected: std::collections::BTreeSet<usize> = (0..self.universe).collect();
        if all_fragments != expected {
            let missing: Vec<usize> = expected.difference(&all_fragments).copied().collect();
            return Err(format!("fragments not covered by any image: {missing:?}"));
        }

        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_continuous() -> SimsProblem {
        SimsProblem {
            num_images: 3,
            universe: 4,
            images: vec![vec![0, 1], vec![1, 2], vec![2, 3]],
            costs: vec![1000.0, 2000.0, 1500.0],
            cloud_coverages: vec![10.0, 0.0, 25.0],
            areas: vec![100_000.0, 200_000.0, 150_000.0, 180_000.0],
            resolution: vec![0.5, 1.0, 0.8],
            incidence_angle: vec![15.5, 20.0, 12.3],
            max_cloud_area: 630_000.0,
        }
    }

    #[test]
    fn test_discretize_scaling() {
        let p = sample_continuous();
        let d = p.discretize(42);

        assert_eq!(d.costs, vec![1000, 2000, 1500]);
        assert_eq!(d.areas, vec![100_000, 200_000, 150_000, 180_000]);
        assert_eq!(d.resolution, vec![50, 100, 80]); // × 100 then truncate
        assert_eq!(d.incidence_angle, vec![155, 200, 123]); // × 10 then round
    }

    #[test]
    fn test_validate_ok() {
        let p = sample_continuous();
        let d = p.discretize(0);
        assert!(d.validate().is_ok());
    }

    #[test]
    fn test_validate_missing_fragments() {
        let mut d = sample_continuous().discretize(0);
        d.universe = 5; // fragment 4 is never covered
        assert!(d.validate().is_err());
    }
}
