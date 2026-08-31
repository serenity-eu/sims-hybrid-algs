//! Rectangles in the projected criterion space.
//!
//! The search partitions the criterion space projected onto objectives
//! `1..p` -- the first objective is the one minimised in the first stage, so it
//! needs no bound of its own. For two objectives the projection is
//! one-dimensional and a "rectangle" is an interval.
//!
//! A rectangle is stored as its two opposite corners, `l` and `u`, and nothing
//! else. That is deliberate. An earlier version of this crate stored a box as a
//! reference corner plus two known points, with the invariant that each point
//! fixed one coordinate at the reference; reading the box's extent off the
//! wrong pair of coordinates then measured the two sides that are zero by
//! construction, and every box tested as empty. The search discarded its
//! initial box without a single solve and reported a complete front containing
//! only the corners it started from. With explicit corners the extent is
//! `u - l` and there are no coordinate roles to confuse.

/// A box in the projected criterion space, given by opposite corners.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rectangle {
    l: Vec<i64>,
    u: Vec<i64>,
}

impl Rectangle {
    /// Build a rectangle from its lower and upper corners.
    ///
    /// # Panics
    /// If the corners have different dimensions.
    #[must_use]
    pub fn new(l: Vec<i64>, u: Vec<i64>) -> Self {
        assert_eq!(l.len(), u.len(), "rectangle corners differ in dimension");
        Self { l, u }
    }

    /// The lower corner.
    #[must_use]
    pub fn lower(&self) -> &[i64] {
        &self.l
    }

    /// The upper corner.
    #[must_use]
    pub fn upper(&self) -> &[i64] {
        &self.u
    }

    /// Volume of the region between a fixed `reference` corner and this
    /// rectangle's upper corner.
    ///
    /// Measured from the global ideal point rather than from `self.l`, matching
    /// the reference implementation: the quantity that orders the search is how
    /// far the rectangle reaches from the ideal point, not how large it is in
    /// isolation. Selecting the largest such region first is what gives a run
    /// stopped by its deadline points spread over the frontier instead of
    /// clustered at one end.
    ///
    /// `i128` because the product of `p` objective ranges overflows `i64` from
    /// three objectives up at these magnitudes; it saturates rather than wraps,
    /// which at worst makes two enormous rectangles compare equal.
    #[must_use]
    pub fn volume(&self, reference: &[i64]) -> i128 {
        self.u
            .iter()
            .zip(reference)
            .map(|(u, r)| i128::from((u - r).max(0)))
            .try_fold(1i128, |acc, side| acc.checked_mul(side))
            .unwrap_or(i128::MAX)
    }

    /// Cut this rectangle in two at `at` along `axis`.
    ///
    /// For a caller that already knows where to cut -- retrying a region as two
    /// smaller questions after a solve timed out on it, say -- rather than
    /// cutting around a point that was found, which is [`update_list`]'s job.
    ///
    /// # Panics
    /// If `axis` is out of range.
    #[must_use]
    pub fn halves(&self, axis: usize, at: i64) -> [Self; 2] {
        assert!(axis < self.l.len(), "axis {axis} is out of range");
        self.split(axis, at)
    }

    /// Whether this rectangle lies entirely within `other`.
    #[must_use]
    pub fn is_subset_of(&self, other: &Self) -> bool {
        self.l.iter().zip(&other.l).all(|(a, b)| a >= b)
            && self.u.iter().zip(&other.u).all(|(a, b)| a <= b)
    }

    /// Split into the part below `at` on `axis` and the part above it.
    ///
    /// Public as [`Rectangle::halves`]; kept private here because the splitting
    /// the search does routinely goes through [`update_list`].
    #[must_use]
    fn split(&self, axis: usize, at: i64) -> [Self; 2] {
        let mut lower_u = self.u.clone();
        lower_u[axis] = at;
        let mut upper_l = self.l.clone();
        upper_l[axis] = at;
        [
            Self::new(self.l.clone(), lower_u),
            Self::new(upper_l, self.u.clone()),
        ]
    }
}

/// Refine `list` around a newly found point.
///
/// Every rectangle is split on each axis the point falls strictly inside, so
/// the region dominated by the point -- which can hold nothing new -- is
/// separated from the regions that can. A rectangle the point misses is carried
/// through untouched.
#[must_use]
pub fn update_list(list: Vec<Rectangle>, point: &[i64]) -> Vec<Rectangle> {
    let mut refined = Vec::with_capacity(list.len());
    for rectangle in list {
        let mut pieces = vec![rectangle];
        for (axis, &coordinate) in point.iter().enumerate() {
            // Strictly inside: a point on the boundary splits nothing off.
            if pieces
                .first()
                .is_some_and(|r| r.l[axis] < coordinate && coordinate < r.u[axis])
            {
                pieces = pieces
                    .iter()
                    .flat_map(|r| r.split(axis, coordinate))
                    .collect();
            }
        }
        refined.extend(pieces);
    }
    refined
}

/// Drop every rectangle contained in `region`, which is known to hold nothing
/// new.
pub fn remove_subsets(list: &mut Vec<Rectangle>, region: &Rectangle) {
    list.retain(|r| !r.is_subset_of(region));
}

#[cfg(test)]
mod tests {
    use super::{Rectangle, remove_subsets, update_list};

    /// The initial rectangle for `lagos_nigeria_150`, projected onto the second
    /// objective: from the ideal cloud value to the cloud value at the
    /// minimum-cost extreme.
    fn initial() -> Rectangle {
        Rectangle::new(vec![620_841], vec![1_428_692_043])
    }

    /// Regression: the extent must be the distance across the rectangle.
    ///
    /// The predecessor of this module derived it from coordinates that are zero
    /// by construction, so every rectangle measured as empty and the search
    /// terminated before its first solve, reporting a complete front.
    #[test]
    fn a_wide_rectangle_has_volume() {
        let r = initial();
        assert_eq!(r.volume(&[620_841]), 1_428_071_202);
        assert!(r.volume(&[620_841]) > 0);
    }

    /// Volume is measured from the reference corner, not the rectangle's own
    /// lower corner, so a rectangle reaching further from the ideal point wins
    /// even when it is the smaller of the two.
    #[test]
    fn volume_is_measured_from_the_reference_corner() {
        let reference = [0];
        let near = Rectangle::new(vec![0], vec![10]);
        let far = Rectangle::new(vec![90], vec![100]);
        assert!(far.volume(&reference) > near.volume(&reference));
    }

    #[test]
    fn volume_saturates_rather_than_overflowing() {
        let huge = Rectangle::new(vec![0; 4], vec![i64::MAX; 4]);
        assert_eq!(huge.volume(&[0; 4]), i128::MAX);
    }

    #[test]
    fn a_degenerate_rectangle_has_no_volume() {
        assert_eq!(Rectangle::new(vec![5], vec![5]).volume(&[5]), 0);
        // Inverted, which `remove_subsets` can leave behind.
        assert_eq!(Rectangle::new(vec![9], vec![5]).volume(&[9]), 0);
    }

    #[test]
    fn a_point_strictly_inside_splits_the_rectangle() {
        let refined = update_list(vec![Rectangle::new(vec![0], vec![100])], &[40]);
        assert_eq!(
            refined,
            vec![
                Rectangle::new(vec![0], vec![40]),
                Rectangle::new(vec![40], vec![100]),
            ]
        );
    }

    #[test]
    fn a_point_on_the_boundary_splits_nothing() {
        let list = vec![Rectangle::new(vec![0], vec![100])];
        assert_eq!(update_list(list.clone(), &[0]), list);
        assert_eq!(update_list(list.clone(), &[100]), list);
        assert_eq!(update_list(list.clone(), &[500]), list, "outside entirely");
    }

    /// With two projected axes a point inside both splits into four.
    #[test]
    fn splitting_is_per_axis() {
        let refined = update_list(vec![Rectangle::new(vec![0, 0], vec![10, 10])], &[5, 5]);
        assert_eq!(refined.len(), 4);
        for r in &refined {
            assert!(r.is_subset_of(&Rectangle::new(vec![0, 0], vec![10, 10])));
        }
    }

    #[test]
    fn subsets_of_an_exhausted_region_are_removed() {
        let mut list = vec![
            Rectangle::new(vec![10], vec![20]),
            Rectangle::new(vec![20], vec![30]),
            Rectangle::new(vec![5], vec![40]),
        ];
        remove_subsets(&mut list, &Rectangle::new(vec![10], vec![30]));
        assert_eq!(list, vec![Rectangle::new(vec![5], vec![40])]);
    }

    #[test]
    fn a_rectangle_is_a_subset_of_itself() {
        let r = initial();
        assert!(r.is_subset_of(&r));
    }
}
