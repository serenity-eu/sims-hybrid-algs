//! # Solution Representation Module
//!
//! This module provides structures and traits for representing optimization solutions,
//! Pareto fronts, and performing multi-objective solution analysis.
//!
//! ## Overview
//!
//! The solution system includes:
//! - **[`HasObjectives`]**: Trait for types that expose objective values
//! - **[`MoSolution`]**: Trait for multi-objective dominance operations
//! - **[`Solution`]**: Individual solution with objective values, decision variables, and metadata
//! - **[`ParetoFront`]**: A non-dominated set that maintains the Pareto invariant on every insertion
//!
//! ## Design Principles
//!
//! 1. **Invariant-preserving insertion** – [`ParetoFront::try_insert`] atomically checks
//!    dominance, removes newly-dominated incumbents, and inserts the new solution.  The
//!    front is *always* a valid non-dominated set after every public mutation.
//!
//! 2. **Mixed objective directions** – Each objective can independently be `Minimize` or
//!    `Maximize`.  Dominance comparisons respect the per-objective direction.
//!
//! 3. **Configurable precision** – Duplicate detection rounds objective values to a
//!    configurable number of decimal places so that near-identical solutions from
//!    different solver runs are collapsed.
//!
//! 4. **Quality metrics** – Hypervolume indicator, spacing, spread, and maximum-gap
//!    measures are available for assessing representation quality.
//!
//! ## Quick Example
//!
//! ```rust
//! use augmecon::solution::{Solution, ParetoFront};
//! use augmecon::model::ObjectiveDirection;
//! use std::collections::HashMap;
//!
//! let dirs = vec![ObjectiveDirection::Minimize, ObjectiveDirection::Minimize];
//! let mut front = ParetoFront::new(dirs);
//!
//! // First solution is always accepted
//! let s1 = Solution::new(vec![1.0, 4.0], HashMap::new());
//! assert!(front.try_insert(s1));
//!
//! // Dominated solution is rejected
//! let s2 = Solution::new(vec![2.0, 5.0], HashMap::new());
//! assert!(!front.try_insert(s2));
//!
//! // Non-dominated solution is accepted
//! let s3 = Solution::new(vec![3.0, 2.0], HashMap::new());
//! assert!(front.try_insert(s3));
//!
//! assert_eq!(front.len(), 2);
//! ```

use std::collections::HashMap;
use std::fmt;
use std::ops::Index;

// ---------------------------------------------------------------------------
// Free function: dominance check with explicit directions
// ---------------------------------------------------------------------------

/// Check if solution `a` dominates solution `b` given the specified objective
/// directions.  This is a free function so it can be called without borrowing
/// the entire `ParetoFront`.
fn dominates_with_directions(
    a: &Solution,
    b: &Solution,
    directions: &[crate::model::ObjectiveDirection],
) -> bool {
    let a_obj = &a.objective_values;
    let b_obj = &b.objective_values;
    if a_obj.len() != b_obj.len() {
        return false;
    }

    let mut at_least_as_good = true;
    let mut strictly_better = false;

    for (k, (av, bv)) in a_obj.iter().zip(b_obj.iter()).enumerate() {
        if av.is_nan() || bv.is_nan() {
            return false;
        }
        let is_max = k < directions.len()
            && matches!(directions[k], crate::model::ObjectiveDirection::Maximize);
        if is_max {
            if av < bv {
                at_least_as_good = false;
                break;
            } else if av > bv {
                strictly_better = true;
            }
        } else if av > bv {
            at_least_as_good = false;
            break;
        } else if av < bv {
            strictly_better = true;
        }
    }

    at_least_as_good && strictly_better
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Trait for types that have objective values.
pub trait HasObjectives {
    /// Get the objective values for this solution.
    fn objectives(&self) -> &[f64];
}

/// Trait for multi-objective solution operations.
///
/// All methods require an `is_maximizing` slice that indicates the optimisation
/// sense for each objective (`true` = maximise, `false` = minimise).
pub trait MoSolution: HasObjectives {
    /// Returns `true` if `self` dominates `other`.
    ///
    /// Domination means *at least as good* in every objective **and** *strictly
    /// better* in at least one.
    fn dominates(&self, other: &Self, is_maximizing: &[bool]) -> bool {
        let self_obj = self.objectives();
        let other_obj = other.objectives();

        if self_obj.len() != other_obj.len() || self_obj.len() != is_maximizing.len() {
            return false;
        }

        let mut at_least_as_good = true;
        let mut strictly_better = false;

        for ((s, o), &is_max) in self_obj
            .iter()
            .zip(other_obj.iter())
            .zip(is_maximizing.iter())
        {
            if s.is_nan() || o.is_nan() {
                return false;
            }
            if is_max {
                if s < o {
                    at_least_as_good = false;
                    break;
                } else if s > o {
                    strictly_better = true;
                }
            } else {
                if s > o {
                    at_least_as_good = false;
                    break;
                } else if s < o {
                    strictly_better = true;
                }
            }
        }

        at_least_as_good && strictly_better
    }

    /// Returns `true` if `self` is dominated by `other`.
    fn is_dominated_by(&self, other: &Self, is_maximizing: &[bool]) -> bool {
        other.dominates(self, is_maximizing)
    }

    /// Returns `true` if `self` covers `other` (dominates **or** equals).
    fn covers(&self, other: &Self, is_maximizing: &[bool]) -> bool {
        let self_obj = self.objectives();
        let other_obj = other.objectives();

        if self_obj.len() != other_obj.len() || self_obj.len() != is_maximizing.len() {
            return false;
        }

        for ((s, o), &is_max) in self_obj
            .iter()
            .zip(other_obj.iter())
            .zip(is_maximizing.iter())
        {
            if s.is_nan() || o.is_nan() {
                return false;
            }
            if is_max {
                if s < o {
                    return false;
                }
            } else if s > o {
                return false;
            }
        }

        true
    }

    /// Returns `true` if `self` is covered by `other`.
    fn is_covered_by(&self, other: &Self, is_maximizing: &[bool]) -> bool {
        other.covers(self, is_maximizing)
    }

    // -- Convenience methods assuming all-minimisation --

    /// Dominates assuming minimisation for every objective.
    fn dominates_min(&self, other: &Self) -> bool {
        let is_min = vec![false; self.objectives().len()];
        self.dominates(other, &is_min)
    }

    /// Is dominated assuming minimisation for every objective.
    fn is_dominated_by_min(&self, other: &Self) -> bool {
        other.dominates_min(self)
    }

    // -- Convenience methods assuming all-maximisation --

    /// Dominates assuming maximisation for every objective.
    fn dominates_max(&self, other: &Self) -> bool {
        let is_max = vec![true; self.objectives().len()];
        self.dominates(other, &is_max)
    }

    /// Is dominated assuming maximisation for every objective.
    fn is_dominated_by_max(&self, other: &Self) -> bool {
        other.dominates_max(self)
    }
}

// ---------------------------------------------------------------------------
// Solution
// ---------------------------------------------------------------------------

/// Represents a single solution to the multi-objective optimisation problem.
#[derive(Debug, Clone)]
pub struct Solution {
    /// Values of the objective functions for this solution.
    pub objective_values: Vec<f64>,
    /// Values of the decision variables for this solution.
    pub decision_variables: HashMap<String, f64>,
    /// Whether this solution satisfies all constraints.
    pub feasible: bool,
    /// Free-form metadata (e.g. solver name, iteration found, …).
    pub metadata: HashMap<String, String>,
}

impl Solution {
    /// Create a new **feasible** solution.
    #[must_use]
    pub fn new(objective_values: Vec<f64>, decision_variables: HashMap<String, f64>) -> Self {
        Self {
            objective_values,
            decision_variables,
            feasible: true,
            metadata: HashMap::new(),
        }
    }

    /// Create an **infeasible** placeholder with `NaN` objectives.
    #[must_use]
    pub fn infeasible(num_objectives: usize) -> Self {
        Self {
            objective_values: vec![f64::NAN; num_objectives],
            decision_variables: HashMap::new(),
            feasible: false,
            metadata: HashMap::new(),
        }
    }

    /// Get the value of a specific decision variable by name.
    #[must_use]
    pub fn get_variable(&self, name: &str) -> Option<f64> {
        self.decision_variables.get(name).copied()
    }

    /// Get the value of a specific objective by index.
    #[must_use]
    pub fn get_objective(&self, index: usize) -> Option<f64> {
        self.objective_values.get(index).copied()
    }

    /// Attach a key-value metadata pair (builder-style).
    #[must_use]
    pub fn with_metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

impl HasObjectives for Solution {
    fn objectives(&self) -> &[f64] {
        &self.objective_values
    }
}

impl MoSolution for Solution {}

impl fmt::Display for Solution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, v) in self.objective_values.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{v:.4}")?;
        }
        write!(f, ")")
    }
}

impl PartialEq for Solution {
    fn eq(&self, other: &Self) -> bool {
        self.objective_values.len() == other.objective_values.len()
            && self
                .objective_values
                .iter()
                .zip(other.objective_values.iter())
                .all(|(a, b)| (a - b).abs() < f64::EPSILON || (a.is_nan() && b.is_nan()))
    }
}

// ---------------------------------------------------------------------------
// ParetoFront
// ---------------------------------------------------------------------------

/// A collection of non-dominated solutions that preserves the Pareto invariant.
///
/// Every public mutation (`try_insert`, `try_insert_with_precision`, `merge`,
/// `add_solution`, `add_solution_with_precision`) ensures that the internal
/// `solutions` vec contains **only** mutually non-dominated, feasible solutions
/// afterwards.
///
/// Solutions are kept sorted by the first objective (ascending for minimisation,
/// descending for maximisation) after each insertion so that iteration order is
/// deterministic and quality-metric computations can assume sorted input.
#[derive(Debug, Clone)]
pub struct ParetoFront {
    /// The non-dominated solution set.
    pub solutions: Vec<Solution>,
    /// Payoff table showing extreme values for each objective.
    pub payoff_table: Vec<Vec<f64>>,
    /// Per-objective optimisation direction.
    pub objective_directions: Vec<crate::model::ObjectiveDirection>,
    /// Number of decimal places used for duplicate detection.
    precision: u32,
}

impl ParetoFront {
    // ── Construction ────────────────────────────────────────────────────

    /// Create a new empty Pareto front.
    #[must_use]
    pub const fn new(objective_directions: Vec<crate::model::ObjectiveDirection>) -> Self {
        Self {
            solutions: Vec::new(),
            payoff_table: Vec::new(),
            objective_directions,
            precision: 9,
        }
    }

    /// Set the number of decimal places used when detecting duplicate objective
    /// vectors.  The default is 9.
    #[must_use]
    pub const fn with_precision(mut self, decimal_places: u32) -> Self {
        self.precision = decimal_places;
        self
    }

    // ── Helpers (private) ───────────────────────────────────────────────

    /// Build the `is_maximizing` flags from `objective_directions`.
    fn is_maximizing_flags(&self) -> Vec<bool> {
        self.objective_directions
            .iter()
            .map(|d| matches!(d, crate::model::ObjectiveDirection::Maximize))
            .collect()
    }

    /// Remove incumbents dominated by `candidate`, returning the number removed.
    fn remove_dominated_by(&mut self, candidate: &Solution) -> usize {
        let dirs = &self.objective_directions;
        let before = self.solutions.len();
        self.solutions
            .retain(|incumbent| !dominates_with_directions(candidate, incumbent, dirs));
        before - self.solutions.len()
    }

    /// Re-sort solutions by first objective (ascending for min, descending for max).
    fn sort_solutions(&mut self) {
        if self.solutions.is_empty() {
            return;
        }
        let first_is_max = !self.objective_directions.is_empty()
            && matches!(
                self.objective_directions[0],
                crate::model::ObjectiveDirection::Maximize
            );
        self.solutions.sort_by(|a, b| {
            let va = a.objective_values.first().copied().unwrap_or(f64::NAN);
            let vb = b.objective_values.first().copied().unwrap_or(f64::NAN);
            if first_is_max {
                vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            }
        });
    }

    // ── Invariant-preserving insertion ──────────────────────────────────

    /// Try to insert a solution into the Pareto front.
    ///
    /// Returns `true` if the solution was **accepted** (non-dominated and not a
    /// duplicate).  Any incumbents that the new solution dominates are removed.
    ///
    /// Infeasible solutions and solutions containing `NaN` objectives are silently
    /// rejected.
    pub fn try_insert(&mut self, solution: Solution) -> bool {
        self.try_insert_with_precision(solution, self.precision)
    }

    /// Like [`try_insert`](Self::try_insert) but with an explicit precision for
    /// duplicate detection.
    pub fn try_insert_with_precision(&mut self, solution: Solution, decimal_places: u32) -> bool {
        // Reject infeasible solutions
        if !solution.feasible {
            log::trace!("Rejected infeasible solution");
            return false;
        }

        // Reject solutions with NaN objectives
        if solution.objective_values.iter().any(|v| v.is_nan()) {
            log::trace!("Rejected solution with NaN objective(s)");
            return false;
        }

        // Round for comparison
        #[allow(clippy::cast_possible_wrap)]
        let factor = 10_f64.powi(decimal_places as i32);
        let rounded: Vec<f64> = solution
            .objective_values
            .iter()
            .map(|&v| (v * factor).round() / factor)
            .collect();

        // Duplicate check
        #[allow(clippy::cast_possible_wrap)]
        let eps = 0.5 / 10_f64.powi(decimal_places as i32);
        let is_dup = self.solutions.iter().any(|existing| {
            existing.objective_values.len() == rounded.len()
                && existing
                    .objective_values
                    .iter()
                    .zip(rounded.iter())
                    .all(|(a, b)| (a - b).abs() < eps)
        });
        if is_dup {
            log::trace!("Rejected duplicate solution: {rounded:?}");
            return false;
        }

        // Build rounded candidate for dominance checks
        let candidate = Solution {
            objective_values: rounded.clone(),
            decision_variables: solution.decision_variables.clone(),
            feasible: true,
            metadata: solution.metadata,
        };

        // Check if any incumbent dominates the candidate
        for incumbent in &self.solutions {
            if dominates_with_directions(incumbent, &candidate, &self.objective_directions) {
                log::trace!("Solution {rounded:?} is dominated by existing solution");
                return false;
            }
        }

        // Remove incumbents dominated by the candidate
        let removed = self.remove_dominated_by(&candidate);
        if removed > 0 {
            log::debug!("New solution {rounded:?} removed {removed} dominated incumbent(s)");
        }

        self.solutions.push(candidate);
        self.sort_solutions();

        log::trace!(
            "Accepted solution {rounded:?}  (front size: {})",
            self.solutions.len()
        );
        true
    }

    // ── Legacy API (delegates to try_insert) ────────────────────────────

    /// Add a solution, performing dominance filtering automatically.
    ///
    /// This is a convenience wrapper around [`try_insert`](Self::try_insert)
    /// that matches the legacy API.
    pub fn add_solution(&mut self, solution: Solution) {
        self.try_insert(solution);
    }

    /// Add a solution with configurable precision, performing dominance filtering
    /// automatically.
    ///
    /// This is a convenience wrapper around
    /// [`try_insert_with_precision`](Self::try_insert_with_precision) that
    /// matches the legacy API.
    pub fn add_solution_with_precision(&mut self, solution: Solution, decimal_places: u32) {
        self.try_insert_with_precision(solution, decimal_places);
    }

    // ── Batch dominance filter (legacy / safety-net) ────────────────────

    /// Filter dominated solutions from the front.
    ///
    /// If you always use [`try_insert`](Self::try_insert), this is a no-op.
    /// It exists as a safety-net for code that mutates `self.solutions` directly.
    pub fn filter_dominated_solutions(&mut self) {
        let n = self.solutions.len();
        if n <= 1 {
            return;
        }

        log::debug!("Batch-filtering Pareto front with {n} solutions");

        let mut is_dominated = vec![false; n];

        for i in 0..n {
            if is_dominated[i] {
                continue;
            }
            for j in (i + 1)..n {
                if is_dominated[j] {
                    continue;
                }
                let dirs = &self.objective_directions;
                if dominates_with_directions(&self.solutions[i], &self.solutions[j], dirs) {
                    is_dominated[j] = true;
                } else if dominates_with_directions(&self.solutions[j], &self.solutions[i], dirs) {
                    is_dominated[i] = true;
                    break;
                }
            }
        }

        self.solutions = self
            .solutions
            .drain(..)
            .enumerate()
            .filter_map(|(i, s)| if is_dominated[i] { None } else { Some(s) })
            .collect();

        self.sort_solutions();

        log::debug!("Batch filter: {n} -> {} solutions", self.solutions.len());
    }

    // ── Merge ───────────────────────────────────────────────────────────

    /// Merge another Pareto front into this one, preserving non-dominance.
    pub fn merge(&mut self, other: &Self) {
        for solution in &other.solutions {
            self.try_insert(solution.clone());
        }
    }

    // ── Payoff table ────────────────────────────────────────────────────

    /// Set the payoff table.
    pub fn set_payoff_table(&mut self, payoff_table: Vec<Vec<f64>>) {
        self.payoff_table = payoff_table;
    }

    // ── Accessors ───────────────────────────────────────────────────────

    /// Number of solutions in the front.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.solutions.len()
    }

    /// Returns `true` when the front contains no solutions.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    /// Iterate over the solutions.
    pub fn iter(&self) -> std::slice::Iter<'_, Solution> {
        self.solutions.iter()
    }

    /// Get a solution by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Solution> {
        self.solutions.get(index)
    }

    /// Get the objective values of every solution as a row-major matrix.
    #[must_use]
    pub fn objective_matrix(&self) -> Vec<Vec<f64>> {
        self.solutions
            .iter()
            .map(|s| s.objective_values.clone())
            .collect()
    }

    /// Sort the front by a given objective index.
    ///
    /// The sort direction respects the objective's optimisation sense:
    /// ascending for minimisation, descending for maximisation.
    pub fn sort_by_objective(&mut self, obj_index: usize) {
        let is_max = obj_index < self.objective_directions.len()
            && matches!(
                self.objective_directions[obj_index],
                crate::model::ObjectiveDirection::Maximize
            );
        self.solutions.sort_by(|a, b| {
            let va = a
                .objective_values
                .get(obj_index)
                .copied()
                .unwrap_or(f64::NAN);
            let vb = b
                .objective_values
                .get(obj_index)
                .copied()
                .unwrap_or(f64::NAN);
            if is_max {
                vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            }
        });
    }

    // ── Quality Metrics ─────────────────────────────────────────────────

    /// Compute the hypervolume indicator relative to a reference point.
    ///
    /// For a **bi-objective minimisation** problem the hypervolume equals the
    /// area of the union of rectangles between each solution and the reference
    /// point.  For higher dimensions a simple inclusion-exclusion sweep is used
    /// which is exact for 2-D and 3-D and provides a reasonable approximation
    /// for higher dimensions (though the complexity is exponential).
    ///
    /// Returns `None` if the front is empty or the reference point has the wrong
    /// dimensionality.
    #[must_use]
    pub fn hypervolume(&self, reference_point: &[f64]) -> Option<f64> {
        if self.solutions.is_empty() {
            return None;
        }
        let ndim = reference_point.len();
        if ndim == 0 {
            return None;
        }
        // Make sure all solutions have the same dimensionality
        if self
            .solutions
            .iter()
            .any(|s| s.objective_values.len() != ndim)
        {
            return None;
        }

        // Normalise: for maximisation objectives negate both solution and ref so
        // that we can compute hypervolume uniformly in the "lower-is-better" sense.
        let is_max_flags = self.is_maximizing_flags();
        let normalised: Vec<Vec<f64>> = self
            .solutions
            .iter()
            .map(|s| {
                s.objective_values
                    .iter()
                    .enumerate()
                    .map(|(k, &v)| {
                        if k < is_max_flags.len() && is_max_flags[k] {
                            -v
                        } else {
                            v
                        }
                    })
                    .collect()
            })
            .collect();
        let norm_ref: Vec<f64> = reference_point
            .iter()
            .enumerate()
            .map(|(k, &v)| {
                if k < is_max_flags.len() && is_max_flags[k] {
                    -v
                } else {
                    v
                }
            })
            .collect();

        // Filter out solutions that are worse than the reference in any objective
        let valid: Vec<&Vec<f64>> = normalised
            .iter()
            .filter(|s| s.iter().zip(norm_ref.iter()).all(|(sv, rv)| sv < rv))
            .collect();

        if valid.is_empty() {
            return Some(0.0);
        }

        if ndim == 1 {
            // 1-D: just the distance to the reference
            let best = valid.iter().map(|s| s[0]).fold(f64::INFINITY, f64::min);
            return Some(norm_ref[0] - best);
        }

        if ndim == 2 {
            return Some(Self::hypervolume_2d(&valid, &norm_ref));
        }

        // General n-D: use the HSO (Hypervolume by Slicing Objectives) algorithm
        // which is exact but exponential.  For practical SIMS problems n <= 4.
        Some(Self::hypervolume_nd(&valid, &norm_ref))
    }

    /// Efficient 2-D hypervolume (sweep-line).  Input must be in min-form.
    fn hypervolume_2d(points: &[&Vec<f64>], reference: &[f64]) -> f64 {
        // Sort by first objective ascending, then filter to the non-dominated
        // staircase (running minimum on second objective).
        let mut pts: Vec<(f64, f64)> = points.iter().map(|p| (p[0], p[1])).collect();
        pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut nd: Vec<(f64, f64)> = Vec::with_capacity(pts.len());
        let mut min_y = f64::INFINITY;
        for &(x, y) in &pts {
            if y < min_y {
                nd.push((x, y));
                min_y = y;
            }
        }

        let mut hv = 0.0;
        for i in 0..nd.len() {
            let x_lo = nd[i].0;
            let x_hi = if i + 1 < nd.len() {
                nd[i + 1].0
            } else {
                reference[0]
            };
            let y_height = reference[1] - nd[i].1;
            hv += (x_hi - x_lo) * y_height;
        }

        hv
    }

    /// General n-D hypervolume using a simple recursive inclusion-exclusion
    /// approach (HSO).  Exact but exponential – acceptable for n ≤ 5.
    fn hypervolume_nd(points: &[&Vec<f64>], reference: &[f64]) -> f64 {
        let n = reference.len();
        if n == 0 || points.is_empty() {
            return 0.0;
        }
        if n == 1 {
            let best = points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
            return reference[0] - best;
        }

        // Sort by last objective ascending
        let last = n - 1;
        let mut sorted: Vec<&Vec<f64>> = points.to_vec();
        sorted.sort_by(|a, b| {
            a[last]
                .partial_cmp(&b[last])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut hv = 0.0;
        let mut prev_bound = reference[last];

        for i in (0..sorted.len()).rev() {
            let slice_height = prev_bound - sorted[i][last];
            if slice_height <= 0.0 {
                continue;
            }

            // Project points[0..=i] onto the first (n-1) dimensions
            let sub_ref: Vec<f64> = reference[..last].to_vec();
            let sub_points: Vec<Vec<f64>> =
                sorted[..=i].iter().map(|p| p[..last].to_vec()).collect();
            // Filter to those that are within the sub-reference
            let valid_sub: Vec<&Vec<f64>> = sub_points
                .iter()
                .filter(|p| p.iter().zip(sub_ref.iter()).all(|(pv, rv)| pv < rv))
                .collect();

            if !valid_sub.is_empty() {
                let sub_hv = if last - 1 == 1 {
                    Self::hypervolume_2d(&valid_sub, &sub_ref)
                } else {
                    Self::hypervolume_nd(&valid_sub, &sub_ref)
                };
                hv += sub_hv * slice_height;
            }

            prev_bound = sorted[i][last];
        }

        hv
    }

    /// Compute the **spacing** metric (Schott, 1995).
    ///
    /// Spacing measures how uniformly the solutions are distributed.  A value
    /// of 0 means perfectly uniform spacing.
    ///
    /// Returns `None` if fewer than 2 solutions exist.
    #[must_use]
    pub fn spacing(&self) -> Option<f64> {
        let n = self.solutions.len();
        if n < 2 {
            return None;
        }

        // d_i = min distance from solution i to any other solution
        let distances: Vec<f64> = (0..n)
            .map(|i| {
                (0..n)
                    .filter(|&j| j != i)
                    .map(|j| self.euclidean_distance(i, j))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        #[allow(clippy::cast_precision_loss)]
        let n_f = n as f64;
        let d_mean: f64 = distances.iter().sum::<f64>() / n_f;
        let variance = distances.iter().map(|d| (d - d_mean).powi(2)).sum::<f64>() / (n_f - 1.0);
        Some(variance.sqrt())
    }

    /// Compute the **spread** (Δ) metric.
    ///
    /// Spread measures the extent and distribution of the Pareto front.
    /// Lower values indicate better distribution.  For bi-objective problems
    /// this uses the extreme-point formulation.
    ///
    /// Returns `None` if fewer than 2 solutions exist.
    #[must_use]
    pub fn spread(&self) -> Option<f64> {
        let n = self.solutions.len();
        if n < 2 {
            return None;
        }

        // Nearest-neighbour distances (sorted by first objective)
        let distances: Vec<f64> = (0..n)
            .map(|i| {
                (0..n)
                    .filter(|&j| j != i)
                    .map(|j| self.euclidean_distance(i, j))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        #[allow(clippy::cast_precision_loss)]
        let n_f = n as f64;
        let d_mean: f64 = distances.iter().sum::<f64>() / n_f;

        // Extreme-point distances (first and last solution to the boundary)
        let d_first = distances.first().copied().unwrap_or(0.0);
        let d_last = distances.last().copied().unwrap_or(0.0);

        let numerator =
            d_first + d_last + distances.iter().map(|d| (d - d_mean).abs()).sum::<f64>();
        let denominator = n_f.mul_add(d_mean, d_first + d_last);

        if denominator.abs() < f64::EPSILON {
            Some(0.0)
        } else {
            Some(numerator / denominator)
        }
    }

    /// Compute the **maximum gap** between consecutive solutions (sorted by the
    /// first objective).
    ///
    /// This is particularly useful for GPBA-A which aims to minimise the maximum
    /// gap.  Returns `None` if fewer than 2 solutions exist.
    #[must_use]
    pub fn max_gap(&self) -> Option<f64> {
        if self.solutions.len() < 2 {
            return None;
        }

        let mut max_d: f64 = 0.0;
        for i in 0..self.solutions.len() - 1 {
            let d = self.euclidean_distance(i, i + 1);
            if d > max_d {
                max_d = d;
            }
        }
        Some(max_d)
    }

    /// Compute the **minimum gap** between any two consecutive solutions (sorted
    /// by first objective).
    ///
    /// Useful for GPBA-B which aims to maximise the minimum distance.
    /// Returns `None` if fewer than 2 solutions exist.
    #[must_use]
    pub fn min_gap(&self) -> Option<f64> {
        if self.solutions.len() < 2 {
            return None;
        }

        let mut min_d = f64::INFINITY;
        for i in 0..self.solutions.len() - 1 {
            let d = self.euclidean_distance(i, i + 1);
            if d < min_d {
                min_d = d;
            }
        }
        Some(min_d)
    }

    /// Euclidean distance between solutions at indices `i` and `j`.
    fn euclidean_distance(&self, i: usize, j: usize) -> f64 {
        self.solutions[i]
            .objective_values
            .iter()
            .zip(self.solutions[j].objective_values.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Compute ideal (best-per-objective) and nadir (worst-per-objective) points
    /// from the current solutions.
    ///
    /// Returns `(ideal, nadir)` or `None` if the front is empty.
    #[must_use]
    pub fn ideal_nadir(&self) -> Option<(Vec<f64>, Vec<f64>)> {
        if self.solutions.is_empty() {
            return None;
        }
        let ndim = self.solutions[0].objective_values.len();
        let is_max_flags = self.is_maximizing_flags();

        let mut ideal = vec![f64::NAN; ndim];
        let mut nadir = vec![f64::NAN; ndim];

        for (k, (id, nad)) in ideal.iter_mut().zip(nadir.iter_mut()).enumerate() {
            let is_max = k < is_max_flags.len() && is_max_flags[k];
            let vals: Vec<f64> = self
                .solutions
                .iter()
                .map(|s| s.objective_values[k])
                .collect();
            if is_max {
                *id = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                *nad = vals.iter().copied().fold(f64::INFINITY, f64::min);
            } else {
                *id = vals.iter().copied().fold(f64::INFINITY, f64::min);
                *nad = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            }
        }

        Some((ideal, nadir))
    }

    // ── Validation ──────────────────────────────────────────────────────

    /// Assert that no solution in the front is dominated by another.
    ///
    /// This is a debugging aid: if the invariant is maintained correctly this
    /// should always pass.
    ///
    /// # Panics
    ///
    /// Panics if any pair of solutions has a dominance relationship.
    pub fn validate(&self) {
        let dirs = &self.objective_directions;
        for (i, a) in self.solutions.iter().enumerate() {
            for (j, b) in self.solutions.iter().enumerate() {
                if i != j {
                    assert!(
                        !dominates_with_directions(a, b, dirs),
                        "Pareto front invariant violated: solution {i} ({:?}) dominates solution {j} ({:?})",
                        a.objective_values,
                        b.objective_values,
                    );
                }
            }
        }
    }
}

// ── Default ─────────────────────────────────────────────────────────────

impl Default for ParetoFront {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

// ── Display ─────────────────────────────────────────────────────────────

impl fmt::Display for ParetoFront {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ParetoFront ({} solutions):", self.solutions.len())?;
        let dirs: Vec<&str> = self
            .objective_directions
            .iter()
            .map(|d| match d {
                crate::model::ObjectiveDirection::Minimize => "min",
                crate::model::ObjectiveDirection::Maximize => "max",
            })
            .collect();
        if !dirs.is_empty() {
            writeln!(f, "  Directions: [{}]", dirs.join(", "))?;
        }
        for (i, sol) in self.solutions.iter().enumerate() {
            writeln!(f, "  [{i:3}] {sol}")?;
        }
        if let Some((ideal, nadir)) = self.ideal_nadir() {
            writeln!(f, "  Ideal: {ideal:?}")?;
            writeln!(f, "  Nadir: {nadir:?}")?;
        }
        Ok(())
    }
}

// ── Index ───────────────────────────────────────────────────────────────

impl Index<usize> for ParetoFront {
    type Output = Solution;

    fn index(&self, index: usize) -> &Self::Output {
        &self.solutions[index]
    }
}

// ── IntoIterator ────────────────────────────────────────────────────────

impl IntoIterator for ParetoFront {
    type Item = Solution;
    type IntoIter = std::vec::IntoIter<Solution>;

    fn into_iter(self) -> Self::IntoIter {
        self.solutions.into_iter()
    }
}

impl<'a> IntoIterator for &'a ParetoFront {
    type Item = &'a Solution;
    type IntoIter = std::slice::Iter<'a, Solution>;

    fn into_iter(self) -> Self::IntoIter {
        self.solutions.iter()
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ObjectiveDirection;
    use std::collections::HashMap;
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn init_test_logging() {
        INIT.call_once(|| {
            env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
                .is_test(true)
                .try_init()
                .ok();
        });
    }

    fn sol(objs: Vec<f64>) -> Solution {
        Solution::new(objs, HashMap::new())
    }

    fn min_front() -> ParetoFront {
        ParetoFront::new(vec![
            ObjectiveDirection::Minimize,
            ObjectiveDirection::Minimize,
        ])
    }

    fn max_front() -> ParetoFront {
        ParetoFront::new(vec![
            ObjectiveDirection::Maximize,
            ObjectiveDirection::Maximize,
        ])
    }

    fn mixed_front() -> ParetoFront {
        ParetoFront::new(vec![
            ObjectiveDirection::Minimize,
            ObjectiveDirection::Maximize,
        ])
    }

    // ── HasObjectives trait ─────────────────────────────────────────────

    #[test]
    fn test_has_objectives_trait() {
        init_test_logging();
        let s = sol(vec![1.0, 2.0, 3.0]);
        assert_eq!(s.objectives(), &[1.0, 2.0, 3.0]);
    }

    // ── MoSolution trait ────────────────────────────────────────────────

    #[test]
    fn test_dominates_minimization() {
        init_test_logging();
        let s1 = sol(vec![1.0, 2.0]);
        let s2 = sol(vec![2.0, 3.0]);
        let is_max = vec![false, false];
        assert!(s1.dominates(&s2, &is_max));
        assert!(!s2.dominates(&s1, &is_max));
    }

    #[test]
    fn test_dominates_maximization() {
        init_test_logging();
        let s1 = sol(vec![2.0, 3.0]);
        let s2 = sol(vec![1.0, 2.0]);
        let is_max = vec![true, true];
        assert!(s1.dominates(&s2, &is_max));
        assert!(!s2.dominates(&s1, &is_max));
    }

    #[test]
    fn test_no_dominance() {
        init_test_logging();
        let s1 = sol(vec![1.0, 3.0]);
        let s2 = sol(vec![2.0, 2.0]);
        let is_max = vec![true, true];
        assert!(!s1.dominates(&s2, &is_max));
        assert!(!s2.dominates(&s1, &is_max));
    }

    #[test]
    fn test_covers() {
        init_test_logging();
        let s1 = sol(vec![1.0, 2.0]);
        let s2 = sol(vec![1.0, 3.0]);
        let is_max = vec![false, false];
        assert!(s1.covers(&s2, &is_max));
        assert!(!s2.covers(&s1, &is_max));
    }

    #[test]
    fn test_covers_equal() {
        init_test_logging();
        let s1 = sol(vec![1.0, 2.0]);
        let s2 = sol(vec![1.0, 2.0]);
        let is_max = vec![false, false];
        assert!(s1.covers(&s2, &is_max));
        assert!(s2.covers(&s1, &is_max));
    }

    #[test]
    fn test_convenience_methods() {
        init_test_logging();
        let s1 = sol(vec![1.0, 2.0]);
        let s2 = sol(vec![2.0, 3.0]);
        assert!(s1.dominates_min(&s2));
        assert!(s2.is_dominated_by_min(&s1));
        assert!(s2.dominates_max(&s1));
        assert!(s1.is_dominated_by_max(&s2));
    }

    #[test]
    fn test_nan_handling() {
        init_test_logging();
        let s1 = sol(vec![1.0, f64::NAN]);
        let s2 = sol(vec![2.0, 3.0]);
        let is_max = vec![true, true];
        assert!(!s1.dominates(&s2, &is_max));
        assert!(!s2.dominates(&s1, &is_max));
    }

    // ── Solution struct ─────────────────────────────────────────────────

    #[test]
    fn test_solution_new() {
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 1.0);
        let s = Solution::new(vec![10.0, 20.0], vars);
        assert!(s.feasible);
        assert_eq!(s.get_objective(0), Some(10.0));
        assert_eq!(s.get_variable("x"), Some(1.0));
        assert_eq!(s.get_variable("y"), None);
    }

    #[test]
    fn test_solution_infeasible() {
        let s = Solution::infeasible(3);
        assert!(!s.feasible);
        assert!(s.objective_values.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_solution_with_metadata() {
        let s = sol(vec![1.0]).with_metadata("solver", "cbc");
        assert_eq!(s.metadata.get("solver").map(String::as_str), Some("cbc"));
    }

    #[test]
    fn test_solution_display() {
        let s = sol(vec![1.5, 2.75]);
        let text = format!("{s}");
        assert!(text.contains("1.5000"));
        assert!(text.contains("2.7500"));
    }

    // ── ParetoFront – basic insertion ───────────────────────────────────

    #[test]
    fn test_empty_front() {
        let f = min_front();
        assert!(f.is_empty());
        assert_eq!(f.len(), 0);
    }

    #[test]
    fn test_insert_single() {
        let mut f = min_front();
        assert!(f.try_insert(sol(vec![1.0, 2.0])));
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn test_insert_non_dominated() {
        let mut f = min_front();
        assert!(f.try_insert(sol(vec![1.0, 4.0])));
        assert!(f.try_insert(sol(vec![3.0, 2.0])));
        assert_eq!(f.len(), 2);
        f.validate();
    }

    #[test]
    fn test_reject_dominated() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 2.0]));
        assert!(!f.try_insert(sol(vec![2.0, 3.0])));
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn test_remove_dominated_incumbents() {
        let mut f = min_front();
        f.try_insert(sol(vec![5.0, 5.0]));
        f.try_insert(sol(vec![3.0, 8.0]));
        assert_eq!(f.len(), 2);
        // Insert a solution that dominates the first
        assert!(f.try_insert(sol(vec![2.0, 2.0])));
        // (5,5) should have been removed because (2,2) dominates it
        // (3,8) should have been removed because (2,2) dominates it
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].objective_values, vec![2.0, 2.0]);
    }

    #[test]
    fn test_reject_duplicate() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 2.0]));
        assert!(!f.try_insert(sol(vec![1.0, 2.0])));
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn test_reject_infeasible() {
        let mut f = min_front();
        assert!(!f.try_insert(Solution::infeasible(2)));
        assert!(f.is_empty());
    }

    #[test]
    fn test_reject_nan_objectives() {
        let mut f = min_front();
        assert!(!f.try_insert(sol(vec![1.0, f64::NAN])));
        assert!(f.is_empty());
    }

    // ── Maximisation ────────────────────────────────────────────────────

    #[test]
    fn test_max_front_basic() {
        let mut f = max_front();
        f.try_insert(sol(vec![1.0, 4.0]));
        f.try_insert(sol(vec![3.0, 2.0]));
        assert_eq!(f.len(), 2);
        // (1,4) and (3,2) are non-dominated for max
        // Now insert (4,5) which dominates both
        assert!(f.try_insert(sol(vec![4.0, 5.0])));
        assert_eq!(f.len(), 1);
    }

    // ── Mixed directions ────────────────────────────────────────────────

    #[test]
    fn test_mixed_directions() {
        // min obj0, max obj1
        let mut f = mixed_front();
        f.try_insert(sol(vec![1.0, 2.0])); // good obj0, bad obj1
        f.try_insert(sol(vec![3.0, 5.0])); // bad obj0, good obj1
        assert_eq!(f.len(), 2);
        f.validate();

        // (2, 6) dominates (3, 5)? obj0: 2<3 good, obj1: 6>5 good → yes
        assert!(f.try_insert(sol(vec![2.0, 6.0])));
        assert_eq!(f.len(), 2); // (1,2) and (2,6) remain, (3,5) removed
        f.validate();
    }

    // ── Sorting ─────────────────────────────────────────────────────────

    #[test]
    fn test_sorted_by_first_objective_min() {
        let mut f = min_front();
        f.try_insert(sol(vec![5.0, 1.0]));
        f.try_insert(sol(vec![1.0, 5.0]));
        f.try_insert(sol(vec![3.0, 3.0]));
        // Should be sorted ascending by obj0
        assert_eq!(f[0].objective_values[0], 1.0);
        assert_eq!(f[1].objective_values[0], 3.0);
        assert_eq!(f[2].objective_values[0], 5.0);
    }

    #[test]
    fn test_sorted_by_first_objective_max() {
        let mut f = max_front();
        f.try_insert(sol(vec![1.0, 5.0]));
        f.try_insert(sol(vec![5.0, 1.0]));
        f.try_insert(sol(vec![3.0, 3.0]));
        // Should be sorted descending by obj0
        assert_eq!(f[0].objective_values[0], 5.0);
        assert_eq!(f[1].objective_values[0], 3.0);
        assert_eq!(f[2].objective_values[0], 1.0);
    }

    #[test]
    fn test_sort_by_objective() {
        let mut f = min_front();
        f.try_insert(sol(vec![5.0, 1.0]));
        f.try_insert(sol(vec![1.0, 5.0]));
        f.try_insert(sol(vec![3.0, 3.0]));
        f.sort_by_objective(1); // sort by second objective ascending
        assert_eq!(f[0].objective_values[1], 1.0);
        assert_eq!(f[1].objective_values[1], 3.0);
        assert_eq!(f[2].objective_values[1], 5.0);
    }

    // ── Merge ───────────────────────────────────────────────────────────

    #[test]
    fn test_merge() {
        let mut f1 = min_front();
        f1.try_insert(sol(vec![1.0, 5.0]));
        f1.try_insert(sol(vec![5.0, 1.0]));

        let mut f2 = min_front();
        f2.try_insert(sol(vec![2.0, 2.0]));
        f2.try_insert(sol(vec![6.0, 6.0])); // dominated by (1,5) and (5,1)

        f1.merge(&f2);
        // (2,2) is non-dominated with (1,5) and (5,1)
        // (6,6) is dominated and not added
        assert_eq!(f1.len(), 3);
        f1.validate();
    }

    #[test]
    fn test_merge_dominating() {
        let mut f1 = min_front();
        f1.try_insert(sol(vec![5.0, 5.0]));
        f1.try_insert(sol(vec![3.0, 8.0]));

        let mut f2 = min_front();
        f2.try_insert(sol(vec![1.0, 1.0])); // dominates everything in f1

        f1.merge(&f2);
        assert_eq!(f1.len(), 1);
        assert_eq!(f1[0].objective_values, vec![1.0, 1.0]);
    }

    // ── Batch filter (legacy) ───────────────────────────────────────────

    #[test]
    fn test_filter_dominated_solutions_legacy() {
        let mut f = min_front();
        // Bypass the invariant by pushing directly
        f.solutions.push(sol(vec![1.0, 5.0]));
        f.solutions.push(sol(vec![2.0, 6.0])); // dominated by (1,5)
        f.solutions.push(sol(vec![5.0, 1.0]));
        f.solutions.push(sol(vec![6.0, 2.0])); // dominated by (5,1)

        f.filter_dominated_solutions();
        assert_eq!(f.len(), 2);
        f.validate();
    }

    // ── Precision ───────────────────────────────────────────────────────

    #[test]
    fn test_precision_duplicate_detection() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 2.0]));
        // Very close solution should be treated as duplicate
        assert!(!f.try_insert_with_precision(sol(vec![1.0000001, 2.0000001]), 6));
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn test_precision_distinct_solutions() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 2.0]));
        // Different enough even with low precision
        assert!(f.try_insert_with_precision(sol(vec![2.0, 1.0]), 0));
        assert_eq!(f.len(), 2);
    }

    #[test]
    fn test_with_precision_builder() {
        let f = min_front().with_precision(3);
        assert_eq!(f.precision, 3);
    }

    // ── Iteration ───────────────────────────────────────────────────────

    #[test]
    fn test_iter() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 5.0]));
        f.try_insert(sol(vec![5.0, 1.0]));
        let objs: Vec<_> = f.iter().map(|s| s.objective_values.clone()).collect();
        assert_eq!(objs.len(), 2);
    }

    #[test]
    fn test_into_iter() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 5.0]));
        f.try_insert(sol(vec![5.0, 1.0]));
        let collected: Vec<Solution> = f.into_iter().collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn test_ref_into_iter() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 5.0]));
        f.try_insert(sol(vec![5.0, 1.0]));
        let count = (&f).into_iter().count();
        assert_eq!(count, 2);
    }

    // ── Indexing ────────────────────────────────────────────────────────

    #[test]
    fn test_index() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 5.0]));
        assert_eq!(f[0].objective_values, vec![1.0, 5.0]);
    }

    #[test]
    fn test_get() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 5.0]));
        assert!(f.get(0).is_some());
        assert!(f.get(1).is_none());
    }

    // ── objective_matrix ────────────────────────────────────────────────

    #[test]
    fn test_objective_matrix() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 5.0]));
        f.try_insert(sol(vec![5.0, 1.0]));
        let m = f.objective_matrix();
        assert_eq!(m.len(), 2);
        assert_eq!(m[0], vec![1.0, 5.0]);
        assert_eq!(m[1], vec![5.0, 1.0]);
    }

    // ── Display ─────────────────────────────────────────────────────────

    #[test]
    fn test_display() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 5.0]));
        f.try_insert(sol(vec![5.0, 1.0]));
        let text = format!("{f}");
        assert!(text.contains("ParetoFront (2 solutions)"));
        assert!(text.contains("min"));
        assert!(text.contains("Ideal"));
        assert!(text.contains("Nadir"));
    }

    // ── ideal_nadir ─────────────────────────────────────────────────────

    #[test]
    fn test_ideal_nadir_min() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 5.0]));
        f.try_insert(sol(vec![5.0, 1.0]));
        f.try_insert(sol(vec![3.0, 3.0]));
        let (ideal, nadir) = f.ideal_nadir().unwrap();
        assert_eq!(ideal, vec![1.0, 1.0]);
        assert_eq!(nadir, vec![5.0, 5.0]);
    }

    #[test]
    fn test_ideal_nadir_max() {
        let mut f = max_front();
        f.try_insert(sol(vec![1.0, 5.0]));
        f.try_insert(sol(vec![5.0, 1.0]));
        f.try_insert(sol(vec![3.0, 3.0]));
        let (ideal, nadir) = f.ideal_nadir().unwrap();
        assert_eq!(ideal, vec![5.0, 5.0]);
        assert_eq!(nadir, vec![1.0, 1.0]);
    }

    #[test]
    fn test_ideal_nadir_empty() {
        let f = min_front();
        assert!(f.ideal_nadir().is_none());
    }

    // ── Hypervolume ─────────────────────────────────────────────────────

    #[test]
    fn test_hypervolume_empty() {
        let f = min_front();
        assert!(f.hypervolume(&[10.0, 10.0]).is_none());
    }

    #[test]
    fn test_hypervolume_single_point() {
        let mut f = min_front();
        f.try_insert(sol(vec![2.0, 3.0]));
        let hv = f.hypervolume(&[10.0, 10.0]).unwrap();
        // Rectangle: (10-2) * (10-3) = 8 * 7 = 56
        assert!((hv - 56.0).abs() < 1e-9, "hv = {hv}");
    }

    #[test]
    fn test_hypervolume_two_points() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 5.0]));
        f.try_insert(sol(vec![3.0, 2.0]));
        let hv = f.hypervolume(&[6.0, 6.0]).unwrap();
        // Staircase: (3-1)*(6-5) + (6-3)*(6-2) = 2*1 + 3*4 = 2 + 12 = 14
        assert!((hv - 14.0).abs() < 1e-9, "hv = {hv}");
    }

    #[test]
    fn test_hypervolume_three_points() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 6.0]));
        f.try_insert(sol(vec![3.0, 4.0]));
        f.try_insert(sol(vec![5.0, 2.0]));
        let hv = f.hypervolume(&[7.0, 7.0]).unwrap();
        // Staircase:
        //   (3-1)*(7-6) = 2*1 = 2
        //   (5-3)*(7-4) = 2*3 = 6
        //   (7-5)*(7-2) = 2*5 = 10
        //   Total = 18
        assert!((hv - 18.0).abs() < 1e-9, "hv = {hv}");
    }

    #[test]
    fn test_hypervolume_reference_inside() {
        let mut f = min_front();
        f.try_insert(sol(vec![2.0, 3.0]));
        // Reference worse in obj0 only – solution point (2,3) has obj1=3 < ref1=1? No, 3>1
        // Actually ref = (10, 1) means obj1 ref is 1 which is BETTER than 3 for minimisation
        // So the solution is worse than the reference in obj1 → HV contribution = 0
        let hv = f.hypervolume(&[10.0, 1.0]).unwrap();
        assert!((hv - 0.0).abs() < 1e-9, "hv = {hv}");
    }

    #[test]
    fn test_hypervolume_maximization() {
        let mut f = max_front();
        f.try_insert(sol(vec![5.0, 3.0]));
        // Reference point: (0, 0) (worst for maximisation)
        // After negation: solution=(-5,-3), ref=(0,0)
        // Rectangle: (0-(-5)) * (0-(-3)) = 5*3 = 15
        let hv = f.hypervolume(&[0.0, 0.0]).unwrap();
        assert!((hv - 15.0).abs() < 1e-9, "hv = {hv}");
    }

    // ── Spacing ─────────────────────────────────────────────────────────

    #[test]
    fn test_spacing_too_few() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 2.0]));
        assert!(f.spacing().is_none());
    }

    #[test]
    fn test_spacing_uniform() {
        let mut f = min_front();
        // Three equally spaced points along the line y = 6-x
        f.try_insert(sol(vec![1.0, 5.0]));
        f.try_insert(sol(vec![3.0, 3.0]));
        f.try_insert(sol(vec![5.0, 1.0]));
        let sp = f.spacing().unwrap();
        // All nearest-neighbour distances are equal → spacing ~ 0
        assert!(sp < 1e-9, "spacing = {sp}");
    }

    // ── Spread ──────────────────────────────────────────────────────────

    #[test]
    fn test_spread_too_few() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 2.0]));
        assert!(f.spread().is_none());
    }

    #[test]
    fn test_spread_computes() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 5.0]));
        f.try_insert(sol(vec![3.0, 3.0]));
        f.try_insert(sol(vec![5.0, 1.0]));
        let sp = f.spread().unwrap();
        assert!(sp.is_finite());
    }

    // ── Max/Min gap ─────────────────────────────────────────────────────

    #[test]
    fn test_max_gap() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 5.0]));
        f.try_insert(sol(vec![3.0, 3.0]));
        f.try_insert(sol(vec![5.0, 1.0]));
        let gap = f.max_gap().unwrap();
        let expected = (4.0_f64 + 4.0_f64).sqrt(); // sqrt(8)
        assert!((gap - expected).abs() < 1e-9, "max_gap = {gap}");
    }

    #[test]
    fn test_min_gap() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 5.0]));
        f.try_insert(sol(vec![3.0, 3.0]));
        f.try_insert(sol(vec![5.0, 1.0]));
        let gap = f.min_gap().unwrap();
        let expected = (4.0_f64 + 4.0_f64).sqrt();
        assert!((gap - expected).abs() < 1e-9, "min_gap = {gap}");
    }

    #[test]
    fn test_gap_too_few() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 2.0]));
        assert!(f.max_gap().is_none());
        assert!(f.min_gap().is_none());
    }

    // ── Validate ────────────────────────────────────────────────────────

    #[test]
    fn test_validate_passes_on_good_front() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 5.0]));
        f.try_insert(sol(vec![5.0, 1.0]));
        f.validate(); // should not panic
    }

    #[test]
    #[should_panic(expected = "Pareto front invariant violated")]
    fn test_validate_catches_dominated() {
        let mut f = min_front();
        // Bypass the invariant on purpose
        f.solutions.push(sol(vec![1.0, 2.0]));
        f.solutions.push(sol(vec![2.0, 3.0])); // dominated
        f.validate();
    }

    // ── Large front stress test ─────────────────────────────────────────

    #[test]
    fn test_large_front() {
        let mut f = min_front();
        // Insert 100 points on the convex front y = 100 - x
        for i in 0..100 {
            let x = i as f64;
            let y = 100.0 - x;
            f.try_insert(sol(vec![x, y]));
        }
        assert_eq!(f.len(), 100);
        f.validate();

        // Now insert a point that dominates many
        assert!(f.try_insert(sol(vec![0.0, 0.0])));
        // Everything should be dominated by (0,0)
        assert_eq!(f.len(), 1);
    }

    // ── Payoff table ────────────────────────────────────────────────────

    #[test]
    fn test_set_payoff_table() {
        let mut f = min_front();
        f.set_payoff_table(vec![vec![1.0, 5.0], vec![5.0, 1.0]]);
        assert_eq!(f.payoff_table.len(), 2);
    }

    // ── Legacy API compatibility ────────────────────────────────────────

    #[test]
    fn test_legacy_add_solution() {
        let mut f = min_front();
        f.add_solution(sol(vec![1.0, 5.0]));
        f.add_solution(sol(vec![5.0, 1.0]));
        f.add_solution(sol(vec![3.0, 6.0])); // dominated
        assert_eq!(f.len(), 2);
        f.validate();
    }

    #[test]
    fn test_legacy_add_solution_with_precision() {
        let mut f = min_front();
        f.add_solution_with_precision(sol(vec![1.0, 5.0]), 0);
        f.add_solution_with_precision(sol(vec![5.0, 1.0]), 0);
        assert_eq!(f.len(), 2);
    }

    // ── 3-D hypervolume ─────────────────────────────────────────────────

    #[test]
    fn test_hypervolume_3d_single() {
        let dirs = vec![
            ObjectiveDirection::Minimize,
            ObjectiveDirection::Minimize,
            ObjectiveDirection::Minimize,
        ];
        let mut f = ParetoFront::new(dirs);
        f.try_insert(sol(vec![1.0, 1.0, 1.0]));
        let hv = f.hypervolume(&[2.0, 2.0, 2.0]).unwrap();
        // Cube: 1*1*1 = 1
        assert!((hv - 1.0).abs() < 1e-9, "hv = {hv}");
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn test_equal_objectives_not_dominated() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 2.0]));
        // Same objectives = duplicate, not domination
        assert!(!f.try_insert(sol(vec![1.0, 2.0])));
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn test_weakly_dominated_not_inserted() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 2.0]));
        // Same in obj0, worse in obj1 → dominated
        assert!(!f.try_insert(sol(vec![1.0, 3.0])));
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn test_weakly_dominating_replaces() {
        let mut f = min_front();
        f.try_insert(sol(vec![1.0, 3.0]));
        // Same in obj0, better in obj1 → dominates incumbent
        assert!(f.try_insert(sol(vec![1.0, 2.0])));
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].objective_values, vec![1.0, 2.0]);
    }

    #[test]
    fn test_default_front() {
        let f = ParetoFront::default();
        assert!(f.is_empty());
        assert!(f.objective_directions.is_empty());
    }
}
