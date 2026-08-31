//! Problem definition, bounds, and configuration.

use good_lp::{Constraint, Expression, ProblemVariables, Variable};
use std::collections::HashMap;
use std::time::Duration;

/// A multiobjective integer program, every objective minimised.
///
/// Minimisation only, deliberately: the rectangle geometry is stated that way,
/// and negating a maximised objective at construction is both trivial and less
/// error-prone than carrying a direction flag through every comparison.
pub struct Problem {
    /// Variable pool the expressions refer to.
    pub variables: ProblemVariables,
    /// Structural constraints.
    pub constraints: Vec<Constraint>,
    /// The objectives, all to minimise. At least two.
    pub objectives: Vec<Expression>,
    /// Variable names, for reporting decision values.
    pub var_names: HashMap<String, Variable>,
}

impl Problem {
    /// Number of objectives.
    #[must_use]
    pub fn num_objectives(&self) -> usize {
        self.objectives.len()
    }

    /// Evaluate every objective at a variable assignment.
    ///
    /// Integer-valued by construction: this algorithm is for *discrete*
    /// problems, and it relies on that in the `delta` step (one unit is exact
    /// and skips nothing). Typing the criterion space as integral makes the
    /// rectangle comparisons exact instead of approximate.
    #[must_use]
    pub fn objective_values(&self, values: &HashMap<Variable, f64>) -> Vec<i64> {
        self.objectives
            .iter()
            .map(|e| Self::eval(e, values))
            .collect()
    }

    /// Evaluate every objective without rounding.
    ///
    /// The criterion values are sums of floating-point coefficients -- areas
    /// here -- so they are not integers, and rounding them is only safe for
    /// reporting and for the rectangle geometry, where `delta` absorbs the
    /// half-unit. It is *not* safe for bounding a subsequent solve: a bound
    /// rounded down by a fraction excludes the very solution it came from.
    #[must_use]
    pub fn objective_values_exact(&self, values: &HashMap<Variable, f64>) -> Vec<f64> {
        self.objectives
            .iter()
            .map(|e| Self::eval_exact(e, values))
            .collect()
    }

    /// Evaluate one expression, constant term included.
    ///
    /// The constant matters. `linear_coefficients` yields only the variable
    /// terms, and an objective carrying an offset -- which a negated
    /// maximisation objective generally does -- would then read off by exactly
    /// that offset. Bounding such a value in a subproblem makes it infeasible
    /// rather than merely wrong, which is a far harder failure to trace.
    fn eval(expression: &Expression, values: &HashMap<Variable, f64>) -> i64 {
        objective_from_f64(Self::eval_exact(expression, values))
    }

    /// Evaluate one expression to full precision, constant term included.
    ///
    /// Variable values within [`INTEGRALITY_TOLERANCE`] of an integer are
    /// snapped to it first. Solvers return integer variables a hair off, and
    /// summed against large coefficients those hairs become tens of units --
    /// enough to misplace a criterion vector. Genuinely fractional values are
    /// further than the tolerance from any integer and pass through untouched.
    fn eval_exact(expression: &Expression, values: &HashMap<Variable, f64>) -> f64 {
        let linear: f64 = good_lp::IntoAffineExpression::linear_coefficients(expression)
            .map(|(v, c)| c * snap(values.get(&v).copied().unwrap_or(0.0)))
            .sum();
        linear + good_lp::IntoAffineExpression::constant(expression)
    }

    /// Decision variables by name, for the caller.
    #[must_use]
    pub fn named_values(&self, values: &HashMap<Variable, f64>) -> HashMap<String, f64> {
        self.var_names
            .iter()
            .map(|(name, var)| (name.clone(), values.get(var).copied().unwrap_or(0.0)))
            .collect()
    }
}

/// Bounds on the nondominated set, one entry per objective.
///
/// `anti_ideal` must be an upper bound on the nondominated values of each
/// objective, not necessarily tight. For two objectives the lexicographic
/// extremes give it exactly. For more, the true nadir is itself hard, and
/// maximising each objective gives a safe if loose substitute -- which is what
/// this crate computes when the caller supplies nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bounds {
    /// Per-objective minimum over the feasible set.
    pub ideal: Vec<i64>,
    /// Per-objective upper bound over the nondominated set.
    pub anti_ideal: Vec<i64>,
}

/// The outcome of a feasibility check.
///
/// Three states, not two. The third is what distinguishes this from a method
/// that must solve every subproblem to optimality: a check that runs out of
/// time without settling the question leaves the region *undecided* rather than
/// empty, and the search subdivides it instead of discarding it. Nothing is
/// lost, and the search never stalls on one hard node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feasibility {
    /// A feasible point was found in the region.
    Feasible,
    /// The region provably contains no feasible point.
    Infeasible,
    /// The check ran out of time. The region may or may not be empty.
    Undecided,
}

/// Why a solve stopped.
///
/// The distinction is the point: a point may only enter the front when its
/// solve was `Optimal`. A `Deadline` result carries an incumbent that is
/// feasible but unproven, and emitting those is exactly how an exact method
/// ends up reporting dominated solutions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveStatus {
    /// Proven optimal.
    Optimal,
    /// Time limit hit; any solution found is unproven.
    Deadline,
    /// No feasible solution exists.
    Infeasible,
}

/// Result of one scalarisation.
pub struct SolveResult {
    /// Why the solve stopped.
    pub status: SolveStatus,
    /// Present only when a feasible solution was found.
    pub values: Option<HashMap<Variable, f64>>,
    /// The objective value the *solver* reports, when it reports one.
    ///
    /// Preferred over re-evaluating the objective at the returned variable
    /// values, and the difference is not small. A solver returns binaries a
    /// hair off their integers -- 0.999999 rather than 1 -- and with a few
    /// hundred variables at coefficients around `1e6` that error accumulates
    /// into tens of units. A bound taken from the re-evaluated figure can
    /// therefore sit *below* the true optimum and exclude the very solution it
    /// came from, which the next solve reports as infeasible.
    pub objective_value: Option<f64>,
}

/// How far from an integer a variable value may sit and still be treated as
/// that integer.
///
/// Matches the usual solver integrality tolerance.
pub const INTEGRALITY_TOLERANCE: f64 = 1e-6;

/// Round a variable value to an integer when it is within
/// [`INTEGRALITY_TOLERANCE`] of one.
#[must_use]
pub fn snap(value: f64) -> f64 {
    let rounded = value.round();
    if (value - rounded).abs() < INTEGRALITY_TOLERANCE {
        rounded
    } else {
        value
    }
}

/// Objective magnitude beyond which the `i64` <-> `f64` conversions in this
/// crate would start losing integers.
///
/// `f64` represents every integer up to 2^53 exactly. Objective values here are
/// costs and areas -- in the instances this was built for they run to about
/// 1e9, seven orders below the limit -- so the conversions between the integral
/// criterion space and the solver's floating-point model are lossless in
/// practice. The debug assertions state that rather than leaving it implied.
pub const MAX_EXACT_OBJECTIVE: i64 = 1 << 53;

/// [`MAX_EXACT_OBJECTIVE`] as `f64`.
///
/// Spelled out rather than cast: 2^53 is itself exactly representable, but
/// writing the cast invites the reader (and the lint) to wonder.
pub const MAX_EXACT_OBJECTIVE_F64: f64 = 9_007_199_254_740_992.0;

/// `i64` -> `f64` for handing an objective value to the solver.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "values are bounded well below 2^53; see MAX_EXACT_OBJECTIVE"
)]
pub fn objective_to_f64(value: i64) -> f64 {
    debug_assert!(
        value.abs() < MAX_EXACT_OBJECTIVE,
        "objective {value} exceeds the exactly-representable range"
    );
    value as f64
}

/// `f64` -> `i64` for reading an objective value back from the solver.
///
/// Rounds rather than truncates: the solver returns values a hair off the
/// integers the model guarantees, and truncation would turn 41.999... into 41.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    reason = "rounded first, and bounded below 2^53; see MAX_EXACT_OBJECTIVE"
)]
pub fn objective_from_f64(value: f64) -> i64 {
    let rounded = value.round();
    debug_assert!(
        rounded.abs() < MAX_EXACT_OBJECTIVE_F64,
        "objective {rounded} exceeds the exactly-representable range"
    );
    rounded as i64
}

/// A step for one objective, derived from its range.
///
/// A millionth of the range: comfortably above the solver's resolution at these
/// magnitudes -- integrality tolerance times the coefficient sizes works out
/// around one unit -- and orders below the spacing of real nondominated points,
/// so it excludes a known point without stepping over its neighbour. Never
/// below one, which is the exact step when the values really are integral.
#[must_use]
pub fn scaled_delta(ideal: i64, anti_ideal: i64) -> i64 {
    const RELATIVE: f64 = 1e-6;
    let range = objective_to_f64((anti_ideal - ideal).max(0));
    objective_from_f64((range * RELATIVE).ceil()).max(1)
}

/// Configuration.
///
/// Every field defaults to "unset": no deadline, no per-solve cap, a step
/// derived from each objective's range, and no caller-supplied bounds or seeds.
#[derive(Default)]
pub struct Config {
    /// Global wall-clock budget. Enforced across all solves, not per solve.
    pub deadline: Option<Duration>,
    /// Cap on any single solve, so one hard subproblem cannot eat the budget.
    pub per_solve: Option<Duration>,
    /// Step separating a new point from a known one, `delta` in the reference.
    ///
    /// `None` derives one per objective from that objective's range; `Some`
    /// applies a fixed step to all of them.
    ///
    /// The reference calls this tolerance "really important", and it is: too
    /// small and the search re-finds points it already has, too large and it
    /// steps over them. Its `delta = 1` is right when the criterion values are
    /// genuinely discrete with a meaningful unit. These are not -- they are
    /// sums of floating-point areas reaching `1e9`, where one unit is *below*
    /// the solver's own numerical resolution. A step that small leaves the
    /// epsilon bound a hair from the point it is meant to exclude: the first
    /// stage returns essentially that point again, and the second stage's bound
    /// lands so tight against it that the solver reports infeasible.
    /// [`scaled_delta`] keeps the step above the noise and far below the gaps
    /// between genuine points.
    pub delta: Option<i64>,
    /// Bounds on the nondominated set, when the caller already has them.
    ///
    /// Supplying them saves `2p` solves. Omitting them makes this crate compute
    /// them, at the cost of those solves against its own deadline.
    pub bounds: Option<Bounds>,
    /// Points already known to be nondominated, reported as part of the front.
    ///
    /// The lexicographic extremes typically. They are not re-derived, but they
    /// also do not narrow the search: the initial rectangle already excludes
    /// them by construction.
    pub seeds: Vec<crate::front::Solution>,
    /// Where the epsilon bound is placed inside the chosen rectangle.
    pub split: Split,
}

/// Where the first stage's epsilon bound is placed inside a rectangle.
///
/// The choice does not affect which points are reported -- both settings report
/// only nondominated points, and both terminate with the complete front -- but
/// it decides the *order*, which is everything for a run stopped by a deadline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Split {
    /// Bound one `delta` below the rectangle's upper corner, as Kirlik & Sayin
    /// state it.
    ///
    /// Each solve returns the point nearest that corner, so the search walks
    /// the frontier from one end in `delta`-sized steps and a run cut short has
    /// covered a `delta`-scaled prefix of it. Measured on `lagos_nigeria_150`:
    /// eight points spanning 0.2% of the cost range in 200s, because at two
    /// objectives the projected space is a line, the rectangle list never holds
    /// more than one interval, and the "largest rectangle first" rule has
    /// nothing to choose between.
    Sweep,
    /// Bound at the midpoint of the rectangle's widest axis.
    ///
    /// The solve then splits that axis in two rather than shaving one step off
    /// its end, so the list grows, selection has something to choose between,
    /// and the points arrive spread across the frontier instead of in order
    /// from one end. This is the balanced-box idea of Boland, Charkhgard &
    /// Savelsbergh (2015) carried into the rectangle search.
    ///
    /// Exhausting the front costs somewhat more solves than [`Split::Sweep`],
    /// because a bisection can rediscover a known point where the sweep never
    /// does. That is the trade: worse for running to completion, far better for
    /// stopping on a clock.
    ///
    /// The default: a caller that passes a deadline at all is in the regime
    /// this setting is for.
    #[default]
    Bisect,
}

#[cfg(test)]
mod tests {
    use super::{MAX_EXACT_OBJECTIVE, Problem};
    use good_lp::{ProblemVariables, variable};
    use std::collections::HashMap;

    /// An objective carrying a constant offset must evaluate to include it.
    ///
    /// Regression: evaluating only the variable terms made a bounded subproblem
    /// infeasible, because the value bounded differed from the value the
    /// constraint computes by exactly the offset. A negated maximisation
    /// objective usually carries one.
    #[test]
    fn objective_values_include_the_constant_term() {
        let mut vars = ProblemVariables::new();
        let x = vars.add(variable().binary());

        let problem = Problem {
            // `2*x - 10` and `-(3*x + 7)`, the latter as a negated maximisation
            // objective reaches this crate.
            objectives: vec![2.0 * x - 10.0, -(3.0 * x + 7.0)],
            var_names: HashMap::from([("x".to_string(), x)]),
            variables: vars,
            constraints: Vec::new(),
        };

        assert_eq!(
            problem.objective_values(&HashMap::from([(x, 1.0)])),
            vec![-8, -10]
        );
        assert_eq!(
            problem.objective_values(&HashMap::from([(x, 0.0)])),
            vec![-10, -7]
        );
    }

    /// The step must clear the solver's numerical resolution at these
    /// magnitudes. `delta = 1` against a range of `1.4e9` leaves the epsilon
    /// bound a hair from the point it excludes -- measured on
    /// `lagos_nigeria_150` as the search returning the same extreme back and
    /// the second stage then reporting infeasible.
    #[test]
    fn the_step_scales_with_the_objective_range() {
        use super::scaled_delta;
        assert_eq!(scaled_delta(620_841, 1_428_692_042), 1429);
        assert_eq!(scaled_delta(6_400_456, 31_286_476), 25);
        // Genuinely small ranges keep the exact unit step.
        assert_eq!(scaled_delta(0, 10), 1);
        assert_eq!(scaled_delta(7, 7), 1, "a collapsed range still steps");
        assert_eq!(scaled_delta(9, 7), 1, "and so does an inverted one");
    }

    /// Regression: a solver returns integer variables a hair off, and against
    /// large coefficients those hairs become tens of units.
    ///
    /// Measured on `lagos_nigeria_150`: re-evaluating the cost objective at the
    /// returned values gave 6400456.22 where the solver reported the true
    /// optimum, and bounding the next solve at the lower figure excluded the
    /// solution it came from.
    #[test]
    fn snapping_recovers_the_exact_combinatorial_value() {
        use super::snap;
        assert!((snap(0.999_999_9) - 1.0).abs() < f64::EPSILON);
        assert!((snap(1e-9) - 0.0).abs() < f64::EPSILON);
        // Genuinely fractional values pass through.
        assert!((snap(0.5) - 0.5).abs() < f64::EPSILON);
        assert!((snap(3.25) - 3.25).abs() < f64::EPSILON);

        // 225 binaries at 1e6, each returned one part in 1e7 low.
        let drift: f64 = (0..225).map(|_| 1e6 * (1.0 - 1e-7)).sum();
        let exact: f64 = 225.0 * 1e6;
        assert!(
            exact - drift > 20.0,
            "the accumulated error is tens of units"
        );
        let snapped: f64 = (0..225).map(|_| 1e6 * snap(1.0 - 1e-7)).sum();
        assert!((snapped - exact).abs() < 1e-6, "snapping removes it");
    }

    #[test]
    fn exact_objective_bound_is_two_to_the_fifty_third() {
        assert_eq!(MAX_EXACT_OBJECTIVE, 9_007_199_254_740_992);
    }
}

#[cfg(test)]
mod exactness_tests {
    use super::Problem;
    use good_lp::{ProblemVariables, variable};
    use std::collections::HashMap;

    /// Regression: a bound taken from a rounded objective can exclude the very
    /// solution it came from.
    ///
    /// Criterion values here are sums of floating-point areas. Rounding
    /// `6400456.3` to `6400456` and imposing it as `f <= 6400456` cuts off the
    /// point that produced it, and the next solve reports infeasible on a
    /// region that demonstrably contains a solution -- indistinguishable, to
    /// the caller, from a region that is genuinely empty.
    #[test]
    fn exact_evaluation_does_not_round_the_bound_below_the_solution() {
        let mut vars = ProblemVariables::new();
        let x = vars.add(variable().binary());
        let problem = Problem {
            objectives: vec![6_400_456.3 * x, 2.0 * x],
            var_names: HashMap::from([("x".to_string(), x)]),
            variables: vars,
            constraints: Vec::new(),
        };
        let at_one = HashMap::from([(x, 1.0)]);

        let exact = problem.objective_values_exact(&at_one)[0];
        let rounded = problem.objective_values(&at_one)[0];

        assert!((exact - 6_400_456.3).abs() < 1e-6);
        assert_eq!(rounded, 6_400_456);
        assert!(
            (rounded as f64) < exact,
            "the rounded bound falls below the solution it came from"
        );
    }
}
