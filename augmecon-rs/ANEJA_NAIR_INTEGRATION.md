# Integrating Anytime Aneja & Nair into `augmecon-rs`

**Why:** the PPSN paper's *winning* first-phase method is **Anytime Aneja & Nair (A&N)**, but our Rust pipeline only implements GPBA-A. Closing this gap is the prerequisite for "Path B" in `paper/ppsn2026/PAPER_REVISION_PLAN.md` (framing the paper in our HiGHS + tuned-timeout protocol). This document evaluates how to add A&N and proposes a clean integration that maximises reuse of the existing exact-phase machinery.

---

## 1. What A&N is (and how it differs from GPBA-A)

Both methods produce a **well-distributed representative subset of the Pareto front within a timeout**, on the *same* MILP model, with the *same* solver. They differ only in the scalarisation they solve and how they pick the next one:

| | GPBA-A (have) | Anytime A&N (add) |
|---|---|---|
| Scalarisation | **ε-constraint** (`min f₁ s.t. f₂ − s = ε`) | **weighted sum** (`min w₁f₁ + w₂f₂`) |
| Next point chosen by | largest unexplored interval in **f₂** | largest gap between adjacent points in **objective space** |
| Finds | supported **and** unsupported Pareto points | **supported only** (convex hull of the front) |
| Solve type | single-objective + slack + augmentation | plain single-objective (linear scalarisation) |

The "supported only" difference is deliberate — the paper's whole point is to test whether it matters (it barely does; A&N+PLS ≻ GPBA-A+PLS by a hair). A&N is *simpler* to solve than GPBA-A (no slack, no augmentation term).

### The anytime algorithm (Dubois-Lacoste et al., 2011)
1. Solve each objective individually (lexicographically refined) → the two **extreme** Pareto points.
2. Maintain the found points sorted by f₁ ascending (⇒ f₂ descending), and a priority queue of **gaps** between adjacent points.
3. Repeatedly take the **largest** gap `(a, b)`, solve the weighted sum with the weight that makes `a` and `b` equal, and:
   - if a **new** point `z` strictly below the line `ab` is found → insert `z`, split the gap into `(a,z)` and `(z,b)`;
   - else → the gap has **no** further supported solution (exact solver guarantees this) → close it.
4. Stop at timeout or when all gaps are closed.

Filling the *largest* gap first is what gives the anytime behaviour: the front is well-spread at any interruption point.

---

## 2. Reuse vs. new — the integration surface

The good news: A&N reuses almost everything. Only the *driver* and a *weighted-sum solve* are new.

| Component | Action | Where |
|---|---|---|
| MILP model + solver backends (HiGHS `presolve=off`, CBC) | **reuse** | `single_objective.rs` already builds+solves an objective on `MultiObjectiveProblem` |
| `MultiObjectiveProblem` (`variables`, `objectives: Vec<(Expression, ObjectiveDirection)>`) | **reuse** | `model.rs:71` |
| `ParetoFront`, `Solution`, `Timer`, `Options` (solver choice) | **reuse** | unchanged |
| Lexicographic extreme points | **reuse** | `solve_objective` (`single_objective.rs:63`) + `solve_objective_with_constraints` (`:363`) |
| Per-solve timeout cap + incumbent acceptance | **reuse** | same pattern as GPBA's `per_solve_timeout` |
| HiGHS presolve-off fix | **inherited free** | A&N solves go through the same single-objective path |
| **Weighted-sum solve** | **add (small)** | `solve_weighted_sum(weights, timeout)` in `single_objective.rs` |
| **A&N driver** | **add (new module)** | `aneja_nair.rs` mirroring `GpbaA` |
| PyO3 exposure + hybrid seeding | **add (wiring)** | `sims-problem/src/solver.rs` |

---

## 3. Proposed design

### 3.1 `solve_weighted_sum` — the one new solve primitive
`single_objective::solve_objective` already builds one objective's expression, applies direction, sets solver + timeout, solves, and extracts a `Solution` (`single_objective.rs:63`). Add a sibling that builds a **linear combination** of the objectives instead of a single one:

```rust
// single_objective.rs
/// Minimise Σ wₗ · fₗ(x) over the feasible set. Weights are in *minimisation*
/// form (objectives already normalised to minimisation upstream). Reuses the
/// exact same model-build + solver + timeout path as `solve_objective`.
pub fn solve_weighted_sum(
    &self,
    weights: &[f64],            // one per objective, ≥ 0
    timeout: Option<Duration>,
) -> Result<Solution> {
    let expr: Expression = self.problem.objectives.iter()
        .zip(weights)
        .map(|((obj_expr, dir), &w)| match dir {
            ObjectiveDirection::Minimize =>  w * obj_expr.clone(),
            ObjectiveDirection::Maximize => -w * obj_expr.clone(),
        })
        .sum();
    // …identical to solve_objective from here: minimise `expr`, apply
    //   effective_timeout, `presolve=off`, extract Solution…
}
```
Factor the shared tail of `solve_objective` (build model → set options → solve → extract) into a private `solve_expression(expr, timeout)` so both entry points call it — no duplication.

### 3.2 `aneja_nair.rs` — the driver, mirroring `GpbaA`
Same public shape as `GpbaA` so the two are interchangeable at the call site:

```rust
pub struct AnejaNairConfig {
    pub per_solve_timeout: Option<Duration>,  // same cap semantics as GpbaConfig
    pub target_solutions: Option<usize>,      // optional early stop
}

pub struct AnejaNair { config: AnejaNairConfig, timer: Option<Timer> }

impl AnejaNair {
    pub fn new(config: AnejaNairConfig) -> Self { … }
    pub fn with_timeout(mut self, t: Duration) -> Self { … }

    pub fn generate_representation(
        &mut self,
        problem: &MultiObjectiveProblem,
        options: &Options,
    ) -> Result<ParetoFront> {
        let solver = SingleObjectiveSolver::new(problem, options);

        // 1. Two lexicographic extreme points (reuse existing solves)
        let p_min_f1 = lexicographic_extreme(&solver, 0, 1, budget)?; // min f1, tie-break min f2
        let p_min_f2 = lexicographic_extreme(&solver, 1, 0, budget)?; // min f2, tie-break min f1
        let mut front = ParetoFront::new();
        front.add(p_min_f1); front.add(p_min_f2);

        // 2. Gap priority queue, largest first
        let mut gaps = BinaryHeap::new();
        gaps.push(Gap::new(&p_min_f1, &p_min_f2)); // priority = objective-space area/spread

        // 3. Anytime loop
        while let Some(gap) = gaps.pop() {
            if self.timed_out() { break; }
            let w = equalizing_weight(&gap.a, &gap.b);      // see §4.1
            let z = solver.solve_weighted_sum(&w, self.per_solve())?;
            if is_new_between(&z, &gap.a, &gap.b) {
                front.add_solution(z.clone());
                gaps.push(Gap::new(&gap.a, &z));
                gaps.push(Gap::new(&z, &gap.b));
            } // else: gap closed, discard
        }
        Ok(front)
    }
}
```

### 3.3 Optional: a shared `FirstPhaseSolver` trait
Both drivers already have identical signatures. A one-method trait lets `solve_with_milp` dispatch without duplicating the tail:
```rust
pub trait FirstPhaseSolver {
    fn generate_representation(&mut self, p: &MultiObjectiveProblem, o: &Options) -> Result<ParetoFront>;
}
impl FirstPhaseSolver for GpbaA { … }
impl FirstPhaseSolver for AnejaNair { … }
```
Nice-to-have, not required.

### 3.4 PyO3 exposure (`sims-problem/src/solver.rs`)
`solve_with_milp` currently hard-codes `GpbaA::new(config)…generate_representation` (~`solver.rs:2035`). Add a **method selector**:
```rust
// new pyfunction arg:  method: &str = "gpba"   // "gpba" | "aneja"
let front = match method {
    "aneja" => AnejaNair::new(an_cfg).with_timeout(timeout).generate_representation(&problem, &options)?,
    _       => GpbaA::new(gpba_cfg).with_timeout(timeout).generate_representation(&problem, &options)?,
};
```
The downstream conversion to Python `Solution`s (timestamp plumbing, objective recompute) is **unchanged** — both return a `ParetoFront`.

### 3.5 Pseudo-dataset for the hybrid seeding
The hybrid configs read a pre-generated exact front from `sims-core/tests/data/gpbaa_2d_highs/`. Generate the A&N analogue with the **same generator** plus a `--method aneja` flag → `sims-core/tests/data/an_2d_highs/`, and register it as a `--pseudo-source` choice (mirrors `gpbaa_2d_highs`). Then `Hybrid A&N + PLS` slots into the experiment harness exactly like the GPBA hybrids.

---

## 4. Correctness & numerical care (the parts that bite)

### 4.1 Equalizing weight
For adjacent points `a=(a₁,a₂)`, `b=(b₁,b₂)` with `a₁<b₁`, `a₂>b₂`, the weight normal to segment `ab` is
`w = (a₂ − b₂, b₁ − a₁)` — both components positive. Minimising `w·f` finds the supported point in the gap (or returns `a`/`b` if none exists).

### 4.2 ⚠️ Weight/objective magnitude — real overflow risk
Objectives here are **large integers**: cost ≈ 8·10⁶, cloud ≈ 1.4·10⁹. The raw equalizing weights are objective *differences* (up to ~10⁹), so the scalarised coefficient `w·f` reaches ≈ 10¹⁶ — **past the 2⁵³ ≈ 9·10¹⁵ exact-integer range of `f64`**. good_lp/HiGHS carry coefficients as `f64`, so this silently loses precision and can return a **wrong "optimum"**, breaking A&N's supported-point guarantee.
**Mitigation (required):** normalise each weight vector before solving — divide by `gcd(w₁, w₂)` and/or by the objective **ranges** (`wₗ/rₗ`, exactly what AUGMECON does). Keep the scalarised objective's coefficients well under 2⁵³. This is the single most important implementation detail.

### 4.3 Weakly-dominated tie-break
Weighted-sum optima can be weakly dominated when multiple supported solutions share a `w·f` value; the paper notes A&N returns "one of them." If a Pareto-*optimal* representative is wanted, add a tiny lexicographic tie-break (min f₁ then f₂ among `w`-optima) — reuse `solve_objective_with_constraints`. Optional for the comparison; the ε-augmentation trick is *not* needed (that's GPBA's mechanism).

### 4.4 Timeout & incumbent
Bound each weighted-sum solve with the same `per_solve_timeout` cap as GPBA, and accept a feasible incumbent on timeout (good_lp maps HiGHS `ReachedTimeLimit → Ok`). The `presolve=off` fix is inherited, so A&N won't hang on the large models the way un-fixed GPBA did.

### 4.5 Degenerate fronts
Single-point front (both extremes coincide) → return it, no gaps. All-supported small fronts terminate quickly with every gap closed — expected.

---

## 5. Testing

- **Unit:** `equalizing_weight` sign/normalisation; `is_new_between`; gap-heap ordering.
- **Cross-check vs GPBA-A on small instances:** every A&N point must be **supported** and must lie **on or above GPBA-A's front** (A&N ⊆ convex hull of GPBA's front). GPBA finds ⊇ A&N's points.
- **Timeout monotonicity:** more budget ⇒ ≥ as many points, HV non-decreasing.
- **Overflow guard:** an instance with large objective spreads must still return provably-optimal supported points (compare scalarised objective value against a brute-force check on a tiny instance).

---

## 6. Effort & sequence

| Step | Est. |
|---|---|
| `solve_weighted_sum` + factor `solve_expression` | ~40 lines |
| `aneja_nair.rs` driver (gap heap, weight, loop, extremes) | ~250 lines |
| Weight normalisation + tie-break | ~30 lines |
| `FirstPhaseSolver` trait (optional) | ~15 lines |
| PyO3 `method` arg + dispatch | ~40 lines |
| A&N pseudo-dataset generator flag + regen | reuse `generate_pseudo.py` + `--method` |
| Tests | ~150 lines |

**Sequence:** `solve_weighted_sum` → driver → unit tests → PyO3 wiring → cross-check vs GPBA on 2–3 small instances → generate `an_2d_highs` → wire `--pseudo-source an_2d_highs` into the experiment harness → run `A&N` and `A&N+PLS` under the paper's protocol.

---

## 7. Open questions
- **Reuse GPBA's `BoundsCalculator` for the extremes, or A&N-local lexicographic solves?** The latter is self-contained (no coupling to GPBA internals) — recommended.
- **Weight normalisation scheme:** `gcd` (exact, integer) vs `wₗ/rₗ` (range-based, matches AUGMECON). Recommend **gcd then cap** to keep coefficients exact and small.
- **Gap priority metric:** objective-space area of the triangle vs L∞ box vs Euclidean segment length. Dubois-Lacoste uses the *hypervolume gap*; the triangle area is the cheap, standard proxy — recommend that.
- **Solver for A&N in the paper:** HiGHS `presolve=off` (consistent with our GPBA runs) — confirms Path B's "open-source, reproducible" framing, but A&N's `min w·f` is an easier MILP than GPBA's, so Gurobi-vs-HiGHS should matter even less here.
