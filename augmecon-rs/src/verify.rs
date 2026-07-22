//! Lightweight runtime counters to verify which AUGMECON-family techniques are
//! actually executed on the live GPBA call graph (not merely present in source).
//! Enabled by the `GPBA_VERIFY` env var; zero cost otherwise (atomic adds only).

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

pub static SLACK_SOLVES: AtomicU64 = AtomicU64::new(0); // #1/#2/#3 augmentation+slack ε-solve
pub static AUGMENTATION_BUILT: AtomicU64 = AtomicU64::new(0); // #1 augmented objective assembled
pub static INTERVAL_REMOVALS: AtomicU64 = AtomicU64::new(0); // #4/#5 bypass / bouncing
pub static RELAXATION_REUSE: AtomicU64 = AtomicU64::new(0); // #6 SAUGMECON Lemma 1 (feasible reuse)
pub static INFEASIBLE_PROP: AtomicU64 = AtomicU64::new(0); // #6/#7 SAUGMECON Lemma 2 (infeasible skip)
pub static CASCADE_EXITS: AtomicU64 = AtomicU64::new(0); // #7 early exit / dimension cascade

// ── Phase timers (nanoseconds accumulated across the solve) ──────────────────
pub static PAYOFF_NS: AtomicU64 = AtomicU64::new(0); // ideal-bounds / payoff-table solves
pub static EPS_CALL_NS: AtomicU64 = AtomicU64::new(0); // full ε-subproblem call (build + MILP solve)
pub static SOLVE_NS: AtomicU64 = AtomicU64::new(0); // just the MILP `model.solve()`
pub static TOTAL_NS: AtomicU64 = AtomicU64::new(0); // whole generate_representation

#[inline]
fn on() -> bool {
    std::env::var_os("GPBA_VERIFY").is_some()
}

#[inline]
pub fn bump(c: &AtomicU64) {
    if on() {
        c.fetch_add(1, Ordering::Relaxed);
    }
}

#[inline]
pub fn add_ns(c: &AtomicU64, d: Duration) {
    if on() {
        c.fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
    }
}

pub fn report() {
    if std::env::var_os("GPBA_VERIFY").is_none() {
        return;
    }
    eprintln!("=== GPBA technique execution counters (this solve) ===");
    eprintln!(
        "  #1/2/3 augmentation+slack ε-solves : {}",
        SLACK_SOLVES.load(Ordering::Relaxed)
    );
    eprintln!(
        "  #1     augmented objective built   : {}",
        AUGMENTATION_BUILT.load(Ordering::Relaxed)
    );
    eprintln!(
        "  #4/5   bypass/interval removals     : {}",
        INTERVAL_REMOVALS.load(Ordering::Relaxed)
    );
    eprintln!(
        "  #6     relaxation reuse (Lemma 1)   : {}",
        RELAXATION_REUSE.load(Ordering::Relaxed)
    );
    eprintln!(
        "  #6/7   infeasible propagation (L2)  : {}",
        INFEASIBLE_PROP.load(Ordering::Relaxed)
    );
    eprintln!(
        "  #7     cascade / early exit         : {}",
        CASCADE_EXITS.load(Ordering::Relaxed)
    );

    let total = TOTAL_NS.load(Ordering::Relaxed).max(1);
    let payoff = PAYOFF_NS.load(Ordering::Relaxed);
    let eps_call = EPS_CALL_NS.load(Ordering::Relaxed);
    let solve = SOLVE_NS.load(Ordering::Relaxed);
    let build = eps_call.saturating_sub(solve); // ε-model construction (builder + constraints)
    let overhead = total.saturating_sub(payoff + eps_call); // loop bookkeeping: relaxation, intervals, front
    let n = SLACK_SOLVES.load(Ordering::Relaxed).max(1);
    let ms = |ns: u64| ns as f64 / 1e6;
    let pct = |ns: u64| 100.0 * ns as f64 / total as f64;
    eprintln!("  --- WHERE TIME GOES ---");
    eprintln!("  TOTAL                     : {:9.1} ms", ms(total));
    eprintln!(
        "  payoff table (ideal)      : {:9.1} ms  ({:5.1}%)",
        ms(payoff),
        pct(payoff)
    );
    eprintln!(
        "  ε MILP solve  ({:>4} solves): {:9.1} ms  ({:5.1}%)   avg {:.1} ms/solve",
        SLACK_SOLVES.load(Ordering::Relaxed),
        ms(solve),
        pct(solve),
        ms(solve) / n as f64
    );
    eprintln!(
        "  ε model build             : {:9.1} ms  ({:5.1}%)   avg {:.1} ms/solve",
        ms(build),
        pct(build),
        ms(build) / n as f64
    );
    eprintln!(
        "  loop overhead (relax/ivl) : {:9.1} ms  ({:5.1}%)",
        ms(overhead),
        pct(overhead)
    );
}

/// Reset all counters (call at the start of a solve so numbers are per-run).
pub fn reset() {
    for c in [
        &SLACK_SOLVES,
        &AUGMENTATION_BUILT,
        &INTERVAL_REMOVALS,
        &RELAXATION_REUSE,
        &INFEASIBLE_PROP,
        &CASCADE_EXITS,
        &PAYOFF_NS,
        &EPS_CALL_NS,
        &SOLVE_NS,
        &TOTAL_NS,
    ] {
        c.store(0, Ordering::Relaxed);
    }
}
