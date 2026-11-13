//! Cross-worker deduplication of explored solutions.
//!
//! Each worker already refuses to re-explore a solution it has seen, through
//! [`crate::explored_solutions_data::ExploredSolutionsData`]. Nothing stops two
//! workers exploring the *same* solution independently, and with a decomposed
//! search they routinely do: a region boundary is not closed under the move
//! operator, so a worker reaches solutions its neighbours own. Measured on
//! `lagos_nigeria_50` at four objectives, 12--25% of single moves leave the
//! region that generated them, against a search where 99% of the work is
//! evaluating solutions that never reach the front. Every such crossing is a
//! full coverage evaluation that some other worker is entitled to repeat.
//!
//! This is the shared claim-check that stops it: the first worker to see a
//! solution claims it, and the others skip it.
//!
//! # Design
//!
//! An insert-only open-addressing table of 64-bit fingerprints, one `AtomicU64`
//! per slot, claimed with a single `compare_exchange`. Insert-only is what makes
//! it simple enough to be fast: there is no deletion, no resize, no epoch, no
//! allocation after construction, and therefore nothing to synchronise beyond
//! the individual slot. A reader that loses a race re-reads one word and either
//! finds its own fingerprint (someone else claimed it -- skip) or probes on.
//!
//! Linear probing keeps the common case to one cache line. The table never
//! grows: at construction it is sized for the run, and if it fills past
//! [`Self::SATURATION`] it stops claiming and reports every solution as new.
//! Degrading to "no deduplication" is the only safe direction -- a full table
//! that kept probing would spin over the whole array per lookup, and one that
//! evicted would silently lose solutions.
//!
//! # What a claim means
//!
//! Claiming happens at *discovery*, not after the neighbourhood is expanded,
//! matching the single-worker semantics of `ExploredSolutionsData`. So a worker
//! that claims a solution and then runs out of time leaves it unexplored, and no
//! other worker will pick it up. That is the same bargain the sequential search
//! already makes with itself; it costs a little coverage at the end of a run and
//! saves the duplicate evaluations throughout it.
//!
//! Fingerprints are 64-bit hashes, so distinct solutions can collide and one
//! will be skipped. At 10^6 solutions the birthday probability of any collision
//! is about 3 x 10^-8; a collision costs one unexplored neighbour, not
//! correctness, since the archive is what defines the result.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// A fingerprint the table can never store, used as the empty marker.
const EMPTY: u64 = 0;

/// Substituted for a fingerprint that happens to be [`EMPTY`].
///
/// One arbitrary non-zero value; the cost is that genuine `0` and
/// `0x9E37_79B9_7F4A_7C15` fingerprints alias, which is one collision in the
/// same class as any other.
const ZERO_SUBSTITUTE: u64 = 0x9E37_79B9_7F4A_7C15;

/// Shared "has anyone explored this?" filter.
#[derive(Debug)]
pub struct SharedVisited {
    slots: Box<[AtomicU64]>,
    /// `capacity - 1`; capacity is a power of two so this masks instead of
    /// dividing.
    mask: usize,
    /// Claimed slots, for saturation checks and reporting.
    claimed: AtomicUsize,
    /// Claims refused because the table was full.
    refused: AtomicUsize,
}

impl SharedVisited {
    /// Load factor past which claiming stops.
    ///
    /// Linear probing degrades sharply as a table fills; past this point the
    /// probe sequences cost more than the duplicate evaluations they prevent.
    pub const SATURATION: f64 = 0.7;

    /// Build a table sized for roughly `expected` solutions.
    ///
    /// Capacity is rounded up to a power of two with headroom for
    /// [`Self::SATURATION`], so a run that explores `expected` solutions does
    /// not saturate. Sizing generously is cheap -- one `u64` per slot -- and
    /// undersizing silently turns deduplication off.
    #[must_use]
    pub fn with_capacity_for(expected: usize) -> Self {
        let wanted = ((expected as f64) / Self::SATURATION).ceil() as usize;
        let capacity = wanted.next_power_of_two().max(1024);
        let mut slots = Vec::with_capacity(capacity);
        slots.resize_with(capacity, || AtomicU64::new(EMPTY));
        Self {
            slots: slots.into_boxed_slice(),
            mask: capacity - 1,
            claimed: AtomicUsize::new(0),
            refused: AtomicUsize::new(0),
        }
    }

    /// Claim `fingerprint` for this worker.
    ///
    /// Returns `true` when the caller is the first to see it and should explore
    /// it, `false` when another worker already has it. Also returns `true` once
    /// the table saturates, which turns the filter off rather than starving the
    /// search.
    pub fn claim(&self, fingerprint: u64) -> bool {
        let key = if fingerprint == EMPTY {
            ZERO_SUBSTITUTE
        } else {
            fingerprint
        };

        if self.claimed.load(Ordering::Relaxed) * 10 >= self.slots.len() * 7 {
            self.refused.fetch_add(1, Ordering::Relaxed);
            return true;
        }

        // Fibonacci hashing spreads fingerprints that share low bits, which
        // matters because linear probing clusters on exactly those.
        let mut idx = (key.wrapping_mul(ZERO_SUBSTITUTE) >> 32) as usize & self.mask;
        loop {
            match self.slots[idx].compare_exchange_weak(
                EMPTY,
                key,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                // Claimed it first.
                Ok(_) => {
                    self.claimed.fetch_add(1, Ordering::Relaxed);
                    return true;
                }
                // Lost the race or found an occupant: equal means someone else
                // owns this solution, otherwise keep probing.
                Err(occupant) => {
                    if occupant == key {
                        return false;
                    }
                    idx = (idx + 1) & self.mask;
                }
            }
        }
    }

    /// Solutions claimed so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.claimed.load(Ordering::Relaxed)
    }

    /// Whether nothing has been claimed yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Claims waved through because the table had saturated.
    ///
    /// Non-zero means the table was undersized and deduplication stopped part
    /// way through the run.
    #[must_use]
    pub fn refused(&self) -> usize {
        self.refused.load(Ordering::Relaxed)
    }

    /// Slots available.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn first_claim_wins_and_repeats_are_refused() {
        let v = SharedVisited::with_capacity_for(1024);
        assert!(v.claim(42), "first sighting is the claimant's");
        assert!(
            !v.claim(42),
            "a second sighting belongs to the first claimant"
        );
        assert!(v.claim(43));
        assert_eq!(v.len(), 2);
    }

    /// A zero fingerprint must not be mistaken for an empty slot.
    #[test]
    fn zero_is_storable() {
        let v = SharedVisited::with_capacity_for(1024);
        assert!(v.claim(0));
        assert!(!v.claim(0), "zero must claim like any other fingerprint");
    }

    /// Exactly one thread may claim a given solution, however many race for it.
    #[test]
    fn concurrent_claims_of_one_fingerprint_yield_one_winner() {
        let v = Arc::new(SharedVisited::with_capacity_for(4096));
        let winners: usize = std::thread::scope(|s| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let v = Arc::clone(&v);
                    s.spawn(move || usize::from(v.claim(7)))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).sum()
        });
        assert_eq!(winners, 1, "exactly one worker may own a solution");
    }

    /// Distinct fingerprints must all be claimable, across threads, with none
    /// lost to probing.
    #[test]
    fn concurrent_distinct_claims_all_succeed() {
        const N: u64 = 4096;
        let v = Arc::new(SharedVisited::with_capacity_for(N as usize));
        let winners: usize = std::thread::scope(|s| {
            let handles: Vec<_> = (0..8u64)
                .map(|t| {
                    let v = Arc::clone(&v);
                    s.spawn(move || {
                        (0..N)
                            .filter(|i| i % 8 == t)
                            .filter(|&i| v.claim(i.wrapping_mul(0x1000_0001)))
                            .count()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).sum()
        });
        assert_eq!(winners as u64, N, "every distinct solution claimed once");
        assert_eq!(v.len() as u64, N);
    }

    /// A saturated table stops filtering rather than spinning or evicting.
    #[test]
    fn saturation_disables_the_filter() {
        let v = SharedVisited::with_capacity_for(16);
        let cap = v.capacity();
        for i in 0..cap as u64 {
            v.claim(i.wrapping_mul(0x9E37_79B9).max(1));
        }
        assert!(v.refused() > 0, "a full table must wave claims through");
        assert!(v.claim(1), "and must never block the search");
    }
}
