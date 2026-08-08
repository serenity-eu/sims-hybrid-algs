//! Random cloud-mask generation — the "random clouds" path ported from Python's
//! `SimsProblem._generate_bool_clouds()`, with zero cloud-imagery processing.
//!
//! For each image the cloudy region is synthesised to match that image's cloud
//! coverage **by area**: with a budget of `coverage% × image_area`, the image's
//! fragments are visited in a uniformly random order and greedily marked cloudy,
//! skipping any fragment whose addition would overshoot the budget by more than
//! 10% (the faithful `1.1 ×` rule).
//!
//! ## Performance / determinism model
//! Cloud generation for each image is independent, so [`generate_clouds`] runs it
//! **in parallel across images** (rayon). Each image draws from its own
//! [`SmallRng`] seeded by a SplitMix64 mix of `(base_seed, image_index)`, so the
//! output is:
//! * **deterministic** for a given `seed`,
//! * **independent of the thread count** and of the number of images, and
//! * free of any cross-image RNG dependency (unlike a single shared generator).
//!
//! Hot-path choices: `SmallRng` (xoshiro, far cheaper than the crypto `StdRng`);
//! an **incremental Fisher–Yates** that needs no `clone` of the fragment list;
//! per-worker **scratch buffers** reused across images (`map_init`); and a mask
//! emit that yields the cloudy set already sorted (no final `sort`), relying on
//! the `images` contract that each fragment list is ascending.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// SplitMix64 finaliser mixing a base seed with an image index into a
/// well-distributed per-image seed (decorrelates adjacent indices).
#[inline]
fn image_seed(base: u64, idx: u64) -> u64 {
    let mut z = base.wrapping_add(idx.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Cloudy-fragment set for a single image (faithful to the Python greedy rule).
///
/// `order` and `mask` are caller-owned scratch buffers reused across images to
/// avoid per-image allocation; their contents on entry are irrelevant.
fn cloudy_for_image(
    image_fragments: &[usize],
    areas: &[f64],
    coverage_pct: f64,
    rng: &mut SmallRng,
    order: &mut Vec<u32>,
    mask: &mut Vec<bool>,
) -> Vec<usize> {
    let f = image_fragments.len();
    if coverage_pct <= 0.0 || f == 0 {
        return Vec::new();
    }
    debug_assert!(
        image_fragments.windows(2).all(|w| w[0] < w[1]),
        "image fragment lists must be strictly ascending for sorted cloud output"
    );

    let image_area: f64 = image_fragments.iter().map(|&fi| areas[fi]).sum();
    let budget = coverage_pct / 100.0 * image_area;
    if budget <= 0.0 {
        return Vec::new();
    }
    let limit = 1.1 * budget; // faithful "skip if it overshoots 110%" threshold

    // Scratch reset (reuses capacity across images within a worker).
    order.clear();
    order.extend(0..f as u32);
    mask.clear();
    mask.resize(f, false);

    let mut cloudy_area = 0.0_f64;
    let mut n_selected = 0usize;

    // Incremental Fisher–Yates: `order[i]` becomes a fresh uniformly-random local
    // position each step, i.e. we walk the image's fragments in shuffled order
    // without materialising a separate shuffled copy.
    for i in 0..f {
        let j = rng.random_range(i..f);
        order.swap(i, j);
        let pos = order[i] as usize;
        let a = areas[image_fragments[pos]];
        if cloudy_area + a >= limit {
            continue; // would overshoot: skip, but keep scanning for smaller ones
        }
        cloudy_area += a;
        mask[pos] = true;
        n_selected += 1;
    }

    // Emit in ascending fragment order (image_fragments is sorted ⇒ no sort).
    let mut out = Vec::with_capacity(n_selected);
    for (pos, &fi) in image_fragments.iter().enumerate() {
        if mask[pos] {
            out.push(fi);
        }
    }
    out
}

/// Generate cloudy fragment sets for every image, in parallel.
///
/// `clouds[i]` is the ascending list of fragment indices that are cloudy for
/// image `i`. Deterministic for a given `seed` (see the module-level note).
///
/// # Arguments
/// * `images`          – ascending fragment-index lists per image (0-based).
/// * `areas`           – area (m²) of each fragment, indexed by fragment index.
/// * `cloud_coverages` – cloud coverage percentage (0–100) per image.
/// * `seed`            – base seed for reproducible generation.
///
/// # Panics
/// Panics if `images.len() != cloud_coverages.len()`.
#[must_use]
pub fn generate_clouds(
    images: &[Vec<usize>],
    areas: &[f64],
    cloud_coverages: &[f64],
    seed: u64,
) -> Vec<Vec<usize>> {
    assert_eq!(
        images.len(),
        cloud_coverages.len(),
        "images ({}) and cloud_coverages ({}) length mismatch",
        images.len(),
        cloud_coverages.len()
    );

    images
        .par_iter()
        .zip(cloud_coverages.par_iter())
        .enumerate()
        .map_init(
            || (Vec::<u32>::new(), Vec::<bool>::new()),
            |(order, mask), (idx, (image_fragments, &coverage_pct))| {
                let mut rng = SmallRng::seed_from_u64(image_seed(seed, idx as u64));
                cloudy_for_image(image_fragments, areas, coverage_pct, &mut rng, order, mask)
            },
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_coverage_produces_empty_clouds() {
        let images = vec![vec![0, 1, 2]];
        let areas = vec![100.0, 200.0, 150.0];
        let clouds = generate_clouds(&images, &areas, &[0.0], 42);
        assert!(clouds[0].is_empty());
    }

    #[test]
    fn full_coverage_covers_all_fragments() {
        let images = vec![vec![0, 1, 2]];
        let areas = vec![100.0, 100.0, 100.0];
        let clouds = generate_clouds(&images, &areas, &[100.0], 42);
        assert_eq!(clouds[0], vec![0, 1, 2]);
    }

    #[test]
    fn clouds_are_sorted_subset_of_image_fragments() {
        let images = vec![vec![0, 1, 2, 3, 4]];
        let areas = vec![50.0, 80.0, 120.0, 90.0, 60.0];
        let clouds = generate_clouds(&images, &areas, &[40.0], 7);
        assert!(
            clouds[0].windows(2).all(|w| w[0] < w[1]),
            "must be ascending"
        );
        for &fi in &clouds[0] {
            assert!(images[0].contains(&fi), "cloudy fragment {fi} not in image");
        }
    }

    #[test]
    fn cloudy_area_respects_110_percent_budget() {
        // 5 equal fragments, 40% coverage → budget = 2 fragments, limit = 2.2 → at
        // most 2 fragments selected (a 3rd would reach 3 ≥ 2.2 and be skipped).
        let images = vec![vec![0, 1, 2, 3, 4]];
        let areas = vec![1.0; 5];
        for seed in 0..32 {
            let clouds = generate_clouds(&images, &areas, &[40.0], seed);
            assert!(
                clouds[0].len() <= 2,
                "budget overshoot at seed {seed}: {:?}",
                clouds[0]
            );
            assert!(!clouds[0].is_empty());
        }
    }

    #[test]
    fn deterministic_and_thread_count_independent() {
        // Per-image seeding ⇒ identical output for a seed, regardless of how rayon
        // schedules the images across threads.
        let images: Vec<Vec<usize>> = (0..50)
            .map(|k| (0..20).map(|j| k * 20 + j).collect())
            .collect();
        let areas: Vec<f64> = (0..1000).map(|i| 1.0 + (i % 7) as f64).collect();
        let cov: Vec<f64> = (0..50).map(|k| (k as f64 * 2.0) % 100.0).collect();

        let a = generate_clouds(&images, &areas, &cov, 12345);
        let b = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| generate_clouds(&images, &areas, &cov, 12345));
        assert_eq!(a, b, "output must be independent of the thread count");
    }
}
